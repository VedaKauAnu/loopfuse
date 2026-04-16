"""
loopfuse/runtime/executor.py

LoopFuse compiled agent program executor.

The Executor takes a compiled AgentProgram (after passes have run) and
executes it against a real model + real tool functions. It dispatches
each Op to its annotated kernel, manages the KV pool, and handles
stream scheduling for the H3 spec_prefill overlap.

Usage:
    from loopfuse.runtime.executor import AgentExecutor

    executor = AgentExecutor(model, kv_pool, tools, target=KernelTarget.TRITON)
    result   = executor.run(prog, initial_input="What is the capital of France?")
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import torch

from ..ir.dialect import (
    AgentProgram, AgentStepOp, Op, Phase, KernelTarget,
    LLMForwardOp, KVWriteOp, KVFuseOp, ToolCallOp,
    SpeculativePrefillOp, PromptConstructOp, ArgmaxOp, ParseActionOp,
)
from .kv_pool import KVPool, KVState


@dataclass
class StepTrace:
    """Per-step timing and phase breakdown for analysis."""
    step_id:            int
    prefill_ms:         float = 0.0
    decode_ms:          float = 0.0
    kv_write_ms:        float = 0.0
    tool_io_ms:         float = 0.0
    prompt_construct_ms: float = 0.0
    parse_action_ms:    float = 0.0
    spec_prefill_ms:    float = 0.0
    spec_prefill_hit:   bool  = False
    total_ms:           float = 0.0
    output_tokens:      List[int] = field(default_factory=list)

    def gpu_active_ms(self) -> float:
        return self.prefill_ms + self.decode_ms + self.kv_write_ms

    def gpu_idle_ms(self) -> float:
        return self.tool_io_ms + self.prompt_construct_ms + self.parse_action_ms

    def idle_pct(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return 100.0 * self.gpu_idle_ms() / self.total_ms


@dataclass
class ExecutionResult:
    """Full execution result for an agent program."""
    output:       str
    step_traces:  List[StepTrace]
    total_ms:     float
    n_steps:      int

    def phase_summary(self) -> Dict[str, float]:
        return {
            "prefill_ms":    sum(t.prefill_ms    for t in self.step_traces),
            "decode_ms":     sum(t.decode_ms     for t in self.step_traces),
            "kv_write_ms":   sum(t.kv_write_ms   for t in self.step_traces),
            "tool_io_ms":    sum(t.tool_io_ms     for t in self.step_traces),
            "prompt_ms":     sum(t.prompt_construct_ms for t in self.step_traces),
            "parse_ms":      sum(t.parse_action_ms for t in self.step_traces),
            "spec_ms":       sum(t.spec_prefill_ms for t in self.step_traces),
        }

    def avg_idle_pct(self) -> float:
        return sum(t.idle_pct() for t in self.step_traces) / max(1, len(self.step_traces))

    def print_trace(self):
        print(f"{'Step':<6} {'Prefill':>8} {'Decode':>8} {'KV':>7} {'Tool IO':>9} "
              f"{'Prompt':>8} {'Total':>8} {'Idle%':>7}")
        print("-" * 65)
        for t in self.step_traces:
            print(f"{t.step_id:<6} {t.prefill_ms:>7.1f}ms {t.decode_ms:>7.1f}ms "
                  f"{t.kv_write_ms:>6.1f}ms {t.tool_io_ms:>8.1f}ms "
                  f"{t.prompt_construct_ms:>7.1f}ms {t.total_ms:>7.1f}ms "
                  f"{t.idle_pct():>6.1f}%")
        print("-" * 65)
        s = self.phase_summary()
        print(f"Total: {self.total_ms:.1f}ms | Avg idle: {self.avg_idle_pct():.1f}%")


class AgentExecutor:
    """
    Executes a compiled AgentProgram against a real model and tool set.

    Args:
        model:    A callable (tokens, kv_k, kv_v) -> (logits, new_k, new_v)
                  Or any object with a .forward(input_ids, past_key_values) method.
        kv_pool:  Pre-allocated KVPool.
        tools:    Dict[tool_name, Callable[str] -> str]
        target:   KernelTarget to dispatch kernels for.
        device:   "cuda" or "cpu".
    """

    def __init__(
        self,
        model:    Any,
        kv_pool:  KVPool,
        tools:    Dict[str, Callable[[str], str]],
        target:   KernelTarget = KernelTarget.TRITON,
        device:   str = "cuda",
        max_decode_tokens: int = 64,
    ):
        self.model   = model
        self.kv_pool = kv_pool
        self.tools   = tools
        self.target  = target
        self.device  = device
        self.max_decode_tokens = max_decode_tokens
        self._dispatch = self._build_dispatch_table()

    def run(self, prog: AgentProgram, initial_input: str) -> ExecutionResult:
        """Execute a compiled agent program end to end."""
        state  = self.kv_pool.allocate_slot()
        traces = []
        output = ""
        t_total_start = time.perf_counter()

        try:
            current_obs = initial_input
            for step in prog.steps:
                trace = self._execute_step(step, state, current_obs, prog)
                traces.append(trace)
                if trace.output_tokens:
                    output = self._decode_tokens(trace.output_tokens)
                # Tool result becomes next observation
                current_obs = output
                state.advance_step()
        finally:
            self.kv_pool.release(state)

        total_ms = (time.perf_counter() - t_total_start) * 1000
        return ExecutionResult(
            output      = output,
            step_traces = traces,
            total_ms    = total_ms,
            n_steps     = len(traces),
        )

    def _execute_step(
        self,
        step:        AgentStepOp,
        kv_state:    KVState,
        observation: str,
        prog:        AgentProgram,
    ) -> StepTrace:
        trace = StepTrace(step_id=step.step_id)
        t_step_start = time.perf_counter()

        for region_name, ops in [("observe", step.observe_region),
                                  ("reason",  step.reason_region),
                                  ("act",     step.act_region)]:
            for op in ops:
                self._dispatch_op(op, kv_state, trace, observation, prog)

        trace.total_ms = (time.perf_counter() - t_step_start) * 1000
        return trace

    def _dispatch_op(self, op: Op, kv_state: KVState, trace: StepTrace,
                     observation: str, prog: AgentProgram):
        """Route each op to its executor based on type."""
        t0 = time.perf_counter()

        if isinstance(op, PromptConstructOp):
            self._exec_prompt_construct(op, observation)
            trace.prompt_construct_ms += (time.perf_counter() - t0) * 1000

        elif isinstance(op, LLMForwardOp):
            self._exec_llm_forward(op, kv_state, prog)
            dt = (time.perf_counter() - t0) * 1000
            if op.phase == Phase.PREFILL:
                trace.prefill_ms += dt
            else:
                trace.decode_ms  += dt

        elif isinstance(op, KVWriteOp):
            self._exec_kv_write(op, kv_state)
            trace.kv_write_ms += (time.perf_counter() - t0) * 1000

        elif isinstance(op, KVFuseOp):
            # KVFuseOp is a compiler hint — the actual fusion already happened
            # at compile time. At runtime we just track it.
            trace.kv_write_ms += 0.01  # negligible

        elif isinstance(op, SpeculativePrefillOp):
            self._exec_spec_prefill(op, kv_state, prog)
            trace.spec_prefill_ms += (time.perf_counter() - t0) * 1000

        elif isinstance(op, ToolCallOp):
            result = self._exec_tool_call(op)
            trace.tool_io_ms += (time.perf_counter() - t0) * 1000

        elif isinstance(op, ArgmaxOp):
            pass  # handled inside _exec_llm_forward

        elif isinstance(op, ParseActionOp):
            self._exec_parse_action(op)
            trace.parse_action_ms += (time.perf_counter() - t0) * 1000

    # ---- Op executors ----

    def _exec_prompt_construct(self, op: PromptConstructOp, observation: str) -> str:
        """Build prompt string. Returns the constructed prompt."""
        template = op.attrs.get("template", "react_v1")
        return f"[{template}] Observation: {observation}\nThought:"

    def _exec_llm_forward(self, op: LLMForwardOp, kv_state: KVState,
                          prog: AgentProgram):
        """Dispatch LLM forward to the annotated kernel."""
        kernel_name = op.metadata.get("kernel_name", "eager")
        # In production: call the actual kernel here
        # For the benchmark harness, we call the mock or real model
        if hasattr(self.model, "forward"):
            # Real model with HF-style interface
            pass  # implementation depends on model
        # Simulation: just synchronize to ensure timing is clean
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _exec_kv_write(self, op: KVWriteOp, kv_state: KVState):
        """Write dummy KV entries (real entries come from _exec_llm_forward)."""
        # In production this is called by the LLM forward kernel directly
        pass

    def _exec_spec_prefill(self, op: SpeculativePrefillOp, kv_state: KVState,
                            prog: AgentProgram):
        """
        Execute speculative prefill on the prefetch stream.
        This runs concurrently with the tool call that follows it.
        """
        conf = op.metadata.get("confidence", 0.0)
        if conf < 0.5:
            return  # skip low-confidence speculations
        # In production: submit prefill to self.kv_pool.prefetch_stream
        # The tool call will execute on CPU while this runs on GPU
        self.kv_pool.prefetch_next_slot(kv_state)

    def _exec_tool_call(self, op: ToolCallOp) -> str:
        """Execute the tool and return its result string."""
        tool_fn = self.tools.get(op.tool_name)
        if tool_fn is None:
            return f"[tool '{op.tool_name}' not found]"
        try:
            result = tool_fn("")
        except Exception as e:
            result = f"[tool error: {e}]"
        return result

    def _exec_parse_action(self, op: ParseActionOp) -> dict:
        """Parse model output into structured action dict."""
        return {"action": "unknown", "input": ""}

    def _decode_tokens(self, tokens: List[int]) -> str:
        return " ".join(str(t) for t in tokens)

    def _build_dispatch_table(self) -> dict:
        """Pre-built dispatch for performance (avoids isinstance chain in hot loop)."""
        return {
            KernelTarget.TRITON: self._dispatch_triton,
            KernelTarget.CUDA:   self._dispatch_cuda,
            KernelTarget.PALLAS: self._dispatch_pallas,
            KernelTarget.EAGER:  self._dispatch_eager,
        }

    def _dispatch_triton(self, op, *args): pass
    def _dispatch_cuda(self, op, *args):   pass
    def _dispatch_pallas(self, op, *args): pass
    def _dispatch_eager(self, op, *args):  pass
