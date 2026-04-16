"""
loopfuse/ir/builder.py

Fluent builder API for constructing AgentProgram instances.

Usage:
    prog = (AgentProgramBuilder("react_hotpotqa", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(step_id=0, tool_name="search", tool_latency_ms=150.0,
                        is_first_step=True, system_prompt_len=128)
        .add_react_step(step_id=1, tool_name="lookup", tool_latency_ms=80.0)
        .add_react_step(step_id=2, tool_name=None)   # final answer, no tool
        .build())
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from .dialect import (
    AgentProgram, AgentStepOp, KernelTarget, Phase,
    LLMForwardOp, KVWriteOp, ToolCallOp, PromptConstructOp,
    ArgmaxOp, ParseActionOp, Value, TensorType, TokenSeqType,
)


class AgentProgramBuilder:
    """
    Fluent builder for AgentProgram.

    Provides high-level helpers for common agent patterns (ReAct)
    while still allowing low-level op insertion for custom programs.
    """

    def __init__(self, name: str, model_config: Dict[str, Any],
                 kv_cache_config: Dict[str, Any]):
        self._prog = AgentProgram(name, model_config, kv_cache_config)
        self._model_config = model_config
        self._kv_config = kv_cache_config

    # ---- High-level helpers ----

    def add_react_step(
        self,
        step_id: int,
        tool_name: Optional[str] = "search",
        tool_latency_ms: float = 100.0,
        is_first_step: bool = False,
        system_prompt_len: int = 64,
    ) -> "AgentProgramBuilder":
        """
        Add one complete ReAct step (Observe → Reason → Act) to the program.

        is_first_step: if True, uses Phase.PREFILL for the LLM forward pass
                       (system prompt + user query = long context).
                       Otherwise uses Phase.DECODE (single-token generation).
        """
        cfg = self._model_config
        kv  = self._prog.kv_state

        # -- OBSERVE region --
        # Dummy input values (real values come from the runtime)
        obs_tokens = Value(f"obs_{step_id}",  TokenSeqType(512, cfg["vocab_size"]))
        mem_tokens = Value(f"mem_{step_id}",  TokenSeqType(512, cfg["vocab_size"]))
        prompt_op  = PromptConstructOp(obs_tokens, mem_tokens, template="react_v1")

        # -- REASON region --
        prompt_out = prompt_op.result()
        phase      = Phase.PREFILL if is_first_step else Phase.DECODE
        seq_len    = system_prompt_len if is_first_step else 1
        llm_op     = LLMForwardOp(prompt_out, kv, cfg, phase=phase, seq_len=seq_len)

        # KV write after forward pass
        dummy_k  = Value(f"keys_{step_id}",   TensorType(
            (cfg["num_layers"], cfg["num_heads"], seq_len, cfg["head_dim"]), "fp16"))
        dummy_v  = Value(f"vals_{step_id}",   TensorType(
            (cfg["num_layers"], cfg["num_heads"], seq_len, cfg["head_dim"]), "fp16"))
        # seq_pos advances with each step
        seq_pos  = step_id * seq_len
        kv_write = KVWriteOp(kv, dummy_k, dummy_v, seq_pos=seq_pos, step_id=step_id)

        argmax_op = ArgmaxOp(llm_op.result())

        # -- ACT region --
        action_tokens = Value(f"action_toks_{step_id}", TokenSeqType(64, cfg["vocab_size"]))
        parse_op      = ParseActionOp(action_tokens, action_schema="react_action_v1")

        act_ops = [parse_op]
        if tool_name is not None:
            tool_input = parse_op.result()
            tool_op    = ToolCallOp(tool_name, tool_input,
                                    estimated_latency_ms=tool_latency_ms,
                                    step_id=step_id)
            act_ops.append(tool_op)

        step = AgentStepOp(
            step_id      = step_id,
            observe_ops  = [prompt_op],
            reason_ops   = [llm_op, kv_write, argmax_op],
            act_ops      = act_ops,
        )
        self._prog.add_step(step)
        return self

    def set_target(self, target: KernelTarget) -> "AgentProgramBuilder":
        self._prog.set_target(target)
        return self

    def build(self) -> AgentProgram:
        return self._prog

    # ---- Low-level helpers ----

    def add_custom_step(self, step: AgentStepOp) -> "AgentProgramBuilder":
        self._prog.add_step(step)
        return self
