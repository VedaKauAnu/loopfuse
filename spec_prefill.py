"""
loopfuse/passes/spec_prefill.py

Speculative Prefill Insertion Pass — H3 optimization.

What it does:
  Scans each AgentStepOp for ToolCallOps (Phase.IO, GPU-idle).
  If the tool's estimated idle window exceeds the cost of prefilling
  the next step's prompt, inserts a SpeculativePrefillOp into the
  idle window — overlapping GPU prefill compute with tool I/O.

Why this beats vLLM sequential:
  vLLM (and every other system) waits for the tool result before
  starting the next LLM call. We start the next step's prefill
  *during* the tool call, hiding the prefill cost behind I/O latency.
  This is only possible because our IR spans multiple agent steps.

Metric:
  Speedup = (tool_latency_ms - overlap_ms) / tool_latency_ms
  End-to-end step latency p50/p99 on A100 (mock tool delays: 50-500ms)

Safety:
  The speculative prefill is only beneficial if the tool result
  doesn't invalidate the speculative prompt. We track confidence
  (set heuristically — production would use a learned predictor).
  If confidence < threshold, the pass is conservative and skips.
"""

from __future__ import annotations
import time
from typing import List, Optional
from .base import Pass, PassResult
from ..ir.dialect import (
    AgentProgram, AgentStepOp, ToolCallOp, SpeculativePrefillOp,
    LLMForwardOp, Phase, Value, TokenSeqType,
)


class SpecPrefillPass(Pass):
    """
    Inserts speculative prefill ops into tool-call idle windows.

    Configuration:
      min_idle_window_ms:   only insert if tool latency > this (default: 30ms)
      confidence_threshold: only insert if confidence > this (default: 0.5)
      estimated_prefill_ms: assumed cost of one prefill step (default: 15ms)
    """

    def __init__(
        self,
        min_idle_window_ms:   float = 30.0,
        confidence_threshold: float = 0.50,
        estimated_prefill_ms: float = 15.0,
    ):
        self.min_idle_window_ms   = min_idle_window_ms
        self.confidence_threshold = confidence_threshold
        self.estimated_prefill_ms = estimated_prefill_ms

    def run(self, prog: AgentProgram) -> PassResult:
        start = time.time()
        annotations: List[str] = []
        changed = False

        for i, step in enumerate(prog.steps):
            # Look for tool calls with sufficient idle windows
            for tool_op in step.tool_ops():
                idle_ms = tool_op.idle_window_ms()

                if idle_ms < self.min_idle_window_ms:
                    continue

                # Idempotency guard: skip if already has a SpeculativePrefillOp
                # immediately before this tool call in the act_region
                from ..ir.dialect import SpeculativePrefillOp as _SP
                act = step.act_region
                tool_idx = act.index(tool_op) if tool_op in act else -1
                if tool_idx > 0 and isinstance(act[tool_idx - 1], _SP):
                    continue

                # Compute confidence that the next step's prompt doesn't
                # depend on this tool's result in a way that invalidates
                # the speculative prefill. In a ReAct loop, the THOUGHT
                # portion is largely determined before the tool result
                # arrives — only the OBSERVATION prefix changes.
                confidence = self._estimate_confidence(step, tool_op, prog)

                if confidence < self.confidence_threshold:
                    continue

                # Can we fit a prefill in the available window?
                available_ms = idle_ms - self.estimated_prefill_ms * 0.2  # 20% margin
                if available_ms < self.estimated_prefill_ms:
                    continue

                # Build the speculative prefill op
                dummy_tokens = Value(
                    f"spec_toks_step{step.step_id}",
                    TokenSeqType(64, prog.model_config["vocab_size"]),
                )
                spec_op = SpeculativePrefillOp(
                    speculative_tokens  = dummy_tokens,
                    kv_state            = prog.kv_state,
                    model_config        = prog.model_config,
                    confidence          = confidence,
                    overlapping_tool    = tool_op,
                )

                # Insert spec_prefill immediately before the tool call
                # so the scheduler knows to overlap them
                inserted = step.insert_before(tool_op, spec_op)
                if inserted:
                    changed = True
                    overlap_pct = min(100, self.estimated_prefill_ms / idle_ms * 100)
                    ann = (
                        f"inserted spec_prefill in step {step.step_id} "
                        f"(confidence={confidence:.2f}, "
                        f"window={idle_ms:.0f}ms, "
                        f"est_overlap={overlap_pct:.0f}%)"
                    )
                    annotations.append(ann)
                    # Only insert one spec_prefill per tool call
                    break

        return self._result(prog, changed, annotations, start)

    def _estimate_confidence(
        self,
        step: AgentStepOp,
        tool_op: ToolCallOp,
        prog: AgentProgram,
    ) -> float:
        """
        Heuristic confidence that speculative prefill is valid.

        In a well-formed ReAct loop:
        - The system prompt and few-shot examples are fixed (high confidence).
        - The Thought prefix is partially predictable (medium confidence).
        - The Observation section depends on the tool result (low confidence).

        We use tool type and loop position as a proxy:
        - search/lookup tools: predictable structure, higher confidence
        - first step: highest confidence (system prompt dominates)
        - later steps: lower (more context-dependent)
        """
        base = 0.85

        # Later steps have more dynamic context
        base -= step.step_id * 0.05

        # Tool type heuristic
        tool_name = tool_op.tool_name.lower()
        if "search" in tool_name or "lookup" in tool_name:
            base += 0.05
        elif "execute" in tool_name or "run" in tool_name:
            base -= 0.15

        # Clamp
        return max(0.0, min(1.0, base))
