"""
loopfuse/ir/printer.py

Human-readable IR dump for AgentProgram.

Output format mirrors MLIR textual IR but stays Python-native.
Compiler-inserted ops are flagged with ★.
Phase and target annotations are shown inline.

Example output:
  // AgentProgram 'react_hotpotqa' target=triton
  // KV: kvcache<L=12,H=12,d=64,seq=1024,fp16>

  agent.step[0] {
    // --- observe ---
    %prompt_construct_... : tokens<len=2048,vocab=50257> = prompt.construct(...)  // [io]
    // --- reason ---
    %llm_forward_... : tensor<128x50257xbf16> = llm.forward(...)  // [prefill]
    %kv_write_... = kv.write(...)  // [io]
    %argmax_... = argmax(...)  // [decode]
    // --- act ---
    %parse_action_... = parse.action(...)  // [io]
    %tool_call_... = tool.call(tool=search, est_latency_ms=150.0)  // [io]
  }

  // Pass annotations:
  //   [KVFusionPass] fused steps [0,1,2] prefix_len=64 saved 49152 HBM bytes
  //   [SpecPrefillPass] inserted spec_prefill in step 0 (confidence=0.85)
"""

from __future__ import annotations
from typing import Optional
from .dialect import (
    AgentProgram, AgentStepOp, Op, Phase,
    LLMForwardOp, KVWriteOp, KVFuseOp, ToolCallOp,
    SpeculativePrefillOp, PromptConstructOp, ArgmaxOp, ParseActionOp,
)


PHASE_COLORS = {
    Phase.PREFILL: "\033[92m",  # green
    Phase.DECODE:  "\033[94m",  # blue
    Phase.IO:      "\033[93m",  # yellow
    Phase.FUSE:    "\033[95m",  # magenta
}
RESET = "\033[0m"
COMPILER_STAR = "\033[95m★\033[0m"


class IRPrinter:
    """
    Prints an AgentProgram to a readable string.

    printer = IRPrinter(use_color=True)
    print(printer.print(program))
    """

    def __init__(self, use_color: bool = False, indent: str = "  "):
        self.use_color = use_color
        self.indent = indent

    def print(self, prog: AgentProgram) -> str:
        lines = []
        tgt = prog.target.value if prog.target else "unset"
        lines.append(f"// AgentProgram '{prog.name}'  target={tgt}")
        lines.append(f"// KV: {prog.kv_state.type}")
        lines.append(f"// Steps: {len(prog.steps)}")

        # Summary counts
        by_phase = prog.ops_by_phase()
        phase_summary = "  ".join(
            f"{p.value}={len(ops)}"
            for p, ops in by_phase.items() if ops
        )
        lines.append(f"// Phases: {phase_summary}")
        ci = prog.compiler_inserted_ops()
        if ci:
            lines.append(f"// Compiler-inserted ops: {len(ci)} ★")
        lines.append("")

        for step in prog.steps:
            lines.extend(self._print_step(step))
            lines.append("")

        if prog.pass_annotations:
            lines.append("// Pass annotations:")
            for ann in prog.pass_annotations:
                lines.append(f"//   {ann}")

        return "\n".join(lines)

    def _print_step(self, step: AgentStepOp) -> list:
        lines = [f"agent.step[{step.step_id}] {{"]
        idle_ms = step.total_gpu_idle_ms()
        if idle_ms > 0:
            lines.append(f"  // GPU-idle window: {idle_ms:.0f}ms")

        regions = [
            ("observe", step.observe_region),
            ("reason",  step.reason_region),
            ("act",     step.act_region),
        ]
        for region_name, ops in regions:
            if ops:
                lines.append(f"  // --- {region_name} ---")
                for op in ops:
                    lines.append(self._print_op(op, prefix=self.indent))

        lines.append("}")
        return lines

    def _print_op(self, op: Op, prefix: str = "  ") -> str:
        # Result names (short)
        if op.results:
            res_parts = [f"%{r.name[:20]}:{r.type}" for r in op.results]
            res_str = ", ".join(res_parts)
        else:
            res_str = "_"

        # Operand names
        op_str = ", ".join(f"%{o.name[:16]}" for o in op.operands)

        # Key attrs (selective, not full dump)
        key_attrs = self._key_attrs(op)
        args = ", ".join(p for p in [op_str, key_attrs] if p)

        phase_str = f"[{op.phase.value}]" if op.phase else ""
        ci_str    = " ★" if op.is_compiler_inserted() else ""

        if self.use_color and op.phase and op.phase in PHASE_COLORS:
            c    = PHASE_COLORS[op.phase]
            line = f"{prefix}{res_str} = {c}{op.name}{RESET}({args})  // {phase_str}{ci_str}"
        else:
            line = f"{prefix}{res_str} = {op.name}({args})"
            if phase_str:
                line += f"  // {phase_str}{ci_str}"

        # Extra metadata for compiler-inserted ops
        if op.is_compiler_inserted():
            if isinstance(op, KVFuseOp):
                saved = op.bytes_saved()
                line += f"  [prefix={op.attrs['prefix_len']} tokens, saved={saved//1024}KB]"
            elif isinstance(op, SpeculativePrefillOp):
                conf   = op.metadata.get("confidence", "?")
                window = op.metadata.get("available_window_ms", "?")
                line  += f"  [confidence={conf:.2f}, window={window}ms]"

        return line

    def _key_attrs(self, op: Op) -> str:
        """Extract the most informative attrs without flooding the output."""
        if isinstance(op, LLMForwardOp):
            return f"model={op.model_config.get('model_name','?')}, seq={op.seq_len}"
        if isinstance(op, ToolCallOp):
            return f"tool={op.tool_name}, lat={op.attrs['est_latency_ms']:.0f}ms"
        if isinstance(op, KVWriteOp):
            return f"step={op.attrs['step_id']}, pos={op.attrs['seq_pos']}"
        if isinstance(op, KVFuseOp):
            return f"steps={op.attrs['steps']}, prefix={op.attrs['prefix_len']}"
        if isinstance(op, SpeculativePrefillOp):
            return f"conf={op.metadata.get('confidence',0):.2f}"
        if isinstance(op, PromptConstructOp):
            return f"tmpl={op.attrs['template']}"
        return ""

    def print_summary_table(self, prog: AgentProgram) -> str:
        """Print a compact phase breakdown table."""
        by_phase = prog.ops_by_phase()
        total_ops = len(prog.all_ops())
        lines = [
            f"{'Phase':<12}{'Ops':>6}{'%':>8}{'Est. FLOP':>14}{'Est. Bytes':>14}",
            "-" * 55,
        ]
        for phase in Phase:
            ops = by_phase[phase]
            if not ops:
                continue
            pct   = 100 * len(ops) / total_ops if total_ops else 0
            flops = sum(o.estimated_flops() for o in ops)
            btes  = sum(o.estimated_bytes() for o in ops)
            lines.append(
                f"{phase.value:<12}{len(ops):>6}{pct:>7.1f}%"
                f"{flops:>14,}{btes:>14,}"
            )
        lines.append("-" * 55)
        ci = prog.compiler_inserted_ops()
        lines.append(f"Compiler-inserted ops: {len(ci)}")
        if prog.pass_annotations:
            lines.append(f"Passes run: {len(prog.pass_annotations)}")
        return "\n".join(lines)
