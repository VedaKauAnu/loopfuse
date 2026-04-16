"""
Notebook 02: LoopFuse IR Demo — build, pass, and print an agent program.

This notebook demonstrates the Agent IR pipeline end-to-end on CPU.
No GPU required. Run this to understand what the compiler is doing
before running the GPU benchmarks in notebooks 03-06.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget, Phase
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.ir.printer import IRPrinter
from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass
from loopfuse.analysis.roofline import RooflineAnalyzer, A100_SPEC


def demo_ir_pipeline():
    print("=" * 60)
    print("  LoopFuse IR Demo")
    print("=" * 60)

    # --- 1. Build the agent program ---
    print("\n[Step 1] Build 3-step ReAct program (HotpotQA + 2 tool calls)")
    prog = (AgentProgramBuilder("react_hotpotqa", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search",  150.0, is_first_step=True, system_prompt_len=64)
        .add_react_step(1, "lookup",   80.0)
        .add_react_step(2, None)   # final answer, no tool
        .set_target(KernelTarget.TRITON)
        .build())

    print(f"  Steps: {len(prog.steps)}")
    print(f"  Ops:   {len(prog.all_ops())}")
    print(f"  GPU idle: {prog.total_gpu_idle_ms():.0f}ms (H3 opportunity)")

    by_phase = prog.ops_by_phase()
    for p in Phase:
        ops = by_phase[p]
        if ops:
            print(f"  {p.value:10s}: {len(ops)} ops")

    # --- 2. Print pre-pass IR ---
    print("\n[Step 2] Pre-pass IR dump:")
    printer = IRPrinter(use_color=False)
    print(printer.print(prog))

    # --- 3. Run passes ---
    print("[Step 3] Running passes...")
    results = PassManager([
        KVFusionPass(min_prefix_len=4),
        SpecPrefillPass(min_idle_window_ms=50.0, confidence_threshold=0.6),
        PhaseSelectPass(),
    ]).run(prog)

    for r in results:
        status = "✓ changed" if r.changed else "- no-op"
        print(f"  [{status}] {r.pass_name}  ({r.duration_ms:.2f}ms)")

    ci = prog.compiler_inserted_ops()
    print(f"\n  Compiler-inserted ops: {len(ci)}")
    for op in ci:
        print(f"    {op}")

    # --- 4. Print post-pass IR ---
    print("\n[Step 4] Post-pass IR dump:")
    print(printer.print(prog))

    # --- 5. Phase summary table ---
    print("[Step 5] Phase summary table:")
    print(printer.print_summary_table(prog))

    # --- 6. Roofline analysis ---
    print("\n[Step 6] Roofline analysis (A100 spec):")
    analyzer = RooflineAnalyzer(A100_SPEC)
    report   = analyzer.analyze_program(prog)
    print(report.phase_breakdown_str())

    # --- 7. Kernel annotations ---
    from loopfuse.ir.dialect import LLMForwardOp
    print("[Step 7] Kernel annotations (from PhaseSelectPass):")
    for op in prog.ops_by_type(LLMForwardOp):
        km = op.metadata.get("kernel_module", "?")
        kn = op.metadata.get("kernel_name", "?")
        ai = op.metadata.get("arithmetic_intensity", 0)
        cb = "compute-bound" if op.metadata.get("is_compute_bound") else "memory-bound"
        print(f"  [{op.phase.value:8s}] {km}.{kn}")
        print(f"            AI={ai:.1f} FLOP/byte  ({cb})")


if __name__ == "__main__":
    demo_ir_pipeline()
    print("\nNotebook 02 DONE. Next: 03_kv_fusion.py (requires T4/A100)")
