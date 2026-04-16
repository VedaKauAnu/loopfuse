"""
Notebook 05: Phase-Specialized Kernels + Roofline (H4).
Requires: A100 (Triton autotune optimal on Ampere).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import torch
if not torch.cuda.is_available():
    print("H4 requires CUDA. Run this notebook on Colab with a GPU runtime.")
else:
    from loopfuse.benchmarks.h4_phase_kernels.run import run
    from loopfuse.analysis.roofline import RooflineAnalyzer, A100_SPEC, T4_SPEC
    from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
    from loopfuse.ir.builder import AgentProgramBuilder
    from loopfuse.passes import PassManager, PhaseSelectPass

    results = run()

    # Roofline plot
    prog = (AgentProgramBuilder("roofline_demo", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search", 150.0, is_first_step=True)
        .add_react_step(1, None)
        .set_target(KernelTarget.TRITON)
        .build())
    PassManager([PhaseSelectPass()]).run(prog)

    hw = A100_SPEC if "A100" in torch.cuda.get_device_name(0) else T4_SPEC
    analyzer = RooflineAnalyzer(hw)
    report   = analyzer.analyze_program(prog)
    fig = analyzer.plot(report)
    if fig:
        fig.savefig("h4_roofline.png", dpi=150, bbox_inches="tight")
        print("Roofline saved to h4_roofline.png")
