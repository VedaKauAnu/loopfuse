"""
LoopFuse: Agent-Loop-Aware Kernel Fusion and Compiler Optimization for LLM Inference.

Quick start:
    from loopfuse.ir import AgentProgram, AgentProgramBuilder, IRPrinter
    from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
    from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass
    from loopfuse.analysis import RooflineAnalyzer, A100_SPEC

    prog = (AgentProgramBuilder("react_demo", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, tool_name="search",     tool_latency_ms=150, is_first_step=True)
        .add_react_step(1, tool_name="lookup",     tool_latency_ms=80)
        .add_react_step(2, tool_name=None)
        .set_target(KernelTarget.TRITON)
        .build())

    PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()]).run(prog)

    print(IRPrinter(use_color=True).print(prog))
"""

__version__ = "0.1.0"
