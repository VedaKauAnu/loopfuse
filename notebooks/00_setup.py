"""Notebook 00: Environment Setup + Hardware Check"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

def detect_hardware():
    report = {}
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            cc   = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            report["gpu"] = {"available": True, "name": name, "memory_gb": round(mem,1),
                             "compute_cap": cc,
                             "tier":   "A100" if "A100" in name else ("T4" if "T4" in name else name),
                             "target": "triton + cuda" if "A100" in name else "triton"}
        else:
            report["gpu"] = {"available": False}
    except ImportError:
        report["gpu"] = {"available": False}
    try:
        import triton
        report["triton"] = {"available": True, "version": triton.__version__}
    except ImportError:
        report["triton"] = {"available": False}
    return report


if __name__ == "__main__":
    hw = detect_hardware()
    print("=" * 55)
    print("  LoopFuse Hardware Check")
    print("=" * 55)
    gpu = hw.get("gpu", {})
    if gpu.get("available"):
        print(f"  GPU:    {gpu['name']}")
        print(f"  Memory: {gpu['memory_gb']} GB  (SM {gpu['compute_cap']})")
        print(f"  Tier:   {gpu.get('tier','?')}  ->  target={gpu.get('target','?')}")
    else:
        print("  GPU:    NOT AVAILABLE (CPU-only mode)")
    tri = hw.get("triton", {})
    print(f"  Triton: {tri.get('version','NOT INSTALLED')}")
    print("=" * 55)

    # IR smoke test — CPU only, no GPU required
    from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
    from loopfuse.ir.builder import AgentProgramBuilder
    from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass

    print("\n[Smoke test] Building 3-step ReAct program...")
    prog = (AgentProgramBuilder("smoke_test", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search", 150.0, is_first_step=True)
        .add_react_step(1, "lookup",  80.0)
        .add_react_step(2, None)
        .set_target(KernelTarget.TRITON).build())
    print(f"  Steps: {len(prog.steps)}  Ops: {len(prog.all_ops())}")

    results = PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()]).run(prog)
    for r in results:
        print(f"  [{'✓' if r.changed else '-'}] {r.pass_name}  {r.duration_ms:.2f}ms")

    ci   = len(prog.compiler_inserted_ops())
    idle = prog.total_gpu_idle_ms()
    print(f"  Compiler-inserted ops: {ci}")
    print(f"  GPU idle window:       {idle:.0f}ms  (H3 opportunity)")
    print("\n[Smoke test] PASSED ✓  —  Ready for Notebook 01")
