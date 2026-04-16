"""
Notebook 06: Cross-Hardware Portability (H5) — end-to-end.
Runs the same IR on whatever hardware is available.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from loopfuse.benchmarks.h5_cross_hardware.run import run
from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.ir.printer import IRPrinter
from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass

import torch
results = run()

print("\n=== Final compiled IR (TRITON target) ===")
prog = (AgentProgramBuilder("final_demo", GPT2_CONFIG, GPT2_KV_CONFIG)
    .add_react_step(0, "search", 150.0, is_first_step=True)
    .add_react_step(1, "lookup",  80.0)
    .add_react_step(2, None)
    .set_target(KernelTarget.TRITON if torch.cuda.is_available() else KernelTarget.EAGER)
    .build())
PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()]).run(prog)
print(IRPrinter(use_color=False).print(prog))
print("\nH5 DONE. All notebooks complete.")
