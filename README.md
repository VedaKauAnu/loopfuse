# LoopFuse

**Agent-Loop-Aware Kernel Fusion and Compiler Optimization for LLM Inference**

> *The first compiler IR that treats the agent loop — not the single forward pass — as the unit of compilation.*

## What is LoopFuse?

LoopFuse is an academic research project demonstrating that standard LLM inference runtimes (vLLM, TensorRT-LLM, SGLang) leave significant performance on the table for agentic workloads because they optimize *individual forward passes*, not the multi-step agent loop that orchestrates them.

LoopFuse introduces:
1. **An Agent IR** — a lightweight SSA-style IR where `AgentStepOp` is the primitive, not a single GEMM
2. **Cross-step compiler passes** — KV fusion (H2), speculative prefill insertion (H3), phase-specialized kernel selection (H4)
3. **Per-phase Triton kernels** — separate compute-optimized (prefill) and bandwidth-optimized (decode) attention kernels
4. **Agent-specific roofline analysis** — first profiling methodology for per-phase agent execution

## Hardware Targets

| Component | T4 (free) | A100 (Pro+) | TPU v4 (Pro+) |
|---|---|---|---|
| IR + passes | ✅ | ✅ | ✅ |
| Triton kernels | ✅ | ✅ | — |
| Raw CUDA kernels | — | ✅ | — |
| JAX/Pallas kernels | — | — | ✅ |

## Colab Notebooks

| Notebook | Hardware | Hypothesis |
|---|---|---|
| `00_setup.py` | Any | Environment check |
| `01_phase_profiling.py` | T4/A100 | H1: Phase waste > 30% |

## Project Structure

```
loopfuse/
├── ir/            # Agent IR dialect (core novelty)
├── passes/        # Optimization passes (KV fusion, spec prefill, phase select)
├── kernels/       # Triton/CUDA/Pallas kernels
├── analysis/      # Roofline analysis + statistical tools
└── notebooks/     # Colab-runnable experiment notebooks
```

## Quick Start

```python
from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.ir.printer import IRPrinter
from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass

prog = (AgentProgramBuilder("react_demo", GPT2_CONFIG, GPT2_KV_CONFIG)
    .add_react_step(0, "search", 150.0, is_first_step=True)
    .add_react_step(1, "lookup",  80.0)
    .add_react_step(2, None)
    .set_target(KernelTarget.TRITON)
    .build())

PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()]).run(prog)
print(IRPrinter(use_color=True).print(prog))
```

## Research Hypotheses

- **H1**: Agent loops waste >30% of wall-clock on non-GPU work (measured in `01_phase_profiling.py`)
- **H2**: Compiler-driven KV prefix fusion reduces HBM bandwidth vs. SGLang RadixAttention
- **H3**: Inserting speculative prefill into tool I/O idle windows reduces step latency vs. vLLM
- **H4**: Phase-specialized attention kernels improve MFU vs. torch.compile default
- **H5**: One LoopFuse IR program lowers to T4/A100/TPU, each outperforming platform torch.compile baseline

## Citation

```bibtex
@misc{loopfuse2025,
  title  = {LoopFuse: Agent-Loop-Aware Kernel Fusion for LLM Inference},
  year   = {2025},
  note   = {https://github.com/your-org/loopfuse}
}
```
