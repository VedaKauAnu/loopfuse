"""
loopfuse/passes/phase_select.py

Phase-Specialized Kernel Selection Pass — H4 optimization.

What it does:
  Annotates each LLMForwardOp with the optimal KernelTarget for its phase
  and hardware target. PREFILL ops get compute-optimized kernels (large
  tile GEMM). DECODE ops get memory-optimized kernels (streaming loads,
  small GEMM with register reuse).

  This is the pass that *selects* which kernel the runtime will dispatch.
  The actual kernels live in loopfuse/kernels/.

Why one general kernel suboptimizes both phases:
  - Decode attention: arithmetic intensity ~1-5 FLOP/byte (memory-bound).
    Optimal: maximize HBM streaming bandwidth. Tile size: small (16x16).
  - Prefill attention: arithmetic intensity ~100+ FLOP/byte (compute-bound).
    Optimal: maximize tensor core utilization. Tile size: large (128x128).
  torch.compile and vLLM use a single FlashAttention kernel for both,
  which makes a compromise and is optimal for neither at small batch sizes.

Metric:
  MFU (Model FLOP Utilization) per phase, measured via Nsight Compute:
    smsp__sass_thread_inst_executed_op_hmma_pred_on.sum
  vs. torch.compile default kernel
"""

from __future__ import annotations
import time
from typing import List
from .base import Pass, PassResult
from ..ir.dialect import (
    AgentProgram, LLMForwardOp, Phase, KernelTarget,
)

# Roofline constants — A100 SXM4
A100_PEAK_FLOPS_BF16 = 312e12   # 312 TFLOPS (tensor cores)
A100_HBM_BANDWIDTH   = 2.0e12   # 2 TB/s
A100_RIDGE_POINT     = A100_PEAK_FLOPS_BF16 / A100_HBM_BANDWIDTH  # ~156 FLOP/byte

# T4
T4_PEAK_FLOPS_FP16 = 65e12
T4_HBM_BANDWIDTH   = 320e9
T4_RIDGE_POINT     = T4_PEAK_FLOPS_FP16 / T4_HBM_BANDWIDTH  # ~203 FLOP/byte


PHASE_KERNEL_MAP = {
    # (hardware_target, phase) -> (kernel_module, kernel_name, notes)
    (KernelTarget.TRITON, Phase.PREFILL): (
        "kernels.triton.prefill_attn",
        "prefill_attention_fwd",
        "large-tile GEMM, block=128x128, compute-optimized",
    ),
    (KernelTarget.TRITON, Phase.DECODE): (
        "kernels.triton.decode_attn",
        "decode_attention_fwd",
        "streaming load, block=16x64, bandwidth-optimized",
    ),
    (KernelTarget.CUDA, Phase.PREFILL): (
        "kernels.cuda.async_prefill",
        "agent_prefill_kernel",
        "async memcpy pipeline, double-buffered, warp-specialized",
    ),
    (KernelTarget.CUDA, Phase.DECODE): (
        "kernels.cuda.ring_kv_cache",
        "ring_kv_decode_kernel",
        "ring-buffer KV, async prefetch, tensor core MMA",
    ),
    (KernelTarget.PALLAS, Phase.PREFILL): (
        "kernels.pallas.agent_attn",
        "pallas_prefill_fwd",
        "TPU VMEM-resident QK^T, column-major K layout",
    ),
    (KernelTarget.PALLAS, Phase.DECODE): (
        "kernels.pallas.agent_attn",
        "pallas_decode_fwd",
        "TPU streaming decode, HBM bandwidth optimized",
    ),
    (KernelTarget.EAGER, Phase.PREFILL): (
        "torch.nn.functional",
        "scaled_dot_product_attention",
        "PyTorch eager baseline",
    ),
    (KernelTarget.EAGER, Phase.DECODE): (
        "torch.nn.functional",
        "scaled_dot_product_attention",
        "PyTorch eager baseline (same kernel — the suboptimality we measure)",
    ),
}


class PhaseSelectPass(Pass):
    """
    Annotates LLMForwardOps with their optimal kernel target.

    After this pass, each LLMForwardOp carries:
        op.metadata["kernel_module"]   -- importable module path
        op.metadata["kernel_name"]     -- function to call
        op.metadata["kernel_notes"]    -- human-readable explanation
        op.metadata["is_compute_bound"]-- True if above ridge point
        op.metadata["roofline_efficiency"] -- estimated efficiency %
    """

    def run(self, prog: AgentProgram) -> PassResult:
        start = time.time()
        annotations: List[str] = []
        changed = False

        target = prog.target or KernelTarget.EAGER
        llm_ops = prog.ops_by_type(LLMForwardOp)

        prefill_count = decode_count = 0

        for op in llm_ops:
            assert isinstance(op, LLMForwardOp)
            phase = op.phase or Phase.DECODE

            key = (target, phase)
            if key not in PHASE_KERNEL_MAP:
                key = (KernelTarget.EAGER, phase)

            module, fn, notes = PHASE_KERNEL_MAP[key]

            op.metadata["kernel_module"] = module
            op.metadata["kernel_name"]   = fn
            op.metadata["kernel_notes"]  = notes
            op.target = target

            # Roofline classification
            ai = op.arithmetic_intensity()
            ridge = self._ridge_point(target)
            is_compute_bound = ai > ridge
            op.metadata["is_compute_bound"] = is_compute_bound
            op.metadata["arithmetic_intensity"] = round(ai, 2)

            # Rough efficiency estimate
            if is_compute_bound:
                peak = self._peak_flops(target)
                eff  = min(1.0, op.estimated_flops() / (peak * 1e-3))  # very rough
            else:
                peak_bw = self._peak_bandwidth(target)
                eff     = min(1.0, op.estimated_bytes() / (peak_bw * 1e-3))
            op.metadata["roofline_efficiency"] = round(eff, 3)

            if phase == Phase.PREFILL:
                prefill_count += 1
            else:
                decode_count += 1
            changed = True

        ann = (f"annotated {prefill_count} PREFILL ops + {decode_count} DECODE ops "
               f"for target={target.value}")
        annotations.append(ann)

        return self._result(prog, changed, annotations, start)

    def _ridge_point(self, target: KernelTarget) -> float:
        return {
            KernelTarget.CUDA:   A100_RIDGE_POINT,
            KernelTarget.TRITON: A100_RIDGE_POINT,
            KernelTarget.PALLAS: 200.0,  # TPU v4 approximate
            KernelTarget.EAGER:  A100_RIDGE_POINT,
        }.get(target, A100_RIDGE_POINT)

    def _peak_flops(self, target: KernelTarget) -> float:
        return {
            KernelTarget.CUDA:   A100_PEAK_FLOPS_BF16,
            KernelTarget.TRITON: A100_PEAK_FLOPS_BF16,
            KernelTarget.PALLAS: 275e12,  # TPU v4
            KernelTarget.EAGER:  A100_PEAK_FLOPS_BF16,
        }.get(target, A100_PEAK_FLOPS_BF16)

    def _peak_bandwidth(self, target: KernelTarget) -> float:
        return {
            KernelTarget.CUDA:   A100_HBM_BANDWIDTH,
            KernelTarget.TRITON: A100_HBM_BANDWIDTH,
            KernelTarget.PALLAS: 1.2e12,  # TPU v4 HBM
            KernelTarget.EAGER:  A100_HBM_BANDWIDTH,
        }.get(target, A100_HBM_BANDWIDTH)
