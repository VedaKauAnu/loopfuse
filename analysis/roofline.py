"""
loopfuse/analysis/roofline.py

Roofline analysis for LoopFuse agent programs.

Extends classical roofline to agent workloads:
  - Per-phase roofline (PREFILL vs DECODE have different positions)
  - Cross-step bandwidth accounting (the H1 measurement)
  - Plots: matplotlib-based, Colab-ready

Usage in notebooks:
    from loopfuse.analysis.roofline import RooflineAnalyzer, HardwareSpec, A100_SPEC

    analyzer = RooflineAnalyzer(A100_SPEC)
    report = analyzer.analyze_program(prog)
    fig = analyzer.plot(report)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..ir.dialect import AgentProgram, LLMForwardOp, Phase


# ---------------------------------------------------------------------------
# Hardware specifications
# ---------------------------------------------------------------------------

@dataclass
class HardwareSpec:
    name:             str
    peak_flops:       float  # FLOP/s (BF16/FP16 tensor cores)
    hbm_bandwidth:    float  # bytes/s
    l2_cache_bytes:   int
    shared_mem_bytes: int    # per SM
    compute_cap:      float  # CUDA compute capability (0 for TPU)

    @property
    def ridge_point(self) -> float:
        """FLOP/byte at which compute-bound and memory-bound meet."""
        return self.peak_flops / self.hbm_bandwidth


A100_SPEC = HardwareSpec(
    name             = "NVIDIA A100 SXM4 80GB",
    peak_flops       = 312e12,   # BF16 tensor cores
    hbm_bandwidth    = 2.0e12,   # HBM2e
    l2_cache_bytes   = 40 * 1024 * 1024,
    shared_mem_bytes = 192 * 1024,
    compute_cap      = 8.0,
)

T4_SPEC = HardwareSpec(
    name             = "NVIDIA T4",
    peak_flops       = 65e12,    # FP16 tensor cores
    hbm_bandwidth    = 320e9,
    l2_cache_bytes   = 4 * 1024 * 1024,
    shared_mem_bytes = 64 * 1024,
    compute_cap      = 7.5,
)

TPU_V4_SPEC = HardwareSpec(
    name             = "Google TPU v4",
    peak_flops       = 275e12,   # BF16 MXU
    hbm_bandwidth    = 1.2e12,
    l2_cache_bytes   = 0,        # TPU has HBM only
    shared_mem_bytes = 0,
    compute_cap      = 0.0,
)


# ---------------------------------------------------------------------------
# Per-op roofline data
# ---------------------------------------------------------------------------

@dataclass
class OpRooflinePoint:
    op_id:                str
    op_name:              str
    phase:                Optional[Phase]
    flops:                int
    bytes_accessed:       int
    arithmetic_intensity: float
    is_compute_bound:     bool
    achieved_efficiency:  float  # 0-1 (estimated)
    bottleneck:           str    # "compute" or "memory"


@dataclass
class RooflineReport:
    hw_spec:         HardwareSpec
    program_name:    str
    op_points:       List[OpRooflinePoint] = field(default_factory=list)
    phase_summary:   Dict[str, Any]        = field(default_factory=dict)
    total_flops:     int = 0
    total_bytes:     int = 0
    gpu_idle_ms:     float = 0.0

    @property
    def overall_arithmetic_intensity(self) -> float:
        if self.total_bytes == 0:
            return float("inf")
        return self.total_flops / self.total_bytes

    @property
    def overall_bottleneck(self) -> str:
        ai = self.overall_arithmetic_intensity
        return "compute" if ai > self.hw_spec.ridge_point else "memory"

    def phase_breakdown_str(self) -> str:
        lines = [f"Roofline Report: {self.program_name} on {self.hw_spec.name}",
                 f"Ridge point: {self.hw_spec.ridge_point:.0f} FLOP/byte",
                 f"Overall arithmetic intensity: {self.overall_arithmetic_intensity:.1f} FLOP/byte",
                 f"Overall bottleneck: {self.overall_bottleneck}",
                 f"GPU idle time (tool I/O): {self.gpu_idle_ms:.1f}ms",
                 ""]
        for phase_name, stats in self.phase_summary.items():
            lines.append(f"  {phase_name:12s}: {stats['n_ops']:3d} ops, "
                         f"AI={stats['mean_ai']:.1f} FLOP/byte, "
                         f"{'compute' if stats['mean_ai'] > self.hw_spec.ridge_point else 'memory'}-bound")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RooflineAnalyzer:
    """
    Computes roofline metrics for an AgentProgram.

    analyzer = RooflineAnalyzer(A100_SPEC)
    report   = analyzer.analyze_program(prog)
    print(report.phase_breakdown_str())
    fig      = analyzer.plot(report)  # requires matplotlib
    """

    def __init__(self, hw_spec: HardwareSpec):
        self.hw = hw_spec

    def analyze_program(self, prog: AgentProgram) -> RooflineReport:
        report = RooflineReport(
            hw_spec      = self.hw,
            program_name = prog.name,
            gpu_idle_ms  = prog.total_gpu_idle_ms(),
        )

        phase_data: Dict[str, List[float]] = {p.value: [] for p in Phase}

        for op in prog.all_ops():
            flops = op.estimated_flops()
            btes  = op.estimated_bytes()
            ai    = op.arithmetic_intensity()
            is_cb = ai > self.hw.ridge_point

            if flops == 0 and btes == 0:
                continue

            eff = self._estimate_efficiency(op, is_cb)
            pt  = OpRooflinePoint(
                op_id                = op.id,
                op_name              = op.name,
                phase                = op.phase,
                flops                = flops,
                bytes_accessed       = btes,
                arithmetic_intensity = ai,
                is_compute_bound     = is_cb,
                achieved_efficiency  = eff,
                bottleneck           = "compute" if is_cb else "memory",
            )
            report.op_points.append(pt)
            report.total_flops += flops
            report.total_bytes += btes

            if op.phase:
                phase_data[op.phase.value].append(ai)

        # Phase summary
        for phase_name, ais in phase_data.items():
            if ais:
                report.phase_summary[phase_name] = {
                    "n_ops":   len(ais),
                    "mean_ai": sum(ais) / len(ais),
                    "min_ai":  min(ais),
                    "max_ai":  max(ais),
                }

        return report

    def _estimate_efficiency(self, op: Any, is_compute_bound: bool) -> float:
        """
        Heuristic efficiency estimate. Real measurement comes from Nsight/XProf.
        Used for plotting expected vs. achievable performance.
        """
        from ..ir.dialect import LLMForwardOp, KVWriteOp, KVFuseOp
        if isinstance(op, LLMForwardOp):
            if op.phase == Phase.PREFILL:
                return 0.72  # FlashAttention-2 typical on A100
            else:
                return 0.55  # decode is harder to saturate
        if isinstance(op, KVFuseOp):
            return 0.85  # compiler-fused is better than naive writes
        if isinstance(op, KVWriteOp):
            return 0.60  # memory-bound write
        return 0.50

    def plot(self, report: RooflineReport, show_phases: bool = True):
        """
        Generate a roofline plot. Returns matplotlib Figure.
        Colab-compatible: call plt.show() or display(fig) after.

        The plot shows:
          - The roofline (memory-bound slope + compute-bound ceiling)
          - One point per LLMForwardOp, colored by phase
          - The ridge point labeled
          - GPU idle time as an annotation
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Roofline bounds
        ai_range = np.logspace(-1, 4, 500)  # 0.1 to 10000 FLOP/byte
        mem_bound  = self.hw.hbm_bandwidth * ai_range
        comp_bound = np.full_like(ai_range, self.hw.peak_flops)
        roofline   = np.minimum(mem_bound, comp_bound)

        ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline", zorder=5)
        ax.axvline(self.hw.ridge_point, color="gray", linestyle="--", alpha=0.6,
                   label=f"Ridge point ({self.hw.ridge_point:.0f} FLOP/byte)")

        # Phase colors
        COLORS = {
            Phase.PREFILL: "#1D9E75",  # green
            Phase.DECODE:  "#3B8BD4",  # blue
            Phase.IO:      "#EF9F27",  # amber
            Phase.FUSE:    "#D05538",  # coral
        }

        # Plot each op
        for pt in report.op_points:
            if pt.flops == 0:
                continue
            color = COLORS.get(pt.phase, "#888") if pt.phase else "#888"
            achieved = pt.achieved_efficiency * min(
                self.hw.peak_flops,
                self.hw.hbm_bandwidth * pt.arithmetic_intensity
            )
            ax.scatter(pt.arithmetic_intensity, achieved,
                       color=color, s=100, alpha=0.8, zorder=10,
                       edgecolors="white", linewidths=0.5)

        # Legend
        patches = [
            mpatches.Patch(color=COLORS[p], label=p.value)
            for p in Phase if any(pt.phase == p for pt in report.op_points)
        ]
        ax.legend(handles=patches + [
            plt.Line2D([0], [0], color="k", linewidth=2, label="Roofline"),
        ], loc="lower right")

        # GPU idle annotation
        if report.gpu_idle_ms > 0:
            ax.annotate(
                f"GPU idle: {report.gpu_idle_ms:.0f}ms\n(tool I/O opportunity)",
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, color="#EF9F27",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", alpha=0.8),
            )

        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
        ax.set_ylabel("Performance (FLOP/s)",              fontsize=12)
        ax.set_title(f"LoopFuse Roofline: {report.program_name}\n{self.hw.name}",
                     fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(0.1, 10000)
        ax.set_ylim(1e9, self.hw.peak_flops * 2)

        plt.tight_layout()
        return fig
