"""
loopfuse/analysis/stats.py

Statistical utilities for LoopFuse benchmarks.

All benchmark comparisons require statistical rigor:
  - Report p50/p99, not mean (mean is skewed by JIT compilation)
  - Welch's t-test for significance (unequal variance across systems)
  - Cohen's d for effect size (is the speedup practically meaningful?)
  - Mandatory warmup before measurement
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    system_name: str
    times_ms:    List[float]

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p90(self) -> float:
        return self._percentile(90)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def mean(self) -> float:
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def std(self) -> float:
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self.times_ms) / len(self.times_ms))

    def _percentile(self, pct: float) -> float:
        s = sorted(self.times_ms)
        idx = (pct / 100) * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] + (idx - lo) * (s[hi] - s[lo])

    def summary_str(self) -> str:
        return (f"{self.system_name:20s}  "
                f"p50={self.p50:.2f}ms  "
                f"p90={self.p90:.2f}ms  "
                f"p99={self.p99:.2f}ms  "
                f"mean={self.mean:.2f}ms  "
                f"std={self.std:.2f}ms  "
                f"n={len(self.times_ms)}")


@dataclass
class ComparisonResult:
    baseline: BenchmarkResult
    treatment: BenchmarkResult
    p_value:   float
    cohens_d:  float
    speedup_p50: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def effect_size_label(self) -> str:
        d = abs(self.cohens_d)
        if d < 0.2: return "negligible"
        if d < 0.5: return "small"
        if d < 0.8: return "medium"
        return "large"

    def summary_str(self) -> str:
        sig = "✓ significant" if self.is_significant else "✗ not significant"
        return (f"Speedup p50: {self.speedup_p50:.2f}x  |  "
                f"p-value: {self.p_value:.4f} ({sig})  |  "
                f"Cohen's d: {self.cohens_d:.2f} ({self.effect_size_label})")


def benchmark(
    fn: Callable,
    n_warmup: int = 20,
    n_measure: int = 100,
    system_name: str = "system",
    sync_fn: Optional[Callable] = None,
) -> BenchmarkResult:
    """
    Benchmark a callable with GPU-aware timing.

    Args:
        fn:          The function to benchmark. Must be callable with no args.
        n_warmup:    Warmup iterations (not measured). 20 is standard for Triton.
        n_measure:   Measured iterations.
        system_name: Label for the result.
        sync_fn:     If provided, called before each timing stop (e.g. torch.cuda.synchronize).

    Usage:
        result = benchmark(lambda: decode_attention_fwd(q, k, v),
                           sync_fn=torch.cuda.synchronize,
                           system_name="LoopFuse decode")
        print(result.summary_str())
    """
    # Warmup — essential for JIT kernels (Triton, torch.compile)
    for _ in range(n_warmup):
        fn()
    if sync_fn:
        sync_fn()

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return BenchmarkResult(system_name=system_name, times_ms=times)


def compare(
    baseline:  BenchmarkResult,
    treatment: BenchmarkResult,
) -> ComparisonResult:
    """
    Statistical comparison of two benchmark results.
    Uses Welch's t-test (doesn't assume equal variance).
    """
    speedup = baseline.p50 / treatment.p50 if treatment.p50 > 0 else float("inf")

    # Welch's t-test
    n1, n2 = len(baseline.times_ms), len(treatment.times_ms)
    m1, m2 = baseline.mean, treatment.mean
    s1, s2 = baseline.std, treatment.std

    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat = (m1 - m2) / se
        # Welch-Satterthwaite degrees of freedom
        df_num = (s1**2 / n1 + s2**2 / n2) ** 2
        df_den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else 1
        # Two-tailed p-value (approximation via Gaussian for large n)
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    # Cohen's d (pooled standard deviation)
    pooled_std = math.sqrt((s1**2 + s2**2) / 2)
    cohens_d   = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0

    return ComparisonResult(
        baseline    = baseline,
        treatment   = treatment,
        p_value     = p_value,
        cohens_d    = cohens_d,
        speedup_p50 = speedup,
    )


def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2) * poly
    return cdf if x >= 0 else 1.0 - cdf


def print_comparison_table(results: List[Tuple[BenchmarkResult, str]]) -> str:
    """Print a formatted comparison table for multiple systems."""
    lines = [
        f"{'System':<22} {'p50':>8} {'p90':>8} {'p99':>8} {'mean':>8} {'std':>7}",
        "-" * 60,
    ]
    baseline_p50 = results[0][0].p50 if results else 1.0
    for res, label in results:
        speedup = baseline_p50 / res.p50 if res.p50 > 0 else 0
        lines.append(
            f"{label:<22} {res.p50:>7.2f}ms {res.p90:>7.2f}ms "
            f"{res.p99:>7.2f}ms {res.mean:>7.2f}ms {res.std:>6.2f}ms"
            + (f"  ({speedup:.2f}x)" if res != results[0][0] else "  (baseline)")
        )
    return "\n".join(lines)
