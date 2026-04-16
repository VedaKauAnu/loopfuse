"""tests/test_stats.py — unit tests for statistical utilities."""
import pytest
import math
from loopfuse.analysis.stats import (
    BenchmarkResult, ComparisonResult, benchmark, compare, print_comparison_table,
)


class TestBenchmarkResult:
    def _make(self, times):
        return BenchmarkResult("sys", times)

    def test_p50(self):
        r = self._make([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r.p50 == pytest.approx(3.0, abs=0.1)

    def test_p99_near_max(self):
        r = self._make(list(range(1, 101)))
        assert r.p99 >= 99.0

    def test_mean(self):
        r = self._make([10.0, 20.0, 30.0])
        assert r.mean == pytest.approx(20.0)

    def test_std(self):
        r = self._make([1.0, 1.0, 1.0])
        assert r.std == pytest.approx(0.0, abs=1e-9)

    def test_summary_str_contains_system_name(self):
        r = BenchmarkResult("my_system", [5.0] * 10)
        assert "my_system" in r.summary_str()


class TestCompare:
    def _two_systems(self, a_times, b_times):
        ra = BenchmarkResult("A", a_times)
        rb = BenchmarkResult("B", b_times)
        return compare(ra, rb)

    def test_significant_difference_detected(self):
        # System A is clearly slower (10ms vs 5ms mean)
        a = [10.0 + 0.1 * i for i in range(100)]
        b = [ 5.0 + 0.1 * i for i in range(100)]
        cmp = self._two_systems(a, b)
        assert cmp.is_significant
        assert cmp.speedup_p50 > 1.0

    def test_no_difference_not_significant(self):
        times = [10.0] * 100
        cmp = self._two_systems(times, times[:])
        assert not cmp.is_significant
        assert cmp.speedup_p50 == pytest.approx(1.0)

    def test_speedup_p50_calculation(self):
        a = [20.0] * 50
        b = [10.0] * 50
        cmp = self._two_systems(a, b)
        assert cmp.speedup_p50 == pytest.approx(2.0)

    def test_cohens_d_large_effect(self):
        a = [10.0] * 100
        b = [20.0] * 100
        cmp = self._two_systems(a, b)
        # d = (10-20)/0 -> undefined, but for non-zero spread use approx values
        a2 = [10.0 + 0.1*i for i in range(100)]
        b2 = [20.0 + 0.1*i for i in range(100)]
        cmp2 = self._two_systems(a2, b2)
        assert abs(cmp2.cohens_d) > 0.8  # "large" effect

    def test_effect_size_label(self):
        a = [10.0 + 0.1*i for i in range(100)]
        b = [20.0 + 0.1*i for i in range(100)]
        cmp = self._two_systems(a, b)
        assert cmp.effect_size_label == "large"

    def test_p_value_range(self):
        a = [10.0 + 0.5*i for i in range(100)]
        b = [12.0 + 0.5*i for i in range(100)]
        cmp = self._two_systems(a, b)
        assert 0.0 <= cmp.p_value <= 1.0


class TestBenchmarkFunction:
    def test_returns_benchmark_result(self):
        r = benchmark(lambda: None, n_warmup=2, n_measure=10)
        assert isinstance(r, BenchmarkResult)
        assert len(r.times_ms) == 10

    def test_times_are_positive(self):
        r = benchmark(lambda: None, n_warmup=2, n_measure=10)
        assert all(t > 0 for t in r.times_ms)

    def test_system_name_preserved(self):
        r = benchmark(lambda: None, n_warmup=1, n_measure=5, system_name="mytest")
        assert r.system_name == "mytest"


class TestPrintComparisonTable:
    def test_returns_string(self):
        ra = BenchmarkResult("A", [10.0]*50)
        rb = BenchmarkResult("B", [ 8.0]*50)
        t = print_comparison_table([(ra, "sys A"), (rb, "sys B")])
        assert isinstance(t, str)
        assert "sys A" in t
        assert "sys B" in t
