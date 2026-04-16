"""tests/test_roofline.py — unit tests for roofline analysis."""
import pytest
from loopfuse.analysis.roofline import RooflineAnalyzer, A100_SPEC, T4_SPEC, TPU_V4_SPEC
from loopfuse.passes import PassManager, PhaseSelectPass


class TestHardwareSpec:
    def test_a100_ridge_point(self):
        assert 150 < A100_SPEC.ridge_point < 165

    def test_t4_ridge_point(self):
        assert 190 < T4_SPEC.ridge_point < 215

    def test_tpu_v4_spec_non_zero(self):
        assert TPU_V4_SPEC.peak_flops > 0
        assert TPU_V4_SPEC.hbm_bandwidth > 0


class TestRooflineAnalyzer:
    def test_returns_report(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert r is not None

    def test_total_flops_positive(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert r.total_flops > 0

    def test_total_bytes_positive(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert r.total_bytes > 0

    def test_gpu_idle_ms_matches_program(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert r.gpu_idle_ms == gpt2_3step.total_gpu_idle_ms()

    def test_agent_workload_is_memory_bound(self, gpt2_3step):
        """Decode-heavy agent loops must be memory-bound overall."""
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert r.overall_bottleneck == "memory"

    def test_phase_summary_has_expected_phases(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert "prefill" in r.phase_summary or "decode" in r.phase_summary

    def test_op_points_non_empty(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        assert len(r.op_points) > 0

    def test_all_op_points_have_positive_ai_or_zero(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        for pt in r.op_points:
            assert pt.arithmetic_intensity >= 0

    def test_phase_breakdown_str(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        s = r.phase_breakdown_str()
        assert "Ridge" in s
        assert "bottleneck" in s

    def test_plot_returns_fig_or_none(self, gpt2_3step):
        r = RooflineAnalyzer(A100_SPEC).analyze_program(gpt2_3step)
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend for CI
            fig = RooflineAnalyzer(A100_SPEC).plot(r)
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not installed")
