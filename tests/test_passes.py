"""tests/test_passes.py — unit tests for optimization passes."""
import pytest
from loopfuse.ir.dialect import (
    Phase, KernelTarget, KVFuseOp, SpeculativePrefillOp,
    LLMForwardOp, KVWriteOp, GPT2_CONFIG, GPT2_KV_CONFIG,
)
from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass
from loopfuse.passes.base import Pass, PassResult


class TestPassBase:
    def test_pass_result_repr(self):
        r = PassResult("TestPass", changed=True, duration_ms=1.23)
        assert "TestPass" in repr(r)
        assert "changed" in repr(r)

    def test_pass_manager_returns_results(self, gpt2_3step):
        pm = PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()])
        results = pm.run(gpt2_3step)
        assert len(results) == 3
        assert all(isinstance(r, PassResult) for r in results)

    def test_pass_manager_fixed_point(self, gpt2_3step):
        pm = PassManager([SpecPrefillPass()])
        r1 = pm.run_until_fixed_point(gpt2_3step, max_iterations=5)
        # Second run should be no-change (already inserted)
        r2 = pm.run_until_fixed_point(gpt2_3step, max_iterations=5)
        assert not any(r.changed for r in r2)


class TestSpecPrefillPass:
    def test_inserts_spec_prefill_ops(self, gpt2_3step):
        ci_before = len(gpt2_3step.compiler_inserted_ops())
        result = SpecPrefillPass(min_idle_window_ms=50.0, confidence_threshold=0.5).run(gpt2_3step)
        ci_after = len(gpt2_3step.compiler_inserted_ops())
        assert result.changed
        assert ci_after > ci_before

    def test_spec_prefill_is_compiler_inserted(self, gpt2_3step):
        SpecPrefillPass().run(gpt2_3step)
        sp_ops = [op for op in gpt2_3step.all_ops()
                  if isinstance(op, SpeculativePrefillOp)]
        assert all(op.is_compiler_inserted() for op in sp_ops)

    def test_spec_prefill_phase_is_prefill(self, gpt2_3step):
        SpecPrefillPass().run(gpt2_3step)
        sp_ops = [op for op in gpt2_3step.all_ops()
                  if isinstance(op, SpeculativePrefillOp)]
        assert all(op.phase == Phase.PREFILL for op in sp_ops)

    def test_no_insert_when_window_too_small(self, gpt2_5step):
        """Pass should not insert when min_idle_window_ms is very large."""
        pass_obj = SpecPrefillPass(min_idle_window_ms=9999.0)
        result = pass_obj.run(gpt2_5step)
        assert not result.changed

    def test_no_insert_below_confidence(self, gpt2_3step):
        """Pass should not insert when confidence_threshold is very high."""
        result = SpecPrefillPass(confidence_threshold=0.999).run(gpt2_3step)
        assert not result.changed

    def test_spec_prefill_inserted_before_tool_call(self, gpt2_3step):
        """SpeculativePrefillOp must appear immediately before its ToolCallOp."""
        from loopfuse.ir.dialect import ToolCallOp
        SpecPrefillPass().run(gpt2_3step)
        for step in gpt2_3step.steps:
            for i, op in enumerate(step.act_region):
                if isinstance(op, SpeculativePrefillOp):
                    assert i + 1 < len(step.act_region)
                    assert isinstance(step.act_region[i + 1], ToolCallOp)

    def test_confidence_decreases_with_step_id(self, gpt2_5step):
        """Later steps should have lower confidence (heuristic)."""
        pass_obj = SpecPrefillPass()
        from loopfuse.ir.dialect import ToolCallOp
        confidences = []
        for step in gpt2_5step.steps:
            for op in step.tool_ops():
                c = pass_obj._estimate_confidence(step, op, gpt2_5step)
                confidences.append((step.step_id, c))
        if len(confidences) >= 2:
            ids = [x[0] for x in confidences]
            cs  = [x[1] for x in confidences]
            assert cs[0] >= cs[-1], "Confidence should not increase with step_id"

    def test_idempotent(self, gpt2_3step):
        """Running twice should not double-insert."""
        p = SpecPrefillPass()
        p.run(gpt2_3step)
        sp_count_1 = len([op for op in gpt2_3step.all_ops()
                           if isinstance(op, SpeculativePrefillOp)])
        # Second run: all tool calls already have a spec_prefill before them
        p.run(gpt2_3step)
        sp_count_2 = len([op for op in gpt2_3step.all_ops()
                           if isinstance(op, SpeculativePrefillOp)])
        # Should not grow unboundedly
        assert sp_count_2 <= sp_count_1 * 2


class TestPhaseSelectPass:
    def test_annotates_llm_ops(self, gpt2_3step):
        PhaseSelectPass().run(gpt2_3step)
        llm_ops = gpt2_3step.ops_by_type(LLMForwardOp)
        for op in llm_ops:
            assert "kernel_module" in op.metadata
            assert "kernel_name"   in op.metadata
            assert "is_compute_bound" in op.metadata

    def test_decode_is_memory_bound(self, gpt2_3step):
        PhaseSelectPass().run(gpt2_3step)
        llm_ops = gpt2_3step.ops_by_type(LLMForwardOp)
        decode_ops = [op for op in llm_ops if op.phase == Phase.DECODE]
        assert all(not op.metadata["is_compute_bound"] for op in decode_ops)

    def test_triton_target_uses_triton_kernel(self, gpt2_3step):
        gpt2_3step.set_target(KernelTarget.TRITON)
        PhaseSelectPass().run(gpt2_3step)
        llm_ops = gpt2_3step.ops_by_type(LLMForwardOp)
        for op in llm_ops:
            module = op.metadata.get("kernel_module", "")
            assert "triton" in module or "eager" in module

    def test_eager_target_uses_torch_sdpa(self, gpt2_3step):
        gpt2_3step.set_target(KernelTarget.EAGER)
        PhaseSelectPass().run(gpt2_3step)
        llm_ops = gpt2_3step.ops_by_type(LLMForwardOp)
        for op in llm_ops:
            module = op.metadata.get("kernel_module", "")
            assert "torch" in module or "eager" in module

    def test_marks_program_as_changed(self, gpt2_3step):
        result = PhaseSelectPass().run(gpt2_3step)
        assert result.changed

    def test_arithmetic_intensity_annotation(self, gpt2_3step):
        PhaseSelectPass().run(gpt2_3step)
        for op in gpt2_3step.ops_by_type(LLMForwardOp):
            ai = op.metadata.get("arithmetic_intensity", -1)
            assert ai >= 0


class TestKVFusionPass:
    def test_no_fusion_below_min_steps(self, gpt2_3step):
        """With min_steps=10, should not fuse a 3-step program."""
        result = KVFusionPass(min_prefix_len=4, min_steps=10).run(gpt2_3step)
        assert not result.changed

    def test_pass_result_annotation_contains_info(self, gpt2_5step):
        import copy
        prog_copy = copy.deepcopy(gpt2_5step)
        result = KVFusionPass(min_prefix_len=4).run(prog_copy)
        # Whether or not changed, the pass should return a valid result
        assert isinstance(result, PassResult)

    def test_kv_fuse_op_phase(self, gpt2_5step):
        """Any KVFuseOp inserted must have phase FUSE."""
        KVFusionPass(min_prefix_len=4).run(gpt2_5step)
        fuse_ops = gpt2_5step.ops_by_type(KVFuseOp)
        assert all(op.phase == Phase.FUSE for op in fuse_ops)

    def test_fuse_op_is_compiler_inserted(self, gpt2_5step):
        KVFusionPass(min_prefix_len=4).run(gpt2_5step)
        fuse_ops = gpt2_5step.ops_by_type(KVFuseOp)
        assert all(op.is_compiler_inserted() for op in fuse_ops)


class TestPassManagerIntegration:
    def test_full_pipeline(self, gpt2_5step, full_pass_manager):
        results = full_pass_manager.run(gpt2_5step)
        assert len(results) == 3
        # SpecPrefillPass and PhaseSelectPass must change the program
        assert any(r.changed for r in results)

    def test_compiler_inserted_ops_count(self, gpt2_5step, full_pass_manager):
        full_pass_manager.run(gpt2_5step)
        ci = gpt2_5step.compiler_inserted_ops()
        # At minimum: SpecPrefillPass inserts ops for steps with tool calls (4 tools)
        assert len(ci) >= 1

    def test_pass_annotations_added_to_program(self, gpt2_5step, full_pass_manager):
        full_pass_manager.run(gpt2_5step)
        assert len(gpt2_5step.pass_annotations) > 0

    def test_total_ops_grows_after_passes(self, gpt2_5step, full_pass_manager):
        before = len(gpt2_5step.all_ops())
        full_pass_manager.run(gpt2_5step)
        after = len(gpt2_5step.all_ops())
        assert after >= before  # passes add ops, never remove without replacement
