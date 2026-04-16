"""tests/test_ir_dialect.py — unit tests for the Agent IR."""
import pytest
from loopfuse.ir.dialect import (
    Phase, KernelTarget, TensorType, KVCacheType, TokenSeqType, ScalarType,
    Value, LLMForwardOp, KVWriteOp, KVFuseOp, ToolCallOp, SpeculativePrefillOp,
    PromptConstructOp, ArgmaxOp, AgentStepOp, AgentProgram,
    GPT2_CONFIG, GPT2_KV_CONFIG,
)


class TestTypes:
    def test_tensor_type_repr(self):
        t = TensorType((1, 50257), "bf16")
        assert "bf16" in repr(t)
        assert "50257" in repr(t)

    def test_tensor_bytes(self):
        t = TensorType((4, 768), "fp16")
        assert t.bytes() == 4 * 768 * 2

    def test_kvcache_bytes_per_token(self):
        kv = KVCacheType(num_layers=12, num_heads=12, head_dim=64,
                         max_seq_len=1024, dtype="fp16")
        # 2 (K+V) * 12 * 12 * 64 * 2 bytes = 36864
        assert kv.bytes_per_token() == 2 * 12 * 12 * 64 * 2

    def test_kvcache_total_bytes(self):
        kv = KVCacheType(12, 12, 64, 1024, "fp16")
        assert kv.total_bytes() == kv.bytes_per_token() * 1024


class TestValues:
    def test_value_repr(self):
        v = Value("x", TensorType((1, 768), "fp16"))
        assert "%x" in repr(v)

    def test_value_no_defining_op(self):
        v = Value("standalone", ScalarType("int32"))
        assert v.defining_op is None


class TestOps:
    def test_llm_forward_phase_prefill(self):
        kv_val = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp    = Value("inp", TokenSeqType(64, 50257))
        op = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.PREFILL, seq_len=64)
        assert op.phase == Phase.PREFILL
        assert op.seq_len == 64
        assert len(op.results) == 1

    def test_llm_forward_phase_decode(self):
        kv_val = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp    = Value("inp", TokenSeqType(1, 50257))
        op = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.DECODE, seq_len=1)
        assert op.phase == Phase.DECODE

    def test_llm_forward_flops(self):
        kv_val = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp    = Value("inp", TokenSeqType(64, 50257))
        op = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.PREFILL, seq_len=64)
        assert op.estimated_flops() > 0

    def test_llm_forward_decode_less_flops_than_prefill(self):
        kv_val = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp    = Value("inp", TokenSeqType(1, 50257))
        dec = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.DECODE,  seq_len=1)
        pre = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.PREFILL, seq_len=64)
        assert dec.estimated_flops() < pre.estimated_flops()

    def test_arithmetic_intensity_decode_below_ridge(self):
        """Decode must be memory-bound (AI < 156 on A100)."""
        kv_val = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp    = Value("inp", TokenSeqType(1, 50257))
        dec = LLMForwardOp(inp, kv_val, GPT2_CONFIG, phase=Phase.DECODE, seq_len=1)
        assert dec.arithmetic_intensity() < 156.0  # below A100 ridge

    def test_tool_call_phase_io(self):
        inp = Value("inp", TokenSeqType(64, 50257))
        op  = ToolCallOp("search", inp, estimated_latency_ms=150.0)
        assert op.phase == Phase.IO
        assert op.metadata["gpu_idle"] is True
        assert op.idle_window_ms() == 150.0

    def test_kv_write_phase_io(self):
        kv  = Value("kv",  KVCacheType(12, 12, 64, 1024))
        k   = Value("k",   TensorType((12, 12, 1, 64), "fp16"))
        v   = Value("v",   TensorType((12, 12, 1, 64), "fp16"))
        op  = KVWriteOp(kv, k, v, seq_pos=0, step_id=0)
        assert op.phase == Phase.IO
        assert op.estimated_bytes() > 0

    def test_kv_fuse_compiler_inserted(self):
        kv = Value("kv", KVCacheType(12, 12, 64, 1024))
        k  = Value("k",  TensorType((12, 12, 1, 64), "fp16"))
        v  = Value("v",  TensorType((12, 12, 1, 64), "fp16"))
        w  = KVWriteOp(kv, k, v, seq_pos=0)
        f  = KVFuseOp(kv, steps_to_fuse=[0, 1, 2], prefix_len=32, replaced_writes=[w])
        assert f.is_compiler_inserted()
        assert f.phase == Phase.FUSE
        assert f.attrs["prefix_len"] == 32

    def test_spec_prefill_compiler_inserted(self):
        toks = Value("t",  TokenSeqType(64, 50257))
        kv   = Value("kv", KVCacheType(12, 12, 64, 1024))
        tool = ToolCallOp("search", toks, estimated_latency_ms=200.0)
        sp   = SpeculativePrefillOp(toks, kv, GPT2_CONFIG, confidence=0.85,
                                    overlapping_tool=tool)
        assert sp.is_compiler_inserted()
        assert sp.phase == Phase.PREFILL
        assert sp.metadata["confidence"] == 0.85
        assert sp.metadata["available_window_ms"] == 200.0

    def test_op_repr_contains_phase(self):
        kv  = Value("kv", KVCacheType(12, 12, 64, 1024))
        inp = Value("inp", TokenSeqType(1, 50257))
        op  = LLMForwardOp(inp, kv, GPT2_CONFIG, phase=Phase.DECODE)
        r   = repr(op)
        assert "decode" in r


class TestAgentStepOp:
    def test_regions(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        assert len(step0.observe_region) > 0
        assert len(step0.reason_region)  > 0
        assert len(step0.act_region)     > 0

    def test_all_ops_covers_all_regions(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        all_ops = step0.all_ops()
        assert len(all_ops) == (len(step0.observe_region) +
                                 len(step0.reason_region) +
                                 len(step0.act_region))

    def test_llm_ops(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        assert len(step0.llm_ops()) >= 1

    def test_tool_ops_present_in_non_final_steps(self, gpt2_3step):
        # Step 0 and 1 have tool calls; step 2 is final answer
        assert len(gpt2_3step.steps[0].tool_ops()) == 1
        assert len(gpt2_3step.steps[1].tool_ops()) == 1
        assert len(gpt2_3step.steps[2].tool_ops()) == 0

    def test_gpu_idle_ms(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        assert step0.total_gpu_idle_ms() == 150.0

    def test_insert_before(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        from loopfuse.ir.dialect import ArgmaxOp, Value, TensorType
        logits = Value("logits", TensorType((1, 50257), "bf16"))
        new_op = ArgmaxOp(logits)
        target = step0.act_region[0]
        orig_len = len(step0.act_region)
        step0.insert_before(target, new_op)
        assert len(step0.act_region) == orig_len + 1
        assert step0.act_region[0] is new_op

    def test_remove_op(self, gpt2_3step):
        step0 = gpt2_3step.steps[0]
        target = step0.act_region[0]
        orig_len = len(step0.act_region)
        removed = step0.remove(target)
        assert removed
        assert len(step0.act_region) == orig_len - 1


class TestAgentProgram:
    def test_total_ops(self, gpt2_3step):
        total = len(gpt2_3step.all_ops())
        assert total > 0

    def test_ops_by_phase(self, gpt2_3step):
        by_phase = gpt2_3step.ops_by_phase()
        # Must have at least one PREFILL and one IO op
        assert len(by_phase[Phase.PREFILL]) >= 1
        assert len(by_phase[Phase.IO]) >= 1

    def test_total_gpu_idle_ms(self, gpt2_3step):
        # steps 0 + 1 have tool calls: 150 + 80 = 230ms
        assert gpt2_3step.total_gpu_idle_ms() == 230.0

    def test_set_target(self, gpt2_3step):
        gpt2_3step.set_target(KernelTarget.CUDA)
        assert gpt2_3step.target == KernelTarget.CUDA

    def test_summary_keys(self, gpt2_3step):
        s = gpt2_3step.summary()
        for key in ["name", "target", "num_steps", "total_ops", "ops_by_phase"]:
            assert key in s

    def test_5step_has_5_steps(self, gpt2_5step):
        assert len(gpt2_5step.steps) == 5

    def test_kv_state_type(self, gpt2_3step):
        from loopfuse.ir.dialect import KVCacheType
        assert isinstance(gpt2_3step.kv_state.type, KVCacheType)
