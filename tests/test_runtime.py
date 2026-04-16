"""tests/test_runtime.py — unit tests for KVPool (CPU-only)."""
import pytest
import torch
from loopfuse.runtime.kv_pool import KVPool, KVState
from loopfuse.ir.dialect import GPT2_KV_CONFIG


@pytest.fixture
def cpu_pool():
    return KVPool.from_config(GPT2_KV_CONFIG, device="cpu", max_slots=4)


class TestKVPool:
    def test_from_config(self, cpu_pool):
        assert cpu_pool.num_layers  == GPT2_KV_CONFIG["num_layers"]
        assert cpu_pool.num_heads   == GPT2_KV_CONFIG["num_heads"]
        assert cpu_pool.head_dim    == GPT2_KV_CONFIG["head_dim"]
        assert cpu_pool.max_seq_len == GPT2_KV_CONFIG["max_seq_len"]

    def test_allocate_slot(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        assert isinstance(state, KVState)
        assert state.slot_id == 0
        assert state.current_len == 0

    def test_allocate_multiple_slots(self, cpu_pool):
        s0 = cpu_pool.allocate_slot()
        s1 = cpu_pool.allocate_slot()
        assert s0.slot_id != s1.slot_id

    def test_exhaust_slots_raises(self, cpu_pool):
        slots = [cpu_pool.allocate_slot() for _ in range(4)]
        with pytest.raises(RuntimeError, match="exhausted"):
            cpu_pool.allocate_slot()

    def test_release_frees_slot(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        cpu_pool.release(state)
        new_state = cpu_pool.allocate_slot()
        assert new_state is not None

    def test_write_advances_seq_len(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        nl, nh, hd = cpu_pool.num_layers, cpu_pool.num_heads, cpu_pool.head_dim
        k = torch.zeros(nl, nh, 8, hd, dtype=torch.float16)
        v = torch.zeros_like(k)
        new_len = cpu_pool.write(state, k, v)
        assert new_len == 8
        assert state.current_len == 8

    def test_write_multiple_times(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        nl, nh, hd = cpu_pool.num_layers, cpu_pool.num_heads, cpu_pool.head_dim
        for _ in range(4):
            k = torch.zeros(nl, nh, 4, hd, dtype=torch.float16)
            cpu_pool.write(state, k, torch.zeros_like(k))
        assert state.current_len == 16

    def test_write_overflow_raises(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        nl, nh, hd = cpu_pool.num_layers, cpu_pool.num_heads, cpu_pool.head_dim
        max_seq = cpu_pool.max_seq_len
        k = torch.zeros(nl, nh, max_seq + 1, hd, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="overflow"):
            cpu_pool.write(state, k, torch.zeros_like(k))

    def test_read_single_layer(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        nl, nh, hd = cpu_pool.num_layers, cpu_pool.num_heads, cpu_pool.head_dim
        k_in = torch.randn(nl, nh, 16, hd, dtype=torch.float16)
        v_in = torch.randn_like(k_in)
        cpu_pool.write(state, k_in, v_in)
        k_out, v_out = cpu_pool.read(state, layer=0)
        assert k_out.shape == (nh, 16, hd)
        assert v_out.shape == (nh, 16, hd)

    def test_read_all_layers(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        nl, nh, hd = cpu_pool.num_layers, cpu_pool.num_heads, cpu_pool.head_dim
        k_in = torch.randn(nl, nh, 16, hd, dtype=torch.float16)
        cpu_pool.write(state, k_in, torch.randn_like(k_in))
        k_out, v_out = cpu_pool.read_all_layers(state)
        assert k_out.shape == (nl, nh, 16, hd)

    def test_hbm_usage_bytes(self, cpu_pool):
        assert cpu_pool.hbm_usage_bytes() > 0

    def test_stats_keys(self, cpu_pool):
        s = cpu_pool.stats()
        for k in ["hbm_mb", "active_slots", "free_slots", "writes"]:
            assert k in s

    def test_advance_step(self, cpu_pool):
        state = cpu_pool.allocate_slot()
        assert state.step_id == 0
        state.advance_step()
        assert state.step_id == 1
