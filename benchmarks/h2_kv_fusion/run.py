"""
benchmarks/h2_kv_fusion/run.py — H2: KV Cache Fusion vs. naive per-step writes.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import torch
from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.passes import PassManager, KVFusionPass
from loopfuse.analysis.stats import benchmark, compare, print_comparison_table
from loopfuse.runtime.kv_pool import KVPool


def run(n_steps=5, seq_new=64, prefix_len=32, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPT2_KV_CONFIG
    pool = KVPool.from_config(cfg, device=device, max_slots=16)
    sync = torch.cuda.synchronize if device == "cuda" else None

    print(f"\nH2: KV Cache Fusion  device={device}  n_steps={n_steps}  prefix={prefix_len}/{seq_new} tokens")

    def _kv():
        nl, nh, hd = cfg["num_layers"], cfg["num_heads"], cfg["head_dim"]
        k = torch.randn(nl, nh, seq_new, hd, dtype=torch.float16, device=device)
        return k, torch.randn_like(k)

    def _kv_slice(seq):
        nl, nh, hd = cfg["num_layers"], cfg["num_heads"], cfg["head_dim"]
        k = torch.randn(nl, nh, seq, hd, dtype=torch.float16, device=device)
        return k, torch.randn_like(k)

    def naive():
        state = pool.allocate_slot()
        for _ in range(n_steps):
            k, v = _kv()
            pool.write(state, k, v)
        pool.release(state)
        if sync: sync()

    def fused():
        state = pool.allocate_slot()
        k_pre, v_pre = _kv_slice(prefix_len)
        pool.write(state, k_pre, v_pre)          # prefix written ONCE
        suffix = seq_new - prefix_len
        if suffix > 0:
            for _ in range(n_steps):
                k_s, v_s = _kv_slice(suffix)
                pool.write(state, k_s, v_s)      # only unique suffix per step
        pool.release(state)
        if sync: sync()

    r_naive = benchmark(naive, n_warmup=20, n_measure=100, system_name="naive", sync_fn=sync)
    r_fused = benchmark(fused, n_warmup=20, n_measure=100, system_name="KVFuse", sync_fn=sync)

    print(print_comparison_table([(r_naive, "naive"), (r_fused, "KVFuse")]))
    cmp = compare(r_naive, r_fused)
    print(f"  {cmp.summary_str()}")

    bytes_per_tok = 2 * cfg["num_layers"] * cfg["num_heads"] * cfg["head_dim"] * 2
    saved = bytes_per_tok * prefix_len * (n_steps - 1)
    print(f"  Theoretical HBM saved: {saved / 1024:.1f} KB")
    print(f"  H2 {'CONFIRMED ✓' if cmp.speedup_p50 > 1.02 else 'inconclusive'}  ({cmp.speedup_p50:.2f}x p50)")
    return {"naive": r_naive, "fused": r_fused, "cmp": cmp}

if __name__ == "__main__":
    run()
