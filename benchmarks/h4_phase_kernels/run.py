"""
benchmarks/h4_phase_kernels/run.py — H4: Phase-specialized kernels vs. torch.compile.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import torch
import torch.nn.functional as F
from loopfuse.analysis.stats import benchmark, compare, print_comparison_table
from loopfuse.analysis.roofline import RooflineAnalyzer, A100_SPEC, T4_SPEC


def run(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("H4 requires CUDA (Triton kernels). Skipping.")
        return {}

    print(f"\nH4: Phase-Specialized Kernels  GPU={torch.cuda.get_device_name(0)}")

    try:
        from loopfuse.kernels.triton.decode_attn  import decode_attention_fwd, verify_decode_kernel
        from loopfuse.kernels.triton.prefill_attn import prefill_attention_fwd, verify_prefill_kernel
    except Exception as e:
        print(f"  Triton kernels unavailable: {e}")
        return {}

    # Verify correctness first
    print("  Verifying decode kernel... ", end="")
    dec_check = verify_decode_kernel()
    print("✓" if dec_check["passed"] else f"✗ err={dec_check['max_abs_error']:.4f}")

    print("  Verifying prefill kernel... ", end="")
    pre_check = verify_prefill_kernel()
    print("✓" if pre_check["passed"] else f"✗ err={pre_check['max_abs_error']:.4f}")

    sync = torch.cuda.synchronize
    results = {}

    # --- DECODE benchmark ---
    print("\n  Decode (seq_q=1, seq_kv=512, heads=12, dim=64)")
    q_dec = torch.randn(1, 12, 1,   64, dtype=torch.float16, device=device)
    k_dec = torch.randn(1, 12, 512, 64, dtype=torch.float16, device=device)
    v_dec = torch.randn_like(k_dec)

    r_eager_dec = benchmark(
        lambda: F.scaled_dot_product_attention(q_dec, k_dec, v_dec),
        sync_fn=sync, system_name="torch SDPA (baseline)", n_warmup=50, n_measure=200)
    r_lfuse_dec = benchmark(
        lambda: decode_attention_fwd(q_dec, k_dec, v_dec),
        sync_fn=sync, system_name="LoopFuse decode", n_warmup=50, n_measure=200)

    print(print_comparison_table([(r_eager_dec, "torch SDPA"), (r_lfuse_dec, "LoopFuse decode")]))
    cmp_dec = compare(r_eager_dec, r_lfuse_dec)
    print(f"  Decode: {cmp_dec.summary_str()}")
    results["decode"] = cmp_dec

    # --- PREFILL benchmark ---
    print("\n  Prefill (seq_q=seq_k=128, heads=12, dim=64)")
    q_pre = torch.randn(1, 12, 128, 64, dtype=torch.float16, device=device)
    k_pre = torch.randn_like(q_pre)
    v_pre = torch.randn_like(q_pre)

    r_eager_pre = benchmark(
        lambda: F.scaled_dot_product_attention(q_pre, k_pre, v_pre, is_causal=True),
        sync_fn=sync, system_name="torch SDPA (baseline)", n_warmup=30, n_measure=100)
    r_lfuse_pre = benchmark(
        lambda: prefill_attention_fwd(q_pre, k_pre, v_pre, causal=True),
        sync_fn=sync, system_name="LoopFuse prefill", n_warmup=30, n_measure=100)

    print(print_comparison_table([(r_eager_pre, "torch SDPA"), (r_lfuse_pre, "LoopFuse prefill")]))
    cmp_pre = compare(r_eager_pre, r_lfuse_pre)
    print(f"  Prefill: {cmp_pre.summary_str()}")
    results["prefill"] = cmp_pre

    # H4 confirmed if EITHER decode OR prefill is significantly faster
    h4 = any(c.speedup_p50 > 1.05 and c.is_significant for c in results.values())
    print(f"\n  H4 {'CONFIRMED ✓' if h4 else 'inconclusive'}")
    return results


if __name__ == "__main__":
    run()
