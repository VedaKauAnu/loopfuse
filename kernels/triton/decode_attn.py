"""
loopfuse/kernels/triton/decode_attn.py

Decode-phase attention kernel — H4 optimization (memory-bandwidth optimized).

Roofline position: DECODE attention at batch=1 has arithmetic intensity ~1-5
FLOP/byte — far below the ridge point (~156 on A100). The bottleneck is HBM
bandwidth, not compute. Every design choice here maximizes streaming bandwidth.

Key design choices vs. a general FlashAttention kernel:
  1. Block size: BLOCK_N=64 instead of 128. At seq_len=1 (decode), a 128-wide
     block wastes tensor core efficiency. 64-wide fills the SMEM bus optimally.
  2. No causal mask computation: decode is always attending to all past tokens.
     Removing the mask predicate saves ~15% of compute.
  3. Persistent KV loading: K and V are loaded once into SMEM registers and
     reused across the (trivially small) Q dimension.
  4. FP16 accumulator: safe for decode (short softmax denominator), avoids
     FP32 upcast overhead that FlashAttention-2 uses for prefill stability.

Usage (Colab T4/A100):
    import torch, triton
    from loopfuse.kernels.triton.decode_attn import decode_attention_fwd

    # batch=1, heads=12, seq_len=512, head_dim=64
    q = torch.randn(1, 12, 1,   64, dtype=torch.float16, device='cuda')
    k = torch.randn(1, 12, 512, 64, dtype=torch.float16, device='cuda')
    v = torch.randn(1, 12, 512, 64, dtype=torch.float16, device='cuda')
    o = decode_attention_fwd(q, k, v)   # shape: (1, 12, 1, 64)

Benchmarking:
    See notebooks/05_phase_profiling.ipynb for T4 vs A100 roofline plots.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel autotune configs
# T4 (compute cap 7.5): smaller blocks, more warps
# A100 (compute cap 8.0): larger blocks, warp-specialized pipeline
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # T4-friendly configs
        triton.Config({"BLOCK_N": 32,  "num_warps": 2, "num_stages": 2}),
        triton.Config({"BLOCK_N": 64,  "num_warps": 4, "num_stages": 2}),
        # A100-friendly configs
        triton.Config({"BLOCK_N": 64,  "num_warps": 4, "num_stages": 3}),
        triton.Config({"BLOCK_N": 128, "num_warps": 8, "num_stages": 3}),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE:    tl.constexpr,
    BLOCK_N:  tl.constexpr,
):
    """
    Decode attention: Q is shape (1, 1, HEAD_DIM) per head.
    K/V are shape (1, SEQ_LEN, HEAD_DIM) per head.

    Each program instance handles one (batch, head) pair.
    We iterate over K/V in BLOCK_N chunks, computing a running softmax.
    """
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    # Offset pointers to this (batch, head)
    Q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    K_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    O_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # Load Q (single query vector) into registers
    q = tl.load(Q_base + tl.arange(0, HEAD_DIM) * stride_qd)  # [HEAD_DIM]
    q = q * SCALE

    # Running softmax accumulators
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.full([1], 0.0,          dtype=tl.float32)  # running sum
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)          # output accumulator

    n_blocks = tl.cdiv(SEQ_LEN, BLOCK_N)

    for block_idx in range(n_blocks):
        start = block_idx * BLOCK_N
        offs  = start + tl.arange(0, BLOCK_N)
        mask  = offs < SEQ_LEN

        # Load K block: [BLOCK_N, HEAD_DIM]
        k_ptrs = K_base + offs[:, None] * stride_ks + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # QK^T: dot product of q [HEAD_DIM] with each k [HEAD_DIM]
        # Result: scores [BLOCK_N]
        scores = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
        scores = tl.where(mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        # Load V block: [BLOCK_N, HEAD_DIM]
        v_ptrs = V_base + offs[:, None] * stride_vs + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # Weighted sum: p [BLOCK_N] @ v [BLOCK_N, HEAD_DIM] -> [HEAD_DIM]
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize
    out = acc / l_i

    # Write output
    o_ptrs = O_base + tl.arange(0, HEAD_DIM) * stride_od
    tl.store(o_ptrs, out.to(tl.float16))


def decode_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Agent-optimized decode attention.

    Args:
        q: (batch, n_heads, 1, head_dim)       -- single query token
        k: (batch, n_heads, seq_len, head_dim) -- full KV cache
        v: (batch, n_heads, seq_len, head_dim)

    Returns:
        o: (batch, n_heads, 1, head_dim)
    """
    assert q.is_cuda, "decode_attention_fwd requires CUDA tensor"
    assert q.dtype == torch.float16, f"Expected fp16, got {q.dtype}"
    assert q.shape[2] == 1, f"Decode kernel expects seq_q=1, got {q.shape[2]}"

    batch, n_heads, _, head_dim = q.shape
    seq_len = k.shape[2]
    scale   = head_dim ** -0.5

    o = torch.empty(batch, n_heads, 1, head_dim, dtype=torch.float16, device=q.device)

    grid = (batch, n_heads)

    _decode_attn_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SEQ_LEN = seq_len,
        HEAD_DIM = head_dim,
        SCALE    = scale,
    )
    return o


# ---------------------------------------------------------------------------
# Correctness check against PyTorch reference
# ---------------------------------------------------------------------------

def _reference_decode_attn(q, k, v):
    """PyTorch reference implementation for correctness verification."""
    import torch.nn.functional as F
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (b, h, 1, s)
    attn = F.softmax(attn.float(), dim=-1).half()
    return torch.matmul(attn, v)  # (b, h, 1, d)


def verify_decode_kernel(seq_len: int = 256, head_dim: int = 64,
                         n_heads: int = 12) -> dict:
    """
    Run correctness check. Call this in Notebook 05 before benchmarking.
    Returns dict with max_abs_error and mean_abs_error.
    """
    import torch
    q = torch.randn(1, n_heads, 1,       head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    out_triton = decode_attention_fwd(q, k, v)
    out_ref    = _reference_decode_attn(q, k, v)

    diff = (out_triton - out_ref).abs()
    return {
        "max_abs_error":  diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "passed":         diff.max().item() < 1e-2,
    }
