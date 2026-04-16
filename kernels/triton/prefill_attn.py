"""
loopfuse/kernels/triton/prefill_attn.py

Prefill-phase attention kernel — H4 optimization (compute-optimized).

Roofline position: PREFILL attention at seq_len=128 has arithmetic intensity
~128 FLOP/byte — approaching the ridge point (~156 on A100). The bottleneck
is tensor core utilization, not HBM bandwidth.

Key design choices vs. decode kernel:
  1. Block size: BLOCK_M=128, BLOCK_N=128 — fills tensor cores fully.
  2. Causal mask: enabled (prefill processes the full prompt causally).
  3. FP32 accumulator: required for stability over long sequences.
  4. Double buffering: prefetch next K/V block while computing current.

This is the FA2-style blocked attention, specialized for the agent prefill
access pattern: one long system prompt followed by a short user query.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "num_warps": 4, "num_stages": 3}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 3}),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "num_warps": 8, "num_stages": 4}),
    ],
    key=["SEQ_LEN_Q", "SEQ_LEN_K", "HEAD_DIM"],
)
@triton.jit
def _prefill_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_K: tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    SCALE:     tl.constexpr,
    CAUSAL:    tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
):
    """
    Prefill blocked attention.
    Each program: one (batch, head, q_block) — processes BLOCK_M query tokens.
    Iterates over K/V in BLOCK_N chunks with online softmax.
    """
    batch_idx  = tl.program_id(0)
    head_idx   = tl.program_id(1)
    q_block    = tl.program_id(2)

    q_start = q_block * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, HEAD_DIM)

    Q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    K_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    O_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # Load Q block [BLOCK_M, HEAD_DIM]
    q_mask  = q_offs < SEQ_LEN_Q
    q_ptrs  = Q_base + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd
    q_block_data = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    q_block_data = q_block_data * SCALE

    # Running softmax state
    m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # Determine how many K blocks to process (causal: only up to q_start + BLOCK_M)
    kv_end = tl.cdiv(
        (q_start + BLOCK_M) if CAUSAL else SEQ_LEN_K,
        BLOCK_N
    )

    for k_block in range(kv_end):
        k_start = k_block * BLOCK_N
        k_offs  = k_start + tl.arange(0, BLOCK_N)
        k_mask  = k_offs < SEQ_LEN_K

        # Load K block [BLOCK_N, HEAD_DIM]
        k_ptrs = K_base + k_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_data = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # QK^T: [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_block_data, tl.trans(k_data))  # [BLOCK_M, BLOCK_N]

        # Causal mask: query i can only attend to key j <= i
        if CAUSAL:
            causal_mask = q_offs[:, None] >= k_offs[None, :]
            scores = tl.where(causal_mask & k_mask[None, :], scores, float("-inf"))
        else:
            scores = tl.where(k_mask[None, :], scores, float("-inf"))

        # Online softmax update
        m_new  = tl.maximum(m_i, tl.max(scores, axis=1))   # [BLOCK_M]
        alpha  = tl.exp(m_i - m_new)                        # [BLOCK_M]
        p      = tl.exp(scores - m_new[:, None])            # [BLOCK_M, BLOCK_N]
        l_new  = alpha * l_i + tl.sum(p, axis=1)

        # Load V block [BLOCK_N, HEAD_DIM]
        v_ptrs = V_base + k_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v_data = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # Weighted sum: p [BLOCK_M, BLOCK_N] @ v [BLOCK_N, HEAD_DIM] -> [BLOCK_M, HEAD_DIM]
        acc   = alpha[:, None] * acc + tl.dot(p, v_data)

        m_i = m_new
        l_i = l_new

    # Normalize
    out = acc / l_i[:, None]

    # Write output [BLOCK_M, HEAD_DIM]
    o_ptrs = O_base + q_offs[:, None] * stride_oq + d_offs[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.float16), mask=q_mask[:, None])


def prefill_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Agent-optimized prefill attention.

    Args:
        q: (batch, n_heads, seq_q,   head_dim)
        k: (batch, n_heads, seq_k,   head_dim)
        v: (batch, n_heads, seq_k,   head_dim)
        causal: apply causal mask (True for standard LM prefill)

    Returns:
        o: (batch, n_heads, seq_q, head_dim)
    """
    assert q.is_cuda, "prefill_attention_fwd requires CUDA"
    assert q.dtype == torch.float16

    batch, n_heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    scale = head_dim ** -0.5

    o = torch.empty_like(q)

    grid = lambda meta: (batch, n_heads, triton.cdiv(seq_q, meta["BLOCK_M"]))

    _prefill_attn_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SEQ_LEN_Q = seq_q,
        SEQ_LEN_K = seq_k,
        HEAD_DIM  = head_dim,
        SCALE     = scale,
        CAUSAL    = causal,
    )
    return o


def verify_prefill_kernel(seq_len: int = 128, head_dim: int = 64,
                          n_heads: int = 12) -> dict:
    """Correctness check vs. PyTorch SDPA. Returns error stats."""
    import torch.nn.functional as F
    q = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    out_triton = prefill_attention_fwd(q, k, v, causal=True)
    out_ref    = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True
    ).half()

    diff = (out_triton - out_ref).abs()
    return {
        "max_abs_error":  diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "passed":         diff.max().item() < 1e-2,
    }
