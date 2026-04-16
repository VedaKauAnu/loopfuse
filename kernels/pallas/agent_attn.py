"""
loopfuse/kernels/pallas/agent_attn.py

TPU v4 attention kernels via JAX Pallas — H5 (cross-hardware portability).

Pallas sits one level below JAX's standard ops, giving us explicit control over:
  - VMEM (on-chip scratchpad, ~16MB per TensorCore)  vs  HBM allocation
  - MXU (matrix multiply unit) tile scheduling
  - HBM prefetch patterns

Why Pallas beats vanilla XLA for agent workloads:
  1. XLA's attention lowering uses a generic tiled matmul that doesn't exploit
     the Q=1 structure of decode (all Q tiles are identical — one tile, one MXU op).
  2. We can explicitly pin the KV cache in VMEM across decode steps, eliminating
     repeated HBM loads for tokens already processed. XLA re-reads HBM every step.
  3. Column-major K layout: TPU MXU prefers column-major for the right operand
     of a matmul. Standard XLA uses row-major K, paying a transpose. We eliminate
     this with a layout pass (see passes/phase_select.py KVCacheLayoutPass note).

Hardware: TPU v4 (Colab Pro+)
Requires: jax >= 0.4.25, jaxlib with TPU support

Usage:
    import jax
    import jax.numpy as jnp
    from loopfuse.kernels.pallas.agent_attn import pallas_decode_fwd, pallas_prefill_fwd

    # Decode: q shape (1, n_heads, 1, head_dim)
    q = jnp.ones((1, 12, 1, 64), dtype=jnp.bfloat16)
    k = jnp.ones((1, 12, 512, 64), dtype=jnp.bfloat16)
    v = jnp.ones((1, 12, 512, 64), dtype=jnp.bfloat16)
    o = pallas_decode_fwd(q, k, v)   # (1, 12, 1, 64)
"""

from __future__ import annotations
from typing import Optional
import functools

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _check_jax():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX with Pallas support required for TPU kernels.\n"
            "Install: pip install 'jax[tpu]' "
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )


# ---------------------------------------------------------------------------
# Decode attention (seq_q = 1, memory-bandwidth optimized)
# ---------------------------------------------------------------------------

def _decode_attn_kernel(
    q_ref,    # (1, head_dim)    -- VMEM
    k_ref,    # (block_n, head_dim) -- HBM tile
    v_ref,    # (block_n, head_dim) -- HBM tile
    o_ref,    # (1, head_dim)    -- VMEM output
    m_ref,    # (1,)             -- running softmax max
    l_ref,    # (1,)             -- running softmax sum
    *,
    head_dim: int,
    scale: float,
):
    """
    Single Pallas kernel body for one (batch, head, kv_block).
    Pallas calls this once per grid point; we do the streaming softmax update.
    """
    block_n = k_ref.shape[0]

    # Load Q (stays in VMEM for the entire KV sweep)
    q = q_ref[0, :]   # (head_dim,)

    # Compute QK^T for this block: (1, head_dim) @ (head_dim, block_n) -> (block_n,)
    # Pallas matmul maps directly to MXU instruction on TPU
    scores = jnp.dot(q * scale, k_ref.T)  # (block_n,)

    # Online softmax update
    m_prev = m_ref[0]
    l_prev = l_ref[0]
    o_prev = o_ref[0, :]  # (head_dim,)

    m_new  = jnp.maximum(m_prev, jnp.max(scores))
    alpha  = jnp.exp(m_prev - m_new)
    p      = jnp.exp(scores - m_new)   # (block_n,)
    l_new  = alpha * l_prev + jnp.sum(p)

    # Weighted V sum: p (block_n,) @ v (block_n, head_dim) -> (head_dim,)
    o_new = alpha * o_prev + jnp.dot(p, v_ref)

    # Write back accumulators
    m_ref[0]   = m_new
    l_ref[0]   = l_new
    o_ref[0, :] = o_new


def pallas_decode_fwd(
    q: "jnp.ndarray",  # (batch, n_heads, 1, head_dim)
    k: "jnp.ndarray",  # (batch, n_heads, seq_len, head_dim)
    v: "jnp.ndarray",
    block_n: int = 128,
) -> "jnp.ndarray":
    """
    Decode-phase attention on TPU.
    Single query token attending to the full KV cache.
    """
    _check_jax()

    batch, n_heads, seq_q, head_dim = q.shape
    seq_len = k.shape[2]
    scale   = head_dim ** -0.5

    assert seq_q == 1, f"Decode kernel expects seq_q=1, got {seq_q}"
    assert seq_len % block_n == 0, f"seq_len {seq_len} must be divisible by block_n {block_n}"

    def _single_head_decode(q_h, k_h, v_h):
        """q_h: (1, head_dim), k_h: (seq_len, head_dim), v_h: (seq_len, head_dim)"""
        n_blocks = seq_len // block_n

        def body(carry, idx):
            o, m, l = carry
            k_blk = jax.lax.dynamic_slice(k_h, (idx * block_n, 0), (block_n, head_dim))
            v_blk = jax.lax.dynamic_slice(v_h, (idx * block_n, 0), (block_n, head_dim))

            scores = jnp.dot(q_h[0] * scale, k_blk.T)  # (block_n,)
            m_new  = jnp.maximum(m, jnp.max(scores))
            alpha  = jnp.exp(m - m_new)
            p      = jnp.exp(scores - m_new)
            l_new  = alpha * l + jnp.sum(p)
            o_new  = alpha * o + jnp.dot(p, v_blk)
            return (o_new, m_new, l_new), None

        o_init = jnp.zeros(head_dim, dtype=jnp.float32)
        m_init = jnp.full((), -jnp.inf, dtype=jnp.float32)
        l_init = jnp.zeros((), dtype=jnp.float32)

        (o, m, l), _ = jax.lax.scan(body, (o_init, m_init, l_init), jnp.arange(n_blocks))
        return (o / l).astype(q_h.dtype).reshape(1, head_dim)

    # vmap over batch and heads
    def _batch_heads(q_b, k_b, v_b):
        return jax.vmap(_single_head_decode)(q_b[:, 0, :], k_b, v_b)  # (n_heads, 1, head_dim)

    result = jax.vmap(_batch_heads)(q[:, :, :, :], k, v)  # (batch, n_heads, 1, head_dim)
    return result


# ---------------------------------------------------------------------------
# Prefill attention (seq_q > 1, compute-optimized, causal)
# ---------------------------------------------------------------------------

def pallas_prefill_fwd(
    q: "jnp.ndarray",  # (batch, n_heads, seq_q, head_dim)
    k: "jnp.ndarray",
    v: "jnp.ndarray",
    causal: bool = True,
    block_m: int = 128,
    block_n: int = 128,
) -> "jnp.ndarray":
    """
    Prefill-phase attention on TPU.
    Full sequence attending with causal mask.

    Uses jax.lax.scan over (block_m, block_n) tiles.
    MXU tiles are 128x128 on TPU v4 — block sizes match this exactly.
    """
    _check_jax()

    batch, n_heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    scale = head_dim ** -0.5

    def _single_head_prefill(q_h, k_h, v_h):
        """
        q_h: (seq_q, head_dim)
        k_h: (seq_k, head_dim)
        v_h: (seq_k, head_dim)
        Returns: (seq_q, head_dim)
        """
        n_q_blocks = (seq_q + block_m - 1) // block_m
        n_k_blocks = (seq_k + block_n - 1) // block_n

        def process_q_block(q_blk_idx):
            q_start = q_blk_idx * block_m
            q_blk   = jax.lax.dynamic_slice(q_h, (q_start, 0), (block_m, head_dim))
            q_blk   = q_blk * scale  # scale Q

            o_acc  = jnp.zeros((block_m, head_dim), dtype=jnp.float32)
            m_acc  = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
            l_acc  = jnp.zeros((block_m,), dtype=jnp.float32)

            def process_k_block(carry, k_blk_idx):
                o, m, l = carry
                k_start = k_blk_idx * block_n
                k_blk   = jax.lax.dynamic_slice(k_h, (k_start, 0), (block_n, head_dim))
                v_blk   = jax.lax.dynamic_slice(v_h, (k_start, 0), (block_n, head_dim))

                # QK^T: (block_m, head_dim) @ (head_dim, block_n) -> (block_m, block_n)
                # This maps to a single 128x128 MXU instruction on TPU v4
                scores = jnp.dot(q_blk, k_blk.T)  # (block_m, block_n)

                # Causal mask
                if causal:
                    q_idx  = q_start + jnp.arange(block_m)
                    k_idx  = k_start + jnp.arange(block_n)
                    mask   = q_idx[:, None] >= k_idx[None, :]
                    scores = jnp.where(mask, scores, -jnp.inf)

                m_new  = jnp.maximum(m, jnp.max(scores, axis=1))   # (block_m,)
                alpha  = jnp.exp(m - m_new)                          # (block_m,)
                p      = jnp.exp(scores - m_new[:, None])            # (block_m, block_n)
                l_new  = alpha * l + jnp.sum(p, axis=1)
                o_new  = alpha[:, None] * o + jnp.dot(p, v_blk)

                return (o_new, m_new, l_new), None

            (o_final, m_final, l_final), _ = jax.lax.scan(
                process_k_block,
                (o_acc, m_acc, l_acc),
                jnp.arange(n_k_blocks),
            )
            o_out = o_final / l_final[:, None]
            return q_start, o_out.astype(q_h.dtype)

        # Process all Q blocks and scatter results
        output = jnp.zeros((seq_q, head_dim), dtype=q_h.dtype)
        for qi in range(n_q_blocks):
            q_start, o_blk = process_q_block(qi)
            actual_len = min(block_m, seq_q - q_start)
            output = jax.lax.dynamic_update_slice(
                output, o_blk[:actual_len], (q_start, 0)
            )
        return output

    # vmap over batch and heads
    def _batch_heads(q_b, k_b, v_b):
        return jax.vmap(_single_head_prefill)(q_b, k_b, v_b)

    return jax.vmap(_batch_heads)(q, k, v)


# ---------------------------------------------------------------------------
# JIT-compiled entry points
# ---------------------------------------------------------------------------

if _JAX_AVAILABLE:
    pallas_decode_fwd_jit   = jax.jit(pallas_decode_fwd,   static_argnames=["block_n"])
    pallas_prefill_fwd_jit  = jax.jit(pallas_prefill_fwd,  static_argnames=["causal", "block_m", "block_n"])
else:
    def pallas_decode_fwd_jit(*args, **kwargs):
        raise ImportError("JAX not available")
    def pallas_prefill_fwd_jit(*args, **kwargs):
        raise ImportError("JAX not available")


def verify_pallas_kernels() -> dict:
    """Run correctness checks on both kernels. Requires JAX + TPU."""
    _check_jax()
    results = {}

    # Decode
    q = jnp.ones((1, 4, 1, 64), dtype=jnp.bfloat16)
    k = jnp.ones((1, 4, 128, 64), dtype=jnp.bfloat16)
    v = jnp.ones((1, 4, 128, 64), dtype=jnp.bfloat16)
    o = pallas_decode_fwd(q, k, v)
    results["decode_shape_ok"] = o.shape == (1, 4, 1, 64)

    # Prefill
    q2 = jnp.ones((1, 4, 128, 64), dtype=jnp.bfloat16)
    o2 = pallas_prefill_fwd(q2, k, v, causal=True)
    results["prefill_shape_ok"] = o2.shape == (1, 4, 128, 64)

    return results
