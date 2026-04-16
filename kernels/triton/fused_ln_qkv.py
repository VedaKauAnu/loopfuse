"""
loopfuse/kernels/triton/fused_ln_qkv.py

Fused LayerNorm + QKV projection kernel.

Eliminates one HBM round-trip per transformer layer by keeping the normalized
activations in registers between LayerNorm and the QKV matrix multiply.

Standard (unfused) pipeline:
    x -> LayerNorm -> write to HBM -> read from HBM -> QKV matmul -> output

Fused pipeline:
    x -> LayerNorm (stays in SMEM/registers) -> QKV matmul -> output

HBM bytes saved per layer per step:
    seq_len * d_model * 2 bytes (FP16 normalized activations never written)

For GPT-2 (d_model=768, seq=1): 768 * 2 = 1.5KB per layer saved.
For Phi-2 (d_model=2560, seq=128 prefill): 2560 * 128 * 2 = 655KB per layer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ln_qkv_kernel(
    X_ptr,       # input activations [seq, d_model]
    W_ln_ptr,    # LayerNorm weight [d_model]
    B_ln_ptr,    # LayerNorm bias   [d_model]
    W_qkv_ptr,   # QKV weight [3*d_model, d_model]
    Out_ptr,     # output [seq, 3*d_model]
    seq_len,
    d_model:   tl.constexpr,
    d_out:     tl.constexpr,  # 3 * d_model
    eps:       tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    """
    Each program: one sequence position.
    Computes LayerNorm(x) in registers, then multiplies by W_qkv.
    """
    pos = tl.program_id(0)
    if pos >= seq_len:
        return

    d_offs = tl.arange(0, BLOCK_D)

    # Load x [d_model]
    x = tl.load(X_ptr + pos * d_model + d_offs,
                 mask=d_offs < d_model, other=0.0).to(tl.float32)

    # LayerNorm: compute mean and variance in registers
    mean = tl.sum(x, axis=0) / d_model
    var  = tl.sum((x - mean) ** 2, axis=0) / d_model
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Apply LN weight and bias
    w = tl.load(W_ln_ptr + d_offs, mask=d_offs < d_model).to(tl.float32)
    b = tl.load(B_ln_ptr + d_offs, mask=d_offs < d_model).to(tl.float32)
    x_scaled = x_norm * w + b  # [d_model] — stays in registers

    # QKV projection: x_scaled @ W_qkv^T
    # For simplicity we compute one output row at a time
    # Production version would use tensor core MMA here
    for out_col in range(d_out):
        w_col = tl.load(W_qkv_ptr + out_col * d_model + d_offs,
                        mask=d_offs < d_model).to(tl.float32)
        val = tl.sum(x_scaled * w_col, axis=0)
        tl.store(Out_ptr + pos * d_out + out_col, val.to(tl.float16))


def fused_layernorm_qkv(
    x:     torch.Tensor,  # (seq, d_model)
    w_ln:  torch.Tensor,  # (d_model,)
    b_ln:  torch.Tensor,  # (d_model,)
    w_qkv: torch.Tensor,  # (3*d_model, d_model)
    eps:   float = 1e-5,
) -> torch.Tensor:
    """
    Fused LayerNorm + QKV projection.
    Returns: (seq, 3*d_model) QKV activations.
    """
    assert x.is_cuda
    seq_len, d_model = x.shape
    d_out = w_qkv.shape[0]  # 3 * d_model

    out = torch.empty(seq_len, d_out, dtype=torch.float16, device=x.device)

    # Round d_model up to next power of 2 for BLOCK_D
    block_d = triton.next_power_of_2(d_model)

    _fused_ln_qkv_kernel[(seq_len,)](
        x.to(torch.float16), w_ln, b_ln, w_qkv, out,
        seq_len,
        d_model = d_model,
        d_out   = d_out,
        eps     = eps,
        BLOCK_D = block_d,
        num_warps = 4 if d_model <= 1024 else 8,
    )
    return out
