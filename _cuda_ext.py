"""
loopfuse/kernels/cuda/_cuda_ext.py

PyTorch extension loader for the LoopFuse CUDA kernels.
Compiles ring_kv_cache.cu and async_prefill.cu on first import.

Usage (A100 Colab, requires nvcc):
    from loopfuse.kernels.cuda import ring_kv_write, async_zero_slot

Requirements:
    - CUDA 12.x
    - SM 80+ (A100)
    - torch 2.x with CUDA support
"""

import os
import torch

_ext = None
_loaded = False

def _load_extension():
    global _ext, _loaded
    if _loaded:
        return _ext

    try:
        from torch.utils.cpp_extension import load
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="loopfuse_cuda_kernels",
            sources=[
                os.path.join(kernel_dir, "ring_kv_cache.cu"),
                os.path.join(kernel_dir, "async_prefill.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_80",
                "--use_fast_math",
                "-std=c++17",
                # Required for cuda::pipeline
                "-I/usr/local/cuda/include",
            ],
            verbose=False,
        )
        _loaded = True
    except Exception as e:
        print(f"[LoopFuse] CUDA extension compile failed: {e}")
        print("[LoopFuse] Falling back to Triton kernels for CUDA ops.")
        _ext = None
        _loaded = True
    return _ext


def is_available() -> bool:
    """True if the CUDA extension compiled successfully."""
    return _load_extension() is not None


def ring_kv_write(
    cache_keys:   torch.Tensor,
    cache_values: torch.Tensor,
    new_keys:     torch.Tensor,
    new_values:   torch.Tensor,
    slot_idx:     int,
    stream:       torch.cuda.Stream = None,
) -> None:
    """
    Write new KV entries into the ring buffer at slot_idx.
    All tensors must be FP16 on CUDA.
    """
    ext = _load_extension()
    if ext is None:
        # Fallback: plain PyTorch copy
        nl, nh, sq, hd = new_keys.shape
        cache_keys [slot_idx, :nl, :nh, :sq, :hd] = new_keys
        cache_values[slot_idx, :nl, :nh, :sq, :hd] = new_values
        return

    s = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
    ext.loopfuse_ring_kv_write(
        cache_keys, cache_values, new_keys, new_values,
        slot_idx,
        new_keys.shape[0],   # num_layers
        new_keys.shape[1],   # num_heads
        new_keys.shape[2],   # seq_new
        new_keys.shape[3],   # head_dim
        cache_keys.shape[3], # max_seq_len
        s,
    )


def async_zero_slot(
    cache_keys:   torch.Tensor,
    cache_values: torch.Tensor,
    next_slot:    int,
    stream:       torch.cuda.Stream = None,
) -> None:
    """
    Zero out the next ring buffer slot asynchronously.
    Called by SpecPrefillPass to prepare the slot during tool I/O.
    """
    ext = _load_extension()
    if ext is None:
        _, nl, nh, ms, hd = cache_keys.shape
        s = stream or torch.cuda.current_stream()
        with torch.cuda.stream(s):
            cache_keys  [next_slot].zero_()
            cache_values[next_slot].zero_()
        return

    s = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
    nl, nh, ms, hd = cache_keys.shape[1:]
    ext.loopfuse_async_zero_slot(
        cache_keys, cache_values, next_slot, nl, nh, ms, hd, s
    )
