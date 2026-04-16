"""loopfuse/kernels/triton/__init__.py"""
try:
    import triton
    from .decode_attn  import decode_attention_fwd,  verify_decode_kernel
    from .prefill_attn import prefill_attention_fwd, verify_prefill_kernel
    from .fused_ln_qkv import fused_layernorm_qkv
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

def is_available() -> bool:
    return _TRITON_AVAILABLE
