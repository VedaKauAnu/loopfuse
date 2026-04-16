"""loopfuse/kernels/pallas/__init__.py"""
try:
    from .agent_attn import (
        pallas_decode_fwd, pallas_decode_fwd_jit,
        pallas_prefill_fwd, pallas_prefill_fwd_jit,
        verify_pallas_kernels,
    )
    _PALLAS_AVAILABLE = True
except ImportError:
    _PALLAS_AVAILABLE = False

def is_available() -> bool:
    return _PALLAS_AVAILABLE
