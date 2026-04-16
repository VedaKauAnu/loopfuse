"""
loopfuse/runtime/kv_pool.py

Ring-buffer KV cache pool for LoopFuse agent programs.

Standard practice: allocate a fresh KV cache per inference call.
LoopFuse practice: one persistent ring buffer, shared across all steps.

The pool pre-allocates the maximum sequence budget at startup:
    bytes = num_layers * 2 * num_heads * head_dim * max_seq_len * bytes_per_element

For GPT-2 (12L, 12H, 64D, 1024 seq, FP16):
    = 12 * 2 * 12 * 64 * 1024 * 2 = 37.7 MB

For Phi-2 (32L, 32H, 80D, 2048 seq, BF16):
    = 32 * 2 * 32 * 80 * 2048 * 2 = 671 MB

The pool also manages two CUDA streams:
    compute_stream: LLM forward passes
    prefetch_stream: async KV slot zeroing (overlaps with tool I/O)
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch


class KVPool:
    """
    Persistent ring-buffer KV cache. Pre-allocated at init, never freed.

    Usage:
        pool = KVPool.from_config(GPT2_KV_CONFIG, device="cuda")

        # Start a new agent trajectory
        state = pool.allocate_slot()

        # After each LLM forward pass
        pool.write(state, step_id, new_keys, new_values)

        # Query KV for attention
        k, v = pool.read(state, layer=0, seq_start=0, seq_end=state.current_len)

        # Release slot when trajectory ends
        pool.release(state)
    """

    def __init__(
        self,
        num_layers:  int,
        num_heads:   int,
        head_dim:    int,
        max_seq_len: int,
        max_slots:   int = 8,   # number of concurrent agent trajectories
        dtype:       torch.dtype = torch.float16,
        device:      str = "cuda",
    ):
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.max_slots   = max_slots
        self.dtype       = dtype
        self.device      = device

        # Pre-allocate: [max_slots, num_layers, num_heads, max_seq_len, head_dim]
        shape = (max_slots, num_layers, num_heads, max_seq_len, head_dim)
        self.keys   = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

        # Slot management
        self._free_slots: list = list(range(max_slots))
        self._active: Dict[int, "KVState"] = {}

        # Two CUDA streams: compute stays hot, prefetch runs in background
        if device == "cuda":
            self.compute_stream  = torch.cuda.Stream(device=device, priority=0)
            self.prefetch_stream = torch.cuda.Stream(device=device, priority=-1)
        else:
            self.compute_stream  = None
            self.prefetch_stream = None

        # Metrics
        self._write_count  = 0
        self._bytes_written = 0

    @classmethod
    def from_config(cls, kv_config: dict, device: str = "cuda",
                    max_slots: int = 8) -> "KVPool":
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32, "int8": torch.int8}
        dtype = dtype_map.get(kv_config.get("dtype", "fp16"), torch.float16)
        return cls(
            num_layers  = kv_config["num_layers"],
            num_heads   = kv_config["num_heads"],
            head_dim    = kv_config["head_dim"],
            max_seq_len = kv_config["max_seq_len"],
            dtype       = dtype,
            device      = device,
            max_slots   = max_slots,
        )

    def allocate_slot(self) -> "KVState":
        """Allocate a ring-buffer slot for a new agent trajectory."""
        if not self._free_slots:
            raise RuntimeError(
                f"KVPool exhausted: all {self.max_slots} slots in use. "
                f"Increase max_slots or wait for a trajectory to complete."
            )
        slot_id = self._free_slots.pop(0)
        # Zero out the slot (async, on prefetch stream)
        self._async_zero_slot(slot_id)
        state = KVState(slot_id=slot_id, pool=self)
        self._active[slot_id] = state
        return state

    def release(self, state: "KVState"):
        """Return slot to the pool when a trajectory ends."""
        if state.slot_id in self._active:
            del self._active[state.slot_id]
            self._free_slots.append(state.slot_id)

    def write(
        self,
        state: "KVState",
        new_keys:   torch.Tensor,   # [num_layers, num_heads, seq_new, head_dim]
        new_values: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """
        Write new KV entries for the next token(s) in this trajectory.
        Returns the new sequence length.
        """
        seq_new = new_keys.shape[2]
        if state.current_len + seq_new > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: current_len={state.current_len} + "
                f"seq_new={seq_new} > max_seq_len={self.max_seq_len}"
            )
        s = stream or self.compute_stream
        with torch.cuda.stream(s) if s else _noop_context():
            self.keys  [state.slot_id, :, :, state.current_len:state.current_len + seq_new, :] = new_keys
            self.values[state.slot_id, :, :, state.current_len:state.current_len + seq_new, :] = new_values

        state.current_len += seq_new
        self._write_count  += 1
        self._bytes_written += new_keys.numel() * 2 * 2  # K+V, 2 bytes per FP16

        return state.current_len

    def read(
        self,
        state: "KVState",
        layer:     int,
        seq_start: int = 0,
        seq_end:   Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read KV entries for a given layer and sequence range.
        Returns (keys, values) each of shape [num_heads, seq_range, head_dim].
        """
        seq_end = seq_end or state.current_len
        k = self.keys  [state.slot_id, layer, :, seq_start:seq_end, :]
        v = self.values[state.slot_id, layer, :, seq_start:seq_end, :]
        return k, v

    def read_all_layers(
        self,
        state: "KVState",
        seq_start: int = 0,
        seq_end:   Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read KV entries for all layers.
        Returns (keys, values) each of shape [num_layers, num_heads, seq_range, head_dim].
        """
        seq_end = seq_end or state.current_len
        k = self.keys  [state.slot_id, :, :, seq_start:seq_end, :]
        v = self.values[state.slot_id, :, :, seq_start:seq_end, :]
        return k, v

    def _async_zero_slot(self, slot_id: int):
        """Zero out a slot asynchronously on the prefetch stream."""
        s = self.prefetch_stream
        with torch.cuda.stream(s) if s else _noop_context():
            self.keys  [slot_id].zero_()
            self.values[slot_id].zero_()

    def prefetch_next_slot(self, state: "KVState"):
        """
        Called by SpecPrefillPass runtime: zero out the next slot that
        will be needed for speculative prefill, overlapping with tool I/O.
        This is the H3 runtime side of the optimization.
        """
        # Find the next slot we might allocate
        if self._free_slots:
            next_slot = self._free_slots[0]
            self._async_zero_slot(next_slot)

    def hbm_usage_bytes(self) -> int:
        """Total HBM used by this pool."""
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def stats(self) -> dict:
        return {
            "hbm_mb":         round(self.hbm_usage_bytes() / 1e6, 1),
            "active_slots":   len(self._active),
            "free_slots":     len(self._free_slots),
            "writes":         self._write_count,
            "mb_written":     round(self._bytes_written / 1e6, 1),
        }

    def __repr__(self):
        s = self.stats()
        return (f"KVPool(layers={self.num_layers}, heads={self.num_heads}, "
                f"dim={self.head_dim}, seq={self.max_seq_len}, "
                f"slots={self.max_slots}, HBM={s['hbm_mb']}MB)")


class KVState:
    """
    Handle to a single trajectory's KV cache slot.
    Holds a reference back to the pool and tracks current sequence length.
    """
    def __init__(self, slot_id: int, pool: KVPool):
        self.slot_id     = slot_id
        self.pool        = pool
        self.current_len = 0
        self.step_id     = 0

    def advance_step(self):
        self.step_id += 1

    def __repr__(self):
        return f"KVState(slot={self.slot_id}, len={self.current_len}, step={self.step_id})"


class _noop_context:
    """Context manager that does nothing (used when no CUDA stream is available)."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
