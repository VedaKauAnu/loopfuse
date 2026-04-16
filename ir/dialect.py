"""
loopfuse/ir/dialect.py

The LoopFuse Agent IR dialect.

This is the core novelty of the project: a lightweight SSA-style intermediate
representation that treats the *agent loop* as the unit of compilation.
Every optimization pass in the system operates on this IR.

Design principles:
  - Pure Python: no MLIR/LLVM dependency, runs on Colab
  - SSA values: each op produces named, typed results
  - Phase-annotated: every op carries a Phase tag (PREFILL/DECODE/IO/FUSE)
  - Target-aware: ops carry KernelTarget (TRITON/CUDA/PALLAS/EAGER)
  - Pass-friendly: programs are mutable graphs; passes annotate in-place

Core IR structure:
  AgentProgram
    KVState (shared across all steps)
    List[AgentStepOp]
      observe_region: List[Op]   -- process incoming observation
      reason_region:  List[Op]   -- LLM forward pass(es)
      act_region:     List[Op]   -- emit action / tool call
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Phase and target enums
# ---------------------------------------------------------------------------

class Phase(Enum):
    """
    Execution phase tag. Primary axis for kernel selection and overlap scheduling.

    PREFILL  -- compute-bound: long prompt processing. Above roofline ridge point.
    DECODE   -- memory-bound: single-token generation. Below ridge point.
    IO       -- GPU-idle: tool calls, JSON parse, prompt construction.
    FUSE     -- compiler-inserted ops overlapping two phases.
    """
    PREFILL = "prefill"
    DECODE  = "decode"
    IO      = "io"
    FUSE    = "fuse"


class KernelTarget(Enum):
    """Hardware backend for code generation."""
    TRITON  = "triton"   # T4 + A100 via OpenAI Triton
    CUDA    = "cuda"     # A100 raw CUDA (CUTLASS/CuTe async pipeline)
    PALLAS  = "pallas"   # TPU v4 via JAX Pallas
    EAGER   = "eager"    # PyTorch eager (baseline, no compilation)


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

class IRType:
    pass


@dataclass
class TensorType(IRType):
    """Dense tensor with static shape."""
    shape: Tuple
    dtype: str  # "bf16" | "fp16" | "fp32" | "int8"

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def bytes(self) -> int:
        bpe = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "int32": 4}
        return self.numel() * bpe.get(self.dtype, 2)

    def __repr__(self):
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.dtype}>"


@dataclass
class KVCacheType(IRType):
    """
    KV cache for all transformer layers.
    Persists across agent steps -- central data structure for H2 (KV fusion).
    """
    num_layers: int
    num_heads:  int
    head_dim:   int
    max_seq_len: int
    dtype: str = "bf16"

    def bytes_per_token(self) -> int:
        bpe = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1}
        return 2 * self.num_layers * self.num_heads * self.head_dim * bpe.get(self.dtype, 2)

    def total_bytes(self) -> int:
        return self.bytes_per_token() * self.max_seq_len

    def __repr__(self):
        return (f"kvcache<L={self.num_layers},H={self.num_heads},"
                f"d={self.head_dim},seq={self.max_seq_len},{self.dtype}>")


@dataclass
class TokenSeqType(IRType):
    max_len:    int
    vocab_size: int

    def __repr__(self):
        return f"tokens<len={self.max_len},vocab={self.vocab_size}>"


@dataclass
class ScalarType(IRType):
    dtype: str

    def __repr__(self):
        return f"scalar<{self.dtype}>"


# ---------------------------------------------------------------------------
# SSA Value
# ---------------------------------------------------------------------------

class Value:
    """SSA value. Every Op result is a Value with a name and type."""

    def __init__(self, name: str, type_: IRType, defining_op=None):
        self.name = name
        self.type = type_
        self.defining_op = defining_op

    def __repr__(self):
        return f"%{self.name}: {self.type}"


# ---------------------------------------------------------------------------
# Op base class
# ---------------------------------------------------------------------------

class Op:
    """
    Base class for all LoopFuse IR operations.

    Subclasses declare specific operands, result types, and attributes.
    Base class handles SSA bookkeeping, phase/target annotation, and
    pass metadata storage.
    """

    def __init__(self, name: str, operands: List[Value],
                 result_types: List[IRType], attrs: Optional[Dict[str, Any]] = None):
        self.id: str = str(uuid.uuid4())[:8]
        self.name: str = name
        self.operands: List[Value] = operands
        self.results: List[Value] = [
            Value(f"{name.replace('.','_')}_{self.id}_{i}", t, self)
            for i, t in enumerate(result_types)
        ]
        self.attrs: Dict[str, Any] = attrs or {}
        self.phase: Optional[Phase] = None
        self.target: Optional[KernelTarget] = None
        self.metadata: Dict[str, Any] = {}  # passes write here

    def result(self, idx: int = 0) -> Value:
        return self.results[idx]

    def is_compiler_inserted(self) -> bool:
        return self.metadata.get("compiler_inserted", False)

    # Override in subclasses for roofline analysis
    def estimated_flops(self) -> int:
        return 0

    def estimated_bytes(self) -> int:
        return 0

    def arithmetic_intensity(self) -> float:
        b = self.estimated_bytes()
        return float("inf") if b == 0 else self.estimated_flops() / b

    def __repr__(self):
        res  = ", ".join(str(r) for r in self.results)
        ops  = ", ".join(f"%{o.name}" for o in self.operands)
        atts = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        args = ", ".join(p for p in [ops, atts] if p)
        ph   = f"  // [{self.phase.value}]" if self.phase else ""
        ci   = " ★" if self.is_compiler_inserted() else ""
        return f"{res} = {self.name}({args}){ph}{ci}"


# ---------------------------------------------------------------------------
# Concrete ops
# ---------------------------------------------------------------------------

class LLMForwardOp(Op):
    """
    Single LLM forward pass.

    Phase=PREFILL for long-prompt processing (compute-bound, above ridge).
    Phase=DECODE for single-token generation (memory-bound, below ridge).

    This distinction drives H4: a single general kernel cannot be optimal
    for both phases simultaneously.
    """

    def __init__(self, input_tokens: Value, kv_state: Value,
                 model_config: Dict[str, Any],
                 phase: Phase = Phase.DECODE, seq_len: int = 1):
        vocab = model_config["vocab_size"]
        super().__init__("llm.forward", [input_tokens, kv_state],
                         [TensorType((seq_len, vocab), "bf16")],
                         dict(model_config))
        self.phase = phase
        self.model_config = model_config
        self.seq_len = seq_len

    def estimated_flops(self) -> int:
        d = self.model_config.get("d_model", 768)
        L = self.model_config.get("num_layers", 12)
        # 2 * seq * L * (attn: 4d^2 + ffn: 8d^2)
        return 2 * self.seq_len * L * 12 * d * d

    def estimated_bytes(self) -> int:
        d = self.model_config.get("d_model", 768)
        L = self.model_config.get("num_layers", 12)
        weight_bytes = 2 * L * 12 * d * d * 2  # BF16
        kv_t = self.operands[1].type
        kv_bytes = kv_t.bytes_per_token() * self.seq_len if isinstance(kv_t, KVCacheType) else 0
        return weight_bytes + kv_bytes


class KVWriteOp(Op):
    """Write new KV entries to cache after a forward pass. Phase: IO."""

    def __init__(self, kv_state: Value, new_keys: Value, new_values: Value,
                 seq_pos: int, step_id: int = 0):
        super().__init__("kv.write", [kv_state, new_keys, new_values],
                         [kv_state.type], {"seq_pos": seq_pos, "step_id": step_id})
        self.phase = Phase.IO
        self.metadata["step_id"] = step_id

    def estimated_bytes(self) -> int:
        k = self.operands[1].type
        v = self.operands[2].type
        return (k.bytes() if isinstance(k, TensorType) else 0) + \
               (v.bytes() if isinstance(v, TensorType) else 0)


class KVFuseOp(Op):
    """
    COMPILER-GENERATED — H2 optimization.

    Replaces N separate KVWriteOps that share a common prefix with a single
    batched write. Only the suffix (unique to each step) is written per-step.

    Emitted by: passes/kv_fusion.py :: KVFusionPass
    Baseline:   SGLang RadixAttention (runtime LRU prefix caching)
    Metric:     HBM bandwidth (Nsight Compute) + KV cache hit rate
    """

    def __init__(self, kv_state: Value, steps_to_fuse: List[int],
                 prefix_len: int, replaced_writes: List[KVWriteOp]):
        super().__init__("kv.fuse", [kv_state], [kv_state.type],
                         {"steps": steps_to_fuse, "prefix_len": prefix_len,
                          "n_fused": len(steps_to_fuse)})
        self.phase = Phase.FUSE
        self.metadata["compiler_inserted"] = True
        self.metadata["fused_steps"] = steps_to_fuse
        self.metadata["shared_prefix_tokens"] = prefix_len
        self.metadata["replaced_writes"] = replaced_writes
        if replaced_writes:
            bytes_each = replaced_writes[0].estimated_bytes()
            self.metadata["hbm_bytes_saved"] = bytes_each * prefix_len * (len(steps_to_fuse) - 1)

    def bytes_saved(self) -> int:
        return self.metadata.get("hbm_bytes_saved", 0)


class ToolCallOp(Op):
    """
    External tool invocation. GPU is idle. Overlap opportunity for H3.

    estimated_latency_ms swept from 50-500ms in H3 benchmark.
    """

    def __init__(self, tool_name: str, tool_input: Value,
                 estimated_latency_ms: float = 100.0, step_id: int = 0):
        super().__init__("tool.call", [tool_input],
                         [TokenSeqType(512, 32000)],
                         {"tool": tool_name, "est_latency_ms": estimated_latency_ms,
                          "step_id": step_id})
        self.phase = Phase.IO
        self.metadata["gpu_idle"] = True
        self.metadata["io_latency_ms"] = estimated_latency_ms
        self.tool_name = tool_name

    def idle_window_ms(self) -> float:
        return self.metadata["io_latency_ms"]


class SpeculativePrefillOp(Op):
    """
    COMPILER-GENERATED — H3 optimization.

    Inserted by SpecPrefillPass into the idle window of a ToolCallOp.
    Overlaps GPU prefill compute with tool I/O.

    Emitted by: passes/spec_prefill.py :: SpecPrefillPass
    Only emitted when: idle_window_ms > estimated_prefill_ms
    Baseline:   vLLM sequential (no overlap)
    Metric:     p50/p99 step latency (ms)
    """

    def __init__(self, speculative_tokens: Value, kv_state: Value,
                 model_config: Dict[str, Any], confidence: float,
                 overlapping_tool: Optional[ToolCallOp] = None):
        vocab = model_config["vocab_size"]
        super().__init__("llm.spec_prefill", [speculative_tokens, kv_state],
                         [TensorType((1, vocab), "bf16")], dict(model_config))
        self.phase = Phase.PREFILL
        self.metadata["compiler_inserted"] = True
        self.metadata["confidence"] = confidence
        self.metadata["overlap_with"] = "tool.call"
        if overlapping_tool is not None:
            self.metadata["overlap_tool_id"] = overlapping_tool.id
            self.metadata["available_window_ms"] = overlapping_tool.idle_window_ms()


class PromptConstructOp(Op):
    """Build next-step prompt. Phase: IO (CPU, GPU idle). Measured in H1."""

    def __init__(self, observation: Value, memory: Value, template: str):
        super().__init__("prompt.construct", [observation, memory],
                         [TokenSeqType(2048, 32000)], {"template": template})
        self.phase = Phase.IO
        self.metadata["gpu_idle"] = True


class ArgmaxOp(Op):
    """Greedy token selection from logits. Phase: DECODE."""

    def __init__(self, logits: Value):
        super().__init__("argmax", [logits], [ScalarType("int32")])
        self.phase = Phase.DECODE


class ParseActionOp(Op):
    """Parse model output into structured action. Phase: IO. Measured in H1."""

    def __init__(self, tokens: Value, action_schema: str):
        super().__init__("parse.action", [tokens],
                         [TokenSeqType(256, 32000)],
                         {"schema": action_schema})
        self.phase = Phase.IO
        self.metadata["gpu_idle"] = True


# ---------------------------------------------------------------------------
# AgentStepOp
# ---------------------------------------------------------------------------

class AgentStepOp(Op):
    """
    One complete agent loop iteration: Observe → Reason → Act.

    Owns three sub-regions. Passes traverse across region boundaries to find
    inter-phase optimization opportunities (the key compiler insight).
    """

    def __init__(self, step_id: int,
                 observe_ops: Optional[List[Op]] = None,
                 reason_ops:  Optional[List[Op]] = None,
                 act_ops:     Optional[List[Op]] = None):
        super().__init__("agent.step", [], [], {"step_id": step_id})
        self.step_id = step_id
        self.observe_region: List[Op] = observe_ops or []
        self.reason_region:  List[Op] = reason_ops  or []
        self.act_region:     List[Op] = act_ops     or []

    def all_ops(self) -> List[Op]:
        return self.observe_region + self.reason_region + self.act_region

    def llm_ops(self)      -> List[LLMForwardOp]: return [o for o in self.all_ops() if isinstance(o, LLMForwardOp)]
    def tool_ops(self)     -> List[ToolCallOp]:   return [o for o in self.act_region if isinstance(o, ToolCallOp)]
    def kv_write_ops(self) -> List[KVWriteOp]:    return [o for o in self.all_ops() if isinstance(o, KVWriteOp)]

    def total_gpu_idle_ms(self) -> float:
        return sum(o.idle_window_ms() for o in self.tool_ops())

    def add_op(self, op: Op, region: str = "reason") -> "AgentStepOp":
        getattr(self, f"{region}_region").append(op)
        return self

    def insert_before(self, target: Op, new_op: Op) -> bool:
        for reg in [self.observe_region, self.reason_region, self.act_region]:
            if target in reg:
                reg.insert(reg.index(target), new_op)
                return True
        return False

    def replace(self, old: Op, new: Op) -> bool:
        for reg in [self.observe_region, self.reason_region, self.act_region]:
            if old in reg:
                reg[reg.index(old)] = new
                return True
        return False

    def remove(self, op: Op) -> bool:
        for reg in [self.observe_region, self.reason_region, self.act_region]:
            if op in reg:
                reg.remove(op)
                return True
        return False

    def __repr__(self):
        return (f"agent.step[{self.step_id}]("
                f"observe={len(self.observe_region)}, "
                f"reason={len(self.reason_region)}, "
                f"act={len(self.act_region)})")


# ---------------------------------------------------------------------------
# AgentProgram
# ---------------------------------------------------------------------------

class AgentProgram:
    """
    Top-level compilation unit. Passes operate on this object.

    Owns: list of AgentStepOps, shared KV cache value, pass annotations,
    model config (for roofline), hardware target.
    """

    def __init__(self, name: str, model_config: Dict[str, Any],
                 kv_cache_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.kv_cache_config = kv_cache_config
        self.steps: List[AgentStepOp] = []
        self.kv_state = Value("kv_cache", KVCacheType(
            num_layers  = kv_cache_config["num_layers"],
            num_heads   = kv_cache_config["num_heads"],
            head_dim    = kv_cache_config["head_dim"],
            max_seq_len = kv_cache_config["max_seq_len"],
            dtype       = kv_cache_config.get("dtype", "bf16"),
        ))
        self.pass_annotations: List[str] = []
        self._target: Optional[KernelTarget] = None

    def add_step(self, step: AgentStepOp) -> "AgentProgram":
        self.steps.append(step); return self

    def set_target(self, target: KernelTarget) -> "AgentProgram":
        self._target = target
        for op in self.all_ops():
            if op.target is None:
                op.target = target
        return self

    @property
    def target(self): return self._target

    def all_ops(self) -> List[Op]:
        ops = []
        for s in self.steps: ops.extend(s.all_ops())
        return ops

    def ops_by_type(self, t: type) -> List[Op]:
        return [o for o in self.all_ops() if isinstance(o, t)]

    def ops_by_phase(self) -> Dict[Phase, List[Op]]:
        res = {p: [] for p in Phase}
        for op in self.all_ops():
            if op.phase: res[op.phase].append(op)
        return res

    def compiler_inserted_ops(self) -> List[Op]:
        return [o for o in self.all_ops() if o.is_compiler_inserted()]

    def total_flops(self) -> int:
        return sum(o.estimated_flops() for o in self.all_ops())

    def total_bytes(self) -> int:
        return sum(o.estimated_bytes() for o in self.all_ops())

    def total_gpu_idle_ms(self) -> float:
        return sum(s.total_gpu_idle_ms() for s in self.steps)

    def add_annotation(self, msg: str):
        self.pass_annotations.append(msg)

    def summary(self) -> Dict[str, Any]:
        by_phase = self.ops_by_phase()
        return {
            "name":                  self.name,
            "target":                self._target.value if self._target else "unset",
            "num_steps":             len(self.steps),
            "total_ops":             len(self.all_ops()),
            "ops_by_phase":          {p.value: len(v) for p, v in by_phase.items()},
            "compiler_inserted_ops": len(self.compiler_inserted_ops()),
            "total_flops":           self.total_flops(),
            "total_bytes":           self.total_bytes(),
            "gpu_idle_ms":           self.total_gpu_idle_ms(),
            "pass_annotations":      self.pass_annotations,
        }

    def __repr__(self):
        return (f"AgentProgram('{self.name}', steps={len(self.steps)}, "
                f"target={self._target})")


# ---------------------------------------------------------------------------
# Canonical model configs (shared across benchmarks and notebooks)
# ---------------------------------------------------------------------------

GPT2_CONFIG = {
    "model_name": "gpt2", "d_model": 768, "num_layers": 12,
    "num_heads": 12, "head_dim": 64, "vocab_size": 50257, "max_seq_len": 1024,
}
GPT2_KV_CONFIG = {
    "num_layers": 12, "num_heads": 12, "head_dim": 64,
    "max_seq_len": 1024, "dtype": "fp16",
}
PHI2_CONFIG = {
    "model_name": "phi-2", "d_model": 2560, "num_layers": 32,
    "num_heads": 32, "head_dim": 80, "vocab_size": 51200, "max_seq_len": 2048,
}
PHI2_KV_CONFIG = {
    "num_layers": 32, "num_heads": 32, "head_dim": 80,
    "max_seq_len": 2048, "dtype": "bf16",
}
