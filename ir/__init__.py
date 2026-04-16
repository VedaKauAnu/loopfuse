from .dialect import (
    Phase, KernelTarget, IRType, TensorType, KVCacheType, TokenSeqType, ScalarType,
    Value, Op, LLMForwardOp, KVWriteOp, KVFuseOp, ToolCallOp,
    SpeculativePrefillOp, PromptConstructOp, ArgmaxOp, ParseActionOp,
    AgentStepOp, AgentProgram,
    GPT2_CONFIG, GPT2_KV_CONFIG, PHI2_CONFIG, PHI2_KV_CONFIG,
)
from .builder import AgentProgramBuilder
from .printer import IRPrinter
