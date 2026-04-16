"""tests/test_builder.py — unit tests for AgentProgramBuilder."""
import pytest
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.ir.dialect import (
    GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget, Phase,
    LLMForwardOp, ToolCallOp, AgentProgram,
)


class TestAgentProgramBuilder:
    def test_build_returns_agent_program(self):
        prog = (AgentProgramBuilder("test", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "search", 100.0, is_first_step=True)
            .build())
        assert isinstance(prog, AgentProgram)

    def test_step_count(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "a", 100.0, is_first_step=True)
            .add_react_step(1, "b",  50.0)
            .add_react_step(2, None)
            .build())
        assert len(prog.steps) == 3

    def test_first_step_is_prefill(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "search", 100.0, is_first_step=True)
            .build())
        step0_llm = prog.steps[0].llm_ops()[0]
        assert step0_llm.phase == Phase.PREFILL

    def test_non_first_step_is_decode(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "search", 100.0, is_first_step=True)
            .add_react_step(1, None)
            .build())
        step1_llm = prog.steps[1].llm_ops()[0]
        assert step1_llm.phase == Phase.DECODE

    def test_no_tool_in_final_step(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "search", 100.0, is_first_step=True)
            .add_react_step(1, None)   # no tool
            .build())
        assert len(prog.steps[1].tool_ops()) == 0

    def test_tool_latency_preserved(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, "search", 250.0, is_first_step=True)
            .build())
        tool = prog.steps[0].tool_ops()[0]
        assert tool.idle_window_ms() == 250.0

    def test_set_target(self):
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, None, is_first_step=True)
            .set_target(KernelTarget.CUDA)
            .build())
        assert prog.target == KernelTarget.CUDA

    def test_kv_state_type_matches_config(self):
        from loopfuse.ir.dialect import KVCacheType
        prog = (AgentProgramBuilder("t", GPT2_CONFIG, GPT2_KV_CONFIG)
            .add_react_step(0, None, is_first_step=True)
            .build())
        assert isinstance(prog.kv_state.type, KVCacheType)
        assert prog.kv_state.type.num_layers == GPT2_KV_CONFIG["num_layers"]
