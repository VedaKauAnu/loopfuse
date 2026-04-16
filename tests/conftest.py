"""tests/conftest.py — shared pytest fixtures (CPU-only, no GPU required)."""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "tpu: requires TPU")
    config.addinivalue_line("markers", "slow: slow test >5s")

def pytest_collection_modifyitems(config, items):
    try:
        import torch; has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU")
    skip_tpu = pytest.mark.skip(reason="no TPU")
    for item in items:
        if "gpu" in item.keywords and not has_cuda: item.add_marker(skip_gpu)
        if "tpu" in item.keywords: item.add_marker(skip_tpu)

@pytest.fixture
def gpt2_3step():
    from loopfuse.ir.builder import AgentProgramBuilder
    return (AgentProgramBuilder("t3", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search", 150.0, is_first_step=True)
        .add_react_step(1, "lookup",  80.0)
        .add_react_step(2, None)
        .set_target(KernelTarget.TRITON).build())

@pytest.fixture
def gpt2_5step():
    from loopfuse.ir.builder import AgentProgramBuilder
    return (AgentProgramBuilder("t5", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search",     150.0, is_first_step=True)
        .add_react_step(1, "lookup",      80.0)
        .add_react_step(2, "calculator",  50.0)
        .add_react_step(3, "search",     200.0)
        .add_react_step(4, None)
        .set_target(KernelTarget.TRITON).build())

@pytest.fixture
def full_pass_manager():
    from loopfuse.passes import PassManager, KVFusionPass, SpecPrefillPass, PhaseSelectPass
    return PassManager([KVFusionPass(min_prefix_len=4), SpecPrefillPass(), PhaseSelectPass()])
