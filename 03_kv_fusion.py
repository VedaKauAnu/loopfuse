"""
Notebook 03: KV Cache Fusion Benchmark (H2).
Requires: T4 (partial) or A100 (full).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from loopfuse.benchmarks.h2_kv_fusion.run import run

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Sweep prefix lengths and step counts
    for n_steps, prefix_len in [(3, 32), (5, 64), (10, 64)]:
        run(n_steps=n_steps, seq_new=128, prefix_len=prefix_len, device=device)
