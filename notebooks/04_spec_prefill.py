"""
Notebook 04: Speculative Prefill Benchmark (H3).
Requires: T4 or A100.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from loopfuse.benchmarks.h3_spec_prefill.run import run

if __name__ == "__main__":
    run(latencies=[50, 100, 150, 250, 500])
