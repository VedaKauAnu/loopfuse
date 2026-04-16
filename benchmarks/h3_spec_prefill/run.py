"""
benchmarks/h3_spec_prefill/run.py — H3: Speculative prefill overlap.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import torch
from loopfuse.analysis.stats import BenchmarkResult, compare, print_comparison_table


def sequential_step(tool_ms, d_model=768, n_layers=12, n_decode=20, device="cuda"):
    t0 = time.perf_counter()
    time.sleep(tool_ms / 1000)                              # tool call: GPU idle
    if device == "cuda":
        x = torch.randn(64, d_model, device=device, dtype=torch.float16)
        w = torch.randn(d_model, d_model, device=device, dtype=torch.float16)
        for _ in range(n_layers): x = torch.mm(x, w)       # prefill
        for _ in range(n_decode):
            x2 = torch.randn(1, d_model, device=device, dtype=torch.float16)
            _ = torch.mm(x2, w)                             # decode
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def overlapped_step(tool_ms, d_model=768, n_layers=12, n_decode=20, device="cuda"):
    t0 = time.perf_counter()
    if device == "cuda":
        bg = torch.cuda.Stream(priority=-1)
        w  = torch.randn(d_model, d_model, device=device, dtype=torch.float16)
        with torch.cuda.stream(bg):                         # spec prefill on bg stream
            x = torch.randn(64, d_model, device=device, dtype=torch.float16)
            for _ in range(n_layers): x = torch.mm(x, w)
        time.sleep(tool_ms / 1000)                          # tool call concurrently
        torch.cuda.current_stream().wait_stream(bg)
        for _ in range(n_decode):
            x2 = torch.randn(1, d_model, device=device, dtype=torch.float16)
            _ = torch.mm(x2, w)
        torch.cuda.synchronize()
    else:
        time.sleep(max(tool_ms, 15) / 1000)
    return (time.perf_counter() - t0) * 1000


def run(latencies=None, device=None, n_warmup=10, n_measure=50):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if latencies is None:
        latencies = [50, 100, 150, 250, 500]

    print(f"\nH3: Spec Prefill Overlap  device={device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"  {'Tool lat':>10}  {'Seq p50':>9}  {'Ovlp p50':>9}  {'Speedup':>8}  H3")
    print("  " + "-" * 52)
    results = []
    for lat in latencies:
        for _ in range(n_warmup):
            sequential_step(lat, device=device)
        ts = [sequential_step(lat, device=device) for _ in range(n_measure)]
        to = [overlapped_step(lat, device=device)  for _ in range(n_measure)]
        rs, ro = BenchmarkResult("seq", ts), BenchmarkResult("overlap", to)
        cmp = compare(rs, ro)
        h3 = "✓" if cmp.speedup_p50 > 1.01 and cmp.is_significant else "✗"
        print(f"  {lat:>9}ms  {rs.p50:>8.1f}ms  {ro.p50:>8.1f}ms  {cmp.speedup_p50:>7.2f}x  {h3}")
        results.append({"lat": lat, "speedup": cmp.speedup_p50, "sig": cmp.is_significant})
    confirmed = sum(1 for r in results if r["speedup"] > 1.01 and r["sig"])
    print(f"\n  H3 confirmed for {confirmed}/{len(results)} configs")
    return results

if __name__ == "__main__":
    run()
