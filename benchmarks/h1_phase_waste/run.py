"""
benchmarks/h1_phase_waste/run.py

H1 Benchmark: Phase waste profiling — the motivating measurement.

Runs a complete instrumented ReAct loop and measures per-phase wall-clock
time breakdown. This is the first experiment to run; its result (GPU idle > 30%)
motivates H2, H3, H4.

Hardware: T4 (free) or A100 (Pro+). CPU fallback with mock timings.
Expected result: GPU idle fraction > 30% across all tool latency configs.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import json, time, argparse
import torch

# ── mock forward passes ─────────────────────────────────────────────────────

def _sync(device):
    if device == "cuda": torch.cuda.synchronize()

def prefill_ms(seq_len=64, d_model=768, n_layers=12, device="cuda"):
    t0 = time.perf_counter()
    x = torch.randn(seq_len, d_model, dtype=torch.float16, device=device)
    w = torch.randn(d_model, d_model * 4, dtype=torch.float16, device=device)
    for _ in range(n_layers):
        x = torch.mm(x, w[:, :d_model]) + x
    _sync(device)
    return (time.perf_counter() - t0) * 1000

def decode_ms(d_model=768, n_layers=12, device="cuda"):
    t0 = time.perf_counter()
    x = torch.randn(1, d_model, dtype=torch.float16, device=device)
    w = torch.randn(d_model, d_model, dtype=torch.float16, device=device)
    for _ in range(n_layers):
        x = torch.mm(x, w) + x
    _sync(device)
    return (time.perf_counter() - t0) * 1000

def kv_write_ms(n_layers=12, n_heads=12, head_dim=64, device="cuda"):
    t0 = time.perf_counter()
    _ = torch.randn(n_layers, 2, n_heads, 1, head_dim, dtype=torch.float16, device=device)
    _sync(device)
    return (time.perf_counter() - t0) * 1000

def tool_io_ms(latency_ms):
    time.sleep(latency_ms / 1000)
    return latency_ms

def json_parse_ms():
    t0 = time.perf_counter()
    data = json.dumps({"result": "x" * 200, "meta": list(range(50))})
    json.loads(data)
    return (time.perf_counter() - t0) * 1000

def prompt_construct_ms(n_tokens=64):
    t0 = time.perf_counter()
    " ".join(["token"] * n_tokens)
    return (time.perf_counter() - t0) * 1000

# ── instrumented loop ────────────────────────────────────────────────────────

def run_one_config(n_steps, tool_latency, n_decode_tokens, device, system_prompt_len=128):
    phases = {k: 0.0 for k in [
        "prefill", "decode", "kv_write", "tool_io", "json_parse", "prompt_build"
    ]}

    for step in range(n_steps):
        # OBSERVE
        phases["prompt_build"] += prompt_construct_ms(
            system_prompt_len if step == 0 else 32)

        # REASON — prefill on first step
        if step == 0:
            phases["prefill"] += prefill_ms(system_prompt_len, device=device)

        # REASON — decode N tokens
        for _ in range(n_decode_tokens):
            phases["decode"] += decode_ms(device=device)

        phases["kv_write"] += kv_write_ms(device=device)

        # ACT — tool call (all but last step)
        if step < n_steps - 1:
            phases["json_parse"] += json_parse_ms()
            phases["tool_io"]    += tool_io_ms(tool_latency)

    total = sum(phases.values())
    idle  = phases["tool_io"] + phases["json_parse"] + phases["prompt_build"]
    return {
        "n_steps":      n_steps,
        "tool_latency": tool_latency,
        "device":       device,
        "phases":       phases,
        "total_ms":     total,
        "idle_ms":      idle,
        "idle_pct":     100.0 * idle / total,
        "h1_confirmed": idle / total > 0.30,
    }

def print_result(r):
    print(f"\n  n_steps={r['n_steps']}  tool_latency={r['tool_latency']}ms  "
          f"device={r['device']}")
    print(f"  {'Phase':<22} {'ms':>8}  {'%':>7}")
    print("  " + "-" * 40)
    for name, ms in sorted(r["phases"].items(), key=lambda x: -x[1]):
        pct = 100 * ms / r["total_ms"]
        bar = "█" * int(pct / 4)
        print(f"  {name:<22} {ms:>7.1f}ms {pct:>6.1f}%  {bar}")
    print("  " + "-" * 40)
    print(f"  GPU idle total:        {r['idle_pct']:>6.1f}%  "
          f"({'H1 ✓ CONFIRMED' if r['h1_confirmed'] else 'H1 ✗ not confirmed'})")

# ── entry point ──────────────────────────────────────────────────────────────

def run(n_steps=5, n_decode_tokens=20, device=None, save_json=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 55)
    print("  LoopFuse H1: Phase Waste Profiling")
    print("=" * 55)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Device: CPU (mock timings — not representative of real GPU)")

    # Warmup
    if device == "cuda":
        print("  Warming up... ", end="", flush=True)
        for _ in range(5):
            prefill_ms(32, device=device)
            decode_ms(device=device)
        print("done")

    results = []
    print(f"\n  Sweeping tool latency: 50ms → 500ms")
    print(f"  {'Latency':>8}  {'Idle%':>7}  H1")
    print("  " + "-" * 25)

    for lat in [50, 100, 150, 250, 500]:
        r = run_one_config(n_steps, lat, n_decode_tokens, device)
        results.append(r)
        h1 = "✓" if r["h1_confirmed"] else "✗"
        print(f"  {lat:>7}ms  {r['idle_pct']:>6.1f}%  {h1}")

    # Detailed breakdown for the 150ms case (paper's primary result)
    r_150 = next(r for r in results if r["tool_latency"] == 150)
    print_result(r_150)

    confirmed = sum(1 for r in results if r["h1_confirmed"])
    print(f"\n  H1 confirmed for {confirmed}/{len(results)} latency configs")

    if save_json:
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {save_json}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps",         type=int,   default=5)
    p.add_argument("--n-decode-tokens", type=int,   default=20)
    p.add_argument("--device",          type=str,   default=None)
    p.add_argument("--save-json",       type=str,   default="results/h1_results.json")
    args = p.parse_args()
    os.makedirs("results", exist_ok=True)
    run(args.n_steps, args.n_decode_tokens, args.device, args.save_json)
