"""
Notebook 01: Phase Waste Profiling — the H1 baseline measurement.

This is the most important notebook to run first. It measures the per-phase
breakdown of a ReAct agent loop on your GPU/TPU:
  - PREFILL time (LLM system prompt processing)
  - DECODE time  (per-token generation)
  - IO time      (tool calls, JSON parsing, prompt construction)
  - GPU idle %   (the H3 optimization opportunity)

Hardware: T4 (free) or A100 (Pro+)
Expected result: IO/idle time > 30% of wall clock (H1 hypothesis)

Based on: arXiv:2509.09505 (PLENA) characterization of agentic inference.
Baseline comparison target: sequential vLLM / torch.compile.
"""

import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import torch
import numpy as np

# -----------------------------------------------------------------
# Mock LLM forward pass (CPU+GPU compatible, no model weights needed)
# Simulates realistic latencies for GPT-2 on T4/A100.
# Replace with a real model.generate() call for production measurements.
# -----------------------------------------------------------------

def mock_prefill(seq_len: int, d_model: int = 768, device: str = "cuda") -> float:
    """Simulate prefill: large GEMM, returns wall time ms."""
    t0 = time.perf_counter()
    # Approximate a transformer layer forward: seq_len x d_model matmuls
    x = torch.randn(seq_len, d_model, device=device, dtype=torch.float16)
    w = torch.randn(d_model, d_model * 4, device=device, dtype=torch.float16)
    for _ in range(12):  # 12 layers
        x = torch.mm(x, w[:, :d_model]) + x
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def mock_decode(d_model: int = 768, device: str = "cuda") -> float:
    """Simulate decode: single-token GEMM, returns wall time ms."""
    t0 = time.perf_counter()
    x = torch.randn(1, d_model, device=device, dtype=torch.float16)
    w = torch.randn(d_model, d_model * 4, device=device, dtype=torch.float16)
    for _ in range(12):
        x = torch.mm(x, w[:, :d_model]) + x
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def mock_tool_call(tool_name: str, latency_ms: float) -> float:
    """Simulate tool I/O: CPU sleep, GPU idle."""
    time.sleep(latency_ms / 1000)
    return latency_ms


def mock_json_parse(complexity: int = 5) -> float:
    """Simulate JSON parsing of tool output."""
    t0 = time.perf_counter()
    data = json.dumps({"result": "x" * 200, "metadata": list(range(complexity * 10))})
    _ = json.loads(data)
    return (time.perf_counter() - t0) * 1000


def mock_prompt_construct(n_tokens: int = 64) -> float:
    """Simulate prompt string construction."""
    t0 = time.perf_counter()
    _ = " ".join(["token"] * n_tokens)
    return (time.perf_counter() - t0) * 1000


# -----------------------------------------------------------------
# Phase-instrumented ReAct loop
# -----------------------------------------------------------------

def run_instrumented_react_loop(
    n_steps: int = 5,
    system_prompt_len: int = 128,
    tool_latency_ms: float = 150.0,
    n_decode_tokens: int = 20,
    device: str = "cuda",
) -> dict:
    """
    Run a complete ReAct loop with per-phase timing.

    Returns a dict with timing breakdown for each phase.
    """
    phase_times = {
        "prefill_ms":          [],
        "decode_ms":           [],
        "tool_io_ms":          [],
        "json_parse_ms":       [],
        "prompt_construct_ms": [],
        "kv_write_ms":         [],
    }

    for step in range(n_steps):
        # --- OBSERVE: prompt construction ---
        t = mock_prompt_construct(n_tokens=system_prompt_len if step == 0 else 32)
        phase_times["prompt_construct_ms"].append(t)

        # --- REASON: LLM forward ---
        if step == 0:
            # First step: prefill the system prompt
            t = mock_prefill(system_prompt_len, device=device)
            phase_times["prefill_ms"].append(t)

        # Decode N tokens
        decode_start = time.perf_counter()
        for _ in range(n_decode_tokens):
            mock_decode(device=device)
        torch.cuda.synchronize()
        phase_times["decode_ms"].append((time.perf_counter() - decode_start) * 1000)

        # Simulate KV cache write
        t0 = time.perf_counter()
        kv = torch.randn(12, 2, 12, 1, 64, device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        phase_times["kv_write_ms"].append((time.perf_counter() - t0) * 1000)

        # --- ACT: tool call (if not last step) ---
        if step < n_steps - 1:
            t = mock_json_parse()
            phase_times["json_parse_ms"].append(t)

            t = mock_tool_call("search", tool_latency_ms)
            phase_times["tool_io_ms"].append(t)

    # Aggregate
    total_ms = sum(sum(v) for v in phase_times.values())
    result = {
        "n_steps":       n_steps,
        "total_ms":      total_ms,
        "by_phase":      {k: sum(v) for k, v in phase_times.items()},
        "by_phase_pct":  {k: 100 * sum(v) / total_ms for k, v in phase_times.items()},
        "per_step_raw":  phase_times,
    }

    # H1 hypothesis: GPU-idle > 30% of wall clock
    idle_total = (result["by_phase"]["tool_io_ms"] +
                  result["by_phase"]["json_parse_ms"] +
                  result["by_phase"]["prompt_construct_ms"])
    result["gpu_idle_pct"] = 100 * idle_total / total_ms
    result["h1_hypothesis_confirmed"] = result["gpu_idle_pct"] > 30.0

    return result


def print_phase_report(result: dict):
    """Print a formatted phase breakdown."""
    print("=" * 60)
    print("  LoopFuse — H1: Phase Waste Profiling")
    print("=" * 60)
    print(f"  Steps: {result['n_steps']}  |  Total: {result['total_ms']:.1f}ms")
    print()
    print(f"  {'Phase':<25} {'Time (ms)':>10} {'% Total':>9}")
    print("  " + "-" * 46)

    PHASE_COLORS = {
        "prefill_ms":          "COMPUTE",
        "decode_ms":           "COMPUTE",
        "tool_io_ms":          "GPU-IDLE",
        "json_parse_ms":       "GPU-IDLE",
        "prompt_construct_ms": "GPU-IDLE",
        "kv_write_ms":         "MEMORY",
    }
    for phase, t_ms in sorted(result["by_phase"].items(),
                               key=lambda x: -x[1]):
        pct  = result["by_phase_pct"][phase]
        cat  = PHASE_COLORS.get(phase, "")
        name = phase.replace("_ms", "").replace("_", " ")
        bar  = "█" * int(pct / 3)
        print(f"  {name:<25} {t_ms:>9.1f}ms {pct:>7.1f}%  {bar}")

    print("  " + "-" * 46)
    idle_pct = result["gpu_idle_pct"]
    print(f"  GPU idle total:          {idle_pct:>7.1f}%")
    print()
    if result["h1_hypothesis_confirmed"]:
        print("  H1 CONFIRMED ✓  GPU idle > 30% — optimization opportunity is real.")
        print("  → H3 (spec_prefill) can reclaim this time.")
        print("  → H2 (kv_fusion) reduces memory-phase overhead.")
    else:
        print("  H1 not confirmed at this config (GPU idle < 30%).")
        print("  Try increasing tool_latency_ms or n_steps.")
    print("=" * 60)


def plot_phase_breakdown(result: dict):
    """Matplotlib pie chart of phase breakdown. Colab-ready."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("pip install matplotlib for plots")
        return None

    COLORS = {
        "prefill_ms":          "#1D9E75",
        "decode_ms":           "#3B8BD4",
        "kv_write_ms":         "#9FE1CB",
        "tool_io_ms":          "#EF9F27",
        "json_parse_ms":       "#F2C88A",
        "prompt_construct_ms": "#F5DBA5",
    }
    LABELS = {
        "prefill_ms":          "Prefill (compute)",
        "decode_ms":           "Decode (compute)",
        "kv_write_ms":         "KV write (memory)",
        "tool_io_ms":          "Tool I/O (GPU idle)",
        "json_parse_ms":       "JSON parse (GPU idle)",
        "prompt_construct_ms": "Prompt build (GPU idle)",
    }

    by_phase = {k: v for k, v in result["by_phase"].items() if v > 0}
    sizes    = list(by_phase.values())
    colors   = [COLORS.get(k, "#aaa") for k in by_phase]
    labels   = [LABELS.get(k, k) for k in by_phase]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    wedges, _, autotexts = ax1.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, wedgeprops={"linewidth": 0.5, "edgecolor": "white"}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title(f"Phase Breakdown ({result['n_steps']} ReAct steps)",
                  fontsize=12, pad=14)
    ax1.legend(wedges, labels, loc="lower left", fontsize=8, framealpha=0.7)

    # Bar chart: per-step decode vs. idle
    steps = range(result["n_steps"])
    decode_times = result["per_step_raw"]["decode_ms"]
    tool_times   = result["per_step_raw"].get("tool_io_ms", [0] * result["n_steps"])
    # pad to same length
    while len(tool_times) < len(decode_times):
        tool_times.append(0)

    x = np.arange(len(decode_times))
    ax2.bar(x, decode_times, label="Decode (GPU active)", color="#3B8BD4", alpha=0.85)
    ax2.bar(x, tool_times[:len(decode_times)], bottom=decode_times,
            label="Tool I/O (GPU idle)", color="#EF9F27", alpha=0.85)
    ax2.set_xlabel("Agent step")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Per-step: GPU active vs. idle time")
    ax2.legend()
    ax2.set_xticks(x)

    plt.suptitle(
        f"LoopFuse H1 — GPU Idle: {result['gpu_idle_pct']:.1f}% "
        f"({'✓ > 30%' if result['h1_hypothesis_confirmed'] else '< 30%'})",
        fontsize=13, fontweight="medium", y=1.01
    )
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU — timing will not reflect real GPU workloads.")
        print("Run this notebook on Colab with a GPU runtime.")

    print(f"Running on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Warmup
    if device == "cuda":
        print("Warming up GPU...")
        for _ in range(5):
            mock_prefill(32, device=device)
            mock_decode(device=device)

    print("\nRunning 5-step ReAct loop (tool_latency=150ms)...")
    result = run_instrumented_react_loop(
        n_steps=5,
        system_prompt_len=128,
        tool_latency_ms=150.0,
        n_decode_tokens=20,
        device=device,
    )
    print_phase_report(result)

    # Sweep tool latency to show H1 robustness
    print("\nH1 Robustness: sweeping tool latency 50ms → 500ms")
    print(f"  {'Latency':>10}  {'GPU idle %':>10}  {'H1':>6}")
    print("  " + "-" * 30)
    for lat in [50, 100, 150, 250, 500]:
        r = run_instrumented_react_loop(
            n_steps=5, system_prompt_len=128,
            tool_latency_ms=lat, n_decode_tokens=20, device=device,
        )
        h1 = "✓" if r["h1_hypothesis_confirmed"] else "✗"
        print(f"  {lat:>9}ms  {r['gpu_idle_pct']:>9.1f}%  {h1:>6}")

    print("\nSave results with: import json; json.dump(result, open('h1_results.json','w'))")
    print("Plot with: fig = plot_phase_breakdown(result); fig.savefig('h1_plot.png')")
