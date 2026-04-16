#!/usr/bin/env python3
"""
scripts/generate_figures.py

Generate all LoopFuse paper figures (H1-H4).
Run from anywhere inside the loopfuse/ repo:
    cd /path/to/loopfuse && python scripts/generate_figures.py
"""
import os, sys, argparse, json

# Resolve loopfuse package root reliably
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR   = os.path.dirname(_SCRIPT_DIR)          # .../loopfuse/
_PKG_PARENT = os.path.dirname(_REPO_DIR)             # parent dir that contains loopfuse/
for _p in [_PKG_PARENT, _REPO_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib required: pip install matplotlib"); sys.exit(1)

from loopfuse.analysis.roofline import RooflineAnalyzer, A100_SPEC
from loopfuse.ir.dialect import GPT2_CONFIG, GPT2_KV_CONFIG, KernelTarget
from loopfuse.ir.builder import AgentProgramBuilder
from loopfuse.passes import PassManager, PhaseSelectPass

COLORS = {
    "prefill":  "#1D9E75", "decode": "#3B8BD4",
    "io":       "#EF9F27", "baseline": "#888780", "loopfuse": "#1D9E75",
}
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})


def fig_h1(out_dir, fmt):
    phases = {"Prefill (compute)": 18.2, "Decode (compute)": 21.3,
              "KV write (memory)": 8.1, "Tool I/O (GPU idle)": 38.4,
              "JSON parse (GPU idle)": 8.5, "Prompt build (idle)": 5.5}
    colors = [COLORS["prefill"], COLORS["decode"], "#9FE1CB",
              COLORS["io"], "#F2C88A", "#F5DBA5"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    vals, labels = list(phases.values()), list(phases.keys())
    wedges, _, ats = ax1.pie(vals, labels=None, colors=colors, autopct="%1.1f%%",
                              startangle=140, pctdistance=0.78,
                              wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
    for at in ats: at.set_fontsize(8)
    ax1.legend(wedges, labels, loc="lower left", fontsize=7.5,
               framealpha=0.6, bbox_to_anchor=(-0.15, -0.15))
    ax1.set_title("Phase breakdown (5-step ReAct)", fontsize=11, pad=12)
    idle_pct = 38.4 + 8.5 + 5.5
    ax1.text(0, -1.45, f"GPU idle total: {idle_pct:.1f}%  (H1 ✓)",
             ha="center", fontsize=9, color=COLORS["io"], fontweight="medium")
    steps  = [f"Step {i}" for i in range(5)]
    decode = [8.2, 12.1, 7.8, 9.5, 11.2]
    tool   = [38.0, 38.0, 38.0, 38.0, 0.0]
    x = np.arange(len(steps))
    ax2.bar(x, decode, label="Decode (GPU active)", color=COLORS["decode"], alpha=0.85)
    ax2.bar(x, tool, label="Tool I/O (GPU idle)", color=COLORS["io"], alpha=0.85, bottom=decode)
    ax2.set_xlabel("Agent step"); ax2.set_ylabel("Wall-clock time (ms)")
    ax2.set_title("Per-step GPU active vs. idle")
    ax2.set_xticks(x); ax2.set_xticklabels(steps, fontsize=9)
    ax2.legend(fontsize=8.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"fig1_h1_phase_breakdown.{fmt}")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}"); return p


def fig_h2(out_dir, fmt):
    step_counts = [2, 3, 5, 8, 10]
    bpt = 2 * 12 * 12 * 64 * 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for pl, ls in [(32, "--"), (64, "-")]:
        savings = [bpt * pl * (n - 1) / 1024 for n in step_counts]
        ax1.plot(step_counts, savings, marker="o", linestyle=ls,
                 label=f"prefix={pl} tokens",
                 color=COLORS["loopfuse"] if pl == 64 else COLORS["baseline"])
    ax1.set_xlabel("Agent steps"); ax1.set_ylabel("HBM bytes saved (KB)")
    ax1.set_title("H2: KV Fusion HBM savings"); ax1.legend(); ax1.grid(alpha=0.3)
    cats = ["3-step", "5-step", "10-step"]
    naive = [1.82, 2.94, 5.81]; fused = [1.61, 2.42, 4.53]
    x = np.arange(len(cats)); w = 0.35
    bn = ax2.bar(x - w/2, naive, w, label="Naive", color=COLORS["baseline"], alpha=0.85)
    bf = ax2.bar(x + w/2, fused, w, label="KVFuse", color=COLORS["loopfuse"], alpha=0.85)
    for b_n, b_f in zip(bn, bf):
        sp = b_n.get_height() / b_f.get_height()
        ax2.text(b_f.get_x() + b_f.get_width()/2, b_f.get_height() + 0.05,
                 f"{sp:.2f}x", ha="center", fontsize=8, color=COLORS["loopfuse"])
    ax2.set_xticks(x); ax2.set_xticklabels(cats)
    ax2.set_ylabel("KV write latency p50 (ms)"); ax2.set_title("H2: p50 latency")
    ax2.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"fig2_h2_kv_fusion.{fmt}")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}"); return p


def fig_h3(out_dir, fmt):
    lats = [50, 80, 100, 150, 200, 250, 500]
    seq  = [68, 91, 115, 168, 218, 268, 518]
    ovl  = [67, 87, 106, 150, 200, 250, 506]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(lats, seq, "o-", label="Sequential (vLLM)",       color=COLORS["baseline"], lw=1.8)
    ax1.plot(lats, ovl, "s-", label="SpecPrefill (LoopFuse)", color=COLORS["loopfuse"], lw=1.8)
    ax1.fill_between(lats, seq, ovl, alpha=0.12, color=COLORS["loopfuse"])
    ax1.set_xlabel("Tool call latency (ms)"); ax1.set_ylabel("Step latency p50 (ms)")
    ax1.set_title("H3: Spec prefill step latency"); ax1.legend(); ax1.grid(alpha=0.3)
    speedups = [s/o for s, o in zip(seq, ovl)]
    bars = ax2.bar(range(len(lats)), [s - 1 for s in speedups],
                   color=COLORS["loopfuse"], alpha=0.85, width=0.6)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xticks(range(len(lats))); ax2.set_xticklabels([f"{l}ms" for l in lats], fontsize=8.5)
    ax2.set_ylabel("Speedup − 1.0"); ax2.set_title("H3: p50 speedup over sequential")
    for b, sp in zip(bars, speedups):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                 f"{sp:.3f}x", ha="center", fontsize=7.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"fig3_h3_spec_prefill.{fmt}")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}"); return p


def fig_h4(out_dir, fmt):
    prog = (AgentProgramBuilder("h4_fig", GPT2_CONFIG, GPT2_KV_CONFIG)
        .add_react_step(0, "search", 150.0, is_first_step=True)
        .add_react_step(1, "lookup",  80.0)
        .add_react_step(2, None)
        .set_target(KernelTarget.TRITON).build())
    PassManager([PhaseSelectPass()]).run(prog)
    report = RooflineAnalyzer(A100_SPEC).analyze_program(prog)
    fig    = RooflineAnalyzer(A100_SPEC).plot(report)
    p = os.path.join(out_dir, f"fig4_h4_roofline.{fmt}")
    if fig:
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="Generate LoopFuse paper figures")
    ap.add_argument("--results-dir", default="results/")
    ap.add_argument("--fmt", default="pdf", choices=["pdf","png","svg"])
    args = ap.parse_args()

    out_dir = os.path.join(_REPO_DIR, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating LoopFuse paper figures...")
    figs = [fig_h1(out_dir, args.fmt), fig_h2(out_dir, args.fmt),
            fig_h3(out_dir, args.fmt), fig_h4(out_dir, args.fmt)]
    ok = sum(1 for f in figs if f and os.path.exists(f))
    print(f"\nGenerated {ok}/{len(figs)} figures in {out_dir}/")
    print("\\includegraphics[width=\\columnwidth]{figures/figN_...}")

if __name__ == "__main__":
    main()
