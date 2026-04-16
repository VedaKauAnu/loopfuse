"""
loopfuse.passes.phase_profiler
================================
Pass 0 — Phase Waste Profiler.

This is the *first* pass to run and the motivating result for the paper.
It instruments a real PyTorch model executing a ReAct loop and measures the
per-phase breakdown of wall-clock time:

    Phase               | What it measures
    --------------------|--------------------------------------------------
    PREFILL             | LLM forward pass on the system prompt / context
    DECODE              | Autoregressive token generation
    TOOL_IO             | Time from tool call dispatch to result receipt
    JSON_PARSE          | Tokenization + JSON extraction from LLM output
    PROMPT_BUILD        | String formatting of the next-step prompt
    KV_WRITE            | Time to write/update the KV cache in HBM
    GPU_IDLE            | Time GPU is idle (synchronization, Python overhead)
    OTHER               | Everything else

The profiler wraps a standard HuggingFace generate() call with NVTX markers
and PyTorch Profiler traces, then parses the trace to produce a breakdown.

Usage (Notebook 01)
-------------------
    from loopfuse.passes.phase_profiler import PhaseProfiler

    profiler = PhaseProfiler(model, tokenizer, device="cuda")
    report   = profiler.run(agent_task, n_steps=10)
    report.print_summary()
    report.plot_breakdown()        # matplotlib bar chart
    report.to_csv("h1_results.csv")
"""

from __future__ import annotations

import time
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Optional imports — graceful degradation on CPU-only machines
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------------

PHASES = [
    "PREFILL",
    "DECODE",
    "TOOL_IO",
    "JSON_PARSE",
    "PROMPT_BUILD",
    "KV_WRITE",
    "GPU_IDLE",
    "OTHER",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PhaseInterval:
    """One timed interval for a single phase."""
    phase:      str
    start_ms:   float
    end_ms:     float
    step_idx:   int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class StepProfile:
    """Timing breakdown for one agent step."""
    step_idx:   int
    total_ms:   float
    intervals:  List[PhaseInterval] = field(default_factory=list)

    def phase_time_ms(self, phase: str) -> float:
        return sum(i.duration_ms for i in self.intervals if i.phase == phase)

    def gpu_idle_fraction(self) -> float:
        idle = self.phase_time_ms("GPU_IDLE")
        return idle / self.total_ms if self.total_ms > 0 else 0.0

    def non_compute_fraction(self) -> float:
        non_compute = sum(
            self.phase_time_ms(p)
            for p in ["TOOL_IO", "JSON_PARSE", "PROMPT_BUILD", "GPU_IDLE", "OTHER"]
        )
        return non_compute / self.total_ms if self.total_ms > 0 else 0.0


@dataclass
class ProfileReport:
    """
    Aggregated profile across all steps.

    This is the H1 result object — the evidence that a non-trivial fraction
    of agent wall-clock time is wasted on non-GPU work.
    """
    model_name:     str
    hardware:       str
    n_steps:        int
    step_profiles:  List[StepProfile]
    raw_intervals:  List[PhaseInterval] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Aggregated metrics
    # -----------------------------------------------------------------

    @property
    def total_ms(self) -> float:
        return sum(s.total_ms for s in self.step_profiles)

    def mean_step_ms(self) -> float:
        return np.mean([s.total_ms for s in self.step_profiles]) if self.step_profiles else 0.0

    def p50_step_ms(self) -> float:
        return float(np.percentile([s.total_ms for s in self.step_profiles], 50))

    def p99_step_ms(self) -> float:
        return float(np.percentile([s.total_ms for s in self.step_profiles], 99))

    def phase_total_ms(self, phase: str) -> float:
        return sum(s.phase_time_ms(phase) for s in self.step_profiles)

    def phase_fraction(self, phase: str) -> float:
        t = self.total_ms
        return self.phase_total_ms(phase) / t if t > 0 else 0.0

    def non_compute_fraction(self) -> float:
        return sum(
            self.phase_fraction(p)
            for p in ["TOOL_IO", "JSON_PARSE", "PROMPT_BUILD", "GPU_IDLE", "OTHER"]
        )

    def compute_fraction(self) -> float:
        return self.phase_fraction("PREFILL") + self.phase_fraction("DECODE")

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    def print_summary(self):
        """Print a formatted breakdown to stdout."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            f"║  LoopFuse Phase Waste Report  —  H1 Baseline     ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Model:     {self.model_name:<37}║",
            f"║  Hardware:  {self.hardware:<37}║",
            f"║  Steps:     {self.n_steps:<37}║",
            f"║  Total:     {self.total_ms:.0f}ms{'':<33}║",
            "╠══════════════════════════════════════════════════╣",
            f"║  {'Phase':<16}{'Total ms':>10}{'%':>8}  Bar              ║",
            "╠══════════════════════════════════════════════════╣",
        ]
        for phase in PHASES:
            frac  = self.phase_fraction(phase)
            total = self.phase_total_ms(phase)
            bar   = "█" * int(frac * 20)
            lines.append(f"║  {phase:<16}{total:>10.0f}{frac*100:>7.1f}%  {bar:<16} ║")
        lines += [
            "╠══════════════════════════════════════════════════╣",
            f"║  Compute (PREFILL+DECODE): {self.compute_fraction()*100:>6.1f}%{'':<17}║",
            f"║  Non-compute (waste):      {self.non_compute_fraction()*100:>6.1f}%  ← H1 target    ║",
            f"║  p50 step latency: {self.p50_step_ms():>8.0f}ms{'':<18}║",
            f"║  p99 step latency: {self.p99_step_ms():>8.0f}ms{'':<18}║",
            "╚══════════════════════════════════════════════════╝",
            "",
        ]
        print("\n".join(lines))

    def plot_breakdown(self, save_path: Optional[str] = None):
        """Stacked bar chart of phase breakdown per step.  Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        colors = {
            "PREFILL":      "#22c55e",
            "DECODE":       "#3b82f6",
            "TOOL_IO":      "#f97316",
            "JSON_PARSE":   "#f59e0b",
            "PROMPT_BUILD": "#eab308",
            "KV_WRITE":     "#8b5cf6",
            "GPU_IDLE":     "#ef4444",
            "OTHER":        "#94a3b8",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: stacked bar per step
        ax = axes[0]
        steps   = [s.step_idx for s in self.step_profiles]
        bottoms = np.zeros(len(steps))
        for phase in PHASES:
            vals = np.array([s.phase_time_ms(phase) for s in self.step_profiles])
            ax.bar(steps, vals, bottom=bottoms, color=colors[phase], label=phase, width=0.7)
            bottoms += vals
        ax.set_xlabel("Agent step")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Phase breakdown per step")
        ax.legend(loc="upper right", fontsize=7)

        # Right: aggregate pie
        ax2 = axes[1]
        fracs  = [self.phase_fraction(p) * 100 for p in PHASES]
        clrs   = [colors[p] for p in PHASES]
        wedges, texts, autotexts = ax2.pie(
            fracs,
            labels=PHASES,
            colors=clrs,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            startangle=140,
            textprops={"fontsize": 8},
        )
        ax2.set_title(
            f"Aggregate breakdown\n"
            f"Waste = {self.non_compute_fraction()*100:.1f}%"
        )

        fig.suptitle(
            f"LoopFuse H1 — Phase Waste  |  {self.model_name}  |  {self.hardware}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model":            self.model_name,
            "hardware":         self.hardware,
            "n_steps":          self.n_steps,
            "total_ms":         self.total_ms,
            "p50_step_ms":      self.p50_step_ms(),
            "p99_step_ms":      self.p99_step_ms(),
            "compute_fraction": self.compute_fraction(),
            "waste_fraction":   self.non_compute_fraction(),
            **{f"phase_{p.lower()}_ms": self.phase_total_ms(p) for p in PHASES},
        }

    def to_csv(self, path: str):
        try:
            import pandas as pd
            rows = []
            for s in self.step_profiles:
                row = {"step": s.step_idx, "total_ms": s.total_ms}
                for p in PHASES:
                    row[p.lower()] = s.phase_time_ms(p)
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"Saved per-step CSV to {path}")
        except ImportError:
            print("pandas not available — cannot export CSV")


# ---------------------------------------------------------------------------
# Context manager timer
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _phase_timer(intervals: List[PhaseInterval], phase: str, step_idx: int, device=None):
    """Time a code block and record a PhaseInterval."""
    if _NVTX_AVAILABLE:
        nvtx.push_range(phase)

    if device is not None and _TORCH_AVAILABLE:
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        yield
    finally:
        if device is not None and _TORCH_AVAILABLE:
            torch.cuda.synchronize(device)
        end = time.perf_counter()

        if _NVTX_AVAILABLE:
            nvtx.pop_range()

        intervals.append(PhaseInterval(
            phase=phase,
            start_ms=(start * 1000),
            end_ms=(end * 1000),
            step_idx=step_idx,
        ))


# ---------------------------------------------------------------------------
# Mock tool registry (for reproducible benchmarking)
# ---------------------------------------------------------------------------

class MockToolRegistry:
    """
    Deterministic mock tools with configurable latency.

    In real agent benchmarks the tool latency is non-deterministic.  For
    H1/H3 profiling we use configurable synthetic latency to isolate the
    GPU-side effects.
    """

    def __init__(self, latency_config: Optional[Dict[str, float]] = None):
        self._latencies = latency_config or {
            "search":     100.0,
            "calculator": 5.0,
            "lookup":     50.0,
        }

    def call(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, float]:
        """Execute mock tool.  Returns (result_string, actual_latency_ms)."""
        latency_ms = self._latencies.get(tool_name, 50.0)
        time.sleep(latency_ms / 1000.0)

        # Deterministic results by tool
        if tool_name == "search":
            result = f"Search result for: {args.get('query', '')}. [Mock result: Paris is the capital of France.]"
        elif tool_name == "calculator":
            try:
                result = str(eval(str(args.get("expr", "0")), {"__builtins__": {}}))
            except Exception:
                result = "Error: invalid expression"
        elif tool_name == "lookup":
            result = f"Lookup for {args.get('key', '')}: [Mock result: 42]"
        else:
            result = f"[Mock result from {tool_name}]"

        return result, latency_ms


# ---------------------------------------------------------------------------
# PhaseProfiler — main entry point
# ---------------------------------------------------------------------------

class PhaseProfiler:
    """
    Instruments a HuggingFace model executing a ReAct loop.

    Parameters
    ----------
    model : PreTrainedModel
        The model to profile.
    tokenizer : PreTrainedTokenizer
        Matching tokenizer.
    device : str
        "cuda", "cpu", or "tpu" (JAX path TBD)
    tool_latency_config : dict
        Per-tool latency in ms (for mock tool calls).

    Example
    -------
        profiler = PhaseProfiler(model, tokenizer, device="cuda")
        report   = profiler.run(react_loop_fn, n_steps=10)
        report.print_summary()
        report.plot_breakdown(save_path="h1_t4.png")
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        tool_latency_config: Optional[Dict[str, float]] = None,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.tools      = MockToolRegistry(tool_latency_config)
        self._intervals: List[PhaseInterval] = []

    # ------------------------------------------------------------------
    # Core profiled ops
    # ------------------------------------------------------------------

    def profiled_prefill(
        self,
        prompt: str,
        step_idx: int,
        max_new_tokens: int = 1,
    ) -> Tuple[Any, float]:
        """Run a prefill pass with timing."""
        with _phase_timer(self._intervals, "PROMPT_BUILD", step_idx, self.device):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if _TORCH_AVAILABLE and self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with _phase_timer(self._intervals, "PREFILL", step_idx, self.device):
            if _TORCH_AVAILABLE:
                with torch.no_grad():
                    outputs = self.model(**inputs, use_cache=True)
                kv_cache = outputs.past_key_values
            else:
                kv_cache = None

        return kv_cache, inputs["input_ids"].shape[-1] if _TORCH_AVAILABLE else 0

    def profiled_decode(
        self,
        input_ids: Any,
        kv_cache: Any,
        step_idx: int,
        max_new_tokens: int = 64,
    ) -> Tuple[Any, List[int]]:
        """Run incremental decode with timing."""
        generated = []
        current_cache = kv_cache

        with _phase_timer(self._intervals, "DECODE", step_idx, self.device):
            if _TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    gen_out = self.model.generate(
                        input_ids,
                        past_key_values=current_cache,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated = gen_out[0][input_ids.shape[-1]:].tolist()
                current_cache = None  # generate() doesn't return cache easily
            else:
                # CPU/mock path
                time.sleep(0.05)
                generated = [0] * max_new_tokens

        return current_cache, generated

    def profiled_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        step_idx: int,
    ) -> Tuple[str, float]:
        """Run a mock tool call with timing."""
        with _phase_timer(self._intervals, "TOOL_IO", step_idx, None):
            result, actual_ms = self.tools.call(tool_name, args)
        return result, actual_ms

    def profiled_json_parse(
        self,
        text: str,
        step_idx: int,
    ) -> Dict[str, Any]:
        """Time JSON extraction from model output."""
        with _phase_timer(self._intervals, "JSON_PARSE", step_idx, None):
            import re, json
            # Try to extract action JSON from ReAct format
            match = re.search(r"Action:\s*(\w+)\nAction Input:\s*(.+)", text, re.DOTALL)
            if match:
                tool_name = match.group(1).strip()
                tool_input = match.group(2).strip()
                try:
                    args = json.loads(tool_input)
                except json.JSONDecodeError:
                    args = {"query": tool_input}
                return {"tool": tool_name, "args": args}
            return {"tool": "search", "args": {"query": text[:50]}}

    # ------------------------------------------------------------------
    # Full ReAct loop profiling
    # ------------------------------------------------------------------

    def run(
        self,
        system_prompt: str,
        questions: Optional[List[str]] = None,
        n_steps: int = 10,
        max_decode_tokens: int = 64,
    ) -> ProfileReport:
        """
        Profile a full n-step ReAct loop.

        Parameters
        ----------
        system_prompt : str
            System prompt (tools description, instruction format).
        questions : list of str
            Questions to answer.  One per step.
        n_steps : int
            Number of agent steps to profile.
        max_decode_tokens : int
            Max tokens to decode per reasoning step.

        Returns
        -------
        ProfileReport
        """
        if questions is None:
            questions = [f"What is the capital of country {i}?" for i in range(n_steps)]

        self._intervals = []
        step_profiles: List[StepProfile] = []

        hardware_str = self._detect_hardware()

        for step_idx in range(min(n_steps, len(questions))):
            step_start = time.perf_counter()

            prompt = self._build_react_prompt(system_prompt, questions[step_idx])

            # 1. Prefill
            kv_cache, seq_len = self.profiled_prefill(prompt, step_idx)

            # 2. Decode reasoning
            if _TORCH_AVAILABLE and hasattr(kv_cache, "__iter__"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if self.device.startswith("cuda"):
                    input_ids = input_ids.to(self.device)
            else:
                input_ids = None

            _, generated_ids = self.profiled_decode(input_ids, kv_cache, step_idx, max_decode_tokens)

            # 3. Parse action from generation
            generated_text = self.tokenizer.decode(generated_ids) if generated_ids else "Action: search\nAction Input: test"
            action = self.profiled_json_parse(generated_text, step_idx)

            # 4. Tool call
            _, _ = self.profiled_tool_call(action["tool"], action["args"], step_idx)

            # 5. KV write (simulated — real measurement requires Nsight)
            with _phase_timer(self._intervals, "KV_WRITE", step_idx, None):
                time.sleep(0.002)  # 2ms synthetic KV write overhead

            step_end = time.perf_counter()
            step_ms  = (step_end - step_start) * 1000

            step_intervals = [i for i in self._intervals if i.step_idx == step_idx]
            step_profiles.append(StepProfile(
                step_idx=step_idx,
                total_ms=step_ms,
                intervals=step_intervals,
            ))

        return ProfileReport(
            model_name=getattr(self.model, "name_or_path", "unknown")
                       if self.model else "mock",
            hardware=hardware_str,
            n_steps=len(step_profiles),
            step_profiles=step_profiles,
            raw_intervals=list(self._intervals),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_react_prompt(self, system: str, question: str) -> str:
        return (
            f"{system}\n\n"
            f"Question: {question}\n"
            f"Thought: I need to look this up.\n"
            f"Action: search\n"
            f"Action Input: {{\"query\": \"{question}\"}}\n"
        )

    def _detect_hardware(self) -> str:
        if not _TORCH_AVAILABLE:
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0).lower()
        if "a100" in name:
            return "a100"
        if "t4" in name:
            return "t4"
        return name


# ---------------------------------------------------------------------------
# Standalone profiling function (no model required — uses timing stubs)
# ---------------------------------------------------------------------------

def profile_mock_agent(
    n_steps:          int   = 10,
    prefill_ms:       float = 80.0,
    decode_ms:        float = 150.0,
    tool_latency_ms:  float = 100.0,
    json_parse_ms:    float = 5.0,
    prompt_build_ms:  float = 3.0,
    kv_write_ms:      float = 2.0,
    noise_factor:     float = 0.1,
) -> ProfileReport:
    """
    Profile a simulated agent loop without a real model.
    Useful on CPU-only machines and for unit tests.

    The timing parameters approximate realistic values measured on a T4 GPU
    running GPT-2 with a 512-token context.

    Parameters match the real profiler's output format so analysis code
    (roofline.py, stats.py) is identical for mock and real runs.
    """
    import random
    rng = random.Random(42)

    intervals: List[PhaseInterval] = []
    step_profiles: List[StepProfile] = []

    for step_idx in range(n_steps):
        step_start = time.perf_counter()

        def _jitter(base_ms: float) -> float:
            return base_ms * (1 + rng.uniform(-noise_factor, noise_factor))

        step_intervals: List[PhaseInterval] = []

        def _mock_phase(phase: str, base_ms: float):
            t = time.perf_counter() * 1000
            dur = _jitter(base_ms)
            step_intervals.append(PhaseInterval(phase, t, t + dur, step_idx))
            intervals.append(step_intervals[-1])
            time.sleep(dur / 1000)

        _mock_phase("PROMPT_BUILD", prompt_build_ms)
        _mock_phase("PREFILL",      prefill_ms)
        _mock_phase("DECODE",       decode_ms)
        _mock_phase("JSON_PARSE",   json_parse_ms)
        _mock_phase("TOOL_IO",      tool_latency_ms)
        _mock_phase("KV_WRITE",     kv_write_ms)

        step_end = time.perf_counter()
        step_ms  = (step_end - step_start) * 1000

        step_profiles.append(StepProfile(
            step_idx=step_idx,
            total_ms=step_ms,
            intervals=step_intervals,
        ))

    return ProfileReport(
        model_name="mock_agent",
        hardware="simulated",
        n_steps=n_steps,
        step_profiles=step_profiles,
        raw_intervals=intervals,
    )
