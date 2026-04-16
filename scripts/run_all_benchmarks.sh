#!/usr/bin/env bash
# scripts/run_all_benchmarks.sh
# Run all LoopFuse benchmarks and save results to results/
# Usage: bash scripts/run_all_benchmarks.sh [--gpu] [--tpu]

set -euo pipefail
ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "==============================="
echo "  LoopFuse Benchmark Suite"
echo "==============================="
echo "Results saved to: $RESULTS"
echo ""

# H1: Phase profiling (T4/A100 or CPU mock)
echo "[H1] Phase waste profiling..."
python "$ROOT/loopfuse/notebooks/01_phase_profiling.py" 2>&1 | tee "$RESULTS/h1_output.txt"
echo ""

# H2: KV cache fusion
echo "[H2] KV cache fusion..."
python "$ROOT/loopfuse/benchmarks/h2_kv_fusion/run.py" 2>&1 | tee "$RESULTS/h2_output.txt"
echo ""

# H3: Speculative prefill
echo "[H3] Speculative prefill..."
python "$ROOT/loopfuse/benchmarks/h3_spec_prefill/run.py" 2>&1 | tee "$RESULTS/h3_output.txt"
echo ""

# H4: Phase-specialized kernels (GPU required)
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[H4] Phase-specialized kernels (GPU)..."
    python "$ROOT/loopfuse/benchmarks/h4_phase_kernels/run.py" 2>&1 | tee "$RESULTS/h4_output.txt"
else
    echo "[H4] SKIPPED — no CUDA GPU detected"
fi
echo ""

# H5: Cross-hardware portability
echo "[H5] Cross-hardware portability..."
python "$ROOT/loopfuse/benchmarks/h5_cross_hardware/run.py" 2>&1 | tee "$RESULTS/h5_output.txt"
echo ""

echo "==============================="
echo "  All benchmarks complete"
echo "  Run: python scripts/generate_figures.py"
echo "==============================="
