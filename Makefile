.PHONY: test test-gpu test-tpu lint notebooks clean install help

help:
	@echo "LoopFuse Makefile"
	@echo ""
	@echo "  make install     Install in editable mode with dev dependencies"
	@echo "  make test        Run CPU unit tests (no GPU required)"
	@echo "  make test-gpu    Run GPU tests (requires CUDA)"
	@echo "  make test-all    Run all tests"
	@echo "  make lint        Ruff style check"
	@echo "  make notebooks   Run CPU-safe notebooks (00, 02)"
	@echo "  make bench-h2    Run H2 KV fusion benchmark"
	@echo "  make bench-h3    Run H3 spec prefill benchmark"
	@echo "  make bench-h4    Run H4 phase kernels benchmark (GPU)"
	@echo "  make bench-h5    Run H5 cross-hardware benchmark"
	@echo "  make paper       Compile LaTeX paper (requires pdflatex)"
	@echo "  make clean       Remove __pycache__ and .pytest_cache"

install:
	pip install -e ".[dev,all]"

test:
	python -m pytest loopfuse/tests/ -m "not gpu and not tpu" -v --tb=short

test-gpu:
	python -m pytest loopfuse/tests/ -m "gpu" -v --tb=short

test-all:
	python -m pytest loopfuse/tests/ -v --tb=short

lint:
	ruff check loopfuse/ --select E,F,W --ignore E501,F401,E741

notebooks:
	python loopfuse/notebooks/00_setup.py
	python loopfuse/notebooks/02_ir_demo.py
	@echo "GPU notebooks (03-06) require Colab with GPU runtime"

bench-h2:
	python loopfuse/benchmarks/h2_kv_fusion/run.py

bench-h3:
	python loopfuse/benchmarks/h3_spec_prefill/run.py

bench-h4:
	python loopfuse/benchmarks/h4_phase_kernels/run.py

bench-h5:
	python loopfuse/benchmarks/h5_cross_hardware/run.py

bench-all: bench-h2 bench-h3 bench-h4 bench-h5

paper:
	cd loopfuse/paper && pdflatex loopfuse.tex && pdflatex loopfuse.tex
	@echo "PDF at loopfuse/paper/loopfuse.pdf"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage
	rm -f loopfuse/paper/*.aux loopfuse/paper/*.log loopfuse/paper/*.bbl
