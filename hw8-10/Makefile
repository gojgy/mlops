PYTHON ?= .venv/bin/python
WORKSPACE ?= context/real
BENCHMARK_OUTPUT ?= context/benchmark
export PREFECT_HOME := $(CURDIR)/context/prefect
export PREFECT_SERVER_ANALYTICS_ENABLED := false
export HF_HUB_OFFLINE := 1
export TOKENIZERS_PARALLELISM := false

.PHONY: install check check-core prepare flow serve benchmark
install:
	uv sync --frozen --extra model
check-core:
	$(PYTHON) -m unittest discover -s tests -p test_release.py -v
check:
	$(PYTHON) -m compileall -q release.py lab.py service.py flow.py real_release.py benchmark.py tests
	$(PYTHON) -m unittest discover -s tests -v
prepare:
	$(PYTHON) real_release.py prepare --workspace $(WORKSPACE)
flow:
	$(PYTHON) flow.py --workspace $(WORKSPACE) --manifest manifest-candidate.json --report report-candidate.json
serve:
	$(PYTHON) service.py --workspace $(WORKSPACE) --backend transformers --port 8091
benchmark:
	$(PYTHON) benchmark.py --output $(BENCHMARK_OUTPUT)
