PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/context/matplotlib
.PHONY: install init build up check integration load hpa report down
install:
	uv sync --frozen --extra analysis
init:
	$(PYTHON) tools/setup.py init
build:
	$(PYTHON) tools/setup.py build
up:
	$(PYTHON) tools/setup.py install
check:
	$(PYTHON) -m compileall -q src tools tests
	$(PYTHON) -m unittest tests.test_guard -v
integration:
	HW11_INTEGRATION=1 $(PYTHON) -m unittest tests.test_cluster -v
load:
	$(PYTHON) tools/run_load.py
hpa:
	$(PYTHON) tools/run_hpa.py
report:
	$(PYTHON) tools/analyze.py
	$(PYTHON) tools/validate_results.py
	$(PYTHON) tools/report.py
down:
	$(PYTHON) tools/setup.py down
