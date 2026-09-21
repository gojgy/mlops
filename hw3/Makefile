.PHONY: install repro v1 v2 diff dag diversity contamination check clean

install:
	uv sync

repro:
	uv run dvc repro

v1:
	uv run python scripts/set_version.py v1
	uv run dvc repro

v2:
	uv run python scripts/set_version.py v2
	uv run dvc repro

# версии зафиксированы git-тегами; другие ревизии: make diff FROM=... TO=...
FROM ?= hw3-data-v1
TO ?= hw3-data-v2
diff:
	uv run dvc metrics diff $(FROM) $(TO)

dag:
	uv run dvc dag

diversity:
	uv run python -m src.diversity

contamination:
	uv run python scripts/check_contamination.py

check:
	bash tests/check.sh

clean:
	rm -rf data metrics/*.json params.yaml.bak params.yaml.orig
