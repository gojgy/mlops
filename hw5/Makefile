HF_HUB_OFFLINE ?= 1
export HF_HUB_OFFLINE

.PHONY: install prepare train train-all train-freeze compare plot check clean distclean prepare-final train-final

install:
	uv sync

prepare:
	uv run python -m src.prepare_data

# Два прогона: адаптеры на всех слоях и с замороженными первыми 14.
train: train-all train-freeze plot

train-all:
	uv run python -m src.train --variant all_layers

train-freeze:
	uv run python -m src.train --variant freeze14

plot:
	uv run python -m src.plot

# Итоговый адаптер на 3072 примерах против базы.
prepare-final:
	uv run python -m src.final prepare

train-final:
	uv run python -m src.final train --variant all_layers

compare:
	uv run python -m src.compare --variant all_layers --out metrics/compare_all_layers.json
	uv run python -m src.final compare --variant all_layers

check:
	bash tests/check.sh

# clean — артефакты контрольного и итогового прогонов.
# distclean — ещё и .venv (~0,8 ГБ): когда ДЗ сдано и место нужнее.
clean:
	rm -rf models metrics docs/curves.png docs/compare.md .check_*.log
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv
