.PHONY: install sample tokenize check clean

install:
	uv sync

# Стадия 0, только для преподавателя: parquet курса -> data/train.jsonl, data/val.jsonl.
# У студента вход другой — его датасет из ДЗ 3, положенный в те же файлы.
sample:
	uv run python -m scripts.make_sample

# Стадия tokenize: JSONL -> токенизированный датасет + метрики + отчёт.
# Та же команда объявлена стадией в dvc.yaml — оттуда её и запускает dvc repro.
tokenize:
	uv run python -m src.tokenize_data

# Токенизатор уже в кеше после make tokenize; онлайн-проверка списка шаблонов
# в HF Hub при плохой сети висит до таймаута и валит проверки. HF_HUB_OFFLINE=0 make check — отключить.
HF_HUB_OFFLINE ?= 1
export HF_HUB_OFFLINE
check:
	bash tests/check.sh

clean:
	rm -rf data metrics/tokenize.json docs/tokenize_report.md
