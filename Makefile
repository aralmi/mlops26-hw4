export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export DVC_SITE_CACHE_DIR := $(CURDIR)/.cache/dvc
export TOKENIZERS_PARALLELISM := false

.PHONY: install sample import-data tokenize check verify repro status

install:
	uv sync --locked

import-data:
	uv run python -m scripts.import_hw3

# Стадия 0, только для преподавателя: parquet курса -> data/train.jsonl, data/val.jsonl.
# У студента вход другой — его датасет из ДЗ 3, положенный в те же файлы.
sample:
	uv run python -m scripts.make_sample

# Стадия tokenize: JSONL -> токенизированный датасет + метрики + отчёт.
# Та же команда объявлена стадией в dvc.yaml — оттуда её и запускает dvc repro.
tokenize:
	uv run python -m src.tokenize_data

check:
	bash tests/check.sh

verify:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v

repro:
	uv run dvc repro

status:
	uv run dvc status
