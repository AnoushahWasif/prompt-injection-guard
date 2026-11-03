.PHONY: install lint format test data baseline finetune serve redteam bench

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	uv run python -m guard.data

baseline:
	uv run python -m guard.baseline

finetune:
	uv run python -m guard.finetune

serve:
	uv run uvicorn --factory guard.api:app_from_env --host 127.0.0.1 --port 8000

redteam:
	uv run pytest tests/redteam -rx

bench:
	uv run python -m guard.bench
