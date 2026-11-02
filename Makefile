.PHONY: install lint format test data baseline finetune

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
