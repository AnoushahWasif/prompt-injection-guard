.PHONY: install lint format test data baseline

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
