# Load local secrets (LATE_DELIVERY_PII_SALT) from .env if present; .env is gitignored.
-include .env
export

.PHONY: setup test lint format typecheck check pipeline rerun training-set train airflow-dag data

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

# Daily pipeline replay. Defaults to the configured replay range.
pipeline:
	uv run late-delivery backfill $(if $(START),--start $(START)) $(if $(END),--end $(END))

rerun:
	uv run late-delivery rerun --ds $(DS)

training-set:
	uv run late-delivery training-set --cutoff $(CUTOFF)

# M3: train, evaluate, gate and write the report to data/models/runs/<run_id>/.
train:
	uv run late-delivery train $(if $(RUN_ID),--run-id $(RUN_ID))

airflow-dag:
	uv run python -m late_delivery.platform.pipeline.airflow > pipelines/airflow_dag.py

data:
	./scripts/download_olist.sh
