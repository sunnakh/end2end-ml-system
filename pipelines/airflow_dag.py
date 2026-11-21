"""Generated from late_delivery.dag.STEPS by `make airflow-dag`. Do not edit."""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# Every task runs inside the project checkout. Set LATE_DELIVERY_PROJECT_DIR on the
# Airflow worker; the PII salt comes from the worker environment or the project's .env.

with DAG(
    dag_id="late_delivery_daily",
    schedule="@daily",
    start_date=datetime(2016, 9, 4),
    end_date=datetime(2018, 10, 17),  # the replay range; no empty runs after it
    catchup=True,
    max_active_runs=1,  # merge_current depends on earlier days
    default_args={
        "retries": 2,
        "depends_on_past": True,  # a failed day blocks the same task on later days
        "wait_for_downstream": True,  # and waits for that day to finish
    },
) as dag:
    tasks = {}
    tasks["simulate"] = BashOperator(
        task_id="simulate",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step simulate --ds {{ ds }}',
    )
    tasks["ingest_bronze"] = BashOperator(
        task_id="ingest_bronze",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step ingest_bronze --ds {{ ds }}',
    )
    tasks["validate_to_silver"] = BashOperator(
        task_id="validate_to_silver",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step validate_to_silver --ds {{ ds }}',
    )
    tasks["merge_current"] = BashOperator(
        task_id="merge_current",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step merge_current --ds {{ ds }}',
    )
    tasks["build_gold_scoring"] = BashOperator(
        task_id="build_gold_scoring",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step build_gold_scoring --ds {{ ds }}',
    )
    tasks["drift_report"] = BashOperator(
        task_id="drift_report",
        bash_command='cd "$LATE_DELIVERY_PROJECT_DIR" && if [ -f .env ]; then set -a; . ./.env; set +a; fi && uv run late-delivery --env "${LATE_DELIVERY_ENV:-local}" --config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" run-step --step drift_report --ds {{ ds }}',
    )
    tasks["simulate"] >> tasks["ingest_bronze"]
    tasks["ingest_bronze"] >> tasks["validate_to_silver"]
    tasks["validate_to_silver"] >> tasks["merge_current"]
    tasks["merge_current"] >> tasks["build_gold_scoring"]
    tasks["build_gold_scoring"] >> tasks["drift_report"]
