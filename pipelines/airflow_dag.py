"""Generated from late_delivery.dag.STEPS by `make airflow-dag`. Do not edit."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="late_delivery_daily",
    schedule="@daily",
    start_date=datetime(2016, 9, 4),
    catchup=True,
    max_active_runs=1,  # merge_current depends on earlier days
    default_args={"retries": 2},
) as dag:
    tasks = {}
    tasks["simulate"] = BashOperator(
        task_id="simulate",
        bash_command="late-delivery run-step --step simulate --ds {{ ds }}",
    )
    tasks["ingest_bronze"] = BashOperator(
        task_id="ingest_bronze",
        bash_command="late-delivery run-step --step ingest_bronze --ds {{ ds }}",
    )
    tasks["validate_to_silver"] = BashOperator(
        task_id="validate_to_silver",
        bash_command="late-delivery run-step --step validate_to_silver --ds {{ ds }}",
    )
    tasks["merge_current"] = BashOperator(
        task_id="merge_current",
        bash_command="late-delivery run-step --step merge_current --ds {{ ds }}",
    )
    tasks["build_gold_scoring"] = BashOperator(
        task_id="build_gold_scoring",
        bash_command="late-delivery run-step --step build_gold_scoring --ds {{ ds }}",
    )
    tasks["drift_report"] = BashOperator(
        task_id="drift_report",
        bash_command="late-delivery run-step --step drift_report --ds {{ ds }}",
    )
    tasks["simulate"] >> tasks["ingest_bronze"]
    tasks["ingest_bronze"] >> tasks["validate_to_silver"]
    tasks["validate_to_silver"] >> tasks["merge_current"]
    tasks["merge_current"] >> tasks["build_gold_scoring"]
    tasks["build_gold_scoring"] >> tasks["drift_report"]
