import ast
from datetime import date
from pathlib import Path

from late_delivery.config import load_settings
from late_delivery.dag import STEPS
from late_delivery.platform.pipeline.airflow import render_airflow_dag, render_for_project

START, END = date(2016, 9, 4), date(2018, 10, 17)


def _src() -> str:
    return render_airflow_dag(STEPS, "late_delivery_daily", start=START, end=END)


def test_render_parses() -> None:
    ast.parse(_src())


def test_uses_airflow_3_imports() -> None:
    src = _src()
    assert "from airflow.sdk import DAG" in src
    assert "from airflow.providers.standard.operators.bash import BashOperator" in src


def test_one_task_per_step() -> None:
    src = _src()
    for step in STEPS:
        assert f'task_id="{step.name}"' in src
        assert f"run-step --step {step.name} --ds {{{{ ds }}}}" in src
    assert src.count("BashOperator(") == len(STEPS)


def test_dependencies_rendered() -> None:
    src = _src()
    assert 'tasks["simulate"] >> tasks["ingest_bronze"]' in src
    assert 'tasks["build_gold_scoring"] >> tasks["drift_report"]' in src


def test_a_failed_day_blocks_later_days() -> None:
    src = _src()
    assert '"depends_on_past": True' in src
    assert '"wait_for_downstream": True' in src
    assert "max_active_runs=1" in src


def test_schedule_bounded_by_replay_range() -> None:
    src = _src()
    assert "start_date=datetime(2016, 9, 4)" in src
    assert "end_date=datetime(2018, 10, 17)" in src
    assert "catchup=True" in src


def test_command_independent_of_working_directory() -> None:
    src = _src()
    assert 'cd "$LATE_DELIVERY_PROJECT_DIR"' in src
    assert '--config-dir "$LATE_DELIVERY_PROJECT_DIR/configs"' in src
    assert "/Users/" not in src and "/home/" not in src  # no machine-specific paths


def test_committed_dag_is_current() -> None:
    expected = render_for_project(load_settings("local"))
    assert Path("pipelines/airflow_dag.py").read_text() == expected
