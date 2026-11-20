import ast

from late_delivery.dag import STEPS
from late_delivery.platform.pipeline.airflow import render_airflow_dag


def _src() -> str:
    return render_airflow_dag(STEPS, "late_delivery_daily")


def test_render_parses() -> None:
    ast.parse(_src())


def test_one_task_per_step() -> None:
    src = _src()
    for step in STEPS:
        assert f'task_id="{step.name}"' in src
        assert f"late-delivery run-step --step {step.name} --ds {{{{ ds }}}}" in src
    assert src.count("BashOperator(") == len(STEPS)


def test_dependencies_rendered() -> None:
    src = _src()
    assert 'tasks["simulate"] >> tasks["ingest_bronze"]' in src
    assert 'tasks["build_gold_scoring"] >> tasks["drift_report"]' in src


def test_sequential_scheduling() -> None:
    src = _src()
    assert "max_active_runs=1" in src
    assert "catchup=True" in src
    assert 'dag_id="late_delivery_daily"' in src


def test_committed_dag_is_current() -> None:
    from pathlib import Path

    assert Path("pipelines/airflow_dag.py").read_text() == _src()
