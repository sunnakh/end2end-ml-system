"""Render an Airflow DAG file from the same step definitions the local runner uses."""

from __future__ import annotations

from late_delivery.platform.pipeline.runner import Step, topological_order

_TEMPLATE = '''"""Generated from late_delivery.dag.STEPS by `make airflow-dag`. Do not edit."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="{dag_id}",
    schedule="{schedule}",
    start_date=datetime({start_year}, {start_month}, {start_day}),
    catchup=True,
    max_active_runs=1,  # merge_current depends on earlier days
    default_args={{"retries": 2}},
) as dag:
    tasks = {{}}
{tasks}
{edges}
'''


def render_airflow_dag(
    steps: list[Step],
    dag_id: str,
    schedule: str = "@daily",
    start: tuple[int, int, int] = (2016, 9, 4),
) -> str:
    ordered = topological_order(steps)
    tasks = "\n".join(
        f'    tasks["{s.name}"] = BashOperator(\n'
        f'        task_id="{s.name}",\n'
        f'        bash_command="late-delivery run-step --step {s.name} --ds {{{{ ds }}}}",\n'
        f"    )"
        for s in ordered
    )
    edges = "\n".join(
        f'    tasks["{dep}"] >> tasks["{s.name}"]' for s in ordered for dep in s.depends_on
    )
    return _TEMPLATE.format(
        dag_id=dag_id,
        schedule=schedule,
        start_year=start[0],
        start_month=start[1],
        start_day=start[2],
        tasks=tasks,
        edges=edges,
    )


def main() -> None:
    from late_delivery.dag import STEPS

    print(render_airflow_dag(STEPS, "late_delivery_daily"), end="")


if __name__ == "__main__":
    main()
