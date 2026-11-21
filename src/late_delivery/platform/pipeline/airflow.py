"""Render an Airflow 3 DAG file from the same step definitions the local runner uses."""

from __future__ import annotations

from datetime import date

from late_delivery.config import Settings
from late_delivery.platform.pipeline.runner import Step, topological_order

DAG_ID = "late_delivery_daily"

_HEADER = '''"""Generated from late_delivery.dag.STEPS by `make airflow-dag`. Do not edit."""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# Every task runs inside the project checkout. Set LATE_DELIVERY_PROJECT_DIR on the
# Airflow worker; the PII salt comes from the worker environment or the project's .env.
'''

# The command does not depend on Airflow's working directory or on a machine-specific path.
_COMMAND = (
    'cd "$LATE_DELIVERY_PROJECT_DIR" && '
    "if [ -f .env ]; then set -a; . ./.env; set +a; fi && "
    'uv run late-delivery --env "${{LATE_DELIVERY_ENV:-local}}" '
    '--config-dir "$LATE_DELIVERY_PROJECT_DIR/configs" '
    "run-step --step {step} --ds {{{{ ds }}}}"
)


def _datetime(d: date) -> str:
    return f"datetime({d.year}, {d.month}, {d.day})"


def render_airflow_dag(
    steps: list[Step], dag_id: str, *, start: date, end: date, schedule: str = "@daily"
) -> str:
    ordered = topological_order(steps)
    lines = [
        _HEADER,
        "with DAG(",
        f'    dag_id="{dag_id}",',
        f'    schedule="{schedule}",',
        f"    start_date={_datetime(start)},",
        f"    end_date={_datetime(end)},  # the replay range; no empty runs after it",
        "    catchup=True,",
        "    max_active_runs=1,  # merge_current depends on earlier days",
        "    default_args={",
        '        "retries": 2,',
        '        "depends_on_past": True,  # a failed day blocks the same task on later days',
        '        "wait_for_downstream": True,  # and waits for that day to finish',
        "    },",
        ") as dag:",
        "    tasks = {}",
    ]
    for s in ordered:
        command = _COMMAND.format(step=s.name)
        lines += [
            f'    tasks["{s.name}"] = BashOperator(',
            f'        task_id="{s.name}",',
            f"        bash_command={command!r},",
            "    )",
        ]
    lines += [f'    tasks["{dep}"] >> tasks["{s.name}"]' for s in ordered for dep in s.depends_on]
    return "\n".join(lines) + "\n"


def render_for_project(settings: Settings) -> str:
    """The DAG for this project's configured replay range."""
    from late_delivery.dag import STEPS

    return render_airflow_dag(STEPS, DAG_ID, start=settings.replay.start, end=settings.replay.end)


def main() -> None:
    from late_delivery.config import load_settings

    print(render_for_project(load_settings("local")), end="")


if __name__ == "__main__":
    main()
