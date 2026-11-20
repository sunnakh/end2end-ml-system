"""The daily pipeline: step definitions shared by the CLI and the generated Airflow DAG."""

from __future__ import annotations

from late_delivery.platform.pipeline.runner import Step
from late_delivery.steps import (
    build_gold_scoring,
    drift_report,
    ingest_bronze,
    merge_current,
    simulate,
    validate_to_silver,
)

STEPS: list[Step] = [
    Step("simulate", simulate),
    Step("ingest_bronze", ingest_bronze, ("simulate",)),
    Step("validate_to_silver", validate_to_silver, ("ingest_bronze",)),
    Step("merge_current", merge_current, ("validate_to_silver",)),
    Step("build_gold_scoring", build_gold_scoring, ("merge_current",)),
    Step("drift_report", drift_report, ("build_gold_scoring",)),
]
# Steps that depend on earlier days and must be refreshed after a mid-range rerun.
DOWNSTREAM_OF_HISTORY = ["merge_current", "build_gold_scoring", "drift_report"]
