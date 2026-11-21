import json
from pathlib import Path

import pytest

from late_delivery.config import TrainingConfig
from late_delivery.modeling.experiment import ExperimentResult, run_experiment
from late_delivery.modeling.report import write_report
from tests.modeling.test_experiment import CFG, GRID, _load

SECTIONS = [
    "## Run", "## Decision", "## Validation candidates", "## Month by month", "## Slices",
    "## Never-delivered orders", "## Leakage check", "## Notes",
]  # fmt: skip


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentResult, Path]:
    models = tmp_path_factory.mktemp("models")
    result = run_experiment(CFG, _load, models, run_id="r1", git_commit="abc", grid=GRID)
    out = models / "runs" / "r1"
    write_report(result, CFG, out)
    return result, out


def test_all_files_written(written: tuple[ExperimentResult, Path]) -> None:
    out = written[1]
    for name in ("report.md", "metrics.json", "run.json", "reliability.png", "pr_curve.png"):
        assert (out / name).stat().st_size > 0, name


def test_report_sections_in_order(written: tuple[ExperimentResult, Path]) -> None:
    text = (written[1] / "report.md").read_text()
    positions = [text.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert ("PASSED" in text) or ("BLOCKED" in text)
    assert "optimistic" in text  # validation gate numbers flagged (selection on same months)


def test_metrics_json_has_every_month(written: tuple[ExperimentResult, Path]) -> None:
    data = json.loads((written[1] / "metrics.json").read_text())
    months = {row["month"] for row in data["monthly"]}
    assert months == {"2018-04", "2018-05", "2018-06", "2018-07", "2018-08"}
    assert data["gate"]["passed"] == written[0].gate.passed
    assert json.loads(json.dumps(data)) == data


def test_run_json_traces_inputs(written: tuple[ExperimentResult, Path]) -> None:
    run = json.loads((written[1] / "run.json").read_text())
    assert run["git_commit"] == "abc"
    assert run["snapshot_hashes"] == written[0].snapshot_hashes
    assert run["config"]["test_start"] == "2018-06-01"
    assert isinstance(TrainingConfig.model_validate(run["config"]), TrainingConfig)


def test_borderline_slice_is_flagged() -> None:
    from late_delivery.modeling.gate import GateCheck, GateResult
    from late_delivery.modeling.report import _gate_section

    g = GateResult(True, [GateCheck("slice region=Center-West", True, "d", -0.094),
                          GateCheck("slice region=South", True, "d", -0.038)])  # fmt: skip
    text = _gate_section(g, tolerance=0.05)
    assert "**Borderline:** slice region=Center-West" in text
    assert "South dropped" not in text


def test_budget_note_comes_from_the_numbers() -> None:
    import polars as pl

    from late_delivery.modeling.report import _budget_note

    def monthly(rate: float) -> pl.DataFrame:
        return pl.DataFrame(
            {"split": ["test"], "model": ["chosen"], "month": ["2018-06"],
             "late_rate": [rate], "contacts": [297.0], "budget": [299.0]}
        )  # fmt: skip

    low = _budget_note(monthly(0.021), CFG)
    assert "2018-06 (2.1% late: 297 contacts of 299 allowed)" in low
    assert "at or above" in _budget_note(monthly(0.08), CFG)
