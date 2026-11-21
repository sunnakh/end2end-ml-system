from datetime import date
from pathlib import Path

import polars as pl
import pytest

from late_delivery.config import TrainingConfig
from late_delivery.modeling.experiment import ExperimentResult, Grid, Snapshot, run_experiment
from late_delivery.modeling.gate import read_champion
from tests.modeling.conftest import synthetic_snapshot

BASE = synthetic_snapshot(9000, seed=11)
CFG = TrainingConfig(bootstrap_resamples=30)
GRID = Grid(logreg_l2=[1e-2], lgbm=[{"num_leaves": 7, "learning_rate": 0.1,
                                     "min_child_samples": 20}])  # fmt: skip
EVAL_CUTOFF = CFG.evaluation_cutoff


def _load(cutoff: date) -> Snapshot:
    # A snapshot cut on `cutoff` only holds orders handed off before it.
    frame = BASE.filter(pl.col("handoff_date") < cutoff).drop("handoff_date")
    return Snapshot(frame=frame, content_hash=f"hash-{cutoff.isoformat()}")


@pytest.fixture(scope="module")
def first(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentResult, Path]:
    models = tmp_path_factory.mktemp("models")
    return run_experiment(CFG, _load, models, run_id="r1", git_commit="abc", grid=GRID), models


def test_no_fit_sees_its_own_test_period(first: tuple[ExperimentResult, Path]) -> None:
    fits = first[0].fits
    fixed = fits.filter(pl.col("purpose") != "walk_forward")
    assert (fixed["max_date"] < CFG.test_start).all()
    wf = fits.filter(pl.col("purpose") == "walk_forward")
    assert wf.height == 2
    assert (wf["max_date"] < wf["scored_month_start"]).all()


def test_train_and_validation_come_from_decision_snapshot(
    first: tuple[ExperimentResult, Path],
) -> None:
    fits = first[0].fits
    assert set(fits.filter(pl.col("purpose") != "walk_forward")["snapshot_cutoff"]) == {
        CFG.test_start
    }
    assert first[0].snapshot_hashes["decision"] == f"hash-{CFG.test_start}"
    assert first[0].snapshot_hashes["evaluation"] == f"hash-{EVAL_CUTOFF}"


def test_test_rows_come_from_evaluation_snapshot(first: tuple[ExperimentResult, Path]) -> None:
    monthly = first[0].monthly
    june = monthly.filter((pl.col("month") == "2018-06") & (pl.col("model") == "chosen"))
    expected = BASE.filter(pl.col("handoff_date").dt.strftime("%Y-%m") == "2018-06").height
    assert june["orders"].item() == expected


def test_monthly_rows(first: tuple[ExperimentResult, Path]) -> None:
    m = first[0].monthly
    for model in ("rule", "chosen"):
        assert m.filter(pl.col("model") == model)["month"].to_list() == [
            "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"
        ]  # fmt: skip
    assert m.filter(pl.col("model") == "walk_forward")["month"].to_list() == ["2018-07", "2018-08"]
    assert {"recall_lo", "recall_hi", "pr_auc", "brier", "budget", "contacts"} <= set(m.columns)


def test_candidates_cover_every_window_and_model(first: tuple[ExperimentResult, Path]) -> None:
    c = first[0].candidates
    assert c.height == len(CFG.window_months) * 2
    assert set(c["window_months"]) == set(CFG.window_months)


def test_calendar_only_is_blocked(first: tuple[ExperimentResult, Path]) -> None:
    assert first[0].calendar_gate.passed is False


def test_slices_undelivered_and_leakage(first: tuple[ExperimentResult, Path]) -> None:
    r = first[0]
    assert set(r.slices["slice_type"]) == {"region", "seller_size"}
    assert set(r.undelivered["rows"]) == {"all", "delivered_only"}
    assert {"time_pr_auc", "random_pr_auc", "time_recall_at_k", "random_recall_at_k"} <= set(
        r.leakage
    )


def test_champion_written_only_on_pass_and_used_next_run(
    first: tuple[ExperimentResult, Path],
) -> None:
    result, models = first
    champ = read_champion(models)
    assert (champ is not None) == result.gate.passed
    if champ is None:
        pytest.skip("synthetic challenger did not pass; nothing to compare against")
    assert (models / champ.gate_model_path).exists() and (models / champ.model_path).exists()
    second = run_experiment(CFG, _load, models, run_id="r2", git_commit="abc", grid=GRID)
    assert "beats champion" in [c.name for c in second.gate.checks]


def test_reruns_are_identical(first: tuple[ExperimentResult, Path], tmp_path: Path) -> None:
    again = run_experiment(CFG, _load, tmp_path, run_id="r1", git_commit="abc", grid=GRID)
    assert again.monthly.equals(first[0].monthly)
    assert again.candidates.equals(first[0].candidates)
