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


def test_champion_written_and_used_next_run(first: tuple[ExperimentResult, Path]) -> None:
    result, models = first
    assert result.gate.passed  # the synthetic signal is strong: the challenger must pass
    champ = read_champion(models)
    assert champ is not None
    # The gate model is the train-only fit; the served model is the refit, a different file.
    assert date.fromisoformat(champ.gate_train_end) < CFG.valid_start
    gate_bytes = (models / champ.gate_model_path).read_bytes()
    assert gate_bytes != (models / champ.model_path).read_bytes()
    assert "seller_prior_median_handling_h" in champ.features
    second = run_experiment(CFG, _load, models, run_id="r2", git_commit="abc", grid=GRID)
    assert "beats champion" in [c.name for c in second.gate.checks]


def test_champion_that_saw_validation_is_refused(tmp_path: Path) -> None:
    run_experiment(CFG, _load, tmp_path, run_id="r1", git_commit="abc", grid=GRID)
    later = CFG.model_copy(update={"valid_start": date(2018, 3, 1)})  # now overlaps r1's train
    with pytest.raises(ValueError, match="trained on rows up to"):
        run_experiment(later, _load, tmp_path, run_id="r2", git_commit="abc", grid=GRID)


def test_champion_missing_features_is_refused(tmp_path: Path) -> None:
    import json

    run_experiment(CFG, _load, tmp_path, run_id="r1", git_commit="abc", grid=GRID)
    path = tmp_path / "champion.json"
    data = json.loads(path.read_text())
    data["features"] = [*data["features"], "feature_from_the_future"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="feature_from_the_future"):
        run_experiment(CFG, _load, tmp_path, run_id="r2", git_commit="abc", grid=GRID)


def test_calibration_trial_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import late_delivery.modeling.experiment as experiment

    monkeypatch.setattr(experiment, "CALIBRATION_ECE", -1.0)  # force a trial
    r = run_experiment(CFG, _load, tmp_path, run_id="r1", git_commit="abc", grid=GRID)
    assert r.calibration["tried"] is True
    assert {"base_ece", "calibrated_ece", "base_recall_at_k", "calibrated_recall_at_k",
            "kept"} <= set(r.calibration)  # fmt: skip


def test_extra_outputs(first: tuple[ExperimentResult, Path]) -> None:
    r = first[0]
    assert "roc_auc" in r.monthly.columns and "roc_auc" in r.candidates.columns
    assert {"candidates", "gate", "test", "total"} <= set(r.timings)
    assert {"time_late_rate", "random_late_rate", "time_pr_auc_lift",
            "random_pr_auc_lift"} <= set(r.leakage)  # fmt: skip
    assert r.calibration["tried"] in (True, False)


def test_reruns_are_identical(first: tuple[ExperimentResult, Path], tmp_path: Path) -> None:
    again = run_experiment(CFG, _load, tmp_path, run_id="r1", git_commit="abc", grid=GRID)
    assert again.monthly.equals(first[0].monthly)
    assert again.candidates.equals(first[0].candidates)
