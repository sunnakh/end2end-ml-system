from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from late_delivery.modeling.metrics import roc_auc
from late_delivery.modeling.models import (
    LightGBMModel,
    LogRegModel,
    PlattCalibrated,
    RuleBaseline,
    calendar_only,
    load_model,
    save_model,
)
from tests.modeling.conftest import synthetic_snapshot

DATA = synthetic_snapshot(6000, seed=0)
TRAIN = DATA.filter(pl.col("handoff_date") < date(2018, 4, 1))
HELD = DATA.filter(pl.col("handoff_date") >= date(2018, 4, 1))


def _rule_rows(**cols: list[object]) -> pl.DataFrame:
    return pl.DataFrame(cols)


def test_rule_ranks_by_excess_over_seller_median() -> None:
    df = _rule_rows(
        handling_time_h=[90.0, 50.0, 10.0],
        seller_prior_median_handling_h=[30.0, 40.0, 40.0],
        same_state_all=[True, True, True],
    )
    s = RuleBaseline().score(df)
    assert s[0] == 60.0 and s[1] == 10.0
    assert s[2] == float("-inf")  # under its median, same state: not flagged


def test_rule_flags_interstate_and_handles_missing_median() -> None:
    df = _rule_rows(
        handling_time_h=[10.0, 10.0],
        seller_prior_median_handling_h=[None, None],
        same_state_all=[False, True],
    )
    s = RuleBaseline().score(df)
    assert s[0] == 0.0  # interstate, no history -> flagged with excess 0
    assert s[1] == float("-inf")
    assert RuleBaseline.uses_threshold is False


@pytest.mark.parametrize(
    "model", [LogRegModel(l2=1e-3), LightGBMModel(num_leaves=15, learning_rate=0.1,
                                                   min_child_samples=20)]
)  # fmt: skip
def test_learned_models_beat_chance(model: LogRegModel | LightGBMModel) -> None:
    model.fit(TRAIN)
    s = model.score(HELD)
    assert ((s >= 0) & (s <= 1)).all()
    assert roc_auc(HELD["late"].to_numpy(), s) > 0.7
    assert model.uses_threshold is True


def test_lightgbm_is_deterministic() -> None:
    a = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20, seed=3)
    b = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20, seed=3)
    a.fit(TRAIN)
    b.fit(TRAIN)
    assert np.array_equal(a.score(HELD), b.score(HELD))


def test_lightgbm_early_stops_on_last_month() -> None:
    m = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20)
    m.fit(TRAIN)
    assert 1 <= m.best_iteration < 2000
    assert m.params["early_stopping_month"] == "2018-03"


def test_unseen_state_scores_without_error() -> None:
    m = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20)
    m.fit(TRAIN)
    odd = HELD.head(3).with_columns(pl.lit("XX").alias("customer_state"))
    assert np.isfinite(m.score(odd)).all()


def test_platt_calibration_fixes_shifted_probabilities() -> None:
    base = LogRegModel(l2=1e-3)
    cal = PlattCalibrated(LogRegModel(l2=1e-3))
    base.fit(TRAIN)
    cal.fit(TRAIN)
    s = cal.score(HELD)
    assert ((s > 0) & (s < 1)).all()
    assert abs(s.mean() - HELD["late"].mean()) < 0.05  # type: ignore[operator]


def test_calendar_only_uses_two_features() -> None:
    m = calendar_only()
    assert m.features == ["handoff_weekday", "handoff_month"]
    m.fit(TRAIN)
    assert m.score(HELD).shape == (HELD.height,)


@pytest.mark.parametrize(
    "model", [RuleBaseline(), LogRegModel(l2=1e-3),
              LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20)]
)  # fmt: skip
def test_save_load_round_trip(tmp_path: Path, model: object) -> None:
    model.fit(TRAIN)  # type: ignore[attr-defined]
    save_model(model, tmp_path / "m.pkl")  # type: ignore[arg-type]
    again = load_model(tmp_path / "m.pkl")
    assert np.array_equal(again.score(HELD), model.score(HELD))  # type: ignore[attr-defined]


def test_lightgbm_keeps_its_own_feature_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import late_delivery.modeling.models as models_module

    m = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20)
    m.fit(TRAIN)
    before = m.score(HELD)
    save_model(m, tmp_path / "m.pkl")
    # A later code version adds a feature (and reorders): the saved model must not change.
    newer = ["brand_new_feature", *reversed(models_module.FEATURE_COLUMNS)]
    monkeypatch.setattr(models_module, "FEATURE_COLUMNS", newer)
    again = load_model(tmp_path / "m.pkl")
    assert np.array_equal(again.score(HELD), before)
    assert again.features == list(reversed(newer[1:]))  # type: ignore[attr-defined]


def test_lightgbm_final_trees_use_every_training_row() -> None:
    m = LightGBMModel(num_leaves=15, learning_rate=0.1, min_child_samples=20)
    m.fit(TRAIN)
    assert m.params["fitted_rows"] == TRAIN.height  # last month is in the trees too
    assert m.params["early_stopping_month"] == "2018-03"
    assert m.num_trees == m.best_iteration


def test_platt_needs_more_than_one_month() -> None:
    one_month = TRAIN.filter(pl.col("handoff_date") >= date(2018, 3, 1))
    with pytest.raises(ValueError, match="at least two months"):
        PlattCalibrated(LogRegModel(l2=1e-3)).fit(one_month)
