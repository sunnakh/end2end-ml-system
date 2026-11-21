import numpy as np
import polars as pl
import pytest

from late_delivery.modeling.design import DesignMatrix


def _df(states: list[str], price: list[float | None], same: list[bool]) -> pl.DataFrame:
    return pl.DataFrame(
        {"customer_state": states, "total_price": price, "same_state_all": same},
        schema={"customer_state": pl.String, "total_price": pl.Float64,
                "same_state_all": pl.Boolean},
    )  # fmt: skip


FEATS = ["customer_state", "total_price", "same_state_all"]
TRAIN = _df(["SP"] * 40 + ["RJ"] * 35 + ["AC"] * 5, [10.0] * 40 + [100.0] * 39 + [None],
            [True, False] * 40)  # fmt: skip


def test_rare_and_unseen_levels_share_one_column() -> None:
    dm = DesignMatrix(FEATS, rare_min_count=30).fit(TRAIN)
    assert [n for n in dm.feature_names if n.startswith("customer_state=")] == [
        "customer_state=RJ", "customer_state=SP", "customer_state=__rare__"
    ]  # fmt: skip
    x = dm.transform(_df(["XX", "AC"], [10.0, 10.0], [True, True]))
    rare = dm.feature_names.index("customer_state=__rare__")
    assert x.shape == (2, len(dm.feature_names))
    assert x[:, rare].tolist() == [1.0, 1.0]


def test_missing_indicator_only_for_features_with_nulls() -> None:
    dm = DesignMatrix(FEATS).fit(TRAIN)
    assert "total_price__missing" in dm.feature_names
    assert "same_state_all__missing" not in dm.feature_names


def test_nulls_imputed_with_training_median() -> None:
    dm = DesignMatrix(FEATS).fit(TRAIN)
    x = dm.transform(_df(["SP", "SP"], [None, 10.0], [True, True]))
    i = dm.feature_names.index("total_price")
    # 40 prices of 10 and 39 of 100: the training median is 10
    assert x[0, i] == pytest.approx(x[1, i])
    assert x[0, dm.feature_names.index("total_price__missing")] == 1.0


def test_numeric_columns_standardised_on_train() -> None:
    dm = DesignMatrix(FEATS).fit(TRAIN)
    x = dm.transform(TRAIN)
    for name in ("total_price", "same_state_all"):
        col = x[:, dm.feature_names.index(name)]
        assert col.mean() == pytest.approx(0.0, abs=1e-12)
        assert col.std() == pytest.approx(1.0, abs=1e-12)


def test_log_features_use_log1p_before_scaling() -> None:
    dm = DesignMatrix(FEATS).fit(TRAIN)
    x = dm.transform(_df(["SP", "SP"], [np.e - 1, np.e**2 - 1], [True, True]))
    i = dm.feature_names.index("total_price")
    step = x[1, i] - x[0, i]  # log1p values 1 and 2, one unit apart before scaling
    assert step == pytest.approx(1.0 / dm.scale_[dm.numeric_names.index("total_price")])


def test_constant_column_does_not_divide_by_zero() -> None:
    df = _df(["SP"] * 50, [5.0] * 50, [True] * 50)
    x = DesignMatrix(FEATS).fit(df).transform(df)
    assert np.isfinite(x).all()


def test_only_categorical_features() -> None:
    df = pl.DataFrame({"handoff_weekday": [1, 2] * 40, "handoff_month": [3] * 80})
    dm = DesignMatrix(["handoff_weekday", "handoff_month"]).fit(df)
    x = dm.transform(df)
    assert x.shape == (80, len(dm.feature_names))
    assert np.isfinite(x).all()


def test_little_time_left_flag_added_next_to_remaining_days() -> None:
    df = pl.DataFrame({"remaining_days": [1.0, 4.9, 5.0, 12.0] * 10})
    dm = DesignMatrix(["remaining_days"]).fit(df)
    assert dm.feature_names == ["remaining_days", "remaining_days__under_5"]
    flag = dm.transform(df)[:4, 1]
    assert flag[0] == flag[1] > flag[2] == flag[3]  # under 5 days vs 5 or more


def test_logreg_learns_a_cliff_it_could_not_draw_as_a_line() -> None:
    from late_delivery.modeling.logreg import fit_logreg, predict_proba
    from late_delivery.modeling.metrics import average_precision

    # Late = a step in days left (under 5) plus a smooth effect of another feature. With days
    # alone, line and step rank orders the same way; the step matters once another feature
    # competes for the ranking, as state and month do on the real data.
    rng = np.random.default_rng(0)
    n = 6000
    days, other = rng.uniform(0, 30, n), rng.normal(0, 1, n)
    logit = -2.6 + 3.0 * (days < 5) + 0.8 * other
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pl.DataFrame({"remaining_days": days, "other": other})
    dm = DesignMatrix(["remaining_days", "other"]).fit(df)
    x_step = dm.transform(df)
    x_line = x_step[:, [dm.feature_names.index("remaining_days"), dm.feature_names.index("other")]]
    ap_line = average_precision(y, predict_proba(x_line, *fit_logreg(x_line, y, l2=1e-3)))
    ap_step = average_precision(y, predict_proba(x_step, *fit_logreg(x_step, y, l2=1e-3)))
    assert ap_step > ap_line + 0.05
