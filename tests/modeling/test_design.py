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
