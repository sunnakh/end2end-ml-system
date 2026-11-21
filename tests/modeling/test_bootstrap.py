from datetime import date, timedelta

import numpy as np
import polars as pl

from late_delivery.modeling.bootstrap import day_bootstrap, paired_day_bootstrap


def _frame(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = [date(2018, 4, 1) + timedelta(days=int(d)) for d in rng.integers(0, 30, 600)]
    return pl.DataFrame({"handoff_date": days, "x": rng.random(600)})


def _mean(df: pl.DataFrame) -> float:
    return float(df["x"].mean())  # type: ignore[arg-type]


def test_same_seed_same_interval() -> None:
    df = _frame(1)
    assert day_bootstrap(df, _mean, n=200, seed=3) == day_bootstrap(df, _mean, n=200, seed=3)


def test_interval_brackets_observed_mean() -> None:
    df = _frame(2)
    lo, hi = day_bootstrap(df, _mean, n=500, seed=0)
    assert lo < _mean(df) < hi


def test_constant_stat_has_zero_width() -> None:
    df = _frame(3)
    assert day_bootstrap(df, lambda d: 1.0, n=100, seed=0) == (1.0, 1.0)


def test_paired_with_itself_is_zero() -> None:
    df = _frame(4)
    assert paired_day_bootstrap(df, df, _mean, n=200, seed=0) == (0.0, 0.0, 0.0)


def test_paired_detects_shift() -> None:
    a = _frame(5)
    b = a.with_columns(pl.col("x") - 0.1)
    diff, lo, hi = paired_day_bootstrap(a, b, _mean, n=200, seed=0)
    assert abs(diff - 0.1) < 1e-12 and lo > 0.09 and hi < 0.11


def test_resamples_whole_days() -> None:
    # Each day has exactly 3 rows, so a resample made of whole days has a multiple of 3 rows.
    days = [date(2018, 4, 1) + timedelta(days=i // 3) for i in range(30)]
    df = pl.DataFrame({"handoff_date": days, "x": np.arange(30.0)})
    whole = lambda d: float(d.height % 3 == 0)  # noqa: E731
    assert day_bootstrap(df, whole, n=50, seed=0) == (1.0, 1.0)
