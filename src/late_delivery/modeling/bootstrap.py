"""Confidence intervals by resampling whole handoff days.

The contact budget is set per day and orders on one day share conditions (carrier load,
weather), so days, not orders, are the independent units.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

Stat = Callable[[pl.DataFrame], float]


def _resamples(df: pl.DataFrame, n: int, seed: int) -> list[np.ndarray]:
    days = np.array(sorted(df["handoff_date"].unique().to_list()))
    rng = np.random.default_rng(seed)
    return [rng.choice(len(days), size=len(days), replace=True) for _ in range(n)]


def _take(df: pl.DataFrame, days: list[object], draw: np.ndarray) -> pl.DataFrame:
    """Rows of the drawn days, a day drawn twice appearing twice."""
    picked = pl.DataFrame(
        {"handoff_date": [days[i] for i in draw]},
        schema={"handoff_date": df.schema["handoff_date"]},
    )
    return picked.join(df, on="handoff_date", how="inner")


def day_bootstrap(df: pl.DataFrame, stat: Stat, *, n: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of `stat` over `n` day resamples."""
    days = sorted(df["handoff_date"].unique().to_list())
    values = [stat(_take(df, days, d)) for d in _resamples(df, n, seed)]
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_day_bootstrap(
    a: pl.DataFrame, b: pl.DataFrame, stat: Stat, *, n: int, seed: int
) -> tuple[float, float, float]:
    """Observed stat(a) - stat(b) and its 95% interval, both frames resampled with the same
    day draws so day-to-day noise shared by the two models cancels out."""
    days = sorted(
        set(a["handoff_date"].unique().to_list()) | set(b["handoff_date"].unique().to_list())
    )
    draws = _resamples(pl.DataFrame({"handoff_date": days}), n, seed)
    diffs = [stat(_take(a, days, d)) - stat(_take(b, days, d)) for d in draws]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return stat(a) - stat(b), float(lo), float(hi)
