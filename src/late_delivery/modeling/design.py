"""Turn snapshot features into a numeric matrix for logistic regression.

Everything is learned on the training rows only (medians, scales, category levels) and
then applied unchanged to validation and test, so no information flows backwards.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Heavy-tailed amounts: log1p makes a 10x difference count as a fixed step.
LOG_FEATURES = ["total_price", "total_freight", "total_weight_g", "max_volume_cm3",
                "max_distance_km"]  # fmt: skip
CATEGORICAL = ["customer_state", "main_payment_type", "handoff_weekday", "handoff_month"]
RARE = "__rare__"
# Lateness is a step, not a slope, in days left at handoff: in training, orders with under
# 5 days left are ~66% late and every longer group ~10%. A straight line cannot draw that
# step, so logistic regression also gets a flag for it.
CLIFF_DAYS = 5.0
CLIFF_FLAG = "remaining_days__under_5"


def _median(s: pl.Series) -> float:
    m = s.drop_nulls().to_numpy()
    return float(np.median(m)) if len(m) else 0.0


class DesignMatrix:
    def __init__(self, features: list[str], *, rare_min_count: int = 30) -> None:
        self.features = features
        self.rare_min_count = rare_min_count
        self.categorical = [f for f in features if f in CATEGORICAL]
        self.numeric_names = [f for f in features if f not in CATEGORICAL]
        if "remaining_days" in self.numeric_names:
            self.numeric_names.append(CLIFF_FLAG)
        self.medians_: dict[str, float] = {}
        self.with_missing_: list[str] = []
        self.levels_: dict[str, list[str]] = {}
        self.center_ = np.zeros(0)
        self.scale_ = np.ones(0)
        self.feature_names: list[str] = []

    def _numeric_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for f in self.numeric_names:
            if f == CLIFF_FLAG:
                exprs.append((pl.col("remaining_days") < CLIFF_DAYS).cast(pl.Float64).alias(f))
                continue
            e = pl.col(f).cast(pl.Float64)
            if f in LOG_FEATURES:
                e = e.clip(lower_bound=0).log1p()
            exprs.append(e.alias(f))
        return df.select(exprs)

    def fit(self, df: pl.DataFrame) -> DesignMatrix:
        raw = self._numeric_raw(df)
        self.medians_ = {f: _median(raw[f]) for f in self.numeric_names}
        self.with_missing_ = [f for f in self.numeric_names if raw[f].null_count() > 0]
        for f in self.categorical:
            counts = df[f].cast(pl.String).value_counts()
            keep = counts.filter(pl.col("count") >= self.rare_min_count)[f].drop_nulls()
            self.levels_[f] = sorted(keep.to_list())
        filled = self._filled(raw, df.height)
        self.center_ = filled.mean(axis=0) if filled.shape[1] else np.zeros(0)
        std = filled.std(axis=0) if filled.shape[1] else np.zeros(0)
        self.scale_ = np.where(std > 0, std, 1.0)  # a constant column stays 0 after centring
        self.feature_names = (
            list(self.numeric_names)
            + [f"{f}__missing" for f in self.with_missing_]
            + [f"{f}={lvl}" for f in self.categorical for lvl in [*self.levels_[f], RARE]]
        )
        return self

    def _filled(self, raw: pl.DataFrame, n_rows: int) -> np.ndarray:
        cols = [raw[f].fill_null(self.medians_[f]).to_numpy() for f in self.numeric_names]
        # An empty select has height 0, so the row count comes from the source frame.
        return np.column_stack(cols) if cols else np.zeros((n_rows, 0))

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        raw = self._numeric_raw(df)
        numeric = (self._filled(raw, df.height) - self.center_) / self.scale_
        missing = [raw[f].is_null().cast(pl.Float64).to_numpy() for f in self.with_missing_]
        onehot = []
        for f in self.categorical:
            known = self.levels_[f]
            values = df[f].cast(pl.String).fill_null(RARE)
            values = values.replace_strict({k: k for k in known}, default=RARE)
            onehot += [(values == lvl).cast(pl.Float64).to_numpy() for lvl in [*known, RARE]]
        parts = [numeric] + [np.column_stack(c) for c in (missing, onehot) if c]
        return np.hstack(parts).astype(np.float64)
