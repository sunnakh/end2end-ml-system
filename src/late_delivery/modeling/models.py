"""The models compared in M3, behind one interface: fit(train) and score(frame).

Scores from the learned models are probabilities of a late delivery; the rule baseline
returns a ranking score only (no probability), so the 0.10 contact threshold skips it.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import polars as pl

from late_delivery.features import FEATURE_COLUMNS
from late_delivery.modeling.design import DesignMatrix
from late_delivery.modeling.logreg import fit_logreg, predict_proba


class Model(Protocol):
    name: str
    uses_threshold: bool

    @property
    def params(self) -> dict[str, Any]: ...

    def fit(self, train: pl.DataFrame) -> None: ...

    def score(self, df: pl.DataFrame) -> np.ndarray: ...


def _last_month_split(train: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """(earlier rows, rows of the last calendar month, that month as YYYY-MM)."""
    last: date = train["handoff_date"].max()  # type: ignore[assignment]
    start = date(last.year, last.month, 1)
    return (
        train.filter(pl.col("handoff_date") < start),
        train.filter(pl.col("handoff_date") >= start),
        start.strftime("%Y-%m"),
    )


class RuleBaseline:
    """Flag slow handling (above the seller's own past median) or interstate shipping;
    rank flagged orders by how far handling exceeds the median."""

    name = "rule"
    uses_threshold = False

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def fit(self, train: pl.DataFrame) -> None:
        return None  # nothing to learn

    def score(self, df: pl.DataFrame) -> np.ndarray:
        excess = (pl.col("handling_time_h") - pl.col("seller_prior_median_handling_h")).fill_null(
            0.0
        )
        slow = (pl.col("handling_time_h") > pl.col("seller_prior_median_handling_h")).fill_null(
            False
        )
        interstate = (~pl.col("same_state_all")).fill_null(False)
        out = df.select(pl.when(slow | interstate).then(excess).otherwise(float("-inf")).alias("s"))
        return out["s"].to_numpy().astype(np.float64)


class LogRegModel:
    """The from-scratch NumPy logistic regression on the prepared design matrix."""

    name = "logreg"
    uses_threshold = True

    def __init__(self, l2: float, features: list[str] | None = None) -> None:
        self.l2 = l2
        self.features = list(features) if features is not None else list(FEATURE_COLUMNS)
        self._design: DesignMatrix | None = None
        self._w = np.zeros(0)
        self._b = 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {"l2": self.l2, "features": len(self.features)}

    def fit(self, train: pl.DataFrame) -> None:
        self._design = DesignMatrix(self.features).fit(train)
        x = self._design.transform(train)
        self._w, self._b = fit_logreg(x, train["late"].to_numpy().astype(np.float64), l2=self.l2)

    def score(self, df: pl.DataFrame) -> np.ndarray:
        assert self._design is not None, "fit first"
        return predict_proba(self._design.transform(df), self._w, self._b)

    def coefficients(self) -> dict[str, float]:
        assert self._design is not None, "fit first"
        return dict(zip(self._design.feature_names, self._w.tolist(), strict=True))


_LGB_CATEGORICAL = ["customer_state", "main_payment_type"]


class LightGBMModel:
    """Gradient boosting. The number of trees is chosen by early stopping on the last
    calendar month of the training rows (never validation); the final trees are then grown
    on every training row, so the newest month is learned from too."""

    name = "lightgbm"
    uses_threshold = True
    max_rounds = 2000
    patience = 50

    def __init__(
        self, num_leaves: int, learning_rate: float, min_child_samples: int, seed: int = 0
    ) -> None:
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.seed = seed
        self.best_iteration = 0
        self.features: list[str] = []  # fixed at fit time: later code may add features
        self._fitted_rows = 0
        self._stop_month = ""
        self._levels: dict[str, list[str]] = {}
        self._booster: lgb.Booster | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "best_iteration": self.best_iteration,
            "early_stopping_month": self._stop_month,
            "fitted_rows": self._fitted_rows,
        }

    @property
    def num_trees(self) -> int:
        assert self._booster is not None, "fit first"
        return int(self._booster.num_trees())

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            if f in _LGB_CATEGORICAL:
                codes = {lvl: i for i, lvl in enumerate(self._levels[f])}
                # Unseen or missing levels get -1, which LightGBM treats as missing.
                col = df[f].cast(pl.String).replace_strict(codes, default=-1, return_dtype=pl.Int64)
                cols.append(col.fill_null(-1).cast(pl.Float64).to_numpy())
            else:
                cols.append(df[f].cast(pl.Float64).to_numpy())
        return np.column_stack(cols)

    def fit(self, train: pl.DataFrame) -> None:
        self.features = list(FEATURE_COLUMNS)
        self._levels = {
            f: sorted(train[f].cast(pl.String).drop_nulls().unique().to_list())
            for f in _LGB_CATEGORICAL
        }
        early, stop, self._stop_month = _last_month_split(train)
        params = {
            "objective": "binary",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbose": -1,
        }
        cat_idx = [self.features.index(f) for f in _LGB_CATEGORICAL]

        def dataset(rows: pl.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
            return lgb.Dataset(
                self._matrix(rows), rows["late"].to_numpy(), categorical_feature=cat_idx,
                reference=reference, free_raw_data=False,
            )  # fmt: skip

        if early.height > 0 and stop.height > 0:
            # 1. Choose the number of trees on the last month, trained on the months before.
            d_early = dataset(early)
            probe = lgb.train(
                params, d_early, num_boost_round=self.max_rounds,
                valid_sets=[dataset(stop, reference=d_early)],
                callbacks=[lgb.early_stopping(self.patience, verbose=False)],
            )  # fmt: skip
            self.best_iteration = max(int(probe.best_iteration), 1)
        else:  # a single month of data: no tail to stop on, use a fixed small budget
            self.best_iteration = 200
            self._stop_month = ""
        # 2. Grow that many trees on every training row, newest month included.
        self._booster = lgb.train(params, dataset(train), num_boost_round=self.best_iteration)
        self._fitted_rows = train.height

    def score(self, df: pl.DataFrame) -> np.ndarray:
        assert self._booster is not None, "fit first"
        out = self._booster.predict(self._matrix(df))
        return np.asarray(out, dtype=np.float64)


class PlattCalibrated:
    """Fits the base model without the last training month, then a one-feature logistic
    regression on logit(score) over that month (never on validation)."""

    uses_threshold = True

    def __init__(self, base: Model) -> None:
        self.base = base
        self.name = f"{base.name}+platt"
        self._a, self._b = 1.0, 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {**self.base.params, "platt_a": self._a, "platt_b": self._b}

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, 1e-9, 1 - 1e-9).astype(np.float64)
        out: np.ndarray = np.log(q / (1 - q))
        return out

    def fit(self, train: pl.DataFrame) -> None:
        early, last, _ = _last_month_split(train)
        if early.height == 0:
            raise ValueError("Platt calibration needs at least two months of training rows")
        self.base.fit(early)
        z = self._logit(self.base.score(last))[:, None]
        w, b = fit_logreg(z, last["late"].to_numpy().astype(np.float64), l2=1e-6)
        self._a, self._b = float(w[0]), b

    def score(self, df: pl.DataFrame) -> np.ndarray:
        z = self._logit(self.base.score(df))[:, None]
        return predict_proba(z, np.array([self._a]), self._b)


def calendar_only(l2: float = 0.01) -> LogRegModel:
    """A deliberately weak model (weekday and month only) the gate must block."""
    m = LogRegModel(l2=l2, features=["handoff_weekday", "handoff_month"])
    m.name = "calendar_only"
    return m


def save_model(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Model:
    """Loads a model this project saved. Pickle runs code on load: never load untrusted files."""
    with path.open("rb") as f:
        model: Model = pickle.load(f)  # noqa: S301 - local, self-written artifacts only
    return model
