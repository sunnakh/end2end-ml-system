"""Synthetic snapshots shaped like gold/training, for model and experiment tests."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from late_delivery.features import FEATURE_COLUMNS, HANDOFF

STATES = ["SP", "RJ", "MG", "BA", "PR", "AM"]


def synthetic_snapshot(
    n: int, seed: int, start: datetime = datetime(2017, 1, 1), days: int = 640
) -> pl.DataFrame:
    """Late depends on handling time, remaining days and state; other columns are noise."""
    rng = np.random.default_rng(seed)
    handling = rng.gamma(2.0, 24.0, n)
    remaining = rng.normal(15, 5, n)
    state = rng.choice(STATES, n)
    median = np.where(rng.random(n) < 0.1, np.nan, rng.gamma(2.0, 24.0, n))
    logit = -3 + 0.03 * (handling - 48) - 0.15 * (remaining - 15) + 1.5 * (state == "BA")
    late = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    day, hour = rng.integers(0, days, n), rng.integers(0, 24, n)
    handoff = [start + timedelta(days=int(d), hours=int(h)) for d, h in zip(day, hour, strict=True)]
    cols = {
        "promised_days": remaining + 8,
        "remaining_days": remaining,
        "approval_delay_h": rng.gamma(1.0, 10.0, n),
        "handling_time_h": handling,
        "n_items": rng.integers(1, 4, n),
        "n_sellers": np.ones(n, dtype=np.int64),
        "total_price": rng.gamma(2.0, 60.0, n),
        "total_freight": rng.gamma(2.0, 10.0, n),
        "total_weight_g": rng.gamma(2.0, 800.0, n),
        "max_volume_cm3": rng.gamma(2.0, 5000.0, n),
        "customer_state": state,
        "same_state_all": rng.random(n) < 0.4,
        "max_distance_km": rng.gamma(2.0, 300.0, n),
        "main_payment_type": rng.choice(["credit_card", "boleto"], n),
        "max_installments": rng.integers(1, 10, n),
        "handoff_weekday": [h.isoweekday() for h in handoff],
        "handoff_month": [h.month for h in handoff],
        "seller_prior_orders": rng.integers(0, 200, n),
        "seller_prior_late_rate": rng.random(n) * 0.2,
        "seller_prior_median_handling_h": median,
    }
    df = pl.DataFrame({"order_id": [f"o{i:06d}" for i in range(n)], HANDOFF: handoff, **cols})
    df = df.with_columns(
        pl.col("seller_prior_median_handling_h").fill_nan(None),
        pl.lit(late).alias("late"),
        pl.lit(False).alias("late_without_delivery"),
        pl.col(HANDOFF).dt.date().alias("handoff_date"),
    )
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    return df
