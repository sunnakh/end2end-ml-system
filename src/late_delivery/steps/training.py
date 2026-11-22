"""Training: labeled snapshot of orders whose outcome is known by a cut-off."""

from __future__ import annotations

import json
from datetime import date, timedelta

import fsspec
import polars as pl

from late_delivery.config import Settings
from late_delivery.features import DELIVERED, ESTIMATED, build_features, label_at_cutoff
from late_delivery.platform.storage import (
    PartitionInfo,
    content_hash,
    read_table,
)
from late_delivery.steps.silver import latest_state


def _silver_up_to(settings: Settings, table: str, up_to: date) -> pl.DataFrame:
    df = read_table(settings.paths.data_root, "silver", table, up_to)
    if df is None:
        raise FileNotFoundError(f"silver/{table} has no partitions up to {up_to}")
    return df


def build_training_set(settings: Settings, cutoff: date) -> PartitionInfo:
    """Labeled snapshot of every order handed off before `cutoff`, using data from days < cutoff."""
    known_until = cutoff - timedelta(days=1)
    changelog = _silver_up_to(settings, "orders_changelog", known_until)
    # Orders with any quarantined event have an untrustworthy outcome: leave them out.
    quarantined = read_table(settings.paths.data_root, "quarantine", "orders", known_until)
    bad_ids = (
        pl.Series("order_id", [], dtype=pl.String)
        if quarantined is None
        else quarantined["order_id"].drop_nulls().unique()
    )
    trusted = changelog.filter(~pl.col("order_id").is_in(bad_ids.implode()))
    # Features come from the handoff event itself: exactly what scoring sees on that day.
    handoffs = trusted.filter(pl.col("_change_type") == "shipped")
    features = build_features(
        handoffs,
        changelog,
        _silver_up_to(settings, "order_items", known_until),
        _silver_up_to(settings, "order_payments", known_until),
        _silver_up_to(settings, "customers", known_until),
        _silver_up_to(settings, "sellers", known_until),
        _silver_up_to(settings, "products", known_until),
        _silver_up_to(settings, "geo_prefix", known_until),
    )
    state = latest_state(trusted.lazy()).collect()
    # Label maturity: keep only orders whose promised day is over by the cut-off. An order
    # not yet due is labeled only if it already arrived (on time), never if it is still
    # travelling, so keeping it would favour early arrivals and understate the late rate.
    due = state.filter(pl.col(ESTIMATED).dt.date() < cutoff).select("order_id")
    excluded_not_yet_due = features.join(due, on="order_id", how="anti").height
    labels = label_at_cutoff(state, cutoff)
    training = (
        features.join(due, on="order_id")
        .join(labels, on="order_id")
        .filter(pl.col("late").is_not_null())
        .sort("order_id")
    )
    # Late because the promised date passed with no delivery yet (in transit or lost).
    # Reported so evaluation can compare results with and without them.
    undelivered = state.filter(pl.col(DELIVERED).is_null()).select("order_id")
    training = training.with_columns(
        ((pl.col("late") == 1) & pl.col("order_id").is_in(undelivered["order_id"].implode())).alias(
            "late_without_delivery"
        )
    )
    late_without_delivery = int(training["late_without_delivery"].sum())

    digest = content_hash(training, ["order_id"])
    path = f"{settings.paths.data_root}/gold/training/snapshot={digest[:12]}"
    fs, fs_path = fsspec.core.url_to_fs(path)
    fs.makedirs(fs_path, exist_ok=True)
    with fs.open(f"{fs_path}/part-0.parquet", "wb") as f:
        training.write_parquet(f, compression="zstd", statistics=False)
    meta = {
        "cutoff": cutoff.isoformat(),
        "rows": training.height,
        "late_rate": training.select(pl.col("late").cast(pl.Float64).mean()).item(),
        "content_hash": digest,
        "late_without_delivery": late_without_delivery,
        "excluded_not_yet_due": excluded_not_yet_due,
    }
    with fs.open(f"{fs_path}/_meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return PartitionInfo(path=path, rows=training.height, content_hash=digest)
