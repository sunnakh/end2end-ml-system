"""Gold: point-in-time features for orders handed off on `ds`."""

from __future__ import annotations

import polars as pl

from late_delivery.features import build_features
from late_delivery.platform.pipeline.runner import Context
from late_delivery.platform.storage import (
    read_partition,
    read_table,
)
from late_delivery.steps._common import (
    _write,
)


def _silver(ctx: Context, table: str) -> pl.DataFrame:
    df = read_table(ctx.root, "silver", table, ctx.ds)
    if df is None:
        raise FileNotFoundError(f"silver/{table} has no partitions up to {ctx.ds}")
    return df


def build_gold_scoring(ctx: Context) -> None:
    """Features for orders handed to the carrier on `ds`, using only data known by then."""
    changelog = _silver(ctx, "orders_changelog")
    handoffs = read_partition(ctx.root, "silver", "orders_changelog", ctx.ds).filter(
        pl.col("_change_type") == "shipped"
    )
    features = build_features(
        handoffs,
        changelog,
        _silver(ctx, "order_items"),
        _silver(ctx, "order_payments"),
        _silver(ctx, "customers"),
        _silver(ctx, "sellers"),
        _silver(ctx, "products"),
        _silver(ctx, "geo_prefix"),
    )
    _write(ctx, features, "gold", "scoring", ["order_id"])
