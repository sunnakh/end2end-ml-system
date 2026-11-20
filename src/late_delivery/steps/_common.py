"""Shared constants and helpers for the pipeline steps."""

from __future__ import annotations

import polars as pl

from late_delivery.platform.pipeline.runner import Context
from late_delivery.platform.storage import (
    write_partition,
)

TABLE_ORDER = (
    "orders",
    "category_translation",
    "sellers",
    "products",
    "geolocation",
    "customers",
    "order_items",
    "order_payments",
    "order_reviews",
)
CUSTOMER_ID_COLS = ("customer_id", "customer_unique_id")
REVIEW_TEXT_COLS = ("review_comment_title", "review_comment_message")
BRONZE_LINEAGE = ("_ds", "_source_file", "_row_number")


def _write(ctx: Context, df: pl.DataFrame, layer: str, table: str, sort_by: list[str]) -> None:
    ctx.outputs.append(write_partition(df, ctx.root, layer, table, ctx.ds, sort_by))
