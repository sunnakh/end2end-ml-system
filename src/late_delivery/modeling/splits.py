"""Time splits and evaluation slices on a training snapshot."""

from __future__ import annotations

from datetime import date

import polars as pl

from late_delivery.features import HANDOFF

_REGIONS = {
    "North": "AC AP AM PA RO RR TO",
    "Northeast": "AL BA CE MA PB PE PI RN SE",
    "Center-West": "DF GO MT MS",
    "Southeast": "ES MG RJ SP",
    "South": "PR RS SC",
}
REGION_OF_STATE: dict[str, str] = {
    uf: region for region, ufs in _REGIONS.items() for uf in ufs.split()
}


def prepare(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Adds handoff_date, month ("YYYY-MM"), region and seller_size to a snapshot."""
    prior = pl.col("seller_prior_orders")
    return snapshot.with_columns(
        pl.col(HANDOFF).dt.date().alias("handoff_date"),
        pl.col(HANDOFF).dt.strftime("%Y-%m").alias("month"),
        pl.col("customer_state")
        .replace_strict(REGION_OF_STATE, default="Unknown", return_dtype=pl.String)
        .fill_null("Unknown")
        .alias("region"),
        pl.when(prior == 0)
        .then(pl.lit("0"))
        .when(prior < 10)
        .then(pl.lit("1-9"))
        .when(prior < 50)
        .then(pl.lit("10-49"))
        .otherwise(pl.lit("50+"))
        .alias("seller_size"),
    )


def months_back(end_exclusive: date, months: int) -> date:
    """First day of the month `months` months before `end_exclusive` (a month start)."""
    index = end_exclusive.year * 12 + (end_exclusive.month - 1) - months
    return date(index // 12, index % 12 + 1, 1)


def between(df: pl.DataFrame, start: date, end_exclusive: date) -> pl.DataFrame:
    """Rows with start <= handoff_date < end_exclusive."""
    return df.filter((pl.col("handoff_date") >= start) & (pl.col("handoff_date") < end_exclusive))
