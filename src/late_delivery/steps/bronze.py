"""Bronze: store each day's files untouched, with lineage columns."""

from __future__ import annotations

import fsspec
import polars as pl

from late_delivery.platform.pipeline.runner import Context
from late_delivery.source.simulator import (
    RAW_FILES,
    landing_dir,
)
from late_delivery.steps._common import (
    _write,
)


def ingest_bronze(ctx: Context) -> None:
    fs, base = fsspec.core.url_to_fs(landing_dir(ctx.root, ctx.ds))
    if not fs.exists(base):
        raise FileNotFoundError(f"no landing files for {ctx.ds}: {base}")
    for table in RAW_FILES:
        with fs.open(f"{base}/{table}.csv", "rb") as f:
            df = pl.read_csv(f, infer_schema=False)
        df = df.with_columns(
            pl.lit(ctx.ds.isoformat()).alias("_ds"),
            pl.lit(f"{table}.csv").alias("_source_file"),
            pl.int_range(pl.len(), dtype=pl.Int64).alias("_row_number"),
        )
        _write(ctx, df, "bronze", table, ["_row_number"])
