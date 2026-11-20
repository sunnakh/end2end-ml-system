"""Monitoring: daily drift report and quarantine rates."""

from __future__ import annotations

import json
from datetime import date, timedelta

import fsspec
import polars as pl

from late_delivery.features import FEATURE_COLUMNS
from late_delivery.platform.drift import psi
from late_delivery.platform.pipeline.runner import Context
from late_delivery.platform.storage import (
    list_partitions,
    read_partition,
)
from late_delivery.steps._common import (
    TABLE_ORDER,
)

# A single day always "drifts" on calendar features versus a multi-month reference.
CALENDAR_FEATURES = ("handoff_weekday", "handoff_month")
DRIFT_FEATURES = [c for c in FEATURE_COLUMNS if c not in CALENDAR_FEATURES]


def _scoring_between(ctx: Context, start: date, end: date) -> pl.DataFrame | None:
    days = [d for d in list_partitions(ctx.root, "gold", "scoring") if start <= d <= end]
    if not days:
        return None
    return pl.concat(
        [read_partition(ctx.root, "gold", "scoring", d) for d in days], how="diagonal_relaxed"
    )


def drift_scores(
    reference: pl.DataFrame | None, window: pl.DataFrame | None, min_rows: int
) -> tuple[dict[str, float | None], str]:
    """PSI per non-calendar feature, and a status: ok, too_few_orders or no_reference."""
    empty: dict[str, float | None] = {col: None for col in DRIFT_FEATURES}
    if reference is None or reference.height == 0:
        return empty, "no_reference"
    if window is None or window.height < max(min_rows, 1):
        return empty, "too_few_orders"
    return {col: psi(reference[col], window[col]) for col in DRIFT_FEATURES}, "ok"


def drift_report(ctx: Context) -> None:
    """Trailing-window PSI of scoring features vs the reference period, plus quarantine rates."""
    cfg = ctx.settings.drift
    current = read_partition(ctx.root, "gold", "scoring", ctx.ds)
    window_start = ctx.ds - timedelta(days=cfg.window_days - 1)
    window = _scoring_between(ctx, window_start, ctx.ds)
    reference = _scoring_between(ctx, cfg.reference_start, cfg.reference_end)
    scores, status = drift_scores(reference, window, cfg.min_rows)
    quarantine: dict[str, dict[str, float | int]] = {}
    for table in TABLE_ORDER:
        bronze_rows = read_partition(ctx.root, "bronze", table, ctx.ds).height
        q_rows = read_partition(ctx.root, "quarantine", table, ctx.ds).height
        rate = q_rows / bronze_rows if bronze_rows else 0.0
        quarantine[table] = {"rows": q_rows, "bronze_rows": bronze_rows, "rate": rate}
    report = {
        "ds": ctx.ds.isoformat(),
        "scoring_rows": current.height,
        "window": {
            "start": window_start.isoformat(),
            "end": ctx.ds.isoformat(),
            "rows": 0 if window is None else window.height,
        },
        "reference": None
        if reference is None
        else {
            "start": cfg.reference_start.isoformat(),
            "end": cfg.reference_end.isoformat(),
            "rows": reference.height,
        },
        "status": status,
        "psi": scores,
        "quarantine": quarantine,
    }
    fs, base = fsspec.core.url_to_fs(f"{ctx.root}/reports/drift/ds={ctx.ds:%Y-%m-%d}")
    fs.makedirs(base, exist_ok=True)
    with fs.open(f"{base}/report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
