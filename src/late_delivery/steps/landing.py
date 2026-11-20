"""Landing: tonight's export arrives (simulated replay of Olist)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

from late_delivery.platform.pipeline.runner import Context
from late_delivery.source.simulator import (
    inject_faults,
    load_raw,
    simulate_day,
    write_landing,
)


@lru_cache(maxsize=2)
def _raw(raw_dir: str) -> dict[str, pl.DataFrame]:
    return load_raw(Path(raw_dir))


def simulate(ctx: Context) -> None:
    s = ctx.settings
    tables = simulate_day(_raw(str(s.paths.raw_dir)), ctx.ds, s.replay.start)
    injected = None
    if s.faults.enabled:
        tables, injected = inject_faults(tables, ctx.ds, s.faults.seed, s.faults.rate)
    write_landing(tables, ctx.root, ctx.ds, injected)
