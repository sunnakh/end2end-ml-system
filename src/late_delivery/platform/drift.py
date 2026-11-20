"""Population stability index (PSI) between a reference sample and a current sample."""

from __future__ import annotations

import math

import polars as pl

EPS = 1e-6
OTHER = "__other__"


def _shares(values: list[str], categories: list[str]) -> list[float]:
    n = len(values)
    counts = {c: 0 for c in categories}
    for v in values:
        counts[v if v in counts else OTHER] += 1
    return [counts[c] / n for c in categories]


def psi(reference: pl.Series, current: pl.Series, bins: int = 10) -> float | None:
    """PSI; numeric columns use reference quantile bins, others use reference categories.

    Returns None when either side has no non-null values.
    """
    ref = reference.drop_nulls()
    cur = current.drop_nulls()
    if ref.len() == 0 or cur.len() == 0:
        return None

    if ref.dtype.is_numeric():
        qs = [ref.cast(pl.Float64).quantile(i / bins, "linear") for i in range(1, bins)]
        edges = pl.Series(sorted({q for q in qs if q is not None}), dtype=pl.Float64)
        # Bin i holds values in (edge[i-1], edge[i]]: index = number of edges below the value.
        ref_bins = edges.search_sorted(ref.cast(pl.Float64), side="left").cast(pl.String).to_list()
        cur_bins = edges.search_sorted(cur.cast(pl.Float64), side="left").cast(pl.String).to_list()
        categories = [str(i) for i in range(edges.len() + 1)]
    else:
        ref_bins = ref.cast(pl.String).to_list()
        cur_bins = cur.cast(pl.String).to_list()
        categories = sorted(set(ref_bins)) + [OTHER]

    total = 0.0
    for r, c in zip(_shares(ref_bins, categories), _shares(cur_bins, categories), strict=True):
        r, c = max(r, EPS), max(c, EPS)
        total += (c - r) * math.log(c / r)
    return total
