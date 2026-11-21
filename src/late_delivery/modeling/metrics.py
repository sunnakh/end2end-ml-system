"""Ranking and calibration metrics, written from scratch (tests check them against sklearn).

y is a 0/1 array of outcomes, s a score array (higher = more likely late), p a probability.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _ranks(s: np.ndarray) -> np.ndarray:
    """1-based ranks, tied values sharing their average rank."""
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average of positions i..j, 1-based
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score of a random late order > score of a random on-time order), ties count 1/2.

    Mann-Whitney form: (sum of late ranks - n_late(n_late+1)/2) / (n_late * n_on_time).
    """
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(np.asarray(s, dtype=np.float64))
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall curve as a step sum: sum over thresholds of
    (recall gained) x (precision at that threshold). Tied scores form one threshold."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    s_sorted, y_sorted = np.asarray(s)[order], y[order]
    tp = np.cumsum(y_sorted)
    # Keep only the last position of each run of equal scores: one point per threshold.
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp, seen = tp[last], np.arange(1, len(y) + 1)[last]
    precision = tp / seen
    recall_gain = np.diff(np.r_[0, tp]) / n_pos
    return float((recall_gain * precision).sum())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability against the 0/1 outcome."""
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y)) ** 2))


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Equal-count bins by predicted probability: mean prediction vs observed late share."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = [
        (b, float(p[idx].mean()), float(np.asarray(y)[idx].mean()), len(idx))
        for b, idx in enumerate(chunks)
        if len(idx)
    ]
    return pl.DataFrame(rows, schema=["bin", "mean_pred", "frac_late", "n"], orient="row")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: row-weighted mean |mean prediction - late share| per bin."""
    bins = reliability_bins(y, p, n_bins)
    n = bins["n"].to_numpy().astype(np.float64)
    gap = np.abs(bins["mean_pred"].to_numpy() - bins["frac_late"].to_numpy())
    return float((gap * n).sum() / n.sum())


def pr_points(y: np.ndarray, s: np.ndarray) -> pl.DataFrame:
    """Precision and recall at every distinct score threshold, for plotting."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    s_sorted, tp = np.asarray(s)[order], np.cumsum(y[order])
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    seen = np.arange(1, len(y) + 1)[last]
    n_pos = max(int(y.sum()), 1)
    return pl.DataFrame({"precision": tp[last] / seen, "recall": tp[last] / n_pos})
