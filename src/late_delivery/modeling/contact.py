"""The daily contact decision and the metrics that judge it.

Each handoff day, contact the top K orders by score, K = max(1, floor(share x day's orders)),
and only those above `min_score` (the break-even probability). A day may use fewer than K.
"""

from __future__ import annotations

import math

import polars as pl


def _daily_budget(budget_share: float) -> pl.Expr:
    return (pl.len().over("handoff_date") * budget_share).floor().clip(lower_bound=1)


def select_contacts(
    scored: pl.DataFrame, *, budget_share: float, min_score: float | None
) -> pl.DataFrame:
    """Adds Boolean `contacted`. `scored` needs order_id, handoff_date, score.

    Ties in score go to the smaller order_id so the choice is reproducible. Scores of -inf
    (the rule baseline's unflagged orders) are never contacted. Row order is preserved.
    """
    threshold = min_score if min_score is not None else float("-inf")
    return (
        scored.with_row_index("_row")
        .sort(["handoff_date", "score", "order_id"], descending=[False, True, False])
        .with_columns(
            (
                (pl.int_range(1, pl.len() + 1).over("handoff_date") <= _daily_budget(budget_share))
                & (pl.col("score") > threshold)
            ).alias("contacted")
        )
        .sort("_row")
        .drop("_row")
    )


def decision_metrics(
    scored: pl.DataFrame, *, budget_share: float, contact_cost: float, late_cost: float
) -> dict[str, float]:
    """Recall@K, precision@K, lift and simulated net savings over a period of days.

    Sums over all days, so busy days weigh more, matching the money actually saved.
    `budget` is the number of contacts allowed; `contacts` how many were used.
    """
    orders = scored.height
    late = int(scored["late"].sum())
    contacts = int(scored["contacted"].sum())
    caught = scored.filter(pl.col("contacted") & (pl.col("late") == 1)).height
    budget_total = (
        scored.group_by("handoff_date")
        .agg((pl.len() * budget_share).floor().clip(lower_bound=1).alias("k"))["k"]
        .sum()
        if orders
        else 0
    )
    late_rate = late / orders if orders else math.nan
    precision = caught / contacts if contacts else math.nan
    lift = precision / late_rate if late and contacts else math.nan
    return {
        "orders": float(orders),
        "late": float(late),
        "late_rate": late_rate,
        "budget": float(budget_total),
        "contacts": float(contacts),
        "caught": float(caught),
        "recall_at_k": caught / late if late else math.nan,
        "precision_at_k": precision,
        "lift_at_k": lift,
        "net_savings": late_cost * caught - contact_cost * contacts,
    }
