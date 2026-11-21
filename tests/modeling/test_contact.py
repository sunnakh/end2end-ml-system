import math
from datetime import date

import polars as pl
import pytest

from late_delivery.modeling.contact import decision_metrics, select_contacts

D1, D2 = date(2018, 4, 2), date(2018, 4, 3)


def _scored(rows: list[tuple[str, date, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["order_id", "handoff_date", "late", "score"], orient="row"
    ).with_columns(pl.col("late").cast(pl.Int8))


def test_budget_is_five_percent_floored() -> None:
    df = _scored([(f"a{i:02d}", D1, 0, 0.5 + i / 100) for i in range(40)])
    out = select_contacts(df, budget_share=0.05, min_score=0.10)
    assert out["contacted"].sum() == 2
    assert set(out.filter("contacted")["order_id"]) == {"a39", "a38"}


def test_small_day_still_gets_one_contact() -> None:
    df = _scored([("a", D1, 0, 0.3), ("b", D1, 1, 0.9), ("c", D1, 0, 0.2)])
    out = select_contacts(df, budget_share=0.05, min_score=0.10)
    assert out.filter("contacted")["order_id"].to_list() == ["b"]


def test_threshold_can_leave_budget_unused() -> None:
    df = _scored([(f"a{i:02d}", D1, 0, 0.05) for i in range(40)])
    out = select_contacts(df, budget_share=0.05, min_score=0.10)
    assert out["contacted"].sum() == 0


def test_no_threshold_fills_budget_exactly() -> None:
    df = _scored([(f"a{i:02d}", D1, 0, 0.05) for i in range(40)])
    assert select_contacts(df, budget_share=0.05, min_score=None)["contacted"].sum() == 2


def test_ties_broken_by_order_id() -> None:
    df = _scored([("c", D1, 0, 0.5), ("a", D1, 0, 0.5), ("b", D1, 0, 0.5)])
    out = select_contacts(df, budget_share=0.05, min_score=None)
    assert out.filter("contacted")["order_id"].to_list() == ["a"]


def test_budget_is_per_day() -> None:
    rows = [(f"x{i}", D1, 0, 0.9) for i in range(20)] + [("y", D2, 0, 0.2)]
    out = select_contacts(_scored(rows), budget_share=0.05, min_score=0.10)
    per_day = out.filter("contacted").group_by("handoff_date").len().sort("handoff_date")
    assert per_day["len"].to_list() == [1, 1]


def test_minus_infinity_scores_never_contacted() -> None:
    df = _scored([("a", D1, 1, float("-inf")), ("b", D1, 0, float("-inf"))])
    assert select_contacts(df, budget_share=0.05, min_score=None)["contacted"].sum() == 0


def _decided(n_late: int, caught: int, contacts: int) -> pl.DataFrame:
    rows = []
    for i in range(n_late):
        rows.append((f"L{i}", D1, 1, 0.0, i < caught))
    for i in range(contacts - caught):
        rows.append((f"N{i}", D1, 0, 0.0, True))
    rows += [(f"Z{i}", D1, 0, 0.0, False) for i in range(50)]
    return pl.DataFrame(
        rows, schema=["order_id", "handoff_date", "late", "score", "contacted"], orient="row"
    )


def test_decision_metrics_hand_example() -> None:
    m = decision_metrics(_decided(10, 4, 6), budget_share=0.05, contact_cost=2.0, late_cost=20.0)
    assert m["late"] == 10 and m["caught"] == 4 and m["contacts"] == 6
    assert m["recall_at_k"] == pytest.approx(0.4)
    assert m["precision_at_k"] == pytest.approx(4 / 6)
    assert m["net_savings"] == pytest.approx(20 * 4 - 2 * 6)
    assert m["lift_at_k"] == pytest.approx((4 / 6) / m["late_rate"])
    assert m["budget"] == 3  # one day of 62 orders: floor(0.05 x 62) = 3


def test_zero_contacts_precision_is_nan() -> None:
    m = decision_metrics(_decided(10, 0, 0), budget_share=0.05, contact_cost=2.0, late_cost=20.0)
    assert math.isnan(m["precision_at_k"])
    assert m["recall_at_k"] == 0.0


def test_zero_late_recall_is_nan() -> None:
    m = decision_metrics(_decided(0, 0, 3), budget_share=0.05, contact_cost=2.0, late_cost=20.0)
    assert math.isnan(m["recall_at_k"])
