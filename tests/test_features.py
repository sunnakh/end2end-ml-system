from datetime import datetime

import polars as pl
import pytest

from late_delivery.features import FEATURE_COLUMNS, build_features, haversine_km

TS = pl.Datetime("us")


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _handoffs(*rows: tuple[str, str, str]) -> pl.DataFrame:
    """rows: (order_id, customer_id, handoff_ts)"""
    return pl.DataFrame(
        {
            "order_id": [r[0] for r in rows],
            "customer_id": [r[1] for r in rows],
            "order_purchase_timestamp": [_dt("2017-03-01 10:00:00")] * len(rows),
            "order_approved_at": [_dt("2017-03-01 12:00:00")] * len(rows),
            "order_delivered_carrier_date": [_dt(r[2]) for r in rows],
            "order_estimated_delivery_date": [_dt("2017-03-21 00:00:00")] * len(rows),
        },
        schema={"order_id": pl.String, "customer_id": pl.String,
                **{c: TS for c in ("order_purchase_timestamp", "order_approved_at",
                                   "order_delivered_carrier_date",
                                   "order_estimated_delivery_date")}},
    )  # fmt: skip


def _deliveries(*rows: tuple[str, str, str]) -> pl.DataFrame:
    """rows: (order_id, delivered_ts, estimated_ts) as changelog 'delivered' events."""
    return pl.DataFrame(
        {
            "order_id": [r[0] for r in rows],
            "order_delivered_customer_date": [_dt(r[1]) for r in rows],
            "order_estimated_delivery_date": [_dt(r[2]) for r in rows],
            "_change_type": ["delivered"] * len(rows),
            "_change_ts": [_dt(r[1]) for r in rows],
        },
        schema={"order_id": pl.String, "order_delivered_customer_date": TS,
                "order_estimated_delivery_date": TS, "_change_type": pl.String,
                "_change_ts": TS},
    )  # fmt: skip


def _items(*rows: tuple[str, str, str]) -> pl.DataFrame:
    """rows: (order_id, seller_id, product_id)"""
    return pl.DataFrame(
        {
            "order_id": [r[0] for r in rows],
            "order_item_id": list(range(1, len(rows) + 1)),
            "product_id": [r[2] for r in rows],
            "seller_id": [r[1] for r in rows],
            "price": [100.0] * len(rows),
            "freight_value": [10.0] * len(rows),
        }
    )


PAYMENTS = pl.DataFrame(
    {
        "order_id": ["cur", "cur"],
        "payment_sequential": [1, 2],
        "payment_type": ["voucher", "credit_card"],
        "payment_installments": [1, 6],
        "payment_value": [20.0, 90.0],
    }
)
CUSTOMERS = pl.DataFrame(
    {"customer_id": ["c"], "customer_zip_code_prefix": ["1310"], "customer_state": ["SP"]}
)
SELLERS = pl.DataFrame(
    {"seller_id": ["s1", "s2"], "seller_zip_code_prefix": ["1037", "20040"],
     "seller_state": ["SP", "RJ"]}
)  # fmt: skip
PRODUCTS = pl.DataFrame(
    {"product_id": ["p"], "product_weight_g": [500.0], "product_length_cm": [10.0],
     "product_height_cm": [10.0], "product_width_cm": [10.0]}
)  # fmt: skip
GEO = pl.DataFrame(
    {"zip_code_prefix": ["1037", "1310", "20040"], "lat": [-23.545, -23.561, -22.903],
     "lng": [-46.639, -46.656, -43.176]}
)  # fmt: skip


def _features(handoffs: pl.DataFrame, deliveries: pl.DataFrame, items: pl.DataFrame) -> dict:
    out = build_features(handoffs, deliveries, items, PAYMENTS, CUSTOMERS, SELLERS, PRODUCTS, GEO)
    return out.filter(pl.col("order_id") == "cur").row(0, named=True)


HISTORY_ITEMS = _items(("old1", "s1", "p"), ("old2", "s1", "p"), ("cur", "s1", "p"))


def test_columns() -> None:
    out = build_features(
        _handoffs(("cur", "c", "2017-03-03 10:00:00")), _deliveries(), HISTORY_ITEMS,
        PAYMENTS, CUSTOMERS, SELLERS, PRODUCTS, GEO,
    )  # fmt: skip
    assert out.columns == ["order_id", "order_delivered_carrier_date", *FEATURE_COLUMNS]


def test_seller_history_excludes_delivery_after_handoff() -> None:
    f = _features(
        _handoffs(("cur", "c", "2017-03-03 10:00:00")),
        _deliveries(("old1", "2017-03-02 09:00:00", "2017-03-01 00:00:00"),
                    ("old2", "2017-03-04 09:00:00", "2017-03-10 00:00:00")),
        HISTORY_ITEMS,
    )  # fmt: skip
    assert f["seller_prior_orders"] == 1
    assert f["seller_prior_late_rate"] == 1.0


def test_seller_history_excludes_delivery_at_exact_handoff_ts() -> None:
    f = _features(
        _handoffs(("cur", "c", "2017-03-03 10:00:00")),
        _deliveries(("old1", "2017-03-03 10:00:00", "2017-03-10 00:00:00")),
        HISTORY_ITEMS,
    )
    assert f["seller_prior_orders"] == 0
    assert f["seller_prior_late_rate"] is None


def test_two_seller_order_takes_max() -> None:
    items = _items(("old1", "s2", "p"), ("cur", "s1", "p"), ("cur", "s2", "p"))
    f = _features(
        _handoffs(("cur", "c", "2017-03-03 10:00:00")),
        _deliveries(("old1", "2017-03-02 09:00:00", "2017-03-10 00:00:00")),
        items,
    )
    assert f["n_sellers"] == 2 and f["n_items"] == 2
    assert f["seller_prior_orders"] == 1
    assert f["same_state_all"] is False
    assert f["max_distance_km"] == pytest.approx(357, abs=15)


def test_order_level_values() -> None:
    f = _features(_handoffs(("cur", "c", "2017-03-03 12:00:00")), _deliveries(), HISTORY_ITEMS)
    assert f["promised_days"] == pytest.approx(19.583, abs=1e-3)
    assert f["remaining_days"] == pytest.approx(17.5)
    assert f["approval_delay_h"] == pytest.approx(2.0)
    assert f["handling_time_h"] == pytest.approx(48.0)
    assert f["total_price"] == 100.0 and f["total_freight"] == 10.0
    assert f["total_weight_g"] == 500.0 and f["max_volume_cm3"] == 1000.0
    assert f["main_payment_type"] == "credit_card" and f["max_installments"] == 6
    assert f["customer_state"] == "SP" and f["same_state_all"] is True
    assert f["handoff_weekday"] == 5 and f["handoff_month"] == 3


def test_haversine_known_pair() -> None:
    df = pl.DataFrame({"a": [-23.55], "b": [-46.63], "c": [-22.91], "d": [-43.17]})
    km = df.select(haversine_km(pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d")))
    assert km.item() == pytest.approx(360, abs=10)


def test_one_row_per_order() -> None:
    items = _items(("cur", "s1", "p"), ("cur", "s1", "p"), ("cur", "s2", "p"))
    out = build_features(
        _handoffs(("cur", "c", "2017-03-03 10:00:00")), _deliveries(), items,
        PAYMENTS, CUSTOMERS, SELLERS, PRODUCTS, GEO,
    )  # fmt: skip
    assert out.height == 1


def test_empty_handoffs_keeps_schema() -> None:
    out = build_features(
        _handoffs(), _deliveries(), HISTORY_ITEMS, PAYMENTS, CUSTOMERS, SELLERS, PRODUCTS, GEO
    )
    assert out.height == 0
    assert out.columns == ["order_id", "order_delivered_carrier_date", *FEATURE_COLUMNS]


def _changelog(*rows: tuple[str, str, str | None, str]) -> pl.DataFrame:
    """rows: (order_id, change_type, delivered_ts or None, estimated_ts)."""
    return pl.DataFrame(
        {
            "order_id": [r[0] for r in rows],
            "_change_type": [r[1] for r in rows],
            "_change_ts": [_dt(r[2]) if r[2] else _dt("2017-02-01 10:00:00") for r in rows],
            "order_delivered_customer_date": [_dt(r[2]) if r[2] else None for r in rows],
            "order_estimated_delivery_date": [_dt(r[3]) for r in rows],
        },
        schema={"order_id": pl.String, "_change_type": pl.String, "_change_ts": TS,
                "order_delivered_customer_date": TS, "order_estimated_delivery_date": TS},
    )  # fmt: skip


HANDOFF_0303 = _handoffs(("cur", "c", "2017-03-03 10:00:00"))


def test_overdue_undelivered_prior_order_counts_as_late() -> None:
    # due 02-20, never delivered: by the 03-03 handoff it is known to be late
    log = _changelog(("old1", "created", None, "2017-02-20 00:00:00"))
    f = _features(HANDOFF_0303, log, HISTORY_ITEMS)
    assert f["seller_prior_orders"] == 1
    assert f["seller_prior_late_rate"] == 1.0


def test_undelivered_prior_order_not_yet_due_not_counted() -> None:
    log = _changelog(("old1", "created", None, "2017-03-10 00:00:00"))
    f = _features(HANDOFF_0303, log, HISTORY_ITEMS)
    assert f["seller_prior_orders"] == 0


def test_late_arrival_after_handoff_already_known_late() -> None:
    # due 02-20, delivered 03-05 (after this handoff): late was known since 02-21
    due = "2017-02-20 00:00:00"
    log = _changelog(
        ("old1", "created", None, due), ("old1", "delivered", "2017-03-05 09:00:00", due)
    )
    f = _features(HANDOFF_0303, log, HISTORY_ITEMS)
    assert f["seller_prior_orders"] == 1
    assert f["seller_prior_late_rate"] == 1.0


def test_due_today_not_yet_late() -> None:
    # due 03-03 (today): not late until the day is over
    log = _changelog(("old1", "created", None, "2017-03-03 00:00:00"))
    assert _features(HANDOFF_0303, log, HISTORY_ITEMS)["seller_prior_orders"] == 0


def test_prior_order_canceled_before_outcome_not_counted() -> None:
    log = _changelog(("old1", "created", None, "2017-02-20 00:00:00"),
                     ("old1", "closed", None, "2017-02-20 00:00:00"))  # fmt: skip
    assert _features(HANDOFF_0303, log, HISTORY_ITEMS)["seller_prior_orders"] == 0


def test_mixed_history_rate() -> None:
    log = _changelog(
        ("old1", "delivered", "2017-02-10 09:00:00", "2017-02-20 00:00:00"),  # on time
        ("old2", "created", None, "2017-02-25 00:00:00"),  # overdue, in transit
    )
    f = _features(HANDOFF_0303, log, HISTORY_ITEMS)
    assert f["seller_prior_orders"] == 2
    assert f["seller_prior_late_rate"] == 0.5
