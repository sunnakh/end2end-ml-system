from pathlib import Path

import polars as pl

from late_delivery.source.simulator import load_raw, order_cdc_events


def _events(mini_olist: Path, order_id: str) -> pl.DataFrame:
    ev = order_cdc_events(load_raw(mini_olist)["orders"])
    return ev.filter(pl.col("order_id") == order_id).sort("_event_rank")


def test_load_raw_all_strings(mini_olist: Path) -> None:
    raw = load_raw(mini_olist)
    assert set(raw) == {
        "orders", "order_items", "order_payments", "customers", "sellers", "products",
        "geolocation", "order_reviews", "category_translation",
    }  # fmt: skip
    assert all(dt == pl.String for df in raw.values() for dt in df.schema.values())


def test_delivered_order_has_four_events(mini_olist: Path) -> None:
    ev = _events(mini_olist, "o1")
    assert ev["_change_type"].to_list() == ["created", "approved", "shipped", "delivered"]
    assert ev["order_status"].to_list() == ["created", "approved", "shipped", "delivered"]
    assert ev["_event_rank"].to_list() == [0, 1, 2, 3]


def test_canceled_order_ends_with_closed(mini_olist: Path) -> None:
    ev = _events(mini_olist, "o3")
    assert ev["_change_type"].to_list() == ["created", "approved", "closed"]
    last = ev.row(-1, named=True)
    assert last["order_status"] == "canceled"
    assert last["_change_ts"] == "2017-01-04 09:00:00"


def test_in_progress_order_has_no_closed(mini_olist: Path) -> None:
    ev = _events(mini_olist, "o5")
    assert ev["_change_type"].to_list() == ["created", "approved", "shipped"]


def test_future_fields_null_on_created(mini_olist: Path) -> None:
    created = _events(mini_olist, "o1").row(0, named=True)
    assert created["_change_ts"] == "2017-01-02 10:00:00"
    assert created["order_approved_at"] is None
    assert created["order_delivered_carrier_date"] is None
    assert created["order_delivered_customer_date"] is None
    assert created["order_estimated_delivery_date"] == "2017-01-15 00:00:00"


def test_shipped_row_has_handoff_not_delivery(mini_olist: Path) -> None:
    shipped = _events(mini_olist, "o1").row(2, named=True)
    assert shipped["order_delivered_carrier_date"] == "2017-01-04 15:00:00"
    assert shipped["order_delivered_customer_date"] is None


def _orders(**cols: str | None) -> pl.DataFrame:
    base: dict[str, str | None] = {
        "order_id": "x",
        "customer_id": "c",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-06-07 19:03:12",
        "order_approved_at": "2018-06-12 23:31:02",
        "order_delivered_carrier_date": "2018-06-11 14:54:00",
        "order_delivered_customer_date": "2018-06-21 15:34:32",
        "order_estimated_delivery_date": "2018-07-05 00:00:00",
    }
    base.update(cols)
    return pl.DataFrame([base], schema={k: pl.String for k in base})


def test_handoff_before_approval_hides_future_approval() -> None:
    ev = order_cdc_events(_orders()).sort("_change_ts")
    assert ev["_change_type"].to_list() == ["created", "shipped", "approved", "delivered"]
    shipped = ev.filter(pl.col("_change_type") == "shipped").row(0, named=True)
    assert shipped["order_approved_at"] is None  # approval had not happened yet
    approved = ev.filter(pl.col("_change_type") == "approved").row(0, named=True)
    assert approved["order_delivered_carrier_date"] == "2018-06-11 14:54:00"  # already known


def test_no_row_shows_a_date_after_its_own_timestamp() -> None:
    ev = order_cdc_events(_orders())
    for row in ev.iter_rows(named=True):
        for col in ("order_approved_at", "order_delivered_carrier_date",
                    "order_delivered_customer_date"):  # fmt: skip
            if row[col] is not None:
                assert row[col] <= row["_change_ts"], (row["_change_type"], col)


def test_delivered_status_without_date_emits_delivered_event() -> None:
    ev = order_cdc_events(_orders(order_delivered_customer_date=None))
    delivered = ev.filter(pl.col("_change_type") == "delivered")
    assert delivered.height == 1
    row = delivered.row(0, named=True)
    assert row["order_delivered_customer_date"] is None
    assert row["_change_ts"] == "2018-06-12 23:31:02"  # latest known timestamp
