from typing import Any

import polars as pl
import pytest

from late_delivery.platform.validation import FileLevelError, validate
from late_delivery.source.contracts import CONTRACTS

ITEM = {
    "order_id": "o1",
    "order_item_id": "1",
    "product_id": "p1",
    "seller_id": "s1",
    "shipping_limit_date": "2017-01-05 10:00:00",
    "price": "10.5",
    "freight_value": "3.0",
}
ORDER = {
    "order_id": "o1",
    "customer_id": "c1",
    "order_status": "shipped",
    "order_purchase_timestamp": "2017-01-01 10:00:00",
    "order_approved_at": "2017-01-01 11:00:00",
    "order_delivered_carrier_date": "2017-01-03 09:00:00",
    "order_delivered_customer_date": "",
    "order_estimated_delivery_date": "2017-01-20 00:00:00",
    "_change_ts": "2017-01-03 09:00:00",
    "_change_type": "shipped",
    "_event_rank": "2",
}
GEO = {
    "geolocation_zip_code_prefix": "01037",
    "geolocation_lat": "-23.54",
    "geolocation_lng": "-46.63",
    "geolocation_city": "sao paulo",
    "geolocation_state": "SP",
}
KNOWN = {
    "orders": pl.Series(["o1", "o2"]),
    "products": pl.Series(["p1"]),
    "sellers": pl.Series(["s1"]),
}


def _bronze(rows: list[dict[str, Any]]) -> pl.DataFrame:
    df = pl.DataFrame(rows, schema={k: pl.String for k in rows[0]})
    return df.with_columns(pl.int_range(pl.len()).cast(pl.String).alias("_row_number"))


def _items(*overrides: dict[str, str]) -> pl.DataFrame:
    rows = [ITEM | {"order_item_id": str(i + 2)} | o for i, o in enumerate(overrides)]
    return _bronze([ITEM, *rows])


def _codes(q: pl.DataFrame) -> list[list[str]]:
    return [sorted(c) for c in q["reason_codes"].to_list()]


def test_type_parse_error() -> None:
    r = validate(_items({"price": "abc"}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["TYPE_PARSE_ERROR"]]
    assert r.valid.height == 1


def test_null_required() -> None:
    r = validate(_items({"product_id": ""}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["NULL_REQUIRED"]]


def test_exact_duplicate_dropped_and_counted() -> None:
    r = validate(_bronze([ITEM, ITEM]), CONTRACTS["order_items"], KNOWN)
    assert r.valid.height == 1
    assert r.duplicates_removed == 1
    assert r.quarantine.height == 0


def test_conflicting_duplicate_quarantined() -> None:
    r = validate(_bronze([ITEM, ITEM | {"price": "99"}]), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["DUPLICATE_KEY"]]
    assert r.valid["price"].to_list() == [10.5]


def test_negative_amount() -> None:
    r = validate(_items({"freight_value": "-1"}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["NEGATIVE_AMOUNT"]]


def test_ts_order_violation_handoff_before_purchase() -> None:
    bad = ORDER | {"order_id": "o2", "order_delivered_carrier_date": "2016-12-30 09:00:00"}
    r = validate(_bronze([ORDER, bad]), CONTRACTS["orders"], KNOWN)
    assert _codes(r.quarantine) == [["TS_ORDER_VIOLATION"]]
    assert r.valid["order_id"].to_list() == ["o1"]


def test_orphan_fk() -> None:
    r = validate(_items({"order_id": "ghost"}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["ORPHAN_FK"]]


def test_invalid_enum() -> None:
    r = validate(
        _bronze([ORDER, ORDER | {"order_id": "o2", "order_status": "lost"}]),
        CONTRACTS["orders"],
        KNOWN,
    )
    assert _codes(r.quarantine) == [["INVALID_ENUM"]]


def test_geo_out_of_range() -> None:
    r = validate(_bronze([GEO, GEO | {"geolocation_lat": "45.0"}]), CONTRACTS["geolocation"], KNOWN)
    assert _codes(r.quarantine) == [["OUT_OF_RANGE"]]


def test_multiple_codes_one_row() -> None:
    r = validate(_items({"order_id": "ghost", "price": "-5"}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["NEGATIVE_AMOUNT", "ORPHAN_FK"]]
    assert r.quarantine["reason_detail"][0] != ""


def test_missing_column_raises_file_level_error() -> None:
    with pytest.raises(FileLevelError, match="price"):
        validate(_items().drop("price"), CONTRACTS["order_items"], KNOWN)


def test_valid_rows_typed() -> None:
    r = validate(_items(), CONTRACTS["order_items"], KNOWN)
    assert r.valid.schema["price"] == pl.Float64
    assert r.valid.schema["order_item_id"] == pl.Int64
    assert r.valid.schema["shipping_limit_date"] == pl.Datetime("us")
    assert "_row_number" not in r.valid.columns


def test_quarantine_keeps_original_values() -> None:
    r = validate(_items({"price": "abc"}), CONTRACTS["order_items"], KNOWN)
    assert r.quarantine["price"].to_list() == ["abc"]
    assert r.quarantine.schema["price"] == pl.String


def test_empty_input_gives_empty_typed_output() -> None:
    empty = _items().clear()
    r = validate(empty, CONTRACTS["order_items"], KNOWN)
    assert r.valid.height == 0 and r.quarantine.height == 0
    assert r.valid.schema["price"] == pl.Float64


def test_delivered_event_without_date_is_null_required() -> None:
    delivered = ORDER | {
        "order_id": "o2",
        "order_status": "delivered",
        "_change_type": "delivered",
        "_event_rank": "3",
        "order_delivered_customer_date": "",
    }
    r = validate(_bronze([ORDER, delivered]), CONTRACTS["orders"], KNOWN)
    assert _codes(r.quarantine) == [["NULL_REQUIRED"]]
    assert "order_delivered_customer_date" in r.quarantine["reason_detail"][0]


def test_handoff_within_two_hours_before_purchase_is_valid() -> None:
    # purchase 10:00, handoff 08:15 -> 1h45m earlier: clock quirk, kept
    quirk = ORDER | {"order_id": "o2", "order_delivered_carrier_date": "2017-01-01 08:15:00"}
    r = validate(_bronze([ORDER, quirk]), CONTRACTS["orders"], KNOWN)
    assert r.quarantine.height == 0
    assert r.valid.height == 2


def test_handoff_more_than_two_hours_before_purchase_quarantined() -> None:
    bad = ORDER | {"order_id": "o2", "order_delivered_carrier_date": "2017-01-01 07:59:59"}
    r = validate(_bronze([ORDER, bad]), CONTRACTS["orders"], KNOWN)
    assert _codes(r.quarantine) == [["TS_ORDER_VIOLATION"]]


def test_delivery_before_handoff_has_no_tolerance() -> None:
    bad = ORDER | {
        "order_id": "o2",
        "order_delivered_carrier_date": "2017-01-03 09:00:00",
        "order_delivered_customer_date": "2017-01-03 08:30:00",
    }
    r = validate(_bronze([ORDER, bad]), CONTRACTS["orders"], KNOWN)
    assert _codes(r.quarantine) == [["TS_ORDER_VIOLATION"]]


@pytest.mark.parametrize("value", ["inf", "1e30", "9999999999999999999", "-inf"])
def test_huge_or_infinite_integer_is_parse_error(value: str) -> None:
    r = validate(_items({"order_item_id": value}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["TYPE_PARSE_ERROR"]]
    assert r.valid.height == 1


@pytest.mark.parametrize("value", ["nan", "NaN", "inf"])
def test_nan_or_infinite_float_is_parse_error(value: str) -> None:
    r = validate(_items({"price": value}), CONTRACTS["order_items"], KNOWN)
    assert _codes(r.quarantine) == [["TYPE_PARSE_ERROR"]]
    assert r.valid.height == 1
