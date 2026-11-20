from datetime import date
from pathlib import Path

import polars as pl

from late_delivery.source.simulator import (
    RAW_FILES,
    REFERENCE_TABLES,
    load_raw,
    simulate_day,
    write_landing,
)

DAY0 = date(2017, 1, 2)


def _day(mini_olist: Path, ds: date) -> dict[str, pl.DataFrame]:
    return simulate_day(load_raw(mini_olist), ds, DAY0)


def test_every_table_emitted_every_day(mini_olist: Path) -> None:
    assert set(_day(mini_olist, date(2017, 1, 15))) == set(RAW_FILES)


def test_order_appears_on_purchase_day_only_as_created(mini_olist: Path) -> None:
    orders = _day(mini_olist, DAY0)["orders"]
    assert sorted(orders["order_id"].to_list()) == ["o1", "o2"]
    assert set(orders["_change_type"].to_list()) == {"created"}


def test_delivery_row_appears_on_delivery_day(mini_olist: Path) -> None:
    orders = _day(mini_olist, date(2017, 1, 8))["orders"]
    assert orders.select("order_id", "_change_type").rows() == [("o1", "delivered")]


def test_items_emitted_with_purchase(mini_olist: Path) -> None:
    tables = _day(mini_olist, date(2017, 1, 4))
    assert tables["order_items"]["order_id"].to_list() == ["o4", "o4"]
    assert tables["customers"]["customer_id"].to_list() == ["c4"]
    assert tables["order_payments"]["order_id"].to_list() == ["o4"]


def test_reviews_emitted_on_answer_day(mini_olist: Path) -> None:
    assert _day(mini_olist, date(2017, 1, 9))["order_reviews"]["review_id"].to_list() == ["r1"]


def test_reference_tables_only_day0(mini_olist: Path) -> None:
    first = _day(mini_olist, DAY0)
    later = _day(mini_olist, date(2017, 1, 3))
    for t in REFERENCE_TABLES:
        assert first[t].height > 0
        assert later[t].height == 0


def test_no_row_dated_after_ds(mini_olist: Path) -> None:
    raw = load_raw(mini_olist)
    for day in range(2, 22):
        ds = date(2017, 1, day)
        t = simulate_day(raw, ds, DAY0)
        stamps = (
            t["orders"]["_change_ts"].to_list()
            + t["order_reviews"]["review_answer_timestamp"].to_list()
        )
        assert all(s[:10] <= ds.isoformat() for s in stamps)


def test_empty_day_writes_header_only_csvs(mini_olist: Path, tmp_path: Path) -> None:
    ds = date(2017, 1, 14)
    tables = _day(mini_olist, ds)
    assert tables["orders"].height == 0
    write_landing(tables, str(tmp_path / "lake"), ds)
    path = tmp_path / "lake" / "landing" / "ds=2017-01-14" / "orders.csv"
    back = pl.read_csv(path, infer_schema=False)
    assert back.height == 0
    assert "_change_type" in back.columns
