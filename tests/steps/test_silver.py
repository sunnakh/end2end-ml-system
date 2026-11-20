from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from late_delivery.config import Settings
from late_delivery.platform.pipeline.runner import new_context
from late_delivery.platform.storage import list_partitions, read_partition, read_table
from late_delivery.steps import (
    ingest_bronze,
    merge_current,
    read_current,
    simulate,
    validate_to_silver,
)
from tests.conftest import FIRST_DAY, LAST_DAY, make_settings, write_mini_olist


def _run(settings: Settings, until: date = LAST_DAY) -> None:
    ds = FIRST_DAY
    while ds <= until:
        ctx = new_context(settings, ds)
        for fn in (simulate, ingest_bronze, validate_to_silver, merge_current):
            fn(ctx)
        ds += timedelta(days=1)


@pytest.fixture(scope="module")
def lake(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    tmp = tmp_path_factory.mktemp("silver")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LATE_DELIVERY_PII_SALT", "test-salt")
        s = make_settings(write_mini_olist(tmp / "raw"), tmp / "lake")
        _run(s)
        yield s


def _root(s: Settings) -> str:
    return s.paths.data_root


def test_bronze_all_strings_with_lineage(lake: Settings) -> None:
    b = read_partition(_root(lake), "bronze", "orders", FIRST_DAY)
    assert {"_ds", "_source_file", "_row_number"} <= set(b.columns)
    assert all(dt == pl.String for c, dt in b.schema.items() if c != "_row_number")
    assert b["_source_file"][0] == "orders.csv"


def test_bad_fixture_row_quarantined_ts_order(lake: Settings) -> None:
    q = read_table(_root(lake), "quarantine", "orders")
    assert q is not None
    o6 = q.filter(pl.col("order_id") == "o6")
    # Handed off 24h before purchase: every event of the order carries the impossible date.
    assert sorted(o6["_change_type"].to_list()) == ["approved", "created", "delivered", "shipped"]
    assert all("TS_ORDER_VIOLATION" in c for c in o6["reason_codes"].to_list())


def test_no_raw_customer_id_in_silver(lake: Settings) -> None:
    for layer in ("silver", "quarantine", "gold"):
        base = Path(_root(lake)) / layer
        for f in base.rglob("*.parquet") if base.exists() else []:
            df = pl.read_parquet(f)
            for col in ("customer_id", "customer_unique_id"):
                if col in df.columns:
                    assert not set(df[col].drop_nulls().to_list()) & {"c1", "c2", "u1", "u2"}


def test_no_review_text_anywhere_past_bronze(lake: Settings) -> None:
    for layer in ("silver", "quarantine"):
        for f in (Path(_root(lake)) / layer).rglob("*.parquet"):
            cols = pl.read_parquet_schema(f)
            assert "review_comment_message" not in cols and "review_comment_title" not in cols


def test_reviews_outcome_has_scores(lake: Settings) -> None:
    r = read_table(_root(lake), "silver", "reviews_outcome")
    assert r is not None
    assert r.columns == ["order_id", "review_score", "review_answer_timestamp"]
    assert sorted(r["review_score"].to_list()) == [1, 5]


def test_same_day_events_current_is_approved(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    _run(s, until=date(2017, 1, 5))
    cur = read_current(_root(s))
    assert cur.filter(pl.col("order_id") == "o5")["order_status"].to_list() == ["approved"]


def test_current_final_states(lake: Settings) -> None:
    cur = read_current(_root(lake)).sort("order_id")
    status = dict(zip(cur["order_id"], cur["order_status"], strict=True))
    assert status == {
        "o1": "delivered", "o2": "delivered", "o3": "canceled", "o4": "delivered",
        "o5": "shipped",
    }  # fmt: skip
    o1 = cur.filter(pl.col("order_id") == "o1").row(0, named=True)
    assert o1["order_approved_at"] is not None and o1["order_delivered_carrier_date"] is not None


def test_only_latest_current_partition_kept(lake: Settings) -> None:
    assert list_partitions(_root(lake), "silver", "orders_current") == [LAST_DAY]


def test_circuit_breaker_fails_above_rate(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(
        mini_olist,
        tmp_path / "lake",
        validation={"max_quarantine_rate": 0.0, "min_rows_for_breaker": 0},
    )
    with pytest.raises(RuntimeError, match="quarantine rate"):
        _run(s, until=date(2017, 1, 5))


def test_empty_day_writes_empty_partitions(lake: Settings) -> None:
    ds = date(2017, 1, 14)
    df = read_partition(_root(lake), "silver", "orders_changelog", ds)
    assert df.height == 0
    assert df.schema["order_purchase_timestamp"] == pl.Datetime("us")


def test_items_of_quarantined_order_are_orphaned(lake: Settings) -> None:
    q = read_table(_root(lake), "quarantine", "order_items")
    assert q is not None
    assert q.select("order_id", "reason_codes").rows() == [("o6", ["ORPHAN_FK"])]
    items = read_table(_root(lake), "silver", "order_items")
    assert items is not None and items.height == 6


def test_geo_prefix_median_excludes_out_of_range(lake: Settings) -> None:
    g = read_table(_root(lake), "silver", "geo_prefix")
    assert g is not None
    row = g.filter(pl.col("zip_code_prefix") == "22041").row(0, named=True)
    assert row["lat"] == pytest.approx(-22.968)
    assert g.height == 4


def test_products_english_category(lake: Settings) -> None:
    p = read_table(_root(lake), "silver", "products")
    assert p is not None
    cats = dict(zip(p["product_id"], p["product_category_name_english"], strict=True))
    assert cats == {"p1": "health_beauty", "p2": "unknown"}


def test_quarantine_counts_recorded(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    _run(s, until=date(2017, 1, 4))
    ctx = new_context(s, date(2017, 1, 5))
    for fn in (simulate, ingest_bronze, validate_to_silver):
        fn(ctx)
    assert ctx.quarantine_counts["orders"] == {"TS_ORDER_VIOLATION": 1}


def test_latest_state_follows_lifecycle_not_clock() -> None:
    from late_delivery.steps import latest_state

    ts = pl.Datetime("us")
    changelog = pl.DataFrame(
        {
            "order_id": ["x", "x", "x"],
            "order_status": ["created", "shipped", "approved"],
            "order_approved_at": [None, None, "2018-06-12 23:31:02"],
            "order_delivered_carrier_date": [None, "2018-06-11 14:54:00", "2018-06-11 14:54:00"],
            "_change_ts": ["2018-06-07 19:03:12", "2018-06-11 14:54:00", "2018-06-12 23:31:02"],
            "_event_rank": [0, 2, 1],
        }
    ).with_columns(
        pl.col("order_approved_at", "order_delivered_carrier_date", "_change_ts").str.to_datetime(
            time_unit="us"
        )
    )
    assert changelog.schema["_change_ts"] == ts
    state = latest_state(changelog.lazy()).collect().row(0, named=True)
    assert state["order_status"] == "shipped"
    assert state["order_approved_at"] is not None


def test_reviews_outcome_one_row_per_order_latest_answer() -> None:
    from late_delivery.steps import latest_review_per_order

    ts = pl.Datetime("us")
    reviews = pl.DataFrame(
        {
            "order_id": ["a", "a", "b"],
            "review_score": [1, 4, 5],
            "review_answer_timestamp": ["2017-01-02 10:00:00", "2017-01-05 10:00:00",
                                        "2017-01-03 10:00:00"],
        }
    ).with_columns(pl.col("review_answer_timestamp").str.to_datetime(time_unit="us"))  # fmt: skip
    out = latest_review_per_order(reviews).sort("order_id")
    assert out.schema["review_answer_timestamp"] == ts
    assert out.select("order_id", "review_score").rows() == [("a", 4), ("b", 5)]


def test_breaker_trip_on_rerun_writes_nothing(mini_olist: Path, tmp_path: Path) -> None:
    import shutil

    s = make_settings(mini_olist, tmp_path / "lake")
    _run(s, until=FIRST_DAY)
    # Remove one already-written partition; a failed rerun must not recreate any of the day.
    shutil.rmtree(tmp_path / "lake" / "quarantine" / "orders" / "ds=2017-01-02")
    strict = make_settings(
        mini_olist, tmp_path / "lake",
        validation={"max_quarantine_rate": 0.0, "min_rows_for_breaker": 0},
    )  # fmt: skip
    ctx = new_context(strict, FIRST_DAY)
    with pytest.raises(RuntimeError, match="geolocation .* quarantine rate"):
        validate_to_silver(ctx)
    assert FIRST_DAY not in list_partitions(_root(s), "quarantine", "orders")


def test_validate_refuses_lake_built_with_other_salt(
    mini_olist: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from late_delivery.platform.pii import SaltMismatchError

    s = make_settings(mini_olist, tmp_path / "lake")
    _run(s, until=FIRST_DAY)
    monkeypatch.setenv("LATE_DELIVERY_PII_SALT", "a-different-salt")
    ctx = new_context(s, date(2017, 1, 3))
    simulate(ctx)
    ingest_bronze(ctx)
    with pytest.raises(SaltMismatchError):
        validate_to_silver(ctx)
    assert date(2017, 1, 3) not in list_partitions(_root(s), "silver", "orders_changelog")
