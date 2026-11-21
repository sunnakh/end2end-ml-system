import json
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from late_delivery.config import Settings
from late_delivery.features import FEATURE_COLUMNS, label_at_cutoff
from late_delivery.platform.pipeline.runner import new_context
from late_delivery.steps import (
    build_training_set,
    ingest_bronze,
    merge_current,
    simulate,
    validate_to_silver,
)
from tests.conftest import FIRST_DAY, LAST_DAY, make_settings, write_mini_olist

TS = pl.Datetime("us")


@pytest.fixture(scope="module")
def lake(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    tmp = tmp_path_factory.mktemp("training")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LATE_DELIVERY_PII_SALT", "test-salt")
        s = make_settings(write_mini_olist(tmp / "raw"), tmp / "lake")
        ds = FIRST_DAY
        while ds <= LAST_DAY:
            ctx = new_context(s, ds)
            for fn in (simulate, ingest_bronze, validate_to_silver, merge_current):
                fn(ctx)
            ds += timedelta(days=1)
        yield s


def _labels(lake: Settings, cutoff: date) -> dict[str, int]:
    info = build_training_set(lake, cutoff)
    df = pl.read_parquet(f"{info.path}/part-0.parquet")
    return dict(zip(df["order_id"], df["late"], strict=True))


def _orders(*rows: tuple[str, str, str | None, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": [r[0] for r in rows],
            "order_status": [r[1] for r in rows],
            "order_delivered_customer_date": [
                datetime.fromisoformat(r[2]) if r[2] else None for r in rows
            ],
            "order_estimated_delivery_date": [datetime.fromisoformat(r[3]) for r in rows],
        },
        schema={"order_id": pl.String, "order_status": pl.String,
                "order_delivered_customer_date": TS, "order_estimated_delivery_date": TS},
    )  # fmt: skip


def test_label_rules_unit() -> None:
    orders = _orders(
        ("late", "delivered", "2017-02-11 08:00:00", "2017-02-10 00:00:00"),
        ("same_day", "delivered", "2017-02-10 23:00:00", "2017-02-10 00:00:00"),
        ("overdue", "shipped", None, "2017-02-10 00:00:00"),
        ("due_today", "shipped", None, "2017-02-15 00:00:00"),
        ("canceled", "canceled", None, "2017-02-01 00:00:00"),
    )
    got = label_at_cutoff(orders, date(2017, 2, 15)).sort("order_id")
    assert dict(zip(got["order_id"], got["late"], strict=True)) == {
        "late": 1, "same_day": 0, "overdue": 1, "due_today": None, "canceled": None,
    }  # fmt: skip
    assert got.schema["late"] == pl.Int8


def test_late_delivered(lake: Settings) -> None:
    assert _labels(lake, date(2017, 1, 21))["o2"] == 1


def test_on_time_delivered(lake: Settings) -> None:
    # cut-off 01-23: o4's promised day (01-22) is over, so its on-time label is mature
    labels = _labels(lake, date(2017, 1, 23))
    assert labels["o1"] == 0 and labels["o4"] == 0


def test_overdue_undelivered_is_late(lake: Settings) -> None:
    assert _labels(lake, date(2017, 1, 15))["o2"] == 1


def test_not_yet_due_is_null_and_dropped(lake: Settings) -> None:
    labels = _labels(lake, date(2017, 1, 21))
    assert "o5" not in labels
    assert "o4" not in _labels(lake, date(2017, 1, 10))


def test_canceled_and_bad_orders_excluded(lake: Settings) -> None:
    labels = _labels(lake, date(2017, 1, 23))
    assert set(labels) == {"o1", "o2", "o4"}


def test_snapshot_has_features_and_meta(lake: Settings) -> None:
    info = build_training_set(lake, date(2017, 1, 23))
    df = pl.read_parquet(f"{info.path}/part-0.parquet")
    assert set(FEATURE_COLUMNS) | {"late"} <= set(df.columns)
    meta = json.loads(Path(info.path, "_meta.json").read_text())
    assert meta["cutoff"] == "2017-01-23" and meta["rows"] == 3
    assert meta["late_rate"] == pytest.approx(1 / 3)
    assert info.path.endswith(f"snapshot={info.content_hash[:12]}")


def test_snapshot_id_stable_across_rebuilds(lake: Settings) -> None:
    a = build_training_set(lake, date(2017, 1, 21))
    b = build_training_set(lake, date(2017, 1, 21))
    assert a == b


EXTRA_ORDERS = [
    # o7: handed to carrier BEFORE payment approval (real Olist pattern, ~1.4% of orders)
    "o7,c7,delivered,2017-01-06 09:00:00,2017-01-08 12:00:00,2017-01-07 10:00:00,"
    "2017-01-12 10:00:00,2017-01-20 00:00:00",
    # o8: status delivered but delivery date missing (8 real Olist orders)
    "o8,c8,delivered,2017-01-06 11:00:00,2017-01-06 12:00:00,2017-01-07 09:00:00,,"
    "2017-01-15 00:00:00",
]


def _append(path: Path, lines: list[str]) -> None:
    with path.open("a") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def anomaly_lake(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    from late_delivery.source.simulator import RAW_FILES

    tmp = tmp_path_factory.mktemp("anomalies")
    raw = write_mini_olist(tmp / "raw")
    _append(raw / RAW_FILES["orders"], EXTRA_ORDERS)
    _append(raw / RAW_FILES["order_items"],
            [f"{o},1,p1,s1,2017-01-09 10:00:00,50.0,10.0" for o in ("o7", "o8")])  # fmt: skip
    _append(raw / RAW_FILES["order_payments"],
            [f"{o},1,credit_card,1,60.0" for o in ("o7", "o8")])  # fmt: skip
    _append(raw / RAW_FILES["customers"],
            [f"{c},u{c},1310,sao paulo,SP" for c in ("c7", "c8")])  # fmt: skip
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LATE_DELIVERY_PII_SALT", "test-salt")
        s = make_settings(raw, tmp / "lake")
        ds = FIRST_DAY
        while ds <= LAST_DAY:
            ctx = new_context(s, ds)
            for fn in (simulate, ingest_bronze, validate_to_silver, merge_current):
                fn(ctx)
            ds += timedelta(days=1)
        yield s


def _snapshot(s: Settings, cutoff: date) -> pl.DataFrame:
    info = build_training_set(s, cutoff)
    return pl.read_parquet(f"{info.path}/part-0.parquet")


def test_training_features_as_of_handoff_not_final_state(anomaly_lake: Settings) -> None:
    o7 = _snapshot(anomaly_lake, date(2017, 1, 21)).filter(pl.col("order_id") == "o7")
    row = o7.row(0, named=True)
    assert row["approval_delay_h"] is None  # approval was in the future at handoff
    assert row["handling_time_h"] == pytest.approx(25.0)  # measured from purchase
    assert row["late"] == 0


def test_delivered_without_date_excluded_not_labeled_late(anomaly_lake: Settings) -> None:
    ids = _snapshot(anomaly_lake, date(2017, 1, 21))["order_id"].to_list()
    assert "o8" not in ids


def _meta(s: Settings, cutoff: date) -> dict:
    info = build_training_set(s, cutoff)
    return json.loads(Path(info.path, "_meta.json").read_text())


def test_meta_counts_overdue_undelivered_labels(lake: Settings) -> None:
    # On 01-15, o2 (due 01-12) is still in transit: labeled late without a delivery date.
    assert _meta(lake, date(2017, 1, 15))["late_without_delivery"] == 1
    # By 01-21 it has been delivered, so no label rests on non-delivery.
    assert _meta(lake, date(2017, 1, 21))["late_without_delivery"] == 0


def test_not_yet_due_order_excluded_even_if_delivered_on_time(lake: Settings) -> None:
    # Cut-off 01-10: o1 (due 01-15) already arrived on 01-08, but o2 (due 01-12) is still
    # travelling. Keeping o1 while o2 cannot be labeled would favour early arrivals.
    labels = _labels(lake, date(2017, 1, 10))
    assert "o1" not in labels
    assert labels == {}


def test_meta_counts_not_yet_due_exclusions(lake: Settings) -> None:
    meta = _meta(lake, date(2017, 1, 10))
    assert meta["excluded_not_yet_due"] == 4  # o1, o2, o4, o5 handed off, all due after 01-10
    assert meta["rows"] == 0


def test_late_without_delivery_column(lake: Settings) -> None:
    # On 01-15, o2 (due 01-12) is still in transit: late, with no delivery date yet.
    info = build_training_set(lake, date(2017, 1, 15))
    df = pl.read_parquet(f"{info.path}/part-0.parquet")
    flagged = set(df.filter(pl.col("late_without_delivery"))["order_id"])
    assert flagged == {"o2"}
    assert df.filter(pl.col("late_without_delivery"))["late"].to_list() == [1]
    meta = _meta(lake, date(2017, 1, 15))
    assert df["late_without_delivery"].sum() == meta["late_without_delivery"]
