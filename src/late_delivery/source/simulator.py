"""Source simulator: replays the Olist dump one day at a time."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import fsspec
import polars as pl

from late_delivery.source.contracts import CONTRACTS

RAW_FILES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

PURCHASE = "order_purchase_timestamp"
APPROVED = "order_approved_at"
CARRIER = "order_delivered_carrier_date"
DELIVERED = "order_delivered_customer_date"
LIFECYCLE = [APPROVED, CARRIER, DELIVERED]
CLOSED_STATUSES = ["canceled", "unavailable"]
REFERENCE_TABLES = ("sellers", "products", "geolocation", "category_translation")


def load_raw(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Read every Olist CSV with all columns as strings; blanks become null."""
    out: dict[str, pl.DataFrame] = {}
    for table, filename in RAW_FILES.items():
        df = pl.read_csv(raw_dir / filename, infer_schema=False)
        out[table] = df.with_columns(
            pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in df.columns
        )
    return out


def _event(
    orders: pl.DataFrame, change_type: str, rank: int, ts: pl.Expr, where: pl.Expr, status: str
) -> pl.DataFrame:
    """One copy of each matching order, showing only dates at or before this event's time.

    Hiding by time (not by lifecycle stage) matters: ~1.4% of Olist orders are handed to the
    carrier before payment approval, and the handoff row must not reveal the later approval.
    """
    return (
        orders.filter(where)
        .with_columns(ts.alias("_change_ts"))
        .with_columns(
            *[
                pl.when(pl.col(c) <= pl.col("_change_ts")).then(pl.col(c)).alias(c)
                for c in LIFECYCLE
            ],
            pl.lit(status).alias("order_status"),
            pl.lit(change_type).alias("_change_type"),
            pl.lit(rank, dtype=pl.Int64).alias("_event_rank"),
        )
    )


def order_cdc_events(orders: pl.DataFrame) -> pl.DataFrame:
    """One row per lifecycle event, on the day it happened; later dates are null on each row."""
    latest_known = pl.max_horizontal(PURCHASE, *LIFECYCLE)
    parts = [
        _event(orders, "created", 0, pl.col(PURCHASE), pl.col(PURCHASE).is_not_null(), "created"),
        _event(
            orders, "approved", 1, pl.col(APPROVED), pl.col(APPROVED).is_not_null(), "approved"
        ),  # fmt: skip
        _event(orders, "shipped", 2, pl.col(CARRIER), pl.col(CARRIER).is_not_null(), "shipped"),
        # A delivered order with no delivery date still emits its event (at the latest known
        # time) so validation quarantines it instead of the order looking forever in transit.
        _event(
            orders,
            "delivered",
            3,
            pl.coalesce(DELIVERED, latest_known),
            pl.col(DELIVERED).is_not_null() | (pl.col("order_status") == "delivered"),
            "delivered",
        ),  # fmt: skip
    ]
    closed = orders.filter(pl.col("order_status").is_in(CLOSED_STATUSES)).with_columns(
        latest_known.alias("_change_ts"),
        pl.lit("closed").alias("_change_type"),
        pl.lit(4, dtype=pl.Int64).alias("_event_rank"),
    )
    cols = orders.columns + ["_change_ts", "_change_type", "_event_rank"]
    return pl.concat([p.select(cols) for p in [*parts, closed]]).sort("order_id", "_event_rank")


def _on(col: str, ds: date) -> pl.Expr:
    return pl.col(col).str.slice(0, 10) == ds.isoformat()


def simulate_day(raw: dict[str, pl.DataFrame], ds: date, day0: date) -> dict[str, pl.DataFrame]:
    """Everything the business learned on `ds`; every table present, possibly empty."""
    orders = raw["orders"]
    events = order_cdc_events(orders)
    purchased = orders.filter(_on(PURCHASE, ds))
    out: dict[str, pl.DataFrame] = {
        "orders": events.filter(_on("_change_ts", ds)),
        "order_items": raw["order_items"].join(purchased.select("order_id"), on="order_id"),
        "order_payments": raw["order_payments"].join(purchased.select("order_id"), on="order_id"),
        "customers": raw["customers"].join(
            purchased.select("customer_id").unique(), on="customer_id"
        ),
        "order_reviews": raw["order_reviews"].filter(_on("review_answer_timestamp", ds)),
    }
    for table in REFERENCE_TABLES:
        out[table] = raw[table] if ds == day0 else raw[table].clear()
    return out


def landing_dir(root: str, ds: date) -> str:
    return f"{root}/landing/ds={ds:%Y-%m-%d}"


def write_landing(
    tables: dict[str, pl.DataFrame],
    root: str,
    ds: date,
    injected: list[InjectedFault] | None = None,
) -> None:
    fs, base = fsspec.core.url_to_fs(landing_dir(root, ds))
    fs.makedirs(base, exist_ok=True)
    for table, df in tables.items():
        with fs.open(f"{base}/{table}.csv", "wb") as f:
            df.write_csv(f)
    marker = f"{base}/_injected.json"
    if injected is not None:
        with fs.open(marker, "w") as f:
            json.dump([asdict(i) for i in injected], f, indent=2, sort_keys=True)
    elif fs.exists(marker):
        fs.rm(marker)  # a clean rerun must not keep a previous run's fault list


# --- fault injection -------------------------------------------------------------------------

ROW_FAULTS = (
    "duplicate", "null_required", "negative_amount", "ts_order", "orphan_fk", "invalid_enum",
)  # fmt: skip
EXPECTED_CODE: dict[str, str | None] = {
    "duplicate": None,
    "null_required": "NULL_REQUIRED",
    "negative_amount": "NEGATIVE_AMOUNT",
    "ts_order": "TS_ORDER_VIOLATION",
    "orphan_fk": "ORPHAN_FK",
    "invalid_enum": "INVALID_ENUM",
    "schema_drift": "FILE_LEVEL",
}
# table -> fault -> (column, replacement); None replacement means null.
_ROW_MUTATIONS: dict[str, dict[str, tuple[str, str | None]]] = {
    "orders": {
        "null_required": ("customer_id", None),
        "invalid_enum": ("order_status", "lost"),
        "ts_order": ("order_approved_at", "<day before purchase>"),
    },
    "order_items": {
        "null_required": ("price", None),
        "negative_amount": ("price", "<negative>"),
        "orphan_fk": ("order_id", "<ghost>"),
    },
    "order_payments": {
        "null_required": ("payment_value", None),
        "negative_amount": ("payment_value", "<negative>"),
        "invalid_enum": ("payment_type", "bitcoin"),
        "orphan_fk": ("order_id", "<ghost>"),
    },
    "customers": {"invalid_enum": ("customer_state", "XX")},
    "order_reviews": {"null_required": ("review_score", None)},
}
_DRIFT_RENAME = ("orders", "order_status", "status")


@dataclass(frozen=True)
class InjectedFault:
    table: str
    key: dict[str, str]
    fault: str
    expected_code: str | None


def _applicable(table: str, row: dict[str, Any], kinds: tuple[str, ...]) -> list[str]:
    out = ["duplicate"] if "duplicate" in kinds else []
    if table == "orders" and row.get("_change_type") == "created":
        # Faulting a created row would orphan that day's items: keep injected sets exact.
        return out
    for fault in _ROW_MUTATIONS.get(table, {}):
        if fault not in kinds:
            continue
        if fault == "ts_order" and not row.get("order_approved_at"):
            continue
        out.append(fault)
    return out


def _mutated_value(placeholder: str | None, row: dict[str, Any], col: str, n: int) -> str | None:
    if placeholder == "<negative>":
        return f"-{row[col]}"
    if placeholder == "<ghost>":
        return f"ghost-{n}"
    if placeholder == "<day before purchase>":
        purchase = datetime.strptime(row[PURCHASE], "%Y-%m-%d %H:%M:%S")
        return (purchase - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    return placeholder


def inject_faults(
    tables: dict[str, pl.DataFrame],
    ds: date,
    seed: int,
    rate: float,
    kinds: tuple[str, ...] = ROW_FAULTS,
) -> tuple[dict[str, pl.DataFrame], list[InjectedFault]]:
    """Corrupt a reproducible random subset of rows; return the tables and what was injected."""
    rng = random.Random(seed * 100_000 + ds.toordinal())
    out = dict(tables)
    injected: list[InjectedFault] = []
    n = 0
    for table in sorted(_ROW_MUTATIONS):
        df = tables[table]
        key_cols = CONTRACTS[table].key
        rows = df.to_dicts()
        new_rows: list[dict[str, Any]] = []
        extra: list[dict[str, Any]] = []
        for row in rows:
            choices = _applicable(table, row, kinds)
            if not choices or rng.random() >= rate:
                new_rows.append(row)
                continue
            fault = rng.choice(choices)
            n += 1
            if fault == "duplicate":
                new_rows.append(row)
                extra.append(dict(row))
                mutated = row
            else:
                col, placeholder = _ROW_MUTATIONS[table][fault]
                mutated = dict(row)
                mutated[col] = _mutated_value(placeholder, row, col, n)
                new_rows.append(mutated)
            key = {k: str(mutated[k]) for k in key_cols}
            injected.append(InjectedFault(table, key, fault, EXPECTED_CODE[fault]))
        if injected and any(f.table == table for f in injected):
            out[table] = pl.DataFrame(new_rows + extra, schema=df.schema, orient="row")
    if "schema_drift" in kinds:
        table, old, new = _DRIFT_RENAME
        out[table] = out[table].rename({old: new})
        injected.append(InjectedFault(table, {}, "schema_drift", EXPECTED_CODE["schema_drift"]))
    return out, injected
