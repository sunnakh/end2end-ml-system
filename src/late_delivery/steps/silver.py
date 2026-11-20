"""Silver: validate, type and redact; quarantine bad rows; current order state."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from late_delivery.platform.pii import ensure_lake_salt, get_salt, hash_ids
from late_delivery.platform.pipeline.runner import Context
from late_delivery.platform.storage import (
    delete_partition,
    list_partitions,
    read_partition,
    read_table,
    scan_table,
    write_partition,
)
from late_delivery.platform.validation import FileLevelError, validate
from late_delivery.source.contracts import CONTRACTS
from late_delivery.source.simulator import (
    REFERENCE_TABLES,
)
from late_delivery.steps._common import (
    CUSTOMER_ID_COLS,
    REVIEW_TEXT_COLS,
    TABLE_ORDER,
    _write,
)


def _known_orders(ctx: Context, today: pl.DataFrame) -> pl.Series:
    lf = scan_table(ctx.root, "silver", "orders_changelog", ctx.ds - timedelta(days=1))
    earlier = (
        pl.Series("order_id", [], dtype=pl.String)
        if lf is None
        else lf.select("order_id").unique().collect()["order_id"]
    )
    return pl.concat([earlier, today["order_id"]]).unique()


def _reference_keys(ctx: Context, table: str, col: str, today: pl.DataFrame | None) -> pl.Series:
    if today is not None and today.height:
        return today[col]
    df = read_table(ctx.root, "silver", table, ctx.ds)
    return pl.Series(col, [], dtype=pl.String) if df is None else df[col]


def _redact(df: pl.DataFrame, salt: str) -> pl.DataFrame:
    df = df.drop([c for c in REVIEW_TEXT_COLS if c in df.columns])
    return df.with_columns(
        hash_ids(pl.col(c), salt).alias(c) for c in CUSTOMER_ID_COLS if c in df.columns
    )


def latest_review_per_order(reviews: pl.DataFrame) -> pl.DataFrame:
    """One review per order: the most recently answered (some Olist orders have several).

    Each day's partition is unique per order; readers spanning days apply this again.
    """
    return (
        reviews.select("order_id", "review_score", "review_answer_timestamp")
        .sort("order_id", "review_answer_timestamp", nulls_last=False)
        .group_by("order_id", maintain_order=True)
        .last()
    )


def _check_breaker(ctx: Context, table: str, valid: int, bad: int) -> None:
    cfg = ctx.settings.validation
    total = valid + bad
    if total >= cfg.min_rows_for_breaker and total and bad / total > cfg.max_quarantine_rate:
        raise RuntimeError(
            f"{table} {ctx.ds}: quarantine rate {bad / total:.1%} exceeds "
            f"{cfg.max_quarantine_rate:.1%}"
        )


def _products_silver(products: pl.DataFrame, translation: pl.DataFrame) -> pl.DataFrame:
    return products.join(translation, on="product_category_name", how="left").with_columns(
        pl.col("product_category_name_english").fill_null("unknown")
    )


def _geo_prefix(geo: pl.DataFrame) -> pl.DataFrame:
    return (
        geo.group_by("geolocation_zip_code_prefix")
        .agg(
            pl.col("geolocation_lat").median().alias("lat"),
            pl.col("geolocation_lng").median().alias("lng"),
        )
        .rename({"geolocation_zip_code_prefix": "zip_code_prefix"})
    )


def validate_to_silver(ctx: Context) -> None:
    salt = get_salt(ctx.settings.pii.salt_env)
    ensure_lake_salt(ctx.root, salt)
    bronze = {t: read_partition(ctx.root, "bronze", t, ctx.ds) for t in TABLE_ORDER}
    for table, df in bronze.items():  # fail the whole day before writing anything
        missing = [c for c in CONTRACTS[table].dtypes if c not in df.columns]
        if missing:
            raise FileLevelError(f"{table} {ctx.ds}: missing columns {missing}")

    valid: dict[str, pl.DataFrame] = {}
    quarantined: dict[str, pl.DataFrame] = {}
    for table in TABLE_ORDER:
        contract = CONTRACTS[table]
        known: dict[str, pl.Series] = {}
        if contract.fk:
            known = {
                "orders": _known_orders(ctx, valid["orders"]),
                "products": _reference_keys(ctx, "products", "product_id", valid.get("products")),
                "sellers": _reference_keys(ctx, "sellers", "seller_id", valid.get("sellers")),
            }
        result = validate(bronze[table], contract, known)
        valid[table] = result.valid
        q = result.quarantine
        counts: dict[str, int] = {}
        for codes in q["reason_codes"].to_list():
            for code in codes:
                counts[code] = counts.get(code, 0) + 1
        if result.duplicates_removed:
            counts["_duplicates_removed"] = result.duplicates_removed
        ctx.quarantine_counts[table] = counts
        ctx.row_counts[table] = {
            "bronze": bronze[table].height,
            "valid": result.valid.height,
            "quarantined": q.height,
            "duplicates_removed": result.duplicates_removed,
        }
        # Check every table before writing anything: a failed day (or failed rerun) must
        # leave the previous silver and quarantine untouched, never half replaced.
        _check_breaker(ctx, table, result.valid.height, q.height)
        quarantined[table] = q

    for table in TABLE_ORDER:
        _write(ctx, _redact(quarantined[table], salt), "quarantine", table, ["_row_number"])
        df = _redact(valid[table], salt)
        key = CONTRACTS[table].key
        if table == "orders":
            _write(ctx, df, "silver", "orders_changelog", ["order_id", "_event_rank"])
        elif table == "order_reviews":
            out = latest_review_per_order(df)
            _write(ctx, out, "silver", "reviews_outcome", ["order_id"])
        elif table in REFERENCE_TABLES:
            if df.height == 0:
                continue
            if table == "products":
                df = _products_silver(df, valid["category_translation"])
            if table == "geolocation":
                _write(ctx, _geo_prefix(df), "silver", "geo_prefix", ["zip_code_prefix"])
                continue
            if table == "category_translation":
                continue
            _write(ctx, df, "silver", table, key)
        else:
            _write(ctx, df, "silver", table, key)


# --- current state ----------------------------------------------------------------------------


def latest_state(changelog: pl.LazyFrame) -> pl.LazyFrame:
    """Furthest lifecycle stage per order; earlier non-null fields carry forward.

    Ordered by stage, not clock: an approval recorded after the handoff must not move a shipped
    order back to 'approved'.
    """
    cols = changelog.collect_schema().names()
    return (
        changelog.sort("order_id", "_event_rank", "_change_ts")
        .with_columns(pl.all().exclude("order_id").forward_fill().over("order_id"))
        .group_by("order_id", maintain_order=True)
        .last()
        .select(cols)
    )


def merge_current(ctx: Context) -> None:
    """Rebuild latest state per order from the full changelog; keep only the newest partition."""
    lf = scan_table(ctx.root, "silver", "orders_changelog")
    if lf is None:
        raise FileNotFoundError("silver/orders_changelog is empty; run validate_to_silver first")
    last_day = list_partitions(ctx.root, "silver", "orders_changelog")[-1]
    current = latest_state(lf).collect()
    info = write_partition(current, ctx.root, "silver", "orders_current", last_day, ["order_id"])
    ctx.outputs.append(info)
    for d in list_partitions(ctx.root, "silver", "orders_current"):
        if d != last_day:
            delete_partition(ctx.root, "silver", "orders_current", d)


def read_current(root: str) -> pl.DataFrame:
    days = list_partitions(root, "silver", "orders_current")
    if not days:
        raise FileNotFoundError("silver/orders_current has no partitions")
    return read_partition(root, "silver", "orders_current", days[-1])
