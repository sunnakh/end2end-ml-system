"""Point-in-time order features at carrier handoff. Shared by training, batch and online scoring."""

from __future__ import annotations

from datetime import date

import polars as pl

PURCHASE = "order_purchase_timestamp"
APPROVED = "order_approved_at"
HANDOFF = "order_delivered_carrier_date"
ESTIMATED = "order_estimated_delivery_date"
DELIVERED = "order_delivered_customer_date"

FEATURE_COLUMNS: list[str] = [
    "promised_days",
    "remaining_days",
    "approval_delay_h",
    "handling_time_h",
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "total_weight_g",
    "max_volume_cm3",
    "customer_state",
    "same_state_all",
    "max_distance_km",
    "main_payment_type",
    "max_installments",
    "handoff_weekday",
    "handoff_month",
    "seller_prior_orders",
    "seller_prior_late_rate",
]

_EARTH_RADIUS_KM = 6371.0


def _hours(later: str, earlier: pl.Expr) -> pl.Expr:
    return (pl.col(later) - earlier).dt.total_seconds() / 3600.0


def haversine_km(lat1: pl.Expr, lng1: pl.Expr, lat2: pl.Expr, lng2: pl.Expr) -> pl.Expr:
    p1, p2 = lat1.radians(), lat2.radians()
    dlat = p2 - p1
    dlng = (lng2 - lng1).radians()
    a = (dlat / 2).sin() ** 2 + p1.cos() * p2.cos() * (dlng / 2).sin() ** 2
    return 2 * _EARTH_RADIUS_KM * a.sqrt().arcsin()


def is_late(delivered: pl.Expr, estimated: pl.Expr) -> pl.Expr:
    """Label rule: delivered on a later calendar day than the estimate."""
    return delivered.dt.date() > estimated.dt.date()


def _known_outcomes(changelog: pl.DataFrame) -> pl.DataFrame:
    """When each order's late/on-time outcome first became known, using the label's own rule.

    - Delivered on or before the promised day: known on time at the delivery moment.
    - Promised day over without a delivery: known late from the start of the next day,
      whether or not it arrives later.
    Counting only delivered orders would under-count late ones (they arrive last).
    An order canceled before its outcome was known never gets one.
    """
    per_order = changelog.group_by("order_id").agg(
        pl.col(ESTIMATED).drop_nulls().first().alias("_estimated"),
        pl.col(DELIVERED).filter(pl.col("_change_type") == "delivered").min().alias("_delivered"),
        pl.col("_change_ts").filter(pl.col("_change_type") == "closed").min().alias("_closed"),
    )
    overdue_from = (pl.col("_estimated").dt.date() + pl.duration(days=1)).cast(pl.Datetime("us"))
    return (
        per_order.filter(pl.col("_estimated").is_not_null())
        .with_columns(overdue_from.alias("_overdue_from"))
        .with_columns(
            pl.min_horizontal("_delivered", "_overdue_from").alias("_known_ts"),
            (pl.col("_delivered").is_null() | (pl.col("_delivered") >= pl.col("_overdue_from")))
            .cast(pl.Int64)
            .alias("_late"),
        )
        .filter(pl.col("_closed").is_null() | (pl.col("_closed") > pl.col("_known_ts")))
        .select("order_id", "_known_ts", "_late")
    )


def _seller_history(
    order_sellers: pl.DataFrame, all_items: pl.DataFrame, changelog: pl.DataFrame
) -> pl.DataFrame:
    """Per (order, seller): the seller's prior orders whose outcome was known strictly before
    this order's handoff, and the late share among them."""
    hist = (
        _known_outcomes(changelog)
        .join(all_items.select("order_id", "seller_id").unique(), on="order_id")
        .sort("_known_ts")
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("seller_id").alias("_n"),
            pl.col("_late").cum_sum().over("seller_id").alias("_n_late"),
        )
        .select("seller_id", "_known_ts", "_n", "_n_late")
    )
    current = order_sellers.select("order_id", "seller_id", HANDOFF).unique().sort(HANDOFF)
    joined = current.join_asof(
        hist,
        left_on=HANDOFF,
        right_on="_known_ts",
        by="seller_id",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,  # both sides sorted globally by their time key above
    )
    return joined.select(
        "order_id",
        "seller_id",
        pl.col("_n").fill_null(0).alias("prior_orders"),
        (pl.col("_n_late") / pl.col("_n")).alias("prior_late_rate"),
    )


def build_features(
    handoffs: pl.DataFrame,
    deliveries: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
    customers: pl.DataFrame,
    sellers: pl.DataFrame,
    products: pl.DataFrame,
    geo_prefix: pl.DataFrame,
) -> pl.DataFrame:
    """One row per handed-off order: order_id, handoff ts and FEATURE_COLUMNS.

    `deliveries` is the orders changelog known at scoring time; only its 'delivered' events
    strictly before each order's handoff feed seller history.
    """
    base = handoffs.select("order_id", "customer_id", PURCHASE, APPROVED, HANDOFF, ESTIMATED)

    geo = geo_prefix.select("zip_code_prefix", "lat", "lng")
    order_sellers = (
        items.join(base.select("order_id", HANDOFF), on="order_id")
        .join(
            sellers.select("seller_id", "seller_zip_code_prefix", "seller_state"),
            on="seller_id",
            how="left",
        )  # fmt: skip
        .join(
            products.select(
                "product_id",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ),
            on="product_id",
            how="left",
        )  # fmt: skip
    )

    cust = customers.select("customer_id", "customer_zip_code_prefix", "customer_state").join(
        geo.rename({"lat": "c_lat", "lng": "c_lng"}),
        left_on="customer_zip_code_prefix",
        right_on="zip_code_prefix",
        how="left",
    )
    with_geo = (
        order_sellers.join(base.select("order_id", "customer_id"), on="order_id")
        .join(cust, on="customer_id", how="left")
        .join(
            geo.rename({"lat": "s_lat", "lng": "s_lng"}),
            left_on="seller_zip_code_prefix",
            right_on="zip_code_prefix",
            how="left",
        )  # fmt: skip
        .with_columns(
            haversine_km(pl.col("s_lat"), pl.col("s_lng"), pl.col("c_lat"), pl.col("c_lng")).alias(
                "_km"
            ),
            (pl.col("seller_state") == pl.col("customer_state")).alias("_same_state"),
            (
                pl.col("product_length_cm")
                * pl.col("product_height_cm")
                * pl.col("product_width_cm")
            ).alias("_volume"),  # fmt: skip
        )
    )
    item_agg = with_geo.group_by("order_id").agg(
        pl.len().cast(pl.Int64).alias("n_items"),
        pl.col("seller_id").n_unique().cast(pl.Int64).alias("n_sellers"),
        pl.col("price").sum().alias("total_price"),
        pl.col("freight_value").sum().alias("total_freight"),
        pl.col("product_weight_g").sum().alias("total_weight_g"),
        pl.col("_volume").max().alias("max_volume_cm3"),
        pl.col("_same_state").all().alias("same_state_all"),
        pl.col("_km").max().alias("max_distance_km"),
    )

    pay_agg = (
        payments.sort(
            ["order_id", "payment_value", "payment_sequential"], descending=[False, True, False]
        )  # fmt: skip
        .group_by("order_id", maintain_order=True)
        .agg(
            pl.col("payment_type").first().alias("main_payment_type"),
            pl.col("payment_installments").max().alias("max_installments"),
        )
    )

    history = (
        _seller_history(order_sellers, items, deliveries)
        .group_by("order_id")
        .agg(
            pl.col("prior_orders").max().alias("seller_prior_orders"),
            pl.col("prior_late_rate").max().alias("seller_prior_late_rate"),
        )
    )

    out = (
        base.join(item_agg, on="order_id", how="left")
        .join(pay_agg, on="order_id", how="left")
        .join(customers.select("customer_id", "customer_state"), on="customer_id", how="left")
        .join(history, on="order_id", how="left")
        .with_columns(
            ((pl.col(ESTIMATED) - pl.col(PURCHASE)).dt.total_seconds() / 86400.0).alias(
                "promised_days"
            ),
            ((pl.col(ESTIMATED) - pl.col(HANDOFF)).dt.total_seconds() / 86400.0).alias(
                "remaining_days"
            ),
            _hours(APPROVED, pl.col(PURCHASE)).alias("approval_delay_h"),
            _hours(HANDOFF, pl.coalesce(APPROVED, PURCHASE)).alias("handling_time_h"),
            pl.col(HANDOFF).dt.weekday().cast(pl.Int64).alias("handoff_weekday"),
            pl.col(HANDOFF).dt.month().cast(pl.Int64).alias("handoff_month"),
            pl.col("seller_prior_orders").fill_null(0),
        )
    )
    return out.select("order_id", HANDOFF, *FEATURE_COLUMNS)


CLOSED_STATUSES = ("canceled", "unavailable")


def label_at_cutoff(orders: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Label known using data before `cutoff` (exclusive): order_id, late (Int8, null = unknown).

    Delivered -> compare dates. Not delivered and the estimated day already passed -> late.
    Not delivered and not yet due, or canceled/unavailable -> null.
    """
    delivered = pl.col(DELIVERED)
    estimated = pl.col(ESTIMATED)
    late = (
        pl.when(pl.col("order_status").is_in(CLOSED_STATUSES))
        .then(None)
        .when(delivered.is_not_null())
        .then(is_late(delivered, estimated).cast(pl.Int8))
        .when(estimated.dt.date() < cutoff)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(None)
    )
    return orders.select("order_id", late.cast(pl.Int8).alias("late"))
