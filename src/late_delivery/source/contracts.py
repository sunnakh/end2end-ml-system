"""Data contracts for every Olist table: types, keys and row-level rules."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from late_delivery.platform.validation import Contract

TS = pl.Datetime("us")
STR = pl.String()
INT = pl.Int64()
FLOAT = pl.Float64()

BR_STATES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}  # fmt: skip
ORDER_STATUSES = {
    "created", "approved", "invoiced", "processing", "shipped", "delivered", "canceled",
    "unavailable",
}  # fmt: skip
CHANGE_TYPES = {"created", "approved", "shipped", "delivered", "closed"}
NO_TOLERANCE = timedelta(0)
PAYMENT_TYPES = {"credit_card", "boleto", "voucher", "debit_card", "not_defined"}


CONTRACTS: dict[str, Contract] = {
    "orders": Contract(
        table="orders",
        key=["order_id", "_change_type"],
        dtypes={
            "order_id": STR,
            "customer_id": STR,
            "order_status": STR,
            "order_purchase_timestamp": TS,
            "order_approved_at": TS,
            "order_delivered_carrier_date": TS,
            "order_delivered_customer_date": TS,
            "order_estimated_delivery_date": TS,
            "_change_ts": TS,
            "_change_type": STR,
            "_event_rank": INT,
        },
        required=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "_change_ts",
            "_change_type",
            "_event_rank",
        ],  # fmt: skip
        enums={"order_status": ORDER_STATUSES, "_change_type": CHANGE_TYPES},
        required_when=[("order_delivered_customer_date", "_change_type", "delivered")],
        ts_order=[
            ("order_purchase_timestamp", "order_approved_at", NO_TOLERANCE),
            # 160 of 166 real orders with handoff before purchase are off by <= 2 hours, falling
            # off smoothly: a clock/time-zone quirk between systems, not corruption. Only the
            # 6 larger gaps are quarantined.
            ("order_purchase_timestamp", "order_delivered_carrier_date", timedelta(hours=2)),
            ("order_delivered_carrier_date", "order_delivered_customer_date", NO_TOLERANCE),
            ("order_purchase_timestamp", "order_delivered_customer_date", NO_TOLERANCE),
        ],
    ),
    "order_items": Contract(
        table="order_items",
        key=["order_id", "order_item_id"],
        dtypes={
            "order_id": STR,
            "order_item_id": INT,
            "product_id": STR,
            "seller_id": STR,
            "shipping_limit_date": TS,
            "price": FLOAT,
            "freight_value": FLOAT,
        },
        required=["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
        non_negative=["price", "freight_value"],
        fk={"order_id": "orders", "product_id": "products", "seller_id": "sellers"},
    ),
    "order_payments": Contract(
        table="order_payments",
        key=["order_id", "payment_sequential"],
        dtypes={
            "order_id": STR,
            "payment_sequential": INT,
            "payment_type": STR,
            "payment_installments": INT,
            "payment_value": FLOAT,
        },
        required=["order_id", "payment_sequential", "payment_type", "payment_value"],
        enums={"payment_type": PAYMENT_TYPES},
        non_negative=["payment_value", "payment_installments"],
        fk={"order_id": "orders"},
    ),
    "customers": Contract(
        table="customers",
        key=["customer_id"],
        dtypes={
            "customer_id": STR,
            "customer_unique_id": STR,
            "customer_zip_code_prefix": STR,
            "customer_city": STR,
            "customer_state": STR,
        },
        required=[
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_state",
        ],
        enums={"customer_state": BR_STATES},
    ),
    "sellers": Contract(
        table="sellers",
        key=["seller_id"],
        dtypes={
            "seller_id": STR,
            "seller_zip_code_prefix": STR,
            "seller_city": STR,
            "seller_state": STR,
        },
        required=["seller_id", "seller_zip_code_prefix", "seller_state"],
        enums={"seller_state": BR_STATES},
    ),
    "products": Contract(
        table="products",
        key=["product_id"],
        dtypes={
            "product_id": STR,
            "product_category_name": STR,
            "product_name_lenght": INT,
            "product_description_lenght": INT,
            "product_photos_qty": INT,
            "product_weight_g": FLOAT,
            "product_length_cm": FLOAT,
            "product_height_cm": FLOAT,
            "product_width_cm": FLOAT,
        },
        required=["product_id"],
        non_negative=[
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],  # fmt: skip
    ),
    "geolocation": Contract(
        table="geolocation",
        key=[],
        dtypes={
            "geolocation_zip_code_prefix": STR,
            "geolocation_lat": FLOAT,
            "geolocation_lng": FLOAT,
            "geolocation_city": STR,
            "geolocation_state": STR,
        },
        required=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
        enums={"geolocation_state": BR_STATES},
        ranges={"geolocation_lat": (-34.0, 6.0), "geolocation_lng": (-74.0, -34.0)},
    ),
    "order_reviews": Contract(
        table="order_reviews",
        key=["review_id", "order_id"],
        dtypes={
            "review_id": STR,
            "order_id": STR,
            "review_score": INT,
            "review_comment_title": STR,
            "review_comment_message": STR,
            "review_creation_date": TS,
            "review_answer_timestamp": TS,
        },
        required=["review_id", "order_id", "review_score"],
        ranges={"review_score": (1.0, 5.0)},
        fk={"order_id": "orders"},
    ),
    "category_translation": Contract(
        table="category_translation",
        key=["product_category_name"],
        dtypes={"product_category_name": STR, "product_category_name_english": STR},
        required=["product_category_name", "product_category_name_english"],
    ),
}
