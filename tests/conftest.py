"""Synthetic Olist-shaped fixture: six orders covering the pipeline's edge cases."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from late_delivery.config import Settings
from late_delivery.source.simulator import RAW_FILES

FIRST_DAY = date(2017, 1, 2)
LAST_DAY = date(2017, 1, 21)

ORDERS = [
    # o1 on time; o2 late; o3 canceled after approval; o4 two sellers;
    # o5 created+approved same day, still shipped at dump time; o6 handoff before purchase (bad).
    ("o1", "c1", "delivered", "2017-01-02 10:00:00", "2017-01-03 09:00:00",
     "2017-01-04 15:00:00", "2017-01-08 12:00:00", "2017-01-15 00:00:00"),
    ("o2", "c2", "delivered", "2017-01-02 11:00:00", "2017-01-03 10:00:00",
     "2017-01-05 10:00:00", "2017-01-20 10:00:00", "2017-01-12 00:00:00"),
    ("o3", "c3", "canceled", "2017-01-03 08:00:00", "2017-01-04 09:00:00",
     "", "", "2017-01-25 00:00:00"),
    ("o4", "c4", "delivered", "2017-01-04 09:00:00", "2017-01-05 08:00:00",
     "2017-01-06 14:00:00", "2017-01-12 10:00:00", "2017-01-22 00:00:00"),
    ("o5", "c5", "shipped", "2017-01-05 09:00:00", "2017-01-05 09:30:00",
     "2017-01-07 10:00:00", "", "2017-01-30 00:00:00"),
    ("o6", "c6", "delivered", "2017-01-06 10:00:00", "2017-01-06 12:00:00",
     "2017-01-05 10:00:00", "2017-01-10 10:00:00", "2017-01-20 00:00:00"),
]  # fmt: skip
ORDER_COLS = [
    "order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]  # fmt: skip

ITEMS = [
    ("o1", "1", "p1", "s1", "2017-01-05 10:00:00", "50.0", "10.0"),
    ("o2", "1", "p2", "s2", "2017-01-05 11:00:00", "80.0", "20.0"),
    ("o3", "1", "p1", "s1", "2017-01-06 08:00:00", "50.0", "10.0"),
    ("o4", "1", "p1", "s1", "2017-01-07 09:00:00", "50.0", "10.0"),
    ("o4", "2", "p2", "s2", "2017-01-07 09:00:00", "80.0", "15.0"),
    ("o5", "1", "p2", "s2", "2017-01-08 09:00:00", "80.0", "20.0"),
    ("o6", "1", "p1", "s1", "2017-01-09 10:00:00", "50.0", "10.0"),
]
ITEM_COLS = [
    "order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price",
    "freight_value",
]  # fmt: skip

PAYMENTS = [(f"o{i}", "1", "credit_card", "2", "60.0") for i in range(1, 7)]
PAYMENT_COLS = [
    "order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value",
]  # fmt: skip

CUSTOMERS = [
    ("c1", "u1", "1310", "sao paulo", "SP"),
    ("c2", "u2", "22041", "rio de janeiro", "RJ"),
    ("c3", "u3", "1310", "sao paulo", "SP"),
    ("c4", "u4", "22041", "rio de janeiro", "RJ"),
    ("c5", "u5", "1310", "sao paulo", "SP"),
    ("c6", "u1", "1310", "sao paulo", "SP"),
]
CUSTOMER_COLS = [
    "customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city",
    "customer_state",
]  # fmt: skip

SELLERS = [("s1", "1037", "sao paulo", "SP"), ("s2", "20040", "rio de janeiro", "RJ")]
SELLER_COLS = ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]

PRODUCTS = [
    ("p1", "beleza_saude", "40", "300", "2", "500", "20", "10", "15"),
    ("p2", "", "", "", "", "1200", "30", "20", "25"),
]
PRODUCT_COLS = [
    "product_id", "product_category_name", "product_name_lenght", "product_description_lenght",
    "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm",
]  # fmt: skip

GEO = [
    ("1037", "-23.545", "-46.639", "sao paulo", "SP"),
    ("1037", "-23.546", "-46.640", "sao paulo", "SP"),
    ("1037", "-23.546", "-46.640", "sao paulo", "SP"),
    ("1310", "-23.561", "-46.656", "sao paulo", "SP"),
    ("20040", "-22.903", "-43.176", "rio de janeiro", "RJ"),
    ("22041", "-22.968", "-43.183", "rio de janeiro", "RJ"),
    ("22041", "42.000", "-8.700", "rio de janeiro", "RJ"),
]
GEO_COLS = [
    "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city",
    "geolocation_state",
]  # fmt: skip

REVIEWS = [
    ("r1", "o1", "5", "", "otimo produto", "2017-01-09 00:00:00", "2017-01-09 15:00:00"),
    ("r2", "o2", "1", "atraso", "chegou muito tarde", "2017-01-21 00:00:00",
     "2017-01-21 09:00:00"),
]  # fmt: skip
REVIEW_COLS = [
    "review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
    "review_creation_date", "review_answer_timestamp",
]  # fmt: skip

CATEGORIES = [("beleza_saude", "health_beauty")]
CATEGORY_COLS = ["product_category_name", "product_category_name_english"]

TABLES: dict[str, tuple[list[str], list[tuple[str, ...]]]] = {
    "orders": (ORDER_COLS, ORDERS),
    "order_items": (ITEM_COLS, ITEMS),
    "order_payments": (PAYMENT_COLS, PAYMENTS),
    "customers": (CUSTOMER_COLS, CUSTOMERS),
    "sellers": (SELLER_COLS, SELLERS),
    "products": (PRODUCT_COLS, PRODUCTS),
    "geolocation": (GEO_COLS, GEO),
    "order_reviews": (REVIEW_COLS, REVIEWS),
    "category_translation": (CATEGORY_COLS, CATEGORIES),
}


def write_mini_olist(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for table, (cols, rows) in TABLES.items():
        df = pl.DataFrame(rows, schema={c: pl.String for c in cols}, orient="row")
        df.write_csv(raw_dir / RAW_FILES[table])
    return raw_dir


@pytest.fixture
def mini_olist(tmp_path: Path) -> Path:
    return write_mini_olist(tmp_path / "raw")


def make_settings(raw_dir: Path, data_root: Path, **overrides: object) -> Settings:

    base: dict[str, object] = {
        "paths": {"raw_dir": raw_dir, "data_root": str(data_root)},
        "replay": {"start": FIRST_DAY, "end": LAST_DAY},
        "drift": {"reference_start": FIRST_DAY, "reference_end": date(2017, 1, 8)},
    }
    for key, value in overrides.items():
        base[key] = value
    return Settings.model_validate(base)


@pytest.fixture(autouse=True)
def pii_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATE_DELIVERY_PII_SALT", "test-salt")
