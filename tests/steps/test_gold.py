from collections.abc import Iterator
from datetime import date, timedelta

import pytest

from late_delivery.config import Settings
from late_delivery.features import FEATURE_COLUMNS
from late_delivery.platform.pipeline.runner import new_context
from late_delivery.platform.storage import read_partition
from late_delivery.steps import (
    build_gold_scoring,
    ingest_bronze,
    merge_current,
    simulate,
    validate_to_silver,
)
from tests.conftest import FIRST_DAY, LAST_DAY, make_settings, write_mini_olist


@pytest.fixture(scope="module")
def lake(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    tmp = tmp_path_factory.mktemp("gold")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LATE_DELIVERY_PII_SALT", "test-salt")
        s = make_settings(write_mini_olist(tmp / "raw"), tmp / "lake")
        ds = FIRST_DAY
        while ds <= LAST_DAY:
            ctx = new_context(s, ds)
            for fn in (simulate, ingest_bronze, validate_to_silver, merge_current,
                       build_gold_scoring):  # fmt: skip
                fn(ctx)
            ds += timedelta(days=1)
        yield s


def test_scoring_only_orders_shipped_on_ds(lake: Settings) -> None:
    g = read_partition(lake.paths.data_root, "gold", "scoring", date(2017, 1, 4))
    assert g["order_id"].to_list() == ["o1"]


def test_bad_handoff_not_scored(lake: Settings) -> None:
    g = read_partition(lake.paths.data_root, "gold", "scoring", date(2017, 1, 5))
    assert g["order_id"].to_list() == ["o2"]


def test_scoring_features_present(lake: Settings) -> None:
    g = read_partition(lake.paths.data_root, "gold", "scoring", date(2017, 1, 6))
    assert g["order_id"].to_list() == ["o4"]
    assert set(FEATURE_COLUMNS) <= set(g.columns)
    assert g["n_sellers"].to_list() == [2]
    assert g["customer_state"].to_list() == ["RJ"]


def test_empty_scoring_day(lake: Settings) -> None:
    g = read_partition(lake.paths.data_root, "gold", "scoring", date(2017, 1, 14))
    assert g.height == 0 and set(FEATURE_COLUMNS) <= set(g.columns)
