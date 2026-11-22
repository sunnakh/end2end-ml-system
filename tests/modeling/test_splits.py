from datetime import date, datetime

import polars as pl
import pytest

from late_delivery.modeling.splits import REGION_OF_STATE, between, months_back, prepare

UFS = ("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO").split()


def _snap(states: list[str | None], prior: list[int]) -> pl.DataFrame:
    n = len(states)
    return pl.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "order_delivered_carrier_date": [datetime(2018, 4, 30, 23, 59)] * n,
            "customer_state": states,
            "seller_prior_orders": prior,
        }
    )


def test_all_27_states_have_a_region() -> None:
    assert set(REGION_OF_STATE) == set(UFS)
    assert set(REGION_OF_STATE.values()) == {
        "North", "Northeast", "Center-West", "Southeast", "South"
    }  # fmt: skip
    assert REGION_OF_STATE["SP"] == "Southeast" and REGION_OF_STATE["DF"] == "Center-West"


def test_unknown_or_missing_state_is_unknown() -> None:
    out = prepare(_snap(["XX", None], [0, 0]))
    assert out["region"].to_list() == ["Unknown", "Unknown"]


@pytest.mark.parametrize(
    ("prior", "bucket"),
    [(0, "0"), (1, "1-9"), (9, "1-9"), (10, "10-49"), (49, "10-49"), (50, "50+"), (900, "50+")],
)
def test_seller_size_buckets(prior: int, bucket: str) -> None:
    assert prepare(_snap(["SP"], [prior]))["seller_size"].item() == bucket


def test_prepare_adds_date_and_month() -> None:
    out = prepare(_snap(["SP"], [3]))
    assert out["handoff_date"].item() == date(2018, 4, 30)
    assert out["month"].item() == "2018-04"


def test_months_back_gives_window_starts() -> None:
    assert months_back(date(2018, 4, 1), 15) == date(2017, 1, 1)
    assert months_back(date(2018, 4, 1), 9) == date(2017, 7, 1)
    assert months_back(date(2018, 4, 1), 6) == date(2017, 10, 1)
    assert months_back(date(2018, 6, 1), 15) == date(2017, 3, 1)


def test_between_is_start_inclusive_end_exclusive() -> None:
    df = pl.DataFrame({"handoff_date": [date(2018, 3, 31), date(2018, 4, 1), date(2018, 6, 1)]})
    out = between(df, date(2018, 4, 1), date(2018, 6, 1))
    assert out["handoff_date"].to_list() == [date(2018, 4, 1)]
