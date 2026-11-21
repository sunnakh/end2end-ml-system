from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from late_delivery.config import TrainingConfig
from late_delivery.modeling.gate import (
    Champion,
    GateResult,
    read_champion,
    run_gate,
    write_champion,
)

CFG = TrainingConfig(bootstrap_resamples=200)


def _region(i: int) -> str:
    # per day of 100 orders: 55 Southeast, 20 South, 15 Center-West, 10 North
    k = i % 100
    return "Southeast" if k < 55 else "South" if k < 75 else "Center-West" if k < 90 else "North"


def _frame(catch: Callable[[int, str], bool], score: float = 0.1,
           days: int = 40) -> pl.DataFrame:  # fmt: skip
    """`days` x 100 orders. Late: every 10th order, except none in Center-West.
    `catch(j, region)` says whether the j-th late order is contacted."""
    rows, j = [], 0
    for i in range(days * 100):
        region = _region(i)
        late = i % 10 == 0 and region != "Center-West"
        contacted = False
        if late:
            contacted = catch(j, region)
            j += 1
        rows.append((f"o{i:05d}", date(2018, 4, 1) + timedelta(days=i // 100), int(late),
                     score, contacted, region, "50+"))  # fmt: skip
    cols = ["order_id", "handoff_date", "late", "score", "contacted", "region", "seller_size"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def _rate(pct: int) -> Callable[[int, str], bool]:
    return lambda j, region: j % 100 < pct


def _check(result: GateResult, name: str) -> bool | None:
    return next(c.passed for c in result.checks if c.name == name)


def test_clearly_better_challenger_passes() -> None:
    r = run_gate(_frame(_rate(40)), {"rule": _frame(_rate(30))}, cfg=CFG, train_late_rate=0.1)
    assert r.passed
    assert _check(r, "beats rule") is True
    assert [c.name for c in r.checks if c.name.startswith("beats")] == ["beats rule"]


def test_too_small_gain_fails_margin() -> None:
    r = run_gate(_frame(_rate(31)), {"rule": _frame(_rate(30))}, cfg=CFG, train_late_rate=0.1)
    assert not r.passed
    assert _check(r, "beats rule") is False


def test_champion_adds_a_check() -> None:
    refs = {"rule": _frame(_rate(10)), "champion": _frame(_rate(40))}
    r = run_gate(_frame(_rate(41)), refs, cfg=CFG, train_late_rate=0.1)
    assert _check(r, "beats rule") is True
    assert _check(r, "beats champion") is False
    assert not r.passed


def test_slice_clearly_worse_fails_and_names_it() -> None:
    no_south = lambda j, region: region != "South" and j % 100 < 45  # noqa: E731
    r = run_gate(_frame(no_south), {"rule": _frame(_rate(30))}, cfg=CFG, train_late_rate=0.1)
    assert _check(r, "slice region=South") is False
    assert _check(r, "slice region=Southeast") is True
    assert not r.passed


def test_slice_drop_within_noise_passes() -> None:
    # North has one late order a day (40 in total). The challenger misses 3 that the rule
    # catches: -0.075, beyond the 0.05 tolerance, but 3 orders are not clear evidence.
    missed: list[int] = []

    def three_fewer_in_north(j: int, region: str) -> bool:
        if region == "North" and j % 100 < 30 and len(missed) < 3:
            missed.append(j)
            return False
        return j % 100 < 30

    r = run_gate(
        _frame(three_fewer_in_north), {"rule": _frame(_rate(30))}, cfg=CFG, train_late_rate=0.1
    )
    north = next(c for c in r.checks if c.name == "slice region=North")
    assert "-0.075" in north.detail
    assert north.passed is True


def test_small_slice_ignored_and_no_late_slice_skipped() -> None:
    worse_north = lambda j, region: region != "North" and j % 100 < 45  # noqa: E731
    r = run_gate(
        _frame(worse_north, days=29), {"rule": _frame(_rate(30), days=29)},
        cfg=CFG, train_late_rate=0.1,
    )  # fmt: skip
    names = [c.name for c in r.checks]
    assert "slice region=North" not in names  # 10 x 29 = 290 rows, below the 300 minimum
    assert _check(r, "slice region=Center-West") is None  # no late orders: cannot judge


def test_brier_worse_than_constant_fails() -> None:
    r = run_gate(
        _frame(_rate(40), score=0.9), {"rule": _frame(_rate(30))}, cfg=CFG, train_late_rate=0.1
    )
    assert _check(r, "calibration") is False
    assert not r.passed


def test_champion_round_trip(tmp_path: Path) -> None:
    assert read_champion(tmp_path) is None
    c = Champion(
        run_id="r1", model_path="runs/r1/model.pkl", gate_model_path="runs/r1/gate_model.pkl",
        snapshot_hashes={"decision": "abc"}, valid_metrics={"recall_at_k": 0.25},
    )  # fmt: skip
    write_champion(tmp_path, c)
    assert read_champion(tmp_path) == c
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_champion_raises(tmp_path: Path) -> None:
    (tmp_path / "champion.json").write_text("{not json")
    with pytest.raises(ValueError, match="champion.json"):
        read_champion(tmp_path)
    (tmp_path / "champion.json").write_text('{"run_id": "r1"}')
    with pytest.raises(ValueError, match="champion.json"):
        read_champion(tmp_path)
