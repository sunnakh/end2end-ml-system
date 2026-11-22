import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from late_delivery.platform.drift import psi
from late_delivery.platform.pipeline.runner import new_context
from late_delivery.steps import (
    build_gold_scoring,
    drift_report,
    ingest_bronze,
    merge_current,
    simulate,
    validate_to_silver,
)
from tests.conftest import FIRST_DAY, make_settings


def test_psi_identical_is_zero() -> None:
    s = pl.Series(range(1000), dtype=pl.Float64)
    assert psi(s, s) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_above_0_2() -> None:
    ref = pl.Series([float(i % 100) for i in range(1000)])
    cur = ref + 60
    value = psi(ref, cur)
    assert value is not None and value > 0.2


def test_psi_categorical_unseen_category() -> None:
    ref = pl.Series(["SP"] * 50 + ["RJ"] * 50)
    cur = pl.Series(["SP"] * 50 + ["XX"] * 50)
    value = psi(ref, cur)
    assert value is not None and value > 0.2


def test_empty_current_gives_null() -> None:
    assert psi(pl.Series([1.0, 2.0]), pl.Series([], dtype=pl.Float64)) is None


def test_all_null_current_gives_null() -> None:
    assert psi(pl.Series([1.0, 2.0]), pl.Series([None, None], dtype=pl.Float64)) is None


def _run_days(s: object, last: date) -> None:
    ds = FIRST_DAY
    while ds <= last:
        ctx = new_context(s, ds)
        for fn in (simulate, ingest_bronze, validate_to_silver, merge_current,
                   build_gold_scoring, drift_report):  # fmt: skip
            fn(ctx)
        ds += timedelta(days=1)


def _report(tmp_path: Path, ds: str) -> dict:
    return json.loads(
        (tmp_path / "lake" / "reports" / "drift" / f"ds={ds}" / "report.json").read_text()
    )


def test_drift_report_written(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(
        mini_olist, tmp_path / "lake",
        drift={"reference_start": FIRST_DAY, "reference_end": date(2017, 1, 4), "min_rows": 0},
    )  # fmt: skip
    _run_days(s, date(2017, 1, 7))
    path = tmp_path / "lake" / "reports" / "drift" / "ds=2017-01-05" / "report.json"
    report = json.loads(path.read_text())
    assert report["ds"] == "2017-01-05"
    assert report["scoring_rows"] == 1
    assert report["status"] == "ok"
    # trailing 7-day window 2016-12-30..2017-01-05: handoffs on 01-04 (o1) and 01-05 (o2)
    assert report["window"] == {"start": "2016-12-30", "end": "2017-01-05", "rows": 2}
    assert "remaining_days" in report["psi"]
    assert "handoff_weekday" not in report["psi"] and "handoff_month" not in report["psi"]
    assert report["quarantine"]["orders"] == {"rows": 1, "bronze_rows": 5, "rate": 0.2}


def test_no_reference_until_reference_period_is_over(mini_olist: Path, tmp_path: Path) -> None:
    # The reference partitions 01-02..01-07 exist on disk when 01-05 runs, but on 01-05 the
    # reference period (to 01-08) is not over: using it would look into the future, and the
    # report would depend on whether later days had already been processed.
    s = make_settings(
        mini_olist, tmp_path / "lake",
        drift={"reference_start": FIRST_DAY, "reference_end": date(2017, 1, 8), "min_rows": 0},
    )  # fmt: skip
    _run_days(s, date(2017, 1, 7))
    ctx = new_context(s, date(2017, 1, 5))
    drift_report(ctx)  # rerun after later days exist
    report = _report(tmp_path, "2017-01-05")
    assert report["status"] == "no_reference"
    assert report["reference"] is None


def _scoring(n: int, shift: float = 0.0) -> pl.DataFrame:
    from late_delivery.features import FEATURE_COLUMNS

    data: dict[str, list[object]] = {}
    for c in FEATURE_COLUMNS:
        if c in ("customer_state", "main_payment_type"):
            data[c] = ["SP"] * n
        elif c == "same_state_all":
            data[c] = [True] * n
        else:
            data[c] = [float(i % 50) + shift for i in range(n)]
    return pl.DataFrame(data)


def test_drift_scores_skip_calendar_features() -> None:
    from late_delivery.steps.monitoring import drift_scores

    scores, status = drift_scores(_scoring(500), _scoring(300), min_rows=200)
    assert status == "ok"
    assert "handoff_weekday" not in scores and "handoff_month" not in scores
    assert scores["remaining_days"] == pytest.approx(0.0, abs=1e-9)


def test_drift_scores_too_few_orders() -> None:
    from late_delivery.steps.monitoring import drift_scores

    scores, status = drift_scores(_scoring(500), _scoring(150, shift=40), min_rows=200)
    assert status == "too_few_orders"
    assert all(v is None for v in scores.values())


def test_drift_scores_detect_real_shift() -> None:
    from late_delivery.steps.monitoring import drift_scores

    scores, _ = drift_scores(_scoring(500), _scoring(300, shift=40), min_rows=200)
    assert scores["remaining_days"] is not None and scores["remaining_days"] > 0.25


def test_feature_missing_from_reference_is_null_not_an_error() -> None:
    from late_delivery.steps.monitoring import drift_scores

    old = _scoring(500).drop("seller_prior_median_handling_h")  # built before the feature
    scores, status = drift_scores(old, _scoring(300), min_rows=200)
    assert status == "ok"
    assert scores["seller_prior_median_handling_h"] is None
    assert scores["remaining_days"] == pytest.approx(0.0, abs=1e-9)
