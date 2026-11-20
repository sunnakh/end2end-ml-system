"""End-to-end M2 exit checks on the synthetic fixture."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from late_delivery.cli import backfill, rerun, run_step
from late_delivery.config import Settings
from late_delivery.platform.pii import hash_ids
from late_delivery.platform.storage import content_hash, list_partitions, read_partition, read_table
from late_delivery.source.contracts import CONTRACTS
from late_delivery.source.simulator import inject_faults, load_raw, simulate_day, write_landing
from tests.conftest import FIRST_DAY, LAST_DAY, make_settings

FAULTS = {"enabled": True, "seed": 7, "rate": 0.4}


def _lake_hashes(root: Path) -> dict[str, str]:
    out = {}
    for f in sorted(root.rglob("*.parquet")):
        if "/gold/training/" in str(f):
            continue
        df = pl.read_parquet(f)
        out[str(f.relative_to(root))] = content_hash(df, [])
    return out


def _manifests(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "manifests").glob("*.json"))]


def test_backfill_runs_all_days_with_manifests(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    assert backfill(s, FIRST_DAY, LAST_DAY)
    manifests = _manifests(tmp_path / "lake")
    assert len(manifests) == (LAST_DAY - FIRST_DAY).days + 1
    assert all(m["ok"] for m in manifests)
    assert {m["ds"] for m in manifests} == {
        f"2017-01-{d:02d}" for d in range(FIRST_DAY.day, LAST_DAY.day + 1)
    }
    assert (tmp_path / "lake" / "reports" / "drift" / "ds=2017-01-21" / "report.json").exists()


def test_rerun_day_identical_hashes(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake", faults=FAULTS)
    assert backfill(s, FIRST_DAY, LAST_DAY)
    first = _lake_hashes(tmp_path / "lake")
    assert backfill(s, FIRST_DAY, LAST_DAY)
    assert _lake_hashes(tmp_path / "lake") == first
    assert rerun(s, date(2017, 1, 6))
    assert _lake_hashes(tmp_path / "lake") == first


def _quarantine_set(s: Settings) -> set[tuple[str, tuple[str, ...], str]]:
    out = set()
    for table, contract in CONTRACTS.items():
        q = read_table(s.paths.data_root, "quarantine", table)
        if q is None:
            continue
        for row in q.iter_rows(named=True):
            key = tuple(str(row[k]) for k in contract.key)
            for code in row["reason_codes"]:
                out.add((table, key, code))
    return out


def test_quarantine_matches_injected_exactly(mini_olist: Path, tmp_path: Path) -> None:
    clean = make_settings(mini_olist, tmp_path / "clean")
    assert backfill(clean, FIRST_DAY, LAST_DAY)
    natural = _quarantine_set(clean)

    faulty = make_settings(mini_olist, tmp_path / "faulty", faults=FAULTS)
    assert backfill(faulty, FIRST_DAY, LAST_DAY)

    injected: set[tuple[str, tuple[str, ...], str]] = set()
    duplicates = 0
    for f in sorted((tmp_path / "faulty" / "landing").glob("ds=*/_injected.json")):
        for fault in json.loads(f.read_text()):
            if fault["fault"] == "duplicate":
                duplicates += 1
                continue
            key = dict(fault["key"])
            if "customer_id" in key:
                key["customer_id"] = (
                    pl.DataFrame({"x": [key["customer_id"]]})
                    .select(hash_ids(pl.col("x"), "test-salt"))
                    .item()
                )
            ordered = tuple(key[k] for k in CONTRACTS[fault["table"]].key)
            injected.add((fault["table"], ordered, fault["expected_code"]))
    assert injected, "seed should inject at least one row fault"
    assert _quarantine_set(faulty) == natural | injected

    def removed(lake: str) -> int:
        return sum(
            counts.get("_duplicates_removed", 0)
            for m in _manifests(tmp_path / lake)
            for counts in m["quarantine_counts"].values()
        )

    assert removed("faulty") - removed("clean") == duplicates


def test_schema_drift_fails_partition_writes_no_silver(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    assert backfill(s, FIRST_DAY, date(2017, 1, 3))
    ds = date(2017, 1, 4)
    tables, faults = inject_faults(
        simulate_day(load_raw(mini_olist), ds, FIRST_DAY), ds, seed=1, rate=0.0,
        kinds=("schema_drift",),
    )  # fmt: skip
    write_landing(tables, s.paths.data_root, ds, faults)
    assert run_step(s, "ingest_bronze", ds)
    assert not run_step(s, "validate_to_silver", ds)
    for table in ("orders_changelog", "order_items", "customers"):
        assert ds not in list_partitions(s.paths.data_root, "silver", table)
    failed = [m for m in _manifests(tmp_path / "lake") if not m["ok"]]
    assert failed[-1]["failed_step"] == "validate_to_silver"
    assert "missing columns" in failed[-1]["error"]


def test_missing_landing_fails_ingest(mini_olist: Path, tmp_path: Path) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    assert not run_step(s, "ingest_bronze", date(2017, 1, 9))
    failed = _manifests(tmp_path / "lake")[-1]
    assert failed["ok"] is False and "FileNotFoundError" in failed["error"]


def test_reconciliation_landing_eq_silver_plus_quarantine_plus_dupes(
    mini_olist: Path, tmp_path: Path
) -> None:
    s = make_settings(mini_olist, tmp_path / "lake", faults=FAULTS)
    assert backfill(s, FIRST_DAY, LAST_DAY)
    silver_name = {"orders": "orders_changelog", "order_items": "order_items",
                   "order_payments": "order_payments", "customers": "customers"}  # fmt: skip
    for m in _manifests(tmp_path / "lake"):
        ds = date.fromisoformat(m["ds"])
        for table, c in m["row_counts"].items():
            assert c["bronze"] == c["valid"] + c["quarantined"] + c["duplicates_removed"], (
                table,
                ds,
            )
            if table in silver_name:
                silver = read_partition(s.paths.data_root, "silver", silver_name[table], ds)
                assert silver.height == c["valid"]


def test_rerun_midrange_day_updates_later_current(mini_olist: Path, tmp_path: Path) -> None:
    clean_a = make_settings(mini_olist, tmp_path / "a")
    assert backfill(clean_a, FIRST_DAY, LAST_DAY)
    expected = _lake_hashes(tmp_path / "a")

    clean_b = make_settings(mini_olist, tmp_path / "b")
    faulty_b = make_settings(mini_olist, tmp_path / "b",
                             faults={"enabled": True, "seed": 2, "rate": 1.0})  # fmt: skip
    assert backfill(clean_b, FIRST_DAY, LAST_DAY)
    assert rerun(faulty_b, date(2017, 1, 5))
    assert _lake_hashes(tmp_path / "b") != expected
    assert rerun(clean_b, date(2017, 1, 5))
    assert _lake_hashes(tmp_path / "b") == expected


@pytest.mark.parametrize("bad", ["nope"])
def test_unknown_step_rejected(mini_olist: Path, tmp_path: Path, bad: str) -> None:
    s = make_settings(mini_olist, tmp_path / "lake")
    with pytest.raises(ValueError, match="unknown step"):
        run_step(s, bad, FIRST_DAY)
