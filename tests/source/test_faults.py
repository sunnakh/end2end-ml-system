import json
from datetime import date
from pathlib import Path

import polars as pl

from late_delivery.source.simulator import (
    ROW_FAULTS,
    inject_faults,
    load_raw,
    simulate_day,
    write_landing,
)

DAY0 = date(2017, 1, 2)


def _tables(mini_olist: Path, ds: date = DAY0) -> dict[str, pl.DataFrame]:
    return simulate_day(load_raw(mini_olist), ds, DAY0)


def test_same_seed_same_faults(mini_olist: Path) -> None:
    a_tables, a = inject_faults(_tables(mini_olist), DAY0, seed=7, rate=0.5)
    b_tables, b = inject_faults(_tables(mini_olist), DAY0, seed=7, rate=0.5)
    assert a == b and len(a) > 0
    assert all(a_tables[t].equals(b_tables[t]) for t in a_tables)


def test_different_day_different_faults(mini_olist: Path) -> None:
    _, a = inject_faults(_tables(mini_olist), DAY0, seed=7, rate=0.5)
    _, b = inject_faults(_tables(mini_olist), date(2017, 1, 3), seed=7, rate=0.5)
    assert a != b


def test_rate_zero_changes_nothing(mini_olist: Path) -> None:
    clean = _tables(mini_olist)
    out, faults = inject_faults(clean, DAY0, seed=7, rate=0.0)
    assert faults == []
    assert all(out[t].equals(clean[t]) for t in clean)


def test_each_fault_changes_its_row(mini_olist: Path) -> None:
    clean = _tables(mini_olist)
    out, faults = inject_faults(clean, DAY0, seed=3, rate=1.0)
    kinds = {f.fault for f in faults}
    assert kinds <= set(ROW_FAULTS)
    for f in faults:
        if f.fault == "duplicate":
            assert out[f.table].height > clean[f.table].height
        else:
            assert not out[f.table].equals(clean[f.table])


def test_created_order_rows_never_faulted(mini_olist: Path) -> None:
    _, faults = inject_faults(_tables(mini_olist), DAY0, seed=3, rate=1.0)
    assert not [f for f in faults if f.table == "orders" and f.fault != "duplicate"]


def test_schema_drift_renames_a_column(mini_olist: Path) -> None:
    out, faults = inject_faults(
        _tables(mini_olist), DAY0, seed=1, rate=0.0, kinds=("schema_drift",)
    )
    assert [f.expected_code for f in faults] == ["FILE_LEVEL"]
    assert "order_status" not in out["orders"].columns


def test_injected_json_written(mini_olist: Path, tmp_path: Path) -> None:
    tables, faults = inject_faults(_tables(mini_olist), DAY0, seed=3, rate=1.0)
    write_landing(tables, str(tmp_path), DAY0, faults)
    data = json.loads((tmp_path / "landing" / "ds=2017-01-02" / "_injected.json").read_text())
    assert len(data) == len(faults)
    assert {"table", "key", "fault", "expected_code"} <= set(data[0])
