from datetime import date
from pathlib import Path

import polars as pl
import pytest

from late_delivery.platform.storage import (
    content_hash,
    list_partitions,
    partition_path,
    read_partition,
    read_table,
    write_partition,
)

DS = date(2017, 3, 1)


def _df() -> pl.DataFrame:
    return pl.DataFrame({"id": ["b", "a", "c"], "v": [2, 1, 3]})


def test_partition_path_layout() -> None:
    assert partition_path("root", "silver", "orders", DS) == "root/silver/orders/ds=2017-03-01"


def test_hash_ignores_row_order() -> None:
    df = _df()
    assert content_hash(df, ["id"]) == content_hash(df.reverse(), ["id"])


def test_hash_changes_with_values() -> None:
    df = _df()
    other = df.with_columns(pl.when(pl.col("id") == "a").then(9).otherwise(pl.col("v")).alias("v"))
    assert content_hash(df, ["id"]) != content_hash(other, ["id"])


def test_write_twice_replaces(tmp_path: Path) -> None:
    root = str(tmp_path)
    write_partition(_df(), root, "bronze", "t", DS, ["id"])
    second = pl.DataFrame({"id": ["z"], "v": [0]})
    write_partition(second, root, "bronze", "t", DS, ["id"])
    assert read_partition(root, "bronze", "t", DS).to_dicts() == [{"id": "z", "v": 0}]


def test_rewrite_same_data_same_hash(tmp_path: Path) -> None:
    root = str(tmp_path)
    a = write_partition(_df(), root, "bronze", "t", DS, ["id"])
    b = write_partition(_df().reverse(), root, "bronze", "t", DS, ["id"])
    assert a.content_hash == b.content_hash
    assert a.rows == 3


def test_written_rows_sorted(tmp_path: Path) -> None:
    write_partition(_df(), str(tmp_path), "bronze", "t", DS, ["id"])
    assert read_partition(str(tmp_path), "bronze", "t", DS)["id"].to_list() == ["a", "b", "c"]


def test_empty_frame_roundtrips_schema(tmp_path: Path) -> None:
    empty = pl.DataFrame(schema={"id": pl.String, "v": pl.Int64})
    info = write_partition(empty, str(tmp_path), "bronze", "t", DS, ["id"])
    back = read_partition(str(tmp_path), "bronze", "t", DS)
    assert info.rows == 0
    assert back.schema == empty.schema


def test_read_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_partition(str(tmp_path), "bronze", "t", DS)


def test_read_table_up_to(tmp_path: Path) -> None:
    root = str(tmp_path)
    for i, d in enumerate([date(2017, 3, 1), date(2017, 3, 2), date(2017, 3, 3)]):
        write_partition(pl.DataFrame({"id": [str(i)], "v": [i]}), root, "silver", "t", d, ["id"])
    got = read_table(root, "silver", "t", up_to=date(2017, 3, 2))
    assert sorted(got["id"].to_list()) == ["0", "1"]
    assert "ds" not in got.columns
    assert read_table(root, "silver", "t").height == 3


def test_read_table_missing_returns_none(tmp_path: Path) -> None:
    assert read_table(str(tmp_path), "silver", "nope") is None


def test_crash_during_swap_keeps_previous_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fsspec.implementations.local import LocalFileSystem

    root = str(tmp_path)
    write_partition(_df(), root, "bronze", "t", DS, ["id"])
    real_mv = LocalFileSystem.mv

    def crash_on_swap_in(self: LocalFileSystem, path1: str, path2: str, **kw: object) -> None:
        if ".tmp-" in str(path1):
            raise OSError("simulated crash while moving the new partition in")
        real_mv(self, path1, path2, **kw)

    monkeypatch.setattr(LocalFileSystem, "mv", crash_on_swap_in)
    with pytest.raises(OSError, match="simulated crash"):
        write_partition(pl.DataFrame({"id": ["z"], "v": [0]}), root, "bronze", "t", DS, ["id"])
    assert read_partition(root, "bronze", "t", DS).height == 3  # old data still there


def test_leftover_temp_dirs_cleaned(tmp_path: Path) -> None:
    base = tmp_path / "bronze" / "t"
    (base / "ds=2017-03-01.tmp-dead0001").mkdir(parents=True)
    (base / "ds=2017-03-01.old-dead0002").mkdir(parents=True)
    write_partition(_df(), str(tmp_path), "bronze", "t", DS, ["id"])
    assert sorted(p.name for p in base.iterdir()) == ["ds=2017-03-01"]


def test_list_partitions_ignores_swap_dirs(tmp_path: Path) -> None:
    write_partition(_df(), str(tmp_path), "bronze", "t", DS, ["id"])
    (tmp_path / "bronze" / "t" / "ds=2017-03-02.old-dead0002").mkdir()
    (tmp_path / "bronze" / "t" / "ds=2017-03-03.tmp-dead0003").mkdir()
    assert list_partitions(str(tmp_path), "bronze", "t") == [DS]
