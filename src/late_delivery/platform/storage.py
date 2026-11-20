"""Hive-partitioned Parquet storage with atomic writes and content hashes."""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass
from datetime import date

import fsspec
import polars as pl

PARQUET_FILE = "part-0.parquet"


@dataclass(frozen=True)
class PartitionInfo:
    path: str
    rows: int
    content_hash: str


def partition_path(root: str, layer: str, table: str, ds: date) -> str:
    return f"{root}/{layer}/{table}/ds={ds:%Y-%m-%d}"


def _canonical(df: pl.DataFrame, sort_by: list[str]) -> pl.DataFrame:
    keys = sort_by + [c for c in df.columns if c not in sort_by]
    return df.sort(keys, nulls_last=True, maintain_order=True)


def content_hash(df: pl.DataFrame, sort_by: list[str]) -> str:
    """SHA-256 of the canonically sorted rows, independent of row order and file metadata."""
    buf = io.BytesIO()
    _canonical(df, sort_by).write_ipc(buf, compression="uncompressed")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _swap_dirs(fs: fsspec.AbstractFileSystem, target_path: str) -> list[str]:
    """Leftover temp/old folders for this partition from an interrupted write."""
    parent, name = target_path.rsplit("/", 1)
    if not fs.exists(parent):
        return []
    return [
        str(p).rstrip("/")
        for p in fs.ls(parent, detail=False)
        if str(p).rstrip("/").rsplit("/", 1)[-1].startswith((f"{name}.tmp-", f"{name}.old-"))
    ]


def write_partition(
    df: pl.DataFrame, root: str, layer: str, table: str, ds: date, sort_by: list[str]
) -> PartitionInfo:
    """Replace the partition without ever leaving the day missing.

    Write the new data to a temp folder, rename the current folder aside, rename the temp
    folder into place, then delete the old one. If the swap-in fails, the old folder is
    renamed back, so readers always see either the old or the new partition.
    """
    target = partition_path(root, layer, table, ds)
    fs, target_path = fsspec.core.url_to_fs(target)
    for leftover in _swap_dirs(fs, target_path):
        fs.rm(leftover, recursive=True)

    token = uuid.uuid4().hex[:8]
    tmp_path, old_path = f"{target_path}.tmp-{token}", f"{target_path}.old-{token}"
    canonical = _canonical(df, sort_by)
    fs.makedirs(tmp_path, exist_ok=True)
    with fs.open(f"{tmp_path}/{PARQUET_FILE}", "wb") as f:
        canonical.write_parquet(f, compression="zstd", statistics=False)

    had_previous = fs.exists(target_path)
    if had_previous:
        fs.mv(target_path, old_path, recursive=True)
    try:
        fs.mv(tmp_path, target_path, recursive=True)
    except Exception:
        if had_previous:
            fs.mv(old_path, target_path, recursive=True)
        raise
    if had_previous:
        fs.rm(old_path, recursive=True)
    return PartitionInfo(path=target, rows=canonical.height, content_hash=content_hash(df, sort_by))


def read_partition(root: str, layer: str, table: str, ds: date) -> pl.DataFrame:
    target = partition_path(root, layer, table, ds)
    fs, target_path = fsspec.core.url_to_fs(target)
    file_path = f"{target_path}/{PARQUET_FILE}"
    if not fs.exists(file_path):
        raise FileNotFoundError(file_path)
    with fs.open(file_path, "rb") as f:
        return pl.read_parquet(f)


def list_partitions(root: str, layer: str, table: str) -> list[date]:
    fs, base = fsspec.core.url_to_fs(f"{root}/{layer}/{table}")
    if not fs.exists(base):
        return []
    out: list[date] = []
    for entry in fs.ls(base, detail=False):
        name = str(entry).rstrip("/").rsplit("/", 1)[-1]
        if name.startswith("ds=") and ".tmp-" not in name and ".old-" not in name:
            out.append(date.fromisoformat(name[3:]))
    return sorted(out)


def scan_table(root: str, layer: str, table: str, up_to: date | None = None) -> pl.LazyFrame | None:
    """Lazily scan all partitions with ds <= up_to; None if the table has no partitions."""
    days = [d for d in list_partitions(root, layer, table) if up_to is None or d <= up_to]
    if not days:
        return None
    paths = [f"{partition_path(root, layer, table, d)}/{PARQUET_FILE}" for d in days]
    return pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")


def read_table(root: str, layer: str, table: str, up_to: date | None = None) -> pl.DataFrame | None:
    """Concatenate all partitions with ds <= up_to; None if the table has no partitions."""
    lf = scan_table(root, layer, table, up_to)
    return None if lf is None else lf.collect()


def delete_partition(root: str, layer: str, table: str, ds: date) -> None:
    fs, path = fsspec.core.url_to_fs(partition_path(root, layer, table, ds))
    if fs.exists(path):
        fs.rm(path, recursive=True)
