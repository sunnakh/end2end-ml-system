"""Row-level validation: split bronze rows into typed valid rows and quarantined rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandera.polars as pa
import polars as pl

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
INT64_SAFE_LIMIT = 9.0e18  # just under 2**63, exactly representable as a float
REASON_CODES = (
    "TYPE_PARSE_ERROR",
    "NULL_REQUIRED",
    "DUPLICATE_KEY",
    "NEGATIVE_AMOUNT",
    "TS_ORDER_VIOLATION",
    "ORPHAN_FK",
    "INVALID_ENUM",
    "OUT_OF_RANGE",
)


@dataclass(frozen=True)
class Contract:
    table: str
    key: list[str]
    dtypes: dict[str, pl.DataType]
    required: list[str]
    enums: dict[str, set[str]] = field(default_factory=dict)
    non_negative: list[str] = field(default_factory=list)
    # (earlier, later, tolerance): violation when earlier > later + tolerance.
    ts_order: list[tuple[str, str, timedelta]] = field(default_factory=list)
    fk: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # (column, when_column, when_value): column is required on rows where when_column == value.
    required_when: list[tuple[str, str, str]] = field(default_factory=list)


class FileLevelError(Exception):
    """The file as a whole breaks the contract (e.g. a missing column)."""


@dataclass(frozen=True)
class ValidationResult:
    valid: pl.DataFrame
    quarantine: pl.DataFrame
    duplicates_removed: int


def _blank_to_null(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.when(c.str.strip_chars() == "").then(None).otherwise(c)


def _cast(col: str, dtype: pl.DataType) -> pl.Expr:
    raw = _blank_to_null(col)
    if dtype == pl.Datetime("us"):
        return raw.str.to_datetime(TS_FORMAT, time_unit="us", strict=False)
    if dtype == pl.Int64:
        # Accept "3" and "3.0" (pandas-style exports); reject "3.5", inf, nan and values
        # outside the Int64 range. Unparseable values become null -> TYPE_PARSE_ERROR.
        f = raw.cast(pl.Float64, strict=False)
        whole = f.is_finite() & (f == f.floor()) & (f.abs() < INT64_SAFE_LIMIT)
        return pl.when(whole).then(f).otherwise(None).cast(pl.Int64, strict=False)
    if dtype == pl.Float64:
        # "nan" and "inf" parse as floats but are never valid amounts or measurements.
        v = raw.cast(pl.Float64, strict=False)
        return pl.when(v.is_finite()).then(v).otherwise(None)
    return raw.cast(dtype, strict=False)


def _pandera_schema(contract: Contract) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            col: pa.Column(dtype, nullable=col not in contract.required)
            for col, dtype in contract.dtypes.items()
        },
        strict=True,
    )


def validate(
    bronze: pl.DataFrame, contract: Contract, known_keys: dict[str, pl.Series]
) -> ValidationResult:
    cols = list(contract.dtypes)
    missing = [c for c in cols if c not in bronze.columns]
    if missing:
        raise FileLevelError(f"{contract.table}: missing columns {missing}")

    raw = bronze.with_columns(pl.col(cols).cast(pl.String))
    before = raw.height
    raw = raw.unique(subset=cols, keep="first", maintain_order=True)
    duplicates_removed = before - raw.height

    typed_names = {c: f"__t_{c}" for c in cols}
    work = raw.with_columns([_cast(c, contract.dtypes[c]).alias(typed_names[c]) for c in cols])

    def t(c: str) -> pl.Expr:
        return pl.col(typed_names[c])

    checks: list[tuple[str, pl.Expr, str]] = []
    for c in cols:
        present = _blank_to_null(c).is_not_null()
        checks.append(("TYPE_PARSE_ERROR", present & t(c).is_null(), f"{c} unparseable"))
        if c in contract.required:
            checks.append(("NULL_REQUIRED", ~present, f"{c} is null"))
    for c, when_col, when_value in contract.required_when:
        absent = (pl.col(when_col) == when_value) & _blank_to_null(c).is_null()
        checks.append(("NULL_REQUIRED", absent, f"{c} is null when {when_col}={when_value}"))
    if contract.key:
        dup = pl.int_range(pl.len()).over([t(k) for k in contract.key]) > 0
        checks.append(("DUPLICATE_KEY", dup, f"duplicate key {contract.key}"))
    for c in contract.non_negative:
        checks.append(("NEGATIVE_AMOUNT", t(c) < 0, f"{c} < 0"))
    for left, right, tolerance in contract.ts_order:
        late_by = f" by more than {tolerance}" if tolerance else ""
        checks.append(
            ("TS_ORDER_VIOLATION", t(left) > t(right) + tolerance, f"{left} > {right}{late_by}")
        )
    for c, ref in contract.fk.items():
        if ref in known_keys:
            orphan = t(c).is_not_null() & ~t(c).is_in(known_keys[ref].implode())
            checks.append(("ORPHAN_FK", orphan, f"{c} not in {ref}"))
    for c, allowed in contract.enums.items():
        bad = t(c).is_not_null() & ~t(c).is_in(sorted(allowed))
        checks.append(("INVALID_ENUM", bad, f"{c} not allowed"))
    for c, (lo, hi) in contract.ranges.items():
        checks.append(("OUT_OF_RANGE", (t(c) < lo) | (t(c) > hi), f"{c} outside [{lo}, {hi}]"))

    flagged = work.with_columns(
        pl.concat_list(
            [pl.when(cond.fill_null(False)).then(pl.lit(code)) for code, cond, _ in checks]
        )
        .list.drop_nulls()
        .list.unique(maintain_order=True)
        .alias("reason_codes"),
        pl.concat_list(
            [pl.when(cond.fill_null(False)).then(pl.lit(detail)) for _, cond, detail in checks]
        )
        .list.drop_nulls()
        .list.join("; ")
        .alias("reason_detail"),
    )
    is_bad = pl.col("reason_codes").list.len() > 0

    valid = flagged.filter(~is_bad).select([t(c).alias(c) for c in cols])
    valid = _pandera_schema(contract).validate(valid)
    quarantine = flagged.filter(is_bad).select(
        [c for c in raw.columns] + ["reason_codes", "reason_detail"]
    )
    return ValidationResult(
        valid=valid, quarantine=quarantine, duplicates_removed=duplicates_removed
    )
