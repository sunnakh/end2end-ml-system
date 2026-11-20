from pathlib import Path

import polars as pl
import pytest

from late_delivery.platform.pii import get_salt, hash_ids


def _hash(values: list[str | None], salt: str) -> list[str | None]:
    df = pl.DataFrame({"id": values}, schema={"id": pl.String})
    return df.select(hash_ids(pl.col("id"), salt))["id"].to_list()


def test_same_input_same_hash() -> None:
    a, b = _hash(["x", "x"], "s")
    assert a == b
    assert a is not None and len(a) == 64 and a != "x"


def test_salt_changes_hash() -> None:
    assert _hash(["x"], "s1") != _hash(["x"], "s2")


def test_null_stays_null() -> None:
    assert _hash([None], "s") == [None]


def test_missing_salt_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SALT_VAR", raising=False)
    with pytest.raises(RuntimeError, match="SALT_VAR"):
        get_salt("SALT_VAR")


def test_empty_salt_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SALT_VAR", "")
    with pytest.raises(RuntimeError, match="SALT_VAR"):
        get_salt("SALT_VAR")


def test_fingerprint_is_stable_and_hides_salt() -> None:
    from late_delivery.platform.pii import salt_fingerprint

    fp = salt_fingerprint("my-secret")
    assert fp == salt_fingerprint("my-secret")
    assert fp != salt_fingerprint("other-secret")
    assert "my-secret" not in fp and len(fp) == 16


def test_first_run_records_fingerprint(tmp_path: Path) -> None:
    from late_delivery.platform.pii import ensure_lake_salt, salt_fingerprint

    ensure_lake_salt(str(tmp_path), "s1")
    assert (tmp_path / "_salt_fingerprint").read_text().strip() == salt_fingerprint("s1")
    ensure_lake_salt(str(tmp_path), "s1")  # same salt again: fine


def test_mismatched_salt_refused(tmp_path: Path) -> None:
    from late_delivery.platform.pii import SaltMismatchError, ensure_lake_salt

    ensure_lake_salt(str(tmp_path), "s1")
    with pytest.raises(SaltMismatchError, match="different PII salt"):
        ensure_lake_salt(str(tmp_path), "s2")
