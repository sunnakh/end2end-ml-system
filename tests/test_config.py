from pathlib import Path

import pytest
from pydantic import ValidationError

from late_delivery.config import load_settings

BASE = """
paths:
  raw_dir: data/raw/olist
  data_root: data/base
replay:
  start: 2016-09-04
  end: 2018-10-17
drift:
  reference_start: 2017-06-01
  reference_end: 2017-08-31
"""


def _write(tmp_path: Path, base: str, local: str) -> Path:
    (tmp_path / "base.yaml").write_text(base)
    (tmp_path / "local.yaml").write_text(local)
    return tmp_path


def test_env_overrides_base(tmp_path: Path) -> None:
    d = _write(tmp_path, BASE, "paths:\n  data_root: data/local\n")
    s = load_settings("local", d)
    assert s.paths.data_root == "data/local"
    assert s.paths.raw_dir == Path("data/raw/olist")
    assert s.validation.max_quarantine_rate == 0.05
    assert s.pii.salt_env == "LATE_DELIVERY_PII_SALT"


def test_unknown_field_rejected(tmp_path: Path) -> None:
    d = _write(tmp_path, BASE, "paths:\n  bogus: 1\n")
    with pytest.raises(ValidationError):
        load_settings("local", d)


def test_rate_bounds(tmp_path: Path) -> None:
    d = _write(tmp_path, BASE, "validation:\n  max_quarantine_rate: 1.5\n")
    with pytest.raises(ValidationError):
        load_settings("local", d)


def test_repo_configs_load() -> None:
    s = load_settings("local")
    assert s.replay.start < s.replay.end
