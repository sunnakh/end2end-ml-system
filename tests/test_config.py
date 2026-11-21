from datetime import date
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


def test_training_defaults_match_design() -> None:
    t = load_settings("local", Path("configs")).training
    assert t.train_start == date(2017, 1, 1)
    assert t.valid_start == date(2018, 4, 1)
    assert t.test_start == date(2018, 6, 1)
    assert t.test_end == date(2018, 8, 31)
    assert t.evaluation_cutoff == date(2018, 10, 17)
    assert t.window_months == [15, 9, 6]
    assert (t.budget_share, t.min_score) == (0.05, 0.10)
    assert (t.contact_cost, t.late_cost) == (2.0, 20.0)
    assert (t.bootstrap_resamples, t.seed) == (1000, 0)
    assert t.models_dir == "data/models"
    assert (t.gate.margin, t.gate.slice_tolerance, t.gate.min_slice_rows) == (0.02, 0.05, 300)


def test_training_rejects_unknown_key(tmp_path: Path) -> None:
    d = _write(tmp_path, BASE, "training:\n  bogus: 1\n")
    with pytest.raises(ValidationError):
        load_settings("local", d)


def test_training_defaults_without_yaml_block(tmp_path: Path) -> None:
    s = load_settings("local", _write(tmp_path, BASE, ""))
    assert s.training.test_start == date(2018, 6, 1)
