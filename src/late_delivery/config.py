"""Typed settings loaded from configs/base.yaml merged with configs/<env>.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    raw_dir: Path
    data_root: str


class ReplayConfig(_Strict):
    start: date
    end: date


class PiiConfig(_Strict):
    salt_env: str = "LATE_DELIVERY_PII_SALT"


class ValidationConfig(_Strict):
    max_quarantine_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    min_rows_for_breaker: int = Field(default=100, ge=0)


class FaultsConfig(_Strict):
    enabled: bool = False
    seed: int = 0
    rate: float = Field(default=0.01, ge=0.0, le=1.0)


class DriftConfig(_Strict):
    reference_start: date
    reference_end: date
    # Daily samples are too small for PSI (random variation looks like drift), so each
    # report compares a trailing window of days, and stays silent below min_rows orders.
    window_days: int = Field(default=7, ge=1)
    min_rows: int = Field(default=200, ge=0)


class GateConfig(_Strict):
    margin: float = Field(default=0.02, ge=0.0)  # recall@K a challenger must add
    slice_tolerance: float = Field(default=0.05, ge=0.0)  # worst allowed drop on any slice
    min_slice_rows: int = Field(default=300, ge=1)  # smaller slices are too noisy to judge


class TrainingConfig(_Strict):
    # Splits by handoff date; the decision snapshot is cut at test_start.
    train_start: date = date(2017, 1, 1)
    valid_start: date = date(2018, 4, 1)
    test_start: date = date(2018, 6, 1)
    test_end: date = date(2018, 8, 31)  # inclusive
    evaluation_cutoff: date = date(2018, 10, 17)  # test labels, once they have resolved
    window_months: list[int] = Field(default_factory=lambda: [15, 9, 6], min_length=1)
    # Daily contact decision (simulated costs, see design 03).
    budget_share: float = Field(default=0.05, gt=0.0, le=1.0)
    min_score: float = Field(default=0.10, ge=0.0, le=1.0)
    contact_cost: float = Field(default=2.0, ge=0.0)
    late_cost: float = Field(default=20.0, ge=0.0)
    bootstrap_resamples: int = Field(default=1000, ge=1)
    seed: int = 0
    models_dir: str = "data/models"
    gate: GateConfig = GateConfig()


class Settings(_Strict):
    paths: PathsConfig
    replay: ReplayConfig
    pii: PiiConfig = PiiConfig()
    validation: ValidationConfig = ValidationConfig()
    faults: FaultsConfig = FaultsConfig()
    drift: DriftConfig
    training: TrainingConfig = TrainingConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_settings(env: str = "local", config_dir: Path = Path("configs")) -> Settings:
    merged = _deep_merge(
        _read_yaml(config_dir / "base.yaml"), _read_yaml(config_dir / f"{env}.yaml")
    )
    return Settings.model_validate(merged)
