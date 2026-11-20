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


class Settings(_Strict):
    paths: PathsConfig
    replay: ReplayConfig
    pii: PiiConfig = PiiConfig()
    validation: ValidationConfig = ValidationConfig()
    faults: FaultsConfig = FaultsConfig()
    drift: DriftConfig


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
