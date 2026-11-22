"""The evaluation gate: a challenger replaces the champion only if it is clearly better on
validation, no major slice gets much worse, and its probabilities are not worse than a
constant guess. Test months never enter here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from late_delivery.config import TrainingConfig
from late_delivery.modeling.bootstrap import paired_day_bootstrap
from late_delivery.modeling.metrics import brier

SLICE_TYPES = ("region", "seller_size")


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool | None  # None: skipped, the data cannot judge it (e.g. no late orders)
    detail: str
    diff: float | None = None  # challenger minus reference recall@K, where one applies


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)


def recall_at_k(df: pl.DataFrame) -> float:
    late = int(df["late"].sum())
    if late == 0:
        return float("nan")
    return df.filter(pl.col("contacted") & (pl.col("late") == 1)).height / late


def _beats(
    name: str, challenger: pl.DataFrame, ref: pl.DataFrame, cfg: TrainingConfig
) -> GateCheck:
    if int(challenger["late"].sum()) == 0:
        return GateCheck(
            f"beats {name}", False,
            "validation has no late orders: recall@K is undefined, so no improvement can be shown",
        )  # fmt: skip
    diff, lo, hi = paired_day_bootstrap(
        challenger, ref, recall_at_k, n=cfg.bootstrap_resamples, seed=cfg.seed
    )
    ok = diff >= cfg.gate.margin and lo > 0
    detail = (
        f"recall@K {recall_at_k(challenger):.3f} vs {recall_at_k(ref):.3f}: "
        f"diff {diff:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), needs >= {cfg.gate.margin:+.3f} "
        "with CI above 0"
    )
    return GateCheck(f"beats {name}", ok, detail, diff)


def _slices(challenger: pl.DataFrame, ref: pl.DataFrame, ref_name: str,
            cfg: TrainingConfig) -> list[GateCheck]:  # fmt: skip
    checks = []
    for kind in SLICE_TYPES:
        for value in sorted(challenger[kind].unique().to_list()):
            c = challenger.filter(pl.col(kind) == value)
            if c.height < cfg.gate.min_slice_rows:  # too few orders to judge fairly
                checks.append(GateCheck(
                    f"slice {kind}={value}", None,
                    f"{c.height} orders < {cfg.gate.min_slice_rows}: not judged",
                ))  # fmt: skip
                continue
            r = ref.filter(pl.col(kind) == value)
            name = f"slice {kind}={value}"
            rc, rr = recall_at_k(c), recall_at_k(r)
            if np.isnan(rc) or np.isnan(rr):
                checks.append(GateCheck(name, None, f"{c.height} orders, no late orders: skipped"))
                continue
            # Fails only if the drop is beyond the tolerance AND clearly real: a slice with
            # a few dozen late orders swings by several points on a handful of orders.
            diff, lo, hi = paired_day_bootstrap(
                c, r, recall_at_k, n=cfg.bootstrap_resamples, seed=cfg.seed
            )
            ok = not (diff < -cfg.gate.slice_tolerance and hi < 0)
            detail = (
                f"recall@K {rc:.3f} vs {ref_name} {rr:.3f} ({diff:+.3f}, 95% CI {lo:+.3f} to "
                f"{hi:+.3f}); fails if below -{cfg.gate.slice_tolerance:.3f} with CI below 0; "
                f"{c.height} orders, {int(c['late'].sum())} late"
            )
            checks.append(GateCheck(name, ok, detail, diff))
    return checks


def run_gate(
    challenger: pl.DataFrame,
    references: dict[str, pl.DataFrame],
    *,
    cfg: TrainingConfig,
    train_late_rate: float,
) -> GateResult:
    """Frames are scored validation rows with late, score, contacted, handoff_date and the
    slice columns. `references` has "rule" and, after the first run, "champion" (scored by
    the champion's train-only gate model, which has never seen validation)."""
    checks = [_beats("rule", challenger, references["rule"], cfg)]
    if "champion" in references:
        checks.append(_beats("champion", challenger, references["champion"], cfg))
    slice_ref = "champion" if "champion" in references else "rule"
    checks += _slices(challenger, references[slice_ref], slice_ref, cfg)
    y = challenger["late"].to_numpy()
    b = brier(y, challenger["score"].to_numpy())
    b0 = brier(y, np.full(len(y), train_late_rate))
    checks.append(
        GateCheck(
            "calibration",
            b <= b0,
            f"Brier {b:.4f} vs constant training late rate {train_late_rate:.3f}: {b0:.4f}",
        )  # fmt: skip
    )
    return GateResult(all(c.passed is not False for c in checks), checks)


@dataclass(frozen=True)
class Champion:
    run_id: str
    model_path: str  # refit on train + validation: what gets served
    gate_model_path: str  # train only: what future challengers are compared against
    snapshot_hashes: dict[str, str]
    valid_metrics: dict[str, float]
    gate_train_end: str  # last handoff date the gate model trained on (ISO)
    features: list[str]  # columns the gate model needs


def read_champion(models_dir: Path) -> Champion | None:
    """None on the first run. A file that exists but cannot be read is an error, not a
    reason to treat the next challenger as the first one."""
    path = models_dir / "champion.json"
    if not path.exists():
        return None
    try:
        return Champion(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"{path} is unreadable: {e}") from e


def write_champion(models_dir: Path, champion: Champion) -> None:
    """Atomic: a crash mid-write leaves the previous champion in place."""
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = models_dir / "champion.json.tmp"
    tmp.write_text(json.dumps(asdict(champion), indent=2, sort_keys=True))
    os.replace(tmp, models_dir / "champion.json")
