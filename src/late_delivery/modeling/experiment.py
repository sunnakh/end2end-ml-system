"""One M3 run: compare candidates on validation, gate the best, refit, test month by month,
run the walk-forward and leakage diagnostics, and keep the champion if the gate passes.

Decisions use only the decision snapshot (cut at the test start) and validation months.
Test months are scored after every choice is fixed and never feed back.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from late_delivery.config import TrainingConfig
from late_delivery.features import FEATURE_COLUMNS
from late_delivery.modeling.bootstrap import day_bootstrap
from late_delivery.modeling.contact import decision_metrics, select_contacts
from late_delivery.modeling.gate import (
    SLICE_TYPES,
    Champion,
    GateResult,
    read_champion,
    recall_at_k,
    run_gate,
    write_champion,
)
from late_delivery.modeling.metrics import (
    average_precision,
    brier,
    ece,
    pr_points,
    reliability_bins,
    roc_auc,
)
from late_delivery.modeling.models import (
    LightGBMModel,
    LogRegModel,
    Model,
    PlattCalibrated,
    RuleBaseline,
    calendar_only,
    load_model,
    save_model,
)
from late_delivery.modeling.splits import between, months_back, prepare

CALIBRATION_ECE = 0.02  # above this validation ECE, a Platt-calibrated version is tried


def _default_lgbm() -> list[dict[str, float]]:
    grid = itertools.product([15, 31, 63], [0.03, 0.1], [20, 100])
    return [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mc} for nl, lr, mc in grid]


@dataclass(frozen=True)
class Grid:
    logreg_l2: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    lgbm: list[dict[str, float]] = field(default_factory=_default_lgbm)


@dataclass(frozen=True)
class Snapshot:
    frame: pl.DataFrame  # gold/training rows, incl. late_without_delivery
    content_hash: str
    meta: dict[str, Any] = field(default_factory=dict)  # the snapshot's _meta.json


@dataclass(frozen=True)
class Spec:
    """A model type plus settings: enough to build a fresh, unfitted copy."""

    kind: str  # "logreg" | "lightgbm"
    params: dict[str, float]
    calibrated: bool = False

    def build(self, seed: int) -> Model:
        base: Model
        if self.kind == "logreg":
            base = LogRegModel(l2=self.params["l2"])
        else:
            base = LightGBMModel(
                num_leaves=int(self.params["num_leaves"]),
                learning_rate=self.params["learning_rate"],
                min_child_samples=int(self.params["min_child_samples"]),
                seed=seed,
            )
        return PlattCalibrated(base) if self.calibrated else base

    def label(self) -> str:
        p = ", ".join(f"{k}={v:g}" for k, v in sorted(self.params.items()))
        return f"{self.kind}({p}){' + platt' if self.calibrated else ''}"


@dataclass
class ExperimentResult:
    run_id: str
    git_commit: str
    snapshot_hashes: dict[str, str]
    counts: dict[str, int]
    candidates: pl.DataFrame
    chosen: dict[str, Any]
    gate: GateResult
    calendar_gate: GateResult
    monthly: pl.DataFrame
    slices: pl.DataFrame
    undelivered: pl.DataFrame
    leakage: dict[str, float]
    curves: dict[str, pl.DataFrame]
    fits: pl.DataFrame
    champion_written: bool
    calibration: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    snapshot_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


class _Run:
    """Holds the run's config and records every fit, so tests can audit what each model saw."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.fit_rows: list[dict[str, Any]] = []

    def fit(self, model: Model, train: pl.DataFrame, purpose: str, cutoff: date,
            scored_month_start: date | None = None) -> Model:  # fmt: skip
        model.fit(train)
        self.fit_rows.append({
            "purpose": purpose, "model": model.name, "rows": train.height,
            "min_date": train["handoff_date"].min(), "max_date": train["handoff_date"].max(),
            "snapshot_cutoff": cutoff, "scored_month_start": scored_month_start,
        })  # fmt: skip
        return model

    def decide(self, model: Model, rows: pl.DataFrame) -> pl.DataFrame:
        """Scores rows and applies the daily contact rule."""
        scored = rows.with_columns(pl.Series("score", model.score(rows), dtype=pl.Float64))
        min_score = self.cfg.min_score if model.uses_threshold else None
        return select_contacts(scored, budget_share=self.cfg.budget_share, min_score=min_score)

    def metrics(self, decided: pl.DataFrame, probabilities: bool) -> dict[str, float]:
        c = self.cfg
        m = decision_metrics(
            decided, budget_share=c.budget_share, contact_cost=c.contact_cost, late_cost=c.late_cost
        )
        y, s = decided["late"].to_numpy(), decided["score"].to_numpy()
        finite = np.where(np.isfinite(s), s, -1e18)
        m["pr_auc"] = average_precision(y, finite)
        m["roc_auc"] = roc_auc(y, finite)
        m["brier"] = brier(y, s) if probabilities else float("nan")
        m["ece"] = ece(y, s) if probabilities else float("nan")
        return m


def _check_champion_usable(champion: Champion, cfg: TrainingConfig, df: pl.DataFrame) -> None:
    """Refuse a champion the gate cannot judge fairly: one whose gate model trained on rows
    inside the current validation months (it would score its own training data), or one
    that needs columns the snapshot no longer has."""
    if date.fromisoformat(champion.gate_train_end) >= cfg.valid_start:
        raise ValueError(
            f"champion {champion.run_id} was trained on rows up to {champion.gate_train_end}, "
            f"inside the validation period starting {cfg.valid_start}: comparing on validation "
            "would favour it. Retire it (move champion.json aside) or restore the earlier dates."
        )
    missing = [f for f in champion.features if f not in df.columns]
    if missing:
        raise ValueError(
            f"champion {champion.run_id} needs features the snapshot lacks: {missing}. "
            "Rebuild the snapshot with them, or retire the champion."
        )


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def _months(start: date, end_exclusive: date) -> list[date]:
    out, d = [], _month_start(start)
    while d < end_exclusive:
        out.append(d)
        d = _next_month(d)
    return out


def run_experiment(
    cfg: TrainingConfig,
    load_snapshot: Callable[[date], Snapshot],
    models_dir: Path,
    *,
    run_id: str,
    git_commit: str,
    grid: Grid | None = None,
) -> ExperimentResult:
    grid = grid or Grid()
    run = _Run(cfg)
    clock = time.perf_counter()
    started, timings = clock, {}

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round(now - clock, 2)
        clock = now

    test_end = cfg.test_end + timedelta(days=1)  # exclusive

    decision = load_snapshot(cfg.test_start)
    df = prepare(decision.frame).filter(pl.col("handoff_date") >= cfg.train_start)
    valid = between(df, cfg.valid_start, cfg.test_start)
    evaluation = load_snapshot(cfg.evaluation_cutoff)
    ev = prepare(evaluation.frame)
    test = between(ev, cfg.test_start, test_end)

    # 1. Candidates: every window x every model setting, judged on validation.
    specs = [Spec("logreg", {"l2": l2}) for l2 in grid.logreg_l2] + [
        Spec("lightgbm", dict(p)) for p in grid.lgbm
    ]
    rows, fitted = [], {}
    for window in cfg.window_months:
        train = between(df, months_back(cfg.valid_start, window), cfg.valid_start)
        for spec in specs:
            model = run.fit(spec.build(cfg.seed), train, "candidate", cfg.test_start)
            decided = run.decide(model, valid)
            m = run.metrics(decided, probabilities=True)
            rows.append({"model": spec.label(), "kind": spec.kind, "window_months": window, **m})
            fitted[(spec.label(), window)] = (spec, model, decided, train)
    candidates = pl.DataFrame(rows).sort(
        ["recall_at_k", "pr_auc", "model", "window_months"], descending=[True, True, False, False]
    )
    best = candidates.row(0, named=True)
    spec, model, chosen_valid, chosen_train = fitted[(best["model"], best["window_months"])]
    window = int(best["window_months"])

    lap("candidates")

    # 2. Calibration, only if the chosen model's probabilities are clearly off. Both
    #    versions are reported whenever it is tried.
    calibration: dict[str, Any] = {"tried": False, "base_ece": best["ece"],
                                   "threshold_ece": CALIBRATION_ECE}  # fmt: skip
    if best["ece"] > CALIBRATION_ECE:
        cal_spec = Spec(spec.kind, spec.params, calibrated=True)
        cal = run.fit(cal_spec.build(cfg.seed), chosen_train, "calibration", cfg.test_start)
        cal_valid = run.decide(cal, valid)
        cm = run.metrics(cal_valid, probabilities=True)
        kept = cm["recall_at_k"] >= best["recall_at_k"] and cm["ece"] < best["ece"]
        calibration.update({
            "tried": True, "kept": kept,
            "base_recall_at_k": best["recall_at_k"], "calibrated_recall_at_k": cm["recall_at_k"],
            "calibrated_ece": cm["ece"], "base_pr_auc": best["pr_auc"],
            "calibrated_pr_auc": cm["pr_auc"],
        })  # fmt: skip
        if kept:
            spec, model, chosen_valid = cal_spec, cal, cal_valid

    # 3. Gate on validation: vs the rule, vs the champion's train-only model, and the
    #    deliberately weak calendar-only model, which must be blocked.
    rule = RuleBaseline()
    rule_valid = run.decide(rule, valid)
    references = {"rule": rule_valid}
    champion = read_champion(models_dir)
    if champion is not None:
        _check_champion_usable(champion, cfg, df)
        references["champion"] = run.decide(
            load_model(models_dir / champion.gate_model_path), valid
        )
    train_late_rate = float(chosen_train["late"].mean())  # type: ignore[arg-type]
    gate = run_gate(chosen_valid, references, cfg=cfg, train_late_rate=train_late_rate)
    weak = run.fit(calendar_only(), chosen_train, "calendar_only", cfg.test_start)
    calendar_gate = run_gate(
        run.decide(weak, valid), {"rule": rule_valid, "champion": chosen_valid},
        cfg=cfg, train_late_rate=train_late_rate,
    )  # fmt: skip

    lap("gate")

    # 4. Refit on train + validation (same window length), score the test months.
    refit_train = between(df, months_back(cfg.test_start, window), cfg.test_start)
    final = run.fit(spec.build(cfg.seed), refit_train, "refit", cfg.test_start)
    chosen_test = run.decide(final, test)
    rule_test = run.decide(rule, test)

    # 5. Walk-forward: retrain before each later test month on what was known then.
    walk = []
    for month in _months(cfg.test_start, test_end)[1:]:
        snap = prepare(load_snapshot(month).frame)
        wf_train = between(snap, months_back(month, window), month)
        wf = run.fit(spec.build(cfg.seed), wf_train, "walk_forward", month, month)
        walk.append(run.decide(wf, between(test, month, _next_month(month))))

    lap("test")

    # 6. Monthly table.
    ev_counts = dict(ev.group_by("month").len().iter_rows())
    monthly_rows = []

    def add_months(split: str, name: str, decided: pl.DataFrame, probabilities: bool) -> None:
        for (month,), part in sorted(decided.group_by("month"), key=lambda kv: str(kv[0][0])):
            m = run.metrics(part, probabilities)
            lo, hi = day_bootstrap(part, recall_at_k, n=cfg.bootstrap_resamples, seed=cfg.seed)
            monthly_rows.append({
                "split": split, "month": month, "model": name, **m,
                "recall_lo": lo, "recall_hi": hi,
                "orders_in_eval_snapshot": ev_counts.get(month, 0),
                "mean_promised_days": float(part["promised_days"].mean()),  # type: ignore[arg-type]
            })  # fmt: skip

    add_months("validation", "rule", rule_valid, False)
    add_months("validation", "chosen", chosen_valid, True)
    add_months("test", "rule", rule_test, False)
    add_months("test", "chosen", chosen_test, True)
    for part in walk:
        add_months("test", "walk_forward", part, True)
    monthly = pl.DataFrame(monthly_rows).sort(["model", "month"])

    # 7. Slices, undelivered and curves.
    slice_rows = []
    for split, frames in (
        ("validation", (rule_valid, chosen_valid)),
        ("test", (rule_test, chosen_test)),
    ):
        for name, decided in zip(("rule", "chosen"), frames, strict=True):
            for kind in SLICE_TYPES:
                for (value,), part in decided.group_by(kind):
                    slice_rows.append({
                        "split": split, "model": name, "slice_type": kind, "slice": value,
                        "orders": part.height, "late": int(part["late"].sum()),
                        "recall_at_k": recall_at_k(part),
                    })  # fmt: skip
    slices = pl.DataFrame(slice_rows).sort(["split", "slice_type", "slice", "model"])

    undelivered_rows = []
    for split, decided in (("validation", chosen_valid), ("test", chosen_test)):
        for label, part in (
            ("all", decided), ("delivered_only", decided.filter(~pl.col("late_without_delivery")))
        ):  # fmt: skip
            m = run.metrics(part, probabilities=True)
            undelivered_rows.append(
                {"split": split, "rows": label, "orders": part.height,
                 "recall_at_k": m["recall_at_k"], "pr_auc": m["pr_auc"]}
            )  # fmt: skip
    undelivered = pl.DataFrame(undelivered_rows)

    curves = {}
    for split, decided in (("validation", chosen_valid), ("test", chosen_test)):
        y, s = decided["late"].to_numpy(), decided["score"].to_numpy()
        curves[f"reliability_{split}"] = reliability_bins(y, s)
        curves[f"pr_{split}"] = pr_points(y, s)

    # 8. Leakage demonstration: the same rows, split at random instead of by time.
    pool = between(df, months_back(cfg.valid_start, window), cfg.test_start)
    rng = np.random.default_rng(cfg.seed)
    in_train = rng.random(pool.height) < 0.8
    rand = run.fit(
        spec.build(cfg.seed), pool.filter(pl.Series(in_train)), "leakage", cfg.test_start
    )
    rand_m = run.metrics(run.decide(rand, pool.filter(pl.Series(~in_train))), probabilities=True)
    chosen_m = run.metrics(chosen_valid, probabilities=True)
    # PR-AUC rises with the late rate itself, so each is also shown divided by its late rate.
    leakage = {
        "time_pr_auc": chosen_m["pr_auc"], "random_pr_auc": rand_m["pr_auc"],
        "time_recall_at_k": chosen_m["recall_at_k"], "random_recall_at_k": rand_m["recall_at_k"],
        "time_late_rate": chosen_m["late_rate"], "random_late_rate": rand_m["late_rate"],
        "time_pr_auc_lift": chosen_m["pr_auc"] / chosen_m["late_rate"],
        "random_pr_auc_lift": rand_m["pr_auc"] / rand_m["late_rate"],
    }  # fmt: skip
    lap("diagnostics")

    # 9. Keep the champion only if the gate passed.
    if gate.passed:
        run_dir = Path("runs") / run_id
        save_model(model, models_dir / run_dir / "gate_model.pkl")
        save_model(final, models_dir / run_dir / "model.pkl")
        write_champion(
            models_dir,
            Champion(
                run_id=run_id,
                model_path=str(run_dir / "model.pkl"),
                gate_model_path=str(run_dir / "gate_model.pkl"),
                snapshot_hashes={
                    "decision": decision.content_hash,
                    "evaluation": evaluation.content_hash,
                },  # fmt: skip
                valid_metrics={k: float(v) for k, v in chosen_m.items()},
                gate_train_end=chosen_train["handoff_date"].max().isoformat(),  # type: ignore[union-attr]
                features=list(FEATURE_COLUMNS),
            ),
        )

    timings["total"] = round(time.perf_counter() - started, 2)
    return ExperimentResult(
        run_id=run_id,
        git_commit=git_commit,
        snapshot_hashes={"decision": decision.content_hash, "evaluation": evaluation.content_hash},
        counts={
            "train": chosen_train.height,
            "validation": valid.height,
            "test": test.height,
            "refit": refit_train.height,
        },  # fmt: skip
        candidates=candidates,
        chosen={
            "model": spec.label(),
            "kind": spec.kind,
            "window_months": window,
            "calibrated": spec.calibrated,
            "params": model.params,
        },  # fmt: skip
        gate=gate,
        calendar_gate=calendar_gate,
        monthly=monthly,
        slices=slices,
        undelivered=undelivered,
        leakage=leakage,
        curves=curves,
        fits=pl.DataFrame(run.fit_rows),
        champion_written=gate.passed,
        calibration=calibration,
        timings=timings,
        snapshot_meta={"decision": decision.meta, "evaluation": evaluation.meta},
    )
