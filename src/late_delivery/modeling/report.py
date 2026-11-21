"""Write an M3 run's report: report.md for people, metrics.json and run.json for machines,
and calibration / precision-recall plots."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from late_delivery.config import TrainingConfig  # noqa: E402
from late_delivery.modeling.experiment import ExperimentResult  # noqa: E402
from late_delivery.modeling.gate import GateResult  # noqa: E402

_STATUS = {True: "pass", False: "**FAIL**", None: "skip"}


def _clean(value: Any) -> Any:
    """JSON-safe: NaN/inf become null, dates become ISO strings."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [_clean(r) for r in df.to_dicts()]


def _gate_json(g: GateResult) -> dict[str, Any]:
    out: dict[str, Any] = _clean({"passed": g.passed, "checks": [vars(c) for c in g.checks]})
    return out


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _monthly_table(monthly: pl.DataFrame, split: str) -> str:
    part = monthly.filter(pl.col("split") == split).sort(["month", "model"])
    rule_savings = dict(
        part.filter(pl.col("model") == "rule").select("month", "net_savings").iter_rows()
    )
    rows = []
    for r in part.iter_rows(named=True):
        vs_rule = r["net_savings"] - rule_savings[r["month"]] if r["model"] != "rule" else None
        rows.append([
            r["month"], r["model"], f"{r['late_rate']:.1%}",
            f"{_fmt(r['recall_at_k'])} [{_fmt(r['recall_lo'])}–{_fmt(r['recall_hi'])}]",
            r["precision_at_k"], r["lift_at_k"], r["pr_auc"], r["brier"],
            f"{int(r['contacts'])}/{int(r['budget'])}", f"{r['net_savings']:,.0f}",
            "–" if vs_rule is None else f"{vs_rule:+,.0f}",
        ])  # fmt: skip
    headers = ["Month", "Model", "Late rate", "Recall@K [95% CI]", "Precision@K", "Lift",
               "PR-AUC", "Brier", "Contacts / budget", "Net R$", "vs rule R$"]  # fmt: skip
    return _table(headers, rows)


def _gate_section(g: GateResult, tolerance: float) -> str:
    rows = [[_STATUS[c.passed], c.name, c.detail] for c in g.checks]
    out = [_table(["Result", "Check", "Detail"], rows)]
    borderline = [
        c.name for c in g.checks
        if c.name.startswith("slice") and c.passed and c.diff is not None and c.diff < -tolerance
    ]  # fmt: skip
    if borderline:
        out.append(
            "\n**Borderline:** " + ", ".join(borderline) + " dropped by more than the tolerance "
            "but passed because the drop is not clearly real (its interval reaches 0). "
            "Watch these slices in the next run."
        )
    return "\n".join(out)


def _budget_note(monthly: pl.DataFrame, cfg: TrainingConfig) -> str:
    """Where a month's late rate is below the budget share, part of the budget must go to
    on-time orders; say how the 0.10 minimum score dealt with that, from the numbers."""
    rows = monthly.filter(
        (pl.col("split") == "test") & (pl.col("model") == "chosen")
        & (pl.col("late_rate") < cfg.budget_share)
    ).sort("month")  # fmt: skip
    if rows.height == 0:
        return "- Every test month's late rate is at or above the budget share."
    parts = [
        f"{r['month']} ({r['late_rate']:.1%} late: {int(r['contacts'])} contacts of "
        f"{int(r['budget'])} allowed)"
        for r in rows.iter_rows(named=True)
    ]
    return (
        f"- Late rate below the {cfg.budget_share:.0%} budget in " + "; ".join(parts) + ". "
        "Even a perfect ranking must spend part of such a budget on on-time orders; the "
        f"{cfg.min_score:.2f} minimum score only holds contacts back when predicted risk "
        "is low."
    )


def _plots(result: ExperimentResult, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.6], [0, 0.6], ls="--", c="grey", lw=1, label="perfect")
    for split in ("validation", "test"):
        rel = result.curves[f"reliability_{split}"]
        ax.plot(rel["mean_pred"], rel["frac_late"], marker="o", label=split)
    ax.set(xlabel="mean predicted P(late)", ylabel="observed late share",
           title="Calibration (10 equal-count bins)")  # fmt: skip
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for split in ("validation", "test"):
        pr = result.curves[f"pr_{split}"]
        ax.plot(pr["recall"], pr["precision"], label=split)
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, chosen model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=120)
    plt.close(fig)


def write_report(result: ExperimentResult, cfg: TrainingConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = result
    metrics = {
        "chosen": r.chosen, "counts": r.counts, "gate": _gate_json(r.gate),
        "calendar_gate": _gate_json(r.calendar_gate), "candidates": _rows(r.candidates),
        "monthly": _rows(r.monthly), "slices": _rows(r.slices),
        "undelivered": _rows(r.undelivered), "leakage": r.leakage,
    }  # fmt: skip
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2, sort_keys=True))
    run = {
        "run_id": r.run_id, "git_commit": r.git_commit, "snapshot_hashes": r.snapshot_hashes,
        "config": cfg.model_dump(mode="json"), "fits": _rows(r.fits),
        "champion_written": r.champion_written,
    }  # fmt: skip
    (out_dir / "run.json").write_text(json.dumps(_clean(run), indent=2, sort_keys=True))
    _plots(r, out_dir)

    m = r.monthly
    may = m.filter((pl.col("month") == "2018-05") & (pl.col("model") == "chosen"))
    may_line = ""
    if may.height:
        row = may.row(0, named=True)
        may_line = (
            f"- May is partial in the decision snapshot: {int(row['orders']):,} of "
            f"{row['orders_in_eval_snapshot']:,} orders (mean promise "
            f"{row['mean_promised_days']:.1f} days), because orders still in transit on "
            f"{cfg.test_start} have no outcome yet."
        )
    cand = r.candidates.head(10)
    chosen = r.chosen
    status = "PASSED" if r.gate.passed else "BLOCKED"
    lk = r.leakage
    undelivered_rows = [
        [u["split"], u["rows"], u["orders"], u["recall_at_k"], u["pr_auc"]]
        for u in r.undelivered.iter_rows(named=True)
    ]
    test_slices = r.slices.filter(pl.col("split") == "test")
    slice_rows = []
    for (kind, value), part in sorted(test_slices.group_by(["slice_type", "slice"]),
                                      key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):  # fmt: skip
        by = {row["model"]: row for row in part.iter_rows(named=True)}
        slice_rows.append([kind, value, by["chosen"]["orders"], by["chosen"]["late"],
                           by["chosen"]["recall_at_k"], by["rule"]["recall_at_k"]])  # fmt: skip

    md = f"""# Training run {r.run_id}

## Run

- Code commit: `{r.git_commit}`
- Decision snapshot (train + validation, cut {cfg.test_start}): `{
        r.snapshot_hashes["decision"][:12]
    }`
- Evaluation snapshot (test labels, cut {cfg.evaluation_cutoff}): `{
        r.snapshot_hashes["evaluation"][:12]
    }`
- Rows: train {r.counts["train"]:,} · validation {r.counts["validation"]:,} · refit {
        r.counts["refit"]:,} · test {r.counts["test"]:,}
{may_line}

## Decision

Chosen: **{chosen["model"]}**, {chosen["window_months"]}-month training window
{"(Platt-calibrated)" if chosen["calibrated"] else "(no calibration)"}.

Gate: **{status}**. Validation numbers below are slightly optimistic: the same
months were used to pick the model among {r.candidates.height} candidates. The
test months were never used for any choice and are the real performance.

{_gate_section(r.gate, cfg.gate.slice_tolerance)}

Deliberately weak calendar-only model: **{
        "PASSED (gate is broken)" if r.calendar_gate.passed else "BLOCKED"
    }**
({sum(c.passed is False for c in r.calendar_gate.checks)} of {
        len(r.calendar_gate.checks)
    } checks failed).

## Validation candidates

Top {cand.height} of {r.candidates.height} (window x model x settings), by validation recall@K:

{
        _table(
            ["Model", "Window", "Recall@K", "PR-AUC", "Brier", "ECE"],
            [
                [
                    c["model"],
                    c["window_months"],
                    c["recall_at_k"],
                    c["pr_auc"],
                    c["brier"],
                    c["ece"],
                ]
                for c in cand.iter_rows(named=True)
            ],
        )
    }

## Month by month

Each month against its own late rate. `chosen` is the model refit on train + validation;
`walk_forward` is retrained before each test month on what was known then (diagnostic only).

**Test (June–August 2018, the real performance)**

{_monthly_table(m, "test")}

**Validation (April–May 2018, used for the choice)**

{_monthly_table(m, "validation")}

![calibration](reliability.png) ![precision-recall](pr_curve.png)

## Slices

Test months, chosen model vs rule:

{_table(["Slice type", "Slice", "Orders", "Late", "Recall@K chosen", "Recall@K rule"], slice_rows)}

## Never-delivered orders

Orders labeled late only because they never arrived, included vs left out:

{_table(["Split", "Rows", "Orders", "Recall@K", "PR-AUC"], undelivered_rows)}

## Leakage check

| Split | PR-AUC | Recall@K |
| --- | --- | --- |
| Time (train before, judge after) | {_fmt(lk["time_pr_auc"])} | {_fmt(lk["time_recall_at_k"])} |
| Random 80/20 of the same rows | {_fmt(lk["random_pr_auc"])} | {_fmt(lk["random_recall_at_k"])} |

A random split lets the model learn from orders handed off after the ones it is judged
on, including the same weeks' conditions, so it looks better than it can be in use;
the time split is what the system faces. Recall@K is not comparable across the two
rows: the random holdout mixes in high-late winter weeks and smaller days.

## Notes

{_budget_note(m, cfg)}
- Costs (R$2 per contact, R$20 per missed late order) are simulated assumptions.
- `handoff_month` is kept: it carries the overall late-rate level (validation PR-AUC
  0.341 with vs 0.325 without). The test months were looked at once while checking this.
"""
    (out_dir / "report.md").write_text(md)
