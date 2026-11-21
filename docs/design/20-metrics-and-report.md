# Metrics and report (M3)

## The daily contact decision

Each handoff day: K = 5% of that day's handoffs, rounded down, at least 1.
Contact the top K orders by score, but only those with score > 0.10 (break-even
from the R$2 / R$20 cost ratio). A day may use fewer than K contacts.

In June–July 2018 the late rate (2–3%) is below the 5% budget, so the threshold,
not the budget, decides how many are contacted. This is expected behaviour and is
reported as contacts used vs budget per month.

## Metrics (written from scratch, checked against scikit-learn in tests)

| Metric | Definition | Role |
| --- | --- | --- |
| Recall@K | late orders contacted ÷ all late orders, summed over the days of the period | primary, used by the gate |
| PR-AUC | average precision over all orders of the period | primary |
| Precision@K | late orders contacted ÷ contacts sent | guardrail |
| ROC-AUC | rank-based (Mann–Whitney) | guardrail |
| Brier score, reliability curve | 10 equal-count bins | calibration guardrail |
| Net savings (simulated) | R$20 × late orders contacted − R$2 × contacts | business, shown against the rule |

**Confidence intervals:** bootstrap over handoff **days** (whole days resampled,
since the budget is per day), 1,000 resamples, 95% percentile intervals. Model
comparisons use the paired difference on the same resampled days.

## How results are shown

- **Per month** of validation and test, each with its own late rate, so a low
  precision@K in June (2% late) is not read as a failure.
- **Lift over the month's base rate** for precision@K (precision@K ÷ late rate),
  comparable across months.
- **Fixed split vs walk-forward** side by side per test month.
- **Training windows** (full / 9 / 6 months) compared on validation.
- **With and without** orders labeled late only because they never arrived.
- **Slices:** region and seller size (see [Data and splits](18-training-data-and-splits.md)).
- **Leakage demonstration:** the chosen model type retrained on a random 80/20
  split of the same rows (train + validation period), PR-AUC and recall@K
  compared with the time split, and the gap explained in the report.

## The report

Written for every run to `data/models/runs/<run_id>/`:

- `report.md`: what was trained, on which snapshots, the tables above, the gate
  decision and its reasons.
- `metrics.json`: every number in the report, machine-readable.
- `reliability.png`, `pr_curve.png`: calibration and PR curves on validation and test.
- `run.json`: snapshot hashes, config, git commit, seed, timings.
