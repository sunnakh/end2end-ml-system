# Training and evaluation overview (M3)

Turn the point-in-time training data from M2 into a model that decides which
orders get a proactive contact, and show honestly whether it beats the rule
baseline under the delivery-policy change of mid-2018.

## Goals

- A rule baseline, a logistic regression written in NumPy, and a LightGBM model,
  all trained and scored on the same splits and features.
- Results reported **month by month** across the test window, each month
  against its own late rate (see [Open items](09-open-items.md)).
- An evaluation gate that blocks a deliberately worse model.
- An evaluation report (Markdown + JSON + plots) written for every run.
- The random-split vs time-split gap shown and explained.

## Non-goals

- Experiment tracking and a model registry (M4). M3 writes models and reports
  to local files.
- Serving, batch contact lists, monitoring (M5).
- New feature engineering beyond the one feature the rule baseline needs
  (see [Data and splits](18-training-data-and-splits.md)).

## Anti-goals (failure even if the numbers look good)

- Any choice (model, window, hyperparameters, threshold) tuned on the test months.
- A model that only works under the old promise policy, hidden by a good overall
  number.
- Shipping a model that does not beat the rule baseline.

## Constraints

- Laptop scale: about 95k labeled orders, each model trains in seconds.
- Every number is reproducible from a snapshot hash, a config and a commit.
- New dependencies: `numpy`, `lightgbm`, `matplotlib`; `scikit-learn` only in the
  dev group, as the reference the hand-written code is tested against.
- Business costs are simulated (see [Business assumptions](03-business-assumptions.md)).

## Approaches considered

| Approach | What it is | Trade-off |
| --- | --- | --- |
| A. Fixed split | Train once on 2017-01 → 2018-03, choose on 2018-04 → 05, test once on 2018-06 → 08 per month | Simple and clean, but a model frozen in March meets August's halved promises cold; it understates what weekly retraining would achieve. |
| B. Walk-forward only | Retrain before each test month on everything before it | Matches weekly retraining, but more models, a harder gate, and a pull toward tuning on test months. |
| **C. Fixed split + walk-forward diagnostic (chosen)** | A makes every decision; a walk-forward run over the test months is reported afterwards and never feeds back | Clean decisions plus an answer to "does retraining fix the drift?" at little extra cost. |

## Sections

- [Data and splits](18-training-data-and-splits.md)
- [Models](19-models.md)
- [Metrics and report](20-metrics-and-report.md)
- [Evaluation gate](21-evaluation-gate.md)
- [Exit checks](22-training-exit-checks.md)
