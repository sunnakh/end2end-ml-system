# Open items (resolved during EDA, Phase 1)


- Exact late rate, date coverage and final split dates.
- Whether the May 2018 strike shows up in the data, and how much.
- Date-level vs timestamp-level label comparison (check how often they differ).
- Confirmation that late orders have markedly worse review scores.

## Must handle in M3: the business changed during the test window

Found by the drift report on the full replay (Phase 1 pipeline run).

The planned test window (June–August 2018) is when Olist's delivery operations
changed most:

| Period | Avg promised days | Median seller handling | Drifting features (PSI > 0.25) |
| --- | --- | --- | --- |
| Jun–Aug 2017 (reference) | ~23 | ~46 h | ~0 |
| Dec 2017–Jan 2018 | ~26–27 | ~53 h | ~3 |
| Jun 2018 | 28.4 | 27 h | ~5 |
| Aug 2018 | 15.5 | 31 h | ~5 |

In June–August 2018 these features drift on most days: seller late rate and
total freight (every day), promised and remaining days (~85%), seller order
volume and approval delay (~45%). Promises were stretched in June (likely after
the late-May truckers' strike) and almost halved by August; sellers got much
faster.

A model trained on 2017 will meet promise lengths it has never seen. M3 must:

- report results **month by month** across the test window, not one number;
- compare a training window ending close to the test period (e.g. the last
  6–9 months) with the full history;
- treat a model that only works under the old promise policy as a failure,
  even if its overall metric looks good.

The base rate moves too. Late share by handoff month in the 2018-09-01
snapshot: 4–5% in mid-2017, 11–13% in Nov 2017–Feb 2018, **22% in March 2018**,
7% in April–May, **2% in June 2018** (promises stretched), 3% in July and 10%
in August (promises cut). Metrics such as precision@K depend on the base rate,
so M3 compares months against their own base rate, not one global number.

## Follow-up after M3: why the model fails in August 2018 (post-hoc)

The M3 model beats the rule in June–July 2018 but loses in August (recall@K
0.186 vs 0.250), when promises were cut to ~15 days; monthly retraining does not
fix it, and validation (April–May, old policy) could not tell the candidates
apart. Any change chosen by looking at June–August makes them no longer a clean
test, and the data has no later months, so the result is labelled **post-hoc**
and kept separate from the M3 numbers.

**Done (post-hoc).** Cause: lateness is a step in days left at handoff, not a
slope. Orders with under 5 days left are ~66% late in training and ~26% in
August; every longer group is ~10% in training and ≤5% in August. Logistic
regression models days left as a straight line, so it barely used it, and state
and month dominated. Under the old policy few orders were near the step (10th
percentile ~10 days left), so validation could not show the gap; August's short
promises put 19% of orders, and 68% of its late orders, under 5 days. Trees draw
steps natively, which is why LightGBM did well in August.

Fix: a 0/1 "under 5 days left" flag next to `remaining_days` for logistic
regression (threshold read from training alone). Results in
[Results on real data](23-training-results.md#august-check-post-hoc).

## Open in M4: promoting the August fix

The official run with the flag (`20261121T110611Z-f0e249b`) was blocked on
"beats champion": on April–May it ties the current champion (−0.004), so the
champion stays the model that loses August to the rule. Validation cannot see
the policy change, so the gate cannot settle this.

Decision: the gate is not loosened and `champion.json` is not edited by hand.
M4 adds a manual promotion, built with the model registry:

- `promote <run_id> --reason "…"`: refused unless the run exists, its model
  loads, its snapshots match, and it passed every gate check except "beats
  champion"; a reason is required.
- The champion record keeps `promoted_by: manual`, the reason, the previous
  champion and the time; every change of champion, automatic or manual, is
  appended to a promotions log.
- Candidate reason for this run: the post-hoc August check (0.394 vs 0.186),
  labelled post-hoc.
