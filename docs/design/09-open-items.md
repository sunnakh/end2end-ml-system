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
