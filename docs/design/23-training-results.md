# Training results on real data (M3)

Official run `20261121T070731Z-a8dff84` (`make train`, commit `a8dff84`, 62 s).
Decision snapshot `dab4bde7db75` (cut 2018-06-01), evaluation snapshot
`05fca140a8e8` (cut 2018-10-17). Business numbers are simulated (R$2 per contact,
R$20 per missed late order).

## Decision

- **Chosen:** LightGBM (15 leaves, learning rate 0.03, min child samples 100),
  9-month training window, no calibration. Best of 48 candidates on validation
  recall@K (0.269); logistic regression was a close second (0.266).
- **Gate: passed.** Beats the rule by +0.083 recall@K (95% CI +0.054 to +0.111);
  every slice and calibration check passed. **Center-West is borderline:** −0.094
  vs the rule on 32 late orders, with an interval reaching 0.
- **Calendar-only model: blocked** (10 of 11 checks failed).
- **A second identical run was blocked** on "beats champion" (+0.000); the champion
  file was left unchanged.

## Test months (never used for any choice)

| Month | Late rate | Recall@K model [95% CI] | Recall@K rule | Walk-forward | Net R$ vs rule |
| --- | --- | --- | --- | --- | --- |
| Jun 2018 | 2.1% | 0.372 [0.279–0.454] | 0.202 | – | +444 |
| Jul 2018 | 3.4% | 0.268 [0.182–0.350] | 0.195 | 0.278 | +302 |
| Aug 2018 | 7.3% | 0.386 [0.329–0.465] | 0.250 | 0.400 | +1,400 |

- The model beats the rule in every test month, through the 2018 promise-policy
  change (promises ~28 days in June, ~15 in August). The rule loses money in June.
- Monthly retraining adds about +0.01: the model trained to May holds up.
- Validation numbers (Apr 0.272, May 0.263) are slightly optimistic: the same
  months picked the model.

## Findings

- **Leakage:** a random split shows PR-AUC 0.434 vs 0.343 for the time split, about
  27% better than the system can really do.
- **Never-delivered orders** are the hardest: test recall@K 0.356 with them,
  0.412 without.
- **Calibration:** the model over-predicts its riskiest orders (top validation bin
  ~0.40 predicted vs ~0.29 observed), but its Brier score beats a constant guess.
- **Budget in low-late months:** in June and July (2–3% late) the model used 297 of
  299 and 293 of 294 contacts. With fewer late orders than contact slots, part of the
  budget necessarily goes to on-time orders; the 0.10 minimum score held back only a
  few.
- **The month feature** carries the late-rate level (validation PR-AUC 0.341 with,
  0.325 without). The test months were looked at once while checking it.
- **May validation is partial** (3,278 of 6,981 orders, short promises), as decided.

## Fixed along the way

- **Drift look-ahead (from M2):** drift reports read the reference period before
  it had happened, so a day's report depended on which later days existed on disk.
  Now `no_reference` until the reference period is over; reports for days up to
  2017-08-31 changed accordingly. A feature missing from the reference gets no PSI
  instead of failing the report.
- **Re-backfill:** all 774 days rebuilt with the new seller feature; rerunning two
  days through every step reproduced all 50 files byte for byte, and two training
  runs produced byte-identical metrics.

## Known limitations

- **No lock on the champion file:** two training runs started at the same moment
  both saw no champion and both wrote one (identical models this time). The M4
  model registry should serialise promotions.
- The gate's validation margin is slightly optimistic (see
  [Evaluation gate](21-evaluation-gate.md)).
