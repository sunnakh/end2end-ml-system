# Training results on real data (M3)

Official run `20261121T090615Z-8bee679` (`make train`, commit `8bee679`, about a
minute), after the fixes from the whole-branch review. Decision snapshot
`dab4bde7db75` (cut 2018-06-01), evaluation snapshot `05fca140a8e8` (cut
2018-10-17). Business numbers are simulated (R$2 per contact, R$20 per missed late
order).

That run compared 48 candidates (12 LightGBM settings). The code was then slimmed
to 4 LightGBM settings (24 candidates) and Platt calibration was removed; a rerun
of the slimmed code on the same snapshots chose the same model and produced
identical validation, gate and month-by-month results (35 s instead of 56 s).

## Headline

**On the honest test (June–August 2018) the model and the rule baseline are tied:**
recall@K 0.231 vs 0.230, net savings R$2,078 vs R$2,024. The model wins clearly in
June and July and loses in August, after the delivery-promise cut. The gate passed
on validation, which could not see the policy change. Under the design's
anti-goals ("a model that only works under the old promise policy", "a model that
does not beat the rule being shipped"), this model should not replace the rule
in serving without the follow-up below.

## Decision

- **Chosen:** logistic regression (L2 = 1e-4), 9-month window, no calibration
  (validation ECE 0.015). Validation recall@K 0.266; the best LightGBM scored 0.265
  in the 48-candidate run (0.262 with the slimmed grid). The top candidates are
  within 0.005 of each other, inside the noise, yet behave very differently on
  August: validation cannot separate them.
- **Gate: passed.** Beats the rule by +0.080 recall@K (95% CI +0.049 to +0.109);
  every slice and calibration check passed; the North (103 orders) was too small
  to judge and is named as such.
- **Calendar-only model: blocked.** A second identical run was **blocked** on
  "beats champion" (+0.000), and the champion file was left unchanged.

## Test months (never used for any choice)

| Month | Late rate | Recall@K model [95% CI] | Recall@K rule | Walk-forward | Net R$ vs rule |
| --- | --- | --- | --- | --- | --- |
| Jun 2018 | 2.1% | 0.364 [0.292–0.438] | 0.202 | – | +454 |
| Jul 2018 | 3.4% | 0.259 [0.174–0.341] | 0.195 | 0.249 | +260 |
| Aug 2018 | 7.3% | **0.186** [0.140–0.249] | **0.250** | 0.188 | **−660** |

- Promises were ~28 days in June and ~15 in August; the model, fitted under ~23-day
  promises, ranks well when promises stretch and badly when they shrink.
- Monthly retraining does not help in August (0.188): the model retrained up to
  July has seen too little of the new policy.
- By slice, the model loses in the Southeast (0.191 vs 0.233), the largest region,
  and the South; it wins strongly in the Northeast (0.438 vs 0.188).
- Validation numbers (Apr 0.291, May 0.211) are slightly optimistic: the same
  months picked the model.

## Other findings

- **Leakage check:** the random split's PR-AUC is higher (0.401 vs 0.341), but its
  holdout is 9.7% late against 6.9%. Divided by the late rate, the random split is
  *not* better (4.12 vs 4.96). Earlier "27% better" was mostly base rate. Without
  future seller statistics (features are point-in-time), random splitting leaks
  little here; the real risk the time split exposes is the policy change.
- **Never-delivered orders:** test recall@K 0.231 with them, 0.251 without.
- **LightGBM with March in the trees:** refitting on every row put March 2018 (22%
  late) into the 9-month models' trees, and their validation calibration error
  rose to ~0.096.
- **Label maturity:** the decision snapshot left out 3,837 orders not yet due; May
  validation holds 3,278 of 6,981 orders (short promises), as decided.

## August check (post-hoc)

After the results above, August was examined to find why the model lost there
(details in [Open items](09-open-items.md)). Everything in this section is
**post-hoc**: June–August were used to find the cause, so they are no longer a
clean test, and the data has no later months to retest on.

- **Cause:** lateness is a step at under 5 days left at handoff. The logistic
  regression drew it as a straight line, and in August spent most of its budget
  on orders with 5–20 days left (265 contacts, 33 late) while contacting only 69
  of the 1,331 orders under 5 days left (63 of them late).
- **Fix:** a 0/1 "under 5 days left" flag for logistic regression.

| Logistic regression | Validation recall@K | Validation PR-AUC | Jun | Jul | Aug |
| --- | --- | --- | --- | --- | --- |
| Without the flag | 0.266 | 0.341 | 0.364 | 0.259 | 0.186 |
| With the flag | 0.258 | 0.355 | 0.364 | 0.263 | **0.338** |

The flag costs nothing on validation beyond noise and fixes the August failure
(rule: 0.250).

**Rerunning the normal experiment with the flag** (all choices still on
validation) picks LightGBM (15 leaves, learning rate 0.03, 15-month window,
validation recall@K 0.262). It beats the rule by +0.076, but the gate blocks it
(see below). Post-hoc test:

| Month | Model | Rule | Walk-forward |
| --- | --- | --- | --- |
| Jun 2018 | 0.341 | 0.202 | – |
| Jul 2018 | 0.293 | 0.195 | 0.302 |
| Aug 2018 | 0.394 | 0.250 | 0.394 |
| **Jun–Aug** | **0.362** (R$4,470) | 0.230 (R$2,024) | |

The choice among the top candidates is still fragile: they sit within noise on
validation, and the flag's small dip in logistic regression's recall flipped the
winner to LightGBM, which handles the step natively. Validation still cannot
show how a candidate behaves under the new promise policy.

**Official run with the flag:** `20261121T110611Z-f0e249b` (`make train`,
commit `f0e249b`, same snapshots) reproduces the numbers above. Gate:
**blocked** on "beats champion": recall@K 0.262 vs the champion's 0.266, a
difference of −0.004 (95% CI −0.026 to +0.021) where +0.020 with the interval
above 0 is required. Every other check passed (rule +0.076, all slices,
calibration), and the calendar-only model was blocked as well.

The champion therefore stays the logistic regression without the flag, the
model that loses August to the rule (0.186 vs 0.250). The gate followed its
rules: it judges on April–May only, and those months cannot show the policy
change. The gate is not loosened after seeing this result; promotion is
handled as a recorded decision (see [Open items](09-open-items.md)).

## Fixed during the whole-branch review

- LightGBM read its feature list from the current code, so the stored champion
  would have broken the next time a feature was added; it now keeps its own list.
- LightGBM's "refit on train + validation" never fitted trees on its newest month;
  the final trees now use every row. This changed which model won (LightGBM before,
  logistic regression now) and the test results above.
- The champion file now records its training end and features; a champion that
  trained inside the current validation months, or needs missing columns, is refused.
- The report no longer prints hardcoded findings or dates; it shows ROC-AUC,
  calibration trials, label-maturity counts and unjudged slices; `run.json` has
  timings.
- Earlier in Task 10: drift look-ahead (reports read the reference period before
  it had happened) fixed; all 774 days re-backfilled, reruns byte-identical.

## Known limitations

- No lock on the champion file: two runs started together can both write it.
- The gate's validation margin is slightly optimistic (see
  [Evaluation gate](21-evaluation-gate.md)), and validation cannot detect the
  August policy change. Follow-up: [Open items](09-open-items.md).
