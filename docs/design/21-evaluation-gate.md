# Evaluation gate (M3)

A challenger becomes the champion only if all of these hold on **validation**:

1. **Beats the rule baseline** on recall@K by at least 0.02 (absolute), and the
   lower end of the paired bootstrap 95% interval of the difference is above 0.
2. **Beats the current champion** by the same rule. On the first run there is no
   champion, so only rule 1 applies.
3. **No major slice much worse:** for every slice with at least 300 validation
   orders, the challenger fails if its recall@K is more than 0.05 below the
   reference (champion, or the rule on the first run) **and** the paired-bootstrap
   95% interval of that drop is entirely below 0. A slice with a few dozen late
   orders moves several points on a handful of orders, so a drop must be both
   large and clearly real. Slices with no late orders are skipped.
4. **Calibration sane:** Brier score no worse than predicting the training late
   rate for everyone.

The test months never enter the gate; they are reported after the decision.
The gate judges the model fitted on train only; if it passes, the model refit on
train + validation (same settings) is the one recorded as champion.

## Champion

The champion is recorded in `data/models/champion.json` (run id, model path,
snapshot hashes, validation metrics). A blocked challenger leaves it unchanged;
the report states which rule failed and by how much.

## Deliberately worse model

An exit check: a model trained on calendar features only (`handoff_weekday`,
`handoff_month`) is run through the gate against the champion and must be
blocked, with the failed rule named in its report.

## Known limitation: the gate's margin is slightly optimistic

The model is picked as the best of many candidates on the validation months, and
the gate then judges it on those same months, so part of its lead is luck on
those weeks. The required margin and interval limit the effect; the candidates
are close to each other, so it is small. Test months were never used for any
choice: the report headlines them as the real performance and labels the
validation gate numbers as optimistic.
