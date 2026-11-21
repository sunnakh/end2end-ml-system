# Evaluation gate (M3)

A challenger becomes the champion only if all of these hold on **validation**:

1. **Beats the rule baseline** on recall@K by at least 0.02 (absolute), and the
   lower end of the paired bootstrap 95% interval of the difference is above 0.
2. **Beats the current champion** by the same rule. On the first run there is no
   champion, so only rule 1 applies.
3. **No major slice much worse:** for every slice with at least 300 validation
   orders, the challenger's recall@K is no more than 0.05 below the reference it
   is compared with (champion, or the rule on the first run).
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
