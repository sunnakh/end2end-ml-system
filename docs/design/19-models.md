# Models (M3)

All three models expose the same interface: `fit(train_df)` and
`score(df) -> P(late)-like score`, so the evaluation code treats them alike.

## Rule baseline (non-ML)

From the [framing](01-framing.md):

- **Flag** an order if `handling_time_h > seller_prior_median_handling_h`, or the
  order ships between states (`same_state_all` is false).
- **Rank** flagged orders by handling-time excess
  (`handling_time_h − seller_prior_median_handling_h`; 0 when the seller has no
  history). Ties broken by `order_id` so the ranking is deterministic.
- Unflagged orders get the lowest score. The rule has no probability, so the
  0.10 minimum score does not apply to it; it fills the daily budget from
  flagged orders only.

## Logistic regression in NumPy (from scratch)

- L2-regularised log loss, fitted with Newton's method (IRLS): about 50
  coefficients after one-hot encoding, so each step is cheap and converges in
  under ~20 iterations.
- Preprocessing learned on the training rows only: numeric features
  median-imputed with a missing-indicator where nulls occur (seller history),
  heavy-tailed ones log-transformed (price, freight, weight, volume, distance),
  then standardised; categorical features one-hot with a rare-level bucket.
- Checks: a numerical gradient check, and coefficients and predictions matching
  scikit-learn's `LogisticRegression` on the same design matrix within a small
  tolerance.
- Tuned: the L2 strength, from a small grid, on validation.

## LightGBM

- Native categorical handling, early stopping on a time-ordered tail of the
  training rows (the last month of the window), never on validation.
- Tuned on validation from a small grid: number of leaves, learning rate,
  minimum child samples.
- Class weights off: the decision needs calibrated probabilities for the 0.10
  threshold.

## Calibration

The 0.10 threshold assumes calibrated scores. Brier score and a reliability
curve are reported for each model on validation and test. If a model is clearly
miscalibrated on validation, Platt scaling is fitted on the last month of the
training window (not on validation) and both versions are reported.

## Chosen model

The model type, training window and settings with the best validation
recall@K, provided it passes the [gate](21-evaluation-gate.md).
