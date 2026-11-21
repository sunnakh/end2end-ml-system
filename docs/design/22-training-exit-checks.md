# Training and evaluation exit checks (M3)

* [x] One command (`make train`) builds both snapshots, trains the rule, logistic
      regression and LightGBM, runs the gate and writes the report
* [x] The NumPy logistic regression passes a gradient check and matches
      scikit-learn's coefficients and predictions within tolerance
* [x] Every hand-written metric matches scikit-learn on random inputs
* [x] No train or validation label comes from after 2018-06-01; test labels come
      from the evaluation snapshot
* [x] Results are reported per month of validation and test, each against its own
      late rate, with bootstrap intervals
* [x] Fixed split and walk-forward are shown side by side for each test month
* [x] The chosen model beats the rule baseline through the gate, or the report
      says plainly that it does not
* [x] A calendar-only model is blocked by the gate
* [x] The random-split vs time-split gap is shown and explained
* [x] Rerunning with the same snapshots, config and seed gives identical metrics
