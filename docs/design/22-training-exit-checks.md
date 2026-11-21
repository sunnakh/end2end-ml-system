# Training and evaluation exit checks (M3)

* [ ] One command (`make train`) builds both snapshots, trains the rule, logistic
      regression and LightGBM, runs the gate and writes the report
* [ ] The NumPy logistic regression passes a gradient check and matches
      scikit-learn's coefficients and predictions within tolerance
* [ ] Every hand-written metric matches scikit-learn on random inputs
* [ ] No train or validation label comes from after 2018-06-01; test labels come
      from the evaluation snapshot
* [ ] Results are reported per month of validation and test, each against its own
      late rate, with bootstrap intervals
* [ ] Fixed split and walk-forward are shown side by side for each test month
* [ ] The chosen model beats the rule baseline through the gate, or the report
      says plainly that it does not
* [ ] A calendar-only model is blocked by the gate
* [ ] The random-split vs time-split gap is shown and explained
* [ ] Rerunning with the same snapshots, config and seed gives identical metrics
