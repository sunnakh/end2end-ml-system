# Data and splits (M3)

## Two snapshots: what was known vs what happened

A model trained on 2018-06-01 could only use labels known by then. Its test
months are judged later, once they have resolved. So M3 uses two snapshots
built by the existing `build_training_set` (label maturity applies to both):

| Snapshot | Cut-off | Used for |
| --- | --- | --- |
| Decision snapshot | 2018-06-01 (start of test) | train and validation rows and labels |
| Evaluation snapshot | 2018-10-17 (end of replay) | test rows and labels |

Why not one snapshot: the 2018-09-01 snapshot drops August orders whose promise
runs past September (2,001 orders), leaving mostly short-promise August orders;
and its labels for May orders include outcomes not yet known on 2018-06-01.

## Fixed split (decides everything)

By handoff date (`order_delivered_carrier_date`):

| Split | Months | Rows (2018-09-01 snapshot) | Late rate |
| --- | --- | --- | --- |
| Train | 2017-01 → 2018-03 | ~64k | 9.0% |
| Validation | 2018-04 → 2018-05 | ~14k | 7.3% |
| Test | 2018-06 → 2018-08 | ~17k | 2.1% / 3.4% / 10.1% by month |

**May is partial in the decision snapshot.** May orders whose promise runs into
June have no outcome on 2018-06-01 and are left out (about 3,300 of 7,000 kept,
mostly short promises: ~17 vs ~23 days). This is kept on purpose: it is what a
team deciding on June 1 would have. The report shows May's coverage against the
evaluation snapshot and its mean promised days, so the skew is visible. April is
complete and carries most of the decision.

Orders before 2017-01 (about 270) are dropped. Exact counts come from the two
snapshots above and are written into every report.

**Training window** is a choice made on validation, from three candidates:
full (2017-01 → 2018-03), last 9 months (2017-07 → 2018-03) and last 6 months
(2017-10 → 2018-03).

After all choices are fixed on validation, the chosen model is **refit on train +
validation** (same window length, ending 2018-05) and scored once on test.

## Walk-forward diagnostic (decides nothing)

For each test month M (June, July, August): train the chosen model type with the
chosen settings on a snapshot cut at the first day of M, on the chosen window
ending the month before M, and score month M from the evaluation snapshot.
Reported next to the fixed-split result for the same month. Never used to pick
anything. June's walk-forward model is the same as the refit fixed-split model
(both trained up to May), so the comparison is informative for July and August.

## Slices

- **Customer region:** the five Brazilian regions (North, Northeast, Center-West,
  Southeast, South) mapped from `customer_state`.
- **Seller size:** buckets of `seller_prior_orders`: 0, 1–9, 10–49, 50+.
- **Undelivered:** headline metrics with and without orders labeled late only
  because they never arrived.

## New feature for the rule baseline

The rule needs each seller's historical median handling time, which the snapshot
does not carry. Add `seller_prior_median_handling_h` to `build_features`: the
median `handling_time_h` over the seller's handoffs **strictly before** this
order's handoff (null if none; multi-seller orders take the max across sellers,
like the other seller features). Handoff times are known at handoff, so no
outcome leaks. Models may use it too. This changes `FEATURE_COLUMNS` and the
snapshot hash; both snapshots are rebuilt.
