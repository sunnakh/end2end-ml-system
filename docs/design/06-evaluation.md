# Evaluation design


- **Time-based split by handoff date** (dates final after EDA):
  train 2017-01 → 2018-03, validation 2018-04 → 2018-05, test 2018-06 → 2018-08.
  2016 data is too sparse to use.
- **Label maturity:** near the end of the data, orders may not have resolved
  yet. Keep only orders whose estimated delivery date is before the data
  cut-off, so every evaluated order has a known outcome. Without this rule a
  not-yet-due order enters only if it already arrived (on time), never while
  it is still travelling: on the real data the last weeks before a 2018-09-01
  cut-off showed 5–8% late instead of the true 11–17%. Each snapshot records
  how many orders were set aside (`excluded_not_yet_due`).
- **Untrusted outcomes:** an order with any quarantined lifecycle event (e.g.
  delivered with no delivery date) is left out of training; its label cannot
  be trusted.
- **Undelivered orders:** labels that rest on non-delivery (overdue, never
  delivered) are reported per snapshot; headline metrics are shown with and
  without them, so a data gap in delivery records cannot drive conclusions.
- **Leakage demonstration:** also train on a random split and report the metric
  gap against the time split, with an explanation.
- **Known risk:** a nationwide truckers' strike in late May 2018 may cause a
  late-delivery spike inside the validation window. Check in EDA; if
  confirmed, report its slice separately rather than tuning to it.
- **Gate:** a challenger is registered only if it beats both the rule baseline
  and the current champion on validation recall@K by a set margin, with no
  major slice (customer region, seller size) much worse.
