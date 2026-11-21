# Data pipeline results (real data)

Full replay of the Olist dump through the M2 pipeline, run with
`make pipeline` on a laptop (Apple Silicon), private PII salt, faults off.

## Run

| | |
| --- | --- |
| Days replayed | 774 (2016-09-04 → 2018-10-17) |
| Failed days | 0 |
| Wall time | 395 s (~0.5 s per day) |
| Peak memory | 1.7 GB (day 0 loads 1M geolocation rows) |
| Lake size | landing 188 MB · bronze 87 MB · silver 66 MB · gold 23 MB |

## Validation

| Table | Rows in | Valid | Quarantined | Exact duplicates removed |
| --- | --- | --- | --- | --- |
| orders (lifecycle events) | 394,098 | 394,043 | 55 | 0 |
| order_items | 112,650 | 112,644 | 6 | 0 |
| order_payments | 103,886 | 103,880 | 6 | 0 |
| order_reviews | 99,214 | 99,148 | 66 | 0 |
| customers | 99,441 | 99,441 | 0 | 0 |
| geolocation | 1,000,163 | 738,299 | 33 | 261,831 |
| products · sellers · categories | 36,117 | 36,117 | 0 | 0 |

Why rows were quarantined:

- **Orders, 37 orders (0.04%)**: 23 delivered before the carrier received
  them; 8 marked delivered with no delivery date; 6 handed to the carrier more
  than 2 hours before purchase. 160 more orders with handoff up to 2 hours
  before purchase are kept: a clock or time-zone difference between systems
  (see [Validation](13-validation-and-quarantine.md)).
- **Items, payments, reviews**: rows whose order is unknown — mostly belonging
  to quarantined orders; some reviews were answered before the order existed.
- **Geolocation**: 33 points outside Brazil. A quarter of the table was exact
  duplicates; it collapses to 19,010 zip-prefix points.

## Gold

- Daily scoring sets: 97,629 handed-off orders with 19 point-in-time features.
- Training snapshot, cut-off 2018-09-01: **95,543 rows, late rate 8.0%**;
  2,001 not-yet-due orders set aside; 1,160 labels rest on non-delivery
  (overdue, never delivered).

## Drift

- 585 days with a full report; 189 days (mostly 2016) below the 200-order
  minimum for the 7-day window.
- Reference months (Jun–Aug 2017) show ~0 drifting features; June–August 2018
  shows 4–5, driven by a change in delivery promises.

## What the real run changed

Running every step on real data, not only on test fixtures, found problems
the tests could not: a too-strict timestamp rule (would have dropped 160 valid
orders), seller history biased low (late orders arrive last; 4.4% → 6.4%),
daily drift reports raising false alarms (fixed with a 7-day window), and a
training set that kept early arrivals near the cut-off (late rate in the final
weeks 5–8% → 11–17%). Each fix has a test.

## Airflow

The generated DAG (`pipelines/airflow_dag.py`) was run in Airflow 3.4 with
`airflow dags test` for 2018-08-01 to 2018-08-03: all 18 tasks succeeded, and
all 73 data partitions for those days were byte-for-byte identical (same
content hashes) to the ones the CLI produced. The DAG stops a day's tasks when
the previous day failed (`depends_on_past`), ends at the replay end date, and
runs from `$LATE_DELIVERY_PROJECT_DIR`, so the file holds no machine-specific
paths.

## Carried into M3

- The delivery-promise policy changed inside the planned test window, and the
  late rate swings from 2% to 22% by month: see
  [Open items](09-open-items.md).
- Orders handed off after their promised date are certainly late; report
  metrics with and without them.
- Report metrics with and without the 1,160 non-delivery labels.
