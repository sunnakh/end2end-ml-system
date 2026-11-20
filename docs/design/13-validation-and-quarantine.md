# Validation and quarantine

Validation runs between bronze and silver, per table and per `ds`.

## Two levels

| Level | Examples | Action |
| --- | --- | --- |
| **File level** | Missing or renamed column, unreadable file, empty file when rows were expected | Fail the step for that partition; nothing written to silver; alert |
| **Row level** | Bad values in otherwise valid files | Move the row to quarantine, keep the rest |

## Reason codes

| Code | Rule |
| --- | --- |
| `TYPE_PARSE_ERROR` | Value cannot be cast to its declared type (including `nan`, `inf` and integers beyond the 64-bit range) |
| `NULL_REQUIRED` | Required field is null (including a `delivered` event with no delivery date) |
| `DUPLICATE_KEY` | Key already seen in this partition (first copy kept) |
| `NEGATIVE_AMOUNT` | Price, freight or payment value below zero |
| `TS_ORDER_VIOLATION` | Lifecycle out of order (approved < purchase, delivered < handoff). Handoff may precede purchase by up to **2 hours**: 160 of 166 such real orders fall within it, a clock or time-zone difference between the website and carrier systems; only larger gaps are quarantined |
| `ORPHAN_FK` | Item, payment or review for an unknown order |
| `INVALID_ENUM` | Status or state code outside the allowed set |
| `OUT_OF_RANGE` | Lat/lng outside Brazil, impossible weight or size |

A row with several problems is recorded once, with all its codes.

## Quarantine format

`quarantine/<table>/ds=…`: the original bronze row, plus `reason_codes`,
`reason_detail`; the run that produced it is recorded in the manifest. Quarantined rows are never silently fixed; a fix
is a code or contract change followed by a rerun of the partition.

## Circuit breaker

If more than 5% of a table's rows for one `ds` are quarantined, the run fails
instead of continuing with a thin partition. The threshold lives in config;
it applies only to table-days with at least 100 rows, so sparse early days are
not stopped by a single bad row. Every table is checked before anything is
written, so a failed day leaves the previous silver and quarantine untouched.

## Real-data issues

Olist contains genuine anomalies (for example lifecycle timestamps out of
order, products without category or dimensions, geolocation points outside
Brazil). EDA counts them; each is either quarantined by a rule above or
handled explicitly (e.g. null category → `unknown`) and documented in the data
contract.

## Drift report

Each day a report compares the scoring features against a reference period
(June–August 2017) using PSI, and records row counts and quarantine rates.
M2 writes the report; alerting on it comes in M5.

- **Trailing 7-day window, not one day.** A single day (~150 orders) is too
  small: comparing the reference period with one of its own days gave PSI
  0.19, close to the "moderate drift" line, although nothing had changed. A
  week of the same period gave 0.03.
- **Calendar features are excluded.** One day's weekday and month always
  differ from a multi-month reference.
- **Minimum sample.** Below 200 orders in the window the status is
  `too_few_orders` and no PSI is reported (most of 2016).
- **Reading PSI:** < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.
