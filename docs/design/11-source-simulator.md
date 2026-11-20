# Source simulator

Replays the Olist dump one day at a time. For each `ds` it writes
`landing/ds=YYYY-MM-DD/` containing only rows the business could have seen by
the end of that day. Range: first to last date present in the data.

## What appears when

| Table | Emitted on | Notes |
| --- | --- | --- |
| `orders` | Each day the order changes (see CDC below) | CDC rows, not one final row |
| `order_items` | Purchase date | With the order's first row |
| `order_payments` | Purchase date | |
| `customers` | Purchase date of the order | One row per `customer_id` |
| `order_reviews` | `review_answer_timestamp` date | Kept for the business check only; never reaches silver features |
| `sellers`, `products`, `geolocation`, `product_category_name_translation` | Day 0, full load | Simplification: reference data treated as known from the start; attributes are static, so no leakage |

## Order changes as CDC rows

An order emits one row per lifecycle event, on the day it happened. Each row
has the full order columns, with fields not yet known set to null, plus
`_change_ts` and `_change_type`.

| `_change_type` | Day | Fields filled in |
| --- | --- | --- |
| `created` | `order_purchase_timestamp` | id, customer, purchase ts, estimated delivery date, status |
| `approved` | `order_approved_at` | + approved ts |
| `shipped` | `order_delivered_carrier_date` | + carrier handoff ts (the prediction moment) |
| `delivered` | `order_delivered_customer_date` | + delivery ts (the label arrives) |
| `closed` | Last known timestamp of the order | Final status for canceled / unavailable orders |

The `closed` timing is an assumption: Olist does not record when an order was
canceled. Status values on intermediate rows follow the event (`created`,
`approved`, `shipped`, `delivered`), not the dump's final status.

## Fault injection

Off by default; enabled per run in config with a seed, so injected rows are
reproducible and the quarantine check can compare exact row sets.

| Fault | Effect |
| --- | --- |
| Duplicate rows | Same CDC row emitted twice |
| Missing required value | Null `order_id` or price |
| Negative amount | Negative price or freight |
| Timestamp order violation | Approved before purchase |
| Unknown foreign key | Item for an order that never existed |
| Invalid enum | Unknown status or state code |
| Schema drift | Renamed or missing column in one file |
| Late file | A day's file delivered two days late |

The simulator writes `landing/ds=…/_injected.json` listing every injected row
key, used by tests and never read by the pipeline.
