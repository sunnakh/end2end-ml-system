# Layers and tables

## Bronze — raw and append-only

- One table per landing file, partitioned by `ds`.
- Every column stored as string; nothing is dropped or fixed.
- Lineage columns: `_ds`, `_source_file`, `_row_number`. The run ID is kept
  in the manifest, not in the data, since it differs on every rerun.
- No wall-clock timestamps in the data (they break identical reruns); run
  times live only in the manifest.
- Review text exists only here and is never read downstream.

## Silver — typed, valid, redacted

| Table | Key | Contents |
| --- | --- | --- |
| `orders_changelog` | `order_id`, `_change_type` | Valid typed lifecycle events (one per stage per order), deduplicated, partitioned by `ds` |
| `orders_current` | `order_id` | Latest state per order as of the last processed `ds` (merge of changelog) |
| `order_items` | `order_id`, `order_item_id` | Typed items |
| `order_payments` | `order_id`, `payment_sequential` | Typed payments |
| `customers` | `customer_id` | Hashed IDs, zip prefix, city, state |
| `sellers` | `seller_id` | Zip prefix, city, state |
| `products` | `product_id` | Category (English), weight, dimensions |
| `geo_prefix` | `zip_code_prefix` | One point per prefix: median lat/lng of in-Brazil rows |
| `reviews_outcome` | `order_id` | Review score and answer time, for the business check; the latest answered review when an order has several |

Point-in-time questions ("what did we know at time t?") are answered from
`orders_changelog`, never from `orders_current`.

## PII handling

- `customer_id` and `customer_unique_id` replaced with salted SHA-256.
- The salt comes from an environment variable or secret, never the repo
  (locally: `.env`, gitignored; `.env.example` documents the variable).
- The lake records a one-way fingerprint of its salt (`_salt_fingerprint`).
  A run with a different salt is refused: mixing salts would silently break
  customer joins, so a new salt means rebuilding the lake from day 0.
- Review title and message dropped at silver.
- Zip prefix (5 digits), city and state kept: coarse location is needed for
  distance features.

## Gold — model-ready

| Table | Rows | Used by |
| --- | --- | --- |
| `gold/scoring/ds=…` | Orders handed off on `ds`, features as of their handoff | Daily batch scoring |
| `gold/training/snapshot=…` | Orders handed off up to a cut-off, features as of handoff, label where resolved | Training (weekly) |

Both are built by the same feature code in `features.py`, which also backs the
online API, so training and serving cannot compute features differently.
Feature rules follow [Feature rules](05-feature-rules.md); label maturity
follows [Evaluation design](06-evaluation.md).
