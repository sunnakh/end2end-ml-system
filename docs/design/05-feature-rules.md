# Feature rules (point-in-time correctness)


A feature is allowed only if its value is known at the order's handoff time.

| Allowed (examples) | Forbidden |
| --- | --- |
| Promised days (estimated − purchase), remaining days (estimated − handoff) | `order_delivered_customer_date` (it defines the label) |
| Approval delay, seller handling time (handoff − approval) | `order_reviews` (score, text, dates) |
| Items, sellers, price, freight, product weight and size, category | Final `order_status` |
| Customer and seller state, same-state flag, seller–customer distance | Any aggregate computed over the full dataset |
| Payment type and installments | |
| Calendar: weekday, month, days to known peaks (e.g. Black Friday) | |
| Seller history: late rate and volume from **earlier orders whose outcome was known before this handoff** | |

The seller-history rule is the main leakage trap, and it uses the label's own
definition of "late". An earlier order counts once its outcome is known before
the current order's handoff:

- delivered on or before its promised day → known **on time** at delivery;
- promised day over with no delivery → known **late** from the next day,
  whether or not it arrives afterwards;
- canceled before its outcome was known → not counted.

Counting only orders already delivered would bias the rate low, because late
orders are, by definition, the last to arrive. On the real data this moved the
average seller late rate from 4.4% to 6.4% (actual late share: 7.8%).
