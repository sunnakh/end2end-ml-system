# Framing block


| | |
| --- | --- |
| **Business goal** | Reduce the damage of late deliveries: fewer 1–2 star reviews and support complaints caused by orders that miss their promised date. |
| **Decision** | Every day, choose which orders handed to the carrier that day receive a proactive heads-up message (apology + small voucher), within a fixed daily budget of K contacts. |
| **Prediction moment** | Once per order, at carrier handoff (`order_delivered_carrier_date`). Only information available at that moment may be used. |
| **Input** | One order at handoff: its items, sellers, products, payment, customer and seller locations, timestamps up to handoff, and seller history known before handoff. |
| **Output** | Calibrated P(late) for the order; the daily batch also outputs the ranked contact list. |
| **ML task** | Binary classification with calibrated probabilities, used for top-K ranking. |
| **Label** | `late = order_delivered_customer_date > order_estimated_delivery_date` (date-level comparison). An order still undelivered after its promised date is also late; see [Data](04-data.md), item 6. |
| **Primary offline metric** | PR-AUC and recall@K (share of late orders caught within the budget). |
| **Guardrail metrics** | Calibration (Brier score, reliability curve), precision@K, ROC-AUC. |
| **Business metric (simulated)** | Net savings vs the rule baseline = R$20 × late orders caught − R$2 × contacts sent. |
| **Non-ML baseline** | Rule: contact if the seller handling time is above that seller's historical median, or the order ships between states; rank by handling-time excess. |
