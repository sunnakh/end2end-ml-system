# Business assumptions (simulated)


| Assumption | Value | Reasoning |
| --- | --- | --- |
| Cost of one contact | R$2 | Voucher + message cost. |
| Cost of an uncontacted late order | R$20 | Expected loss from a bad review (churn, support time). |
| Daily budget K | 5% of the day's handoffs | Below the late rate (~8%), so the budget actually binds and ranking matters. |
| Minimum score to contact | P(late) > 0.10 | Break-even from the cost ratio 2 / 20; below it, contact is not worth its cost even if budget remains. |

Supporting evidence to check in EDA: late orders should show a much higher
share of 1–2 star reviews than on-time orders. Reviews are used **only** for
this check, never as features (see [Feature rules](05-feature-rules.md)).
