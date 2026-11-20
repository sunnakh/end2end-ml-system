# Data


**Source:** Brazilian E-Commerce Public Dataset by Olist (Kaggle), ~100k orders,
2016–2018, nine CSV tables: `orders`, `order_items`, `order_payments`,
`order_reviews`, `customers`, `sellers`, `products`, `geolocation`,
`product_category_name_translation`.

**Facts that shape the design:**

1. **Orders can have several items, each from a different seller.** The label
   and the decision are per order; seller and product features are aggregated
   per order (e.g. worst seller handling time, number of sellers, maximum
   seller–customer distance, total weight).
2. **Reviews are written after delivery or after the estimated date.** Review
   score and text are outcomes of lateness and are leakage.
3. **Geolocation has many lat/lng rows per zip prefix.** Collapse to one point
   per prefix (median lat/lng) before computing distances.
4. **Customers have two IDs.** `customer_id` is per order;
   `customer_unique_id` identifies the person. Customer history uses the latter.
5. **The data is static and historical.** "Daily" batch runs and the online API
   replay past days as if live. This is stated in the README.
6. **Not every order is delivered.** How an undelivered order is labeled:

   | Situation at the training cut-off | Label |
   | --- | --- |
   | Canceled or unavailable | none — excluded |
   | Not delivered, promised date already passed | **late (1)** |
   | Not delivered, promised date not yet reached | none — wait until it resolves |

   About 1,100 orders were handed to the carrier and never delivered in the
   data; all are overdue, so they are labeled late. For the customer this is the
   worst kind of late, and at training time nobody can know whether a parcel is
   lost or just delayed, so excluding them would use knowledge from the future
   and hide the riskiest cases. Some may be deliveries that were never recorded;
   each snapshot reports how many labels rest on non-delivery
   (`late_without_delivery`) so evaluation can check their effect.

**Out of scope:** the Marketing Funnel dataset (seller acquisition, unrelated to
delivery).
