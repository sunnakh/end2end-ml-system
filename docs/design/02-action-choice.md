# Why this action, and why not intervention


The contact is a **heads-up message**, not an intervention (expedite, carrier
switch). The message does not change when the order arrives, so:

- historical labels stay valid training data, with no feedback loop from the
  model's own choices;
- no random holdout group is needed;
- nothing about Olist's real operations has to be invented.

**If contact became an intervention**, the design would change: the model's
selections would alter the outcomes it later trains on, so a random holdout
group (no treatment) would be required for unbiased labels and evaluation, and
the task would shift toward uplift modeling (who benefits from the action, not
who is at risk). This is covered as a design extension, not built.
