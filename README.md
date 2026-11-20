# end2end-ml-system — Late Delivery Risk

Predict, at carrier handoff, whether an Olist e-commerce order will arrive after
its promised date, and choose which orders get a proactive heads-up message
each day within a fixed budget.

Data is real (Olist, anonymised); the daily flow is a replay of history and all
business numbers are simulated. Full design: [docs/design](docs/design/README.md).

## Quick start

```bash
make setup                 # install dependencies with uv
make check                 # lint, type check, tests
cp .env.example .env       # then set LATE_DELIVERY_PII_SALT (see the file)
make data                  # download Olist from Kaggle (needs a Kaggle API token)
make pipeline              # replay 2016-09 → 2018-10 into data/lake
```
