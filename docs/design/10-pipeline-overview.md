# Data pipeline overview

Milestone M2. Turns the static Olist CSVs into a daily data flow, validated
and versioned, ending in a point-in-time training set and a daily scoring set.

## Why a simulated source

Olist is one historical dump. Processed as-is, every row is already final:
deliveries are filled in, labels are known instantly, nothing ever arrives late
or gets updated. A real pipeline faces the opposite. A **source simulator**
replays history one day at a time, emitting only what the business knew on
that day, so the pipeline must handle daily partitions, updates to existing
orders and labels that arrive days or weeks after the prediction.

## Flow

```text
 Olist CSVs (raw, SHA-256 recorded)
        │
        ▼
 ┌──────────────┐   per ds: only rows known on that day,
 │  simulator   │   order changes as CDC rows, optional fault injection
 └──────┬───────┘
        ▼
 landing/ds=YYYY-MM-DD/*.csv
        │  ingest
        ▼
 bronze/<table>/ds=…      raw, all strings, append-only, + lineage columns
        │  validate + type + redact
        ├──────────────► quarantine/<table>/ds=…   bad rows + reason codes
        ▼
 silver/<table>_changelog/ds=…   typed, valid, deduplicated changes
 silver/<table>_current           latest state as of the last processed ds
        │  build features (point-in-time)
        ▼
 gold/scoring/ds=…          orders handed off on ds, features as of handoff
 gold/training/snapshot=…   labeled orders up to a cut-off (weekly)
        │
        ▼
 reports/drift/ds=…  ·  manifests/<run_id>.json
```

## Technology

| Concern | Choice |
| --- | --- |
| Processing | Polars |
| Storage | Parquet, Hive partitions `ds=YYYY-MM-DD`, paths through `fsspec` (local now, S3/Blob in M5 by config) |
| Validation | pandera (Polars backend) plus custom checks |
| Orchestration | Own step runner; Airflow DAG generated from the same step definitions at the end of M2 |
| Versioning | Content hashes and a run manifest; no DVC |

## Daily DAG (one `ds`)

```text
simulate(ds) → ingest_bronze(ds) → validate_to_silver(ds) → merge_current(ds)
            → build_gold_scoring(ds) → drift_report(ds) → write_manifest
```

`simulate` exists only in replay; in a real deployment the landing files come
from the source system. A separate weekly DAG builds `gold/training`.

`merge_current` depends on the previous day, so a backfill runs days in order;
all other steps are independent per `ds`.

## Code layout

```text
src/late_delivery/
├── platform/              # generic building blocks → ml-platform-kit (Task 17)
│   ├── storage.py         # partitioned Parquet, atomic writes, content hashes
│   ├── validation.py      # Contract shape, validate(), quarantine reason codes
│   ├── pii.py             # salted hashing, salt fingerprint guard
│   ├── drift.py           # PSI
│   └── pipeline/          # runner.py, manifest.py, airflow.py
├── source/                # Olist-specific input
│   ├── simulator.py       # day-by-day replay + fault injection
│   └── contracts.py       # rulebook for the 9 Olist tables
├── steps/                 # one module per layer, in pipeline order
│   ├── landing.py  bronze.py  silver.py  gold.py  training.py  monitoring.py
├── features.py            # the features and the label rule (shared with serving)
├── config.py              # project settings (typed YAML)
├── dag.py                 # step order (CLI and Airflow)
└── cli.py                 # backfill, rerun, run-step, training-set
pipelines/airflow_dag.py   # generated from dag.py
configs/                   # base.yaml, local.yaml
tests/                     # mirrors src: platform/, source/, steps/, plus end-to-end
```

`platform/` must not depend on Olist-specific code. One coupling remains:
the pipeline runner and manifest take this project's `Settings`; extraction
(Task 17) replaces that with a small generic settings interface.

Package name `late_delivery` is a proposal; the platform pieces (runner,
validation/quarantine, storage, manifest) move to `ml-platform-kit` at the end
of M2.
