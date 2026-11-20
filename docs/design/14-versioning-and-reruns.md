# Versioning and reruns

## Idempotent writes

- Every step writes whole partitions: write to a temporary folder, rename the
  current folder aside, rename the new one in, then delete the old one. If the
  swap fails, the old folder is renamed back, so a reader always finds either
  the old or the new partition. Leftover temp folders from a crash are removed
  on the next write. A rerun replaces, never appends a second copy.
- Output rows sorted by key before writing; fixed Parquet writer settings.
- No wall-clock values or random IDs in data columns.

## Content hashes

Each output partition gets a content hash computed from its canonically
sorted rows (not the file bytes, which can carry writer metadata). Rerunning
the same `ds` on the same input must reproduce every hash exactly. Hashes come
from Polars' serialization, so compare them only within the same lockfile: a
Polars upgrade may change every hash without any data change.

## Run manifest

`manifests/<run_id>.json` per run:

- run ID, `ds`, steps executed, start and end times
- git commit, config hash and the PII salt's fingerprint
- raw CSV SHA-256 hashes
- input and output partitions with content hashes and row counts
- quarantine counts per reason code

## Snapshots

A `gold/training` snapshot ID is the hash of its contents. MLflow (M4) logs
this ID with every training run, so each model traces back to its exact data.

## Backfill

`make pipeline START=… END=…` runs the daily DAG for each `ds` in order.
Rerunning a single day reruns that day and every later day's
`merge_current`, since current state depends on earlier days.
