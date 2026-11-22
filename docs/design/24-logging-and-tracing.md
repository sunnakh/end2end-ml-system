# Logging and prediction tracing

Cross-cutting: applies to the pipeline (M2), training (M3), tracking (M4) and
serving (M5).

## Why

Two questions must be answerable without rerunning anything:

1. **Why did this run fail?** What ran, what finished, where it stopped, with
   which inputs and which error.
2. **Which model made this prediction?** For any scored order: the model run,
   its code commit and data snapshot, the score, whether the order was
   contacted, and whether the rule fallback answered instead.

## What exists today

| Part | Durable record | Event log |
| --- | --- | --- |
| Pipeline | `manifests/<run_id>.json`, drift reports | One JSON line per step, to the terminal only; a failure keeps `Type: message`, not the traceback |
| Training | `report.md`, `metrics.json`, `run.json` (written only on success) | Two `print` lines at the end |
| Serving | not built | not built |

Records are strong; the event trail is missing, and nothing survives a crash.

## Goals

- Every pipeline day and training run leaves a JSON-lines event log on disk,
  written as events happen, so it survives a crash.
- A failure log line carries the error type, message and full traceback.
- Every prediction (batch or API) is a row in a prediction table that names the
  exact model that produced it.
- One `run_id` links the event log, the run record (manifest or `run.json`),
  the MLflow run (M4) and the predictions (M5).

## Non-goals

- Log servers or hosted tools (ELK, Loki, Datadog), dashboards, alerting.
- Cross-service tracing (OpenTelemetry): there is one API.
- Monitoring logic itself (drift on scores, live recall): M5, which reads the
  prediction table.

## Anti-goals (failure even if it works)

- Logs that contain the PII salt, raw customer identifiers or whole data rows.
- Logs treated as the source of truth for results: metrics live in
  `metrics.json` / MLflow, decisions in the prediction table; a log line only
  narrates.
- A prediction that cannot be tied to one exact model run.
- Logging that pushes the API past its p95 < 100 ms target.

## Constraints

- No new dependency: Python's standard `logging` module.
- Local files only; in the cloud deploy the same lines go to stdout, which the
  cloud's own logging collects ("deploy, verify, tear down").
- Rerunning a pipeline day or a batch must not duplicate prediction rows.

## Approaches

**How to record predictions.**

- A. One JSON log line per prediction: no extra code, but finding one order or
  joining with outcomes means parsing text, and reruns leave duplicates.
- **B. A Parquet table, partitioned by `ds` (chosen).** Predictions are business
  decisions, looked up by order and joined with outcomes; partitions are
  replaced on rerun, using the pipeline's existing writer.

**Which logging tool.**

- **Standard `logging` with a small JSON formatter (chosen):** already used by
  the pipeline, no dependency.
- structlog: nicer API for bound fields; the upgrade if the helper gets clumsy.
- Hosted log tools: need a running server or account; ruled out by the
  constraints.

**Rule:** "what happened / why did it break" → event log; "what was decided,
for whom, by which model" → prediction table; "how was this model built" →
MLflow.

## Design

### Event logs

**Format.** One JSON object per line:

| Field | Always | Example |
| --- | --- | --- |
| `ts` | yes | `2026-11-21T11:06:12Z` (UTC, seconds) |
| `level` | yes | `INFO`, `WARNING`, `ERROR` |
| `component` | yes | `pipeline`, `training`, `serving` |
| `run_id` | yes | `20261121T110611Z-f0e249b` |
| `event` | yes | `step_finished`, `candidate_scored`, `gate_checked` |
| `ds` | pipeline, batch | `2018-08-03` |
| event fields | – | `rows`, `seconds`, `model`, `recall_at_k` … |
| `error_type`, `error`, `traceback` | on failure | |

Counts, hashes, timings and identifiers only: never data rows, the salt, or
raw customer IDs (the salt fingerprint is allowed, as in the manifest).

**Where.** Next to the run's record, so one folder tells the whole story:

- Pipeline: `<data_root>/manifests/<run_id>.log.jsonl`
- Training: `<models_dir>/runs/<run_id>/train.log.jsonl` (the folder is created
  at the start of the run, so a failed run still leaves its log)

**Terminal.** A short readable line per event (`training gate_checked
result=BLOCKED failed=beats_champion`); the file keeps the full JSON.

**Helper** (`platform/logs.py`, generic, no project logic): the JSON formatter,
a context manager that attaches the file handler for one run and binds
`run_id` / `component`, and `log_event(event, **fields)`. Uncaught exceptions
inside the context are logged with their traceback, then re-raised.

**Pipeline events:** `day_started`, `step_finished` (seconds; row counts stay
in the manifest, which already records them), `step_failed` (with traceback),
`day_finished`. A backfill writes one log per day, as it writes one manifest
per day.

Log files carry wall-clock times, so they are not content-hashed and do not
affect the "reruns are byte-identical" check, which covers data outputs only.

**Training events:** `run_started` (commit, snapshot cut-offs),
`snapshot_loaded` (cut-off, rows, hash, seconds), `candidate_scored` (model,
settings, window, validation recall@K and PR-AUC, seconds) for each candidate,
`model_chosen`, `gate_checked` (each check and its result), `champion_written`
or `champion_kept`, `calendar_check`, `report_written`, `run_finished`
(seconds); `run_failed` with traceback.

### Prediction table (M5)

`<data_root>/predictions/ds=<ds>/`, one row per scored order:

| Column | Meaning |
| --- | --- |
| `order_id` | the order |
| `ds` | handoff day |
| `model_run_id` | training run of the model used, or `rule` |
| `model_commit`, `snapshot_hash` | from that run, so the row stands alone |
| `score` | model score (null for the rule) |
| `rank` | position in the day's ranking |
| `contacted` | chosen for contact within the budget |
| `fallback_used` | the rule answered because the model or a dependency failed |
| `source` | `batch` or `api` |
| `request_id` | API calls only; matches the API's event log line |
| `served_at` | UTC time of scoring |

The batch job writes a whole partition with the pipeline's writer (replaced on
rerun, content-hashed, listed in the manifest). API rows are buffered and
written to the same table; how often they are flushed is decided in M5 against
the latency target. API request events (`request_id`, `latency_ms`,
`model_run_id`, `fallback_used`, error) go to the event log.

### MLflow link (M4)

The MLflow run is tagged with the same `run_id`, and `train.log.jsonl` is
logged as an artifact, so the log travels with the model.

## Phases

1. **Now (own branch, before M4):** `platform/logs.py`; pipeline and training
   write event logs; pipeline failures keep the traceback.
2. **M4:** MLflow run tagged with `run_id`, log file stored as an artifact.
3. **M5:** prediction table and API request events.

## Tests (phase 1)

- Formatter: every line is valid JSON with the required fields; a failure line
  has a traceback.
- A run's log file is written; a run that raises still leaves a log ending in
  `run_failed` / `step_failed`.
- A training run's log holds one `candidate_scored` per candidate and a
  `gate_checked` line matching the report.
- No log line contains the salt value.

## Left to the builder

Exact terminal line format, log levels for individual events, and the helper's
function names. Anything that reads run folders must tolerate a failed run's
folder, which holds a log but no `run.json`.
