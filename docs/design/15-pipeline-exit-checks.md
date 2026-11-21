# Data pipeline exit checks (M2)

* [x] Full backfill runs from the first to the last day of the data with one command
* [x] Rerunning any day produces identical content hashes for every output
* [x] With fault injection on, quarantine contains **exactly** the injected bad rows (compared against `_injected.json`), each with the right reason code
* [x] A schema-drift file fails its partition and writes nothing to silver
* [x] Row reconciliation per table and day: landing rows = silver rows + quarantined rows + removed duplicates
* [x] Point-in-time test: on synthetic data, seller history features never count a delivery after the order's handoff
* [x] No review text or raw customer ID exists past bronze; the salt is not in the repo
* [x] A per-day drift report and a manifest exist for every processed day
* [x] The Airflow DAG generated from the step definitions runs the same chain
