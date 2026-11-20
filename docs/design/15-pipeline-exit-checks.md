# Data pipeline exit checks (M2)

* [ ] Full backfill runs from the first to the last day of the data with one command
* [ ] Rerunning any day produces identical content hashes for every output
* [ ] With fault injection on, quarantine contains **exactly** the injected bad rows (compared against `_injected.json`), each with the right reason code
* [ ] A schema-drift file fails its partition and writes nothing to silver
* [ ] Row reconciliation per table and day: landing rows = silver rows + quarantined rows + removed duplicates
* [ ] Point-in-time test: on synthetic data, seller history features never count a delivery after the order's handoff
* [ ] No review text or raw customer ID exists past bronze; the salt is not in the repo
* [ ] A per-day drift report and a manifest exist for every processed day
* [ ] The Airflow DAG generated from the step definitions runs the same chain
* [ ] Runner, validation and quarantine, storage helpers and the manifest are extracted into `ml-platform-kit`, tagged, and this repo uses the tagged version
