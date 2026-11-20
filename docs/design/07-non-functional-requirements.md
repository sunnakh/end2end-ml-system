# Non-functional requirements


| Requirement | Target |
| --- | --- |
| Online API latency | p95 < 100 ms per order, checked by load test |
| API fallback | Rule baseline answers if the model or a dependency fails |
| Daily batch | One replayed day scored and ranked in < 5 minutes |
| Reproducibility | Rerunning the data pipeline on the same input gives identical outputs |
| Traceability | Every served model traces to its code commit and data snapshot |
| Retraining | Weekly in the replay, through the evaluation gate |
| Cloud cost | Deploy, verify, tear down; no long-running resources |
