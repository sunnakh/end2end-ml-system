# Late Delivery Risk — Design

Status: draft · Stage 1 (requirements and framing)

Source of truth for what the system does and why. All business numbers are
**simulated assumptions**, labeled as such; the data itself is real (Olist, anonymised).

## Sections

1. [Framing block](01-framing.md)
2. [Why this action, and why not intervention](02-action-choice.md)
3. [Business assumptions (simulated)](03-business-assumptions.md)
4. [Data](04-data.md)
5. [Feature rules (point-in-time correctness)](05-feature-rules.md)
6. [Evaluation design](06-evaluation.md)
7. [Non-functional requirements](07-non-functional-requirements.md)
8. [Goals, non-goals, anti-goals](08-goals-and-scope.md)
9. [Open items (resolved during EDA, Phase 1)](09-open-items.md)

## Data pipeline (M2)

10. [Pipeline overview](10-pipeline-overview.md)
11. [Source simulator](11-source-simulator.md)
12. [Layers and tables](12-layers-and-schemas.md)
13. [Validation and quarantine](13-validation-and-quarantine.md)
14. [Versioning and reruns](14-versioning-and-reruns.md)
15. [Exit checks](15-pipeline-exit-checks.md)
16. [Results on real data](16-pipeline-results.md)

## Training and evaluation (M3)

17. [Overview](17-training-overview.md)
18. [Data and splits](18-training-data-and-splits.md)
19. [Models](19-models.md)
20. [Metrics and report](20-metrics-and-report.md)
21. [Evaluation gate](21-evaluation-gate.md)
22. [Exit checks](22-training-exit-checks.md)
23. [Results on real data](23-training-results.md)

## Added as built

Tracking and packaging (M4) ·
Serving, monitoring and deployment (M5).
