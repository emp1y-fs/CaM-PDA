# Evaluation records

The final proof's benchmark protocols are in [paper/manifests](../paper/manifests/index.json): **DREDS11520 / NYUv2 654 / ICL80 / STD4500 / procedural600**. [BENCHMARKS.md](../docs/BENCHMARKS.md) explains the frozen input schema, units, masks and strict final-set evaluator.

The JSON/CSV records already in this directory are historical released-weight smoke checks and selected evaluations. They are retained for provenance, not relabeled as the complete final-paper result. In particular, the automatic `reproduce.py` suite has **844 targets, including only DREDS110**.

- [All final-proof table CSVs](../paper/tables/README.md) preserve the displayed results and units.
- [Downloadable source data](../docs/PAPER_DATA.md) include formal DREDS per-frame comparisons, three-seed shared-confidence summaries, geometry and routing diagnostics.
- [Model card](../docs/MODEL_CARD.md) identifies the unchanged formal checkpoint.
- [Reproducibility guide](../docs/REPRODUCIBILITY.md) distinguishes new inference, archived-output replay, full-set evaluation and retraining.

Printed DREDS MAE/RMSE use millimetres; generic evaluation exports use metres. Regional regressions and adverse cases are retained. No new benchmark scores are implied by this documentation/data release.
