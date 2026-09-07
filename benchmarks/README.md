# Recorded paper benchmarks

`paper_metrics.csv` preserves the archived numeric measurements for retained CaM-PDA and external algorithms. The source is the frozen 2026-09-07 comparison record; T1 is renamed CaM-PDA and T2/T3 iteration rows are excluded from this public method-comparison table. No number is regenerated from rounded manuscript values. The `origin` column identifies reused verified results; those methods were not all rerun during packaging.

Evaluation uses fixed inputs, original sensor sampling masks and predefined full-image / sensor-hole / challenging-material / depth-boundary domains where available. AbsRel, MAE and RMSE are computed per frame in FP64 on valid reference pixels, then averaged over eligible frames. Region masks are not changed based on prediction quality or gate activations. Reference depth is never supplied to the predictor. Native relative Depth Anything V2 outputs do not receive fabricated metre-scale scores.

The example selector takes the first, middle and last entries of the sorted 110-frame DREDS manifest. The included RGB-D arrays retain prior 392×392 preprocessing and archived sampling masks/seeds. These three examples illustrate the interface; they cannot estimate the complete-set scores. Their cache does not record transformed camera calibration, so no approximate PLY is presented as calibrated geometry.

## Interpretation

CaM-PDA has lower full-image AbsRel than official PDA on DREDS-CatNovel, NYUv2 and ICL-NUIM in this protocol. DREDS hole/material accuracy also improves relative to official PDA. OMNI-DC v1.1 has lower full-image error on NYUv2, and official PDA has lower NYUv2 hole AbsRel than CaM-PDA. Effects differ by region and metric.

VGGT jointly receives two RGB views and is assigned only a sensor-derived multiplicative scale. It is a contextual different-input comparison, not an equal-modality ranking. PromptDA raw/pooled prompts are explicitly separate adapters. Missing dataset/method measurements stay absent; they are not zeros.

The paper additionally reports guarded ICL multiview refinement on 80 targets. The no-external-pose temporal protocol reduced hole AbsRel from 0.034829 to 0.033168 (4.77%), with 76 fused targets and four exact fallbacks. Full-image RMSE increased from 0.037344 to 0.047694 m; restoration of raw accepted anchors contributes outlier tails. These results do not establish that multiview improves every metric. The interactive terminal, CLI and Python API support optional reference refinement with pose estimated from raw RGB-D.

The author blade sequence has no traceable reference depth. Its depth/point-cloud examples demonstrate qualitative structure and cannot certify dimensional measurement accuracy.
