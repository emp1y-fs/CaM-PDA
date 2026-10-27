# Reproducibility guide

The final-paper companion is a versioned source snapshot dated **27 October 2026**. It preserves the released **026 / T1 / step 600** inference weights and follows the final proof's printed figure/table numbering.

## What can be replayed

| Level | Command / record | Scope |
|---|---|---|
| Paper assets | [Figure catalogue](../paper/figures/README.md), [table CSVs](../paper/tables/README.md) | All original final-paper figures and displayed table values |
| Numerical source data | `python -m reproduction.paper download --root ./paper_data` | Verified capture inputs, predictions, clouds, reference, routing and selected benchmark examples |
| Blade surface agreement | `python -m reproduction.paper geometry --root ./paper_data` | Distances at the archived rigid pose; no independent new registration |
| Expert routing | `python -m reproduction.paper routing --root ./paper_data` | All three blocks, all gate probabilities/activations and eight-state counts |
| New inference | `python -m cam_pda example examples/paper_blade --output outputs/paper_blade` | Run the unchanged formal model on the paper capture |
| Full final benchmark | `python -m reproduction.final_benchmark --help` | Strict verifier/evaluator for the five final manifests; prepared full sets acquired separately |
| Legacy quick check | `python reproduce.py --root ./cam_pda_test` | Older 844-target suite; DREDS110 is not final DREDS11520 |
| Training and retrained comparisons | [Training provenance](TRAINING.md) | Recorded splits and adaptation schedule, not a one-command retraining release |

See [Paper data](PAPER_DATA.md) for the figure-by-figure data index, numerical file formats and replay commands.

## Models and numerical behaviour

The formal original checkpoint SHA256 is `4f406c8778dd0de1b3215f8540b325b5f24043f689a30126a407a9995bc6fe48`; its tensor-only export is `0ed8d6299f49d69e092720b54e7edd6dd732a51ed20d50434b90a0f13a0cfacc`. The frozen monocular-prior SHA256 is `0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328`. File verification and strict state loading precede inference. All 281 retained tensors are preserved. T2/T3 continuation candidates and later experimental architectures are not substituted.

Original archived outputs are included where available. A fresh run on a different OS/GPU/kernel may show small numerical differences due to attention, nearest-neighbour ties, alignment and mixed precision. CUDA inference uses the retained FP32-prior/BF16-fine policies; CPU uses FP32. Compare numerical arrays and regional errors, not PNG file hashes. A color-map difference is not evidence of a metric-depth difference.

The benchmark grid is 392×392 with frozen masks; native author captures are 1280×720. Resizing those captures or drawing a new mask does not reproduce the benchmark. Calibration and metric units must be preserved for point clouds. A colorized depth PNG is never a numerical input.

## Legacy automatic benchmark

```console
python -m pip install ".[benchmark]"
python reproduce.py --root ./cam_pda_test
```

This uses [paper-tests-v1](https://github.com/emp1y-fs/CaM-PDA/releases/tag/paper-tests-v1), verifies the old manifest, obtains NYUv2 images/depth from its official labelled MAT, and evaluates 844 targets. The first download is about 3.83 GB including weights. `--dataset icl80`, `--prepare-only`, `--metrics-only`, `--device cpu`, `--nyu-mat`, `--checkpoint` and `--mde-checkpoint` remain available. Results include per-frame/aggregate CSV and JSON plus optional predicted maps. No sign-in is needed for public releases.

This historical convenience path retains its own preprocessing precision and observation masks. Use [the final guide](BENCHMARKS.md) when reporting final-paper full-set results.

## Scientific interpretation

Native PDA, shared-confidence PDA without experts, retrained equal-capacity dense controls and same-checkpoint expert disabling answer different questions. They must not be combined under a single “baseline” label. The paper includes region-specific regressions; source tables preserve those values. The physical comparison is a single visible-surface capture with inherited preprocessing and a pose fitted to the prediction. It does not demonstrate complete blade thickness, unseen-surface recovery or independently calibrated industrial accuracy. Additional engine components have no reference meshes and support qualitative observations only.
