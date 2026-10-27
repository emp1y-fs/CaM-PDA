# Final-paper figures and numerical source data

This index follows the **33-page final author proof, 27 October 2026**, not older figure numbering. [Figure catalogue](../paper/figures/README.md) · [Table CSVs](../paper/tables/README.md) · [Provenance](../paper/manifest.json).

## Download

[Versioned source-data release: paper-final-20261027](https://github.com/emp1y-fs/CaM-PDA/releases/tag/paper-final-20261027). Download the ZIP directly, or use the size/SHA256-checked command from the source checkout:

```console
python -m reproduction.paper download --root ./paper_data
python -m reproduction.paper verify --root ./paper_data
```

The [data manifest](../paper/data_manifest.json) lists every numerical/companion file and its checksum. It also pins the archive. Files are stored without machine-specific absolute paths. All NPY/NPZ records are readable with `numpy.load(..., allow_pickle=False)`.

## Figure-to-data map

Paths below are relative to the extracted `paper_data` directory. Publication images themselves are in the repository, under `paper/figures/originals/`.

| Final figure | Data and replay coverage |
|---|---|
| 1 | Original workflow artwork; algorithm described in [Architecture](ARCHITECTURE.md). This schematic is not measured data. |
| 2 | Original strict-screening illustration. No separately verified strict-only numerical mask is bundled; balanced diagnostic inputs must not be relabeled as that mask. |
| 3 | Original model schematic; released inference implementation and hash-verified formal weights. |
| 4 | Published native-benchmark values in `paper/tables/table_1.csv`–`table_3.csv`, full protocols in `paper/manifests/`. |
| 5–6 | Original representative comparisons and 64×64 local windows; DREDS inputs `benchmark_examples/dreds11520_00172_0000.npz` and `dreds11520_00164_0128.npz`; NYUv2 targets `0541` and `0379` are obtained from the distributor using the recorded protocol. Exact historical retrained equal-capacity MoE/dense prediction arrays are not included in this release. Running the formal deployment checkpoint on these inputs is a new demonstration, not a numerical replay of those paired retrained arms. |
| 7 | `paper_blade/`: same-capture RGB, raw ToF, native PDA and formal CaM-PDA depth, calibration, sampled/accepted masks and original color previews. |
| 8 | Original acquisition-hardware photograph panels; no new measurement data are implied by the images. |
| 9 | `paper_blade/geometry/`: 25,656 retained prediction points, 137,383 visible reference points, registered arrays, rigid transform, region/section data. |
| 10 | Original author geometry illustrations; measured visible-surface data are the same as Figures 7/9. No unseen geometry or thickness ground truth is supplied. |
| 11 | `routing_frame32/`: separate RGB-D diagnostic capture, formal T1 output, archived conditions, all gate probabilities/activations at blocks 6/8/10. |
| 12 | `engine_component_01`–`04`: RGB, numerical raw/completed depth, calibration, original previews and colored PLY; qualitative only. |
| A1 | `paper_blade/comparison_depths/`, `geometry/*_roi.npz`, error-map CSVs and fixed-pose protocol: raw/native PDA/shared-confidence PDA/CaM-PDA. |

All 13 original figure assets are preserved byte for byte from the final source package. PNG previews are rendered copies for browsing, not recovered numerical depth. CSV tables transcribe displayed, rounded paper values; they do not replace per-frame observations.

## Numerical files and units

| File | Meaning |
|---|---|
| `rgb.png` | Aligned true RGB, 1280×720 for author captures |
| `sensor_depth.npy` | float32 HW measured depth in **metres**; zero is missing |
| `cam_pda_depth.npy` | Archived full-resolution prediction in **metres** |
| `sampled_mask.npy` | Frozen Boolean observation mask; use it rather than drawing a new mask |
| `accepted_mask.png` / `.npy` | Archived accepted-anchor mask, not dense reference truth |
| `camera.json` | Calibrated `fx`, `fy`, `cx`, `cy`, `width`, `height` |
| `*_preview.png` | Original depth visualization; not an input to inference or scoring |
| `*_mm.png` | Convenience integer-millimetre export; float-metre NPY is the numerical source |
| `pointcloud_rgb_m.ply` / `prediction_camera_m.ply` | Camera coordinates in **metres**, x right / y down / z forward |
| `geometry/reference_visible_mm.ply` | Blue-light visible-surface reference in **millimetres** |
| `geometry/registered_arrays.npz` | Predicted/reference points and archived rigid pose in **millimetres**, plus display coordinates |
| `routing_6.npz`, `routing_8.npz`, `routing_10.npz` | 2442×3 gate probability and activation matrices; spatial grid 37×66 |

Figure 7 uses crop `[480, 260, 755, 565]` and shared display limits `[0.15850081890821457, 1.344947165250779]` m with reversed TURBO. These limits apply only to the preview. Saved numerical predictions are not clipped to them.

`matched_dense_040` is the historical filename for **shared-confidence PDA without experts** in the physical four-way comparison. It is **not** the retrained equal-capacity dense control of Tables 4–5 / Figures 5–6. The public protocol records this distinction explicitly.

## Run and inspect

```console
python -m cam_pda example examples/paper_blade --output outputs/paper_blade
python -m cam_pda example examples/engine_component_01 --output outputs/component_01
python -m reproduction.paper geometry --root ./paper_data
python -m reproduction.paper routing --root ./paper_data
```

The geometry replay uses SciPy nearest-neighbour distances at the **archived pose**, with no fresh fit. It returns approximately **3.5141 mm** prediction→reference and **3.0416 mm** reference→prediction (symmetric mean **3.2778 mm**). The pose was historically fitted using the CaM-PDA prediction. This is exploratory visible-surface agreement, not independent metrological accuracy. The cloud had SOR with 30 neighbours / 1.0 standard deviation and manual segmentation, removing part of the root; hidden surfaces, thickness and the complete removed root are not validated.

Routing replay reports each specialist's activation percentage and all eight R/N/E states. These are learned patch decisions, not semantic material/curvature truth. The diagnostic capture has no dense measured depth reference.

## Benchmark demonstrations and attribution

`benchmark_examples/` contains two DREDS examples under **CC BY-NC 4.0** and one ICL-NUIM example under **CC BY 3.0**, with their frozen final-protocol identities. RGB in these prepared benchmark NPZs is **CHW uint8**, unlike the HWC image API. Transpose it with `np.moveaxis(rgb, 0, -1)` before inference. `gt_m` and masks are for scoring, never model inputs.

`benchmark_masks/nyu654/` contains all 654 frozen protocol-mask records. NYUv2 RGB and depth are not mirrored; the [final preparation command](BENCHMARKS.md#prepare-distributor-data) obtains them from the [official distributor](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and verifies the exact final arrays, using the recorded split mapping and frozen masks.

Four author-generated examples in `procedural_examples/` expose the recorded geometry/material diagnostic schema. The complete 600-target identities are in the final manifest; this is not an untouched real-world generalization test.

Author-owned material is downloadable without an added license; author rights remain reserved. Existing third-party terms and required attribution are retained in `NOTICE.txt` and [licenses](../licenses/README.md). This release is an inference/source-data companion, not a claim to reproduce every training campaign or comparison checkpoint.
