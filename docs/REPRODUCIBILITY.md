# Reproducing CaM-PDA

## 1. Install and obtain the model

Use the [installation instructions](INSTALL.md) in an existing Python environment or one stored at a location of your choice. The application is a Python package for Windows and Linux; it does not require building a custom CUDA extension.

The [weight manifest](../src/cam_pda/resources/models.json) records both download locations and SHA256 values. The released CaM-PDA file is an inference export of the selected T1 checkpoint; its 281 retained model tensors are unchanged. Optimizer state is omitted. The frozen monocular-prior weight is also required.

```console
python -m cam_pda download --cache-dir /path/to/weights
```

While this repository is private, CaM-PDA weight downloads require access to its private release. Alternatively provide already downloaded files when the program asks for their paths. The [API guide](API.md) documents explicit checkpoint arguments.

## 2. Reproduce the supplied captures

```console
python run.py
```

Choose `blade32` or one of `engine_component_01`–`engine_component_04`, then enter the output and weight locations. The program uses the included RGB, numerical sensor depth, camera intrinsics and seed. No source-code edits are needed. Each run writes a new result folder containing depth maps and a point cloud.

The four [additional components](ENGINE_COMPONENTS.md) are independent single-frame examples in a relatively uncluttered setting. Their provenance files identify the source capture and expected numerical output. Their point-cloud screenshots were framed in CloudCompare; terminal export reproduces the underlying geometry, not an interactive viewer's camera position.

## 3. Obtain the paper data and verify identities

Download only the required portions from [Data sources](DATASETS.md), preserving the upstream split and the repository's selected frame identities. The two training manifests specify 756 initial-stage and 962 geometry-stage records; the test manifest specifies 844 targets. The later stage deliberately replays initial training/development data.

```console
python -m reproduction.audit
python -m reproduction.audit --initial-root /path/to/data_root/prepared
```

The first command checks manifest counts and scene/trajectory separation. The second also verifies the four prepared files for every initial-stage frame against their recorded hashes. A mismatch stops verification; it is not replaced by a newly chosen image.

## 4. Evaluate predictions

The paper uses fixed **392 × 392** test inputs and frozen observation masks. Preserve the preprocessing grid, depth units, sampled mask, full reference domain, sensor-hole domain, material labels where available and geometric boundary definition. In particular, resizing the supplied 720p examples or drawing a fresh 50,000-point mask does not reproduce the benchmark protocol.

[`reproduction/evaluate.py`](../reproduction/evaluate.py) evaluates the released checkpoint against prepared, named `.npz` records. Each contains HWC `rgb` uint8, HW `sensor_m` float32 metres, `sampled` bool, `gt_m` float32 and bool masks `full`, `sensor_hole`, `challenging_material`, `depth_boundary`. The model receives only RGB, sensor depth, sampled observations and the recorded seed; reference arrays remain in the metric calculation.

```console
python -m reproduction.evaluate --data-root /path/to/prepared/paper_test --checkpoint /path/to/cam_pda_v1.pt --mde-checkpoint /path/to/depth_anything_v2_vitb.pth --output /path/to/new_evaluation
```

The command verifies each input and reference array against the manifest before loading weights. It stops on a failed frame and never silently drops failures from an average. `--dataset dreds110`, `nyu654` or `icl80` evaluates a complete individual set. The output preserves per-frame metrics and reports frame-macro AbsRel (dimensionless), MAE and RMSE (metres). Regional metrics require at least 32 reference pixels. No ground-truth scale/shift fit, clipping or prediction-dependent masking is applied.

**Prepared benchmark caches are not distributed in this release.** Raw download links and sample IDs alone do not reproduce the historical letterboxing, sensor simulation and frozen masks byte for byte. The evaluation command is usable with correctly prepared records; it is not a raw-dataset converter. Full paper-score replay therefore still requires that preprocessing/cache release. This distinction is recorded here so a different preprocessing run is not mistaken for exact reproduction.

The recorded [per-frame CaM-PDA metrics](../benchmarks/cam_pda_per_frame.csv) cover all 844 targets. Their frame-macro means reproduce all 30 available CaM-PDA aggregate entries in [paper_metrics.csv](../benchmarks/paper_metrics.csv) to within 1e-12. This verifies aggregation of the recorded measurements; it is separate from rerunning model inference.

## 5. Reproduce the training design

The model chain is **official PDA v1.1 → initial three-expert adaptation (selected epoch 11) → geometry/material adaptation (selected update 600)**. [Training](TRAINING.md) specifies data, supervision, optimizer groups, warmup, precision and checkpoint selection. The repository includes:

- [Initial-stage configuration](../reproduction/configs/initial_stage.json) and [validation history](../reproduction/configs/initial_history.json).
- [Geometry-stage configuration](../reproduction/configs/geometry_stage.json) and [selection record](../reproduction/configs/geometry_selection.json).
- [Initial proxy and loss definitions](../reproduction/methods/initial_losses.py), [geometry label rules](../reproduction/methods/geometry_labels.py), and [geometry-stage losses](../reproduction/methods/geometry_losses.py), extracted from the executed research code.
- Exact training/development [manifests](../reproduction/manifests), with separate source and condition-cache hashes.

Training uses the same balanced confidence rule and three condition channels as inference. For each stage, prepare source depth/reference fields, freeze observation sampling and confidence conditions, construct supervision from the specified fields, then train and select using that stage's development criteria. Evaluate external targets only after internal selection. The independent balanced-confidence PDA control starts from official PDA and follows the same data/exposure schedule; disabling trained experts is not that control.

**The training definitions are reference methods, not a complete training CLI.** The original per-source converters, complete frozen caches and resumable research trainer have not yet been packaged here. The supplied manifests and method code make the experiment identifiable and reviewable, but downloading raw datasets is insufficient for an exact retraining run. The inference checkpoint and included examples can be reproduced independently of this remaining training-release work.

## Numerical verification

Application 0.4.0 was checked on the original Linux CUDA environment with PyTorch 2.7.1+cu128. Frame32 and all four new component outputs matched their stored depth arrays **bitwise**, with **921,600 calibrated points per capture**. The package contains Python source and uses native PyTorch attention; no custom compiled extension is bundled.

The built wheel was also run outside the source checkout through the terminal prompts and reproduced frame32 exactly. The new evaluator was checked on one frozen frame from each of the three test sets; all three full-image metrics matched the archived values exactly. These are regression checks, not a new whole-set benchmark run. [Verification record](../benchmarks/single_frame_verification.json).

Windows and Linux may choose different attention kernels. In the earlier controlled frame32 comparison, default predictions differed by a mean absolute 0.2473 mm; disabling Linux Flash Attention reproduced the Windows output to a maximum difference of 5.96e-8 m. These are cross-platform differences, not errors against ground truth. Published aggregate scores refer to the recorded Linux environment and frozen preprocessing. Functional Windows support does not imply bitwise parity across hardware and numerical backends.
