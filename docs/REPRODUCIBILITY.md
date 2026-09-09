# Reproducing CaM-PDA

## One-command benchmark

From the cloned/extracted repository, install once with `python -m pip install ".[benchmark]"` after installing PyTorch, then run:

```console
python reproduce.py --root ./cam_pda_test
```

This downloads the three selected test subsets and both released weights, checks file and array hashes, evaluates all 844 targets, and exports `results.md`, `summary.csv`, `per_frame.csv`, `frames.jsonl`, `summary.json`, `run.json`, and per-frame numerical/color depth maps. All files stay under `--root`; each evaluation creates a new result folder. These tools are part of the source repository/archive, so run the command from that folder.

| Option | Purpose |
|---|---|
| `--root "D:/CaM-PDA-test"` | Select a Windows storage drive; Linux paths work too |
| `--dataset icl80` | Evaluate only ICL80; also accepts `dreds110`, `nyu654`, or default `all` |
| `--device cpu` | Use CPU; default `auto` selects CUDA when available |
| `--prepare-only` | Download/prepare/verify data and weights without inference |
| `--metrics-only` | Export all scores without saving predicted depth maps |
| `--nyu-mat /path/to/nyu_depth_v2_labeled.mat` | Verify and reuse an existing official MAT file |
| `--checkpoint /path/to/cam_pda_v1.pt` | Verify and reuse the released CaM-PDA weight |
| `--mde-checkpoint /path/to/depth_anything_v2_vitb.pth` | Verify and reuse the frozen prior weight |
| `--github-user YOUR_GITHUB_NAME` | Select an authorized Git Credential Manager account during private review |

The full first download is approximately 3.83 GB including both weights; reserve about 6 GB for data and outputs. DREDS110 and ICL80 are supplied as prepared archives under their original licenses, with attribution and processing notices. NYUv2 images/depth are obtained directly from the original distributor: the 2.97-GB labeled MAT is converted locally using the fixed official test indices and supplied Boolean protocol masks. All 654 converted records match the archived arrays exactly. No training is performed.

Interrupted downloads retain `.partial` files and resume on the next invocation. Complete verified downloads and prepared records are reused. Evaluation runs create separate result folders; interrupted **inference** is rerun, not resumed. Incomplete runs are marked in `run.json` and do not receive a completed result report. Private assets require a GitHub account granted access, using existing Git Credential Manager credentials or `GH_TOKEN`/`GITHUB_TOKEN`. Public assets need no credentials.

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

For testing, the one-command workflow above downloads and prepares the exact required records. For training, obtain the required portions from [Data sources](DATASETS.md), preserving the upstream split and the repository's selected frame identities. The two training manifests specify 756 initial-stage and 962 geometry-stage records; the test manifest specifies 844 targets. The later stage deliberately replays initial training/development data.

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

**The paper test data can now be prepared automatically** using `python reproduce.py --root ./cam_pda_test --prepare-only`. The versioned [test asset manifest](../reproduction/test_assets.json) pins download locations, sizes and hashes. DREDS110/ICL80 use the supplied prepared records; NYUv2 is reconstructed from the official MAT with frozen protocol masks. This preserves the historical grid, millimetre round-trip, FP16 cache precision and sampling masks rather than drawing a new subset. The lower-level evaluation command above remains available for existing prepared data.

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

The one-command workflow was checked with real release downloads of all three test assets and both weights. Every prepared input/reference array matches the 844-frame manifest; all 654 NYUv2 conversions from the official MAT are byte-exact. A complete new ICL80 inference run reproduced all recorded per-frame AbsRel, MAE and RMSE values exactly and exported 80 numerical depth maps and 80 color previews. This integration check reran ICL80; it does not claim a new inference pass over DREDS110 or NYUv2. [Verification record](../benchmarks/one_command_verification.json).

Application 0.4.0 was checked on the original Linux CUDA environment with PyTorch 2.7.1+cu128. Frame32 and all four new component outputs matched their stored depth arrays **bitwise**, with **921,600 calibrated points per capture**. The package contains Python source and uses native PyTorch attention; no custom compiled extension is bundled.

The built wheel was also run outside the source checkout through the terminal prompts and reproduced frame32 exactly. The new evaluator was checked on one frozen frame from each of the three test sets; all three full-image metrics matched the archived values exactly. These are regression checks, not a new whole-set benchmark run. [Verification record](../benchmarks/single_frame_verification.json).

Windows and Linux may choose different attention kernels. In the earlier controlled frame32 comparison, default predictions differed by a mean absolute 0.2473 mm; disabling Linux Flash Attention reproduced the Windows output to a maximum difference of 5.96e-8 m. These are cross-platform differences, not errors against ground truth. Published aggregate scores refer to the recorded Linux environment and frozen preprocessing. Functional Windows support does not imply bitwise parity across hardware and numerical backends.
