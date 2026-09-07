# Python and command-line interfaces

## Interactive program

Run `python run.py` from the source checkout, or `cam-pda` / `python -m cam_pda` after installation. Choose a language and a data source, then enter file paths at the terminal prompts. There is no source configuration to edit. The output and model storage locations are selected at runtime.

Every interactive run creates a unique subfolder inside the selected output directory. Existing results are preserved. Small preference files store the last language, output folder and model paths, not image data or credentials. Location: `%APPDATA%/cam-pda/settings.json` on Windows; `$XDG_CONFIG_HOME/cam-pda/settings.json` or `~/.config/cam-pda/settings.json` on Linux. Override it with `CAM_PDA_SETTINGS` if needed.

## Path-based Python workflow

`run_from_paths(rgb_path, depth_path, output_dir, camera_path=None, depth_scale=None, seed=0, sampled_mask_path=None, references=(), model=None, model_options=None, progress=print)` reads files, validates them, predicts and exports into a new run subfolder. It returns the result-folder `Path`. All arguments after `output_dir` are keyword-only. An existing `CaMPDA` instance can be passed as `model` to process several scenes without reloading weights; otherwise use `model_options` for checkpoint/device/cache configuration.

`run_example(folder, output_dir, **options)` additionally loads the example's recorded seed and optional `sampled_mask.npy`. Use this helper for supplied DREDS cases so the frozen sampling protocol is preserved. The terminal's example selection does the same automatically.

For reference refinement, `references` contains dictionaries with `rgb_path`, `depth_path`, `camera_path` and optional `depth_scale`. Target and reference calibration are required. Examples with a frozen sampled mask support single-view inference through this helper; combining one with references is rejected instead of silently discarding that mask.

## Single view

`CaMPDA(checkpoint=None, mde_checkpoint=None, device="auto", maximum_samples=50000, cache_dir=None, allow_download=True, memory_efficient=True)` loads the retained model and frozen prior. `predict(rgb, depth_m, seed=0, sampled_mask=None)` expects:

| Input | Contract |
|---|---|
| `rgb` | H×W×3 NumPy uint8 in true RGB order |
| `depth_m` | H×W NumPy float32 in metres; finite, nonnegative; zero is missing |
| `sampled_mask` | Optional H×W boolean mask over valid original observations |
| `seed` | Nonnegative integer, used for fixed sampling and historical alignment |

There must be at least 17 valid original observations. Without a supplied mask, the API selects at most 50,000 valid pixels using NumPy PCG64 without replacement. It performs no ground-truth fit, output scale fitting, manual ROI correction, or point addition. Inputs are copied and not modified. Different sampling or numerical environments can change output; archived external comparisons use their frozen masks and original seeds.

`DepthResult` contains `depth_m`, `sampled`, `accepted`, `raw_accepted`, `condition` (3×H×W), normalization values and `metadata`. All arrays are on CPU. `accepted` records the effective historical fallback mask, whereas `raw_accepted` retains the pre-fallback selection. `model.last_routing` contains independent R/N/E probabilities and activations at each specialist block, excluding the CLS token. These are learned gates, not guaranteed semantic segmentations.

`predict_files(image, depth, depth_scale=None, seed=0)` applies the same pipeline using the file loaders. Float NPY depth always uses metres; integer PNG requires an explicit conversion factor. Colorized depth images are not sensor inputs.

## Export

`export_result(result, rgb, output, camera=None, point_stride=1)` creates an **empty/new** output directory; existing content is rejected so files from different frames cannot be mixed. The float NPY is authoritative. Millimetre PNG is a rounded convenience copy and is omitted if positive values cannot be represented without clipping. Color preview uses the 2nd–98th percentiles and does not alter numeric depth.

`CameraIntrinsics(fx, fy, cx, cy, width, height)` describes the camera on the aligned image grid. Its width and height must match the arrays. PLY uses camera coordinates: x right, y down, z forward, all in metres. No mesh, smoothing or planar postprocessing is applied. `point_stride` uniformly subsamples only the exported point cloud and never changes predicted depth.

```json
{"fx":606.21575928,"fy":606.29565430,"cx":642.39007568,"cy":364.54278564,"width":1280,"height":720}
```

This calibration belongs only to the included author blade sequence. Supply your own calibrated intrinsics for another camera.

## Optional multiview

```python
references = [{"rgb": reference_rgb, "raw_m": reference_depth_m, "camera": reference_camera}]
result = model.predict_multiview(rgb, depth_m, camera, references, seed=0)
```

References must show the same static scene with overlap. ORB matches, PnP and guarded ICP use RGB and raw depth, with a fixed RANSAC seed. The routine selects a usable reference, predicts both views independently, and applies depth-domain reprojection. It uses a 60 mm agreement gate and restores accepted raw anchors. Insufficient correspondences/support produce an exact single-view fallback with a recorded reason. The low-level `multiview.fuse_reference` transform convention uses **millimetre translation**; the high-level API keeps input and output depths in metres.

This optional sequence export includes practical baseline/rotation guards and does not by itself reproduce every manuscript benchmark protocol. The benchmark's fixed pairs and eligibility rules are documented separately. Restoring raw anchors can increase RMSE through sensor outliers; fusion is not universally better.

## CLI

Run `cam-pda --help`, `cam-pda infer --help`, or `python -m cam_pda --help`. `--save-routing` saves block-level arrays for single-view inspection. `--references` takes example-format folders with RGB, depth and calibration. Use a fresh output folder for each run.
