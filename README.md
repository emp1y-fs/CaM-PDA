# CaM-PDA

**Align visual depth with real sensor measurements. Recover dense metric depth from RGB-D.**

Python · Windows & Linux · Single-frame inference

[简体中文](README.zh-CN.md) · [Install](docs/INSTALL.md) · [Data & downloads](docs/DATASETS.md) · [Reproduce](docs/REPRODUCIBILITY.md) · [API](docs/API.md)

![RGB, measured ToF depth and CaM-PDA depth in a real engine-blade scene](assets/blade_depth_showcase.png)

CaM-PDA extends [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything) with a **balanced-confidence front end** and an **expert matrix**. The confidence front end selects sensor observations for aligning the visual depth prior with measured depth. The expert matrix introduces independently gated reflective, non-flat and edge experts into the conditioned network to predict dense metric depth.

**Input:** one RGB image and its registered sensor depth. **Output:** a depth preview, numerical metric depth and, with camera calibration, a colored point cloud. RGB and sensor depth must already share the same pixel grid; the alignment inside CaM-PDA recovers depth scale and structure, not camera-to-camera image registration.

## Overall workflow

![CaM-PDA workflow: balanced-confidence screening, depth alignment and prefilling, and conditioned depth estimation with reflective, non-flat and edge experts](assets/cam_pda_overall_workflow.png)

## Run CaM-PDA

Use Python **3.10–3.12**. Clone the repository, install PyTorch and the package, then start the program:

```console
git clone https://github.com/emp1y-fs/CaM-PDA.git
cd CaM-PDA
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python run.py
```

The commands above use the NVIDIA CUDA build. See [installation](docs/INSTALL.md) for CPU support, isolated environments and weight downloads. Both model files together occupy about **0.8 GB**.

The program asks you to choose an included example or **enter input paths at runtime**, followed by the output directory and model locations. English and Chinese prompts are available. `cam-pda` and `python -m cam_pda` start the same program after installation.

| Input | Format |
|---|---|
| RGB image | PNG or JPEG |
| Registered sensor depth | Floating-point `.npy` in metres, or a single-channel integer `.png` with its unit specified; zero means missing |
| Camera intrinsics | JSON with `fx`, `fy`, `cx`, `cy`, `width`, `height`; required for point-cloud export |

A colorized depth image is only a preview. Supply numerical sensor depth to recover metric scale. Use your camera's calibration for your own captures.

Each run creates a separate folder under the selected output directory:

```text
result_folder/
├── rgb.png              Input RGB
├── depth_color.png      Depth preview
├── depth_m.npy          Float32 depth in metres
├── depth_mm.png         Rounded millimetre depth, when representable
├── point_cloud.ply      Calibrated colored point cloud
├── accepted_mask.png    Retained sensor observations
└── metadata.json        Calibration, units and prediction settings
```

Without intrinsics, the program saves depth maps only. Point-cloud coordinates are in metres: x right, y down, z forward. Export preserves the predicted depth; preview colors do not modify its values.

## Engine components

![Four single-frame captures: RGB, raw ToF depth, CaM-PDA depth and point clouds](assets/engine_components.png)

These additional engine-component captures appear in Section 5 of the manuscript. They were acquired in a **relatively uncluttered tabletop setting**. Each row is processed independently from one RGB-D frame. The examples illustrate completion of missing sensor depth and the resulting surface structure; no reference geometry is available for a quantitative accuracy claim.

Choose `engine_component_01` through `engine_component_04` in the program to reproduce the four rows. The original RGB, sensor depth, calibration and sampling seed are included in both the source checkout and installed package. [Example details and provenance](docs/ENGINE_COMPONENTS.md).

Additional [real material examples](docs/GALLERY.md) are available in the source checkout.

## Reproduction and implementation

- [Reproduction guide](docs/REPRODUCIBILITY.md): installation → weights → examples → evaluation → training records.
- [Dataset sources](docs/DATASETS.md): original download locations, exact sample selections and required files.
- [Training](docs/TRAINING.md): both adaptation stages, expert supervision and checkpoint selection.
- [Architecture](docs/ARCHITECTURE.md): balanced confidence, three conditioning channels and independent R/N/E experts.
- [Python API](docs/API.md): file and array interfaces for integration.
- [Model card](docs/MODEL_CARD.md): intended use and limitations.

The released weights are the manuscript's CaM-PDA checkpoint. Application updates do not retrain the model.

## Acknowledgements and terms

Built on [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and DINOv2. [Upstream notices and data terms](licenses/README.md) apply; ViT-B weights and DREDS data carry **CC BY-NC 4.0** terms.

This repository is private during author review. Licensing for new CaM-PDA code and author-owned examples is pending; no additional redistribution rights are granted by their inclusion here.
