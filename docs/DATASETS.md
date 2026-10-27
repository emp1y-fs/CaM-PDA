# Data sources

Final-proof benchmark descriptions are in [BENCHMARKS.md](BENCHMARKS.md). The [paper-data guide](PAPER_DATA.md) maps the publication figures to downloadable numerical files. Full third-party raw datasets are obtained from their distributors; author captures and selected prepared examples are available in the [versioned source-data release](https://github.com/emp1y-fs/CaM-PDA/releases/tag/paper-final-20261027).

## Training sources

Initial adaptation uses **96 training and 12 development frames from each of seven sources**. The exact frame IDs, source groups, sampling seeds, source metadata and prepared-file hashes are in [`initial_train_val.json`](../reproduction/manifests/initial_train_val.json). Those 756 records take precedence over a newly drawn random subset.

| Source and download | Required data | Selection used here |
|---|---|---|
| [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) | Living-room TUM-compatible RGB-D archives, with and without synthetic noise | `lr_kt0/1` training; `lr_kt2` development; `lr_kt3` reserved for testing |
| [DIODE](https://diode-dataset.org/) · [validation archive](https://diode-dataset.s3.amazonaws.com/val.tar.gz) | RGB, depth and validity files | Scene-separated subset of the downloaded **validation archive**, used for adaptation; not a claim to use DIODE's official training split |
| [BIDCD](https://zenodo.org/records/5172207) | `scenes0130_0139.tar.gz`, `scenes0140_0149.tar.gz` | Selected scenes 130–148; recorded scene separation. This is BIDCD, not BIDCD-SI |
| [IndustryShapes](https://zenodo.org/records/20268306) | `IndustryShapes_Classic_train.zip` and `IndustryShapes_cad_models.zip` | Training scenes 1–6, development 7–8; sensor depth and reference rendered from supplied CAD, poses and visible masks |
| [TartanAir](https://tartanair.org/) · [download tools](https://github.com/castacks/tartanair_tools) | IndustrialHangar, easy trajectories, front-left RGB and depth | Trajectory-separated frames in the manifest |
| [TODD / TranspareNet](https://www.pair.toronto.edu/TranspareNet/) | Official training RGB, measured depth, completed reference and masks | Session-separated subset; no test frames used for adaptation |
| [LingBot subset](https://huggingface.co/datasets/Voxel51/lingbot-depth-subset) | RGB, input depth and reference depth | Sequence-separated selection; distributor revision `ba6cc914c3fb958391cb0b4826fa2b1d91b489ba` |

The LingBot subset originates from [Robbyant MDM depth](https://huggingface.co/datasets/robbyant/mdm_depth); the recorded upstream revision is `5dd6192bdaef7b84828bd33657adc7be17623a41`.

Geometry/material adaptation retains the 672/84 initial training/development frames and adds:

| Source and download | Files needed | Selection |
|---|---|---|
| [Hypersim](https://github.com/apple/ml-hypersim) | HDR RGB, radial distance, clean no-bump camera normals, semantic instance IDs and camera parameters | 118 training frames and 24 development frames from two held-out scenes; [exact paths](../reproduction/manifests/hypersim_files.json) |
| [DREDS-CatKnown](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/) | Training part0 RGB, simulated sensor depth, rendered reference depth, instance masks and material metadata | 64 training frames; [exact paths](../reproduction/manifests/dreds_material_files.json) |

Use [`geometry_train_val.json`](../reproduction/manifests/geometry_train_val.json) for the combined **854/108** split and source/condition-cache hashes. [`dataset_sources.json`](../reproduction/dataset_sources.json) contains download URLs, archive names and pinned revisions in a machine-readable form. It lists only sources, not credentials or machine-specific directories.

## Final tests and historical subsets

The final targets are **DREDS11520, NYUv2 654, ICL80, STD4500 and procedural600**. The complete identities and fingerprints are in [paper/manifests](../paper/manifests/index.json). The older [paper-tests-v1](https://github.com/emp1y-fs/CaM-PDA/releases/tag/paper-tests-v1) bundles DREDS110/ICL80 and NYUv2 masks for an **844-frame quick check**, not the complete final benchmark. Do not report DREDS110 scores as the 11,520-frame result.

The source-data release includes the two DREDS representative inputs shown in Figures 5–6, one ICL input, and all 654 final NYUv2 protocol masks. NYUv2 RGB/depth are obtained from the [official distributor](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html); they are not mirrored here. The [preparation entry point](BENCHMARKS.md#prepare-distributor-data) reconstructs DREDS/STD from separately obtained distributor files, and acquires/prepares ICL/NYUv2 with final-array verification. It does not automatically mirror every full raw dataset or recreate all procedural renders.

## Rights and attribution

Keep original dataset licenses and citation requirements. DREDS uses CC BY-NC 4.0, ICL-NUIM CC BY 3.0, and Hypersim CC BY-SA 3.0. The author's code, trained derivatives, figures and captures receive no new license in this snapshot; author rights remain reserved. See [third-party notices](../licenses/README.md). A public URL does not relicense upstream content.
