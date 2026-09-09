# Data sources and selected subsets

Start with the included examples if you only want to run CaM-PDA. Reproducing the paper experiments additionally requires the source data listed below. Obtain third-party data from its original distributor under its terms; the full datasets are not mirrored in this repository.

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

## Test sets

| Dataset | Download | Frozen evaluation subset |
|---|---|---|
| DREDS-CatNovel | [Official PKU mirror](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatNovel/) | 110 targets; separate from CatKnown training |
| NYUv2 | [Official page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) · [labeled MAT file](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) | Official 654 test images; the 428-GB raw archive is not needed for this subset |
| ICL-NUIM | [Clean lr_kt3](https://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz) · [noisy lr_kt3](https://www.doc.ic.ac.uk/~ahanda/living_room_traj3n_frei_png.tar.gz) | 80 fixed target frames, each evaluated independently |

[`paper_test.json`](../reproduction/manifests/paper_test.json) fixes all **844** target identities and seeds. NYUv2 entries include the one-based `labeled.mat` index and position in official `testNdxs`; all 654 mappings were verified against the original RGB pixels after recorded letterboxing. The [official split file](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) and its SHA256 are recorded in the manifest. ICL shares the living-room scene with adaptation trajectories; its separation is by trajectory, not by unseen scene. The three [ClearGrasp](https://sites.google.com/view/cleargrasp/data) real-test illustrations are gallery examples, not another whole-set benchmark.

## Local organization

Choose any storage drive and keep source archives outside the source checkout:

```text
data_root/
├── raw/                 Original dataset files, grouped by source
├── prepared/
│   ├── initial/         <dataset>/<split>/<frame>/rgb.png, raw_depth_mm.png,
│   │                    gt_depth_mm.png, workspace_mask.png
│   └── paper_test/      <dataset>/<frame>.npz
└── weights/             CaM-PDA and frozen monocular-prior checkpoints
```

File hashes in the manifests refer to their stated source or prepared representation. A prepared-file hash is not the hash of the distributor's ZIP, and a seed alone cannot recover a frozen sampling mask. Follow the [reproduction guide](REPRODUCIBILITY.md) before comparing aggregate scores.

## Terms and attribution

Follow each distributor's current license and access conditions. Hypersim uses CC BY-SA 3.0, DREDS uses CC BY-NC 4.0 and ICL-NUIM uses CC BY 3.0. The included engine captures are author-owned qualitative examples; their release terms are listed in [licenses](../licenses/README.md). Cite the original datasets as well as CaM-PDA. A repository link does not replace permission to redistribute another dataset.
