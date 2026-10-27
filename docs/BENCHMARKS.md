# Final-proof benchmark protocols

The authoritative protocol index is [`paper/manifests/index.json`](../paper/manifests/index.json), with five compressed JSONL manifests, **17,354 unique targets** in total. It follows the final proof dated 27 October 2026. Input fingerprints lock the representation used for scoring; an archive hash and an array hash are different quantities.

| Dataset / manifest key | Frames | Source and separation |
|---|---:|---|
| `dreds11520` | 11,520 | [DREDS-CatNovel](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatNovel/), complete 45-group evaluation; distinct from CatKnown adaptation. Material scores have 11,508 eligible frames. |
| `nyu654` | 654 | [NYUv2 official labeled MAT and split](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), official 654 test targets. |
| `icl80` | 80 | [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), fixed `lr_kt3` frames; same living-room scene as other adaptation trajectories. |
| `std4500` | 4,500 | [STD-CatNovel distribution](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/STD-CatNovel/), complete downloaded archive, not all possible real transparent-object scenes. |
| `heldout600` | 600 | Original procedural role diagnostics. These data had developmental exposure; do not call them an untouched real-world test. |

## Frozen representation

Prepared files are stored as `data/<dataset>/<sample_id>.npz` beneath a user-selected data root. Benchmark RGB is **CHW uint8**, input/reference depth is **HW float32 metres**, and masks are Boolean on a **392×392 letterboxed** grid. The model operates internally at 518. Preserve the recorded sensor depth, sampling mask and per-target seed; at most 50,000 observations are supplied. Full/missing/material/boundary regions are reference-defined, with at least 32 valid pixels for an eligible regional score. Ground truth is not forwarded to the model.

Each JSONL record contains a stable target ID, group/scene, seed, relative prepared path, prepared-file SHA256, and (for real-dataset records) the exact array fingerprints. DREDS/STD entries also retain relative distributor filenames. Array fingerprints are `SHA256(str(dtype).encode() + repr(shape).encode() + contiguous_array.tobytes())`. A byte-identical NPZ has the recorded archive hash; an equivalent repack must match every recorded array fingerprint.

The complete raw third-party datasets and the full prepared caches are not mirrored in this release. Obtain sources under the distributor's terms and use the preparation entry point below; **do not assume that arbitrary resizing plus the same seed reproduces the archived arrays**. The source-data release supplies the demonstrated DREDS inputs, a frozen ICL input and all final NYUv2 masks. The verifier is intentionally strict: it stops when the target or representation is missing or differs, rather than presenting an approximation as the paper protocol.

## Prepare distributor data

The source checkout includes the registered DREDS/STD resize, EXR decoding, material-mask and boundary-mask functions, with [original source fingerprints](../paper/manifests/preprocessing_sources.json). Preparation checks **every array** against the final manifest before saving it. The raw root contains `graspnet_generate_test_catnovel_0202/` for DREDS or `real_data_novel/` for STD.

```console
python -m reproduction.prepare_final --dataset dreds11520 --raw-root /path/to/dreds/raw --root ./final_inputs
python -m reproduction.prepare_final --dataset std4500 --raw-root /path/to/std/raw --root ./final_inputs
python -m reproduction.prepare_final --dataset icl80 --root ./final_inputs
python -m reproduction.prepare_final --dataset nyu654 --paper-data ./paper_data --root ./final_inputs
```

ICL reuses the attributed prepared archive and verifies the final representation after CHW conversion. NYUv2 uses the official 2.97-GB labelled MAT, its recorded official split mapping, exact quantization and the final frozen masks; add `--mat /path/to/nyu_depth_v2_labeled.mat` to reuse an existing file. `--sample-id dreds11520_00172_0000` prepares a single recorded demonstration instead of the full set. This option changes preparation coverage, not the full-set evaluator's frame requirements. Four procedural records are downloadable, but automatic reconstruction of all 600 original rendered diagnostic records is not included.

The source distributor describes STD-CatNovel as 11,000 images; the downloaded archive used in this paper contains 4,500 pairs across five sequences. The manifest and claims are limited to those 4,500 pairs; neither number is silently substituted for the other. See the [upstream dataset description](https://github.com/PKU-EPIC/DREDS#std-dataset-real).

## Verify and evaluate a complete prepared set

From the source checkout, after installing `.[reproduction]`:

```console
python -m reproduction.final_benchmark --dataset dreds11520 --data-root /path/to/final_inputs --verify-only
python -m reproduction.final_benchmark --dataset dreds11520 --data-root /path/to/final_inputs --checkpoint /path/to/cam_pda_v1.pt --mde-checkpoint /path/to/depth_anything_v2_vitb.pth --output /path/to/new_results --save-predictions
```

Replace the dataset with `nyu654`, `icl80`, `std4500` or `heldout600` as needed. On Windows, use a quoted path such as `"D:/CaM-PDA/final_inputs"`. The root contains the `data/` directory, not just one dataset's files. No GPU is needed for `--verify-only`.

The evaluator verifies the entire selected set before constructing the model, exposes only RGB/sensor depth/sampled mask/seed to inference, and exports per-frame metrics plus frame-macro aggregates. MAE/RMSE are exported in **metres**, including for DREDS; the paper's DREDS table displays those quantities in **millimetres**, so multiply by 1000 when comparing those columns. AbsRel is dimensionless. No ground-truth alignment, prediction clipping, or prediction-dependent mask is applied. Nonfinite outputs prevent a valid aggregate. The procedural file schema uses `valid` and known role labels; it is not silently treated as a real material-labelled dataset.

This entry point evaluates the **released formal 026 checkpoint**. It does not rerun all retrained three-seed, equal-capacity, baseline or sensitivity arms; those are separate experiments in the paper. Table transcriptions and available per-frame statistics retain that distinction.

## Historical quick-check suite

`python reproduce.py --root ./cam_pda_test` still downloads/evaluates **DREDS110 + NYUv2 654 + ICL80 = 844 targets** using the older frozen manifests and [`paper-tests-v1`](https://github.com/emp1y-fs/CaM-PDA/releases/tag/paper-tests-v1). It is useful for a convenient functional benchmark. Its `dreds110` is not the final 11,520-target result, and the legacy representation must not replace the final fingerprints without validation. The old filename `reproduction/manifests/paper_test.json` is retained for compatibility, not as an assertion of final-paper coverage.
