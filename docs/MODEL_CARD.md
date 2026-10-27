# CaM-PDA model card

CaM-PDA completes aligned RGB-D observations into full-resolution metric depth. Optional calibrated PLY export represents the same depth values as a colored point cloud. Intended uses include RGB-D research, scene inspection and reconstruction experiments.

## Released model

The model name is **CaM-PDA**. Its retained development checkpoint is T1 / campaign 026 / step 600. The final-paper source-data snapshot provides single-frame inference with the same weights. T2/T3 continuation candidates are not deployed by this package.

| Component | Identity |
|---|---|
| Confidence | Balanced analytic geometric screening |
| Conditions | Accepted mask, normalized globally aligned disparity, normalized prefilled disparity |
| Specialists | Reflection, non-flat geometry and geometric edge |
| Routing | Independent sigmoid gates; overlapping selections |
| Backbone | Conditioned ViT-B with rank-32 residual adapters at zero-based blocks 6, 8 and 10 |
| Original checkpoint SHA256 | `4f406c8778dd0de1b3215f8540b325b5f24043f689a30126a407a9995bc6fe48` |
| Inference checkpoint SHA256 | `0ed8d6299f49d69e092720b54e7edd6dd732a51ed20d50434b90a0f13a0cfacc` |

The inference checkpoint preserves all 281 retained model tensors; training optimizer state and private filesystem paths are excluded. The checked model manifest is `src/cam_pda/resources/models.json`. Model files use the original weight release independently of application updates.

## Inputs and scope

Provide RGB and registered sensor depth on the same image grid. The pipeline needs at least 17 valid observations. It samples up to 50,000 observations, applies the retained confidence rule and predicts dense depth. Camera calibration is required to create a geometrically meaningful point cloud.

Single RGB images, colorized depth previews, unregistered RGB-D pairs and guessed intrinsics are outside the stated input contract. The terminal checks formats and dimensions; it cannot certify that supplied calibration or depth units are correct.

There is no universal accuracy guarantee across materials or domains. Specialist gates are learned routing decisions, not exact semantic segmentations. The final-paper physical blade capture has a blue-light visible-surface reference and archived registration; the separate frame32 routing diagnostic has no dense reference depth. Surface agreement is not independently calibrated metrological accuracy. The specimen is a first-stage high-pressure compressor blade, with an untraceable exact model. See [evaluation data](../benchmarks/README.md), [training provenance](TRAINING.md) and [cross-platform numerical boundaries](REPRODUCIBILITY.md).

Processing uses local files. Downloading missing weights requires a network connection; existing verified weights support offline inference. Predictions are saved directly, without numerical clipping or manual plane fitting. Color normalization affects previews only.

## Final-proof evaluation

The final protocol covers DREDS-CatNovel 11,520, NYUv2 654, ICL80, STD-CatNovel 4,500 and procedural role diagnostics 600. The legacy DREDS110 convenience subset is not the final benchmark. See [full-set protocols](BENCHMARKS.md) and [source-data coverage](PAPER_DATA.md).

No new author-owned model/data license is granted by this release; upstream weight restrictions remain applicable.
