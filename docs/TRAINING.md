# Training, expert supervision and evaluation provenance

## Retained model

The release is the retained **T1** checkpoint from the geometry/material adaptation campaign, selected at step 600. Its initialization chain is official PDA v1.1 → initial three-expert adaptation → geometry/material adaptation. T2/T3 continuation candidates are not part of this checkpoint.

| Stage | Data and executed budget | State inherited by this release |
|---|---|---|
| Initial adaptation | Seven sources; 96 training and 12 development frames per source, 672/84 total; 12 epochs, 2,016 updates, effective batch 4 | Epoch 11, 1,848 updates |
| Geometry/material adaptation | 672 replay + 118 Hypersim + 64 DREDS-CatKnown training frames (854 total); 84 prior + 24 Hypersim development frames (108 total); 1,000 executed updates | Internally selected step 600 |

The first 100 adaptation updates train routers with soft routing and two Hypersim plus two DREDS samples per batch. Later batches contain two replay, one Hypersim and one DREDS sample, with hard forward routing and straight-through gradients. The joint objective combines full-image AbsRel, 0.05 class/expert-balanced routing BCE and 0.10 plane depth-gradient loss. The plane loss preserves the reference slope of tilted planes. Confidence selection and native three-channel construction remain fixed.

Geometry/material adaptation uses seed 2026090626 and AdamW. Learning rates are 1e-6 for the backbone, 2e-6 for the decoder, 2.5e-5 for routers, 5e-5 for residual adapters and 2.5e-4 for residual gates. Weight decay is 1e-4 except for gates (zero). The router warmup uses fixed rates; the joint stage follows a cosine schedule to 10% of the initial rates. Source frames use a 392×392 training grid and the conditioned ViT processes a 37×37 spatial token grid. These settings describe the actual selected chain, not a newly executed training run.

## Labels

The dataset does **not** directly provide a ready-made R/N/E expert segmentation. Specialist supervision is deterministically derived from original renderer/material/geometry fields using fixed rules. It is not manually painted from image colors.

| Specialist | Positive evidence | Reliable negative / unknown treatment |
|---|---|---|
| Reflective R | DREDS material class 3 | Diffuse class 1 is negative; classes 0/2 and unsupported background remain unknown |
| Non-flat N | Hypersim clean no-bump normals showing sustained multiscale variation with valid depth and instance support | Geometrically flat regions across multiple windows are negatives; folds, mixed instances and unsupported regions are excluded |
| Edge E | Clean depth discontinuities and, in Hypersim, normal folds | RGB texture edges alone are not labels; insufficient geometry is unknown |

Hypersim radial distance is converted to axial camera depth using its camera-ray model. Invalid normal fields are quarantined. Unknown supervision contributes no class label gradient. Replay in the later geometry/material phase supplies depth preservation, rather than newly invented specialist labels.

The initial adaptation did use historical sensor-error and local-depth proxies. Consequently, the final weight history cannot be described as renderer-only supervision from the start. The geometry/material stage improves supervision of large planes, but gate activations remain learned processing decisions; they are not semantic labels guaranteed to match every real scene.

## Data separation and selection

The 110 DREDS-CatNovel targets and 654 official NYUv2 test images do not enter adaptation gradients. The new Hypersim development split uses two complete scenes. ICL lr_kt3 test targets are separated from training trajectories within a previously seen living-room scene; they are not an unseen-scene claim. Author blade frames are diagnostics, not gradient or checkpoint-selection data.

Step 600 was selected internally before external scoring, using the original development-depth constraint, Hypersim scene-level recall constraints and plane false-activation criteria. The later decision to retain T1 over continued-training candidates also considered previously viewed external results. This is not a newly blinded benchmark. One training seed was used, and the new DREDS material subset has no separate new material validation split.

## Reproduction boundary

The package reproduces inference for the released checkpoint and includes frozen input/output regression evidence. The small distributed examples are insufficient to reproduce training or whole-test-set aggregate scores. Full training replay requires the original source datasets, frozen scene/frame manifests, initial PDA weights, both stage configurations, loss/label conversion code, optimizer schedules and selection records. This release records those requirements rather than substituting a new approximate training recipe.

A subsequent independent balanced-confidence PDA control was trained from the original PDA through both stages, using the same training frames, splits, sampled batch identities and inherited common-parameter update counts. Stage one completed 12 epochs; epoch 11 was preselected to match the formal model's inherited parent. Stage two completed 1,000 batches; batch 600 was preselected, including 100 initial frozen-common-parameter batches followed by 500 common-parameter updates. Thus the evaluated dense control inherits 1,848 + 500 common-parameter updates. These choices were fixed before its external evaluation to match training exposure, rather than chosen for the dense model's own best validation score. No CaM-PDA expert weights were used to initialize it.

CaM-PDA reduces full-image AbsRel relative to this control by 6.98%, 5.78% and 5.00% on DREDS-CatNovel, NYUv2 and ICL-NUIM. The dense control has slightly lower ICL hole AbsRel (0.034363 versus 0.034829). [All three-method metrics](../benchmarks/matched_training_metrics.csv) retain regional exceptions and unavailable material labels. This compares complete training configurations with matched data and common-parameter exposure; it does not isolate parameter count or provide an optimally tuned dense-model comparison. Historical confidence-network and four-channel studies remain development-stage evidence.

Sources: [PDA](https://github.com/SpatialVision/Prior-Depth-Anything), [Hypersim](https://github.com/apple/ml-hypersim), [DREDS](https://github.com/PKU-EPIC/DREDS). Dataset labels and evaluation ground truth never enter the public prediction API.
