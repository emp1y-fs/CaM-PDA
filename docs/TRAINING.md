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

A fully matched final-chain dense/no-expert training control would need both stages and the same data budget from the original initialization. Switching experts off in the final checkpoint is an inference intervention, not an independently trained dense baseline. Likewise, historical confidence-network and four-channel studies are development-stage evidence, not final-chain matched ablations.

Sources: [PDA](https://github.com/SpatialVision/Prior-Depth-Anything), [Hypersim](https://github.com/apple/ml-hypersim), [DREDS](https://github.com/PKU-EPIC/DREDS). Dataset labels and evaluation ground truth never enter the public prediction API.
