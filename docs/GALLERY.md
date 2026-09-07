# Material gallery

The homepage shows three actual CaM-PDA results from the held-out DREDS-CatNovel set. In the current source checkout, run `python run.py`, select **Included example**, and choose:

| Row | Example folder | Saved seed |
|---|---|---:|
| 1 | `dreds_catnovel_00144_0171` | 20301187 |
| 2 | `dreds_catnovel_00163_0128` | 20314304 |
| 3 | `dreds_catnovel_00185_0128` | 20364754 |

The program loads the saved sampling mask and seed automatically. These examples have no sealed calibration for their processed grid, so they export depth maps. The calibrated blade example also exports a point cloud.

## Selection and display

The three scenes were selected by inspecting the RGB images of the frozen 110-frame test set, before generating the new predictions. Selection favored recognizable objects, adequate illumination and varied shapes/materials. This is a qualitative showcase, not random sampling or a performance ranking. Model weights and aggregate test results are unchanged. No example enters training.

The frozen inputs are 392 × 392 pixels, with artificial padding above row 85 and from row 306 onward. All gallery columns show the same full-width rectangle `[0, 85, 392, 306)`; no scene content is excluded. Inference still receives the complete original input, and exported NPY depth retains the complete grid.

Within each row, both depth panels share TURBO colors and limits computed from the 2nd and 98th percentiles of positive observed and predicted depth in the displayed rectangle. Cool colors are near; warm colors are far; black denotes missing observations. This shared comparison scale differs from the application's independently normalized default previews. The figure performs no numerical smoothing, local recoloring, generative editing or geometric postprocessing. Fine structures in complex objects can remain imperfect.

The original manifest-position examples `00070_0000`, `00164_0000` and `00186_0171` remain available with their original inputs, masks and seeds. [Provenance](../assets/preview_provenance.json) records input/output hashes, checkpoint identity, display bounds and color limits. [Dataset attribution and terms](../licenses/README.md) apply to all six material cases.
