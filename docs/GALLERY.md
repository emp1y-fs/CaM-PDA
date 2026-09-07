# Real material gallery

The homepage shows actual CaM-PDA results from upright tabletop views in [ClearGrasp real-test](https://sites.google.com/view/cleargrasp/data). In the current source checkout, run `python run.py`, select **Included example**, and choose:

| Row | Example folder | Seed |
|---|---|---:|
| 1 | `cleargrasp_d415_000000055` | 0 |
| 2 | `cleargrasp_d415_000000071` | 0 |
| 3 | `cleargrasp_d415_000000081` | 0 |

The program applies the saved seed automatically, with its standard maximum of 50,000 sensor samples. These examples contain RGB and sensor depth at their native 848 × 480 resolution. They have no bundled camera calibration, so they export depth maps. The calibrated blade example also exports a point cloud.

## Selection and display

The selection favors natural, upright viewing directions and recognizable completed surfaces. We inspected the RGBs of 113 ClearGrasp real-test frames and 40 uniformly spaced NYUv2 test frames, then ran six shortlisted scenes with the fixed formal model. The three ClearGrasp scenes above were retained after inspecting the RGB/output pairs. Thus this is a selected qualitative showcase, not random sampling, a blinded evaluation or a quantitative ranking. The unused candidate results are retained in the experiment record. No model, threshold, training set or aggregate benchmark score was changed.

All panels show the full original frame, with no rotation or cropping. The sensor input is the prepared integer-millimetre observation converted to float32 metres; ground truth is not used for inference or color selection. Each example remains excluded from training.

Within a row, the observed and predicted depth share TURBO colors, with limits set to the 2nd and 98th percentiles of their positive values. Cool colors are near, warm colors are far, and black denotes missing observations. This shared comparison scale differs from the application's independently normalized default previews. Numerical outputs are saved unchanged; the figure performs no smoothing, local recoloring, generative editing or geometric postprocessing. Readable completion does not imply exact recovery of every transparent surface or fine structure.

The original DREDS examples `00070_0000`, `00164_0000` and `00186_0171` remain available with their original inputs, masks and seeds. The intermediate tilted DREDS showcase was replaced by these upright real scenes. [Provenance](../assets/preview_provenance.json) records input/output hashes, checkpoint identity, selection and color limits. See [dataset attribution and terms](../licenses/README.md).
