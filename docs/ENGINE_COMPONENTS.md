# Engine-component examples

The manuscript figure contains these independently processed RGB-D captures:

| Figure row | Example folder | Recorded capture |
|---|---|---|
| 1 | `engine_component_01` | O1V2 |
| 2 | `engine_component_02` | O2V2 |
| 3 | `engine_component_03` | O3V1 |
| 4 | `engine_component_04` | O3V2 |

![RGB, measured depth, completed depth and point clouds](../assets/engine_components.png)

These author-owned captures were taken in a relatively uncluttered tabletop setting. They are qualitative examples outside the quantitative benchmark and were not used to train or select this checkpoint. The last two captures show the same component from different directions, each inferred separately.

Run `python run.py` and choose an example, or use:

```console
cam-pda example examples/engine_component_01 --output outputs/component_01
```

For installed-package examples, use the terminal selector. Each folder supplies `rgb.png`, float32 `sensor_depth.npy` in metres, `camera.json` and `provenance.json`. Sampling uses seed 0 and at most 50,000 original observations. The native image grid is 1280 × 720. The model and confidence rules are the same as for blade32.

The figure is the original high-resolution manuscript export. It combines depth previews with recorded point-cloud views; the application exports PLY geometry rather than recreating a particular CloudCompare camera view. No reference depth or reference mesh is available for these captures. [Input and prediction hashes](../assets/engine_components_provenance.json) identify the recorded files. Windows and Linux attention kernels may produce small numerical differences; see the [reproduction guide](REPRODUCIBILITY.md).
