# Cross-platform numerical verification

The distributed inference checkpoint preserves all 281 retained model tensors exactly. Removing optimizer state changes the file SHA256 but not a single model tensor. Both hashes are recorded in the model manifest.

The portable Linux CUDA pipeline reproduced the original real frame-32 result exactly: 921,600 depth pixels, all three conditions and 46,970 accepted anchors. Linux CPU inference was exercised on a 90×160 aligned view using FP32. Windows native CUDA inference was verified in the original package release.

Application v0.2.0 adds runtime terminal path entry without changing the model. Its real `run.py` process was exercised with stdin input for the included blade32 example, user-entered paths (including spaces and Chinese characters), and reference refinement. Both single-view paths reproduced the retained depth bitwise; the reference-view path fused successfully. All three DREDS examples preserved their saved seeds and masks and matched the original release depth hashes. Each blade export contained 921,600 calibrated points. Existing files in the selected result folder remained untouched.

Default Windows and Linux BF16 predictions are **not bitwise identical**. On the checked frame-32 input, the three conditions were identical; the mean absolute output difference was 0.2473 mm, the median 0.00425 mm, the 95th percentile 1.2364 mm and the maximum 5.6811 mm. These are differences between implementations, not errors against ground truth.

The official PyTorch 2.7.1 CUDA wheels expose Flash Attention on the tested Linux installation but not on Windows. A controlled Linux run with only Flash SDPA disabled reproduced the Windows result to a maximum difference of 5.96e-8 m and a mean difference of 6.07e-9 m. This identifies the different attention kernels as the source of the measured platform discrepancy. The public package retains the evaluated default Linux path and native PyTorch fallback on Windows; weights, confidence thresholds and input conditions are unchanged.

The archived paper scores refer to their Linux evaluation environment and frozen masks/seeds. Windows support means a tested functional implementation, not a promise of bitwise reproduction of Linux aggregate scores or identical dimensional measurements. Other hardware, precision policies or package versions may produce additional numerical differences.

See `benchmarks/release_verification.json` for historical numerical checks and `benchmarks/interactive_verification.json` for the current application checks. The unit suite tests units, calibration, output isolation, deterministic sampling, CPU KNN edge cases, raw-anchor restoration and runtime path prompts without downloading weights. Historical web-interface measurements in the original record concern v0.1.0; that interface has been removed from v0.2.0.
