# Changelog

## 0.3.0

- Replace hard reference-depth replacement and raw-anchor restoration with continuous, edge-aware depth correction.
- Preserve the single-view network, model weights, confidence screening and runtime path prompts.
- Record reference support, correction size and solver residuals; return the exact single-view prediction when registration/support is unavailable or the solver fails its acceptance check.
- Add SciPy's standard sparse linear solver dependency; no custom compiled extension is introduced.
- Include reference frame blade20 beside blade32 in the Python wheel so the optional two-view example also works from an installed package.
- Document the frozen 80-target evaluation, regional trade-offs and real frame-32 comparison.

The low-level hard-fusion helper `multiview.fuse_reference` is retired. Use `CaMPDA.predict_multiview` or `run_from_paths(..., references=...)`; their input paths and metre-depth conventions are unchanged.

## 0.2.0

- Start with `python run.py`, `cam-pda` or `python -m cam_pda`; enter local input and output paths when prompted.
- Choose an included RGB-D example or your own files, with English / 中文 prompts, remembered storage settings and optional reference refinement.
- Preserve existing results using a new subfolder per interactive run.
- Include the calibrated blade32 example in the Python wheel.
- Add a public-facing depth gallery, model card and path-based Python helpers.
- Remove the browser service, HTML/JavaScript frontend and web dependencies. Existing `cam-pda serve` callers should use the terminal entry point.
- Reject fixed sampled-mask examples combined with reference refinement consistently across CLI and Python workflows.

Model weights and inference operations are unchanged. Application version 0.2.0 uses the same verified model files from the original weight release.

## 0.1.0

Initial portable inference package, examples, Python/CLI interfaces and recorded Windows/Linux model verification.
