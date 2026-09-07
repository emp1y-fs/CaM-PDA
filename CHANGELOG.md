# Changelog

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
