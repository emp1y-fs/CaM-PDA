# Installation

Use Python 3.10, 3.11 or 3.12 on Windows or Linux. All CaM-PDA runtime code is Python. Dependencies use standard Python wheels; no WSL bridge, compiler, `torch_cluster`, `xformers`, or manually built CUDA extension is required.

## Environment

Create a virtual environment from the repository root:

```console
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

If local PowerShell policy prevents activation, call `.\.venv\Scripts\python.exe` directly. No system policy change is necessary. Linux activation:

```bash
source .venv/bin/activate
```

For an NVIDIA GPU with a compatible driver:

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ".[web]"
```

For CPU:

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[web]"
```

These commands follow the [official PyTorch 2.7.1 wheel matrix](https://pytorch.org/get-started/previous-versions/). Choose a wheel supported by your GPU and driver. CaM-PDA defaults to CUDA when available, otherwise CPU. CPU inference is substantially slower. The default memory-efficient mode keeps the coarse and fine networks on the GPU sequentially. Full-resolution KNN working memory also depends on image size and sensor samples; an 8 GB GPU was used for the supplied 720p regression case.

## Weights

The inference checkpoint is approximately 392 MB and the frozen monocular prior is approximately 390 MB. Both files are checked against `src/cam_pda/resources/models.json`; mismatched weights are rejected. Default cache: `~/.cache/cam-pda`, overridable with `CAM_PDA_HOME` or `--cache-dir`.

While this repository is private, use an account that has access:

```console
cam-pda download --github-user emp1y-fs
```

This reads an already authorized Git Credential Manager account for the fixed GitHub API destination. It neither displays nor writes the token into project files. Alternatively, a `GH_TOKEN` environment variable can be used by an automated environment. Authentication is removed on redirects to a different host.

Manual alternative: download `cam_pda_v1.pt` from this repository's GitHub release and the prior from the [official PDA model host](https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth). Then:

```console
cam-pda example examples/blade32 --checkpoint weights/cam_pda_v1.pt --mde-checkpoint weights/depth_anything_v2_vitb.pth --output outputs/blade32
```

Explicit paths require existing, hash-matching files; they are never replaced by an automatic download. Git does not track model files or generated outputs.

## Browser and CLI

```console
cam-pda serve
cam-pda example examples/blade32 --output outputs/blade32
cam-pda example examples/blade32 --references examples/blade20 --output outputs/blade32_two_view
```

The browser opens at `http://127.0.0.1:7860`; the server binds to localhost. One prediction runs at a time. First use loads the weights and is slower than subsequent calls. Results are stored under `outputs/web/<run-id>` and downloadable as a ZIP. The browser accepts up to 40 MB per request and an aligned grid up to 1920×1080; use the Python interface for other controlled workloads.

If memory is insufficient, reduce the input resolution with a calibrated preprocessing step and update camera intrinsics accordingly, or use CPU. The program does not silently resize an unregistered RGB-D pair or fabricate calibration. Automatic downloading can be disabled in Python with `allow_download=False`.

## Developer checks

```console
python -m pip install ".[web,test]" build httpx
python -m pytest
python -m build
```

The portable unit checks do not download weights. Actual checkpoint regression evidence is recorded separately in `benchmarks/release_verification.json`.
