# Installation and first run

CaM-PDA supports **Python 3.10–3.12 on Windows and Linux**. All application source is Python, packaged as a platform-independent wheel. PyTorch, NumPy and OpenCV use their standard prebuilt dependencies. No JavaScript frontend, local web server, Visual Studio build or custom CUDA extension is needed.

## 1. Create an environment

Download the source ZIP or clone this repository, open a terminal in its folder, then:

```console
python -m venv .venv
```

Choose an environment location on a drive with enough free space; replace `.venv` with that path if desired. Activate it:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```console
.venv\Scripts\activate.bat
```

Linux:

```bash
source .venv/bin/activate
```

If PowerShell prevents activation, use `.venv\Scripts\python.exe` directly. A system policy change is unnecessary.

## 2. Install dependencies

NVIDIA GPU with a compatible driver:

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
```

CPU:

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

These are the [official PyTorch 2.7.1 wheels](https://pytorch.org/get-started/previous-versions/). Use the NVIDIA build only with a supported GPU and driver. CPU works but is slower. The validated 720p example used an 8 GB NVIDIA GPU; memory use also depends on image size and observation density. Model stages are transferred to the GPU sequentially to reduce peak weight memory.

## 3. Run CaM-PDA and enter paths

```console
python run.py
```

The program asks for:

1. English or 中文.
2. An included example, or your own RGB and registered sensor-depth paths.
3. Optional camera calibration and reference view.
4. The folder where results will be saved.
5. A model-storage folder or two existing model files.
6. Automatic device selection or CPU.

Paths are entered **after the program starts**. You do not edit Python variables. Quoted paths, spaces, `~` and environment variables are accepted. Output runs use unique child folders, so reusing a storage location keeps earlier results.

After package installation, `cam-pda` or `python -m cam_pda` starts the same prompt. The current source checkout includes nine examples. The v0.3.0 wheel includes calibrated blade32 and its blade20 reference, so the optional multiview example is available after package installation. Use the source checkout for the additional material gallery cases. SciPy is installed automatically as a standard dependency for the continuous solver; no custom C/CUDA extension or separate CUDA Toolkit build is required.

## Model storage

The inference checkpoint is approximately 392 MB and the frozen prior approximately 390 MB. The program asks where to store them before downloading. Already valid files are reused. Both are SHA256-checked against `src/cam_pda/resources/models.json`.

To run completely offline, choose **Use two existing weight files** and enter:

- `cam_pda_v1.pt` from the [model release](https://github.com/emp1y-fs/CaM-PDA/releases/tag/v0.1.0).
- `depth_anything_v2_vitb.pth` from the [official PDA model host](https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth).

Application v0.3.0 uses the same retained model as the original model release. An application update does not require a new checkpoint.

While the repository is private, downloading its model needs an account with repository access. In the automatic-download flow, enter **your own** authorized GitHub username when prompted. Sign in with Git Credential Manager first; the program reads its saved credential in memory, without saving a token in project files. A `GH_TOKEN` environment variable is also supported for automation. For public releases, the username can be left blank. Manual download and local-file selection remain available.

For automated installation:

```console
cam-pda download --cache-dir weights --github-user YOUR_GITHUB_USERNAME
```

The runtime prompt remembers selected storage locations. Preferences are a small JSON file under `%APPDATA%/cam-pda` on Windows or `~/.config/cam-pda` on Linux (respecting `XDG_CONFIG_HOME`). `CAM_PDA_SETTINGS` can select a different preferences file. Images and depth arrays are never stored in this preferences file.

## Input checklist

- RGB and depth must have the same resolution and already be registered.
- NPY depth is a two-dimensional floating-point array in metres.
- Sensor PNG is a single-channel integer array; select the correct units when prompted.
- Zero depth means missing observations; at least 17 valid observations are needed.
- Camera JSON must describe the aligned grid. Without calibration, export depth only.
- Reference images must show the same static scene with overlap. Unsuitable registration produces a recorded single-view fallback.

See [API.md](API.md) for exact file contracts and optional CLI automation. Use full paths if the current terminal directory differs from your data directory. `run.py` locates its own package and examples even when launched from another folder.

## Developer checks

```console
python -m pip install ".[test]" build
python -m pytest -q
python -m build
```

Unit checks do not download model weights. Real-model regression evidence is documented separately in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
