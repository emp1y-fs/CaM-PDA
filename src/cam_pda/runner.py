"""File-based workflows shared by the terminal application and Python users."""
from datetime import datetime
from pathlib import Path
import json
import re
import uuid

import numpy as np

from .io import CameraIntrinsics, export_result, read_depth, read_rgb
from .types import prepare_inputs


def load_example(folder):
    """Keep the shipped example's original seed and optional sampling mask."""
    folder = Path(folder).expanduser().resolve()
    provenance_file = folder / 'provenance.json'
    provenance = json.loads(provenance_file.read_text(encoding='utf-8')) if provenance_file.is_file() else {}
    return dict(rgb_path=folder/'rgb.png', depth_path=folder/'sensor_depth.npy',
                camera_path=folder/'camera.json' if (folder/'camera.json').is_file() else None,
                seed=provenance.get('seed', 0),
                sampled_mask_path=folder/'sampled_mask.npy' if (folder/'sampled_mask.npy').is_file() else None)


def _load_files(rgb_path, depth_path, camera_path=None, depth_scale=None):
    rgb = read_rgb(Path(rgb_path).expanduser())
    depth = read_depth(Path(depth_path).expanduser(), scale=depth_scale)
    if rgb.shape != (*depth.shape, 3):
        raise ValueError('RGB and depth must already be aligned and have the same resolution.')
    camera = CameraIntrinsics.from_json(Path(camera_path).expanduser()) if camera_path else None
    if camera:
        camera.validate_shape(depth.shape)
    return rgb, depth, camera


def run_from_paths(rgb_path, depth_path, output_dir, *, camera_path=None,
                   depth_scale=None, seed=0, sampled_mask_path=None,
                   model=None, model_options=None, progress=print):
    """Save one prediction under a unique child folder and return that path.

    Input validation runs before the model is loaded.
    """
    if model is not None and model_options:
        raise ValueError('Pass either an existing model or model_options, not both.')
    progress('1/4  Reading RGB-D and checking calibration...')
    rgb, depth, camera = _load_files(rgb_path, depth_path, camera_path, depth_scale)
    sampled = np.load(Path(sampled_mask_path).expanduser(), allow_pickle=False) if sampled_mask_path else None
    prepare_inputs(rgb, depth, seed=seed, sampled_mask=sampled)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A unique child directory keeps repeated runs separate.
    name = re.sub(r'[^\w.-]', '_', Path(rgb_path).stem)[:40].strip('. ') or 'scene'
    run = root / f'{datetime.now():%Y%m%d-%H%M%S}_{name}_{uuid.uuid4().hex[:8]}'
    run.mkdir()
    try:
        progress('2/4  Loading the verified CaM-PDA model...')
        if model is None:
            from .pipeline import CaMPDA
            model = CaMPDA(**(model_options or {}))
        progress('3/4  Predicting dense metric depth...')
        result = model.predict(rgb, depth, seed=seed, sampled_mask=sampled)
        progress('4/4  Saving depth maps' + (' and colored point cloud...' if camera else '...'))
        export_result(result, rgb, run, camera)
    except Exception:
        if not any(run.iterdir()):
            run.rmdir()
        raise
    progress(f'Saved: {run}')
    return run


def run_example(folder, output_dir, **options):
    """Run an example with its recorded seed and optional sampling mask."""
    return run_from_paths(**load_example(folder), output_dir=output_dir, **options)
