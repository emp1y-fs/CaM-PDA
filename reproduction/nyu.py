"""Reconstruct the frozen NYUv2 grid from the distributor's labeled MAT file."""
from pathlib import Path

import cv2
import numpy as np

from .evaluate import array_sha


def from_mat(dataset, row, masks):
    index = row['original_labeled_index_one_based'] - 1
    rgb = np.asarray(dataset['images'][index]).transpose(2, 1, 0)
    mode = {'area': cv2.INTER_AREA, 'linear': cv2.INTER_LINEAR}[row['rgb_resize_interpolation']]
    arrays = dict(masks)
    arrays['rgb'] = np.zeros((392, 392, 3), np.uint8)
    arrays['rgb'][49:343] = cv2.resize(rgb, (392, 294), interpolation=mode)
    for source, key in (('rawDepths', 'sensor_m'), ('depths', 'gt_m')):
        depth = np.asarray(dataset[source][index], np.float32).T
        # Preserve the original millimetre PNG round-trip and FP16 cache precision.
        mm = np.where(np.isfinite(depth) & (depth > 0), depth * 1000., 0.)
        mm = np.rint(np.clip(mm, 0, 65535)).astype(np.uint16)
        grid = np.zeros((392, 392), np.float32)
        grid[49:343] = cv2.resize(mm.astype(np.float32) / 1000., (392, 294), interpolation=cv2.INTER_NEAREST)
        arrays[key] = grid.astype(np.float16).astype(np.float32)
    for key, expected in row['array_sha256'].items():
        if array_sha(arrays[key]) != expected:
            raise ValueError(f"NYUv2 conversion differs: {row['id']} / {key}")
    return arrays


def prepare(mat_path, mask_root, destination, rows):
    import h5py
    from .evaluate import load_record
    destination = Path(destination)
    with h5py.File(mat_path, 'r') as dataset:
        for number, row in enumerate(rows, 1):
            target = destination / row['prepared_file']
            if target.is_file():
                load_record(destination, row)
                continue
            with np.load(Path(mask_root) / row['prepared_file'], allow_pickle=False) as source:
                masks = {key: source[key] for key in source.files}
            arrays = from_mat(dataset, row, masks)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.npz.partial')
            with partial.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            partial.replace(target)
            if number % 100 == 0 or number == len(rows):
                print(f'NYUv2 preparation: {number}/{len(rows)}', flush=True)
