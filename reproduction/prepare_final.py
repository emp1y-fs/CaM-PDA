"""Prepare final-protocol inputs from distributor data, verifying every array."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np

from .final_benchmark import records, load_record
from .evaluate import array_sha

REPO = Path(__file__).resolve().parents[1]


def native_arrays(row, raw_root):
    from .final_preprocessing import full_frame_letterbox, std_arrays, boundary_domain
    files = {k: str(Path(raw_root)/v) for k, v in row['source_files'].items()}
    if row['dataset'] == 'dreds11520':
        rgb, sensor, gt, material = full_frame_letterbox(files, 392, row['seed'])
    else:
        rgb, sensor, gt, material = std_arrays(files)
    valid = np.flatnonzero(sensor > 0)
    if len(valid) < 17:
        raise ValueError('Insufficient measurements: no target omission or synthetic top-up')
    sampled = np.zeros(sensor.shape, bool)
    sampled.flat[np.random.default_rng(row['seed']).choice(valid, min(50000, len(valid)), replace=False)] = True
    full = np.isfinite(gt) & (gt > 1e-5)
    return dict(rgb=rgb.transpose(2,0,1).copy(), sensor_m=sensor, sampled=sampled, gt_m=gt,
                full=full, sensor_hole=full & (sensor <= 0),
                challenging_material=full & material.astype(bool), depth_boundary=boundary_domain(gt, full))


def check_arrays(arrays, row):
    for key, expected in row['array_sha256'].items():
        if key not in arrays or array_sha(arrays[key]) != expected:
            raise ValueError(f"Preprocessing differs: {row['sample_id']}/{key}. No approximation is saved.")


def save(arrays, row, root):
    check_arrays(arrays, row)
    target = Path(root)/row['input']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        load_record(root, row)
        return
    path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.npz.partial', delete=False) as temporary:
            path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
        # Atomic publication of a same-filesystem file; never replace a user file.
        try:
            os.link(path, target)
        except FileExistsError:
            load_record(root, row)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['dreds11520','std4500','nyu654','icl80'])
    parser.add_argument('--root', type=Path, required=True, help='Destination parent of data/<dataset>/')
    parser.add_argument('--raw-root', type=Path, help='Parent of graspnet_generate_test_catnovel_0202 or real_data_novel')
    parser.add_argument('--mat', type=Path, help='Existing official NYUv2 labelled MAT; otherwise download it')
    parser.add_argument('--paper-data', type=Path, help='Verified source-data download containing benchmark_masks/')
    parser.add_argument('--sample-id', action='append', help='Prepare only a named recorded target; repeat for multiple examples')
    args = parser.parse_args(argv)
    rows = records(args.dataset)
    if args.sample_id:
        selected = set(args.sample_id)
        rows = [r for r in rows if r['sample_id'] in selected]
        if len(rows) != len(selected):
            parser.error('Unknown or duplicate target identity')
    if args.dataset in ('dreds11520','std4500'):
        if args.raw_root is None:
            parser.error('--raw-root is required for distributor archives')
        for number, row in enumerate(rows, 1):
            save(native_arrays(row, args.raw_root), row, args.root)
            if number % 100 == 0 or number == len(rows):
                print(f'Verified prepared inputs: {number}/{len(rows)}', flush=True)
        return 0
    from .download import fetch, sha256, unpack
    assets = json.loads((REPO/'reproduction/test_assets.json').read_text(encoding='utf8'))
    legacy = json.loads((REPO/'reproduction/manifests/paper_test.json').read_text(encoding='utf8'))['records']
    legacy = [r for r in legacy if r['dataset'] == args.dataset]
    if args.dataset == 'icl80':
        archive = fetch(assets['datasets']['icl80'], args.root/'.downloads')
        with tempfile.TemporaryDirectory(prefix='icl80-') as temporary:
            unpack(archive, temporary, [r['prepared_file'] for r in legacy])
            by_id = {r['id'].split('/')[-1]: r for r in legacy}
            for row in rows:
                old = by_id[row['sample_id'].removeprefix('icl80_')]
                with np.load(Path(temporary)/old['prepared_file'], allow_pickle=False) as source:
                    arrays = {k: source[k] for k in source.files}
                arrays['rgb'] = arrays['rgb'].transpose(2,0,1).copy()
                save(arrays, row, args.root)
        print(f'Prepared and fingerprint-verified {len(rows)} final ICL targets.')
        return 0
    if args.paper_data is None:
        parser.error('--paper-data is required for the final NYUv2 masks')
    from .paper import verify
    verify(args.paper_data)
    item = assets['nyu_source']
    mat = args.mat or fetch(item, args.root/'.downloads')
    if mat.stat().st_size != item['bytes'] or sha256(mat) != item['sha256']:
        raise ValueError('Official NYUv2 MAT failed verification')
    import h5py
    from .nyu import from_mat
    by_id = {r['id'].split('/')[-1]: r for r in legacy}
    with h5py.File(mat, 'r') as dataset:
        for number, row in enumerate(rows, 1):
            old = by_id[row['sample_id'].removeprefix('nyu654_')]
            with np.load(args.paper_data/'benchmark_masks/nyu654'/(row['sample_id']+'.npz'), allow_pickle=False) as source:
                masks = {k: source[k] for k in source.files}
            arrays = from_mat(dataset, old, masks)
            arrays['rgb'] = arrays['rgb'].transpose(2,0,1).copy()
            save(arrays, row, args.root)
            if number % 100 == 0 or number == len(rows):
                print(f'Verified final NYUv2 inputs: {number}/{len(rows)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
