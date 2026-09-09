"""Verify released sample identities and optionally prepared training files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_records(name):
    return json.loads((ROOT / 'manifests' / name).read_text(encoding='utf8'))['records']


def verify_manifests():
    initial = read_records('initial_train_val.json')
    geometry = read_records('geometry_train_val.json')
    test = read_records('paper_test.json')
    if Counter(r['split'] for r in initial) != {'train': 672, 'val': 84}:
        raise ValueError('Initial split differs from 672/84')
    if Counter(r['split'] for r in geometry) != {'train': 854, 'val': 108}:
        raise ValueError('Geometry split differs from 854/108')
    if Counter(r['dataset'] for r in test) != {'dreds110': 110, 'nyu654': 654, 'icl80': 80}:
        raise ValueError('Test set differs from 110/654/80')
    for records in (initial, geometry, test):
        keys = [(r['dataset'], r['id']) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate frame identity')
    for source in {r['dataset'] for r in initial}:
        rows = [r for r in initial if r['dataset'] == source]
        if Counter(r['split'] for r in rows) != {'train': 96, 'val': 12}:
            raise ValueError('Initial per-source count differs: ' + source)
        groups = {s: {r['group'] for r in rows if r['split'] == s} for s in ('train', 'val')}
        if groups['train'] & groups['val']:
            raise ValueError('Training/development groups overlap: ' + source)
    hs = [r for r in geometry if r['source'] == 'hypersim']
    scenes = {s: {r['scene'] for r in hs if r['split'] == s} for s in ('train', 'val')}
    if scenes['train'] & scenes['val'] or len(scenes['val']) != 2:
        raise ValueError('Hypersim scene separation differs')
    if any(r['group'] == 'lr_kt3' for r in initial if r['dataset'] == 'icl'):
        raise ValueError('ICL test trajectory enters adaptation')
    if any(r['source'] != 'lr_kt3' for r in test if r['dataset'] == 'icl80'):
        raise ValueError('Unexpected ICL test trajectory')
    nyu_indices = [r['original_labeled_index_one_based'] for r in test if r['dataset'] == 'nyu654']
    if len(set(nyu_indices)) != 654 or any(not 1 <= i <= 1449 for i in nyu_indices):
        raise ValueError('Invalid NYUv2 original image mapping')
    keys = {'rgb', 'sensor_m', 'sampled', 'gt_m', 'full', 'sensor_hole', 'challenging_material', 'depth_boundary'}
    if any(set(r['array_sha256']) != keys for r in test):
        raise ValueError('Incomplete evaluation array hashes')
    return {'initial': len(initial), 'geometry': len(geometry), 'test': len(test)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-root', type=Path)
    args = parser.parse_args(argv)
    result = verify_manifests()
    if args.initial_root:
        for row in read_records('initial_train_val.json'):
            for name, expected in row['prepared_sha256'].items():
                path = args.initial_root / row['prepared_directory'] / name
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError('Prepared file hash mismatch: ' + str(path))
        result['prepared_initial_files_verified'] = 756 * 4
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
