"""Evaluate the released formal checkpoint on the final proof's frozen inputs.

Source datasets must be acquired separately. This command never substitutes
the legacy DREDS110 suite for the complete 11,520-frame CatNovel evaluation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

from .evaluate import array_sha, summarize
from .metrics import depth_metrics

REPO = Path(__file__).resolve().parents[1]


def records(dataset):
    index = json.loads((REPO/'paper/manifests/index.json').read_text(encoding='utf8'))
    item = index['datasets'][dataset]
    content = (REPO/item['file']).read_bytes()
    if hashlib.sha256(content).hexdigest() != item['sha256']:
        raise ValueError('Final protocol manifest changed')
    raw = gzip.decompress(content)
    if hashlib.sha256(raw).hexdigest() != item['uncompressed_sha256']:
        raise ValueError('Final protocol content changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != item['records'] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate target identities')
    return rows


def load_record(root, row):
    path = Path(root)/row['input']
    if not path.is_file():
        raise ValueError('Missing prepared target: '+row['input']+'. See docs/BENCHMARKS.md.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    expected = row.get('array_sha256', {})
    # Arrays, not ZIP container bytes, determine numerical equivalence.
    if expected:
        for key, value in expected.items():
            if key not in arrays or array_sha(arrays[key]) != value:
                raise ValueError(f"Changed array: {row['sample_id']}/{key}")
    elif digest != row['cache_sha256']:
        raise ValueError('Changed prepared archive: '+row['sample_id'])
    if arrays['rgb'].shape != (3, 392, 392) or arrays['rgb'].dtype != np.uint8:
        raise ValueError('Expected recorded CHW uint8 RGB at 392 x 392')
    for key in ('sensor_m', 'gt_m'):
        if arrays[key].shape != (392, 392) or arrays[key].dtype != np.float32:
            raise ValueError('Expected recorded float32 metre depths')
    if arrays['sampled'].dtype != bool or arrays['sampled'].shape != (392, 392):
        raise ValueError('Invalid frozen observation mask')
    return arrays


def score(prediction, arrays):
    if 'full' not in arrays:
        # Procedural files have a different role-label schema. Keep it explicit.
        masks = dict(full=arrays['valid'], sensor_hole=arrays['valid'] & ~(arrays['sensor_m'] > 0),
                     challenging_material=arrays['valid'] & arrays['reflective'] & arrays['reflective_known'],
                     depth_boundary=arrays['valid'] & arrays['edge'] & arrays['edge_known'])
    else:
        masks = {k: arrays[k] for k in ('full','sensor_hole','challenging_material','depth_boundary')}
    return {key: depth_metrics(prediction, arrays['gt_m'], mask) for key, mask in masks.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['dreds11520','nyu654','icl80','std4500','heldout600'])
    parser.add_argument('--data-root', required=True, type=Path, help='Parent of data/<dataset>/<sample_id>.npz')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mde-checkpoint', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args(argv)
    rows = records(args.dataset)
    if not args.verify_only and (args.output is None or args.output.exists()):
        parser.error('Select a new --output directory')
    for row in rows:
        load_record(args.data_root, row)
    print(f'Verified {len(rows)} frozen {args.dataset} inputs.', flush=True)
    if args.verify_only:
        return 0
    from cam_pda import CaMPDA
    model = CaMPDA(checkpoint=args.checkpoint, mde_checkpoint=args.mde_checkpoint, device=args.device)
    args.output.mkdir(parents=True)
    results = []
    with (args.output/'frames.jsonl').open('x', encoding='utf8') as stream:
        for row in rows:
            arrays = load_record(args.data_root, row)
            prediction = model.predict(np.moveaxis(arrays['rgb'], 0, -1).copy(), arrays['sensor_m'],
                                       seed=row['seed'], sampled_mask=arrays['sampled'])
            result = dict(id=row['sample_id'], dataset=row['dataset'], metrics=score(prediction.depth_m, arrays),
                          prediction_sha256=array_sha(prediction.depth_m))
            stream.write(json.dumps(result, allow_nan=False)+'\n');stream.flush()
            results.append(result)
            if args.save_predictions:
                folder = args.output/'predictions';folder.mkdir(exist_ok=True)
                np.save(folder/(row['sample_id']+'.npy'), prediction.depth_m, allow_pickle=False)
            print(f'{len(results)}/{len(rows)} {row["sample_id"]}', flush=True)
    summary = summarize(results, [dict(id=r['sample_id'], dataset=r['dataset']) for r in rows])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf8')
    (args.output/'run.json').write_text(json.dumps(dict(dataset=args.dataset,frames=len(rows),
        protocol='paper/manifests/index.json',checkpoint_sha256=prediction.metadata['inference_checkpoint_sha256'],
        source_checkpoint_sha256=prediction.metadata['source_checkpoint_sha256'],
        scope='Formal checkpoint only; not all retrained comparison arms'),indent=2)+'\n',encoding='utf8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
