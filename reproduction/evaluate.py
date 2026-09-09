"""Evaluate the retained checkpoint on complete, hash-verified test subsets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import read_records, verify_manifests
from .metrics import depth_metrics

REGIONS = ('full', 'sensor_hole', 'challenging_material', 'depth_boundary')
METRICS = ('AbsRel', 'MAE_m', 'RMSE_m')


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def load_record(root, row):
    with np.load(root / row['prepared_file'], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in row['array_sha256']}
    for key, expected in row['array_sha256'].items():
        if array_sha(arrays[key]) != expected:
            raise ValueError(f"Prepared array differs: {row['id']} / {key}")
    return arrays


def summarize(records, expected):
    if {r['id'] for r in records} != {r['id'] for r in expected} or len(records) != len(expected):
        raise ValueError('Missing, duplicate or unexpected evaluation frames')
    output = {}
    for dataset in sorted({r['dataset'] for r in expected}):
        output[dataset] = {}
        for region in REGIONS:
            values = [r['metrics'][region] for r in records
                      if r['dataset'] == dataset and r['metrics'][region] is not None]
            if any(not r['finite'] for r in values):
                raise ValueError('Nonfinite predictions cannot be excluded from an aggregate')
            output[dataset][region] = dict(frames=len(values), pixels=sum(v['pixels'] for v in values),
                **{m: float(np.mean([v[m] for v in values])) if values else None for m in METRICS})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mde-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=('dreds110', 'nyu654', 'icl80'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-predictions', action='store_true', help='Save numerical depth and color previews for every frame')
    args = parser.parse_args(argv)
    verify_manifests()
    rows = [r for r in read_records('paper_test.json') if args.dataset is None or r['dataset'] == args.dataset]
    if args.output.exists():
        parser.error('Choose a new output directory')
    # Verify the entire subset before creating a model or a partial result directory.
    for row in rows:
        load_record(args.data_root, row)
    from cam_pda import CaMPDA
    model = CaMPDA(checkpoint=args.checkpoint, mde_checkpoint=args.mde_checkpoint,
                  device=args.device, allow_download=False)
    args.output.mkdir(parents=True)
    records = []
    with (args.output / 'frames.jsonl').open('x', encoding='utf8') as stream:
        for row in rows:
            data = load_record(args.data_root, row)
            prediction = model.predict(data['rgb'], data['sensor_m'], seed=row['seed'], sampled_mask=data['sampled'])
            metrics = {key: depth_metrics(prediction.depth_m, data['gt_m'], data[key]) for key in REGIONS}
            record = dict(id=row['id'], dataset=row['dataset'], metrics=metrics,
                          prediction_sha256=array_sha(prediction.depth_m))
            if args.save_predictions:
                from PIL import Image
                from cam_pda.io import depth_preview
                target = args.output / 'predictions' / row['id']
                target.mkdir(parents=True)
                np.save(target / 'depth_m.npy', prediction.depth_m, allow_pickle=False)
                Image.fromarray(depth_preview(prediction.depth_m)).save(target / 'depth_color.png')
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            print(f"{len(records)}/{len(rows)} {row['id']}", flush=True)
    result = summarize(records, rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
