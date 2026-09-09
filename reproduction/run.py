"""Download the paper test subsets and weights, evaluate, and export results."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import uuid

from .audit import read_records, verify_manifests
from .download import fetch, github_token, sha256, unpack
from .evaluate import load_record

HERE = Path(__file__).resolve().parent
DATASETS = ('dreds110', 'nyu654', 'icl80')


def write_report(output, provenance):
    summary = json.loads((output/'summary.json').read_text())
    records = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    fields = ('dataset', 'region', 'frames', 'pixels', 'AbsRel', 'MAE_m', 'RMSE_m')
    with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, regions in summary.items():
            for region, values in regions.items():
                writer.writerow(dict(dataset=dataset, region=region, **values))
    with (output/'per_frame.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ('dataset', 'id', 'region', 'pixels', 'AbsRel', 'MAE_m', 'RMSE_m')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for region, values in row['metrics'].items():
                if values is not None:
                    writer.writerow(dict(dataset=row['dataset'], id=row['id'], region=region,
                                         **{key:values[key] for key in fields[3:]}))
    lines = ['# CaM-PDA test results', '',
             f'Completed {len(records)} fixed test frames using the released checkpoint.', '',
             '| Dataset | Frames | AbsRel | MAE (m) | RMSE (m) |', '|---|---:|---:|---:|---:|']
    for dataset, regions in summary.items():
        values = regions['full']
        lines.append(f"| {dataset} | {values['frames']} | {values['AbsRel']:.8f} | {values['MAE_m']:.8f} | {values['RMSE_m']:.8f} |")
    lines += ['', 'The table uses per-frame macro averages on the fixed full reference domain. AbsRel is dimensionless; MAE and RMSE are in metres.',
              '', 'Regional metrics are in `summary.csv`; per-frame scores are in `per_frame.csv`. Numerical depths and color previews are under `predictions/` unless `--metrics-only` was selected.',
              '', 'Depth colors are independently normalized previews. Compare numerical arrays and metrics, not preview colors. Point clouds are not exported because the benchmark records do not provide calibrated intrinsics.',
              '', 'Hardware and attention kernels can affect numerical results. Paper scores refer to the recorded Linux CUDA environment; this run reports its own measurements.', '']
    (output/'results.md').write_text('\n'.join(lines), encoding='utf8')
    provenance.update(status='complete', evaluated_frames=len(records))
    (output/'run.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf8')
    print('\n'+ '\n'.join(lines[4:4+2+len(summary)]), flush=True)
    print(f'\nResults: {output.resolve()}\nOpen results.md or summary.csv to see the scores.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('./cam_pda_test'), help='All downloads, prepared data, weights and results stay here')
    parser.add_argument('--dataset', choices=('all',)+DATASETS, default='all')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--github-user', help='Optional Git Credential Manager account during private review')
    parser.add_argument('--prepare-only', action='store_true', help='Download and verify data and weights without inference')
    parser.add_argument('--metrics-only', action='store_true', help='Skip saving predicted depth maps')
    parser.add_argument('--nyu-mat', type=Path, help='Reuse an existing official labeled MAT instead of downloading it')
    parser.add_argument('--checkpoint', type=Path, help='Reuse the released CaM-PDA weight file')
    parser.add_argument('--mde-checkpoint', type=Path, help='Reuse the frozen monocular weight file')
    args = parser.parse_args(argv)
    verify_manifests()
    datasets = DATASETS if args.dataset == 'all' else (args.dataset,)
    if 'nyu654' in datasets:
        try:
            import h5py
        except ImportError:
            parser.error('NYUv2 needs h5py. Install once with: python -m pip install ".[benchmark]"')
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE/'test_assets.json').read_text())
    token = github_token(args.github_user)
    rows = [row for row in read_records('paper_test.json') if row['dataset'] in datasets]
    print(f'Storage folder: {root}\nTest targets: {len(rows)}', flush=True)
    if 'nyu654' in datasets:
        print('NYUv2 is downloaded from its official MAT source (2.97 GB) and converted automatically; RGB and depth are not mirrored by this project.', flush=True)
    for dataset in datasets:
        subset = [row for row in rows if row['dataset'] == dataset]
        complete = all((root/'data'/row['prepared_file']).is_file() for row in subset)
        if complete:
            for row in subset:
                load_record(root/'data', row)
            print(f'Verified prepared subset: {dataset} ({len(subset)} frames)', flush=True)
            continue
        archive = fetch(manifest['datasets'][dataset], root/'downloads', token)
        if dataset == 'nyu654':
            from .nyu import prepare
            unpack(archive, root/'nyu_masks', [row['prepared_file'] for row in subset])
            if args.nyu_mat:
                mat = args.nyu_mat.expanduser().resolve()
                if not mat.is_file() or mat.stat().st_size != manifest['nyu_source']['bytes'] or sha256(mat) != manifest['nyu_source']['sha256']:
                    raise ValueError('Existing NYUv2 MAT failed source verification')
            else:
                mat = fetch(manifest['nyu_source'], root/'downloads')
            prepare(mat, root/'nyu_masks', root/'data', subset)
        else:
            unpack(archive, root/'data', [row['prepared_file'] for row in subset])
            # Each source keeps its own license notice.
            (root/'data'/'NOTICE.txt').replace(root/'data'/dataset/'NOTICE.txt')
        for row in subset:
            load_record(root/'data', row)
    from cam_pda.weights import manifest as model_manifest
    weights = model_manifest()
    paths = []
    for key, explicit in (('ca_m_pda', args.checkpoint), ('mde', args.mde_checkpoint)):
        item = weights[key]
        if explicit:
            path = explicit.expanduser().resolve()
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise ValueError('Existing weight failed verification: '+str(path))
        else:
            path = fetch(item, root/'weights', token)
        paths.append(path)
    print('All selected input arrays and both model weights passed SHA256 verification.', flush=True)
    if args.prepare_only:
        print('Preparation complete. Rerun without --prepare-only to evaluate.', flush=True)
        return
    from .evaluate import main as evaluate
    import torch
    output = root/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    arguments = ['--data-root', str(root/'data'), '--checkpoint', str(paths[0]),
                 '--mde-checkpoint', str(paths[1]), '--output', str(output), '--device', args.device]
    if args.dataset != 'all':
        arguments += ['--dataset', args.dataset]
    if not args.metrics_only:
        arguments += ['--save-predictions']
    provenance = dict(status='running', datasets=list(datasets), expected_frames=len(rows),
        python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, requested_device=args.device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        test_manifest_sha256=sha256(HERE/'manifests/paper_test.json'),
        weight_sha256={key:item['sha256'] for key,item in weights.items()},
        saved_predictions=not args.metrics_only)
    try:
        evaluate(arguments)
        write_report(output, provenance)
    except BaseException:
        if output.is_dir():
            provenance['status'] = 'incomplete'
            (output/'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
        raise


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'CaM-PDA test stopped: {error}', file=sys.stderr)
        sys.exit(1)
