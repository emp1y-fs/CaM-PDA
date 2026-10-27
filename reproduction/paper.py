"""Download, verify and inspect the versioned final-paper source data."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def data_manifest():
    return json.loads((REPO/'paper/data_manifest.json').read_text(encoding='utf8'))


def verify(root, manifest=None):
    root = Path(root).resolve()
    manifest = data_manifest() if manifest is None else manifest
    for row in manifest['files']:
        path = root/row['path']
        if not path.resolve().is_relative_to(root):
            raise ValueError('Data path escapes the selected directory')
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Missing or changed paper-data file: '+row['path'])
    return dict(verified_files=len(manifest['files']), bytes=sum(r['bytes'] for r in manifest['files']))


def extract_verified(archive, destination, manifest):
    """Reject changed, duplicate, linked or unlisted files before publishing data."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Extraction requires a new destination directory')
    expected = {r['path']: r for r in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate data-manifest path')
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) != len(expected) or {i.filename for i in members} != set(expected):
            raise ValueError('Archive members differ from the frozen manifest')
        for item in members:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe archive path')
            if item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive links and directory entries are not supported')
            if item.file_size != expected[item.filename]['bytes']:
                raise ValueError('Archive member size differs')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.paper-data-', dir=destination.parent) as temporary:
            stage = Path(temporary)/'data'
            for item in members:
                target = stage/item.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            verify(stage, manifest)
            stage.rename(destination)
    return verify(destination, manifest)


def download(root):
    from .download import fetch
    root = Path(root).resolve()
    if root.exists():
        return verify(root)
    manifest = data_manifest()
    archive = fetch(manifest['archive'], root.parent/'.paper-downloads')
    return extract_verified(archive, root, manifest)


def geometry(root):
    """Replay point-to-point surface agreement at the archived rigid pose."""
    from scipy.spatial import cKDTree
    folder = Path(root)/'paper_blade/geometry'
    with np.load(folder/'registered_arrays.npz', allow_pickle=False) as source:
        prediction, reference = source['aligned_mm'], source['reference_mm']
    if prediction.shape != (25656, 3) or reference.shape != (137383, 3):
        raise ValueError('Unexpected point counts')
    reference_tree = cKDTree(reference)
    forward = reference_tree.query(prediction)[0]
    reverse = cKDTree(prediction).query(reference)[0]
    result = dict(unit='mm', prediction_points=len(prediction), reference_points=len(reference),
                  prediction_to_reference_mean=float(forward.mean()),
                  reference_to_prediction_mean=float(reverse.mean()),
                  symmetric_mean=float((forward.mean()+reverse.mean())/2),
                  interpretation='Visible-surface agreement at an archived, prediction-fitted rigid pose; not independent metrological accuracy.')
    result['same_capture'] = {}
    for path in sorted(folder.glob('*_valid_roi.npz')):
        with np.load(path, allow_pickle=False) as source:
            points = source['aligned_mm']
        distances = reference_tree.query(points)[0]
        result['same_capture'][path.stem] = dict(points=len(points), mean_mm=float(distances.mean()))
    return result


def routing(root):
    result = {}
    for block in (6, 8, 10):
        with np.load(Path(root)/f'routing_frame32/routing_{block}.npz', allow_pickle=False) as source:
            probability = source['probabilities']
            active = source['activations'].astype(bool)
        if probability.shape != (2442, 3) or active.shape != probability.shape:
            raise ValueError('Unexpected routing grid')
        if not np.array_equal(active, probability >= .5):
            raise ValueError('Activation threshold does not match the archive')
        state = active[:, 0].astype(int) + 2*active[:, 1] + 4*active[:, 2]
        result[str(block)] = dict(tokens=len(state), experts=['R', 'N', 'E'],
            activation_percent=(active.mean(axis=0)*100).tolist(),
            state_bits='R + 2*N + 4*E', eight_state_counts=np.bincount(state, minlength=8).tolist())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'verify', 'geometry', 'routing'])
    parser.add_argument('--root', type=Path, required=True, help='Directory containing the extracted source-data files')
    args = parser.parse_args(argv)
    if args.action == 'download':
        result = download(args.root)
    else:
        check = verify(args.root)
        result = check if args.action == 'verify' else globals()[args.action](args.root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
