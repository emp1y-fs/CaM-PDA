import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from reproduction.paper import extract_verified, verify
from reproduction.final_benchmark import load_record, records, score
from reproduction.evaluate import array_sha


def manifest_for(path, name='depth.npy'):
    return dict(files=[dict(path=name, bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())])


def test_archive_verified_before_destination_is_created(tmp_path):
    original = tmp_path/'depth.npy'
    np.save(original, np.ones((10,10), np.float32))
    manifest = manifest_for(original)
    archive = tmp_path/'data.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('depth.npy', b'x'*original.stat().st_size)
    destination = tmp_path/'extracted'
    with pytest.raises(ValueError, match='changed'):
        extract_verified(archive, destination, manifest)
    assert not destination.exists()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.write(original, 'depth.npy')
    assert extract_verified(archive, destination, manifest)['verified_files'] == 1
    with pytest.raises(ValueError, match='new destination'):
        extract_verified(archive, destination, manifest)


def test_archive_rejects_unexpected_and_unsafe_paths(tmp_path):
    original = tmp_path/'depth.npy';np.save(original, np.zeros((4,4), np.float32))
    manifest = manifest_for(original, '../depth.npy')
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.write(original, '../depth.npy')
    with pytest.raises(ValueError, match='Unsafe'):
        extract_verified(archive, tmp_path/'new', manifest)
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.write(original, 'depth.npy');stream.writestr('extra.txt','unexpected')
    with pytest.raises(ValueError, match='members differ'):
        extract_verified(archive, tmp_path/'new', manifest_for(original))


def test_complete_final_identities():
    expected = dict(dreds11520=11520, nyu654=654, icl80=80, std4500=4500, heldout600=600)
    assert sum(len(records(name)) for name in expected) == 17354
    for name, count in expected.items():
        rows = records(name)
        assert len(rows) == count
        assert {row['dataset'] for row in rows} == {name}


def test_reference_and_frozen_mask_changes_are_rejected(tmp_path):
    arrays = dict(rgb=np.zeros((3,392,392),np.uint8), sensor_m=np.ones((392,392),np.float32),
                  gt_m=np.ones((392,392),np.float32), sampled=np.ones((392,392),bool))
    row = dict(sample_id='test', input='data/test.npz', cache_sha256='unused',
               array_sha256={key:array_sha(value) for key,value in arrays.items()})
    path = tmp_path/row['input'];path.parent.mkdir()
    np.savez_compressed(path, **arrays)
    assert load_record(tmp_path,row)['rgb'].shape == (3,392,392)
    arrays['gt_m'][0,0] = 2
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='Changed array: test/gt_m'):
        load_record(tmp_path,row)


def test_metrics_keep_fixed_domain_and_nonfinite_predictions():
    gt = np.full((8,8),2,np.float32);full=np.ones((8,8),bool)
    arrays = dict(gt_m=gt, full=full, sensor_hole=full,
                  challenging_material=full, depth_boundary=full)
    result = score(np.full((8,8),3,np.float32),arrays)['full']
    assert result['AbsRel'] == .5 and result['MAE_m'] == 1 and result['pixels'] == 64
    prediction=gt.copy();prediction[0,0]=np.nan
    result=score(prediction,arrays)['full']
    assert result['finite'] is False and result['pixels'] == 64
