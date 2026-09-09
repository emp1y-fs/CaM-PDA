from pathlib import Path
import sys
import numpy as np
import pytest

# Reproduction tools are distributed in the source archive, not installed as
# part of the prediction API.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit import verify_manifests
from reproduction.evaluate import summarize, array_sha, load_record
from reproduction.metrics import depth_metrics


def test_manifest_split_contract():
    assert verify_manifests() == dict(initial=756, geometry=962, test=844)


def test_metrics_macro_mean_and_failure_retention():
    # Unequal pixel counts distinguish frame-macro from pixel-weighted means.
    one = depth_metrics(np.full((8,8), 2.), np.ones((8,8)), np.ones((8,8),bool))
    two = depth_metrics(np.full((8,16), 3.), np.ones((8,16)), np.ones((8,16),bool))
    rows = [dict(id=str(i), dataset='test', metrics={k:v for k in
        ('full','sensor_hole','challenging_material','depth_boundary')}) for i,v in enumerate((one,two))]
    assert summarize(rows, rows)['test']['full']['AbsRel'] == 1.5
    assert depth_metrics(np.ones((3,3)),np.ones((3,3)),np.ones((3,3),bool)) is None
    with pytest.raises(ValueError, match='Missing, duplicate'):
        summarize(rows[:1], rows)
    rows[0]['metrics']['full'] = dict(one, finite=False)
    with pytest.raises(ValueError, match='Nonfinite'):
        summarize(rows, rows)


def test_changed_reference_or_sample_mask_fails_hash_check(tmp_path):
    arrays = dict(sampled=np.ones((8,8),bool), gt_m=np.ones((8,8),np.float32))
    row = dict(id='f', prepared_file='f.npz', array_sha256={k:array_sha(v) for k,v in arrays.items()})
    np.savez(tmp_path/'f.npz', **arrays)
    assert np.array_equal(load_record(tmp_path,row)['gt_m'], arrays['gt_m'])
    arrays['sampled'][0,0] = False
    np.savez(tmp_path/'f.npz', **arrays)
    with pytest.raises(ValueError, match='Prepared array differs'):
        load_record(tmp_path,row)
