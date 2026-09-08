"""Public pipeline state must agree with refined and fallback outputs."""
import threading
import numpy as np
import pytest
from cam_pda.io import CameraIntrinsics
from cam_pda.types import DepthResult, _array_sha


@pytest.mark.parametrize('case', ['refined', 'no_registration', 'no_support', 'solver_failure'])
def test_multiview_output_and_target_state(monkeypatch, case):
    from cam_pda.pipeline import CaMPDA
    import cam_pda.multiview as mv
    rgb = np.zeros((6, 8, 3), np.uint8)
    depth = np.ones((6, 8), np.float32)
    mask = np.ones_like(depth, dtype=bool)
    condition = np.zeros((3, 6, 8), np.float32)
    single = DepthResult(depth, mask, mask, mask, condition, 0., 1.,
                         dict(depth_sha256=_array_sha(depth), runtime_s=0.1))
    target_routing = {'target': True}
    model = object.__new__(CaMPDA)
    model._lock = threading.RLock()
    calls = []

    def predict(*args, **kwargs):
        calls.append(1)
        model.last_routing = target_routing if len(calls) == 1 else {'reference': True}
        return single

    model.predict = predict
    camera = CameraIntrinsics(20, 20, 3.5, 2.5, 8, 6)
    source = dict(rgb=rgb, raw_m=depth, K=camera.matrix, frame_id=1)
    registration = None if case == 'no_registration' else {'T_source_to_target': np.eye(4)}
    monkeypatch.setattr(mv, 'select_reference', lambda *args: (registration, source, []))

    def refine(*args, **kwargs):
        if case == 'solver_failure':
            raise RuntimeError('Unconverged solve: 0.1')
        correction = np.full_like(depth, .002 if case == 'refined' else 0.)
        return depth + correction, correction, dict(
            status='refined' if case == 'refined' else 'exact_single_view_fallback',
            support_pixels=[12 if case == 'refined' else 0])

    monkeypatch.setattr(mv, 'refine_reference', refine)
    result = model.predict_multiview(rgb, depth, camera, [dict(rgb=rgb, raw_m=depth, camera=camera)])
    expected = depth + np.float32(.002) if case == 'refined' else depth
    np.testing.assert_array_equal(result.depth_m, expected)
    assert result.metadata['depth_sha256'] == _array_sha(expected)
    assert result.metadata['fused_pixels'] == (depth.size if case == 'refined' else 0)
    assert result.metadata['multiview_status'] == ('fused' if case == 'refined' else 'single_view_fallback')
    assert result.accepted is mask and result.condition is condition
    assert model.last_routing is target_routing
    assert len(calls) == (1 if case == 'no_registration' else 2)
    if case == 'solver_failure':
        assert result.metadata['fallback_reason'] == 'continuous_solver_failed'
    np.testing.assert_array_equal(depth, np.ones((6, 8), np.float32))
