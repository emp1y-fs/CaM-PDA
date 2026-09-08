import numpy as np
from cam_pda.continuous_multiview import refine,Config


def scene():
    target=np.full((48,64),2.,np.float32)
    rgb=np.full((48,64,3),100,np.uint8)
    K=np.array([[60.,0,32],[0,60,24],[0,0,1]])
    return target,rgb,K


def test_no_support_preserves_original_exactly():
    target,rgb,K=scene()
    result,_,meta=refine(target,rgb,np.zeros_like(target),[],K)
    assert np.array_equal(result,target)
    assert meta['status']=='exact_single_view_fallback'


def test_anchor_outlier_not_reinjected_and_smooth_reference_improves():
    target,rgb,K=scene();raw=target.copy();raw[20,20]=3.
    ref=dict(depth_m=target+.02,rgb=rgb,K=K,T_m=np.eye(4))
    result,delta,_=refine(target,rgb,raw,[ref],K)
    assert result[20,20]<2.02
    assert np.mean(np.abs(result[8:-8,8:-8]-2.02))<.02
    assert np.max(np.abs(np.diff(delta[8:-8,8:-8],axis=1)))<.001


def test_disoccluded_inconsistent_depth_has_no_influence():
    target,rgb,K=scene()
    ref=dict(depth_m=target+1.,rgb=rgb,K=K,T_m=np.eye(4))
    result,_,_=refine(target,rgb,target,[ref],K)
    assert np.array_equal(result,target)


def test_correction_does_not_cross_existing_depth_step():
    target,rgb,K=scene();target[:,32:]=3.
    source=target.copy();source[:,:32]+=.02
    ref=dict(depth_m=source,rgb=rgb,K=K,T_m=np.eye(4))
    result,_,_=refine(target,rgb,target,[ref],K)
    assert np.max(np.abs(result[:,35:]-target[:,35:]))<1e-5
    assert np.mean(result[10:-10,10:20]-target[10:-10,10:20])>.001


def test_nonfinite_reference_rejected_before_solver():
    import pytest
    target,rgb,K=scene();source=target.copy();source[15,15]=np.nan
    ref=dict(depth_m=source,rgb=rgb,K=K,T_m=np.eye(4))
    with pytest.raises(ValueError,match='Finite positive predicted source'):
        refine(target,rgb,target,[ref],K)


def test_registration_millimetres_converted_without_changing_caller_transform():
    from cam_pda.multiview import refine_reference
    target,rgb,K=scene();T=np.eye(4);T[2,3]=20.
    before=T.copy()
    result,_,meta=refine_reference(target,rgb,target,target,rgb,K,T)
    assert np.array_equal(T,before)
    assert meta['status']=='refined' and np.mean(result[10:-10,10:-10]-target[10:-10,10:-10])>.001
    assert not meta['raw_anchor_restoration']


def test_nonfinite_raw_observations_are_treated_as_holes():
    from cam_pda.multiview import refine_reference
    target,rgb,K=scene();raw=np.full_like(target,np.inf)
    first=refine_reference(target,rgb,raw,target+.02,rgb,K,np.eye(4))[0]
    second=refine_reference(target,rgb,np.zeros_like(raw),target+.02,rgb,K,np.eye(4))[0]
    assert np.array_equal(first,second) and np.isinf(raw).all()


def test_nonrigid_registration_rejected():
    import pytest
    from cam_pda.multiview import refine_reference
    target,rgb,K=scene();T=np.eye(4);T[0,0]=2
    with pytest.raises(ValueError,match='rigid'):
        refine_reference(target,rgb,target,target,rgb,K,T)
