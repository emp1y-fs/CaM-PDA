import json
import numpy as np
import pytest
from PIL import Image
from cam_pda.types import prepare_inputs,DepthResult
from cam_pda.io import read_depth,CameraIntrinsics,write_ply,export_result


def test_sampling_reproducible_and_input_untouched():
    rgb=np.zeros((20,20,3),np.uint8);depth=np.ones((20,20),np.float32)
    first=prepare_inputs(rgb,depth,maximum_samples=50,seed=7)
    second=prepare_inputs(rgb,depth,maximum_samples=50,seed=7)
    assert np.array_equal(first[2],second[2]) and first[2].sum()==50
    first[1][0,0]=999
    assert depth[0,0]==1
    with pytest.raises(ValueError):prepare_inputs(rgb,depth*0)
    with pytest.raises(TypeError):prepare_inputs(rgb,depth.astype(np.float64))


def test_depth_units_and_calibrated_cloud(tmp_path):
    path=tmp_path/'sensor.png';Image.fromarray(np.full((5,5),2000,np.uint16)).save(path)
    with pytest.raises(ValueError):read_depth(path)
    depth=read_depth(path,scale=.001)
    assert depth.dtype==np.float32 and np.all(depth==2)
    camera=CameraIntrinsics(2,2,2,2,5,5);rgb=np.full((5,5,3),42,np.uint8)
    ply=tmp_path/'cloud.ply';assert write_ply(ply,depth,rgb,camera)==25
    content=ply.read_bytes();start=content.index(b'end_header\n')+len(b'end_header\n')
    dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')]
    points=np.frombuffer(content[start:],dtype=dtype)
    assert (points[12]['x'],points[12]['y'],points[12]['z'])==(0,0,2)
    assert points[12]['r']==42
    with pytest.raises(ValueError):CameraIntrinsics(2,2,2,2,6,5).validate_shape(depth.shape)


def test_export_rejects_mixed_runs_and_does_not_clip(tmp_path):
    z=np.full((5,5),70,np.float32);mask=np.ones((5,5),bool);rgb=np.zeros((5,5,3),np.uint8)
    result=DepthResult(z,mask,mask,mask,np.zeros((3,5,5),np.float32),0,1,{})
    output=export_result(result,rgb,tmp_path/'run')
    assert not (output/'depth_mm.png').exists()
    assert np.all(np.load(output/'depth_m.npy')==70)
    with pytest.raises(FileExistsError):export_result(result,rgb,output)


def test_download_redirect_strips_credentials():
    from urllib.request import Request
    from cam_pda.weights import _SafeRedirect
    request=Request('https://api.github.com/a',headers={'Authorization':'Bearer synthetic-test-value'})
    redirected=_SafeRedirect().redirect_request(request,None,302,'Found',{},'https://release-assets.githubusercontent.com/x')
    assert redirected.get_header('Authorization') is None


def test_cpu_knn_all_accepted_and_one_missing():
    import torch
    from cam_pda._vendor.pda.completion import DepthCompletion
    engine=DepthCompletion.__new__(DepthCompletion);torch.nn.Module.__init__(engine)
    engine.device=torch.device('cpu')
    sparse=torch.arange(1,26,dtype=torch.float32).reshape(1,5,5)
    mask=torch.ones_like(sparse,dtype=torch.bool)
    completed=engine.kss_completer(sparse,sparse,~mask,mask,K=5)
    assert torch.equal(completed,sparse)
    mask[0,2,2]=False
    completed=engine.kss_completer(torch.where(mask,sparse,0),sparse,~mask,mask,K=5)
    assert torch.isfinite(completed).all() and torch.equal(completed[mask],sparse[mask])
    assert abs(float(completed[0,2,2])-13)<.01


def test_multiview_preserves_accepted_raw_anchor():
    from cam_pda.multiview import fuse_reference
    target=np.full((5,5),2,np.float32);source=np.full((5,5),2.03,np.float32)
    mask=np.zeros((5,5),bool);mask[2,2]=True;raw=target.copy();raw[2,2]=1.9
    fused,reliable,_=fuse_reference(target,raw,mask,source,np.eye(3),np.eye(4))
    assert fused[2,2]==raw[2,2] and not reliable[2,2]
    assert np.allclose(fused[~mask],2.03,atol=1e-6)
