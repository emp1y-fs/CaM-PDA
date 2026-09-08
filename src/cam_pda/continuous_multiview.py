"""Experimental edge-aware continuous multiview correction; all distances m.

This module accepts predictions, RGB and calibrated transforms, never GT or ROIs.
The optimized variable is a correction to a depth prior, not a smoothed depth map.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class Config:
    gate_m: float = .03
    gate_relative: float = .01
    roundtrip_px: float = 2.
    source_edge_relative: float = .015
    smoothness: float = 64.
    target_weight: float = 1.
    hole_weight: float = .5
    reference_weight: float = 1.
    iterations: int = 200


def evidence(target, rgb, source, source_rgb, K, source_K, T_m, config=Config()):
    """Inverse sample a source, then check reprojection cycle and visibility.

    T_m maps source camera to target camera, with translation in metres.
    Bilinear sampling is rejected near source depth discontinuities; target->source
    projection chooses visible support without forward-warp raster holes.
    """
    source=np.asarray(source,dtype=np.float32)
    if source.ndim!=2 or not np.isfinite(source).all() or np.any(source<=0):
        raise ValueError('Finite positive predicted source depth required; raw sensor holes are not predictions')
    if source_rgb.shape!=(*source.shape,3):raise ValueError('Unaligned source RGB and prediction')
    h,w=target.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    ray=np.stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)),-1)
    points=ray*target[...,None]
    inv=np.linalg.inv(T_m)
    ps=points@inv[:3,:3].T+inv[:3,3]
    zs=ps[...,2]
    u=(source_K[0,0]*ps[...,0]/np.maximum(zs,1e-6)+source_K[0,2]).astype(np.float32)
    v=(source_K[1,1]*ps[...,1]/np.maximum(zs,1e-6)+source_K[1,2]).astype(np.float32)
    def sample(a):return cv2.remap(a,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    sampled=sample(source)
    ps_observed=np.stack(((u-source_K[0,2])*sampled/source_K[0,0],
                          (v-source_K[1,2])*sampled/source_K[1,1],sampled),-1)
    pt=ps_observed@T_m[:3,:3].T+T_m[:3,3]
    z=pt[...,2]
    ux=K[0,0]*pt[...,0]/np.maximum(z,1e-6)+K[0,2]
    vy=K[1,1]*pt[...,1]/np.maximum(z,1e-6)+K[1,2]
    cycle=np.sqrt((ux-xx)**2+(vy-yy)**2)
    kernel=np.ones((3,3),np.uint8)
    local_span=cv2.dilate(source,kernel)-cv2.erode(source,kernel)
    edge=(local_span>config.source_edge_relative*np.maximum(source,.1)).astype(np.float32)
    source_edge=sample(cv2.dilate(edge,kernel))
    residual=(z-target).astype(np.float32)
    gate=config.gate_m+config.gate_relative*target
    valid=(np.isfinite(residual)&(target>0)&(zs>0)&(sampled>0)&(z>0)&
           (u>=1)&(u<source.shape[1]-2)&(v>=1)&(v<source.shape[0]-2)&
           (cycle<config.roundtrip_px)&(np.abs(residual)<gate)&(source_edge<.01))
    color=np.mean(np.abs(sample(source_rgb.astype(np.float32)/255)-rgb.astype(np.float32)/255),axis=-1)
    weight=valid.astype(np.float32)*np.exp(-.5*(residual/gate)**2)*np.exp(-.5*(cycle/config.roundtrip_px)**2)
    weight*=1/(1+(color/.12)**2)
    return np.where(valid,residual,0).astype(np.float32),weight.astype(np.float32)


def refine(target,rgb,raw_m,references,K,config=Config()):
    target=np.asarray(target,dtype=np.float32)
    if target.ndim!=2 or not np.isfinite(target).all() or np.any(target<=0):
        raise ValueError('Finite positive target depth required')
    if rgb.shape!=(*target.shape,3) or raw_m.shape!=target.shape:
        raise ValueError('Unaligned target inputs')
    if min(config.target_weight,config.hole_weight)<=0 or config.smoothness<0:
        raise ValueError('Positive prior and nonnegative regularization required')
    numerator=np.zeros_like(target); total=np.zeros_like(target); supports=[]
    for ref in references:
        delta,weight=evidence(target,rgb,ref['depth_m'],ref['rgb'],K,ref['K'],ref['T_m'],config)
        numerator+=weight*delta;total+=weight
        supports.append(int(np.count_nonzero(weight)))
    metadata=dict(config=asdict(config),support_pixels=supports,raw_anchor_restoration=False,
                  method='continuous edge-aware depth correction',ground_truth_used=False)
    if not np.any(total):
        return target.copy(),np.zeros_like(target),dict(metadata,status='exact_single_view_fallback')
    # Cap aggregate support to avoid arbitrary changes of prior strength with view count.
    normalization=np.maximum(total,1.)
    numerator=numerator/normalization*config.reference_weight
    data_weight=total/normalization*config.reference_weight
    prior=np.where(raw_m>0,config.target_weight,config.hole_weight).astype(np.float32)
    scale=config.smoothness*(target.shape[1]/640)**2
    def links(axis):
        a,b=(target[:,:-1],target[:,1:]) if axis==1 else (target[:-1],target[1:])
        ca,cb=(rgb[:,:-1],rgb[:,1:]) if axis==1 else (rgb[:-1],rgb[1:])
        relative=np.abs(a-b)/np.maximum(np.minimum(a,b),.1)
        color=np.mean(np.abs(ca.astype(np.float32)-cb.astype(np.float32)),axis=-1)/255
        return (scale*np.exp(-(relative/.006)**2)/(1+(color/.1)**2)*(relative<.015)).astype(np.float32)
    wx,wy=links(1),links(0)
    diagonal=prior+data_weight
    diag=diagonal.copy();diag[:,:-1]+=wx;diag[:,1:]+=wx;diag[:-1]+=wy;diag[1:]+=wy
    def matvec(flat):
        x=flat.reshape(target.shape);result=diagonal*x
        dx=wx*(x[:,:-1]-x[:,1:]);dy=wy*(x[:-1]-x[1:])
        result[:,:-1]+=dx;result[:,1:]-=dx;result[:-1]+=dy;result[1:]-=dy
        return result.ravel()
    size=target.size
    matrix=LinearOperator((size,size),matvec=matvec,dtype=np.float32)
    precond=LinearOperator((size,size),matvec=lambda x:x/diag.ravel(),dtype=np.float32)
    correction,info=cg(matrix,numerator.ravel(),M=precond,rtol=1e-5,atol=1e-8,maxiter=config.iterations)
    if info<0 or not np.isfinite(correction).all():raise RuntimeError('Continuous correction solver failed')
    relative_residual=float(np.linalg.norm(matvec(correction)-numerator.ravel())/max(np.linalg.norm(numerator),1e-12))
    # Do not silently publish an unconverged solve as a valid reconstruction.
    if info>0 and relative_residual>1e-3:raise RuntimeError(f'Unconverged solve: {relative_residual}')
    correction=correction.reshape(target.shape)
    result=target+correction
    if not np.isfinite(result).all() or np.any(result<=0):raise RuntimeError('Invalid refined depth')
    metadata.update(status='refined',cg_info=int(info),solver_relative_residual=relative_residual,
                    changed_pixels=int(np.count_nonzero(result!=target)),
                    correction_abs_p95_mm=float(np.percentile(np.abs(correction),95)*1000))
    return result.astype(np.float32),correction,metadata
