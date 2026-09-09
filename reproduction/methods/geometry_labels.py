"""Offline renderer-derived supervision; no RGB or sensor inputs for N/E labels."""
import numpy as np
from scipy.ndimage import uniform_filter,minimum_filter,maximum_filter,binary_dilation,binary_erosion
SIZE = 392

def camera_rays(row,h,w):
 m=np.array([[float(row[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)],np.float64)
 u=(np.arange(w)+.5)*2/w-1;v=1-(np.arange(h)+.5)*2/h
 uu,vv=np.meshgrid(u,v);r=np.stack([uu,vv,np.ones_like(uu)],-1)@m.T
 assert np.isfinite(r).all() and (r[...,2]<0).all()
 return r.astype(np.float32)

def geometry_targets(depth,norm,valid,instance):
 """Multi-scale normal dispersion with no depth jumps/instance mixing.

 Curvature here is a renderer-derived local geometric label, not object identity.
 """
 depth=depth.astype(np.float32);valid=valid.astype(bool);n=np.nan_to_num(norm.astype(np.float32))
 unit=np.linalg.norm(n,axis=2);good=valid&(unit>.975)&(unit<1.025)
 n=np.where(good[...,None],n/np.maximum(unit[...,None],1e-8),0)
 jump=np.zeros(depth.shape,bool);fold=np.zeros(depth.shape,bool);paired=good.copy()
 for axis in (0,1):
  a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1);a=tuple(a);b=tuple(b)
  ok=good[a]&good[b];delta=np.abs(depth[a]-depth[b]);scale=np.minimum(depth[a],depth[b])
  j=ok&(delta>np.maximum(.015,.025*scale));f=ok&((n[a]*n[b]).sum(-1)<np.cos(np.deg2rad(10)))
  jump[a]|=j;jump[b]|=j;fold[a]|=f;fold[b]|=f
 edge=binary_dilation(jump|fold,iterations=1)&valid
 edge_known=binary_erosion(good,iterations=1)
 near_edge=binary_dilation(jump|fold,iterations=3)
 angles=[];supports=[];pure=[]
 ids=instance.astype(np.int32)
 for size in (7,15,31,61):
  support=uniform_filter(good.astype(np.float32),size=size,mode='constant')
  mean=np.stack([uniform_filter(n[...,k],size=size,mode='constant') for k in range(3)],-1)/np.maximum(support[...,None],1e-6)
  # RMS angular dispersion approximates sqrt(2*(1-|mean normal|)).
  angular=np.sqrt(np.maximum(2*(1-np.clip(np.linalg.norm(mean,axis=2),0,1)),0))*180/np.pi
  stable=(minimum_filter(ids,size,mode='constant',cval=-999)==maximum_filter(ids,size,mode='constant',cval=-999))
  angles.append(angular);supports.append(support);pure.append(stable)
 # Semantic index -1 can be a perfectly valid wall/floor. Geometry, not the
 # presence of an object annotation, determines the negative label.
 plane=good&~near_edge&(supports[3]>.98)&pure[3]&(angles[0]<1)&(angles[1]<1.5)&(angles[2]<2)&(angles[3]<2)
 curved=np.zeros(depth.shape,bool)
 for j,(size,threshold) in enumerate(((7,6),(15,8),(31,6))):
  # A square window must exclude a discontinuity anywhere in that same window.
  clear=maximum_filter((jump|fold).astype(np.uint8),size=size,mode='constant')==0
  curved|=good&clear&(supports[j]>.98)&pure[j]&(angles[j]>threshold)&(angles[j]<35)
 assert not (plane&curved).any()
 return plane,curved,edge,edge_known

def simulate_observation(gt,valid,seed):
 """Fixed synthetic observation; no geometry/material target masks are read."""
 rng=np.random.default_rng(seed);h,w=gt.shape
 sensor=gt*(1+rng.normal(0,.006,gt.shape)).astype(np.float32)+rng.normal(0,.002,gt.shape).astype(np.float32)
 keep=valid&(gt>=.2)&(gt<=10)&(rng.random(gt.shape)>.20)
 yy,xx=np.mgrid[:h,:w]
 for _ in range(5):
  x,y=rng.uniform(0,w),rng.uniform(0,h);rx,ry=rng.uniform(.03,.10)*w,rng.uniform(.03,.10)*h
  keep &= ((xx-x)/rx)**2+((yy-y)/ry)**2>1
 bad=(rng.random(gt.shape)<.025)&keep
 sensor[bad]*=rng.uniform(.7,1.3,int(bad.sum())).astype(np.float32)
 return np.where(keep&np.isfinite(sensor)&(sensor>0),sensor,0).astype(np.float32)

def token_targets(plane,curved,edge,edge_known,refl_pos,refl_known,grid=37):
 # Adaptive pooling uses exactly the model's token grid.
 import torch
 import torch.nn.functional as F
 def pool(a):return F.adaptive_avg_pool2d(torch.from_numpy(a.astype(np.float32))[None,None],(grid,grid))[0,0].numpy()
 y=np.zeros((3,grid,grid),np.float32);k=np.zeros_like(y,bool)
 support=pool(refl_known);f=pool(refl_pos)/np.maximum(support,1e-6)
 k[0]=(support>=.5)&((f>=.25)|(f<=.05));y[0]=(f>=.25)
 pp,cc,ee=pool(plane),pool(curved),pool(edge)
 k[1]=((pp>=.9)|(cc>=.30))&(ee<.1);y[1]=(cc>=.30)
 es=pool(edge_known);ef=ee/np.maximum(es,1e-6)
 k[2]=(es>=.9)&((ef>=.10)|(ef<=.01));y[2]=(ef>=.10)
 return y,k
