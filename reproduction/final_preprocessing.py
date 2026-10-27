"""Registered final-test resize and reference-mask functions.

Extracted unchanged from the authors' frozen preprocessing sources; the
portable caller substitutes only user-selected paths. See source hashes below.
"""
import pathlib
from pathlib import Path
import cv2
import Imath
import OpenEXR
import numpy as np
from PIL import Image

def full_frame_letterbox(record: dict, size: int, seed: int):
    """Resize the complete frame with aspect-ratio preserving zero padding."""
    sensor = read_exr_depth(record["sensor_depth"])
    gt = read_exr_depth(record["gt_depth"])
    challenge = material_mask(record["instance_mask"], record["meta"])
    height, width = sensor.shape
    if gt.shape != (height, width) or challenge.shape != (height, width):
        raise ValueError(
            f"Depth/mask alignment mismatch: sensor={sensor.shape}, gt={gt.shape}, mask={challenge.shape}"
        )

    rgb = Image.open(record["rgb"]).convert("RGB")
    rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb, dtype=np.uint8).copy()

    scale = min(size / max(width, 1), size / max(height, 1))
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    rgb_resized = np.asarray(
        Image.fromarray(rgb).resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    sensor_resized = cv2.resize(
        sensor, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    gt_resized = cv2.resize(
        gt, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    challenge_resized = cv2.resize(
        challenge, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )

    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    sensor = np.zeros((size, size), dtype=np.float32)
    gt = np.zeros((size, size), dtype=np.float32)
    challenge = np.zeros((size, size), dtype=np.uint8)
    rows = slice(top, top + resized_height)
    cols = slice(left, left + resized_width)
    rgb[rows, cols] = rgb_resized
    sensor[rows, cols] = sensor_resized
    gt[rows, cols] = gt_resized
    challenge[rows, cols] = challenge_resized

    sensor[~np.isfinite(sensor) | (sensor <= 0.0)] = 0.0
    gt[~np.isfinite(gt) | (gt <= 0.0)] = 0.0
    return rgb, sensor.astype(np.float32), gt.astype(np.float32), challenge.astype(np.uint8)

def boundary_proxies_numpy(gt, valid):
    """Return (positive, known), preserving any leading dimensions and H,W.

    All comparisons deliberately use FP32, matching the training Torch path.
    Image borders and neighborhoods containing invalid GT are unknown. Missing
    GT is not silently classified as a negative boundary label.
    """
    gt=np.asarray(gt,dtype=np.float32)
    valid=np.asarray(valid,dtype=np.bool_)
    if gt.shape!=valid.shape or gt.ndim<2:
        raise ValueError('GT and valid must share a shape ending in H,W')
    positive=np.zeros(gt.shape,dtype=np.bool_)
    known=np.zeros(gt.shape,dtype=np.bool_)
    h,w=gt.shape[-2:]
    if h<3 or w<3:return positive,known
    finite_valid=valid & np.isfinite(gt) & (gt>0)
    safe=np.where(finite_valid,gt,np.float32(0))
    center=safe[...,1:-1,1:-1]
    complete=np.ones(center.shape,dtype=np.bool_)
    jump=np.zeros(center.shape,dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            complete &= finite_valid[...,dy:dy+h-2,dx:dx+w-2]
            if (dy,dx)!=(1,1):
                delta=np.abs(safe[...,dy:dy+h-2,dx:dx+w-2]-center)
                np.maximum(jump,delta,out=jump)
    known[...,1:-1,1:-1]=complete
    threshold=np.maximum(np.float32(.01),np.float32(.02)*center)
    positive[...,1:-1,1:-1]=complete & (jump>threshold)
    return positive,known

def boundary_domain(gt, valid):
    """Pure NumPy geometric-edge evaluation domain; no prediction is read."""
    return boundary_proxies_numpy(gt,valid)[0]

def exr_depth(path,channel):
    f=OpenEXR.InputFile(str(path))
    try:
        h=f.header();assert channel in h['channels'], (path,channel,list(h['channels']))
        w=h['dataWindow'];shape=(w.max.y-w.min.y+1,w.max.x-w.min.x+1)
        return np.frombuffer(f.channel(channel,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(shape).copy()
    finally:f.close()

def material_ids(path):
    selected=[]
    for line in pathlib.Path(path).read_text().splitlines():
        fields=line.split()
        if len(fields)<3:continue
        if int(fields[5] if len(fields)>5 else fields[2]) in (2,3):selected.append(int(fields[0]))
    return selected

def dreds_material(instance_path,meta_path):
    ids=np.rint(exr_depth(instance_path,'R')*255).astype(np.int32)
    return np.isin(ids,material_ids(meta_path)).astype(np.uint8)

def std_arrays(files):
    sensor=exr_depth(files['sensor_depth'],'Y');gt=exr_depth(files['gt_depth'],'Y');ids=np.asarray(Image.open(files['instance_mask']))
    assert ids.ndim==2 and ids.shape==gt.shape
    h,w=gt.shape;rgb=Image.open(files['rgb']).convert('RGB');assert rgb.size==(sensor.shape[1],sensor.shape[0])
    assert abs(w/h-sensor.shape[1]/sensor.shape[0])<1e-9
    sensor=cv2.resize(sensor,(w,h),interpolation=cv2.INTER_NEAREST)
    rgb=rgb.resize((w,h),Image.Resampling.BILINEAR)
    challenge=np.isin(ids,material_ids(files['meta'])).astype(np.uint8)
    scale=min(392/w,392/h);rw,rh=max(1,min(392,round(w*scale))),max(1,min(392,round(h*scale)));left,top=(392-rw)//2,(392-rh)//2
    out_rgb=np.zeros((392,392,3),np.uint8);out_sensor=np.zeros((392,392),np.float32);out_gt=np.zeros_like(out_sensor);out_challenge=np.zeros((392,392),np.uint8)
    rows=slice(top,top+rh);cols=slice(left,left+rw)
    out_rgb[rows,cols]=np.asarray(rgb.resize((rw,rh),Image.Resampling.BILINEAR),np.uint8)
    for src,out in ((sensor,out_sensor),(gt,out_gt),(challenge,out_challenge)):out[rows,cols]=cv2.resize(src,(rw,rh),interpolation=cv2.INTER_NEAREST)
    out_sensor[~np.isfinite(out_sensor)|(out_sensor<=0)]=0;out_gt[~np.isfinite(out_gt)|(out_gt<=0)]=0
    return out_rgb,out_sensor,out_gt,out_challenge

def read_exr_depth(path):
    return exr_depth(path, 'B')


def material_mask(instance_path, meta_path):
    return dreds_material(instance_path, meta_path)
