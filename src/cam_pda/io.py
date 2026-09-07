"""Image, metric-depth and calibrated point-cloud file interfaces."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self):
        if not np.isfinite([self.fx,self.fy,self.cx,self.cy]).all() or min(self.fx,self.fy)<=0:
            raise ValueError('Focal lengths must be positive and all camera parameters finite.')
        if min(self.width,self.height)<=0:
            raise ValueError('Camera width and height must be positive.')

    @property
    def matrix(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float64)

    @classmethod
    def from_json(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{key:data[key] for key in ('fx','fy','cx','cy','width','height')})

    def validate_shape(self,shape):
        if tuple(shape)!=(self.height,self.width):
            raise ValueError('Calibration resolution must match the aligned RGB-D images.')


def read_rgb(path):
    # Do not rotate EXIF automatically: depth must keep exactly the same grid.
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def read_depth(path,*,scale=None):
    """NPY is float metres; integer PNG requires an explicit units-to-metres scale."""
    path=Path(path)
    if path.suffix.lower()=='.npy':
        value=np.load(path,allow_pickle=False)
        if not np.issubdtype(value.dtype,np.floating):
            raise ValueError('NPY depth must contain floating point metres.')
        if scale is not None and scale!=1:
            raise ValueError('NPY depth already uses metres; do not specify another scale.')
        scale=1.
    elif path.suffix.lower()=='.png':
        with Image.open(path) as image:value=np.asarray(image).copy()
        if value.ndim!=2 or not np.issubdtype(value.dtype,np.integer):
            raise ValueError('Sensor PNG must be a single-channel integer depth image.')
        if scale is None:
            raise ValueError('Specify depth_scale, e.g. 0.001 for millimetre PNG.')
    else:
        raise ValueError('Supported depth files: .npy (float metres), .png (integer sensor units).')
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Depth scale must be positive and finite.')
    value=np.asarray(value,dtype=np.float32)*np.float32(scale)
    if value.ndim!=2 or not np.isfinite(value).all() or (value<0).any():
        raise ValueError('Depth must be finite, nonnegative HW; missing observations are zero.')
    return value


def depth_preview(depth):
    import cv2
    valid=np.isfinite(depth)&(depth>0)
    if not valid.any():return np.zeros((*depth.shape,3),np.uint8)
    low,high=np.percentile(depth[valid],[2,98])
    mapped=np.clip((depth-low)/max(float(high-low),1e-6),0,1)
    colors=cv2.applyColorMap((mapped*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1].copy()
    colors[~valid]=0
    return colors


def write_ply(path,depth_m,rgb,camera,*,stride=1):
    camera.validate_shape(depth_m.shape)
    if type(stride) is not int or stride<1:raise ValueError('Point stride must be a positive integer.')
    if rgb.shape!=(*depth_m.shape,3) or rgb.dtype!=np.uint8:
        raise ValueError('Point-cloud color must be RGB uint8 on the same grid as depth.')
    v,u=np.mgrid[0:camera.height:stride,0:camera.width:stride]
    z=depth_m[::stride,::stride]
    valid=np.isfinite(z)&(z>0)
    x=(u-camera.cx)*z/camera.fx
    y=(v-camera.cy)*z/camera.fy
    points=np.empty(int(valid.sum()),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for name,array in [('x',x),('y',y),('z',z)]:points[name]=array[valid]
    colors=rgb[::stride,::stride][valid]
    for i,name in enumerate(('red','green','blue')):points[name]=colors[:,i]
    header=('ply\nformat binary_little_endian 1.0\ncomment units metres; camera x right, y down, z forward\n'
            f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'));stream.write(points.tobytes())
    return len(points)


def export_result(result,rgb,output,camera=None,*,point_stride=1):
    """Save float depth, depth preview, masks and optional calibrated binary PLY."""
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output folder must be empty; use a new folder for each prediction.')
    output.mkdir(parents=True,exist_ok=True)
    if camera is not None:camera.validate_shape(result.depth_m.shape)
    np.save(output/'depth_m.npy',result.depth_m,allow_pickle=False)
    Image.fromarray(depth_preview(result.depth_m)).save(output/'depth_color.png')
    Image.fromarray(rgb).save(output/'rgb.png')
    Image.fromarray(result.accepted.astype(np.uint8)*255).save(output/'accepted_mask.png')
    meta=dict(result.metadata)
    rounded=np.rint(result.depth_m.astype(np.float64)*1000)
    # Never silently clip long-range metric output to the uint16 PNG range.
    if np.all((rounded>=1)&(rounded<=65535)):
        Image.fromarray(rounded.astype(np.uint16)).save(output/'depth_mm.png')
        meta['depth_png']='uint16 millimetres; NPY preserves original float32 values'
    else:
        meta['depth_png']='omitted: depths outside positive uint16 millimetre range'
    if camera is not None:
        meta['point_count']=write_ply(output/'point_cloud.ply',result.depth_m,rgb,camera,stride=point_stride)
        meta['camera']=vars(camera)
        meta['point_stride']=point_stride
    meta['point_cloud_coordinates']='metres; x right, y down, z forward'
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
    return output
