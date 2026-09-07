"""Cross-platform command line entry point."""
import argparse,json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(prog='cam-pda',description='CaM-PDA: aligned RGB-D to metric depth and colored point cloud')
    sub=parser.add_subparsers(dest='command',required=True)
    download=sub.add_parser('download',help='Download and SHA256-verify inference weights')
    download.add_argument('--cache-dir',type=Path)
    download.add_argument('--github-user',help='Use this account from Git Credential Manager for the private release')
    for name in ('infer','example','serve'):
        item=sub.add_parser(name)
        item.add_argument('--checkpoint',type=Path)
        item.add_argument('--mde-checkpoint',type=Path)
        item.add_argument('--cache-dir',type=Path)
        item.add_argument('--device',default='auto')
        if name=='serve':
            item.add_argument('--port',type=int,default=7860)
            item.add_argument('--examples',type=Path,default=Path('examples'))
            item.add_argument('--output',type=Path,default=Path('outputs/web'))
            continue
        item.add_argument('--output',type=Path,required=True)
        item.add_argument('--seed',type=int)
        item.add_argument('--point-stride',type=int,default=1)
        item.add_argument('--save-routing',action='store_true')
        item.add_argument('--references',type=Path,nargs='*',default=[],help='Aligned RGB-D example folders for pose-free refinement')
        if name=='example':
            item.add_argument('folder',type=Path,help='Folder with rgb.png, sensor_depth.npy and camera.json')
        else:
            item.add_argument('--rgb',type=Path,required=True)
            item.add_argument('--depth',type=Path,required=True)
            item.add_argument('--depth-scale',type=float)
            item.add_argument('--camera',type=Path,help='Calibration JSON; required for point-cloud output')
    args=parser.parse_args(argv)
    if args.command=='download':
        import os
        from .weights import resolve_weights
        if args.github_user:os.environ['CAM_PDA_GITHUB_USER']=args.github_user
        paths=resolve_weights(cache_dir=args.cache_dir)
        print(json.dumps({'checkpoint':str(paths[0]),'mde_checkpoint':str(paths[1]),'verified':True}))
        return 0
    config={key:getattr(args,key) for key in ('checkpoint','mde_checkpoint','device','cache_dir')}
    if args.command=='serve':
        from .web import serve
        return serve(config,args.examples,args.output,args.port)
    from . import CaMPDA
    from .io import read_rgb,read_depth,CameraIntrinsics,export_result
    import numpy as np
    sampled=None
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty. Select a new output folder.')
    if args.command=='example':
        rgb=read_rgb(args.folder/'rgb.png');depth=read_depth(args.folder/'sensor_depth.npy')
        camera=CameraIntrinsics.from_json(args.folder/'camera.json') if (args.folder/'camera.json').exists() else None
        provenance=json.loads((args.folder/'provenance.json').read_text()) if (args.folder/'provenance.json').exists() else {}
        if args.seed is None:args.seed=provenance.get('seed',0)
        if (args.folder/'sampled_mask.npy').exists():sampled=np.load(args.folder/'sampled_mask.npy',allow_pickle=False)
    else:
        rgb=read_rgb(args.rgb);depth=read_depth(args.depth,scale=args.depth_scale)
        camera=CameraIntrinsics.from_json(args.camera) if args.camera else None
    if args.seed is None:args.seed=0
    if camera:camera.validate_shape(depth.shape)
    refs=[]
    for folder in args.references:
        refs.append(dict(rgb=read_rgb(folder/'rgb.png'),raw_m=read_depth(folder/'sensor_depth.npy'),
                         camera=CameraIntrinsics.from_json(folder/'camera.json')))
    if refs and camera is None:parser.error('Multiview requires target calibration.')
    model=CaMPDA(**config)
    if refs and args.save_routing:parser.error('Save routing for single-view calls; multiview also runs a separate reference prediction.')
    result=model.predict_multiview(rgb,depth,camera,refs,seed=args.seed) if refs else model.predict(rgb,depth,seed=args.seed,sampled_mask=sampled)
    export_result(result,rgb,args.output,camera,point_stride=args.point_stride)
    if args.save_routing:
        import numpy as np
        for block,arrays in model.last_routing.items():np.savez_compressed(args.output/f'routing_block{block}.npz',**arrays)
    print(json.dumps({'output':str(args.output.resolve()),**result.metadata},indent=2))
    return 0
