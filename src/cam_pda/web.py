"""Local browser interface. Uploaded images stay on this computer."""
from pathlib import Path
from importlib.resources import files
import json,threading,uuid,zipfile
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory


def create_app(config,examples_dir,output_dir):
    from fastapi import FastAPI,Request,HTTPException
    from fastapi.responses import FileResponse,HTMLResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from .io import read_rgb,read_depth,CameraIntrinsics,export_result
    app=FastAPI(title='CaM-PDA',docs_url=None,redoc_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
    examples_dir=Path(examples_dir).resolve();output_dir=Path(output_dir).resolve()
    executor=ThreadPoolExecutor(max_workers=1)
    lock=threading.Lock();jobs={};state={'model':None,'busy':False}
    examples={p.name:p for p in examples_dir.glob('*') if all((p/file).is_file() for file in ('rgb.png','sensor_depth.npy'))}

    @app.middleware('http')
    async def local_only(request:Request,call_next):
        origin=request.headers.get('origin')
        if origin and origin!=f'{request.url.scheme}://{request.headers.get("host")}':
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Cross-origin requests are disabled.'},status_code=403)
        return await call_next(request)

    @app.get('/')
    def index():return HTMLResponse(files('cam_pda').joinpath('web_static/index.html').read_text(encoding='utf-8'))

    @app.get('/api/examples')
    def list_examples():return [{'id':key,'label':key.replace('_',' ')} for key in examples]

    def run(job_id,rgb,depth,camera,seed=0,sampled=None):
        try:
            jobs[job_id]={'status':'running'}
            if state['model'] is None:
                from . import CaMPDA
                state['model']=CaMPDA(**config)
            result=state['model'].predict(rgb,depth,seed=seed,sampled_mask=sampled)
            output=output_dir/job_id
            export_result(result,rgb,output,camera)
            with zipfile.ZipFile(output/'results.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in output.iterdir():
                    if path.name!='results.zip':archive.write(path,path.name)
            jobs[job_id]={'status':'complete','metadata':result.metadata,
                          'files':[p.name for p in output.iterdir()]}
        except Exception as error:
            # Surface actionable errors locally; never include request credentials.
            jobs[job_id]={'status':'failed','error':str(error)}
        finally:
            with lock:state['busy']=False

    @app.post('/api/predict')
    async def predict(request:Request):
        with lock:
            if state['busy']:raise HTTPException(409,'A prediction is already running. Wait for it to finish.')
            state['busy']=True
        try:
            # Stream the body with a hard limit before multipart parsing.
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>40*1024*1024:raise HTTPException(413,'Upload limit is 40 MB.')
            request._body=bytes(body)
            form=await request.form()
            seed=0;sampled=None
            example=form.get('example','')
            if example:
                if example not in examples:raise ValueError('Unknown example.')
                folder=examples[example]
                rgb=read_rgb(folder/'rgb.png');depth=read_depth(folder/'sensor_depth.npy')
                camera=CameraIntrinsics.from_json(folder/'camera.json') if (folder/'camera.json').exists() else None
                if (folder/'provenance.json').exists():seed=json.loads((folder/'provenance.json').read_text()).get('seed',0)
                if (folder/'sampled_mask.npy').exists():
                    import numpy as np
                    sampled=np.load(folder/'sampled_mask.npy',allow_pickle=False)
            else:
                with TemporaryDirectory(prefix='cam-pda-upload-') as temporary:
                    temporary=Path(temporary)
                    for key in ('rgb','depth','camera'):
                        upload=form.get(key)
                        if upload is None or not getattr(upload,'filename',None):
                            raise ValueError('Upload RGB, aligned sensor depth, and camera JSON.')
                        suffix=Path(upload.filename).suffix.lower()
                        allowed={'rgb':{'.png','.jpg','.jpeg'},'depth':{'.png','.npy'},'camera':{'.json'}}
                        if suffix not in allowed[key]:raise ValueError(f'Unsupported {key} file format.')
                        path=temporary/(key+suffix);path.write_bytes(await upload.read())
                    rgb=read_rgb(next(temporary.glob('rgb.*')))
                    depth=read_depth(next(temporary.glob('depth.*')),scale=float(form.get('depth_scale') or 0.001) if (temporary/'depth.png').exists() else None)
                    camera=CameraIntrinsics.from_json(temporary/'camera.json')
            if camera:camera.validate_shape(depth.shape)
            if depth.size>1920*1080:raise ValueError('The browser demo accepts at most 1920×1080 pixels; use Python for larger grids.')
            if rgb.shape!=(*depth.shape,3):raise ValueError('RGB and depth must already be aligned.')
            job_id=uuid.uuid4().hex
            jobs[job_id]={'status':'queued'}
            executor.submit(run,job_id,rgb,depth,camera,seed,sampled)
            return {'id':job_id}
        except Exception as error:
            with lock:state['busy']=False
            if isinstance(error,HTTPException):raise
            raise HTTPException(400,str(error)) from None

    @app.get('/api/jobs/{job_id}')
    def status(job_id:str):
        if job_id not in jobs:raise HTTPException(404,'Unknown job.')
        return jobs[job_id]

    @app.get('/outputs/{job_id}/{name}')
    def output(job_id:str,name:str):
        record=jobs.get(job_id,{})
        if record.get('status')!='complete' or name not in record.get('files',[]):raise HTTPException(404)
        return FileResponse(output_dir/job_id/name)
    return app


def serve(config,examples_dir,output_dir,port=7860):
    import uvicorn
    print(f'CaM-PDA is available locally at http://127.0.0.1:{port}')
    uvicorn.run(create_app(config,examples_dir,output_dir),host='127.0.0.1',port=port)
