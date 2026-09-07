"""Explicit, SHA256-verified model acquisition; no credentials are logged."""
from importlib.resources import files
from pathlib import Path
import hashlib,json,os,urllib.request
from urllib.parse import urlsplit


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
        if redirected is not None and urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()

def manifest():
    return json.loads(files('cam_pda').joinpath('resources/models.json').read_text())

def download(url,target,expected):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    headers={'User-Agent':'CaM-PDA/0.1'}
    if url.startswith('https://api.github.com/'):
        headers['Accept']='application/octet-stream'
        token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
        if not token and os.environ.get('CAM_PDA_GITHUB_USER'):
            import subprocess
            user=os.environ['CAM_PDA_GITHUB_USER']
            if not user.replace('-','').isalnum():raise ValueError('Invalid GitHub username')
            query=f'protocol=https\nhost=github.com\nusername={user}\n\n'
            env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
            try:
                credentials=subprocess.run(['git','credential','fill'],input=query,text=True,
                    capture_output=True,env=env,timeout=30,check=True)
            except (OSError,subprocess.SubprocessError) as error:
                raise RuntimeError('Git credentials are unavailable. Sign in with Git Credential Manager first.') from None
            values=dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
            token=values.get('password')
        if token:headers['Authorization']='Bearer '+token
    request=urllib.request.Request(url,headers=headers)
    partial=target.with_suffix(target.suffix+'.partial')
    opener=urllib.request.build_opener(_SafeRedirect())
    if url.startswith('https://api.github.com/') and '/releases/tags/' in url:
        metadata_headers=dict(headers,Accept='application/vnd.github+json')
        with opener.open(urllib.request.Request(url,headers=metadata_headers),timeout=60) as response:
            release=json.load(response)
        assets=[asset for asset in release.get('assets',[]) if asset['name']==target.name]
        if len(assets)!=1:raise FileNotFoundError('The named model asset is not available in this release.')
        asset_url=assets[0]['url']
        if not asset_url.startswith('https://api.github.com/'):
            raise ValueError('Unexpected release asset host.')
        request=urllib.request.Request(asset_url,headers=headers)
    with opener.open(request,timeout=60) as response,partial.open('wb') as output:
        for block in iter(lambda:response.read(4<<20),b''):output.write(block)
    if file_sha256(partial)!=expected:
        raise ValueError('Downloaded weight SHA256 mismatch; incomplete file was not promoted.')
    partial.replace(target)

def resolve_weights(checkpoint=None,mde_checkpoint=None,*,cache_dir=None,allow_download=True):
    metadata=manifest()
    cache=Path(cache_dir or os.environ.get('CAM_PDA_HOME',Path.home()/'.cache/cam-pda'))
    paths=[]
    for key,explicit in [('ca_m_pda',checkpoint),('mde',mde_checkpoint)]:
        item=metadata[key]
        path=Path(explicit).expanduser() if explicit is not None else cache/item['filename']
        if not path.is_file():
            url=item.get('url')
            if explicit is not None or not allow_download or not url:
                raise FileNotFoundError(f'Model file is required: {path}. See docs/INSTALL.md or run cam-pda download.')
            download(url,path,item['sha256'])
        if file_sha256(path)!=item['sha256']:
            raise ValueError(f'Model SHA256 mismatch: {path.name}')
        paths.append(path)
    return *paths,metadata
