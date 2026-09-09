"""Download and verify benchmark assets, retaining partial files for resumption."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit
import zipfile


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != 'https':
            raise ValueError('Download redirected to a non-HTTPS address')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def github_token(username=None):
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    if username and not re.fullmatch(r'[A-Za-z0-9-]+', username):
        raise ValueError('Invalid GitHub username')
    query = 'protocol=https\nhost=github.com\n'
    if username:
        query += f'username={username}\n'
    try:
        result = subprocess.run(['git', 'credential', 'fill'], input=query+'\n', text=True,
            capture_output=True, timeout=20, env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never'))
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        return values.get('password')
    return None


def resolve_request(item, token, opener):
    url = item['url']
    if urlsplit(url).scheme != 'https':
        raise ValueError('Expected an HTTPS download source')
    headers = {'User-Agent': 'CaM-PDA-benchmark'}
    if urlsplit(url).netloc == 'api.github.com':
        headers['Accept'] = 'application/octet-stream'
        if token:
            headers['Authorization'] = 'Bearer '+token
        if '/releases/tags/' in url:
            request = urllib.request.Request(url, headers=dict(headers, Accept='application/vnd.github+json'))
            try:
                with opener.open(request, timeout=60) as response:
                    release = json.load(response)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403, 404):
                    raise RuntimeError('GitHub release is inaccessible. During private review, sign in with an account granted repository access (Git Credential Manager or GH_TOKEN). Public releases need no sign-in.') from None
                raise
            matches = [a for a in release.get('assets', []) if a['name'] == item['filename']]
            if len(matches) != 1 or matches[0]['size'] != item['bytes']:
                raise ValueError('Release asset is missing or has a different size: '+item['filename'])
            url = matches[0]['url']
            if not url.startswith('https://api.github.com/'):
                raise ValueError('Unexpected GitHub asset address')
    return url, headers


def fetch(item, folder, token=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = item['filename']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Invalid asset filename')
    target = folder/name
    expected_size = item['bytes']
    if target.is_file():
        if target.stat().st_size != expected_size or sha256(target) != item['sha256']:
            raise ValueError(f'Existing file failed verification: {target}. Move it aside before retrying.')
        print(f'Verified cached file: {name}', flush=True)
        return target
    partial = folder/(name+'.partial')
    opener = urllib.request.build_opener(SafeRedirect())
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            if sha256(partial) == item['sha256']:
                break
            rejected = partial.with_name(partial.name+'.corrupt-'+uuid.uuid4().hex[:8])
            partial.rename(rejected)
            print('Corrupt partial preserved; restarting download: '+name, flush=True)
            offset = 0
        if offset > expected_size:
            raise ValueError('Partial file exceeds the expected size: '+str(partial))
        if shutil.disk_usage(folder).free < expected_size-offset+(64 << 20):
            raise OSError('Insufficient space in the selected download folder')
        url, headers = resolve_request(item, token, opener)
        if offset:
            headers['Range'] = f'bytes={offset}-'
        print(f'Downloading {name}: {offset/1e6:.1f}/{expected_size/1e6:.1f} MB', flush=True)
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                        raise ValueError('Invalid Content-Range from download source')
                    mode = 'ab'
                elif status == 200:
                    offset, mode = 0, 'wb'
                else:
                    raise ValueError(f'Unexpected download status: {status}')
                last = time.monotonic()
                with partial.open(mode) as stream:
                    for block in iter(lambda: response.read(4 << 20), b''):
                        stream.write(block)
                        offset += len(block)
                        if offset > expected_size:
                            raise ValueError('Download exceeds the expected size')
                        if time.monotonic()-last >= 5:
                            print(f'  {offset/1e6:.1f}/{expected_size/1e6:.1f} MB', flush=True)
                            last = time.monotonic()
            if offset == expected_size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise RuntimeError(f'Download failed (HTTP {error.code}): {name}. Partial data is retained; check access and rerun.') from None
            if attempt == 3:
                raise RuntimeError(f'Download interrupted: {name}. Rerun the same command to resume.') from None
        if attempt < 3:
            time.sleep(2**attempt)
    if not partial.exists() or partial.stat().st_size != expected_size or sha256(partial) != item['sha256']:
        raise ValueError(f'Download failed SHA256 verification: {partial}. No unverified data will be used.')
    partial.replace(target)
    print('SHA256 verified: '+name, flush=True)
    return target


def unpack(archive_path, destination, expected_files):
    destination = Path(destination).resolve()
    expected = set(expected_files) | {'NOTICE.txt'}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if set(names) != expected or len(names) != len(expected):
            raise ValueError('Archive members do not match the selected test subset')
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Unsafe archive member')
            target = (destination/relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive path escapes the selected folder')
            if info.file_size > 4 << 20:
                raise ValueError('Unexpectedly large benchmark record')
        for info in archive.infolist():
            target = destination/info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix+'.partial')
            with archive.open(info) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)
