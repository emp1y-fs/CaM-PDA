import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction import download
from reproduction.run import write_report


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def test_resume_range_and_cached_reuse(tmp_path, monkeypatch):
    body = b'correct complete archive'
    item = dict(filename='test.zip', url='https://example.org/test.zip', bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest())
    (tmp_path/'test.zip.partial').write_bytes(body[:7])
    class Opener:
        def open(self, request, timeout):
            assert request.get_header('Range') == 'bytes=7-'
            return Response(body[7:], 206, {'Content-Range':f'bytes 7-{len(body)-1}/{len(body)}'})
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *a: Opener())
    target = download.fetch(item, tmp_path)
    assert target.read_bytes() == body
    assert download.fetch(item, tmp_path) == target
    assert not (tmp_path/'test.zip.partial').exists()


def test_server_ignoring_range_restarts_cleanly(tmp_path, monkeypatch):
    body = b'complete data'
    item = dict(filename='test.zip', url='https://example.org/test.zip', bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest())
    (tmp_path/'test.zip.partial').write_bytes(body[:5])
    class Opener:
        def open(self, request, timeout):
            return Response(body)
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *a: Opener())
    assert download.fetch(item, tmp_path).read_bytes() == body


def test_complete_corrupt_partial_is_quarantined_and_retried(tmp_path, monkeypatch):
    item = dict(filename='test.zip', url='https://example.org/test.zip', bytes=4,
                sha256=hashlib.sha256(b'good').hexdigest())
    (tmp_path/'test.zip.partial').write_bytes(b'evil')
    class Opener:
        def open(self, request, timeout):
            assert request.get_header('Range') is None
            return Response(b'good')
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *a: Opener())
    assert download.fetch(item, tmp_path).read_bytes() == b'good'
    assert next(tmp_path.glob('*.corrupt-*')).read_bytes() == b'evil'


def test_corrupt_download_never_promoted(tmp_path, monkeypatch):
    item = dict(filename='test.zip', url='https://example.org/test.zip', bytes=4,
                sha256=hashlib.sha256(b'good').hexdigest())
    class Opener:
        def open(self, request, timeout):
            return Response(b'evil')
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *a: Opener())
    with pytest.raises(ValueError, match='SHA256'):
        download.fetch(item, tmp_path)
    assert not (tmp_path/'test.zip').exists()


def test_redirect_drops_github_credential():
    request = urllib.request.Request('https://api.github.com/asset', headers={'Authorization':'Bearer secret'})
    redirect = download.SafeRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://release-assets.githubusercontent.com/asset')
    assert not redirect.has_header('Authorization')
    with pytest.raises(ValueError, match='HTTPS'):
        download.SafeRedirect().redirect_request(request, None, 302, 'Found', {}, 'http://example.org/asset')


def test_unpack_rejects_extra_traversal_and_duplicates(tmp_path):
    for name in ('../outside.npz', '/absolute.npz', 'unexpected.npz'):
        archive = tmp_path/'bad.zip'
        with zipfile.ZipFile(archive, 'w') as f:
            f.writestr('NOTICE.txt', 'notice')
            f.writestr(name, b'bad')
        with pytest.raises(ValueError):
            download.unpack(archive, tmp_path/'destination', ['expected.npz'])
    assert not (tmp_path/'outside.npz').exists()


def test_unpack_complete_subset(tmp_path):
    archive = tmp_path/'ok.zip'
    with zipfile.ZipFile(archive, 'w') as f:
        f.writestr('NOTICE.txt', 'notice')
        f.writestr('test/frame.npz', b'example')
    download.unpack(archive, tmp_path/'data', ['test/frame.npz'])
    assert (tmp_path/'data/test/frame.npz').read_bytes() == b'example'


def test_reports_preserve_numeric_units_and_frame_counts(tmp_path):
    metrics = dict(frames=1, pixels=100, AbsRel=.01, MAE_m=.002, RMSE_m=.003)
    (tmp_path/'summary.json').write_text(json.dumps({'icl80':{'full':metrics}}))
    frame_metrics = {k:v for k,v in metrics.items() if k!='frames'}
    row = dict(dataset='icl80', id='icl80/example', metrics={'full':frame_metrics})
    (tmp_path/'frames.jsonl').write_text(json.dumps(row)+'\n')
    write_report(tmp_path, {'status':'running'})
    assert '0.00200000' in (tmp_path/'results.md').read_text()
    assert 'MAE_m' in (tmp_path/'per_frame.csv').read_text(encoding='utf-8-sig')
    assert json.loads((tmp_path/'run.json').read_text())['evaluated_frames'] == 1
