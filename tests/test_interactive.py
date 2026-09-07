import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cam_pda.interactive import clean_path, main, read_settings
from cam_pda.runner import run_example, run_from_paths
from cam_pda.types import DepthResult


@pytest.fixture
def example(tmp_path):
    folder = tmp_path/'input with spaces'
    folder.mkdir()
    Image.fromarray(np.full((5,5,3), 128, np.uint8)).save(folder/'rgb.png')
    np.save(folder/'sensor_depth.npy', np.full((5,5), 2, np.float32))
    (folder/'camera.json').write_text(json.dumps(dict(fx=2,fy=2,cx=2,cy=2,width=5,height=5)))
    (folder/'provenance.json').write_text(json.dumps(dict(seed=42)))
    return folder


class FakeModel:
    def predict(self, rgb, depth, seed=0, sampled_mask=None):
        self.seed, self.mask = seed, sampled_mask
        valid = np.ones(depth.shape, bool)
        return DepthResult(depth.copy(), valid, valid, valid, np.zeros((3,*depth.shape),np.float32),0,1,{})


def test_repeated_runs_preserve_existing_files_and_example_protocol(example, tmp_path):
    mask=np.ones((5,5),bool);mask[0,0]=False
    np.save(example/'sampled_mask.npy',mask)
    out=tmp_path/'results';out.mkdir();(out/'keep.txt').write_text('keep me')
    model=FakeModel()
    first=run_example(example,out,model=model,progress=lambda _:None)
    second=run_example(example,out,model=model,progress=lambda _:None)
    assert first!=second and first.parent==second.parent==out
    assert (out/'keep.txt').read_text()=='keep me'
    assert (first/'point_cloud.ply').exists() and (second/'depth_color.png').exists()
    assert model.seed==42 and np.array_equal(model.mask,mask)
    assert np.array_equal(np.load(first/'depth_m.npy'),np.load(example/'sensor_depth.npy'))


def test_invalid_inputs_rejected_before_model_or_output(example,tmp_path):
    Image.fromarray(np.zeros((4,5,3),np.uint8)).save(example/'rgb.png')
    with pytest.raises(ValueError,match='same resolution'):
        run_from_paths(example/'rgb.png',example/'sensor_depth.npy',tmp_path/'out',progress=lambda _:None)
    assert not (tmp_path/'out').exists()


def test_reference_requires_calibration_and_preserves_frozen_mask(example,tmp_path):
    ref=dict(rgb_path=example/'rgb.png',depth_path=example/'sensor_depth.npy',camera_path=example/'camera.json')
    with pytest.raises(ValueError,match='target camera'):
        run_from_paths(ref['rgb_path'],ref['depth_path'],tmp_path/'out',references=[ref],model=FakeModel(),progress=lambda _:None)
    np.save(example/'sampled_mask.npy',np.ones((5,5),bool))
    with pytest.raises(ValueError,match='single-view'):
        run_example(example,tmp_path/'out',references=[ref],model=FakeModel(),progress=lambda _:None)


def test_interactive_paths_are_requested_at_runtime(example,tmp_path):
    output=tmp_path/'chosen output'
    weight=tmp_path/'model.pt';weight.write_bytes(b'test placeholder')
    # English, own files, quoted paths, no calibration, output, local weights, auto, exit.
    answers=iter(['1','2',f'"{example / "rgb.png"}"',str(example/'sensor_depth.npy'),'',
                  str(output),'2',str(weight),str(weight),'1','0'])
    seen={};messages=[]
    def execute(**kwargs):
        seen.update(kwargs)
        return output/'new_run'
    settings=tmp_path/'preferences.json'
    assert main(input_fn=lambda prompt:next(answers),output_fn=messages.append,settings_path=settings,execute=execute)==0
    assert seen['rgb_path']==example/'rgb.png' and seen['camera_path'] is None
    assert seen['output_dir']==output and seen['references']==[]
    assert seen['model_options']['allow_download'] is False
    assert json.loads(settings.read_text(encoding='utf-8'))['output_dir']==str(output)
    assert not any('point_cloud.ply' in message for message in messages)


def test_prompt_example_keeps_seed_and_mask(example,tmp_path):
    from cam_pda import interactive
    np.save(example/'sampled_mask.npy',np.ones((5,5),bool))
    weight=tmp_path/'model.pt';weight.touch()
    answers=iter(['2','1','1',str(tmp_path/'out'),'2',str(weight),str(weight),'2','0'])
    seen={}
    def execute(**kwargs):
        seen.update(kwargs);return tmp_path/'out'/'run'
    # Point discovery at an isolated folder; the packaged example can also exist.
    original=interactive.example_folders
    interactive.example_folders=lambda explicit=None:[example]
    try:
        main(input_fn=lambda _:next(answers),output_fn=lambda _:None,settings_path=tmp_path/'prefs.json',execute=execute)
    finally:
        interactive.example_folders=original
    assert seen['seed']==42 and seen['sampled_mask_path']==example/'sampled_mask.npy'
    assert seen['model_options']['device']=='cpu'


def test_path_quotes_corrupt_preferences_and_eof(tmp_path,monkeypatch):
    monkeypatch.setenv('CAM_PDA_TEST_PATH',str(tmp_path))
    assert clean_path('  "$CAM_PDA_TEST_PATH/a b"  ')==tmp_path/'a b'
    path=tmp_path/'bad.json';path.write_text('[]');assert read_settings(path)=={}
    def eof(_):raise EOFError
    assert main(input_fn=eof,output_fn=lambda _:None,settings_path=path)==0


def test_cli_does_not_discard_frozen_example_mask(example,tmp_path,monkeypatch,capsys):
    import cam_pda
    from cam_pda.cli import main as cli_main
    np.save(example/'sampled_mask.npy',np.ones((5,5),bool))
    def unexpected_model(**kwargs):
        raise AssertionError('Input rejection must precede model loading')
    monkeypatch.setitem(cam_pda.__dict__,'CaMPDA',unexpected_model)
    with pytest.raises(SystemExit) as error:
        cli_main(['example',str(example),'--output',str(tmp_path/'out'),'--references',str(example)])
    assert error.value.code==2
    assert 'single-view' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()
