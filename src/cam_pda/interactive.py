"""A Python terminal application: ask for local paths at runtime."""
from importlib.resources import files
from pathlib import Path
import json
import os


def clean_path(value):
    """Accept pasted/dragged paths, quotes, environment variables and spaces."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return Path(os.path.expandvars(value)).expanduser().resolve()


def settings_file():
    override = os.environ.get('CAM_PDA_SETTINGS')
    if override:
        return clean_path(override)
    base = Path(os.environ.get('APPDATA') or os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
    return base/'cam-pda/settings.json'


def read_settings(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            return {}
        return {k: v for k, v in value.items() if isinstance(v, str)}
    except (OSError, ValueError):
        return {}


def example_folders(explicit=None):
    # The source checkout provides additional examples; the wheel includes blade32.
    candidates = [Path(explicit)] if explicit else [Path(__file__).resolve().parents[2]/'examples']
    candidates.append(Path(str(files('cam_pda').joinpath('resources/examples'))))
    found = {}
    for root in candidates:
        for folder in sorted(root.glob('*')):
            if (folder/'rgb.png').is_file() and (folder/'sensor_depth.npy').is_file():
                found.setdefault(folder.name, folder)
    order = {'blade32': 0, 'blade20': 1, 'blade30': 2}
    return sorted(found.values(), key=lambda p: (order.get(p.name, 3), p.name))


class Prompts:
    def __init__(self, read, write, chinese=False):
        self.read, self.write, self.chinese = read, write, chinese

    def text(self, en, zh):
        return zh if self.chinese else en

    def choice(self, title, options, default='1'):
        self.write(title)
        for key, label in options.items():
            self.write(f'  {key}  {label}')
        while True:
            value = self.read(self.text(f'Choose [{default}]: ', f'请选择 [{default}]：')).strip() or default
            if value in options:
                return value
            self.write(self.text('Enter one of the listed numbers.', '请输入列表中的数字。'))

    def path(self, title, default=None, *, optional=False, directory=False):
        while True:
            hint = f' [{default}]' if default else ''
            value = self.read(title + hint + ': ').strip()
            if not value and optional and default is None:
                return None
            if not value and default is not None:
                value = str(default)
            if not value:
                self.write(self.text('A path is required.', '请输入路径。'))
                continue
            try:
                path = clean_path(value)
                if directory:
                    if path.exists() and not path.is_dir():
                        raise ValueError(self.text('Choose a folder, not a file.', '这里需要文件夹路径。'))
                elif not path.is_file():
                    raise ValueError(self.text('File not found. Check the full path.', '文件不存在，请检查完整路径。'))
                return path
            except (OSError, ValueError) as error:
                self.write(str(error))

    def own_inputs(self, *, reference=False):
        self.write(self.text('\nReference view' if reference else '\nYour RGB-D files',
                             '\n参考视角' if reference else '\n输入自己的 RGB-D 文件'))
        rgb = self.path(self.text('RGB image path', 'RGB 照片路径'))
        depth = self.path(self.text('Sensor depth path (.npy or integer .png)', '传感深度路径（.npy 或单通道整数 .png）'))
        scale = None
        if depth.suffix.lower() == '.png':
            choice = self.choice(self.text('Depth PNG units', '深度 PNG 的原始单位'), {
                '1': self.text('Millimetres', '毫米'), '2': self.text('Metres', '米'),
                '3': self.text('Custom conversion to metres', '自定义转为米的倍率')})
            scale = .001 if choice == '1' else 1.
            if choice == '3':
                import math
                while True:
                    try:
                        scale = float(self.read(self.text('Units-to-metres multiplier: ', '转为米的倍率：')))
                        if not math.isfinite(scale) or scale <= 0:
                            raise ValueError
                        break
                    except ValueError:
                        self.write(self.text('Enter a positive finite number.', '请输入有效的正数。'))
        camera = self.path(self.text('Camera JSON path' + ('' if reference else ' (Enter: depth only)'),
                                     '相机内参 JSON 路径' + ('' if reference else '（回车：仅生成深度图）')),
                           optional=not reference)
        return dict(rgb_path=rgb, depth_path=depth, depth_scale=scale, camera_path=camera)


def main(*, input_fn=input, output_fn=print, settings_path=None, examples_dir=None, execute=None):
    """Start the interactive prompt. No GUI, local server or source edits."""
    path = Path(settings_path) if settings_path else settings_file()
    settings = read_settings(path)
    p = Prompts(input_fn, output_fn)
    try:
        output_fn('\nCaM-PDA\nRGB-D  >  Metric depth  >  Colored point cloud\n')
        language = p.choice('Language / 语言', {'1': 'English', '2': '简体中文'}, settings.get('language', '1'))
        p.chinese = language == '2'
        settings['language'] = language
        while True:
            mode = p.choice(p.text('\nHow would you like to start?', '\n选择数据来源'), {
                '1': p.text('Try an included example', '使用自带案例'),
                '2': p.text('Enter my data paths', '输入自己的数据路径'),
                '0': p.text('Exit', '退出')})
            if mode == '0':
                return 0
            try:
                from .runner import load_example, run_from_paths
                folders = example_folders(examples_dir)
                selected = None
                if mode == '1':
                    if not folders:
                        raise FileNotFoundError('No example files found. Use your own data or download the source examples.')
                    names = {}
                    for i, folder in enumerate(folders, 1):
                        label = ('Engine blade / 发动机叶片 — ' + folder.name.removeprefix('blade')) if folder.name.startswith('blade') else folder.name
                        if not (folder/'camera.json').is_file():
                            label += p.text(' (depth only)', '（仅深度）')
                        names[str(i)] = label
                    chosen = p.choice(p.text('\nChoose an example', '\n选择案例'), names)
                    selected = folders[int(chosen)-1]
                    inputs = load_example(selected)
                else:
                    inputs = p.own_inputs()
                references = []
                if inputs.get('camera_path') and not inputs.get('sampled_mask_path'):
                    view = p.choice(p.text('\nReconstruction mode', '\n重建方式'), {
                        '1': p.text('Single view', '单视角'),
                        '2': p.text('Use an additional view of the same static scene', '增加同一静态场景的参考视角')})
                    if view == '2':
                        ref = next((f for f in folders if f.name == 'blade20'), None)
                        if selected and selected.name == 'blade32' and ref:
                            r = load_example(ref)
                            references.append({k: r[k] for k in ('rgb_path', 'depth_path', 'camera_path')})
                            output_fn(p.text('Using included blade20 as the reference.', '使用自带的第20帧作为参考视角。'))
                        else:
                            references.append(p.own_inputs(reference=True))
                output = p.path(p.text('\nSave results in folder', '\n结果保存文件夹'),
                                settings.get('output_dir', str(Path.cwd()/'outputs')), directory=True)
                options = {}
                source = p.choice(p.text('\nModel files', '\n模型文件'), {
                    '1': p.text('Use a model folder; download missing weights', '指定模型存储文件夹，缺少时下载'),
                    '2': p.text('Use two existing weight files', '指定已有的两个权重文件')},
                    settings.get('weight_source', '1'))
                settings['weight_source'] = source
                if source == '1':
                    cache = p.path(p.text('Model folder (about 0.8 GB for both weights)', '模型存储文件夹（两个权重合计约0.8 GB）'),
                                   settings.get('cache_dir', os.environ.get('CAM_PDA_HOME', str(Path.home()/'.cache/cam-pda'))), directory=True)
                    options['cache_dir'] = cache
                    settings['cache_dir'] = str(cache)
                    if not (cache/'cam_pda_v1.pt').is_file():
                        user = input_fn(p.text('GitHub username for private-release access (Enter to skip): ',
                                               '私有发布的 GitHub 用户名（公开发布时可直接回车）：')).strip()
                        if user:
                            os.environ['CAM_PDA_GITHUB_USER'] = user
                else:
                    options['checkpoint'] = p.path(p.text('CaM-PDA weight file', 'CaM-PDA 权重文件'), settings.get('checkpoint'))
                    options['mde_checkpoint'] = p.path(p.text('Monocular prior weight file', '冻结单目先验权重文件'), settings.get('mde_checkpoint'))
                    options['allow_download'] = False
                    settings.update({k: str(options[k]) for k in ('checkpoint', 'mde_checkpoint')})
                device = p.choice(p.text('\nCompute device', '\n计算设备'), {
                    '1': p.text('Automatic (NVIDIA GPU when available)', '自动（有 NVIDIA GPU 时优先使用）'),
                    '2': p.text('CPU (slower)', 'CPU（较慢）')}, settings.get('device_choice', '1'))
                options['device'] = 'auto' if device == '1' else 'cpu'
                settings.update(output_dir=str(output), device_choice=device)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError:
                    output_fn(p.text('Preferences could not be saved; this run can continue.', '无法保存偏好设置，本次运行仍可继续。'))
                output_fn(p.text('\nStarting. First model load can take a little longer.\n', '\n开始处理，首次加载模型可能需要更长时间。\n'))
                run = (execute or run_from_paths)(**inputs, output_dir=output, references=references,
                                                 model_options=options, progress=output_fn)
                output_fn(p.text('\nComplete. Your result folder:', '\n处理完成，结果文件夹：'))
                output_fn(str(run))
                output_fn('depth_color.png  |  depth_m.npy' + ('  |  point_cloud.ply' if inputs.get('camera_path') else ''))
            except ModuleNotFoundError as error:
                output_fn(p.text(f'Missing Python dependency: {error.name}. Follow docs/INSTALL.md in this repository.',
                                 f'缺少 Python 依赖：{error.name}。请按仓库 docs/INSTALL.md 安装。'))
            except Exception as error:
                output_fn(p.text('Could not finish: ', '未能完成：') + str(error))
            again = p.choice(p.text('\nRun another scene?', '\n继续处理其他数据？'),
                             {'1': p.text('Yes', '是'), '0': p.text('Exit', '退出')}, '0')
            if again == '0':
                return 0
    except (KeyboardInterrupt, EOFError):
        output_fn(p.text('\nStopped.', '\n已停止。'))
        return 0
