# CaM-PDA

CaM-PDA 使用平衡置信、原生三条件通道，以及反射／非平面／边缘三个独立专家，将对齐的 RGB-D 输入补全为米制深度，并利用相机内参生成彩色点云。正式权重对应历史 T1（026 step600）。本仓库先设为私有，待作者检查后决定公开。

已验证 Windows 原生 CUDA 和 Linux CUDA／CPU。Linux 移植与正式第32帧逐位一致；Windows 因 PyTorch 注意力内核不同，与 Linux 的深度平均差约0.247毫米、最大差约5.68毫米，条件通道一致。详见[复现记录](docs/REPRODUCIBILITY.md)，这不是相对真值的精度声明。

## 安装和使用

Windows、Linux 使用同一 Python 包，不需要 WSL 桥接、自编译 CUDA 算子或本机实验目录。建议 Python 3.10–3.12；详细环境与激活方法见 [安装说明](docs/INSTALL.md)。

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ".[web]"
cam-pda download --github-user emp1y-fs
cam-pda example examples/blade32 --output outputs/blade32
cam-pda serve
```

浏览器打开 **http://127.0.0.1:7860**，选择案例或上传自己的照片、已对齐传感器深度和标定 JSON。输入在本机处理，输出可下载成 ZIP。仅有 RGB 照片无法提供当前方法所需的米制观测锚点。

```console
cam-pda infer --rgb photo.png --depth sensor.png --depth-scale 0.001 --camera camera.json --output outputs/my_scene
```

深度 PNG 是单通道整数观测值，0 表示缺失；`0.001` 将毫米转为米。NPY 必须是二维浮点米制深度。标定 JSON 的字段为 `fx, fy, cx, cy, width, height`，必须对应对齐后的分辨率。不能把深度彩色预览当作传感器原始深度。

输出包括：完整精度深度 NPY、深度彩色预览、满足 uint16 范围时的毫米 PNG、接受锚点掩膜、运行记录，以及提供正确标定时的彩色 PLY 点云。点云单位是米，可在 CloudCompare 中打开。程序不会进行平面后处理。

## 已包含的内容

- 发动机叶片第 32 帧，以及可作为多视角参考的第 20 帧（另附第 30 帧的小基线回退案例），附真实 RGB-D 和对应标定。
- 三张 DREDS-CatNovel 测试材料示例，按冻结清单首／中／末位置抽取，不按效果挑图。其旧缓存没有封存变换后的内参，因此提供深度示例，不猜测内参生成点云。
- [架构](docs/ARCHITECTURE.md)、[Python 接口](docs/API.md)、[训练及标注口径](docs/TRAINING.md)、[论文测试数据](benchmarks/README.md)。

对外名称统一为 CaM-PDA；原 T2/T3 不替换正式模型。模型仍有跨域与局部指标的取舍，专家调用图也不等同于精确语义分割。第三方结果和全部区域指标保留在 CSV 中，不把局部改善写成所有场景均领先。

代码整理参考官方 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything)。上游代码、模型与数据的许可分别保留；ViT-B 权重和 DREDS 数据涉及 CC BY-NC 4.0。作者新代码与自有案例的公开许可尚待作者决定。当前是私有审阅版本。
