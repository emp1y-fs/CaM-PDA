# CaM-PDA

**将 RGB-D 图像补全为米制深度图，并生成彩色点云。**

Python · Windows / Linux · 本地处理

[English](README.md) · [安装说明](docs/INSTALL.md) · [Python 接口](docs/API.md) · [模型架构](docs/ARCHITECTURE.md) · [完整测试结果](benchmarks/README.md)

![真实发动机叶片场景：RGB、传感深度与CaM-PDA补全深度](assets/blade_depth_showcase.png)

CaM-PDA 基于 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything)，结合平衡置信筛选，以及反射、非平面、边缘三个专家，从彩色图像与已对齐的传感深度生成稠密米制深度。

**启动程序后，按提示输入数据路径和保存路径即可。也可以直接选择自带案例。**

## 可以得到什么

| 稠密深度图 | 彩色点云 | 简单的 Python 使用方式 |
|---|---|---|
| 原始分辨率的米制深度，以及便于查看的彩色预览 | 提供相机内参后生成 `.ply`，可在 CloudCompare 等软件中查看 | 运行时输入路径，中英文提示，记住输出和模型存储位置 |

## 开始使用

使用 **Python 3.10–3.12**，支持 Windows 与 Linux。从 GitHub 的 **Code → Download ZIP** 下载并解压，或运行：

```console
git clone https://github.com/emp1y-fs/CaM-PDA.git
cd CaM-PDA
```

先安装适合设备的 PyTorch，再安装 CaM-PDA。以下为已验证的 NVIDIA GPU 环境：

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python run.py
```

[安装说明](docs/INSTALL.md)提供独立环境、CPU 安装和权重配置方法。两个权重合计约 **0.8 GB**；程序会询问模型存储文件夹，也可以直接指定已有权重文件。

### 1. 启动后选择数据

运行 `python run.py`，选择中文后会看到：

```text
选择数据来源
  1  使用自带案例
  2  输入自己的数据路径
  0  退出
```

选择 **1** 可体验发动机叶片案例；选择 **2** 后依次输入照片、原始传感深度和可选的相机内参路径。程序随后询问结果保存文件夹、权重位置与计算设备。

**路径是在程序运行过程中输入的，不需要打开或修改 Python 源码。** 支持含空格的路径，也支持直接粘贴带引号的路径。安装后输入 `cam-pda` 或 `python -m cam_pda` 也可启动同一程序。

### 2. 准备自己的 RGB-D

| 输入 | 格式与要求 |
|---|---|
| 彩色照片 | PNG、JPEG 等 RGB 图像 |
| 原始传感深度 | 二维浮点米制 `.npy`，或单通道整数 `.png`；PNG 的单位由程序询问 |
| 相机内参 | 可选 JSON，包含 `fx、fy、cx、cy、width、height`；生成点云时需要 |

照片与深度必须事先对齐到同一像素网格，深度中的 0 表示缺失。请提供原始深度数值，不能把彩色深度预览当作输入。当前方法需要米制观测锚点，仅有 RGB 照片不足以运行这一补全流程。

不提供内参时输出深度图；提供对应内参后同时输出点云。程序还支持增加同一静态场景的参考视角，自带叶片案例已配有参考帧。其他相机需要使用自己的标定参数。

### 3. 在指定文件夹查看结果

每次运行都会新建独立子文件夹，保留已有结果：

```text
你选择的保存文件夹/
└── 20260907-220000_rgb_a1b2c3d4/
    ├── rgb.png              输入照片
    ├── depth_color.png      彩色深度预览
    ├── depth_m.npy          完整精度的米制深度
    ├── depth_mm.png         可表示时保存的毫米深度 PNG
    ├── point_cloud.ply      提供内参时生成的彩色点云
    ├── accepted_mask.png    接受的传感观测点
    └── metadata.json        单位、相机与运行记录
```

点云坐标单位为米，x 向右、y 向下、z 向前。数值深度保留原始预测，不做平面平滑或几何后处理。

## 材料场景展示

![正向真实桌面场景的RGB、原始深度与CaM-PDA深度结果](assets/material_depth_showcase.png)

这三张来自当前源码中可运行的 ClearGrasp 真实测试案例，按自然正向的观看角度和清楚可辨的补全效果选出。每个面板均展示完整原始画幅。同一行的原始深度和 CaM-PDA 深度共用色标，黑色表示缺失观测；未旋转图片、平滑深度或进行几何后处理。

这些精选图用于展示，与下方保持不变的完整基准测试统计分开，并继续排除于训练。原始三张 DREDS 案例仍然保留。详见[样例编号与选样说明](docs/GALLERY.md)和[展示图来源](assets/preview_provenance.json)。上方真实叶片场景没有真值深度，仅用于定性展示。

## 已记录的测试结果

固定留出测试输入的全图 **AbsRel，越低越好**：

| 方法 | DREDS-CatNovel · 110帧 | NYUv2 · 654帧 | ICL-NUIM · 80帧 |
|---|---:|---:|---:|
| **CaM-PDA** | **0.014226** | 0.023305 | **0.009010** |
| Official PDA | 0.030987 | 0.054981 | 0.067776 |
| OMNI-DC v1.1 | 0.035454 | **0.016802** | — |
| IP-Basic | 0.078946 | 0.031725 | — |
| Marigold-DC · 10步 | 0.096877 | 0.059145 | — |

CaM-PDA 在这些测试集上相对官方 PDA 降低了全图误差；OMNI-DC 在 NYUv2 全图指标上更低。各区域的取舍和更多算法见[完整数据与协议](benchmarks/README.md)。展示图用于说明模型行为，定量结论以冻结测试记录为依据。

## 给开发者的接口

需要集成到其他 Python 程序时，可以调用：

```python
from cam_pda import run_from_paths

result_folder = run_from_paths(
    rgb_path="my_scene/rgb.png",
    depth_path="my_scene/sensor_depth.npy",
    camera_path="my_scene/camera.json",
    output_dir="results",
)
```

普通使用直接启动交互程序即可。数组接口、批处理命令、案例固定采样和参考视角功能见 [API 文档](docs/API.md)。

## 进一步了解

- [模型卡](docs/MODEL_CARD.md)：正式权重、适用范围与已知限制。
- [架构](docs/ARCHITECTURE.md)：平衡置信、三通道和 R/N/E 专家。
- [训练与标注](docs/TRAINING.md)：训练数据、专家监督及测试隔离。
- [跨平台复现](docs/REPRODUCIBILITY.md)：Windows / Linux 验证和数值差异。

## 致谢与使用条款

本项目基于 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything)、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 和 DINOv2。上游声明保留在 [licenses](licenses/README.md)；ViT-B 权重与 DREDS 数据涉及 **CC BY-NC 4.0** 条款。

仓库目前仍处于访问受限的发布检查阶段。新增代码与自有案例的公开许可尚待确定，上游条款继续有效；正式论文与引用信息确定后补充。
