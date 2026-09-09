# CaM-PDA

**将视觉深度先验与实测深度对齐，从 RGB-D 恢复稠密的米制深度。**

Python · Windows / Linux · 单帧推理

[English](README.md) · [安装](docs/INSTALL.md) · [数据下载](docs/DATASETS.md) · [复现流程](docs/REPRODUCIBILITY.md) · [API](docs/API.md)

![真实叶片场景的 RGB、ToF 实测深度与 CaM-PDA 深度](assets/blade_depth_showcase.png)

CaM-PDA 在 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything) 的基础上引入了**平衡置信前置**和**专家矩阵**。置信前置筛选传感器观测，用于视觉深度先验与实测深度的对齐；专家矩阵在条件深度网络中引入独立门控的反射、非平面和边缘专家，进一步预测稠密的米制深度。

**输入：** 一张 RGB 图像及对应的传感器深度。**输出：** 深度预览、数值深度；提供相机内参时还可生成彩色点云。RGB 与传感器深度需要预先配准到同一像素网格。CaM-PDA 内部进行的是深度尺度与结构对齐，不负责两台相机之间的图像配准。

## 总体流程

![CaM-PDA 总体流程：平衡置信筛选、深度对齐与预填充，以及引入反射、非平面和边缘专家的条件深度估计](assets/cam_pda_overall_workflow.png)

## 运行程序

使用 Python **3.10–3.12**，下载仓库并安装依赖：

```console
git clone https://github.com/emp1y-fs/CaM-PDA.git
cd CaM-PDA
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python run.py
```

以上使用 NVIDIA CUDA 版 PyTorch。CPU 安装、独立环境和权重获取见[安装说明](docs/INSTALL.md)。两个模型文件合计约 **0.8 GB**。

程序启动后选择自带案例，或**在运行时输入数据路径**，然后选择保存目录和模型位置。无需修改 Python 源文件。安装后也可以使用 `cam-pda` 或 `python -m cam_pda` 启动，提供中英文提示。

| 输入 | 格式 |
|---|---|
| RGB 图像 | PNG、JPEG |
| 已配准的传感器深度 | 以米为单位的浮点 `.npy`，或注明单位的单通道整数 `.png`；0 表示缺失 |
| 相机内参 | 包含 `fx`、`fy`、`cx`、`cy`、`width`、`height` 的 JSON；生成点云时必需 |

彩色深度图只用于展示，不能作为数值深度输入。自己的数据应使用实际相机内参；仅凭一张 RGB 照片不能获得本方法所需的实测尺度。

每次运行都在所选保存目录中创建独立文件夹：

```text
result_folder/
├── rgb.png              输入 RGB
├── depth_color.png      深度预览
├── depth_m.npy          Float32 米制深度
├── depth_mm.png         四舍五入后的毫米深度，数值可表示时输出
├── point_cloud.ply      标定后的彩色点云
├── accepted_mask.png    保留的传感器观测
└── metadata.json        内参、单位和推理设置
```

未提供相机内参时只生成深度图。点云单位为米，x 向右、y 向下、z 向前。导出保留模型预测数值，预览配色不会改变深度值。

## 其他发动机构件

![四组单帧案例：RGB、原始 ToF 深度、CaM-PDA 深度和点云](assets/engine_components.png)

以上为论文第 5 节补充的发动机内部构件，拍摄于**杂物相对较少的干净桌面环境**。每行均由一帧 RGB-D 独立推理，展示缺失深度的补全与重建结构。这些场景没有参考几何，不能据此给出定量精度结论。

在程序中选择 `engine_component_01` 至 `engine_component_04`，即可分别运行图中四行案例。源码和安装包都附带原始 RGB、实测深度、内参与采样种子。详见[案例说明及来源](docs/ENGINE_COMPONENTS.md)。

源码中还保留了[真实材料场景案例](docs/GALLERY.md)。

## 复现与代码

- [复现流程](docs/REPRODUCIBILITY.md)：环境 → 权重 → 案例 → 测试 → 训练记录。
- [数据下载](docs/DATASETS.md)：原始数据入口、精确样本清单和所需文件。
- [训练方法](docs/TRAINING.md)：两个适配阶段、专家监督与权重选择。
- [架构](docs/ARCHITECTURE.md)：平衡置信、三通道条件、独立 R/N/E 专家。
- [Python API](docs/API.md)：文件和数组接口。
- [模型说明](docs/MODEL_CARD.md)：适用范围与局限。

发布权重对应论文中的 CaM-PDA；应用更新不改变模型权重。

## 致谢与使用条款

基于 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything)、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 和 DINOv2。[上游版权与数据条款](licenses/README.md)继续适用，其中 ViT-B 权重和 DREDS 数据采用 **CC BY-NC 4.0**。

仓库目前保持私有供作者检查。CaM-PDA 新增代码及作者自采案例的公开许可尚未确定，收录在此不代表额外授予再分发权限。
