# CaM-PDA

**将视觉深度先验与实测深度对齐，从 RGB-D 恢复稠密的米制深度。**

Python · Windows / Linux · 单帧推理

[快速上手](#快速上手) · [一键测试](#一条命令下载测试集并评估) · [运行自己的数据](#3-运行自己的数据) · [English](README.md) · [安装](docs/INSTALL.md) · [数据下载](docs/DATASETS.md) · [复现流程](docs/REPRODUCIBILITY.md) · [API](docs/API.md)

## 总体流程

![CaM-PDA 总体流程：平衡置信筛选、深度对齐与预填充，以及引入反射、非平面和边缘专家的条件深度估计](assets/cam_pda_overall_workflow.png)

## 快速上手

在 Python 终端运行 CaM-PDA，**程序启动后再输入数据路径和保存位置，无需修改源代码**。建议先使用自带案例：RGB、实测深度和相机内参均已备好。

### 1. 获取代码并安装

Windows、Linux 均可使用已有的 **Python 3.10–3.12** 环境。在准备存放项目的文件夹中打开终端：

```console
git clone https://github.com/emp1y-fs/CaM-PDA.git
cd CaM-PDA
```

也可以点击本页 **Code → Download ZIP**，解压后在项目文件夹中打开终端。

使用 **NVIDIA 显卡及兼容驱动**时，执行：

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ".[benchmark]"
```

使用 **CPU** 时，改为下面两条命令，推理速度会慢一些：

```console
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[benchmark]"
```

无需另外安装 CUDA Toolkit 或使用 Visual Studio 编译。需要独立环境时，可参考[环境设置](docs/INSTALL.md#1-choose-a-python-environment)。

### 2. 启动 CaM-PDA

在项目文件夹中运行：

```console
python run.py
```

安装后，`cam-pda` 或 `python -m cam_pda` 也能启动同一个程序。

### 3. 运行自己的数据

准备一张 RGB 图、与其配准的数值深度，以及实际相机内参 JSON。下面假设文件位于 `D:\RGBD\scene01\rgb.png`、`D:\RGBD\scene01\sensor_depth.npy`、`D:\RGBD\scene01\camera.json`。**请替换为你电脑上的真实完整路径。** 保存位置填写文件夹，输入数据和权重填写具体文件。

这个示例使用已有权重，请先下载两个模型文件，放到你选择的目录：

- **`cam_pda_v1.pt`**：[CaM-PDA 模型发布页](https://github.com/emp1y-fs/CaM-PDA/releases/tag/v0.1.0)。
- **`depth_anything_v2_vitb.pth`**：[PDA 模型下载地址](https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth)。

执行 `python run.py` 后，按下表**逐项输入，每输入一项就按回车**。这些内容是在 CaM-PDA 程序内回答提示，不是整段粘贴到 PowerShell 的命令，也不用修改 Python 文件。下表列出菜单和提示名称，默认选项可能沿用上一次运行的设置。

| 程序询问什么 | 你输入什么（Windows 示例） |
|---|---|
| Language / 语言 | **`2` — 简体中文** |
| 选择数据来源 | **`2` — 输入自己的数据路径** |
| RGB 照片路径 | `D:\RGBD\scene01\rgb.png` |
| 传感深度路径（.npy 或单通道整数 .png） | `D:\RGBD\scene01\sensor_depth.npy` |
| 相机内参 JSON 路径（回车：仅生成深度图） | `D:\RGBD\scene01\camera.json` |
| 结果保存文件夹 | **`D:\RGBD\results`** |
| 模型文件 | **`2` — 指定已有的两个权重文件** |
| CaM-PDA 权重文件 | `D:\CaM-PDA\weights\cam_pda_v1.pt` |
| 冻结单目先验权重文件 | `D:\CaM-PDA\weights\depth_anything_v2_vitb.pth` |
| 计算设备 | `1` — 自动选择，或 `2` — CPU |

Linux 操作相同，将路径换为实际 Linux 路径，例如 `/home/you/RGBD/scene01/rgb.png`、`/home/you/RGBD/scene01/sensor_depth.npy`、`/home/you/RGBD/scene01/camera.json`、`/home/you/RGBD/results`，两个模型文件放在 `/home/you/CaM-PDA/weights/` 下。含空格的路径可以连同外层引号一起粘贴。

**结果保存到哪里？** 按上述 Windows 示例，程序会新建类似 `D:\RGBD\results\<时间>_rgb_<编号>\` 的文件夹，完成后在终端打印完整位置。选择保存路径不会改变输出文件名和格式。程序会记住存储位置；下次在同一提示处输入新路径即可更换。

**深度和内参要求：** RGB 与实测深度须分辨率相同、已配准到同一像素网格。`.npy` 深度是以**米**为单位的二维浮点数组。若输入单通道整数深度 `.png`，填写路径后会立即多出单位选择；文件实际以毫米存储时，才选择 **1 — 毫米**。0 表示缺失，彩色预览图不是数值深度。相机 JSON 包含真实标定得到的 `fx`、`fy`、`cx`、`cy`、`width`、`height`。在相机路径处留空回车可仅生成深度图；点云需要内参。

**希望自动下载权重？** 在“模型文件”处选择 **1 — 指定模型存储文件夹，缺少时下载**，再输入例如 `D:\CaM-PDA\weights` 或 `/home/you/CaM-PDA/weights`。两个文件合计约 **0.8 GB**。仓库保持私有期间，CaM-PDA 权重需要访问权限：可通过浏览器手动下载，也可使用已通过 Git Credential Manager 登录的授权账号自动下载。发布公开后，用户名提示直接留空即可。

**想先跑自带案例？** 在数据来源处选择 **1 — 使用自带案例**，从列表选择 `blade32` 或 `engine_component_01`–`engine_component_04`，然后按上表填写保存目录和模型位置。案例的 RGB、实测深度和内参会自动载入。

### 4. 查看结果

处理完成后，终端会显示结果位置。每次运行都会在指定保存目录下新建一个子文件夹：

```text
result_folder/
├── rgb.png              输入 RGB
├── depth_color.png      深度预览
├── depth_m.npy          Float32 米制深度
├── depth_mm.png         四舍五入后的毫米深度，数值可表示时输出
├── point_cloud.ply      彩色点云，提供内参时输出
├── accepted_mask.png    保留的传感器观测
└── metadata.json        内参、单位和推理设置
```

双击 **`depth_color.png`** 查看深度预览；使用 CloudCompare 等 PLY 查看器打开 **`point_cloud.ply`**。数值计算使用 **`depth_m.npy`**。点云单位为米，x 向右、y 向下、z 向前；导出保留预测的深度值。

## 一条命令下载测试集并评估

完成上方安装后，**在克隆或解压得到的 CaM-PDA 项目文件夹内**运行：

```console
python reproduce.py --root ./cam_pda_test
```

程序会自动下载固定测试数据和两个权重，校验 SHA256，评估 **844 张正式测试图像**，最后输出指标和预测深度图。无需手动解压、挑选样本或修改代码。`--root` 指定**全部下载文件、数据、权重和结果**的保存目录。例如 Windows 可直接运行 `python reproduce.py --root "D:/CaM-PDA-test"`，Linux 可运行 `python reproduce.py --root /home/you/CaM-PDA-test`。

| 测试集 | 帧数 | 程序自动下载的文件 |
|---|---:|---|
| DREDS-CatNovel | 110 | [整理后的测试输入与参考掩码，30.2 MB](https://github.com/emp1y-fs/CaM-PDA/releases/download/paper-tests-v1/cam_pda_dreds110_v1.zip) |
| NYUv2 | 654 | [官方标注 MAT，2.97 GB](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) + [固定协议掩码，15.8 MB](https://github.com/emp1y-fs/CaM-PDA/releases/download/paper-tests-v1/cam_pda_nyu654_masks_v1.zip)，下载后自动转换 |
| ICL-NUIM | 80 | [整理后的测试输入与参考掩码，28.0 MB](https://github.com/emp1y-fs/CaM-PDA/releases/download/paper-tests-v1/cam_pda_icl80_v1.zip) |

首次完整运行连同两个权重下载约 **3.83 GB**。NYUv2 官方 MAT 含全部标注图像，程序只提取其中 654 张官方测试图像；本项目不另行分发其 RGB 和深度数组。整理后的三套测试记录合计约 375 MB。下载、整理数据和保存深度预测总共建议预留约 **6 GB**。已校验文件会复用，下载中断后重新执行相同指令即可继续。完整测试建议使用 NVIDIA 显卡；CPU 也支持，但耗时较长。

**结果在哪里？** 完成后终端会显示指标表和完整保存位置。打开 `cam_pda_test/results/<本次运行>/results.md` 查看汇总，`summary.csv` 查看全图与各区域指标，`per_frame.csv` 查看逐帧指标。`predictions/` 下按测试清单中的帧 ID 建立文件夹，保存 `depth_m.npy`（米制数值深度）和 `depth_color.png`（彩色预览）。每次运行新建结果文件夹。这些固定测试记录没有相机标定内参，因此测试命令输出深度图和评估结果；需要点云时使用上方自带采集案例或自己的标定数据。

想先跑较小的 ICL 80 帧，可运行 `python reproduce.py --root ./cam_pda_test --dataset icl80`；也可选择 `dreds110` 或 `nyu654`。`--prepare-only` 表示只下载准备，`--metrics-only` 表示不保存预测深度图。已有权重或 NYUv2 MAT 文件可直接复用，详见[完整测试选项](docs/REPRODUCIBILITY.md#one-command-benchmark)。

**仓库私有期间：** 需要已获得仓库访问权限的 GitHub 账号。程序读取已有的 Git Credential Manager 登录或 `GH_TOKEN` / `GITHUB_TOKEN`；多个账号时可加 `--github-user 你的GitHub用户名`。结果中不会保存凭据。仓库及发布文件公开后，相同命令无需 GitHub 登录。如果此前安装的是未带测试依赖的版本，先执行一次 `python -m pip install ".[benchmark]"`。

![真实叶片场景的 RGB、ToF 实测深度与 CaM-PDA 深度](assets/blade_depth_showcase.png)

CaM-PDA 在 [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything) 的基础上引入了**平衡置信前置**和**专家矩阵**。置信前置筛选传感器观测，用于视觉深度先验与实测深度的对齐；专家矩阵在条件深度网络中引入独立门控的反射、非平面和边缘专家，进一步预测稠密的米制深度。

**输入：** 一张 RGB 图像及对应的传感器深度。**输出：** 深度预览、数值深度；提供相机内参时还可生成彩色点云。RGB 与传感器深度需要预先配准到同一像素网格。CaM-PDA 内部进行的是深度尺度与结构对齐，不负责两台相机之间的图像配准。

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
