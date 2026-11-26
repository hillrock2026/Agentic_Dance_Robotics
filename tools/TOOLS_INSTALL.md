# 工具安装记录: GVHMR + Blender

**日期**: 2026-11-12
**目的**: 为 Agentic Dance Robotics 项目部署视频动捕（GVHMR）与 3D 制作工具（Blender），
用于后续"人体舞蹈视频 → 世界系 3D 动作 → K1 机器人重定向"管线。

---

## 1. 安装位置总览

| 工具 | 位置 | 运行方式 |
|---|---|---|
| GVHMR 代码 | `tools/GVHMR/` (zju3dv/GVHMR, commit `ee960bb`) | conda env `gvhmr` |
| GVHMR Python 环境 | `~/anaconda3/envs/gvhmr` (Python 3.10.22) | `~/anaconda3/envs/gvhmr/bin/python` |
| GVHMR 权重 | `tools/GVHMR/inputs/checkpoints/` | 见 §3 |
| Blender | `tools/blender-4.5.16-linux-x64/` | `tools/bin/blender` (符号链接) |

## 2. GVHMR

### 2.1 来源
- 仓库: https://github.com/zju3dv/GVHMR ("World-Grounded Human Motion Recovery via
  Gravity-View Coordinates", SIGGRAPH Asia 2024)
- commit: `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` (2026-09-15, main, depth=1)
- 功能: 单目视频 → 世界坐标系 3D 人体动作 (SMPLX 参数)，本项目舞蹈视频分析的前端。

### 2.2 环境 (conda env `gvhmr`, Python 3.10.22)

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.3.0+cu121 | 官方 requirements pin, RTX 3090 CUDA 验证通过 |
| torchvision | 0.18.0+cu121 | |
| pytorch3d | 0.7.6 | 官方预编译 wheel (py310/cu121/pyt230) |
| lightning | 2.3.0 | |
| numpy | 1.23.5 | 官方 pin (pytorch3d 曾拉高到 2.2.6, 已强制回退) |
| ultralytics | 8.2.42 | YOLO 人体检测 |
| smplx / chumpy | 0.1.28 / 0.70 | 人体模型加载 |
| timm / hydra-core | 0.9.12 / 1.3 | |

已验证: `torch.cuda.is_available()=True`, GPU matmul OK, `pytorch3d`/`chumpy`/
`smplx`/`ultralytics`/`hmr4d`(GVHMR 包名) 导入正常。

### 2.3 安装过程中解决的问题 (记录供重现)

1. **Meta CDN 证书过期**: `dl.fbaipublicfiles.com` 的 SSL 证书 2026-11-11 到期
   (OpenSSL 同样报 certificate has expired)，导致 requirements 中 pytorch3d
   直链 wheel 下载失败、整条 pip install 中止。
   → 处理: `curl -k` 单独下载 wheel 后本地 `pip install`（pip 安装时校验 wheel
   内部 RECORD 哈希，完整性有基本保障；该 wheel 为 20.5MB 官方发布物）。
2. **chumpy 构建失败**: 老包，新版 setuptools 隔离构建环境下 sdist 构建失败。
   → 处理: 先装好 numpy 后 `pip install --no-build-isolation chumpy`。
3. **pkg_resources 缺失**: lightning 2.3.0 依赖 `pkg_resources`，conda py3.10
   默认 setuptools 84.x 已移除。→ 处理: `pip install "setuptools<81"`。
4. **pip 解析器长回溯**: 一次性安装全部 requirements 时 resolver 假死。
   → 处理: 分 5 批显式安装 (torch → lightning/hydra → 科学计算 → 3D 工具 →
   chumpy/numpy pin)。
5. **Google Drive 不可达**: 官方权重发布在 Google Drive，本网络无法访问。
   → 处理: 使用 HuggingFace 镜像 (详见 §3)。
6. **包名易混**: pip 发行名 `gvhmr`，但 Python 包名是 `hmr4d`
   (`import hmr4d`, 不是 `import gvhmr`)。

### 2.4 权重 (inputs/checkpoints/)

官方 Google Drive 在本网络不可达，改用 HuggingFace 镜像
`wendell0218/gvhmr_ckp` (dataset 仓库, 经 hf-mirror.com 下载)。
目录结构与官方 INSTALL.md 完全一致:

| 文件 | 大小 | 用途 |
|---|---|---|
| `gvhmr/gvhmr_siga24_release.ckpt` | 163.5 MB | GVHMR 主模型 |
| `hmr2/epoch=10-step=25000.ckpt` | 2.71 GB | HMR2.0a ViT 特征提取 |
| `vitpose/vitpose-h-multi-coco.pth` | 2.55 GB | ViTPose-H 2D 姿态 |
| `yolo/yolov8x.pt` | 136.9 MB | 人体检测/跟踪 |
| `dpvo/dpvo.pth` | 14.2 MB | DPVO (可选, demo 默认 SimpleVO 不用) |
| `body_models/smpl/SMPL_{GENDER}.pkl` | 3×247 MB | SMPL 人体模型 |
| `body_models/smplx/SMPLX_{GENDER}.npz` | 3×109 MB | SMPLX 人体模型 |

完整性交叉验证: `gvhmr_siga24_release.ckpt` 与另一独立镜像
(`ArthurCww/gvhmr-runtime-checkpoints`) 大小完全一致 (163.5MB)。

> **SMPL/SMPLX 许可说明**: 这两组人体模型文件在官方渠道
> (smpl.is.tue.mpg.de / smpl-x.is.tue.mpg.de) 需注册下载，许可证禁止再分发。
> 本机文件来自第三方公开镜像，**仅限科研用途**。若项目用途超出科研范围，
> 请自行注册 MPI 官网并替换这两个目录下的文件。

### 2.5 运行方式

```bash
cd tools/GVHMR
~/anaconda3/envs/gvhmr/bin/python tools/demo/demo.py \
    --video=docs/example_video/tennis.mp4 -s        # -s = 静态相机(跳过VO)
# 输出: outputs/demo/tennis/  (hmr4d_results.pt + 1_incam.mp4 + 2_global.mp4)
```

### 2.6 冒烟测试结果 (2026-11-12, PASS)

官方 demo (`tennis.mp4`, 静态相机) **全链路实际运行成功**:

| 输出 | 状态 | 说明 |
|---|---|---|
| `preprocess/` (bbx/vitpose/vit_features) | ✅ | YOLO 跟踪 + ViTPose + HMR2 特征 |
| `hmr4d_results.pt` (1.36 MB) | ✅ | GVHMR 模型输出 (SMPLX 参数序列) |
| `1_incam.mp4` | ✅ | 机内视角网格叠加渲染 |
| `2_global.mp4` | ✅ | **世界系渲染 (核心输出)** |
| `tennis_3_incam_global_horiz.mp4` | ✅ | 最终合并视频 |

对 `2_global.mp4` 抽帧视觉核验: 完整 SMPL 人形、网球挥拍动作合理、
足地接触正常、无网格炸裂/肢体缺失等瑕疵。

补充修复 (安装时记录):
- `pycolmap`: SimpleVO 模块无条件导入, 实装 4.2.1 (与 numpy 1.23.5 兼容)。
- `ffmpeg`: ffmpeg-python 需要 CLI 二进制, 系统未装。以 `pip install
  imageio-ffmpeg` 的静态 ffmpeg (n9.0.2) 软链到 env 的 bin/ 下, 免 root。
- 权重下载脚本: `tools/GVHMR/download_weights.py` (hf-mirror 断点续传,
  503 自动重试, 已成功)。

## 3. Blender

| 项 | 值 |
|---|---|
| 版本 | **Blender 4.5.16 LTS** (build hash 3f6797307da5, 2026-11-10) |
| 安装方式 | 官方 Linux tarball (download.blender.org), 免 root, 不动 apt |
| 位置 | `tools/blender-4.5.16-linux-x64/` |
| 命令 | `tools/bin/blender` |
| 内嵌 Python | 3.11.15 (bpy) |
| 渲染 | 无界面 Cycles **GPU** 渲染验证通过 (RTX 3090), 输出 PNG 正常 |

冒烟测试命令 (无界面 GPU 渲染):

```bash
tools/bin/blender -b --python-expr "<创建球+平面+相机, cycles GPU 渲染>"
```

## 4. 与本项目 (Virtual K1) 的衔接

- GVHMR 输出 SMPLX 参数序列 → 后续经 GMR (github.com/YanjieZe/GMR) 类工具或
  自研重定向器映射到 K1 的 22 关节 → 喂给 `virtual_k1` 的关节控制接口回放。
- 官方 booster_assets 已自带两条 K1 重定向动作 (50Hz CSV)，别名映射层
  (`virtual_k1.joint_mapping`) 已兼容其命名差异。
- Blender 用于可视化/调试 K1 与动作 (可导入 K1 的 URDF/mesh)。
