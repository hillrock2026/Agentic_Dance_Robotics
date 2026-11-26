# Motion Studio V1 — 使用文档

Motion Studio 将普通街舞视频转换为**可观察、可编辑、可导出**的标准化人体三维动作数据：

```
dance_video.mp4 → GVHMR 动作恢复 → SMPL 参数 → MotionData 统一结构
  → Blender 可视化/编辑 → 脚部接触检测 → clean_motion.npz
```

## 1. 环境与安装

本仓库使用**三个互不干扰的运行时**（Virtual K1 的 `.venv` 环境不受影响）：

| 运行时 | 解释器 | 用途 |
|---|---|---|
| core | `.venv`（Python 3.14，项目根） | Motion Studio 核心：数据结构、编辑、导出、测试 |
| gvhmr | `/home/liu/anaconda3/envs/gvhmr`（Python 3.10 + torch 2.3.0+cu121） | GVHMR 官方推理（子进程调用，Motion Studio 不直接 import torch） |
| blender | `tools/blender-4.5.16-linux-x64/blender`（内置 Python） | 骨架导入、动作播放、FK 验证、渲染 |

已有资产（无需再次下载）：

- GVHMR 官方仓库：`tools/GVHMR`（commit `ee960bb`），预训练权重 6.2 GB 已在 `tools/GVHMR/inputs/checkpoints/`（来源 `wendell0218/gvhmr_ckp`）。
- SMPL 身体模型：`tools/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`（官方许可模型）。
- Motion Studio 核心用纯 numpy SMPL 前向模型（由适配器从上述 pkl 导出）：`motion_studio/assets/smpl/smpl_neutral_core.npz`。

核心环境依赖（已装好）：numpy、scipy、pyyaml、pillow、imageio、imageio-ffmpeg、pytest。

## 2. 快速开始

### 2.1 单视频动作恢复

```bash
cd /home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics
.venv/bin/python motion_studio/scripts/process_video.py --video inputs/hiphop87_elements/segment_071.mp4
```

输出 job 目录：`motion_studio/outputs/gvhmr/<视频名>_<sha256前8位>/`，包含官方 `hmr4d_results.pt`、
适配层产物 `gvhmr_raw.npz`、预览视频（`1_incam.mp4` 叠加网格 / `2_global.mp4` 世界视角）、
`job.json`（状态 PENDING/RUNNING/COMPLETED/FAILED、日志、sha256、推理耗时）。

摄像机运动视频（跟拍）去掉 `-s`：`--moving-cam`（启用 SimpleVO 相机估计）。

### 2.2 批量处理

```bash
.venv/bin/python motion_studio/scripts/process_batch.py --folder motion_studio/assets/test_videos
```

### 2.3 端到端管线（恢复 → 接触检测 → 导出 → Blender → FK 验证 → 编辑）

```bash
.venv/bin/python motion_studio/scripts/pipeline.py --job segment_071_343ee79d \
    --edit motion_studio/outputs/edits/arm_wave_edits.json
```

产物在 `motion_studio/outputs/motions/<job>/`：

| 文件 | 内容 |
|---|---|
| `clean_motion.npz` | 标准化动作（SMPL 参数 + 关节位置 + 接触估计 + 全部元数据） |
| `clean_motion.json` | 人类可读元数据、数组形状、编辑日志、接触区间 |
| `clean_motion.pc2` | 精确 SMPL 网格逐帧序列（Blender Mesh Cache 直接回放） |
| `clean_motion.blend` | 已绑定的骨架动画场景（骨架+接触标记+网格+相机） |
| `edited_motion.*` | 编辑后的版本（原始文件永不覆盖） |
| `blender_fk_report.npz` | Blender↔numpy FK 一致性验证报告 |
| `pipeline_summary.json` | 全流程量化摘要 |

### 2.4 在 Blender 中查看 / 编辑

```bash
# 交互式打开（播放、逐帧、姿态编辑）
.venv/bin/python motion_studio/scripts/launch_blender.py --blend motion_studio/outputs/motions/segment_071_343ee79d/clean_motion.blend

# 从 npz 重建场景
.venv/bin/python motion_studio/scripts/launch_blender.py --npz motion_studio/outputs/motions/<job>/clean_motion.npz
```

场景内容：`MS_<joint>` 骨架（24 骨骼，四元数关键帧，根骨骼含平移通道；显示为细线条 STICK 模式）、
**姿态估计风格线条人**（`STICK_*` 彩色肢体线段 + `JOINT_*` 关节点球，默认模式，替代人体网格）、
4 个接触状态球（绿=左支撑 红=右支撑 蓝=双支撑 灰=腾空，按帧缩放动画）、
`CAM_FRONT/CAM_SIDE/CAM_TOP` 相机。时间轴已按动作帧率设置（25 fps）。
需要人体网格时加 `--mode mesh`（`SMPL_MESH` 骨架蒙皮近似网格 + `SMPL_EXACT_MESH` PC2 精确网格）。

**2D 线条骨骼图/GIF（OpenPose 风格，无需 Blender）**：

```bash
.venv/bin/python motion_studio/export/stick_figure.py --npz <clean_motion.npz> --gif out.gif --sheet sheet.png --view front
```

## 3. 编辑操作（edits JSON）

```jsonc
[
  {"op": "trim", "start_frame": 10, "end_frame": 110},
  {"op": "set_speed", "factor": 0.7},                          // >1 加速, <1 减速
  {"op": "resample_fps", "new_fps": 30.0},                     // 时长不变
  {"op": "scale_amplitude", "factor": 0.5, "joint_group": "arms",
   "frame_range": [30, 90]},                                    // 可选时间窗
  {"op": "smooth", "window": 7, "sigma": 1.5},
  {"op": "keyframes", "add": [0, 100], "set_pose": [
      {"frame": 0, "body_pose": [/* 63 轴角 */], "joints": [16,18,20]}],
   "interpolate": true}
]
```

关节组：`all / root / upper_body / lower_body / arms / left_arm / right_arm / legs /
left_leg / right_leg / spine / head_neck / left_foot / right_foot`（定义见
`motion_studio/core/motion_schema.py::JOINT_GROUPS`）。

旋转编辑全部使用四元数 SLERP / 四元数幂（先规范化到 w≥0 半球），**从不**对欧拉角或四元数分量做线性插值/缩放。

## 4. Python API

```python
import sys; sys.path.insert(0, '.')
from motion_studio.core.motion_data import load_gvhmr_raw, MotionData
from motion_studio.core.smpl_model import SMPLForwardKinematics
from motion_studio.export.npz_exporter import export_motion, load_motion, validate_motion
from motion_studio.processing.foot_contact import FootContactDetector
from motion_studio.processing.motion_retimer import set_speed, trim
from motion_studio.processing.motion_scaler import scale_amplitude
from motion_studio.processing.keyframe_editor import KeyframeEditor
from motion_studio.processing.motion_filter import smooth
from motion_studio.gvhmr.processor import GVHMRProcessor

fk = SMPLForwardKinematics('motion_studio/assets/smpl/smpl_neutral_core.npz')
proc = GVHMRProcessor()
job = proc.process_video('my_dance.mp4')            # 或 get_result(job_id) 复用已完成结果
md = load_gvhmr_raw(job['raw_npz'], smpl_fk=fk)     # 内置 numpy-FK 与官方 smplx 对拍
md.contacts = FootContactDetector().detect(md.joint_positions, md.motion_fps)
edited = set_speed(md, 0.8)                          # 编辑产生新版本，原数据不动
export_motion(edited, 'edited.npz')                  # + 同名 .json
md2 = load_motion('edited.npz')                      # 独立程序可直接读回
```

## 5. 数据结构（clean_motion.npz）

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `motion_fps` / `source_fps` | f8 | 动作帧率 / 原视频帧率（帧与原视频 1:1 对应） |
| `frame_count` / `duration` / `timestamps` | i8 / f8 / (T,) | 时间轴（均匀采样） |
| `global_orient` | (T,3) f8 | 根朝向，SMPL 轴角，世界系 |
| `body_pose` | (T,63) f8 | 21 个身体关节局部旋转，轴角 |
| `betas` | (T,10) f8 | SMPL 形状参数（静态形状，逐帧保留） |
| `transl` | (T,3) f8 | 根（骨盆）平移，米 |
| `joint_positions` | (T,24,3) f8 | 世界系关节位置（FK 重建） |
| `contact_left/right` | (T,) bool | 脚部接触**估计**（高度+速度+滞回，非力测量） |
| `contact_conf_left/right` | (T,) f8 | 置信度 [0,1] |
| `contact_state` | (T,) i8 | 0腾空 1左支撑 2右支撑 3双支撑 |
| `contact_intervals_json` | str | 接触区间 [起帧,止帧] |
| `joint_names` / `parents` | (24,) | SMPL 24 关节名与父索引 |
| `coordinate_json` | str | 坐标系定义：**+Y up、右手系、米、SMPL +Z 朝向** |
| `rotation_representation` | str | `smpl_axis_angle`（导出不改变旋转表示） |
| `metadata_json` | str | 来源 sha256、GVHMR commit/ckpt、编辑日志、版本 |

## 6. 测试

```bash
.venv/bin/python -m pytest motion_studio/tests/ -v
```

覆盖：旋转数学（往返/SLERP 测地线性质）、numpy SMPL FK 对拍官方 smplx、数据校验器
（NaN/形状/缺失/非法轴角）、脚部接触（合成交替步态）、重定时（SLERP 抽样恒等性）、
幅度（恒等/角幅度/窗口无跳变/单关节弧长单调）、关键帧（中点 SLERP、原始数据不可变）、
平滑（抖动下降）、导出（逐位 roundtrip、独立加载、字段完整性）、GVHMR 处理器（探测/校验/失败路径）。

## 7. 已知限制

- **单目恢复的固有误差**：GVHMR 从单视角视频恢复，脚部接触为几何估计（已标记 `is_estimate`），非真实力测量。
- **画面中的静态人形干扰**：合集视频左侧的静态人物/海报会误导 GVHMR 的单人跟踪（YOLO one-track），导致恢复成"冻结姿态"。解决：裁剪输入画面只保留舞者（见 `assets/test_videos/*_cropped.mp4`）。
- **多人视频**：当前不支持，处理器会在跟踪歧义时产生不可靠结果。
- `SMPL_MESH`（骨架蒙皮网格）不含 pose blendshape（误差毫米-厘米级）；`SMPL_EXACT_MESH`（PC2）为逐帧精确网格。
