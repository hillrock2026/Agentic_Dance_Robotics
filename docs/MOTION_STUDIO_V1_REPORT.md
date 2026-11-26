# MOTION_STUDIO_V1 交付报告

日期：2026-11-13 ｜ 平台：Ubuntu 24.04 (linux 6.x) ｜ GPU：NVIDIA GeForce RTX 3090 24 GB（驱动 595.84 / CUDA 13.2 runtime，PyTorch cu121 轮子自带运行时）

---

## 1. 实际安装的依赖与版本

| 组件 | 版本 | 位置 / 说明 |
|---|---|---|
| GVHMR 运行时 | Python 3.10.22, torch 2.3.0+cu121, pytorch3d 0.7.6, lightning 2.3.0, numpy 1.23.5, ultralytics 8.2.42, smplx | `/home/liu/anaconda3/envs/gvhmr`（conda 环境，独立于 V1） |
| Motion Studio 核心 | Python 3.14.6, numpy 2.5.4, scipy 1.18.1, pyyaml, pillow, imageio + imageio-ffmpeg, pytest 9.2.0 | 项目 `.venv`（与 Virtual K1 V1 共用、未改动其包集合，仅新增 scipy/pillow/pyyaml） |
| Blender | 4.5.16 LTS（内置 Python，含 numpy） | `tools/blender-4.5.16-linux-x64/blender` |
| Virtual K1 (V1) | mujoco 4.0.1，未做任何修改 | `.venv` / `virtual_k1/` |

V1 兼容性：未改动 `virtual_k1/` 任何文件；`.venv` 仅追加三个纯 Python 依赖，MuJoCo 栈完好。

## 2. GVHMR 官方仓库与预训练权重

- 仓库：`https://github.com/zju3dv/GVHMR`，commit **`ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`**（main），**未修改任何官方源码**。
- 权重（6.2 GB，HuggingFace `wendell0218/gvhmr_ckp`，经 hf-mirror 下载）：`gvhmr_siga24_release.ckpt`、`hmr2/epoch=10-step=25000.ckpt`、`vitpose-h-multi-coco.pth`、`yolov8x.pt`、`dpvo/dpvo.pth`。
- 身体模型：**SMPL**（非 SMPL-X）——GVHMR 输出 `body_pose (T,63)`=21 关节轴角、`betas (T,10)`，配合 `SMPL_NEUTRAL.pkl`（官方许可模型，随权重仓库分发）前向。适配层实测确认（含将 63 维补零至 69 维以喂入 smplx 官方实现）。
- 调用方式：子进程执行官方 `tools/demo/demo.py`（静态机位加 `-s` 跳过 VO；PATH 注入 ffmpeg shim 修复官方 `merge_videos_horizontal` 对系统 ffmpeg 的依赖——不改源码）。适配脚本 `motion_studio/gvhmr/export_raw_npz.py` 在 torch 环境内把 `hmr4d_results.pt` 转为可移植 `gvhmr_raw.npz`（完整保留 incam 参数与 K 矩阵），并用官方 smplx 计算权威关节位置作为对拍基准；同时导出纯 numpy SMPL 模型数组。
- 推理速度（RTX 3090，123 帧/4.92 s 视频）：预处理+推理+渲染 全程 7.8–52 s；纯 HMR4D 推理 0.5–0.8 s。显存峰值 < 6 GB。

## 3. 统一数据结构与坐标/关节定义

- `MotionData`（`motion_studio/core/motion_data.py`）：SMPL 参数（global_orient (T,3)、body_pose (T,63)、betas (T,10)、transl (T,3)、joint_positions (T,24,3)，f8）+ 接触估计 + 编辑日志 + 来源元数据。
- 坐标系 `MS_WORLD`：**+Y up（重力对齐）、右手系、单位米、T-pose 面向 +Z**；Blender 转换 `(x,y,z)→(x,−z,y)`（Y-up→Z-up，右手性保持），往返可逆。
- 旋转表示：SMPL 轴角（导出不变换表示）；插值/编辑内部用 wxyz 四元数（SLERP、四元数幂、特征向量四元数均值——先规范化 w≥0 半球避免主值歧义）。
- 关节顺序：SMPL 24 关节（`pelvis, left_hip, right_hip, spine1, …`，见 `motion_schema.py`），父索引与官方 kintree 一致。
- 校验器：字段完整性、形状、NaN/Inf、duration=fps/frames 一致性、轴角幅值 < 2π、接触时间轴对齐（`validate_motion_dict`，43 项测试覆盖）。

## 4. 三方一致性（核心质量证据）

| 动作 | 官方 smplx ↔ numpy FK | Blender ↔ 导出 npz | 导出 ↔ 源（roundtrip） |
|---|---|---|---|
| Arm Wave (071) | 3.17e-07 m | 1.47e-06 m | 0.00e+00 |
| Indian Step (082) | 2.67e-07 m | 1.38e-06 m | 0.00e+00 |
| Robocop (056) | 2.69e-07 m | 2.14e-06 m | 0.00e+00 |

即：GVHMR 官方前向、Motion Studio 纯 numpy 前向、Blender 骨架动画三者关节位置在**微米级**一致（剩余误差为 float32 与 Blender 矩阵分解精度）。Blender 导入公式经实证探针确认：根骨 `pose=rest@basis`，子骨 `pose=parent_pose@(rest_p⁻¹rest_i)@basis`（Blender 4.5 实测）。

## 5. 三种街舞动作端到端验证

素材：《87 个 HIPHOP 元素合集》Bobylien 舞蹈教室（Bilibili BV1794y1h77B），1920×1080@25fps，静态机位，单人，全身可见，无遮挡。元素-分段映射经本地 OCR（RapidOCR）确认（segment N = 元素 N）。

| 项目 | A. Arm Wave (seg71) | B. Indian Step/Top Rock (seg82) | C. Robocop/Popping (seg56) |
|---|---|---|---|
| 测试目的 | 上肢恢复+幅度编辑 | 下肢+脚部接触+速度调整 | 快速局部运动+时间轴编辑 |
| 前处理 | 直接使用 | **画面左侧静态人形干扰 YOLO 跟踪 → 裁剪 x≥770** | 同左（裁剪） |
| 恢复质量 | 腕速 1.02 m/s，上肢>下肢 ✓ | 踝高振荡 0.21–0.27 m（跨步）✓ | 腕速 2.09 m/s（快速收缩）✓ |
| 接触统计 | L99%/R70%/双70%/空1% | L56%/R66%/双51%/空29%（交替支撑）✓ | L37%/R48%/空48%（快速移重心）✓ |
| 编辑操作 | arms 幅度 0.5× | trim(10,110)+速度 0.7×（4.92→5.72s） | 局部 arms 0.6×+25→30fps 重采样 |
| 编辑量化 | 肩角幅度 1.293→0.647 rad（精确 0.5×） | 帧数 123→143，接触区间随时间轴重检 | 帧数 122→146，时长守恒 4.88s |
| 导出 | clean+edited npz/json/pc2/blend | 同左 | 同左 |

裁剪说明：082/056 原始段中 GVHMR 的 YOLO 单目标跟踪锁定画面左侧静态人形（bbx 宽度 std 仅 2px，人体相对姿态 std 0.007——刚体冻结），恢复结果忠实于"静止目标"。裁剪掉干扰区域后恢复正确。该限制已记录于 README §7，属 GVHMR 单人跟踪在干扰人形场景的已知能力边界，**未伪造任何通过**。

## 6. 自动化测试

`43 passed`（`.venv/bin/python -m pytest motion_studio/tests/`），分四组：旋转数学（往返、SLERP 等角速度/端点/符号、四元数幂、均值）、核心（numpy FK 对拍 smplx、校验器 5 种故障注入、Y-up 断言、合成交替步态接触）、编辑+导出（速度恒等/抽样、trim、重采样、幅度恒等/角幅度/窗口、关键帧 SLERP 中点、平滑、roundtrip 逐位、独立加载字段完整性）、GVHMR 处理器（探测、job 状态、校验、失败路径、来源记录）。

## 7. 验收清单

| # | 验收项 | 结论 | 证据 |
|---|---|---|---|
| Test 1 | GVHMR 集成（官方调用/街舞输入/真实恢复/读取转换/失败处理/日志） | **PASS** | 3 job COMPLETED + job.json/job.log + FAILED 路径测试 |
| Test 2 | 数据结构（字段/维度/时间轴/坐标/单位/无 NaN） | **PASS** | 校验器 + 43 测试 + 三个 npz 校验 |
| Test 3 | Blender 可视化（骨架/动画/逐帧/坐标旋转正确/无扭曲镜像） | **PASS** | 3 个 .blend + FK 验证 ≤2.1e-6 m + 54 张渲染图 |
| Test 4 | 脚部接触（分脚/区间/显示/置信度/标注为估计） | **PASS** | 接触区间+状态动画标记+`is_estimate` 标记+合成步态测试 |
| Test 5 | 动作编辑（关键帧/时间轴/速度/幅度/可重播/原始不破坏） | **PASS** | edited_motion.* 三例 + 编辑日志 + 量化指标 |
| Test 6 | 标准化导出（npz+json/独立加载/一致性/坐标关节明确） | **PASS** | roundtrip 0 误差 + 独立加载测试 |
| Test 7 | 三种街舞端到端 | **PASS**（082/056 需裁剪预处理） | §5 表格 + outputs/motions/ |
| — | 多人视频 | **NOT TESTED**（明确不支持，范围外） | README §7 |
| — | 动态机位（SimpleVO 路径） | **NOT TESTED**（接口已留 `--moving-cam`，测试视频均为静态机位） | processor.py |

## 8. 尚未解决 / 待改进

1. 画面含静态人形干扰时 GVHMR 跟踪锁错目标——需人工裁剪或换多人跟踪前端（V2 可做自动人物选择）。
2. `SMPL_MESH` 蒙皮网格不含 pose blendshape（毫米-厘米级偏差）；精确视觉以 `SMPL_EXACT_MESH`(PC2) 为准。
3. Foot Contact 为几何估计；未来 K1 落地时可结合 GVHMR 的 static_conf 或物理仿真做二次校准。
4. 编辑操作目前通过 JSON 管线驱动；Blender 内交互编辑（NLA/姿态编辑器回写 MotionData）为 V2 议题。
5. 未执行任何 git commit/push（遵守边界）。

## 9. 交付物索引

```
motion_studio/
├── assets/smpl/smpl_neutral_core.npz        纯 numpy SMPL 模型（自官方 pkl 导出）
├── assets/test_videos/                      裁剪后的测试视频（082/056）
├── config/motion_studio.yaml                路径/阈值配置
├── core/  motion_schema / motion_data / smpl_model / rotations / coordinate_system
├── gvhmr/ processor.py（批量接口+job 状态）/ export_raw_npz.py（torch 侧适配器）
├── blender/ import_motion / validate_fk / render_frames
├── processing/ foot_contact / motion_retimer / motion_scaler / keyframe_editor / motion_filter
├── export/ npz_exporter（export/load/validate）/ mesh_cache（PC2）
├── scripts/ process_video / process_batch / pipeline / export_motion / launch_blender
├── tests/  conftest + 4 测试模块（43 项）
├── docs/README.md                           安装+使用文档
└── outputs/ gvhmr/<jobs>/ + motions/<jobs>/（全部中间与最终产物）
```
