# K1_ARM_WAVE_ROOT_REFERENCE_CORRECTION_01 — 执行结果报告

日期：2026-11-18
前置：`K1_ARM_WAVE_RL_PIPELINE_01`（训练/导出/首次 sim2sim）、
`K1_ARM_WAVE_RL_CYCLIC_SIM2SIM_01`（三周期循环验证）。
约束遵守：未重训、未修改 `model_2999.pt`/JIT（部署 hash 复核一致）、未覆盖任何原始
数据（原 CSV/NPZ/cyclic 文件全部原样保留）、未触及其他动作任务与物理模型。

## 0. 结论摘要

1. **Root 异常来源（Task A）**：原始参考 root 的 xy 游走（路径 1.47 m）与 yaw 净转
   **+309.6°**（peak-to-peak 352.7°）都是**真实传播自源动作**——舞者在源视频中确实
   走位并整体转身近一圈；`upper_body_only` 重定向在钉死腿部站姿的同时把 pelvis
   平移与 heading 1:1 写入 root（代码 `k1_action/retargeting.py::_retarget_root`），
   两者共同构成"腿固定、root 走动转圈"的语义不一致参考。转换链本身（坐标变换、
   四元数顺序、重采样、csv↔npz 一致性）未发现 bug。
2. **新参考（Task B/C）**：`k1_arm_wave_upper_body_stationary_root` —— root 固定为
   官方站姿（pos (0,0,0.555023) m、identity 四元数），22 关节轨迹**逐帧零差异**
   保留原 Arm Wave；body 参考用部署同款 K1_22dof.xml 重新 FK，root/body/关节语义
   自洽；足底距地 +5.2 mm 恒定、与原站姿几何一致；循环 transition 重新生成。
3. **对照（Task D）**：新旧 22 关节轨迹最大差异**恰好 0.0**（手臂动作原样保留）；
   root xy 路径 1.472 m→0、yaw 净转 309.6°→0°；旧参考下双脚在世界系中被迫
   "滑行" 1.41/1.78 m（语义不一致的几何证明），新参考为 0。
4. **现有 JIT 基线（Task E）**：同一未重训策略在新参考下三周期全部完成，无摔倒；
   关节 MAE all/arms/legs = 71.5–72.6 / 128.0–129.2 / 34.6–35.3 mrad（旧参考：
   85–90 / 144–150 / 50–56），root 偏差均值从 ~165 mm 塌缩至 **19–23 mm**，躯干
   朝向误差从 68.7°/峰值 179° 降至 **3.3°**。边界窗口 MAE 67.7 mrad 低于动作段
   均值（边界惩罚消失）。**注意：误差变化证明旧参考在指令层就物理不自洽，不能
   归因为策略能力变化**——本实验只是同一策略换了语义一致的新指令。

## 1. Task A — Root 异常追溯（详见 `docs/arm_wave_root_anomaly_trace.md`）

数据链逐环节核验：`segment_071.mp4` → GVHMR（pelvis xy 路径 1.311 m，真实）→
motion_studio 编辑（pelvis 与 GVHMR 完全一致；sha256 与 metadata 记录相符）→
`_retarget_root()`（yaw=pelvis 前向投影 heading、xy=pelvis 位移 1:1、z=常数站姿高）
→ trajectory.csv → booster CSV → NPZ（npz trunk 与 csv root 逐帧相对旋转 <0.09°）。

关键数值（符号安全 yaw + `np.unwrap(period=360)`）：
- root xy：路径 1.472 m，净位移 232 mm，y 向最大偏移 431 mm；z 恒 0.555023 m；
- root yaw：起始 −7.13° → 最低 −50.2° → 终值 +302.5°，**净转 +309.6°**，峰值
  速率 429°/s（第 0 帧）；roll/pitch 全程恒 0（yaw-only 重定向）；
- 源视频人眼核验：舞者从 ~45° 侧身开始、结束时背对镜头且位置移动，与上述一致。

**分析方法更正记录**：本任务中途曾用"度值 + 默认 2π 周期的 `np.unwrap`"得出
"净 +31.3°、无整圈转体"的错误读数；正确重算后确认任务书给出的 +309.6° 属实。
相关脚本（`trace_root_anomaly.py`、`make_cyclic_motion.py`、`analyze_trunk_yaw.py`）
已全部修正为显式 `period=360`（度）或 `2π`（弧度），并回归验证与上一轮报告
数值一致（上轮"每循环转体约一圈"结论成立）。

## 2. Task B — Standing Arm Wave 参考定义

- **Root 位置 = 常量 (0, 0, 0.555023) m**：xy 取重定向器自身的站姿原点（frame-0
  `stance_xy=(0,0)`），z 取官方站姿 root 高度（官方 CSV 首行，即 upper_body_only
  一直在用的常数）——不是盲目清零；
- **Root 朝向 = 常量 identity**（xyzw [0,0,0,1]）：官方 K1 站姿朝向约定，同时与
  MuJoCo 部署默认初始朝向一致；
- **22 关节轨迹逐帧原样保留**：手臂波浪的相位、节奏、幅度不变（对照差异=0）；
  腿部本就是常值站姿；头部关节保留；
- **躯干局部运动的说明**：K1 为 22 自由度（头 2 + 臂 8 + 腿 12），**无躯干关节**
  （retarget 时 spine 已列入 dropped_human_dof）。原参考中"躯干运动"只存在于
  root 的 yaw/xy 通道：roll/pitch 恒为 0（无局部侧倾可保留），yaw 是与钉死腿部
  矛盾的全局转体——按站姿语义固定。上肢剩余协调信息由保留的头/臂关节承载。

## 3. Task C — 新数据生成与自检

新增文件（原始数据零改动）：

| 文件 | 说明 |
|---|---|
| `datasets/k1_dance_v1/arm_wave_upper_body_stationary_root/trajectory.csv` | 31 列，246 帧 @50 Hz，4.92 s，格式同原 trajectory |
| `datasets/k1_dance_v1/arm_wave_upper_body_stationary_root/metadata.json` | 派生说明 |
| `virtual_k1/assets/booster_assets/motions/K1/k1_arm_wave_upper_body_stationary_root.csv` | 官方 29 列格式 |
| `tools/booster_deploy/.../motions/k1_arm_wave_upper_body_stationary_root.npz` | 部署格式（MuJoCo FK 重算 body 参考） |
| `tools/booster_deploy/.../motions/k1_arm_wave_upper_body_stationary_root_cyclic.npz` | 循环版（271 帧 = 246+25 过渡） |

自检结果：
- joint_pos / joint_vel 与源 npz 最大差异 **0.0**（原样复制）；
- trunk pos 恒 (0,0,0.555023)、trunk quat 恒 identity（最大偏差 0.0）；
- 足底几何：左右脚底最低点 +5.17/+5.20 mm，全程恒定，与原参考 frame-0 站姿几何
  完全一致（无穿地、无悬空）；
- 循环边界：过渡段为 Hermite 插值（与旧版相同的关节过渡质量）；root 过渡为
  常数→常数，body pos 过渡最大帧步 14.1 mm（旧版 53.7 mm），wrap 步 16.4 mm
  （旧 64.7 mm），yaw 过渡 0.00°/帧——**边界比旧循环更平滑**；
- 关节速度边界连续：进入过渡 0.49 rad/s，过渡末 7.79 → frame0 9.79 rad/s
  （Hermite 端点速度匹配，与关节数据固有特性一致）。

## 4. Task D — 新旧参考对照

数据（`virtual_k1/outputs/arm_wave_root_correction/compare_summary.json`）：

| 项 | 原参考 | 新参考（stationary root） |
|---|---:|---:|
| root xy 路径 / 净位移 | 1.472 m / 232 mm | 0 / 0 |
| root z 范围 | 0.555023（恒） | 0.555023（恒） |
| root yaw 净转 / peak-to-peak | +309.6° / 352.7° | 0° / 0° |
| 22 关节位置/速度最大差异 | — | **0.0 rad / 0.0 rad/s** |
| 双脚世界系滑行路径（左/右） | 1.41 m / 1.78 m | 0 / 0 |
| 脚底高度（左/右） | +5.17 / +5.20 mm | +5.17 / +5.20 mm（相同） |

可视化：`reference_comparison.mp4`（左=原参考 ghost 橙色：平移+整圈转体；右=新参考
ghost 绿色：原地站姿挥手；固定机位、同步 246 帧），
`compare_root_curves.png`（root xyz+yaw 曲线）、`compare_foot_traces.png`
（足底 xy 轨迹与高度）、`compare_joint_diff.png`（逐帧关节差异=0）。

结论：新版本**完全消除**了与站姿语义矛盾的全局平移/转体（脚底滑行 1.4–1.8 m
的定量化证明旧参考的"腿部常值"与"root 游走"互相矛盾），同时 Arm Wave 手臂
动作逐位保留，足底接触几何不变。

## 5. Task E — 现有 JIT 在新参考下的三周期基线（可选验证，已执行）

设置：`--task k1_arm_wave_usr`（新注册，仅新增配置，同一 `k1_arm_wave.jit`、
同 PD/增益），MuJoCo 初始化于参考 frame 0，连续 3×271 帧，无重置。

| 周期 | 段 | MAE all | arms | legs | root err mean/max (mm) | 躯干朝向 (°) |
|---|---|---:|---:|---:|---:|---:|
| 1 | 动作 | 71.5 | 128.0 | 34.6 | 20/44 | 3.3 |
| 1 | 过渡 | 64.5 | 98.6 | 42.6 | 15/24 | 2.5 |
| 2 | 动作 | 72.6 | 129.2 | 35.1 | 25/44 | 3.3 |
| 2 | 过渡 | 64.4 | 98.5 | 42.2 | 16/24 | 2.5 |
| 3 | 动作 | 72.6 | 129.0 | 35.3 | 25/44 | 3.4 |
| 3 | 过渡 | 64.4 | 98.5 | 42.2 | 16/24 | 2.5 |

稳定性：摔倒 0 步；最小 base height 0.549 m；触地时脚底滑动峰值 4–6 mm/s
（旧参考 90 mm/s）；瞬时离地左 2 步/右 1 步。循环边界 ±20 帧窗口 MAE 67.7 mrad，
**低于**动作段均值——边界惩罚消失。

与旧参考同策略对比（动作段均值）：

| 指标 | 旧参考 | 新参考 | 变化 |
|---|---:|---:|---:|
| MAE all / arms / legs (mrad) | 89.5 / 150.0 / 54.6 | 72.6 / 129.2 / 35.1 | −19% / −14% / −36% |
| root 偏差 mean / max (mm) | 163 / 489 | 25 / 44 | −85% / −91% |
| 躯干朝向误差 mean / max (°) | 68.7 / 179 | 3.3 / ~7 | −95% |
| 边界窗口 MAE (mrad) | 107.6 | 67.7 | −37% |

**归因声明（按任务要求）**：以上改善来自参考指令语义一致化（旧参考让观测中的
参考朝向命令持续旋转、参考位置持续游走，且物理上要求钉死的双脚滑行），不能
解读为策略能力变化。反向说明：上一轮测得的 ~165 mm root 偏差与 69° 朝向误差
主要是**参考不自洽**的度量，而非机器人失稳（机器人自身漂移当时仅 11–14 mm）。

逐帧数据：`k1_arm_wave_usr_mujoco_cyclic.metrics.npz`；视频
`k1_arm_wave_usr_mujoco_cyclic.mp4`（813 帧，HUD 标注）；分周期 csv/图表
（`k1_arm_wave_usr_*`）均在 `virtual_k1/outputs/arm_wave_root_correction/`。

## 6. 修改文件清单

| 文件 | 修改 |
|---|---|
| `tools/booster_deploy/scripts/trace_root_anomaly.py` | 新增：Task A 全链追溯 |
| `tools/booster_deploy/scripts/make_stationary_root_reference.py` | 新增：Task B+C 参考生成（CSV×2+NPZ+自检） |
| `tools/booster_deploy/scripts/make_cyclic_motion.py` | 参数化（--src/--dst/--transition-frames）；修正验证段 unwrap 单位 bug |
| `tools/booster_deploy/scripts/compare_references.py` | 新增：Task D 对照（曲线/足底/关节差异/对比视频） |
| `tools/booster_deploy/scripts/analyze_cyclic_tracking.py` | 参数化（--metrics/--outdir/--prefix） |
| `tools/booster_deploy/scripts/analyze_trunk_yaw.py` | 修正 unwrap 周期显式化 |
| `tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py` | 新增 `k1_arm_wave_usr` 任务（同 JIT/增益，指向新参考） |
| `docs/arm_wave_root_anomaly_trace.md` | 新增：Task A 追溯记录 |

**未修改**：训练侧代码与配置、`model_2999.pt`/`k1_arm_wave.jit`、原始
`trajectory.csv`/`k1_arm_wave.npz`/`k1_arm_wave_cyclic.npz`、`k1_arm_wave` 任务
配置、其他街舞动作、K1 物理模型。

## 7. 复现方法

```sh
cd tools/booster_deploy
python scripts/trace_root_anomaly.py                       # Task A 追溯
python scripts/make_stationary_root_reference.py           # Task B+C 新参考
python scripts/make_cyclic_motion.py --src \
    tasks/beyond_mimic/robots/k1/motions/k1_arm_wave_upper_body_stationary_root.npz
MUJOCO_GL=egl python scripts/compare_references.py         # Task D 对照
MUJOCO_GL=egl python scripts/deploy_mujoco_record_cyclic.py --task k1_arm_wave_usr \
    --output ../../virtual_k1/outputs/arm_wave_root_correction/k1_arm_wave_usr_mujoco_cyclic.mp4 \
    --cycles 3                                             # Task E 基线
python scripts/analyze_cyclic_tracking.py --metrics \
    ../../virtual_k1/outputs/arm_wave_root_correction/k1_arm_wave_usr_mujoco_cyclic.metrics.npz \
    --prefix k1_arm_wave_usr
```

## 8. 遗留与建议

1. 新参考已具备重训条件：若以 `k1_arm_wave_upper_body_stationary_root` 训练新策略，
   参考语义将完全自洽（预期可进一步压低手臂误差）；应作为新实验，不覆盖本基线。
2. `k1_arm_wave_usr` 任务的策略仍是在旧参考上训练的，其在新参考下的表现为
   zero-shot 基线；正式采用新参考语义应重训。
3. 源动作的转身/走位若未来要做全身版本，需走全身重定向（含腿部参考），不属于
   本任务范围。
