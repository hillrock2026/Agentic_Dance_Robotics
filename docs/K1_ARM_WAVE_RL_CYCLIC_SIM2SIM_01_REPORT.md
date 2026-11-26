# K1_ARM_WAVE_RL_CYCLIC_SIM2SIM_01 — 执行结果报告

日期：2026-11-18
前置任务：`K1_ARM_WAVE_RL_PIPELINE_01`（训练、导出与首次 MuJoCo sim2sim）
策略：`model_2999.pt` 导出的 JIT（未重训、未改动，SHA256 与上轮部署文件一致：
`1fa8b882948ff63f36164ac61ba0ecfe64c15fd8aaeba03e2be6426db7c4a705`）

## 0. 结论摘要

**现有策略在 MuJoCo 中真正连续完成了三个完整 Arm Wave 周期（16.26 s / 813 策略步），全程未摔倒、
未手动重置、未隐藏循环边界。** 三周期的关节跟踪误差几乎相同（手臂 MAE 144.1 / 150.0 / 149.3 mrad），
机器人自身漂移每周期结束时仅 11–14 mm，无逐周期恶化。**保留 model_2999 作为基线，本轮不需要继续训练。**

| 报告问题 | 回答 |
|---|---|
| 是否真正完成三个周期？ | **是**。参考帧按 271 帧周期取模回绕，813 步 = 3×271 帧；逐帧手臂 MAE 的周期间相关系数 0.942 / 0.999，动作真实重复 |
| 循环边界是否平滑？ | **基本平滑**。边界前后 ±20 步窗口 MAE 107.6 / 107.5 mrad，对比动作段基线 88 mrad 仅轻度抬升，无跳变、无摔倒、无异常动作 |
| 当前策略能否重复执行 Arm Wave？ | **能**。三周期误差与行为高度重复（cycle2≈cycle3，相关 0.999） |
| 手臂跟踪误差是否改善或恶化？ | **不恶化**。144.1→150.0→149.3 mrad（cycle1 受益于从参考首帧初始化；2、3 周期持平）。快速挥臂段（参考速度 >10 rad/s）误差升至 258 mrad，属策略能力上限但逐周期稳定 |
| Root 漂移是否逐周期累积？ | **不累积**。机器人自身漂移每周期末 14 / 12 / 11 mm。相对参考的 root 偏差均值 166 / 163 / 162 mm 持平——该偏差主要来自参考轨迹本身每周期转体 ~360° 并平移 232 mm，在腿部参考为常值站姿的 `upper_body_only` 数据下属结构性不可跟踪项，而非机器人走漂 |
| 是否有必要继续训练？ | **本轮不需要**。触发继续训练的四个条件（边界失效 / 误差逐周期增大 / 漂移累积 / 快速挥臂跟不上导致失败）均未出现。可选后续：从 checkpoint 续训可收紧快速挥臂滞后（>10 rad/s 段误差 258 mrad vs 慢速段 125 mrad），但应作为独立新实验，不覆盖本基线 |

## 1. Task A — 周期播放

### 1.1 首尾连续性检查（修改前必查项）

对训练用 `k1_arm_wave.npz`（246 帧 @ 50 Hz）的首尾帧独立测量：

| 项目 | 首尾差异 | 动作内部最大帧间步长 | 结论 |
|---|---|---|---|
| 右肩 pitch 关节角 | **1249 mrad** | 307 mrad | 直接循环会 4 倍于内部最大步长跳变 |
| 左肩 pitch / 双肩 roll | 529 / 440–474 mrad | 307 / 53–60 mrad | 同上，明显不连续 |
| 末帧关节速度 | 全 0 | — | 首帧右肩 +9.79 rad/s、左肩 −3.64 rad/s，速度不连续 |
| trunk 位置（xy） | 231.6 mm | 帧间最大 ~14 mm | 参考位置回绕跳变 |
| trunk 朝向 | 50.4°（geodesic） | 帧间 8.6° | 朝向回绕跳变 |

**结论：不能直接强循环。**

补充发现（对 root 指标的解释至关重要）：参考 trunk 为纯 yaw 旋转（roll/pitch ≈ 0），yaw 从 −7.1°
经 −50.2° 连续变化至 +302.5°，即参考语义是"每循环整体转体约一圈"。而该动作参考的 12 个腿关节
在全部 246 帧内为常值站姿（`upper_body_only` 重定向），物理上腿部不动则躯干无法完成整圈转体——
root yaw / xy 对实际机器人是**结构性不可完全跟踪项**。

### 1.2 最小过渡处理

新增脚本 `make_cyclic_motion.py` 生成 `k1_arm_wave_cyclic.npz`（271 帧 = 246 原始 + 25 过渡帧，0.5 s）：

- 关节角 / 关节速度 / 各 body 位置：**三次 Hermite 插值**，同时匹配末帧位置（末帧速度为 0）与
  首帧位置 + 首帧速度，边界位置与速度均连续；
- 四元数：最短弧 slerp，使 trunk yaw 沿原转向继续前进 50.4° 闭环（每循环净转 ≈358°）；
- 过渡采样点取 α = i/26（i = 1..25），回绕步长不超过首尾固有速度决定的步长；
- **不修改**训练用 `k1_arm_wave.npz`（SHA256 复核与上轮一致 `1d26a1ed…`）。

过渡段自检：过渡内最大帧间步长 137 mrad < 动作内部最大 307 mrad；回绕步长 175.6 mrad，
与动作首尾固有参考速度（9.79 rad/s × 20 ms ≈ 196 mrad/帧）一致，非不连续。

### 1.3 播放逻辑修改

`beyond_mimic.py` 新增 `BeyondMimicPolicyCfg.loop_motion: bool = False`（默认关闭）与
`_frame_id()`：开启时 `row_ids = current_frame % time_step_total`，否则保持原末帧钳制行为。
仅 `K1ArmWaveControllerCfg` 设 `loop_motion = True` 并指向循环 npz —— **其他任务（k1_mj2、
k1_fight、T1/T2 等）行为不变**。

## 2. Task B — 时间对齐

上轮评审确认的 20 ms 偏差来自：误差数组用 `ctrl_step()` 之后已递增的 `current_frame`，而 ghost
画面用递增前的 command frame。新录制脚本统一时序：

```
step k:  update_state()      -> 机器人 t_k 状态
         policy_step()       -> 命令帧 f_k（ghost 被设为 f_k）
         ctrl_step()         -> 物理推进至 t_{k+1}
记录:     机器人 t_{k+1} 状态  vs  参考帧 f_{k+1} = (k+1) mod 271
画面:     渲染前把 ghost 显式重设为 f_{k+1}
```

即**策略命令、误差数据、视频 ghost 三者对每个采样点使用同一参考索引**，且误差比较的机器人时刻
（t_{k+1}）与参考帧时刻严格对应。逐帧记录同时保存 `abs_frame`、`frame`（取模后）、`cycle`、
`phase`、`in_transition`，任何一条误差记录可还原到确切的参考帧。

另外核查了关节顺序（上轮数据正确性的疑点）：MuJoCo XML 的 qpos 关节顺序 == `joint_names`
（csv/real 序），`motion.joint_pos` 为 sim 序并经 `sim2real` 索引映射；部署观测链路的
`real2sim / sim2real` 映射自洽，上轮录制的关节比较并未错序。

## 3. Task C — 三周期 MuJoCo 执行

- 命令：`MUJOCO_GL=egl python scripts/deploy_mujoco_record_cyclic.py --task k1_arm_wave
  --output virtual_k1/outputs/k1_arm_wave_mujoco_cyclic.mp4 --cycles 3`
- 初始化：机器人精确置于参考帧 0（trunk 位置/朝向 + 参考关节角，零速度），随后**无任何 reset**；
- 813 步 @ 50 Hz = 16.26 s = 3 × 271 帧，视频 813 帧带 HUD（时间 / 周期 / 帧号 / 过渡段标记）；
- 两个内部循环边界（abs frame 271、542）全部保留在视频与数据中，未剪辑、未删除失败片段；
- 结果：`stopped_early=False`，`fallen` 全程 0，最小 base height 0.543 m。

## 4. Task D — 动作质量测量

### 4.1 关节跟踪（分周期，动作段 = 源 246 帧，过渡段单列）

| 周期 | 段 | 全 22 关节 MAE | 手臂 MAE | 腿部 MAE | 肩部 4 关节 MAE |
|---|---|---:|---:|---:|---:|
| 1 | 动作 | 85.0 | 144.1 | 49.7 | 235.5 |
| 1 | 过渡 | 96.7 | 172.1 | 49.4 | 285.7 |
| 2 | 动作 | 89.5 | 150.0 | 54.6 | 243.3 |
| 2 | 过渡 | 98.2 | 177.9 | 49.1 | 296.6 |
| 3 | 动作 | 89.4 | 149.3 | 55.5 | 240.9 |
| 3 | 过渡 | 99.4 | 180.3 | 50.2 | 300.7 |

（单位 mrad；肩部 = 左右 shoulder pitch/roll。逐关节 MAE 与最大绝对误差见
`k1_arm_wave_cyclic_joint_mae_per_cycle.csv`，其中 `aaright_shoulder_pitch` 379.8 / 382.7 / 377.2
mrad、最大绝对误差 ~0.93–0.98 rad，左肩 pitch ~307 mrad、最大 ~1.73 rad —— 快速挥臂滞后为主因，
逐周期稳定。）

**误差 vs 参考速度**（动作段合并）：手臂 MAE 125 mrad（<2 rad/s）→ 169（2–4）→ 258（>10 rad/s）；
`aaright_shoulder_pitch` 在 |v|>6 rad/s 的 45 帧 MAE 494 mrad（全程均值 372）。快速挥臂滞后明显
但**不逐周期增大**。误差-相位曲线见 `k1_arm_wave_cyclic_error_vs_phase.png`；
`aaright_shoulder_pitch` 的参考角/实际角/参考速/实际速/误差同一时间轴见
`k1_arm_wave_cyclic_right_shoulder_pitch.png`。

### 4.2 Root 跟踪（每采样点含符号分量，见 root_deviation.csv）

| 周期 | 段 | err x 均值 | err y 均值 | err z 均值 | 范数均值 / 最大 | 周期末范数 | 躯干朝向误差均值 / 最大 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 动作 | −61 | −127 | −8 | 166 / 490 | — | 68.7° / 179.3° |
| 2 | 动作 | −60 | −127 | −8 | 163 / 489 | — | 68.7° / 179.0° |
| 3 | 动作 | −60 | −125 | −8 | 162 / 488 | 176 mm | 68.6° / 179.0° |

- **z 分量全程 −8.4 mm（最大 11.9 mm）**——高度跟踪极好，是"未退化站立"的硬证据；
- 165 mm 量级的 xy 偏差与 ~69°(峰 179°) 的躯干朝向误差**逐周期持平而非累积**，且方向（y 主导）
  与参考自身每周期 +y 平移 232 mm / 转体 ~358° 一致：参考 root 在"走动 + 转圈"，机器人按其
  固定站姿参考原地执行，属 `upper_body_only` 数据的结构性限制，不是策略失稳；
- **机器人自身漂移**（相对自己起点）：每周期最大 16 / 19 / 17 mm，周期末 14 / 12 / 11 mm——
  实际机器人几乎没挪动。

### 4.3 稳定性

| 指标 | 数值 |
|---|---|
| 摔倒步数（base height < 0.40 m 或 roll/pitch > 45°） | 0 / 813 |
| base height 范围 | 0.543–0.573 m（参考 0.555 m） |
| 双脚触地占比 | 99.4%（左脚离地 5 步、右脚 6 步，均为单帧瞬时） |
| 触地时脚底水平速度 | 均值 0.001 m/s，最大 0.090 m/s（无持续滑动） |
| 循环边界异常 | 无。边界 ±20 步窗口 MAE 107.6 / 107.5 mrad（基线 88），实际关节速度最大帧间变化 2.0 rad/s，未见冲击或姿态突变 |

### 4.4 循环边界连续性

两次回绕（t = 5.42 s、10.84 s）处的表现与动作内部其他快速段同量级；过渡段（Hermite 插值，策略
训练时未见过的参考）手臂 MAE 172–180 mrad，仅比动作段高 ~20%，策略平滑跟过过渡并进入下一周期。
视频 HUD 明确标出 TRANSITION 段，未做任何掩饰。

## 5. Task E — 是否继续训练

判定条件逐项核对：

| 触发条件 | 实测 | 是否触发 |
|---|---|---|
| 循环边界跟踪明显失效 | 边界窗口误差仅轻度抬升，无跳变 | 否 |
| 手臂误差逐周期增大 | 144.1 / 150.0 / 149.3，cycle2≈cycle3 | 否 |
| Root 漂移持续累积 | 自身漂移 14→12→11 mm；相对参考误差持平 | 否 |
| 快速挥臂明显跟不上 | 滞后存在（>10 rad/s 段 258 mrad）但稳定可达姿态，未失败 | 部分（可选优化项） |

**决定：保留 `model_2999.pt` 导出策略为当前基线，不启动继续训练。** 播放逻辑、时间对齐与模型配置
已验证正确；剩余误差中快速挥臂滞后与静态右髋/右膝偏置（~170/140 mrad，逐周期恒定）是策略能力
项，可作为后续独立实验（从 checkpoint 续训、保留原模型不覆盖），但不是本轮循环验证的必要条件。

## 6. 交付物

| 交付物 | 路径（相对仓库根） |
|---|---|
| 三周期视频（813 帧 @ 50 fps，16.26 s，HUD 标注周期/帧号/过渡） | `virtual_k1/outputs/k1_arm_wave_mujoco_cyclic.mp4` |
| 三周期逐帧数据（t / 帧 / 周期 / 相位 / 参考与实际关节角速度 / root 四元数 / 接触 / 力 / 滑动 / 摔倒 / 目标与力矩） | `virtual_k1/outputs/k1_arm_wave_mujoco_cyclic.metrics.npz` |
| Root XYZ 偏差逐点数据（含符号分量、范数、朝向误差、接触） | `virtual_k1/outputs/k1_arm_wave_cyclic_root_deviation.csv` |
| 分周期跟踪指标 | `virtual_k1/outputs/k1_arm_wave_cyclic_per_cycle_metrics.csv` |
| 逐关节分周期 MAE / 最大误差 | `virtual_k1/outputs/k1_arm_wave_cyclic_joint_mae_per_cycle.csv` |
| 循环动作文件（271 帧，Hermite 过渡） | `tools/booster_deploy/tasks/beyond_mimic/robots/k1/motions/k1_arm_wave_cyclic.npz` |
| 分析摘要 | `virtual_k1/outputs/k1_arm_wave_cyclic_analysis.md` |
| 图表（右肩时间轴 / 误差-相位 / root xyz / 朝向与高度 / 稳定性 / 逐关节 MAE） | `virtual_k1/outputs/k1_arm_wave_cyclic_*.png` |
| 本报告 | `docs/K1_ARM_WAVE_RL_CYCLIC_SIM2SIM_01_REPORT.md` |

## 7. 修改文件清单

| 文件 | 修改 |
|---|---|
| `tools/booster_deploy/tasks/beyond_mimic/beyond_mimic.py` | 新增 `loop_motion` 配置与 `_frame_id()` 取模回绕；默认关闭，原有末帧钳制行为不变 |
| `tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py` | `K1ArmWaveControllerCfg` 指向 `k1_arm_wave_cyclic.npz` 并开启 `loop_motion`（仅此任务） |
| `tools/booster_deploy/scripts/make_cyclic_motion.py` | 新增：生成含 Hermite 过渡段的循环 npz + 边界连续性自检 |
| `tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py` | 新增：时间对齐三周期录制 + 逐帧丰富指标 + HUD |
| `tools/booster_deploy/scripts/analyze_cyclic_tracking.py` | 新增：分周期/分段指标、root 偏差 csv、边界连续性、图表 |
| `tools/booster_deploy/scripts/analyze_loop_continuity.py`、`analyze_trunk_yaw.py` | 新增：Task A 前置检查脚本（首尾连续性 / trunk yaw 轨迹） |

**未修改**：训练侧全部代码与配置、`model_2999.pt` / `k1_arm_wave.jit`（hash 复核一致）、
训练用 `k1_arm_wave.npz`、原 `deploy_mujoco_record.py`、其他任务（k1_mj2 / k1_fight / T1 / T2）。

## 8. 复现命令

```sh
cd tools/booster_deploy
# 1) 生成循环动作（幂等，输入输出 hash 见上）
python scripts/make_cyclic_motion.py
# 2) 三周期录制（约 1 分钟）
MUJOCO_GL=egl python scripts/deploy_mujoco_record_cyclic.py --task k1_arm_wave \
    --output ../../virtual_k1/outputs/k1_arm_wave_mujoco_cyclic.mp4 --cycles 3
# 3) 分析出全部 csv / 图表
python scripts/analyze_cyclic_tracking.py
# 交互观看（同样启用循环播放）
python scripts/deploy.py --task k1_arm_wave --mujoco
```

## 9. 已知限制

1. 过渡段参考是插值构造的（策略训练未见），其 25 帧内手臂误差略高于动作段；
2. root yaw / xy 参考每周期转体 ~358° 并平移 232 mm，在腿部常值参考下不可物理跟踪，
   相对参考的 root 偏差 ~165 mm 属结构性项；若要消除需含腿部运动的全身参考并重训；
3. 快速挥臂（>10 rad/s）滞后 ~258 mrad、`aaright_shoulder_pitch` MAE ~380 mrad，为策略能力上限；
4. 循环周期为 5.42 s（动作 4.92 s + 过渡 0.5 s），"三周期"含三段过渡，已在视频 HUD 与数据中
   明确标注，未与原始 4.92 s 混淆。
