# K1_ACTION_DATA_CONVERSION_V1 — 交付报告

日期：2026-11-16 ｜ 运行环境：项目 `.venv`（Python 3.14, mujoco 4.0.1）+ Virtual K1 V1（只读复用，未修改）

---

## 1. 实际完成的转换链

```
Motion Studio clean_motion.npz (SMPL 24关节, 25fps)
  → load_human_motion/validate (结构+数值+时间轴)
  → FullBodyRetargeter (分段方向 IK, 运行时坐标系标定)
  → 5帧滑动平均 + 2%限位裕度 (越限逐帧记录)
  → resample 25→50 Hz (时长守恒, root四元数SLERP)
  → validate_k1_trajectory (限位/速度/加速度/NaN/时间轴)
  → export: trajectory.csv(官方29列+时间轴) + metadata.json + validation.json
  → Virtual K1: A运动学回放 + B物理回放(PD, 只读复用V1)
  → datasets/k1_dance_v1 + manifest.json
```

## 2. 输入/目标格式实测结论（详见 K1_ACTION_FORMAT_ANALYSIS_V1.md）

- **输入**：`clean_motion.npz` — SMPL（非 SMPL-X），(T,24,3) 世界关节位置 + 轴角参数，25fps，MS_WORLD(+Y up, 米)，含脚部接触估计与完整溯源。逐字段实测确认。
- **目标**：官方 `k1_fight_001.csv`/`k1_mj2_seg1.csv` — **29 列**（root_pos 3 + root_quat **xyzw** 4 + 22 关节**绝对弧度角**），首帧=官方站立位（实测逐位一致），**无时间元数据**（fps 未标注），关节列名与 XML 存在 2 处命名差异（`left/right_shoulder_pitch` ↔ `aaleft/aaright_…`）。官方 CSV 是动作序列样例而非带元数据的训练封装格式——本任务输出**官方兼容的 29 列 + 显式时间轴与元数据封装**，两层分开。

## 3. 关节映射与坐标（要点）

- 映射表：`k1_action/config/human_to_k1_joint_mapping.yaml`（22 关节全覆盖 + DOF 缺失清单 + 置信标记）。
- K1 关节顺序与 XML 逐个核对一致、与官方 CSV 列序一致（测试断言）。
- 坐标系：人/K1 躯干坐标系**运行时**由双肩连线+重力标定（GVHMR 世界朝向任意，不猜名字方向）；K1 前向经脚尖几何核实（1.7° 内）；左右不镜像、竖直方向一致均有测试断言。
- 丢弃 DOF（metadata 记录）：脊柱×3、双腕、双手、锁骨；root 仅保留 yaw（roll/pitch 丢弃——K1 无躯干关节且 PD 站立无法承受躯干倾角）。

## 4. 重定向质量（真实数据实测）

| 段 | 方法 | 中位方向误差 |
|---|---|---|
| 上臂 | 肩 2DOF IK | **2–6°**（精确） |
| 前臂 | 肘 2DOF IK | 39–70°（K1 无腕，尽力而为，如实标记） |
| 大腿 | 髋 3DOF IK | **0.3–1.1°**（精确） |
| 小腿 | 膝 1DOF IK | 28–31°（尽力而为） |
| 脚 | 踝 2DOF IK | 6–64°（尽力而为） |

## 5. 三种动作转换与验证结果

| | Arm Wave | Top Rock | Popping |
|---|---|---|---|
| 输入 | seg071 edited(0.5×幅度) | seg082-cropped edited(减速0.7×) | seg056-cropped edited |
| 模式 | upper_body（GVHMR 下肢噪声 49° 超 PD 站立能力，如实声明） | full_body | full_body |
| 输出 | 246帧@50Hz / 4.92s | 286帧@50Hz / 5.72s | 243帧@50Hz / 4.86s |
| 限位/越限 | 0 越限、0 饱和 | 0 越限 | 0 越限 |
| **A 运动学回放** | **PASS**（L/R 正确、贴地 0.023m） | **PASS**（L/R 正确、贴地 0.003m） | **PASS** |
| **B 物理回放(PD)** | **PASS**（未摔倒 0.547-0.551m，跟踪 rmse 0.023rad，接触吻合 L99%/R71%） | **FAIL（摔倒 0.060m）** | **FAIL（摔倒 0.060m）** |
| 数据集状态 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW |

**状态如实说明**：
- Arm Wave 的格式、数值、运动学、物理回放全部通过；`NEEDS_REVIEW` 仅因肩关节速度 15.3 rad/s 超过 8 rad/s 检查阈值——需要对照 K1 电机额定速度人工判定（未虚标为 PASS）。
- Top Rock / Popping 物理失败是**预期内的能力边界**：动态步伐/快速全身动作需要平衡控制器（本阶段明确不做，V1 PD 站位控制不足以支撑单支撑相）。**未修改任何物理参数来制造通过**。其轨迹数据本身（格式+限位+运动学）是有效数据，供后续带平衡控制器的训练使用。
- `CONVERTED` 未被自动升级：所有状态均由对应检查实际产生。

## 6. 已实现 vs 已运行 vs 已通过

| 项 | 实现 | 运行 | 结果 |
|---|---|---|---|
| 格式分析报告 | ✓ | ✓ | docs/K1_ACTION_FORMAT_ANALYSIS_V1.md |
| 映射表/坐标模块/重定向/限位/轨迹/导出/数据集/验证器 | ✓ | ✓ | k1_action/ 包 |
| 官方 CSV 双向兼容 | ✓ | ✓ | 列序测试+官方文件直读通过 |
| Arm Wave 端到端 | ✓ | ✓ | 运动学+物理双过（速度阈值待人工） |
| Top Rock/Popping 端到端 | ✓ | ✓ | 运动学过/物理如实 FAIL |
| 自动化测试 | ✓ | ✓ | **17/17 通过**（真实数据端到端，非随机数组） |
| V1 兼容性 | ✓ | ✓ | virtual_k1 只读复用，35 项 V1 测试不受影响 |

## 7. 遗留问题与需人工判断项

1. **肩关节速度**：三个动作的最大肩速 9.8–17.9 rad/s 需对照 K1 电机额定速度（Booster 规格书）判定是否需要时间缩放（Motion Studio 已有 set_speed 工具可直接生成慢速版）。
2. **前臂/脚方向误差**：K1 肘部实为"肘弯+腕转"复合（4DOF 中仅 1 个定位 DOF）、踝 2DOF，结构上无法完整表达人体对应段——误差是硬件边界而非软件缺陷，已按 best_effort 标记。
3. **GVHMR 下肢噪声**：单目恢复的下肢漂移（实测 49°）对 upper_body_mode 动作需抑制；若需忠实腿部数据，建议上游（Motion Studio smooth 工具）先做时序平滑再转换。
4. **Popping 帧间跳变 0.36 rad**：快速收缩动作固有，若用于实机需限速处理。
5. root 位置为重定向近似（yaw+平移+限幅起伏），**不是**动力学可行轨迹——训练时如需物理一致的 root，应由控制器产生。

## 8. 交付物索引

```
docs/K1_ACTION_FORMAT_ANALYSIS_V1.md      格式分析（输入/目标/差异/可转换性）
docs/K1_ACTION_DATA_CONVERSION_V1_REPORT.md 本报告
k1_action/                                转换工具包
  config/human_to_k1_joint_mapping.yaml   关节映射（22关节全覆盖+DOF缺失清单）
  skeleton_mapping.py  coordinate_transform.py  motion_loader.py
  retargeting.py  joint_limits.py  trajectory_processing.py
  action_exporter.py  dataset_builder.py  validators.py  virtual_k1_replay.py
  scripts/convert_motion_to_k1.py  scripts/validate_k1_action.py  scripts/inspect_k1_action.py
  tests/test_pipeline.py（17 项）  docs/README.md（使用说明）
datasets/k1_dance_v1/                     三动作数据集 + manifest.json
```
