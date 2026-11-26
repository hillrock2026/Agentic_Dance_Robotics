# VIRTUAL_K1_V1 交付报告

**项目**: Booster K1 Agentic Dance Robotics — Virtual-First 阶段 V1
**日期**: 2026-11-12
**执行范围**: 虚拟环境建立与控制接口实现 (未连接任何实体机器人)

---

## 0. 验收结论总览

| 验收项 | 状态 | 依据 |
|---|---|---|
| Test 1 模型加载 | **PASS** | `tests/test_model_loading.py` 5 项通过; XML/Mesh 解析、MjModel/MjData/mj_forward 全部正常 |
| Test 2 关节结构 | **PASS** | 22 DoF 说法**得到验证**: 22 驱动 hinge + 1 root free; 35 项测试通过 |
| Test 3 物理属性 | **PASS** | 质量/惯性/碰撞检查全部通过, 详见 [K1_MODEL_PHYSICS.md](K1_MODEL_PHYSICS.md) |
| Test 4 足部接触 | **PASS** | 双足接触可读, 法向力合计与重力吻合 (192.9N), 随姿态变化更新 |
| Test 5 关节控制 | **PASS** | 目标 Δ+0.5 rad → 实际 Δ+0.5000 rad, 闭环验证通过 |
| Test 6 状态读取与重置 | **PASS** | 全状态接口验证, reset 误差 0.0 (逐位一致) |
| Test 7 持续运行 10s | **PASS** | 无界面 10.0s 仿真, 0 NaN/Inf, 全程双足接触, 实时率 11.7x |
| MuJoCo Viewer 显示 | **PASS** | 离屏渲染 PNG 已验证 (见 docs/k1_render_standing.png); GUI viewer 脚本就绪 |

**核心结论: Virtual K1 基础环境已建立 —— 机器人可以被 Python 真实力矩控制、
按关节名读取状态、检测足地接触、可重复重置, 10 秒量级仿真稳定。**

---

## 1. 实际完成的工作

1. 检查并配置开发环境 (独立 venv, 未触碰系统/conda base 环境)。
2. 克隆 Booster 官方 `booster_assets` 仓库并确认 K1 资源 (模型/mesh/动作数据)。
3. 加载官方 `K1_22dof.xml`, 验证初始化、渲染、几何无异常。
4. 提取完整关节映射 (23 关节 + 22 执行器) 并生成文档。
5. 完成质量/惯性/碰撞模型检查并生成文档。
6. 完成足地接触检查 (初始接触、静态站立、姿态变化、自由落体四种场景)。
7. 实现 `virtual_k1` Python 包: K1Sim 仿真器 + 关节映射 + PD 控制器 + 状态接口。
8. 建立初始站立姿态 (来源: 官方动作数据首帧 + 运动学计算, 已记录)。
9. 编写 35 项自动化测试 + 3 个演示脚本, 全部实际执行并通过。

## 2. 环境与依赖版本 (实际运行版本)

| 组件 | 版本 | 说明 |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS (kernel 7.0.0-31-generic) | |
| GPU | NVIDIA RTX 3090 (driver 595.84) | EGL 离屏渲染 |
| Python | 3.14.6 | 项目独立 venv: `<项目>/Agentic_Dance_Robotics/.venv` |
| mujoco | **4.0.1** (mujoco-core 4.0.1) | pip 安装, 官方 Python bindings |
| numpy | 2.5.4 | |
| pyyaml | 6.0.3 | |
| pytest | 9.2.0 | |
| 渲染 | EGL (headless) + GLFW/X11 (GUI) | `MUJOCO_GL=egl` 用于离屏 |

未修改 conda base / 系统 Python; 官方模型文件零改动 (git 工作区干净)。

## 3. 官方资源来源记录

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/BoosterRobotics/booster_assets.git |
| 分支 / 深度 | main / shallow (depth=1) |
| **commit SHA** | **12a97516f57469078707afb587a5f84c1cb110d1** |
| commit 日期 | 2026-09-26 |
| 使用模型 | `robots/K1/K1_22dof.xml` (K1 专属; 仓库另含 T1/T2) |
| Mesh | `robots/K1/meshes/` 52 个 STL, 24 个被模型引用, 全部解析成功 |
| 附带资源 | `motions/K1/k1_fight_001.csv`, `k1_mj2_seg1.csv` (50Hz 官方重定向动作, 含 MJ 舞蹈片段) |
| 备选模型 | `K1_22dof_parallel.xml` (31 关节并联踝, 22 motor) — V1 未采用, 已记录 |

资源核对: 模型自述 `model="K1"`, 关节拓扑 (头 2 + 臂 4×2 + 腿 6×2) 与 K1 官方
README 表格一致, 确认为 K1 而非 T1/T2。

## 4. 关键模型参数 (实际加载)

| 参数 | 值 |
|---|---|
| 驱动自由度 | **22** (hinge) — 官方 "22 DoF" 说法验证成立 |
| Root | free joint `world_joint` → nq=29, nv=28 |
| 执行器 | nu=22, **全部 torque/motor** (gear=1, ctrl 不限幅) |
| 力矩限幅 | 头 ±6 / 臂 ±14 / hip_pitch ±68 / hip_roll ±43 / hip_yaw·ankle ±38.3 / knee ±112 Nm |
| 总质量 | 19.666 kg (躯干 6.5, 单腿 4.503×2, 单臂 1.58×2, 头 1.0) |
| 物理设置 | timestep 0.001s, Newton solver, impratio 10, 腿关节有 armature |
| 碰撞 | 21/45 geom 参与碰撞; 足底为不可见 box (18×7×3.6 cm) |
| 传感器 | imu site: framequat + gyro (官方带噪声模型) |
| 官方初始姿态 | **无 keyframe**; qpos0 为 root z=1.0m 悬空态 (足底离地 0.448m) |

关节映射全表: [K1_JOINT_MAPPING.md](K1_JOINT_MAPPING.md)
物理属性全表: [K1_MODEL_PHYSICS.md](K1_MODEL_PHYSICS.md)

**模型检查中发现并记录的官方资源特性** (非错误, 但必须知晓):
- XML 命名 `aa` 前缀怪癖 (`aahead_*`, `aaleft/aaright_shoulder_pitch_joint`)。
- 官方动作 CSV 与 XML 关节命名不一致 (肩 pitch 无前缀) — 映射层已兼容。
- qpos0 悬空 0.45m, 官方未提供站立姿态 — 本项目依据官方动作数据构建 (见 §6)。
- 地面 plane condim=1 与足部 geom condim=3 组合后接触为 condim=3, 摩擦合成正常。

## 5. 控制接口设计

包结构 (`virtual_k1/virtual_k1/`):

| 模块 | 内容 |
|---|---|
| `simulator.K1Sim` | 仿真器: load/reset/step/状态读取/接触读取/离屏渲染 |
| `joint_mapping.JointMapping` | 从实际模型提取关节↔qpos/qvel↔执行器映射 + 别名解析 |
| `controllers.JointPDController` | 软件 PD 位置控制器 (+可选重力补偿), 输出力矩 |
| `robot_state` | 统一状态字典结构与单位约定 |

关键设计决策 (均以实际模型为依据):

1. **力矩控制语义**: 官方 22 个执行器经参数检查 (gainprm/biasprm) 确认全部为
   torque/motor, 因此 `K1Sim.step(control)` 的 `control` 是**力矩 (Nm)**,
   支持按关节名 dict 或 (nu,) 数组。未修改官方 actuator 类型。
2. **位置控制**: 官方无位置执行器 → 项目内实现 `JointPDController`
   (kp/kd 可配, 叠加 qfrc_bias 重力补偿, 输出被官方 forcerange 截断)。
   PD 参数在 `config/k1_sim.yaml`, 属本项目软件层, 非模型修改。
3. **`set_joint_positions` 为运动学写入** (qpos + mj_forward), 用于姿态设置/重置,
   与力矩控制通道明确区分, 文档已注明。
4. **无界面运行**: 默认纯物理, 不创建任何窗口; 需要图像时用 `render()`
   (EGL 离屏), GUI 调试用 `scripts/view_k1.py --interactive`。
5. **状态单位** (MuJoCo 仿真数据, 非实体传感器): 角度 rad, 角速度 rad/s,
   力矩 Nm, 位置 m, 线速度 m/s; base 线/角速度统一换算到**世界系**
   (mj_objectVelocity, flg_local=0)。

## 6. 初始站立姿态 (来源已记录)

| 项 | 值 | 来源 |
|---|---|---|
| 腿部 12 关节 | 轻微屈膝 (hip≈-0.10, knee≈0.207, ankle≈-0.10 rad) | 官方 `k1_fight_001.csv` 第 1 帧 |
| 臂/头 10 关节 | 全零 (自然下垂, 在限位内) | 模型 qpos0 |
| root 位置 | (0, 0, 0.5510) m | 运动学计算: 足底最低点 0.4495@trunk1.0 → 1.0-0.4495+0.5mm 间隙 |
| root 姿态 | 单位四元数 | — |

交叉验证: 官方动作首帧 root z=0.5550 m, 与计算值 0.5505 差 4.5mm (动作数据的
动态浮起), 互相印证。配置位于 `config/k1_sim.yaml` (provenance 注释完整)。

## 7. 测试命令与实际结果

```bash
cd virtual_k1/
PY=../.venv/bin/python

# 全部 35 项自动化测试
$PY -m pytest tests/ -v            # → 35 passed in 3.80s

# 模型检查 (生成 docs/_model_inspection.json)
$PY scripts/inspect_model.py

# 关节控制闭环演示
$PY scripts/test_joint_control.py

# 重置可重复性演示
$PY scripts/test_reset.py

# 离屏渲染 (无 GUI 环境用 MUJOCO_GL=egl)
MUJOCO_GL=egl $PY scripts/view_k1.py --pose standing

# 文档再生成 (从实际加载 JSON)
$PY scripts/generate_docs.py
```

### 关键测试输出 (实际运行摘录)

**关节控制闭环 (Test 5)** — `scripts/test_joint_control.py`:
```
[1] 初始站立 (PD 保持 2.0 s):
  root_z=0.5483m 左脚接触=True 右脚接触=True 法向力 L=96.1N R=96.8N
[2] 目标变化: 左肩 pitch 0 → +0.5 rad, 头 yaw 0 → +0.5 rad:
  左肩pitch=+0.5000 rad (Δ=+0.5000, 目标Δ=+0.5)   ← 闭环证据
  头yaw=+0.5000 rad (Δ=+0.5000, 目标Δ=+0.5)
[3] 重置: 最大关节复位误差=0.00e+00 rad (PASS)
```
双足法向力 96.1 + 96.8 = 192.9 N = 19.666 kg × 9.81 —— 与总重力精确吻合。

**足部接触 (Test 4)** — `tests/test_foot_contact.py -s`:
```
静态站立: 双脚 in_contact=True, 法向 ≈[0,0,+1]
下蹲后:   right 脚: 接触点 4→2, 法向力 96.9→95.3 N   ← 接触随姿态更新
自由落体: qpos0 释放 1.5s 后足地接触建立
```

**重置 (Test 6)** — `scripts/test_reset.py`:
```
扰动后: root=[0.0663 -0.3294 0.0882]   (机器人被随机力矩踢倒)
第1次 reset: qpos误差=0.00e+00 qvel误差=0.00e+00 time误差=0.00e+00 → PASS
第2次 reset: (同上) → PASS               ← 确定性重置
```

**10 秒持续运行 (Test 7)** — `tests/test_sustained_run.py -s`:
```
10s 仿真完成: 墙钟 0.9s, 实时率 11.7x
  t=  0.5s root_z=0.5492 双足接触=True
  ... (每 0.5s 采样共 19 个点) ...
  t=  9.5s root_z=0.5487 双足接触=True
qpos/qvel/qacc/ctrl 全程无 NaN/Inf; root 高度波动 < 2mm
```

**渲染验证** — `docs/k1_render_standing.png` (960×720):
经视觉核验: 完整人形 (头/双臂/双腿/双足), 双足触地, 轻微屈膝站姿, 无部件缺失、
无穿地、无姿态异常。

## 8. 尚未解决的问题

1. **官方站立姿态非正式发布物**: booster_assets 无 keyframe; 本项目姿态来自
   官方动作数据首帧 + 运动学推算, 已记录 provenance, 但若 Booster 后续发布
   官方 standing pose 应以官方为准。
2. **PD 增益为工程初值**: kp/kd 由本项目按关节力矩限幅量级选取并实测稳定,
   非官方整定值; 后续动作开发时需再整定。
3. **碰撞几何简化**: 大腿段与上臂段无碰撞体 (官方建模如此)。V1 站立/基础动作
   不受影响; 街舞动作若涉及大腿/上臂触地或肢体互碰, 需要补碰撞 (后续决策)。
4. **悬空 qpos0**: 官方默认初始状态为 0.45m 悬空。这是官方设定而非模型缺陷;
   本项目以 reset 到站立姿态规避, 未修改官方 XML。
5. `K1_22dof_parallel.xml` (并联踝, 31 关节含被动) 尚未深入分析, V1 采用标准版。

## 9. 下一阶段建议

1. **官方动作数据回放**: `motions/K1/k1_mj2_seg1.csv` (50Hz, MJ 舞蹈片段) 与
   本 K1Sim 关节映射/别名层直接对接, 验证轨迹回放 —— 是通向舞蹈生成的最短路径。
2. 引入整定后的全身 PD / 任务空间控制器, 支撑单腿平衡等舞蹈基础动作。
3. 添加批量 rollout 接口 (并行多实例) 为强化学习做准备。
4. 评估 parallel 踝模型与标准模型的差异, 决定 Sim-to-Real 阶段用哪个。
5. 将 `get_robot_state()` 作为唯一状态源接入后续音乐-动作同步模块。

## 10. 目录结构 (交付物)

```
virtual_k1/
├── assets/booster_assets/          # 官方资源 (独立 git clone, 未修改)
├── config/k1_sim.yaml              # 仿真配置 (含姿态 provenance 与 PD 增益)
├── virtual_k1/                     # Python 包 (本项目代码)
│   ├── __init__.py / simulator.py / joint_mapping.py
│   ├── robot_state.py / controllers.py
├── scripts/
│   ├── inspect_model.py            # 模型全面检查 → JSON
│   ├── view_k1.py                  # 离屏渲染 / GUI viewer
│   ├── test_joint_control.py       # 闭环控制演示
│   ├── test_reset.py               # 重置演示
│   └── generate_docs.py            # 文档生成 (从实际加载 JSON)
├── tests/                          # 35 项 pytest
│   ├── test_model_loading.py / test_joint_mapping.py / test_model_physics.py
│   ├── test_joint_control.py / test_foot_contact.py
│   └── test_state_and_reset.py / test_sustained_run.py
└── docs/
    ├── K1_JOINT_MAPPING.md         # 关节映射 (实际加载)
    ├── K1_MODEL_PHYSICS.md         # 物理/碰撞检查 (实际加载)
    ├── k1_render_standing.png      # 站立渲染证据
    ├── k1_render_qpos0.png         # 官方默认姿态渲染
    └── VIRTUAL_K1_V1_REPORT.md     # 本报告
```
