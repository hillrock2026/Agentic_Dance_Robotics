# K1 Dynamics Model — Week 1 File Inventory (Repository Inspection Report)

- 检查日期: 2026-11-23
- 检查方式: 只读 Repository Inspection（目录遍历 + XML include 链检查 + 符号检索 + 文件阅读 + 一次性模型静态加载验证）
- 项目根目录: `/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics`（非 git 仓库）
- MuJoCo 版本: **4.0.1**（项目 `.venv` 与 conda env `booster_train` 均实测 4.0.1；与 `virtual_k1/docs/_model_inspection.json:3` 记录一致）

> 验证说明：本报告中标注 **[RUNTIME-VERIFIED]** 的事实来自本次检查中的一次性模型加载
> （`mujoco.MjModel.from_xml_path("virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml")`，
> 未执行任何轨迹仿真），其余事实均直接引自列出的源文件行号。无法确认的内容统一归入
> §H Unresolved Questions，未做任何虚构。

---

## 0. Week 1 六项核心结论速览

| # | 需确定的问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | 广义位置/速度/加速度维度 | **nq=29, nv=28, nu=22**；qpos = root(3+4 quat) + 22 hinge；qvel/qacc = root(6) + 22；qacc 维度 = nv = 28 | [RUNTIME-VERIFIED]；`virtual_k1/docs/_model_inspection.json:5-8` |
| 2 | 执行器输入 → 广义力映射 | 官方 XML 22 个 `motor`，joint transmission，gear=1，gainprm=[1,0,0]，biasprm=[0,0,0]，ctrl 不限幅但 forcerange 限幅 ⇒ `qfrc_actuator[dof_i] = clip(ctrl[i], forcerange)`，数值上 ctrl 即关节力矩 (Nm)。两条控制管线（virtual_k1 软件 PD、booster_deploy 软件 PD）都是先在软件里算 PD 力矩再写 `data.ctrl` | `K1_22dof.xml:188-211`；`virtual_k1/virtual_k1/simulator.py:5-14,180-199`；`tools/booster_deploy/.../mujoco_controller.py:221-246` |
| 3 | 刚体动力学方程所需物理量 | 26 body / 总质量 19.666 kg，每个 body 均有显式 `<inertial>`（pos/quat/mass/diaginertia），全部正定；关节带 armature（腿 0.028–0.096，臂/头 0.001–0.002，**armature 计入质量矩阵对角**）；主 XML 无 damping/spring/frictionloss ⇒ `qfrc_passive = 0`；重力 [0,0,-9.81] 含于 `qfrc_bias`。`qM` 目前**无任何代码读取**（需用 `mj_fullM` 展开） | `K1_22dof.xml` 全部 `<inertial>`/armature；`virtual_k1/docs/K1_MODEL_PHYSICS.md:1-49`；§E |
| 4 | 足底接触力与接触约束 | 每足 1 个不可见碰撞 box（半尺寸 0.09×0.035×0.018 m，踝前偏 0.026 m）；接触 condim=3（运行时取 max(地面 condim=1, 足盒 condim=3)）；**有效滑动摩擦 μ = max(0.4, 1.0) = 1.0**（不是 XML 注释里的 0.4），spin 0.005 / roll 0.0001；每足最多 4 角点接触（实测站立位 8 contacts/双足）；力经 `mj_contactForce` 在接触坐标系读出，法向已换算到世界系 | [RUNTIME-VERIFIED]；`K1_22dof.xml:43,143,179`；`virtual_k1/virtual_k1/simulator.py:275-320` |
| 5 | MLP/PINN 所需数据字段 | 50 Hz 参考轨迹（qpos 级）完备；sim2sim metrics.npz 已含 qpos/qvel/qfrc_actuator/足底法向力/接触 bool（50 Hz）；**缺失：qacc、qfrc_bias、qfrc_constraint、qM、ctrl（物理步级）、多接触点位置/法向、sensordata** —— 需新增导出通道（见 §F.3） | §E/§F；`tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py:207-216` |
| 6 | MuJoCo 能否提供可靠物理残差验证 | 能，但有三处必须先决策的口径问题：(a) 三套物理栈不一致（见 §H Q1）；(b) 现有 `pd_replay_reference.py` 的力矩注入方式**已被实测证伪**（`qfrc_actuator` 写入被 `mj_step` 覆盖）；(c) 残差验证所需字段（qM/qfrc_bias/qfrc_constraint/qacc）尚无导出代码 | §E/§H；实测：写入 `d.qfrc_actuator=1.234` 后 `mj_step` → 0.0 |

---

## A. Repository Overview

### A.1 目录结构与职责

```
Agentic_Dance_Robotics/                      # 项目根目录
├── virtual_k1/                              # [P0] MuJoCo 官方模型 + K1Sim 仿真封装（本项目自研）
│   ├── assets/booster_assets/               # Booster 官方资源 (github BoosterRobotics/booster_assets,
│   │   │                                    #   commit 12a9751, 2026-09-26, 见 k1_sim.yaml:8-12)
│   │   ├── robots/K1/K1_22dof.xml           # ★ 主 MJCF 模型（本报告 B 节）
│   │   ├── robots/K1/K1_22dof_parallel.xml  #   并联踝变体（含 equality 闭环约束，未被任何代码使用）
│   │   ├── robots/K1/K1_22dof.urdf          #   URDF（仅 IsaacLab 训练用）
│   │   └── motions/K1/                      #   官方 29 列 CSV + FK npz 动作（k1_top_rock.csv/.npz 等）
│   ├── config/k1_sim.yaml                   # ★ K1Sim 配置（模型路径/初始姿态/接触检测/PD 增益）
│   ├── virtual_k1/                          # K1Sim 包: simulator / controllers / joint_mapping /
│   │                                        #   foot_geometry / robot_state
│   ├── docs/                                # ★ K1_MODEL_PHYSICS.md / K1_JOINT_MAPPING.md /
│   │                                        #   _model_inspection.json（模型静态检查权威输出）
│   └── tests/                               # 模型加载/足接触/关节控制等 pytest
├── tools/
│   ├── booster_deploy/                      # [P0] MuJoCo sim2sim 部署/录制（policy 回放 + metrics.npz）
│   ├── booster_train/                       # IsaacLab + rsl_rl RL 训练（PhysX/URDF 栈, conda env）
│   ├── GVHMR/  IsaacLab/  blender.../       # 第三方 vendored 工具
├── k1_action/                               # [P1] SMPL→K1 重定向管线 + 轨迹回放验证
├── k1_skills/                               # 技能框架（skill.yaml 驱动; 调度 train/deploy/export）
├── skills/                                  # 技能定义 (arm_wave / mj_dance_002 / top_rock / _profiles)
├── datasets/k1_dance_v1/                    # [P1] 参考轨迹数据集 (manifest + 每技能 trajectory.csv)
├── motion_studio/                           # GVHMR 视频重建 + 动作编辑 (SMPL npz)
├── k1_runs/                                 # 运行产物: tasks/logs/outputs(metrics.npz,mp4)/policies
├── docs/                                    # 项目历史任务报告 (MOTION_RETARGETING / FOOT_GEOMETRY_FIX 等)
├── inputs/                                  # 源视频 (hiphop87_elements) + 开发需求 PDF
└── .venv/                                   # 项目 venv (mujoco 4.0.1, torch, ...)
```

### A.2 MuJoCo 模型文件分布（全仓 XML，已排除 Blender/IsaacLab/GVHMR vendored 目录）

| 文件 | 用途 | 是否被代码引用 |
|---|---|---|
| `virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml` | **主模型**，无 include（mesh 通过 `meshdir="meshes/"` 相对路径加载 24 个 STL，非 XML include） | 是（k1_sim.yaml:14、k1_action/retargeting.py:27、booster_deploy/robots/k1.py:65、pd_replay_reference.py:24、convert_k1_trajectory.py:23） |
| `.../K1/K1_22dof_parallel.xml` | 并联踝变体（4 个 `<connect>` equality 闭环 + 每踝 2 个驱动 hinge + 2 个 rod ball joint；踝 pitch/roll 改为 damping=0.2 无 armature） | **否**（全仓无引用） |
| `.../K1/K1_22dof.urdf` | IsaacLab 训练用（26 link，含 effort+velocity 限位；velocity 限位在 MJCF 中不存在） | 是（booster_train booster_k1.py:18） |
| `.../T1/*.xml, .../T2/*.xml` | 其他机器人，与本项目无关 | 否 |

include 链结论：**主 MJCF 无任何 `<include>`**，单文件自包含；mesh 为 STL 二进制资产。

---

## B. MJCF Model Inventory（主模型 K1_22dof.xml）

### B.1 全局定义

| 项 | 值 | 位置 |
|---|---|---|
| `<compiler>` | angle=radian, meshdir="meshes/", autolimits=true | `K1_22dof.xml:2` |
| `<option>` | timestep=**0.001 s**, tolerance=1e-6, impratio=10, solver=Newton | `K1_22dof.xml:4` |
| `<default>` | geom solimp="0.90 0.95 0.001" solref="0.001 1" | `K1_22dof.xml:5-7` |
| gravity | [0, 0, -9.81]（默认，未覆写） | `_model_inspection.json:17-21` |
| 模型统计 | **nq=29, nv=28, nu=22, njnt=23 (1 free + 22 hinge), nbody=26 (world+25), ngeom=45 (21 个参与碰撞), nmesh=24, nsensor=2, nkey=0, neq=0, ntendon=0, npair=0** | [RUNTIME-VERIFIED]; `_model_inspection.json:4-23` |
| 总质量 | **19.666 kg**（23 个有质量 body + 2 个零质量相机 link） | 同上 |

### B.2 Root / 浮动基座

- `<body name="trunk" pos="0 0 1.0">`（`K1_22dof.xml:45`），`<joint name="world_joint" type="free" limited="false">`（`:48`）——**root 为 free joint**，qpos[0:3]=位置(世界系, m)、qpos[3:7]=四元数 **wxyz**，qvel[0:6]=世界系 6D 速度（MuJoCo free joint 约定）。
- trunk `<inertial>`（`:47`）：mass 6.5 kg，ipos (-0.0043, -0.0007, 0.0657)，iquat+diaginertia 完整给出。
- IMU site `imu`（`:46`），传感器：`framequat`(orientation) + `gyro`(angular-velocity)，带 noise 参数（`:212-215`）；`sensordata` 目前无代码读取。

### B.3 关节清单（22 hinge，XML 定义顺序 = qpos[7:29] 顺序）

| qpos idx | dof idx | 关节 (XML 名) | 轴 | range (rad) | armature |
|---|---|---|---|---|---|
| 7/6 | 6 | aahead_yaw_joint | z | [-1.012, 1.012] | 0.002 |
| 8/7 | 7 | aahead_pitch_joint | y | [-0.314, 0.794] | 0.002 |
| 9/8 | 8 | aaleft_shoulder_pitch_joint | y | [-2.932, 1.196] | 0.001 |
| 10/9 | 9 | left_shoulder_roll_joint | x | [-1.642, 1.629] | 0.001 |
| 11/10 | 10 | left_elbow_pitch_joint | y | [-1.885, 1.885] | 0.001 |
| 12/11 | 11 | left_elbow_yaw_joint | z | [-2.242, 0.816] | 0.001 |
| 13/12 | 12 | aaright_shoulder_pitch_joint | y | [-2.932, 1.196] | 0.001 |
| 14/13 | 13 | right_shoulder_roll_joint | x | [-1.642, 1.629] | 0.001 |
| 15/14 | 14 | right_elbow_pitch_joint | y | [-1.885, 1.885] | 0.001 |
| 16/15 | 15 | right_elbow_yaw_joint | z | [-0.816, 2.242] | 0.001 |
| 17/16 | 16 | left_hip_pitch_joint | y | [-2.958, 2.226] | 0.0478125 |
| 18/17 | 17 | left_hip_roll_joint | x | [-0.375, 1.536] | 0.0339552 |
| 19/18 | 18 | left_hip_yaw_joint | z | [-1.012, 1.012] | 0.0282528 |
| 20/19 | 19 | left_knee_pitch_joint | y | [0, 2.321] | 0.095625 |
| 21/20 | 20 | left_ankle_pitch_joint | y | [-0.87, 0.345] | 0.0565 |
| 22/21 | 21 | left_ankle_roll_joint | x | [-0.345, 0.345] | 0.0565 |
| 23/22 | 22 | right_hip_pitch_joint | y | [-2.958, 2.226] | 0.0478125 |
| 24/23 | 23 | right_hip_roll_joint | x | [-1.536, 0.375] | 0.0339552 |
| 25/24 | 24 | right_hip_yaw_joint | z | [-1.012, 1.012] | 0.0282528 |
| 26/25 | 25 | right_knee_pitch_joint | y | [0, 2.321] | 0.095625 |
| 27/26 | 26 | right_ankle_pitch_joint | y | [-0.87, 0.345] | 0.0565 |
| 28/27 | 27 | right_ankle_roll_joint | x | [-0.345, 0.345] | 0.0565 |

（来源：`K1_22dof.xml:56-176` + `_model_inspection.json:24-551`，两者一致）

- **命名怪癖**：head 两个关节与左右肩 pitch 带 `aa` 前缀；官方动作 CSV 对肩 pitch 无前缀。别名解析在 `virtual_k1/virtual_k1/joint_mapping.py:188-212` 与 `k1_action/config/human_to_k1_joint_mapping.yaml`（`csv_to_xml_name`）。
- 主 XML 关节**无 damping / springdamper / frictionloss** ⇒ `qfrc_passive ≡ 0`；armature 进入质量矩阵对角（PINN 刚体模型必须包含，见 §H Q4）。
- 关节限位（jnt_limited）全部启用（autolimits + range 属性）——属于**非足底约束**，其约束力包含在 `qfrc_constraint` 中（目前无代码读取，见 §E）。
- 无 equality / tendon / 自定义 contact pair（[RUNTIME-VERIFIED] neq=0, ntendon=0, npair=0）；`K1_22dof_parallel.xml:233-238` 例外（未被使用）。

### B.4 刚体惯量

全部 23 个有质量 body 的 mass / ipos / iquat / diaginertia 已由 `virtual_k1/scripts/inspect_model.py` 提取至
`virtual_k1/docs/_model_inspection.json:1170-1924`（bodies 数组）并在
`virtual_k1/docs/K1_MODEL_PHYSICS.md:20-46` 汇总成表（惯性全部正定、满足三角不等式）。
质量分布：躯干 6.5 kg (33.1%)、单腿 4.503 kg×2、单臂 1.58 kg×2、头 1.0 kg。
本报告不重复该表——**动力学规范应以 `_model_inspection.json` 数字为准**（其注明"实际加载提取, 非人工誊写"）。

### B.5 并联踝变体（未使用，仅登记）

`K1_22dof_parallel.xml`：每踝增加 drive_a/drive_b 两个驱动 hinge（armature 0.0282528, damping 0.1）+ 两个 rod（ball joint, damping 0.01），通过 4 个 `<connect>` equality 闭环连接 ankle 上的 anchor site；踝 pitch/roll 本体关节改 damping=0.2（去掉 armature）。执行器数量仍 22（4 个踝电机换成 drive_a/b）。**该模型 nq/nv/nu 与主模型不同（额外 +4 hinge +2 ball×4=+8 qpos/+8 nv 量级）**，当前无任何代码引用，动力学建模 V1 不使用。

---

## C. Actuator Inventory

### C.1 官方模型执行器表（K1_22dof.xml:188-211，全部为 `<motor>`）

| Actuator (idx) | Type | Joint | Gear | Ctrl Range | Force Range (Nm) |
|---|---|---|---|---|---|
| aahead_yaw_joint (0) | motor (torque) | aahead_yaw_joint | 1.0 | 不限幅 (ctrllimited=false) | ±6.0 |
| aahead_pitch_joint (1) | motor | aahead_pitch_joint | 1.0 | 不限幅 | ±6.0 |
| aaleft_shoulder_pitch_joint (2) | motor | 同名 joint | 1.0 | 不限幅 | ±14.0 |
| left_shoulder_roll_joint (3) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| left_elbow_pitch_joint (4) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| left_elbow_yaw_joint (5) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| aaright_shoulder_pitch_joint (6) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| right_shoulder_roll_joint (7) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| right_elbow_pitch_joint (8) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| right_elbow_yaw_joint (9) | motor | 同名 | 1.0 | 不限幅 | ±14.0 |
| left_hip_pitch_joint (10) | motor | 同名 | 1.0 | 不限幅 | ±68.0 |
| left_hip_roll_joint (11) | motor | 同名 | 1.0 | 不限幅 | ±43.0 |
| left_hip_yaw_joint (12) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |
| left_knee_pitch_joint (13) | motor | 同名 | 1.0 | 不限幅 | ±112.0 |
| left_ankle_pitch_joint (14) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |
| left_ankle_roll_joint (15) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |
| right_hip_pitch_joint (16) | motor | 同名 | 1.0 | 不限幅 | ±68.0 |
| right_hip_roll_joint (17) | motor | 同名 | 1.0 | 不限幅 | ±43.0 |
| right_hip_yaw_joint (18) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |
| right_knee_pitch_joint (19) | motor | 同名 | 1.0 | 不限幅 | ±112.0 |
| right_ankle_pitch_joint (20) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |
| right_ankle_roll_joint (21) | motor | 同名 | 1.0 | 不限幅 | ±38.3 |

全部执行器 gaintype=0/biastype=0, gainprm=[1,0,0], biasprm=[0,0,0]（`_model_inspection.json:552-1168`）⇒
**`actuator_force = ctrl`（clip 到 forcerange）**，gear=1 joint transmission ⇒
**`qfrc_actuator[dof_i] = actuator_force[i]`**。四个概念的当前口径：

| 概念 | 当前项目中的事实 |
|---|---|
| Control command `ctrl` | 软件 PD 算出的力矩 (Nm) 直接写入 `data.ctrl`（两处：`simulator.py:199`、`mujoco_controller.py:239-243`）；K1Sim 在 ctrllimited 时才 clip（官方模型不触发） |
| Actuator force `actuator_force` | = clip 后的 ctrl；读取处 `simulator.py:245-248` |
| `qfrc_actuator` | gear=1 ⇒ 数值等于 actuator_force，映射到关节 dof；读取处 `simulator.py:237-243`、`mujoco_controller.py:164,205`、`deploy_mujoco_record_cyclic.py:275-276` |
| Joint torque（关节合力矩） | 本项目所有"joint_torques"字段实际都是 `qfrc_actuator`（不含 bias/constraint）；与广义合力 `qfrc_actuator + qfrc_constraint - qfrc_bias ≠ M·qacc` 意义上的"关节等效力矩"不同 |

### C.2 控制器实现（位置控制 = 软件 PD，不改模型）

1. **virtual_k1 `JointPDController`**（`virtual_k1/virtual_k1/controllers.py:16-71`）：
   `τ = kp(q_t − q) − kd·dq [+ qfrc_bias]`（重力+科氏补偿可开关，读 `data.qfrc_bias`，`:70`），输出 clip 到 actuator forcerange × gear（`:41-42,71`）。增益表在 `virtual_k1/config/k1_sim.yaml:81-105`（如 knee kp=180/kd=6）。
2. **booster_deploy `MujocoController.ctrl_step`**（`tools/booster_deploy/booster_deploy/controllers/mujoco_controller.py:221-246`）：
   `ctrl = clip(kp·(target − q) − kd·dq, ±effort_limit)`，每个 policy step 循环 **decimation 次 `mj_step`**（physics_dt=0.005×4 ⇒ 50 Hz 控制）。增益来源见 §C.3。
3. **booster_train（IsaacLab 训练）**（`tools/booster_train/.../assets/robots/booster_k1.py:14-105`）：
   `BoosterDelayedPDActuatorCfg`（**随机延迟 2–8 步**的 PD 执行器），Kp/Kd 由 Booster 关节模型（E6408/E4315/E4310/E6416 + 踝并联 wrapper）按 natural_freq/damping_ratio 和等效惯量解析计算 —— 这正是 skill.yaml/deploy 里那组 `3.9478…/30.2/21.4/…` 增益的来源。

### C.3 三套控制参数对照（对动力学数据采集至关重要）

| 管线 | 物理引擎/模型 | timestep | 控制频率 | Kp (腿 例) | effort limit (腿 例) |
|---|---|---|---|---|---|
| virtual_k1 K1Sim (V1 验证) | MuJoCo K1_22dof.xml | **0.001 s**（模型原值） | 1 kHz（n_substeps=1, `k1_sim.yaml:56-59`） | knee 180, hip_p 150（`k1_sim.yaml:94-105`） | 模型 forcerange（同 §C.1） |
| booster_deploy sim2sim | MuJoCo K1_22dof.xml（同文件，**运行时改 timestep=0.005**，`mujoco_controller.py:21`） | 0.005 s × decimation 4 | 50 Hz | knee 60.4, hip_p 30.2（BOOSTER_K1_CFG 换算，`tasks/beyond_mimic/robots/k1/__init__.py:118-142`；top_rock 任务继承于 `:232-248`） | hip_p 68, knee 112…（`:136-142`） |
| booster_train RL 训练 | **PhysX / K1_22dof.urdf** | IsaacLab env 步进 | 50 Hz | 同上（deploy 镜像训练值） | 同上 |

> ⚠️ 注意 booster_deploy `booster_deploy/robots/k1.py:29-44` 内还有一套**未被 top_rock 用**的默认增益（4/80/30, effort 30/35/20/40/20/20），仅 `K1BeyondMimicControllerCfg` 基类/旧任务使用；top_rock/arm_wave 走 `K1ArmWaveControllerCfg` 覆盖值。

---

## D. Contact Model Inventory

### D.1 碰撞几何

- **地面**：`<geom name="ground" type="plane" pos="0 0 0" size="0 0 1" condim="1" friction="0.4 0.005 0.0001">`（`K1_22dof.xml:43`）——无限大水平面 z=0。
- **足底（左/右各 1 box）**：body `left_ankle_roll_link` / `right_ankle_roll_link` 下不可见 box（rgba alpha=0 但 contype/conaffinity 默认 1）：
  `pos="0.026 0 -0.02" size="0.09 0.035 0.018"`（`K1_22dof.xml:143` 左、`:179` 右；geom id 33/44，`_model_inspection.json:2786-2811,3072-3097`）。足长≈0.18 m、宽≈0.07 m、厚≈0.036 m。足部 STL mesh 为 contype=0 纯视觉。
- **其他碰撞体**（自碰/摔倒时会与地面接触，动力学上属非足底接触）：躯干 box+cylinder、头 mesh×2、肩 roll cylinder×2、肘下 mesh×4、髋 roll/yaw cylinder、膝 cylinder+box（左右各一套），共 21 个碰撞 geom（`K1_MODEL_PHYSICS.md:51-83` 表）。大腿（hip_yaw→knee 段）与上臂无碰撞体（该文档 §3.4 已记录此限制）。
- **无 `<contact>`/`<exclude>`/pair 配置**：父子 geom 由 MuJoCo 默认排除，其余身体 geom 之间允许自碰（与 IsaacLab 训练侧 `enabled_self_collisions=True` 一致，`booster_k1.py:30`）。

### D.2 接触参数（[RUNTIME-VERIFIED]，站立位下压实测 8 contacts）

| 量 | 值 | 说明 |
|---|---|---|
| contact condim | **3**（1 法向 + 2 切向） | 取 max(ground condim=1, foot condim=3)；**有摩擦接触** |
| 有效摩擦 friction | **[1.0, 1.0, 0.005, 0.0001, 0.0001]**（μ_slide=1.0, μ_spin=0.005, μ_roll=0.0001） | MuJoCo 逐元素取两 geom 摩擦的 max：foot 默认 1.0 **覆盖** ground 的 0.4。`K1_MODEL_PHYSICS.md:83`"摩擦系数 0.4"的说法对足-地接触**不成立** |
| solref / solimp | [0.001, 1] / [0.9, 0.95, 0.001, 0.5, 2.0] | `<default>`（`K1_22dof.xml:5-7`），两 geom 相同无混合歧义 |
| solver | Newton, tolerance 1e-6, **impratio=10**（摩擦约束刚度放大） | `K1_22dof.xml:4` |
| 每足接触点 | box-plane 生成**最多 4 个角点接触** | 实测每足 4 个，位置即穿透角点 |
| 接触穿透 | 允许软穿透（solimp width 0.001 m 量级）；站立配置刻意留 0.5 mm 间隙（`k1_sim.yaml:24-27`） | |

### D.3 接触判定与力读取（代码位置）

| 问题 | 答案 | 代码 |
|---|---|---|
| 如何判断左右脚是否接触 | 遍历 `data.ncon`/`data.contact[i]`，geom 属于 `left/right_ankle_roll_link` 且对端 geom ∈ {ground}；`in_contact = 任一接触点法向力 ≥ min_normal_force (0.5 N, 可配)` | `virtual_k1/virtual_k1/simulator.py:275-320`（配置 `k1_sim.yaml:61-70`）；deploy 侧 `deploy_mujoco_record_cyclic.py:165-182`（阈值不同：n>0 即接触） |
| 接触点位置 | `con.pos`（**世界系**，m） | `simulator.py:313` |
| 接触力 | `mj_contactForce(model, data, i, wrench)` —— **接触坐标系下 6D wrench**（[法向, tan1, tan2, spin矩, roll矩1, roll矩2]）；项目当前只用 wrench[0] 法向力 | `simulator.py:303-304`；`deploy_mujoco_record_cyclic.py:177-178` |
| 力的坐标系 | wrench 在**接触坐标系**；法向方向 `con.frame[0:3]`（世界系），geom1 为足时取反使法向 地面→足(+z) | `simulator.py:305-309`（含方向注释） |
| 多接触点处理 | 每足累加 `total_normal_force`，保留逐点 normal_forces/positions/normals 列表 | `simulator.py:310-319` |
| 广义接触约束力 | `qfrc_constraint` / `efc_force` / `cfrc_ext` **全仓无读取代码**（见 §E） | — |
| 非足底约束 | 关节限位（22 关节全部 limited）约束力含于 `qfrc_constraint`；无 equality/tendon | §B.3 |
| 几何接触 vs 物理接触 | 独立双通道：`get_foot_contacts()`（物理力）与 `get_foot_geometry()`（8 角点几何间隙/穿透，PENETRATION/CONTACT/AIRBORNE，容差 2 mm）**明令禁止混用** | `simulator.py:322-337`；`virtual_k1/virtual_k1/foot_geometry.py:1-44`（docstring 详述换链：local→collision frame→foot link→world） |

---

## E. MuJoCo Dynamics Interface（已使用的动力学字段及位置）

全仓检索范围：`virtual_k1/ k1_action/ k1_skills/ tools/booster_deploy/ motion_studio/ tests_k1_skills/`（排除 vendored、__pycache__）。

### E.1 字段使用矩阵

| 字段 | 是否使用 | 位置与用途 |
|---|---|---|
| `qpos` | ✅ 广泛 | 写入：`simulator.py:145-151`(reset)、`:213`(set_joint_positions)、`mujoco_controller.py:26-32`；读取：`simulator.py:262-263`、`mujoco_controller.py:162,166-167`、recorder `deploy_mujoco_record_cyclic.py:229,233-234`、`virtual_k1_replay.py:55-56` |
| `qvel` | ✅ 广泛 | 同上（`:234`、`mujoco_controller.py:163,168-169`、recorder `:230`）；`get_base_state` 用 `mj_objectVelocity` 换世界系 6D 速度（`simulator.py:264-273`） |
| `qacc` | ⚠️ 仅有限性检查 | `simulator.py:377`、`virtual_k1/tests/test_sustained_run.py:44` —— **无导出、无学习用途** |
| `ctrl` | ✅ 写入 | `simulator.py:199`（含 ctrllimited clip 逻辑 `:194-198`）、`mujoco_controller.py:239-243`（clip 到 effort_limit）—— **recorder 未记录 ctrl 本身**（只记 dof_target 策略输出） |
| `actuator_force` | ✅ 读取 | `simulator.py:245-248`（`get_actuator_forces`）；测试 `test_state_and_reset.py:66-70` |
| `qfrc_actuator` | ✅ 读取（3 处） | `simulator.py:242`（get_joint_torques）、`mujoco_controller.py:164,205`（feedback_torque/log_states）、recorder `:275-276`（metrics.npz 的 joint_torque 字段） |
| `qfrc_bias` | ✅ 仅重力补偿 | `controllers.py:70`（PD 叠加 qfrc_bias） |
| `qfrc_passive` | ❌ 未使用 | （主 XML 下恒为 0，见 §B.3） |
| `qfrc_applied` | ❌ 未使用 | （`pd_replay_reference.py:98` 误用 `qfrc_actuator` 直写，见 §H Q7） |
| `qfrc_constraint` | ❌ 未使用 | **PINN 残差验证的关键缺口** |
| `qM` / `mj_fullM` | ❌ 未使用 | 质量矩阵无任何读取；MuJoCo 中 M 以自定义稀疏格式存于 `MjData.qM`，需 `mj_fullM(m, M_dense, d.qM)` 展开（28×28） |
| `efc_force` / `cfrc_ext` | ❌ 未使用 | — |
| `sensordata` | ❌ 未使用 | IMU framequat/gyro 定义于 XML 但无读取代码 |
| `contact`/`ncon`/`mj_contactForce` | ✅ | 见 §D.3 |
| `mj_step` | ✅ | `simulator.py:176`（1 ms×n_substeps）、`mujoco_controller.py:244`（5 ms×decimation）、`pd_replay_reference.py:99`、`virtual_k1_replay.py:152` |
| `mj_step1`/`mj_step2` | ❌ 未使用 | — |
| `mj_forward` | ✅ | reset/FK/ghost：`simulator.py:152,216`、`mujoco_controller.py:33,44,128`、recorder `:150`、`virtual_k1_replay.py:56`、`convert_k1_trajectory.py`（FK 生成 npz） |

### E.2 隐含的动力学方程口径（供 V1 规范引用）

MuJoCo 每步满足：`M(q)·qacc + qfrc_bias(q,qvel) = qfrc_passive + qfrc_actuator + qfrc_applied + qfrc_constraint`。
本项目主模型下：`qfrc_passive = 0`（无 damping/spring/frictionloss）、`qfrc_applied = 0`（无外力接口）、
armature ∈ M 对角、重力+科氏 ∈ qfrc_bias。因此
**`M(q)·qacc + qfrc_bias = qfrc_actuator + qfrc_constraint`** 就是 V1 残差验证的目标方程；
其中 M 需经 `mj_fullM(d.qM)` 取得，`qfrc_constraint` 由接触（足底 4×2 角点 + 关节限位）产生。

### E.3 仿真程序入口与版本

| 入口 | 作用 | timestep | 位置 |
|---|---|---|---|
| `K1Sim`（virtual_k1） | 官方模型封装：reset/step/状态/接触/渲染 | 0.001（模型原值不改，`simulator.py:371-374`） | `virtual_k1/virtual_k1/simulator.py` |
| `virtual_k1_replay.py` | 参考轨迹 A 运动学回放 / B PD 物理回放（1 kHz 步进、~4 ms 采样记录） | 0.001 | `k1_action/virtual_k1_replay.py:31-215` |
| `deploy_mujoco_record_cyclic.py` | 策略 sim2sim 录制（`MujocoController` + ghost 渲染 + metrics.npz） | 0.005（运行时覆写 `mujoco_controller.py:21`） | `tools/booster_deploy/scripts/` |
| `deploy.py` / `play_reference_motion.py` / `pd_replay_reference.py` | 交互查看 / 参考播放 / PD 回放（后者有缺陷 §H Q7） | 0.005 | 同上 |
| `scripts/inspect_model.py` | 模型静态检查（生成 `_model_inspection.json`） | — | `virtual_k1/scripts/inspect_model.py` |
| booster_train（IsaacLab） | RL 训练（PhysX 栈，与 MuJoCo 数据无关但产策略） | env 级 | `tools/booster_train/.../beyond_mimic/robots/k1/top_rock/env_cfg.py`（复用 mj_dance_002 配置，仅换 motion npz） |

解释器分布（`k1_skills/paths.py:33-69`）：训练用 conda `booster_train`（IsaacLab），sim2sim 优先 conda、回退项目 `.venv`；两者 mujoco 均 4.0.1 [RUNTIME-VERIFIED]。

---

## F. Data Pipeline Inventory

### F.1 Reference Motion（参考轨迹，非物理）

**数据链**：`inputs/hiphop87_elements/segment_*.mp4`（87 段街舞视频）
→ GVHMR 重建 + motion_studio 编辑（`motion_studio/outputs/motions/segment_082_cropped_*/edited_motion.npz`：SMPL 24 关节 world 坐标 (T,24,3)、global_orient、transl、body_pose、contact_state 估计、25 fps；加载器 `k1_action/motion_loader.py:25-74`）
→ **k1_action 重定向**（`k1_action/retargeting.py`：per-chain direction IK，MuJoCo Jacobian + 阻尼最小二乘；限位来自 XML `joint_limits.py`；root yaw-only、xy 取人体骨盆、z=站高+缩放弹跳）
→ `datasets/k1_dance_v1/<skill>/trajectory.csv`。

**trajectory.csv 格式（top_rock 实测，286 行 + 表头）**：
`frame_index, timestamp_s, root_pos_x/y/z, root_quat_x/y/z/w（注意 xyzw 顺序）, <22 关节列（官方 CSV 命名，肩 pitch 无 aa 前缀）>`，50 Hz，单位 rad/m（`k1_action/skeleton_mapping.py:36-40` 列定义；`datasets/k1_dance_v1/manifest.json` 数据集级约定 + sha256）。
配套 `metadata.json`（重定向配置/质量/来源 sha）、`validation.json`、`trajectory_original.csv`（top_rock 修复前原件，fix_history 见 `metadata.json:71-85`）。
**注意**：这是**纯运动学参考**，无速度/加速度/力；速度在转 npz 时由 FK+差分产生。

**转换链（CSV → 训练/部署 npz）**：`tools/booster_deploy/scripts/convert_k1_trajectory.py`（MuJoCo FK 生成 body 参考；hinge 地址按名字解析——曾因按序号写错 18/22 个地址修复过，见 `metadata.json:76`）输出：
- 官方 29 列 CSV：`virtual_k1/assets/booster_assets/motions/K1/k1_top_rock.csv`
- train/deploy npz（两者字段一致，实测）：`fps, joint_pos (286,22), joint_vel (286,22), body_pos_w (286,23,3), body_quat_w (286,23,4), body_lin_vel_w, body_ang_vel_w, joint_names(22, sim 顺序), body_names(23)`
  - train: `virtual_k1/assets/booster_assets/motions/K1/k1_top_rock.npz`（IsaacLab 训练输入）
  - deploy: `tools/booster_deploy/tasks/beyond_mimic/robots/k1/motions/k1_top_rock.npz`（sim2sim 参考与 ghost）
- npz 读取器：`tools/booster_deploy/booster_deploy/utils/motion_loader.py:9-125`（`time_step_total`、按名字索引 body/joint、首帧对齐）
- **关节顺序陷阱**：npz 内 joint_pos 为 **sim 顺序**（`K1_CFG.sim_joint_names`，`booster_deploy/robots/k1.py:45-54`），与 CSV/qpos 顺序（`joint_names`，`:7-17`）不同，需 `sim2real` 索引映射（recorder `:134-136`；`k1_skills/robot.py:11-22` 注明 CSV 分组顺序）。

**技能定义**：`skills/top_rock/skill.yaml`（motion 资源四角色 dataset_trajectory/booster_csv/train_npz/deploy_npz `:26-32`；control_profile `:40-66`；训练任务 Booster-K1-Top_Rock-v0 `:76-83`；资源集加载/校验框架 `k1_skills/motion.py:25-41`）。root_mode: free，oneshot 286 帧。

### F.2 Physics Rollout（物理回放，已有产物）

| 管线 | 采样 | 记录字段 | 产物 |
|---|---|---|---|
| `k1_action/virtual_k1_replay.py::physical_replay`（PD+重力补偿, 1 kHz） | ~4 ms | base_z、跟踪误差、左右接触 bool；（可选 trace CSV：time、接触 bool、左右法向力、几何间隙、base_z `:203-214`） | `datasets/.../replay/physical_replay.gif`、`k1_runs/outputs/...`；top_rock 走过 fix01 的 pd_replay（但该脚本有缺陷，见 Q7；virtual_k1_replay 未受影响） |
| `deploy_mujoco_record_cyclic.py`（策略 sim2sim, 5 ms×4） | 50 Hz（policy step 后采样） | metrics.npz 实测字段：`t, abs_frame, frame, cycle, phase, in_transition, ref_joint/act_joint (N,22), ref/act_joint_vel, ref/act_root_xyz, root_err_xyz, ref/act_root_quat(wxyz), trunk_ori_err_deg, base_roll/pitch/yaw_deg, base_height, left/right_contact(bool), left/right_force_n, left/right_slide, fallen, dof_target(策略输出), joint_torque(=qfrc_actuator[6:])` + 元数据 `joint_names_csv/sim, sim2real, period, n_source_frames, fps, cycles, stopped_early`（`:207-216,335-347`） | `k1_runs/outputs/{arm_wave,mj_dance_002}/…metrics.npz`（**top_rock 尚无**——无已训练策略）；`k1_runs/tasks|logs` 留痕 |
| `mujoco_controller.log_states` | 100 步快照 | root/joint pos/vel、joint_torque、dof_targets → npz（`:186-219`，默认未启用） | — |

### F.3 缺口（Week 1 规范必须补齐的数据通道）

现有导出**不含**：`qacc`、`qfrc_bias`、`qfrc_constraint`、`qM`、物理步级 `ctrl`（有 50 Hz dof_target + 已知 PD 增益可重构 ctrl，但含 clip/重算细节）、逐接触点位置/法向/切向力（recorder 只存足底合计法向力）、`sensordata`、1 kHz 物理步级全量状态。**归一化/预处理/训练数据读取代码：不存在**（尚无动力学训练数据集）。参考轨迹 50 Hz vs 物理 1 kHz 的对齐策略未定义（Q6）。

---

## G. Important File Index

| 编号 | 文件路径（仓库相对） | 关键行 | 用途摘要 | 重要程度 |
|---|---|---|---|---|
| 1 | `virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml` | 2, 4-7, 43, 45-48, 143, 179, 188-211, 212-215 | 主 MJCF：option/地面/自由关节/足碰撞盒/22 motor 执行器/IMU 传感器 | ★★★ P0 |
| 2 | `virtual_k1/docs/_model_inspection.json` | 4-23 (stats), 24-551 (joints), 552-1168 (actuators), 1170-1924 (bodies), 1927-3098 (geoms) | 模型静态检查权威数据（nq/nv/nu、惯量、geom、传感器）——动力学参数的事实来源 | ★★★ P0 |
| 3 | `virtual_k1/virtual_k1/simulator.py` | 5-14, 157-178, 180-199, 237-248, 275-320, 322-337, 339-352 | K1Sim：ctrl 语义、step、qfrc_actuator/actuator_force 读取、足接触力读取、状态规范 | ★★★ P0 |
| 4 | `virtual_k1/virtual_k1/joint_mapping.py` | 104-129, 131-186, 188-212 | 从实际模型提取 joint/actuator 映射与控制语义判定、CSV↔XML 别名 | ★★★ P0 |
| 5 | `virtual_k1/config/k1_sim.yaml` | 8-15, 28-53, 56-59, 61-76, 81-105 | 模型来源 commit、初始姿态、n_substeps、接触检测配置、V1 PD 增益 | ★★★ P0 |
| 6 | `virtual_k1/docs/K1_MODEL_PHYSICS.md` | 8-49, 51-97 | 质量分布表、碰撞模型表、检查结论（注意"摩擦 0.4"对足-地接触不成立，见 §D.2） | ★★☆ P0 |
| 7 | `virtual_k1/virtual_k1/controllers.py` | 16-71 | 软件 PD（qfrc_bias 重力补偿、forcerange 截断） | ★★☆ P0 |
| 8 | `virtual_k1/virtual_k1/foot_geometry.py` | 1-44 (docstring) | 足底 8 角点几何检测（与物理接触独立）；足盒尺寸/偏移的事实依据 | ★★☆ P0 |
| 9 | `tools/booster_deploy/booster_deploy/controllers/mujoco_controller.py` | 19-33, 161-184, 186-219, 221-246 | sim2sim 控制器：timestep 覆写、状态读取（qfrc_actuator）、ctrl 写入与 PD/decimation 循环 | ★★★ P0 |
| 10 | `tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py` | 152-182, 207-216, 225-276, 335-347 | Physics rollout 录制器：接触判定、全部 metrics 字段、npz 导出 | ★★★ P1 |
| 11 | `tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py` | 101-155, 232-248 | top_rock/arm_wave sim2sim 任务的增益、effort、decimation=4×5 ms | ★★☆ P0 |
| 12 | `tools/booster_deploy/booster_deploy/robots/k1.py` | 4-80 | K1_CFG（joint/body 名序、mjcf 路径、默认增益——注意非 top_rock 所用） | ★★☆ P0 |
| 13 | `tools/booster_train/.../assets/robots/booster_k1.py` | 14-105 | IsaacLab 训练执行器（URDF、延迟 PD、Booster 关节模型——增益解析来源） | ★★☆ P1 |
| 14 | `k1_action/virtual_k1_replay.py` | 31-92, 95-215 | 参考轨迹 A/B 回放验证（K1Sim 物理回放 + trace CSV） | ★★☆ P1 |
| 15 | `k1_action/retargeting.py` | 25-52 (K1Trajectory), 头注 | SMPL→K1 重定向方法与数据结构（root quat xyzw yaw-only） | ★★☆ P1 |
| 16 | `k1_action/skeleton_mapping.py` | 26-40 | CSV 列序/命名映射的单一事实来源 | ★★☆ P1 |
| 17 | `datasets/k1_dance_v1/manifest.json` + `top_rock/{metadata.json,trajectory.csv,trajectory_original.csv,validation.json}` | manifest 全文; metadata 37-50, 71-85 | 参考数据集约定（joint_order、rad、50 Hz、sha256、修复历史） | ★★★ P1 |
| 18 | `skills/top_rock/skill.yaml` | 22-32, 40-66, 76-99 | top_rock 技能定义：资源角色、控制 profile、训练任务、评估指标 | ★★☆ P1 |
| 19 | `virtual_k1/assets/booster_assets/motions/K1/k1_top_rock.{csv,npz}` 与 `tools/booster_deploy/.../motions/k1_top_rock.npz` | — | 参考动作的官方 CSV/训练 npz/部署 npz（字段见 §F.1） | ★★★ P1 |
| 20 | `k1_skills/paths.py` + `k1_skills/motion.py` + `k1_skills/robot.py` | paths:15-34; motion:25-41; robot:11-22 | 技能框架：双解释器、资源集校验、CSV 顺序约定 | ★☆☆ P1 |
| 21 | `tools/booster_deploy/scripts/pd_replay_reference.py` | 27-34, 92-99, 112-121 | ⚠️ 有缺陷的 PD 回放器（力矩写入无效，§H Q7）；作为反例登记 | ★☆☆ P1 |
| 22 | `k1_runs/outputs/{arm_wave,mj_dance_002}/*.metrics.npz`、`k1_runs/outputs/top_rock/{kinematic_checks.json,fix01/*}` | — | 现有 physics rollout 产物与 top_rock 运动学检查（top_rock 无 metrics.npz） | ★☆☆ P1 |
| 23 | `virtual_k1/virtual_k1/robot_state.py` | 1-34 | V1 状态字典与单位约定 | ★☆☆ P1 |
| 24 | `tools/booster_deploy/scripts/convert_k1_trajectory.py` | 1-16, 37-80+ | CSV→官方 29 列 CSV+FK npz 转换（按名解析 qpos 地址） | ★☆☆ P1 |
| 25 | `virtual_k1/assets/booster_assets/robots/K1/K1_22dof_parallel.xml` | 233-238 (equality) 等 | 并联踝变体（未被使用，登记备查） | ☆☆☆ 参考 |
| 26 | `docs/TOP_ROCK_REFERENCE_FEASIBILITY_FIX_01_REPORT.md` 等 `docs/*.md` | — | 各历史任务的执行/评审报告（含 top_rock 足底穿透修复） | ☆☆☆ 参考 |

---

## H. Unresolved Questions（现有文件无法确认、需 Week 1 决策）

1. **物理栈口径**（最重要）：项目存在三套不一致的物理环境——virtual_k1（MuJoCo 1 kHz + V1 PD 增益 kp=150~180）、booster_deploy（同一 MJCF 但 timestep 覆写为 5 ms + BOOSTER_K1_CFG PD kp=30~60）、booster_train（PhysX + URDF + 延迟 2–8 步 PD）。**V1 动力学模型以哪一套为 ground truth？** 同一参考轨迹在三套环境下的 qfrc_* 数据不可互换。
2. **有效摩擦系数意图**：XML 地面写 0.4 但足-地接触实测 μ_slide=1.0（geom 摩擦取 max 规则，[RUNTIME-VERIFIED]）。若研究意图是"地面 μ=0.4"，需要给足盒显式 friction 或确认接受 1.0。PINN 接触约束中的 μ 取值必须先定。
3. **qM / qfrc_bias / qfrc_constraint / qacc 导出**：全仓无任何代码读取（§E.1）。物理残差验证数据通道需要新写（建议挂在 K1Sim 或 recorder 上，`mj_fullM` 展开 28×28 M）。目前无法仅凭现有文件给出"MLP/PINN 训练数据集"的具体 schema——这属于 Week 1 规范设计，不是文件盘点能确认的。
4. **armature 的模型归属**：主 XML 关节 armature（腿 0.028–0.096）会进 MuJoCo 质量矩阵对角。PINN 的解析刚体模型（若用 RNEA/CRBA 自算）必须加同值 armature，否则残差有系统性偏差；URDF 侧 IsaacLab 又另有执行器模型换算——两边口径需统一核对。
5. **URDF 与 MJCF 一致性未验证**：`K1_22dof.urdf` 仅被训练栈使用，本次未逐项对比其惯量/碰撞/限位与 MJCF 是否一致（URDF 另有关节速度限位如 knee 7.85/33.51 rad/s，MJCF 无此限位）。
6. **采样与对齐策略**：参考 50 Hz vs MuJoCo 1 kHz（virtual_k1）/200 Hz 物理步（deploy）。qacc 在 policy step 边界与 substep 内含义不同；学习用数据取哪个时刻、是否需要 1 kHz 全量记录，未在任何文档定义。
7. **`pd_replay_reference.py` 力矩注入无效（已证伪）**：`d.qfrc_actuator[...] = tau` 后调用 `mj_step` 会被 `mj_forward` 内部的 `mj_fwdActuation` 重算覆盖——实测写入 1.234 后归零（ctrl=0）[RUNTIME-VERIFIED]。因此 `k1_runs/outputs/top_rock/fix01/pd_replay_after.json` 的"269/286 帧跌倒"**反映的是零力矩自由落体，而非参考轨迹物理不可行**。该脚本结论在动力学规范中不应引用；正确做法是写 `qfrc_applied` 或经 `data.ctrl`（`virtual_k1_replay.py` 即正确路径）。
8. **接触力完整信息未采集**：现只取每足法向力合计；切向摩擦力、接触点位置/法向（recorder 未存）、6D wrench、非足底接触（躯干/膝/肘触地，top_rock 这类全身动作可能出现）均无记录。残差验证是否需要这些，待规范决定。
9. **传感器通道闲置**：IMU framequat/gyro（带噪声模型）定义于 XML 但无人读取；若 V1 需要模拟真机观测，需补读取代码。
10. **`K1_22dof_parallel.xml` 的定位**：更接近真机踝并联结构（等效力矩经连杆分配），但未使用；V1 用串联简化模型的误差影响未评估。
11. **top_rock 无 policy rollout 数据**：RL 训练（Booster-K1-Top_Rock-v0）尚未执行（skill.yaml:98-99 "baseline: none yet"），故 top_rock 目前只有运动学验证数据；arm_wave/mj_dance_002 的 metrics.npz 是仅有的 physics rollout 样本。

---

*报告结束。本报告只登记事实与缺口，不包含任何模型设计、代码修改或数据生成。*
