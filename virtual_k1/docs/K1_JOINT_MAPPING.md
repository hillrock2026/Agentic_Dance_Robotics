# K1 关节映射表 (实际加载结果)

> 数据来源: Booster 官方仓库 [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets)
> commit `12a97516f57469078707afb587a5f84c1cb110d1` (main, 2026-09-26),
> 模型文件 `robots/K1/K1_22dof.xml`, 由 `scripts/inspect_model.py` 实际加载
> MuJoCo 4.0.1 后提取, 非人工誊写。

## 1. 维度统计

| 项 | 值 | 说明 |
|---|---|---|
| njnt | 23 | 1 free (root) + 22 hinge |
| nq | 29 | 22 hinge + 7 (free joint 位姿) |
| nv | 28 | 22 hinge + 6 (free joint 速度) |
| nu | 22 | 22 个 motor (力矩) 执行器 |
| nbody | 26 | world + 25 body |
| ngeom | 45 | 其中 21 个参与碰撞 |
| nmesh | 24 | 视觉 STL mesh |
| nkey | 0 | 官方模型无 keyframe 初始姿态 |

**结论**: K1 的 "22 自由度" 指 **22 个驱动旋转关节** (头 2 + 臂 4×2 + 腿 6×2)。
加上 root free joint 后广义坐标为 nq=29 / nv=28。22 个执行器与 22 个
驱动关节一一对应 (执行器名 == 关节名), 无被动关节、无固定关节冗余。

## 2. 全部关节映射表

| Index | Joint Name | Body | Type | qpos Address | qvel Address | Lower Limit | Upper Limit | Actuator |
|---|---|---|---|---|---|---|---|---|
| 0 | `world_joint` | trunk | free | 0 | 0 | — | — | - |
| 1 | `aahead_yaw_joint` | aahead_yaw_link | hinge | 7 | 6 | -1.012 | 1.012 | aahead_yaw_joint |
| 2 | `aahead_pitch_joint` | aahead_pitch_link | hinge | 8 | 7 | -0.314 | 0.794 | aahead_pitch_joint |
| 3 | `aaleft_shoulder_pitch_joint` | aaleft_shoulder_pitch_link | hinge | 9 | 8 | -2.932 | 1.196 | aaleft_shoulder_pitch_joint |
| 4 | `left_shoulder_roll_joint` | left_shoulder_roll_link | hinge | 10 | 9 | -1.642 | 1.629 | left_shoulder_roll_joint |
| 5 | `left_elbow_pitch_joint` | left_elbow_pitch_link | hinge | 11 | 10 | -1.885 | 1.885 | left_elbow_pitch_joint |
| 6 | `left_elbow_yaw_joint` | left_elbow_yaw_link | hinge | 12 | 11 | -2.242 | 0.816 | left_elbow_yaw_joint |
| 7 | `aaright_shoulder_pitch_joint` | aaright_shoulder_pitch_link | hinge | 13 | 12 | -2.932 | 1.196 | aaright_shoulder_pitch_joint |
| 8 | `right_shoulder_roll_joint` | right_shoulder_roll_link | hinge | 14 | 13 | -1.642 | 1.629 | right_shoulder_roll_joint |
| 9 | `right_elbow_pitch_joint` | right_elbow_pitch_link | hinge | 15 | 14 | -1.885 | 1.885 | right_elbow_pitch_joint |
| 10 | `right_elbow_yaw_joint` | right_elbow_yaw_link | hinge | 16 | 15 | -0.816 | 2.242 | right_elbow_yaw_joint |
| 11 | `left_hip_pitch_joint` | left_hip_pitch_link | hinge | 17 | 16 | -2.958 | 2.226 | left_hip_pitch_joint |
| 12 | `left_hip_roll_joint` | left_hip_roll_link | hinge | 18 | 17 | -0.375 | 1.536 | left_hip_roll_joint |
| 13 | `left_hip_yaw_joint` | left_hip_yaw_link | hinge | 19 | 18 | -1.012 | 1.012 | left_hip_yaw_joint |
| 14 | `left_knee_pitch_joint` | left_knee_pitch_link | hinge | 20 | 19 | 0.000 | 2.321 | left_knee_pitch_joint |
| 15 | `left_ankle_pitch_joint` | left_ankle_pitch_link | hinge | 21 | 20 | -0.870 | 0.345 | left_ankle_pitch_joint |
| 16 | `left_ankle_roll_joint` | left_ankle_roll_link | hinge | 22 | 21 | -0.345 | 0.345 | left_ankle_roll_joint |
| 17 | `right_hip_pitch_joint` | right_hip_pitch_link | hinge | 23 | 22 | -2.958 | 2.226 | right_hip_pitch_joint |
| 18 | `right_hip_roll_joint` | right_hip_roll_link | hinge | 24 | 23 | -1.536 | 0.375 | right_hip_roll_joint |
| 19 | `right_hip_yaw_joint` | right_hip_yaw_link | hinge | 25 | 24 | -1.012 | 1.012 | right_hip_yaw_joint |
| 20 | `right_knee_pitch_joint` | right_knee_pitch_link | hinge | 26 | 25 | 0.000 | 2.321 | right_knee_pitch_joint |
| 21 | `right_ankle_pitch_joint` | right_ankle_pitch_link | hinge | 27 | 26 | -0.870 | 0.345 | right_ankle_pitch_joint |
| 22 | `right_ankle_roll_joint` | right_ankle_roll_link | hinge | 28 | 27 | -0.345 | 0.345 | right_ankle_roll_joint |

free joint (`world_joint`) 占用 qpos[0:7] (位置 xyz + 四元数 wxyz), 
qvel[0:6] (世界系线速度 + 体坐标系角速度), 无执行器。

## 3. 执行器表

| Index | Actuator | Joint | 类型 | 语义 | Force Range (Nm) | Gear | Ctrl Range |
|---|---|---|---|---|---|---|---|
| 0 | `aahead_yaw_joint` | `aahead_yaw_joint` | motor | torque (ctrl=力矩) | [-6.0, 6.0] | 1 | 不限幅 |
| 1 | `aahead_pitch_joint` | `aahead_pitch_joint` | motor | torque (ctrl=力矩) | [-6.0, 6.0] | 1 | 不限幅 |
| 2 | `aaleft_shoulder_pitch_joint` | `aaleft_shoulder_pitch_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 3 | `left_shoulder_roll_joint` | `left_shoulder_roll_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 4 | `left_elbow_pitch_joint` | `left_elbow_pitch_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 5 | `left_elbow_yaw_joint` | `left_elbow_yaw_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 6 | `aaright_shoulder_pitch_joint` | `aaright_shoulder_pitch_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 7 | `right_shoulder_roll_joint` | `right_shoulder_roll_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 8 | `right_elbow_pitch_joint` | `right_elbow_pitch_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 9 | `right_elbow_yaw_joint` | `right_elbow_yaw_joint` | motor | torque (ctrl=力矩) | [-14.0, 14.0] | 1 | 不限幅 |
| 10 | `left_hip_pitch_joint` | `left_hip_pitch_joint` | motor | torque (ctrl=力矩) | [-68.0, 68.0] | 1 | 不限幅 |
| 11 | `left_hip_roll_joint` | `left_hip_roll_joint` | motor | torque (ctrl=力矩) | [-43.0, 43.0] | 1 | 不限幅 |
| 12 | `left_hip_yaw_joint` | `left_hip_yaw_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |
| 13 | `left_knee_pitch_joint` | `left_knee_pitch_joint` | motor | torque (ctrl=力矩) | [-112.0, 112.0] | 1 | 不限幅 |
| 14 | `left_ankle_pitch_joint` | `left_ankle_pitch_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |
| 15 | `left_ankle_roll_joint` | `left_ankle_roll_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |
| 16 | `right_hip_pitch_joint` | `right_hip_pitch_joint` | motor | torque (ctrl=力矩) | [-68.0, 68.0] | 1 | 不限幅 |
| 17 | `right_hip_roll_joint` | `right_hip_roll_joint` | motor | torque (ctrl=力矩) | [-43.0, 43.0] | 1 | 不限幅 |
| 18 | `right_hip_yaw_joint` | `right_hip_yaw_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |
| 19 | `right_knee_pitch_joint` | `right_knee_pitch_joint` | motor | torque (ctrl=力矩) | [-112.0, 112.0] | 1 | 不限幅 |
| 20 | `right_ankle_pitch_joint` | `right_ankle_pitch_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |
| 21 | `right_ankle_roll_joint` | `right_ankle_roll_joint` | motor | torque (ctrl=力矩) | [-38.3, 38.3] | 1 | 不限幅 |

**控制语义**: 全部 22 个执行器为 `<motor>` (direct force/torque), 
`data.ctrl[i]` 即关节力矩 (Nm), 受 `forcerange` 截断, `gear=1`。
官方模型不含位置/速度执行器 —— 位置控制需在上层实现
(本项目提供 `virtual_k1.controllers.JointPDController`)。

## 4. 关节分组

| 部位 | 数量 | 关节 |
|---|---|---|
| 头部 | 2 | `aahead_yaw_joint`, `aahead_pitch_joint` |
| 左臂 | 4 | `aaleft_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_elbow_pitch_joint`, `left_elbow_yaw_joint` |
| 右臂 | 4 | `aaright_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_elbow_pitch_joint`, `right_elbow_yaw_joint` |
| 左腿 | 6 | `left_hip_pitch_joint`, `left_hip_roll_joint`, `left_hip_yaw_joint`, `left_knee_pitch_joint`, `left_ankle_pitch_joint`, `left_ankle_roll_joint` |
| 右腿 | 6 | `right_hip_pitch_joint`, `right_hip_roll_joint`, `right_hip_yaw_joint`, `right_knee_pitch_joint`, `right_ankle_pitch_joint`, `right_ankle_roll_joint` |

## 5. 命名与关节语义注意事项

1. **XML `aa` 前缀怪癖**: 头部两关节 (`aahead_*`) 与两个肩 pitch 关节
   (`aaleft/aaright_shoulder_pitch_joint`) 带 `aa` 前缀, 其余关节不带。
   `virtual_k1` 的映射层同时接受两种命名 (`resolve_joint`)。
2. **官方动作 CSV 命名差异**: `booster_assets/motions/K1/*.csv` 表头对肩 pitch
   使用无前缀名 `left/right_shoulder_pitch_joint` (头仍为 `aahead_*`), 
   读取官方动作数据时需做名称映射。
3. **左右限位镜像**: hip_roll / elbow_yaw 左右限位范围互为镜像 
   (如左 hip_roll [-0.375, 1.536], 右 [-1.536, 0.375]), 属正常镜像结构。
4. **膝关节零位**: knee_pitch 限位 [0, 2.321], 零位为完全伸直且不允许过伸。
5. **手臂关节语义 (FK 实测, 重要!)**: 手臂零位为 **T 字平举** (沿 ±Y 轴),
   不是自然下垂。各关节实际语义:
   - `shoulder_roll`: 抬/放手臂。左臂 -1.45=下垂, 0=水平, +1.05≈高举过头;
     右臂镜像 (+1.45=下垂, 负=高举)。官方动作数据的下垂位即 roll=∓1.449。
   - `shoulder_pitch`: 悬垂/举起状态下前后摆臂 (矢状面摆动)。
   - `elbow_pitch`: **前臂自旋** (转轴永远平行于前臂长轴, 不产生可见屈肘)。
   - `elbow_yaw`: **实际可见的屈肘关节** (左负=前弯, 右正=前弯)。
   - 编排举手动作请用 `shoulder_roll` 抬臂 + `elbow_yaw` 摆动。
6. **平足约束**: 矢状面腿部三关节角之和 hip+knee+ankle ≈ 0 时脚掌贴地
   (站立姿态 -0.099+0.207-0.099≈0.009)。编排下蹲/弹跳时必须保持该约束,
   否则脚掌翘起导致失稳 (已在 demo 开发中实测验证)。
