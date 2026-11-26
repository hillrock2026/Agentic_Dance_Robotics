# K1_ACTION_FORMAT_ANALYSIS_V1 — 输入/目标动作格式分析报告

日期：2026-11-16 ｜ 所有结论均来自实际文件检查（非文档推断）

---

## 1. 输入端：Motion Studio 实际输出

文件：`motion_studio/outputs/motions/<job>/clean_motion.npz`（及编辑版 `edited_motion.npz`）

实测字段（以 Arm Wave `segment_071_343ee79d` 为例，T=123 帧）：

| 字段 | 形状 | 语义 |
|---|---|---|
| `motion_fps` / `source_fps` | 标量 | 25.0（帧与源视频 1:1） |
| `frame_count` / `duration` / `timestamps` | 标量 / (T,) | 123 / 4.92 s / 均匀时间轴 |
| `global_orient` | (T,3) | SMPL 根朝向，**轴角**，MS_WORLD |
| `body_pose` | (T,63) | 21 个身体关节局部旋转（轴角），顺序=SMPL 24 关节去掉 root 与双手 |
| `betas` | (T,10) | SMPL 形状参数 |
| `transl` | (T,3) | 根（骨盆）世界平移，米 |
| `joint_positions` | (T,24,3) | **世界系 24 关节位置**（米）——重定向的主要输入 |
| `contact_left/right`, `contact_state` | (T,) bool / int8 | 脚部接触**估计**（0腾空/1左/2右/3双支撑） |
| `joint_names` / `parents` | (24,) | SMPL 关节名与父索引 |
| `coordinate_json` | str | MS_WORLD：**+Y up、右手系、米、T-pose 面向 +Z** |
| `metadata_json` | str | 来源 sha256、GVHMR commit、编辑日志等完整溯源 |

身体模型：**SMPL（非 SMPL-X）**，24 关节（pelvis, left_hip … right_hand），旋转为轴角。

## 2. 目标端：K1 官方动作 CSV 实测

文件（`virtual_k1/assets/booster_assets/motions/K1/`，来自 Booster 官方仓库 booster_assets）：

- `k1_fight_001.csv`：706 数据行
- `k1_mj2_seg1.csv`：1261 数据行

**两文件格式完全一致**：29 列，无表头以外元数据，无时间戳列。

### 2.1 列顺序（实测）

```
1-3   root_pos_x, root_pos_y, root_pos_z        # 世界位置（米）
4-7   root_quat_x, root_quat_y, root_quat_z, root_quat_w   # xyzw 四元数
8-9   aahead_yaw_joint, aahead_pitch_joint                  # 头 2
10-13 left_shoulder_pitch, left_shoulder_roll,
      left_elbow_pitch, left_elbow_yaw                      # 左臂 4
14-17 right_shoulder_pitch, right_shoulder_roll,
      right_elbow_pitch, right_elbow_yaw                    # 右臂 4
18-23 left_hip_pitch, left_hip_roll, left_hip_yaw,
      left_knee_pitch, left_ankle_pitch, left_ankle_roll    # 左腿 6
24-29 right_hip_pitch, right_hip_roll, right_hip_yaw,
      right_knee_pitch, right_ankle_pitch, right_ankle_roll # 右腿 6
```

### 2.2 语义（实测确认）

- **角度单位：弧度，绝对关节角**（非相对初始姿态偏移）。证据：`k1_fight_001.csv` 首行 22 个关节值与 `K1_22dof.xml` 加载后的 V1 站立位形逐位一致（如 `left_shoulder_roll = -1.448789…`）——首帧即官方站立姿态，说明是绝对角。
- **root**：`root_pos` 为世界系位置（首帧 z=0.555 = K1 站立高度；root_z 变化范围 0.473–0.633 表示含蹲起动作）；`root_quat` 为 xyzw 四元数（首帧含 ≈112° yaw，机器人初始斜向站立）。**root 是动作数据的一部分**。
- **帧率：文件中未标注**。两文件均无时间戳/帧号列。相邻帧关节变化（fight 首两帧 Δ≈0.0016 rad）与常见 50 Hz 采样一致但无法证实。⚠️ 官方 CSV 是**动作序列样例**，不构成带元数据的训练数据集封装格式。
- **关节命名差异**：CSV 列名与 XML 驱动关节名**不完全一致**——CSV 用 `left_shoulder_pitch_joint`，XML 用 `aaleft_shoulder_pitch_joint`（左/右肢 CSV 去掉了 `aa` 前缀；头部两者均带 `aa`）。转换时必须做名称规范化（见 `k1_action/skeleton_mapping.py`）。

### 2.3 XML 关节顺序（K1_22dof.xml 实测，V1 验收文档一致）

22 驱动关节顺序 = 头(2) → 左臂(4) → 右臂(4) → 左腿(6) → 右腿(6)，与 CSV 列顺序**逐列核对一致**（经名称规范化后）。关节限位（rad）见 `k1_action/joint_limits.py`（从 XML 运行时读取，非硬编码）。

## 3. 两种格式的差异与转换判定

| 维度 | Motion Studio（输入） | K1 CSV（目标） | 处理 |
|---|---|---|---|
| 模型 | SMPL 24 关节人体 | K1 22 驱动关节（+root free joint） | **重定向**（非重排） |
| 旋转表示 | 轴角（局部/全局） | 标量关节角（弧度，绝对） | 分段方向 + 数值 IK |
| 时间轴 | 25 fps + timestamps | 无时间元数据 | 重采样至 50 fps + 内部元数据 |
| root | transl + global_orient | root_pos + root_quat(xyzw) | 坐标系转换 + 高度/朝向映射 |
| 接触信息 | 有（估计） | 无 | 存入 metadata（供训练用） |
| 单位 | 米/弧度 | 米/弧度 | 一致 |

### 3.1 自由度缺失清单（无法直接转换，必须近似）

| 人体部位 | 人体 DOF | K1 DOF | 处理策略 |
|---|---|---|---|
| 脊柱（3 关节） | 3×3 旋转 | **0**（无躯干关节） | 躯干弯曲并入 root orientation 近似（只取 yaw，roll/pitch 丢弃——K1 腿部/平衡无法承受躯干倾角） |
| 颈 | 1×3 | 2（yaw,pitch） | 颈→头方向向量取 yaw/pitch，roll 丢弃 |
| 肩 | 3 旋转 | 2（pitch,roll） | 上臂方向 IK，不可达时最近方向 |
| 肘+腕 | 肘1+腕2 | 2（pitch,yaw） | 前臂方向尽力而为（K1 肘 yaw 实为腕转） |
| 手/手指 | 15×3 | 0 | 丢弃 |
| 髋 | 3 旋转 | 3（pitch,roll,yaw） | 大腿方向 IK（可精确） |
| 膝 | 1 | 1（pitch） | 小腿方向（可精确） |
| 踝 | 2 旋转 | 2（pitch,roll） | 脚掌方向尽力而为 |
| 比例 | 人臂展≈1.7 m | K1 臂展≈0.4 m | 方向重定向天然比例无关 |

### 3.2 结论：格式分层

1. **K1 关节动作序列格式**（本任务兼容目标）：官方 29 列 CSV。内部标准在此基础上**前置 `frame_index, timestamp_s` 两列**（官方读取器按列名解析不受影响），形成 31 列 `trajectory.csv`。
2. **K1 动作训练数据集封装格式**（内部新建）：`action/trajectory.csv + metadata.json + validation.json`，元数据补齐官方缺失的 fps/单位/溯源/验证状态。
