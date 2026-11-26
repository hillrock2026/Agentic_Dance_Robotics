# K1_DANCE_SKILL_003_MOTION_RETARGETING_01_GPT_REVIEW_REPORT

日期：2026-11-20

## 总体结论

**PARTIAL PASS**

Top Rock 的源视频、GVHMR/Motion Studio 编辑、286 帧时间轴、训练 NPZ 的 22-DOF 映射与 Root 数据均有可独立追溯的证据；Root 路径、转体、关节限位、速度、跳变和节奏相关性的数值基本可复算。

但本任务仍存在两个直接阻塞训练验收的问题：

1. **足底参考严重穿地**。报告中的 10.5/3.3 mm 来自旋转不敏感、以第 0 帧固定 ankle-to-sole 偏移的错误检查；逐帧旋转 box 角点计算得到左/右最大穿地约 **58.03/58.39 mm**，分别发生在 **254/257 个非负地面以下帧**。现有物理 kinematic/PD 回放记录也显示机器人会跌倒。
2. **部署 NPZ 的 body FK 数组是错误的**。`joint_pos` 的名称与列映射正确，但 [convert_k1_trajectory.py](tools/booster_deploy/scripts/convert_k1_trajectory.py#L78) 将 sim-order joint array 直接写入 MuJoCo XML qpos 顺序。两者并非同一顺序，导致部署 NPZ 的非 trunk body position/orientation 与正确 FK 相差最多约 **0.391 m / 120.7°**。现有 trunk-only 自检无法检测该问题。

因此，虽然训练命令、训练 NPZ 和任务注册文件已存在，**当前参考轨迹不应被认定为已经具备直接启动 RL Training 的基本质量条件**。

**结论：`top_rock` 当前不应启动 RL Training。**

| 评审项 | 结论 | 证据摘要 |
|---|---|---|
| A. 动作资源和数据链 | **PASS WITH QUALIFICATIONS** | 原视频—cropped—edited—trajectory 链及 SHA256 可追溯；trim/slowdown/帧数一致；许可文本缺失 |
| B. K1 Retargeting | **PARTIAL** | 训练 NPZ 的 22 列/Root 映射正确；部署 NPZ body FK 因 sim/XML 顺序混用错误 |
| C. 运动学质量 | **PARTIAL** | 关节限位、速度、跳变、Root、rhythm 数值可复算；足底与回放物理状态被错误低估 |
| D. 足底行为 | **FAIL** | 真实 box 角点显示约 58 mm、近乎全段穿地；原 10.5 mm 结论不成立 |
| E. 参考回放 | **PASS WITH QUALIFICATIONS** | 286 帧/50 Hz 视频可读且 24 帧抽检连续、无明显身体破碎；它不验证 body FK，并低估足底穿地 |
| F. Skill 接入 | **PARTIAL** | registry/validate/测试/训练命令通过，但 deploy 运动资源 body references 错误，现有验证未捕获 |

## A. 动作资源和数据链

### 已验证

- `inputs/hiphop87_elements/manifest.csv` 第 82 条记录 `segment_082.mp4`，区间 411.44–416.36 秒，长度 4.92 秒。
- 原始 `segment_082.mp4` 为 123 帧、25 fps、4.92 秒；视频中可见完整的 B-boy footwork/step-dance 片段。
- Motion Studio 的 cropped 视频 SHA256 为 `72c3b3ef…1a6d72`，与 [clean_motion.json](motion_studio/outputs/motions/segment_082_cropped_72c3b3ef/clean_motion.json) 和 [edited_motion.json](motion_studio/outputs/motions/segment_082_cropped_72c3b3ef/edited_motion.json) 记录一致。
- 编辑日志实际包含 `trim 10–110` 与 `set_speed 0.7`。编辑后为 143 帧、25 fps、5.72 秒；时间戳从 0 到 5.68 秒，符合 143 个采样点的离散时间表示。
- edited motion NPZ SHA256 为 `7cec0f61…e3b6`，与 [top_rock metadata](datasets/k1_dance_v1/top_rock/metadata.json) 的 `source_sha256` 精确一致。
- 最终 trajectory 有 286 行、31 列，时间戳 0–5.70 秒，步长 0.02 秒；按 `286/50=5.72s` 表示持续时间一致。
- [trajectory.csv](datasets/k1_dance_v1/top_rock/trajectory.csv) 与 29 列 booster CSV 保留数据最大差约 `5e-9`。

### 证据限制

- `inputs/hiphop87_elements` 目录未发现授权、版权或许可文本。可确认资源存在和数据关系，不可据此推定再分发或商业使用许可。
- SMPL 为单目重建，绝对尺度/深度不确定是固有限制；本任务报告已披露。
- 源视频画面自身使用的标签包含多个街舞元素名称；本次证据支持其为全身 step/Top Rock 类素材，但不构成逐帧语义标注证明。

## B. K1 Retargeting

### 正确部分

- 训练 NPZ 的 `joint_names` 为 canonical K1 22-DOF sim 顺序。
- CSV 官方列映射到 train NPZ 时，最大数值差约 `1.43e-6 rad`，属于 float32 精度；未发现左右交换。
- train NPZ 的 trunk position 与 trajectory root position 最大差约 `1.46e-7 m`，根姿态 quaternion dot 接近 1，free-root 未被固定或重置。
- 使用正确 joint-name→MuJoCo qpos 地址映射复算 train NPZ 的各 body FK，最大位置误差约 `4.46e-7 m`；训练 NPZ body references 是可信的。

### 关键缺陷

[convert_k1_trajectory.py](tools/booster_deploy/scripts/convert_k1_trajectory.py#L78) 中：

```python
npz_joint_names = ...   # canonical sim order
joints_sim = joints_csv[:, csv2sim]
d.qpos[7:] = joints_sim
```

但 MuJoCo XML 的 hinge qpos 顺序并不是 K1 canonical sim order。独立得到 sim→XML qpos 地址：

```text
[7, 9, 13, 17, 23, 8, 10, 14, 18, 24, 11, 15, 19, 25, 12, 16, 20, 26, 21, 27, 22, 28]
```

部署 NPZ 的 joint array 本身顺序正确，而生成其中 `body_pos_w/body_quat_w` 时却未经该映射直接写入 `d.qpos[7:]`。独立复算后，部署 NPZ 存储值与正确 FK 的差异为：

- trunk：0（因此报告的 trunk 自检“零偏差”成立，但无诊断能力）；
- left elbow link：最大约 67.1 mm；
- left knee link：最大约 82.1 mm；
- left ankle roll link：最大约 141.5 mm；
- right ankle roll link：最大约 391.1 mm；
- 最大 body orientation error：约 120.7°。

这不是轻微数值误差，而是非根 body 参考的实质错配。`joint_pos` 和回放 renderer 按 joint name 映射，所以播放视频看似连贯，并不能证明 NPZ 的 body references 正确。

现有 oneshot 资源验证没有比较 deploy NPZ 的 body FK 与 train NPZ，因此 `SkillRegistry.validate("top_rock")` 仍返回全绿。

## C. 运动学质量

### 可复算且一致的指标

使用部署 NPZ 的 joint trajectory、K1 XML 范围和 `dt=0.02s` 独立复算：

| 指标 | 独立结果 | 报告结果 |
|---|---:|---:|
| 关节限位违例 | 0 | 0 |
| 最小裕量 | 0.0207000089 rad | 0.0207 rad |
| 最大关节速度 | 3.24300006 rad/s | 3.24 rad/s |
| 最大加速度 | 81.0750015 rad/s² | 81.1 rad/s² |
| 最大帧间跳变 | 0.06486000 rad | 0.0649 rad |
| >0.2 rad 跳变 | 0 | 0 |
| Root xy 路径 | 1.2607691 m | 1.261 m |
| Root 净位移 | 0.2163111 m | 0.216 m |
| Root z 范围 | 0.536156–0.597759 m | 0.536–0.598 m |
| yaw 净转 | -86.6128° | -86.6° |
| yaw 跨度 | 100.1717° | 100.2° |
| yaw 峰值速率 | 101.4902°/s | 101.5°/s |
| 节奏包络相关 | 0.74808934 | 0.748 |

这些数字在数值上可信。Root 平移/转体与 sampled 视频中的前后移位、侧移和转身阶段相容。

### 解释限制

- `0.748` 是 SMPL body pose gradient 能量包络与 K1 joint velocity 范数的时序相关，不是姿态保真度或“动作像 Top Rock”的直接证明。
- trajectory 的 `retarget_quality` 中 forearm、shank、foot 中位误差分别约 53.0°、27.5°、42.5°。报告正文只强调 upper-arm 2.1° 和 thigh 0.3°，没有同时呈现这些较大的链误差，容易高估全身保真度。
- [validation.json](datasets/k1_dance_v1/top_rock/validation.json) 与 dataset manifest 仍标记 `status=NEEDS_REVIEW`；其物理回放 `pass=false`、`fell=true`。这些不是“全项干净”的全部证据。

## D. 足底行为

报告所称 10.5/3.3 mm 来自 `check_reference_quality.py`：在第 0 帧计算一次 `ankle_z - sole_z` 偏移，然后对所有帧直接从 ankle z 减去该固定偏移。此方法没有考虑足部 box 随 ankle/root 姿态旋转时的最低角点变化。

GPT-6 使用 K1 XML 中实际 box half-size、`geom_xpos`、`geom_xmat`，逐帧计算每只脚 8 个世界角点，结果为：

| Foot | 最低 sole z | 最大穿地深度 | 负高度帧数 |
|---|---:|---:|---:|
| left | -58.032 mm | 58.032 mm | 254 / 286 |
| right | -58.391 mm | 58.391 mm | 257 / 286 |

该结果与仓库已有独立足几何产物 [top_rock_foot_audit.json](virtual_k1/outputs/foot_geometry_fix01/top_rock_foot_audit.json) 一致。

支撑相滑动 0.20–0.25 m/s 同样没有稳固的物理定义，因为固定-offset 方法把大量穿地帧都判为“contact”。使用旋转 box 的 `<0.02m` 接触代理独立得到约 0.30 m/s 的均值量级；两种结果的差异说明该指标强依赖不可靠的接触分类，不应作为精确的滑动证明。

“与 42.5° 足链误差具有相关性”目前只是合理原因解释，缺少逐帧、逐阶段的统计相关分析，不能提升为已证实因果。

结论：原报告低估了约 5.5 倍的最大穿地。穿地覆盖约 89–90% 的动作帧，并且现有物理 replay 已记录跌倒；这不是“轻微、可忽略的训练前瑕疵”，而是参考轨迹物理可行性的直接阻塞证据。

## E. 参考回放

已独立读取 `k1_runs/outputs/top_rock/reference_playback.mp4`：

- 286 帧、50 fps、5.72 秒；
- 抽取 9 帧和 24 帧代表性 contact sheet，覆盖完整时间轴；
- 没有看到模型结构缺失、肢体断裂、左右肢体整体交换或突然 teleport；
- Root 移动与转体连续，步法交替和手臂摆动在若干阶段可辨；
- 视频 renderer 使用 joint-name 映射，因此 joint pose 本身没有出现问题。

局限：

- 视频不是 body FK consistency 测试；错误的 body arrays 可能不显现。
- MuJoCo 地面渲染与相机角度无法可靠暴露几十毫米穿地，视觉抽样不能否定逐帧 box 角点结果。
- 足部/肘部姿态相较源舞者仍明显受 K1 IK 能力限制，retarget quality 的大 forearm/foot 误差与此相符。

因此回放支持“连续且未破碎”，但不足以支持“物理可行或全身忠实”。

## F. Skill 接入

### 正确且可复现

- `skills/top_rock/skill.yaml` 为 v2、oneshot、free-root，训练 NPZ 路径正确，source_frames=286、fps=50。
- `Booster-K1-Top_Rock-v0` env 配置实际引用 `k1_top_rock.npz`。
- `k1_top_rock_trained` deploy slot 引用对应 deploy NPZ、oneshot 配置。
- `k1_skills validate top_rock` 独立执行返回 `valid=true, problems=[]`。
- `k1_skills list` 显示 `arm_wave`、`mj_dance_002`、`top_rock`。
- `motion.process action=inspect` 能识别 train/deploy 两份 NPZ、286 帧、50 Hz、22 joints。
- `train_command` 独立生成：4096 env、3000 iterations、seed 42、Top Rock task id，训练 NPZ 实际存在。
- 全量测试独立执行为 **161 passed, 0 failed**。

### 但接入验收不完整

- validate 仅证明 schema、joint identity、步进和资源存在；它未验证 oneshot deploy NPZ 的 body FK。
- deploy NPZ body arrays 与正确 FK 大幅不一致，因此 deploy task 槽目前没有接入一个与训练 NPZ 相一致的完整 body reference。
- 训练任务注册源代码存在，但在非 Isaac AppLauncher 环境直接 import 会缺少 `pxr`；本评审没有启动训练，无法把“任务可注册”误写为已创建真实 Isaac 环境。

## 核心问题回答

1. **Top Rock 资源链是否可信？** 部分可信。源、编辑、trajectory 和训练 NPZ 的链路可追溯；许可证缺失；retarget/deploy 质量没有达到报告声称程度。
2. **K1 22-DOF Retargeting 是否正确？** joint columns、joint order、训练 NPZ 和 Root 正确；部署 NPZ body FK 错误，整体判定不通过。
3. **Root 运动是否保留并符合动作意图？** 是。路径、转体和时间轴独立复算一致，视频亦与移动/转体语义相容。
4. **运动学检查数值是否可信？** 关节/Root/rhythm 数值可信；足底数值和“全项干净”结论不可信。
5. **足底穿地和滑动是否属于可接受的训练前瑕疵？** 否。约 58 mm、近乎整段穿地，且已有物理 replay 跌倒证据。
6. **回放是否确实表现出连贯的全身 Top Rock？** 连贯、无破碎；姿态保真度仍受大链误差和足底限制，不能据此认定训练参考正确。
7. **`top_rock` 是否正确接入现有 Skill Framework？** 接线和 schema 正确，但部署 body reference 有实质错误，整体验收不通过。
8. **是否已经具备启动 RL Training 的基本条件？** **否。**

## 阻塞启动的问题

- 参考动作左右脚最大穿地约 58 mm，分别覆盖 254/257 帧，物理 replay 跌倒。
- 部署 NPZ 非根 body FK 与正确 MuJoCo FK 最大偏差约 0.391 m，最大 orientation 偏差约 120.7°。

因此本次任务为 **PARTIAL PASS**，`top_rock` 当前**不应启动 RL Training**。 
