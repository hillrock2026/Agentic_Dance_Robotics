# K1_DANCE_SKILL_003_MOTION_RETARGETING_01_GPT_INDEPENDENT_REVIEW

日期：2026-11-20

## 1. 总体结论

**PARTIAL PASS**

Top Rock 的源动作链、286 帧/50 Hz 时间轴、训练 NPZ 的 22-DOF joint/root 数据和大部分运动学数值均有实际文件及独立复算支持。但两个问题阻止将当前参考运动视为已达到 RL Training 准入条件：

1. deploy NPZ 的 `body_pos_w/body_quat_w` 确实由 sim-order joints 错写进 MuJoCo XML qpos 顺序而生成。该错误真实、可精确复现，非根 body references 不能继续作为可信参考。
2. 使用 K1 XML 中实际 foot collision box 的旋转世界角点复算后，左右脚分别最大穿地约 **58.03 / 58.39 mm**，且分别有 **254 / 257** 帧处于负高度。该结论不是固定坐标偏移或视觉几何误解。

因此：

- 训练 NPZ 的 joint/root/body 参考基本可信；
- 当前 deploy NPZ 的非根 body references 不可信；
- 当前参考动作存在严重的足底几何不可行性；
- `validate` 和测试通过不能推翻上述直接 FK/几何证据。

## 2. 评审范围与证据来源

已直接检查：

- 原始 manifest、`segment_082.mp4`、cropped 视频；
- Motion Studio clean/edited JSON 与 edited NPZ；
- top_rock metadata、trajectory、training/deploy NPZ、validation；
- `K1_22dof.xml` 的 joint qpos addresses、foot collision box；
- `convert_k1_trajectory.py`、`check_reference_quality.py`；
- top_rock Skill、训练 task、deploy task；
- `reference_playback.mp4`；
- `top_rock_foot_audit.json` 与逐帧 foot CSV；
- 现有 `k1_skills validate` 和两项 joint identity 测试。

独立复算使用实际 MuJoCo FK、实际 geom center/matrix/half-size 和 NumPy 数值计算。没有重新训练或修改任何文件。

## 3. A. 动作资源和数据链

**结论：PASS WITH QUALIFICATIONS**

已验证：

- manifest 记录 `segment_082.mp4` 为 411.44–416.36 秒、4.92 秒片段；视频元数据为 123 帧、25 fps、4.92 秒。
- cropped 视频 SHA256 与 Motion Studio JSON 的 `source_hash=72c3b3ef…1a6d72` 一致。
- edit journal 含 trim 10–110 和 speed factor 0.7；输出为 143 帧、25 fps、5.72 秒，时间戳从 0 到 5.68 秒。
- edited NPZ SHA256 为 `7cec0f618c756ee4fa…e3b6`，与 top_rock metadata 完全一致。
- K1 trajectory 为 286×31，0–5.70 秒，0.02 秒间隔；`286/50=5.72` 秒的帧周期表示成立。
- 29 列 booster CSV 与 trajectory 保留列最大差约 `5e-9`。

资源链可信，但 HIPHOP 输入目录没有发现可证明再分发/商业使用权限的许可证文本。单目 GVHMR 的绝对尺度和深度本身也存在估计不确定性；这两个是已记录的非阻塞证据限制。

## 4. B. K1 Retargeting

**结论：PARTIAL**

### 4.1 joint identity 与 trajectory

canonical K1 sim order、training NPZ 和 deploy NPZ 的 `joint_names` 一致，22 个名称逐项匹配。train/deploy `joint_pos` 最大差为约 `1.43e-6 rad`，属于 float32/CSV 转换量级，没有左右交换证据。

trajectory 官方 CSV 顺序映射到 train NPZ 时也没有发现列错位。root position、quaternion 从 trajectory 到 train NPZ 的保存基本一致，free-root 未被固定。

### 4.2 deploy body FK 顺序错误

该错误真实存在。

MuJoCo XML 前 10 个 hinge joints 是 CSV/real 顺序风格：

```text
aahead_yaw, aahead_pitch, left_shoulder_pitch, left_shoulder_roll,
left_elbow_pitch, left_elbow_yaw, right_shoulder_pitch,
right_shoulder_roll, right_elbow_pitch, right_elbow_yaw, ...
```

而 `joint_names`/`joint_pos` 是 sim order：

```text
aahead_yaw, left_shoulder_pitch, right_shoulder_pitch,
left_hip_pitch, right_hip_pitch, aahead_pitch, ...
```

使用 XML joint name→实际 qpos address 得到：

```text
[7, 9, 13, 17, 23, 8, 10, 14, 18, 24, 11, 15,
 19, 25, 12, 16, 20, 26, 21, 27, 22, 28]
```

与报告所列完全一致。

[convert_k1_trajectory.py](tools/booster_deploy/scripts/convert_k1_trajectory.py#L96) 将 `joints_sim` 直接放入 `d.qpos[7:]`。独立复算证明：

- 当前 deploy NPZ 存储的 body arrays 与这个**错误、直接写入**的 FK 结果完全吻合；
  - position 最大差仅 `6.29e-8 m`；
  - quaternion 最大角差约 `0.033°`（float32 存储量级）。
- 按正确 name→qpos address 映射重新 FK 后：
  - trunk position 误差仍为 0；
  - left elbow：约 67.083 mm；
  - left knee：约 82.140 mm；
  - left ankle：约 141.461 mm；
  - right ankle：约 391.131 mm；
  - 最大全 body position error：0.39113135 m；
  - 最大 orientation error：120.7248°。

这些数字可复现，因此原评审报告并非夸大。

### 4.3 影响范围

- **joint_pos/joint_vel 本身正确**；错误发生在 deploy NPZ 的 body reference 生成阶段。
- 训练 NPZ 未受此特定 converter bug 影响；用正确映射复算 train NPZ body FK 最大位置误差约 `4.46e-7 m`。
- 当前 deploy NPZ 的：
  - `body_pos_w`
  - `body_quat_w`
  - 由它们差分得到的 `body_lin_vel_w`
  - `body_ang_vel_w`
  均不可视为完整可信。
- trunk correct 是因为 root body 不受 22 个 hinge joint 顺序影响，所以 trunk-only self-check 完全无法捕获该错误。
- 当前 `play_reference_motion.py` 使用 joint-name→qpos address 映射重放 joint trajectory，并只从 body array 读取 trunk；因此视频可连续播放，不能作为 deploy body FK 正确性的证据。

### 4.4 是否需要重生成 deploy NPZ

**是。** 最小必要修复是按 joint name 写入 qpos address 后重新生成全套 body positions/quaternions/velocities，并增加非根 body FK 一致性检查。错误不要求修改 joint trajectory 本身。

## 5. C. 运动学质量

**结论：PASS WITH QUALIFICATIONS**

以下数值在 `dt=0.02s`、实际 joint arrays 上可复现：

| 指标 | 独立结果 |
|---|---:|
| 关节限位违例 | 0 |
| 最小限位裕量 | 0.0207000089 rad，left ankle roll |
| 最大速度 | 3.24300006 rad/s |
| 最大加速度 | 81.0750015 rad/s² |
| 最大帧间 jump | 0.06486000 rad |
| >0.2 rad jumps | 0 |
| Root xy path | 1.2607691 m |
| Root 净位移 | 0.2163111 m |
| Root z | 0.53615594–0.59775920 m |
| yaw net | -86.61281° |
| yaw span | 100.17168° |
| max yaw rate | 101.49021°/s |
| rhythm envelope corr | 0.74808934 |

方法核查：

- path 是逐帧 xy 位移范数之和，不是 net displacement；两者已区分。
- yaw 使用 unwrap 后计算 net/span/rate，未被 ±180°跳变污染。
- 最大速度/加速度使用 `np.gradient`；最大加速度为二次差分量，对夹紧/饱和附近的离散噪声敏感，不能直接等同动态可控性证明。
- 相关性 0.748 定义为 SMPL `body_pose` 梯度范数包络与 K1 joint velocity 范数包络的 Pearson correlation。它仅支持节奏包络相似，不能单独证明姿态或动作语义忠实。

retarget metadata 中：

- forearm median/mean error 约 52.97° / 61.52°；
- shank median/mean error 约 27.45° / 29.84°；
- foot median/mean error 42.50° / 50.31°。

这些大误差意味着“全身链方向保真”存在明显限制，不能只引用 upper-arm 2.12° 和 thigh 0.33° 来描述全身 retarget。该限制影响动作忠实度，但单独不构成改变本次训练准入结论的证据；严重穿地和 deploy FK 错误才是直接阻塞项。

## 6. D. 足底行为

**结论：FAIL**

### 6.1 几何计算核查

K1 XML 中左右足实际碰撞 geom 是 box：

```text
pos = (0.026, 0, -0.02)
size/half-extents = (0.09, 0.035, 0.018)
```

visual mesh 明确为 `contype=0, conaffinity=0`，而该 box 没有关闭碰撞，是 MuJoCo 接触/physics 使用的足底表示。

独立步骤为：

1. 使用 train NPZ joint/root；
2. 以 correct joint-name→qpos-address 写入 `d.qpos`；
3. `mj_forward`；
4. 对每只脚 box 的 8 个局部 half-size 符号角点执行：

   `world_corner = geom_xpos + geom_xmat @ corner`；

5. 取 world z 最小值作为 ground plane z=0 下的最低碰撞角点。

结果：

| 指标 | Left | Right |
|---|---:|---:|
| 最低 z | -58.03174 mm | -58.39094 mm |
| 最大穿地深度 | 58.03174 mm | 58.39094 mm |
| 负 z 帧数 | 254/286 | 257/286 |
| 最深帧 | 155 | 124 |
| 最深帧 foot-ground contacts | 4 | 4 |
| 最大 world corner z | 14.337 mm | 40.810 mm |

结果与仓库的 `top_rock_foot_audit.json` 逐帧产物一致，且在最深帧 MuJoCo contact distance 达约 -58 mm。重复检查 train NPZ而非错误 deploy body arrays，确认穿地来自 joint/root reference 本身，而非 deploy FK bug。

### 6.2 58 mm 是否假象

不是假象：

- 使用了实际 collision box 而非可见 mesh；
- `geom_xmat` 已包括 root、hip、knee、ankle pitch/roll 的完整旋转；
- 每个角点只应用一次世界变换，没有重复旋转；
- ground plane 是 z=0；
- MuJoCo contact manifold 在负深度处直接复现穿透；
- 结果不依赖 deploy NPZ 的错误 body arrays。

固定 ankle-to-sole offset 方法在第 0 帧计算 `ankle_z - sole_z`，之后忽略 box orientation 对最低角点的影响，因此原 10.5/3.3 mm 只能是旋转不敏感的近似，不能作为当前参考的足底上限。

### 6.3 支撑相滑动

滑动必须与穿地分别评价。

`check_reference_quality.py` 将 `sole<0.02m` 全部视为 contact；由于严重穿地，这个 mask 把穿地帧和真正支撑帧混在一起。其 0.20–0.25 m/s 因而不是可靠的独立物理支撑滑动统计。

使用 source edited motion 的 contact 估计重新映射到 286 帧后，ankle xy speed 在标记为 support 的帧约为：

- left mean 0.255 m/s；p95 0.726 m/s；max 0.945 m/s；
- right mean 0.270 m/s；p95 0.925 m/s；max 1.113 m/s。

这只证明 reference kinematics 存在明显的水平足部平移；source contact 本身是 height/velocity 估计，不是力测量，故不能将 p95/max 直接判为物理“支撑滑动”。关于“足链 42.5°误差导致穿地/滑动”的说法仍是未建立统计因果链的假设。

结论：穿地是强、独立、可复算的阻塞证据；滑动是有限证据的已知瑕疵，不应与穿地合并或夸大为同样确定的因果结论。

## 7. E. 参考回放

**结论：PASS WITH QUALIFICATIONS**

已读取实际 `reference_playback.mp4`：

- 286 帧，50 fps，5.72 秒；
- 对 9 帧及更密集的 24 帧抽样覆盖完整时长；
- 未见结构破碎、joint teleport、左右整体交换或躯干翻转；
- Root 平移和 yaw 转动连续；手臂与步法在多个阶段有基本协调；
- 可辨认持续 step/turn 的 Top Rock 类动作语义。

该视频只证明 CSV/joint trajectory 在 joint-name mapping 下的连续播放。它不证明：

- deploy body FK arrays 正确；
- 足底 collision geometry 不穿地；
- 任何 trained policy 的物理执行可行。

地面网格、相机距离和模型 visual geometry 可以掩盖毫米至厘米级的 collision box 穿地。因此视觉通过不能覆盖 D 项失败。

## 8. F. Skill 接入

**结论：PARTIAL**

已验证：

- `skill.yaml` 是 oneshot/free-root，fps=50，source_frames=286；
- training NPZ 路径存在且为 286×22；
- Top Rock task env 实际替换 motion file 为该 training NPZ；
- deploy slot 引用对应 deploy NPZ，oneshot/loop false；
- `k1_skills validate top_rock` 当前返回 `valid=true`；
- `list` 和 `motion.process inspect` 能识别 Top Rock；
- joint identity 测试扩展到 Top Rock 并通过；
- 训练命令可生成 4096 env、3000 iter、seed 42、正确 task id。

但现有 validate 只检查 schema、资源、joint names/order、frame/fps 等，没有验证：

- oneshot deploy NPZ 的 body FK 是否能从 joint_pos/root 重新生成；
- train/deploy body reference 一致性；
- foot collision geometry 的严重穿地。

因此当前 Skill 的“训练 wiring”部分通过，但“完整训练/部署参考一致性”未通过。

**Training impact：** train env 消费 training NPZ，该文件的 body FK 没有发现顺序错误，因此 deploy body FK bug 本身不直接污染 RL 训练张量；它直接影响 deploy 资源完整性和训练/部署一致性。足底穿地则直接存在于 training NPZ/reference root/joints，是训练目标的实际问题。

## 9. 核心争议复核结果

### 争议一：deploy NPZ body FK 顺序

- **错误是否真实：是。**
- **证据：** XML qpos 地址独立确认；当前 stored arrays 与错误 direct-write FK 精确匹配；正确映射下出现大误差。
- **影响 joint_pos：否。** joint arrays 保持 sim order，train/deploy joint trajectories 一致。
- **影响 body refs：是。** `body_pos_w/body_quat_w` 及派生速度错误；trunk 除外。
- **trunk 自检为何失效：** trunk 是 root/free body，不随 hinge order 改变。
- **是否重新生成 deploy NPZ：是。** 当前非根 body arrays 不能作为可信 resource。
- **对 training 的直接影响：** 不由 deploy body bug造成；train NPZ 独立且 FK 正确。对 deploy/reference 一致性有实质影响。

### 争议二：58 mm 穿地

- **是否真实：是。**
- **证据：**正确 joint mapping、实际 K1 collision box、8 个旋转 world corners、ground z=0、独立 MuJoCo contact distances。
- **不是仅算法假象：**消除固定 sole offset；使用实际碰撞 geom；在 254/257 帧及最深帧 contact manifold 中均复现。
- **是否自动导致物理 policy 必然失败：** 不能从静态参考穿地单独断言训练策略必然跌倒；但 reference reward target 已包含不可同时满足的足-地关系，且现有 PD replay 记录跌倒。因此不能在当前状态称为“基本可训练参考”。
- **滑动：** 0.20–0.30 m/s 量级的数据依赖 contact estimate，不能与穿地结论等同，但也不能据此反驳穿地。

## 10. 训练准入结论

训练 task 文件、训练 NPZ 和 CLI wiring 已存在，但这不是运动参考达到训练准入质量的充分条件。

当前阻塞项：

1. training reference 左右脚碰撞 box 近乎全段穿地，最大约 58 mm，且已有 physical replay fall 证据；
2. deploy NPZ 的非根 body reference 由错误 qpos mapping 生成，训练/部署参考资源不一致。

训练 NPZ 的 22-DOF joint/root 数据不受 deploy converter bug 影响，因此无需否定整个数据链或重新做 GVHMR/Retargeting。但是，当前 Top Rock reference 未达到可直接进入 RL Training 的基本物理/资源一致性要求。

## 11. 必要的最小修复建议

仅限本任务直接相关的最小工作：

1. 在 `convert_k1_trajectory.py` 中用 joint name→MuJoCo qpos address 写入每帧 joints；重新生成 deploy NPZ 的 body position/quaternion/linear/angular velocity。
2. 增加至少一个非根 body 的 FK round-trip 检查，避免 trunk-only false pass。
3. 对 training/deploy 的 joint trajectory、root、body references 做一致性检查。
4. 使用实际旋转 collision-box corners 修正足底质量检查，替代固定 frame-0 ankle offset。
5. 根据逐帧穿地区间，对足底参考的 root height/ankle pitch-roll 做限定于该轨迹的可行性修正或明确验证可接受性，再重新跑物理 kinematic replay。
6. 将 support/contact/sliding 指标与 ground penetration 分开报告，并说明 contact mask 来源。

## 12. 证据限制

- 未重新训练，因此无法直接判断 RL 优化后对穿地的容忍或补偿能力；现有结论基于 static reference geometry 和已有 PD replay。
- direct `isaaclab` task import 在当前普通 Python 环境缺少 `pxr`；本评审验证了注册源码和 `train_command`，未创建 Isaac 实际训练环境。
- 原始视频许可文本缺失；不影响工程数值复核，但影响资源使用许可确认。
- 0.748 correlation 和视觉连续性不代表逐帧动作保真度或物理可行性。
- physical replay 的 V1 controller/gains 与未来 trained policy 不同，因此 `fell=true` 是风险证据，不是预测训练失败概率。

## 最终决定

**暂不允许进入 RL Training。**

允许条件不是框架改造，而是至少完成：deploy NPZ 的正确 FK 重生成，以及对约 58 mm / 近 90% 帧穿地的参考足底进行复核并解决或取得有数据支持的明确物理可接受结论。
