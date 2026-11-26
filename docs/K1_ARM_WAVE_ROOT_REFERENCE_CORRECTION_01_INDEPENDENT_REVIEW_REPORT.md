# K1_ARM_WAVE_ROOT_REFERENCE_CORRECTION_01 — 独立评审报告

## A. 总体结论

**PARTIAL PASS**

本次确实定位并修正了 Arm Wave 参考中的核心语义问题：旧轨迹将源 SMPL pelvis 的全局运动传给 K1 Root，同时把 12 个腿部关节固定在站姿；新参考将 Root 固定、保留 22 个关节轨迹，并以部署模型 FK 重建 body reference。对新参考的静态几何和新策略运行数据也能独立复核。

但 Task E 的关键对照不满足“只改变 Root 参考、其余控制条件相同”的要求。新任务类没有继承原 Arm Wave 的增益、力矩限幅和 decimation 配置；同时原配置在更改 decimation 后没有重算 `physics_dt`。这会改变策略 action scale、PD 响应及参考推进相对于物理时间的比例。因此，新旧误差确有下降，但不能将其作为严格受控对照，也不能把改善全部归因于 Root 修正。

另有两项数据验证瑕疵：Task A 追溯脚本把 SMPL 的 Y-up 数据按 Z-up 的 x/y 轴计算水平路径；足底自检将 box 半尺寸在旋转后逐轴缩放，导致声称的 `+5.2 mm` 不是正确的 box 支持点高度。修正这两项后，核心判断仍成立：SMPL/K1 Root 有真实全局运动；新参考固定 Root 且 body FK 自洽；正确足底间隙约 `+4.69/+4.76 mm`，无穿透。

## B. Task A–E 验收表

| 任务 | 结论 | 关键证据 | 限制 |
|---|---|---|---|
| A：Root 异常来源 | **PARTIALLY VERIFIED** | GVHMR 与 edited SMPL 的 `transl` 完全相同；独立重跑 [FullBodyRetargeter](./k1_action/retargeting.py#L403-L432) 后，50 Hz retarget Root 与原 `trajectory.csv` 最大差约 `5e-10 m`；根位置路径 `1.471913 m`、净位移 `0.231597 m`，K1 yaw 净变化 `+309.600°` | [trace_root_anomaly.py](./tools/booster_deploy/scripts/trace_root_anomaly.py#L47-L55) 将 MS_WORLD 的 Y-up 数据误按 `x/y` 作为水平、`z` 作为竖直；其“源 pelvis 水平路径 1.311 m”和源 pelvis 垂直范围解释有误。视频可定性佐证动作变化，不能单独证明精确 309.6° |
| B：Standing Arm Wave 定义 | **PARTIALLY VERIFIED** | [新参考生成脚本](./tools/booster_deploy/scripts/make_stationary_root_reference.py#L65-L69) 固定 Root `(0,0,0.555023)`、identity，并原样保留头/臂/腿关节轨迹；与“站姿上肢挥臂”任务定义相符 | `z=0.555023 m` 有官方 K1 CSV 高度来源；`x/y=0` 是本地站姿原点，identity 是所选 K1/MuJoCo 中性朝向。未找到证据证明这两个分量来自一条官方 canonical stand pose，不应笼统称整个 Root 位姿为官方标准姿态 |
| C：新参考生成 | **VERIFIED（带几何数值更正）** | [生成脚本](./tools/booster_deploy/scripts/make_stationary_root_reference.py#L121-L179) 通过同 XML FK；246 帧、50 Hz；22 关节 NPZ 位置/速度逐元素差为 0；时间戳一致；全部 23 个 body FK 位置最大误差 `0.83 µm`，四元数最大 geodesic 差 `0.033°` | 角点自检公式错误（见该脚本 L202-L219）；正确最低点间隙是左 `4.686 mm`、右 `4.763 mm`，不是 `5.17/5.20 mm` |
| D：新旧参考对照 | **PARTIALLY VERIFIED** | 新旧同为 246 帧；reference joints/velocities 在 metrics 中完全相同；Root path `1.471913 m→0`、yaw `+309.600°→0°`；固定相机对比视频同步同帧 | [compare_references.py](./tools/booster_deploy/scripts/compare_references.py#L78-L90) 的 `1.408/1.781 m` 是碰撞 geom **中心轨迹长度**，不是已发生的接触滑移；参考足底实际高于地面约 4.7 mm，没有真实地面接触。全角点轨迹也确认旧 Root 目标造成显著世界位移 |
| E：现有 JIT 基线 | **PARTIALLY VERIFIED / 对照主张 CONTRADICTED** | 新旧均指向相同 JIT；哈希 `1fa8b882…a705`；两组 NPZ 的帧索引、时间标签、reference joint/velocity 数组逐元素相同；新运行 813 步、`stopped_early=False`、`fallen=0` | 新 task 继承通用 K1 配置而不是 Arm Wave 专用配置；PD、effort limit、action scale、decimation 不同。旧配置还存在 decimation 更改后未重算 physics timestep 的问题，报告未记录 MuJoCo 实际 `data.time`。因此新旧跟踪数值不是同条件对照 |

## C. 关键独立复算结果

### C.1 源运动与 Root

- 源文件 [segment_071.mp4](./inputs/hiphop87_elements/segment_071.mp4) 为 123 帧、25 fps；其哈希与 GVHMR 元数据记录一致。对源视频采样可见舞者姿态和身体位置变化；视频只提供定性支持。
- [GVHMR raw](./motion_studio/outputs/gvhmr/segment_071_343ee79d/segment_071/gvhmr_raw.npz) 与 [edited_motion.npz](./motion_studio/outputs/motions/segment_071_343ee79d/edited_motion.npz) 的 `transl` 逐元素相同。edited 文件哈希与 Arm Wave metadata 的 `source_sha256` 一致。
- Motion Studio 元数据明确 MS_WORLD 为 `+Y up`、米制、右手系。因此正确水平位移应取 XZ：
  - GVHMR/edited SMPL 水平路径：`1.471913 m`；
  - 净水平位移：`0.231597 m`；
  - SMPL `transl` 的竖直 Y 范围：`1.287703–1.372609 m`，相对起点约 `−15.7/+69.2 mm`。
- 原 [trajectory.csv](./datasets/k1_dance_v1/arm_wave/trajectory.csv) Root XY 路径：`1.471913 m`，净位移 `0.231597 m`；Root Z 固定约 `0.555023 m`。K1 yaw 正确按角度周期 `360°` 展开后净变化 `+309.600°`、peak-to-peak `352.649°`。
- 我用当前 [retargeting.py](./k1_action/retargeting.py) 从 edited SMPL 离线重算 retarget + 50 Hz resample，生成 Root 与现有 trajectory 的最大位置差约 `5e-10 m`。这支持 Root 的 XZ 平移传播和 K1 Root 输出一致；不是对整个上游 IK/模型精度的全面验证。
- yaw 独立复算：对源 `global_orient` 计算前向 `R@[0,0,1]`，按 MS_WORLD 的 Y-up 投影掉 Y，再用 `atan2(f_x,f_z)` 得到角度并以 `period=360` unwrap，源净 heading 变化为 `+309.600°`、peak-to-peak `352.649°`；与 K1 Root yaw 净变化吻合。因此 `+309.6°` 成立，不是把度误用默认 `2π` 的结果。
- 但 `trace_root_anomaly.py:rpy_deg()`（L39-L45）用 ZYX RPY 公式，并由 `report()`（L47-L55）同时处理 Y-up 源数据和 Z-up K1 数据。它对 K1 Root 的 yaw 计算可用；对源 SMPL 的 raw roll/pitch/yaw 分量标签不适用 Y-up 坐标约定。源数据的精确 heading 应以上述 forward-projection 复算为准，不能用该函数报告的 raw RPY 分量证明。
- 同一 `report()` 把 Y-up SMPL 的 `x/y` 误作水平、把 Z 当竖直；正确原始水平路径为 `1.472 m`，不是文档所述 `1.311 m`。源 `transl` 的竖直坐标是 Y，而不是该脚本打印的 Z。

### C.2 关节、时间轴与 FK

- 原始与新 `trajectory.csv` 均为 `(246,31)`，header 一致；`frame_index/timestamp_s` 之外的 22 个关节列最大文本解析差为 `5e-9 rad`（格式输出精度），时间戳逐项一致，范围 `0.0…4.9 s`，`246/50=4.92 s`。
- 原始与新 NPZ 的 22 维 `joint_pos` 和 `joint_vel` 最大绝对差均为 **0**。只可称“22 关节轨迹未改变”；Root 固定后全局 body positions/orientations 本来就改变了，不能称整个机器人动作完全未改变。
- 新 NPZ body names 全部存在于 [部署 XML](./virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml)。逐帧将 NPZ 的 sim-order 关节按 name 映射回 MuJoCo XML/qpos order 后独立 `mj_forward()`：
  - body positions 最大差 `8.27e-7 m`；
  - body orientation 最大 geodesic 差 `0.033°`。
 这验证了新 stationary-root NPZ 的 body reference 与它保存的 Root/关节状态相符，也验证了不是重复叠加 Root 变换。
- XML 文件路径与生成脚本指定的 FK 模型相同；部署 `K1_CFG.mjcf_path` 也引用同一 `{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml`。

### C.3 足底几何与“滑行”

- 部署 XML 中左右脚实际碰撞几何均为 ankle-roll link 上的 box，half-size `[0.09, 0.035, 0.018] m`、局部位置 `[0.026,0,-0.02] m`。独立读取 `geom_xpos/xmat` 并用正确角点公式 `geom_xpos + R @ (corner * half_size)`：
  - 新固定 Root 最低角点：左脚 `z=+4.686 mm`，右脚 `z=+4.763 mm`，246 帧恒定；没有穿透，但存在约 4.7 mm 的几何间隙。
- [make_stationary_root_reference.py](./tools/booster_deploy/scripts/make_stationary_root_reference.py#L202-L219) 和 [compare_references.py](./tools/booster_deploy/scripts/compare_references.py#L95-L110) 的自检表达式写成 `(R @ corners.T).T * half_size`，即先旋转符号角点、再按世界 xyz 分量缩放；非立方 box 旋转时这不是实际 box 角点。其产生的 `+5.171/+5.196 mm` 被独立复现为错误公式的输出。
- `compare_references.py` 的 `1.408216/1.780524 m`（L78-L90）可由我使用真实碰撞 geom center `geom_xpos` 的逐帧 XY 轨迹长度复现。全 8 个局部角点分别追踪时，旧 Root 下各角点轨迹长度范围左脚 `1.467–1.788 m`、右脚 `1.697–2.149 m`；新 Root 下均为 0。
- 这些是对**运动学参考几何位置轨迹**的路径统计，不是“足底与地面发生真实接触后的滑移距离”。最低角点仍高于地面约 4.7 mm，故报告应改称“参考下脚部几何世界轨迹长度”。它能说明 Root 与固定腿部组合造成脚部参考随 Root 移动/转向，但不能当作真实 contact manifold 滑动量。

### C.4 新旧 JIT sim2sim 数据

我独立读取两组逐帧 NPZ，并使用相同分析定义复算（动作段每周期 246 帧，transition 每周期 25 帧；all=22 joints、arms=索引 2–9、legs=10–21）：

| 参考 | 周期 | 全关节 MAE | 手臂 MAE | 腿部 MAE | Root 误差范数均值/最大 | 姿态误差均值/最大 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 | 1 | 84.98 mrad | 144.13 mrad | 49.72 mrad | 165.79 / 489.64 mm | 68.65° / 179.34° |
| 旧 | 2 | 89.46 mrad | 150.03 mrad | 54.63 mrad | 162.89 / 488.62 mm | 68.74° / 178.98° |
| 旧 | 3 | 89.43 mrad | 149.27 mrad | 55.51 mrad | 162.33 / 487.63 mm | 68.57° / 178.99° |
| 新 | 1 | 71.54 mrad | 128.01 mrad | 34.62 mrad | 20.85 / 32.61 mm | 2.82° / 4.09° |
| 新 | 2 | 72.62 mrad | 129.16 mrad | 35.14 mrad | 25.04 / 43.88 mm | 3.33° / 5.82° |
| 新 | 3 | 72.63 mrad | 129.04 mrad | 35.26 mrad | 25.00 / 43.89 mm | 3.35° / 5.84° |

- 两组数据均 813 行，时间标签、`abs_frame/frame/cycle/in_transition` 完全相同；`ref_joint` 和 `ref_joint_vel` 最大差为 0。三周期 hand/arm error 都可复算，周期 2、3 稳定。
- 误差下降在产物里真实存在：全关节约 `89.5→72.3 mrad`，手臂约 `150.0→129.2 mrad`，Root 参考误差范数均值约 `163→25 mm`（按周期动作段均值）；躯干姿态误差约 `68.7→3.3°`。新运行 `stopped_early=False`、`fallen=0`，height `0.5488–0.5530 m`。
- 但这不是只切换 Root 参考的对照。原始 JIT SHA256 与训练导出文件相同；**控制配置不同**（见 Task E），因此以上数值是两次实际运行结果，不是可归因于 Root 修正单一变量的实验。

## D. 对 GLM-5.3 核心主张的判断

| 主张 | 判断 | 理由 |
|---|---|---|
| Root 异常来自原始动作的全局运动 | **成立（SMPL 估计数据层面）** | 源 `transl` 与 edited motion 完全一致；正确 XZ 路径 `1.472 m`、净移 `0.232 m`，K1 Root 路径数值一致；全局 heading 序列净展开 `+309.6°` |
| 转换链没有相关 bug | **部分成立** | `_retarget_root()` 对 pelvis XZ 到 K1 XY、heading 投影的实现与重跑 CSV 一致；但 Task A 报告追溯脚本的源坐标水平/竖直轴标签错误，不能沿用其全部源 pelvis 数值或姿态描述 |
| 旧参考语义不一致 | **成立** | 源/转换动作 12 个腿关节恒定而 Root 有显著平移和 yaw；旧参考的脚部 collision geometry world path 非零；这是针对站姿 Arm Wave 的不一致 |
| 新参考 Root 固定正确 | **成立（作为显式 Standing Arm Wave 定义）** | 新 CSV/NPZ Root 恒定 `(0,0,0.555023)`、identity；该站姿坐标是可用定义。高度取值有官方数据支撑，但 XY 原点及 identity 更准确地说是本任务选择的坐标约定 |
| 22 关节动作完整保留 | **成立（仅指关节位置/速度）** | NPZ 位置和速度全帧最大差均为 0；trajectory 时间戳和 50 Hz 时间轴一致。Root 与全局 body motion 则被有意改变 |
| body reference 与 FK 一致 | **成立** | 部署同款 XML 逐帧独立 FK，最大位置差 `<1 µm`，四元数最大差 `<0.033°` |
| 新参考消除了足底滑行 | **部分成立** | 固定 Root 后 collision geom center 和 8 角点参考轨迹均不再移动；但旧路径是运动学 world path 而非真实地面接触滑移，足底实际离地约 4.7 mm |
| 新参考下现有 JIT 指标改善 | **指标下降成立；因果归因不成立** | 新 NPZ 中实测 MAE/Root/姿态误差确实下降；但新旧部署增益、action scale、effort limit、decimation/timestep 不同 |
| 新参考可作为 Standing Arm Wave 版本 | **成立（范围受限）** | Root/关节/body FK 自洽，且明确对应原地站姿上肢动作；不代表复现源视频的腿部动作、走位或转身，也不代表达到高精度跟踪 |
| 新旧指标证明只因 Root 参考变合理 | **不成立** | task config 不是同控制参数；旧物理每策略步仿真时长与新配置亦不相同，根因变量不受控 |

## E. 必须修正的问题

### 必须修正后，才可接受 Task E 的“同配置对照”结论

1. **统一任务配置。**  
   新的 `K1ArmWaveStationaryRootControllerCfg`（[定义](./tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py#L155-L172)）继承 `K1BeyondMimicControllerCfg`，而原 `K1ArmWaveControllerCfg`（[定义](./tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py#L101-L145)）才定义 Arm Wave 专用 stiffness、damping、effort limit 和 decimation。新任务因未继承原专用类，继承通用 K1 参数：
   - 原 Arm Wave：上肢 `Kp≈3.948, Kd≈0.2513`；髋/膝/踝使用逐关节自定义值；髋 pitch effort `68 N·m`、hip yaw `38.3 N·m`、knee `112 N·m` 等。
   - 新任务的基类：上肢 `Kp=4, Kd=1`；腿部 `Kp=80/30, Kd=2`；髋 pitch effort `30 N·m`、hip yaw `20 N·m`、knee `40 N·m` 等。
   - action scale 由 `0.25*effort_limit/Kp` 计算。示例：hip pitch 原/新约 `0.563/0.0938`，hip yaw `0.537/0.0625`，ankle roll `1.341/0.167`。差异足以改变策略输出目标和跟踪误差。
2. **修正并记录实际物理步长。**  
   [ControllerCfg.__post_init__()](./tools/booster_deploy/booster_deploy/controllers/controller_cfg.py#L111-L112) 按默认 `decimation=10` 计算 `physics_dt=0.02/10=0.002 s`。原 Arm Wave 子类之后才将 decimation 改成 4（[K1 config L144](./tools/booster_deploy/tasks/beyond_mimic/robots/k1/__init__.py#L144)），没有重算 `physics_dt`；MuJoCo controller 又直接把该值设给 `mj_model.opt.timestep`，并按 `self.decimation` 调用 `mj_step`（[mujoco_controller.py L21-L22, L238-L244](./tools/booster_deploy/booster_deploy/controllers/mujoco_controller.py#L21-L22)）。按当前代码，原配置为 `4×2 ms=8 ms/策略步`，而新配置默认 `10×2 ms=20 ms/策略步`。录制脚本的 `t` 字段只写 `(step+1)*policy_dt`（[deploy_mujoco_record_cyclic.py L213](./tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py#L213)），未保存 MuJoCo `data.time`。因此原报告把两者都当作相同的 20 ms 仿真推进/50 Hz 物理时间没有证据支持。
3. 要作严格旧/新 Root 参考比较，需使用完全相同的 K1 Arm Wave 控制参数和一致的 `physics_dt × decimation`，并在两次记录中保存实际 `data.time`。在此之前，旧/新 MAE 改善只能作为各自配置下的观察结果，不能用于单因素归因。

### 应更正的证据/表述

1. 将 trace 脚本的源坐标分析改为 MS_WORLD `XZ` 水平、`Y` 竖直；更正源 pelvis 水平路径为 `1.472 m`，并重算垂直范围。K1 Root 的 yaw `+309.6°` 独立复算成立。
2. 将两处 box 支持点公式改为先在局部坐标按 half-size 缩放再乘旋转矩阵；并将足底结果更正为约 `+4.69/+4.76 mm`。当前值仍无穿透，但“官方脚底间隙精确 +5.2 mm”不成立。
3. 将 `1.408/1.781 m` 明确为“collision geom center 的运动学参考世界轨迹长度”，不要称为实际接触滑移。

### 可接受的已知限制

- 新参考是 **Standing Arm Wave**，不是原视频完整全身动作：原视频/SMPL 中有腿部和全局位移信息，而新目标明确把 Root 固定、腿部保持常值。
- 现有 JIT 是在旧参考/旧训练配置下训练的；新参考运行是 zero-shot 行为样本。即使配置匹配，也不应把它解释成针对新参考训练完成。
- Root identity 与原点是新任务的选定坐标约定；只有 Root 高度数值可直接追溯到官方 K1 CSV 高度字段。

## F. 最终验收

1. **本次任务是否可以验收？**  
   **部分验收。** Root 异常定位的核心原因、新 stationary-root reference、22 关节保留、部署模型 FK 一致性和新 JIT 运行事实成立；Task E 的“同一控制配置下对照改善”暂不能验收。
2. **新参考数据是否可以保留为 Standing Arm Wave 参考版本？**  
   可以。Root 固定、关节轨迹原样保留、body reference 与 K1 XML FK 一致。保留时应把足底间隙修正为 `+4.69/+4.76 mm`，并明确它不是原舞者完整全身动作。
3. **原始参考是否应保留作为原始舞者动作数据，而不再直接用作当前站姿技能目标？**  
   是。原 Root 的全局位移和转向来自源估计动作，作为原始动作数据应保留；但在腿部保持站姿时，不应把该 Root 轨迹直接解释为站姿技能目标。
4. **现有 JIT 是否可以作为新参考的基线？**  
   可保留为一个**未针对新参考重训的策略样本**。当前新参考 metrics 证明该 JIT 在本次运行设置下可连续运行且未触发摔倒判定；但由于控制配置不一致，不能把它当作与旧参考严格可比的同配置基线。
5. **报告中的误差改善是否有足够证据支持？**  
   数字本身可从两个 metrics NPZ 复算：全关节动作段约 `89.5→72.3 mrad`、手臂约 `150.0→129.2 mrad`、Root 误差范数约 `163→25 mm`、躯干姿态误差约 `68.7→3.3°`。但由于 controller 参数和每步物理推进不同，**不足以支持“改善仅由 Root 参考修正造成”**。完成相同配置、相同物理时间基准的最小复测后，才能给出该因果结论。

本次评审只读完成；未修改工程代码、配置、模型、原始动作数据或仿真产物，也未运行实体机器人。
