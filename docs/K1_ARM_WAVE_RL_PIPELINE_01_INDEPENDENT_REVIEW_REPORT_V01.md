# K1_ARM_WAVE_RL_PIPELINE_01 — 独立技术评审报告

评审日期：2026-11-17  
评审对象：Arm Wave RL Training and MuJoCo Sim2Sim  
评审范围：仅本次 Arm Wave 数据、训练、导出、部署和演示结果。

## 1. 总体结论

**PARTIAL PASS**

证据支持：Arm Wave 数据被转换为 Booster 可读取的 motion NPZ；训练任务注册与 K1 BeyondMimic 配方大体一致；日志含完整 3000 个迭代的 reward/跟踪指标和最终 checkpoint；TorchScript 导出文件与部署策略文件字节一致；16 秒 MuJoCo 录制确实加载部署策略并运行，机器人在视频中保持直立。

但不能接受报告所称“约 16 秒、3.25 个动作循环的 Arm Wave 跟踪展示”作为已证明结果。部署策略对参考帧使用末帧钳制，不会循环；动作约 4.92 秒后参考姿态保持不变，视频后段也基本静止。报告所称第 2–3 循环的较低误差实际是末帧保持误差，不是重复动作跟踪误差。此外，训练目标数据的 12 个腿关节在全部 246 帧内都是常值，元数据明确注明 `upper_body_only`；因此该策略并未被训练去复现原视频中的腿部步伐。你的腿部不匹配观察有直接证据支持。

因此：**训练与策略部署的技术链路部分成立，单次 MuJoCo 执行和站立现象真实；但重复动作跟踪、完整视频动作复现及下肢动作匹配的主张不成立或证据不足。**

## 2. 分项评审结果

| 评审项 | 结论 | 关键证据与发现 |
|---|---|---|
| 数据转换 | **VERIFIED** | 原始 CSV 31 列、246 行；删去 `frame_index`、`timestamp_s` 后，其余 29 列与 Booster 输入 CSV 的列顺序和样本值一致。官方转换器重排四元数 xyzw→wxyz，并按关节名校验。 |
| NPZ 生成及读取 | **PARTIALLY VERIFIED** | 训练配置指向生成的 `k1_arm_wave.npz`；训练与部署 NPZ 的 SHA256 相同，均含 246×22 关节轨迹和 50 Hz。由于没有本次 `csv_to_npz` 命令原始 stdout/运行清单，生成过程本身只能由代码、文件内容及下游路径交叉证明。 |
| RL 任务注册 | **VERIFIED** | `Booster-K1-Arm_Wave-v0` 和 Play 变体已在 gym 注册；Arm Wave env 继承 `mj_dance_002` 对应配置，覆盖 motion 文件路径。训练运行参数与 TensorBoard 日志、checkpoint 数量相符。 |
| 训练配置一致性 | **PARTIALLY VERIFIED** | BeyondMimic 的基础 observation、奖励、rough terrain、延迟执行器随机化和 PPO 配方来自 K1 `mj_dance_002`。Arm Wave 实际跑 4096 env、3000 iteration，属于训练运行参数覆盖；不能称为“除 motion 文件外所有运行参数完全相同”。 |
| 训练过程与收敛 | **VERIFIED（训练日志数值）** | TensorBoard 文件有 3000 个 `Train/mean_reward` 样本：step 0 为 −11.27、step 2999 为 62.30；step 2999 `Metrics/motion/error_body_pos` 为 0.02937 m。训练用时约 2.6 小时没有独立墙钟日志佐证。Reward 不是动作跟踪质量的单独证明。 |
| 策略导出 | **PARTIALLY VERIFIED** | 最终 `model_2999.pt`、JIT、ONNX 均存在；导出代码传入 observation normalizer。部署 `.jit` 与本次 run 的导出 `.pt` SHA256 完全相同。ONNX 生成代码和文件存在，但实际 MuJoCo 运行使用 JIT；当前评审解释器缺少 Torch/ONNX Runtime，未能独立离线加载、比对推理输出。 |
| 部署控制一致性 | **PARTIALLY VERIFIED** | Kp、Kd、effort limit、动作缩放公式和控制频率的源码配置相符；50 Hz 策略、4×5 ms MuJoCo 步进已配置。训练包含随机执行延迟/增益，MuJoCo 部署是固定 PD；训练用 Isaac/PhysX 与 URDF，部署用 MuJoCo/XML，故只能称参数对齐，不能称运行时动力学完全相同。 |
| MuJoCo sim2sim | **PARTIALLY VERIFIED** | MP4 为 16.0 s、1280×720、50 fps；录制脚本由部署 controller 执行策略、叠加其 MotionLoader 参考 ghost，并同步保存 800 步误差。策略大约在第一个动作结束后保持末姿态，不是 3.25 个循环。 |
| 动作跟踪质量 | **PARTIALLY VERIFIED / 报告表述被反证** | 复算整体误差 74.46 mrad，但跨越 800 帧，其中 554 帧以上已处于参考末帧保持。第一个 246 步动作段全关节 MAE 92.66 mrad、手臂 MAE 160.02 mrad、腿部 MAE 54.48 mrad。所称后续循环手臂 71 mrad 不是循环跟踪。 |
| 身体稳定性及 Root 偏差 | **PARTIALLY VERIFIED** | 视频可见机器人未倒地；记录的参考躯干位置误差范数均值/最大值 221.67/493.30 mm，末段约 245 mm。记录只保存误差范数，不保存带符号 root 位置分量，故横向方向、累积规律和“为摆臂动量平衡”的因果解释无法独立验证。 |
| 腿部动作相对原视频 | **CONTRADICTED（全身匹配主张）** | 同一 Arm Wave 训练参考 CSV 中 12 个腿关节 246 帧均为严格常数；元数据标注 `upper_body_only`。关联原始视频 `0_input_video.mp4` 中可见舞者跨步/改变站姿。训练目标本身没有这些腿部动作。 |
| 可复现性 | **PARTIALLY VERIFIED** | 训练 event log、checkpoint、JIT/ONNX、motion NPZ、MP4 和 metrics 文件存在；hash 可建立导出 JIT→部署 JIT 的同一性。没有原训练命令终端日志或完整离线依赖环境，无法复跑训练/推理。 |

## 3. 关键指标独立复算

### 3.1 输入、时间轴与格式

- [arm_wave/trajectory.csv](datasets/k1_dance_v1/arm_wave/trajectory.csv)：31 列、246 行；第 0 帧 timestamp 为 `0.000000 s`，末帧为 `4.900000 s`，间隔严格 `0.02 s`，共 245 个间隔。
- 以帧周期定义动作长度：`246/50 = 4.92 s`；以首末采样时间差定义跨度：`4.90 s`。两者分别是 246 个采样周期和首末样本时间，不矛盾。报告计算 16 秒/4.92 秒约 3.25 次，算术正确，但部署实际没有实现循环。
- 删除前两列后，输出 29 列与 `virtual_k1/assets/booster_assets/motions/K1/k1_arm_wave.csv` 表头、每行数值一致。22 个关节保持 Booster CSV 顺序；肩部历史命名差异由 `csv_to_npz.py` 对预期和实际名称统一去除 `aa` 前缀后校验。
- 原始 CSV 四元数列为 xyzw；逐帧范数范围 `0.9999999994–1.0000000007`。转换器将其重排为 wxyz。转换器以 `(input_frames−1)/input_fps` 得到 4.9 s，并用 `round(duration*output_fps)+1` 输出 246 帧，未出现额外帧或丢失末帧。
- 训练 NPZ 和部署 NPZ均为 246×22，`fps=50`，两文件 SHA256 均为 `1d26a1ed56281f1f4c1fa706d7f3adc4d5bebcf103852eebd73e1296c83d8872`。
- 按 NPZ `joint_names`（将 CSV 中左右肩 pitch 的历史名称映射到 `aa` 前缀名称）逐关节比较，NPZ 与 CSV 的最大角度差为 `4.68e-6 rad`，符合 float32 保存精度。

### 3.2 训练日志

直接从 `tools/booster_train/logs/rsl_rl/k1_arm_wave/2026-11-17_13-20-05/events.out.tfevents...` 解析 TensorBoard scalar：

| 项目 | 独立结果 |
|---|---:|
| `Train/mean_reward` 标量记录数 | 3000 |
| reward，step 0 | −11.2715 |
| reward，step 2999 | 62.2980 |
| `Metrics/motion/error_body_pos`，step 2999 | 0.029372 m |
| `Metrics/motion/error_anchor_pos`，step 2999 | 0.197372 m |
| 最终 checkpoint | `model_2999.pt`，存在 |

配置保存的 `params/agent.yaml` 为 `max_iterations: 3000`；该 run 的 checkpoint 和 TensorBoard 样本均覆盖 step 0–2999。Hydra 原始 `.hydra/config.yaml` 仍显示 agent 默认 `max_iterations: 20000`，与实际运行参数文件不同；结合实际 3000 个事件和 `model_2999.pt`，执行次数可确认为 3000，但两份配置快照不一致应避免混用。

`error_body_pos` 不是全局 root 位置误差：训练代码先将被跟踪 body 转到 anchor-relative 表示，再对 body 位置误差取范数并求所选 body 的均值。部署 `anchor_err` 则是世界坐标下躯干/root 三维位置差的范数。29 mm 与 222 mm 因参考系和被测对象不同，不能解释成同一个指标的训练—部署恶化率。mean reward 汇总多个动作跟踪、姿态、力矩、动作率、关节限位等项；reward 上升只证明优化过程有进展，不能证明部署多循环跟踪成功。

### 3.3 部署逐帧误差

`k1_arm_wave_mujoco.metrics.npz` 实际保存 `joint_err (800,22)` 与 `anchor_err (800,)`。录制代码逐帧取绝对关节误差，再对元素求均值；不是 RMS。

| 指标 | 独立重算 | 说明 |
|---|---:|---|
| 全 800 帧、22 关节平均绝对误差 | 74.464 mrad | 被后续末帧保持段拉低 |
| 全 800 帧手臂误差 | 98.200 mrad | 与逐循环摘要口径不同 |
| 全 800 帧腿部误差 | 61.122 mrad | 受静态参考末帧阶段影响 |
| 第一个 246 步动作段，全关节 | 92.657 mrad | 更接近实际动作执行段 |
| 第一个动作段，手臂 | 160.024 mrad | MAE；最大绝对误差 1.730 rad |
| 第一个动作段，腿部 | 54.481 mrad | MAE；RMS 75.364 mrad |
| root/anchor 位置误差范数 mean/max | 221.665/493.303 mm | 仅有无方向范数 |
| 参考末帧保持阶段 anchor error | 约 245 mm | 第 492–799 步均值约 245.14 mm |

第一个动作段的腿部逐关节 MAE：左髋 pitch 69.3、左髋 roll 8.2、左髋 yaw 18.9、左膝 58.4、左踝 pitch 45.0、左踝 roll 18.3 mrad；右髋 pitch **180.6**、右髋 roll 49.9、右髋 yaw 23.4、右膝 **122.9**、右踝 pitch 18.3、右踝 roll 40.5 mrad。右腿髋 pitch 和膝是明显偏差项。由于当前参考的腿关节全为常值，这些误差说明机器人连静态目标姿势也未精确匹配；不能据此推导腿部舞蹈动作被正确跟踪。

### 3.4 Root 偏差与视频

- 保存的 `anchor_err` 是 `norm(robot_root_xyz - ghost_root_xyz)`。最大值出现在第 206 个采样附近，约 493 mm；首个动作段均值约 168 mm；第 2/3 个按帧索引切片的均值约 245 mm。
- 由于未保存 robot/reference 的 xyz 分量或 root pose 序列，无法独立确认偏差是横向、前后、垂直中的哪一项，也无法验证报告所称的摆臂动量补偿因果关系。视频能显示机器人与 ghost 分离，但不能给出精确轴向位移。
- 视频文件长度 16.0 s、50 fps；`metrics.npz` 有 800 步。录制脚本只有一次 `ctrl.start()` 和单个 rollout 循环，无循环内 reset。参考索引在动作末帧后保持不变，且 MP4 约 6 秒以后的采样画面基本静止。因此“不倒地”是这个策略执行后站立/保持的证据，不是 3 次动态 Arm Wave 循环均稳定的证据。

## 4. 分项技术核查

### A. 数据转换与腿部参考语义

**结论：数据转换格式通过；全身参考语义不支持“匹配原视频腿部动作”。**

转换输入与输出 CSV 的 29 列、顺序及样本数值一致。`csv_to_npz.py` 按列名转换四元数，读取 K1 CSV 关节顺序并经 Isaac articulation 做运动学 FK，保存 joint/body positions、velocities 和名字数组。训练配置实际 motion path 指向该 NPZ；部署 NPZ 与其 hash 一致。

更重要的是，[arm_wave/metadata.json](datasets/k1_dance_v1/arm_wave/metadata.json) 写明 retarget mode 为 `upper_body_only`，并注明腿部保持 official stance、输入腿部幅度超过其站立 PD 支持范围。对输出 CSV 独立统计，12 个髋/膝/踝角度的 246 帧 min=max，均严格为常数。关联源动作 [0_input_video.mp4](motion_studio/outputs/gvhmr/segment_071_343ee79d/segment_071/0_input_video.mp4) 时长约 4.1 秒，可见舞者有步伐和站姿变化。

因此，sim2sim ghost 本身已不是原视频的全身动作参考；其腿部保持站立是转换目标的明示语义。当前 RL 训练不能凭空学出已从参考轨迹中去掉的腿部动作。你观察到 K1 腿部和原视频不匹配，是事实层面正确的；这不是 MP4 单纯角度造成的错觉。

### B. 任务注册、训练配置及结果

**结论：训练流程确实执行；配置继承成立，但 reward/训练误差不能替代部署动作跟踪。**

- `arm_wave/__init__.py` 注册训练任务和 `-Play` 任务；训练配置继承 `mj_dance_002` 的 Rough/No-State-Estimation K1 环境，仅替换 motion path。PPO 文件改了实验名，并保存 3000 次训练参数；这不是逐字不变的官方默认运行配置。
- 解析后的训练配置显示 4096 环境、physics `dt=0.005 s`、decimation 4、运动参考路径正确、actor/critic observation normalization 开启、动作 scale 按 K1 actuator 配置生成。
- 基础 K1 任务含 rough terrain、无状态估计 actor observation、2–8 步执行延迟，以及材料/初始姿态/COM/增益/推力等随机化。训练 Play 环境继承 flat/no-state-estimation 配置并关闭 push event；MuJoCo 部署无 Isaac 的 terrain/randomization，延迟也没有按训练随机化。这是 sim2sim 域差异，报告只能说部分控制参数对齐。
- reward 和 `error_body_pos` 标量真实存在并已独立复算；`body_pos` 是 anchor-relative 多 body 指标，不包括以同一形式报告的 world root 漂移。29 mm 不是“整机位置误差 29 mm”。

### C. 导出、关节顺序与运行时控制

**结论：JIT 复制身份通过；normalizer/控制配置由代码支持，但离线模型推理未独立复跑。**

- 训练最终 checkpoint `model_2999.pt`、同 run 导出的 JIT/ONNX 文件存在。`play.py` 从 checkpoint 加载 PPO runner，读取 actor observation normalizer 并将其传给 JIT/ONNX exporter。
- 导出 JIT SHA256 为 `1fa8b882948ff63f36164ac61ba0ecfe64c15fd8aaeba03e2be6426db7c4a705`；部署 `models/k1_arm_wave.jit` SHA256 完全相同。部署配置引用此 JIT；ONNX 文件没有被此 MuJoCo 录制路径使用。
- 训练 NPZ 标记 joint names；deploy MotionLoader 按名称映射到 sim joint order，推理输出再按 sim→robot order 重排。MuJoCo K1 XML 的 hinge joint 顺序与 `RobotCfg.joint_names` 的官方 CSV 顺序一致；录制指标也按此顺序对照。没有发现 22 关节错序证据。
- 部署动作缩放按 `0.25 * effort_limit / stiffness` 计算，数值与训练 K1 action scale 对齐；PD Kp/Kd 和 effort limit 数值亦逐关节匹配训练配置。策略 20 ms 更新一次，每次 4 个 5 ms MuJoCo 子步，符合训练 `dt=5 ms, decimation=4` 的时间设置。
- 但训练执行器随机增益/延迟，部署使用固定增益、无延迟模型；训练通过 URDF/PhysX，部署加载共享 Booster K1 XML/MuJoCo。因此数字相同不等于控制器内部时序、接触、动力学行为完全相同。
- 当前评审环境没有 Torch、ONNX Runtime、Isaac Sim，不执行模型推理或重训。JIT 实际参与录制的证据来自 deployment controller 代码、hash 相同的导出文件和同次生成的误差数据。

### D. MuJoCo 视频及循环播放

**结论：策略确实在 MuJoCo 执行；“3.25 个循环”被代码和视频反证。**

部署配置的 `stop_at_motion_end` 默认 `False`。`BeyondMimicPolicy._set_command()` 使用：

`row_ids = min(current_frame, time_step_total - 1)`

`inference()` 只递增 `current_frame`，到结尾不回到 0；未设置停止时，控制器会继续运行但始终使用最后一帧 joint/root 参考。录制脚本以 50 Hz 跑 800 步，因此视频总长 16 秒；它没有自行实现参考动作循环。

`deploy_mujoco_record.py` 的误差计算在 `ctrl_step()` 后使用已递增的 `policy.current_frame`，而 ghost 在 `inference()` 中使用递增前的 command frame。因此误差数组与画面中的 ghost 存在约 1 个策略周期（20 ms）的时间对齐偏差；不会解释数秒的静止，但会影响精确跟踪数值。

视频与逐帧误差的时间切片显示：

- 0–约 4.92 s：执行一次 Arm Wave 参考；
- 约 4.92 s 以后：命令/ghost 停在参考末帧；机器人也收敛在该末姿态附近；
- 800 步“总体”误差包含后面约 11 秒的末帧保持。

因此，报告中循环 2/3 的 71 mrad 手臂误差不构成第二/第三次动作跟踪；把全 16 秒称为 3.25 个动作循环不准确。视频没有发现 rollout 代码内隐藏 reset；MP4 文件与 800 步指标长度相符，但缺少原始录制命令 stdout，不能单独证明 MP4 文件从未被外部剪辑。

## 5. 重大问题与一般问题

### MAJOR — 部署 motion 播放器末帧钳制而非循环，循环结果和整体误差结论失效

- **事实：**参考帧索引超过总帧数后固定在最后一帧，`stop_at_motion_end=False` 使 controller 继续但不循环。
- **证据：**`tools/booster_deploy/tasks/beyond_mimic/beyond_mimic.py` 的 `_set_command()`、`inference()`；`deploy_mujoco_record.py` 固定录制 duration；MP4 约 5 秒之后静止；metrics 第 246 帧后进入末姿态保持误差区。
- **影响：**3.25 循环的视觉/指标主张不成立；整体 74 mrad 与“循环 2–3 为 71 mrad”把动作跟踪与静态保持混在一起，低估动态阶段误差。
- **严重程度：**MAJOR。
- **是否需修复才能接受完整 sim2sim 动作跟踪成果：**是。最小修正是让参考帧按 246 帧周期回绕（或只录制一周期），并以明确的 loop phase 记录参考/实际状态后重新生成视频和指标。

### MAJOR — Arm Wave 参考轨迹没有腿部动作，无法复现原视频的下肢表现

- **事实：**训练数据在 246 帧中将 12 个腿关节全部固定为常量；retarget 元数据明确声明 `upper_body_only`。
- **证据：**`datasets/k1_dance_v1/arm_wave/trajectory.csv` 的逐列 min/max；`datasets/k1_dance_v1/arm_wave/metadata.json` 的 `retargeting_config.mode` 与腿部说明；关联原视频可见跨步/站姿变化。
- **影响：**RL 任务只收到固定腿部姿势作为目标，不可能学习视频中的腿部动作。deployment 的右髋 pitch/knee 对该静态目标也分别有约 181/123 mrad 首段 MAE。
- **严重程度：**MAJOR（若验收定义是全身 Arm Wave 与原视频匹配）；不否定训练过程本身执行。
- **是否需修复才能接受全身动作跟踪主张：**是。必须先让本次参考目标实际包含所需腿部运动，再训练/验证与该目标一致的策略；仅更改展示视频或循环误差无效。

### MAJOR — Root/anchor 偏差显著，且方向和成因不可由当前产物确认

- **事实：**部署 root 参考误差范数 mean 221.7 mm、max 493.3 mm，末段约 245 mm。
- **证据：**指标保存脚本直接以 robot 与 ghost root 世界坐标之差计算范数；`metrics.npz` 中 800 个范数值。
- **影响：**“站立”不能代表 Root 跟踪良好；约 0.2–0.5 m 的位置偏差是本次展示的重要跟踪限制。数据不含 xyz 分量，不能证明偏差全部横向、逐循环累积或确由摆臂补偿造成。
- **严重程度：**MAJOR。
- **是否需修复才能接受高质量跟踪主张：**是；最小证据修正是保存同步的 reference/actual root xyz、base orientation、关节位置和接触记录，并在正确循环窗口重算。因果解释需额外证据，当前不作归因。

### MINOR — 全程/循环误差口径和 ghost 对齐不一致

- **事实：**第一个动作段误差较高；整段 800 步含长末帧保持；误差索引在物理 step 后读取 `current_frame`，ghost 则是当前 command frame。
- **证据：**录制脚本调用顺序与 metrics 数组长度/分段统计。
- **影响：**比较值有一帧（20 ms）偏移，且跨动作与 hold 的均值不代表标准周期性能。
- **严重程度：**MINOR（相对不循环的重大问题）。
- **是否需修复：**若再提交跟踪精度声明，需要用同一 policy frame/time 对齐 reference 与 actual，并分开报告动作阶段、保持阶段。

### OBSERVATION — 训练 body position 29 mm 与部署 root position 222 mm 不可横向比较

- **事实：**训练 `error_body_pos` 计算 tracked body 相对 anchor 的姿态位置误差均值；部署 `anchor_err` 是世界 root 三维位置误差范数。
- **影响：**两数各自可能真实，但不能据此说 sim2sim 相比训练恶化了某个确定倍数。训练日志中的 `error_anchor_pos` 是另一项指标，最后值约 197 mm。
- **严重程度：**OBSERVATION；应校正报告解释，不要求改训练代码。

## 6. 最终成果判定

1. **是否可以确认 Arm Wave RL 训练完成？**可以确认该 run 确实完成 3000 个记录迭代并保存了最终 checkpoint；但本次训练目标是 upper-body-only 版本，不是原视频全身动作。
2. **是否可以确认导出的策略在 MuJoCo 运行？**可以。部署 JIT 与该训练 run 导出的 JIT hash 一致；录制脚本创建部署 controller 并从 policy rollout 生成 800 步指标。
3. **是否可以确认 sim2sim 展示对应本次训练策略？**有强证据支持：部署配置选用该 JIT，文件 hash 一致，视频与同次 metrics 数量相符；缺少原始命令日志和离线推理复跑，故不宣称密码学意义的录制来源证明。
4. **机器人是否在 16 秒展示中保持站立？**视频中可见其保持直立、没有明显摔倒；录制循环不含 reset。缺少逐帧 base height/contact/reset 日志，故这是视频范围内的视觉结论，不是严格的接触/稳定性统计。
5. **挥手方向和节奏是否与 ghost 一致？**首个动作段视觉方向总体相符，但量化手臂 MAE 约 160 mrad；动作结束后 ghost 和机器人停在末帧。不能说 16 秒中完成三轮匹配。
6. **腿部是否与原视频匹配？**不匹配。下肢动作在目标数据中被明确固定为站姿，源视频有步伐。你的观察正确。
7. **能否称为成功的 Arm Wave RL 动作跟踪策略？**可以称为“已训练并在 MuJoCo 执行的一次 upper-body-only Arm Wave 候选策略”；在循环播放没有实现、下肢参考无动作且 root 误差较大的现状下，不足以称为完整、成功的原视频 Arm Wave 全身跟踪策略。
8. **GLM-5.3 执行报告是否准确完整？**训练 reward、训练误差、MP4 时长、JIT 复制及误差数组汇总大体有直接证据；“3.25 个循环”“后续循环跟踪改善”不准确；若将其展示理解为腿部也匹配原视频，则目标数据与表述明显不符。Root 偏移的方向和因果解释缺乏保存的数据支撑。

## 7. 最小修正要求

仅针对本次交付：

1. 修正 MuJoCo 参考帧播放为真实循环，或将视频和指标严格限定为一个动作周期；同步修复 reference/actual 时间索引后重录。
2. 若本次验收目标包含与原视频的腿部匹配，当前固定腿部轨迹不能满足目标；必须使用包含期望腿部运动的本次训练参考，才能据此声称全身匹配。若本次任务本来只验上肢，应把限制明确写入交付结论。
3. 重新采集逐帧 robot/reference root xyz、base height、关节位置及接触状态，分动作循环、腿/臂分别输出误差。当前仅需此项以支持所声称的 Root 漂移方向、循环跟踪与物理稳定性结论。

本报告不评价其他动作或项目后续路线。
