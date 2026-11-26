# K1_DYNAMICS_MODEL_SPEC_V1 — Booster K1 全身动力学模型技术规范（第 1 版）

- 版本: V1（规范设计稿，不含实现）
- 日期: 2026-11-23
- 依据: `K1_DYNAMICS_MODEL_WEEK1_FILE_INVENTORY.md`（文件盘点报告，下称 [INV]）
- 标注 `[RUNTIME-VERIFIED 2026-11-23]` 的事实为规范撰写时在项目 `.venv`（MuJoCo 4.0.1）中加载主模型实际验证；其余设计决策为本规范裁定。

---

## 0. 范围

### 0.1 一个月目标

| 周次 | 交付 |
|---|---|
| W1 | 数据采集器 `k1_dynamics/collector.py` + 不变量测试通过 + 试点数据集（~50k 样本）+ MLP 冒烟训练 |
| W2 | 完整数据集 D1（≈0.5–0.6 M 样本）+ MLP baseline（实验 E1）+ 评价工具链 |
| W3 | Residual-based Soft-constrained PINN（实验 E2）+ λ_dyn 扫描（实验 E3） |
| W4 | 数据规模对照（实验 E4）+ 跨动作泛化评价 + 结果报告 `K1_DYNAMICS_MODEL_V1_RESULTS.md` |

### 0.2 Non-goals（V1 明确不做）

- 不实现接触力预测网络（qfrc_constraint 作为训练时已知量，见 §3.5）；
- 不做可微仿真 / 解析可微动力学（M 与 qfrc_bias 取自 MuJoCo，梯度只回传到 qacc_pred）；
- 不学习质量矩阵 M（M 来自 `mj_fullM`，是固定物理量）；
- 不做 sim2real、真机部署、IsaacLab 侧改动；
- 不修改任何现有 MJCF / 训练配置 / 动作资产；新代码与新数据全部落在新增目录（§10）；
- 不将 Top Rock 286 帧参考运动学轨迹当作动力学训练数据（参考只用于生成 PD 目标，§6.2）。

---

## 1. 物理基线（唯一 ground truth 栈）

**决策 D7：V1 动力学数据的唯一来源是 `virtual_k1` 的 MuJoCo 主模型。**

| 项 | 值 | 依据 |
|---|---|---|
| 模型 | `virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml`（未修改） | [INV] §B |
| 引擎 | MuJoCo **4.0.1**，integrator = **Euler**（XML 未指定，枚举 0）[RUNTIME-VERIFIED] | [INV] §A |
| 维度 | nq=29, nv=28, nu=22；qpos = root(3+4) + 22 hinge | [RUNTIME-VERIFIED] |
| timestep | **0.001 s**（模型原值，采集器不得覆写） | `K1_22dof.xml:4` |
| 执行器 | 22 × motor, gear=1, joint transmission ⇒ `qfrc_actuator[6+i] = clip(ctrl[i], forcerange_i)` | [INV] §C |
| 质量 | 19.666 kg；armature 计入 M 对角 | [INV] §B |
| 接触 | 足盒 condim=3，有效 μ_slide=1.0 / μ_spin=0.005 / μ_roll=0.0001；每足最多 4 角点 | [INV] §D |
| 重力 | [0, 0, −9.81]（含于 qfrc_bias） | [INV] §B |

**选择 virtual_k1 的理由**：(a) 官方 XML 原样加载，timestep 保持 0.001 s（booster_deploy 会在运行时覆写为 0.005 s 且用另一套 PD 增益，IsaacLab 是 PhysX+URDF+延迟执行器，三栈数据不可互换，[INV] §H Q1）；(b) `K1Sim` 已暴露全部所需字段读取接口（[INV] §E）；(c) 主 XML 无 passive 力 ⇒ 残差方程最简（§3）。**适用范围**：本规范产出的数据、模型与结论只在该 MuJoCo 栈内有效，不外推到 IsaacLab 训练环境或真机。Top Rock RL 训练按现有 IsaacLab 流程独立推进，互不阻塞。

---

## 2. 坐标系、维度与顺序约定

### 2.1 dof 布局与分组（评价与误差分解均按此表）

| dof 索引 | 内容 | 坐标系 | 单位 |
|---|---|---|---|
| 0:3 | root 线速度 / 线加速度 | **世界系**（free joint 线速度分量在世界轴上）[RUNTIME-VERIFIED] | m/s, m/s² |
| 3:6 | root 角速度 / 角加速度 | **root 局部系**（free joint 角速度分量在 body 系上）[RUNTIME-VERIFIED] | rad/s, rad/s² |
| 6:8 | 头 yaw/pitch | 关节轴 | rad, rad/s, rad/s² |
| 8:12 / 12:16 | 左臂 / 右臂（shoulder p/r, elbow p/yaw） | 关节轴 | 同上 |
| 16:22 / 22:28 | 左腿 / 右腿（hip p/r/y, knee, ankle p/r） | 关节轴 | 同上 |

**决策 D1（qvel/qacc 坐标约定）**：
- 模型全部使用 MuJoCo dof 空间量：`qvel[0:3]` 为世界系线速度，`qvel[3:6]` 为局部系角速度；`qacc` 是 `qvel` 各分量的时间导数（dof 空间广义加速度），与 qvel 同坐标系布局。root 角加速度（dof 3:6）是**局部系角速度分量的导数**（隐含 ω×Ω 项），不等于世界系角加速度——评价"root 角加速度误差"时即指 dof 3:6 误差，不做坐标系再换算。
- 实证方法（已做，写入采集器单测）：设 trunk quat 为非 yaw 旋转（如绕 x 30°）、`qvel[3:6]=[0,0.2,0]`，`mj_objectVelocity(flg_local=0)` 返回的世界系角速度等于该向量经 quat 旋转的结果，而线速度等于 `qvel[0:3]` 原值。
- **禁止**从 29 维 qpos 直接差分构造 28 维 qvel（四元数差分与坐标系都会错）；所有 qvel/qacc 直接读 `MjData`。

### 2.2 顺序与命名规则

- 一切索引映射以**关节/执行器名称**解析（`virtual_k1/virtual_k1/joint_mapping.py` 的 `JointMapping`），禁止位置硬编码。事实核查：主模型执行器顺序与 hinge qpos/dof 顺序逐名一致 [RUNTIME-VERIFIED]，即 `ctrl[i] ↔ dof[6+i]`，但代码仍必须走名称映射。
- 输入中的 22 维关节角/力矩按 **XML hinge 顺序**（= dof 6:28 顺序）。与官方 CSV 顺序的差异（肩 pitch 的 `aa` 前缀）由 `JointMapping` 别名处理，不进入模型数据。
- 四元数一律 **wxyz**；数据集内 qpos 原样存储；作为网络输入特征时做**半球归一化（w≥0）**消除双覆盖跳变（写数据加载器时处理，参考 `k1_action/virtual_k1_replay.py:52` 的既有做法）。

### 2.3 符号表

| 符号 | 含义 | 来源 |
|---|---|---|
| q ∈ R²⁹ | 广义位置（root pos + quat wxyz + 22 关节） | `d.qpos` |
| v ∈ R²⁸ | 广义速度 | `d.qvel` |
| a ∈ R²⁸ | 广义加速度（标签） | `d.qacc`（mj_step1 时刻，见 §6.4） |
| τ ∈ R²² | 实际执行器力矩 | `d.qfrc_actuator[6:28]` |
| M(q) ∈ R²⁸ˣ²⁸ | 含 armature 的质量矩阵 | `mj_fullM(m, d, M_dense)` |
| b(q,v) ∈ R²⁸ | bias 力（重力+科氏） | `d.qfrc_bias` |
| f_c ∈ R²⁸ | 约束力（接触+关节限位） | `d.qfrc_constraint` |
| u ∈ R²² | 控制指令（= 软件 PD 输出，写 `d.ctrl`） | `d.ctrl` |

---

## 3. 动力学方程与残差的物理定义

### 3.1 基本方程与已验证恒等式

MuJoCo Euler 步内成立（主 XML 下 `qfrc_passive ≡ 0` 精确为 0、`qfrc_applied ≡ 0` [RUNTIME-VERIFIED]）：

```
M(q)·a + qfrc_bias(q,v) = qfrc_actuator + qfrc_constraint        … (E)
```

**实证**：随机 (q,v)、ctrl=5 Nm、足底压实（8 contacts）下，`max|M·qacc + qfrc_bias − qfrc_actuator − qfrc_constraint| = 6.6e-10`，M 对称（atol 1e-12）、正定（eigmin≈2.0e-3）。该恒等式同时是采集器的逐样本断言（§6.5）。

### 3.2 质量矩阵展开（决策 D3）

MuJoCo 4.0.1 Python API（注意与旧版文档不同）：

```python
M = np.empty((m.nv, m.nv))        # 28×28, C-contiguous, float64
mujoco.mj_fullM(m, d, M)          # 签名 (model, data, dst)；无 d.qM 属性
```

- `d.M`（内部稀疏缓冲）在 `mj_step1`/`mj_forward` 后有效；`mj_fullM` 展开为对称稠密阵。
- 存储格式：数据集中存**上三角 406 元素 float32**（§6.3），加载时重建对称阵。

### 3.3 armature（决策 D4）

- MuJoCo CRBA 在 `d.M` 中**已把 `dof_armature` 加进对角**（K1 腿关节 0.028–0.096、臂/头 0.001–0.002，[INV] §B.3）。因此 V1 只要使用 `mj_fullM` 输出，armature 自动包含，无手工处理。
- 禁止在 V1 中另用解析 CRBA/RNEA 自算 M/b（若未来替换，必须显式 `M_ii += dof_armature`，否则残差有系统性偏差）。小腿/踝等连杆惯量极小（4e-6）的 dof 上 armature 主导惯量（ankle dof M_ii≈0.06 vs 连杆惯量 4e-6），漏加即数量级错误。

### 3.4 控制量口径（输入 τ 的定义）

- **网络输入 τ ≡ `d.qfrc_actuator[6:28]`**（dof 顺序，Nm）。gear=1 下数值等于 clip 后的 `ctrl`。
- **禁止**把 `dof_target`（PD 目标角）当作力矩；数据集中不存 dof_target 作为模型特征（可存作 episode 元数据供审计）。
- `ctrl` 由软件 PD 每物理步计算并 clip 到该关节 forcerange 后写入（沿用 `virtual_k1/virtual_k1/controllers.py:49-71` 的 `JointPDController`，含 `qfrc_bias` 重力补偿开关，见 §6.2 场景定义）。

### 3.5 qfrc_constraint 的内容与使用边界（决策 D2）

**内容**：主模型中 `qfrc_constraint` = 足底（及其他身体部位）与地面接触的广义约束力 + 22 个关节限位约束力。求解器为 Newton（tol 1e-6, impratio 10, solref [0.001,1], solimp [0.9,0.95,0.001,0.5,2]），是**软约束**解，不是精确互补解。

**训练时使用边界（三条红线）**：
1. `qfrc_constraint` 在训练/验证时**作为已知量直接取自数据集记录值**，只进入损失函数（式 §5.1），**不作为网络输入特征**，**不由网络预测**，也**不在训练时用 mj_forward 重算**——接触求解依赖 warmstart（`qacc_warmstart`）与迭代路径，重算值在接触建立/分离瞬间可能与 rollout 记录值不一致；rollout 记录值是权威值。
2. 与之相对，`M(q)` 与 `qfrc_bias(q,v)` 是 (q,v) 的**路径无关光滑函数**，可存可重算；V1 选择存储（§6.3），训练循环不依赖 MuJoCo。
3. 诚实的性质说明：由于 `a* = M⁻¹(τ_dof + f_c − b)`，在训练数据上 `L_dyn` 是数据误差的 M 加权形式（见 §5.2）；PINN 的价值在于给主干网络注入"按动力学重要性加权"的归纳偏置，以及允许在仿真器内以物理残差口径评价。**纯推理（脱离仿真器）时 f_c 不可得**，V1 不声称推理期物理一致性——这是 soft-constrained 训练，不是推理期硬约束。

---

## 4. 模型定义（MLP 与 PINN 共用主干）

### 4.1 输入 / 输出张量

```
x ∈ R^79 →  Network  → ŷ ∈ R^28（归一化 qacc），反归一化得 a_pred
```

| 输入切片 | 内容 | 维度 | 预处理 |
|---|---|---|---|
| x[0:3] | root position（世界系, m） | 3 | z-score（train split 统计） |
| x[3:7] | root quaternion **wxyz** | 4 | 单位化 + w≥0 半球归一化，**不做 z-score** |
| x[7:29] | 22 关节角（rad, XML hinge 序） | 22 | z-score |
| x[29:57] | qvel（dof 序，m/s · rad/s） | 28 | z-score |
| x[57:79] | τ = qfrc_actuator[6:28]（Nm, dof 序） | 22 | z-score |

输出：28 维 qacc（dof 序）。**输出层预测 z-score 归一化后的 qacc**，物理量 `a_pred = ŷ ⊙ σ_a + μ_a`；所有指标在物理单位下计算。

已知简化（记录在案，不做处理）：绝对 xy 与 yaw 对平面地面动力学是冗余坐标（平面无穷大、摩擦各向同性），V1 保留原始特征让网络自行处理；坐标规范化（xy 相对 episode 起点、yaw 消除）留作 V1.1 消融。接触状态不进输入（用户规定的 79 维输入），接触信息仅通过训练时的 f_c 与标签结构进入——这是 V1 明确接受的建模难度，见 §7.4 分层评价与 §11 风险。

### 4.2 网络结构

| 项 | 值 |
|---|---|
| 结构 | MLP: 79 → 512 → 512 → 512 → 512 → 28 |
| 激活 | SiLU（隐藏层），输出层线性 |
| 参数量 | ≈ 0.9 M |
| 初始化 | PyTorch 默认（kaiming-uniform 线性层）；seed 控制 |
| 规范化 | 无 BatchNorm/LayerNorm（保持 V1 简单；LN 留作消融） |

### 4.3 训练配置（E1/E2 共用，E3/E4 只动 λ 或数据量）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，lr 3e-4，weight decay 1e-5 |
| 调度 | cosine 退火至 3e-5 |
| batch | 4096 |
| 轮数 | max 400 epoch，早停 patience 40（监控 val L_data） |
| 梯度裁剪 | 全局范数 1.0 |
| 精度 | float32 |
| 种子 | {0, 1, 2} 三种子，报告 mean±std |
| 硬件 | 单 GPU（0.9M 参数 CPU 亦可行，不作为瓶颈） |
| 环境 | 项目 `.venv`（mujoco 4.0.1 + torch；MuJoCo 仅在采集与评价期使用） |

---

## 5. 损失函数

### 5.1 定义

```
L_data = MSE(a_pred, a_gt)                    （物理单位；等价于归一化输出的逐 dof 加权 MSE）
r      = M(q)·a_pred + b(q,v) − τ_dof − f_c   ∈ R^28,  τ_dof = qfrc_actuator
L_dyn  = (1/28) Σ_i ( r_i / s_i )²            （逐 dof 归一化力残差 MSE）

MLP:  L = L_data
PINN: L = L_data + λ_dyn · L_dyn
```

### 5.2 残差的等价解释

由恒等式 (E)：`r = M·(a_pred − a_gt)`。即 L_dyn 是以质量矩阵加权的加速度误差（单位 N / Nm）。它的加权与 L_data（归一化输出空间的均匀加权）不同：重肢体 dof（M 对角大）的误差被放大。两个损失互补，这正是引入它的动机，须在结果报告中按此解释。

### 5.3 残差归一化（s_i 的定义）

`s_i = RMS over train split of (M·qacc)_i`（第 i 个 dof 上惯性力的均方根，物理单位 N/Nm），随 `norm_stats.json` 一起产出并固定。理由：(a) 消除 dof 间 N 与 Nm、大腿与手腕的量级差（本模型 dof 力量级跨度约 1e-2…1e2）；(b) 完全数据驱动、无超参。结果同时报告未归一化残差 `mean |r_i|`（物理单位）供审阅。

### 5.4 λ_dyn 选择（决策）

- 默认 **λ_dyn = 1.0**（两损失均已归一化到 O(1) 量级）。
- E3 扫描 {0.1, 0.3, 1, 3, 10}；选择规则：**在 val L_data（任务指标）不劣于 MLP baseline 5% 以内的前提下取最大 λ**（最大化物理正则强度）。同时报告各 λ 的 train/val L_dyn 曲线。
- 监控项（非 V1 机制）：记录每步 `||∇_θ L_data|| / ||∇_θ (λ L_dyn)||`，若持续 >10 或 <0.1 提示 λ 失衡，写入日志供分析（不做自动调节，GradNorm 类方法列为 V1.1）。

---

## 6. 数据集规范 D1（`k1_dynamics_dataset_v1`）

### 6.1 采集引擎

复用 `K1Sim`（`virtual_k1/virtual_k1/simulator.py`），新增薄封装 `k1_dynamics/collector.py`（只新增代码，不改 virtual_k1）。约束：
- 模型 timestep 保持 0.001，`n_substeps=1`；
- 每物理步完整采样（stride=1）；
- 全部字段从 `MjData` 直读，禁止任何重构/差分/平滑。

### 6.2 场景与 Episode 设计

三类场景（episode = 一次完整 rollout，含随机化与元数据）：

**C1 参考 PD 跟踪**（主体数据，分布贴近舞蹈任务）：
- 参考：4 条现有轨迹 `datasets/k1_dance_v1/{top_rock, arm_wave, popping, arm_wave_upper_body_stationary_root}/trajectory.csv`（50 Hz，仅作 PD 目标源）。
- 控制器：`JointPDController`（V1 增益 `k1_sim.yaml:81-105`，gravity_compensation=True），每个 1 ms 物理步重算力矩；50 Hz 参考帧间目标角**线性插值**到当前时刻。
- episode 结构（沿用 `virtual_k1_replay.py` 的成熟模式）：reset 到参考帧 0 → hold 0.5 s → ramp 0.5 s → 跟踪全轨迹 → hold 0.4 s。
- 随机化（每 episode）：初始关节角 N(0, 0.02 rad)，root xy N(0, 0.02 m)、z +N(0, 0.005 m)，root quat 小角扰动 N(0, 2°)；PD 增益逐关节 ×U(0.8, 1.2)。

**C2 扰动跟踪**（提高激励与接触事件覆盖）：
- C1 基础上追加：力矩噪声 σ = 3% 该关节 forcerange（高斯，clip 回 forcerange）；随机时刻速度踢（在两物理步之间直接 `qvel += Δv`，Δv ~ N(0, diag(root lin 0.15 m/s, root ang 0.3 rad/s, joints 0.3 rad/s))）。
- 用速度踢而非 `qfrc_applied` 力脉冲，保证 `qfrc_applied ≡ 0`、残差方程 (E) 不引入新项（qfrc_applied 字段仍记录并断言为 0，§6.5）。

**C3 随机力矩探索**（覆盖远离参考的状态与摔倒前接触）：
- 一阶 OU 过程逐关节生成带限力矩：`u_i(k+1) = ρ u_i(k) + σ_i √(1−ρ²) ε`，ρ=0.99（≈16 Hz 带宽），σ_i = 0.3×forcerange_i，clip ±0.5×forcerange_i；root 无控制。
- episode 上限 6 s；跌倒判据（base_z < 0.42 m 或 |roll|,|pitch| > 45°，与 `virtual_k1_replay.py:28`、recorder 一致）触发终止；终止前的帧全部保留（含冲击接触），终止后不采样。

规模预算：

| 场景 | episode 数 | 估计样本 |
|---|---|---|
| C1（4 动作 × 12） | 48 | ≈ 360k |
| C2 | 16 | ≈ 120k |
| C3（含提前终止） | 16 | ≈ 60–100k |
| **合计** | ~80 | **≈ 0.55–0.6 M**（≈ 3.1 kB/样本 ⇒ **≈ 1.7–1.9 GB**） |

采集耗时估计：MuJoCo 1 kHz 单线程 ≫ 实时（约 20–50× 含 Python 接触遍历开销），全程 < 30 min。试点集（pipeline 验证）：每场景 2 episode ≈ 50k 样本。

### 6.3 字段表（每 episode 一个 NPZ，定长字段 + 固定槽位接触数组）

| 字段 | 形状 | dtype | 单位 | 来源 / 说明 |
|---|---|---|---|---|
| `t` | (T,) | f64 | s | `d.time`（step2 之前读，即本行动力学求解时刻 t_k） |
| `qpos` | (T,29) | f32 | m/rad | `d.qpos`（root quat wxyz 原样） |
| `qvel` | (T,28) | f32 | m/s, rad/s | `d.qvel` |
| `qacc` | (T,28) | f32 | m/s², rad/s² | `d.qacc` |
| `ctrl` | (T,22) | f32 | Nm | `d.ctrl`（写入值，= PD 输出 clip 后） |
| `actuator_force` | (T,22) | f32 | Nm | `d.actuator_force`（=clip(ctrl)，验证用） |
| `qfrc_actuator` | (T,28) | f32 | N/Nm | `d.qfrc_actuator`（root 6 维应为 0） |
| `qfrc_bias` | (T,28) | f32 | N/Nm | `d.qfrc_bias` |
| `qfrc_constraint` | (T,28) | f32 | N/Nm | `d.qfrc_constraint` |
| `qfrc_passive` | (T,28) | f32 | N/Nm | 应恒 0（审计字段） |
| `qfrc_applied` | (T,28) | f32 | N/Nm | 应恒 0（审计字段，V1 无外力输入） |
| `M_upper` | (T,406) | f32 | kg, kg·m² | `mj_fullM` 结果上三角（行优先 i≤j），加载端重建对称阵 |
| `ncon` | (T,) | i32 | — | 总接触数 |
| `ncon_nonfoot` | (T,) | i32 | — | 非足底接触数（躯干/膝等触地诊断） |
| `foot_in_contact_{l,r}` | (T,) | bool | — | 法向力合计 ≥ 0.5 N（沿用 `k1_sim.yaml` 阈值） |
| `foot_normal_force_{l,r}` | (T,) | f32 | N | 该足接触点法向力合计 |
| `foot_contact_pos_{l,r}` | (T,4,3) | f32 | m | 世界系，NaN 填充空槽 |
| `foot_contact_normal_{l,r}` | (T,4,3) | f32 | — | 地面→足方向（世界系），NaN 填充 |
| `foot_contact_wrench_{l,r}` | (T,4,6) | f32 | N, Nm | `mj_contactForce` 接触坐标系 6D wrench，NaN 填充 |

不采集：`dof_target`（不作特征，见 §3.4）、渲染帧、sensordata（V1 闲置，[INV] §H Q9）。

Episode 元数据写入 `index.json`：`{episode_id, scenario, motion, seed, n_steps, ended_by: {none|fall}, contact_stats, fall_step?, sha256}`。

### 6.4 状态-控制-加速度时间对齐（逐物理步协议）

```python
# 第 k 步（t_k = k * 0.001）：
u_k = pd.compute(q_k, v_k, ref_interp(t_k))      # 软件 PD
d.ctrl[:] = clip(u_k, forcerange)                 # 输入②：控制
mujoco.mj_step1(m, d)                             # 在 (q_k, v_k, u_k) 上求解前向动力学
snapshot(...)                                     # 采样本行：q_k, v_k, u_k, qacc, qfrc_*, M, contacts
mujoco.mj_step2(m, d)                             # 积分 → (q_{k+1}, v_{k+1})，不属于本行
```

| 量 | 语义 |
|---|---|
| q_k, v_k, u_k | 第 k 行**输入**（积分前状态 + 本步控制） |
| qacc, qfrc_bias/actuator/constraint, M, contact* | 第 k 行**标签与物理量**，由 mj_step1 在该输入上求出 |
| (q_{k+1}, v_{k+1}) | 仅作下一行输入，不重复记录为输出 |

合法性说明：主模型 integrator=Euler [RUNTIME-VERIFIED]，`mj_step ≡ mj_step1 + mj_step2`（MuJoCo 文档对 Euler 成立；若未来换 implicit 积分器此协议需复核）。禁止的替代写法：先 `mj_step` 再读 qacc（qpos/qvel 已前进，字段与状态错位一拍）；直接写 `qfrc_actuator`（会被重算覆盖，[INV] §H Q7 已实证）。

### 6.5 采集器不变量（逐样本断言，W1 验收标准）

| # | 断言 | 容差 |
|---|---|---|
| I1 | `max|M·qacc + qfrc_bias − qfrc_actuator − qfrc_constraint| < 1e-6`（M 从本行 M_upper 重建） | 实测 6.6e-10 |
| I2 | `max|qfrc_passive| == 0` 且 `max|qfrc_applied| == 0` | 精确 |
| I3 | `max|actuator_force − ctrl| < 1e-9` 且 `max|qfrc_actuator[6:] − ctrl| < 1e-9`（gear=1） | float 精度 |
| I4 | M 对称（重建后 atol 1e-6）、eigmin > 0 | |
| I5 | 全部字段有限；NaN 只允许出现在接触空槽 | |
| I6 | `|‖quat‖ − 1| < 1e-6` | |
| I7 | `ncon == 足底计入数 + ncon_nonfoot`（接触簿记闭合） | |
| I8 | episode 元数据含 seed 与 ended_by；跌倒后无样本行 | |

违反任一断言即抛错终止采集（不静默跳过），单测 `tests/test_collector_invariants.py` 以 3 个短 episode 覆盖全部断言。

### 6.6 数据切分（防泄漏）

- **按 episode 切分**（绝不在 episode 内切）：train 70% / val 15% / test 15%，按 (scenario × motion) 分层抽样，保证三种场景与四条动作在三个 split 中均出现。
- 相邻帧泄漏控制：同一 episode 的帧只进同一 split；`index.json` 记录 split 归属；切分脚本固定 seed 并产出 `splits.json`（含各 split episode 清单与帧数）。
- **跨动作泛化集（额外）**：train/val 取 3 条动作 + C2/C3，test 仅 top_rock 的 C1 episode——评价模型对未见动作的泛化（top_rock 是研究目标动作，且是唯一 free-root 大位移轨迹）。
- 归一化统计（μ/σ、s_i）**只从 train split 计算**，一次性产出 `norm_stats.json`（含逐特征 mean/std 与 28 维 s_i），E3/E4 全部实验复用同一份，消除统计口径漂移。

---

## 7. 评价协议

### 7.1 误差指标（全部物理单位，test split 上报告）

| 指标 | 定义 |
|---|---|
| qacc RMSE（总） | sqrt(mean over 样本×28 dof (a_pred − a_gt)²) |
| 分组 qacc RMSE | 按 §2.1 分组（root_lin / root_ang / head / 左右臂 / 左右腿）分别报告 |
| root 平移 / 角加速度误差 | 即 dof 0:3（m/s²）与 dof 3:6（rad/s²）的 RMSE |
| 动力学残差 | mean_i |r_i| / s_i 与 mean |r_i|（物理单位），r 按 §5.1 |
| 参照基线 | (a) 训练集均值预测器；(b) 持续加速度预测器（ŷ_k = a_{k−1}）。所有模型误差必须相对这两个基线报告，避免裸 RMSE 无参照 |

### 7.2 接触状态分层评价

- 按 `(foot_in_contact_l, foot_in_contact_r)` 四象限分层报告 qacc RMSE 与残差；
- **近冲击子集**：本行与前/后 5 ms 内足底接触数发生变化的样本，单独报告（预期最难，见 §9 D5）。

### 7.3 短时程积分（rollout）测试

协议：取 test episode 的初始 (q_0, v_0) 与**记录的 ctrl 序列**；逐步 `a_pred = model(q_k, v_k, ctrl_k)`，以与采集一致的积分方式推进：
- `q_{k+1}`：`mujoco.mj_integratePos(m, qpos, qvel, dt)`（四元数正确积分，禁止手工欧拉 quat 更新）；
- `v_{k+1} = v_k + dt · a_pred`。
视程 0.1 s 与 0.5 s（100/500 步）。指标：关节角 RMSE（rad）、root 位置误差（m）、root quat 测地角（°）、base 高度最小值、跌倒判定一致率（与真实 rollout 对比）。同时以"真实 a 序列"积分作对照上限（数值积分误差）。

### 7.4 对照实验矩阵

| 实验 | 内容 | 控制变量 |
|---|---|---|
| E1 | MLP baseline（L_data only）+ 两个参照基线 | 固定一切 |
| E2 | PINN（λ_dyn=1） | 与 E1 同种子同切分，仅加 L_dyn |
| E3 | λ_dyn ∈ {0.1, 0.3, 1, 3, 10} | 其余同 E2；选 λ 规则见 §5.4 |
| E4 | 训练数据规模 {10%, 25%, 50%, 100%}（按 train episode 均匀子采样，保持场景配比） | MLP 与最优 PINN 各跑一遍；norm stats 固定用 100% train 的版本 |

每组 3 种子，报告 mean±std；统一产出 `runs/k1_dynamics/<exp_id>/` 下 `config.yaml / metrics.json / per_group_metrics.json / rollout_metrics.json`。

---

## 8. 关键决策记录（对应任务书 §7 的七项）

| # | 问题 | V1 裁定 | 章节 |
|---|---|---|---|
| D1 | qvel/qacc 坐标系 | root 线速度世界系、角速度 root 局部系（均实测确认）；qacc 为 dof 空间导数，与 qvel 同布局；禁止 qpos 差分 | §2.1 |
| D2 | qfrc_constraint 来源与用法 | = 接触 + 关节限位的软约束广义力；rollout 记录值为权威，训练时作已知量入损失；不作输入、不预测、不重算 | §3.5 |
| D3 | M 的展开 | `mujoco.mj_fullM(m, d, M)`（4.0.1 签名，无 d.qM）；数据集存上三角 406 f32 | §3.2, §6.3 |
| D4 | armature | 已含于 mj_fullM 输出（CRBA 加对角）；禁用无 armature 的解析 M | §3.3 |
| D5 | 接触突变与残差连续性 | 不平滑标签、不删冲击帧；靠 C2/C3 提高冲击覆盖；分层评价暴露该难度；接触标志输入留作 V1.1 消融 | §6.2, §7.2, §11 |
| D6 | 1 kHz 数据 vs 50 Hz 参考 | 参考仅线性插值生成 1 kHz PD 目标；数据集与模型均为 1 kHz 物理步；参考轨迹本身不进训练数据 | §6.2, §0.2 |
| D7 | 物理栈统一 | 统一 virtual_k1 MuJoCo（0.001 s、官方 XML、无 passive 力）；结论不外推 IsaacLab/真机 | §1 |

---

## 9. 新增代码与产物布局（全部新增，不改现有文件）

```
k1_dynamics/                     # 新顶层包（.venv 解释器）
├── __init__.py
├── collector.py                 # K1DynamicsCollector: §6.1/6.4 采样协议 + §6.5 断言
├── scenarios.py                 # C1/C2/C3 episode 生成（随机化表 §6.2）
├── dataset.py                   # NPZ 加载、M_upper 重建、切分加载、z-score/半球归一化
├── model.py                     # MLP（§4.2）
├── residual.py                  # r 与 L_dyn（§5），s_i 加载
├── train.py                     # E1/E2/E3/E4 入口（config 驱动）
├── evaluate.py                  # §7 全部指标 + rollout 测试
├── configs/                     # dataset.yaml / mlp.yaml / pinn.yaml / e3_sweep.yaml / e4_scale.yaml
└── tests/test_collector_invariants.py
datasets/k1_dynamics_v1/         # 新数据集目录（与 k1_dance_v1 并列，不动旧数据）
├── episodes/*.npz  index.json  splits.json  norm_stats.json
runs/k1_dynamics/                # 训练产物与指标
```

依赖方式：`import virtual_k1`（复用 `k1_action/virtual_k1_replay.py:20-23` 的 sys.path 注入模式，或以 `-m` 从仓库根运行）。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 接触冲击使 qacc 标签不连续，MLP 上限受限 | E1/E2 误差集中在近冲击帧 | C2/C3 增加冲击覆盖；§7.2 分层报告使结论诚实；V1.1 可加接触标志输入 |
| top_rock 无已训练策略 ⇒ C1 数据是 PD 跟踪分布，非策略分布 | 泛化到未来 RL 策略 rollout 未知 | 跨动作泛化集显式测；C3 随机力矩拓宽覆盖；结论限定分布内 |
| M 条件数大（eigmin≈2e-3）⇒ 力残差被小惯量 dof 主导 | L_dyn 不稳定 | s_i 归一化正是为此；监控 §5.4 梯度比 |
| 数据量/存储超预算（>2 GB） | 磁盘与 IO | M 上三角 f32 已压；必要时 C1 降到 8 eps/动作（仍 >0.4M 样本） |
| 训练循环误引入 MuJoCo 依赖 | 复杂度 | M/b/f_c 全部离线存储，训练仅用 numpy/torch |
| mj_step1/2 协议被后续维护者改成 mj_step 后读数 | 数据错位一拍 | I1 恒等式断言会立即失败（错位时不再闭合） |

---

## 11. 验收清单（W1–W4 逐项打勾）

- [ ] 采集器 8 条不变量在试点集 100% 通过（I1–I8）
- [ ] 试点 50k 样本 + splits.json + norm_stats.json 产出
- [ ] E1 MLP 与两参照基线在 test 上报告完整指标（§7.1/7.2）
- [ ] E2 PINN(λ=1) 同口径完成；train/val L_dyn 曲线存档
- [ ] E3 λ 扫描完成并按 §5.4 规则选出 λ*
- [ ] E4 数据规模曲线（MLP 与 PINN×λ*）完成
- [ ] 跨动作（train 3 动作 → test top_rock）结果单独成表
- [ ] rollout 0.1/0.5 s 积分测试完成（含真实加速度序列对照上限）
- [ ] `K1_DYNAMICS_MODEL_V1_RESULTS.md` 汇总全部表格，失败/异常如实报告

---

*本规范到此为止；不包含任何已实现的采集/训练代码，现有 MJCF、训练配置与动作资产零改动。*
