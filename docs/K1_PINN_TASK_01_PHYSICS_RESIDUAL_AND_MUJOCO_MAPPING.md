# K1_PINN_TASK_01 — 物理残差的数学定义与 MuJoCo 实现映射（技术报告）

- 日期: 2026-11-23
- 模型: `virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml`（未修改）
- 环境: 项目 `.venv`，MuJoCo **4.0.1**（Python binding 实测行为为最高依据）
- 验证程序: `tools/pinn/verify_k1_pinn_physics_residual.py` → **49/49 PASS**，摘要 `tools/pinn/verify_summary.json`
- 结论级别标注：**[V]** = 本机运行验证过；**[D]** = 文档/设计裁定。

---

## 1. 任务结论与推荐方案

**总判定：PASS（49/49 检查通过）。**

1. K1 广义动力学方程与 MuJoCo 4.0.1 实际计算**逐项一致**，完整残差
   `r = M·a + qfrc_bias − qfrc_actuator − qfrc_passive − qfrc_applied − qfrc_constraint (+ xfrc 项)`
   在 MuJoCo 自身 qacc 上为 **0（≤1.5e-9）** [V]。
2. **关键发现（退化问题）**：在训练样本上（各力项均取自同一 forward 求解步），该残差**精确等于** `M(q)·(â − a_mujoco)`（数值验证 1.45e-9）[V]。因此把它叫"物理约束损失"是误导——它只是**质量矩阵加权的数据加速度误差**。本文建议命名为 **dynamics consistency loss**（§8）。
3. 第一版 PINN 采用 **Free-Dynamics PINN**（§9）：仅在无活动约束样本上施加**不包含 qfrc_constraint** 的残差 `r_free = M·â + qfrc_bias − qfrc_actuator`（本模型 passive=applied=xfrc=0），这是真正独立于求解器输出的物理约束 [V：air 状态 r_free(qacc)=4.5e-15]。
4. 有接触样本保留数据监督，做接触分层评价；第一版不建立接触物理残差，不假装学到了接触动力学（§10）。

---

## 2. K1 自由度与广义坐标定义 [V]

| 量 | 值 | 说明 |
|---|---|---|
| nq / nv / nu / na | **29 / 28 / 22 / 0** | na=0：22 个 motor 无 activation 状态，`data.act` 形状 (0,) |
| 关节 | 1 free joint（`world_joint`，trunk）+ 22 hinge | 全部 hinge 有限位（autolimits+range） |
| 约束源 | 接触（任意 geom 对，含自碰）+ 关节限位 | neq=0、ntendon=0、npair=0；无 equality/tendon/frictionloss |
| q 布局 | qpos[0:3]=root 位置（世界系，m），qpos[3:7]=四元数 **wxyz**，qpos[7:29]=22 关节角（rad，XML hinge 序） | |
| v 布局 | qvel[0:3]=root **线速度（世界系分量）**，qvel[3:6]=root **角速度（root 局部系分量）**，qvel[6:28]=关节角速度 | 线速度世界/角速度局部的约定已实测（非 yaw 旋转下与 `mj_objectVelocity(flg_local=0)` 对比）[V] |
| a 布局 | qacc 是 **qvel 各分量的时间导数**（dof 空间），与 v 同坐标系布局；root 角加速度（dof 3:6）是局部系角速度分量的导数（含 ω×Ω 项），**不是**世界系角加速度 | 评价口径即按 dof 3:6 报告 |

**单位差异（28 维残差混合两种单位）**：dof 0:2（root 平移）的广义力单位是 **N**；dof 3:27（root 旋转 + 22 关节）是 **N·m**。量级也悬殊：M 对角从 ~2e-3（小惯量 dof，armature 主导）到 19.67（root 平动=总质量）[V]。这是残差必须逐 dof 归一化的根本原因（§11）。

---

## 3. 完整动力学方程（与 MuJoCo 4.0.1 一致）[V]

MuJoCo 的每步前向动力学满足：

```
M(q)·a = qfrc_smooth + qfrc_constraint
qfrc_smooth = qfrc_passive + qfrc_actuator + qfrc_applied + J_com^T·xfrc_applied(世界系) − qfrc_bias
```

移项得到本研究的**完整残差定义**：

```
r_dyn(â) = M(q)·â + qfrc_bias − qfrc_actuator − qfrc_passive − qfrc_applied − f_xfrc − qfrc_constraint
f_xfrc := Σ_body jacBodyCom(body)^T · xfrc_applied[body][0:3]  (+ jr^T·torque)
```

当前模型与采集条件下：`qfrc_passive ≡ 0`、`qfrc_applied ≡ 0`、`xfrc_applied ≡ 0`（三者均逐状态断言为精确 0 [V]），方程简化为

```
r_dyn(â) = M·â + qfrc_bias − qfrc_actuator − qfrc_constraint
```

各项维度均为 R²⁸；`qfrc_bias = 重力 + 科氏/离心（C(q,v)v + g(q)）`；`qfrc_actuator` 的 root 6 维恒为 0（无 root 执行器）[V]。

**验证数值**：air / contact(8 触点, ‖qfrc_constraint‖∞=21286 N) / joint_limit(限位力 29.63 N·m) 三种状态下 `r_dyn(a_mujoco)` 的 ∞ 范数分别为 4.5e-15 / 1.45e-9 / 1.49e-13 [V]。

---

## 4. 数学项 ↔ MuJoCo 4.0.1 字段映射 [V]

| 数学项 | MuJoCo 字段 / API | 维度 | 实测备注 |
|---|---|---:|---|
| q | `data.qpos` | 29 | quat wxyz |
| v | `data.qvel` | 28 | §2 坐标系 |
| a | `data.qacc` | 28 | mj_forward/mj_step 求解后的本步加速度（§12 时序） |
| u | `data.ctrl` | 22 | 本模型语义=力矩指令（gear=1 motor） |
| activation | `data.act` | **0** | na=0；字段存在但为空，不采集 |
| actuator force | `data.actuator_force` | 22 | =clip(ctrl, forcerange)，与 `qfrc_actuator[6:28]` 逐元素相等 [V] |
| τ_act | `data.qfrc_actuator` | 28 | root 6 维为 0 |
| c(q,v) | `data.qfrc_bias` | 28 | |
| τ_passive | `data.qfrc_passive` | 28 | 恒 0（无 damping/spring/frictionloss）[V] |
| τ_applied | `data.qfrc_applied` | 28 | 用户广义外力；恒 0（采集条件）[V] |
| τ_constraint | `data.qfrc_constraint` | 28 | 接触 + 关节限位的**软约束**广义力（≠纯足底接触力，见下） |
| 外部体坐标系 wrench | `data.xfrc_applied` | 26×6 | 世界系 wrench 作用于各 body **质心**（`mj_jacBodyCom` 转置精确匹配 1.4e-17；作用于 frame 原点的假设被排除，误差 0.043）[V]；不进入 `qfrc_applied` 字段 |
| 接触列表/计数 | `data.contact[i]`, `data.ncon` | 可变 | `contact.frame`(9)=(n,t1,t2) 行向量、`pos`、`dist`、`dim`、`geom1/2`、`efc_address` |
| 约束力 | `data.efc_force` | nefc | 金字塔面乘子（本模型 ≥0 [V]） |
| 约束雅可比 | `data.efc_J` | **扁平 (nefc·nv,)** | 需 reshape(nefc, nv)；不是现成二维矩阵 [V] |
| 逆向动力学 | `mj_inverse` → `data.qfrc_inverse` | 28 | = M·a + bias − passive − applied − constraint（逆向重算），与 forward 的 `qfrc_actuator` 差 ~1.2e-9 [V]；**副作用：调用后覆写 `data.qfrc_constraint`** [V] |

**`qfrc_constraint` 的组成（不得简称足底接触力）**：在本模型 = (a) 任意 geom 对的接触广义力（足底 4 角点盒、躯干/膝/肘/头等非足底接触、自碰）+ (b) 22 个关节限位约束力。不存在 equality/tendon/frictionloss 项（neq=ntendon=0 [V]）。实证：构造 knee=−0.02 rad（越下限）状态，ncon=0 但 nefc=1、限位广义力 29.63 N·m 精确落在 knee dof 上、其 efc 行 = 该 dof 的单位行向量（误差 0）[V]。

**`xfrc_applied` 的处理**：它是 26×6 的用户外力接口，以 `jacBodyCom^T`（世界系、质心）进入 `qfrc_smooth`，**不体现在 `qfrc_applied` 里**。本研究的采集器将其保持恒零，并作为实验条件写入数据集元数据（`xfrc_applied_zero: true`）[D，验证程序演示了若非零时的完整方程闭合]。

---

## 5. Armature 处理 [V]

- MuJoCo 的 CRBA 把 `model.dof_armature` 直接加进 `d.M`（内部稀疏质量矩阵）的对角，因此 **`mj_fullM` 的输出已经包含 armature**。
- **不需要、也绝不能**在残差里再加 armature 项。数值证明：把 `diag(dof_armature)` 从 M 中减去后，`r(M−diag(arm), a_mujoco)` 不再为 0，而精确等于 `−diag(arm)·a_mujoco`（验证误差 1.45e-9，扰动幅度 ‖·‖∞=166）[V]。K1 的 armature 不容忽略：ankle dof 连杆惯量 ~4e-6 而 armature=0.0565，漏加即数量级错误。

---

## 6. 质量矩阵获取方法 [V]

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml")
d = mujoco.MjData(m)
# ... 设置 qpos/qvel/ctrl 后调用 mj_forward（或 mj_step，见 §12）...
M = np.zeros((m.nv, m.nv), dtype=np.float64)   # C 连续 float64
mujoco.mj_fullM(m, d, M)                        # 4.0.1 签名: (model, data, dst)
```

- **`d.M` 不是普通稠密矩阵**：它是内部稀疏（CSC 类）缓冲；4.0.1 Python binding 没有 `d.qM`（旧文档名），直接当 28×28 用是错的。唯一正确入口是 `mj_fullM(m, d, dst)`。
- 验证项与数值 [V]：对称（误差恰为 0）、正定（eigmin=1.97e-3, eigmax=19.93）、有限；root 平移块对角均值 = 总质量 19.666 kg（结构自检）。
- **PyTorch 用法**：M 是 (q) 的函数、与 â 无关 ⇒ `M·â` 对 â 是线性算子。V1 把每样本 M 的上三角 406 元素存入数据集，训练时 `torch.einsum("bij,bj->bi", M_batch, a_batch)`；无需对 M 求梯度。
- **`mj_mulM` 可替代稠密展开**：`mj_mulM(m, d, res, vec)` 给出 `M@vec`，与稠密结果差 7.1e-15 [V]。适合只做前向残差的在线评价；但 PINN 训练需要 M 参与 autograd 的线性组合，存稠密（或上三角）更直接 [D]。

---

## 7. 接触力、约束力与 Jacobian 的区别 [V]

三层量，**不可互换**：

| 层 | 量 | 内容 | 关系 |
|---|---|---|---|
| 接触层 | `mj_contactForce(m, d, i, wrench6)` | 第 i 个接触的 6 维 wrench，**接触坐标系**（前 3 = 法向 + 2 切向；condim=3 时后 3 维无效为 0 [V]） | 由该接触的 efc 金字塔乘子换算回 (fn, ft1, ft2) |
| 约束层 | `efc_force`（nefc 维）+ `efc_J`（nefc×nv） | 全部约束（接触面 + 关节限位等）的乘子与雅可比 | `efc_J^T @ efc_force == qfrc_constraint`，**精确为 0**（dense 1.1e-13，`mj_mulJacTVec` 恰为 0）[V] |
| 广义层 | `qfrc_constraint`（28 维） | 约束对广义坐标的合力 | 进入动力学方程 (§3) |

- **efc 行布局（重要版本行为）**：condim=3 的接触在本版本占 **4 行**（`efc_address` stride=4 [V]），行方向为金字塔面方向 n±t1、n±t2，乘子非负（实测 min=377 [V]）；`opt.cone=0`。**不要用 `3×ncon` 之类的公式推 nefc**，必须读 `efc_address`/`efc_id`。
- **版本差异记录（如实登记，不选边）**：`efc_type` 数值标签（接触行=6、关节限位行=3）与 MuJoCo 4.0 之前公开文档的经典枚举（pyramidal=5、limit_joint=2）不一致，两处观测均 +1，与枚举插入新类型一致，但在当前环境无法从文档确证。**结论：禁止依赖 efc_type 的数值标签做语义判断；用结构特征（J 行 = 关节 dof 单位行 ⇒ 限位行；属于某 contact 的 efc_address 区间 ⇒ 接触行）识别** [V]。
- **接触力 → 世界系**：`F_world = frame_matrix^T @ wrench[0:3]`，frame 行 = (n, t1, t2)（世界系方向；本模型 n=+z 从 geom1 指向 geom2 [V]）。`mj_contactForce` 的 wrench 作用在 **geom2 所属 body** 上（符号 +1 实证）[V]。
- **多接触聚合**：Σᵢ jacp(con.posᵢ, foot_body)ᵀ·F_worldᵢ == 接触部分 qfrc_constraint（无活动限位状态下即全量，误差 1.2e-13）[V]；法向合力 Σframe_row0·fn = 21286.33 N 与 `qfrc_constraint[2]` 精确相等 [V]。
- **V1 是否直接操作 Jacobian**：不需要。残差只用 `qfrc_constraint`；Jacobian（`mj_mulJacVec`/`mj_mulJacTVec`，前向 J·v / 转置 Jᵀ·f）仅在（a）验证上述恒等式、（b）未来接触力监督 ablation 中使用 [D]。

**正逆动力学时序**：`mj_forward(q,v,u)` → 算出 qacc/qfrc_*（不推进时间 [V]）；`mj_step` = forward + Euler 积分（推进 0.001 s [V]）；`mj_inverse(q,v,a)` → `qfrc_inverse`（= M·a + bias − passive − applied − constraint(逆算)，≈ forward 的 `qfrc_actuator`，差为正/逆约束求解的数值差 ~1e-9 [V]）。差异来源：约束力在 inverse 中由给定 qacc 经 `mj_invConstraint` 重算（与 forward 解差 ~2e-10），且 **调用后 `data.qfrc_constraint` 被覆写**——凡后续要读 forward 值必须先快照 [V]。

---

## 8. solver-consistency residual 的退化分析（核心问题）

**数学推导**：训练样本的 M, bias, actuator, passive, applied, constraint 全部取自**同一个 forward 求解步**，满足恒等式 `M·a* + bias − act − passive − applied − constraint = 0`（a*=qacc，残差 ≤1.5e-9 [V]）。于是对任意预测 â：

```
r_dyn(â) = M·â + (bias − act − passive − applied − constraint)
         = M·â − M·a*        （用恒等式替换括号）
         = M·(â − a*)
```

**数值验证** [V]：构造 â = a* + δ（δ~N(0,1)²⁸），`‖r(â) − M·δ‖∞ = 1.45e-09`（air 2.7e-14、joint_limit 1.4e-13）——精确线性关系。

**五个问题的回答**：

1. **是否只是 M 加权的数据加速度误差？** 是。在该使用方式下 `λ·L_dyn` 与 `L_data` 的差别仅是加权矩阵（M² vs 单位阵）与归一化尺度，不引入新信息。
2. **是否含独立于数据标签的物理信息？** 该形式不含（M 是 q 的确定函数，无标签独立性可言）。**但 Free-Dynamics 形式（去掉 constraint 项）含**：M(q)、bias(q,v) 只依赖状态，残差对任何 â 都给出不依赖标签的物理约束——这是 V1 的推荐方案。
3. **用了 solver 输出的 qfrc_constraint 是否等于学会了接触动力学？** 否。网络从未预测接触力；constraint 项是外生已知量。网络学到的仍是 a 的映射，接触物理只通过标签结构与损失加权间接体现。
4. **该残差能用于什么 / 不能用于什么？** 能：动力学一致性正则（按动力学重要性加权的归纳偏置）、数据审计（恒等式破坏 ⇒ 采集协议错误）、课程设计。不能：宣称接触物理建模、推理期（无仿真器时）物理一致性、对 MuJoCo 之外的物理栈的泛化主张。
5. **命名**：应命名为 **dynamics consistency loss**（solver-consistency），不得称独立 contact physics loss。

**四种概念的区分表**（后续文档与代码命名统一遵守）：

| 概念 | 定义 | 是否独立物理约束 | V1 用途 |
|---|---|---|---|
| 独立物理约束 | `r_free = M·â + bias − act`（无 solver 输出项），仅在无活动约束样本 | **是** | PINN 物理损失 |
| MuJoCo solver consistency | `r_dyn`（含记录的 qfrc_constraint） | 否（= M(â−a*)） | 审计/诊断/正则（可作对照） |
| 数据监督损失 | `L_data = MSE(â, a*)` | — | 主损失 |
| 接触力监督损失 | 预测 qfrc_constraint 或接触 wrench，以 solver 输出为标签 | 否（蒸馏 solver） | 可选 ablation，必须标注 solver-force-supervised |

---

## 9. 第一版 PINN：Free-Dynamics PINN

**残差**（本模型 passive=applied=xfrc=0 已断言，公式保留通式以便审计字段非零时立即暴露）：

```
r_free(â) = M(q)·â + qfrc_bias(q,v) − qfrc_actuator      ∈ R^28
```

**mask 规则（生成）** [V：air=True，contact/joint_limit=False]：

```
free_dynamics_mask = (ncon == 0) AND (nefc == 0) AND (||qfrc_constraint||_inf == 0)
```

- 三条件在验证状态中完全一致地同时成立/不成立；主判据用 nefc==0（涵盖接触、限位与任何未来 equality 行），另两条为冗余保险 [D]。
- **阈值问题不存在**：判据是精确零比较，无超参 ⇒ 无需用任何数据"选阈值"，天然无泄漏。若未来需要软判据（如 ‖qfrc_constraint‖<ε），ε 只能在 train split 上定并冻结 [D]。
- **mask 只用于训练样本选择**（L_free 的求和集合 F）；**不作为网络推理输入**（V1 输入保持 79 维状态+力矩）；mask 与 `qfrc_constraint_norm` 逐样本存入数据集 [D]。
- **test 数据不参与 mask 统计或任何阈值确定**；test 只按同样确定性规则打标后用于分层评价 [D]。

**验证** [V]：air 状态 `r_free(a*)` = 4.5e-15；接触/限位状态 mask=False 被正确排除。

---

## 10. 接触状态的第一版处理原则

- 有接触（或限位等活动约束）样本：**只保留 L_data 监督**；完整保存 qfrc_constraint、接触点位置/法向/wrench、free_dynamics_mask，用于**接触状态分层评价**（单/双足、近冲击帧）与后续研究。
- 明确声明：V1 未建立接触物理残差；`qfrc_constraint` 作为已知量的边界见 §8/§4。
- 可选 ablation（不在本任务实现）：**solver-force-supervised** —— 网络加一头预测 qfrc_constraint（或逐接触 wrench），以 solver 输出为监督。报告与代码中必须用该名称，不得表述为独立物理约束。

---

## 11. Loss 归一化与权重方案

```
L_data = (1/28) Σ_i ((â_i − a_i)/s_{a,i})²                      （全样本）
L_free = (1/(28·N_f)) Σ_{n∈F} Σ_i (r_{n,i}/s_{r,i})²            （仅 free 样本）
L_PINN = L_data + λ_dyn · L_free
```

1. **为什么不能裸 MSE**：残差 28 维混合 N（dof 0:2）与 N·m（其余），且 M 对角跨 4 个数量级（2e-3…19.7）、实测加速度幅值可达 ~1.6e3 rad/s²（air+2Nm 状态 [V]）；未归一化时残差被 root 平动 dof 支配，小惯量 dof 的物理错误被淹没。
2. **尺度参数估计** [D]：`s_{a,i} = std(a_i)`（train split，全样本）；`s_{r,i} = RMS_{n∈F_train}((M·a)_{n,i})`（train split 的 free 样本；与 L_free 的作用域一致）。二者随 `norm_stats.json` 一次性产出冻结；**禁止用 test episode 估计**。
3. **N_f 不足的处理** [D]：先报告 N_f 与占比；若 free 样本 <5% 或 <10k，(a) 优先扩充含腾空相的随机激励 episode，(b) 过渡期将 λ_dyn 减半并在结果中标注，不得静默丢弃该损失。
4. **λ_dyn 候选**：{0.1, 1, 10}。
5. **选择规则** [D]：以 **validation episode 的 L_data**（任务指标）为准，在不劣于 MLP baseline 5% 的范围内取最大 λ；同时报告 val L_free。
6. **数值监控** [D]：每 epoch 记录 train 的 L_data、λ·L_free 数值量级与各自对主干参数的梯度范数比 `‖∇(λ L_free)‖/‖∇L_data‖`；持续 >10 或 <0.1 时人工复核 λ（不做自动调节）。两条 loss 均为 O(1) 量级归一化后，默认 λ=1。
7. **防支配**：除归一化与上述监控外，训练日志保留两 loss 分开的曲线；若 L_data 收敛明显劣化即回退 λ。

---

## 12. 数据采集时序（pre-step 协议，已验证）[V]

**推荐协议 P-A（任务书建议的流程，验证正确）**：

```python
data.ctrl[:] = u_t                        # ① 本步控制
mujoco.mj_forward(model, data)            # ② 在 (q_t, v_t, u_t) 上求前向动力学（不推进时间）
qpos_t  = data.qpos.copy(); qvel_t  = data.qvel.copy()
qacc_t  = data.qacc.copy()                # ③ 快照：状态、加速度、
M_t     = full_M(); forces = ...copy()    #    M、各广义力、接触 —— 全部对应 (q_t, v_t, u_t)
mujoco.mj_step(model, data)               # ④ 积分到 t+dt（内部会重算 forward，结果与 ② 一致）
qpos_next = data.qpos.copy(); qvel_next = data.qvel.copy()
```

**验证事实** [V]：
- `mj_forward` 不推进 `data.time`；`mj_step` 恰好推进 `opt.timestep`=0.001 s；
- `mj_step` 之后 `data.qacc` 仍等于 ② 的快照（差恰为 0.0）——**qacc 属于 pre-step 状态**；在新状态上重新 forward 得到的 qacc 与其差 2.6 rad/s²（混淆 post-step 状态与 pre-step 加速度会引入真实错误）；
- 协议 P-B（先存 q,v → `mj_step` → 事后读动力学字段）与 P-C（`mj_step1` → 可改 ctrl → `mj_step2`）与 P-A 的 qacc/next-state **完全一致（差 0.0）**；P-A 代价是 forward 算两遍，数据量大时可换 P-B/P-C；
- **禁止在 `mj_step1` 与 `mj_step2` 之间快照动力学字段**：step1 后 M 已有效但 qfrc_actuator/qfrc_constraint/qacc 尚未计算（fresh data 上为 0）[V]；
- **禁止**直接写 `data.qfrc_actuator` 注入力矩（会被 step 内部重算覆盖，Week1 报告已证）。

**加速度标签**：V1 直接用 `data.qacc`（A 方案）。B 方案（状态差分）只作后续多步预测研究，且必须用 `mj_differentiatePos`（四元数正确差分；`mj_integratePos`/`mj_differentiatePos` 往返误差恰为 0 [V]；对 quat 分量裸差分给出 `[-0.005,0,0,1]` 之类无物理意义的量 [V]）。

**逐 episode 需记录的上下文**：`time`、`episode_id`、`step_id`、场景/控制策略标识（PD 跟踪哪条参考、随机力矩等）、接触状态、随机种子、split 归属 [D]。

---

## 13. 第一版数据字段规范

| 类别 | 字段 | 状态 |
|---|---|---|
| **训练必需** | `episode_id, step_id, time, qpos(29), qvel(28), ctrl(22), qacc(28), M_upper(406,f32), qfrc_bias(28), qfrc_actuator(28), qfrc_constraint(28), free_dynamics_mask(bool), qfrc_constraint_norm(f32)` | 主干输入输出 + 残差全部物理量 + mask |
| **训练必需（监督完整性）** | `qfrc_passive(28), qfrc_applied(28)` | 断言恒 0 的审计字段（防采集协议漂移） |
| **推荐保存** | `qpos_next(29), qvel_next(28)`（多步/积分评价）、`actuator_force(22)`（与 ctrl 互验）、`ncon, nefc`、每足足底接触 4 槽位 `pos/normal/wrench` 与 `in_contact/normal_force` 汇总（分层评价） | |
| **可离线重算（不存）** | `M_dense`（可由 `M_upper` 重建；亦可用 (qpos)+`mj_forward` 重算——路径无关）、`qfrc_bias`（(q,v) 路径无关，可重算；V1 为去 MuJoCo 依赖选择存储）、`efc_J/efc_force`（重算依赖 warmstart 路径，不建议作真值重算，见 §7） | |
| **元数据（每 episode）** | XML sha256、MuJoCo 版本、timestep、na=0、控制策略标识、激励/动作类型、seed、`xfrc_applied_zero: true`（实验条件）、`ended_by`、split id | |

`act` 字段：na=0，不存在，不采集（元数据记录 na=0）[V]。规模控制：仅上三角 M（406 f32）而非稠密 784。

---

## 14. 最小验证程序及运行结果

`tools/pinn/verify_k1_pinn_physics_residual.py`（只读，不改 XML/控制器/物理参数），11 节 49 项检查 **全部 PASS**，摘要写入 `tools/pinn/verify_summary.json`。关键数值：

| 检查 | 结果 |
|---|---|
| 维度 nq/nv/nu/na | 29/28/22/0 [V] |
| M 对称/正定 | 对称误差 0；eigmin 1.97e-3, eigmax 19.93 [V] |
| armature 已含于 mj_fullM | 去掉后残差 = −diag(arm)·a（1.45e-9），扰动幅值 166 [V] |
| r_dyn(a*) 三状态 | 4.5e-15 / 1.45e-9 / 1.5e-13（air/contact/limit）[V] |
| 退化 r(â)=M(â−a*) | 1.45e-9（+人工扰动 δ）[V] |
| free mask | air=True；contact、joint_limit=False [V] |
| Jᵀ·efc_force=qfrc_constraint | dense 1.1e-13 / mulJacTVec 恰 0 [V] |
| 接触世界力聚合 ↔ qfrc_constraint | 1.2e-13；wrench 作用于 geom2 body（符号 +）[V] |
| 法向合力 ↔ root-z 约束力 | 21286.3 N 精确相等 [V] |
| xfrc 会计 | jacBodyCom（质心、世界系）1.4e-17；frame 原点假设被排除 [V] |
| mj_inverse | = M·a+bias−passive−applied−constraint（2.3e-10）≈ qfrc_actuator（1.2e-9）；覆写 qfrc_constraint 警告 [V] |
| 时序协议 | P-A/P-B/P-C 三协议等价（差 0.0）；step1 中途字段无效；forward 不推进时间 [V] |
| 四元数 | integrate/differentiate 往返 0；裸差分错误示范 [V] |
| mj_mulM | 7.1e-15 [V] |

覆盖状态：无接触（air）、足底接触（stance 下压，8 触点）、活动关节限位（knee 越下限，nefc=1）。**equality 约束无法构造**（neq=0，主模型无此项）——如实说明，未伪造。

---

## 15. 已知限制和下一步建议

**限制**：
1. equality/tendon 约束在主模型不存在，相关映射只验证了"不存在"本身；`K1_22dof_parallel.xml` 有 connect 闭环但未被任何代码使用，未纳入。
2. `efc_type` 数值标签与 4.0 前文档枚举不一致（§7），已改用结构识别；若未来版本再变，结构识别规则仍成立。
3. 接触部分与限位部分在 `qfrc_constraint` 中不显式可分，需 efc 行簿记（本报告给出了方法与验证）。
4. `mj_inverse` 的覆写副作用要求"先快照后调用"。
5. Free-dynamics 样本占比取决于采集场景设计（舞蹈类 PD 跟踪腾空相少），Task 2 需显式包含随机激励场景以保证 N_f。
6. 全部结论限定于本 MuJoCo 栈（0.001 s、官方 XML、Newton/impratio=10），不外推 IsaacLab/真机。

**下一步（Task 2）**：数据采集接口与最小数据集——按 §12/§13 实现采集器（协议 P-A，含 8 条审计断言：恒等式、passive/applied/xfrc≡0、actuator_force≈ctrl、M 对称正定、mask 簿记），先出 ~50k 试点集并跑通 free-mask 统计与 norm_stats 产出，为 MLP/PINN 对照实验准备数据。
