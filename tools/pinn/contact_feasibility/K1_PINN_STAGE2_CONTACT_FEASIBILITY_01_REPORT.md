# K1_PINN_STAGE2_CONTACT_FEASIBILITY_01_REPORT — Contact-aware PINN 数学与数据可行性分析

- 日期: 2026-11-24
- 性质: 只读数学推导 + 已有数据数值验证；**未训练任何模型、未修改任何既有文件**
- 分析脚本/结果: `tools/pinn/contact_feasibility/{feasibility_analysis.py, contact_feasibility_results.json, force_scale_by_class.csv}`
- 数据: `tools/pinn/stage1_data/` 全部 30 episodes / 24,578 samples（Oracle 全量；等价性验证用 TEST 3,152 样本 + Stage1-02 checkpoint `pinn_lam1_seed0` 的真实预测）

---

## 1. 核心结论（对应任务书 6.1）

| 问题 | 结论 |
|---|---|
| 现有数据是否支持构造 Contact-aware residual？ | **字段完整**：M(q)（以 M_upper 406 f64/样本保存）、qfrc_bias/actuator/passive/applied/constraint、qacc、互斥三类标签全部全量存在，无 NaN/Inf，时间严格对齐（dt 违例 0），同快照同源性由采集协议保证 |
| Oracle residual 是否接近数值零？ | **是（三类全部）**：Free max 2.7e-5 N（相对 |M·a*| 4.3e-8）；Contact-exclusive max 0.123 N、RMS 6.8e-4 N（相对 2.1e-5）；Impact max 1.3e-3（相对 4.4e-8）。量级 = Newton 求解器容差，相对力尺度（Contact-ex 约束力 RMS 43.6 N、Impact 1137 N）为 1e-5 量级 ⇒ 实用意义上精确为零 |
| Contact-aware residual 是否等价于 mass-weighted acceleration error？ | **是（代数恒等 + 数值验证）**：`r_contact(â) ≡ M(q)(â − a*)`，差值 = Oracle 残差本身（见 §2） |
| qfrc_constraint 是否是独立于 qacc 的物理信息？ | **对模型而言不是**（同一次求解的会计恒等项）；**作为监督标签是**——它是 (q,v,ctrl) 经求解器的确定性函数，含接触/关节限位信息，模型输入中不含 |
| 是否存在训练/推理信息不一致？ | 情况 I/II 无不一致（constraint 项只在训练期进入损失/辅助头）；情况 III（推理期残差）不可行且被拒绝 |
| 是否值得进入下一阶段实验？ | **CONDITIONAL GO**——只做损失函数设计对照（mass-weighted acceleration loss）与明确的 solver-force-supervised 辅助监督消融；**不得作为"独立接触物理模型"立项**（§4 最小方案） |

## 2. 关键推导与数值验证

**代数推导**：训练样本所有力项、M、a* 取自同一次前向求解，满足 `M·a* + bias = act + passive + applied + constraint`。代入残差定义：

```
r_contact(â) = M·â + bias − act − passive − applied − constraint
             = M·â − M·a*        （用恒等式替换力项组合）
             = M(q)·(â − a*)
```

**数值验证**（两个独立途径）：
1. **Oracle**（a = a*）：三类残差全部为求解器容差级零（§1 表；数据记录为 float64，残差仅来自 step 内二次求解的收敛差）。
2. **真实 checkpoint**（PINN λ=1 seed0 在 TEST 的 3,152 个预测）：`r_contact(â) − M(â−a*)` 的逐类 max = 2.4e-5（free）/ 1.0e-1（contact-ex）/ 1.3e-3（impact）——**恰等于该类 Oracle 残差**，即恒等式精确成立。

**结论**：在本数据定义下，拟研究的 Contact-aware residual **不含任何独立于加速度监督的物理信息**；把 `qfrc_constraint` 放进残差只是把"逐 dof 标准化加权"换成"质量矩阵二次型加权"的另一种损失设计。qfrc_constraint 的组成 = 足底/身体接触 + 22 个关节限位的软约束广义力（Task 1 §4 已验证，equality/tendon=0），不能未经分解就当作纯接触力。

**质量矩阵加权的定量分析**（TEST，真实预测误差 e = â − a*）：

| 类 | L_acc（Σe² 均值） | L_mass（eᵀMe 均值） | 对角近似/全阵 | L_mass/L_acc |
|---|---|---|---|---|
| Free | 1.20e5 | 1.52e3 | 2.35 | 1.3e-2 |
| Contact-exclusive | 9.64e5 | 1.47e4 | 1.36 | 1.5e-2 |
| Impact | 2.17e8 | 1.10e7 | 1.72 | 5.1e-2 |
| 全部 | 4.38e6 | 1.98e5 | 1.71 | 4.5e-2 |

两个事实：(a) **M 不可当对角阵**——对角近似高估真实二次型 1.36–2.35 倍（M 的非对角元平均为负，耦合项显著）；(b) **mass 加权会进一步放大 Impact 域的相对权重**（impact/free 的损失比从 L_acc 的 1799 升至 L_mass 的 7244），与 Stage1 目标（改善 contact 域常规预测）方向相反——纯 M 加权作为损失设计大概率是有害的，除非配类平衡/鲁棒尺度。

## 3. qfrc_constraint 的信息属性（任务 D）

1–2. 是求解器输出（Newton + warmstart 路径），由真实状态、接触几何、接触参数与求解过程共同决定；同模型内非路径无关（step 内二次求解与快照一次求解差 ~3.5e-6 rad/s² / 0.123 N，WM_01/本报告实测）。
3. **不能**仅凭 (q,v,ctrl) 在部署推理时独立获得——必须内嵌 MuJoCo 约束求解器（含接触检测与 warmstart）。
4. 属于**训练期 privileged information**。
5. 情况 I/II 下无泄漏、无训练/推理不一致（它从不进入网络输入，推理期不需要）；情况 III 下模型退化为"包着网络的 MuJoCo"，被拒绝。
6. **可作为辅助监督目标**（预测 qfrc_constraint 的辅助头，推理期丢弃）——这是唯一可能带来超出损失重加权之归纳偏置的用法（迫使网络内部表征接触状态），但本质是 solver-force-supervised 蒸馏，不得表述为独立接触物理。
7. 不依赖求解器输出的精确构造在现有数据内**不存在**（数据未存 efc 行，无法从存储值分离"纯接触力"与"关节限位力"；近似合规模型需推理期接触检测，仍是仿真器知识）。若未来需要纯接触力辅助目标，需在采集端记录 efc 行分解——记录为数据缺口，不在本任务补。

**三情况裁定**：情况 I = 损失重加权（与本报告 §2 等价性结论绑定）；情况 II = 有研究价值的辅助监督消融（命名 solver-force-supervised）；情况 III = 不可行，拒绝。

## 4. 下一阶段最小对照方案（CONDITIONAL GO，本任务不执行）

| 组 | 输入 | 监督目标 | 物理残差/额外损失 | 训练期字段 | 推理期字段 | 验证的科学问题 |
|---|---|---|---|---|---|---|
| G1 MLP | 81 维（同 Stage1-02） | a*（per-dof 标准化 MSE） | 无 | 现有 | 现有 | 基线 |
| G2 Free-PINN（现有） | 同上 | 同上 | r_free（Free 域） | 现有 | 现有 | 已有结果，作参照 |
| G3 Mass-weighted loss | 同上 | a*，损失改为/叠加 `L_mass = eᵀM e`（全阵） | **无**（与 G1 差异仅在加权） | 现有（M 可离线算） | 现有 | 质量加权是否改善 contact 域预测（纯损失设计对照；**预期风险**：放大 impact 权重，须配类平衡或鲁棒尺度） |
| G4 Solver-force-supervised | 同上 + 辅助头 ĝ = g(q,v,ctrl) | a* 主损失 + μ·MSE(ĝ, qfrc_constraint) 辅助损失 | 无部署期残差（辅助头推理期丢弃） | qfrc_constraint 标签（已有） | 现有 | privileged 接触力标签作为辅助监督能否改善 contact/impact 域主任务（**不得宣称学到独立接触物理**） |

评价指标沿用既有口径：Free / Contact-exclusive / Impact 分层 + bulk（≤5σ）+ ≥5 seeds + episode 级隔离 + （可选）H=1/2/4/8 rollout。**若 G3/G4 改善 contact 域，正确解释是损失设计/辅助监督效应，不是模型获得了新的接触动力学知识**；G3 与"Contact-aware residual"是同一事物的两种说法，不得包装成两个独立物理模型。

## 5. 证据边界

- **数学恒等式**：r_contact(â) ≡ M(â−a*)（同求解前提下精确成立）。
- **数值验证**：Oracle 三类 ≤ 求解器容差（§1 表）；恒等式差 = Oracle 残差；真实 checkpoint 复核一致。
- **数据字段事实**：全部字段存在且 f64 无损；M 已保存（M_upper）；`qfrc_constraint` 含关节限位贡献、efc 分解未存储（纯接触力标签的缺口）。
- **对未来的假设**：G3/G4 的潜在收益是假设（G4 有 ROLLOUT-01 的 root 旋转 contact/impact 优势作为弱旁证；G3 存在放大 impact 权重的定量风险），均需实验检验。

**最终判定：CONDITIONAL GO。** 不建议以"Contact-aware PINN"名义新增物理模型组；建议以 G3（损失设计对照，须处理 impact 权重放大风险）与 G4（solver-force-supervised 辅助监督）作为下一阶段的两组最小实验，评价与命名遵循本报告约束。是否启动由 Owner 决定。
