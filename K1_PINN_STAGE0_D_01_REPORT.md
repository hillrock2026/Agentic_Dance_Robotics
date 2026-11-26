# K1_PINN_STAGE0_D_01_REPORT — MLP 与 Free-Dynamics PINN 最小对照实验

- 日期: 2026-11-24
- 执行环境: conda env `booster_train`（MuJoCo 4.0.1 + PyTorch 2.7.0 + CUDA），K1 官方 XML 未修改
- 数据/代码: 全部在 `tools/pinn/`（采集、切分、训练、评价）；数据 `tools/pinn/stage0_data/`；产物 `tools/pinn/stage0_runs/`
- 上游依据: `K1_PINN_TASK_01_PHYSICS_RESIDUAL_AND_MUJOCO_MAPPING.md`（Free-Dynamics residual 与采集时序）

---

## 0. 总结论

| 问题 | 答案 |
|---|---|
| F1 数据可行性 | **基本可行但不足**：管线正确（逐步残差闭合审计通过），但 15.6k 强相关样本 / 20 episodes 支撑不起强泛化；无约束样本 14.3% 且 70% 集中在 drop 类 |
| F2 MLP 可行性 | **可行**：训练稳定收敛，在常规样本（<5σ）上解释测试基线 MSE 的 7.8%（val 上 22%），无约束样本 RMSE 显著优于均值基线（163 vs 213）；但**全测试集 RMSE/MAE 输给"预测均值"基线**（冲击带无预测力 + 跨动作分布偏移） |
| F3 PINN 可行性 | **可稳定加入**（λ≤1 无发散），物理损失在训练集稳定下降；但收益是**残差一致性改善（1.76→1.38–1.58），无加速度预测改善**；λ=10 三种子 best_epoch 全为 0——物理梯度压制数据训练（梯度比 ~346×） |
| F4 研究价值 | 当前形态的 Free-Dynamics PINN **不足以作为论文核心方案的最终形态**：它本质是"训练时正则"，在推理期无独立物理约束（Task 1 已论证退化问题），且本实验未见预测收益 |
| F5 Stage1 决策 | **CONDITIONAL GO**（见 §6） |

---

## 1. 数据（Task A）

### 1.1 采集

`collect_stage0_data.py`，严格 Task 1 P-A 时序（`ctrl → mj_forward → 快照 → mj_step`），1 kHz，每步断言：完整残差闭合（相对阈值 1e-3，实测动态接触下 Newton 容差内 ~2e-4 相对）、`qfrc_passive=qfrc_applied=xfrc_applied=0`、字段有限。20 episodes / 15,600 步 / 45 MB / 字段含 qpos/qvel/ctrl/qacc/M_upper/qfrc_bias/qfrc_actuator/qfrc_constraint/ncon/nefc/time/actuator_force/residual_closure。

三类场景：**pd_***（软件 PD+重力补偿跟踪 4 条既有参考轨迹，gain ×U(0.8,1.2)、初始扰动）、**drop**（空中随机姿态初速、零力矩抛落）、**ou_torque**（站立 OU 力矩探索）。

### 1.2 无约束样本统计（Task A5，mask: ncon=0 ∧ nefc=0 ∧ ‖qfrc_constraint‖∞=0）

| 类别 | episodes | 步数 | free 样本 | free 占比 |
|---|---|---|---|---|
| pd_top_rock / pd_arm_wave / pd_popping / pd_usr | 6 | 10,884 | 524 | 4.8% |
| drop | 8 | 3,252 | **1,561 (70% 的全部 free)** | 48.0% |
| ou_torque | 6 | 1,464 | 145 | 9.9% |
| **合计** | 20 | **15,600** | **2,230 (14.3%)** | |

**集中性**：free 样本高度依赖 drop 类；PD 跟踪腾空相极少（arm_wave/usr 仅 0.7–1.9%）。**覆盖**（train 内 free vs contact）：关节速度 absmax 50.9 vs 112.3、root 线速度 2.39 vs 2.42 m/s、root 角速度 7.2 vs 9.6 rad/s——量级相当但 free 未覆盖 contact 的高速端。**未人为放宽 mask**；OU episode 站立不稳快速跌倒（137–353 步）是真实行为，如实保留。

### 1.3 训练过滤规则（冻结、全模型一致、如实报告）

跌落/落地终止帧的归一化 qacc 达 10–133σ，**0.63% 的 val 样本贡献 87% 的 MSE**。冻结规则：`max|a_norm| ≤ 10σ` 的样本参与**训练与早停**（train 移除 118/9732、val 移除 28/4418）；**全部指标仍在完整 val/test 上计算**（含冲击带），并单列冲击带指标。规则写入 `norm_stats.json`。此规则不是为了让结果好看——它在 MLP 与 PINN 间完全对称，且冲击带指标单列呈现（结果：所有模型在该带均≈均值基线，无一受益）。

---

## 2. 方法（Task B/C/D）

- **输入 81 维**：root pos(3) + 四元数→**6D 连续旋转表示**(6) + 关节角(22) + qvel(28) + **ctrl(22，真实 MuJoCo ctrl，非 actuator_force/dof_target)**；输出 28 维 qacc（z-score 归一化）。
- **划分（episode 级、动作不相交）**：TEST={pd_top_rock, drop_d0, ou_o0}（1450 步）、VAL={pd_arm_wave, drop_d1, ou_o1}（4418）、TRAIN=其余 14 episodes（9732，free 1546=15.9%）。同一 episode 不跨集合；train/val/test 比例 62/28/9（受 episode 粒度限制，偏离 70/15/15，test 保证独立且含全部三类场景）。
- **归一化**：仅 train 统计；`s_a=std(a)`、`s_r=RMS_train,free((M·qacc)_i)`；ε 下限 1e-3；冻结于 `norm_stats.json`。
- **网络**（MLP 与 PINN 完全相同）：81→3×256(SiLU)→28，159,772 参数。
- **优化**：AdamW lr 1e-3 wd 1e-5，batch 256，≤2000 epoch，早停 patience 200（监控 val L_data），grad-clip 1.0，3 种子。超参经 4 个变体小扫描（仅 MLP seed0、仅用 val 选择），val 差异 ≤0.008，全部停在 ~0.124–0.129 平台——**该平台是数据规模下的泛化地板而非调参问题**，故保留默认配置。
- **PINN 损失**：`L = L_data + λ·L_free`，`r_free = M·a_pred + qfrc_bias − qfrc_actuator`（**不含 qfrc_constraint**，未使用完整 solver residual），仅作用于 batch 内 free 样本（无 free 样本的 batch 跳过；每 epoch 参与样本数记录）。监控项：两损失数值、各自梯度范数、free 样本数、lr。λ∈{0.1, 1, 10}。

---

## 3. 结果（Task E，test 三 episodes，3 种子均值±std）

### 3.1 主对照表（物理单位；RMSE 单位混合 m/s²·rad/s²，被冲击带主导）

| 配置 | MAE | RMSE | base_lin | base_ang | joints | **free RMSE** | contact RMSE | **≤5σ RMSE** | 5–10σ | >10σ | free 残差(归一) | free 残差(N/Nm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 85.9±2.2 | 548.5±2.3 | 28.9 | 182.2 | 615.0 | **163.3±7.0** | 607.2 | **174.1±4.5** | 1069 | 4119 | 1.763±0.07 | 5.52 |
| PINN-0.1 | 81.1±1.6 | 548.8±0.8 | 29.0 | 181.7 | 615.4 | 173.1±8.2 | 606.9 | 175.3±4.7 | 1058 | 4126 | 1.575±0.08 | 5.75 |
| PINN-1 | 80.0±3.1 | 550.3±4.6 | 29.1 | 181.5 | 617.1 | 191.1±12.2 | 607.2 | 186.6±4.4 | 1046 | 4113 | 1.528±0.15 | 4.93 |
| PINN-10 | 75.2±0.5 | 543.6±0.5 | 29.1 | 181.6 | 609.5 | 214.6±3.8 | 597.8 | 183.5±0.7 | 1010 | 4080 | **1.382±0.02** | 3.54 |
| **均值基线** | **73.0** | **543.0** | 28.8 | 181.5 | 608.8 | 213.4 | 597.2 | 183.2 | 1009 | 4076 | 1.400 | 8.45 |

oracle 参照：真值加速度在 test free 样本上的归一化残差 = **0.000**（free 标签无求解器噪声，残差指标干净）。contact 残差（诊断）：11.8–12.0，各配置无差异。

### 3.2 关键读数

1. **MLP 在 free 样本和常规带上确实优于基线**：free RMSE 163 vs 213（好 23%），≤5σ 带 174 vs 183；test ≤5σ 带上解释基线 MSE 的 **7.8%**（val 上 22.5%）。**但全测试集 MAE/RMSE 输给均值基线**——误差集中在 >10σ 冲击带（RMSE ~4100，与基线持平）与跨动作 contact 样本（607 vs 597）。
2. **λ 单调换准确率换一致性**：λ: 0→0.1→1→10 对应 free 残差 1.76→1.58→1.53→1.38，free RMSE 163→173→191→215。物理损失买到的是残差一致性，**没有**买到预测精度（λ=10 的 free RMSE 214.6 与基线 213.4 持平）。
3. **λ=10 压制训练**：三种子 best_epoch 全为 0，模型停在近似初始状态（≈均值预测器），故其指标全面贴近基线。监控的梯度范数比（早期 epoch）：λ=0.1→5.4×、λ=1→32×、λ=10→346×——**L_free 梯度支配是直接原因**。
4. **物理损失可稳定优化但泛化不出训练 free 样本**：train_free 0.15→0.008（下降 ✓），val_free 却从 0.03 **升至** 0.5–6.0。机理：free 残差在 PD 跟踪腾空段要求 a≈M⁻¹(act−bias)≈小值（重力补偿使 act≈bias），而模型从 contact 冲击数据学到的"大加速度"行为被错误外推到 airborne 状态——**L_free 与 L_data 存在真实张力，物理损失实际上在教"接触状态感知"，而 81 维输入（无接触信息）+ 当前数据量不足以表达该边界**。
5. **种子一致性**：上述全部结论在 3 种子上方向一致（表格 std 即证）；无种子翻转现象。
6. **训练时长**：12 个 run 共 ~4 分钟 GPU（每 run 12–23 s），最佳 epoch 0–13 即出现，之后 overfit。

---

## 4. 重点问题逐条回答（任务书 §7）

- **PINN 是否降低测试集加速度误差？** 没有。全指标上无一 λ 配置同时优于 MLP；λ≥1 反而劣化 free RMSE。
- **是否只改善无约束样本？** 改善的是无约束样本的**残差**（诊断量），但其**加速度精度反而随 λ 变差**。
- **physics loss 是否稳定下降？** 训练集上是（0.15→0.008）；验证集上不降反升——物理一致性与数据泛化在本数据上不同步。
- **λ 的规律？** 单调正则化/收缩效应，λ=10 处梯度支配导致训练失效。λ=0.1 是唯一"无害"档（MAE 略降、残差略降、RMSE 持平），但增益不含物理意义信号。
- **当前无约束样本是否具代表性？** 不足：70% 来自零力矩抛落（一个动力学域），PD 跟踪腾空几乎缺失；且 val/test 的 free 残差行为与 train 不同（controller 语义：重力补偿与否）主导该指标。

---

## 5. 真实运行记录

**已运行**：采集 20 episodes（45 MB）；切分/归一化；4 变体超参扫描（MLP seed0）；最终网格 4 配置 × 3 种子 = **12 次完整训练**（每次含完整早停与监控）；统一测试评价（13 行 × 27 指标，`metrics_per_run.csv` / `metrics_summary.json`）；oracle 残差与梯度比分析。
**未运行**：多步积分 rollout 评价（任务书列为不作为完成条件）；更多 episode / 更大数据规模；λ 连续扫描。
**失败与修正记录**（全部保留在代码与本文）：初版断言阈值过紧（Newton 容差在动态接触下 ~2e-4 相对，探针曾误判协议漂移——两个探针自身漏写 mj_step 的 bug 已定位）；episode 时长 bug（286 帧被截成 150 步）；OU forcerange 维度 bug；冲击带主导 MSE 导致训练失效（→ 冻结过滤规则）。未发生 PINN 数值发散；未使用 test 做任何选择。

---

## 6. Stage1 决策：**CONDITIONAL GO**

数据管线、时序协议、残差定义、训练与评价链路全部打通且审计干净（oracle 残差=0），MLP 可学、PINN 可训——**具备继续的条件**；但"Free-Dynamics PINN 提升 K1 动力学预测"这一命题在本规模下**不成立**，直接放大到 Stage1 正式数据集大概率重复本结果。进入 Stage1 前必须针对以下明确发现调整：

1. **数据覆盖（最重要）**：正式数据集必须包含系统性的腾空/多接触事件（当前 free 仅 14.3% 且 70% 来自抛落）。建议：跳跃/摆腿激励 episode、PD 参考加入含腾空相的动作、contact 状态进入观测的消融（81→83 维：左右足接触 flag）。
2. **残差设计**：任务书第四部分预判的"退化"已被实验证实为实际约束——qfrc_constraint 已知量口径下物理损失只能在 free 样本施压，而模型无接触输入导致两损失张力。Stage1 应把 **contact-aware residual**（用记录的 qfrc_constraint 作为监督/约束的 Contact-aware PINN）列为与 Free-Dynamics 并列的正式研究臂，并把"接触状态感知"作为设计目标而非副作用。
3. **λ 治理**：梯度比监控应成为 λ 选择的硬性依据（本实验 30× 即已有害）；建议按 residual 加权（per-dof s_r 之外再除以 ‖∇L_data‖ 归一）或 GradNorm 类自适应，而非手选 λ。
4. **样本规模**：相邻 1 kHz 样本高度相关（有效样本数远小于名义数），Stage1 按 episode/段采样或 stride>1，并以"独立段数"而非步数报告规模。
5. **评价口径**：全量 RMSE 被冲击带主导（0.6% 样本 → 87% 损失），论文指标应固定为：bulk（≤5σ）+ free/contact 分层 + 冲击带单列 + 残差一致性（附 oracle=0 参照）——本报告的表结构可直接沿用。

**NO-GO 不适用**：管线无实质性问题，数据缺陷（覆盖不足）可通过采集设计解决。

---

## 7. 交付物清单

| 项 | 路径 |
|---|---|
| 采集程序 | `tools/pinn/collect_stage0_data.py` |
| 数据集（20 npz + index/splits/norm_stats） | `tools/pinn/stage0_data/` |
| 共享库（特征/6D 旋转/模型/损失） | `tools/pinn/stage0_common.py` |
| 切分+归一化+free 统计 | `tools/pinn/prepare_stage0_splits.py` |
| 训练脚本（MLP/PINN 统一入口） | `tools/pinn/train_stage0.py` |
| 评价脚本（统一 test + 基线 + 汇总） | `tools/pinn/evaluate_stage0.py` |
| 12 个 run（权重/配置/训练曲线/test 预测） | `tools/pinn/stage0_runs/<run>/`（best.pt, config.json, history.json, test_preds.npz） |
| 指标表（per-run / summary） | `tools/pinn/stage0_runs/results/metrics_per_run.{csv,json}`, `metrics_summary.json` |
| Task 1 验证程序（沿用未改） | `tools/pinn/verify_k1_pinn_physics_residual.py` |

复现命令（conda `booster_train` 解释器）：
```bash
python tools/pinn/collect_stage0_data.py
python tools/pinn/prepare_stage0_splits.py
python tools/pinn/train_stage0.py --model mlp --seed 0            # + pinn --lambda_dyn {0.1,1,10}
python tools/pinn/evaluate_stage0.py
```
