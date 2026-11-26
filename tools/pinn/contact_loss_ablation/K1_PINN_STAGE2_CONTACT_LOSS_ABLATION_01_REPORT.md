# K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_01_REPORT — 三种物理损失设计的对照实验

- 日期: 2026-11-24
- 新增训练: G3/G4/G5 × 5 seeds（0–4）× λ=1 = **15 次，全部完成**（`tools/pinn/contact_loss_ablation/runs/`）
- 复用: G1（MLP）、G2（Free-PINN λ=1）直接沿用 Stage1-02 checkpoint 与指标；G1/G2 的 rollout 结果沿用 ROLLOUT_01 原始逐窗口产物（未重算）
- 判定: **PASS**（完成标准 6 项全部满足）；核心结论——**三种损失设计均未改善 Contact-exclusive 域预测**

---

## 1. 实验目的与数学定义

研究问题：不同物理损失形式与接触求解器辅助监督，能否改善 PINN 在 Contact-exclusive 域的预测。基础监督 `L_acc` 与 Stage1-02 完全一致（per-dof 标准化 qacc MSE + 冻结 10σ 冲击帧过滤），五组唯一差异如下（λ 统一 = 1，见 §2 归一化）：

- **G3** `L = L_acc + λ·(1/N)Σ eᵀM(q)e / s₃²`：全阵二次型（禁止对角近似），e 为物理单位误差；`s₃² = mean_TRAIN((a*−μ_a)ᵀ M (a*−μ_a))` = **9,695**。
- **G4** `L = L_acc + λ·(1/N)Σ ‖M(q)e‖² / s₄²`：全阵残差平方；`s₄² = mean_TRAIN(‖M(a*−μ_a)‖²)` = **3.181e4**。与 G3 数值上不同（Step-2 验证：随机 e 下相对差 17.5×）。
- **G5** `L = L_acc + λ·MSE(ĝ, qfrc_constraint)`：共享主干（81→3×256）+ 主头 28 维 qacc + 辅助头 28 维 qfrc_constraint（TRAIN 统计 z-score），辅助头仅见主干特征、推理期丢弃。**命名：solver-force privileged supervision**，非独立接触力观测。

归一化/尺度全部来自 TRAIN（s₃²、s₄²、μ_g/σ_g），VAL/TEST 未参与；无类别重加权； Impact 样本保留；基线 G1/G2 的原始结果未覆盖未重算。

## 2. Step-2 数学单元验证（训练前，全部通过）

M 重建后对称（误差 0）；G3 与蛮算 `eᵀMe` 最大差 3.8e-6；G4 与蛮算 `‖Me‖²` 最大差 1.0e-4；同批随机 e 下 |L3−L4|/L3 = 17.5（证实 G3≠G4）；G5 主/辅输出均 28 维，辅助头输入 = 主干特征（无真值耦合）。

## 3. 单步加速度 TEST 结果（5 seeds 均值±std）

| 组 | Overall RMSE | **Free RMSE** | **Contact-ex RMSE** | Impact RMSE | ≤5σ RMSE |
|---|---|---|---|---|---|
| G1 MLP | — | 78.36±4.86 | 176.50±4.35 | 2771.7 | 102.65±3.31 |
| G2 Free-PINN | — | **70.92±4.55** | 176.61±5.01 | 2788.3 | **99.92±5.14** |
| G3 mass-weighted | 392.28±0.31 | 122.16±1.29 | 177.43±1.43 | **2739.5** | 121.88±1.30 |
| G4 residual-squared | 393.57±2.42 | 93.16±14.28 | 180.21±7.14 | 2763.9 | 112.66±3.13 |
| G5 solver-force-aux | 395.92±1.38 | 133.73±4.57 | 182.73±2.80 | 2746.4 | 130.53±3.15 |
| 均值基线 | — | 121.93 | 175.99 | 2746.0 | 119.78 |

逐 seed Contact-exclusive RMSE：G1 174.1/173.9/184.8/172.8/177.0；G2 185.6/174.8/178.2/172.0/172.4；G3 176.1/176.5/178.7/176.3/179.6；G4 181.1/190.8/177.0/169.1/183.0；G5 181.3/179.1/182.3/183.3/187.6。

**要点**：(a) 三组新增损失在 Contact-exclusive 上**无一改善**（G3 177.4、G4 180.2、G5 182.7 vs G1 176.5）；G4 的 seed3=169.1 是唯一低于 G1 均值的单点，但组内方差极大（±7.1）且其余 seed 更差。(b) **G3 五种子 best_epoch 全为 0**、val_data 0.140–0.144（劣于 G1 的 0.121–0.130）——与可行性报告预警一致：质量加权放大 Impact/root 权重（impact/free 损失比 1799→7244），物理项在 λ=1 下主导梯度，训练退化为向均值-惯性收缩（Free RMSE 122 ≈ 均值基线 121.9）。(c) **G5 在全部域一致变差**（Free 133.7 > 基线 121.9、≤5σ 130.5、val 0.160 5/5 一致），辅助监督未迁移到主任务。(d) G4 方向正确（le5 112.7 优于 G3/G5）但幅度小且 seed 不稳定（Free ±14.3）。

## 4. 开环 rollout（H=1/2/4/8，3,110 共享窗口，5-seed 均值）

### Joint velocity RMSE (rad/s)

| 组 | H=1 | H=2 | H=4 | H=8 |
|---|---|---|---|---|
| G1 | 0.436 | 0.668 | 1.069 | 1.797±0.021 |
| G2 | 0.438 | 0.673 | 1.082 | 1.831±0.024 |
| G3 | 0.436 | 0.667 | **1.067** | **1.799**（H4 亦最优） |
| G4 | 0.438 | 0.672 | 1.079 | 1.826±0.044 |
| G5 | 0.440 | 0.677 | 1.091 | 1.853±0.016 |

### H=8 分起点 joint velocity RMSE (rad/s)

| 组 | Free | Contact-exclusive | Impact |
|---|---|---|---|
| G1 | 1.107 | 1.610 | 8.370 |
| G2 | **1.071** | 1.637 | 8.713 |
| G3 | 1.216 | **1.604** | **8.135** |
| G4 | 1.133 | 1.654 | 8.344 |
| G5 | 1.300 | 1.654 | 8.215 |

（joint position H=8：G1 0.0085 / G2 0.0087 / G3 0.0085 / G4 0.0086 / G5 0.0088 rad，差异微弱；root geo/angv/linv/pos 各组差异亦在噪声内，全表见 `results/rollout_five_groups.json`。）

## 5. 科学问题回答（任务 §7）

1. **G3 是否改善 Contact-exclusive？** 否。单步 177.4 vs G1 176.5（+0.5%）；rollout H8 jv 1.604 vs 1.610（−0.4%，收益可忽略）；代价是 Free 域大幅退化（78→122）与验证拟合崩塌（best_epoch=0 ×5）。
2. **G4 是否改善 Contact-exclusive？** 否。单步 180.2（+2.1%）；rollout H8 1.654（+2.7%）；仅 ≤5σ 带较 G3 稳定。
3. **G3 与 G4 哪个更适合？** 都不适合作为"接触域改善"手段；G3 有一致性收缩效应（impact accel −1.2%、rollout impact jv −2.8%）但以 Free 域为代价；G4 数值上更接近中性。二者不可互代（Step-2 已证损失不同）。
4. **G5 辅助监督是否改善主输出？** 否。5/5 seeds 全域变差（Free +70%、≤5σ +27%），辅助头在 λ=1 下干扰共享主干。
5. **辅助任务收益是否迁移？** 未观察到任何迁移；辅助头本身拟合了 qfrc_constraint（train aux loss 下降），但主任务无受益——privileged 信息未转化为可迁移表征（在 λ=1、当前数据规模下）。
6. **Free/Impact 是否退化？** Free：G3/G5 明显退化，G4 高方差；Impact 单步：G3 −1.2%（收缩效应）、G5 ≈基线；无一组在 Impact 域有超出噪声的稳定收益。
7. **单步还是多步改善？** 均没有。rollout 差异全部在 ±3% 内（除 G5/G3 的域间转移），远小于 ROLLOUT-01 中 G2 在 Free 域的稳定优势。
8. **种子一致性？** 每个负面/中性结论在 5 seeds 上一致；唯一的正向单点（G4-seed3 cex 169.1）不具种子间可复现性。5 seeds 是训练重复，方向一致性不构成统计显著性。

**结论排序**：Contact-exclusive 域上 G1 ≈ G2 ≈ G3 > G4 > G5（差异微小）；Free 域 G2 ≫ G1 > G4 > G3 > G5。**没有任何新增损失设计改善了 Contact-exclusive 域。**

## 6. 证据边界

- G3/G4/G5 仅在 **λ=1**（统一初始权重）下测试；λ 扫描不在本任务范围。G3 的 best_epoch=0 表明其失败模式是权重过高而非损失本身无信息，但结合 Stage1-02 历史（错误目标物理损失的 λ 依赖性）与可行性报告的 impact 放大预警，没有理由预期调 λ 能翻转结论。
- "G5 未迁移"限于共享主干+线性辅助头的架构与 24.6k 样本规模。
- G3 的 impact rollout 改善（jv −2.8%）是收缩效应的伴随现象，不构成接触物理学习的证据。
- 五 seeds 方向一致性 ≠ 统计显著性（同 TEST pool、窗口时间重叠）。

## 7. 交付物

| 项 | 路径 |
|---|---|
| 训练/评价代码 | `contact_loss_ablation/{train_ablation.py, evaluate_ablation.py}` |
| 15 次 run（checkpoint/config/history） | `contact_loss_ablation/runs/<run>/` |
| 单步加速度指标（25 行，含 G1/G2 合并） | `contact_loss_ablation/results/accel_metrics.{csv,json}` |
| 五组 rollout 汇总 | `contact_loss_ablation/results/rollout_five_groups.json` |
| 新 checkpoint 逐窗口 rollout 原始结果 | `contact_loss_ablation/rollout_raw_<run>.npz`（15 个） |
| 损失尺度记录 | 各 run config.json `normalization` 字段（s₃²=9,695；s₄²=3.181e4；μ_g/σ_g TRAIN 统计） |

复现：`python train_ablation.py --group {g3|g4|g5} --lambda_dyn 1 --seed {0..4}`；`python train_ablation.py --group g3 --verify`（Step-2 检查）；`python evaluate_ablation.py`。

---

**最终回答**：在相同数据、划分、网络与训练条件下，质量矩阵加权损失（G3）、全阵残差平方损失（G4）与求解器力辅助监督（G5）**均未能改善 Contact-exclusive 域的动力学预测**；G3 呈现收缩退化、G5 全域变差、G4 中性高方差。G2（Free-PINN）仍是唯一有稳定域内收益的物理损失设计。据此，论文的物理损失章节应以 Free-Dynamics 残差为主线，contact 域改善需另寻机制（如 efc 级接触力分解标签的采集支持），而非损失函数形式的继续变体。
