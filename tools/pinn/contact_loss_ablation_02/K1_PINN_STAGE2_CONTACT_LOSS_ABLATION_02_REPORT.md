# K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_02_REPORT — 修正实现后的 G3/G4/G5 对照实验

- 日期: 2026-11-25
- 性质: ABLATION_01 实现错误修复 + 15 次重训 + G1–G5 统一评价
- 模型: G3/G4/G5 × 5 seeds（新训）+ G1/G2 沿用 Stage1-02 checkpoint（未动）
- 数据: Stage1 TEST 6 episodes（单步 3,152 样本）+ ROLLOUT 同款 3,110 个 H=8 窗口
- 判定: **PASS**（实现修复、15 次训练、统一评价全部完成）；**核心结论：修复后三种损失设计仍未改善 Contact-exclusive 域，G1/G2 的结论稳健**

---

## 1. ABLATION_01 的实质性实现错误与修复

**错误定位**：`train_ablation.py`（01 版）中 10σ keep 掩码只应用到 `X/qacc标签/M/qfrc_constraint/free` 的一部分——**`tr["qacc"]` 未过滤**，而训练循环的 `idx` 是过滤后索引空间的随机数。于是 G3/G4 的物理损失

```
e_phys = a_phys − tr["qacc"][idx]
```

在首个被删样本（189 个 spike 散布）之后**系统性错位**：过滤行 i 的真值应取自原始行 keep_idx[i]，代码却取了原始行 i。主损失（过滤后的 tr_y）正确，物理损失错误——G3 的崩缩（best_epoch=0 ×5、val 0.142）与 G4 的高方差（Free ±14.3）由此产生。

**修复**：单一共享 keep 掩码同步子集化全部逐样本字段（X、qacc、M、qfrc_constraint、free），keep 索引持久化（`keep_indices.npy` + config 记录），TRAIN/VAL 对齐规则一致。损失定义未变（G3 `mean(eᵀMe)`、G4 `mean(‖Me‖²)`、G5 solver-force 辅助头）。G5 的 mu_g/std_g 本就 TRAIN-only 且过滤对齐无误；本次补齐其**持久化**（config.json `g5_mu_g/g5_std_g`）与**辅助任务评价**（反标准化 RMSE/MAE）。

**修复有效性的直接证据**：修复前后 G3 seed0 对照——错误版 val_data=0.1419/best_epoch=0（5/5 seeds 崩缩）；修正版 30-epoch 冒烟即达 val 0.1248/best_epoch=5，完整训练 5 seeds val 0.125–0.132 全部健康。

## 2. 训练执行

15/15 完成（每组 ~35 s，总 ~9 min）；配置 = Stage1-02（AdamW 1e-3/wd 1e-5/batch 256/≤2000 epoch/patience 200/clip 1.0/backbone 81→3×256→28；G5 共享主干+辅助头）；λ=1 统一；归一化尺度 TRAIN-only：s₃²=9,695、s₄²=3.181e4（keep 样本计算）、μ_g/σ_g TRAIN-only。每 run 保存 config（含 seed、尺度、fix 说明、keep 索引散列）、history（主损失/物理或辅助损失/val 两项/逐 epoch 探针梯度范数——**同参数集分别反传测得**/类内贡献）、best.pt、keep_indices.npy。无失败 seed。

## 3. 单步加速度 TEST 结果（5 seeds 均值±std）

| 组 | Free RMSE | **Contact-ex RMSE** | Impact RMSE | Overall MAE |
|---|---|---|---|---|
| G1 MLP | 78.36±4.86 | 176.50±4.35 | 2771.7±3.8 | 52.20 |
| G2 Free-PINN | **70.92±4.55** | 176.61±5.01 | 2788.3±9.8 | 52.20 |
| **G3 修正** | 84.08±5.03 | **176.30±5.93** | 2768.3±3.6 | 53.94 |
| G4 修正 | 89.31±7.81 | 179.55±4.62 | 2767.9±4.7 | 55.16 |
| G5 修正 | 99.65±7.49 | 177.60±4.39 | **2760.6±5.8** | **52.09** |

逐 seed Contact-ex RMSE：G1 174.1/173.9/184.8/172.8/177.0；G2 185.6/174.8/178.2/172.0/172.4；G3 173.9/181.9/183.5/175.2/**167.1**；G4 178.9/178.4/184.8/171.9/183.8；G5 182.0/171.6/182.5/173.5/178.4。

**要点**：修复后 G3 回到合理区间（Free 84.1，介于 G1 78.4 与 G2 70.9 之间），但 Contact-ex 176.30 vs G1 176.50 差异 0.1%——**仍无改善**；G4 179.6（+1.7%）、G5 177.6（+0.6%），同样无改善。G5 的 Overall MAE 52.09 是全场最低（vs G1 52.20），但差异 0.2% 在 seed 方差内，不构成有效改善。逐 dof RMSE（28 维）存于 `accel_metrics.json` 的 `per_dof_RMSE`。

## 4. 开环 rollout（H=1/2/4/8，3,110 共享窗口）

### Joint velocity RMSE (rad/s)

| 组 | H=1 | H=2 | H=4 | H=8 |
|---|---|---|---|---|
| G1 | 0.436 | 0.668 | 1.069 | 1.797±0.021 |
| G2 | 0.438 | 0.673 | 1.082 | 1.831±0.024 |
| G3 | 0.436 | 0.668 | 1.069 | 1.799±0.034 |
| G4 | 0.438 | 0.672 | 1.077 | 1.811±0.027 |
| G5 | 0.437 | 0.669 | 1.071 | 1.802±0.028 |

### H=8 分起点 jv / jp 与 root 指标（5-seed 均值）

| 组 | jv Free | jv C-ex | jv Impact | jp (rad) | geo (rad) | root angv | root linv | root pos (m) |
|---|---|---|---|---|---|---|---|---|
| G1 | 1.107 | **1.610** | 8.370 | 0.0085 | 0.0039 | 0.4581 | 0.0782 | 4.1e-4 |
| G2 | **1.071** | 1.637 | 8.713 | 0.0087 | 0.0039 | **0.4491** | 0.0842 | 4.3e-4 |
| G3 | 1.101 | 1.618 | 8.351 | 0.0085 | 0.0039 | 0.4591 | **0.0773** | **4.0e-4** |
| G4 | 1.120 | 1.630 | 8.369 | 0.0086 | 0.0039 | 0.4506 | 0.0771 | 4.0e-4 |
| G5 | 1.138 | 1.615 | 8.311 | 0.0085 | 0.0040 | 0.4644 | 0.0778 | 4.0e-4 |

全部 rollout 指标（含 4 horizon × 4 class × 6 变量）见 `results/rollout_five_groups.json`。逐 seed H8 cex jv：G1 1.62/1.60/1.67/1.58/1.59；G3 1.61/1.67/1.67/1.61/1.54；G5 1.66/1.56/1.67/1.59/1.60——组间不可区分。

## 5. G5 辅助任务评价（反标准化，TEST）

| seed | qfrc_constraint RMSE | MAE | Free | Contact-ex | Impact |
|---|---|---|---|---|---|
| 0–4（均值） | **192.6** | 7.7 | 13.0 | 27.7 | **1461.8** |

参照：qfrc_constraint 的"预测零"基线 RMS = 140.3（全数据）、Impact 域 1137。**辅助头在 TEST 上比预测零更差**（总 RMSE 192.6 > 140.3；Impact 1462 > 1137）——privileged 辅助任务自身未泛化到未见 episode，更谈不上迁移。这与可行性报告的机制判断一致：qfrc_constraint 由接触几何与求解路径决定，(q,v,ctrl) 输入不含判定接触状态所需的足够信息（81 维输入无接触标志）。

## 6. 科学问题回答

1. **G3 是否改善 Contact-exclusive？** 否（176.30 vs G1 176.50，差 0.1%，seed 方差内；逐 seed 2/5 胜）。修复后 G3 是一个干净的"质量加权损失"测试，结论与 01 版（受 bug 污染）一致。
2. **G4 是否改善？** 否（179.55，+1.7%）。
3. **G3 vs G4 哪个更适合？** G3 略好（Free 84.1 vs 89.3、cex 176.3 vs 179.6、方差更小），但两者都不值得引入。
4. **G5 辅助监督是否改善主输出？** 否（cex 177.6 vs 176.5；Free 劣化 78→100）；且辅助任务本身在 TEST 未泛化（§5）。
5. **改善是否迁移？** 无迁移。rollout 差异全部 ≤3%，方向不一致。
6. **Free/Impact 是否退化？** Free：G4/G5 轻中度退化（89/100 vs 78）；Impact：G5 −0.4%、G3 −0.1%，均在噪声内。
7. **单步与多步？** 两类指标结论一致：无改善。
8. **种子一致性？** 所有"无改善/退化"结论 5 seeds 方向一致；个别正向单点（G3-seed4 cex 167.1）不具可复现性。

## 7. 论文可用的结论

> 在相同数据、划分、网络与训练条件下，对 Contact-exclusive 域的三种物理损失变体——质量矩阵加权加速度损失（G3）、全阵动力学残差平方损失（G4）、约束求解器力辅助监督（G5）——均未产生超越基线的改善（差异 ≤2%，seed 方差内）。其中 mass-weighted 损失在修正标签对齐后表现正常（Free 84.1），证明此前实验的异常源于实现错误而非损失形式本身有信息。约束求解器力的辅助监督在 TEST 上连辅助目标本身都无法泛化（RMSE 192.6 vs 零预测 140.3），表明接触约束力对 (q,v,ctrl) 输入而言本质上是欠定的——缺少接触状态信息是结构性缺口，不是损失设计问题。

**机制层面的推论（证据充分）**：三个损失全部以"重加权同一监督信号"或"增加同源 privileged 标签"为手段，均无效；结合 ROLLOUT-01 中 Free 域收益与约束域精确对应的现象，contact 域改善的结构性前提是让模型在输入层面获得接触状态/几何信息（如接触标志、efc 分解标签的采集支持）——这属于数据与输入表示问题，不是损失形式问题。

## 8. 交付物

| 项 | 路径 |
|---|---|
| 修复版训练代码 | `contact_loss_ablation_02/train_ablation2.py` |
| 统一评价代码 | `contact_loss_ablation_02/evaluate_ablation2.py` |
| 15 次 run 产物 | `contact_loss_ablation_02/runs/<run>/`（best.pt/config/history/keep_indices.npy） |
| 加速度指标（25 行含 G1/G2） | `contact_loss_ablation_02/results/accel_metrics.{csv,json}` |
| G5 辅助任务指标 | `contact_loss_ablation_02/results/g5_aux_metrics.json` |
| 五组 rollout 全表 | `contact_loss_ablation_02/results/rollout_five_groups.json` |
| 新 rollout 逐窗口原始结果 | `contact_loss_ablation_02/rollout_raw_<run>.npz`（15 个） |
| 修正说明 | 本报告 §1 |

复现：`python train_ablation2.py --group {g3|g4|g5} --lambda_dyn 1 --seed {0..4}`；`python evaluate_ablation2.py`。范围遵守：未重训 G1/G2、未改数据/划分/物理参数/残差定义、未引入新损失组。
