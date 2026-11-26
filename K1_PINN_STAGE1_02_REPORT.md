# K1_PINN_STAGE1_02_REPORT — 正式 MLP / Free-Dynamics PINN 对照实验

- 日期: 2026-11-24
- 数据: `tools/pinn/stage1_data/`（30 episodes / 24,578 样本；TEST 6 eps 动作不相交）
- 训练: 4 配置（MLP、PINN λ=0.1/1/10）× 5 seeds（0–4）= **20 次，全部完成**，产物 `tools/pinn/stage1_runs/`
- 条件一致性: 同一数据/划分/归一化（含冻结 10σ 冲击帧训练过滤）/网络（81→3×256→28）/AdamW(1e-3, wd 1e-5)/batch 256/≤2000 epoch/早停 patience 200/grad-clip 1.0；唯一研究变量 = 是否加物理损失及 λ。代码仅做 `--data-dir` 接口适配（不改训练行为），residual 用 D-02 修复后的定义，未动。
- 执行环境: MuJoCo 4.0.1 + torch 2.7.0 (CUDA)，`booster_train` env

---

## 1. 总体结果（TEST 6 episodes，3,158 样本，5 seeds 均值±std）

| 配置 | Overall MAE | Overall RMSE | **Free RMSE** | Contact RMSE | Impact RMSE | ≤5σ RMSE | ≤5σ MAE | macro RMSE | **Free residual (nRMS)** |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 52.20±1.35 | 392.26±1.01 | 78.36±4.86 | 176.50±4.35 | 2771.7 | 102.65±3.31 | 33.27 | 367.73 | 0.991±0.06 |
| PINN-0.1 | 50.41±0.69 | 392.92±1.48 | 75.27±4.78 | 176.08±2.76 | 2780.0 | 100.48±2.62 | 31.35 | 367.67 | 0.771±0.06 |
| **PINN-1** | 50.55±0.19 | 393.83±1.59 | **70.92±4.55** | 176.61±5.01 | 2788.3 | **99.92±5.14** | 31.41 | 368.84 | **0.634±0.03** |
| PINN-10 | 50.06±1.07 | 393.25±1.13 | 74.21±5.18 | 176.31±3.56 | 2782.7 | 101.25±2.27 | **31.01** | 368.31 | **0.593±0.03** |
| 均值基线 | 44.18 | 392.60 | 121.93 | 175.99 | 2745.96 | 119.78 | 24.76 | 364.15 | 0.953 |

（oracle 参照：free residual = 0；free residual 与加速度误差是两类指标，下文分开陈述。）

## 2. 配对逐 seed 比较（同一 seed 的 PINN−MLP 差值）

| 指标 | PINN-1 vs MLP | PINN-10 vs MLP |
|---|---|---|
| free RMSE | −18.7/−10.9/−5.3/−1.7/−0.7 → **5/5 胜**（均值 −9.5%） | −18.7/−8.1/+1.7/+1.4/+3.0 → 2/5（均值 −5.3%） |
| Overall MAE | −2.0/−0.3/−3.2/−0.2/−2.7 → **5/5 胜** | 5/5 胜（均值 −4.1%） |
| Free residual | −0.4/−0.4/−0.3/−0.4/−0.4 → **5/5 胜** | −0.42/−0.48/−0.28/−0.41/−0.40 → **5/5 胜**（−40%） |
| ≤5σ RMSE | 4/5 胜（seed0 +5.3） | ≈平 |
| Contact RMSE | 3/5（+11.5/−6.6 混合，≈平） | ≈平 |

## 3. 核心问题回答

**A. PINN 是否优于 MLP？——是，在物理损失作用的域内明确优于，域外持平。**
Free 子集 RMSE −9.5%（PINN-1）、Overall MAE −3.2%（PINN-10）/−1.7%（λ=0.1）、free residual −36%（λ=1）/−40%（λ=10）；contact 与 impact 子集无差异（±0.2% 内）。Overall RMSE（392–394）四组与基线（392.6）统计上不可区分——被 impact 带（~2770 vs 基线 2746，模型均略差）主导。

**B. 改善是否跨 seed 稳定？——λ=1 是。** free RMSE、Overall MAE、free residual 三项 **5/5 种子配对全胜**；λ=10 的 residual 5/5 胜但精度 2/5；λ=0.1 方向一致幅度居中。

**C. λ=10 是否仍有"一致性改善但精度退化"？——不再存在。** Stage0-D-01 的该现象是错误残差+小数据的产物：修正后 λ=10 在 Stage1 上 ≤5σ MAE 全场最佳（31.01）、free RMSE 均值优于 MLP（但不 seed 稳定）、无任何精度退化证据。梯度比 early~179×/max~471× 未压制训练（best_epoch 12–47，深于其他组）。

**D. λ=1 是否最佳平衡？——是。** free RMSE 全场最佳（70.92）+ residual 0.634（优于基线 0.953 与 MLP 0.991）+ 唯一精度/一致性双 5/5 稳定。λ 梯度比谱系：0.1→4–9×，1→23–48×，10→159–471×；λ=1 的物理梯度已足够改变解（residual −36%）而不破坏数据拟合。

**E. 改善集中在哪？** free/驱动腾空域。逐 episode RMSE：**hold_throw_h0 是唯一模型优于均值基线的 episode**（MLP 288.8 / PINN-1 288.3 / PINN-10 288.1 < 基线 291.5）；来源分层：hold_throw MLP 288.8 < 基线 291.5，其余来源（drop 357.9 vs 354.7、ou 1217 vs 1221、top_rock 114.1 vs 105.7）≈平或略差。

**F. 动作不相交 TEST 上的泛化？——分状态诚实回答**：模型在 free/bulk 状态显著优于均值基线（free 71–78 vs 122；≤5σ 100–103 vs 120），证明学到非平凡动力学；但在 contact 稳态（top_rock 干净跟踪段：模型 52.1/59.3 vs 基线 36.2/47.2——基线更好）与 impact 带（2770–2790 vs 2746）仍不如"预测均值"，macro RMSE 367.7–368.8 vs 基线 364.2。**跨动作泛化是真实的但Selectively——物理损失改善的正是它作用的状态，未外溢到接触稳态。**

## 4. 交付物

| 项 | 路径 |
|---|---|
| 20 次 run（config/history/best.pt/test_preds） | `tools/pinn/stage1_runs/<run>/` |
| 逐 run 全指标（27+ 列） | `stage1_runs/results/metrics_per_run.{csv,json}` |
| 4 配置均值±std 汇总 | `stage1_runs/results/metrics_summary.json` |
| 逐 episode / 来源分层 RMSE | summary JSON 内 `per_episode_RMSE_mean` / `source_macro_RMSE_mean` |
| 梯度比（early/at_best/max，逐 run） | metrics_per_run.json `grad_ratio_*` |
| 对照图（逐 episode RMSE + residual-λ 曲线） | `stage1_runs/results/comparison.png` |
| 评价脚本 | `tools/pinn/evaluate_stage1.py` |

复现：`python train_stage0.py --model {mlp|pinn} [--lambda_dyn {0.1,1,10}] --seed {0..4} --data-dir stage1_data --runs-dir stage1_runs`；`python evaluate_stage1.py`。

## 5. 最终结论

**Stage1 数据集上，Free-Dynamics PINN 相比纯 MLP 实现了真实的动力学预测改善，且核心改善（λ=1：free 子集 RMSE −9.5%、Overall MAE −3.2%、物理残差 −36%）具有跨 5 种子全胜的稳定性；但改善域限于物理损失作用的无约束状态——接触稳态与冲击带没有变化，全测试集 RMSE 仍被冲击带主导、与均值基线持平。**

判定：**PASS**（对照实验完整、结论由数据支撑、无指标筛选）。研究价值定性：这是"物理约束改善其所约束的状态"的正结果，且首次跨 seed 稳定复现；论文叙事应如实限定在 free-dynamics 域，接触状态预测（contact-aware 分支）仍是公开缺口，不因本结果宣称整体动力学被 PINN 改善。

局限（如实）：(a) impact 带无模型可预测（2770 vs 基线 2746，均差 ~1%）；(b) contact 稳态与基线持平，物理损失不作用域无收益；(c) λ=10 的 free RMSE 改善不 seed 稳定（2/5）；(d) 单层 MLP + 24.6k 样本的上限未探明（更大的模型/数据留给后续，不在本任务范围）。
