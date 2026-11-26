# K1_PINN_STAGE1_02 — GPT Independent Technical Review

**Review date:** 2026-11-24  
**Scope:** Stage1-02 execution, metric recomputation, statistical evidence, and conclusion wording only.

## A. 总体判定

**PASS WITH QUALIFICATIONS**

20 次训练及其配置、checkpoint、history、原始数据和评价实现总体一致；从全部 20 个 `best.pt` 独立重算的指标与保存结果吻合，最大绝对差约 `2.4e-4`。Free residual 的输入项、作用 mask 和 train-only normalization 均经核查，Oracle 检验通过。λ=1 对 MLP 的 Free RMSE、Overall MAE、Free residual 在当前 TEST 上确为 5/5 配对改善。

但是报告需要三类实质限定：(1) TEST 只有 episode 级隔离，并非 action-type 不相交，不能据此宣称未见动作泛化；(2) 5/5 是同一 6 个 TEST episode 上的跨随机初始化稳定趋势，不是统计显著性的证明；(3) 部分措辞夸大了领域对应关系、λ=10 的变化以及 episode 结论。无需重做 20 次训练，也无需重新采集数据。

## B. 实验执行正确性

### 已确认

- `stage1_runs` 中确有 20 个独立 run，分别覆盖 MLP、PINN λ=0.1/1/10 与 seed 0–4；每个目录均含 `best.pt`、`config.json`、`history.json`。
- 20 组配置完全一致：`81→3×256→28`、159,772 参数、AdamW、lr `1e-3`、weight decay `1e-5`、batch 256、最多 2000 epoch、patience 200、grad clip 1.0、冻结 10σ 训练/early-stopping 过滤。PINN 的 λ 是预期研究变量。
- 训练实际使用 15,978 个样本（原 train 16,167，排除 189 个 >10σ 样本）、2,704 个 Free 训练样本；20 run 的划分、normalization 文件和过滤数一致。
- `best.pt` 可被现有评价器重新加载；保存结果与 checkpoint 重算结果一致。history 和权重均未见 NaN/Inf，所有 run 正常早停；best epoch 为 2–47。
- 物理项确为 `M(q) @ a_pred + qfrc_bias - qfrc_actuator`，仅在 `(ncon == 0) AND (nefc == 0) AND ||qfrc_constraint||∞ == 0` 的样本计算。见 [stage0_common.py](tools/pinn/stage0_common.py) 和 [train_stage0.py](tools/pinn/train_stage0.py)。Oracle raw-data 复核通过：float64 nRMS `8.64e-8`，float32 训练路径 `1.79e-7`。
- 测试指标全部使用 3,158 个 TEST 样本；训练/early stopping 不读取 TEST。normalization 的 `mu_x/std_x/mu_a/std_a/s_r` 仅由 TRAIN 计算。Mean baseline 合法地使用 TRAIN 的 `mu_a`，但其名称 `zero_accel_baseline` 不准确：它预测的是训练集均值加速度，不是零加速度。

### 配对证据的准确含义

相同 seed 并非仅有编号相同。实现中先以该 seed 设置 `torch.manual_seed`/`numpy` seed 并实例化网络，随后以相同 seed 驱动独立的 `torch.Generator` 生成 epoch permutation。因此，在同一 seed 下，MLP 与各 PINN 名义上共享初始参数和 minibatch 顺序。物理项和不同 λ 会改变更新，且没有保存初始 checkpoint 或固定 CUDA 确定性配置，因此不应将其解释为逐位可复现的受控因果配对。

更恰当的表述是：同一五组随机初始化及数据顺序下的配对比较。五个 seed 重复的是训练随机性，不是五个 TEST 数据集；TEST episode 只有六个，且每个 seed 都重复评价同一组 episode。

### 可信度问题

- **没有保存 test predictions。** 报告第 4 节声称存在 `test_preds`，但每个 run 只有 config/history/checkpoint。该缺失不改变已由 checkpoint 重算验证的数值结论，只说明交付包不如报告所称完整，并降低直接抽查预测的便利性。
- **“动作不相交”不成立。** episode ID 是隔离的，但 action type 不是：`drop`、`ou_torque`、`hold_throw` 都同时出现在 TRAIN/VAL/TEST；只有 `pd_top_rock` 在 TRAIN/VAL 中缺席。故只能称 episode-disjoint TEST，并可讨论有限的 action-type 外测试，不能宣称广泛跨动作泛化。

## C. 核心结果独立复核

下表为样本×自由度加权、physical-unit 指标。序列顺序为 seed 0–4。括号中的 ±SD 为保存产物所用的**总体标准差**（`np.std` 默认 `ddof=0`），与文中口语“标准差”一致；样本 SD 应分别乘以 `sqrt(5/4)=1.118`。

| 配置 | Free RMSE（五 seed） | Overall MAE（五 seed） | Free residual nRMS（五 seed） | Contact RMSE（五 seed） |
|---|---|---|---|---|
| MLP | 84.230, 83.031, 71.742, 74.103, 78.709；**78.36±4.86** | 52.650, 50.937, 53.812, 50.329, 53.296；**52.20±1.35** | 1.0198, 1.0405, 0.8735, 0.9862, 1.0362；**0.9912±0.0619** | 174.079, 173.858, 184.768, 172.842, 176.962；**176.50±4.35** |
| PINN λ=0.1 | 74.504, 76.261, 82.133, 67.231, 76.233；**75.27±4.78** | 50.958, 51.059, 50.723, 49.183, 50.112；**50.41±0.69** | 0.8632, 0.7763, 0.7261, 0.6823, 0.8053；**0.7706±0.0626** | 177.266, 176.068, 180.510, 172.445, 174.087；**176.08±2.76** |
| PINN λ=1 | 65.567, 72.174, 66.399, 72.446, 78.014；**70.92±4.55** | 50.617, 50.683, 50.641, 50.173, 50.616；**50.55±0.19** | 0.6307, 0.6778, 0.5830, 0.6181, 0.6597；**0.6339±0.0330** | 185.590, 174.829, 178.208, 171.979, 172.434；**176.61±5.01** |
| PINN λ=10 | 65.506, 74.904, 73.475, 75.518, 81.664；**74.21±5.18** | 48.891, 50.272, 51.124, 48.736, 51.265；**50.06±1.07** | 0.5976, 0.5589, 0.5960, 0.5772, 0.6346；**0.5929±0.0252** | 178.952, 174.918, 181.121, 170.721, 175.832；**176.31±3.56** |
| Train-mean baseline | **121.93** | **44.18** | **0.9531** | **175.99** |

样本 SD 对照：Free RMSE 分别为 `5.44 / 5.34 / 5.08 / 5.79`；Overall MAE 为 `1.51 / 0.78 / 0.21 / 1.20`；Free residual 为 `0.069 / 0.070 / 0.037 / 0.028`。

### 配对结果

| 指标（PINN−MLP） | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|
| Free RMSE | −9.73, −6.77, +10.39, −6.87, −2.48；4/5 | **−18.66, −10.86, −5.34, −1.66, −0.69；5/5** | −18.72, −8.13, +1.73, +1.41, +2.96；2/5 |
| Overall MAE | −1.69, +0.12, −3.09, −1.15, −3.18；4/5 | **−2.03, −0.25, −3.17, −0.16, −2.68；5/5** | −3.76, −0.67, −2.69, −1.59, −2.03；5/5 |
| Free residual | 全部为负；5/5 | −0.389, −0.363, −0.291, −0.368, −0.377；5/5 | −0.422, −0.482, −0.278, −0.409, −0.402；5/5 |
| Contact RMSE | 3/5，均值 −0.43 | 3/5，均值 +0.11 | 3/5，均值 −0.19 |

λ=1 Free RMSE 的组均值从 78.36 降至 70.92（均值对均值为 **−9.5%**），但 seed 配对的相对改善为约 0.9%–22.2%，中位数约 7.4%。seed 0 占总绝对改善的约 50%，seed 0/1 合计约 79%；因此平均效果幅度有明显 seed 异质性，但改善方向在五组均为正。

### 统计口径与样本

- 指标确为 sample×DOF 加权，未混用 episode macro。`macro_RMSE` 则是六个 episode RMSE 的非加权均值，二者不可互换。
- TEST 子集计数：Free 1,028、Contact 2,076、Impact 54、≤5σ 3,062。≤5σ 排除 96 个样本，但仍含 12 个 Impact 样本。
- 54 个 Impact 样本仅占 TEST 的 1.7%，却贡献约 85% 的 Overall RMSE 平方误差，因此不能仅用 Overall RMSE 判断所有域的变化。
- λ=10 的 ≤5σ MAE 为 `31.01`，在模型组中最低；相应基线为 `24.76`。故“最佳”只能指四个已学习模型之间，而非优于 baseline。

## D. 主要结论逐条判断

### 1. “PINN 真正改善了动力学预测。”
**有条件成立。** λ=1 确实改善当前 TEST 的 Free RMSE、Overall MAE 和 Free residual；但 Overall RMSE 未改善，Contact 持平，Impact 略差。可保留为“在 Free-Dynamics 子域改善了加速度预测和物理一致性”，不能无限定地表述为整体动力学预测改善。

### 2. “改善域限定在无约束状态。”
**有条件成立。** Free 子集改善明确；Contact 均值仅在约 ±0.1 内波动，方向混合；Impact 的均值 RMSE 分别比 MLP 差约 0.3%–0.6%，并且 PINN 的各 seed 都略差。证据支持“显著收益集中于 Free，域外没有证据获得同等收益”，但不支持“域外绝对零变化”。

### 3. “λ=1 首次跨 5 种子稳定复现。”
**第一部分有条件成立，历史“首次”证据不足。** 当前实验三项指标均 5/5，方向稳定；但 Free-RMSE 改善幅度很不均匀，Overall-MAE 的两个改善值仅 0.16/0.25。若“首次”意味着跨阶段历史比较，现有评审边界下未建立该优先序；应写“在 Stage1-02 的五组 seed 中稳定同向”。

### 4. “λ=10 的 Stage0 旧病理在 Stage1 消失。”
**证据不足且归因过强。** Stage1 中 λ=10 的 Overall/≤5σ MAE 与 residual 改善明显，但 Free RMSE 仅 2/5 优于 MLP，尚不能说所有预测精度均稳定。Stage0-D-02 已使用修复 residual，不能将 D-01 错误 residual 混入对比；Stage0 与 Stage1 的数据分布也不同。因此只能说“Stage1 未观察到与其所有指标同时恶化的病理”，不能单独归因于 residual 修复或数据扩充。

### 5. “λ=1 是最佳平衡点。”
**有条件成立。** 它在三项核心指标上取得唯一的 5/5 组合，并具有最低 Free RMSE；λ=10 的 residual、Overall MAE 和 ≤5σ MAE 更优，但 Free RMSE 不 seed 稳定。可将其称为本次三档 λ 中的“平衡选择”，不应宣称通用最优 λ 或已求得最优区间。

### 6. “改善域精确对应物理损失作用域。”
**有实验支持的关联，未证明严格因果。** 代码确实只对 Free mask 施加损失，且收益集中于 Free；但没有干预、反事实域或跨 mask 训练实验足以证明收益的边界完全由该 mask 因果决定。论文应使用“与物理损失作用域一致/高度吻合”。

### 7. “跨动作泛化是真实的但选择性的。”
**当前措辞不成立，需要纠正。** 三类 TEST action（drop、ou_torque、hold_throw）在 TRAIN 中均出现，不能当作未见 action；只有 `pd_top_rock` 三个 TEST episode 没有同 action 的 TRAIN/VAL episode，而其中结果并不一致。支持的是 episode 级外推、已见动作家族中另一 episode 的泛化，以及非常有限的 `pd_top_rock` 状态外测试；不是一般性的跨动作泛化。

### 8. “整体动力学没有改善，论文应限定在 Free-Dynamics 域。”
**核心限定方向正确，但措辞需精确。** Overall RMSE 与 baseline/MLP 差异很小且模型略差，Contact 持平；但 Overall MAE 确实改善，Free RMSE 大幅改善。因此应说“全 TEST 的 RMSE 未改善，且收益并非覆盖全部动力学状态；可强陈述的优势限于 Free-Dynamics 和 MAE 型指标”，而非“整体动力学完全没有改善”。

## E. 必须修正的问题

### BLOCKER

无。现有数值和实验执行足以支持限定后的核心结论，不需要重跑 20 次。

### MAJOR

1. **删除“TEST 动作不相交/未见动作泛化”表述。** 实际仅是 episode 级隔离；drop/OU/hold_throw 的 action type 在 train/val/test 中重叠。只保留 episode-level holdout 和有限的 `pd_top_rock` 外动作族证据。
2. **纠正 episode 唯一性结论。** `hold_throw_h0` 不是唯一优于 train-mean baseline 的 episode；`pd_top_rock_s76` 上所有五-seed 均值模型组 RMSE 230.8–232.8，也优于 baseline 233.82。`hold_throw_h0` 可作为窄义整体 RMSE 略优的例子，不可作为“唯一模型优于 baseline 的动作域”。
3. **收窄 λ=10 结论。** 物理一致性 5/5，Overall/≤5σ MAE 改善，但 Free RMSE 只有 2/5；不能概括为“精度退化病理消失”或精度全面稳定。更不能将其变化归因于数据扩充或 residual 的某一个因素。
4. **将“改善域精确对应”改为“与物理损失作用域高度一致”。** 当前证据是关联和实施域匹配，不是严格因果识别。
5. **不要把 5/5 写成统计显著。** 它是五个训练重复在同一六 episode TEST 上的稳定同向证据。可在描述性层面说“重复实验稳定”，不能推广为五个独立测试集，也不宜据 n=5 构造强显著性结论。

### MINOR

1. 明示保存汇总使用总体 SD；若论文惯例使用样本 SD，按上述 `sqrt(5/4)` 修正并注明口径。
2. 将 baseline 名称改为 `train-mean-acceleration baseline`；它不是零加速度。
3. 报告声称保存 `test_preds`，实际缺失。可由 checkpoint 再生成；若要保持交付可审计，应补产物或删除该声明。
4. “top_rock 干净跟踪段”并未在评价代码中定义独立 segment。现有结果是整个 `pd_top_rock` 和 `pd_top_rock_s77` episode RMSE；可结合 stable-support 标签讨论，但不能把它们称为专门切出的干净跟踪段。
5. 梯度比只能作为训练诊断。实际定义是固定前 1024 个过滤后训练样本上的 `||∇(λL_free)||₂ / ||∇L_data||₂`；数据梯度覆盖全部 1024，物理梯度只覆盖其中 129 个 Free 样本；采样点在 epoch 更新后、不会应用实际 1.0 clip。λ=0.1/1/10 的 early 比值约 `4/23/179`，at-best 约 `4/26/159`，最大值约 `9/48/471`。这与趋势一致，但三档 λ、五个 seed、单一探针且不控制模型状态的差异，不足以建立治理硬阈值。

## 领域与泛化的具体核查

- **hold_throw_h0：** 384 样本，其中 335 Free、42 Contact、7 Impact。其整体 RMSE 仅由 baseline 291.53 降至 MLP 288.82 / λ=1 288.33；Free 子集从 baseline 68.04 降至 MLP 53.99 / λ=1 47.80，才是较明确的改善来源。Impact 子集约 2095–2102，未改善。该 episode 不能代表整个动作域。
- **pd_top_rock_s76：** 450 样本中 368 为 Free，Free RMSE baseline 39.51、MLP 25.96、λ=1 18.07；Contact 则从 baseline 90.01 变为 MLP 73.61、λ=1 111.08。该 episode 说明 Free/Contact 的方向可相反，而非整体一致改善。
- **pd_top_rock / s77：** baseline 的完整 episode RMSE 为 36.24/47.15，λ=1 为 51.89/59.26。现有评价支持 baseline 在这些以稳定支撑为主的 episode 上更好，但不支持称已完成“干净段”特定切片。
- **Impact：** 总计 54 个，分布于所有六个 TEST episode，最多为 `ou_o0` 的 33 个。Impact RMSE：baseline 2745.96、MLP 2771.74、λ=0.1/1/10 为 2780.00/2788.30/2782.67；幅度差仅约 0.3%–1.5%，不能仅凭较大 RMSE 断言模型完全没有学到冲击动力学，但可以断言本实验未显示冲击域改善。

## F. 最终可接受的论文结论

在 Stage1-02 的 episode-level holdout TEST 上，相比使用同一数据划分、归一化、结构和优化设置的 MLP，Free-Dynamics PINN 以 λ=1 使 Free 子集 RMSE 降低约 9.5%，并使 Free residual 降低约 36%；Free RMSE、Overall MAE 和 Free residual 的改善在当前五组共享初始化/数据顺序的 seed 中均为 5/5 同向，说明这是在该测试集上相对稳定的重复实验趋势。收益集中于物理损失直接覆盖的 Free 样本；Contact RMSE 无实质改善，Impact RMSE 略差，Overall RMSE 未超过 MLP 或 train-mean baseline。该结果支持 Free-Dynamics 预测精度和物理一致性在当前数据上的有意义改善，以及这种改善与物理损失作用域高度吻合，但不能仅凭五个 seed 断言统计显著、通用 λ 阈值、严格因果的域边界或全面未见动作/未见物理分布泛化；现有 split 只是 episode-disjoint，并非 action-type-disjoint。
