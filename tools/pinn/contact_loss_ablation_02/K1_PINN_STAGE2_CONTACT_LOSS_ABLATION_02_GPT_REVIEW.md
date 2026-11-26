# K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_02 — GPT-6 独立技术评审

评审范围：仅限 Ablation 02 的损失实现、15 次训练、统一 TEST/rollout 评价和相应结论。检查了报告、训练/评价代码、全部 15 组 checkpoint/history/config/keep-index 文件、指标产物、rollout 原始文件，并对核心指标进行了独立复算。

## A. 总体判定

**PASS WITH QUALIFICATIONS**

Ablation 01 的样本错位在 Ablation 02 训练代码中已修复；G3/G4 数学算子、G5 标准化辅助头、15 次训练产物和单步 TEST 预测总体可信。五组 rollout 也可由逐窗口原始产物复算。

但报告有几项必须修正的结论/证据问题：

1. G5 的“零预测基线 140.3”与同一 TEST 样本、28 维 pooled RMSE 定义不符。复算值为 195.20；G5 平均为 192.62，略优于零预测，而不是更差。Contact-exclusive 区域的辅助头误差也明显低于零预测。因此“辅助任务未泛化”不能按报告中的证据成立。
2. 由 G5 辅助预测失败推导输入映射“本质欠定”没有直接证据。
3. G2 的 Overall MAE 报为 52.20，实际 Stage1-02 原始记录均值为 50.55；“G5 Overall MAE 全场最低”也因此不成立。
4. 报告的 TEST 样本数 3,152 与评价代码及指标对应的 3,158 不一致。
5. G3/G4 的训练目标有 TRAIN-only 标量归一化，且物理项梯度明显大于数据项；应把结论限定为当前归一化和 λ=1，而不能暗示 λ=1 使两项贡献相当。

上述问题不需要重新训练；主要是修订报告和补齐索引/指标产物的可追溯性。

## B. 实现修复审查

| 审查项目 | 判定 | 证据与问题 |
|---|---|---|
| G3/G4 统一过滤与样本对齐 | **通过** | [train_ablation2.py](./train_ablation2.py) 先从 `tr_y` 生成一个 `tr_keep`，再用该 mask 同步索引 `X/qacc/M/qfrc_constraint/free`；之后从已过滤的 `tr["qacc"][idx]` 构造物理误差。VAL 也按对应的 `va_keep` 同步过滤标签、M、辅助标签及 Free mask。独立从 TRAIN 数据重建的 10σ 索引与所有 15 个文件中的 int64 索引逐项一致，均为 15,978 个保留、189 个丢弃。故 \(\hat a\) 与 \(a_{\rm true}\) 来自相同原始样本，Ablation 01 的错位已修复。 |
| keep 索引可复核性 | **有条件通过** | 文件 `keep_indices.npy` 内容与重建索引相同，但代码以 `write_bytes(keep_idx.tobytes())` 保存，没有 NumPy `.npy` 头；标准 `np.load()` 会失败。config 的 `keep_index_sha` 实际只是所有索引之和 128235344，不是 hash/checksum。当前索引可用已知 `int64` 从原始字节恢复，但命名和“hash”记录不准确，降低了归档可移植性，不影响已训练模型。 |
| G3 数学实现 | **通过，需注明缩放** | `loss_g3` 使用全矩阵 batch 二次型 `einsum("bi,bij,bj->b",e,M,e).mean()`。`rebuild_M` 从上三角重建完整对称 28×28 矩阵；维度与 28 维 qacc 一致。训练物理误差从归一化主输出反标准化后，与过滤后的物理 qacc 相减。无对角近似或错误广播。 |
| G4 数学实现 | **通过，需注明缩放** | `loss_g4` 实际计算 `M @ e` 后逐维平方求和再对 batch 取 mean，确为 \(\|Me\|^2=e^TM^TMe\)，并未误用 G3。 |
| G3/G4 训练目标中的额外缩放 | **已披露，但结论须限定** | 训练使用 `loss_g3/s3_scale` 与 `loss_g4/s4_scale`，其中 TRAIN-kept 的 \(s_3^2=9694.76\)、\(s_4^2=31805.05\)。因此实现的总目标是相应物理项除以 TRAIN 标量后再乘 λ，而不是任务书所写未缩放的裸公式。统计量仅来自 TRAIN，且缩放不改变各物理项单独的最优点；但会改变其相对 \(L_{acc}\) 的贡献。报告应将实验称为“TRAIN 尺度归一化的 G3/G4，λ=1”。 |
| G5 统计量及辅助头 | **通过** | `mu_g/std_g` 在过滤后的 TRAIN qfrc_constraint 上计算，std 下限为 \(10^{-3}\)（实际最小约 0.416），VAL 共用 TRAIN 统计量。TRAIN/VAL 标签均转换到 z-score。配置保存 28 维统计量；TEST 预测按同一组统计量反标准化后与物理 qfrc_constraint 比较。主干、28 维主头及 28 维辅助头均在 checkpoint 中。普通 `forward()` 只返回 qacc；rollout 只用该主输出，未发现辅助标签输入或测试泄漏。 |

特别说明：M 与加速度误差均以 qacc 的 28 维 generalized-coordinate 顺序输入；损失函数有完整 batch 维处理。所有 15 个 history 中的记录项有限，无 NaN/Inf。

## C. 训练有效性审查

- **15/15 运行存在**：G3/G4/G5 各有 seed 0–4；每个 run 均含可加载 checkpoint、完整 history、配置和索引文件。配置的 λ=1、batch=256、lr=1e-3、AdamW、weight decay \(10^{-5}\)、patience=200、gradient clipping=1、宽度 256、深度 3，与指定条件一致。
- G3/G4 主干为 159,772 参数；G5 为 166,968 参数，增加 7,196 个辅助头参数，主 qacc 网络仍为 81→3×256→28。G5 参数增量与辅助任务相符。
- 每次运行 history 长度为 202–214 epoch，符合记录的 best epoch（G3 5–10、G4 9–13、G5 1–7）和 patience 早停；所有 checkpoint 均能用于评价。G3 已从 Ablation 01 的 best_epoch=0 恢复到正常早停区间。
- TEST 不参与过滤阈值、qacc 输入/输出归一化、s3/s4 或 G5 辅助统计量。
- `L_acc` 继续采用 per-dof z-scored qacc MSE。物理项/辅助项的 probe 梯度范数是在同一模型参数集合上分别反传测得；但这不意味着它们贡献相等。best epoch 上 physics/data 梯度范数比值：G3 约 10.6–12.0，G4 约 22.4–46.9；G5 aux/data 约 3.2–4.4。λ=1 只是相同标量系数，不是相同梯度影响。G3/G4 归一化后物理梯度仍明显占优，Free 域退化不能仅归因于损失表达式本身。

**训练完整性结论：通过。** 就本轮固定的 λ=1 和尺度归一化而言，训练确已完成且具备可复核产物；训练条件大体一致，但各损失项有效梯度贡献并不相同。

## D. 单步评价审查

### 数据、mask 与算法

- 评价代码对全部 TEST 样本使用相同输入、qacc 物理标签、TRAIN 归一化及 RMSE/MAE 计算。`Free = free`、`Contact-exclusive = ~free & ~impact`、`Impact = impact`，互斥且覆盖全体。
- 代码及 Stage1-02 源评价均使用相同的 Contact-exclusive 语义；G1/G2 的 `contact_RMSE` 映射正确，已核实其源函数定义为 `(~free) & (~impact)`。
- 直接检查的 TEST 数量为 **3,158**，其中 Free 1,028、Contact-exclusive 2,076、Impact 54。报告写 3,152，应修正。三个区域的不平衡及 Impact 小样本数须在解释中考虑。
- 新组 15 个 checkpoint 的 CSV 指标由本地重新推理复算一致。均值±std 是先逐 seed 计算 RMSE，再跨五个 seed 汇总，std 为 population std。相同 TEST 样本和 episodes 被所有 seed 共用；这不是五份独立测试集。

### 复核后的五组关键 TEST 结果

单位为 qacc 物理单位；± 为五个训练 seed 间的 population std。

| 组 | Free RMSE | Contact-exclusive RMSE | Impact RMSE | Overall RMSE | Overall MAE |
|---|---:|---:|---:|---:|---:|
| G1 | 78.36±4.86 | 176.50±4.35 | 2771.7±3.8 | 392.26±1.01 | 52.20±1.35 |
| G2 | 70.92±4.55 | 176.61±5.01 | 2788.3±9.8 | 393.83±1.59 | **50.55±0.19** |
| G3 | 84.08±5.03 | 176.30±5.93 | 2768.3±3.6 | 392.18±1.63 | 53.94±1.44 |
| G4 | 89.31±7.81 | 179.55±4.62 | 2767.9±4.7 | 393.48±1.76 | 55.16±1.96 |
| G5 | 99.65±7.49 | 177.60±4.39 | 2760.6±5.8 | 392.82±1.57 | 52.09±1.41 |

G1/G2 Overall 指标来自 Stage1-02 原始 `metrics_per_run.json`，而不是 Ablation 02 合并 CSV（后者只为基线填入 Free/Contact/Impact RMSE）。原始文件证实 G2 Overall MAE 均值为 50.55，而非报告的 52.20；因此报告所称 G5 Overall MAE 52.09“全场最低”错误，G2 才最低。

G3 的 Contact-exclusive 点估计比 G1 低 0.20（约 0.11%），比 G2 低 0.30；G4 比 G1 高约 1.73%，G5 高约 0.62%。这些差异均不能仅凭五个 seed 宣称统计显著。报告写 G3 对 G1 为“2/5 seed 胜”也不准确：按逐 seed 配对值，G3 相对 G1 是 **3/5** 胜；相对 G2 才是 2/5。G3/G4/G5 的“方向 5/5 一致”不适用于 Contact-exclusive 配对结果。

### G3 与 G4 的 episode-level 核对

按六个 TEST episode 分别计算 Contact-exclusive RMSE，再对该 episode 内五个 checkpoint 取均值，G3 仅在 2/6 episodes 低于 G4；G4 在其余 4/6 低于 G3。示例均值（各 episode Contact-exclusive 样本数依次为 863、78、834、72、187、42）：

| Episode | G3 | G4 |
|---|---:|---:|
| pd_top_rock | 57.81 | 55.14 |
| pd_top_rock_s76 | 74.14 | 81.78 |
| pd_top_rock_s77 | 64.57 | 62.14 |
| drop_d0 | 119.24 | 112.13 |
| ou_o0 | 548.49 | 562.73 |
| hold_throw_h0 | 98.34 | 95.96 |

不同 episode 的样本量和误差尺度相差很大；pooled RMSE 的排序会受大 episode/高误差 episode 权重影响。无论 pooled 还是 episode-level 结果，都不支持“G3 稳健优于 G4”；最多可说 pooled 点估计较低、episode 结果不一致。以上 episode 划分仍只有六个 TEST episodes，不能当作大量独立观测做强显著性推断。

## E. Rollout 审查

- 检查到五组共 25 个 raw rollout 文件（含基线原始文件），其 `ep_idx/t0` 完全一致，均是相同的 3,110 个 H=8 有效窗口；由 baseline raw 文件中的起点 mask 复算出的五组 H8 汇总与 `rollout_five_groups.json` 一致。
- 实现每个窗口均由真实 \(q_t,v_t\) 独立初始化；后续按 `v += dt*a_hat`，再对更新后的 v 调用 `mj_integratePos(q,v,dt)`。仅使用记录控制序列，不使用后续真实状态。窗口按 episode 和 t0 构造，不跨 episode 递推。
- root 平移、线速度、角速度分别比较；root 姿态用绝对四元数点积的 geodesic 误差，joint qpos/qvel 使用相应 22 维切片。此处是给定记录控制下的**开环状态预测**，不是闭环控制评价。
- H=1/2/4/8 所有组的 jv all 均值与报告一致：G1 0.436/0.668/1.069/1.797，G2 0.438/0.673/1.082/1.831，G3 0.436/0.668/1.069/1.799，G4 0.438/0.672/1.077/1.811，G5 0.437/0.669/1.071/1.802。
- G4 对 G2 并非各域都“没有任何收益”：all-window jv 的 G4 点估计在 H1–H8 均稍低，但配对 seed 胜数仅 2/5、3/5、3/5、3/5；Free 域 G4 在 H1–H8 均更差，且 H8 为 1.120 vs 1.071；Contact-exclusive H8 为 1.630 vs 1.637，差异很小且配对胜数 2/5。Impact H8 则为 8.369 vs 8.713，G4 在五个训练 seed 上均较低；全体 H=1/2/4/8 的 impact 方向也相同。Impact 起点总共仅 50 个，分布在六 episodes，故这是应报告的域特定点结果，不足以据此宣称跨 episode 的稳定显著收益。

**结论：** rollout 执行方式与对照窗口公平，summary 可由 raw 复算。报告的“差异全在 3% 内、方向不一致”不适用于 G4-vs-G2 的 Impact rollout；应披露其约 4% 的 H8 点改善，同时说明 Free 域退化和小型、共享 TEST 窗口的证据边界。

## F. 研究结论审查

### G3 对 G4

G3 Contact-exclusive pooled RMSE 为 176.30，G4 为 179.55，G3 数值较低；但差距小于组内 seed 波动量级，且 episode-level 排序反转（G3 只在 2/6 episodes 胜）。可以写“本次 pooled 点估计 G3 较低”，不应写成“G3 明确优于 G4”或将差异解释为损失机理优劣。

### G3/G4 对 Contact-exclusive 基线

当前数据、模型和归一化 λ=1 条件下，没有证据显示 G3/G4 带来明确、可重复的 Contact-exclusive 改善。G3 的 0.1% 左右差异应视作近似持平；G4 的点估计偏差约 1.7%，但没有独立 TEST episodes 的显著性分析。可作受限的“未观察到清晰改善”结论，不能泛化到未归一化损失或其他训练权重。

### G4 对 G2

单步 qacc 上，G4 的 Free 和 Contact-exclusive 点估计均比 G2 差；Impact 点估计更低，且五个 seed 均低于 G2。rollout 中 G4 的 Free 持续较差、Contact-exclusive 近似持平，而 Impact H8 有上述同向降低。故“G4 没有超过 G2”只有在限定为**没有全域一致的主任务优势**时成立；若概括为任何域/任何 rollout 指标都无额外收益，则不准确。也没有足够独立 TEST episodes 证明等效或显著差异。

### G5 主任务及辅助任务

主 qacc 方面，G5 Free RMSE 明显高于 G1/G2，Contact-exclusive 约持平但没有明确优势；不能把 Overall MAE 的低值说成 G5 最优，因为 G2 的实际 Overall MAE 更低。

辅助任务方面，`g5_aux_metrics.json` 中 RMSE/MAE 可复核。独立按同一 TEST 样本、28 维 pooled 口径计算：

- 全 TEST \(N=3158\)：零预测 RMSE = **195.20**、MAE = **9.30**；G5 五 seed 平均 RMSE = **约 192.62**、MAE = **约 7.70**。G5 约有 1.3% RMSE 改善，而不是 192.6 大于 140.3。
- Contact-exclusive：零预测 RMSE = **42.98**，G5 平均约 **27.7**。
- Impact（54 个样本）：零预测 RMSE = **1468.79**，G5 平均约 **1461.8**，改善很小；Free 上真值约为零，G5 会引入非零误差。
- 各 generalized-coordinate 尺度高度不均；例如第三个分量的零预测 RMSE 约 982，其余很多维度小得多。不得把按维度宏平均、限定子集或不同样本的 baseline 与 pooled 28 维 RMSE 混用。报告给出的“零预测 140.3 / Impact 1137”无法由当前同口径 TEST 标签复现。

因此，G5 辅助头的绝对 pooled 误差仍高，但当前原始数据并不支持“测试表现比零预测差”或“辅助目标完全没有泛化”。它在 Contact-exclusive 上优于零预测，整体仅有小幅改善且域间不均；此外 checkpoint 按主任务验证损失选择，并非按辅助验证损失选择。合适结论是“本实验未证明辅助监督能稳定改善主 qacc；辅助 qfrc_constraint 预测有一定域别能力，但总体误差大、泛化收益有限且依赖评价域”。

### 输入映射“本质欠定”

现有实验没有构造相同/近似 \((q,v,ctrl)\) 对应显著不同 qfrc_constraint 的配对证据，也没有分析求解器历史状态或映射多值性。qpos/qvel 含有几何与运动信息，ctrl 也可能携带间接状态相关信息；仅“没有显式 contact flag”不能证明输入理论上不足。G5 辅助误差、特别是本次错误的零预测对比，最多支持“当前网络、数据和训练目标下辅助力预测不够准确/收益有限”，不支持“本质欠定”的理论断言。

## G. 论文可用结论

> 在当前 K1 Stage1 episode 划分、81→3×256→28 网络、五个训练种子及 TRAIN 尺度归一化的 λ=1 设置下，G3 与 G4 均未显示出明确且可重复的 Contact-exclusive 单步加速度改善：其 pooled RMSE 分别为 176.30 和 179.55，而 G1/G2 约为 176.50/176.61；差异较小，且 G3/G4 的 episode-level 排序并不一致。G3/G4 的物理项采用 TRAIN-only 标量归一化，probe 梯度范数显示该项相对数据项仍占主导，因此结果仅适用于所测试的具体尺度和权重。G5 的主任务未显示 Contact-exclusive 优势，且 Free 预测误差增加。G5 辅助 qfrc_constraint 预测的全 TEST pooled RMSE 约为 192.6，略低于同一 28 维 pooled 口径的零预测基线 195.2；Contact-exclusive 区域误差低于零预测，而 Impact 区域改善很小。该结果提示辅助预测效果受区域影响，但不能据此认定约束力关于 \((q,v,ctrl)\) 本质欠定。开环 rollout 总体差异有限且域间存在权衡：例如 G4 的 Free 误差高于 G2，但 Impact H8 jv 点估计较低；由于这些结果共享少量 TEST episodes/window，尚不足以声称普遍优势或等效性。

## H. 必须修复的问题

| 问题 | 影响 | 是否阻塞实验验收 | 最小必要修复 |
|---|---|---|---|
| G5 零预测 baseline 140.3/Impact 1137 与同口径 TEST 标签不符；“比零预测差”“辅助目标未泛化”推论不成立 | 影响 G5 辅助泛化结论 | **阻塞该项结论/论文表述，不阻塞已完成训练及主 qacc 对照** | 在报告中明确 pooled 定义和 3158×28 样本范围，改用 195.20/1468.79 的零预测基线及相同 mask；修订为“整体误差高但略优于零预测，区域表现不均”。无需重训。 |
| “qfrc_constraint 对输入本质欠定”及“缺少接触状态是结构性缺口”被写成已证实结论 | 超出实验能证明的范围 | **阻塞该因果/理论结论** | 删除确定性表述，改为当前表示下辅助预测仍有限、输入信息不足仅为待证假设。无需新实验才能保留有限经验结论。 |
| G2 Overall MAE、G5 最低 Overall MAE、TEST 样本数及 G3 paired wins 写错 | 指标与报告叙述不一致 | 不阻塞模型训练结果，但阻塞报告定稿 | 将 G2 Overall MAE 改为 50.55；删除 G5 “最低”表述；TEST 数量改为 3,158；明确 G3 对 G1 为 3/5 seed wins（对 G2 为 2/5）。 |
| “G3 略优 G4”及“G4 无任何超出 G2 的收益”过宽 | 忽略 episode reversal 和 G4 Impact rollout H8 的同向较低点结果 | 不阻塞总体“无 Contact-exclusive 明确改善”结论 | 改为 pooled 数值排序而非显著优劣；注明 G4 Impact rollout 点结果及 Free 域权衡、小样本限制。 |
| keep-index 文件扩展名/格式与 checksum 字段不实 | 归档可移植性和完整性验证不足 | 不阻塞本轮结果，源码掩码可重建 | 将索引用 `np.save()` 正式写成 `.npy`，并把真正的文件哈希与索引和分开记录。无需重训；按当前 keep mask 重存即可。 |
| G3/G4 的物理项缩放及明显梯度占比未在论文措辞中充分限定 | λ=1 易被误解为两项贡献平衡或等同裸公式 | 不阻塞当前归一化版本结果 | 方法和结论明确写出 `Lphys/s3²`、`Lphys/s4²` 及梯度诊断；结论限于当前 TRAIN scale、λ=1。 |

**无需额外重训。** 修正的是报告中的对照基线、统计口径、结论边界及索引归档格式；现有 checkpoint、history、统一指标和 rollout 原始数据足以支撑范围受限的论文结果。
