# K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_01 — GPT 独立技术评审

## 1. 总体判定

**判定：PARTIAL PASS。**

G3/G4 的底层数学算子和 M 的全矩阵重建是正确的；15 个 run 均有 history、checkpoint 和 rollout 原始产物；TEST 评价在统一样本与 mask 上可复核，原始指标基本正确。但训练管线存在两个影响核心结论的 **P1** 问题：

1. G3/G4 在 10σ 过滤后没有同步过滤 `tr["qacc"]`，物理损失的 `e` 将过滤后的输入预测与未过滤目标错位配对；
2. G5 计算了 TRAIN 的 `mu_g/std_g`，但训练和验证均未使用，实际优化的是未经标准化的 qfrc_constraint MSE。

因此，本任务的 15 次运行真实发生了，但 G3/G4 的结果不是规定损失定义下的有效实验，G5 的结果也不是报告所声明的 z-scored auxiliary loss。报告的主要科学结论不能按现状通过。

## 2. G3/G4/G5 数学实现

### G3

[loss_g3](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:67) 使用 `einsum("bi,bij,bj->b", e, M, e).mean()`，完整实现了逐样本 \(e^TMe\)，没有对角近似，转置与 reduction 正确；M 从 `M_upper` 重建为对称 28×28 全矩阵。

但训练组合并不正确。过滤代码只替换了 `tr_x/tr_y` 和 `tr["M"]` 等项，却未替换 `tr["qacc"]`；物理误差仍以 `tr["qacc"][idx]` 计算。见 [训练数据过滤](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:116) 和 [G3/G4 误差构造](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:205)。独立核查显示 15,978 个保留样本中 15,363 个（96.15%）的物理目标不是对应输入目标。

**回答：数学原语正确，但实际 G3 训练未正确实现任务定义的 `e = a_hat-a_true`。**

### G4

[loss_g4](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:72) 先做 `M @ e`，再平方求和，确实实现了 \(\lVert Me\rVert^2=e^TM^2e\)，并与 G3 明确不同。

然而它继承了与 G3 相同的 `qacc` 目标错位，故其训练结果同样不是规定 G4 实验的有效结果。

**回答：数学原语正确；实际 G4 训练受同一 P1 目标错位问题影响。**

### G5

[AuxAccelMLP](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:43) 的共享主干为 81→3×256，主、辅头均为 28 维。真值只进入 loss，不进入主干或主头；普通 `forward` 只返回主 qacc，评价也只消费主输出。未发现输入泄漏。

但 [G5 统计量计算](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:132) 后从未应用于标签。训练直接使用 `mse_loss(aux, raw_qfrc_constraint)`，验证也同样如此，见 [G5 训练损失](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:205) 和 [G5 验证损失](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/train_ablation.py:248)。这与报告及 config 声明的 TRAIN z-score 不符。

**回答：双头数据流正确，但损失归一化和 λ=1 的实验定义未正确实现。**

## 3. 数学自检

- 对称误差 0、G3/G4 的蛮算差量级可接受，且与浮点精度一致。
- `--verify` 确实调用了真实的两个 loss 原语，但只使用随机 `e`；没有覆盖物理误差构造、10σ 过滤后的索引对齐、scale 的组合应用或训练 batch。
- 自检只有输出打印，没有断言/阈值失败机制。
- G5 只检查了输出维度和静态头部结构，完全没有验证 `mu_g/std_g` 是否真正作用于标签。

因此，两项 P1 问题都被现有自检漏掉。浮点末位差异本身不是问题，但报告将数学原语通过误表述为整条训练实现通过。

## 4. 训练公平性

### 已确认公平或一致的部分

- 15 个目录均包含 `best.pt`、`config.json`、`history.json`；每个 history 覆盖实际 early stopping 长度（G3 201、G4 206–228、G5 206–214 epoch）。
- 均为 G3/G4/G5 × seed 0–4、λ=1，TRAIN/VAL split 和 10σ mask 的来源一致。
- 主干、batch=256、AdamW、lr=1e-3、wd=1e-5、clip=1.0、patience=200 与 Stage1-02 一致；G3/G4 参数量均 159,772。
- `L_acc` 本身和 Stage1-02 一致；未发现 TEST 参与归一化。
- G5 参数增加为 7,196（159,772→166,968），仅是 28 维线性辅助头，额外容量有限；这不应被解释为“完全相同参数量”，但可接受为任务定义的辅助架构。

### 使对照失效的部分

**P1-1：G3/G4 训练目标错位。** 这不是小实现差异，而是物理损失与 L_acc 使用不同 `qacc` 配对。验证物理损失却使用正确过滤后的目标，形成 train/val 语义不一致。

**P1-2：G5 未归一化辅助标签。** 保留 TRAIN 数据上 raw target element-MSE 约 1692.37，而 z-scored target 为 1；各 DOF 标准差约 0.416–150.24。实际 G5 auxiliary loss 在 best checkpoint 附近仍约 680–720，训练后期约 366–398，明显压倒 `L_acc`（约 0.08–0.27）。所以该实验不能判断“统一 λ=1 的 z-scored solver-force supervision”。

另外，G3/G4 的物理项在首个 epoch 分别约 12 和 16，而 L_acc 约 0.15；G5 更高。即使消除实现错误，λ=1 的三个组也不会自动获得相近的有效梯度贡献。现有代码未记录分项梯度范数，报告也不能据此声称各组已充分尺度对齐。

## 5. TEST 结果复核

- TEST 为 3,158 个样本，Free / Contact-exclusive / Impact 分别为 1,028 / 2,076 / 54；三个 mask 互斥、定义一致。
- 独立从 15 个新 checkpoint 重算，CSV 中 G3/G4/G5 的 Free、Contact-exclusive、Impact、Overall 与现有产物一致到报告精度。
- ≤5σ 由逐样本 `max(|(a_true-mu_a)/sigma_a|) <= 5` 再跨元素 RMSE；定义清楚。
- 报告均值与 std 是先逐 seed 计算 RMSE，再在 seed 间统计；std 为 population std。5 个 seed 无缺失。
- 25 个 G1/G2/G3/G4/G5 rollout 原始文件均为 3,110 个完全相同窗口（`ep_idx/t0` 哈希一致），数据有限。独立由 raw NPZ 复算的 H=8 关键结果与 summary 一致。

评价本身未发现把窗口视作独立 seed 统计的问题：报告至少以 seed 为训练重复汇总。但 [evaluate_ablation.py](/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics/tools/pinn/contact_loss_ablation/evaluate_ablation.py) 只产出新 raw rollout，并未记录 `rollout_five_groups.json` 的聚合来源；该汇总缺少脚本内可追溯元数据，属于 P2 证据链不足。

另有一个 P2 导入键错误：G1/G2 baseline 源字段为 `le5sigma_RMSE`，评价代码读取不存在的 `band_le5_RMSE`，导致合并 CSV 中 G1/G2 ≤5σ 为 NaN。报告表中的 G1/G2 ≤5σ 数值可从 Stage1-02 JSON 独立核实，但并非由当前合并脚本正确产出。

## 6. G3/G4/G5 结果解释

### G3

- 现有五个 best_epoch=0、Free RMSE 接近均值基线、Free 预测几乎无相关性这一组事实可以从产物复核。
- 但当前物理损失配错了目标，不能将此失败因果归于“质量矩阵加权造成收缩”。
- G3 预测方差确实很小、误差接近均值解，支持“输出收缩”的描述；Impact 1.2%左右改善与 2.8% rollout 数值改善只是伴随现象，且可能仍是随机/尺度效应。
- 没有梯度分解、预测幅度按 DOF 对照或消除错位后的重复，因此报告中“已证实的收缩副效应因果”过强，应降级为相关性解释。

### G4

- G4 的数学原语正确，但训练目标错位使五 seed 的 Free 和 Contact-exclusive 结果不能支持关于正确 G4 损失的科学结论。
- Contact-exclusive seed 值为约 169.1–190.8，区间与 G1/G2 重叠且 n=5；不能声称统计显著。
- 即使不考虑 P1，物理损失随训练并不稳定下降，在部分 seed 的 best checkpoint 仍接近首轮水平。

因此，“当前正确实现的 G4 在本任务条件下无收益”**不成立**；只有“当前含错位缺陷的运行未观察到稳定收益”成立。

### G5

- TRAIN auxiliary loss 从约 1300 降到约 370，说明网络会降低 raw-force MSE；但这不足以证明 TEST 辅助任务已良好泛化。
- 按当前主干/标签尺度独立评估五个 best checkpoint，其 TEST raw-force RMSE 约 192.4–192.7，几乎与“恒预测 TRAIN 均值”的 192.9 相同；每个维度 z-score 后 RMSE 约 4.6。报告所称辅助任务“合理拟合”缺乏 TEST 证据。
- 五个 seed 主任务方向一致变差，足以支持“此次实际 raw-MSE λ=1 运行发生负迁移/竞争”的观察。
- 但因为真实运行未进行 z-score，不能支持对“solver-force auxiliary supervision（按报告定义）未迁移”的结论，也不能推广到权重或共享方式不合适之外的情形。

## 7. 统计与证据边界

- 五个 seed 是训练随机性重复，而不是五组独立 TEST 数据；TEST episodes 和 rollout 窗口完全共享。
- 相关 TEST 窗口存在时间重叠，不能将 3,158 个样本或 3,110 个窗口当作独立有效样本量计算显著性。
- 方向一致性不等于统计显著性；报告总体注意到这一点，但仍在“排序”和“收缩效应”措辞中部分过度解释小差异。
- Contact-exclusive 的 G1/G2/G3/G4/G5 差异与 seed 方差重叠；无法据此建立稳定排序，尤其不能把 G1≈G2≈G3>G4>G5 当作统计成立的排序。
- Overall、Free、Contact-exclusive、Impact 不能互换；Impact 单步和 rollout 的小幅改善不能证明接触物理更好。

任何修正后的结论也仍必须限定于当前 K1 模型、Stage1 数据、网络、训练配置、指定 TEST episodes、五个 seed，以及明确的 λ=1。报告在这一边界意识上方向正确，但由于两项 P1 实验实现偏差，当前限定的结论仍不可靠。

## 8. 必须修正的问题

| 优先级 | 问题 | 影响 |
|---|---|---|
| **P1** | 过滤后未更新 `tr["qacc"]`，导致 G3/G4 预测与目标错位 | G3/G4 训练及归因结论无效 |
| **P1** | G5 的 `mu_g/std_g` 未用于 auxiliary 标签、预测解码或验证 | 任务未测试声明的 z-scored λ=1 辅助监督，G5 公平性和迁移结论无效 |
| **P2** | 自检只验证算子/shape，未检测过滤对齐与完整 normalization | 两项关键缺陷都可逃过验证 |
| **P2** | λ=1 未经实际梯度量级证实等价，报告又将相近贡献写得偏强 | 容易把 loss-scale/优化问题误归因于物理形式 |
| **P2** | G1/G2 ≤5σ 合并键错误；rollout summary 缺少生成/聚合脚本 | 原始数据大多可靠，但完整证据链不自洽 |
| **P2** | G3 收缩因果、G5 auxiliary 拟合、Contact 排序等表述强于证据 | 需改为相关性/未观察到收益，并修正结论对象 |

本次评审不建议修改已有 checkpoint 或重做大规模实验；最低限度的报告修正应明确这些训练不是既定定义下的有效对照，并撤回基于 G3/G4/G5 的因果结论。

## 9. 最终结论

1. **G3 是否正确实现 `e^TMe`？** 算子是；实际训练中的 `e` 不对应相同样本。
2. **G4 是否正确实现 `|Me|^2`？** 算子是；实际训练同样受目标错位影响。
3. **G5 是否正确实现 solver-force auxiliary supervision？** 双头数据流是，但标签归一化缺失，实际实现不是报告定义。
4. **15 次训练是否全部真实完成且可复现？** 运行和产物真实完整；脚本/配置可追踪，但 GPU 完全逐位复现未由现有内容保证。
5. **训练条件是否足够公平？** 基础条件大体一致，但两项 P1 使新增组没有测试任务书所声明的相同条件。
6. **TEST 评价是否使用相同数据、mask 和指标？** 新组以及 rollout 原始结果是；G1/G2 ≤5σ 导入键需要修正，汇总生成需补证据链。
7. **G3 退化及 Impact 改善解释是否有证据？** 收缩现象有证据，质量加权因果没有；Impact 改善不足以解释为物理学习。
8. **G4 无收益结论是否成立？** 对当前错误实现只能说未观察到稳定收益；不能支持正确 G4 的结论。
9. **G5 未迁移结论是否成立？** 对 raw-MSE λ=1 的实际运行成立于“未见主任务收益”层面；对报告声明的 z-scored G5 不成立。
10. **报告是否正确限定 λ=1 的边界？** 已提到 λ=1 和五 seed，但执行偏离了声明的 λ=1 定义，故边界描述不完整且“物理损失无效”的解释过强。
11. **本次任务是否达到原定完成标准？** **否。** 15 次训练已执行且评价产物可复核，但核心损失对照的有效性被两个 P1 问题破坏。

最终评审：**PARTIAL PASS**。