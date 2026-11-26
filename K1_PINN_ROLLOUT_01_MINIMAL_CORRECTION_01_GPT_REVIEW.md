# K1_PINN_ROLLOUT_01_MINIMAL_CORRECTION_01 — GPT Independent Review

**Review date:** 2026-11-24  
**Scope:** minimal correction of classification, aggregation, error-growth and result wording only.

## A. 总体结论

**PASS WITH QUALIFICATIONS**

修正代码从 10 个原始 `raw_<run>.npz` 重新聚合，成功解决原报告的核心 Contact/Impact 分类错误；三类互斥、覆盖完整，H=8 Joint velocity 三类数值与独立复算一致；误差累积与 Contact-exclusive Root linear velocity 等关键结论也正确。修正未重新训练或重跑 rollout，五项交付物可执行、可追溯且内部数值一致。

仍需一个小范围文字修正：最终报告对 H=8 全窗口 root angular velocity 的胜场在不同段落中分别写作 4/5 和 3/5。代码、JSON 与独立复算均为 **3/5**；geodesic 为 **4/5**。这不改变主要关节状态结论，但应统一，以免论文误引。其余限定不构成阻塞。

## B. 分类与样本数量核验

### Mask 核验

原始 `raw_mlp_seed0.npz` 与其余九个 run 的 `free0/impact0` 完全相同。独立检查结果：

```text
free               = free0
impact             = impact0
contact_exclusive  = (~free0) & (~impact0)
```

| 类别 | Windows | 与其他类交集 | 核验 |
|---|---:|---:|---|
| Free | 1,024 | Free∩Contact=0；Free∩Impact=0 | 通过 |
| Contact-exclusive | 2,036 | Contact∩Impact=0 | 通过 |
| Impact | 50 | Impact∩Free=0 | 通过 |
| Total / union | 3,110 | 三类 union 覆盖全部 3,110 | 通过 |

- `1024 + 2036 + 50 = 3110`，且逐窗口并、交结果验证了互斥完整，而不只是数量相加。
- `free0`、`impact0` 来源于原始 rollout 保存的起点标签；未依据预测误差或模型输出重新分类。
- 同一 mask 对 H=1/2/4/8 使用；原 rollout 每个 horizon 已从相同 H=8-valid 起点读取。
- 50 个 Impact 均属于原报告错误的 `~free` Contact 集，因此原 Contact 并非 Contact-exclusive。
- 分类是 rollout **起点**标签，不代表整个 8-step 窗口保持同一状态。

## C. 指标复算核验

### 聚合实现

[correct_aggregation.py](tools/pinn/rollout/correct_aggregation.py) 直接读取 raw NPZ 的 `jp/jv/rp/rlv/rav/geo`，对选中窗口在指定 step 计算：

`RMSE = sqrt(mean(stored per-window component squared error))`

这等价于先在 window×component 上聚合平方误差再开方；不是平均逐窗口 RMSE。各 seed 和 MLP/PINN 使用完全相同代码路径。

独立复算所有 160 个长表行（10 run × 4 class × 4 horizon），6 个指标均无差异；最大偏差在 `1e-12` 阈值内为 0。CSV 数据行 160，与 JSON `per_run` 一致。

### 关键 H=8 Joint velocity（rad/s）

| category | 原报告 `Contact` 口径 | GLM 修正结果 | GPT 独立复算 | 差异 | 结论影响 |
|---|---:|---:|---:|---|---|
| Free | 1.1072 / 1.0705 | 1.107184 / 1.070523 | 1.107184 / 1.070523 | 0 | 无 |
| Contact-exclusive | 原报告无；混合 Contact 为 2.0521 / 2.1063 | 1.610420 / 1.636974 | 1.610420 / 1.636974 | 0 | 修正必要；结论由“0/5、+2.64%”变为“2/5、+1.65%” |
| Impact | 8.3701 / 8.7130 | 8.370070 / 8.712970 | 8.370070 / 8.712970 | 0 | 无；PINN 劣势 0/5 成立 |

每格为 `MLP / PINN λ=1`。

### 其他核心 H=8 指标

| class | Joint pos MLP→PINN / wins | Joint vel MLP→PINN / wins | Contact/Impact 语义 |
|---|---|---|---|
| Free | 0.0044994→0.0043165；5/5 | 1.107184→1.070523；5/5 | 正确 |
| Contact-exclusive | 0.0076202→0.0077184；2/5 | 1.610420→1.636974；2/5 | 无稳定优势，PINN mean 略差 |
| Impact | 0.0417550→0.0431808；0/5 | 8.370070→8.712970；0/5 | 一致劣势，但 n=50 |

Root geodesic 使用 raw NPZ 已保存的 quaternion geodesic squared error，原 rollout 使用 `2*acos(abs(dot(q1,q2)))`。修正脚本没有改变其定义。

## D. 误差累积分析核验

### Free joint velocity

修正脚本对五-seed mean 的 H1/H8 ratio 计算：

| model | H1 mean | H8 mean | ratio of means |
|---|---:|---:|---:|
| MLP | 0.0882723 | 1.1071839 | 12.5428×（报告 12.54×） |
| PINN | 0.0798747 | 1.0705228 | 13.4025×（报告 13.40×） |

另一个合理但不同的口径——先计算每个 seed 的 H8/H1，再取五-seed mean——得到 MLP 12.5886×、PINN 13.4535×。GLM 报告没有混淆这两个定义：其 12.54/13.40 明确来自汇总均值。应在论文中写明“ratio of group-mean RMSE”，避免与 per-seed mean ratio 混淆。

### 相对改善

`(MLP-PINN)/MLP` 独立复算：

| H | Free joint velocity advantage |
|---:|---:|
| 1 | 9.513% |
| 2 | 2.796% |
| 4 | 2.911% |
| 8 | 3.30098% |

与报告的 `9.51%, 2.80%, 2.91%, 3.31%` 一致。趋势并非逐 horizon 单调收窄：H1→H2 明显收窄，H2→H4/H8 略回升。因此准确说法是“总体由 H1 大幅收窄至后续约 3%，在 H2/H4/H8 间非单调”，而不是“每一步持续收窄”。

以下三项成立：

1. PINN 在已测试的所有 horizon 上具有更低的 **pooled Free joint velocity RMSE**，且 5/5 seed；
2. PINN 没有表现出比 MLP 更慢的相对误差增长；ratio of means 为 13.40× vs 12.54×；
3. H=8 的绝对优势仍约 3.31%，但不能称为“减缓误差累积”。

## E. Root 状态分析核验

### H=8 root 指标

| domain | metric | MLP | PINN | relative change | paired wins |
|---|---|---:|---:|---:|---:|
| Free | root position | 0.000367164 | 0.000371668 | +1.23% | 0/5 |
| Free | root linear velocity | 0.0853677 | 0.0867479 | +1.62% | 0/5 |
| Free | root angular velocity | 0.352234 | 0.357848 | +1.59% | 0/5 |
| Free | geodesic | 0.00265337 | 0.00269487 | +1.56% | 0/5 |
| Contact-exclusive | root linear velocity | 0.0590381 | 0.0696937 | +18.05% | 0/5 |
| Contact-exclusive | root angular velocity | 0.421165 | 0.409565 | −2.75% | 3/5 |
| Contact-exclusive | geodesic | 0.00311335 | 0.00301568 | −3.14% | 3/5 |
| Impact | root angular velocity | 1.812279 | 1.758037 | −2.99% | 5/5 |
| Impact | geodesic | 0.0204682 | 0.0201405 | −1.60% | 5/5 |
| All | root angular velocity | 0.458087 | 0.449088 | −1.96% | **3/5** |
| All | geodesic | 0.00392481 | 0.00385582 | −1.76% | **4/5** |

### 核验结论

- Free H=8 四个 root variables 均为 0/5 PINN 优，结论正确。
- Impact 的 root angular velocity/geodesic 在全部已测 horizon 均为 5/5；Contact-exclusive 为 3/5。由窗口加权的全窗口结果为 H8 angular `3/5`、geodesic `4/5`。
- Contact-exclusive root linear velocity 的 `0.0697 vs 0.0590` 和约 +18% 正确。
- 上述结果是区域性能差异的观察。它不证明 contact/impact 中的 PINN 优势由 constraint force、Free residual 或 contact physics 所导致，也不证明 Contact 动力学被正确建模。
- Impact n=50，且窗口时间重叠；5/5 只能说明当前这组起点/seed 的方向一致。

### 需要文字统一

最终报告第 3.3 和代码表中的 H8 angular 为 3/5，但报告其他位置仍使用“全窗口 root ang vel / geodesic 优势 4/5”或概括为“4/5”。应明确写为：

> H=8：root angular velocity 3/5，root orientation geodesic 4/5；H=1/2/4 的 angular velocity 及四个 horizon 的 geodesic 分别为 4/5。

这是 **MINOR** 表述问题，不影响关键关节状态指标。

## F. 统计边界核验

### 通过

- 五个 seed 被表述为训练重复，不是五个独立 TEST set。
- 所有 seed 共用六个 TEST episode / 同一 3,110 起点池。
- 5/5 被限定为 paired seed 方向一致，没有包装成统计显著。
- 3,110 窗口时间重叠，未被作为独立随机重复或用来构造显著性。
- Impact n=50 的代表性限制已说明。
- Free 是起点标签，不代表整窗全程 Free。
- 实验被限定为 recorded-control open-loop rollout，不是闭环控制。
- 8 steps 对应约 8 ms，未扩展为长期预测。
- Contact-exclusive 混合、Impact joint-state 劣势、root state 非全面改善和无累计速度减缓均被保留。

### 需要最小修正

- 统一 H8 root angular velocity `3/5` 与 geodesic `4/5`。
- 对 12.54/13.40 标明是 ratio of seed-mean H8/H1 RMSE；若论文采用 per-seed mean ratios，则应报告约 12.59/13.45。

## G. 最终结论核验

可以采用的准确结论：

> 在当前 Booster K1 Stage1 TEST 的 Free-Dynamics 起点窗口中，PINN λ=1 相比配对 MLP 在 H=1/2/4/8 的 joint position 与 joint velocity open-loop RMSE 上表现出 5/5 训练 seed 同向改善；H=8 joint velocity 降低约 3.31%。但 PINN 的 H1→H8 相对误差增长并不低于 MLP（ratio of group means 13.40× vs 12.54×），故这不是更慢的误差累积。Contact-exclusive 的关节预测没有稳定优势，Impact 起点的 joint prediction 一致较差。Root metrics 没有全面改善：Free H=8 为 0/5，H=8 全窗口 root angular velocity/geodesic 分别为 3/5/4/5，且伴随 Contact-exclusive root linear velocity 约 18% 的劣势。结果仅限于当前六个 TEST episode、五个训练重复和约 8 ms 的开环短时预测，不构成统计显著性、闭环控制性能或全动力学域优势的证明。

## H. 执行合规性核验

- 未重新训练模型；`correct_aggregation.py` 只读取 raw NPZ。
- 未修改 Stage1-02 checkpoint、Stage1 split、原始物理数据、积分方法或 rollout 定义。
- 保留了根目录原报告，仅增加指向最终修正报告的注记。
- 创建了 FINAL_REPORT、CORRECTION_LOG、JSON、CSV、Python 聚合代码。
- JSON/CSV/code/FINAL_REPORT 的核心数值一致；脚本可执行并重建产物。
- 本次评审执行脚本重新生成了 corrected JSON/CSV，但没有重新训练或重跑 rollout。
- 任务范围总体合规。

## I. GLM 下一步建议的证据边界

“可以把 Contact/Impact 的区域差异作为提出 Contact-aware PINN 数学可行性问题的动机”是有限、可接受的建议。观察到：

- Free joint prediction 受益；
- Impact joint prediction 较差；
- Contact/Impact 的部分 root rotational metrics 方向不同；
- Contact-exclusive root linear velocity 明显较差。

这些结果只支持提出“是否缺少 constraint-force 项或其他接触建模成分”的**待验证假设**，不足以支持：

- 当前 Contact 问题的根因就是缺少 constraint force；
- Contact residual 一定能改善预测；
- 当前 root rotation 差异来自接触约束被正确利用；
- PINN 已具有普遍 Contact 优势。

因此建议的结论边界应明确标为 hypothesis/motivation，而非已经由本实验验证的机制。无需据此在本修正任务中设计或执行新实验。

## J. 最终验收意见

- **本次最小修正可否关闭：** 可以，在完成一处最小文字统一后关闭。
- **是否需要重跑 rollout：** 不需要。
- **是否需要重聚合：** 不需要；当前 JSON/CSV/code 已正确。
- **唯一必须修改：** [K1_PINN_ROLLOUT_01_FINAL_REPORT.md](tools/pinn/rollout/K1_PINN_ROLLOUT_01_FINAL_REPORT.md) 中所有 H=8 all-root angular velocity 的表述统一为 **3/5**；geodesic 保持 **4/5**。同时可在 growth 段注明使用 ratio of group means。
- 这不改变最终实验结论或研究判断。
