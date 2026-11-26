# K1_PINN_ROLLOUT_01_FINAL_REPORT — 修正后的开环多步状态预测对照结果

- 日期: 2026-11-24
- 性质: K1_PINN_ROLLOUT_01 的最小修正（分类/聚合/结论表述），实验本身未重跑
- 模型: Stage1-02 的 10 个 checkpoint（MLP seed0–4、PINN λ=1 seed0–4），未修改
- 数据: Stage1 TEST 6 episodes，全部 3,110 个合法 8 步窗口起点；H∈{1,2,4,8} 配对
- 修正依据: `rollout_corrected_metrics.json` / `corrected_metrics.csv`（自原始逐窗口误差重聚合）；修正过程见 `K1_PINN_ROLLOUT_01_CORRECTION_LOG.md`
- 原报告: `K1_PINN_ROLLOUT_01_REPORT.md`（保留作历史记录，其 §2/§3 中受影响结论以本报告为准）

---

## 1. 实验目的与固定条件

回答：Stage1-02 中 PINN 的单步加速度优势能否转化为多步状态预测优势。固定条件：开环状态递推（初始真实状态一次性，此后仅预测状态自馈 + 记录 ctrl 序列）；`v̂+=Δt·â; q̂=mj_integratePos(q̂,v̂,Δt)`；dt 为记录 f64 时间戳差；81 维输入契约与冻结归一化不变；无未来泄漏（真值加速度积分重建 max|Δv|=9.2e-14、max|Δq|=1.3e-15）；H=1 与独立单步实现差异=0；非有限预测 0 个。**本实验是相同记录控制序列下的开环预测，不代表闭环控制性能。**

## 2. 正确的互斥分类（任务 A）

起点标签分类（不使用预测状态）：Free = 冻结 Free-Dynamics mask；Impact = 原 impact 标签；Contact-exclusive = ~Free ∧ ~Impact。

| 类别 | Windows | 与复核基准 |
|---|---|---|
| Free | **1,024** | ✓ |
| Contact-exclusive | **2,036** | ✓ |
| Impact | **50** | ✓ |
| 合计 | 3,110 | ✓（Free∩Impact 重叠 = 0，逐项核对通过） |

全部指标自原始逐窗口逐步误差重算；Contact-exclusive **不是**从 ~free 指标扣除 Impact 得到。

## 3. 各 Horizon 分类指标

### 3.1 Joint velocity RMSE (rad/s)

| 起点 | H=1 | H=2 | H=4 | H=8 | 配对胜场 |
|---|---|---|---|---|---|
| Free | 0.0883 / **0.0799** | 0.3066 / **0.2980** | 0.5797 / **0.5628** | 1.1072 / **1.0705** | **5/5 × 4H** |
| Contact-exclusive | 0.1988 / **0.1986** | **0.5403** / 0.5430 | **1.0008** / 1.0105 | **1.6104** / 1.6370 | 3/2/2/2（5 seed 混合） |
| Impact | **3.1710** / 3.1913 | **3.7323** / 3.7878 | **4.8355** / 4.9797 | **8.3701** / 8.7130 | **0/5 × 4H** |

（每格：MLP / PINN λ=1。H=8 三类数值与 Owner 复核基准 1.107184/1.070523、1.610420/1.636974、8.370070/8.712970 **精确一致**。）

### 3.2 Joint position RMSE (rad)

| 起点 | H=1 | H=2 | H=4 | H=8 | 配对胜场 |
|---|---|---|---|---|---|
| Free | 9.0e-5 / **8.1e-5** | 3.6e-4 / **3.4e-4** | 1.27e-3 / **1.22e-3** | 4.50e-3 / **4.32e-3** | **5/5 × 4H** |
| Contact-exclusive | **1.9e-4** / 1.9e-4 | 3.7e-4 / 3.7e-4（2/5） | 1.05e-3 / 1.08e-3（2/5） | 3.4e-3 / 3.5e-3（2/5） | 3/2/2/2 |
| Impact | MLP 全部占优 | 0/5 × 4H | | | |

### 3.3 Root 状态（分变量、分域，配对胜场 PINN vs MLP）

| 变量 | 域 | H=1 | H=2 | H=4 | H=8 |
|---|---|---|---|---|---|
| root position (m) | Free | **5/5** | 4/5 | 3/5 | **0/5** |
| | Contact-ex / Impact | 0/5, 0/5 | 0/5, 0/5 | 0/5, 0/5 | 0/5, 1/5 |
| root linear vel (m/s) | Free | **5/5** | 3/5 | **0/5** | **0/5** |
| | Contact-ex / Impact | 0/5, 0/5 | 0/5, 0/5 | 0/5, 1/5 | 0/5, 3/5 |
| root angular vel (rad/s) | Free | 1/5 | 1/5 | 0/5 | **0/5** |
| | Contact-ex / Impact | 3/5, **5/5** | 3/5, **5/5** | 3/5, **5/5** | 3/5, **5/5** |
| root orientation geodesic (rad) | Free | 1/5 | 1/5 | 1/5 | **0/5** |
| | Contact-ex / Impact | 3/5, **5/5** | 3/5, **5/5** | 3/5, **5/5** | 3/5, **5/5** |

H=8 数值示例：Free 域 root 四指标 PINN 均无优势（angv 0.358 vs 0.352，geo 2.69e-3 vs 2.65e-3，均 0/5）；Contact-ex 域 root lin vel PINN 明显更差（0.0697 vs 0.0590，+18%）；**Impact 域 PINN 的 root 角速度（1.758 vs 1.812，−3.0%）与姿态（2.014e-2 vs 2.047e-2，−1.6%）在全部 horizon 5/5 占优**。geodesic 为四元数测地误差的 RMS，非分量 RMSE。

## 4. 误差累积的修正表述（任务 B）

Free 域 joint velocity RMSE H1→H8 增长倍数（自原始逐窗口全精度重算）：**MLP 12.54×（0.0883→1.1072），PINN λ=1 13.40×（0.0799→1.0705）**。Owner 复核参考值 ~12.59×/~13.45× 与本复算差 <0.5%（复核值疑似由舍入后表格值推得），不影响任何结论。

**正确表述**：
1. PINN 在全部 horizon 上保持 Free 域 joint velocity RMSE 绝对优势（5/5）；
2. 相对优势由 H=1 的 **9.51%** 变为 H=2 2.80%、H=4 2.91%、H=8 **3.31%**（非单调收窄，约收窄至 1/3）；
3. **PINN 没有减缓 Free 域的误差累积**——其相对增长倍数更高（13.40× vs 12.54×），只是起点更低；
4. 不得把较低的绝对 RMSE 写成"误差增长更慢"；8 ms 为短时预测，不得引申长时能力。

Free 域 joint position 的相对优势为 9.51% → 4.78% → 3.98% → 4.06%（5/5 × 4H），形态相同。

## 5. 五个 seed 的方向一致性（任务 C/D 的统计边界内）

| 结论 | 方向一致性 | 含义 |
|---|---|---|
| Free 域 joint pos/vel 优势 | **5/5 seeds × 4 horizons** | 训练重复间方向完全一致 |
| Impact 域 joint pos/vel 劣势 | 0/5（即 MLP 5/5 优）× 4H | 同样完全一致（反向） |
| 全窗口 root ang vel / geodesic 优势 | 4/5（H8 angv 3/5） | 方向多数一致 |
| Contact-ex 域 root lin vel 劣势 | 0/5 × 4H | 一致（反向） |

**统计边界（必须随任何引用一同声明）**：5 个 seed 是训练重复而非独立测试集；全部 seed 共用同一 TEST episode pool；"5/5"指同一 pool 上训练种子间方向一致，**不构成统计显著性检验**；3,110 个窗口存在时间重叠，不是独立实验重复，窗口数不得当独立样本量使用；Impact 仅 50 个起点，只作困难区域观测，不可推广；Free 起点不代表窗口后续全程无接触；本实验为记录控制序列下的开环预测，非闭环控制性能。

## 6. 最终研究结论（修正后）

> 在 Booster K1 当前 MuJoCo 数据集及 episode-level TEST 条件下，Free-Dynamics 物理约束训练改善了关节加速度预测，并在 Free-Dynamics 起点的短时开环 rollout 中表现出关节位置和关节速度预测优势（joint velocity 相对优势 9.5%→3.3%，joint position 9.5%→4.1%，均 5/5 seeds × 全部 horizon）。该优势在预测 Horizon 增大时有所收窄，且没有表现出更慢的误差累积（Free 域 H1→H8 增长 13.40× vs MLP 12.54×）。对于 Contact-exclusive 和 Impact 区域的关节状态预测，PINN 未体现出相同的优势（Contact-exclusive 混合、Impact 一致劣势）。Root 状态没有全面稳定的改善：Free 域 root 优势仅在 H=1 存在并于 H≥4 消失；全窗口 root 角速度/姿态的 4/5 优势来自 Impact（5/5）与 Contact-exclusive（3/5）窗口而非 Free 域，且伴随 Contact-exclusive 域 root 线速度的明显劣势（+18%）。8 ms 为短时预测窗口；本实验为记录控制序列下的开环预测。

与推荐表述的一致性：完全一致，且补充了 Owner 关注的 root 分域分解（修正新发现：全窗口 root 旋转优势的来源是 contact/impact 域而非 free 域）。

## 7. 可复算产物

| 文件 | 内容 |
|---|---|
| `rollout_corrected_metrics.json` | 类别核对、逐 run/逐 seed 全指标、配对差值与胜场、free 增长率、root 逐 seed 分域结果 |
| `corrected_metrics.csv` | 长表（run × horizon × class × 指标） |
| `correct_aggregation.py` | 修正聚合代码（自 raw_*.npz 重算，可追溯） |
| `raw_<run>.npz`（10 个） | 原始逐窗口逐步误差（未改动） |
| `K1_PINN_ROLLOUT_01_CORRECTION_LOG.md` | 修正记录 |
