# K1_PINN_ROLLOUT_01_CORRECTION_LOG — 修正记录

- 日期: 2026-11-24
- 范围: 结果分类/聚合/结论表述的最小修正；**未重新训练、未重新执行 rollout、未修改 checkpoint/数据/积分方法**
- 修正后结论见同目录 `K1_PINN_ROLLOUT_01_FINAL_REPORT.md`

---

## 1. 原报告（K1_PINN_ROLLOUT_01_REPORT.md）的问题

| # | 问题 | 原文位置 | 严重性 |
|---|---|---|---|
| P1 | Contact 类定义用 `~free`，把 50 个 Impact 起点窗口混入 Contact（报告为 "Contact (2086)"） | §2 分层表、§3-B/C | 高（类别不互斥，数值非同类可比） |
| P2 | 误差累积表述错误：原文"free 域 PINN 增长 12.1×（MLP 12.5×）"暗示 PINN 累积更慢；实测 PINN 相对增长 **13.40×** 快于 MLP **12.54×**，仅绝对 RMSE 更低 | §3-C、§4 | 高（方向性错误） |
| P3 | Root 结论域归属缺失/误导：全窗口 root ang vel (4/5) 与 geodesic (4/5) 优势未指出来源域；修正后证明**不来自 Free 域**（free 域 H8 为 0/5），而来自 Impact（5/5 × 4H）与 Contact-exclusive（3/5） | §2 Root 表、§3-B/E | 高（归因错误） |
| P4 | Root 各变量（pos/lin vel/ang vel/orientation）未分域讨论，仅给全窗口表 | §2 Root 表 | 中 |
| P5 | "PINN 优势是否减缓误差累积"的回答（§3-C）混用绝对 RMSE 与增长速度两个概念 | §3-C | 中 |

## 2. 修正内容

1. **互斥三类分类**：Free（冻结 mask）/ Impact（原 impact 标签）/ Contact-exclusive（~Free ∧ ~Impact）。核对：1,024 / 2,036 / 50，合计 3,110，Free∩Impact 重叠 = 0——与 Owner 基准**完全一致**。
2. **自原始逐窗口误差重算**全部指标（jp/jv/rp/rlv/rav/geodesic × H∈{1,2,4,8} × 4 类 × 10 run），Contact-exclusive 为直接重算，非扣减近似。
3. **误差累积修正**：Free 域 jv H1→H8 增长 MLP **12.54×**、PINN **13.40×**（Owner 复核参考 ~12.59×/~13.45×；差异 <0.5%，疑为参考值由舍入表格推算，以本全精度复算为准）；相对优势 9.51%→2.80%→2.91%→3.31%（非单调收窄）。
4. **Root 分域分解**（新增）：见最终报告 §3.3。关键修正发现：全窗口 root 旋转优势（ang vel 4/5、geodesic 4/5）来源为 Impact 5/5 + Contact-ex 3/5；Free 域 H=8 四个 root 变量均为 0/5，root pos/lin vel 的 Free 域优势仅存在于 H=1 并随 horizon 消失。
5. **结论重写**：采用 Owner 推荐表述并补充 root 分域发现；明确"未减缓误差累积""8 ms 短时""开环非闭环"等限定。

## 3. 涉及的文件

| 文件 | 变更 |
|---|---|
| `rollout/correct_aggregation.py` | **新增**（修正聚合，自 raw_*.npz 重算） |
| `rollout/rollout_corrected_metrics.json` | **新增**（可复算结果：类别核对/逐 seed/配对/增长/Root 分域） |
| `rollout/corrected_metrics.csv` | **新增**（长表） |
| `rollout/K1_PINN_ROLLOUT_01_FINAL_REPORT.md` | **新增**（最终报告） |
| `K1_PINN_ROLLOUT_01_REPORT.md`（根目录） | 仅加头部指针注记，正文保留作历史记录 |
| `raw_<run>.npz`、`run_rollout.py`、`aggregate_rollout.py`、Stage1-02 checkpoint、Stage1 数据 | **未改动** |

## 4. 新旧结果差异（同类指标对照）

| 指标 | 原报告 | 修正后 |
|---|---|---|
| Contact 类窗口数 | 2,086（含 Impact） | **2,036**（exclusive；50 归入 Impact） |
| H=8 jv "Contact" | 2.106 vs 2.052（PINN 劣 2.6%，实际是含 Impact 的混合值） | Contact-ex **1.610 vs 1.637**（PINN 劣 1.7%，2/5）；Impact 8.370 vs 8.713（劣 4.1%，0/5）单独列出 |
| Free jv H1→H8 增长 | "12.1×（MLP 12.5×）"（表述混乱） | PINN **13.40×**、MLP **12.54×**——PINN 相对累积更快 |
| Root ang vel 4/5 归属 | 未指明（上下文紧随 free 域讨论，易误读为 free 域收益） | **来自 Impact 5/5 + Contact-ex 3/5；Free 域 H8 为 0/5** |

## 5. 执行方式确认

- 未重新执行 rollout（全部自 `raw_<run>.npz` 重聚合）；未复用 checkpoint 重跑评价；
- 未重新训练、未改 λ/结构/特征/划分/积分/残差；
- 原始结果文件全部保留；修正代码与结果独立命名，与原结果可区分。

## 6. 未解决问题（只记录，不实施）

1. Impact 域 root 旋转（ang vel/geodesic）5/5 优势的机理未研究（样本仅 50，且可能是物理约束对"高加速度旋转状态"的间接正则效应——仅为假说）；
2. Contact-exclusive 域 root lin vel 的 PINN 劣势（+18%）机理未研究；
3. Contact-exclusive jv 的 seed 混合胜场（3/2/2/2）表明该域两模型无稳定差异，非"PINN 略优/略劣"可概括；
4. 全部统计边界见最终报告 §5（5 seeds=训练重复、窗口时间重叠、无显著性检验、Impact n=50）。
