# K1_PINN_ROLLOUT_01_REPORT — MLP/PINN 多步状态预测对照实验

> **[2026-11-24 修正注记]** 本报告的 Contact 分类（~free，混入 Impact）、误差累积表述与 Root 结论域归属存在已确认问题，修正后的最终结论以 `tools/pinn/rollout/K1_PINN_ROLLOUT_01_FINAL_REPORT.md` 与 `K1_PINN_ROLLOUT_01_CORRECTION_LOG.md` 为准。本文件保留作历史记录。

- 日期: 2026-11-24
- 模型: Stage1-02 的 10 个 checkpoint（MLP seed0–4、PINN λ=1 seed0–4），从 `stage1_runs/<run>/{best.pt,config.json}` 原始产物加载，结构与归一化按训练配置重建，未重训/未筛选 seed
- 数据: Stage1 TEST 6 episodes，**全部 3,110 个合法 8 步窗口起点**（`RolloutWindowDataset(horizon=8)` 索引），H∈{1,2,4,8} 共用相同起点配对比较
- 代码/产物: `tools/pinn/rollout/`（run_rollout.py、aggregate_rollout.py、raw_<run>.npz、results/、rollout_manifest.json、README.md）
- 判定: **PASS**（完成标准 7 项全部满足；核心答案见 §C——**单步优势只在 Free-Dynamics 域转化为多步优势，全域不成立**）

---

## 1. 实验设置与实现正确性检查

- **契约**：每步输入 = 81 维 [root pos 3 | rot6 6 | joints 22 | qvel 28 | ctrl 22]，由**预测状态**构建，冻结 train z-score；输出反归一化为物理单位加速度。MuJoCo 仅用于 `mj_integratePos` 位置积分（无 mj_step/mj_forward ⇒ 未来动力学不可能进入预测状态）；dt 用记录的 **float64** 时间戳差。
- **检查结果（§7 全部通过）**：
  1. 10 个 checkpoint 全部加载，结构与 Stage1-02 config 一致；
  2. 输出 28 维反归一化后为物理单位；
  3. H=1 rollout 与独立单步实现差异 = **0.0**（逐 run 验证）；
  4. **真值加速度积分重建**：3,110 窗 × 8 步，max|Δv|=**9.2e-14**、max|Δq|=**1.3e-15**——比 WM_01 报告值（3.5e-6/7e-9）更严。差异根因已定位：**WM_01 物化文件 `transitions_*.npz` 的 dt 字段是 float32 量化值**（引入 ~3.5e-6 速度重建误差），本实验按任务书要求使用记录 f64 时间差后误差消失。**此为对 K1_WM_01 接口的一个具体勘误：使用该文件的 dt 字段时须注意 f32 量化，精确复算应读 episode 时间戳**（本实验不受影响）；
  5. 积分后四元数模长合法（`mj_integratePos` 保范）；
  6. 窗口全部满足 episode/split/时间连续性（构造性保证 + 索引记录）；
  7. 零未来泄漏（实现仅以记录 ctrl 与预测状态为输入）；
  8. 非有限预测：**0 个窗口**（10 run 全部），无静默删除。
- **实验边界（必须声明）**：这是**相同已记录控制序列下的开环状态预测**，控制不随预测状态重算；结果不可解释为闭环控制性能。

## 2. 主结果（TEST 3,110 窗口，5 seeds 均值±std；配对胜场 = 同 seed PINN−MLP<0 的 seed 数）

### Joint position RMSE (rad)

| H | MLP | PINN-1 | 配对 |
|---|---|---|---|
| 1 | 0.0004 | 0.0004 | 1/5 |
| 2 | 0.0010 | 0.0011 | 1/5 |
| 4 | 0.0028 | 0.0029 | 1/5 |
| 8 | 0.0085±0.0001 | 0.0087±0.0001 | 1/5 |

### Joint velocity RMSE (rad/s)

| H | MLP | PINN-1 | 配对 |
|---|---|---|---|
| 1 | 0.4360±0.0012 | 0.4378±0.0018 | 1/5 |
| 2 | 0.6679±0.0031 | 0.6731±0.0045 | 1/5 |
| 4 | 1.0688±0.0080 | 1.0824±0.0108 | 1/5 |
| 8 | 1.7967±0.0212 | 1.8312±0.0241 | **0/5** |

### Root 状态 RMSE

| H | root pos (m) MLP→PINN | root lin vel (m/s) | root ang vel (rad/s) | geodesic (rad) |
|---|---|---|---|---|
| 1 | 3.0e-5 → 3.0e-5 (0/5) | 0.0311 → 0.0314 (0/5) | 0.1465 → **0.1459** (4/5) | 2.5e-4 → 2.5e-4 (4/5) |
| 4 | 1.6e-4 → 1.6e-4 (0/5) | 0.0548 → 0.0570 (0/5) | 0.3029 → **0.2990** (4/5) | 1.44e-3 → **1.42e-3** (4/5) |
| 8 | 4.1e-4 → 4.3e-4 (0/5) | 0.0782 → 0.0843 (0/5) | 0.4581 → **0.4491** (3/5) | 3.92e-3 → **3.86e-3** (4/5) |

误差增长（H1→H8）：joint vel ×4.1、joint pos ×21、root lin vel ×2.5、root pos ×13——**速度类变量对误差累积最敏感的是 joint velocity 与 root ang vel 的绝对增长率，root position 的相对增长率最高**（逐 horizon 曲线见 `results/error_growth.png`，含 5-seed ±std 带）。

### 起点状态分层（joint pos / joint vel RMSE，配对胜场 PINN vs MLP）

| 起点 | H=1 | H=2 | H=4 | H=8 |
|---|---|---|---|---|
| **Free (1024)** | jp **5/5**, jv **5/5** | jp **5/5**, jv **5/5** | jp **5/5**, jv **5/5** | jp **5/5**, jv **5/5**（jv 1.071 vs 1.107） |
| **Contact (2086)** | 1/5, 1/5 | 1/5, 1/5 | 1/5, 1/5 | 0/5, 0/5（jv 2.106 vs 2.052） |
| **Impact (50)** | 0/5, 0/5 | 0/5, 0/5 | 0/5, 0/5 | 0/5, 0/5（jv 8.713 vs 8.370） |

### 逐 episode（joint pos RMSE @H=8，5-seed 均值）

PINN 更好：drop_d0（−1e-4）、pd_top_rock/s77（≈0）；PINN 更差：pd_top_rock_s76（+3e-4，混沌翻滚段）、ou_o0（+5e-4）；hold_throw_h0 ≈0。全表见 `results/metrics_by_horizon.csv`。

## 3. 核心问题回答

**A. 单步加速度优势是否延续到 H=1 状态预测？——全域不延续，分层延续。** 全窗口层面 PINN 在 joint pos/vel（1/5）与 root lin vel（0/5）上无优势；但 **free 起点窗口 5/5 全胜**（jv 0.080 vs 0.088），root ang vel 与四元数测地角 4/5 胜。单步*加速度*优势（Stage1-02 free 子集 −9.5%）转化为*状态*优势的充分条件是窗口确实处于无约束飞行段。

**B. 随 Horizon 增加优势如何变化？——分层固化。** free 起点的 5/5 全胜在 H=1→8 全程保持（幅度同步增长：jv −3.3% at H8）；contact 起点的劣势随 H 轻微扩大（H1 1/5 → H8 0/5）；root ang vel 优势由 4/5 缓释至 3/5。

**C. PINN 能否减缓误差累积？——仅 free 域。** free 起点 jv H1→H8 增长 12.1×（MLP 12.5×），H8 绝对误差 −3.3%；contact/impact 域 PINN 累积略快（contact jv H8 +2.6%）。

**D. 优势是否主要来自 Free-Dynamics 区域？——是，完全一致于 Stage1-02 与 D-02 的结论链**：物理损失作用的域 = 多步优势的域，没有任何证据显示该优势外溢。

**E. Contact/Impact 是否仍是主要困难区域？——是**：impact 起点窗口 H=1 的 jv RMSE 已达 3.17（全窗均值 0.44 的 7 倍），H=8 达 8.4；两模型在 impact 域差异极小且 MLP 略优。

**F. 是否存在变量间分化？——是**：PINN 在 root 角速度与姿态（geodesic）上占优（4/5），MLP 在 joint pos/vel 与 root 线速度上占优（0–1/5 反向）。两类结论并存，报告必须同时呈现。

**G. 论文能支持什么 / 不能支持什么？**
- 能支持："Physics-informed training improves open-loop multi-step state prediction **in the free-flight regime**, consistently across 5 independent seeds and all tested horizons (1–8 ms)"——free 域 5/5 配对稳定 + 物理残差改善 + 机制解释（约束域=改善域）三者自洽。
- 能支持（辅助）："root rotational prediction is also improved (4/5)"。
- **不能支持**："PINN improves multi-step prediction overall"（全窗口 joint vel 上 PINN 一致性劣于 MLP，0/5–1/5）；"改善源于接触动力学学习"（无接触残差建模，contact 域无收益且略负）；任何"长时间预测"表述（8 ms 只是 8 步）；将 5 seeds 方向一致宣称为"统计显著"（未做假设检验，仅报告配对方向一致性）。

## 4. 产物清单

| 项 | 路径 |
|---|---|
| rollout 实现 | `tools/pinn/rollout/run_rollout.py` |
| 汇总实现 | `tools/pinn/rollout/aggregate_rollout.py` |
| checkpoint/数据/契约记录 | `tools/pinn/rollout/rollout_manifest.json` |
| 逐窗口原始误差（10 run） | `tools/pinn/rollout/raw_<run>.npz` |
| 逐 seed/episode/horizon/class 指标 | `tools/pinn/rollout/results/metrics_by_horizon.{csv,json}` |
| 配对差值与胜场 | `tools/pinn/rollout/results/summary_paired.json` |
| 误差增长图 | `tools/pinn/rollout/results/error_growth.png` |
| 说明文档 | `tools/pinn/rollout/README.md` |

历史产物（stage1_runs/、stage1_data/、world_model/）零改动。代码仅新增，未修改既有接口。

## 5. 对下一阶段的具体建议

1. **论文口径**：主结果表采用"全窗口 + free 域"双栏呈现；contact/impact 的中性偏负结果如实放入，作为"物理约束域=收益域"论证的一部分而非遮蔽项。
2. **接触域改善的正确路径**：本实验再次确认收益边界由训练约束决定——contact-aware 残差（用记录 qfrc_constraint 监督/约束）是把收益扩展到接触域的最小改动方向（Stage1-01 建议的延续，仍属未来工作）。
3. **rollout 教师强制混合训练**（scheduled grounding）：多步误差累积主因是分布漂移（预测状态离开训练流形后加速度预测失准，impact 域尤甚）；若继续世界模型路线，短期多步微调属于新任务，不在 PINN 论文现有范围。
4. **复算注意**：使用 K1_WM_01 `transitions_*.npz` 时 dt 为 f32 量化（见 §1 检查 4），精确积分一律读 episode 时间戳（本实验已按此执行）。

**完成标准 7 项全部满足。任务完成，停止；等待 Owner 研究决策，不进入 Stage2。**
