# K1_PINN_ROLLOUT_01 — GPT Independent Technical Review

**Review date:** 2026-11-24  
**Scope:** checkpoint reuse, rollout implementation, state integration, test-window alignment, raw metric recomputation, paired MLP/PINN comparison and conclusion limits.

## A. 总体判定

**PASS WITH QUALIFICATIONS**

实验确实实现了真正的开环多步 rollout：只有窗口起点使用真实状态，之后模型输入使用预测的 `q_hat/v_hat`；记录控制序列逐步注入；没有 `mj_step/mj_forward`、未来 q/v/qacc 或 teacher forcing。10 个 Stage1-02 `best.pt`、normalization、TEST 起点和配对顺序均正确。Free 起点的 joint position/velocity RMSE 在 H=1/2/4/8 均为 5/5 seed 的 PINN 更优，H=8 joint velocity 为 1.0705 vs 1.1072（−3.31%），数值可信。

但有三个必须限定或修正的结论：

1. 官方 `contact` mask 实际是 `~free`，包含全部 50 个 Impact 起点，因此不是互斥的 Contact 类；报告中的 Contact 0/5 和 H8 `+2.6%` 是被 Impact 污染后的结果。
2. 证据不支持“PINN 减缓 Free 误差累积”。Free H8 仍低于 MLP，但 H1→H8 的相对增长率 PINN 反而略大；正确表述是相对优势由 H1 的约 9.5% 收窄到 H8 的约 3.3%。
3. Free 起点优势只在 joint position/velocity 上稳定；Free 的 root rotation/linear metrics 并非同步受益，尤其 H8 root angular velocity 与 geodesic 均为 0/5。全窗口、Contact、Impact 不支持总体优势。

因此本实验可作为 Stage1 PINN 在当前 TEST 上“Free 起点 joint-state 短时开环预测优势”的证据，不能泛化为全状态或全动力学域优势。

## B. 七个方面的评审结论

### A. Checkpoint 与模型输入输出一致性 — 通过

| Run | model / λ | seed | checkpoint | arch / params | loading / normalization |
|---|---|---:|---|---|---|
| `mlp_seed0` | MLP / 0 | 0 | `stage1_runs/mlp_seed0/best.pt` | 81×256×3×28 / 159,772 | 原始 checkpoint；Stage1 train stats |
| `mlp_seed1` | MLP / 0 | 1 | `stage1_runs/mlp_seed1/best.pt` | 同上 | 同上 |
| `mlp_seed2` | MLP / 0 | 2 | `stage1_runs/mlp_seed2/best.pt` | 同上 | 同上 |
| `mlp_seed3` | MLP / 0 | 3 | `stage1_runs/mlp_seed3/best.pt` | 同上 | 同上 |
| `mlp_seed4` | MLP / 0 | 4 | `stage1_runs/mlp_seed4/best.pt` | 同上 | 同上 |
| `pinn_lam1_seed0` | PINN / 1 | 0 | `stage1_runs/pinn_lam1_seed0/best.pt` | 81×256×3×28 / 159,772 | 原始 checkpoint；Stage1 train stats |
| `pinn_lam1_seed1` | PINN / 1 | 1 | `stage1_runs/pinn_lam1_seed1/best.pt` | 同上 | 同上 |
| `pinn_lam1_seed2` | PINN / 1 | 2 | `stage1_runs/pinn_lam1_seed2/best.pt` | 同上 | 同上 |
| `pinn_lam1_seed3` | PINN / 1 | 3 | `stage1_runs/pinn_lam1_seed3/best.pt` | 同上 | 同上 |
| `pinn_lam1_seed4` | PINN / 1 | 4 | `stage1_runs/pinn_lam1_seed4/best.pt` | 同上 | 同上 |

- 10 个 checkpoint 与 manifest/config 完全对应；均为 8 个预期 state tensors，未重训。
- 输入为 `[root pos 3 | rot6 6 | joint pos 22 | qvel 28 | ctrl 22]=81`；与 Stage1-02 训练接口一致。
- normalization 来自 `stage1_data/norm_stats.json`，即原训练目录的 train-only statistics；模型输出先乘 `std_a`、加 `mu_a`，得到 28 维 physical acceleration。
- MLP/PINN 对 q/v/ctrl、模型路径和推理代码的唯一模型变量是 checkpoint。没有未来 qacc/q_next/v_next 或标签进入 `build_X`。

### B. Rollout 是否真正使用预测状态 — 通过

[run_rollout.py](tools/pinn/rollout/run_rollout.py) 每个 run：

1. `q_hat=q0`, `v_hat=v0`；
2. 用 `build_X(q_hat,v_hat,ctrl_seq[:,k])` 预测 `a_hat`；
3. `v_hat += dt*a_hat`；
4. `q_hat = mj_integratePos(q_hat,v_hat,dt)`；
5. 下一迭代继续使用更新后的预测状态。

没有 teacher forcing、真实状态重置、未来 q/v/qacc 注入；`mj_forward/mj_step` 未在模型 rollout 中使用。`true_a` 只在独立 sanity-check 中用于验证积分，并不参与模型预测指标。H=8 先连续 rollout，再读取 step 1/2/4/8；不是重复 teacher-forced H=1。

10 个 raw 文件的 `ep_idx/t0/dt_seq/free0/impact0` 全部逐元素相同，配对窗口严格一致。Selective rerun 亦再现原 raw errors（仅 GPU/batch 数值级微差）。

### C. 积分与浮动基座状态正确性 — 通过

- q/v 维度和布局正确：qpos 29、qvel/qacc 28；quaternion 为 MuJoCo wxyz；root angular velocity 未被错误世界/局部转换。
- 采用半隐式 Euler，位置积分调用 `mj_integratePos`；没有 componentwise quaternion update。
- `dt` 直接由 episode 原始 float64 timestamp diff 计算。
- 报告的真加速度积分重建量级（max `|Δv|≈9.2e-14`, `|Δq|≈1.3e-15`）可由当前协议复现；该检查只验证时间、积分和状态布局，不代表模型准确度。
- geodesic 使用 `2*acos(abs(dot(q1,q2)))`，能处理 q/-q；输入 quaternions 在机器精度内归一，故未显式除 norm 不影响结论。
- H=1 rollout 与“独立单步”比较为 0，只说明两个代码路径一致；其数学计算几乎相同，不能作为模型精度证据。

### D. 测试窗口与数据统计 — 有条件通过

- TEST 6 个 episode；所有 H=8 合法起点均被使用，共 3,110；各 horizon 由同一组起点配对读取。
- 每 episode 的合法范围为 `t0=0..T-9`；无跨 episode/split、无重复时间戳、无窗口越界。
- 所有 `dt` 与原始 f64 timestamps 逐元素吻合。
- 10 个 raw run 均无非有限预测；没有静默删除窗口。

**问题：官方类别不是互斥的。**

| 类 | 官方定义 | 窗口数 |
|---|---|---:|
| Free | `free` | 1,024 |
| 官方 Contact | `~free` | 2,086 |
| Impact | `impact` | 50 |
| Contact-exclusive | `~free & ~impact` | 2,036 |
| All | — | 3,110 |

`Free∩Impact=0`，而 `Impact⊆official Contact`。因此三类数量相加得到 3,160，而真实 all 只有 3,110。报告中的 Free 和 Impact 数值有效，但 2,086 窗的 Contact 数值混入 Impact，不应作为纯 Contact 统计。

此外，标签是 rollout **起点**状态；窗口内部可转入其他状态。50 个 Impact 起点分散于六 episode，其中仍然是少量样本，只可作困难域观测，不足以支持强稳定性或统计显著性结论。

### E. MLP/PINN 指标与配对比较 — 有条件通过

- 独立从 10 个 raw NPZ 的逐窗口逐步平方误差重算官方全窗口、Free、Impact 指标，最大差约 `1e-16`。
- RMSE 为窗口×component 上的平方误差均值再开方：joint 22 components、root velocity/position 3 components、geodesic 为角度 squared error。
- 不同物理单位没有被加总；JP、JV、root pos/vel、angle 分列。
- `np.std` 为 5 个 seed 的总体 SD，不是 sample SD；论文若采用 sample SD 需乘 `sqrt(5/4)`。
- H1 joint position 与 joint velocity 并非独立证据：dt=1 ms 时二者精确相差常数尺度，分别报告可读性更好，但不应计为两个独立统计观测。
- 5/5 表示五个训练 seed 的同一 3,110 个 TEST 起点池上配对 RMSE 方向一致，不是五个 episode 或五个独立 test set。窗口彼此高度重叠且来自仅六个 episode，不能用 3,110 当作独立实验重复。
- 无需进行常规显著性检验；当前证据支持描述性的 seed 方向一致与 H=8 短时效应。

### F. 核心结果复核 — 有条件通过

- **Free joint position/velocity：** H=1/2/4/8 均 5/5 PINN 优，H8 数值正确。
- **Contact-exclusive：** joint position/velocity 没有稳定优势。H8 配对为 2/5，不是官方污染样本上的 0/5；均值仍为 PINN 略差 1.3%/1.7%。
- **Impact：** joint position/velocity 各 H 均 0/5，H8 +3.42%/+4.12%；但仅 50 个起点，不能作广泛统计推广。
- **全窗口：** joint position H8 为 1/5，joint velocity H8 为 0/5；root linear velocity 为 0/5。没有全域优势。
- **Root orientation：** H8 geodesic 4/5 PINN 优，全窗口 mean `0.003925→0.003856`。root angular velocity H8 为 **3/5**，不是 4/5；4/5 只出现在 H=1/2/4。
- **Free root 状态：** H8 root pos/linvel/angular/geodesic 均为 0/5 PINN 优。故“Free 域多步状态优势”若无度量限定会误导。
- Free H8 joint velocity 逐 episode 不具有统一 5/5 episode-direction；5/5 是 pooled seed-level free RMSE。部分 episode（尤其 `pd_top_rock_s76`）PINN 反而更差，说明 pooled result 受 episode 组成和误差加权影响。

### G. 结果解释与论文边界 — 有条件通过

可直接保留：

> 在当前 Stage1 TEST 的 Free-Dynamics 起点子集上，PINN λ=1 相比配对 MLP 对 1/2/4/8 步的 joint position/velocity open-loop rollout 表现出一致的 5/5 seed 方向；H=8 joint velocity RMSE 降低约 3.3%。

必须限定：

- 使用 **Free-Dynamics regime**，不要改写为 `free-flight`；当前 mask 是 no-constraint，不等同于纯腾空飞行或模型输入窗口全程无接触。
- 只到 8 steps/约 8 ms，不是长期 rollout。
- 只支持 joint position/velocity；不支持全部 root/state variables。
- 这是当前 6 个 TEST episode 和 5 个训练 seed 的描述性重复趋势，不是 statistical significance。
- 当前关联与受控 MLP/PINN 对照相容，但不能仅凭 rollout 结果证明“物理约束导致改善”的机制因果。

不能保留：

- “PINN improves multi-step prediction overall”；
- “Free 域误差累积被减缓”；
- “Contact 无优势为 0/5 且 +2.6%”（official contamination）；
- “root angular H8 4/5”；
- 长 horizon、free-flight、全动力学域或实际控制性能结论。

## C. 核心结果独立复算

### Joint position / velocity，按真实互斥起点类别

数值为五-seed mean；括号中为 joint-pos / joint-vel 配对胜场。

| class | H | MLP joint pos | PINN joint pos | MLP joint vel | PINN joint vel | wins |
|---|---:|---:|---:|---:|---:|---|
| Free (1024) | 1 | 0.0000883 | 0.0000799 | 0.088272 | 0.079875 | 5/5, 5/5 |
|  | 2 | 0.0003589 | 0.0003417 | 0.306621 | 0.298047 | 5/5, 5/5 |
|  | 4 | 0.0012685 | 0.0012180 | 0.579713 | 0.562837 | 5/5, 5/5 |
|  | 8 | 0.0044994 | 0.0043165 | 1.107184 | 1.070523 | 5/5, 5/5 |
| Contact-exclusive (2036) | 1 | 0.0001988 | 0.0001986 | 0.198761 | 0.198568 | 3/5, 3/5 |
|  | 2 | 0.0006947 | 0.0006977 | 0.540329 | 0.543049 | 2/5, 2/5 |
|  | 4 | 0.0023554 | 0.0023751 | 1.000840 | 1.010541 | 2/5, 2/5 |
|  | 8 | 0.0076202 | 0.0077184 | 1.610420 | 1.636974 | 2/5, 2/5 |
| Impact (50) | 1 | 0.0031710 | 0.0031913 | 3.170959 | 3.191328 | 0/5, 0/5 |
|  | 2 | 0.0067436 | 0.0068194 | 3.732284 | 3.787766 | 0/5, 0/5 |
|  | 4 | 0.0154031 | 0.0157195 | 4.835456 | 4.979692 | 0/5, 0/5 |
|  | 8 | 0.0417550 | 0.0431808 | 8.370070 | 8.712970 | 0/5, 0/5 |
| All (3110) | 1 | 0.0004360 | 0.0004378 | 0.436015 | 0.437816 | 1/5, 1/5 |
|  | 2 | 0.0010438 | 0.0010511 | 0.667881 | 0.673061 | 1/5, 1/5 |
|  | 4 | 0.0028243 | 0.0028558 | 1.068830 | 1.082437 | 1/5, 1/5 |
|  | 8 | 0.0085276 | 0.0086679 | 1.796749 | 1.831217 | 1/5, 0/5 |

### 全窗口六项指标 H=1/2/4/8

| H | model | joint pos | joint vel | root pos | root lin vel | root ang vel | geodesic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | MLP | 0.0004360 | 0.436015 | 3.1088e-5 | 0.031088 | 0.146477 | 0.0002537 |
| 1 | PINN | 0.0004378 | 0.437816 | 3.1384e-5 | 0.031384 | 0.145879 | 0.0002527 |
| 2 | MLP | 0.0010438 | 0.667881 | 6.6897e-5 | 0.039430 | 0.206012 | 0.0005716 |
| 2 | PINN | 0.0010511 | 0.673061 | 6.8012e-5 | 0.040233 | 0.204459 | 0.0005678 |
| 4 | MLP | 0.0028243 | 1.068830 | 1.5893e-4 | 0.054755 | 0.302849 | 0.0014358 |
| 4 | PINN | 0.0028558 | 1.082437 | 1.6383e-4 | 0.056969 | 0.299024 | 0.0014200 |
| 8 | MLP | 0.0085276 | 1.796749 | 4.0701e-4 | 0.078172 | 0.458087 | 0.0039248 |
| 8 | PINN | 0.0086679 | 1.831217 | 4.3101e-4 | 0.084248 | 0.449088 | 0.0038558 |

H8 Free relative change：joint position `−4.064%`，joint velocity `−3.311%`。  
H8 corrected Contact relative change：joint position `+1.299%`，joint velocity `+1.663%`，不是官方污染结果的 `+2.6%`。

H1→H8 joint velocity growth，按各 seed 后平均：MLP `12.59×`，PINN `13.45×`。因此 PINN 不是降低相对累积率，而是在 H1 已有较低误差后至 H8 保持较小绝对误差；优势幅度从约 9.5% 收窄到 3.3%。

## D. 问题清单

### MAJOR — Contact 类包含全部 Impact 起点

- **位置：** [aggregate_rollout.py](tools/pinn/rollout/aggregate_rollout.py) 的 `classes = {"contact": ~free0, ...}`。
- **证据：** 3,110 all = 1,024 free + 2,086 official contact，但 impact 的 50 个全部落在 official contact 中；真正互斥 contact 仅 2,036。
- **影响：** 官方 Contact H8 joint velocity 2.052/2.106、0/5 和 +2.6% 对纯 Contact 无效。真实 Contact H8 为 1.610/1.637、2/5、+1.66%。Free、Impact、All 结果不受影响。
- **最小修正：** `contact = (~free0) & (~impact0)`；重新聚合并更新 Contact 数值/文本。无需重跑 rollout。

### MAJOR — “减缓 Free 误差累积”与数据相反

- **位置：** [K1_PINN_ROLLOUT_01_REPORT.md](K1_PINN_ROLLOUT_01_REPORT.md) §3C。
- **证据：** free joint velocity H1→H8 growth：MLP 12.59×，PINN 13.45×；joint position 同样 PINN 54.23× vs MLP 51.13×。
- **影响：** 不能声称 PINN 降低累积速度/更慢增长，只能说它从 H1 已较低且在 H8 仍较低，但相对差距缩小。
- **最小修正：** 将结论改为“H1–H8 每个 horizon 的 pooled Free joint RMSE 均低，H8 约 −3.3%；绝对优势未随 horizon 扩大”。

### MINOR — Root rotation/Free 域的报告表述过宽

- **位置：** report §2/§3G。
- **证据：** Free H8 root ang vel/geodesic 都是 0/5；全窗口 H8 root ang vel 为 3/5，geodesic 4/5；不能把整体 root rotational direction 归到 Free 域。
- **影响：** 宽泛的“Free state prediction”会暗示全部 state variables 改善。
- **最小修正：** 明确写 joint position/velocity，并将 root metrics 单列为全窗口辅助结果。

### MINOR — H=1 一致性检查不是独立准确度验证

- **位置：** [run_rollout.py](tools/pinn/rollout/run_rollout.py) H=1 check。
- **证据：** direct path 与 rollout step 1 使用相同初始状态、同一 `build_X/model/integrate_batch` 逻辑，预期应为 0。
- **影响：** 不影响指标；只影响对“独立验证”的解释。
- **最小修正：** 称为 implementation self-consistency check，而非 independent accuracy test。

### MINOR — 标准差/显著性和 free-flight 用语

- **位置：** report §2/§3G。
- **证据：** summary 用 population SD；Free mask 是 no-constraint 起点，不代表整窗 free-flight；5 seed 共享同一 TEST episode pool。
- **影响：** 可能导致统计和动力学语义过度泛化。
- **最小修正：** 标注 population SD、训练重复方向一致、6 episode/8 ms/current dataset；改用 “Free-Dynamics regime”。

## E. 最终验收

1. **是否真正实现开环多步 rollout？** 是。t0 后仅使用模型预测状态；无 teacher forcing、`mj_forward/mj_step` 或未来状态泄漏。
2. **10 个 checkpoint 是否正确复用？** 是。均为 Stage1-02 对应 best checkpoint，五个 MLP 和五个 λ=1 PINN，输入/输出维度、normalization 和反归一化正确。
3. **状态积分和浮动基座误差计算是否正确？** 是。f64 dt、半隐式 Euler、`mj_integratePos`、MuJoCo quaternion convention 和 geodesic RMSE 均正确。
4. **配对结果是否可信？** 核心 raw data、同起点、同 controls、5 seed pairing、Free/Impact/All 指标可信；Contact 需用互斥 mask 重算。
5. **Free 域优势是否得到支持？** **对 joint position/velocity 支持。** H=1/2/4/8 均 5/5 seed，H8 JV −3.31%；对全部 root/state metrics 不支持。
6. **全域优势是否得到支持？** 否。joint pos/vel 和 root linear velocity PINN 劣于 MLP；只有 root angular/geodesic 有局部全窗口优势。
7. **哪些论文结论可以采信？** 可采信当前 TEST、8 ms、Free 起点的 joint-state 短时 open-loop prediction 方向一致；不可声称全域、长期、free-flight 全状态、统计显著或错误累积减缓。
8. **是否可以关闭？** 在完成 Contact mask 重聚合、修正 H8 root ang/accumulation/Contact wording 后可以关闭。**无需重跑模型 rollout**，只需重新执行 aggregation 并更新报告。
