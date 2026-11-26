# K1_PINN_STAGE1_01_REPORT — 正式动力学数据集扩展（覆盖盘点 → 缺口补充 → Stage1 数据集）

- 日期: 2026-11-24
- 执行环境: conda env `booster_train`（MuJoCo 4.0.1），K1 XML/物理参数零改动
- 判定: **PASS**（D1–D4 全部达成，存在已记录的残余限制，见 §D）
- 上游: Stage0-D-02（residual 修复与 Oracle 验证方法直接复用）

---

## A. 覆盖盘点（原 20 episodes，基于实际字段而非名称）

### A.0 盘点过程中发现的数据缺陷（重要）

Stage0 npz 中记录的 `foot_l/foot_r` 字段**损坏**：采集器把左右足 geom 混入同一集合，且 MuJoCo 接触对为 `geom1=ground, geom2=foot`，导致 `foot_l` 恒为 0、任一足触地时 `foot_r` 置 1（实测：single_L=0、double=0、single_R=11,379——机器人不可能全程单右足站立）。**原 npz 保留未动**；Stage1 的支撑标签改用**几何重建**（复用 `virtual_k1/foot_geometry.py` 的角点方法：mj_kinematics + 足盒 8 角点世界坐标最小离地高度 ≤ 2 mm 容差），并在采集器中修复了左右足判定（新 episode 起生效）。方法交叉验证见 §B/C（新旧一致性 95.93%）。

### A.1 覆盖矩阵（几何支撑标签 + 冻结 free mask 定义，15,600 样本）

| 状态类 | 样本 | 占比 | episode 数 | 集中度问题 |
|---|---|---|---|---|
| Free（冻结定义） | 2,230 | 14.3% | 20 | — |
| ├ 零力矩 free | 1,561 | 10.0% | 8（全部 drop） | **70% 的 free 来自零力矩抛落** |
| ├ 驱动 free | 669 | 4.3% | 12 | **51% 集中在 2 个 episode** |
| 稳定双足支撑 | 9,399 | 60.2% | 18 | **76% 来自 2 个 arm_wave episode** |
| 单足支撑 | 3,185 | 20.4% | 20 | 尚可 |
| 支撑转换（逐步标签） | **103** | **0.7%** | 20 | 每 episode 平均 ~5 步，极稀疏 |
| 冲击（10σ 或转换±2步 ∧ ΔF>100N） | 216 | 1.4% | 19 | 全部为摔落/抛落冲击，**无受控落地** |
| 躯干等非足接触 | 1,176 | 7.5% | 14 | — |

足支撑类别：double 60.5% / single_L 7.5% / single_R 12.9% / none 19.1%。Free 样本速度覆盖：drop 类 |ctrl|≡0、ou 类高角速度（p50 15.7 rad/s）高力矩（21.7 Nm）、pd 类低速近接触——**缺少"正常驱动水平下的受控腾空"**。

### A.2 缺口结论（证据驱动）

1. 驱动腾空 free 样本少且集中（669，无受控飞行段）；
2. 支撑转换样本极稀疏（103），无受控落地→支撑重建过程；
3. 冲击全部来自失控摔落，无"驱动状态下的落地冲击"；
4. 稳定支撑高度集中于 2 个 episode。

可复用资源（第二阶段）：`collect_stage0_data.py` 采集器与 4 条已验证 50 Hz 轨迹（top_rock/arm_wave/popping/arm_wave_usr）直接复用；官方 `k1_fight_001.csv`（29 列、**无时间戳列**）因帧率不可验证而**不采用**（避免以未验证假设造数据）。

---

## B. 扩展执行（+10 episodes，全部复用现有采集器）

| 新 episode | seeds | 步数 | free | 目的 |
|---|---|---|---|---|
| **hold_throw h0–h5** | 70–75 | 206–1,002 | 23–95%（合计 1,552） | 空中随机姿态/速度初始化，**PD+重力补偿全程保持站姿**：受控驱动腾空 + 受控落地冲击 + 支撑重建 |
| pd_top_rock_s76/s77 | 76/77 | 450/874 | 81.8%/4.2% | 研究目标动作（步伐转换）的 TEST 侧覆盖 |
| pd_popping_s78 | 78 | 900 | 8.6% | TRAIN 稳定支撑多样性 |
| pd_arm_wave_usr_s79 | 79 | 3,600 | 1.9% | TRAIN 稳定支撑多样性 |

每个新 episode 的动作来源、seed、控制配置（V1 PD 增益 ×U(0.8,1.2)）、ended_by 均记录于 `stage1_data/index.json`；采集时序、字段与审计断言（残差闭合、passive/applied≡0）与 Stage0 完全一致。**未新增字段需求**（现有字段已足够支撑标签），采集器仅修复左右足判定 bug。

新增覆盖实测：actuated free **669 → 2,772**（4.1×，分布于 22 episodes，top2 集中度 51%→26%）；hold_throw free 的 |ctrl| p50/p95 = 5.3/29.0 Nm、|vlin| p50/p95 = 1.36/2.58 m/s——正是 Stage0 缺失的"正常驱动受控腾空"；impact 216→345 且新增受控落地类。

## C. Stage1 数据集

- **目录**：`tools/pinn/stage1_data/`（30 npz；Stage0 原始 `stage0_data/` 保留未动）
- **清单与元数据**：`index.json`（30 episodes，含 origin/动作/seed/控制器/ended_by/n_steps/free 数）
- **划分**：`splits.json` — episode 级、动作不相交、三类场景齐全：
  TEST 6 eps（pd_top_rock×3 + drop_d0 + ou_o0 + hold_throw_h0）3,158 样本；
  VAL 4 eps（pd_arm_wave + drop_d1 + ou_o1 + hold_throw_h1）5,253；
  TRAIN 20 eps 16,167（66%）
- **归一化**：`norm_stats.json`（仅 train 统计；μ/σ_x、μ/σ_a、s_r、ε 下限、冻结 10σ 训练过滤规则、impact 判定规则）
- **状态标签**：`stage1_labels.npz`（逐样本：free / actuated_free / zero_ctrl_free / support{none,sL,sR,double} / transition / impact / stable_support / band / max_ctrl；标签定义见文件头注释，非互斥，统计时按集合显式区分）
- **支撑标签方法与验证**：全部 episode 用几何重建；在 10 个新 episode 上与修复后的物理接触标志对比，**一致率 95.93%**（17,226/17,956，分歧集中于 2 mm 容差边界的近接触态）
- **完整性**：`integrity.json` — 30 episodes 全部字段无 NaN/Inf；时间戳严格 1 kHz 单调；passive/applied 精确为 0；`actuator_force≡ctrl`（≤1.9e-6）；逐步残差闭合最大相对误差 1.94e-2（接触态 Newton 容差内；**free 子集见下为数值零**）
- **物理一致性核验**（Oracle，复用 D-02 程序）：4,333 free 样本 × 28 dof，`M·a* + bias − actuator` 归一化 RMS = **8.6e-8**（f64）/ 1.8e-7（训练 f32 路径）→ **PASS**
- **规模**：30 episodes / 24,578 样本 / 24.6 s 仿真时长（1 kHz）

### Stage1 vs Stage0 覆盖对比

| 类 | Stage0 | Stage1 | 变化 |
|---|---|---|---|
| 样本 / episodes | 15,600 / 20 | 24,578 / 30 | +58% / +10 |
| free | 2,230 (14.3%) | 4,333 (17.6%) | +94% |
| 驱动 free | 669（51% 集中 2 eps） | **2,772**（22 eps，top2 26%） | 4.1× 且去集中 |
| 零力矩 free | 1,561（70% 的 free） | 1,561（36% 的 free） | 占比降为 1/3 |
| 受控落地冲击 | 0 | hold_throw 落地段 | 新增 |
| 稳定支撑 top2 集中度 | 76% | 46% | 改善 |
| 支撑转换（逐步） | 103 | 131 | 仍稀疏（见 §D.4） |

---

## D. 阶段结论

1. **Stage1 数据集是否足以支持正式 MLP/PINN 对照实验？——是（PASS）**。覆盖缺口按证据补齐，全部质量门槛（完整性/快照一致性/Oracle 残差）通过；划分保持 episode 级隔离、动作不相交、test 含全部状态类型（含研究目标动作 top_rock 三个 seed 与受控腾空）。规模限制（24.6k 强相关样本）仍是泛化上限约束，但它不是"覆盖"缺陷，不应以继续加量为由推迟实验。
2. **Free 样本是否摆脱零力矩抛落依赖？——是**：驱动 free 占比 30%→64%，来源 12→22 个 episode，并新增受控腾空域（hold_throw：中速、中力矩、重力补偿驱动）。
3. **接触与支撑转换是否足够？——基本足够，带保留**：单足支撑 4,181 样本（30 eps 全覆盖）、受控落地→支撑重建过程已进入数据；但"转换发生步"本身仍仅 131 样本（0.5%）——1 kHz 逐点标签下转换是瞬态事件，其邻域（±2 步）已通过 impact 规则纳入。若 Stage1 实验显示转换邻域误差突出，再考虑加密采集，不在本任务预判。
4. **冲击是否仍过度集中？——否**：345 个 impact 样本分布于 29 episodes（top2 26%），且新增受控落地类（旧数据只有失控摔落）。
5. **残余限制**：(a) 总体规模仍小，跨动作泛化上限受约束（Stage0-D-01 已证）；(b) 零力矩 free 仍只来自 drop 域（其角色是弹道覆盖，可接受）；(c) seed 间训练方差（D-02 发现）是数据层面的现实，Stage1 实验设计必须 ≥5 seeds；(d) Stage0 老数据的 foot_l/foot_r 字段不可用（已由几何标签替代并记录）。

---

## E. 下一步建议（仅限正式对照实验所需）

1. 直接以 `stage1_data/` 启动正式 MLP / Free-Dynamics PINN 对照（复用 `train_stage0.py --runs-dir` 与 `evaluate_stage0.py`，norm/splits 已就绪）；
2. 每配置 ≥5 seeds（D-02 的种子方差结论）；
3. 评价指标沿用 D-02 口径：bulk（≤5σ）/free/contact/impact 分层 + free 残差（oracle=0 参照）+ 梯度比监控；λ 治理以梯度比为硬依据；
4. 报告中把 **Stage0 结果作为独立基准保留**（stage0_runs/ 与 norm_stats 未动，可随时复跑对照）。

## 复现命令

```bash
cd tools/pinn  # 解释器: /home/liu/anaconda3/envs/booster_train/bin/python
python analyze_stage1_coverage.py --out stage0_data/coverage_stage0.json          # Stage0 盘点
python collect_stage1_data.py                                                     # 复制20+采集10
python build_stage1_dataset.py                                                    # splits/labels/norm/integrity
python analyze_stage1_coverage.py --data-dir stage1_data --stats stage1_data/norm_stats.json \
    --out stage1_data/coverage_stage1.json                                        # Stage1 覆盖评价
python oracle_free_residual.py --data-dir stage1_data                             # 物理一致性核验
```

修改文件：`collect_stage0_data.py`（foot 左右足判定修复）、`stage0_common.py`（load_episodes 支持 extra_fields）、`oracle_free_residual.py`（--data-dir）；新增 `analyze_stage1_coverage.py`、`collect_stage1_data.py`、`build_stage1_dataset.py`。原始 Stage0 数据与全部旧实验产物零改动；未训练任何模型。
