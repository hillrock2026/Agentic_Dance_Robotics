# K1_PINN_STAGE2_CONTACT_FEASIBILITY_01 — GPT Independent Technical Review

**Review date:** 2026-11-24  
**Scope:** contact equation, Stage1 field alignment, Oracle/equivalence calculations, mass loss interpretation, `qfrc_constraint` information boundary and G1–G4 definitions.

## 1. 总体判定

**PARTIAL PASS**

动力学符号、同快照字段可用性以及

`r_contact(a_pred) = M(q)(a_pred-a_true) + r_oracle`

成立并可独立复算。当前数据支持用完整广义动力学方程构造训练期 residual。

但三类 full-dataset mask 并非互斥，Oracle 中的 M reconstruction 又被静默降至 float32，报告所呈现的 Free/Impact residual 并非真实 float64 solver residual。

但是报告把接触 residual 与其建议的 `e^T M e` mass-weighted loss 当作同一对象，数学上不成立。对于通常使用的逐分量平方 residual：

`||r_contact||² = ||M e||² = e^T M^T M e = e^T M² e`，而不是 `e^T M e`（oracle error 忽略时）。因此 G3 是一个独立的 loss-design comparator，不是“Contact-aware residual 的另一种说法”。这会实质影响 G3/G4 的科学定义、Impact 权重结论的外推和 `CONDITIONAL GO` 的措辞。

此外，情况 I/II 的“无训练/推理不一致”需要限定为“推理计算不需要该字段”，不能擦掉 privileged-supervision 信息边界。`CONDITIONAL GO` 对“进行 G3/G4 定义的损失/辅助监督消融”在概念上仍可保留，但应修正为：数学可行，实验效果未知，且 G3 与 residual-squared 尚不能等同。

## 2. 数学推导评审 — PASS WITH QUALIFICATIONS

### 通过项

MuJoCo generalized-force balance and the implemented sign convention are correct:

`M(q)a + qfrc_bias = qfrc_actuator + qfrc_passive + qfrc_applied + qfrc_constraint`。

因此：

`r_contact(a_pred) = M a_pred + qfrc_bias - qfrc_actuator - qfrc_passive - qfrc_applied - qfrc_constraint`

且恒有：

`r_contact(a_pred) = M(a_pred-a_true) + r_oracle`，

其中 `r_oracle` 是记录的 `a_true` 代入相同残差。

- 力项符号正确，无遗漏，维度均为 28。
- 原始 `M_upper` 为 406 项 float64，索引 `np.triu_indices(28)` 正确；原始 storage 可精确恢复为对称矩阵，独立 float64 reconstruction 的 max asymmetry 为 0。
- raw arrays、M、qacc、force fields 来自采集器同一次 pre-step `mj_forward` snapshot；训练预测 TEST 中使用相同 row indexing。
- **但分析脚本没有真正保留该 float64 精度：** `stage0_common.rebuild_M` 内部预先分配 float32，传入 float64 `M_upper` 仍会被 cast。当前报告的 2.7e-5 Free、1.3e-3 Impact residual 及部分 identity numbers 因此大多是矩阵重建量化误差，不是 solver error。原始 float64 reconstruction 的 Free/Impact residual 分别约为 2.84e-13 / 4.06e-10。
- `qfrc_constraint` 被解释为 generalized constraint force，报告已指出它包含足底/身体 contact 与 joint-limit contributions，而非纯足底接触力。

### 需限定

身份式里的 `r_oracle` 不是严格为零；完整表述应始终保留该项。等式“`r_contact ≈ M e`”是数值近似，仅因 oracle residual 相对于主力尺度很小。

## 3. 数据完整性与 Oracle residual — PASS WITH QUALIFICATIONS

### 数据核验

- 30 episodes / 24,578 samples；`qpos(29)`, `qvel/qacc/28 force fields(28)`, `M_upper(406)`, `ctrl/actuator_force(22)`。
- 原始 NPZ 动力学字段均 f64，独立复算 nonfinite=0、dt violation=0。
- **raw masks 不是全量互斥：** `Free∩Impact=3`。重叠为 `hold_throw_h5[378]`、`ou_o3[33]`、`pd_popping_s30[622]`。Free 4,333、Impact 345、`~free&~impact` 19,903 相加为 24,581，而不是 24,578。
- 若优先将 Impact 独立定义，互斥 Free 应为 `free & ~impact`，计数 4,330；Impact 345；Contact-exclusive 19,903，总计 24,578。
- TEST 恰好没有 Free/Impact overlap，所以 TEST 的 1,028 Free、2,076 Contact-exclusive、54 Impact 是有效互斥类别。

### Oracle 数值

脚本输出（float32 reconstruction）为：

| class | reported n | max residual | RMS residual | RMS residual / RMS `M a*` |
|---|---:|---:|---:|---:|
| Free raw | 4,333 | 2.66485e-5 | 1.64721e-6 | 4.33367e-8 |
| Contact-exclusive | 19,903 | 0.122625 | 6.80675e-4 | 2.13387e-5 |
| Impact | 345 | 1.25902e-3 | 4.96728e-5 | 4.42986e-8 |

独立 **float64** reconstruction gives:

| class | mutually exclusive n | max residual | RMS residual | RMS/RMS Ma |
|---|---:|---:|---:|---:|
| Free-exclusive | 4,330 | 2.84217e-13 | 1.43533e-14 | 3.77630e-16 |
| Contact-exclusive | 19,903 | 0.122615 | 6.80675e-4 | 2.13387e-5 |
| Impact | 345 | 4.05635e-10 | 1.35843e-11 | 1.21146e-14 |

Free 和 Impact 的报告值并不反映 solver residual 主导项，主要反映 `rebuild_M` 的 float32 cast。Contact-exclusive 几乎不变，仍是实际 solver closure 量级。RMS relative-to-force scale shows residual small, but the reported 2.1e-5 is **RMS/RMS**, not the maximum relative residual.该口径需明确。
Contact-exclusive 的约 0.123 generalized-force residual 与 constraint-bearing samples 的 solver closure tolerance 兼容；它只占 RMS force scale 的约 2.1e-5。Free/Impact 在 true float64 rebuild 下接近 machine precision。此处单位混合了 generalized force components，不能无条件都表述为 N；root rotational DOFs correspond to N·m。

### 需修正的来源说明

报告把 residual 量级归结为“step 内二次求解/二次收敛”并引用约 `3.5e-6 rad/s²`。本分析重算的是同一 `mj_forward` snapshot 的 `M a + forces` closure；记录字段都来自那次 solve 前的同一状态/求解结果。约 0.123 N 的残差首先是该次 Newton constraint solve 的收敛/closure error；不能用 rollout/W​​M 的 float32 dt 或 step-internal re-solve 差异来确定其来源，后者并未在脚本中被量化。

## 4. 等价性验证评审 — PASS

两项验证通过：

- random `a_pred`：`r_contact(a_pred)-M(a_pred-a_true)` 等于同一数据的 oracle residual；global max 为 0.122625。
- Stage1-02 PINN λ=1 seed0 的 3,158 TEST predictions：Free max 2.36e-5、Contact-exclusive max 0.10456、Impact max 1.259e-3；数值是各类 TEST oracle residual。

矩阵乘法和 shape 正确，三个域分别计算。虽然代码未实际 assert `equals_oracle_residual`，而是将该布尔字段硬编码为 `True`，独立逐样本复算确认该结论成立；这是实现严谨性细节，不改变数学结论。

该验证只证明：

1. residual 实现与完整动力学 accounting 一致；
2. residual 在忽略 oracle tolerance 后是 `M e`；
3. 不证明该 residual/loss 会提升模型预测性能。

## 5. Mass-weighted loss 评审 — PARTIAL PASS

### `e^T M e` 的计算通过

脚本定义和矩阵计算正确：

- `L_acc = mean(sum(e²))`，不是除以 28；
- `L_mass = mean(e^T M e)`；
- diagonal approximation = `mean(sum(M_ii e_i²))`；
- class averages each normalize by class sample count。

TEST seed0 独立复算（CPU/GPU 末位可能略异）：

| class | n | L_acc | L_mass | diag/full |
|---|---:|---:|---:|---:|
| Free | 1,028 | 120,373 | 1,519.04 | 2.35 |
| Contact-exclusive | 2,076 | 964,425 | 14,734.95 | 1.36 |
| Impact | 54 | 216,562,725 | 10,996,528 | 1.72 |
| All | 3,158 | 4,376,275 | 198,215 | 1.71 |

### 对 Stage1 G1 normalization 的比较 — PASS WITH QUALIFICATIONS

The report's `L_acc=mean(sum(e²))` is an unnormalized physical-unit sum over DOFs. The actual Stage1-02 G1 data loss is per-dof normalized MSE, so its class-level normalized losses are not numerically the same object. Independently, normalized elementwise MSE gives Free≈0.030066 and Impact≈133.2306, an Impact/Free ratio about 4,431, not 1,799. Therefore 1,799 is a valid ratio only for the physical `sum e²` diagnostic, not a direct statement about the G1 normalized loss.

### 对角矩阵不可忽略的结论 — PASS WITH QUALIFICATIONS

对当前 checkpoint/test error vectors，diagonal-only approximation 对 full quadratic form 高估 1.36–2.35 倍，因此不能把这类数据上的 `e^T M e` 简化为 M diagonal。报告对“当前样本误差”有证据。

但脚本没有实际统计“所有非对角元均值”，因此“非对角元均值为负”只是注释式说明。独立检查其确实约为负（约 `-2.85e-3`），但仅由均值不能推断每个误差向量上的耦合效应；报告应以 quadratic-form ratio 为主。

### 关键数学错误：`||r||² != e^T M e`

Oracle tolerance 忽略时：

`r = M e`。

常用 residual square loss 是

`||r||² = e^T M^T M e = e^T M² e`，

而报告分析的是

`L_mass = e^T M e`。

To obtain the latter from residual one would need a nontrivial inverse-mass form such as `r^T M^{-1} r`（实际还需选择 normalization/scaling）。因此：

- `r_contact ≈ M e` 正确；
- `e^T M e` 的 full-matrix weighted loss calculation 正确；
- “Contact-aware residual and G3 mass loss are the same thing” **不正确**；
- `L_mass` 的 Impact weighting result cannot be attributed to naively squaring the contact residual without separately analyzing `e^T M² e`.

Independent seed0 TEST recomputation gives:

- `L_acc impact/free ≈ 1799`；
- `L_mass impact/free ≈ 7239`（report rounds/quotes 7244, a small model-execution numeric difference）；
- but `||M e||² impact/free ≈ 28,477`。

所以“质量加权放大 Impact”的方向性判断对 G3 `e^T M e` 成立，但报告若将它说成 Contact-aware residual squared loss 的量化影响，则错误且明显低估 `||Me||²` 下的 domain skew。

Risk wording is valid only as a loss-design risk, not as an observed training-performance result. Class-mean ratios are not class-total loss ratios: sample counts have been removed by mean normalization; this is stated implicitly by mean formulas but should be explicit.

## 6. qfrc_constraint 信息属性 — PASS WITH QUALIFICATIONS

### 正确认识

- `qfrc_constraint` 是 MuJoCo constraint solver output，由 state/contact/joint-limit geometry、model parameters and solver path 决定。
- 它包含 generalized constraint forces，不是独立 external force measurement，也不是 pure foot force。
- 当前 arrays did not store `efc_*` decomposition；cannot isolate pure contact vs joint-limit components.
- `qfrc_constraint` is not an independent observation from `qacc` once the rest of the force-balance snapshot is fixed.
- At deployment it is not obtainable from the existing network input alone without reproducing contact detection/constraint solving.

### 情况 I/II/III

- **I: training-only residual/loss:** if the residual is algebraically rewritten to `M e` or another expression using recorded `a_true` and training-only M, the deployed predictor does not need `qfrc_constraint`. The present report establishes this rewriting for residual equality, but should explicitly say: the literal residual construction uses privileged solver output during training; deployment consistency is obtained by showing the training loss is algebraically equivalent to an available-target loss, not merely because the field is “only used in training.”
- **II: auxiliary head:** there is no inference input requirement if the auxiliary head/output is discarded, but the optimization uses privileged supervision. Calling it “no information mismatch” is too absolute; there is an information asymmetry in training, though no operational train/inference input/output interface mismatch.
- **III: inference residual requiring constraint force:** rejected correctly. It requires solver access and ceases to be a pure learned-only inference interface.

Thus the report’s privileged/training/inference distinction is mostly sound, but I/II require these explicit qualifications.

## 7. G1–G4 定义评审 — PARTIAL PASS

| group | assessment |
|---|---|
| G1 MLP | **PASS** as baseline using same 81-dim input, per-DOF normalized acceleration MSE. |
| G2 Free-PINN | **PASS** as existing reference residual on Free samples. |
| G3 `e^T M e` | **PASS WITH QUALIFICATIONS** as a pure loss-weighting comparator, **not** as an independent contact physics model and **not equivalent** to the usual squared contact residual. |
| G4 auxiliary constraint head | **PASS WITH QUALIFICATIONS** as solver-force-supervised privileged auxiliary target. It should explicitly say shared representation/training objective changes while inference drops the head. It is not a contact-physics residual. If warmstart/history makes the stored solver output path-dependent, the 81-dim current-state/control may not be a strictly sufficient deterministic input; this is not required for a supervised auxiliary feasibility question but should temper a claim of exact `g(q,v,ctrl)` mapping. |

- M in G3 is available from training data and only enters the loss; no deployment leakage in the proposed state predictor.
- G4 inputs are unchanged; the predicted auxiliary target is an output, not an input. No direct feature leakage is identified.
- The plan correctly preserves Free/Contact-exclusive/Impact evaluation.
- G3’s exact physics/scientific question must be “does an `M`-metric/kinetic-error weighting alter accuracy?” G4 asks a distinct representation-supervision question.

## 8. 必须修正的问题

### MAJOR 1 — Distinguish `e^T M e` from squared contact residual

- **Location:** report §§1–2 and §4, especially “G3 and Contact-aware residual are the same thing.”
- **Evidence:** `||M e||² = e^T M² e`, while the analyzed `L_mass=e^T M e`.
- **Impact:** The report misidentifies the mechanism and under-attributes Impact weighting; `||Me||²` has impact/free ratio about 28,477, versus L_mass 7,239.
- **Minimum correction:** define G3 explicitly as an `M`-metric supervised loss comparator; do not call it equivalent to residual-square. If a residual loss is contemplated, analyze `e^T M² e` or `r^T W r` with the exact W/normalization.

### MAJOR 2 — Qualify training/inference information consistency

- **Location:** report §3, situation I/II “no inconsistency.”
- **Evidence:** training uses solver-derived qfrc_constraint as target/residual information unavailable at deployment, even if no runtime input is required.
- **Impact:** Overstates compatibility and conflates no-input-dependency with no privileged-supervision asymmetry.
- **Minimum correction:** say “no required runtime input mismatch”; explicitly retain “privileged training supervision.” For I, state the algebraic elimination; for II, retain the auxiliary-head training-only nature.

### MAJOR 3 — Oracle precision and class partition are incorrectly stated

- **Location:** field inventory, oracle block, `class_masks`, and report §1/§2.
- **Evidence:** `rebuild_M` returns float32 despite float64 input; full-dataset raw Free/Impact overlap has 3 samples. Reported `L`/residual values therefore mix cast-induced errors and overlapping partitions.
- **Impact:** Free/Impact oracle values do not measure actual float64 solver closure, and “three mutually exclusive classes” is false for the full dataset. TEST happens to be mutually exclusive but cannot justify the all-data statement.
- **Minimum correction:** implement float64 M reconstruction locally; define Free-exclusive as `free & ~impact`; rerun JSON/CSV and update counts/Oracle values. Keep the current TEST-only loss counts if their overlap is zero.

### MINOR 1 — Oracle metric and provenance wording

- State that the reported relative residual is RMS/RMS, not max relative.
- Attribute the 0.123 contact residual to same-forward-solve closure/convergence; do not cite the WM float32-dt artifact as evidence for its source.
- Do not use N-only units for mixed generalized forces.

### MINOR 2 — Normalization and reproduction details

- Clarify that `L_acc=sum(e²)` is a physical-unit diagnostic, not the Stage1 G1 normalized MSE; report the normalized class ratio separately.
- The 7244 Impact/Free G3 ratio should be stated as approximate or use the regenerated JSON ratio about 7239; the conclusion is unchanged.
- Replace hardcoded `equals_oracle_residual=True` with an actual assertion if the artifact is intended as executable evidence.
- Correct the report's TEST sample count from 3,152 to 3,158.

## 9. 最终结论

1. **Oracle residual negligible?** PASS WITH QUALIFICATIONS. The report's reported script values are partly float32-cast artifacts, but a true float64 rebuild gives Free/Impact near machine precision and Contact-exclusive RMS relative residual `2.13e-5`; acceptable after correcting the method and class partition.
2. **Contact residual equivalent to M-weighted acceleration error?** PASS if “weighted residual vector” means `M e`. The squared loss equivalence requires clarification.
3. **Is qfrc_constraint independent contact information?** FAIL as phrased. It is solver-derived, algebraically redundant with qacc/other same-snapshot fields, privileged generalized constraint force, and not a pure contact-force observation.
4. **I/II/III information boundaries?** PASS WITH QUALIFICATIONS; I/II require no-runtime-input but retain privileged-supervision wording.
5. **Full M usage?** FAIL for float precision claims, PASS for matrix topology and formulas. The raw 406 values were used but silently downcast to float32 by `rebuild_M`.
6. **Impact amplification evidence?** PASS WITH QUALIFICATIONS for the physical-unit G3 `e^T M e` diagnostic; not directly for the normalized G1 loss or a generic residual-square PINN without exact norm/weighting.
7. **G1–G4 definitions?** PASS WITH QUALIFICATIONS, except G3 must no longer be identified with contact residual-squared.
8. **CONDITIONAL GO?** **Not yet on the present artifacts.** The underlying data support a corrected, tightly defined feasibility analysis, but MAJOR 1–3 must first be fixed: residual-squared vs M metric, full-data class partition, and float64 reconstruction. After those minimal numerical corrections, a conditional go for G3/G4 loss-design/privileged-supervision comparisons remains reasonable; no training performance has yet been demonstrated.
