# K1_DANCE_SKILL_FRAMEWORK_02_MINIMAL_FIX_01_EXECUTION_REPORT

日期：2026-11-19

## 1. 总体结论

**PASS**。三个 Major 问题全部修复，各有针对性测试捕获原始错误并验证修复；两个现有 Skill 回归通过；完整测试集 72 项全过（新增 54 项）；8 个受保护产物 SHA256 前后一致。

与任务书的两处差异（以实际代码为准，均已记录）：
- 工作区**不是 git 仓库**（任务书假设存在 worktree/分支）——以 SHA256 快照执行产物保护；
- 问题 B 的复现症状：完全相同的 2D 误差序列在旧实现下返回 **NaN** 而非评审所述 -1（机制一致且更严重：`np.corrcoef(sa, sb)[0,1]` 计算的是 `sa` 自身前两个时间步之间的相关，**根本不依赖第二个周期**）。

## 2. 三个 Major 修复

### A. oneshot 末帧索引绕回（`tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py`）

- **问题**：记录行 k 的参考帧用 `f = abs_frame % period`。oneshot 1260 帧跑到第 1260 步时 `1260 % 1260 = 0`，末行（机器人最终状态）被拿来与第 0 帧参考比较，误差/ghost 全部错位。策略命令本身正确钳制在末帧（`beyond_mimic._frame_id` 的 `min(...)`），仅录制器取模错误。
- **修复**：新增纯函数 `record_frame_index(abs_frame, period, oneshot)`——cyclic 保持取模绕回（语义不变），oneshot 单调递增并在末帧钳制；主循环改用该函数。ghost 渲染与误差行共用此索引，一并对齐。
- **测试**：`TestRecordFrameIndex`（6 项）覆盖首/中/末/超出保持/1260 全序列单调性/cyclic 绕回回归/真实新旧 npz 对应关系（t、abs_frame、frame、行号公式化互证）。

### B. cycle_consistency 数值计算（`k1_skills/evaluation.py`）

- **问题**：`np.corrcoef(sa, sb)[0,1]` 将 2D 输入按行作变量，该元素 = `sa` 第 0、1 两个时间步的关节模式相关——与周期间一致性无关（且与 sb 无关）。
- **定义（明确化并写入代码注释）**：一个周期的误差模式 = 该周期动作帧（cyclic 剔除过渡帧）× 组内关节的绝对跟踪误差矩阵展平向量；周期间一致性 = 两周期展平模式的 Pearson 相关（两周期一致 ⇔ 模式正仿射等价，corr=+1）。多关节聚合方式即展平（2D 时空结构整体作为签名）。
- **修复**：`_error_pattern_correlation()`——非有限值/样本不足/零方差 → 返回 `None`（计入 `undefined_pairs`），绝不伪造分数；`cycle_consistency()` 重写为逐周期对计算，新增 `mean_correlation`、`undefined_pairs`、`pattern_samples_per_cycle`；`per_cycle_arm_mae_mrad` 语义不变。
- **测试**：7 项合成数值测试（相同=1、反向=-1、独立≈0（|r|<0.3）、多关节异质幅度恒等=1、常量=None 不伪造、NaN=None、arm_wave 真实数据值域合法）+ 真实数据重算。

### C. Schema 约束与关节顺序契约（`k1_skills/spec.py`、`k1_skills/motion.py`、`skills/mj_dance_002/skill.yaml`）

- **C1**：`_is_int/_is_num/_isfinite/_check_pos_int/_check_pos_num` 类型化校验：fps（数、有限、>0）、source_frames（整数、>0）、transition_frames（整数、≥0）、physics_dt（>0 有限）、decimation（整数 >0）、num_envs/max_iterations（整数 >0）、seed（整数 ≥0）、三组 22 元增益数组（逐元素有限数值；stiffness/effort>0、damping≥0）。bool 不再被当整数、数值字符串/NaN/±Inf 全部拒绝（参数化测试 31 项）。
- **C2**：基于仓库资源实测事实建立契约——**所有 npz（train/deploy/cyclic，arm_wave 与 mj2 通用）共享同一 sim 序** `joint_names`（逐字节一致），CSV/评价序是同一 22 关节的另一排列。契约实现（`MotionResourceSet._verify_joint_order`）：① 有名称的资源必须名称+顺序完全一致（train↔deploy 不一致即拒绝）；② 无名称资源必须在 YAML 声明 `{path, joint_order: <角色>}` 并以**数据交叉验证**（共同帧 joint_pos allclose ≤1e-3 rad）证实声明；③ 无名称且无声明 → 明确报错，绝不凭长度 22 猜测。单一顺序来源即 npz 自带名称 + deploy 机器人配置，未复制第三份列表。
- **测试**：`TestJointOrderContract` 6 项（一致通过/错序拒绝/无名称无声明拒绝/声明+数据匹配通过/声明被数据反驳拒绝/双真实 Skill 通过）。

## 3. 关键验证数据

**MJ 末帧与误差（baseline 策略重跑，仅末行参考对齐不同）：**

| 指标 | 修复前（末行 ref=frame0，受 bug 影响） | 修复后（末行 ref=frame1259） |
|---|---:|---:|
| Root endpoint error | 414.3 mm | **218.5 mm** |
| Root maximum error | 414.3 mm（被末行污染） | **268.1 mm** |
| Torso orientation endpoint | 89.36° | **4.45°** |
| Torso orientation maximum | 89.36°（被末行污染） | **13.94°** |
| 最终参考帧索引 | 0 | **1259** |

新旧 1260 行仅最后 1 行 frame 不同（idx 1259）；`t[k]=(k+1)·0.02`、`abs[k]=k+1`、`frame[k]=min(k+1,1259)` 全序列公式化验证通过。产物：`k1_runs/outputs/mj_dance_002/minimal_fix_01_taskA_before_after.json`（含 skill/policy/动作资源/代码版本/仿真参数/最终帧索引元数据）；旧 metrics 原样保留并标记为受影响数据。

**cycle_consistency 合成数据**：相同=1.000、反向=-1.000、独立 |r|<0.3、多关节恒等=1.000、常量/NaN=None + undefined_pairs 计数——全部数值断言通过。

**arm_wave 真实数据重算**（未重跑物理）：baseline 0.951/0.951/1.000（mean 0.967），trained 0.943/0.943/1.000（mean 0.962）；F02 报告的旧值（0.969/0.969/0.871 与 0.928/0.928/0.995）为错误实现的产物，已在 `minimal_fix_01_taskB_cycle_consistency.json` 中并列记录。

**22-DOF 顺序**：arm_wave 三份 npz 名称逐字节一致 ✓；mj2 声明 `joint_order: train_npz` 经数据交叉验证通过（max diff 3.46e-05 rad ≤ 1e-3）✓。

## 4. 两个 Skill 回归

| 项 | arm_wave | mj_dance_002 |
|---|---|---|
| validate（含新契约） | ✓ 全绿 | ✓（trained 未供给的生命周期警告，预期内） |
| 资源加载 + 关节顺序 | ✓ | ✓（1261 train / 1260 deploy，±1 容忍） |
| 策略加载（现有 JIT/PT） | ✓（哈希未变） | ✓（哈希未变） |
| Runtime | ✓（cyclic 绕回归测试通过，未重跑） | ✓ 重跑 1260/1260 步（修复验证运行） |
| Evaluation | ✓ arms 75.4 mrad 与 F01 一致；cycle 指标已按修复语义重算 | ✓ 六族指标 + cycle 自动降级 |

未重新训练任何策略；mj2 的运行明确标注为 baseline 策略的修复验证运行，非 trained-policy 验证。

## 5. 完整测试结果

`python -m pytest tests_k1_skills -q`：**72 passed / 0 failed / 0 skipped**（原 18 项 + 本次新增 54 项，其中含 31 项参数化约束测试）。无历史失败，无断言放宽。

## 6. 修改文件清单与产物

| 文件 | 修改 |
|---|---|
| `tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py` | 新增 `record_frame_index()`，主循环替换取模 |
| `k1_skills/evaluation.py` | `_error_pattern_correlation` + `cycle_consistency` 重写（定义入注释） |
| `k1_skills/spec.py` | C1 类型/范围校验 + resources 支持 `{path, joint_order}` |
| `k1_skills/motion.py` | 关节顺序契约（`_verify_joint_order`）、结构校验整理 |
| `skills/mj_dance_002/skill.yaml` | deploy_npz 加 `joint_order: train_npz` 声明 |
| `tests_k1_skills/test_minimal_fix_01.py` | 新增（54 项） |

新产物：`k1_runs/outputs/mj_dance_002/`（修复后 mp4+metrics+log+before/after JSON）、`k1_runs/outputs/*/minimal_fix_01_task*.json`、`k1_runs/minimal_fix_01_{baseline,after}_sha256.txt`。

## 7. 保护结果

8 个受保护产物（两个 JIT、mj2 预训练 PT、model_2999 checkpoint、mj2 两份原始 npz、两份原始 metrics）修复前后 SHA256 **全部一致**（清单存 `k1_runs/`）。原始 baseline 运行记录未覆盖，新 metrics 为独立文件。

## 8. 尚未解决的问题

1. `mj_dance_002` 的 trained 角色仍无全量训练策略（smoke 12 iter 不作为策略使用）——非本任务范围。
2. cycle_consistency 对"零误差（常量）周期"按 Pearson 语义返回 undefined；此类情形的信息由 per-cycle MAE 承载——已定义并文档化，非缺陷，如需"常量周期=完全一致"的扩展语义需另行决策。
3. 仓库无版本控制，本任务的代码变更无 git 提交记录可追溯（建议尽早 `git init`，与 Agent-Kernel 侧治理待办同源）。
