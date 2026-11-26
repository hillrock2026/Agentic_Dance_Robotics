# K1_DANCE_SKILL_FRAMEWORK_02_FINAL_REMEDIATION_01_REPORT

日期：2026-11-20。执行环境：实际仓库 `/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics`
（任务书所写 `Nao_Program/nao_gazebo_ws` 与实际工作区不符，按真实结构执行，已如实记录）。

## A. 执行摘要

| 项 | 状态 |
|---|---|
| B：Evaluation 非有限值隔离（velocity 帧账目 + amplitude 三态） | **完成** |
| D：Joint Identity（canonical 唯一性 + content_sha256 内容绑定） | **完成** |
| 12 资产完整 SHA256 证据 | **完成**（历史仅 16 位前缀，如实披露） |
| 阻塞问题 | 无 |

全量测试 **150 passed / 0 failed / 0 skipped**（原 128 + 新增 22）。未重新训练任何策略；唯一被修改的受保护资产是 `skills/mj_dance_002/skill.yaml`（按任务 §7.3 允许的最小元数据：为无名称部署资源补 content_sha256 绑定，变更已披露并附前后哈希）。

## B. B 项实施结果

**修改文件**：`k1_skills/evaluation.py`（`arm_motion_velocity`、`arm_motion_amplitude` 两个函数）。

**B-1 `arm_motion_velocity`**（根因：`speed = ref_v.max(axis=1)` 使 NaN/±Inf 帧落入不了任何速度箱，200 帧静默变 199 帧）：
- 新增硬不变量 **报告箱帧数 + 小箱省略帧数 + 非有限速度帧数 == 总帧数**（违反即抛 EVAL_FAILED，测试断言恒等式）；
- 非有限速度帧显式计数（`nonfinite_speed_frames`），不再静默消失；
- 每个速度箱带 `nonfinite_err_samples`；箱内误差含非有限 → `mae_mrad: None`（明确无效态）而非有限均值伪装；
- 顶层报告 `total_frames / binned_frames / small_bin_frames_omitted / nonfinite_speed_frames / nonfinite_err_samples`，输入帧数与有效分段一一可对账；
- 峰值速度保持严格 max 语义（含 ±Inf → None）。

**B-2 `arm_motion_amplitude`**（根因：参考序列含 NaN/Inf 的关节 `ref_amp=NaN` → `moving=False` → 被误归入"静态关节"桶）：
- 三态分类：**有效移动 / 有效静态 / 无效**（无效 = 参考或实际序列含任何非有限值）；
- 静态桶只收有效静态关节；无效关节进入 `invalid_reference_joints` / `invalid_actual_joints` 显式列表，幅度条目为 None；
- 新增 `static_arm_joints` 计数；`mean_amplitude_ratio` 只在有限比值上计算，无比值时 None。

**正常数据兼容性**：arm_wave trained arms MAE 75.4358…、mj2 baseline 68.4856… 与修复前逐位一致；moving/static 分类在有限数据下与原实现相同（测试覆盖）。

**新增测试**（`tests_k1_skills/test_final_remediation_01.py`）：velocity NaN/+Inf/−Inf/部分/全部/正常 6 项 + 幅度 NaN/+Inf/−Inf（参数化）/全无效/act 无效/正常 6 项，全部断言**计数值、恒等式与桶语义**（非仅"不抛异常"），并全部经 `json.dumps(allow_nan=False)`。

## C. D 项实施结果

**D-1 canonical 唯一性**（根因：只查长度 + 排序相等，重复名配缺失名可通过）：
`k1_skills/robot.py::canonical_joint_orders` 现在在信任前强制：每列表**恰 22 项**、**名称逐一唯一**（显式重复检测）、两列表同关节集合；任何违例 raise（fail-closed）。测试以 monkeypatch 注入重复/缺数/集合不匹配三种畸形 canonical，全部拒绝。

**D-2 无名称资源内容绑定**（根因：`joint_order: train_npz` 声明只做数值交叉验证，声明与文件内容无强绑定——列交换（含相同列）后原声明仍被接受）：
- Schema（`k1_skills/spec.py`）：资源条目新增可选 `content_sha256`（强制 64 位 hex 格式校验，坏格式拒绝）；
- 契约（`k1_skills/motion.py::_verify_joint_order`）：无名称资源的 `joint_order` 声明**必须**携带 `content_sha256`，加载时**重算文件摘要**——不匹配即拒绝（"identity claim is VOID"）；声明链仍须指向通过 canonical 验证的具名资源；数值交叉验证保留为辅助证据（矛盾报错、吻合不证明身份）；
- 最小元数据变更：`skills/mj_dance_002/skill.yaml` 为部署 npz 补充其真实摘要
  `22840f7311ce87131c29341a8aa26de28a25c9f7fa999e13e52abcdbd2253224`（动作数据本身零改动）。

**5.4 相同轨迹列交换测试**（`test_identical_column_swap_rejected_by_binding`）：两列完全相同 → 合法声明+绑定建立 → 交换两列并保留原声明 → **字节变化 → 摘要不匹配 → 拒绝**（数值路径完全无法察觉该交换，内容绑定可以）；显式重新声明新哈希才重新接地（人工行为，非数值推断）。

**新增测试**：声明缺绑定拒绝 / 单 float 修改即失效 / 相同列交换拒绝+重声明 / 绑定格式校验 / 真实 mj2 绑定通过 / canonical 畸形拒绝，共 10 项。

**合法资源兼容性**：arm_wave（三份具名 npz）与 mj_dance_002（声明+绑定）`validate` 全绿；三个 deploy 任务策略加载并 5 步执行探针通过（`k1_arm_wave_usr` / `k1_arm_wave_usr_v2` / `k1_mj2`）。

## D(SHA256). 12 资产完整哈希证据

当前哈希独立重算两次一致（`k1_runs/final_remediation_01_{pre,post}_fix_full_sha256.txt`）；历史记录（remediation_01 系列文件）仅存 16 位前缀，12/12 与修复前完整哈希前缀吻合——**历史完整性只能证明到前缀级别**，不伪造历史完整哈希。

| Asset | Full SHA256（当前，修复后） | vs 修复前快照 | 历史证据 |
|---|---|---|---|
| arm_wave_train_npz | `1d26a1ed56281f1f4c1fa706d7f3adc4d5bebcf103852eebd73e1296c83d8872` | 一致 | 前缀匹配 |
| arm_wave_usr_train_npz | `f84d9aff1d6b79a7211a9bc636913d098b103ad2d302bad414b288b4ad0ef8a5` | 一致 | 前缀匹配 |
| arm_wave_usr_cyclic_npz | `9fa4b74b408d8f536035984e564131e12c2f0e61afae97de8d26ca44ba56049a` | 一致 | 前缀匹配 |
| arm_wave_baseline_jit | `1fa8b882948ff63f36164ac61ba0ecfe64c15fd8aaeba03e2be6426db7c4a705` | 一致 | 前缀匹配 |
| arm_wave_trained_jit | `2de5bc07ab5d6aed493fe676ad1aafff881d5720596991277cb4a72b1bfae381` | 一致 | 前缀匹配 |
| mj2_train_npz | `99532a4280c6504f8ac822f1f84dc7c016f86818570348bc31345c5ee45505e6` | 一致 | 前缀匹配 |
| mj2_deploy_npz | `22840f7311ce87131c29341a8aa26de28a25c9f7fa999e13e52abcdbd2253224` | 一致 | 前缀匹配 |
| mj2_pretrained_pt | `bfb9b6ad86f7ef66cd8f97489878d2de15c253e435a04054306f38b2f6f29e15` | 一致 | 前缀匹配 |
| arm_wave_model2999 | `0dec324338d3618c0b3ccea4fa9708cad8d4a42198d5734f8fe9bdd593d283b4` | 一致 | 前缀匹配 |
| arm_wave_skill_yaml | `fc545ea7b1794e2ec41df99bebab17ff7709441f92c209fc39a1178fa3b54bf5` | 一致 | 前缀匹配 |
| mj2_skill_yaml | `e1a0c71a4cf7c5344d6afd2e99948cbb3ebce1c33e8f0a9035d4b10f1b0f1fcc` | **变更（本任务 §7.3 允许的最小元数据：+content_sha256 两行）**；修复前 `001b0fbe259a24e39bb36447f4e8af4c302e9c051eae9ce3e1285fc7eb12e565` | 前缀匹配（修复前） |
| k1_default_profile | `52fe46f2121411efbc3af75fa511e93ea72016dd0d5e001279d7beb3d4ccd2c2` | 一致 | 前缀匹配 |

## E. 测试结果

| 阶段 | 命令/范围 | 结果 |
|---|---|---|
| B/D 定向 | `pytest tests_k1_skills/test_final_remediation_01.py` | 22 passed |
| 旧契约升级回归 | `pytest tests_k1_skills/test_minimal_fix_01.py test_remediation_01.py`（3 处按新绑定契约补 content_sha256，原断言意图保留） | 128 passed |
| 双 Skill 最小回归 | validate×2 全绿；3 个 deploy 任务策略 5 步探针 OK；双 Skill 评价严格 JSON 且数值不变 | 通过 |
| 全量 | `pytest tests_k1_skills -q`（booster_train conda env） | **150 passed / 0 failed / 0 skipped** |

未执行项：无（未重新训练，按任务边界）。

## F. 最终状态

- **B：PASS**（无效帧/关节不静默消失、明确无效态、计数可对账、正常数据不变、严格 JSON）
- **D：PASS**（canonical 22 唯一、具名逐项匹配、声明链 + SHA256 内容绑定、相同列交换拒绝、全链 fail-closed）
- **SHA256：PASS WITH QUALIFICATIONS**（当前完整哈希独立验证；历史完整性仅前缀级证据，如实披露）

**已达到 Framework 02 关闭条件**（待独立评审确认）。
