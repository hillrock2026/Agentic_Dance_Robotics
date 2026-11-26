# K1_DANCE_SKILL_FRAMEWORK_02_FINAL_INDEPENDENT_REVIEW_REPORT_01

日期：2026-11-20。评审人角色：独立技术评审（GLM-5.3）。评审对象：`K1_DANCE_SKILL_FRAMEWORK_02_FINAL_REMEDIATION_01` 的三项交付（B / D / SHA256）。评审方式：直接读代码 + 全新探针（不复用工程侧测试断言）+ 独立三次哈希计算。范围限定：不重审已通过的 A、C、双 Skill 回归及历史项目。

> 环境差异记录：任务书工作区 `Nao_Program/nao_gazebo_ws` 与实际不符；实际仓库为 `/home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics`（非 git 仓库，以 SHA256 快照承担可追溯性）。

## A. 总体结论

# **PASS WITH QUALIFICATIONS — Framework 02 可以正式关闭**

qualifications 仅一条且不阻塞：SHA256 的**历史**完整性证据只存在 16 位前缀（历史记录如此），无法独立证明完整 64 位前后一致；当前哈希已独立验证。按任务书第 8 节原则，这不构成回滚或重训理由。

## B. B 项结果：PASS

**代码检查**（`k1_skills/evaluation.py`）：

- `arm_motion_velocity`：实现了硬性帧账目不变量（报告箱 + 小箱省略 + 非有限 == 总数，违反即抛错）；每箱带 `nonfinite_err_samples`，箱内含非有限 → `mae_mrad: None`；顶层 5 个计数字段使输入帧数与有效分段一一可对账。评审确认旧行为（NaN/±Inf 帧静默脱离所有速度箱、MAE 照常输出）已消除。
- `arm_motion_amplitude`：三态分类（有效移动/有效静态/无效），静态桶只收有效静态关节；`invalid_reference_joints`/`invalid_actual_joints` 显式列出无效关节；无效参考不再因 `ref_amp=NaN` 被归入静态。
- 严格 JSON：`evaluate()` 兜底 + CLI `allow_nan=False` 双层。

**全新探针**（独立构造，非工程测试复用）：

| 探针 | 结果 |
|---|---|
| 200 帧 + 1 NaN 速度帧 | `nonfinite=1, binned=199, total=200`，不变量成立，峰值 None ✓ |
| 追加 +Inf 速度帧（合计 2 无效） | `binned=198`，不变量 198+0+2=200 成立 ✓ |
| NaN 参考关节（幅度） | 进入 `invalid_reference_joints=[4]`，static=6（8−1移动−1无效），静态桶长 6 且全为数值 ✓ |
| 真实 mj2 评价输出 | `json.dumps(allow_nan=False)` 通过 ✓ |

**正常数据回归**：arm_wave arms 75.4358 / mj2 arms 68.4856 mrad 与历史逐位一致；工程侧全量 150 项测试中 B 定向 12 项断言了计数值与恒等式（非仅无异常）。

## C. D 项结果：PASS

**Canonical 唯一性**：`canonical_joint_orders()` 在信任前强制 22 项、名称逐一唯一（显式重复检测，不再只比排序列表）、两序同集合。探针：真实 npz/csv 序各 22 且唯一 ✓；monkeypatch 注入重复名 → SkillError 拒绝 ✓；canonical 不可加载 → fail-closed（既有测试覆盖，代码路径确认）。

**具名资源**：逐项匹配 canonical（缺/多/重复/未知/错序在既有测试中各自拒绝；本轮抽验 arm_wave 三份 npz validate 全绿）。

**无名称资源身份链**（本轮重点）：声明 = `joint_order: train_npz`（指向通过 canonical 验证的具名资源）+ `content_sha256`（64 hex，schema 校验格式）→ 加载时重算摘要。四条全新攻击探针全部正确拒绝：

| 攻击探针 | 结果 |
|---|---|
| 真实 mj2 部署 npz 交换两列（保留原声明哈希）→ 加载 | **拒绝**（字节变化→摘要失配；数值层不可辨的交换同样被捕获）✓ |
| YAML 中声明哈希被篡改为 64 个 0 | **拒绝** ✓ |
| 声明缺少 content_sha256 | **拒绝**（"lacks content_sha256 binding"）✓ |
| 真实文件中相同值列对检测 | 实际不存在相同列对；即便存在，字节级绑定仍可检测交换（合成测试已证） |

**5.4 相同轨迹列交换专项**：工程侧测试（合法声明+绑定 → 交换相同值两列 → 保留原声明 → 拒绝；显式重声明才重新接地）语义正确——身份由声明链+内容绑定承载，**不从数值推断**，符合"不得声称可从相同轨迹推断身份"的要求。

**fail-closed 复核**：无名称无声明 / 声明指向未验证资源 / 绑定失配 / canonical 不可用，四类均明确报错（探针+代码路径双确认）。

## D. SHA256 结果

评审独立第三次计算与工程侧快照 12/12 一致；历史 16 位前缀与修复前完整哈希 12/12 匹配；**历史完整 64 位哈希不存在**（历次记录仅存前缀）。

| Asset | Current SHA256（评审独立计算） | Historical Evidence | Result |
|---|---|---|---|
| arm_wave_train_npz | `1d26a1ed56281f1f4c1fa706d7f3adc4d5bebcf103852eebd73e1296c83d8872` | 前缀匹配（16/16 位） | 当前已验证；历史前缀级 |
| arm_wave_usr_train_npz | `f84d9aff1d6b79a7211a9bc636913d098b103ad2d302bad414b288b4ad0ef8a5` | 前缀匹配 | 同上 |
| arm_wave_usr_cyclic_npz | `9fa4b74b408d8f536035984e564131e12c2f0e61afae97de8d26ca44ba56049a` | 前缀匹配 | 同上 |
| arm_wave_baseline_jit | `1fa8b882948ff63f36164ac61ba0ecfe64c15fd8aaeba03e2be6426db7c4a705` | 前缀匹配 | 同上 |
| arm_wave_trained_jit | `2de5bc07ab5d6aed493fe676ad1aafff881d5720596991277cb4a72b1bfae381` | 前缀匹配 | 同上 |
| mj2_train_npz | `99532a4280c6504f8ac822f1f84dc7c016f86818570348bc31345c5ee45505e6` | 前缀匹配 | 同上 |
| mj2_deploy_npz | `22840f7311ce87131c29341a8aa26de28a25c9f7fa999e13e52abcdbd2253224` | 前缀匹配 | 同上 |
| mj2_pretrained_pt | `bfb9b6ad86f7ef66cd8f97489878d2de15c253e435a04054306f38b2f6f29e15` | 前缀匹配 | 同上 |
| arm_wave_model2999 | `0dec324338d3618c0b3ccea4fa9708cad8d4a42198d5734f8fe9bdd593d283b4` | 前缀匹配 | 同上 |
| arm_wave_skill_yaml | `fc545ea7b1794e2ec41df99bebab17ff7709441f92c209fc39a1178fa3b54bf5` | 前缀匹配 | 同上 |
| mj2_skill_yaml | `e1a0c71a4cf7c5344d6afd2e99948cbb3ebce1c33e8f0a9035d4b10f1b0f1fcc` | 修复前完整哈希 `001b0fbe…b12e565`（本轮快照）+ 更早前缀匹配；变更 = 本任务允许的最小元数据（+content_sha256 两行），已披露 | 有意变更，已说明 |
| k1_default_profile | `52fe46f2121411efbc3af75fa511e93ea72016dd0d5e001279d7beb3d4ccd2c2` | 前缀匹配 | 当前已验证；历史前缀级 |

无文件缺失、无路径混淆。未修改任何受保护资产以制造一致（mj2_skill_yaml 的变更属任务 §7.3 明示允许且已披露）。

## E. 最终关闭建议

1. **阻塞 Framework 02 关闭的问题：无。**
2. **不阻塞的证据局限**（如实披露，无需行动）：历史 SHA256 仅 16 位前缀——自本轮起的完整 64 位快照已建立（`k1_runs/final_remediation_01_*_full_sha256.txt`），未来不再有此局限。
3. **不需要继续处理的问题**：mj2 trained 角色无全量训练策略（主线工作）、仓库无版本控制（历史遗留，建议尽早 `git init` 但不属 Framework 02 范畴）。

**建议：正式关闭 Framework 02。** 项目主线转入 Booster K1 实际街舞动作开发——降低新 Skill 接入成本、接入真实街舞动作资源、完成 K1 Retargeting、RL 训练与 MuJoCo 执行，以真实动作质量作为下一阶段的衡量标准。
