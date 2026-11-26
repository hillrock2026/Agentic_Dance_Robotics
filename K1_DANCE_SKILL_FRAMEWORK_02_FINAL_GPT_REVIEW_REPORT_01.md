# K1_DANCE_SKILL_FRAMEWORK_02_FINAL_GPT_REVIEW_REPORT_01

日期：2026-11-20

评审范围仅限 FINAL_REMEDIATION_01 的 B、D 和 12 个受保护资产 SHA256。未重审 A、C、双 Skill 完整回归或其他 Framework 工作。

## A. 总体结论

**结论：PARTIAL PASS**

- **FINAL_REMEDIATION_01 是否通过：否，未完整通过。**
- **Framework 02 是否可以关闭：NO。**
- 阻塞项：B 的 amplitude 三态分类仍允许实际轨迹非有限的关节进入“有效静态”计数/列表。
- D 已通过，且相同数值列交换的报告解释基本严谨。
- SHA256 为 **PASS WITH QUALIFICATIONS**；当前完整证据正确，历史仅有 16 位前缀的限制已披露。

## B. B 项：PARTIAL

### 实际代码证据

#### `arm_motion_velocity()`

[当前实现](k1_skills/evaluation.py#L192)正确完成以下行为：

- 对参考速度先取绝对值，再以逐行最大值生成分箱速度。
- NaN、`+Inf` 和 `-Inf` 分别显式表现为非有限速度帧；`-Inf` 经 `abs()` 后为 `Inf`，不会进入有限箱。
- 每个输入帧被划分为：已报告箱、被省略的小箱、或非有限帧。
- 帧账目硬不变量为：

  `binned_frames + small_bin_frames_omitted + nonfinite_speed_frames == total_frames`

  违反时抛 `EVAL_FAILED`。
- 每个已报告箱包含 `nonfinite_err_samples`；箱内误差包含非有限值时 `mae_mrad=None`。
- `ref_peak_speed_rad_s`、`act_peak_speed_rad_s` 使用严格 max 语义，含非有限值时返回 `None`。

GPT-6 独立探针确认：分别注入 NaN、`+Inf`、`-Inf` 后均为 `total_frames=200`、`nonfinite_speed_frames=1`，账目始终精确为 `199 + 0 + 1 = 200`；严格 JSON 序列化通过。

[定向测试](tests_k1_skills/test_final_remediation_01.py#L42)覆盖了三类非有限值、部分无效、全部无效、箱内误差非有限和有限正常数据。定向文件本次独立执行结果为 **22 passed**。

结论：velocity 部分满足任务要求，未发现剩余阻塞问题。

#### `arm_motion_amplitude()`

[当前实现](k1_skills/evaluation.py#L147)分别计算 `ref_ok`、`act_ok`，并输出 `invalid_reference_joints`、`invalid_actual_joints`。参考序列非有限的关节不会进入移动或静态集合；全部参考无效时移动数、静态数均为 0，均值为 `None`。

正常有限数据与旧公式独立复算一致：Arm Wave moving 数 4/4、`mean_amplitude_ratio=0.9424192670413977`；mj2 moving 数 8/8、ratio=`0.8966770149443131`。严格 JSON 通过。

但是实现中的静态分类为：

```python
static = ref_ok & ~(ref_amp > 0.1)
```

它只依赖参考序列，不要求 `act_ok`。因此，实际轨迹非有限但参考轨迹有效静态的关节仍被计入 `static_arm_joints`，其 residual 仅被输出为 `None`。

GPT-6 独立探针将一个有效静态关节的实际轨迹设为 NaN，得到：

- `invalid_actual_joints = [1]`；
- `static_arm_joints = 7`，仍包含该无效实际关节；
- `static_joint_residual_amplitude_rad` 中出现 `None`，列表仍含 7 项。

这与本次任务要求的三态分类（有效移动 / 有效静态 / 无效）以及“无效关节不进入静态关节列表”不一致。现有测试只在移动关节上污染 actual，因此没有捕获该分支。

### B 判定

**PARTIAL**。Velocity 修复通过，但 amplitude 对实际序列无效的静态关节仍混入有效静态桶。该项是关闭 Framework 02 的实际阻塞问题。

## C. D 项：PASS

### Canonical 唯一性

[canonical_joint_orders()](k1_skills/robot.py#L88)在信任顺序前检查：

- npz/csv 各恰好 22 项；
- 每个列表逐一唯一；
- 显式检测重复名称；
- 两列表包含相同的 22 关节集合。

GPT-6 使用配置的 booster_train Python 独立验证真实 `K1_CFG`：npz 为 22/22 唯一，csv 为 22/22 唯一，集合相同。定向测试还注入重复、缺项和集合不匹配，均被拒绝。canonical 加载异常时 `MotionResourceSet.from_spec()` 记录为不可用，verify 保持 fail-closed。

### 具名资源验证

[资源验证](k1_skills/motion.py#L204)要求具名 `joint_names` 与 canonical 列表逐项完全相等，不使用排序后相等或数值相似度作为身份。GPT-6 独立验证：Arm Wave 三份具名 npz、mj2 的具名 train npz 均通过；缺失、重复、未知和错序由既有测试拒绝。

### 无名称资源身份声明与 `content_sha256`

[Schema](k1_skills/spec.py#L265)接受可选的 64 位十六进制摘要，并拒绝错误类型、长度或非十六进制内容。

[Joint-order 契约](k1_skills/motion.py#L235)进一步要求无名称资源：

1. 指向已通过 canonical 验证的具名资源；
2. 携带 `content_sha256`；
3. 每次加载时从实际文件重新计算 SHA256；
4. 摘要不同即将身份声明视为失效；
5. 数值交叉验证仅作为矛盾检测，不能单独证明身份。

[mj2 skill.yaml](skills/mj_dance_002/skill.yaml#L31)中的声明摘要为当前 deploy npz 的实际摘要，GPT-6 独立验证链路成立。

### 相同轨迹列交换测试的正确解释

[专项测试](tests_k1_skills/test_final_remediation_01.py#L272)首先建立有效声明和原摘要，使两列轨迹完全相同，然后交换列并保留旧声明。测试明确记录“字节确实变化，因此摘要不匹配并拒绝”，符合任务书的区分要求。

GPT-6 独立复现得到：

- 交换后 `joint_pos` 的数组字节完全相同；
- 由于 `np.savez` 重写了压缩包容器，整个 `.npz` 文件字节不同；
- 因此旧文件摘要失效，资源被拒绝。

这必须解释为**文件内容绑定检测到了容器字节变化**，不能解释为从数值相同的两条轨迹推断出了交换。如果操作完全不改变最终文件字节，SHA256 无法检测该操作；此时系统依赖原始受信任的 `joint_order` 声明来解释列身份。当前代码和专项测试没有依赖不可能的数值推断。

### 身份信任链

信任链成立：

`受信任的 skill 配置声明` → `canonical 验证后的具名资源` → `无名称资源的 content_sha256` → `实际文件逐次重算 SHA256`。

SHA256 只提供内容绑定，不单独提供身份来源；身份来源来自受信任配置中的 `joint_order` 声明及其指向的 canonical 具名资源。显式修改声明和摘要属于修改受信任配置，而不是未声明路径绕过。

### D 判定

**PASS**。未发现本任务范围内的残余阻塞问题。

## D. SHA256 核查

GPT-6 独立计算当前 `final_remediation_01_post_fix_full_sha256.txt` 中的 12 个路径，结果为 **12/12、64 位、路径唯一且与实际文件一致**。历史 16 位前缀为 **12/12 匹配**。`pre_fix` 清单也有 12 个完整记录；其中 mj2 skill.yaml 按预期与当前不同，其他 11 项与实际当前文件一致。

| Asset | 当前 SHA256 | 核查结果 |
|---|---|---|
| arm_wave_train_npz | `1d26a1ed56281f1f4c1fa706d7f3adc4d5bebcf103852eebd73e1296c83d8872` | 一致 |
| arm_wave_usr_train_npz | `f84d9aff1d6b79a7211a9bc636913d098b103ad2d302bad414b288b4ad0ef8a5` | 一致 |
| arm_wave_usr_cyclic_npz | `9fa4b74b408d8f536035984e564131e12c2f0e61afae97de8d26ca44ba56049a` | 一致 |
| arm_wave_baseline_jit | `1fa8b882948ff63f36164ac61ba0ecfe64c15fd8aaeba03e2be6426db7c4a705` | 一致 |
| arm_wave_trained_jit | `2de5bc07ab5d6aed493fe676ad1aafff881d5720596991277cb4a72b1bfae381` | 一致 |
| mj2_train_npz | `99532a4280c6504f8ac822f1f84dc7c016f86818570348bc31345c5ee45505e6` | 一致 |
| mj2_deploy_npz | `22840f7311ce87131c29341a8aa26de28a25c9f7fa999e13e52abcdbd2253224` | 一致 |
| mj2_pretrained_pt | `bfb9b6ad86f7ef66cd8f97489878d2de15c253e435a04054306f38b2f6f29e15` | 一致 |
| arm_wave_model2999 | `0dec324338d3618c0b3ccea4fa9708cad8d4a42198d5734f8fe9bdd593d283b4` | 一致 |
| arm_wave_skill_yaml | `fc545ea7b1794e2ec41df99bebab17ff7709441f92c209fc39a1178fa3b54bf5` | 一致 |
| mj2_skill_yaml | `e1a0c71a4cf7c5344d6afd2e99948cbb3ebce1c33e8f0a9035d4b10f1b0f1fcc` | 一致；为本轮唯一有意变更项 |
| k1_default_profile | `52fe46f2121411efbc3af75fa511e93ea72016dd0d5e001279d7beb3d4ccd2c2` | 一致 |

### mj2 skill.yaml 变更

从当前文件逆向仅还原旧注释块并移除新增 `content_sha256` 行后，得到完整哈希：

`001b0fbe259a24e39bb36447f4e8af4c302e9c051eae9ce3e1285fc7eb12e565`

该值与 [pre-fix 完整快照](k1_runs/final_remediation_01_pre_fix_full_sha256.txt)完全一致。实际变更除新增摘要行外，还更新了直接说明该摘要语义的注释；没有发现动作数据、训练配置或其他业务定义变更。工程报告称“仅增加两行”略有文字不准确，因为 YAML 注释块也被改写，但该差异只涉及本任务要求的身份绑定说明，不影响资产保护结论。

### 历史证据局限

旧记录 [remediation_01_baseline_sha256.txt](k1_runs/remediation_01_baseline_sha256.txt) 和 [remediation_01_after_sha256.txt](k1_runs/remediation_01_after_sha256.txt) 只有 16 位前缀，不能证明更早时期完整 64 位一致性。本轮已建立 pre/post 完整快照，因此当前结论可独立复算；不得将前缀匹配误写为历史完整哈希证明。

**SHA256 结论：PASS WITH QUALIFICATIONS。**

## E. 最终验收

| 项目 | 结论 |
|---|---|
| B | **PARTIAL** |
| D | **PASS** |
| SHA256 | **PASS WITH QUALIFICATIONS** |
| FINAL_REMEDIATION_01 | **PARTIAL PASS** |
| Framework 02 关闭 | **NO** |

### 唯一阻塞关闭的问题

`arm_motion_amplitude()` 对“实际序列非有限、参考序列有效静态”的关节仍计算为有效静态，并将其计入 `static_arm_joints`。这违背三态分类与无效关节不得进入静态桶的要求；现有定向测试没有覆盖该分支。

历史 SHA256 仅有 16 位前缀属于已披露的非阻塞证据局限；D 中相同数值列交换的不可辨识边界不构成额外工程要求。
