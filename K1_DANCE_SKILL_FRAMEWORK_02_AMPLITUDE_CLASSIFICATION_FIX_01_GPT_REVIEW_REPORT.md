# K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01_GPT_REVIEW_REPORT

日期：2026-11-20

## 1. 总体结论

**PASS**

`K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01` 已真正消除原始缺陷：静态参考轨迹配合非有限实际轨迹时，该关节不再进入 static 类，也不再进入静态 residual 数值统计。moving、static、invalid 三类互斥且完备，正常 Skill 数据结果保持逐位不变。

本任务可以正式验收关闭。

## 2. 分项结论

| 评审项 | 结论 | 证据摘要 |
|---|---|---|
| A. 修复代码 | **PASS** | `valid=ref_ok & act_ok`；moving/static 均以 valid 为前提；静态阈值保留 `<= 0.1` |
| B. 原始缺陷路径 | **PASS** | 静态 ref + NaN/+Inf/-Inf actual 均只进入 invalid，不进入 static；residual 全为有限值 |
| C. 三态互斥与完备性 | **PASS** | moving∩static、moving∩invalid、static∩invalid 均为空；moving∪static=valid，invalid=~valid |
| D. 正常数据回归 | **PASS** | Arm Wave 4 / `0.9424192670413977`；mj2 8 / `0.8966770149443131`，独立复算逐位一致 |
| E. 输出兼容性 | **PASS** | 字段结构未变；无效关节不进入 static residual；严格 JSON 通过 |
| F. 修改范围与测试证据 | **PASS** | 修复局限于 amplitude 分类及直接相关测试/报告；11 项定向测试、161 项全量测试通过；12 个受保护资产与前次快照一致 |

## 3. 代码核验

### A. 修复代码

[arm_motion_amplitude()](k1_skills/evaluation.py#L147) 当前关键逻辑为：

```python
ref_ok = np.array([_series_ok(ref[:, j]) for j in range(n_joints)])
act_ok = np.array([_series_ok(act[:, j]) for j in range(n_joints)])
valid = ref_ok & act_ok
moving = valid & (ref_amp > 0.1)
static = valid & (ref_amp <= 0.1)
```

核查结果：

- `ref_ok` 与 `act_ok` 均要求对应完整关节序列全部有限，语义与 `valid` 定义一致。
- moving 仍按参考幅度阈值 `ref_amp > 0.1` 判断，没有改变移动判定标准。
- static 同时要求 reference 和 actual 有效。
- invalid 等价于 `~(ref_ok & act_ok)`，因此同时覆盖单侧无效和双侧无效。
- `ratios` 仅对 moving 计算；moving 已包含 `act_ok`，因此 ratio 不会重新接纳无效 actual。
- residual 只从 `act_amp[static]` 生成，static 内每项均满足 reference/actual 有限，因此不会混入 `None`。

### 阈值边界

GPT-6 独立构造精确/相邻浮点幅度，结果为：

- `nextafter(0.1, 0)`：static；
- `0.1`：static；
- `nextafter(0.1, +inf)`：moving。

这与原分类约定 `~(ref_amp > 0.1)` 对有效数据的语义一致，且避免了 NaN 被该比较式误归入 static。

## 4. 原始缺陷路径与分类不变量

GPT-6 对静态参考和移动参考分别注入 NaN、`+Inf`、`-Inf` 实际数据。每种情况下：

- 被投毒关节出现在 `invalid_actual_joints`；
- 不计入 `moving_arm_joints`；
- 不计入 `static_arm_joints`；
- 不进入 `static_joint_residual_amplitude_rad`；
- residual 列表中的有效项均不为 `None`。

独立构造 8 个关节、3 moving + 2 static + 3 invalid（参考无效、实际无效、双方均无效）后，得到：

```text
moving=3
static=2
invalid union={4,5,6}
3 + 2 + 3 = 8
```

由实现直接可得：

```text
moving & static = empty
moving & invalid = empty
static & invalid = empty
moving | static = valid
invalid = ~valid
```

`invalid_reference_joints` 与 `invalid_actual_joints` 保留各自语义；单个关节若两边同时无效会出现在两个列表中，但三态 invalid 使用两个列表的集合并，不会产生重复分类。

现有新增测试 [test_amplitude_classification_fix_01.py](tests_k1_skills/test_amplitude_classification_fix_01.py) 调用真实生产函数 `ev.arm_motion_amplitude`，不是脱节的辅助实现。A–C 的三种静态 actual 投毒测试、E 的 reference 投毒、F 的六种 placement 均通过。

## 5. 正常数据回归

GPT-6 直接加载实际 metrics NPZ，并独立按以下方式复算：

```text
valid = ref_fully_finite & actual_fully_finite
moving = valid & (reference_amplitude > 0.1)
ratio = mean(actual_amplitude / reference_amplitude) over moving
```

结果：

| 实际数据 | moving | mean_amplitude_ratio | arms MAE 抽样 |
|---|---:|---:|---:|
| Arm Wave trained 三周期 | 4 | `0.9424192670413977` | `75.43583497931853` mrad |
| mj_dance_002 baseline | 8 | `0.8966770149443131` | `68.48562276321793` mrad |

两项 ratio 与修复前记录逐位相等，ratio 的分母和 moving 集合未改变。两 Skill 的完整评价对象均通过 `json.dumps(..., allow_nan=False)`。

## 6. 输出兼容性

本次修改未改变以下字段的名称或结构：

- `moving_arm_joints`
- `static_arm_joints`
- `invalid_reference_joints`
- `invalid_actual_joints`
- `mean_amplitude_ratio`
- `amplitude_ratio_per_joint`
- `static_joint_residual_amplitude_rad`

正常有限数据下：

- reference_amplitude / actual_amplitude 的计算方式未变；
- moving 阈值未变；
- ratio 仍只对有限 moving ratio 求均值；
- static residual 只增加更严格的有效性前提，不引入旧输出之外的字段。

非有限数据下：invalid 关节不会通过 ratio 或 static residual 再进入正常数值统计。

## 7. 修改范围与测试证据

实际执行了：

```text
python -m pytest tests_k1_skills/test_amplitude_classification_fix_01.py -q
11 passed
```

以及全量：

```text
python -m pytest tests_k1_skills -q
161 passed, 2 warnings
```

两次命令退出成功；报告中的 `161 passed / 0 failed` 得到独立复核。

直接相关修改可见于：

- [k1_skills/evaluation.py](k1_skills/evaluation.py#L147)：amplitude 三态分类；
- [test_amplitude_classification_fix_01.py](tests_k1_skills/test_amplitude_classification_fix_01.py)：11 项专项测试；
- [test_final_remediation_01.py](tests_k1_skills/test_final_remediation_01.py#L139)：既有 moving-actual 无效断言更新；
- [执行报告](K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01_REPORT.md)。

本工作区不是 Git repository，因此无法提供 `git diff`。范围核验改用可用的实际证据：

- `arm_motion_velocity()` 当前实现与上轮已核验实现一致，未发现本次触碰痕迹或语义变化；
- Joint Identity、manifest 和 content hash 代码未被纳入本修复；
- 12 个受保护资产相对上一轮完整 post-fix SHA256 快照为 **12/12 完全不变**；
- 未发现测试删除、skip 标记、断言降低、动作资源、策略、Skill 配置或训练参数变更。

## 8. 最终验收意见

1. **原始 Amplitude 静态分类缺陷是否消除？** 是。
2. **moving / static / invalid 是否互斥且完备？** 是。
3. **正常 Skill 评价结果是否保持不变？** 是，两个目标值逐位一致。
4. **是否存在本任务范围内阻塞问题？** 否。
5. **本任务是否可以正式验收关闭？** 是。

**K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01：PASS，可关闭。**
