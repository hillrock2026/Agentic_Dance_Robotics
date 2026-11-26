# K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01_REPORT

日期：2026-11-20

## 1. 根因

`k1_skills/evaluation.py::arm_motion_amplitude` 的静态分类为
`static = ref_ok & ~(ref_amp > 0.1)` —— 只检查参考序列有效性。参考有效静态、
实际序列含 NaN/±Inf 的关节因此**同时**出现在 `invalid_actual_joints` 与
`static_arm_joints`（静态桶内混入 None 幅度条目）。复现实测：组内 8 关节含 1
个该情形时，`invalid_actual=[3]` 且 `static_arm_joints=7`（1+7=8，无效关节被
重复计入有效类）。

## 2. 修改文件与代码差异

仅 `k1_skills/evaluation.py` 的 `arm_motion_amplitude()`，三行语义变化：

```diff
-    moving = ref_ok & (ref_amp > 0.1)
-    static = ref_ok & ~(ref_amp > 0.1)          # valid static only
-    ratios = np.where(moving & act_ok, act_amp / ..., np.nan)
+    valid = ref_ok & act_ok
+    moving = valid & (ref_amp > 0.1)
+    static = valid & (ref_amp <= 0.1)           # valid static only
+    ratios = np.where(moving, act_amp / ..., np.nan)
```

有效性 = 参考**与**实际序列均完全有限；移动/静态/无效三类互斥且完备
（moving ∪ static = valid，invalid = ~valid）。`ratios` 中 `act_ok` 冗余条件
随之移除（moving ⊆ valid 已保证 act_amp 有限）。字段命名与数据结构零变化。

## 3. 新增测试及结果

`tests_k1_skills/test_amplitude_classification_fix_01.py`（11 项，全过）：

| 测试 | 内容 | 结果 |
|---|---|---|
| A/B/C（参数化 3 项） | 静态参考 + 实际 NaN/+Inf/−Inf → 入 invalid_actual、**不入** static（static=6、桶长 6 且全数值、moving=1） | ✓ |
| D | 参考与实际均有效 → moving 1 / static 7 / ratio≈1.0，分类不变 | ✓ |
| E | 参考无效 + 实际有效 → 入无效、不入移动/静态（static=6） | ✓ |
| F（参数化 6 项） | 互斥完整性：各种无效投放位置下 `moving + static + 无效数 == 组内关节数`，静态桶无 None；moving 关节实际无效时 moving=0 且 ratio=None | ✓ |

另有一处**既有测试断言更新**：`test_final_remediation_01.py::test_invalid_act_joint_ratio_none`
原断言 `moving_arm_joints == 1`（钉住"moving 只看参考"的被取代语义），按本修复
契约更新为 `moving_arm_joints == 0, static_arm_joints == 7`。属本修复的直接
对应测试，非放宽。

## 4. 正常数据回归结果

| Skill | moving（期望） | mean_amplitude_ratio（期望） | 结果 |
|---|---|---|---|
| arm_wave（trained 3 周期实测数据） | 4（4） | 0.9424192670413977（同） | **逐位一致** |
| mj_dance_002（baseline 实测数据） | 8（8） | 0.8966770149443131（同） | **逐位一致** |

## 5. 严格 JSON

两个 Skill 的完整评价对象与全部 161 项测试中的每项 amplitude 输出均通过
`json.dumps(..., allow_nan=False)`（evaluate 内置兜底 + 测试内联门）。

## 6. 是否改变其他评价指标

没有。修改仅涉及 `arm_motion_amplitude` 内部分类布尔式；抽样核验
`joint_tracking_error.mae_arms = 75.4358…` 与历史逐位一致，`arm_motion_velocity`
及所有其他指标族代码未触碰。测试总数 150 → **161 passed / 0 failed**（新增
11；1 项断言按新契约更新）。未重新训练任何策略；未修改动作资源、模型、
Skill 配置、manifest、content_sha256。
