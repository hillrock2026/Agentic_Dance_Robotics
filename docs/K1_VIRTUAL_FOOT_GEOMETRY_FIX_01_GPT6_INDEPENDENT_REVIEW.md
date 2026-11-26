# K1_VIRTUAL_FOOT_GEOMETRY_FIX_01 — 独立技术评审

## 1. 总体结论

**PASS WITH QUALIFICATIONS**

本次修复正确解决了 Virtual K1 当前模型足底碰撞盒的中心高度误判、旋转遗漏和左右脚合并问题。独立复算表明，修复器对真实 K1 左右足的全部 8 个角点与 MuJoCo 的 `geom_xpos/geom_xmat` 结果一致，最大坐标误差为 `2.78e-17 m`；Arm Wave 和 Top Rock 的关键结果也得到独立复现。

有两项局部问题需要在解释证据时保留限定：无效 NaN 输入会被成功形状地分类为 `AIRBORNE`；Arm Wave 物理 trace 的汇总过滤器把动作结束后的 0.4 秒保持段计入“运动段”。这些不推翻当前 K1 box、水平地面下正常有限数据的核心几何修复，但降低了异常输入鲁棒性和所报接触统计的准确性。

## 2. 评审范围与方法

仅审查足底几何修复及直接调用链、测试和复核数据。没有修改项目源代码、配置、模型或动作数据；物理回放复验写到系统临时目录后删除。

检查和独立执行包括：

- `virtual_k1/virtual_k1/foot_geometry.py`、`virtual_k1/virtual_k1/simulator.py`、`k1_action/virtual_k1_replay.py`、`k1_action/scripts/foot_geometry_audit.py`。
- 官方 K1 XML 中左右脚 body、碰撞 geom 类型/尺寸/偏移，以及足部是否另有参与碰撞的 geom。
- 新增测试和两套直接相关回归测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest virtual_k1/tests/test_foot_geometry.py virtual_k1/tests/ k1_action/tests/ -q`，结果 **77 passed**。
- 从真实模型当前运行态逐角点比较检测器与 MuJoCo `geom_xpos/geom_xmat`；对 Arm Wave、Top Rock 原始轨迹逐帧复算。
- Arm Wave PD 物理回放独立重跑一次，临时 trace 与已交付 trace 字节一致。
- 按 manifest 重新计算三份 trajectory 和 metadata 的 SHA256。

**证据边界：**工作区根目录不是 Git 仓库，无法独立比较本次所有源码/配置改动的 diff 或证明修改范围绝无旁逸；官方资产子仓库状态干净。因而“动作文件与 manifest 一致”已验证，但“所有列出的文件均为本任务唯一改动”只能部分验证。

## 3. 原始错误核实

**结论：VERIFIED。**

- 当前运行环境 MuJoCo 版本为 `4.0.1`；其官方枚举值为 `mjGEOM_BOX=6`、`mjGEOM_CAPSULE=3`。修复前用数值 `3` 判断 K1 足部 box 的报告描述符合 MuJoCo 类型定义。
- [K1_22dof.xml](virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml#L138) 左足和 [同文件右足定义](virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml#L174) 各有一个启用碰撞的 box：`pos="0.026 0 -0.02"`、`size="0.09 0.035 0.018"`。两侧视觉 mesh 均 `contype=0, conaffinity=0`，不是碰撞体；每只足当前只有该一个有效碰撞 geom。
- 实际加载模型识别的 geom ID 为左 `33`、右 `44`，两者类型均为 6，尺寸、局部位移和单位四元数与 XML 相符。代码通过配置中的足部 body 名称和运行时碰撞 mask 枚举 geom，并未把这些 ID 写死。
- 新实现保留各侧独立结果，并按 geom 的实际局部位置、姿态和尺寸计算，不依赖 geom 排列顺序。

## 4. 几何变换独立复算

**结论：VERIFIED（当前 K1 box 和水平地面）。**

修复器在 [FootGeometryDetector._geom_world_corners](virtual_k1/virtual_k1/foot_geometry.py#L204) 实现：

`world = foot_xpos + foot_xmat · (geom_pos + geom_rotation · local_corner)`

其中局部 box 角点由模型运行时 `geom_size` 生成，body 的 `xpos/xmat` 来自 MuJoCo FK。变换链没有再重复叠加 root；root 已包含在 FK 的 body 世界变换中。

独立方法从实际加载模型的 `data.geom_xpos`、`data.geom_xmat` 和 `model.geom_size` 重建 8 个角点，并在 reset、左右踝姿态变化等 6 个足部姿态（两侧）比较全部 48 个角点：最大绝对坐标误差 `2.7755575615628914e-17 m`。检测器的 `verify_transform_chain()` 也以 MuJoCo geom 变换作交叉核验。Top Rock 每帧另以直接 `geom_xpos/xmat` 公式计算最低角点，得到与检测器完全一致的最低间隙、帧号和坐标（见第 9 节）。

对水平地面，代码按所有角点的有符号距离取最小值；穿透深度为 `max(0, -clearance)`。球体则使用球心到平面的有符号距离减去半径，间隙计算正确。倾斜平面从一般平面距离公式看可用于“到该平面的几何间隙”，但没有倾斜地面端到端测试；当前 K1 场景是 XML 中的水平地面，故该限制不影响本次使用。

## 5. 三态分类审查

**结论：PARTIALLY VERIFIED。**

- `PENETRATION` / `CONTACT` / `AIRBORNE` 在 [evaluate](virtual_k1/virtual_k1/foot_geometry.py#L239) 中按 `contact_tolerance_m` 分类。`|clearance| <= 2 mm` 为几何 `CONTACT`，不是 MuJoCo manifold 接触或受力；几何和物理接触的语义在 API 文档中区分。
- 代码对 `clearance=-2 mm`（恰在负边界）判 `CONTACT`，超出负边界判 `PENETRATION`；正边界同样由绝对值条件纳入 `CONTACT`。测试有精确负 2 mm 球体边界用例，但未专门用例覆盖正 2 mm。
- 默认 `numeric_tolerance_m=1e-6` 与接触容差字段分开保存；但 `numeric_tolerance_m` 在实现中没有被后续计算使用，测试的理论值容差来自测试文件中的独立常量。它不会错误影响三态分类，但当前不能称为被检测逻辑实际使用的数值容差。
- **异常输入存在明确缺陷：**`GroundPlane` 检查法向量单位长度，却未验证 offset 有限；检测器也未验证接触容差/计算结果有限。独立将 plane offset 设为 NaN，或将仿真 `qpos[0]` 设为 NaN 后执行 `mj_forward()`，`evaluate()` 均返回 `AIRBORNE` 且 `clearance=inf, penetration=0`，没有报错或标记无效。负值或 NaN 接触容差也被接受。此结果会把无效状态伪装成有效悬空状态，不符合异常输入应明确失败的要求。

## 6. Virtual K1 调用链审查

**结论：VERIFIED（调用集成和物理/几何区分）；trace 窗口统计有一项局部错误。**

- [K1Sim 初始化](virtual_k1/virtual_k1/simulator.py#L81)按配置足 body 创建检测器，[get_foot_geometry](virtual_k1/virtual_k1/simulator.py#L322)只返回几何结果，不推进仿真、不改 qpos/qvel/time。
- [kinematic_replay](k1_action/virtual_k1_replay.py#L31)逐帧 `mj_forward()` 后调用新 API，逐侧统计 clearance、penetration 和状态；旧中心高度检测不再用于该调用路径。检索结果未发现其他 replay 调用点继续依赖旧中心高度判定；旧公式仅保留在审计脚本作为对照。
- [physical_replay](k1_action/virtual_k1_replay.py#L95)的 trace 只有在 `trace_path` 非空时采集；每条记录在同一仿真步之后读取 `get_foot_contacts()` 的 manifold/法向力和 `get_foot_geometry()` 的几何间隙。默认关闭时，不执行新增 trace 采集。重新运行 Arm Wave 后，结果 `pass=True`、base z 范围 `0.546550–0.550990 m`，临时 trace 与交付 trace 字节完全相同。
- 接触力来自实际 MuJoCo `mj_contactForce` 路径，不是由质量除以二推算；几何 clearance 与 manifold 接触分别输出。Arm Wave 的运动学姿态逐帧没有 manifold 接触，但 PD 物理回放有接触，二者是不同仿真状态，不构成矛盾。

**局部统计缺陷：**`foot_geometry_audit.py` 对 trace 仅筛选 `time_s >= 1.5`，未限制动作结束时间；而 replay 在 246 帧动作结束后还运行 `run("hold", 0.4)`。因此报告的 `1577` 条“motion”记录含结束后的约 0.4 秒保持段。按动作区间 `[1.50, 6.42)` 独立重算为 `1475` 条，左/右有接触记录分别 `1469/1475`、`1474/1475`；包含端点 6.42 秒时为 1476 条。trace 的采样间隔也不是恒定 4 ms，实际出现 1、3、4 ms 间隔（每帧额外记录末步）。因此接触比例仍约为 99.6% / 99.9%，但报告的分母和“运动段 1.50–6.82 s / 1577 条”不准确。

## 7. 测试质量和独立复跑结果

**结论：VERIFIED（77 项通过；对异常值的覆盖不充分）。**

- 新增 `test_foot_geometry.py` 有 **25 项**测试。覆盖水平 box 间隙/穿透、X/Y/Z 与组合旋转、碰撞原点平移/旋转、左右脚独立、root/踝姿态变化、真实 K1 模型多姿态、sphere 间隙和 capsule 显式拒绝。
- 直接运行新增、Virtual K1 回归和 k1_action 回归，得到 **77 passed in 15.69s**，与执行报告的 25 + 35 + 17 项一致。
- 测试的理论旋转部分使用测试侧 Rodrigues/角点计算；真实 K1 姿态测试使用 XML 中的独立常量。此次额外独立比较全部 8 个真实 geom 角点与 MuJoCo 的 `geom_xpos/xmat`，补强了测试未逐个角点比较 MuJoCo 变换结果的证据。
- 没有发现测试失败、跳过或 xfail 证据。现有测试未覆盖 NaN root/plane offset/容差输入，也未覆盖倾斜地面的返回字段语义；这与第 5 节发现一致。

## 8. Arm Wave 数据独立复核

**结论：几何数据 VERIFIED；运动学与物理接触统计需要严格区分，物理 trace 汇总窗口 PARTIALLY VERIFIED。**

源文件 `datasets/k1_dance_v1/arm_wave/trajectory.csv` 逐帧加载后进行 `qpos + mj_forward`，共 246 帧。独立重算：

| 指标 | 左足 | 右足 |
|---|---:|---:|
| 最小 clearance | 4.685578 mm | 4.762979 mm |
| 最大穿透 | 0 mm | 0 mm |
| 几何 AIRBORNE 帧数 | 246/246 | 246/246 |
| 几何 CONTACT（±2 mm）帧数 | 0 | 0 |
| 运动学姿态中的物理 manifold 接触帧 | 0 | 0 |

数值与 `arm_wave_foot_audit.json` 和逐帧 CSV 一致；`arm_wave_foot_frames.csv` 行数与 246 帧吻合。独立物理回放 trace 的合法动作窗内有接触记录（见第 6 节），且法向力均从引擎 contact 数据读取。修复后的结论应表述为：**该动作的直接运动学回放足底实际高于地面约 4.7 mm；另一次自由 root 的 PD 物理回放中，机器人沉降后实际产生双足接触。**不能把后者当作前者的接触证据。

## 9. Top Rock 穿透结果复核

**结论：VERIFIED。**

对原始 `top_rock/trajectory.csv` 的 286 帧逐帧执行真实模型 FK，并直接以 `data.geom_xpos + local_corners @ data.geom_xmat.T` 计算 box 角点：

- 左足最大穿透 `58.031666 mm`，第 155 帧；最低角点世界坐标约 `(0.372434563, 0.093992536, -0.058031666) m`。
- 右足最大穿透 `58.390849 mm`，第 124 帧；最低角点世界坐标约 `(0.173927764, -0.077181732, -0.058390849) m`。

两项均与新检测器逐帧输出一致，右足结果复现报告中的约 58.391 mm。该证据只用于核实几何检测结果，不对动作本身作进一步评价。

## 10. 数据完整性及修改范围

**结论：数据 hash VERIFIED；完整源码修改范围 PARTIALLY VERIFIED。**

- 当前三个 `trajectory.csv` SHA256 与 manifest 中相应值全部 MATCH：Arm Wave `7afe1bb3…276dde1`；Top Rock `a25c91cc…38c022`；Popping `b5e12611…90062d`。三份 `metadata.json` SHA256 亦全部与 manifest MATCH。
- 当前官方 XML 足部定义符合报告参数，Booster 资产子仓库 `git status --short` 无输出；本次审查没有修改 XML、配置、映射或动作文件。
- 由于工程根目录不是 Git 仓库，无法独立证明执行者声称的完整修改清单、Motion Studio 输入或未列出的其他文件相对任务前版本未改变。mtime 不能替代 diff/hash 基线，故这部分不作超出证据的确认。

## 11. 发现的问题（按严重程度）

### MINOR — 无效数值状态被伪装为有效 AIRBORNE

- **位置：**[foot_geometry.py](virtual_k1/virtual_k1/foot_geometry.py#L58) 的 `GroundPlane` 校验、[检测器初始化](virtual_k1/virtual_k1/foot_geometry.py#L160)、[evaluate](virtual_k1/virtual_k1/foot_geometry.py#L239)。
- **直接证据：**NaN plane offset 和含 NaN root qpos 的实测都返回 `state=AIRBORNE, clearance=inf, penetration=0`；非有限/负容差未拒绝。
- **影响：**调用方可能把无效仿真状态误认为脚悬空。正常有限的当前 K1 box 输入不受影响。
- **最小建议：**在构造时验证 offset/容差有限且容差非负；evaluate 对输入角点和距离做 finite 检查，异常时抛明确错误。增加 NaN/Inf 测试。

### MINOR — Arm Wave 物理 trace 统计窗口多算动作后的保持段

- **位置：**[foot_geometry_audit.py](k1_action/scripts/foot_geometry_audit.py#L180)；replay 在 `physical_replay()` 动作循环后另有 `run("hold", 0.4)`。
- **直接证据：**过滤器只有 `time_s >= 1.5`，实际把结束时间 6.42 秒之后的 102 条记录纳入。报告所用 1577 条中包含该段；严格 `[1.5, 6.42)` 为 1475 条。
- **影响：**报告的分母和“运动段”时间范围不精确；按正确区间重算接触比例仍接近 99.6% / 99.9%，不改变“该次 PD 回放中存在双足真实接触”的结论。
- **最小建议：**按 `1.5 <= time_s < 1.5 + T/fps` 限定窗口，并从时间戳计算样本统计；不要把约 4 ms 标称值描述为严格等间隔。

### OBSERVATION — 几何结果中的最高点字段对 sphere/倾斜平面不准确

- **位置：**[FootGeometryResult 生成](virtual_k1/virtual_k1/foot_geometry.py#L239)。
- **直接证据：**`max_corner_z` 赋值来自最大有符号距离 `max_d`；sphere 的单个代表点距离已减半径，故该值是球底间隙而非球顶高度。斜平面时它是平面法向距离而非 world Z。
- **影响：**当前 K1 使用水平地面上的 box，现有 clearance/penetration 与三态不受影响；但 `max_corner_z_m` 字段在已声称支持的 sphere/倾斜平面情形下容易误解。
- **最小建议：**明确字段定义为“最大平面有符号距离”，或按 world Z 和各几何的真实最高支持点单独计算；为 sphere/斜面结果增加测试。

## 12. 最终判定

1. **足底几何修复是否真实有效：是。**针对当前 Booster K1 单 box 足底和水平地面的核心计算正确。
2. **类型识别、碰撞参数和左右脚区分：通过。**使用官方枚举及运行时 geom/body 关系；实际各侧单一有效 box 被独立识别。
3. **旋转、collision origin 和 root/FK 变换：通过。**全部角点独立对照误差在机器精度量级。
4. **Arm Wave 复核：**运动学回放 246 帧均悬空约 4.69/4.76 mm；独立物理 PD 回放真实产生接触。原报告将动作后保持段纳入“motion”计数，但修正窗口后结论仍成立。
5. **Top Rock 58.391 mm 右足穿透：通过复核。**最大帧和最低角点坐标已由 MuJoCo 实际 geom 变换复算。
6. **重复性与回归：**77 项测试通过，Arm Wave 物理 trace 复跑与交付文件完全一致；三种轨迹及 metadata hash 匹配 manifest。
7. **资格限定：**需避免让非有限输入返回成功形状的 AIRBORNE，并修正物理 trace 的统计窗口。工程源码的全量改动范围因缺少根仓库版本基线而未能验证。

因此，本专项判定为 **PASS WITH QUALIFICATIONS**，而不是无保留 PASS。该结论仅限 `K1_VIRTUAL_FOOT_GEOMETRY_FIX_01`，不代表任一动作通过完整动作验收。
