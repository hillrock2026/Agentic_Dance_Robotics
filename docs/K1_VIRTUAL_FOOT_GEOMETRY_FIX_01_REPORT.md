# K1_VIRTUAL_FOOT_GEOMETRY_FIX_01_REPORT — Virtual K1 足底几何检测修复工程报告

- 任务编号: K1_VIRTUAL_FOOT_GEOMETRY_FIX_01
- 日期: 2026-11-17
- 范围: 仅足底几何检测修复。本报告判定不代表 Arm Wave / Top Rock / Popping 任一动作通过完整动作质量验收。
- 复现入口:
  - 测试: `python -m pytest virtual_k1/tests/ k1_action/tests/`
  - Arm Wave 复核: `python k1_action/scripts/foot_geometry_audit.py --action datasets/k1_dance_v1/arm_wave --physical`
  - 产物目录: `virtual_k1/outputs/foot_geometry_fix01/`

---

## 1. 原始错误定位

错误实现位于 `k1_action/virtual_k1_replay.py::kinematic_replay`（修复前第 53–62 行），是 A 类运动学回放验证的唯一足-地检测。原代码：

```python
z_min = min(
    d.geom_xpos[g][2] - d.geom_size[g][1] if m.geom_type[g] == 3 else d.geom_xpos[g][2]
    for g in range(m.ngeom)
    if m.geom_bodyid[g] in (left/right_ankle_roll_link) and m.geom_contype[g])
foot_pen.append(float(z_min))
```

对真实模型（K1_22dof.xml）实测确认存在**三处独立错误**：

1. **几何类型常量错误，条件永远为假**。MuJoCo 中 `mjGEOM_BOX = 6`，常量 `3` 是 `mjGEOM_CAPSULE`。K1 足部碰撞体是 box（type 6，实测 geom id 33/44），因此 `== 3` 分支从不执行，检测退化为**纯几何中心高度** `geom_xpos[z]`——连半尺寸都没有减。这正是 Arm Wave 旧报告值 `ground_penetration_min_m = +0.0233 m` 的来源：它就是足部碰撞盒中心的实际世界高度（实测中心 z = 0.0192 m @ root 0.551，root 抬到 0.555 后 ≈ 0.0233 m）。
2. **无旋转处理**。即使减半尺寸，也只对未旋转 box 成立。足踝 pitch/roll 或 root 非偏航旋转后，真实底面角点为 `center_z − (hx·|sinθ| + hz·|cosθ|)`（绕 Y 旋转 θ），不再等于中心 z 减固定半尺寸。Top Rock 的 58.4 mm 穿透即由此漏检。
3. **左右脚混取全局 min**，无法输出左右足各自的间隙/穿透/接触状态。

另注：`virtual_k1/tests/test_foot_contact.py:45` 使用 `geom_xpos[2] − geom_size[2]`（正确的 z 半尺寸索引），仅在 reset 平放足姿态下近似成立，属既有测试、本次未改动（见 §6 回归）。

真实足部碰撞几何（K1_22dof.xml 138–144 / 174–180 行，左右相同）：

```xml
<geom type="box" pos="0.026 0 -0.02" size="0.09 0.035 0.018"/>
```

- box（非 mesh；足 mesh `contype=0` 不参与碰撞），半尺寸 (0.09, 0.035, 0.018) m；
- **collision origin 存在非零平移偏移** (0.026, 0, −0.02) m —— 足部 link 原点（踝关节）不与 box 中心重合；
- geom 无 `quat` 属性（局部旋转为单位元），但 link 以下整个 FK 链 + root 姿态会旋转 box；
- 地面：worldbody 唯一无限水平 plane `ground`（z=0）。

## 2. 原实现的问题（对照任务背景的定量复核）

用修复后检测器对三个已导出动作逐帧复核（仅加载 trajectory.csv，不重新生成）：

| 动作 | 旧方法报告值 | 修复后真实值 | 旧方法最大误差 |
|---|---:|---|---:|
| arm_wave (246 帧) | min = **+23.29 mm**（无穿透） | 左 min **+4.69 mm** / 右 **+4.76 mm**，全程悬空，0 穿透 | 18.61 mm（虚报间隙） |
| top_rock (286 帧) | min = **+2.86 mm**（A-pass） | 左 max 穿透 **58.03 mm** / 右 **58.39 mm**，286 帧中 254/257 帧有足-地物理碰撞对 | 78.68 mm（漏检穿透） |
| popping (243 帧) | min = **+57.98 mm** | 左 min +1.35 mm / 右 +2.58 mm，0 穿透 | 79.00 mm（虚报间隙） |

- Top Rock 的 **58.39 mm 最大穿透**与任务背景“最大穿透约 58.4 mm”及独立评审文档（docs/K1_ACTION_DATA_CONVERSION_V1_INDEPENDENT_REVIEW.md）一致；
- Arm Wave 的 **4.7 mm 间隙**（4.69/4.76 mm）与任务背景“约 4.7 mm 间隙”一致——该间隙真实存在，不是检测误差。

## 3. 修复方案

新增 `virtual_k1/virtual_k1/foot_geometry.py`（核心）+ `K1Sim.get_foot_geometry()` 接口，`kinematic_replay` 改用该接口：

1. **角点级几何计算**：对每只足的每个碰撞 geom（`contype|conaffinity > 0`），box 生成局部 8 角点，经完整变换链到世界系后对地面平面取有符号距离；clearance = 最低角点距离，penetration = max(0, −clearance)。
2. **左右脚独立结果**：`{left, right}` 各自输出 state / clearance / penetration / 最高角点 / 角点坐标。
3. **状态三分类**（容差参数化，默认 `contact_tolerance_m = 0.002`、`numeric_tolerance_m = 1e-6`，可由 `config/k1_sim.yaml contact.foot_geometry` 覆盖）：
   - `PENETRATION`：penetration > 接触容差；
   - `CONTACT`：|clearance| ≤ 接触容差（几何接近，**不等于物理接触力**）；
   - `AIRBORNE`：clearance > 接触容差。
4. **几何与物理证据分离**：物理接触仍由既有 `K1Sim.get_foot_contacts()`（MuJoCo contact manifold + 法向力）独立读取，两套接口互不冒充。
5. **不支持类型显式失败**：足部碰撞 geom 若为 capsule/cylinder/mesh 等未实现精确边界计算的类型，构造时抛 `UnsupportedFootGeomError`（明确列出类型），**绝不静默退化为中心高度检测**。当前支持 box（精确 8 角点）与 sphere（精确球面），K1 官方模型为单 box。
6. **地面模型**：`GroundPlane`（单位法向量 n、有符号距离 n·x + d，地面上方为正）。Virtual K1 当前唯一地面形态是水平 plane z=0，默认实例化为水平地面，并在模块文档与接口中明示；倾斜地面属尚不存在场景，未做超前抽象。

## 4. 坐标变换与几何计算方法

显式变换链（每层只应用一次）：

```
corner_world = foot_xpos + foot_xmat @ ( geom_pos + R(geom_quat) @ corner_local )
               └── world ← link (FK, 含 root 姿态) ──┘  └ collision origin ┘
```

- `corner_local ∈ {±hx, ±hy, ±hz}⁸`（box 半尺寸，MuJoCo `geom_size` 即半尺寸）；
- `geom_pos / geom_quat` 是 collision frame 相对足 link frame 的平移/旋转（collision origin 偏移）；
- `data.xpos / data.xmat`（足 link）已由 MuJoCo FK 合成 root free-joint 位姿与全链关节角，即 Robot/root frame → World frame；
- `verify_transform_chain()` 提供交叉校验：显式链结果 vs MuJoCo `geom_xpos/geom_xmat` 合成结果，实测最大角点位置差 **2.8e-17 m**（机器精度），证明各层变换无遗漏、无重复。

水平地面判定：`clearance = min(corner_world_z) − ground_z`；`penetration_depth = max(0, −clearance)`。最低角点用于越面判定；完整接触判据按 §3.3 的容差带分类，并保留全部角点（`include_corners`）供上层分析。

## 5. 修改文件列表

| 文件 | 状态 | 内容 |
|---|---|---|
| `virtual_k1/virtual_k1/foot_geometry.py` | 新增 | GroundPlane / FootGeometryDetector / 结果结构与分类 |
| `virtual_k1/virtual_k1/simulator.py` | 修改 | K1Sim 构造时初始化检测器；新增 `get_foot_geometry()` |
| `virtual_k1/virtual_k1/__init__.py` | 修改 | 导出新符号 |
| `virtual_k1/config/k1_sim.yaml` | 修改 | 新增 `contact.foot_geometry` 容差配置（带注释） |
| `virtual_k1/tests/test_foot_geometry.py` | 新增 | 25 项自动化测试（§6） |
| `k1_action/virtual_k1_replay.py` | 修改 | `kinematic_replay` 改用角点级检测并输出左右脚逐项统计；`physical_replay` 增加可选 `trace_path` 逐帧记录（默认关闭，物理行为零改动） |
| `k1_action/scripts/foot_geometry_audit.py` | 新增 | 动作复核 CLI（新旧对照 + 物理证据） |
| `virtual_k1/outputs/foot_geometry_fix01/*` | 新增 | 三动作复核 JSON + 逐帧 CSV + Arm Wave 物理回放 trace |

未改动：K1 模型 XML、22-DOF 映射、Motion Studio→SMPL→K1 主链路、动作数据（三个 trajectory.csv 的 sha256 与 manifest 全部 MATCH，已复核）、物理参数/质量/摩擦/PD 增益。旧行为保留证据：旧公式以 `_old_method_value()` 形式仅存在于复核脚本中用于对照。

## 6. 测试结果

**总计 77 通过 / 0 失败**：
- 新增 `virtual_k1/tests/test_foot_geometry.py`：**25 通过**；
- 既有 `virtual_k1/tests/` 回归：**35 通过**（test_foot_contact / test_joint_control / test_joint_mapping / test_model_loading / test_model_physics / test_state_and_reset / test_sustained_run，未删除、未跳过、未弱化）；
- `k1_action/tests/` 回归：**17 通过**，其中 `test_virtual_k1_replay_kinematic` 直接运行修复后的 `kinematic_replay` 对 Arm Wave 全量重定向轨迹（246 帧）并保持 A-pass。

### 6.1 基础几何（理论值 vs 实际值，全部 |误差| ≤ 1e-9 m）

| 案例 | 理论 clearance | 实际 | 状态 |
|---|---:|---:|---|
| 水平放置底面恰好触地 | 0 | 0.000000000 | CONTACT |
| 悬空 5 mm | +0.005 | +0.005000000 | AIRBORNE |
| 下穿 5 mm | −0.005 | −0.005000000 | PENETRATION (深度 5.000 mm) |
| 悬空 50 mm | +0.050 | +0.050000000 | AIRBORNE |
| 下穿 50 mm | −0.050 | −0.050000000 | PENETRATION (深度 50.000 mm) |

### 6.2 旋转（理论值 vs 实际值）

| 案例 | 理论 | 实际 | 误差 |
|---|---:|---:|---:|
| 绕 X 25° | z_c −(hy·sin25°+hz·cos25°) = 0.018895 | 0.018895 | <1e-9 |
| 绕 Y 30° | z_c −(hx·sin30°+hz·cos30°) = −0.000588 | −0.000588 | <1e-9 |
| 绕 Z 73° | 与未旋转一致（偏航不改底面） | 一致 | <1e-9 |
| 多轴组合 (X20°+Y35°+Z−50°) | 测试侧独立 Rodrigues 组合逐角点计算 | min/max 角点一致 | <1e-9 |
| collision origin 平移 (0.026,0,−0.02) | z_body−0.02−hz | 一致（漏掉偏移会高 20 mm） | <1e-9 |
| collision origin 平移+body 旋转 20° | 独立组合计算 | 一致 | <1e-9 |
| collision origin 旋转 (geom quat 绕 Y 30°) | z−(hx·sin30°+hz·cos30°) | 一致；与 body 旋转叠加不重复应用 | <1e-9 |
| **旧方法盲区**：中心同高 0.030 m，平放 vs 绕 Y 25° | 平放 +12.0 mm / 斜放 **−24.35 mm 穿透** | 一致 | <1e-3 |
| （盲区量化） | 旧方法两姿态同值 0.030，漏检 54.35 mm | — | — |

另含：sphere 支持（0.005 悬空 / 2 mm 恰在容差带内判 CONTACT / 3 mm 判 PENETRATION）、capsule 显式抛 `UnsupportedFootGeomError`（不静默退化）、GroundPlane 单位法向量校验与符号约定。

### 6.3 机器人级（真实 K1 模型与关节变换）

- **reset 官方站立**：左 0.483 mm / 右 0.740 mm，均判 CONTACT、0 穿透（与配置注释“0.5 mm 设计间隙”一致）；
- **root 平移精确性**：root ±5 mm / +50 mm → 双足 clearance 精确变化同值（误差 <1e-9）；
- **root 偏航不变性**：45°/90°/−170° 偏航下 clearance 不变（<1e-9）；
- **root 俯仰/横滚/组合**：与独立角点计算一致；
- **左/右脚独立**：单脚构造检测器与双脚结果逐位一致（<1e-12）；左踝 pitch −0.35 rad 后左脚判 PENETRATION 而右脚 clearance 不变（<1e-9）；左脚角点质心 y>0、右脚 y<0；
- **多姿态一致性**：6 组姿态（reset、踝 pitch ±、踝 roll、膝+踝、髋+膝+踝+对侧髋 roll）下检测器与测试侧从 XML 常量独立变换的 8 角点逐位一致（<1e-9）；
- **变换链一次性**：`verify_transform_chain` 最大角点位置差 2.8e-17 m；
- **姿态理论**：纯 pitch 足姿态下 `bottom = center_z −(hx·|sinθ|+hz·|cosθ|)` 与检测器一致（<1e-9）。

### 6.4 回归

virtual_k1 35 项 + k1_action 17 项全部通过，无删除/跳过/弱化。未发现旧测试本身错误；`test_foot_contact.py:45` 的中心高−z 半尺寸写法仅在平放足姿态下是精确的（其断言场景恰为 reset 平放），本次保留不动，新的角点级检测已由 `test_foot_geometry.py` 覆盖。

## 7. Arm Wave 复核结果（原始数据逐帧复算）

来源：`datasets/k1_dance_v1/arm_wave/trajectory.csv`（sha256 与 manifest MATCH），246 帧 @ 50 Hz，4.92 s；逐帧 qpos + mj_forward；产物 `virtual_k1/outputs/foot_geometry_fix01/arm_wave_foot_frames.csv`（逐帧）与 `arm_wave_foot_audit.json`（汇总）。

| 指标 | 左脚 | 右脚 |
|---|---:|---:|
| 最小足底间隙 | **4.686 mm** | **4.763 mm** |
| 最大穿透深度 | **0.000 mm** | **0.000 mm** |
| 悬空帧数 (AIRBORNE) | 246 / 246 | 246 / 246 |
| 几何接近帧数 (\|c\|≤2 mm 容差带) | 0 | 0 |
| 运动学 mj_forward 物理 manifold 接触帧数 | 0 | 0 |
| 物理回放（PD+重力，未改任何物理参数）运动段接触记录 | 1571 / 1577 (99.62%) | 1576 / 1577 (99.94%) |
| 接触持续时间（运动学几何接触） | **0 s** | **0 s** |

（物理回放记录间隔 4 ms，运动段 1.50–6.82 s；左右脚平均法向力 96.1 / 96.6 N ≈ 体重/2，最大穿透 ≤0.01 mm，base_z_min = 0.5466 m 与既有 validation.json 记录完全一致——回放确定性复现。）

**结论**：
1. 任务背景中的“约 4.7 mm 间隙”属实（4.686/4.763 mm，源自 root z=0.555 高于官方站立 0.551 约 4.0 mm + 设计间隙 ~0.5–0.7 mm）。**运动学回放全程无足底几何接触**——这是真实结果，未通过放大容差伪装成接触。
2. 旧检测把该间隙虚报为 23.29 mm（中心高 vs 底面，误差 18.61 mm）。
3. 物理回放中双足确实产生动态接触（重力使 root 沉降贴地），但这是控制器+重力的结果，与运动学轨迹本身“无接触”是两件事，已分别记录、未混用。
4. 肩部速度峰值 15.3 rad/s 独立问题仅在此记录，未处理。

## 8. 剩余问题

1. **Arm Wave 运动学无足底接触**：若动作质量标准要求运动学轨迹即含足底接触，需在重定向层解决（root 高度或腿部 IK），不属于本任务，也未擅自修改。
2. **`penetration_ok` 政策阈值仍为 −0.02 m**（沿袭原策略未动）：含义是“最深穿透 20 mm 内仍判 ok”。修复后 Top Rock 实测 −58.4 mm 会正确 FAIL，但 20 mm 的容忍度本身值得在动作质量标准中重新审视（政策问题，非几何问题）。
3. Top Rock 58.4 mm 穿透与 Popping 6.667 ms 重采样缩短问题仍未解决（超出本任务范围，检测器已能为其提供可信度量）。
4. 足部若未来改用 capsule/cylinder/mesh 碰撞体，当前显式报错，需扩展对应精确边界计算。
5. `kinematic_replay` 的 `contact_match` 字段仍为占位（原状保留）。

## 9. 下一步建议

1. **Arm Wave 完整验证**：以本检测器输出为准，先决策“运动学是否要求足底接触”；若要求，调整重定向 root 高度策略（如按足底间隙闭环校正 root z，目标 clearance 进入 ±2 mm 容差带），再重跑 A/B 两类回放；肩部 15.3 rad/s 速度问题另行立项。
2. 用同一检测器重审 Top Rock（运动学穿透 58.4 mm 必须先消除，物理摔倒复核才有意义）与 Popping（1.4–2.6 mm 间隙接近容差带，可先做与 Arm Wave 相同的 root 高度决策）。
3. 将 `foot_geometry` 逐帧统计并入 `validate_k1_action` 输出，作为三个动作的常规质量指标。
4. 考虑收紧或参数化 §8.2 的 20 mm 穿透政策阈值。

## 10. 最终工程判定

**PASS** —— 几何检测实现正确：25 项新测试（基础/旋转/机器人级/盲区判别）全部通过与理论值吻合（≤1e-9 m 量级），52 项既有回归全部通过；三动作复核数值与独立评审文档交叉一致（4.7 mm / 58.4 mm）；变换链经机器精度交叉验证；不支持几何类型显式失败、几何与物理证据分离。

（该 PASS 仅针对足底几何检测修复任务本身；Arm Wave 仍存在运动学无足底接触与肩部速度两个独立未决问题，不得据此视为动作验收通过。）
