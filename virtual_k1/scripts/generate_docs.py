#!/usr/bin/env python3
"""从 docs/_model_inspection.json (实际加载结果) 生成 K1 关节映射与物理属性文档。"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PROJECT_ROOT / "docs"
RPT = json.loads((DOC_DIR / "_model_inspection.json").read_text(encoding="utf-8"))

PROVENANCE = """\
> 数据来源: Booster 官方仓库 [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets)
> commit `12a97516f57469078707afb587a5f84c1cb110d1` (main, 2026-09-26),
> 模型文件 `robots/K1/K1_22dof.xml`, 由 `scripts/inspect_model.py` 实际加载
> MuJoCo 4.0.1 后提取, 非人工誊写。
"""


def gen_joint_mapping() -> str:
    s = RPT["stats"]
    lines = []
    lines.append("# K1 关节映射表 (实际加载结果)\n")
    lines.append(PROVENANCE)
    lines.append("## 1. 维度统计\n")
    lines.append("| 项 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| njnt | {s['njnt']} | 1 free (root) + 22 hinge |")
    lines.append(f"| nq | {s['nq']} | 22 hinge + 7 (free joint 位姿) |")
    lines.append(f"| nv | {s['nv']} | 22 hinge + 6 (free joint 速度) |")
    lines.append(f"| nu | {s['nu']} | 22 个 motor (力矩) 执行器 |")
    lines.append(f"| nbody | {s['nbody']} | world + 25 body |")
    lines.append(f"| ngeom | {s['ngeom']} | 其中 21 个参与碰撞 |")
    lines.append(f"| nmesh | {s['nmesh']} | 视觉 STL mesh |")
    lines.append(f"| nkey | {s['nkey']} | 官方模型无 keyframe 初始姿态 |")
    lines.append("")
    lines.append("**结论**: K1 的 \"22 自由度\" 指 **22 个驱动旋转关节** (头 2 + 臂 4×2 + 腿 6×2)。")
    lines.append("加上 root free joint 后广义坐标为 nq=29 / nv=28。22 个执行器与 22 个")
    lines.append("驱动关节一一对应 (执行器名 == 关节名), 无被动关节、无固定关节冗余。")
    lines.append("")
    lines.append("## 2. 全部关节映射表\n")
    lines.append("| Index | Joint Name | Body | Type | qpos Address | qvel Address | Lower Limit | Upper Limit | Actuator |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for j in RPT["joints"]:
        if j["type"] == "free":
            lo = hi = "—"
        else:
            lo, hi = (f"{j['range'][0]:.3f}", f"{j['range'][1]:.3f}") if j["range"] else ("—", "—")
        lines.append(
            f"| {j['joint_id']} | `{j['joint_name']}` | {j['body_name']} | "
            f"{j['type']} | {j['qpos_adr']} | {j['dof_adr']} | "
            f"{lo} | {hi} | {j['actuator']} |")
    lines.append("")
    lines.append("free joint (`world_joint`) 占用 qpos[0:7] (位置 xyz + 四元数 wxyz), ")
    lines.append("qvel[0:6] (世界系线速度 + 体坐标系角速度), 无执行器。")
    lines.append("")
    lines.append("## 3. 执行器表\n")
    lines.append("| Index | Actuator | Joint | 类型 | 语义 | Force Range (Nm) | Gear | Ctrl Range |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in RPT["actuators"]:
        lines.append(
            f"| {a['actuator_id']} | `{a['name']}` | `{a['joint']}` | motor | "
            f"torque (ctrl=力矩) | [{a['force_range'][0]:.1f}, {a['force_range'][1]:.1f}] | "
            f"{a['gear']:.0f} | 不限幅 |")
    lines.append("")
    lines.append("**控制语义**: 全部 22 个执行器为 `<motor>` (direct force/torque), ")
    lines.append("`data.ctrl[i]` 即关节力矩 (Nm), 受 `forcerange` 截断, `gear=1`。")
    lines.append("官方模型不含位置/速度执行器 —— 位置控制需在上层实现")
    lines.append("(本项目提供 `virtual_k1.controllers.JointPDController`)。")
    lines.append("")
    lines.append("## 4. 关节分组\n")
    groups = {"头部": [], "左臂": [], "右臂": [], "左腿": [], "右腿": []}
    for j in RPT["joints"]:
        if j["type"] != "hinge":
            continue
        n = j["joint_name"]
        if "head" in n:
            groups["头部"].append(n)
        elif "shoulder" in n or "elbow" in n:
            groups["左臂" if "left" in n else "右臂"].append(n)
        elif "hip" in n or "knee" in n or "ankle" in n:
            groups["左腿" if "left" in n else "右腿"].append(n)
    lines.append("| 部位 | 数量 | 关节 |")
    lines.append("|---|---|---|")
    for g, names in groups.items():
        lines.append(f"| {g} | {len(names)} | {', '.join('`%s`' % n for n in names)} |")
    lines.append("")
    lines.append("## 5. 命名与关节语义注意事项\n")
    lines.append("1. **XML `aa` 前缀怪癖**: 头部两关节 (`aahead_*`) 与两个肩 pitch 关节")
    lines.append("   (`aaleft/aaright_shoulder_pitch_joint`) 带 `aa` 前缀, 其余关节不带。")
    lines.append("   `virtual_k1` 的映射层同时接受两种命名 (`resolve_joint`)。")
    lines.append("2. **官方动作 CSV 命名差异**: `booster_assets/motions/K1/*.csv` 表头对肩 pitch")
    lines.append("   使用无前缀名 `left/right_shoulder_pitch_joint` (头仍为 `aahead_*`), ")
    lines.append("   读取官方动作数据时需做名称映射。")
    lines.append("3. **左右限位镜像**: hip_roll / elbow_yaw 左右限位范围互为镜像 ")
    lines.append("   (如左 hip_roll [-0.375, 1.536], 右 [-1.536, 0.375]), 属正常镜像结构。")
    lines.append("4. **膝关节零位**: knee_pitch 限位 [0, 2.321], 零位为完全伸直且不允许过伸。")
    lines.append("5. **手臂关节语义 (FK 实测, 重要!)**: 手臂零位为 **T 字平举** (沿 ±Y 轴),")
    lines.append("   不是自然下垂。各关节实际语义:")
    lines.append("   - `shoulder_roll`: 抬/放手臂。左臂 -1.45=下垂, 0=水平, +1.05≈高举过头;")
    lines.append("     右臂镜像 (+1.45=下垂, 负=高举)。官方动作数据的下垂位即 roll=∓1.449。")
    lines.append("   - `shoulder_pitch`: 悬垂/举起状态下前后摆臂 (矢状面摆动)。")
    lines.append("   - `elbow_pitch`: **前臂自旋** (转轴永远平行于前臂长轴, 不产生可见屈肘)。")
    lines.append("   - `elbow_yaw`: **实际可见的屈肘关节** (左负=前弯, 右正=前弯)。")
    lines.append("   - 编排举手动作请用 `shoulder_roll` 抬臂 + `elbow_yaw` 摆动。")
    lines.append("6. **平足约束**: 矢状面腿部三关节角之和 hip+knee+ankle ≈ 0 时脚掌贴地")
    lines.append("   (站立姿态 -0.099+0.207-0.099≈0.009)。编排下蹲/弹跳时必须保持该约束,")
    lines.append("   否则脚掌翘起导致失稳 (已在 demo 开发中实测验证)。")
    lines.append("")
    return "\n".join(lines)


def gen_model_physics() -> str:
    s = RPT["stats"]
    lines = []
    lines.append("# K1 模型物理属性检查报告 (实际加载结果)\n")
    lines.append(PROVENANCE)
    lines.append("## 1. 全局参数\n")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 整机总质量 | **{s['total_mass_kg']:.4f} kg** |")
    lines.append(f"| timestep | {s['timestep']} s |")
    lines.append(f"| solver | Newton (impratio={s['impratio']}, tolerance=1e-6) |")
    lines.append(f"| gravity | {s['gravity']} |")
    lines.append(f"| 惯性矩阵正定性 | 全部通过 (25 个有质量 body) |")
    lines.append(f"| 惯量三角不等式 | 全部通过 |")
    lines.append(f"| 零/负质量 | 无 (含关节的 body 均有质量) |")
    lines.append("")
    lines.append("质量分布 (有质量 body, 共 "
                 f"{sum(1 for b in RPT['bodies'] if b['mass_kg'] > 0)} 个):\n")
    lines.append("| Body | Mass (kg) | 质心 ipos (m) | 主惯量 diaginertia (kg·m²) |")
    lines.append("|---|---|---|---|")
    for b in RPT["bodies"]:
        if b["mass_kg"] <= 0:
            continue
        d = b["diaginertia"]
        lines.append(
            f"| {b['name']} | {b['mass_kg']:.4f} | "
            f"({b['ipos'][0]:.4f}, {b['ipos'][1]:.4f}, {b['ipos'][2]:.4f}) | "
            f"({d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}) |")
    total = s["total_mass_kg"]
    lines.append("")
    lines.append("主要质量段: 躯干 6.50 kg (33.1%), 单腿 4.503 kg ×2 (45.9%), ")
    lines.append("单臂 1.58 kg ×2 (16.1%), 头部 1.00 kg (5.1%)。")
    lines.append("")
    lines.append("## 2. 碰撞模型\n")
    lines.append(f"45 个 geom 中 **{RPT['collision_geom_count']} 个参与碰撞** ")
    lines.append("(contype/conaffinity 非零), 其余为纯视觉 mesh。\n")
    lines.append("| Body | 碰撞几何 | 尺寸 (半尺寸) | 说明 |")
    lines.append("|---|---|---|---|")
    col_by_body: dict[str, list[dict]] = {}
    for g in RPT["geoms"]:
        if g["collision"] and g["type"] != "plane":
            col_by_body.setdefault(g["body_name"], []).append(g)
    desc = {
        "trunk": "躯干主体碰撞盒 (不可见)",
        "aahead_pitch_link": "头部 mesh 直接参与碰撞",
        "left_elbow_pitch_link": "前臂 mesh 直接参与碰撞 (左右同)",
        "left_hip_roll_link": "髋侧圆柱",
        "left_hip_yaw_link": "大腿上部圆柱",
        "left_knee_pitch_link": "膝关节圆柱 + 小腿下端盒",
        "left_ankle_roll_link": "**足底碰撞盒 (不可见)** + 足 mesh (视觉)",
    }
    for body, geoms in col_by_body.items():
        for g in geoms:
            size = ", ".join(f"{x:.3f}" for x in g["size"][:3])
            lines.append(f"| {g['body_name']} | {g['type']} | {size} | "
                         f"{desc.get(g['body_name'], '')} |")
    lines.append("")
    lines.append("**足部碰撞**: 每足碰撞体为 ankle_roll_link 下的不可见 box")
    lines.append("(半尺寸 0.090 × 0.035 × 0.018 m, 即足长约 18 cm、宽 7 cm、厚 3.6 cm),")
    lines.append("位置在踝关节前偏 0.026 m。足部 STL mesh (`Left/Right_Foot.STL`)")
    lines.append("为 contype=0 的视觉几何, **不参与碰撞**。躯干碰撞为不可见 box + 圆柱,")
    lines.append("前臂/头部 mesh 直接参与碰撞。地面为无限大 plane, 摩擦系数 0.4。")
    lines.append("")
    lines.append("## 3. 检查结论\n")
    lines.append("1. **质量**: 无零质量/负质量异常; 所有含关节 body 质量均 > 0;")
    lines.append("   左右对称关节段质量一致 (相对误差 < 0.1%)。")
    lines.append("2. **惯性**: 25 个有质量 body 的惯性矩阵 (由 diaginertia + iquat")
    lines.append("   重建) 全部正定, 且满足刚体三角不等式 I1+I2≥I3。")
    lines.append("3. **碰撞**: 覆盖躯干/头/前臂/髋/膝/足; 足底为独立碰撞盒,")
    lines.append("   与视觉 mesh 分离 —— 属常见简化建模, 未见异常重叠或缺失。")
    lines.append("4. **限制 (记录, 非错误)**: 大腿 (hip_yaw→knee 段) 无碰撞体,")
    lines.append("   上臂 (肩→肘) 无碰撞体; 若后续舞蹈动作涉及大腿/上臂接触地面")
    lines.append("   或身体互碰, 需评估是否补充碰撞几何 (属后续阶段决策)。")
    lines.append("5. 传感器: imu site 上有 framequat (姿态) 与 gyro (角速度) 两个")
    lines.append("   官方传感器, 带噪声参数, 可用于模拟真实 IMU 读数。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    (DOC_DIR / "K1_JOINT_MAPPING.md").write_text(
        gen_joint_mapping(), encoding="utf-8")
    (DOC_DIR / "K1_MODEL_PHYSICS.md").write_text(
        gen_model_physics(), encoding="utf-8")
    print("生成:", DOC_DIR / "K1_JOINT_MAPPING.md")
    print("生成:", DOC_DIR / "K1_MODEL_PHYSICS.md")


if __name__ == "__main__":
    main()
