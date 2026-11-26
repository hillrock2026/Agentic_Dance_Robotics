#!/usr/bin/env python3
"""K1 模型全面检查脚本。

从实际加载的 MuJoCo 模型提取:
  1. 模型规模统计 (nq/nv/nu/njnt/nbody/ngeom/...)
  2. 全部关节表 (含 root free joint)
  3. 执行器表 (类型/控制语义/参数)
  4. Body 质量/惯性表 + 惯性正定性检查
  5. 碰撞几何表
  6. 传感器表

输出 JSON 到 docs/_model_inspection.json 供文档生成, 并打印可读摘要。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1.joint_mapping import JointMapping  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"
OUT_JSON = PROJECT_ROOT / "docs/_model_inspection.json"


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mapping = JointMapping(m)

    report: dict = {"model_file": str(MODEL_PATH), "mujoco_version": mujoco.__version__}

    # ---------------------------------------------------------------- stats
    report["stats"] = {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "njnt": int(m.njnt), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh), "nsensor": int(m.nsensor), "nkey": int(m.nkey),
        "timestep": float(m.opt.timestep),
        "solver": int(m.opt.solver),  # 0 PGS, 1 CG, 2 Newton
        "impratio": float(m.opt.impratio),
        "gravity": m.opt.gravity.tolist(),
        "total_mass_kg": float(m.body_mass.sum()),
    }

    # --------------------------------------------------------------- joints
    joints = []
    for j in mapping.joints:
        act = mapping.actuator_of_joint(j.name) if j.name in mapping.actuated_joints else None
        joints.append({
            "joint_id": j.joint_id, "joint_name": j.name, "type": j.type,
            "body_id": j.body_id, "body_name": j.body_name,
            "qpos_adr": j.qpos_adr, "dof_adr": j.dof_adr, "nq": j.nq, "nv": j.nv,
            "limited": j.limited,
            "range": list(j.range) if j.limited else None,
            "axis": list(j.axis), "armature": j.armature,
            "actuator": act.name if act else "-",
        })
    report["joints"] = joints

    # ------------------------------------------------------------ actuators
    acts = []
    for a in mapping.actuators:
        acts.append({
            "actuator_id": a.actuator_id, "name": a.name, "joint": a.joint_name,
            "semantics": a.control_semantics, "gear": a.gear,
            "ctrl_limited": a.ctrl_limited, "ctrl_range": list(a.ctrl_range),
            "force_limited": a.force_limited, "force_range": list(a.force_range),
            "gainprm": a.gainprm, "biasprm": a.biasprm, "note": a.bias_note,
        })
    report["actuators"] = acts

    # ------------------------------------------------------- bodies/inertia
    bodies = []
    inertia_issues = []
    for bid in range(m.nbody):
        mass = float(m.body_mass[bid])
        ipos = m.body_ipos[bid].tolist()
        iquat = m.body_iquat[bid].tolist()
        diag = m.body_inertia[bid].tolist()
        # 惯性矩阵 (body 局部惯性主轴坐标系 → body 坐标系)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[bid])
        R = R.reshape(3, 3)
        I_diag = np.diag(diag)
        I_full = R @ I_diag @ R.T
        eig = np.linalg.eigvalsh(I_full)
        pd = bool(np.all(eig > 0)) if mass > 0 else None
        if mass < 0:
            inertia_issues.append(f"body {bid} 负质量 {mass}")
        if mass > 0 and not pd:
            inertia_issues.append(f"body {bid} 惯性矩阵非正定, 特征值 {eig.tolist()}")
        if mass > 0 and diag[2] + diag[1] < diag[0] * 0.999:  # 三角不等式
            inertia_issues.append(
                f"body {bid} 惯量违反三角不等式: {diag}")
        parent = int(m.body_parentid[bid])
        bodies.append({
            "body_id": bid,
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid),
            "parent": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, parent),
            "mass_kg": mass, "ipos": ipos, "iquat_wxyz": iquat,
            "diaginertia": diag,
            "inertia_eigvals": eig.tolist(),
            "inertia_positive_definite": pd,
            "nchild_geoms": int(np.sum(m.geom_bodyid == bid)),
        })
    report["bodies"] = bodies
    report["inertia_issues"] = inertia_issues

    # ---------------------------------------------------------------- geoms
    geoms = []
    for gid in range(m.ngeom):
        body = int(m.geom_bodyid[gid])
        col = (int(m.geom_contype[gid]) > 0) or (int(m.geom_conaffinity[gid]) > 0)
        t = int(m.geom_type[gid])
        tname = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
                 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}.get(t, str(t))
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        mesh_name = None
        if tname == "mesh":
            mid = int(m.geom_dataid[gid])
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        geoms.append({
            "geom_id": gid, "name": gname or "",
            "body_id": body,
            "body_name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
            "type": tname, "size": m.geom_size[gid].tolist(),
            "collision": col, "mesh": mesh_name,
            "friction": m.geom_friction[gid].tolist(),
            "condim": int(m.geom_condim[gid]),
            "rgba": m.geom_rgba[gid].tolist(),
        })
    report["geoms"] = geoms
    report["collision_geom_count"] = int(sum(1 for g in geoms if g["collision"]))

    # -------------------------------------------------------------- sensors
    sensors = []
    for sid in range(m.nsensor):
        sensors.append({
            "sensor_id": sid,
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, sid),
            "type": int(m.sensor_type[sid]),
            "objtype": int(m.sensor_objtype[sid]),
            "objname": mujoco.mj_id2name(m, m.sensor_objtype[sid],
                                         int(m.sensor_objid[sid])),
            "dim": int(m.sensor_dim[sid]),
            "adr": int(m.sensor_adr[sid]),
        })
    report["sensors"] = sensors

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ------------------------------------------------------------- summary
    s = report["stats"]
    print("=" * 72)
    print("K1 模型检查摘要 (实际加载结果)")
    print("=" * 72)
    print(f"MuJoCo {mujoco.__version__} | model: {MODEL_PATH.name}")
    print(f"nq={s['nq']} nv={s['nv']} nu={s['nu']} njnt={s['njnt']} "
          f"nbody={s['nbody']} ngeom={s['ngeom']} nmesh={s['nmesh']} "
          f"nsensor={s['nsensor']} nkey={s['nkey']}")
    print(f"timestep={s['timestep']} solver={'Newton' if s['solver']==2 else s['solver']} "
          f"impratio={s['impratio']} gravity={s['gravity']}")
    print(f"总质量: {s['total_mass_kg']:.4f} kg")
    print(f"碰撞 geom 数: {report['collision_geom_count']} / {s['ngeom']}")
    print(f"惯性问题: {inertia_issues if inertia_issues else '无'}")

    print("\n--- 关节 (全部 %d 个) ---" % len(joints))
    for j in joints:
        rng = f"[{j['range'][0]:.3f}, {j['range'][1]:.3f}]" if j["range"] else "-"
        print(f"  jid{j['joint_id']:2d} {j['joint_name']:32s} {j['type']:5s} "
              f"{j['body_name']:26s} qpos@{j['qpos_adr']:2d} dof@{j['dof_adr']:2d} "
              f"range={rng:22s} act={j['actuator']}")

    print("\n--- 执行器 (全部 %d 个) ---" % len(acts))
    for a in acts:
        print(f"  u{a['actuator_id']:2d} {a['name']:32s} joint={a['joint']:32s} "
              f"{a['semantics']:8s} force=[{a['force_range'][0]:.1f},{a['force_range'][1]:.1f}] "
              f"gear={a['gear']} ctrl_limited={a['ctrl_limited']}")

    print("\n--- Body 质量 ---")
    for b in bodies:
        if b["mass_kg"] > 0:
            print(f"  b{b['body_id']:2d} {str(b['name']):26s} mass={b['mass_kg']:8.4f} kg "
                  f"diagI=[{b['diaginertia'][0]:.6f},{b['diaginertia'][1]:.6f},{b['diaginertia'][2]:.6f}] "
                  f"PD={'Y' if b['inertia_positive_definite'] else 'N'}")

    print(f"\nJSON 已写入: {OUT_JSON}")


if __name__ == "__main__":
    main()
