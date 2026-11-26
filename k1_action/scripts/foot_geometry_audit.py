#!/usr/bin/env python
"""CLI: 用修复后的足底几何检测器逐帧复核已导出的 K1 动作 (FIX_01)。

不重新生成动作、不修改重定向、不修改轨迹数据: 只加载
datasets/<action>/trajectory.csv, 逐帧运动学摆位 (qpos + mj_forward),
输出左右足逐帧间隙/穿透/状态, 并与旧实现 (box 中心高度) 对照。

可选 --physical: 追加一次 PD 物理回放 (不修改任何物理参数),
逐帧记录 MuJoCo contact manifold 的真实接触状态, 作为独立物理证据。

用法:
  python k1_action/scripts/foot_geometry_audit.py --action datasets/k1_dance_v1/arm_wave
  python k1_action/scripts/foot_geometry_audit.py --action ... --physical
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "virtual_k1"))

import mujoco  # noqa: E402

from k1_action.action_exporter import load_k1_action  # noqa: E402
from k1_action.skeleton_mapping import K1_JOINT_ORDER, xml_name_to_csv  # noqa: E402
from virtual_k1 import K1Sim  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "virtual_k1/outputs/foot_geometry_fix01"


def _old_method_value(model, data, foot_body_ids) -> float:
    """复现修复前 k1_action/virtual_k1_replay.py 的检测值 (保留作对照证据)。

    旧代码: min(geom_xpos[z] - geom_size[1] if geom_type==3 else geom_xpos[z]);
    mjGEOM_BOX=6 而条件比较 ==3 (capsule), 对 K1 足 box 恒走 else 分支
    => 实际返回碰撞盒中心高度 (min over 双足全部碰撞 geom)。
    """
    box_t = int(mujoco.mjtGeom.mjGEOM_CAPSULE)  # 旧代码的常量错误: 3 是 capsule
    vals = []
    for g in range(model.ngeom):
        if (int(model.geom_bodyid[g]) in foot_body_ids.values()
                and int(model.geom_contype[g]) > 0):
            if int(model.geom_type[g]) == box_t:
                vals.append(float(data.geom_xpos[g][2] - model.geom_size[g][1]))
            else:
                vals.append(float(data.geom_xpos[g][2]))
    return min(vals)


def _physics_foot_ground_contacts(sim: "K1Sim") -> dict[str, dict]:
    """mj_forward 后的 MuJoCo contact manifold (足-地面碰撞对), 独立物理证据。

    注意: 这是物理引擎的碰撞检测输出 (kinematic pose 下), 不是接触力;
    动态接触力证据用 --physical 的 PD 回放。
    """
    out = {"left": {"n_contacts": 0}, "right": {"n_contacts": 0}}
    for i in range(sim.data.ncon):
        con = sim.data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        side = sim._foot_geom_sides.get(g1) or sim._foot_geom_sides.get(g2)
        if side is None:
            continue
        other = g2 if side == sim._foot_geom_sides.get(g1) else g1
        if other in sim._ground_geom_ids:
            out[side]["n_contacts"] += 1
    return out


def audit(action_dir: str, out_dir: str | Path, physical: bool = False) -> dict:
    act = load_k1_action(action_dir)
    rows = act["rows"]
    T = len(rows)
    name = act["metadata"].get("action_name", Path(action_dir).name)

    sim = K1Sim()
    m, d = sim.model, sim.data
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qadr = {}
    for jn in K1_JOINT_ORDER:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr[jn] = int(m.jnt_qposadr[jid])

    frame_rows = []
    per_side = {"left": [], "right": []}
    old_vals = []
    for t, row in enumerate(rows):
        q = np.zeros(m.nq)
        q[:3] = [float(row["root_pos_x"]), float(row["root_pos_y"]), float(row["root_pos_z"])]
        qwxyz = [float(row["root_quat_w"]), float(row["root_quat_x"]),
                 float(row["root_quat_y"]), float(row["root_quat_z"])]
        q[3:7] = max(qwxyz[0], 0.0), qwxyz[1], qwxyz[2], qwxyz[3]
        for jn in K1_JOINT_ORDER:
            q[qadr[jn]] = float(row[xml_name_to_csv(jn)])
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        geo = sim.get_foot_geometry()
        phys = _physics_foot_ground_contacts(sim)
        old_vals.append(_old_method_value(m, d, sim._foot_body_ids))
        rec = {"frame": t, "timestamp_s": float(row["timestamp_s"]),
               "old_method_min_m": old_vals[-1]}
        for side in ("left", "right"):
            g = geo[side]
            per_side[side].append(g)
            rec[f"{side}_clearance_m"] = g["clearance_m"]
            rec[f"{side}_penetration_m"] = g["penetration_depth_m"]
            rec[f"{side}_state"] = g["state"]
            rec[f"{side}_physics_ncon"] = phys[side]["n_contacts"]
        frame_rows.append(rec)
    sim.close()

    # ---- per-frame CSV ----
    csv_path = out / f"{name}_foot_frames.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        w.writeheader()
        w.writerows(frame_rows)

    # ---- summary ----
    summary: dict = {"action": name, "frames": T, "source_csv": act["path"],
                     "per_foot": {}, "old_method": {}}
    for side in ("left", "right"):
        cl = np.array([g["clearance_m"] for g in per_side[side]])
        states = [g["state"] for g in per_side[side]]
        nc = np.array([r[f"{side}_physics_ncon"] for r in frame_rows])
        summary["per_foot"][side] = {
            "min_clearance_m": float(cl.min()),
            "max_clearance_m": float(cl.max()),
            "max_penetration_depth_m": float(np.maximum(-cl, 0.0).max()),
            "penetration_frames": int((cl < 0.0).sum()),
            "contact_band_frames": int(states.count("CONTACT")),
            "airborne_frames": int(states.count("AIRBORNE")),
            "airborne_duration_s": float(states.count("AIRBORNE") / 50.0),
            "physics_manifold_contact_frames_kinematic": int((nc > 0).sum()),
        }
    old = np.array(old_vals)
    new_min = np.minimum(
        np.array([r["left_clearance_m"] for r in frame_rows]),
        np.array([r["right_clearance_m"] for r in frame_rows]))
    summary["old_method"] = {
        "definition": "min(geom center z) over both feet (bug: mjGEOM_BOX=6 vs "
                      "compared ==3, no rotation, no left/right split)",
        "min_m": float(old.min()),
        "max_m": float(old.max()),
        "error_vs_new_at_min_m": float(old.min() - new_min.min()),
        "max_abs_error_m": float(np.max(np.abs(old - new_min))),
    }
    summary["frames_csv"] = str(csv_path)

    # ---- optional physical replay for true dynamic contact evidence ----
    if physical:
        from k1_action.retargeting import K1Trajectory
        from k1_action.virtual_k1_replay import physical_replay

        joints = {jn: np.array([float(r[xml_name_to_csv(jn)]) for r in rows])
                  for jn in K1_JOINT_ORDER}
        traj = K1Trajectory(
            fps=50.0,
            timestamps=np.array([float(r["timestamp_s"]) for r in rows]),
            joints_raw=joints, joints=joints,
            root_pos=np.array([[float(r["root_pos_x"]), float(r["root_pos_y"]),
                                float(r["root_pos_z"])] for r in rows]),
            root_quat=np.array([[float(r["root_quat_x"]), float(r["root_quat_y"]),
                                 float(r["root_quat_z"]), float(r["root_quat_w"])]
                                for r in rows]),
        )
        trace_csv = out / f"{name}_physical_trace.csv"
        res = physical_replay(traj, trace_path=trace_csv)
        tr = list(csv.DictReader(open(trace_csv)))
        # motion window only (after hold+ramp): hold 1.0 + ramp 0.5
        motion = [r for r in tr if float(r["time_s"]) >= 1.5]
        summary["physical_replay"] = {
            "pass": res["pass"], "details": res["details"],
            "trace_csv": str(trace_csv),
            "left_contact_records_motion": sum(int(r["left_in_contact"]) for r in motion),
            "right_contact_records_motion": sum(int(r["right_in_contact"]) for r in motion),
            "motion_records": len(motion),
            "record_interval_s": 0.004,
        }

    json_path = out / f"{name}_foot_audit.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_json"] = str(json_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, help="exported action dir or trajectory.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--physical", action="store_true",
                    help="追加 PD 物理回放, 记录真实动态接触 (不修改物理参数)")
    args = ap.parse_args()

    s = audit(args.action, args.out, physical=args.physical)
    print(json.dumps(s, indent=2, default=str))
    pf = s["per_foot"]
    print("\n==== 汇总 (新旧对照) ====")
    print(f"frames={s['frames']}")
    for side in ("left", "right"):
        p = pf[side]
        print(f"{side:5s}: min_clearance={p['min_clearance_m']*1000:8.3f} mm  "
              f"max_pen={p['max_penetration_depth_m']*1000:7.3f} mm  "
              f"airborne={p['airborne_frames']:4d}  contact_band={p['contact_band_frames']:4d}  "
              f"physics_ncon_frames={p['physics_manifold_contact_frames_kinematic']}")
    om = s["old_method"]
    print(f"old   : min={om['min_m']*1000:8.3f} mm  max|err|={om['max_abs_error_m']*1000:7.3f} mm")


if __name__ == "__main__":
    main()
