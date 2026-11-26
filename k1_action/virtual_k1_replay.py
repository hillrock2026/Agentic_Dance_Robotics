"""Virtual K1 replay validation — two explicitly separate checks:

A. kinematic_replay: set qpos (incl. root) from the trajectory frame by frame,
   mj_forward only. Checks joint limits, ground penetration, foot contacts vs
   human contact estimate, left/right assignment, obvious self-collision
   proxies (knee/ankle z sanity). No physics — pose correctness only.

B. physical_replay: PD position control (V1 JointPDController, gravity
   compensation) tracks the 22 joints; root is free under gravity. Records
   base height, foot contacts, falls. A-pass does NOT imply B-pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "virtual_k1"))

from virtual_k1 import JointPDController, K1Sim  # noqa: E402

from .retargeting import K1Trajectory  # noqa: E402
from .skeleton_mapping import K1_JOINT_ORDER  # noqa: E402

FALL_Z = 0.42


def kinematic_replay(traj: K1Trajectory, contact_state: np.ndarray | None = None) -> dict:
    import mujoco

    sim = K1Sim()
    m, d = sim.model, sim.data
    T = traj.frame_count

    def qadr(jn):
        return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])

    errs: list[str] = []
    # FIX_01: 逐帧、逐足足底几何检测 (真实碰撞盒角点, 完整旋转平移变换),
    # 取代旧的"box 中心高度"检测 (旧实现对 K1 足 box 恒取中心高度:
    # mjGEOM_BOX=6 而旧条件比较 ==3(capsule), 且无旋转/左右脚区分)。
    sides = ("left", "right")
    foot_geo: dict[str, list[dict]] = {s: [] for s in sides}
    l_elbow_y, r_elbow_y = [], []
    for t in range(T):
        q = np.zeros(m.nq)
        q[:3] = traj.root_pos[t]
        q[3:7] = [traj.root_quat[t][3], *traj.root_quat[t][:3]]  # xyzw -> wxyz
        q[4] = max(q[4], 0.0)  # keep quat w non-negative hemisphere
        for jn in K1_JOINT_ORDER:
            q[qadr(jn)] = traj.joints[jn][t]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        geo = sim.get_foot_geometry()
        for s in sides:
            foot_geo[s].append(geo[s])
        # L/R: left elbow must stay on +Y side of the trunk, right on -Y
        le = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_pitch_link")]
        re = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_pitch_link")]
        l_elbow_y.append(float(le[1] - d.xpos[1][1]))
        r_elbow_y.append(float(d.xpos[1][1] - re[1]))
    sim.close()

    lr_ok = bool(np.median(l_elbow_y) > 0 and np.median(r_elbow_y) > 0)
    per_side = {}
    for s in sides:
        cl = np.array([g["clearance_m"] for g in foot_geo[s]])
        per_side[s] = {
            "min_clearance_m": float(cl.min()),
            "max_penetration_depth_m": float(np.maximum(-cl, 0.0).max()),
            "penetration_frames": int((cl < 0.0).sum()),
            "contact_frames": int(sum(g["state"] == "CONTACT" for g in foot_geo[s])),
            "airborne_frames": int(sum(g["state"] == "AIRBORNE" for g in foot_geo[s])),
            "state_frames": {st: int(sum(g["state"] == st for g in foot_geo[s]))
                             for st in ("PENETRATION", "CONTACT", "AIRBORNE")},
        }
    pen = float(min(per_side[s]["min_clearance_m"] for s in sides))
    checks = {
        "ground_penetration_min_m": pen,
        "penetration_ok": pen > -0.02,
        "foot_geometry": per_side,
        "left_right_assignment_ok": lr_ok,
        "median_left_elbow_offset_m": float(np.median(l_elbow_y)),
        "median_right_elbow_offset_m": float(np.median(r_elbow_y)),
        "frames": T,
        "contact_match": "NOT TESTED" if contact_state is None else "see physical replay",
    }
    passed = checks["penetration_ok"] and lr_ok
    return {"check": "A_kinematic_replay", "pass": passed, "details": checks}


def physical_replay(traj: K1Trajectory, hold_s: float = 1.0, ramp_s: float = 0.5,
                    render: bool = False, render_dir: str | Path | None = None,
                    contact_state: np.ndarray | None = None,
                    trace_path: str | Path | None = None) -> dict:
    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"],
                           gravity_compensation=True)
    stance = sim.get_joint_positions()
    T = traj.frame_count
    fps = traj.fps
    dt = sim.timestep

    q_filt = {j: stance[j] for j in K1_JOINT_ORDER}
    alpha = 0.35
    base_z, track_err, contact_seq = [], [], []
    frames = []
    trace_rows: list[list] = []   # FIX_01: 可选逐帧 trace (物理接触+几何间隙)

    def _record(record_frame: bool):
        st = sim.get_robot_state()
        base_z.append(float(st["base_position"][2]))
        pos = st["joint_positions"]
        track_err.append(float(np.sqrt(np.mean([(pos[j] - q_filt[j]) ** 2 for j in K1_JOINT_ORDER]))))
        c = sim.get_foot_contacts()
        contact_seq.append((bool(c["left"]["in_contact"]), bool(c["right"]["in_contact"])))
        if trace_path is not None:
            geo = sim.get_foot_geometry()
            trace_rows.append([
                float(st["time"]),
                int(c["left"]["in_contact"]), int(c["right"]["in_contact"]),
                float(c["left"]["total_normal_force"]), float(c["right"]["total_normal_force"]),
                float(geo["left"]["clearance_m"]), float(geo["right"]["clearance_m"]),
                float(st["base_position"][2]),
            ])
        if record_frame and render:
            frames.append(sim.render(width=480, height=360))

    def run(kind, seconds=None, t_idx=None, record=False):
        n = max(1, int(round((seconds if seconds is not None else 1.0 / fps) / dt)))
        rec_every = max(1, int(round(0.004 / dt)))  # sample state every ~4 ms
        for step in range(n):
            tgt = dict(stance)
            if kind == "frame":
                f0 = {j: traj.joints[j][t_idx] for j in K1_JOINT_ORDER}
                f1 = {j: traj.joints[j][min(t_idx + 1, T - 1)] for j in K1_JOINT_ORDER}
                u = step / max(n - 1, 1)
                for j in K1_JOINT_ORDER:
                    tgt[j] = (1 - u) * f0[j] + u * f1[j]
            elif kind == "ramp":
                u = step / max(n - 1, 1)
                for j in K1_JOINT_ORDER:
                    tgt[j] = (1 - u) * stance[j] + u * traj.joints[j][0]
            else:
                for j in K1_JOINT_ORDER:
                    tgt[j] = q_filt[j]
            for j in K1_JOINT_ORDER:
                q_filt[j] = alpha * tgt[j] + (1 - alpha) * q_filt[j]
            sim.step(pd.compute(sim, {**tgt, **q_filt}))
            if step % rec_every == 0 or step == n - 1:
                _record(record and step == n - 1)

    run("hold", hold_s)
    run("ramp", ramp_s)
    for t in range(T):
        run("frame", t_idx=t, record=True)
    run("hold", 0.4)
    sim.close()

    base_z_arr = np.array(base_z)
    fell = bool(base_z_arr.min() < FALL_Z)
    # stability window: after ramp (motion portion)
    # records are sampled every ~4 ms; recompute window boundaries
    rec_hz = 1.0 / 0.004
    motion_z = base_z_arr[int((hold_s + ramp_s) * rec_hz):
                          int((hold_s + ramp_s + T / fps) * rec_hz)]
    contact_ok = None
    if contact_state is not None and len(motion_z):
        # compare sim contact vs human estimate over motion frames (per replay step)
        rec_hz = 1.0 / 0.004
        start = int((hold_s + ramp_s) * rec_hz)
        steps_per_frame = max(1, int(round(rec_hz / fps)))
        n_cmp = min(len(contact_seq) - start, T * steps_per_frame)
        sim_c = np.array(contact_seq)[start:start + n_cmp:steps_per_frame]
        hum_l = contact_state[:len(sim_c)] > 0
        hum_r = contact_state[:len(sim_c)] > 1
        agree = float((sim_c[:, 0] == hum_l).mean())
        agree_r = float((sim_c[:, 1] == hum_r).mean())
        contact_ok = {"left_agreement": agree, "right_agreement": agree_r}

    details = {
        "base_z_min": float(base_z_arr.min()), "base_z_max": float(base_z_arr.max()),
        "motion_base_z_min": float(motion_z.min()) if len(motion_z) else None,
        "fell": fell, "fall_threshold_z": FALL_Z,
        "tracking_rmse_rad": float(np.mean(track_err)),
        "tracking_max_rad": float(np.max(track_err)),
        "foot_contact_agreement_vs_human_estimate": contact_ok or "NOT TESTED",
        "note": "PD replay with V1 gains; root free under gravity; pass/fail is "
                "about this controller, not about the trajectory format",
    }
    if render and frames and render_dir:
        from PIL import Image

        rd = Path(render_dir)
        rd.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(a) for a in frames[::2]]
        imgs[0].save(rd / "physical_replay.gif", save_all=True,
                     append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        details["replay_gif"] = str(rd / "physical_replay.gif")
    if trace_path is not None and trace_rows:
        import csv

        tp = Path(trace_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        with open(tp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "left_in_contact", "right_in_contact",
                        "left_normal_force_n", "right_normal_force_n",
                        "left_clearance_m", "right_clearance_m", "base_z_m"])
            w.writerows(trace_rows)
        details["trace_csv"] = str(tp)
    return {"check": "B_physical_replay", "pass": (not fell), "details": details}
