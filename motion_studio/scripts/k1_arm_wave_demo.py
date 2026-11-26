#!/usr/bin/env python
"""K1 Arm Wave demo — retarget recovered human arm motion to the Booster K1
upper body in MuJoCo (Virtual K1 V1 environment, used read-only).

Pipeline:
  1. Per motion frame, take SMPL upper-arm / forearm unit direction vectors
     (shoulder->elbow, elbow->wrist) from clean_motion.npz.
  2. Map directions from the human torso frame to the K1 torso frame
     (frames auto-calibrated from shoulder positions + gravity up).
  3. Damped least-squares IK on 4 arm DOF per side with DIRECTION-only
     residuals (K1's shoulder chain has offset anchors, so link lengths are
     pose-dependent — fixed-radius position targets are over-constrained and
     leave a large error floor; direction targets converge cleanly).
  4. Temporal smoothing of the IK joint trajectories (moving average) —
     per-frame IK can jump between solution branches and excite the PD.
  5. Track with the V1 PD controller (gravity compensation); targets are
     linearly interpolated inside each 40 ms frame (first-order hold) to
     avoid zero-order-hold staircase jitter. Legs/head hold the stance.

Outputs (out_dir): k1_arm_wave.gif, k1_vs_human.gif, k1_arm_wave_sheet.jpg,
k1_arm_wave_metrics.json (stability + direction-tracking diagnostics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "virtual_k1"))
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
from virtual_k1 import JointPDController, K1Sim  # noqa: E402

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 16, 17, 18, 19, 20, 21
SIDES = {"left": (L_SHO, L_ELB, L_WRI), "right": (R_SHO, R_ELB, R_WRI)}

K1_ARM_JOINTS = {
    "left": ["aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_elbow_pitch_joint", "left_elbow_yaw_joint"],
    "right": ["aaright_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_elbow_pitch_joint", "right_elbow_yaw_joint"],
}
K1_ELBOW_BODY = {"left": "left_elbow_pitch_link", "right": "right_elbow_pitch_link"}
K1_FOREARM_BODY = {"left": "left_elbow_yaw_link", "right": "right_elbow_yaw_link"}
ALL_ARM_JOINTS = K1_ARM_JOINTS["left"] + K1_ARM_JOINTS["right"]


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_qposadr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def joint_anchor(model, data, name):
    """Hinge rotation center (anatomical joint center), pose-dependent."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return data.xanchor[jid].copy()


def forearm_local_end(model, data, link_name):
    """Distal end of the forearm link in its local frame (lowest world-z mesh
    vertex at the calibration pose)."""
    bid = body_id(model, link_name)
    lowest = (np.inf, None)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[va:va + vn].astype(np.float64) * model.mesh_scale[mid]
        gp, gq = model.geom_pos[g].astype(np.float64), model.geom_quat[g].astype(np.float64)
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, gq)
        vb = (verts @ Rg.reshape(3, 3).T) + gp
        vw = (vb @ data.xmat[bid].reshape(3, 3).T) + data.xpos[bid]
        k = int(np.argmin(vw[:, 2]))
        if vw[k, 2] < lowest[0]:
            lowest = (float(vw[k, 2]), vb[k].copy())
    if lowest[1] is None:
        raise RuntimeError(f"no mesh geom on {link_name}")
    return lowest[1]


def torso_frames(model, data, jp):
    """R_map: human torso frame -> K1 torso frame (columns front/left/up)."""
    def frame(left_vec, up_vec):
        left = left_vec / np.linalg.norm(left_vec)
        front = np.cross(left, up_vec)
        front /= np.linalg.norm(front)
        left2 = np.cross(up_vec, front)
        return np.column_stack([front, left2, up_vec])

    left_k1 = data.xpos[body_id(model, "aaleft_shoulder_pitch_link")] - \
        data.xpos[body_id(model, "aaright_shoulder_pitch_link")]
    F_k1 = frame(left_k1, np.array([0.0, 0.0, 1.0]))
    left_h = jp[0, L_SHO] - jp[0, R_SHO]
    F_h = frame(left_h, np.array([0.0, 1.0, 0.0]))
    return F_k1 @ F_h.T


def calibrate(sim: K1Sim):
    """Arm geometry at the zero pose (arms hanging straight)."""
    d0 = mujoco.MjData(sim.model)
    mujoco.mj_forward(sim.model, d0)
    arm = {}
    for side in K1_ARM_JOINTS:
        sh = joint_anchor(sim.model, d0, K1_ARM_JOINTS[side][1])
        el = joint_anchor(sim.model, d0, K1_ARM_JOINTS[side][2])
        le = forearm_local_end(sim.model, d0, K1_FOREARM_BODY[side])
        fb = body_id(sim.model, K1_FOREARM_BODY[side])
        end = d0.xpos[fb] + d0.xmat[fb].reshape(3, 3) @ le
        arm[side] = {"local_end": le,
                     "L_upper": float(np.linalg.norm(el - sh)),
                     "L_fore": float(np.linalg.norm(end - el))}
    return arm


class ArmIK:
    def __init__(self, sim: K1Sim, R_map, arm_geoms):
        self.model = sim.model
        self.R_map = R_map
        self.arm = arm_geoms
        self.data_ik = mujoco.MjData(sim.model)
        self.addr = {s: [joint_qposadr(sim.model, j) for j in js]
                     for s, js in K1_ARM_JOINTS.items()}
        self.lim = {s: np.array([sim.model.jnt_range[
                         mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                         for j in js]) for s, js in K1_ARM_JOINTS.items()}

    def _arm_points(self, side, d=None):
        """Anatomical points on the K1 arm (evaluated on `d`, default: the
        solver's internal data — pass an explicit MjData for evaluation!):
        sh  = shoulder ROLL anchor, el = elbow PITCH anchor,
        end = distal forearm point (local_end in elbow_yaw_link)
        """
        m = self.model
        d = self.data_ik if d is None else d
        sh = joint_anchor(m, d, K1_ARM_JOINTS[side][1])   # roll anchor
        el = joint_anchor(m, d, K1_ARM_JOINTS[side][2])   # elbow_pitch anchor
        # (the pitch anchor moves only with the shoulder joints -> the
        # upper-arm reference decouples cleanly from the elbow DOF)
        fb = body_id(m, K1_FOREARM_BODY[side])
        end = d.xpos[fb] + d.xmat[fb].reshape(3, 3) @ self.arm[side]["local_end"]
        return sh, el, end, fb

    def solve(self, q_base, du, df, iters=60, refine=False):
        """Two-stage decoupled direction IK on the anatomical segments:

        stage 1: shoulder (pitch, roll) aligns UPPER-ARM direction
                 (shoulder roll anchor -> elbow yaw anchor, the physical
                 upper-arm segment of K1's compact arm)
        stage 2: elbow (pitch, yaw) aligns FOREARM direction
                 (elbow yaw anchor -> hand end)

        K1's forearm has effectively one positioning DOF plus a twist, so
        forearm matching is best-effort; upper-arm matching is exact on
        reachable frames.
        """
        m = self.model
        q = np.zeros(m.nq)
        for jname, val in q_base.items():
            qadr, _ = joint_qposadr(m, jname)
            q[qadr] = val
        out = {}
        margin = 0.03
        for side in K1_ARM_JOINTS:
            span = self.lim[side][:, 1] - self.lim[side][:, 0]
            lo = self.lim[side][:, 0] + margin * span
            hi = self.lim[side][:, 1] - margin * span
            tu = self.R_map @ du[side]
            tf = self.R_map @ df[side]
            d = self.data_ik
            dof_idx = [da for _, da in self.addr[side]]
            # ---- stage 1: shoulder -> upper-arm direction ----
            for it in range(iters):
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                sh, el, end, fb = self._arm_points(side)
                v1 = el - sh
                r1 = v1 - np.linalg.norm(v1) * tu
                if np.linalg.norm(r1) < 1e-4:
                    break
                Jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, Jp, None, el, body_id(m, K1_ELBOW_BODY[side]))
                J = Jp[:, dof_idx[:2]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), -r1)
                step = 0.6 if it < 30 else 0.25
                for k in (0, 1):
                    qadr = self.addr[side][k][0]
                    q[qadr] = np.clip(q[qadr] + step * dq[k], lo[k], hi[k])
            # ---- stage 2: elbow -> forearm direction (best effort) ----
            for it in range(iters // 2):
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                sh, el, end, fb = self._arm_points(side)
                v2 = end - el
                r2 = v2 - np.linalg.norm(v2) * tf
                if np.linalg.norm(r2) < 1e-4:
                    break
                Jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, Jp, None, end, fb)
                J = Jp[:, dof_idx[2:]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), -r2)
                step = 0.4
                for k in (2, 3):
                    qadr = self.addr[side][k][0]
                    q[qadr] = np.clip(q[qadr] + step * dq[k - 2], lo[k], hi[k])
            out.update({jn: q[qadr] for (qadr, _), jn in zip(self.addr[side], K1_ARM_JOINTS[side])})
        return out


def solve_targets(sim, ik, jp, stance):
    """IK over all frames + moving-average smoothing. Returns (targets, diag)."""
    T = len(jp)
    raw = []
    cur = dict(stance)
    for t in range(T):
        du, df = {}, {}
        for side, (S, E, W) in SIDES.items():
            u = jp[t, E] - jp[t, S]
            f = jp[t, W] - jp[t, E]
            du[side] = u / np.linalg.norm(u)
            df[side] = f / np.linalg.norm(f)
        cur = ik.solve(cur, du, df)
        raw.append(dict(cur))

    k = 5
    pad = k // 2
    targets = [dict(r) for r in raw]
    for j in ALL_ARM_JOINTS:
        arr = np.array([r[j] for r in raw])
        smoothed = np.convolve(np.pad(arr, pad, mode="edge"), np.ones(k) / k, mode="valid")
        for i in range(T):
            targets[i][j] = float(smoothed[i])

    jump_raw = np.mean([abs(raw[i + 1][j] - raw[i][j])
                        for i in range(T - 1) for j in ALL_ARM_JOINTS])
    jump_sm = np.mean([abs(targets[i + 1][j] - targets[i][j])
                       for i in range(T - 1) for j in ALL_ARM_JOINTS])
    diag = {"target_jump_raw_rad_per_frame": float(jump_raw),
            "target_jump_smoothed_rad_per_frame": float(jump_sm)}
    return targets, diag


def direction_errors(sim, ik, targets, jp, R_map):
    """Achieved upper-arm / forearm direction match under the targets."""
    m = sim.model
    d = mujoco.MjData(m)
    q = np.zeros(m.nq)
    stance = sim.get_joint_positions()
    for jn, v in stance.items():
        qadr, _ = joint_qposadr(m, jn)
        q[qadr] = v
    errs_u, errs_f = [], []
    for t, tgt in enumerate(targets):
        for jn, v in tgt.items():
            qadr, _ = joint_qposadr(m, jn)
            q[qadr] = v
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for side, (S, E, W) in SIDES.items():
            sh, el, end, _ = ik._arm_points(side, d)
            gu = (el - sh) / np.linalg.norm(el - sh)
            gf = (end - el) / np.linalg.norm(end - el)
            u = jp[t, E] - jp[t, S]
            f = jp[t, W] - jp[t, E]
            tu = R_map @ (u / np.linalg.norm(u))
            tf = R_map @ (f / np.linalg.norm(f))
            errs_u.append(np.degrees(np.arccos(np.clip(gu @ tu, -1, 1))))
            errs_f.append(np.degrees(np.arccos(np.clip(gf @ tf, -1, 1))))
    return {"upper_deg_median": float(np.median(errs_u)),
            "upper_deg_mean": float(np.mean(errs_u)),
            "fore_deg_median": float(np.median(errs_f)),
            "fore_deg_mean": float(np.mean(errs_f)),
            "upper_within10_pct": float(np.mean(np.array(errs_u) < 10))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(
        PROJECT_ROOT / "motion_studio/outputs/motions/segment_071_343ee79d/edited_motion.npz"),
        help="default: the 0.5x-amplitude edited export (K1-arm-workspace adapted)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--hold_s", type=float, default=1.0)
    ap.add_argument("--ramp_s", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    jp = z["joint_positions"].astype(np.float64)
    fps = float(z["motion_fps"])
    T = int(z["frame_count"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.npz).parent / "k1_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"],
                           gravity_compensation=True)
    stance = sim.get_joint_positions()
    arm = calibrate(sim)
    for s, g in arm.items():
        print(f"[geom] {s}: upper {g['L_upper']:.3f} m, fore {g['L_fore']:.3f} m")
    R_map = torso_frames(sim.model, sim.data, jp)
    ik = ArmIK(sim, R_map, arm)

    print("[ik] solving + smoothing ...")
    targets, tdiag = solve_targets(sim, ik, jp, stance)
    derr = direction_errors(sim, ik, targets, jp, R_map)
    print(f"[ik] upper-arm dir err: median {derr['upper_deg_median']:.1f}° mean {derr['upper_deg_mean']:.1f}° "
          f"({derr['upper_within10_pct']:.0%} frames <10°) | forearm best-effort median {derr['fore_deg_median']:.1f}°")
    print(f"[ik] target jump raw {tdiag['target_jump_raw_rad_per_frame']:.3f} -> "
          f"smoothed {tdiag['target_jump_smoothed_rad_per_frame']:.3f} rad/frame")

    dt = sim.timestep
    frames_img = []
    base_zs = []
    track_errs = []
    q_prev = {j: stance[j] for j in ALL_ARM_JOINTS}
    q_filt = {j: stance[j] for j in ALL_ARM_JOINTS}  # low-pass state
    alpha = 0.35  # per-ms smoothing factor (tau ~ 2.5 ms * (1-a)/a ~ 4.6 ms)

    def run(kind, seconds=None, frame_idx=None, record=False):
        nonlocal q_prev, q_filt
        n = max(1, int(round((seconds if seconds is not None else 1.0 / fps) / dt)))
        for step in range(n):
            tgt = dict(stance)
            if kind == "frame":
                f0, f1 = targets[frame_idx], targets[min(frame_idx + 1, T - 1)]
                u = step / max(n - 1, 1)
                for j in ALL_ARM_JOINTS:
                    tgt[j] = (1 - u) * f0[j] + u * f1[j]
            elif kind == "ramp":
                u = step / max(n - 1, 1)
                for j in ALL_ARM_JOINTS:
                    tgt[j] = (1 - u) * stance[j] + u * targets[0][j]
            else:
                for j in ALL_ARM_JOINTS:
                    tgt[j] = q_prev[j]
            for j in ALL_ARM_JOINTS:  # first-order low-pass on arm targets
                q_filt[j] = alpha * tgt[j] + (1 - alpha) * q_filt[j]
            sim.step(pd.compute(sim, {**tgt, **q_filt}))
            q_prev = {j: tgt[j] for j in ALL_ARM_JOINTS}
        state = sim.get_robot_state()
        base_zs.append(float(state["base_position"][2]))
        pos = state["joint_positions"]
        track_errs.append(float(np.sqrt(np.mean([(pos[j] - q_prev[j]) ** 2
                                                 for j in ALL_ARM_JOINTS]))))
        if record:
            frames_img.append(sim.render(width=640, height=480))

    print("[sim] running timeline ...")
    run("hold", args.hold_s)
    run("ramp", args.ramp_s)
    for t in range(T):
        run("frame", frame_idx=t, record=True)
    run("hold", 0.5)

    base_z = np.array(base_zs)
    metrics = {
        "frames": T, "fps": fps,
        "base_z_min": float(base_z.min()), "base_z_max": float(base_z.max()),
        "fell": bool(base_z.min() < 0.42),
        "pd_tracking_rmse_rad": float(np.mean(track_errs)),
        "pd_tracking_max_rad": float(np.max(track_errs)),
        "direction_error_deg": derr,
        "target_smoothness": tdiag,
    }
    with open(out_dir / "k1_arm_wave_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    from PIL import Image, ImageDraw

    gif_imgs = []
    for i, arr in enumerate(frames_img):
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 262, 22], fill=(0, 0, 0))
        d.text((6, 5), f"K1 Arm Wave  frame {i + 1}/{T}", fill=(255, 255, 0))
        gif_imgs.append(im)
    gif_imgs[0].save(out_dir / "k1_arm_wave.gif", save_all=True,
                     append_images=gif_imgs[1:], duration=int(1000 / fps), loop=0)

    keys = frames_img[::max(1, len(frames_img) // 8)][:8]
    sheet = Image.new("RGB", (4 * 320, 2 * 240), (10, 10, 10))
    for i, arr in enumerate(keys):
        sheet.paste(Image.fromarray(arr).resize((320, 240)), ((i % 4) * 320, (i // 4) * 240))
    sheet.save(out_dir / "k1_arm_wave_sheet.jpg", quality=90)

    # --- side-by-side comparison GIF: human stick | K1 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from motion_studio.export.stick_figure import _project, draw_pose

    state = z["contact_state"].astype(int)
    allxy = _project(jp.reshape(-1, 3), "front")
    xmin, xmax = allxy[:, 0].min() - 0.15, allxy[:, 0].max() + 0.15
    ymin, ymax = allxy[:, 1].min() - 0.3, allxy[:, 1].max() + 0.1
    W, H = 640, 360
    cmp_imgs = []
    for i in range(0, T, 2):
        fig, ax = plt.subplots(figsize=(4, 3.6), dpi=100)
        draw_pose(ax, jp[i], "front", contact_state=int(state[i]), lw=6, joint_size=30)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        stick = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        gif_imgs[i].seek(0) if False else None
        k1 = np.asarray(gif_imgs[i].resize((W, H - 30)))
        canvas = Image.new("RGB", (W * 2 + 20, H), (8, 8, 8))
        d = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(stick).resize((W, H - 30)), (0, 30))
        canvas.paste(Image.fromarray(k1), (W + 20, 30))
        d.text((10, 8), f"HUMAN (GVHMR)   frame {i + 1}/{T}", fill=(255, 255, 120))
        d.text((W + 30, 8), "BOOSTER K1 (MuJoCo, IK retarget)", fill=(120, 255, 120))
        cmp_imgs.append(canvas.resize((960, 270)))
    cmp_imgs[0].save(out_dir / "k1_vs_human.gif", save_all=True,
                     append_images=cmp_imgs[1:], duration=80, loop=0)

    print(f"[done] out={out_dir}")
    print(f"[done] base_z {metrics['base_z_min']:.3f}-{metrics['base_z_max']:.3f} "
          f"fell={metrics['fell']} | pd rmse {metrics['pd_tracking_rmse_rad']:.3f} rad "
          f"max {metrics['pd_tracking_max_rad']:.3f} rad")


if __name__ == "__main__":
    main()
