"""Full-body Human(SMPL) -> K1 retargeting via per-chain direction IK.

Method (all coordinates resolved numerically against the real model — no
name-based sign guessing):

  arms : shoulder(2) -> upper-arm dir (exact on reachable frames);
         elbow(2)   -> forearm dir (best effort, K1 has no wrist)
  legs : hip(3)     -> thigh dir; knee(1) -> shank dir;
         ankle(2)   -> foot dir (best effort)
  head : yaw/pitch derived from neck->head direction
  root : yaw-only quaternion from human facing; xy from human pelvis (1:1,
         via torso-frame map); z = stance height + scaled human bounce

The IK machinery is the MuJoCo Jacobian + damped least-squares approach
validated in the K1 Arm Wave demo (upper-arm median error ~6 deg).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coordinate_transform import yaw_quat_xyzw
from .joint_limits import apply_limits, load_limits_from_xml
from .motion_loader import HumanMotion
from .skeleton_mapping import SMPL_IDS, load_mapping_config

K1_XML = Path(__file__).resolve().parents[1] / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"


@dataclass
class K1Trajectory:
    """Retargeted K1 action in internal standard form."""

    fps: float
    timestamps: np.ndarray                 # (T,)
    joints_raw: dict[str, np.ndarray]      # {joint: (T,)} before limits
    joints: dict[str, np.ndarray]          # after limits+smoothing
    root_pos: np.ndarray                   # (T,3) world, meters
    root_quat: np.ndarray                  # (T,4) xyzw, yaw-only
    quality: dict = field(default_factory=dict)   # direction errors etc.
    limit_report: dict | None = None
    retarget_config: dict = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0] + 1 / self.fps)

    def joint_matrix(self, order: list[str]) -> np.ndarray:
        """(T, 22) joint angles in the given order."""
        return np.stack([self.joints[j] for j in order], axis=1)


class FullBodyRetargeter:
    def __init__(self, xml_path: str | Path | None = None, config_path: str | Path | None = None):
        import mujoco

        self.cfg = load_mapping_config(config_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or K1_XML))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.model.jnt_qposadr[0]:self.model.jnt_qposadr[0] + 7] = self.model.qpos0[:7]
        mujoco.mj_forward(self.model, self.data)
        self.limits = load_limits_from_xml(xml_path or K1_XML)
        self._init_stance()
        self._init_geometry()
        self.R_map: np.ndarray | None = None
        self.height_scale: float = 1.0
        # official standing root height (first row of the official CSV); the
        # raw XML free-joint default (z=1.0, mid-air) is NOT a stance
        self.stance_hip_height: float = self._official_stance_root_z()

    # ------------------------------------------------------------------
    def _jid(self, name):
        import mujoco

        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body(self, name):
        import mujoco

        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _anchor(self, d, name):
        return d.xanchor[self._jid(name)].copy()

    def _init_stance(self):
        self.stance = {}
        for jn, (lo, hi) in self.limits.items():
            self.stance[jn] = float(self.model.qpos0[self.model.jnt_qposadr[self._jid(jn)]])
        # official stance = K1Sim standing pose (verified equal to the first
        # row of the official CSVs); qpos0 is the raw XML zero pose — use the
        # official standing values instead where they differ.
        official = self._official_stance()
        if official:
            self.stance.update(official)

    def _official_stance_root_z(self) -> float:
        path = Path(__file__).resolve().parents[1] / (
            "virtual_k1/assets/booster_assets/motions/K1/k1_fight_001.csv")
        if path.exists():
            row = next(csv.DictReader(open(path)))
            return float(row["root_pos_z"])
        return float(self.model.qpos0[2])

    def _official_stance(self) -> dict | None:
        """First data row of k1_fight_001.csv == official standing pose."""
        import csv

        from .skeleton_mapping import csv_name_to_xml

        path = Path(__file__).resolve().parents[1] / (
            "virtual_k1/assets/booster_assets/motions/K1/k1_fight_001.csv")
        if not path.exists():
            return None
        row = next(csv.DictReader(open(path)))
        return {csv_name_to_xml(k): float(v) for k, v in row.items()
                if k.startswith(("aahead", "left_", "right_", "aaleft_", "aaright_"))
                and k.endswith("_joint")}

    def _init_geometry(self):
        """Anatomical reference points, from mesh/geom geometry (local frames)."""
        self.toe_local = self._distal_point("left_ankle_roll_link", prefer="+x")
        self.fore_end_local = self._distal_point("left_elbow_yaw_link", prefer="-z")
        self.toe_local_r = self._distal_point("right_ankle_roll_link", prefer="+x")
        self.fore_end_local_r = self._distal_point("right_elbow_yaw_link", prefer="-z")
        self.head_top_local = self._distal_point("aahead_pitch_link", prefer="+z")

    def _distal_point(self, link: str, prefer: str = "+x") -> np.ndarray:
        """Extreme point of a link's geometry in the LINK-local frame."""
        import mujoco

        bid = self._body(link)
        best = None
        axis, sign = {"+x": (0, 1), "-z": (2, -1), "+z": (2, 1)}[prefer]
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[g]
                va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                pts = self.model.mesh_vert[va:va + vn].astype(np.float64) * self.model.mesh_scale[mid]
                Rg = np.zeros(9)
                mujoco.mju_quat2Mat(Rg, self.model.geom_quat[g])
                pts = pts @ Rg.reshape(3, 3).T + self.model.geom_pos[g]
            elif self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                e = self.model.geom_size[g]
                pts = np.array([[sx * e[0], sy * e[1], sz * e[2]]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + self.model.geom_pos[g]
            else:
                continue
            k = int(np.argmax(sign * pts[:, axis]))
            cand = pts[k]
            if best is None or sign * cand[axis] > sign * best[axis]:
                best = cand
        if best is None:
            raise RuntimeError(f"no geometry on {link}")
        return best.astype(np.float64)

    # ------------------------------------------------------------------
    def calibrate(self, human_joints_t0: np.ndarray):
        from .coordinate_transform import calibrate_torso_map

        self.R_map = calibrate_torso_map(self.model, self.data, human_joints_t0)
        hip_h = human_joints_t0[SMPL_IDS["left_hip"]][1]
        self.height_scale = self.stance_hip_height / max(hip_h, 1e-6)
        return self

    # ------------------------------------------------------------------
    def retarget(self, hm: HumanMotion, smooth_window: int = 5,
                 limit_margin: float = 0.02, upper_body_mode: bool = False) -> K1Trajectory:
        import mujoco

        if self.R_map is None:
            self.calibrate(hm.joint_positions[0])
        T = hm.frame_count
        S = SMPL_IDS
        R = self.R_map
        rcfg = self.cfg["retargeting"]

        # ---- per-frame segment direction targets (human, unit) ----
        def seg(a, b):
            v = hm.joint_positions[:, S[b]] - hm.joint_positions[:, S[a]]
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        tgt = {
            ("left_arm", "upper"): seg("left_shoulder", "left_elbow"),
            ("left_arm", "fore"): seg("left_elbow", "left_wrist"),
            ("right_arm", "upper"): seg("right_shoulder", "right_elbow"),
            ("right_arm", "fore"): seg("right_elbow", "right_wrist"),
            ("left_leg", "thigh"): seg("left_hip", "left_knee"),
            ("left_leg", "shank"): seg("left_knee", "left_ankle"),
            ("left_leg", "foot"): seg("left_ankle", "left_foot"),
            ("right_leg", "thigh"): seg("right_hip", "right_knee"),
            ("right_leg", "shank"): seg("right_knee", "right_ankle"),
            ("right_leg", "foot"): seg("right_ankle", "right_foot"),
        }

        q_cur = dict(self.stance)
        raw = {jn: np.zeros(T) for jn in self.limits}
        head_dirs = hm.joint_positions[:, S["head"]] - hm.joint_positions[:, S["neck"]]
        head_dirs /= np.linalg.norm(head_dirs, axis=1, keepdims=True)

        d_ik = mujoco.MjData(self.model)

        def set_q(q):
            d_ik.qpos[:] = 0
            d_ik.qpos[3] = 1.0  # free joint quat w
            for jn, val in q.items():
                d_ik.qpos[self.model.jnt_qposadr[self._jid(jn)]] = val

        tgt_head = head_dirs  # (T,3) human neck->head unit vectors
        chains = ("left_arm", "right_arm", "head") if upper_body_mode else (
            "left_arm", "right_arm", "left_leg", "right_leg", "head")
        for t in range(T):
            for chain in chains:
                self._solve_chain(d_ik, set_q, q_cur, chain, None, tgt, t, R,
                                  iters=rcfg.get("ik_iters", 60),
                                  head_dir=tgt_head[t] if chain == "head" else None)
            for jn in self.limits:
                raw[jn][t] = q_cur[jn]

        # ---- smoothing (moving average) ----
        k = max(1, int(smooth_window) | 1)
        pad = k // 2
        smoothed = {}
        for jn, arr in raw.items():
            smoothed[jn] = np.convolve(np.pad(arr, pad, mode="edge"),
                                       np.ones(k) / k, mode="valid")

        if upper_body_mode:
            for jn in ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                       "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                       "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                       "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"):
                smoothed[jn][:] = self.stance[jn]

        # ---- limits ----
        limited, report = apply_limits(smoothed, self.limits, margin_frac=limit_margin)

        # ---- root: yaw + xy + scaled bounce ----
        root_pos, root_quat = self._retarget_root(hm)
        if upper_body_mode:
            root_pos[:, 2] = self.stance_hip_height  # legs at stance: stay on ground
        else:
            # GVHMR mono-video recovery drifts in absolute height; keep only a
            # bounded bounce around the stance height
            root_pos[:, 2] = self.stance_hip_height + np.clip(
                root_pos[:, 2] - self.stance_hip_height, -0.12, 0.12)

        traj_mode = ("upper_body_only (legs hold official stance; input leg "
                     "motion amplitude exceeded what PD standing supports)"
                     if upper_body_mode else "full_body")
        traj = K1Trajectory(
            fps=hm.fps,
            timestamps=hm.timestamps.copy(),
            joints_raw=raw,
            joints=limited,
            root_pos=root_pos,
            root_quat=root_quat,
            limit_report=report.__dict__ | {"has_severe": report.has_severe},
            retarget_config={"ik_iters": rcfg.get("ik_iters", 60),
                             "smoothing_window": k,
                             "limit_margin": limit_margin,
                             "mode": traj_mode,
                             "method": "per-chain direction IK (MuJoCo Jacobian, damped LS)"},
        )
        self._measure_quality(traj, hm, tgt)
        return traj

    # ------------------------------------------------------------------
    def _segment_refs(self, chain: str):
        """(start_anchor_joint, end_point_joint, end_local, end_body) per chain
        for direction measurement — mirrors the validated arm conventions."""
        if chain == "left_arm":
            return ("left_shoulder_roll_joint", "left_elbow_pitch_joint",
                    self.fore_end_local, "left_elbow_yaw_link")
        if chain == "right_arm":
            return ("right_shoulder_roll_joint", "right_elbow_pitch_joint",
                    self.fore_end_local_r, "right_elbow_yaw_link")
        if chain == "left_leg":
            return ("left_hip_pitch_joint", "left_knee_pitch_joint",
                    None, "left_ankle_pitch_joint")
        if chain == "right_leg":
            return ("right_hip_pitch_joint", "right_knee_pitch_joint",
                    None, "right_ankle_pitch_joint")
        raise KeyError(chain)

    HEAD_STAGE = (("aahead_yaw_joint", "aahead_pitch_joint"),)

    def _chain_stages(self, chain: str):
        if chain == "head":
            return [(self.HEAD_STAGE[0],
                     "aahead_yaw_joint",
                     ("point", "aahead_pitch_link", "head_top"), "head", 0.5)]
        """Stage table per chain: list of
        (control_joints, start_anchor_joint | None, end ('anchor',joint) |
         ('point', body, local), target_key, weight_iters)."""
        if chain == "left_arm":
            return [
                (("aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint"),
                 "left_shoulder_roll_joint", ("anchor", "left_elbow_pitch_joint"), "upper", 1.0),
                (("left_elbow_pitch_joint", "left_elbow_yaw_joint"),
                 "left_elbow_pitch_joint", ("point", "left_elbow_yaw_link", "fore"), "fore", 0.5),
            ]
        if chain == "right_arm":
            return [
                (("aaright_shoulder_pitch_joint", "right_shoulder_roll_joint"),
                 "right_shoulder_roll_joint", ("anchor", "right_elbow_pitch_joint"), "upper", 1.0),
                (("right_elbow_pitch_joint", "right_elbow_yaw_joint"),
                 "right_elbow_pitch_joint", ("point", "right_elbow_yaw_link", "fore"), "fore", 0.5),
            ]
        if chain == "left_leg":
            return [
                (("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"),
                 "left_hip_pitch_joint", ("anchor", "left_knee_pitch_joint"), "thigh", 1.0),
                (("left_knee_pitch_joint",),
                 "left_knee_pitch_joint", ("anchor", "left_ankle_pitch_joint"), "shank", 0.5),
                (("left_ankle_pitch_joint", "left_ankle_roll_joint"),
                 "left_ankle_pitch_joint", ("point", "left_ankle_roll_link", "toe"), "foot", 0.35),
            ]
        if chain == "right_leg":
            return [
                (("right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"),
                 "right_hip_pitch_joint", ("anchor", "right_knee_pitch_joint"), "thigh", 1.0),
                (("right_knee_pitch_joint",),
                 "right_knee_pitch_joint", ("anchor", "right_ankle_pitch_joint"), "shank", 0.5),
                (("right_ankle_pitch_joint", "right_ankle_roll_joint"),
                 "right_ankle_pitch_joint", ("point", "right_ankle_roll_link", "toe"), "foot", 0.35),
            ]
        raise KeyError(chain)

    def _local_point(self, which: str, chain: str) -> tuple[str, np.ndarray]:
        if which == "head_top":
            return ("aahead_pitch_link", self.head_top_local)
        if which == "fore":
            return (("left_elbow_yaw_link", self.fore_end_local) if chain == "left_arm"
                    else ("right_elbow_yaw_link", self.fore_end_local_r))
        return (("left_ankle_roll_link", self.toe_local) if chain == "left_leg"
                else ("right_ankle_roll_link", self.toe_local_r))

    def _solve_chain(self, d_ik, set_q, q_cur, chain, segs, tgt, t, R, iters=60,
                     head_dir=None):
        """Sequential per-stage direction IK for one chain at frame t."""
        import mujoco

        m = self.model
        margin = 0.03
        if chain == "head":
            tgt = {("head", "head"): np.asarray(head_dir)[None, :]}

        def dofadr(jn):
            return int(m.jnt_dofadr[self._jid(jn)])

        for ctrl, start_j, end_spec, key, wfrac in self._chain_stages(chain):
            arr = tgt[(chain, key)]
            tu = R @ (arr[0] if chain == 'head' else arr[t])
            n_it = max(8, int(iters * wfrac))
            for it in range(n_it):
                set_q(q_cur)
                mujoco.mj_forward(m, d_ik)
                sh = self._anchor(d_ik, start_j)
                if end_spec[0] == "anchor":
                    end = self._anchor(d_ik, end_spec[1])
                    end_body_name = end_spec[1].replace("_joint", "_link")
                else:
                    body_name, local = self._local_point(end_spec[1], chain)
                    bid = self._body(body_name)
                    end = d_ik.xpos[bid] + d_ik.xmat[bid].reshape(3, 3) @ local
                    end_body_name = body_name
                v = end - sh
                r = v - np.linalg.norm(v) * tu
                if np.linalg.norm(r) < 1e-4:
                    break
                Jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d_ik, Jp, None, end, self._body(end_body_name))
                J = Jp[:, [dofadr(j) for j in ctrl]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), -r)
                step = 0.6 if it < n_it // 2 else 0.25
                for kk, jn in enumerate(ctrl):
                    lo, hi = self.limits[jn]
                    span = hi - lo
                    q_cur[jn] = float(np.clip(q_cur[jn] + step * dq[kk],
                                              lo + margin * span, hi - margin * span))

    def _segment_body(self, chain: str, stage: str) -> str:
        if chain == "left_arm":
            return "left_elbow_pitch_link"
        if chain == "right_arm":
            return "right_elbow_pitch_link"
        if chain == "left_leg":
            return "left_knee_pitch_link"
        if chain == "right_leg":
            return "right_knee_pitch_link"
        raise KeyError(chain)

    # ------------------------------------------------------------------
    def _retarget_root(self, hm: HumanMotion):
        import mujoco

        from motion_studio.core.rotations import aa_to_mat

        T = hm.frame_count
        S = SMPL_IDS
        R = self.R_map
        root_pos = np.zeros((T, 3))
        root_quat = np.zeros((T, 4))
        # K1 stance root: trunk at stance height; take xy origin at first pose
        stance_xy = np.array([0.0, 0.0])
        pelvis = hm.transl
        pelvis_rel = pelvis - pelvis[0]
        pelvis_h = hm.joint_positions[:, S["left_hip"], 1]  # hip height over time
        h0 = pelvis_h[0]
        for t in range(T):
            # yaw from human facing (global_orient applied to SMPL +Z front)
            Rw = aa_to_mat(hm.global_orient[t][None])[0]
            fwd_ms = Rw @ np.array([0.0, 0.0, 1.0])
            fwd_ms[1] = 0.0
            fwd_ms /= max(np.linalg.norm(fwd_ms), 1e-9)
            fwd_k1 = R @ fwd_ms
            root_quat[t] = yaw_quat_xyzw(fwd_k1)
            # xy: pelvis horizontal displacement mapped through R
            d_ms = np.array([pelvis_rel[t][0], 0.0, pelvis_rel[t][2]])
            d_k1 = R @ d_ms
            root_pos[t, :2] = stance_xy + d_k1[:2]
            # z: stance height + scaled bounce
            root_pos[t, 2] = self.stance_hip_height + (pelvis_h[t] - h0) * self.height_scale
        return root_pos, root_quat

    # ------------------------------------------------------------------
    def _measure_quality(self, traj: K1Trajectory, hm: HumanMotion, tgt):
        """Achieved segment-direction errors under the limited trajectory."""
        import mujoco

        m = self.model
        d = mujoco.MjData(m)
        R = self.R_map
        S = SMPL_IDS
        errs: dict[str, list] = {"upper_arm": [], "forearm": [], "thigh": [], "shank": [], "foot": []}
        for t in range(traj.frame_count):
            d.qpos[:] = 0
            d.qpos[3] = 1.0
            for jn, val in traj.joints.items():
                d.qpos[m.jnt_qposadr[self._jid(jn)]] = val[t]
            mujoco.mj_forward(m, d)
            for chain in ("left_arm", "right_arm"):
                sh = self._anchor(d, "left_shoulder_roll_joint" if chain[0] == "l" else "right_shoulder_roll_joint")
                el = self._anchor(d, "left_elbow_pitch_joint" if chain[0] == "l" else "right_elbow_pitch_joint")
                fbl = self._body("left_elbow_yaw_link" if chain[0] == "l" else "right_elbow_yaw_link")
                loc = self.fore_end_local if chain[0] == "l" else self.fore_end_local_r
                end = d.xpos[fbl] + d.xmat[fbl].reshape(3, 3) @ loc
                gu = (el - sh) / np.linalg.norm(el - sh)
                gf = (end - el) / np.linalg.norm(end - el)
                tu = R @ tgt[(chain, "upper")][t]
                tf = R @ tgt[(chain, "fore")][t]
                errs["upper_arm"].append(np.degrees(np.arccos(np.clip(gu @ tu, -1, 1))))
                errs["forearm"].append(np.degrees(np.arccos(np.clip(gf @ tf, -1, 1))))
            for chain in ("left_leg", "right_leg"):
                side = "left" if chain[0] == "l" else "right"
                hip = self._anchor(d, f"{side}_hip_pitch_joint")
                knee = self._anchor(d, f"{side}_knee_pitch_joint")
                fbl = self._body(f"{side}_ankle_roll_link")
                loc = self.toe_local if side == "left" else self.toe_local_r
                toe = d.xpos[fbl] + d.xmat[fbl].reshape(3, 3) @ loc
                gt = (knee - hip) / np.linalg.norm(knee - hip)
                gs = (toe - knee) / np.linalg.norm(toe - knee)
                gf = (toe - knee) / np.linalg.norm(toe - knee)
                tt = R @ tgt[(chain, "thigh")][t]
                ts = R @ tgt[(chain, "shank")][t]
                tfd = R @ tgt[(chain, "foot")][t]
                ank = self._anchor(d, f"{side}_ankle_pitch_joint")
                gft = (toe - ank) / np.linalg.norm(toe - ank)
                errs["thigh"].append(np.degrees(np.arccos(np.clip(gt @ tt, -1, 1))))
                errs["shank"].append(np.degrees(np.arccos(np.clip(gs @ ts, -1, 1))))
                errs["foot"].append(np.degrees(np.arccos(np.clip(gft @ tfd, -1, 1))))
        traj.quality = {
            f"{k}_deg_median": float(np.median(v)) for k, v in errs.items()
        } | {f"{k}_deg_mean": float(np.mean(v)) for k, v in errs.items()}


def retarget_to_k1(hm: HumanMotion, config_path: str | Path | None = None,
                   xml_path: str | Path | None = None) -> K1Trajectory:
    cfg = load_mapping_config(config_path)
    r = FullBodyRetargeter(xml_path=xml_path, config_path=config_path)
    rc = cfg["retargeting"]
    return r.retarget(hm, smooth_window=rc.get("smoothing_window_frames", 5),
                      limit_margin=cfg["limits"]["margin_frac"])
