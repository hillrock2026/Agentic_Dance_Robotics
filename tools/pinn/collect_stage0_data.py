#!/usr/bin/env python3
"""Stage0-D data collection: small pilot set of K1 dynamics episodes.

Protocol (Task 1 §12, P-A): per physical step (1 kHz, model timestep unchanged)
    set ctrl -> mj_forward -> snapshot (state + dynamics + M) -> mj_step.
All recorded fields belong to the same pre-step solve state. A per-step
solver-consistency assert (|M qacc + bias - act - passive - applied - constraint|
< 1e-5) guards against protocol drift.

Episode types:
  pd_<motion>   software PD (V1 gains, gravity comp) tracks a retargeted
                reference trajectory (datasets/k1_dance_v1/...); root free.
  drop          ballistic: randomized airborne init, zero torque.
  ou_torque     stance init, first-order OU torque exploration.

Outputs one NPZ per episode + index.json under --out-dir.
Read-only w.r.t. model / config / trajectory assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"
DEFAULT_CFG = REPO_ROOT / "virtual_k1/config/k1_sim.yaml"
DATASETS = REPO_ROOT / "datasets/k1_dance_v1"

NV = 28
IU = np.triu_indices(NV)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_joint(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "aa" + name)
    if jid < 0:
        raise KeyError(f"joint not found in model: {name}")
    return jid


def load_reference(path: Path) -> tuple[list[str], np.ndarray]:
    header = path.read_text().splitlines()[0].strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    cols = [k for k, h in enumerate(header) if h.endswith("_joint")]
    return [header[k] for k in cols], data[:, cols]


def rpy_deg(q_wxyz) -> tuple[float, float]:
    w, x, y, z = q_wxyz
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def quat_about_axis(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), *(axis * s)])


class Collector:
    def __init__(self, xml: Path, cfg_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert abs(self.model.opt.timestep - 0.001) < 1e-12, "model timestep changed"
        m = self.model
        self.gains = self.cfg["pd_controller"]["gains"]
        # per-joint PD state in XML hinge order (dof 6..27)
        self.jids: list[int] = []
        self.qadr: list[int] = []
        self.vadr: list[int] = []
        self.kp = np.zeros(m.nu)
        self.kd = np.zeros(m.nu)
        self.flo = np.zeros(m.nu)
        self.fhi = np.zeros(m.nu)
        stance = self.cfg["initial_pose"]["joints"]
        self.stance_q = np.zeros(m.nq)
        for name, val in stance.items():
            jid = resolve_joint(m, name)
            self.jids.append(jid)
            self.qadr.append(int(m.jnt_qposadr[jid]))
            self.vadr.append(int(m.jnt_dofadr[jid]))
            g = self.gains[name]
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.kp[aid] = g["kp"]
            self.kd[aid] = g["kd"]
            self.flo[aid] = m.actuator_forcerange[aid, 0]
            self.fhi[aid] = m.actuator_forcerange[aid, 1]
            self.stance_q[int(m.jnt_qposadr[jid])] = float(val)
        # foot/ground geom ids for contact bookkeeping (per side!)
        self.ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.foot_geom_ids = {"left": set(), "right": set()}
        for side in ("left", "right"):
            bid = m.body(self.cfg["contact"]["foot_bodies"][side]).id
            for gid in range(m.ngeom):
                if int(m.geom_bodyid[gid]) == bid and m.geom_contype[gid]:
                    self.foot_geom_ids[side].add(gid)
        self.all_foot_geom_ids = self.foot_geom_ids["left"] | self.foot_geom_ids["right"]
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------- helpers
    def reset_episode(self, root_z, root_quat, qvel_std, joint_noise=0.0,
                      seed: int = 0) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        rng = np.random.default_rng(seed)
        d.qpos[0:3] = [0.0, 0.0, root_z]
        d.qpos[3:7] = root_quat
        for k, adr in enumerate(self.qadr):
            q = self.stance_q[adr]
            if joint_noise > 0:
                lo, hi = m.jnt_range[self.jids[k]]
                q = np.clip(q + rng.normal(0, joint_noise), lo + 0.02, hi - 0.02)
            d.qpos[adr] = q
        d.qvel[:3] = rng.normal(0, qvel_std[0], 3)
        d.qvel[3:6] = rng.normal(0, qvel_std[1], 3)
        d.qvel[6:] = rng.normal(0, qvel_std[2], m.nv - 6)
        mujoco.mj_forward(m, d)

    def pd_torque(self, target_q: np.ndarray, gain_scale: float) -> np.ndarray:
        m, d = self.model, self.data
        u = np.zeros(m.nu)
        bias = d.qfrc_bias  # one-step stale from previous forward (controller detail)
        for k in range(len(self.jids)):
            aid = k  # actuator order == hinge order (verified in Task 1)
            adr, vadr = self.qadr[k], self.vadr[k]
            u[aid] = (gain_scale * self.kp[aid] * (target_q[adr] - d.qpos[adr])
                      - gain_scale * self.kd[aid] * d.qvel[vadr] + bias[vadr])
        return np.clip(u, self.flo, self.fhi)

    # ------------------------------------------------------------- stepping
    def run_episode(self, ep_id: str, action_type: str, seed: int,
                    controller, n_steps: int) -> dict:
        """controller(t_step) -> torque vector (nu,); None => zero torque."""
        m, d = self.model, self.data
        rec = {k: [] for k in ("qpos", "qvel", "ctrl", "qacc", "M_upper",
                               "qfrc_bias", "qfrc_actuator", "qfrc_constraint",
                               "qfrc_passive", "qfrc_applied", "actuator_force",
                               "ncon", "nefc", "time", "foot_l", "foot_r",
                               "residual_closure")}
        ended_by = "steps"
        for step in range(n_steps):
            u = controller(step)
            if u is None:
                u = np.zeros(m.nu)
            d.ctrl[:] = u
            mujoco.mj_forward(m, d)                       # solve at (q_t, v_t, u_t)
            M = np.zeros((NV, NV))
            mujoco.mj_fullM(m, d, M)
            Mq = M @ d.qacc
            r = (Mq + d.qfrc_bias - d.qfrc_actuator - d.qfrc_passive
                 - d.qfrc_applied - d.qfrc_constraint)
            res_err = float(np.abs(r).max())
            # Newton terminates at tolerance: on contact samples the closure
            # error scales with the constraint-force magnitude (observed
            # ~2e-4 relative in dynamic settling). Allow 1e-3 relative —
            # still catches protocol drift (O(1) relative errors).
            scale = float(np.abs(Mq).max()) + float(np.abs(d.qfrc_constraint).max()) + 1.0
            if res_err > 1e-3 * scale:
                raise RuntimeError(
                    f"{ep_id} step {step}: residual closure {res_err:.3g} "
                    f"(ncon={d.ncon} nefc={d.nefc} |Mq|inf={np.abs(Mq).max():.4g} "
                    f"|con|inf={np.abs(d.qfrc_constraint).max():.4g}) (protocol drift?)")
            if np.abs(d.qfrc_passive).max() != 0.0 or np.abs(d.qfrc_applied).max() != 0.0 \
                    or np.abs(d.xfrc_applied).max() != 0.0:
                raise RuntimeError(f"{ep_id} step {step}: passive/applied/xfrc nonzero")
            rec["qpos"].append(d.qpos.copy())
            rec["qvel"].append(d.qvel.copy())
            rec["ctrl"].append(d.ctrl.copy())
            rec["qacc"].append(d.qacc.copy())
            rec["M_upper"].append(M[IU])
            rec["qfrc_bias"].append(d.qfrc_bias.copy())
            rec["qfrc_actuator"].append(d.qfrc_actuator.copy())
            rec["qfrc_constraint"].append(d.qfrc_constraint.copy())
            rec["qfrc_passive"].append(d.qfrc_passive.copy())
            rec["qfrc_applied"].append(d.qfrc_applied.copy())
            rec["actuator_force"].append(d.actuator_force.copy())
            rec["ncon"].append(d.ncon)
            rec["nefc"].append(d.nefc)
            rec["time"].append(d.time)
            rec["residual_closure"].append(res_err)
            fl = fr = 0
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in self.foot_geom_ids["left"] or g2 in self.foot_geom_ids["left"]) \
                        and self.ground_id in (g1, g2):
                    fl = 1
                if (g1 in self.foot_geom_ids["right"] or g2 in self.foot_geom_ids["right"]) \
                        and self.ground_id in (g1, g2):
                    fr = 1
            rec["foot_l"].append(fl)
            rec["foot_r"].append(fr)
            roll, pitch = rpy_deg(d.qpos[3:7])
            if d.qpos[2] < 0.42 or abs(roll) > 45.0 or abs(pitch) > 45.0:
                ended_by = "fall"
                break
            mujoco.mj_step(m, d)                          # integrate -> t+dt
        out = {k: np.asarray(v) for k, v in rec.items()}
        out["meta"] = np.array([seed, out["time"].shape[0]])
        return {"episode_id": ep_id, "action_type": action_type, "seed": seed,
                "ended_by": ended_by, "data": out}


def pd_episode_factory(col: Collector, motion: str, seed: int):
    """Hold 0.3 s -> ramp 0.3 s -> track up to 3.0 s of the reference."""
    names, ref = load_reference(DATASETS / motion / "trajectory.csv")
    adrs = [int(col.model.jnt_qposadr[resolve_joint(col.model, n)]) for n in names]
    fps = 50.0
    rng = np.random.default_rng(seed)
    gain_scale = float(rng.uniform(0.8, 1.2))
    axis = rng.normal(size=3)
    tilt = np.deg2rad(rng.normal(0, 2.0))
    root_z = float(col.cfg["initial_pose"]["root_position"][2] + rng.normal(0, 0.005))

    def controller(step: int):
        t = step * col.model.opt.timestep
        if t < 0.3:
            return col.pd_torque(col.stance_q.copy(), gain_scale)
        u_ramp = min((t - 0.3) / 0.3, 1.0)
        if t < 0.6:
            tgt = (1 - u_ramp) * col.stance_q + u_ramp * _frame(0.0)
            return col.pd_torque(tgt, gain_scale)
        tt = t - 0.6
        f = tt * fps
        k = min(int(f), len(ref) - 2)
        frac = min(f - k, 1.0)
        tgt = col.stance_q.copy()
        for j, adr in enumerate(adrs):
            tgt[adr] = (1 - frac) * ref[k, j] + frac * ref[k + 1, j]
        return col.pd_torque(tgt, gain_scale)

    def _frame(local_t: float) -> np.ndarray:
        tgt = col.stance_q.copy()
        k = min(int(local_t * fps), len(ref) - 1)
        for j, adr in enumerate(adrs):
            tgt[adr] = ref[k, j]
        return tgt

    col.reset_episode(root_z, quat_about_axis(axis, tilt), qvel_std=(0.0, 0.0, 0.0),
                      seed=seed)
    # hold 300 + ramp 300 + track up to 3.0 s (capped by trajectory length)
    track_steps = min(int(len(ref) / fps * 1000), 3000)
    return controller, 300 + 300 + track_steps


def drop_episode_factory(col: Collector, seed: int):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    tilt = np.deg2rad(rng.uniform(0, 40))
    z = float(0.75 + rng.uniform(0, 0.2))
    col.reset_episode(z, quat_about_axis(axis, tilt),
                      qvel_std=(0.2, 0.5, 0.5), joint_noise=0.1, seed=seed + 1000)
    return (lambda step: None), 1500


def ou_episode_factory(col: Collector, seed: int):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    tilt = np.deg2rad(rng.normal(0, 2.0))
    col.reset_episode(float(col.cfg["initial_pose"]["root_position"][2]),
                      quat_about_axis(axis, tilt), qvel_std=(0.0, 0.0, 0.0),
                      joint_noise=0.02, seed=seed)
    rho, fr = 0.99, col.model.actuator_forcerange.copy()
    sigma = 0.3 * np.abs(fr[:, 1])
    state = np.zeros(col.model.nu)

    def controller(step: int):
        nonlocal state
        state = rho * state + sigma * np.sqrt(1 - rho ** 2) * rng.normal(size=col.model.nu)
        return np.clip(state, 0.5 * fr[:, 0], 0.5 * fr[:, 1])

    return controller, 2000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tools/pinn/stage0_data")
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    col = Collector(args.xml, args.config)
    # factories are called immediately before each episode runs, so each
    # episode starts from its own reset state (no cross-episode leakage)
    plan = []
    plan.append(("pd_top_rock", "pd_top_rock", 10,
                 lambda: pd_episode_factory(col, "top_rock", 10)))
    plan.append(("pd_arm_wave", "pd_arm_wave", 20,
                 lambda: pd_episode_factory(col, "arm_wave", 20)))
    for s in (30, 31):
        plan.append((f"pd_popping_s{s}", "pd_popping", s,
                     lambda s=s: pd_episode_factory(col, "popping", s)))
    for s in (40, 41):
        plan.append((f"pd_arm_wave_usr_s{s}", "pd_arm_wave_usr", s,
                     lambda s=s: pd_episode_factory(
                         col, "arm_wave_upper_body_stationary_root", s)))
    for i, s in enumerate(range(50, 58)):
        plan.append((f"drop_d{i}", "drop", s,
                     lambda s=s: drop_episode_factory(col, s)))
    for i, s in enumerate(range(60, 66)):
        plan.append((f"ou_o{i}", "ou_torque", s,
                     lambda s=s: ou_episode_factory(col, s)))

    index = []
    for ep_id, action_type, seed, factory in plan:
        controller, n_steps = factory()
        ep = col.run_episode(ep_id, action_type, seed, controller, n_steps)
        path = args.out_dir / f"{ep_id}.npz"
        np.savez_compressed(path, **ep["data"])
        info = {k: ep[k] for k in ("episode_id", "action_type", "seed", "ended_by")}
        info["n_steps"] = int(ep["data"]["qpos"].shape[0])
        info["npy"] = path.name
        index.append(info)
        free = int(((ep["data"]["ncon"] == 0) & (ep["data"]["nefc"] == 0)
                    & (np.abs(ep["data"]["qfrc_constraint"]).max(axis=1) == 0.0)).sum())
        print(f"{ep_id:24s} type={action_type:16s} steps={info['n_steps']:5d} "
              f"ended_by={ep['ended_by']:5s} free={free:5d} "
              f"({100 * free / max(info['n_steps'], 1):.1f}%)")

    manifest = {
        "mujoco_version": mujoco.__version__,
        "xml": str(args.xml),
        "xml_sha256": sha256_of(args.xml),
        "timestep": float(col.model.opt.timestep),
        "na": int(col.model.na),
        "xfrc_applied_zero": True,
        "collector_protocol": "ctrl -> mj_forward -> snapshot -> mj_step (Task 1 P-A)",
        "episodes": index,
    }
    (args.out_dir / "index.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(index)} episodes -> {args.out_dir} (index.json written)")


if __name__ == "__main__":
    main()
