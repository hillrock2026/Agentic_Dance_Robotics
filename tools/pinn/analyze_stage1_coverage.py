#!/usr/bin/env python3
"""Stage1 coverage analyzer: classify dynamics states from actual recorded
fields (never from episode names) and produce the coverage matrix.

State labels (crossing allowed, documented):
  free           (ncon==0) & (nefc==0) & (|qfrc_constraint|_inf==0)   [frozen def]
    actuated_free       free & max|ctrl| > 1e-6
    zero_ctrl_free      free & max|ctrl| <= 1e-6
  support        foot contact classes from recorded foot_l/foot_r:
                 none / single-L / single-R / double
    body_contact        no foot contact but ncon>0 (trunk/knee/... contact)
  transition     foot support class changed vs previous step
  impact         normalized max-dof |qacc| > 10 sigma (frozen rule) OR
                 (support transition within +/-2 steps AND
                  |delta qfrc_constraint|_inf > 100 N between adjacent steps)
  stable_support double support & ~impact
Labels are written per-sample; totals use explicit label sets to avoid
double counting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import stage0_common as sc

SUPPORT_NAMES = ["none", "single_L", "single_R", "double"]

# ---------------------------------------------------------------- geometry
_MJ_MODEL = None
_FOOT_BOXES = None      # side -> list of (body_id, geom_pos, geom_quat, corners(8,3))


def _init_geometry():
    """Foot collision box corners in body frame (for geometric support labels).

    Same construction as virtual_k1.foot_geometry: corner_world =
    body_xpos + body_xmat @ (geom_pos + R(geom_quat) @ corner_local).
    """
    global _MJ_MODEL, _FOOT_BOXES
    if _MJ_MODEL is not None:
        return
    import mujoco
    import yaml
    root = Path(__file__).resolve().parents[2]
    xml = root / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"
    cfg = yaml.safe_load((root / "virtual_k1/config/k1_sim.yaml").read_text())
    m = mujoco.MjModel.from_xml_path(str(xml))
    _MJ_MODEL = m
    _FOOT_BOXES = {}
    for side in ("left", "right"):
        bid = m.body(cfg["contact"]["foot_bodies"][side]).id
        boxes = []
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) != bid or int(m.geom_type[gid]) != \
                    int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            sx, sy, sz = m.geom_size[gid]
            corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy)
                                for z in (-sz, sz)])
            boxes.append((bid, m.geom_pos[gid].copy(),
                          m.geom_quat[gid].copy(), corners))
        _FOOT_BOXES[side] = boxes


def reconstruct_support(qpos: np.ndarray, tol: float = 0.002):
    """Per-step (foot_l, foot_r) geometric contact booleans from corner
    clearance to the z=0 ground plane (tolerance 2 mm, cf. k1_sim.yaml).

    Used to replace the corrupted foot_l/foot_r fields in Stage0 episodes
    (collector side-detection bug); cross-validated against the FIXED physical
    flags on Stage1 episodes.
    """
    _init_geometry()
    import mujoco
    m = _MJ_MODEL
    d = mujoco.MjData(m)
    n = qpos.shape[0]
    out = np.zeros((n, 2), dtype=bool)
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for si, side in enumerate(("left", "right")):
            lowest = np.inf
            for bid, gpos, gquat, corners in _FOOT_BOXES[side]:
                Rg = np.zeros(9)
                mujoco.mju_quat2Mat(Rg, gquat)
                Rg = Rg.reshape(3, 3)
                Rb = d.xmat[bid].reshape(3, 3)
                world = d.xpos[bid] + (corners @ Rg.T + gpos) @ Rb.T
                lowest = min(lowest, float(world[:, 2].min()))
            out[k, si] = lowest <= tol
    return out


def classify_episode(e: dict, mu_a, std_a, support: np.ndarray,
                     dt: float = 1e-3) -> dict:
    n = e["n_steps"]
    free = e["free"]
    support_l = support[:, 0].astype(int)
    support_r = support[:, 1].astype(int)
    sup = np.full(n, 3, dtype=np.int8)          # 0 none, 1 sL, 2 sR, 3 double
    sup[(support_l == 1) & (support_r == 0)] = 1
    sup[(support_l == 0) & (support_r == 1)] = 2
    sup[(support_l == 0) & (support_r == 0)] = 0
    transition = np.zeros(n, dtype=bool)
    transition[1:] = sup[1:] != sup[:-1]

    band = np.abs((e["qacc"] - mu_a) / std_a).max(axis=1)
    dcon = np.zeros(n)
    dcon[1:] = np.abs(np.diff(e["qfrc_constraint"], axis=0)).max(axis=1)
    trans_window = np.zeros(n, dtype=bool)
    for k in (-2, -1, 0, 1, 2):
        trans_window |= np.roll(transition, k)
    impact = (band > 10.0) | (trans_window & (dcon > 100.0))

    max_ctrl = np.abs(e["X"][:, 59:81]).max(axis=1)
    return {
        "free": free,
        "actuated_free": free & (max_ctrl > 1e-6),
        "zero_ctrl_free": free & (max_ctrl <= 1e-6),
        "support": sup,
        "transition": transition,
        "impact": impact,
        "body_contact": (sup == 0) & (~free),
        "stable_support": (sup == 3) & (~impact),
        "single_support": (sup == 1) | (sup == 2),
        "band": band,
        "max_ctrl": max_ctrl,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=sc.DATA_DIR)
    ap.add_argument("--stats", type=Path, default=sc.DATA_DIR / "norm_stats.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    index = json.loads((args.data_dir / "index.json").read_text())
    names = sorted(e_p["episode_id"] for e_p in index["episodes"])
    stats = sc.load_norm_stats(args.stats)
    mu_a, std_a = stats["mu_a"], stats["std_a"]
    eps = sc.load_episodes(names, data_dir=args.data_dir,
                           extra_fields=("qpos", "qvel", "ncon", "nefc"))
    print("reconstructing geometric foot support labels ...")
    support_by_ep = {n: reconstruct_support(eps[n]["qpos"]) for n in names}

    total = sum(eps[n]["n_steps"] for n in names)
    print(f"episodes={len(names)} total_samples={total}")

    classes = {k: {} for k in ("free", "actuated_free", "zero_ctrl_free",
                               "body_contact", "stable_support", "single_support",
                               "transition", "impact")}
    support_counts = {s: {} for s in SUPPORT_NAMES}
    for name in names:
        e = eps[name]
        lab = classify_episode(e, mu_a, std_a, support_by_ep[name])
        for k in classes:
            c = int(lab[k].sum())
            if c:
                classes[k][name] = c
        for si, sname in enumerate(SUPPORT_NAMES):
            c = int((lab["support"] == si).sum())
            if c:
                support_counts[sname][name] = c

    print("\n=== coverage matrix (samples per class, by episode) ===")
    for k, d in classes.items():
        tot = sum(d.values())
        eps_n = len(d)
        top = sorted(d.items(), key=lambda x: -x[1])
        conc = f" top2={top[0][0]}({100 * top[0][1] / tot:.0f}%),{top[1][0]}({100 * top[1][1] / tot:.0f}%)" if len(top) > 1 else ""
        print(f"{k:16s} {tot:7d} ({100 * tot / total:5.1f}%) episodes={eps_n:2d}{conc}")

    print("\n=== foot support classes ===")
    for sname, d in support_counts.items():
        tot = sum(d.values())
        print(f"{sname:9s} {tot:7d} ({100 * tot / total:5.1f}%) episodes={len(d):2d}")

    # ---- free-sample diversity --------------------------------------------
    # consistency: physical free mask vs geometric 'none' support
    free_none = sum(int((eps[n]["free"] & (support_by_ep[n][:, 0] == 0)
                         & (support_by_ep[n][:, 1] == 0)).sum()) for n in names)
    print(f"\nfree samples with geometric no-foot-contact: {free_none}/{classes['free'] and sum(classes['free'].values())}")

    print("\n=== free-sample diversity (per source type) ===")
    index_by_id = {e_p["episode_id"]: e_p["action_type"] for e_p in index["episodes"]}
    for src in sorted(set(index_by_id.values())):
        names_src = [n for n in names if index_by_id[n] == src]
        arrs = [classify_episode(eps[n], mu_a, std_a, support_by_ep[n]) for n in names_src]
        sel = np.zeros(sum(eps[n]["n_steps"] for n in names_src), dtype=bool)
        free_sel = np.zeros_like(sel)
        vlin, vang, qd, tq = [], [], [], []
        ofs = 0
        for n, lab in zip(names_src, arrs):
            t = eps[n]["n_steps"]
            sel[ofs:ofs + t] = True
            free_sel[ofs:ofs + t] = lab["free"]
            f = lab["free"]
            if f.any():
                vlin.append(np.linalg.norm(eps[n]["qvel"][f][:, 0:3], axis=1))
                vang.append(np.linalg.norm(eps[n]["qvel"][f][:, 3:6], axis=1))
                qd.append(np.abs(eps[n]["qvel"][f][:, 6:]).max(axis=1))
                tq.append(lab["max_ctrl"][f])
            ofs += t
        fr = int(free_sel.sum())
        if fr == 0:
            print(f"  {src:16s} free=0")
            continue
        vl = np.concatenate(vlin); va = np.concatenate(vang)
        qdm = np.concatenate(qd); tqm = np.concatenate(tq)
        print(f"  {src:16s} free={fr:5d}"
              f" |vlin| p50/p95={np.percentile(vl, 50):.2f}/{np.percentile(vl, 95):.2f} m/s"
              f" |vang| p50/p95={np.percentile(va, 50):.2f}/{np.percentile(va, 95):.2f} rad/s"
              f" |qd|max p50/p95={np.percentile(qdm, 50):.2f}/{np.percentile(qdm, 95):.2f} rad/s"
              f" |ctrl|max p50/p95={np.percentile(tqm, 50):.1f}/{np.percentile(tqm, 95):.1f} Nm")

    if args.out:
        out = {
            "classes": {k: d for k, d in classes.items()},
            "support": support_counts,
            "total_samples": total,
            "episodes": len(names),
        }
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
