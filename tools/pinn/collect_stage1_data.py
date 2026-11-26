#!/usr/bin/env python3
"""Stage1-01 data expansion: copy the 20 Stage0 episodes unchanged into
stage1_data/, then collect 10 new episodes chosen by the Stage0 coverage
inventory:

  hold_throw h0..h5   (seeds 70-75)  airborne init, PD holds the stance pose
                      (gravity compensation active) during ballistic flight ->
                      ACTUATED free-flight samples + controlled landing impacts
                      + support re-establishment transitions. Directly fills
                      the two largest Stage0 gaps (actuated free diversity,
                      controlled landing/transition).
  pd_top_rock s76/s77 new seeds of the research-target motion (footwork =
                      frequent support switching); enrich TEST-side coverage.
  pd_popping s78 / pd_arm_wave_usr s79  extra train-side stable-support
                      diversity (Stage0: 76% of stable support came from two
                      arm_wave episodes).

Original stage0_data/ is never modified. Reuses Collector from
collect_stage0_data.py (with the fixed per-side foot contact detection).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import collect_stage0_data as c0
import stage0_common as sc

STAGE1_DIR = REPO_ROOT / "tools/pinn/stage1_data"


def hold_throw_factory(col: c0.Collector, seed: int):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    tilt = np.deg2rad(rng.uniform(0, 45))
    z = float(0.70 + rng.uniform(0, 0.25))
    col.reset_episode(z, c0.quat_about_axis(axis, tilt),
                      qvel_std=(0.3, 0.8, 0.5), joint_noise=0.15,
                      seed=seed + 2000)
    stance = col.stance_q.copy()
    gain_scale = float(rng.uniform(0.8, 1.2))

    def controller(step: int):
        return col.pd_torque(stance, gain_scale)

    return controller, 2000


def main() -> None:
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    # 1) copy the untouched Stage0 episodes
    copied = 0
    for f in sorted(sc.DATA_DIR.glob("*.npz")):
        shutil.copy2(f, STAGE1_DIR / f.name)
        copied += 1
    print(f"copied {copied} Stage0 episodes -> {STAGE1_DIR}")

    # 2) collect the new episodes (fixed collector: correct foot_l/foot_r)
    col = c0.Collector(c0.DEFAULT_XML, c0.DEFAULT_CFG)
    plan = []
    plan.append(("pd_top_rock_s76", "pd_top_rock", 76,
                 lambda: c0.pd_episode_factory(col, "top_rock", 76)))
    plan.append(("pd_top_rock_s77", "pd_top_rock", 77,
                 lambda: c0.pd_episode_factory(col, "top_rock", 77)))
    plan.append(("pd_popping_s78", "pd_popping", 78,
                 lambda: c0.pd_episode_factory(col, "popping", 78)))
    plan.append(("pd_arm_wave_usr_s79", "pd_arm_wave_usr", 79,
                 lambda: c0.pd_episode_factory(
                     col, "arm_wave_upper_body_stationary_root", 79)))
    for i, s in enumerate(range(70, 76)):
        plan.append((f"hold_throw_h{i}", "hold_throw", s,
                     lambda s=s: hold_throw_factory(col, s)))

    index = []
    for ep_id, action_type, seed, factory in plan:
        controller, n_steps = factory()
        ep = col.run_episode(ep_id, action_type, seed, controller, n_steps)
        path = STAGE1_DIR / f"{ep_id}.npz"
        np.savez_compressed(path, **ep["data"])
        info = {k: ep[k] for k in ("episode_id", "action_type", "seed", "ended_by")}
        info["n_steps"] = int(ep["data"]["qpos"].shape[0])
        free = int(((ep["data"]["ncon"] == 0) & (ep["data"]["nefc"] == 0)
                    & (np.abs(ep["data"]["qfrc_constraint"]).max(axis=1) == 0.0)).sum())
        info["free_samples"] = free
        info["origin"] = "stage1_new"
        index.append(info)
        print(f"{ep_id:22s} type={action_type:12s} steps={info['n_steps']:5d} "
              f"ended_by={ep['ended_by']:5s} free={free:5d} "
              f"({100 * free / max(info['n_steps'], 1):.1f}%)")

    # 3) full Stage1 index: old (origin=stage0) + new episodes
    old_index = json.loads((sc.DATA_DIR / "index.json").read_text())
    episodes = []
    for e in old_index["episodes"]:
        e = dict(e)
        e["origin"] = "stage0"
        episodes.append(e)
    episodes.extend(index)
    manifest = {
        "mujoco_version": mujoco.__version__,
        "xml": str(c0.DEFAULT_XML),
        "xml_sha256": c0.sha256_of(c0.DEFAULT_XML),
        "timestep": float(col.model.opt.timestep),
        "na": int(col.model.na),
        "xfrc_applied_zero": True,
        "collector_protocol": "ctrl -> mj_forward -> snapshot -> mj_step (Task 1 P-A)",
        "collector_fixes": [
            "STAGE1: per-side foot contact detection fixed (Stage0 foot_l was "
            "always 0 and foot_r fired for either foot); Stage0 npz keep the "
            "broken fields, Stage1 support labels use geometric reconstruction "
            "for ALL episodes (see analyze_stage1_coverage.py)",
        ],
        "episodes": episodes,
    }
    (STAGE1_DIR / "index.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nStage1 index: {len(episodes)} episodes -> {STAGE1_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
