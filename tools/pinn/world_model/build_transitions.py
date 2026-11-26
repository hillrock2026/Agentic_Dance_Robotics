#!/usr/bin/env python3
"""K1_WM_01 build: materialize single-step transitions + window indices,
run the data inventory and continuity/consistency checks.

Outputs (under tools/pinn/world_model/transitions/):
  transitions_{train,val,test}.npz   stacked single-step transitions
  windows_h8_{train,val,test}.npz    window start indices (episode_index, t0)
  transition_inventory.json          per-episode field/shape/time inventory
  transition_checks.json             continuity + consistency + distribution
  episode_context.json               physical context xi_e
Original Stage1 data is never modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stage0_common as sc

sys.path.insert(0, str(HERE))
from context import episode_context

STAGE1 = HERE.parent / "stage1_data"
OUT = HERE / "transitions"

STACK_SINGLE = {
    "q_t": ("qpos", 0), "v_t": ("qvel", 0), "ctrl_t": ("ctrl", 0),
    "a_t": ("qacc", 0), "qfrc_actuator_t": ("qfrc_actuator", 0),
    "qfrc_bias_t": ("qfrc_bias", 0), "qfrc_constraint_t": ("qfrc_constraint", 0),
    "actuator_force_t": ("actuator_force", 0),
    "q_next": ("qpos", 1), "v_next": ("qvel", 1),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    index = json.loads((STAGE1 / "index.json").read_text())
    meta = {e["episode_id"]: e for e in index["episodes"]}
    splits = json.loads((STAGE1 / "splits.json").read_text())
    labels = np.load(STAGE1 / "stage1_labels.npz")

    inventory = {"episodes": {}}
    checks = {"splits": {}, "issues": []}

    # ---------------- per-episode inventory + continuity ---------------------
    all_names = sorted(meta)
    for n in all_names:
        z = np.load(STAGE1 / f"{n}.npz")
        t = z["time"]
        dt = np.diff(t)
        inv = {
            "origin": meta[n]["origin"], "action_type": meta[n]["action_type"],
            "split": next(s for s, ns in splits.items() if n in ns),
            "n_steps": int(z["qpos"].shape[0]),
            "t_start": float(t[0]), "t_end": float(t[-1]),
            "dt_min": float(dt.min()), "dt_max": float(dt.max()),
            "duplicate_timestamps": int((dt == 0).sum()),
            "jumps_gt_2ms": int((dt > 2e-3).sum()),
            "quat_norm_min": float(np.linalg.norm(z["qpos"][:, 3:7], axis=1).min()),
            "quat_norm_max": float(np.linalg.norm(z["qpos"][:, 3:7], axis=1).max()),
            "nonfinite": int(sum((~np.isfinite(z[k])).sum()
                                 for k in ("qpos", "qvel", "qacc", "ctrl"))),
            "max_abs_passive": float(np.abs(z["qfrc_passive"]).max()),
            "max_abs_applied": float(np.abs(z["qfrc_applied"]).max()),
            "actuator_force_vs_ctrl": float(np.abs(z["actuator_force"] - z["ctrl"]).max()),
            "qfrc_actuator_root_max": float(np.abs(z["qfrc_actuator"][:, :6]).max()),
        }
        inventory["episodes"][n] = inv
        if inv["duplicate_timestamps"] or inv["jumps_gt_2ms"] or inv["nonfinite"]:
            checks["issues"].append(f"{n}: discontinuity/nonfinite")
        if inv["max_abs_passive"] != 0 or inv["max_abs_applied"] != 0:
            checks["issues"].append(f"{n}: passive/applied nonzero")

    # ---------------- materialize transitions + windows per split ------------
    for split, ep_names in splits.items():
        ep_names = sorted(ep_names)
        cols = {k: [] for k in STACK_SINGLE}
        m_up, dt_c, time_t, time_next = [], [], [], []
        ncon_c, nefc_c = [], []
        free_c, support_c, impact_c = [], [], []
        ep_idx_c, step_c = [], []
        windows_ep, windows_t0 = [], []
        for ei, n in enumerate(ep_names):
            z = np.load(STAGE1 / f"{n}.npz")
            T = z["qpos"].shape[0]
            for k, (field, off) in STACK_SINGLE.items():
                cols[k].append(z[field][off:T - 1 + off])
            m_up.append(z["M_upper"][:T - 1].astype(np.float32))
            dt_c.append(np.diff(z["time"]).astype(np.float32))
            time_t.append(z["time"][:T - 1])
            time_next.append(z["time"][1:])
            ncon_c.append(z["ncon"][:T - 1].astype(np.int32))
            nefc_c.append(z["nefc"][:T - 1].astype(np.int32))
            free_c.append(labels[f"{n}__free"][:T - 1])
            support_c.append(labels[f"{n}__support"][:T - 1].astype(np.int8))
            impact_c.append(labels[f"{n}__impact"][:T - 1])
            ep_idx_c.append(np.full(T - 1, ei, dtype=np.int32))
            step_c.append(np.arange(T - 1, dtype=np.int32))
            for t0 in range(T - 8):
                windows_ep.append(ei)
                windows_t0.append(t0)
        data = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
        data.update({
            "M_upper_t": np.concatenate(m_up, axis=0),
            "dt": np.concatenate(dt_c),
            "time_t": np.concatenate(time_t),
            "time_next": np.concatenate(time_next),
            "ncon_t": np.concatenate(ncon_c),
            "nefc_t": np.concatenate(nefc_c),
            "free": np.concatenate(free_c),
            "support": np.concatenate(support_c),
            "impact": np.concatenate(impact_c),
            "episode_index": np.concatenate(ep_idx_c),
            "step": np.concatenate(step_c),
            "episode_names": np.array(ep_names),
            "transition_valid": np.ones(data["q_t"].shape[0], dtype=bool),
        })
        np.savez_compressed(OUT / f"transitions_{split}.npz", **data)
        np.savez_compressed(OUT / f"windows_h8_{split}.npz",
                            episode_index=np.array(windows_ep, dtype=np.int32),
                            t0=np.array(windows_t0, dtype=np.int32),
                            episode_names=np.array(ep_names))
        n = data["q_t"].shape[0]
        checks["splits"][split] = {
            "episodes": len(ep_names),
            "transitions": n,
            "windows_h8": len(windows_ep),
            "free_transitions": int(data["free"].sum()),
            "impact_transitions": int(data["impact"].sum()),
            "support_double": int((data["support"] == 3).sum()),
            "support_single": int(((data["support"] == 1) |
                                   (data["support"] == 2)).sum()),
            "support_none": int((data["support"] == 0).sum()),
            "dt_min": float(data["dt"].min()), "dt_max": float(data["dt"].max()),
        }
        print(f"{split:5s}: {checks['splits'][split]['episodes']:2d} eps, "
              f"{n:6d} transitions, {len(windows_ep):6d} windows(h8), "
              f"free={checks['splits'][split]['free_transitions']:5d}, "
              f"impact={checks['splits'][split]['impact_transitions']:4d}")

    (OUT / "transition_inventory.json").write_text(json.dumps(inventory, indent=2))
    (OUT / "transition_checks.json").write_text(json.dumps(checks, indent=2))
    (HERE / "episode_context.json").write_text(
        json.dumps(episode_context(STAGE1), indent=2))
    total_t = sum(c["transitions"] for c in checks["splits"].values())
    print(f"\ntotal transitions={total_t}, "
          f"issues={checks['issues'] if checks['issues'] else 'NONE'}")
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
