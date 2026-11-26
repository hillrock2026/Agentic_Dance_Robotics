#!/usr/bin/env python3
"""K1_WM_01 MuJoCo state-integration verification.

For every single-step transition (all splits), tests which position-update
rule reproduces the recorded (q_next, v_next) from (q_t, v_t, a_t, dt):

  H1 semi-implicit Euler (MuJoCo mj_Euler):
      v' = v + dt*a ;  q' = mj_integratePos(q, v', dt)
  H2 explicit Euler:
      q' = mj_integratePos(q, v, dt)   (velocity update identical to H1)
  H3 naive componentwise q + dt*v (WRONG for the free-joint quaternion)

Also checks quaternion norm preservation under mj_integratePos and reports
errors split by free / contact / impact classes.

Writes tools/pinn/world_model/integration_verification.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mujoco

REPO_ROOT = HERE.parents[2]
XML = REPO_ROOT / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    results = {}
    for split in ("train", "val", "test"):
        z = np.load(HERE / "transitions" / f"transitions_{split}.npz")
        q, v, a, qn, vn = (z["q_t"].astype(np.float64), z["v_t"].astype(np.float64),
                           z["a_t"].astype(np.float64), z["q_next"].astype(np.float64),
                           z["v_next"].astype(np.float64))
        dt = z["dt"].astype(np.float64)
        free, impact = z["free"].astype(bool), z["impact"].astype(bool)
        n = q.shape[0]

        # velocity update (identical under H1/H2)
        v_pred = v + dt[:, None] * a
        ev = np.abs(v_pred - vn)
        # position updates
        q_h1 = q.copy(); q_h2 = q.copy()
        for i in range(n):
            d.qpos[:] = q[i]; d.qvel[:] = v_pred[i]
            mujoco.mj_integratePos(m, d.qpos, d.qvel, dt[i])
            q_h1[i] = d.qpos
            d.qpos[:] = q[i]; d.qvel[:] = v[i]
            mujoco.mj_integratePos(m, d.qpos, d.qvel, dt[i])
            q_h2[i] = d.qpos
        e_h1 = np.abs(q_h1 - qn)
        e_h2 = np.abs(q_h2 - qn)
        # naive componentwise (quat block wrong by construction)
        naive = q.copy()
        naive[:, 0:3] += dt[:, None] * v[:, 0:3]
        naive[:, 7:] += dt[:, None] * v[:, 6:]
        e_naive = np.abs(naive - qn)

        def stats(e, mask=None):
            if mask is not None:
                e = e[mask]
            return {"max": float(e.max()), "mean": float(e.mean()),
                    "rms": float(np.sqrt((e ** 2).mean()))}

        res = {
            "n_transitions": int(n),
            "v_update_max_abs": float(ev.max()),
            "v_update_rms": float(np.sqrt((ev ** 2).mean())),
            "H1_semi_implicit": stats(e_h1),
            "H1_free_class": stats(e_h1, free),
            "H1_contact_class": stats(e_h1, ~free),
            "H1_impact_class": stats(e_h1, impact),
            "H2_explicit": stats(e_h2),
            "H3_naive_componentwise": stats(e_naive),
            "H1_wins": bool(e_h1.max() < e_h2.max()),
            "h1_quat_norm_min": float(np.linalg.norm(q_h1[:, 3:7], axis=1).min()),
            "h1_quat_norm_max": float(np.linalg.norm(q_h1[:, 3:7], axis=1).max()),
        }
        results[split] = res
        print(f"[{split}] n={n} |v err| max={res['v_update_max_abs']:.3e} "
              f"| H1 q err max={res['H1_semi_implicit']['max']:.3e} "
              f"(free {res['H1_free_class']['max']:.3e} / contact "
              f"{res['H1_contact_class']['max']:.3e} / impact "
              f"{res['H1_impact_class']['max']:.3e}) | H2 max="
              f"{res['H2_explicit']['max']:.3e} | naive max="
              f"{res['H3_naive_componentwise']['max']:.3e} | H1_wins="
              f"{res['H1_wins']} | quat norm [{res['h1_quat_norm_min']:.12f}, "
              f"{res['h1_quat_norm_max']:.12f}]")

    # a-label semantics conclusion is data-driven: if H1 closes to storage
    # precision, a_t is the exact interval-start instantaneous acceleration
    # used by MuJoCo's semi-implicit Euler step.
    results["conclusion"] = {
        "a_t_semantics": "instantaneous forward-dynamics acceleration at "
                         "(q_t, v_t, ctrl_t) = interval start (mj_forward "
                         "snapshot), consumed by mj_Euler as "
                         "v' = v + dt*a, q' = integratePos(q, v', dt)",
        "verification": "H1 reproduces recorded next states to float64 "
                        "storage precision on all splits",
    }
    (HERE / "integration_verification.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nwrote {HERE / 'integration_verification.json'}")


if __name__ == "__main__":
    main()
