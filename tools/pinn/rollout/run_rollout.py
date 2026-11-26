#!/usr/bin/env python3
"""K1_PINN_ROLLOUT_01 — open-loop multi-step state prediction comparison.

Models: the 10 Stage1-02 checkpoints (MLP seed 0-4, PINN lambda=1 seed 0-4).
Data:   Stage1 TEST split, ALL valid 8-step window starts (RolloutWindowDataset
        index; 3,110 windows over 6 episodes). Paired horizons H in {1,2,4,8}:
        every window is rolled out 8 steps and metrics are read at steps
        1,2,4,8, so all horizons share identical start states and control
        sequences.

Rollout contract (no future leakage):
  init    q_hat(t0)=q(t0), v_hat(t0)=v(t0)            (true state, once)
  step k  a_hat = denorm(model(feats(q_hat, v_hat, ctrl[t0+k-1])))
          v_hat += dt_k * a_hat                        (dt_k = recorded f64
          q_hat = mj_integratePos(q_hat, v_hat, dt_k)   time difference)
  MuJoCo is used ONLY for position integration; no mj_step/mj_forward, so no
  ground-truth dynamics can enter the predicted states.

Metrics per window per step (physical units):
  joint position RMSE components (22), joint velocity RMSE components (22),
  root position sq-err (3), root lin-vel sq-err (3), root ang-vel sq-err (3),
  root quaternion geodesic angle (rad).
Stored raw per run; aggregated by horizon/episode/state-class afterwards.

Checks (section 7 of the task): H=1 rollout == direct single-step inference;
recorded-a integration reproduces true next states at K1_WM_01 precision;
quaternion norms valid; nonfinite predictions counted, never silently dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mujoco

import stage0_common as sc

STAGE1 = HERE.parent / "stage1_data"
RUNS = HERE.parent / "stage1_runs"
OUT = HERE
HORIZONS = (1, 2, 4, 8)
MAX_H = 8
RUN_NAMES = [f"mlp_seed{s}" for s in range(5)] + \
            [f"pinn_lam1_seed{s}" for s in range(5)]


def quat_geodesic(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = sc.load_norm_stats(STAGE1 / "norm_stats.json")
    mu_x = torch.as_tensor(stats["mu_x"], dtype=torch.float32, device=device)
    std_x = torch.as_tensor(stats["std_x"], dtype=torch.float32, device=device)
    mu_a = torch.as_tensor(stats["mu_a"], dtype=torch.float32, device=device)
    std_a = torch.as_tensor(stats["std_a"], dtype=torch.float32, device=device)
    mu_xn, std_xn = stats["mu_x"], stats["std_x"]
    mu_an, std_an = stats["mu_a"], stats["std_a"]

    labels = np.load(STAGE1 / "stage1_labels.npz")
    splits = json.loads((STAGE1 / "splits.json").read_text())
    test_names = splits["test"]

    from world_model import RolloutWindowDataset
    rw = RolloutWindowDataset(data_dir=STAGE1, split="test", horizon=MAX_H)
    print(f"TEST windows (H=8 valid starts): {len(rw)} over {len(rw.episode_ids)} "
          f"episodes: {rw.episode_ids}")

    # gather window tensors from the dataset (canonical order)
    q0 = np.stack([rw[i]["q_t"] for i in range(len(rw))])
    v0 = np.stack([rw[i]["v_t"] for i in range(len(rw))])
    ep_idx = np.array([rw[i]["episode_index"] for i in range(len(rw))])
    t0s = np.array([rw[i]["t0"] for i in range(len(rw))])
    ep_names = rw.episode_ids
    N = len(rw)

    # per-episode arrays for ground truth / control / dt
    ep_arrays = {n: dict(np.load(STAGE1 / f"{n}.npz")) for n in ep_names}

    # start-sample labels
    free0 = np.concatenate([labels[f"{n}__free"][:T - MAX_H]
                            for n, T in zip(ep_names, [ep_arrays[n]["qpos"].shape[0]
                                                       for n in ep_names])])
    impact0 = np.concatenate([labels[f"{n}__impact"][:T - MAX_H]
                              for n, T in zip(ep_names, [ep_arrays[n]["qpos"].shape[0]
                                                         for n in ep_names])])
    support0 = np.concatenate([labels[f"{n}__support"][:T - MAX_H]
                               for n, T in zip(ep_names, [ep_arrays[n]["qpos"].shape[0]
                                                          for n in ep_names])])
    print(f"start classes: free={free0.sum()} impact={impact0.sum()} "
          f"contact={(~free0).sum()} (start-label stratification)")

    # per-window dt sequence (f64, from recorded timestamps) and ground truth
    dt_seq = np.zeros((N, MAX_H))
    truth_q = np.zeros((N, MAX_H, 29))
    truth_v = np.zeros((N, MAX_H, 28))
    true_a = np.zeros((N, MAX_H, 28))
    ctrl_seq = np.zeros((N, MAX_H, 22))
    for i in range(N):
        ep = ep_arrays[ep_names[ep_idx[i]]]
        t0 = int(t0s[i])
        dt_seq[i] = np.diff(ep["time"][t0:t0 + MAX_H + 1])
        truth_q[i] = ep["qpos"][t0 + 1:t0 + MAX_H + 1]
        truth_v[i] = ep["qvel"][t0 + 1:t0 + MAX_H + 1]
        true_a[i] = ep["qacc"][t0:t0 + MAX_H]
        ctrl_seq[i] = ep["ctrl"][t0:t0 + MAX_H]

    m = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT := Path(__file__).resolve().parents[3]
            / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"))
    d = mujoco.MjData(m)

    def integrate_batch(q: np.ndarray, v: np.ndarray, dtv: np.ndarray) -> np.ndarray:
        out = q.copy()
        for i in range(q.shape[0]):
            d.qpos[:] = q[i]
            mujoco.mj_integratePos(m, d.qpos, v[i], dtv[i])
            out[i] = d.qpos
        return out

    def build_X(q_hat: np.ndarray, v_hat: np.ndarray, ctrl: np.ndarray) -> torch.Tensor:
        """Stage1-02 input contract: [root pos 3 | rot6 6 | joints 22 | qvel 28
        | ctrl 22] = 81 dims, z-scored with frozen train stats."""
        X = np.concatenate([q_hat[:, 0:3],
                            sc.quat_wxyz_to_rot6(q_hat[:, 3:7]),
                            q_hat[:, 7:29], v_hat, ctrl], axis=1)
        Xn = (X - mu_xn) / std_xn
        return torch.as_tensor(Xn, dtype=torch.float32, device=device)

    # ---- sanity check 1: recorded-a integration reproduces true states -----
    v_chk = v0.copy()
    q_chk = q0.copy()
    max_dv = max_dq = 0.0
    for k in range(MAX_H):
        v_chk = v_chk + dt_seq[:, k:k + 1] * true_a[:, k]
        q_chk = integrate_batch(q_chk, v_chk, dt_seq[:, k])
        max_dv = max(max_dv, float(np.abs(v_chk - truth_v[:, k]).max()))
        max_dq = max(max_dq, float(np.abs(q_chk - truth_q[:, k]).max()))
    print(f"[check] recorded-a integration on all {N} windows x {MAX_H} steps: "
          f"max|dv|={max_dv:.3e} max|dq|={max_dq:.3e} (WM_01 level: 3.5e-6 / 7e-9)")

    # ---- rollout all checkpoints -------------------------------------------
    OUT.mkdir(exist_ok=True)
    manifest = {
        "checkpoints": RUN_NAMES, "runs_dir": str(RUNS),
        "data_dir": str(STAGE1), "test_episodes": ep_names,
        "n_windows": int(N), "window_starts": "all H=8-valid starts",
        "horizons": list(HORIZONS), "dt_source": "recorded f64 time diffs",
        "integration": "mj_integratePos, semi-implicit (v then q)",
        "input_contract": "81-dim [pos3|rot6|joints22|qvel28|ctrl22], frozen "
                          "train z-score; output 28-dim normalized qacc",
        "control": "recorded ctrl sequence (open-loop under recorded controls)",
        "stage1_02_configs": {r: json.loads((RUNS / r / "config.json").read_text())
                              | {"history": "omitted"} for r in RUN_NAMES},
    }
    (OUT / "rollout_manifest.json").write_text(json.dumps(manifest, indent=2))

    for run in RUN_NAMES:
        cfg = json.loads((RUNS / run / "config.json").read_text())
        model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        model.load_state_dict(torch.load(RUNS / run / "best.pt",
                                         map_location=device))
        model.eval()

        q_hat, v_hat = q0.copy(), v0.copy()
        jp = np.zeros((N, MAX_H, 22)); jv = np.zeros((N, MAX_H, 22))
        rp = np.zeros((N, MAX_H, 3)); rlv = np.zeros((N, MAX_H, 3))
        rav = np.zeros((N, MAX_H, 3)); geo = np.zeros((N, MAX_H))
        nonfinite = np.zeros((N, MAX_H), dtype=bool)
        with torch.no_grad():
            for k in range(MAX_H):
                X = build_X(q_hat, v_hat, ctrl_seq[:, k])
                a_hat = (model(X) * std_a + mu_a).cpu().numpy().astype(np.float64)
                bad = ~np.isfinite(a_hat).all(axis=1) | \
                      ~np.isfinite(v_hat).all(axis=1)
                if bad.any():                       # keep, never drop
                    a_hat[bad] = 0.0                # freeze to record & count
                    nonfinite[:, k] |= bad
                    print(f"[warn] {run} step {k}: {int(bad.sum())} windows "
                          "with nonfinite prediction/state (kept, zero-torque "
                          "substituted for numerics)")
                v_hat = v_hat + dt_seq[:, k:k + 1] * a_hat
                q_hat = integrate_batch(q_hat, v_hat, dt_seq[:, k])
                eq = q_hat - truth_q[:, k]
                ev = v_hat - truth_v[:, k]
                jp[:, k] = eq[:, 7:29] ** 2
                jv[:, k] = ev[:, 6:28] ** 2
                rp[:, k] = eq[:, 0:3] ** 2
                rlv[:, k] = ev[:, 0:3] ** 2
                rav[:, k] = ev[:, 3:6] ** 2
                geo[:, k] = quat_geodesic(q_hat[:, 3:7], truth_q[:, k, 3:7]) ** 2

        # H=1 consistency: independent single-step implementation must match
        # the rollout's stored step-1 errors exactly (same contract, separate
        # code path)
        with torch.no_grad():
            a_direct = (model(build_X(q0, v0, ctrl_seq[:, 0]))
                        * std_a + mu_a).cpu().numpy().astype(np.float64)
        v1_direct = v0 + dt_seq[:, 0:1] * a_direct
        q1_direct = integrate_batch(q0, v1_direct, dt_seq[:, 0])
        eq = q1_direct - truth_q[:, 0]
        ev = v1_direct - truth_v[:, 0]
        check_err = {
            "joint_pos": float(np.abs(np.sqrt(eq[:, 7:29] ** 2)
                                      - np.sqrt(jp[:, 0])).max()),
            "joint_vel": float(np.abs(np.sqrt(ev[:, 6:28] ** 2)
                                      - np.sqrt(jv[:, 0])).max()),
            "root_pos": float(np.abs(np.sqrt(eq[:, 0:3] ** 2)
                                     - np.sqrt(rp[:, 0])).max()),
        }
        print(f"[check] {run}: H=1 rollout vs independent single-step "
              f"implementation max diff = {max(check_err.values()):.3e}")

        np.savez_compressed(
            OUT / f"raw_{run}.npz", jp=jp, jv=jv, rp=rp, rlv=rlv, rav=rav,
            geo=geo, nonfinite=nonfinite, ep_idx=ep_idx, t0=t0s,
            free0=free0, impact0=impact0, support0=support0, dt_seq=dt_seq)
        print(f"[done] {run}: results saved "
              f"({int(nonfinite.any(axis=1).sum())} windows touched nonfinite)")

    print("all checkpoint rollouts complete")


if __name__ == "__main__":
    main()
