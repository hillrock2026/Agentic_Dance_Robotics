#!/usr/bin/env python3
"""K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_01 — unified TEST evaluation.

Evaluates the 15 new runs (G3/G4/G5 x seeds 0-4) with the same metrics and
the same TEST windows as Stage1-02 / ROLLOUT_01, and merges with the existing
G1 (MLP) and G2 (Free-PINN lam=1) results for the five-group comparison.
G1/G2 rows are read from stage1_runs artifacts (no re-training, no re-rollout).

Accel metrics: overall / free_ex / contact_ex / impact RMSE + MAE (+<=5sigma),
per-dof RMSE. State metrics: open-loop rollout H=1/2/4/8 on the identical
3,110 TEST window starts (f64 dt, mj_integratePos, no future leakage).
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
NEW_RUNS = HERE / "runs"
STAGE1_RUNS = HERE.parent / "stage1_runs"
ROLL_DIR = HERE.parent / "rollout"
HORIZONS = (1, 2, 4, 8)
MAX_H = 8


def quat_geodesic(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = json.loads((STAGE1 / "splits.json").read_text())
    stats_nm = sc.load_norm_stats(STAGE1 / "norm_stats.json")
    labels = np.load(STAGE1 / "stage1_labels.npz")
    test_names = splits["test"]

    eps = sc.load_episodes(test_names, data_dir=STAGE1,
                           extra_fields=("M_upper", "time"))
    S = sc.stack(eps)
    free = S["free"].astype(bool)
    impact = np.concatenate([labels[f"{n}__impact"] for n in test_names])
    contact_ex = (~free) & (~impact)
    classes = {"free": free, "contact_exclusive": contact_ex,
               "impact": impact, "all": np.ones(len(free), bool)}

    a_true = S["qacc"]
    runs_rows = []

    new_runs = sorted([d.name for d in NEW_RUNS.iterdir()
                       if (d / "config.json").exists()])
    from train_ablation import AuxAccelMLP
    print(f"new runs: {len(new_runs)}")
    for run in new_runs:
        cfg = json.loads((NEW_RUNS / run / "config.json").read_text())
        if cfg["group"] == "g5":
            model = AuxAccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        else:
            model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        model.load_state_dict(torch.load(NEW_RUNS / run / "best.pt",
                                         map_location=device))
        model.eval()
        mu_x = torch.as_tensor(stats_nm["mu_x"], device=device)
        std_x = torch.as_tensor(stats_nm["std_x"], device=device)
        mu_a = torch.as_tensor(stats_nm["mu_a"], device=device)
        std_a = torch.as_tensor(stats_nm["std_a"], device=device)
        with torch.no_grad():
            a_pred = (model((torch.as_tensor(S["X"], device=device) - mu_x) / std_x)
                      * std_a + mu_a).cpu().numpy().astype(np.float64)
        err = a_pred - a_true
        se = err ** 2
        row = {"run": run, "group": cfg["group"], "seed": cfg["seed"],
               "best_epoch": cfg["best_epoch"]}
        for cname, m in classes.items():
            row[f"{cname}_RMSE"] = float(np.sqrt(se[m].mean()))
            row[f"{cname}_MAE"] = float(np.abs(err[m]).mean())
        band = np.abs((a_true - np.asarray(stats_nm["mu_a"])) /
                      np.asarray(stats_nm["std_a"])).max(axis=1)
        row["le5_RMSE"] = float(np.sqrt(se[band <= 5].mean()))
        # per-dof RMSE (28)
        row["per_dof_RMSE"] = json.dumps(np.sqrt(se.mean(axis=0)).round(4).tolist())
        runs_rows.append(row)
        print(f"[accel] {run:22s} RMSE={row['all_RMSE']:8.2f} "
              f"free={row['free_RMSE']:7.2f} cex={row['contact_exclusive_RMSE']:7.2f} "
              f"imp={row['impact_RMSE']:8.2f} le5={row['le5_RMSE']:7.2f}")

    # ---------------- rollout for new runs ------------------------------------
    from world_model import RolloutWindowDataset
    rw = RolloutWindowDataset(data_dir=STAGE1, split="test", horizon=MAX_H)
    q0 = np.stack([rw[i]["q_t"] for i in range(len(rw))])
    v0 = np.stack([rw[i]["v_t"] for i in range(len(rw))])
    ep_idx = np.array([rw[i]["episode_index"] for i in range(len(rw))])
    t0s = np.array([rw[i]["t0"] for i in range(len(rw))])
    ep_names = rw.episode_ids
    ep_arrays = {n: dict(np.load(STAGE1 / f"{n}.npz")) for n in ep_names}
    N = len(rw)
    dt_seq = np.zeros((N, MAX_H)); truth_q = np.zeros((N, MAX_H, 29))
    truth_v = np.zeros((N, MAX_H, 28)); ctrl_seq = np.zeros((N, MAX_H, 22))
    for i in range(N):
        ep = ep_arrays[ep_names[ep_idx[i]]]
        t0 = int(t0s[i])
        dt_seq[i] = np.diff(ep["time"][t0:t0 + MAX_H + 1])
        truth_q[i] = ep["qpos"][t0 + 1:t0 + MAX_H + 1]
        truth_v[i] = ep["qvel"][t0 + 1:t0 + MAX_H + 1]
        ctrl_seq[i] = ep["ctrl"][t0:t0 + MAX_H]
    m = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[3]
            / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"))
    d = mujoco.MjData(m)

    def integrate_batch(q, v, dtv):
        out = q.copy()
        for i in range(q.shape[0]):
            d.qpos[:] = q[i]
            mujoco.mj_integratePos(m, d.qpos, v[i], dtv[i])
            out[i] = d.qpos
        return out

    def features(qh, vh, c):
        X = np.concatenate([qh[:, 0:3], sc.quat_wxyz_to_rot6(qh[:, 3:7]),
                            qh[:, 7:29], vh, c], axis=1)
        return (X - np.asarray(stats_nm["mu_x"])) / np.asarray(stats_nm["std_x"])

    for run in new_runs:
        cfg = json.loads((NEW_RUNS / run / "config.json").read_text())
        if cfg["group"] == "g5":
            model = AuxAccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        else:
            model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        model.load_state_dict(torch.load(NEW_RUNS / run / "best.pt",
                                         map_location=device))
        model.eval()
        qh, vh = q0.copy(), v0.copy()
        jp = np.zeros((N, MAX_H, 22)); jv = np.zeros((N, MAX_H, 22))
        rp = np.zeros((N, MAX_H, 3)); rlv = np.zeros((N, MAX_H, 3))
        rav = np.zeros((N, MAX_H, 3)); geo = np.zeros((N, MAX_H))
        with torch.no_grad():
            for k in range(MAX_H):
                Xn = torch.as_tensor(features(qh, vh, ctrl_seq[:, k]),
                                     dtype=torch.float32, device=device)
                a_hat = (model(Xn) * torch.as_tensor(stats_nm["std_a"], device=device)
                         + torch.as_tensor(stats_nm["mu_a"], device=device)) \
                    .cpu().numpy().astype(np.float64)
                vh = vh + dt_seq[:, k:k + 1] * a_hat
                qh = integrate_batch(qh, vh, dt_seq[:, k])
                eq = qh - truth_q[:, k]; ev = vh - truth_v[:, k]
                jp[:, k] = eq[:, 7:29] ** 2
                jv[:, k] = ev[:, 6:28] ** 2
                rp[:, k] = eq[:, 0:3] ** 2
                rlv[:, k] = ev[:, 0:3] ** 2
                rav[:, k] = ev[:, 3:6] ** 2
                geo[:, k] = quat_geodesic(qh[:, 3:7], truth_q[:, k, 3:7]) ** 2
        np.savez_compressed(HERE / f"rollout_raw_{run}.npz", jp=jp, jv=jv, rp=rp,
                            rlv=rlv, rav=rav, geo=geo, ep_idx=ep_idx, t0=t0s)
        print(f"[rollout] {run}: done")

    # ---------------- merge with G1/G2 ----------------------------------------
    old = json.loads((STAGE1_RUNS / "results" / "metrics_per_run.json").read_text())
    for r in old:
        is_g2 = r["model"] == "pinn" and r["lambda_dyn"] == 1.0
        if (r["model"] == "mlp" or is_g2) and r["seed"] >= 0:
            g = "g1" if r["model"] == "mlp" else "g2"
            row = {"run": f"{g}_seed{r['seed']}", "group": g, "seed": r["seed"]}
            for cname in ("free", "contact_exclusive", "impact"):
                key = {"free": "free", "contact_exclusive": "contact",
                       "impact": "impact"}[cname]
                row[f"{cname}_RMSE"] = r.get(f"{key}_RMSE", float("nan"))
                row[f"{cname}_MAE"] = float("nan")
            row["le5_RMSE"] = r.get("band_le5_RMSE", float("nan"))
            runs_rows.append(row)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with (out / "accel_metrics.csv").open("w") as f:
        cols = [k for k in runs_rows[0] if k != "per_dof_RMSE"]
        f.write(",".join(cols) + "\n")
        for r in runs_rows:
            f.write(",".join(str(r.get(k)) for k in cols) + "\n")
    (out / "accel_metrics.json").write_text(json.dumps(runs_rows, indent=2,
                                                       default=str))
    print(f"\nwrote {out}/accel_metrics.{{csv,json}} ({len(runs_rows)} rows)")


if __name__ == "__main__":
    main()
