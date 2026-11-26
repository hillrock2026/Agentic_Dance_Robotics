#!/usr/bin/env python3
"""Stage0-D Task E: unified evaluation of all runs on the test episodes.

Metrics are computed in physical units on the FULL test set, plus a breakdown
into free/constraint and acceleration-magnitude bands (normalized by train
s_d). Physics residuals:
  free residual    r_free = M a_pred + bias - act        (PINN training target)
  contact residual r_full = r_free - qfrc_constraint      (diagnostic only)
Both reported as normalized RMS (frozen train s_r) and physical mean |r|.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

import stage0_common as sc

GROUPS = {
    "base_lin": slice(0, 3), "base_ang": slice(3, 6),
    "joints": slice(6, 28),
}


def evaluate_run(run_dir: Path, eps: dict, stats, device) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    mu_x = sc.to_torch(stats["mu_x"], device)
    std_x = sc.to_torch(stats["std_x"], device)
    mu_a = sc.to_torch(stats["mu_a"], device)
    std_a = sc.to_torch(stats["std_a"], device)
    s_r = np.asarray(stats["s_r"])

    X = sc.to_torch(eps["X"], device)
    with torch.no_grad():
        a_pred = (model((X - mu_x) / std_x) * std_a + mu_a).cpu().numpy()
    a_true = eps["qacc"]
    free = eps["free"]
    M, bias, act, con = eps["M"], eps["qfrc_bias"], eps["qfrc_actuator"], \
        eps["qfrc_constraint"]

    err = a_pred - a_true
    se = (err ** 2)

    def rmse(m):
        return float(np.sqrt(se[m].mean())) if m.any() else float("nan")

    all_m = np.ones(len(err), bool)
    metrics = {
        "overall_MAE": float(np.abs(err).mean()),
        "overall_RMSE": rmse(all_m),
        "base_lin_RMSE": float(np.sqrt(se[:, 0:3].mean())),
        "base_ang_RMSE": float(np.sqrt(se[:, 3:6].mean())),
        "joints_RMSE": float(np.sqrt(se[:, 6:28].mean())),
        "free_RMSE": rmse(free),
        "contact_RMSE": rmse(~free),
    }

    # residuals (physical), per-sample 28-dim
    r_free = (np.einsum("nij,nj->ni", M, a_pred) + bias - act)
    r_full = r_free - con
    metrics["free_residual_norm_RMS"] = float(
        np.sqrt(((r_free[free] / s_r) ** 2).mean())) if free.any() else float("nan")
    metrics["free_residual_phys_mean"] = float(np.abs(r_free[free]).mean())
    metrics["contact_residual_norm_RMS"] = float(
        np.sqrt(((r_full[~free] / s_r) ** 2).mean())) if (~free).any() else float("nan")
    metrics["contact_residual_phys_mean"] = float(np.abs(r_full[~free]).mean())

    # magnitude bands (normalized by train stats)
    band = np.abs((a_true - stats["mu_a"]) / stats["std_a"]).max(axis=1)
    metrics["band_le5_RMSE"] = rmse(band < 5)
    metrics["band_5_10_RMSE"] = rmse((band >= 5) & (band <= 10))
    metrics["band_gt10_RMSE"] = rmse(band > 10)
    metrics["n_test"] = int(len(err))
    metrics["n_free"] = int(free.sum())

    np.savez_compressed(run_dir / "test_preds.npz", a_pred=a_pred.astype(np.float32),
                        a_true=a_true, free=free,
                        a_zero=stats["mu_a"][None, :].astype(np.float32))
    return metrics


def baseline_metrics(eps: dict, stats) -> dict:
    """Zero-acceleration predictor (predict train mean)."""
    a_true, free = eps["qacc"], eps["free"]
    a_zero = np.asarray(stats["mu_a"], dtype=np.float32)[None, :]
    err = a_zero - a_true
    se = err ** 2
    band = np.abs((a_true - stats["mu_a"]) / stats["std_a"]).max(axis=1)
    r_free = (np.einsum("nij,nj->ni", eps["M"], np.broadcast_to(a_zero, a_true.shape))
              + eps["qfrc_bias"] - eps["qfrc_actuator"])
    r_full = r_free - eps["qfrc_constraint"]
    s_r = np.asarray(stats["s_r"])
    out = {
        "overall_MAE": float(np.abs(err).mean()),
        "overall_RMSE": float(np.sqrt(se.mean())),
        "base_lin_RMSE": float(np.sqrt(se[:, 0:3].mean())),
        "base_ang_RMSE": float(np.sqrt(se[:, 3:6].mean())),
        "joints_RMSE": float(np.sqrt(se[:, 6:28].mean())),
        "free_RMSE": float(np.sqrt(se[free].mean())) if free.any() else float("nan"),
        "contact_RMSE": float(np.sqrt(se[~free].mean())),
        "free_residual_norm_RMS": float(np.sqrt(((r_free[free] / s_r) ** 2).mean())),
        "free_residual_phys_mean": float(np.abs(r_free[free]).mean()),
        "contact_residual_norm_RMS": float(np.sqrt(((r_full[~free] / s_r) ** 2).mean())),
        "contact_residual_phys_mean": float(np.abs(r_full[~free]).mean()),
        "band_le5_RMSE": float(np.sqrt(se[band < 5].mean())),
        "band_5_10_RMSE": float(np.sqrt(se[(band >= 5) & (band <= 10)].mean())),
        "band_gt10_RMSE": float(np.sqrt(se[band > 10].mean())),
        "n_test": int(len(a_true)), "n_free": int(free.sum()),
    }
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=sc.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=sc.DATA_DIR)
    args = ap.parse_args()
    runs_dir = args.runs_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = json.loads((args.data_dir / "splits.json").read_text())
    stats = sc.load_norm_stats(args.data_dir / "norm_stats.json")
    eps_raw = sc.load_episodes(splits["test"], data_dir=args.data_dir)
    eps = sc.stack(eps_raw)

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not (run_dir / "config.json").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        if "_" in cfg["run_name"] and re.search(r"_seed\d+$", cfg["run_name"]) is None:
            continue    # skip tagged sweep runs
        m = evaluate_run(run_dir, eps, stats, device)
        m["run"] = cfg["run_name"]
        m["model"] = cfg["model"]
        m["lambda_dyn"] = cfg["lambda_dyn"]
        m["seed"] = cfg["seed"]
        rows.append(m)
        print(f"{cfg['run_name']:22s} MAE={m['overall_MAE']:9.3f} "
              f"RMSE={m['overall_RMSE']:9.3f} freeRMSE={m['free_RMSE']:9.3f} "
              f"contRMSE={m['contact_RMSE']:9.3f} freeRes={m['free_residual_norm_RMS']:8.3f}")

    bl = baseline_metrics(eps, stats)
    bl.update({"run": "zero_accel_baseline", "model": "baseline", "lambda_dyn": 0.0,
               "seed": -1})
    rows.append(bl)
    print(f"{'zero_accel_baseline':22s} MAE={bl['overall_MAE']:9.3f} "
          f"RMSE={bl['overall_RMSE']:9.3f} freeRMSE={bl['free_RMSE']:9.3f} "
          f"contRMSE={bl['contact_RMSE']:9.3f} freeRes={bl['free_residual_norm_RMS']:8.3f}")

    out_dir = runs_dir / "results"
    out_dir.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "metrics_per_run.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    (out_dir / "metrics_per_run.json").write_text(json.dumps(rows, indent=2))

    # ---- per-config summary: mean +- std over seeds (baseline kept as-is) ----
    def agg(model_key):
        grp = [r for r in rows if (r["model"], r["lambda_dyn"]) == model_key
               and r["seed"] >= 0]
        if not grp:
            return None
        out = {"model": model_key[0], "lambda_dyn": model_key[1],
               "n_seeds": len(grp)}
        for k in keys:
            if k in ("run", "model", "lambda_dyn", "seed"):
                continue
            vals = [r[k] for r in grp]
            out[k] = round(float(np.mean(vals)), 4)
            out[k + "_std"] = round(float(np.std(vals)), 4)
        return out

    summary = []
    seen = []
    for r in rows:
        key = (r["model"], r["lambda_dyn"])
        if r["seed"] >= 0 and key not in seen:
            seen.append(key)
            summary.append(agg(key))
    summary.append(bl)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    hdr = ["config", "MAE", "RMSE", "base_ang", "joints", "free", "contact",
           "le5sig", "freeRes"]
    print(f"\n{'config':20s} " + " ".join(f"{h:>10s}" for h in hdr))
    for s in summary:
        name = (s["run"] if s["model"] == "baseline"
                else f"{s['model']}-lam{s['lambda_dyn']:g}")
        print(f"{name:20s} "
              f"{s['overall_MAE']:10.2f} {s['overall_RMSE']:10.2f} "
              f"{s['base_ang_RMSE']:10.2f} {s['joints_RMSE']:10.2f} "
              f"{s['free_RMSE']:10.2f} {s['contact_RMSE']:10.2f} "
              f"{s['band_le5_RMSE']:10.2f} {s['free_residual_norm_RMS']:10.3f}")
    print(f"\nwrote {out_dir/'metrics_per_run.csv'} ({len(rows)} rows) "
          f"and metrics_summary.json ({len(summary)} configs)")


if __name__ == "__main__":
    main()
