#!/usr/bin/env python3
"""Stage1-02 formal evaluation: MLP vs Free-Dynamics PINN (lambda 0.1/1/10),
5 seeds each, on the Stage1 TEST episodes.

Metrics per run (physical units):
  Overall / Free / Contact / Impact / <=5sigma  MAE and RMSE
  Free residual normalized RMS + physical mean |r| (oracle reference = 0)
  Per-episode RMSE + macro-average RMSE
  Per-action-source macro RMSE (top_rock / arm_wave / drop / ou / hold_throw)
Plus zero-acceleration baseline (predict train mean) with identical metrics.

Outputs under <runs-dir>/results/: metrics_per_run.{csv,json},
metrics_summary.json, per_episode_rmse.csv, per_source_rmse.csv,
gradient_analysis.json, comparison.png.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import stage0_common as sc


def seg_metrics(err: np.ndarray, m: np.ndarray) -> dict:
    se = err[m] ** 2
    ae = np.abs(err[m])
    return {"MAE": float(ae.mean()), "RMSE": float(np.sqrt(se.mean()))}


def eval_pred(a_pred: np.ndarray, a_true: np.ndarray, free: np.ndarray,
              impact: np.ndarray, band: np.ndarray, s_r: np.ndarray,
              M: np.ndarray, bias: np.ndarray, act: np.ndarray) -> dict:
    err = a_pred - a_true
    bulk = band <= 5.0
    contact = (~free) & (~impact)
    out = {}
    for name, m in (("overall", np.ones(len(err), bool)), ("free", free),
                    ("contact", contact), ("impact", impact), ("le5sigma", bulk)):
        out[name] = seg_metrics(err, m)
    r_free = np.einsum("nij,nj->ni", M, a_pred) + bias - act
    if free.any():
        out["free_residual_norm_RMS"] = float(np.sqrt(((r_free[free] / s_r) ** 2).mean()))
        out["free_residual_phys_mean"] = float(np.abs(r_free[free]).mean())
    else:
        out["free_residual_norm_RMS"] = float("nan")
        out["free_residual_phys_mean"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent / "stage1_data")
    ap.add_argument("--runs-dir", type=Path,
                    default=Path(__file__).resolve().parent / "stage1_runs")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = json.loads((args.data_dir / "splits.json").read_text())
    stats = sc.load_norm_stats(args.data_dir / "norm_stats.json")
    s_r = np.asarray(stats["s_r"])
    index = json.loads((args.data_dir / "index.json").read_text())
    action_of = {e["episode_id"]: e["action_type"] for e in index["episodes"]}
    labels = np.load(args.data_dir / "stage1_labels.npz")

    test_names = splits["test"]                       # order = load order below
    eps_raw = sc.load_episodes(test_names, data_dir=args.data_dir)
    segments, ofs = [], 0
    for n in test_names:
        t = eps_raw[n]["n_steps"]
        segments.append((n, ofs, t))
        ofs += t
    eps = sc.stack(eps_raw)
    N = eps["qacc"].shape[0]

    impact = np.concatenate([labels[f"{n}__impact"] for n in test_names])
    band = np.concatenate([labels[f"{n}__band"] for n in test_names])
    free = eps["free"]
    bulk = band <= 5.0
    a_true = eps["qacc"]
    print(f"TEST: {len(test_names)} episodes, {N} samples | free={free.sum()} "
          f"impact={impact.sum()} bulk={bulk.sum()}")

    mu_a = np.asarray(stats["mu_a"], dtype=np.float32)

    def run_metrics(a_pred: np.ndarray, ep_prefix: str = "") -> dict:
        m = eval_pred(a_pred, a_true, free, impact, band, s_r,
                      eps["M"], eps["qfrc_bias"], eps["qfrc_actuator"])
        flat = {}
        for grp, d in m.items():
            if isinstance(d, dict):
                for k, v in d.items():
                    flat[f"{grp}_{k}"] = v
            else:
                flat[grp] = d
        # per-episode RMSE + macro average
        per_ep = OrderedDict()
        for n, o, t in segments:
            e = a_pred[o:o + t] - a_true[o:o + t]
            per_ep[n] = float(np.sqrt((e ** 2).mean()))
        flat[f"{ep_prefix}macro_RMSE"] = float(np.mean(list(per_ep.values())))
        # per-source macro RMSE (over episodes of that source)
        src_rmse = {}
        for src in sorted(set(action_of[n] for n in test_names)):
            vals = [per_ep[n] for n in test_names if action_of[n] == src]
            src_rmse[src] = float(np.mean(vals))
        flat[f"{ep_prefix}source_macro_RMSE"] = json.dumps(src_rmse)
        flat[f"{ep_prefix}per_episode_RMSE"] = json.dumps(per_ep)
        return flat, per_ep

    rows = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not (run_dir / "config.json").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        model.eval()
        mu_x = torch.as_tensor(stats["mu_x"], device=device)
        std_x = torch.as_tensor(stats["std_x"], device=device)
        mu_a_t = torch.as_tensor(stats["mu_a"], device=device)
        std_a_t = torch.as_tensor(stats["std_a"], device=device)
        with torch.no_grad():
            a_pred = (model((torch.as_tensor(eps["X"], device=device) - mu_x) / std_x)
                      * std_a_t + mu_a_t).cpu().numpy().astype(np.float32)
        m, per_ep = run_metrics(a_pred)
        m.update({"run": cfg["run_name"], "model": cfg["model"],
                  "lambda_dyn": cfg["lambda_dyn"], "seed": cfg["seed"],
                  "best_epoch": cfg["best_epoch"]})
        # gradient ratio stats from history
        h = json.loads((run_dir / "history.json").read_text())
        g = [(r["gnorm_phys_x_lambda"], r["gnorm_data"], r["epoch"]) for r in h
             if r.get("gnorm_phys_x_lambda") == r.get("gnorm_phys_x_lambda")
             and r.get("gnorm_data") and r["gnorm_data"] > 0]
        if g:
            ratios = [gp / gd for gp, gd, _ in g]
            early = [gp / gd for gp, gd, ep_ in g if ep_ < 15]
            at_best = [gp / gd for gp, gd, ep_ in g if ep_ <= cfg["best_epoch"] + 2]
            m["grad_ratio_early_mean"] = float(np.mean(early)) if early else float("nan")
            m["grad_ratio_at_best_mean"] = float(np.mean(at_best)) if at_best else float("nan")
            m["grad_ratio_max"] = float(np.max(ratios))
        rows.append(m)
        print(f"{cfg['run_name']:22s} MAE={m['overall_MAE']:8.2f} "
              f"RMSE={m['overall_RMSE']:8.2f} free={m['free_RMSE']:8.2f} "
              f"cont={m['contact_RMSE']:8.2f} imp={m['impact_RMSE']:9.2f} "
              f"le5={m['le5sigma_RMSE']:8.2f} freeRes={m['free_residual_norm_RMS']:.3f} "
              f"macro={m['macro_RMSE']:8.2f}")

    # zero-accel baseline
    bl, _ = run_metrics(np.broadcast_to(mu_a, a_true.shape).copy())
    bl.update({"run": "zero_accel_baseline", "model": "baseline",
               "lambda_dyn": 0.0, "seed": -1})
    rows.append(bl)
    print(f"{'zero_accel_baseline':22s} MAE={bl['overall_MAE']:8.2f} "
          f"RMSE={bl['overall_RMSE']:8.2f} free={bl['free_RMSE']:8.2f} "
          f"cont={bl['contact_RMSE']:8.2f} imp={bl['impact_RMSE']:9.2f} "
          f"le5={bl['le5sigma_RMSE']:8.2f} freeRes={bl['free_residual_norm_RMS']:.3f} "
          f"macro={bl['macro_RMSE']:8.2f}")

    out = args.runs_dir / "results"
    out.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with (out / "metrics_per_run.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k)) for k in keys) + "\n")
    (out / "metrics_per_run.json").write_text(json.dumps(rows, indent=2))

    # ---- summary mean+-std per config ---------------------------------------
    summary = []
    for key in [("mlp", 0.0), ("pinn", 0.1), ("pinn", 1.0), ("pinn", 10.0)]:
        grp = [r for r in rows if (r["model"], r["lambda_dyn"]) == key
               and r["seed"] >= 0]
        if not grp:
            continue
        s = {"model": key[0], "lambda_dyn": key[1], "n_seeds": len(grp)}
        for k in keys:
            if k in ("run", "model", "lambda_dyn", "seed", "best_epoch",
                     "per_episode_RMSE", "source_macro_RMSE"):
                continue
            vals = [r[k] for r in grp]
            s[k] = round(float(np.nanmean(vals)), 4)
            s[k + "_std"] = round(float(np.nanstd(vals)), 4)
        # per-episode RMSE mean across seeds
        pe = {}
        for n in test_names:
            vals = [json.loads(r["per_episode_RMSE"])[n] for r in grp]
            pe[n] = round(float(np.mean(vals)), 2)
        s["per_episode_RMSE_mean"] = pe
        ss = {}
        for src in sorted(set(action_of[n] for n in test_names)):
            vals = [json.loads(r["source_macro_RMSE"])[src] for r in grp]
            ss[src] = round(float(np.mean(vals)), 2)
        s["source_macro_RMSE_mean"] = ss
        summary.append(s)
    summary.append(bl)
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    hdr = ["config", "MAE", "RMSE", "free_R", "contact_R", "impact_R", "le5_R",
           "le5_MAE", "macro_R", "freeRes"]
    print(f"\n{'config':16s}" + "".join(f"{h:>11s}" for h in hdr[1:]))
    for s in summary:
        name = (s.get("run") if s["model"] == "baseline"
                else f"{s['model']}-lam{s['lambda_dyn']:g}")
        print(f"{name:16s}{s['overall_MAE']:11.2f}{s['overall_RMSE']:11.2f}"
              f"{s['free_RMSE']:11.2f}{s['contact_RMSE']:11.2f}"
              f"{s['impact_RMSE']:11.2f}{s['le5sigma_RMSE']:11.2f}"
              f"{s['le5sigma_MAE']:11.2f}{s['macro_RMSE']:11.2f}"
              f"{s['free_residual_norm_RMS']:11.3f}")

    # ---- comparison figure ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        ax = axes[0]
        eps_names = [n for n in test_names]
        width = 0.2
        colors = {"mlp-lam0": "tab:blue", "pinn-lam0.1": "tab:orange",
                  "pinn-lam1": "tab:green", "pinn-lam10": "tab:red",
                  "zero_accel_baseline": "gray"}
        for i, s in enumerate(summary):
            name = (s.get("run") if s["model"] == "baseline"
                    else f"{s['model']}-lam{s['lambda_dyn']:g}")
            pe = s.get("per_episode_RMSE_mean") or {}
            xs = np.arange(len(eps_names)) + (i - 2) * width
            vals = [pe.get(n, np.nan) for n in eps_names]
            ax.bar(xs, vals, width, label=name, color=colors.get(name))
        ax.set_xticks(np.arange(len(eps_names)))
        ax.set_xticklabels([n.replace("pd_", "").replace("hold_", "") for n in eps_names],
                           rotation=20, fontsize=8)
        ax.set_ylabel("test RMSE (mixed units)")
        ax.set_title("Per-episode RMSE (5-seed mean)")
        ax.legend(fontsize=7)
        ax = axes[1]
        lams, res_m, res_s = [], [], []
        for s in summary:
            if s["model"] == "pinn":
                lams.append(s["lambda_dyn"])
                res_m.append(s["free_residual_norm_RMS"])
                res_s.append(s["free_residual_norm_RMS_std"])
        ax.errorbar(lams, res_m, yerr=res_s, marker="o", label="PINN (fixed)")
        ax.axhline(bl["free_residual_norm_RMS"], color="gray", ls="--",
                   label="zero-accel baseline")
        mlp = next(s for s in summary if s["model"] == "mlp")
        ax.axhline(mlp["free_residual_norm_RMS"], color="tab:blue", ls=":",
                   label="MLP")
        ax.axhline(0.0, color="k", lw=0.8, label="oracle (=0)")
        ax.set_xscale("log"); ax.set_xlabel("lambda_dyn"); ax.set_ylabel("free residual nRMS")
        ax.set_title("Physics consistency vs lambda (5-seed mean±std)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=150)
        print(f"\nfigure -> {out / 'comparison.png'}")
    except Exception as exc:  # matplotlib missing etc.
        print(f"figure skipped: {exc}")

    print(f"\nwrote {out}/metrics_per_run.csv, metrics_summary.json")


if __name__ == "__main__":
    main()
