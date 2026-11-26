#!/usr/bin/env python3
"""K1_PINN_ROLLOUT_01 aggregation: per-seed / per-episode / per-horizon /
per-state-class tables, paired PINN-MLP diffs, error-growth figure.

Reads raw_<run>.npz produced by run_rollout.py; writes metrics tables (CSV +
JSON) and comparison.png under tools/pinn/rollout/results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_NAMES = [f"mlp_seed{s}" for s in range(5)] + \
            [f"pinn_lam1_seed{s}" for s in range(5)]
HORIZONS = (1, 2, 4, 8)
MAX_H = 8
METRIC_GROUPS = ("joint_pos", "joint_vel", "root_pos", "root_lin_vel",
                 "root_ang_vel", "root_geodesic")


_RAW_KEY = {"joint_pos": "jp", "joint_vel": "jv", "root_pos": "rp",
            "root_lin_vel": "rlv", "root_ang_vel": "rav",
            "root_geodesic": "geo"}


def rmse(raw: dict, group: str, h: int, mask=None) -> float:
    step = h - 1
    e2 = raw[_RAW_KEY[group]][:, step]
    if mask is not None:
        e2 = e2[mask]
    return float(np.sqrt(e2.mean()))


def main() -> None:
    runs = {r: dict(np.load(HERE / f"raw_{r}.npz")) for r in RUN_NAMES}
    ref = runs[RUN_NAMES[0]]
    ep_idx, ep_names = ref["ep_idx"], [str(x) for x in
                                       json.loads((HERE / "rollout_manifest.json")
                                                  .read_text())["test_episodes"]]
    free0, impact0 = ref["free0"].astype(bool), ref["impact0"].astype(bool)
    classes = {"free": free0, "contact": ~free0, "impact": impact0,
               "all": np.ones(len(free0), bool)}

    # ---------------- per-run aggregate table --------------------------------
    rows = []
    for r in RUN_NAMES:
        raw = runs[r]
        for h in HORIZONS:
            for cname, cmask in classes.items():
                row = {"run": r, "seed": int(r.rsplit("seed", 1)[1]),
                       "model": "mlp" if r.startswith("mlp") else "pinn_lam1",
                       "horizon": h, "class": cname, "n_windows": int(cmask.sum())}
                for g in METRIC_GROUPS:
                    row[g + "_RMSE"] = rmse(raw, g, h, cmask)
                rows.append(row)

    # per-episode, per-horizon (all-class) for each run
    for r in RUN_NAMES:
        raw = runs[r]
        for h in HORIZONS:
            row = next(x for x in rows if x["run"] == r and x["horizon"] == h
                       and x["class"] == "all")
            for ei, n in enumerate(ep_names):
                m = ep_idx == ei
                row[f"ep:{n}"] = rmse(raw, "joint_pos", h, m)

    # ---------------- summary: mean+-std over seeds, paired diffs ------------
    summary = []
    for h in HORIZONS:
        for cname, cmask in classes.items():
            entry = {"horizon": h, "class": cname}
            for model in ("mlp", "pinn_lam1"):
                gr = [r for r in rows if r["model"] == model and r["horizon"] == h
                      and r["class"] == cname]
                for g in METRIC_GROUPS:
                    vals = [x[g + "_RMSE"] for x in gr]
                    entry[f"{model}_{g}_RMSE_mean"] = float(np.mean(vals))
                    entry[f"{model}_{g}_RMSE_std"] = float(np.std(vals))
                entry[f"{model}_seeds"] = {x["seed"]: x[g + "_RMSE"]
                                           for x in gr for g in ["joint_pos"]}
            # paired per-seed diffs
            for g in METRIC_GROUPS:
                d = []
                for seed in range(5):
                    mlp = next(x for x in rows if x["model"] == "mlp"
                               and x["seed"] == seed and x["horizon"] == h
                               and x["class"] == cname)
                    pin = next(x for x in rows if x["model"] == "pinn_lam1"
                               and x["seed"] == seed and x["horizon"] == h
                               and x["class"] == cname)
                    d.append(pin[g + "_RMSE"] - mlp[g + "_RMSE"])
                entry[f"paired_diff_{g}_RMSE"] = [round(x, 6) for x in d]
                entry[f"paired_diff_{g}_wins"] = int(sum(1 for x in d if x < 0))
            summary.append(entry)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with (out / "metrics_by_horizon.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    (out / "metrics_by_horizon.json").write_text(json.dumps(rows, indent=2))
    (out / "summary_paired.json").write_text(json.dumps(summary, indent=2))

    # ---------------- console tables ------------------------------------------
    def fmt(model, h, cname, g):
        e = next(x for x in summary if x["horizon"] == h and x["class"] == cname)
        return (e[f"{model}_{g}_RMSE_mean"], e[f"{model}_{g}_RMSE_std"],
                e[f"paired_diff_{g}_wins"])

    print("=== joint position RMSE (rad), all windows ===")
    print(f"{'H':>3s} {'MLP':>14s} {'PINN-1':>14s} {'paired wins':>12s}")
    for h in HORIZONS:
        m, ms, w = fmt("mlp", h, "all", "joint_pos")
        p, ps, _ = fmt("pinn_lam1", h, "all", "joint_pos")
        print(f"{h:>3d} {m:7.4f}±{ms:.4f} {p:7.4f}±{ps:.4f} {w:>8d}/5")
    print("\n=== joint velocity RMSE (rad/s), all windows ===")
    for h in HORIZONS:
        m, ms, w = fmt("mlp", h, "all", "joint_vel")
        p, ps, _ = fmt("pinn_lam1", h, "all", "joint_vel")
        print(f"{h:>3d} {m:7.4f}±{ms:.4f} {p:7.4f}±{ps:.4f} {w:>8d}/5")
    for g, unit in (("root_pos", "m"), ("root_lin_vel", "m/s"),
                    ("root_ang_vel", "rad/s"), ("root_geodesic", "rad")):
        print(f"\n=== {g} RMSE ({unit}), all windows ===")
        for h in HORIZONS:
            m, ms, w = fmt("mlp", h, "all", g)
            p, ps, _ = fmt("pinn_lam1", h, "all", g)
            print(f"{h:>3d} {m:9.5f}±{ms:.5f} {p:9.5f}±{ps:.5f} {w:>8d}/5")

    print("\n=== start-class stratification (joint pos / joint vel RMSE) ===")
    for cname in ("free", "contact", "impact"):
        for h in HORIZONS:
            m, _, _ = fmt("mlp", h, cname, "joint_pos")
            p, _, w = fmt("pinn_lam1", h, cname, "joint_pos")
            mv, _, wv = fmt("mlp", h, cname, "joint_vel")
            pv, _, _ = fmt("pinn_lam1", h, cname, "joint_vel")
            print(f" {cname:8s} H{h}: jp MLP={m:7.4f} PINN={p:7.4f} ({w}/5) | "
                  f"jv MLP={mv:8.4f} PINN={pv:8.4f} ({wv}/5)")

    print("\n=== per-episode joint pos RMSE at H=8 (5-seed mean) ===")
    for n in ep_names:
        key = f"ep:{n}"
        m = np.mean([r[key] for r in rows if r["model"] == "mlp"
                     and r["horizon"] == 8 and r["class"] == "all"])
        p = np.mean([r[key] for r in rows if r["model"] == "pinn_lam1"
                     and r["horizon"] == 8 and r["class"] == "all"])
        b_bl = None
        print(f" {n:20s} MLP={m:8.4f} PINN1={p:8.4f} diff={p - m:+.4f}")

    # ---------------- figure ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = list(range(1, MAX_H + 1))

    def curve(model, g, cname="all"):
        vals = []
        for h in range(1, MAX_H + 1):
            gr = [r for r in rows if r["model"] == model and r["horizon"] == h
                  and r["class"] == cname]
            vals.append(np.mean([x[g + "_RMSE"] for x in gr]) if h in HORIZONS
                        else np.nan)
        # fill intermediate horizons (2,3,5..8) from per-run rows at that h? we
        # only stored h in HORIZONS for summary; use raw aggregation here:
        return vals

    # direct full-step curves from raw
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, g in zip(axes.ravel(), METRIC_GROUPS):
        for model, color in (("mlp", "tab:blue"), ("pinn_lam1", "tab:red")):
            ys = [rmse(runs[next(r for r in RUN_NAMES if r.startswith(model))],
                       g, h) for h in steps]
            seed_curves = []
            for s in range(5):
                rname = next(r for r in RUN_NAMES
                             if r.startswith(model) and r.endswith(f"seed{s}"))
                seed_curves.append([rmse(runs[rname], g, h) for h in steps])
            ys = np.mean(seed_curves, axis=0)
            ys_std = np.std(seed_curves, axis=0)
            ax.plot(steps, ys, marker="o", color=color, label=model)
            ax.fill_between(steps, np.array(ys) - ys_std,
                            np.array(ys) + ys_std, color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("horizon (steps, 1 ms each)")
        ax.set_ylabel(f"{g} RMSE")
        ax.set_title(g)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("K1 open-loop rollout: error growth (5-seed mean±std, TEST)")
    fig.tight_layout()
    fig.savefig(out / "error_growth.png", dpi=150)
    print(f"\nfigure -> {out / 'error_growth.png'}")
    print(f"wrote {out}/metrics_by_horizon.csv, summary_paired.json")


if __name__ == "__main__":
    main()
