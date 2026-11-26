#!/usr/bin/env python3
"""K1_PINN_ROLLOUT_01_MINIMAL_CORRECTION_01 — re-aggregation with mutually
exclusive start-state classes.

Fixes the original aggregation bug: Contact was defined as ~free, which mixed
Impact-start windows into Contact. Corrected classes (start-label based,
mutually exclusive, covering all windows):
  Free              = free0
  Contact-exclusive = ~free0 & ~impact0
  Impact            = impact0
Recomputes all metrics from the stored per-window per-step errors
(raw_<run>.npz); no checkpoints are re-run, no training is touched.

Also recomputes:
  - free-domain joint-velocity error growth (H1->H8) per model and the
    per-horizon relative advantage
  - per-seed paired results for every root metric (free-start and all)
Outputs rollout_corrected_metrics.json + corrected_metrics.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_NAMES = [f"mlp_seed{s}" for s in range(5)] + \
            [f"pinn_lam1_seed{s}" for s in range(5)]
HORIZONS = (1, 2, 4, 8)
RAW_KEY = {"joint_pos": "jp", "joint_vel": "jv", "root_pos": "rp",
           "root_lin_vel": "rlv", "root_ang_vel": "rav",
           "root_geodesic": "geo"}
METRICS = list(RAW_KEY)


def rmse(raw, key, h, mask):
    e2 = raw[key][:, h - 1]
    if mask is not None:
        e2 = e2[mask]
    return float(np.sqrt(e2.mean()))


def main() -> None:
    runs = {r: dict(np.load(HERE / f"raw_{r}.npz")) for r in RUN_NAMES}
    ref = runs[RUN_NAMES[0]]
    free0 = ref["free0"].astype(bool)
    impact0 = ref["impact0"].astype(bool)
    n = len(free0)
    contact_ex = ~free0 & ~impact0
    overlap = int((free0 & impact0).sum())
    counts = {"free": int(free0.sum()), "contact_exclusive": int(contact_ex.sum()),
              "impact": int(impact0.sum()), "total": n,
              "free_impact_overlap": overlap}
    print(f"class counts: {counts}")
    assert counts["free"] + counts["contact_exclusive"] + counts["impact"] == n

    ep_idx = ref["ep_idx"]
    manifest = json.loads((HERE / "rollout_manifest.json").read_text())
    ep_names = manifest["test_episodes"]

    # ---------------- full re-aggregation ------------------------------------
    long_rows = []
    per_seed = {}
    for r in RUN_NAMES:
        raw = runs[r]
        seed = int(r.rsplit("seed", 1)[1])
        model = "mlp" if r.startswith("mlp") else "pinn_lam1"
        per_seed[r] = {"model": model, "seed": seed}
        for cname, cmask in (("free", free0), ("contact_exclusive", contact_ex),
                             ("impact", impact0), ("all", np.ones(n, bool))):
            for h in HORIZONS:
                row = {"run": r, "model": model, "seed": seed, "horizon": h,
                       "class": cname, "n_windows": int(cmask.sum())}
                for met in METRICS:
                    row[met + "_RMSE"] = rmse(raw, RAW_KEY[met], h, cmask)
                long_rows.append(row)
                per_seed[r].setdefault(cname, {})[h] = {
                    m: row[m + "_RMSE"] for m in METRICS}
        # per-episode joint-pos RMSE at each horizon (all class)
        per_seed[r]["per_episode_joint_pos"] = {
            h: {ep_names[ei]: rmse(raw, "jp", h, ep_idx == ei)
                for ei in range(len(ep_names))} for h in HORIZONS}

    # ---------------- summary: seed-mean +- std + paired wins ----------------
    summary = []
    for cname in ("free", "contact_exclusive", "impact", "all"):
        for h in HORIZONS:
            entry = {"class": cname, "horizon": h,
                     "n_windows": {"free": counts["free"],
                                   "contact_exclusive": counts["contact_exclusive"],
                                   "impact": counts["impact"],
                                   "all": n}[cname]}
            for model in ("mlp", "pinn_lam1"):
                gr = [x for x in long_rows if x["model"] == model
                      and x["horizon"] == h and x["class"] == cname]
                for met in METRICS:
                    vals = [x[met + "_RMSE"] for x in gr]
                    entry[f"{model}_{met}_RMSE_mean"] = float(np.mean(vals))
                    entry[f"{model}_{met}_RMSE_std"] = float(np.std(vals))
                    entry[f"{model}_{met}_per_seed"] = [round(v, 6) for v in vals]
            for met in METRICS:
                d = [next(x for x in long_rows if x["run"] ==
                          f"pinn_lam1_seed{s}" and x["horizon"] == h
                          and x["class"] == cname)[met + "_RMSE"]
                     - next(x for x in long_rows if x["run"] ==
                            f"mlp_seed{s}" and x["horizon"] == h
                            and x["class"] == cname)[met + "_RMSE"]
                     for s in range(5)]
                entry[f"paired_diff_{met}"] = [round(x, 6) for x in d]
                entry[f"paired_wins_{met}"] = int(sum(1 for x in d if x < 0))
            summary.append(entry)

    # ---------------- free-domain error growth --------------------------------
    free_growth = {}
    for model in ("mlp", "pinn_lam1"):
        vals = {h: next(x[f"{model}_joint_vel_RMSE_mean"] for x in summary
                        if x["class"] == "free" and x["horizon"] == h)
                for h in HORIZONS}
        free_growth[model] = {
            "jv_RMSE_by_horizon": {str(h): vals[h] for h in HORIZONS},
            "growth_H1_to_H8": vals[8] / vals[1],
        }
    rel_adv = {h: (next(x["mlp_joint_vel_RMSE_mean"] for x in summary
                        if x["class"] == "free" and x["horizon"] == h)
                   - next(x["pinn_lam1_joint_vel_RMSE_mean"] for x in summary
                          if x["class"] == "free" and x["horizon"] == h))
               / next(x["mlp_joint_vel_RMSE_mean"] for x in summary
                      if x["class"] == "free" and x["horizon"] == h)
               for h in HORIZONS}

    # ---------------- per-seed root metrics, free vs all, H=8 (and all H) ----
    root_by_seed = {}
    for model in ("mlp", "pinn_lam1"):
        for s in range(5):
            r = f"{model}_seed{s}"
            root_by_seed[r] = {
                cname: {h: {met: per_seed[r][cname][h][met]
                            for met in ("root_pos", "root_lin_vel",
                                        "root_ang_vel", "root_geodesic")}
                        for h in HORIZONS}
                for cname in ("free", "all")}

    out = {
        "class_counts": counts,
        "expected_counts_check": {
            "expected": {"free": 1024, "contact_exclusive": 2036, "impact": 50},
            "match": counts["free"] == 1024 and counts["contact_exclusive"] == 2036
                     and counts["impact"] == 50 and overlap == 0,
        },
        "per_run": long_rows,
        "summary": summary,
        "free_domain_growth": {"ratios": free_growth,
                               "relative_advantage_by_horizon": {
                                   str(h): round(v, 6) for h, v in rel_adv.items()}},
        "root_by_seed": root_by_seed,
        "provenance": {
            "source": "raw_<run>.npz per-window per-step errors from "
                      "K1_PINN_ROLLOUT_01 (no re-run, no retraining)",
            "class_definition": "start-label based; free = frozen Free-Dynamics "
                                "mask at t0; impact = original impact label at "
                                "t0; contact_exclusive = ~free & ~impact",
        },
    }
    (HERE / "rollout_corrected_metrics.json").write_text(json.dumps(out, indent=2))
    keys = list(long_rows[0].keys())
    with (HERE / "corrected_metrics.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in long_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---------------- console check tables ------------------------------------
    print("\n=== H=8 joint velocity RMSE (rad/s) ===")
    for cname in ("free", "contact_exclusive", "impact"):
        e = next(x for x in summary if x["class"] == cname and x["horizon"] == 8)
        print(f" {cname:18s} MLP={e['mlp_joint_vel_RMSE_mean']:.6f} "
              f"PINN={e['pinn_lam1_joint_vel_RMSE_mean']:.6f} "
              f"wins={e['paired_wins_joint_vel']}/5")
    print("\n=== free-domain joint velocity: growth and relative advantage ===")
    for model, g in free_growth.items():
        print(f" {model:10s} growth H1->H8 = {g['growth_H1_to_H8']:.2f}x "
              f"({g['jv_RMSE_by_horizon']['1']:.4f} -> "
              f"{g['jv_RMSE_by_horizon']['8']:.4f})")
    print(" relative advantage (MLP-PINN)/MLP by horizon:",
          {h: f"{100 * v:.2f}%" for h, v in rel_adv.items()})
    print("\n=== root metrics paired wins (per horizon) ===")
    for met in ("root_pos", "root_lin_vel", "root_ang_vel", "root_geodesic"):
        line = f" {met:14s}: " + "  ".join(
            f"H{h} free={next(x for x in summary if x['class'] == 'free' and x['horizon'] == h)['paired_wins_' + met]}/5"
            f"/all={next(x for x in summary if x['class'] == 'all' and x['horizon'] == h)['paired_wins_' + met]}/5"
            for h in HORIZONS)
        print(line)
    print(f"\nexpected-count check: {out['expected_counts_check']}")


if __name__ == "__main__":
    main()
