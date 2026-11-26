#!/usr/bin/env python3
"""Stage1-01 dataset build: splits, train-only normalization, per-sample
state labels, integrity checks.

Outputs (all under tools/pinn/stage1_data/):
  splits.json        episode-level split (motion-disjoint, all state types
                     represented in test/val)
  norm_stats.json    train-only statistics (+ frozen 10-sigma filter rule)
  stage1_labels.npz  per-episode per-sample labels:
                     <ep>__free / __actuated_free / __zero_ctrl_free /
                     __support (0 none,1 singleL,2 singleR,3 double) /
                     __transition / __impact / __band / __max_ctrl
                     Support labels are GEOMETRIC (foot-box corner clearance
                     <= 2 mm) for ALL episodes, because Stage0 stored
                     foot_l/foot_r are corrupted by a collector bug; on
                     Stage1 episodes (fixed collector) the physical flags are
                     cross-validated against the geometric labels.
  integrity.json     NaN/Inf, timestamp monotonicity, snapshot-consistency
                     residual closure, field cross-checks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import stage0_common as sc
from analyze_stage1_coverage import reconstruct_support

S1 = REPO_ROOT = Path(__file__).resolve().parents[2] / "tools/pinn/stage1_data"

SPLITS = {
    "test": ["pd_top_rock", "pd_top_rock_s76", "pd_top_rock_s77",
             "drop_d0", "ou_o0", "hold_throw_h0"],
    "val": ["pd_arm_wave", "drop_d1", "ou_o1", "hold_throw_h1"],
    "train": ["pd_popping_s30", "pd_popping_s31", "pd_popping_s78",
              "pd_arm_wave_usr_s40", "pd_arm_wave_usr_s41", "pd_arm_wave_usr_s79",
              "drop_d2", "drop_d3", "drop_d4", "drop_d5", "drop_d6", "drop_d7",
              "ou_o2", "ou_o3", "ou_o4", "ou_o5",
              "hold_throw_h2", "hold_throw_h3", "hold_throw_h4", "hold_throw_h5"],
}

FEAT_FLOOR = 1e-3
EPS_A_FLOOR = 1e-3
SR_FLOOR = 1e-3
FILTER_SIGMA = 10.0
IMPACT_SIGMA = 10.0
IMPACT_DCON_N = 100.0


def main() -> None:
    index = json.loads((S1 / "index.json").read_text())
    names = sorted(e["episode_id"] for e in index["episodes"])
    meta = {e["episode_id"]: e for e in index["episodes"]}
    assert set(sum(SPLITS.values(), [])) == set(names), "split must cover all episodes"
    eps = sc.load_episodes(names, data_dir=S1,
                           extra_fields=("qpos", "qvel", "ncon", "nefc", "time",
                                         "actuator_force", "qfrc_passive",
                                         "qfrc_applied", "residual_closure",
                                         "foot_l", "foot_r"))

    # ---------------- geometric support labels + cross-validation ------------
    labels = {}
    agree = {"n": 0, "match": 0}
    for name in names:
        e = eps[name]
        sup2 = reconstruct_support(e["qpos"])               # (n,2) bool
        n = e["n_steps"]
        sup = np.full(n, 3, dtype=np.int8)
        sup[(sup2[:, 0]) & (~sup2[:, 1])] = 1
        sup[(~sup2[:, 0]) & (sup2[:, 1])] = 2
        sup[(~sup2[:, 0]) & (~sup2[:, 1])] = 0
        transition = np.zeros(n, dtype=bool)
        transition[1:] = sup[1:] != sup[:-1]
        free = e["free"]
        max_ctrl = np.abs(e["X"][:, 59:81]).max(axis=1)
        labels[name] = {
            "free": free, "actuated_free": free & (max_ctrl > 1e-6),
            "zero_ctrl_free": free & (max_ctrl <= 1e-6), "support": sup,
            "transition": transition, "max_ctrl": max_ctrl,
            "phys_foot_l": e["foot_l"].astype(bool), "phys_foot_r": e["foot_r"].astype(bool),
            "geo_foot_l": sup2[:, 0], "geo_foot_r": sup2[:, 1],
        }
        if meta[name]["origin"] == "stage1_new":
            agree["n"] += 2 * n
            agree["match"] += int((sup2[:, 0] == e["foot_l"].astype(bool)).sum()) \
                + int((sup2[:, 1] == e["foot_r"].astype(bool)).sum())

    # ---------------- impact labels need norm stats -> compute stats first ---
    # provisional band with train stats
    tr_names = SPLITS["train"]
    tr = sc.stack({k: eps[k] for k in tr_names})
    mu_a, std_a = tr["qacc"].mean(0), tr["qacc"].std(0)
    std_a = np.maximum(std_a, EPS_A_FLOOR)
    mu_x, std_x = tr["X"].mean(0), tr["X"].std(0)
    std_x = np.maximum(std_x, FEAT_FLOOR)
    free_tr = tr["free"]
    s_r = np.sqrt((np.einsum("nij,nj->ni", tr["M"][free_tr].astype(np.float64),
                             tr["qacc"][free_tr].astype(np.float64)) ** 2).mean(axis=0))
    s_r = np.maximum(s_r, SR_FLOOR)

    for name in names:
        e = eps[name]
        band = np.abs((e["qacc"] - mu_a) / std_a).max(axis=1)
        dcon = np.zeros(e["n_steps"])
        dcon[1:] = np.abs(np.diff(e["qfrc_constraint"], axis=0)).max(axis=1)
        trans_window = np.zeros(e["n_steps"], dtype=bool)
        for k in (-2, -1, 0, 1, 2):
            trans_window |= np.roll(labels[name]["transition"], k)
        impact = (band > IMPACT_SIGMA) | (trans_window & (dcon > IMPACT_DCON_N))
        labels[name]["band"] = band
        labels[name]["impact"] = impact
        labels[name]["stable_support"] = (labels[name]["support"] == 3) & (~impact)

    # ---------------- save labels --------------------------------------------
    flat = {}
    for name, lab in labels.items():
        for k, v in lab.items():
            flat[f"{name}__{k}"] = v
    np.savez_compressed(S1 / "stage1_labels.npz", **flat)

    # ---------------- splits + norm stats ------------------------------------
    (S1 / "splits.json").write_text(json.dumps(SPLITS, indent=2))
    stats = {
        "mu_x": mu_x.tolist(), "std_x": std_x.tolist(),
        "mu_a": mu_a.tolist(), "std_a": std_a.tolist(),
        "s_r": s_r.tolist(),
        "floors": {"feat": FEAT_FLOOR, "accel": EPS_A_FLOOR, "residual": SR_FLOOR},
        "train_filter_max_abs_norm": FILTER_SIGMA,
        "impact_rule": {"band_sigma": IMPACT_SIGMA, "dcon_N": IMPACT_DCON_N,
                        "transition_window_steps": 2},
        "support_label": "geometric: foot box corner min clearance <= 2 mm",
        "n_train_samples": int(tr["X"].shape[0]),
        "n_train_free": int(free_tr.sum()),
    }
    (S1 / "norm_stats.json").write_text(json.dumps(stats, indent=2))

    # ---------------- integrity checks ---------------------------------------
    integ = {"episodes": {}}
    problems = []
    for name in names:
        z = np.load(S1 / f"{name}.npz")
        rep = {}
        for k in ("qpos", "qvel", "qacc", "ctrl", "M_upper", "qfrc_bias",
                  "qfrc_actuator", "qfrc_constraint", "actuator_force"):
            bad = int((~np.isfinite(z[k])).sum())
            rep[f"nonfinite_{k}"] = bad
            if bad:
                problems.append(f"{name}:{k}")
        t = z["time"]
        rep["time_monotonic_1ms"] = bool(np.allclose(np.diff(t), 1e-3, atol=1e-9))
        rep["t_start"] = float(t[0]); rep["t_end"] = float(t[-1])
        rep["max_passive"] = float(np.abs(z["qfrc_passive"]).max())
        rep["max_applied"] = float(np.abs(z["qfrc_applied"]).max())
        rep["actuator_force_vs_ctrl"] = float(np.abs(z["actuator_force"] - z["ctrl"]).max())
        # snapshot consistency: full residual closure recomputed in f64
        M = sc.rebuild_M(z["M_upper"].astype(np.float64))
        r = (np.einsum("nij,nj->ni", M, z["qacc"].astype(np.float64))
             + z["qfrc_bias"] - z["qfrc_actuator"] - z["qfrc_passive"]
             - z["qfrc_applied"] - z["qfrc_constraint"])
        rel = np.abs(r).max(axis=1) / (np.abs(np.einsum(
            "nij,nj->ni", M, z["qacc"].astype(np.float64))).max(axis=1) + 1.0)
        rep["closure_max_abs"] = float(np.abs(r).max())
        rep["closure_max_rel"] = float(rel.max())
        rep["recorded_closure_max"] = float(z["residual_closure"].max())
        integ["episodes"][name] = rep
        if rep["max_passive"] != 0 or rep["max_applied"] != 0:
            problems.append(f"{name}: passive/applied nonzero")
    integ["problems"] = problems
    integ["geo_vs_physical_agreement_stage1_eps"] = round(
        agree["match"] / max(agree["n"], 1), 4)
    (S1 / "integrity.json").write_text(json.dumps(integ, indent=2, default=str))

    # ---------------- summary print ------------------------------------------
    tot = sum(eps[n]["n_steps"] for n in names)
    print(f"Stage1: {len(names)} episodes, {tot} samples")
    for sp, ns in SPLITS.items():
        n_ep = len(ns)
        n_s = sum(eps[n]["n_steps"] for n in ns)
        print(f"  {sp:5s}: {n_ep:2d} episodes, {n_s:6d} samples ({100 * n_s / tot:.0f}%)")
    free_all = sum(int(labels[n]["free"].sum()) for n in names)
    act_free = sum(int(labels[n]["actuated_free"].sum()) for n in names)
    print(f"  free={free_all} ({100 * free_all / tot:.1f}%), "
          f"actuated_free={act_free}")
    print(f"  geo-vs-physical foot agreement (stage1 episodes): "
          f"{agree['match']}/{agree['n']} = {100 * agree['match'] / max(agree['n'], 1):.2f}%")
    print(f"  integrity problems: {problems if problems else 'NONE'}")
    print(f"  closure max rel over all episodes: "
          f"{max(integ['episodes'][n]['closure_max_rel'] for n in names):.2e}")


if __name__ == "__main__":
    main()
