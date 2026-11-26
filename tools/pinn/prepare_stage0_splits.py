#!/usr/bin/env python3
"""Stage0-D Task B: episode-level splits + train-only normalization statistics.

Split design (motion-disjoint PD references, independent episodes everywhere):
  TEST: pd_top_rock, drop_d0, ou_o0
  VAL:  pd_arm_wave, drop_d1, ou_o1
  TRAIN: pd_popping_s30/s31, pd_arm_wave_usr_s40/s41, drop_d2..d7, ou_o2..o5
No episode appears in two splits; each PD motion lives in exactly one split.
Test/val episodes are never used for normalization, residual scales or any
threshold selection.

Outputs: splits.json, norm_stats.json and prints Task A5 coverage statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import stage0_common as sc

SPLITS = {
    "test": ["pd_top_rock", "drop_d0", "ou_o0"],
    "val": ["pd_arm_wave", "drop_d1", "ou_o1"],
    "train": ["pd_popping_s30", "pd_popping_s31", "pd_arm_wave_usr_s40",
              "pd_arm_wave_usr_s41", "drop_d2", "drop_d3", "drop_d4",
              "drop_d5", "drop_d6", "drop_d7", "ou_o2", "ou_o3", "ou_o4",
              "ou_o5"],
}

EPS_A_FLOOR = 1e-3     # m/s^2, rad/s^2
FEAT_FLOOR = 1e-3
SR_FLOOR = 1e-3        # N, Nm


def main() -> None:
    index = json.loads((sc.DATA_DIR / "index.json").read_text())
    names_by_type = {e["episode_id"]: e["action_type"] for e in index["episodes"]}
    all_eps = sc.load_episodes(sorted(names_by_type))

    # ---------------------------------------------------------- A5 statistics
    print("=== per-episode free statistics ===")
    total = free_total = 0
    by_type: dict[str, list[int]] = {}
    for e in index["episodes"]:
        n = e["n_steps"]
        f = int(all_eps[e["episode_id"]]["free"].sum())
        total += n
        free_total += f
        by_type.setdefault(e["action_type"], []).append(f)
        print(f"  {e['episode_id']:22s} {e['action_type']:16s} steps={n:5d} "
              f"free={f:5d} ({100 * f / n:5.1f}%)")
    print(f"TOTAL: {total} samples, free {free_total} ({100 * free_total / total:.1f}%)")
    for t, fs in sorted(by_type.items()):
        print(f"  by type {t:20s}: episodes={len(fs)} free_sum={sum(fs)}")

    train_eps = sc.stack({k: all_eps[k] for k in SPLITS["train"]})
    free = train_eps["free"]
    print("\n=== free-sample coverage in TRAIN (vs contact samples) ===")
    # feature layout: [0:3]=root pos, [3:9]=rot6, [9:31]=joints, [31:59]=qvel, [59:81]=ctrl
    X = train_eps["X"]
    for name, sl in (("joint pos", slice(9, 31)), ("joint vel", slice(37, 59)),
                     ("root lin vel", slice(31, 34)), ("root ang vel", slice(34, 37))):
        f_arr, c_arr = X[free][:, sl], X[~free][:, sl]
        print(f"  {name:12s} free  absmax={np.abs(f_arr).max():8.3f} "
              f"rms={np.sqrt((f_arr ** 2).mean()):8.3f}   "
              f"contact absmax={np.abs(c_arr).max():8.3f} "
              f"rms={np.sqrt((c_arr ** 2).mean()):8.3f}")

    # --------------------------------------------- splits.json / norm_stats.json
    sizes = {k: sum(all_eps[e]["n_steps"] for e in v) for k, v in SPLITS.items()}
    print("\nsplit sizes:", sizes,
          f"(train {100 * sizes['train'] / total:.0f}% / "
          f"val {100 * sizes['val'] / total:.0f}% / "
          f"test {100 * sizes['test'] / total:.0f}%)")

    Xtr = train_eps["X"]
    mu_x, std_x = Xtr.mean(0), Xtr.std(0)
    std_x = np.maximum(std_x, FEAT_FLOOR)
    mu_a, std_a = train_eps["qacc"].mean(0), train_eps["qacc"].std(0)
    std_a = np.maximum(std_a, EPS_A_FLOOR)
    # residual scale: RMS over TRAIN free samples of the inertial term (M qacc)_i
    Mf = train_eps["M"][free]
    inert = np.einsum("nij,nj->ni", Mf, train_eps["qacc"][free].astype(np.float64))
    s_r = np.sqrt((inert ** 2).mean(axis=0))
    s_r = np.maximum(s_r, SR_FLOOR)

    stats = {
        "mu_x": mu_x.tolist(), "std_x": std_x.tolist(),
        "mu_a": mu_a.tolist(), "std_a": std_a.tolist(),
        "s_r": s_r.tolist(),
        "floors": {"feat": FEAT_FLOOR, "accel": EPS_A_FLOOR, "residual": SR_FLOOR},
        # frozen training-filter rule: impact-termination frames (fall/landing)
        # exceed 10 sigma in normalized qacc, dominate MSE gradients (87% of
        # val MSE from 0.6% of samples) and are excluded from TRAINING and
        # early-stopping; ALL metrics are still reported on full sets.
        "train_filter_max_abs_norm": 10.0,
        "n_train_samples": int(Xtr.shape[0]),
        "n_train_free": int(free.sum()),
    }
    (sc.DATA_DIR / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    (sc.DATA_DIR / "splits.json").write_text(json.dumps(SPLITS, indent=2))
    print(f"\nsaved {sc.DATA_DIR / 'splits.json'} and norm_stats.json "
          f"(train {Xtr.shape[0]} samples, free {int(free.sum())}, "
          f"free fraction {100 * free.mean():.1f}%)")


if __name__ == "__main__":
    main()
