#!/usr/bin/env python3
"""Stage0-D-02 Step 3: Oracle residual test (must pass BEFORE PINN retraining).

Substitutes the recorded ground-truth acceleration a* = qacc into the FIXED
free_residual() through the same torch float32 code path used in training,
and compares against a float64 numpy reference, over free-mask samples from
every episode that contains them.

Also verifies the field mappings the residual depends on:
  - qfrc_actuator[6:28] == actuator_force == ctrl  (gear=1 torque actuators)
  - qfrc_actuator[0:6] == 0 (no root actuators)
  - qfrc_passive == qfrc_applied == 0 (no uncounted force terms)
  - normalization roundtrip: a_phys = a_norm * s_a + mu_a reproduces qacc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_common as sc


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=sc.DATA_DIR)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = json.loads((args.data_dir / "splits.json").read_text())
    stats = sc.load_norm_stats(args.data_dir / "norm_stats.json")
    s_r = np.asarray(stats["s_r"])
    mu_a, std_a = np.asarray(stats["mu_a"]), np.asarray(stats["std_a"])
    all_names = sorted(set(splits["train"] + splits["val"] + splits["test"]))
    eps = sc.load_episodes(all_names, data_dir=args.data_dir)

    rows = []
    for name, e in eps.items():
        f = e["free"]
        if not f.any():
            continue
        rows.append((name, e, f))
    print(f"episodes with free samples: {len(rows)}/{len(eps)}")

    # ---- per-episode oracle residuals, float64 numpy reference --------------
    f64_all, f32_all = [], []
    per_ep = []
    for name, e, f in rows:
        a, M, b, act = (e["qacc"][f].astype(np.float64), e["M"][f].astype(np.float64),
                        e["qfrc_bias"][f].astype(np.float64),
                        e["qfrc_actuator"][f].astype(np.float64))
        r64 = np.einsum("nij,nj->ni", M, a) + b - act
        # training code path: torch float32 on the training device
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        r32 = sc.free_residual(t(e["qacc"][f]), t(e["M"][f]), t(e["qfrc_bias"][f]),
                               t(e["qfrc_actuator"][f])).cpu().numpy().astype(np.float64)
        f64_all.append(r64)
        f32_all.append(r32)
        per_ep.append((name, int(f.sum()),
                       float(np.abs(r64).max()),
                       float(np.sqrt(((r32 / s_r) ** 2).mean()))))
        print(f"  {name:22s} n_free={int(f.sum()):5d} "
              f"max|r64|={np.abs(r64).max():.3e} N  RMS(r32/s_r)={per_ep[-1][3]:.3e}")

    R64 = np.concatenate(f64_all)
    R32 = np.concatenate(f32_all)
    n = R64.shape[0]
    print(f"\noracle residual over {n} free samples x 28 dofs:")
    for tag, R in (("float64 (numpy)", R64), ("float32 (train path)", R32)):
        absr = np.abs(R)
        nrm = np.sqrt(((R / s_r) ** 2).mean())
        print(f"  {tag:20s} dtype={R.dtype} max|abs|={absr.max():.3e} "
              f"mean|abs|={absr.mean():.3e} RMS|abs|={np.sqrt((R ** 2).mean()):.3e} "
              f"| normalized RMS={nrm:.3e}")

    # ---- field mapping cross-checks ----------------------------------------
    print("\nfield mapping checks (all free samples, all episodes):")
    A = np.concatenate([e["qfrc_actuator"][f] for _, e, f in rows])
    ctrl = np.concatenate([e2["X"][:, 59:81][f] for _, e2, f in rows])
    af, pp, aa = [], [], []
    for name, e, f in rows:
        z = np.load(args.data_dir / f"{name}.npz")
        af.append(z["actuator_force"][f])
        pp.append(z["qfrc_passive"][f])
        aa.append(z["qfrc_applied"][f])
    af, pp, aa = np.concatenate(af), np.concatenate(pp), np.concatenate(aa)
    checks = {
        "qfrc_actuator[6:28] == actuator_force": float(np.abs(A[:, 6:28] - af).max()),
        "actuator_force == ctrl": float(np.abs(af - ctrl).max()),
        "qfrc_actuator[0:6] == 0": float(np.abs(A[:, 0:6]).max()),
        "qfrc_passive == 0": float(np.abs(pp).max()),
        "qfrc_applied == 0": float(np.abs(aa).max()),
    }
    for k, v in checks.items():
        print(f"  {k}: max abs diff = {v:.3e}")

    # ---- normalization roundtrip -------------------------------------------
    X_err, a_max = [], 0.0
    for _, e, f in rows:
        a_norm = (e["qacc"][f].astype(np.float32) - mu_a) / std_a
        a_phys = a_norm * std_a + mu_a
        X_err.append(np.abs(a_phys - e["qacc"][f]).max())
        a_max = max(a_max, float(np.abs(e["qacc"][f]).max()))
    print(f"\nnormalization roundtrip max |a_phys - a*| = {max(X_err):.3e} "
          f"(on |a*| up to {a_max:.1f}; float32 storage precision)")

    # ---- verdict ------------------------------------------------------------
    r64_nrm = np.sqrt(((R64 / s_r) ** 2).mean())
    r32_nrm = np.sqrt(((R32 / s_r) ** 2).mean())
    ok = (r64_nrm < 1e-6 and r32_nrm < 1e-3
          and checks["qfrc_actuator[6:28] == actuator_force"] < 1e-5
          and checks["actuator_force == ctrl"] < 1e-5
          and checks["qfrc_actuator[0:6] == 0"] == 0.0
          and checks["qfrc_passive == 0"] == 0.0
          and checks["qfrc_applied == 0"] == 0.0)
    print(f"\nORACLE VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(f64 nRMSE={r64_nrm:.2e} < 1e-6, f32 nRMSE={r32_nrm:.2e} < 1e-3, "
          f"field mappings consistent)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
