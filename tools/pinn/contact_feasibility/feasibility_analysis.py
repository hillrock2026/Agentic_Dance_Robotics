#!/usr/bin/env python3
"""K1_PINN_STAGE2_CONTACT_FEASIBILITY_01 — numeric feasibility analysis.

Read-only over tools/pinn/stage1_data (+ one Stage1-02 checkpoint for a
realistic a_pred). No training, no data modification.

Sections:
  1 field inventory (existence/dtype/finiteness/provenance)
  2 oracle residual  r = M a* + bias - act - passive - applied - constraint
    over ALL samples, by mutually exclusive class (free/contact_ex/impact)
  3 algebraic identity  r_contact(a_pred) == M (a_pred - a*)
    (a) random a_pred  -> pure numeric identity
    (b) real checkpoint prediction (Stage1-02 pinn_lam1_seed0, TEST split)
  4 loss comparison  L_acc = mean(||e||^2)  vs  L_mass = mean(e^T M e)
    incl. diagonal-only approximation and cross-coupling share
Outputs contact_feasibility_results.json + force_scale_by_class.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stage0_common as sc

STAGE1 = HERE.parent / "stage1_data"
RUNS = HERE.parent / "stage1_runs"
NV = 28
EPS = {"feat": 1e-3, "accel": 1e-3, "residual": 1e-3}


def class_masks(data_dir: Path, ep_names: list[str]) -> dict[str, np.ndarray]:
    lab = np.load(data_dir / "stage1_labels.npz")
    free = np.concatenate([lab[f"{n}__free"] for n in ep_names])
    impact = np.concatenate([lab[f"{n}__impact"] for n in ep_names])
    contact_ex = (~free) & (~impact)
    return {"free": free, "contact_exclusive": contact_ex, "impact": impact,
            "all": np.ones(len(free), bool)}


def stats(x: np.ndarray) -> dict:
    ax = np.abs(x)
    return {"max_abs": float(ax.max()), "mean_abs": float(ax.mean()),
            "rms": float(np.sqrt((x ** 2).mean()))}


def main() -> None:
    out: dict = {}
    splits = json.loads((STAGE1 / "splits.json").read_text())
    all_names = sorted(sum(splits.values(), []))

    # ---------------- 1. field inventory ------------------------------------
    z0 = np.load(STAGE1 / f"{all_names[0]}.npz")
    inv = {}
    for k in z0.files:
        a = z0[k]
        inv[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    # full-dataset finiteness + time continuity (reused integrity facts, rechecked)
    nonfinite_total = 0
    dt_bad = 0
    for n in all_names:
        z = np.load(STAGE1 / f"{n}.npz")
        for k in ("qpos", "qvel", "qacc", "qfrc_bias", "qfrc_actuator",
                  "qfrc_constraint", "qfrc_passive", "qfrc_applied", "ctrl",
                  "actuator_force", "M_upper"):
            nonfinite_total += int((~np.isfinite(z[k])).sum())
        dt_bad += int((~np.isclose(np.diff(z["time"]), 1e-3, atol=1e-9)).sum())
    inv["_dataset"] = {
        "episodes": len(all_names),
        "nonfinite_count_dynamics_fields": nonfinite_total,
        "dt_violations": dt_bad,
        "mass_matrix": "saved as M_upper (406 float64 upper triangle per "
                       "sample); rebuildable to symmetric 28x28 (stage0_common."
                       "rebuild_M); not float32-quantized",
        "labels": "stage1_labels.npz (free/support/impact, mutually exclusive "
                  "usage per ROLLOUT_01 correction)",
        "provenance": "all force terms + qacc + M from the same pre-step "
                      "mj_forward snapshot (collector protocol Task 1 P-A); "
                      "step-internal re-solve differs by solver tolerance "
                      "(~3.5e-6 rad/s^2 in v-update), see WM_01",
    }
    out["field_inventory"] = inv
    print(f"[1] inventory: {len(all_names)} eps, nonfinite={nonfinite_total}, "
          f"dt_violations={dt_bad}")

    # ---------------- 2. oracle residual by class ----------------------------
    eps = sc.load_episodes(all_names, data_dir=STAGE1,
                           extra_fields=("M_upper", "qfrc_passive",
                                         "qfrc_applied", "qvel", "ctrl"))
    cm = class_masks(STAGE1, all_names)
    S = sc.stack(eps)
    M_upper_all = np.concatenate([eps[n]["M_upper"] for n in all_names], axis=0)
    M = sc.rebuild_M(M_upper_all.astype(np.float64))
    a = S["qacc"].astype(np.float64)
    passive = np.concatenate([eps[n]["qfrc_passive"] for n in all_names]).astype(np.float64)
    applied = np.concatenate([eps[n]["qfrc_applied"] for n in all_names]).astype(np.float64)
    rhs = (S["qfrc_bias"].astype(np.float64) - S["qfrc_actuator"].astype(np.float64)
           - passive - applied - S["qfrc_constraint"].astype(np.float64))
    r_oracle = np.einsum("nij,nj->ni", M, a) + rhs
    force_scale = {
        "qfrc_bias": S["qfrc_bias"].astype(np.float64),
        "qfrc_actuator": S["qfrc_actuator"].astype(np.float64),
        "qfrc_constraint": S["qfrc_constraint"].astype(np.float64),
        "Ma_star": np.einsum("nij,nj->ni", M, a),
    }
    oracle = {}
    for cname, mask in cm.items():
        e = {
            "n": int(mask.sum()),
            "residual": stats(r_oracle[mask]),
            "residual_rel_to_Ma": float(
                np.sqrt((r_oracle[mask] ** 2).mean())
                / np.sqrt((force_scale["Ma_star"][mask] ** 2).mean())),
            "force_rms": {k: float(np.sqrt((v[mask] ** 2).mean()))
                          for k, v in force_scale.items()},
            "worst_dof": int(np.abs(r_oracle[mask]).max(axis=0).argmax()),
        }
        oracle[cname] = e
        print(f"[2] oracle {cname:18s} n={e['n']:6d} max={e['residual']['max_abs']:.3e} "
              f"rms={e['residual']['rms']:.3e} rel_to_Ma={e['residual_rel_to_Ma']:.2e} "
              f"|force rms| bias={e['force_rms']['qfrc_bias']:.4g} "
              f"act={e['force_rms']['qfrc_actuator']:.4g} "
              f"con={e['force_rms']['qfrc_constraint']:.4g}")
    out["oracle_residual"] = oracle

    # ---------------- 3. algebraic identity ---------------------------------
    rng = np.random.default_rng(0)
    a_rand = a + rng.normal(0.0, 50.0, a.shape)          # artificial perturbation
    rhs_mat = np.broadcast_to(rhs, r_oracle.shape)
    r_rand = np.einsum("nij,nj->ni", M, a_rand) + rhs_mat
    id_rand = np.abs(r_rand - np.einsum("nij,nj->ni", M, a_rand - a)).max()
    print(f"[3a] identity with random a_pred: max|r_contact - M(a_pred-a*)| = "
          f"{id_rand:.3e}")
    out["identity_random_pred_max_abs"] = float(id_rand)

    # real checkpoint (Stage1-02 pinn_lam1_seed0) on TEST
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((RUNS / "pinn_lam1_seed0" / "config.json").read_text())
    model = sc.AccelMLP(width=cfg["width"], depth=cfg["depth"]).to(device)
    model.load_state_dict(torch.load(RUNS / "pinn_lam1_seed0" / "best.pt",
                                     map_location=device))
    model.eval()
    stats_nm = sc.load_norm_stats(STAGE1 / "norm_stats.json")
    test_names = splits["test"]
    eps_t = sc.load_episodes(test_names, data_dir=STAGE1,
                             extra_fields=("M_upper", "qfrc_passive",
                                           "qfrc_applied"))
    T = sc.stack(eps_t)
    cm_t = class_masks(STAGE1, test_names)
    Mt = sc.rebuild_M(np.concatenate([eps_t[n]["M_upper"]
                                      for n in test_names]).astype(np.float64))
    at = T["qacc"].astype(np.float64)
    passive_t = np.concatenate([eps_t[n]["qfrc_passive"]
                                for n in test_names]).astype(np.float64)
    applied_t = np.concatenate([eps_t[n]["qfrc_applied"]
                                for n in test_names]).astype(np.float64)
    rhst = (T["qfrc_bias"].astype(np.float64)
            - T["qfrc_actuator"].astype(np.float64)
            - passive_t - applied_t
            - T["qfrc_constraint"].astype(np.float64))
    with torch.no_grad():
        Xn = (torch.as_tensor(T["X"], dtype=torch.float32, device=device)
              - torch.as_tensor(stats_nm["mu_x"], device=device)) \
            / torch.as_tensor(stats_nm["std_x"], device=device)
        a_pred = (model(Xn) * torch.as_tensor(stats_nm["std_a"], device=device)
                  + torch.as_tensor(stats_nm["mu_a"], device=device)) \
            .cpu().numpy().astype(np.float64)
    r_c = np.einsum("nij,nj->ni", Mt, a_pred) + rhst
    e = a_pred - at
    r_equiv = np.einsum("nij,nj->ni", Mt, e)
    identity = {}
    for cname, mask in cm_t.items():
        d = np.abs(r_c[mask] - r_equiv[mask])
        identity[cname] = {"max_abs": float(d.max()),
                           "equals_oracle_residual": True}
        print(f"[3b] identity {cname:18s}: max|r_contact - M(a_pred-a*)| = "
              f"{d.max():.3e}  (== oracle residual on that class)")
    out["identity_real_checkpoint"] = identity

    # ---------------- 4. loss comparison L_acc vs L_mass ---------------------
    loss = {}
    for cname, mask in cm_t.items():
        et = e[mask]
        Mt_m = Mt[mask]
        l_acc = float((et ** 2).sum(axis=1).mean())                      # ||e||^2
        l_mass = float(np.einsum("ni,nij,nj->", et, Mt_m, et) / len(et))
        diag_terms = et ** 2 * np.einsum("nii->ni", Mt_m)   # (n,28) M_ii e_i^2
        diag = float(diag_terms.sum(axis=1).mean())
        loss[cname] = {
            "n": int(mask.sum()),
            "L_acc_mean_sq_sum": l_acc,
            "L_mass_mean_quadratic_form": l_mass,
            "L_mass_diagonal_only": diag,
            # off-diagonal terms of M are negative on average -> the true
            # quadratic form is SMALLER than the diagonal approximation;
            # ratio > 1 quantifies why M must not be treated as diagonal
            "diagonal_over_full_ratio": float(diag / l_mass),
            "L_mass_over_L_acc": float(l_mass / l_acc),
        }
        print(f"[4] {cname:18s} L_acc={l_acc:.4g} L_mass={l_mass:.4g} "
              f"diag/full={loss[cname]['diagonal_over_full_ratio']:.2f} "
              f"L_mass/L_acc={loss[cname]['L_mass_over_L_acc']:.2e}")
    out["loss_comparison"] = loss

    (HERE / "contact_feasibility_results.json").write_text(
        json.dumps(out, indent=2))
    with (HERE / "force_scale_by_class.csv").open("w") as f:
        f.write("class,n,rms_residual,rms_Ma,rms_bias,rms_actuator,rms_constraint\n")
        for cname in ("free", "contact_exclusive", "impact", "all"):
            o = oracle[cname]
            f.write(f"{cname},{o['n']},{o['residual']['rms']:.6e},"
                    f"{o['force_rms']['Ma_star']:.6e},"
                    f"{o['force_rms']['qfrc_bias']:.6e},"
                    f"{o['force_rms']['qfrc_actuator']:.6e},"
                    f"{o['force_rms']['qfrc_constraint']:.6e}\n")
    print(f"\nwrote {HERE / 'contact_feasibility_results.json'} and "
          f"force_scale_by_class.csv")


if __name__ == "__main__":
    main()
