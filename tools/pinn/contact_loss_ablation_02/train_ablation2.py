#!/usr/bin/env python3
"""K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_02 — corrected training for G3/G4/G5.

Fixes the ABLATION_01 implementation error: the 10-sigma keep mask was applied
to X/qacc-labels/M/qfrc_constraint/free inconsistently — M, qfrc_constraint
and free were subsetted but qacc was NOT, so the G3/G4 physics losses used
`e_phys = a_pred - qacc_unfiltered[idx]` with idx in the FILTERED index space,
misaligning labels for every sample after the first dropped spike. Here ALL
per-sample fields are subsetted with one shared keep mask and the keep indices
are persisted for reproducibility.

Loss definitions unchanged:
  G3  L = L_acc + lam * mean(e^T M e) / s3^2,  e physical, full matrix
  G4  L = L_acc + lam * mean(||M e||^2) / s4^2
  G5  shared trunk, main head 28-d qacc + aux head 28-d qfrc_constraint
      (z-scored with TRAIN-only mu_g/sigma_g); aux dropped at inference
Base L_acc identical to Stage1-02 (per-dof z-scored qacc MSE, frozen
10-sigma filter). Diagnostics per epoch: main/physics(aux) losses, class-wise
contributions, and per-component gradient norms measured on the SAME
parameter set over a fixed probe batch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stage0_common as sc

STAGE1 = HERE.parent / "stage1_data"
RUNS = HERE / "runs"
NV = 28


class AuxAccelMLP(torch.nn.Module):
    """Stage1-02 trunk with main qacc head + auxiliary qfrc_constraint head."""

    def __init__(self, in_dim: int = sc.IN_DIM, out_dim: int = NV,
                 width: int = 256, depth: int = 3, aux_dim: int = NV):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        self.trunk = torch.nn.Sequential(*layers)
        self.main_head = torch.nn.Linear(d, out_dim)
        self.aux_head = torch.nn.Linear(d, aux_dim)

    def forward(self, x):
        return self.main_head(self.trunk(x))

    def forward_aux(self, x):
        h = self.trunk(x)
        return self.main_head(h), self.aux_head(h)


def loss_g3(e: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bij,bj->b", e, M, e).mean()


def loss_g4(e: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return (torch.bmm(M, e.unsqueeze(2)).squeeze(2) ** 2).sum(dim=1).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=("g3", "g4", "g5"), required=True)
    ap.add_argument("--lambda_dyn", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = json.loads((STAGE1 / "splits.json").read_text())
    stats_nm = sc.load_norm_stats(STAGE1 / "norm_stats.json")
    mu_x = sc.to_torch(stats_nm["mu_x"], device)
    std_x = sc.to_torch(stats_nm["std_x"], device)
    mu_a = sc.to_torch(stats_nm["mu_a"], device)
    std_a = sc.to_torch(stats_nm["std_a"], device)
    keep_max = float(json.loads((STAGE1 / "norm_stats.json").read_text())
                     ["train_filter_max_abs_norm"])

    def load_split(names):
        eps = sc.load_episodes(names, data_dir=STAGE1,
                               extra_fields=("M_upper", "qfrc_constraint"))
        s = sc.stack(eps)
        t = {k: sc.to_torch(v, device) for k, v in s.items()}
        t["free"] = torch.as_tensor(s["free"], device=device)
        m_up = np.concatenate([eps[n]["M_upper"] for n in names], axis=0)
        t["M"] = sc.to_torch(sc.rebuild_M(m_up), device)
        qc = np.concatenate([eps[n]["qfrc_constraint"] for n in names], axis=0)
        t["qfrc_constraint"] = sc.to_torch(qc, device)
        return t

    tr, va = load_split(splits["train"]), load_split(splits["val"])
    tr_x = (tr["X"] - mu_x) / std_x
    tr_y = (tr["qacc"] - mu_a) / std_a
    va_x = (va["X"] - mu_x) / std_x
    va_y = (va["qacc"] - mu_a) / std_a

    # --- FIX: one shared keep mask synchronizes X, qacc, M, qfrc_constraint,
    # free; keep indices persisted so the filtering is exactly reproducible
    tr_keep = tr_y.abs().max(dim=1).values <= keep_max
    va_keep = va_y.abs().max(dim=1).values <= keep_max
    keep_idx = tr_keep.nonzero(as_tuple=True)[0]
    fields = ("X", "qacc", "M", "qfrc_constraint", "free")
    for k in fields:
        tr[k] = tr[k][tr_keep]
    for k in fields:
        if k in va:
            va[k] = va[k][va_keep]
    va_xe, va_ye = va_x[va_keep], va_y[va_keep]
    tr_x, tr_y = tr_x[tr_keep], tr_y[tr_keep]
    n_drop = int((~tr_keep).sum())

    is_aux = args.group == "g5"
    mu_g = std_g = None
    if is_aux:
        mu_g = tr["qfrc_constraint"].mean(dim=0)          # TRAIN-only stats
        std_g = tr["qfrc_constraint"].std(dim=0).clamp(min=1e-3)
    tr_g = (tr["qfrc_constraint"] - mu_g) / std_g if is_aux else None
    va_g = (va["qfrc_constraint"] - mu_g) / std_g if is_aux else None

    run_name = f"{args.group}_lam{args.lambda_dyn:g}_seed{args.seed}"
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN-only physical scales for the added losses (on kept samples)
    eps_dev = tr["qacc"] - mu_a
    s3_scale = float(torch.einsum("bi,bij,bj->b", eps_dev, tr["M"], eps_dev).mean())
    s4_scale = float((torch.bmm(tr["M"], eps_dev.unsqueeze(2)).squeeze(2) ** 2)
                     .sum(dim=1).mean())
    print(f"{run_name}: n_train={tr_x.shape[0]} (dropped {n_drop}), "
          f"s3={s3_scale:.4g} s4={s4_scale:.4g}")

    model = (AuxAccelMLP(width=args.width, depth=args.depth).to(device)
             if is_aux else sc.AccelMLP(width=args.width, depth=args.depth).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lam = args.lambda_dyn
    n_train = tr_x.shape[0]

    probe = slice(0, 1024)

    def grad_norms_probe() -> dict:
        xb = tr_x[probe]; yb = tr_y[probe]
        Mb = tr["M"][probe]; qb = tr["qacc"][probe]
        gb = tr_g[probe] if is_aux else None
        fb = tr["free"][probe]
        res = {}
        opt.zero_grad(set_to_none=True)
        sc.data_loss(model(xb), yb).backward()
        res["data"] = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1e9))
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        a_phys = pred * std_a + mu_a
        e_phys = a_phys - qb
        if args.group == "g3":
            lp = loss_g3(e_phys, Mb) / s3_scale
        elif args.group == "g4":
            lp = loss_g4(e_phys, Mb) / s4_scale
        else:
            _, aux = model.forward_aux(xb)
            lp = torch.nn.functional.mse_loss(aux, gb if is_aux else qb)
        (lam * lp).backward()
        res["physics_or_aux_x_lambda"] = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1e9))
        opt.zero_grad(set_to_none=True)
        # class-wise data-loss gradients not needed per epoch; keep loss values
        res["probe_data_loss"] = float(sc.data_loss(model(xb), yb))
        res["probe_free_frac"] = float(fb.float().mean())
        opt.zero_grad(set_to_none=True)
        return res

    history = []
    best_val, best_state, best_epoch = math.inf, None, -1
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        ep = {"data": 0.0, "phys": 0.0, "nb": 0}
        cls = {"free": [0.0, 0], "contact": [0.0, 0], "impact": [0.0, 0]}
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = tr_x[idx], tr_y[idx]
            pred = model(xb)
            loss_data = sc.data_loss(pred, yb)
            loss = loss_data
            a_phys = pred * std_a + mu_a
            e_phys = a_phys - tr["qacc"][idx]        # aligned after the FIX
            if args.group == "g3":
                lp = loss_g3(e_phys, tr["M"][idx]) / s3_scale
            elif args.group == "g4":
                lp = loss_g4(e_phys, tr["M"][idx]) / s4_scale
            else:
                _, aux = model.forward_aux(xb)
                lp = torch.nn.functional.mse_loss(aux, tr_g[idx])
            loss = loss_data + lam * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep["data"] += loss_data.item(); ep["phys"] += lp.item(); ep["nb"] += 1
            with torch.no_grad():
                f = tr["free"][idx]
                # impact flag not carried in tensors; free/contact bookkeeping only
                cls["free"][0] += float(((pred - yb)[f] ** 2).mean()) if f.any() else 0.0
                cls["free"][1] += int(f.sum())
                cls["contact"][0] += float(((pred - yb)[~f] ** 2).mean()) if (~f).any() else 0.0
                cls["contact"][1] += int((~f).sum())
        nb = math.ceil(n_train / args.batch)
        model.eval()
        with torch.no_grad():
            vp = model(va_xe)
            val_ld = float(sc.data_loss(vp, va_ye))
            a_phys = vp * std_a + mu_a
            e_phys = a_phys - va["qacc"]
            if args.group == "g3":
                val_lp = float(loss_g3(e_phys, va["M"]) / s3_scale)
            elif args.group == "g4":
                val_lp = float(loss_g4(e_phys, va["M"]) / s4_scale)
            else:
                _, vaux = model.forward_aux(va_xe)
                val_lp = float(torch.nn.functional.mse_loss(vaux, va_g))
        gn = grad_norms_probe()
        history.append({"epoch": epoch, "train_data": ep["data"] / nb,
                        "train_phys_or_aux": ep["phys"] / nb,
                        "val_data": val_ld, "val_phys_or_aux": val_lp,
                        "grad": gn,
                        "train_cls_free_mse": cls["free"][0] / max(cls["free"][1], 1),
                        "train_cls_contact_mse": cls["contact"][0] / max(cls["contact"][1], 1)})
        if val_ld < best_val:
            best_val, best_epoch = val_ld, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)

    cfg = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(args).items()}
    cfg.update({"run_name": run_name, "n_params": sum(p.numel() for p in model.parameters()),
                "n_train": n_train, "n_train_dropped_spikes": n_drop,
                "keep_index_sha": str(keep_idx.sum().item()),
                "keep_max_sigma": keep_max,
                "base_loss": "per-dof z-scored qacc MSE (identical to Stage1-02)",
                "normalization": {"g3_scale_s3_sq_TRAIN_kept": s3_scale,
                                   "g4_scale_s4_sq_TRAIN_kept": s4_scale,
                                   "g5_aux": "qfrc_constraint z-scored by TRAIN-only stats"},
                "fix": "shared keep mask synchronizes X/qacc/M/qfrc_constraint/free",
                "lambda_dyn": args.lambda_dyn, "best_epoch": best_epoch,
                "best_val_data": best_val,
                "train_seconds": round(time.time() - t0, 1)})
    if is_aux:
        cfg["g5_mu_g"] = mu_g.cpu().tolist()
        cfg["g5_std_g"] = std_g.cpu().tolist()
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "keep_indices.npy").write_bytes(keep_idx.cpu().numpy().tobytes())
    torch.save(best_state, out_dir / "best.pt")
    print(f"{run_name}: best_epoch={best_epoch} val_data={best_val:.5f} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
