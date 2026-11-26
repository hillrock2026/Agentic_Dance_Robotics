#!/usr/bin/env python3
"""K1_PINN_STAGE2_CONTACT_LOSS_ABLATION_01 — training for G3/G4/G5.

Base supervision L_acc is IDENTICAL to Stage1-02 (per-dof z-scored qacc MSE,
same 10-sigma impact-frame filter, same optimizer/epochs/patience), so the
only study variable is the added physics term:

  G3  L = L_acc + lam * (1/N) sum e^T M(q) e / s3^2
        e = a_phys_pred - a_true (physical units), full symmetric matrix,
        s3^2 = mean_TRAIN((a*-mu_a)^T M (a*-mu_a))          [TRAIN-only scale]
  G4  L = L_acc + lam * (1/N) sum ||M(q) e||^2 / s4^2
        s4^2 = mean_TRAIN(||M(a*-mu_a)||^2)                  [TRAIN-only scale]
  G5  shared trunk, main head 28-d qacc + auxiliary head 28-d qfrc_constraint
        L = L_acc + lam * MSE_aux (aux target z-scored by TRAIN mu_g/sigma_g)
        auxiliary head sees only the shared trunk features; dropped at
        inference (solver-force privileged supervision).

Run with --verify for the Step-2 mathematical unit checks (no training).
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


# ------------------------------------------------------------------ model
class AuxAccelMLP(torch.nn.Module):
    """Stage1-02 trunk (81->3x256 SiLU) with main qacc head + aux head."""

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
        h = self.trunk(x)
        return self.main_head(h)

    def forward_aux(self, x):
        h = self.trunk(x)
        return self.main_head(h), self.aux_head(h)


# ------------------------------------------------------------------ losses
def loss_g3(e_phys: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(1/N) sum_i e_i^T M e_i  (full symmetric matrix quadratic form)."""
    return torch.einsum("bi,bij,bj->b", e_phys, M, e_phys).mean()


def loss_g4(e_phys: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(1/N) sum_i ||M e_i||^2  (NOT equal to G3 in general)."""
    return (torch.bmm(M, e_phys.unsqueeze(2)).squeeze(2) ** 2).sum(dim=1).mean()


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
    ap.add_argument("--verify", action="store_true",
                    help="Step-2 math checks only (no training)")
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
        t["M"] = sc.to_torch(sc.rebuild_M(m_up), device)      # f32 symmetric
        qc = np.concatenate([eps[n]["qfrc_constraint"] for n in names], axis=0)
        t["qfrc_constraint"] = sc.to_torch(qc, device)
        return t

    tr, va = load_split(splits["train"]), load_split(splits["val"])
    tr_x = (tr["X"] - mu_x) / std_x
    tr_y = (tr["qacc"] - mu_a) / std_a
    va_x = (va["X"] - mu_x) / std_x
    va_y = (va["qacc"] - mu_a) / std_a
    tr_keep = tr_y.abs().max(dim=1).values <= keep_max
    va_keep = va_y.abs().max(dim=1).values <= keep_max
    tr_x, tr_y = tr_x[tr_keep], tr_y[tr_keep]
    for k in ("M", "qfrc_constraint", "free"):
        tr[k] = tr[k][tr_keep]
    va_xe, va_ye = va_x[va_keep], va_y[va_keep]
    va_full = {k: va[k][va_keep] for k in va}
    n_drop = int((~tr_keep).sum())

    # G5 auxiliary target: z-scored qfrc_constraint, TRAIN-only stats
    if args.group == "g5":
        mu_g = tr["qfrc_constraint"].mean(dim=0)
        std_g = tr["qfrc_constraint"].std(dim=0).clamp(min=1e-3)
    else:
        mu_g = std_g = None

    # ---------------- Step 2: mathematical unit checks ----------------------
    if args.verify:
        e = torch.randn(64, NV, device=device) * 0.5
        M = tr["M"][:64]
        sym_err = float(np.abs(M.cpu().numpy() - M.cpu().numpy().transpose(0, 2, 1)).max())
        l3 = loss_g3(e, M)
        l4 = loss_g4(e, M)
        # brute-force reference on 4 samples
        ref3 = ref4 = 0.0
        for i in range(4):
            m_ = M[i].cpu().numpy(); e_ = e[i].cpu().numpy()
            ref3 = max(ref3, abs(float(e_ @ m_ @ e_) - loss_g3(e[i:i+1], M[i:i+1]).item()))
            ref4 = max(ref4, abs(float(np.linalg.norm(m_ @ e_) ** 2)
                                 - loss_g4(e[i:i+1], M[i:i+1]).item()))
        m_aux = AuxAccelMLP(width=args.width, depth=args.depth).to(device)
        out, aux = m_aux.forward_aux(torch.randn(8, sc.IN_DIM, device=device))
        dims_ok = out.shape == (8, NV) and aux.shape == (8, NV)
        # identity: e^T M e == |Me|^2 only for special e (eigenvectors); general random e differs
        print(f"[verify] M symmetric err={sym_err:.1e} (28x28 rebuilt f32)")
        print(f"[verify] G3 vs brute-force e^TMe: max diff={ref3:.2e}")
        print(f"[verify] G4 vs brute-force ||Me||^2: max diff={ref4:.2e}")
        print(f"[verify] G3 != G4 on random e: |L3-L4|/L3 = "
              f"{abs(l3.item()-l4.item())/l3.item():.4f}")
        print(f"[verify] G5 main(8,28)={tuple(out.shape)} aux(8,28)={tuple(aux.shape)} "
              f"ok={dims_ok}; aux head input=trunk features only (no ground truth)")
        return

    # ---------------- training ----------------------------------------------
    run_name = f"{args.group}_lam{args.lambda_dyn:g}_seed{args.seed}"
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    is_aux = args.group == "g5"

    model = (AuxAccelMLP(width=args.width, depth=args.depth).to(device)
             if is_aux else sc.AccelMLP(width=args.width, depth=args.depth).to(device))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lam = args.lambda_dyn

    def s_scales() -> tuple[float, float]:
        """TRAIN-only physical scales for the added losses (frozen), computed
        on the KEPT training samples (10-sigma filtered, same subset the
        physics losses act on)."""
        eps_dev = tr["qacc"][tr_keep] - mu_a             # typical deviation
        s3 = torch.einsum("bi,bij,bj->b", eps_dev, tr["M"], eps_dev).mean()
        s4 = (torch.bmm(tr["M"], eps_dev.unsqueeze(2)).squeeze(2) ** 2) \
            .sum(dim=1).mean()
        return float(s3), float(s4)

    s3_scale, s4_scale = s_scales()
    print(f"{run_name}: n_train={tr_x.shape[0]} (dropped {n_drop} spikes), "
          f"s3={s3_scale:.4g} s4={s4_scale:.4g}")

    history = []
    best_val, best_state, best_epoch = math.inf, None, -1
    gen = torch.Generator().manual_seed(args.seed)
    n_train = tr_x.shape[0]
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        ep = {"data": 0.0, "phys": 0.0, "aux": 0.0, "n_batches": 0}
        cls_sq = {"free": 0.0, "contact": 0.0, "impact": 0.0}
        cls_n = {"free": 0, "contact": 0, "impact": 0}
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = tr_x[idx], tr_y[idx]
            if is_aux:
                pred, aux = model.forward_aux(xb)
                loss_data = sc.data_loss(pred, yb)
                loss_aux = torch.nn.functional.mse_loss(aux, tr["qfrc_constraint"][idx])
                loss = loss_data + lam * loss_aux
                ep["aux"] += loss_aux.item()
            else:
                pred = model(xb)
                loss_data = sc.data_loss(pred, yb)
                a_phys = pred * std_a + mu_a
                e_phys = a_phys - tr["qacc"][idx]
                if args.group == "g3":
                    loss_phys = loss_g3(e_phys, tr["M"][idx]) / s3_scale
                else:
                    loss_phys = loss_g4(e_phys, tr["M"][idx]) / s4_scale
                loss = loss_data + lam * loss_phys
                ep["phys"] += loss_phys.item()
            loss_data_item = loss_data.item()
            ep["data"] += loss_data_item
            ep["n_batches"] += 1
            # class-wise data-loss contribution (bookkeeping)
            with torch.no_grad():
                f = tr["free"][idx]
                cls_sq["free"] += float(((pred - yb)[f] ** 2).mean()) if f.any() else 0.0
                cls_sq["contact"] += float(((pred - yb)[~f] ** 2).mean()) if (~f).any() else 0.0
                cls_n["free"] += int(f.sum()); cls_n["contact"] += int((~f).sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        nb = math.ceil(n_train / args.batch)
        model.eval()
        with torch.no_grad():
            vp = model(va_xe)
            val_ld = sc.data_loss(vp, va_ye).item()
            val_phys = float("nan")
            if args.group in ("g3", "g4"):
                a_phys = vp * std_a + mu_a
                e_phys = a_phys - va["qacc"][va_keep]
                if args.group == "g3":
                    val_phys = float(loss_g3(e_phys, va["M"][va_keep]) / s3_scale)
                else:
                    val_phys = float(loss_g4(e_phys, va["M"][va_keep]) / s4_scale)
            elif is_aux:
                _, vaux = model.forward_aux(va_xe)
                val_phys = float(torch.nn.functional.mse_loss(
                    vaux, va["qfrc_constraint"][va_keep]))
        history.append({"epoch": epoch,
                        "train_data": ep["data"] / nb,
                        "train_phys_or_aux": ep["phys"] / nb if args.group != "g5"
                        else ep["aux"] / nb,
                        "val_data": val_ld, "val_phys_or_aux": val_phys,
                        "train_cls_free": cls_sq["free"] / max(cls_n["free"], 1),
                        "train_cls_contact": cls_sq["contact"] / max(cls_n["contact"], 1),
                        "lr": args.lr})
        if val_ld < best_val:
            best_val, best_epoch = val_ld, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)

    cfg = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(args).items()}
    cfg.update({"run_name": run_name, "n_params": n_params,
                "n_train": n_train, "n_train_dropped_spikes": n_drop,
                "best_epoch": best_epoch, "best_val_data": best_val,
                "train_seconds": round(time.time() - t0, 1),
                "base_loss": "per-dof z-scored qacc MSE (identical to Stage1-02)",
                "normalization": {
                    "g3_scale_s3_sq_TRAIN": s3_scale,
                    "g4_scale_s4_sq_TRAIN": s4_scale,
                    "g5_aux_target": "qfrc_constraint z-scored by TRAIN stats",
                },
                "lambda_dyn": args.lambda_dyn,
                "seeds_note": "training-stochasticity repeats; 5/5 direction "
                              "consistency is not statistical significance"})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(best_state, out_dir / "best.pt")
    print(f"{run_name}: best_epoch={best_epoch} val_data={best_val:.5f} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
