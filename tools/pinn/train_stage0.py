#!/usr/bin/env python3
"""Stage0-D Task C/D: train MLP baseline or Free-Dynamics PINN.

Identical data, splits, architecture and optimization for both; the only
difference is the physics term:

  L      = L_data                                  (mlp)
  L      = L_data + lambda_dyn * L_free            (pinn)
  L_free = mean over free samples in batch of (1/28) sum_i (r_i/s_{r,i})^2,
           r = M a_pred + qfrc_bias - qfrc_actuator  (no qfrc_constraint!)

Per-epoch logs: data/physics/total loss, probe-batch gradient norms of each
term, number of free samples seen, lr. Early stopping on validation L_data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import stage0_common as sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("mlp", "pinn"), required=True)
    ap.add_argument("--lambda_dyn", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs-dir", type=Path, default=sc.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=sc.DATA_DIR)
    args = ap.parse_args()
    assert args.model == "mlp" or args.lambda_dyn is not None, \
        "pinn requires --lambda_dyn"
    if args.model == "mlp":
        args.lambda_dyn = 0.0

    base = (f"{args.model}_lam{args.lambda_dyn:g}_seed{args.seed}"
            if args.model == "pinn" else f"mlp_seed{args.seed}")
    run_name = base + (f"_{args.tag}" if args.tag else "")
    out_dir = args.runs_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    splits = json.loads((args.data_dir / "splits.json").read_text())
    stats = sc.load_norm_stats(args.data_dir / "norm_stats.json")
    mu_x, std_x = sc.to_torch(stats["mu_x"], device), sc.to_torch(stats["std_x"], device)
    mu_a, std_a = sc.to_torch(stats["mu_a"], device), sc.to_torch(stats["std_a"], device)
    s_r = sc.to_torch(stats["s_r"], device)

    def load_split(names):
        eps = sc.load_episodes(names, data_dir=args.data_dir)
        s = sc.stack(eps)
        t = {k: sc.to_torch(v, device) for k, v in s.items()}
        t["free"] = torch.as_tensor(s["free"], device=device)  # keep bool
        return t

    tr, va = load_split(splits["train"]), load_split(splits["val"])
    tr_x = (tr["X"] - mu_x) / std_x
    tr_y = (tr["qacc"] - mu_a) / std_a
    va_x = (va["X"] - mu_x) / std_x
    va_y = (va["qacc"] - mu_a) / std_a

    # frozen rule from norm_stats.json: exclude extreme impact-termination
    # frames (>10 sigma normalized qacc) from training and early stopping;
    # full-set metrics are computed in evaluate_stage0.py
    keep_max = float(json.loads((args.data_dir / "norm_stats.json").read_text())
                     ["train_filter_max_abs_norm"])
    tr_keep = tr_y.abs().max(dim=1).values <= keep_max
    va_keep = va_y.abs().max(dim=1).values <= keep_max
    tr_x, tr_y = tr_x[tr_keep], tr_y[tr_keep]
    for k in ("M", "qfrc_bias", "qfrc_actuator", "free"):
        tr[k] = tr[k][tr_keep]
    va_xe, va_ye = va_x[va_keep], va_y[va_keep]          # early-stop views
    va_full = {k: va[k][va_keep] for k in va}            # for val_free monitor
    n_dropped_train = int((~tr_keep).sum())

    model = sc.AccelMLP(width=args.width, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    is_pinn = args.model == "pinn"
    lam = args.lambda_dyn

    # fixed probe batch for gradient-norm monitoring (same for every model)
    probe_x, probe_y = tr_x[:1024], tr_y[:1024]
    probe_M, probe_bias, probe_act = tr["M"][:1024], tr["qfrc_bias"][:1024], \
        tr["qfrc_actuator"][:1024]
    probe_free = tr["free"][:1024]
    def eval_losses(x, y, full=None):
        """Full-pass data loss; optionally physics loss on all free samples."""
        with torch.no_grad():
            pred = model(x)
            ld = sc.data_loss(pred, y).item()
            lf, nf = float("nan"), 0
            if full is not None and is_pinn:
                f = full["free"]
                if f.any():
                    a_phys = pred[f] * std_a + mu_a
                    r = sc.free_residual(a_phys, full["M"][f], full["qfrc_bias"][f],
                                         full["qfrc_actuator"][f])
                    lf = sc.free_loss(r, s_r).item()
                    nf = int(f.sum())
        return ld, lf, nf

    history = []
    best_val, best_state, best_epoch = math.inf, None, -1
    t0 = time.time()
    n_train = tr_x.shape[0]
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        ep_ld = ep_lf = 0.0
        ep_nf = 0
        n_lf_batches = 0
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = tr_x[idx], tr_y[idx]
            pred = model(xb)
            loss_data = sc.data_loss(pred, yb)
            loss = loss_data
            if is_pinn:
                fb = tr["free"][idx]
                if fb.any():
                    a_phys = pred[fb] * std_a + mu_a
                    r = sc.free_residual(a_phys, tr["M"][idx][fb],
                                         tr["qfrc_bias"][idx][fb],
                                         tr["qfrc_actuator"][idx][fb])
                    loss_free = sc.free_loss(r, s_r)
                    loss = loss + lam * loss_free
                    ep_lf += loss_free.item()
                    n_lf_batches += 1
                    ep_nf += int(fb.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_ld += loss_data.item()
        n_batches = math.ceil(n_train / args.batch)

        # probe gradient norms (data term vs physics term, separately)
        model.eval()
        gnorm_data = gnorm_phys = float("nan")
        if is_pinn and probe_free.any():
            opt.zero_grad(set_to_none=True)
            sc.data_loss(model(probe_x), probe_y).backward()
            gnorm_data = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            opt.zero_grad(set_to_none=True)
            pred_p = model(probe_x)
            a_phys_p = pred_p[probe_free] * std_a + mu_a
            r_p = sc.free_residual(a_phys_p, probe_M[probe_free],
                                   probe_bias[probe_free], probe_act[probe_free])
            (lam * sc.free_loss(r_p, s_r)).backward()
            gnorm_phys = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            opt.zero_grad(set_to_none=True)

        val_ld, val_lf, _ = eval_losses(va_xe, va_ye, full=va_full)
        history.append({
            "epoch": epoch,
            "train_data": ep_ld / n_batches,
            "train_free": ep_lf / max(n_lf_batches, 1) if n_lf_batches else None,
            "train_free_samples": ep_nf,
            "val_data": val_ld, "val_free": val_lf,
            "gnorm_data": gnorm_data, "gnorm_phys_x_lambda": gnorm_phys,
            "lr": args.lr,
        })
        if val_ld < best_val:
            best_val, best_epoch = val_ld, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    dt = time.time() - t0
    model.load_state_dict(best_state)

    cfg = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(args).items()}
    cfg.update({"run_name": run_name, "n_params": n_params,
                "n_train": n_train, "n_train_free": int(tr["free"].sum()),
                "n_train_dropped_spikes": n_dropped_train,
                "train_filter_max_abs_norm": keep_max,
                "n_val": int(va["X"].shape[0]),
                "n_val_dropped_spikes": int((~va_keep).sum()),
                "best_epoch": best_epoch, "best_val_data": best_val,
                "train_seconds": round(dt, 1),
                "arch": f"{sc.IN_DIM}x{args.width}x{args.depth}x{sc.NV}",
                "physics_residual": ("M a + qfrc_bias - qfrc_actuator (free samples only)"
                                     if is_pinn else "none")})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(best_state, out_dir / "best.pt")
    print(f"{run_name}: best_epoch={best_epoch} val_data={best_val:.5f} "
          f"({n_params} params, {dt:.1f}s)")


if __name__ == "__main__":
    main()
