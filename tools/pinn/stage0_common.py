"""Shared utilities for Stage0-D MLP / Free-Dynamics PINN experiments.

Features (81-dim): root pos (3) + root quat as 6D continuous rotation (6)
+ 22 joint positions + 28 qvel + 22 ctrl. Output: 28-dim qacc.

Data: one NPZ per episode under tools/pinn/stage0_data/ (see
collect_stage0_data.py). Splits are episode-level (splits.json); normalization
statistics are computed on the train split only (norm_stats.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tools/pinn/stage0_data"
RUNS_DIR = REPO_ROOT / "tools/pinn/stage0_runs"

NV = 28
NJOINT = 22
NU = 22
IN_DIM = 3 + 6 + NJOINT + NV + NU   # 81
IU = np.triu_indices(NV)


# ------------------------------------------------------------------ features
def quat_wxyz_to_rot6(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz quaternion -> (N,6) first two rotation-matrix columns."""
    w, x, y, z = q.T
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def build_features(qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """(N,29),(N,28),(N,22) -> (N,81)."""
    return np.concatenate([
        qpos[:, 0:3],                       # root position (world, m)
        quat_wxyz_to_rot6(qpos[:, 3:7]),    # root orientation, 6D rotation
        qpos[:, 7:29],                      # 22 joint positions (rad)
        qvel,                               # 28 generalized velocities
        ctrl,                               # 22 control commands (Nm)
    ], axis=1).astype(np.float32)


def rebuild_M(M_upper: np.ndarray) -> np.ndarray:
    """(N,406) upper triangle -> (N,28,28) symmetric dense."""
    n = M_upper.shape[0]
    M = np.zeros((n, NV, NV), dtype=np.float32)
    M[:, IU[0], IU[1]] = M_upper
    M[:, IU[1], IU[0]] = M_upper
    return M


def free_mask_of(npz) -> np.ndarray:
    """Task 1 mask: ncon==0 AND nefc==0 AND ||qfrc_constraint||_inf == 0."""
    return ((npz["ncon"] == 0) & (npz["nefc"] == 0)
            & (np.abs(npz["qfrc_constraint"]).max(axis=1) == 0.0))


# ----------------------------------------------------------------- dataset io
def load_episodes(ep_names: list[str], data_dir: Path = DATA_DIR,
                  extra_fields: tuple = ()) -> dict:
    out = {}
    for name in ep_names:
        z = np.load(data_dir / f"{name}.npz")
        e = {
            "X": build_features(z["qpos"], z["qvel"], z["ctrl"]),
            "qacc": z["qacc"].astype(np.float32),
            "M": rebuild_M(z["M_upper"].astype(np.float32)),
            "qfrc_bias": z["qfrc_bias"].astype(np.float32),
            "qfrc_actuator": z["qfrc_actuator"].astype(np.float32),
            "qfrc_constraint": z["qfrc_constraint"].astype(np.float32),
            "free": free_mask_of(z),
            "n_steps": int(z["qpos"].shape[0]),
        }
        for f in extra_fields:
            e[f] = z[f]
        out[name] = e
    return out


def stack(eps: dict) -> dict:
    """Concatenate episode dicts into stacked arrays (episode order kept)."""
    keys = ("X", "qacc", "M", "qfrc_bias", "qfrc_actuator", "qfrc_constraint", "free")
    return {k: np.concatenate([e[k] for e in eps.values()], axis=0) for k in keys}


def load_norm_stats(path: Path) -> dict:
    s = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.float32)
            for k, v in s.items() if isinstance(v, list)}


# --------------------------------------------------------------------- model
class AccelMLP(torch.nn.Module):
    """81 -> 3x256 (SiLU) -> 28. Shared trunk for MLP baseline and PINN."""

    def __init__(self, in_dim: int = IN_DIM, out_dim: int = NV,
                 width: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------- losses
def data_loss(a_norm_pred: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
    """(1/28) sum_i ((a_i-a_i*)/s_i)^2, averaged over the batch.

    Predictions/targets are already z-scored by s_a, so this is plain MSE
    over all elements (dims x batch) == per-dim weighted MSE in physical units.
    """
    return torch.nn.functional.mse_loss(a_norm_pred, a_norm)


def free_residual(a_phys: torch.Tensor, M: torch.Tensor, bias: torch.Tensor,
                  act: torch.Tensor) -> torch.Tensor:
    """Task 1 free-dynamics residual r = M a + qfrc_bias - qfrc_actuator.

    STAGE0_D_02 FIX: the previous implementation computed baddbmm(bias, M, a)
    == M a + qfrc_bias and silently ignored `act`, so the physics loss pushed
    predictions toward a = -M^-1 qfrc_bias (a gravity-scale target) instead of
    a = M^-1 (qfrc_actuator - qfrc_bias). `act` must be qfrc_actuator
    (28-dim generalized force, dof order; recorded by the collector from
    d.qfrc_actuator at the same mj_forward snapshot as qacc/M/bias).
    """
    return torch.baddbmm(
        (bias - act).unsqueeze(2), M, a_phys.unsqueeze(2)
    ).squeeze(2)


def free_loss(r: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
    """(1/28) sum_i (r_i/s_i)^2 averaged over the free samples in the batch."""
    return (((r / s_r) ** 2).mean(dim=1)).mean()


def to_torch(arr: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)
