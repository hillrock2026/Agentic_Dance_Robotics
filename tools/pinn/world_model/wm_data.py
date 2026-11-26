"""Unified state-transition data interface (K1_WM_01).

State transition definition (MuJoCo free floating base):
    x_t = (q_t, v_t),  q in R^29 (root pos 3 + quat wxyz 4 + 22 hinge),
    v in R^28 (root lin WORLD-frame 3 + root ang LOCAL-frame 3 + 22 joints),
    (q_t, v_t, ctrl_t) --Euler dt=1e-3--> (q_{t+1}, v_{t+1})
    v_{t+1} = v_t + dt * a_t,   q_{t+1} = mj_integratePos(q_t, v_{t+1}, dt)

Field availability at inference time (world model deployment):
  always available   q_t, v_t, ctrl_t, dt, episode context
  derivable          actuator_force_t == qfrc_actuator_t[6:28] == clip(ctrl_t)
                     (gear=1 torque motors, no delay; same-snapshot equality
                     verified in build_transitions)
  model-computable   M_t, qfrc_bias_t (functions of (q_t, v_t) via MuJoCo)
  training-only      a_t, qfrc_constraint_t (solver output), free/contact/
                     impact labels (require simulator knowledge)
  not applicable     act_t (na = 0: motors have no activation state)

The datasets are read-only views over stage1_data npz files; Stage1 data is
never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "tools/pinn/stage1_data"
DEFAULT_TRANSITION_DIR = Path(__file__).resolve().parent / "transitions"

STATE_FIELDS = ("q_t", "v_t", "ctrl_t", "actuator_force_t", "qfrc_actuator_t",
                "qfrc_bias_t", "qfrc_constraint_t", "M_upper_t", "a_t",
                "q_next", "v_next", "dt", "time_t", "time_next", "ncon_t",
                "nefc_t")


def _load_labels(data_dir: Path, ep_names: list[str]) -> dict:
    labels = np.load(data_dir / "stage1_labels.npz")
    out = {}
    for n in ep_names:
        out[n] = {k: labels[f"{n}__{k}"]
                  for k in ("free", "support", "transition", "impact")}
    return out


class _SplitData:
    """In-memory view of one split's episodes + per-sample labels."""

    def __init__(self, data_dir: Path, split: str):
        self.data_dir = data_dir
        self.split = split
        splits = json.loads((data_dir / "splits.json").read_text())
        self.ep_names = list(splits[split])                 # canonical order
        self.eps = {}
        self.labels = _load_labels(data_dir, self.ep_names)
        for n in self.ep_names:
            z = np.load(data_dir / f"{n}.npz")
            self.eps[n] = {k: z[k] for k in z.files}
        self.steps = [self.eps[n]["qpos"].shape[0] for n in self.ep_names]

    def episode_arrays(self, ep_index: int) -> dict:
        return self.eps[self.ep_names[ep_index]]

    def label_arrays(self, ep_index: int) -> dict:
        return self.labels[self.ep_names[ep_index]]


class StateTransitionDataset:
    """Single-step transitions (x_t, u_t) -> x_{t+1} for one split.

    Transition t of an episode pairs row t with row t+1; episodes have no
    gaps (integrity.json), so every consecutive row pair is a valid transition.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, split: str = "train",
                 include_dynamics: bool = True):
        self.data = _SplitData(Path(data_dir), split)
        self.include_dynamics = include_dynamics
        self.split = split
        self.index = [(ei, t)
                      for ei, T in enumerate(self.data.steps)
                      for t in range(T - 1)]
        self.episode_context = None   # filled by caller via context.episode_context

    def __len__(self) -> int:
        return len(self.index)

    @property
    def episode_ids(self) -> list[str]:
        return list(self.data.ep_names)

    def __getitem__(self, i: int) -> dict:
        ei, t = self.index[i]
        ep = self.data.episode_arrays(ei)
        lab = self.data.label_arrays(ei)
        row = {
            "episode_id": self.data.ep_names[ei],
            "episode_index": ei,
            "step": t,
            "split": self.split,
            "dt": float(self.data.eps[self.data.ep_names[ei]]["time"][t + 1]
                        - ep["time"][t]),
            "q_t": ep["qpos"][t], "v_t": ep["qvel"][t],
            "q_next": ep["qpos"][t + 1], "v_next": ep["qvel"][t + 1],
            "a_t": ep["qacc"][t],
            "ctrl_t": ep["ctrl"][t],
            "actuator_force_t": ep["actuator_force"][t],
            "qfrc_actuator_t": ep["qfrc_actuator"][t],
            "time_t": ep["time"][t], "time_next": ep["time"][t + 1],
            "ncon_t": int(ep["ncon"][t]), "nefc_t": int(ep["nefc"][t]),
            "free": bool(lab["free"][t]),
            "support": int(lab["support"][t]),
            "impact": bool(lab["impact"][t]),
            "transition_valid": True,
        }
        if self.include_dynamics:
            row["qfrc_bias_t"] = ep["qfrc_bias"][t]
            row["qfrc_constraint_t"] = ep["qfrc_constraint"][t]
            row["M_upper_t"] = ep["M_upper"][t].astype(np.float32)
        return row


class RolloutWindowDataset:
    """Continuous H-step windows (x_t, u_{t:t+H-1}) -> (x_{t+1}, ..., x_{t+H}).

    Windows never cross episode boundaries; within an episode every window is
    time-contiguous at dt = 1 ms and spans exactly H * dt seconds
    (H = 8 -> 8 ms). Rows are all-valid (integrity.json), so no window is
    masked out; the API still exposes per-step labels for filtering.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, split: str = "train",
                 horizon: int = 8):
        self.data = _SplitData(Path(data_dir), split)
        self.horizon = int(horizon)
        self.split = split
        self.index = [(ei, t)
                      for ei, T in enumerate(self.data.steps)
                      for t in range(T - self.horizon)]
        self.time_span_s = self.horizon * 1e-3

    def __len__(self) -> int:
        return len(self.index)

    @property
    def episode_ids(self) -> list[str]:
        return list(self.data.ep_names)

    def __getitem__(self, i: int) -> dict:
        ei, t0 = self.index[i]
        ep = self.data.episode_arrays(ei)
        lab = self.data.label_arrays(ei)
        H = self.horizon
        sl = slice(t0, t0 + H)
        return {
            "episode_id": self.data.ep_names[ei],
            "episode_index": ei,
            "t0": t0,
            "horizon": H,
            "time_span_s": self.time_span_s,
            "time_seq": ep["time"][t0:t0 + H + 1],            # (H+1,)
            "q_t": ep["qpos"][t0], "v_t": ep["qvel"][t0],
            "ctrl_seq": ep["ctrl"][sl],                        # (H,22)
            "a_seq": ep["qacc"][sl],                           # (H,28)
            "q_next_seq": ep["qpos"][t0 + 1:t0 + H + 1],       # (H,29)
            "v_next_seq": ep["qvel"][t0 + 1:t0 + H + 1],       # (H,28)
            "free_seq": lab["free"][sl],                       # (H,)
            "support_seq": lab["support"][sl],                 # (H,)
            "impact_seq": lab["impact"][sl],                   # (H,)
        }
