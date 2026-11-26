"""Foot contact detection for recovered human motion.

IMPORTANT: this is an ESTIMATE derived from reconstructed 3D joint kinematics
(GVHMR output), not a force/pressure measurement. Confidence scores reflect
geometric plausibility only.

Detection combines, per foot:
  - foot clearance above the estimated ground (min of ankle & toe heights)
  - vertical velocity
  - horizontal speed
  - temporal hysteresis + minimum-interval filtering

All thresholds configurable via config/motion_studio.yaml `foot_contact`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FOOT_JOINTS = {"left_foot": (7, 10), "right_foot": (8, 11)}  # (ankle, toe)


@dataclass
class FootContactConfig:
    height_threshold: float = 0.05
    velocity_threshold: float = 0.30
    horizontal_velocity_threshold: float = 0.60
    min_interval_frames: int = 3
    hysteresis_off_scale: float = 2.0
    confidence_margin: float = 0.02


class FootContactDetector:
    """Stateful per-motion detector; one instance can serve many motions."""

    def __init__(self, config: dict | FootContactConfig | None = None):
        if isinstance(config, dict):
            self.cfg = FootContactConfig(**config)
        elif config is not None:
            self.cfg = config
        else:
            self.cfg = FootContactConfig()

    # ------------------------------------------------------------------
    def detect(self, joint_positions: np.ndarray, fps: float) -> dict:
        """Detect per-foot contacts.

        joint_positions: (T, 24, 3) in MS_WORLD (meters), Y up.
        Returns dict for `MotionData.contacts`:
          {side: {contact (T,) bool, confidence (T,) float,
                  contact_intervals [(s,e)], transitions int, ground_y float,
                  method: str}}
        """
        jp = np.asarray(joint_positions, dtype=np.float64)
        T = jp.shape[0]
        dt = 1.0 / float(fps)

        # --- robust ground estimate: lower envelope of both feet over time ---
        feet_y = np.concatenate(
            [jp[:, FOOT_JOINTS["left_foot"][0], 1], jp[:, FOOT_JOINTS["left_foot"][1], 1],
             jp[:, FOOT_JOINTS["right_foot"][0], 1], jp[:, FOOT_JOINTS["right_foot"][1], 1]]
        )
        ground_y = float(np.quantile(feet_y, 0.01))  # 1% quantile = deepest stance

        out: dict[str, dict] = {}
        for side, (ankle_i, toe_i) in FOOT_JOINTS.items():
            foot_pts = jp[:, [ankle_i, toe_i], :]  # (T,2,3)
            # clearance: lower of ankle/toe is what can touch ground
            height = np.min(foot_pts[:, :, 1], axis=1) - ground_y  # (T,)
            point = foot_pts.mean(axis=1)  # (T,3) representative point
            vel = self._central_diff(point) * fps
            v_y = vel[:, 1]
            v_h = np.linalg.norm(vel[:, [0, 2]], axis=1)

            conf = (
                self._soft_below(height, self.cfg.height_threshold)
                * self._soft_below(np.abs(v_y), self.cfg.velocity_threshold)
                * self._soft_below(v_h, self.cfg.horizontal_velocity_threshold)
            )
            contact = self._hysteresis(height, v_y, v_h, conf)
            intervals = self._intervals(contact)
            out[side] = {
                "contact": contact,
                "confidence": conf,
                "contact_intervals": intervals,
                "ground_y": ground_y,
                "n_transitions": int(np.sum(np.diff(contact.astype(int)) != 0)),
                "method": "height+velocity+hysteresis (geometric estimate)",
            }

        combined = self.contact_state(out, T)
        out["state"] = combined  # 0=air 1=left 2=right 3=double (T,) int8
        out["is_estimate"] = True
        out["fps"] = float(fps)
        return out

    # ------------------------------------------------------------------
    def _central_diff(self, x: np.ndarray) -> np.ndarray:
        v = np.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / 2.0
        v[0] = x[1] - x[0]
        v[-1] = x[-1] - x[-2]
        return v

    def _soft_below(self, x: np.ndarray, thr: float) -> np.ndarray:
        margin = max(self.cfg.confidence_margin, 1e-6)
        return 1.0 / (1.0 + np.exp((x - thr) / margin))

    def _hysteresis(self, height, v_y, v_h, conf) -> np.ndarray:
        T = height.shape[0]
        on_h = self.cfg.height_threshold
        on_v = self.cfg.velocity_threshold
        on_hv = self.cfg.horizontal_velocity_threshold
        s = self.cfg.hysteresis_off_scale
        contact = np.zeros(T, dtype=bool)
        state = False
        for t in range(T):
            if state:
                off = (height[t] < on_h * s) and (abs(v_y[t]) < on_v * s) and (v_h[t] < on_hv * s)
                state = off
            else:
                on = (height[t] < on_h) and (abs(v_y[t]) < on_v) and (v_h[t] < on_hv)
                state = on
            contact[t] = state
        return contact

    def _intervals(self, contact: np.ndarray) -> list[list[int]]:
        ivs: list[list[int]] = []
        start = None
        for t, c in enumerate(contact):
            if c and start is None:
                start = t
            elif not c and start is not None:
                ivs.append([int(start), int(t - 1)])
                start = None
        if start is not None:
            ivs.append([int(start), int(len(contact) - 1)])
        # drop too-short intervals
        return [iv for iv in ivs if iv[1] - iv[0] + 1 >= self.cfg.min_interval_frames]

    @staticmethod
    def contact_state(contacts: dict, T: int) -> np.ndarray:
        left = contacts["left_foot"]["contact"].astype(bool)
        right = contacts["right_foot"]["contact"].astype(bool)
        state = np.zeros(T, dtype=np.int8)
        state[left & ~right] = 1
        state[right & ~left] = 2
        state[left & right] = 3
        return state

    # ------------------------------------------------------------------
    @staticmethod
    def summarize(contacts: dict) -> dict:
        T = len(contacts["left_foot"]["contact"])
        st = contacts["state"]
        return {
            "frames": T,
            "left_contact_ratio": float(contacts["left_foot"]["contact"].mean()),
            "right_contact_ratio": float(contacts["right_foot"]["contact"].mean()),
            "double_support_ratio": float((st == 3).mean()),
            "airborne_ratio": float((st == 0).mean()),
            "left_intervals": [list(map(int, iv)) for iv in contacts["left_foot"]["contact_intervals"]],
            "right_intervals": [list(map(int, iv)) for iv in contacts["right_foot"]["contact_intervals"]],
        }
