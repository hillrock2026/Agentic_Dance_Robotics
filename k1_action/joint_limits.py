"""K1 joint limits read from the official XML at runtime + violation checks.

Never silently clip: raw retarget values, limited values, and per-frame
violation records are all preserved in the trajectory object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton_mapping import K1_JOINT_ORDER

K1_XML = Path(__file__).resolve().parents[1] / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"


def load_limits_from_xml(xml_path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path or K1_XML))
    out = {}
    for name in K1_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name} not in XML")
        lo, hi = model.jnt_range[jid]
        out[name] = (float(lo), float(hi))
    return out


@dataclass
class LimitReport:
    violations: list[dict]       # [{frame, joint, raw, limited, excess}]
    saturated_joints: dict       # {joint: fraction of frames at limit}
    max_excess_rad: float
    worst_joints: list

    @property
    def has_severe(self) -> bool:
        return self.max_excess_rad > 0.35  # >20 deg past limit needs review


def apply_limits(raw: dict[str, np.ndarray], limits: dict[str, tuple[float, float]],
                 margin_frac: float = 0.0) -> tuple[dict[str, np.ndarray], LimitReport]:
    """raw: {joint: (T,) radians}. Returns (limited, report).

    margin_frac shrinks the usable range (2% by default) so the trajectory
    does not sit exactly on hardware stops.
    """
    limited: dict[str, np.ndarray] = {}
    violations: list[dict] = []
    sat: dict[str, float] = {}
    max_excess = 0.0
    T = len(next(iter(raw.values())))
    for jn, arr in raw.items():
        lo, hi = limits[jn]
        span = hi - lo
        m = margin_frac * span
        lo_m, hi_m = lo + m, hi - m
        clipped = np.clip(arr, lo_m, hi_m)
        limited[jn] = clipped
        over = (arr > hi_m + 1e-9) | (arr < lo_m - 1e-9)
        for t in np.where(over)[0]:
            excess = max(arr[t] - hi_m, lo_m - arr[t])
            excess = float(min(excess, max(arr[t] - hi, lo - arr[t]) + m))
            violations.append({"frame": int(t), "joint": jn, "raw": float(arr[t]),
                               "limited": float(clipped[t]), "excess": float(excess)})
            max_excess = max(max_excess, excess)
        at_lim = (np.abs(clipped - lo_m) < 1e-6) | (np.abs(clipped - hi_m) < 1e-6)
        frac = float(at_lim.mean())
        if frac > 0.10:  # >10% of frames riding the limit
            sat[jn] = frac
    worst = sorted(sat.items(), key=lambda kv: -kv[1])[:6]
    return limited, LimitReport(violations, sat, float(max_excess), worst)
