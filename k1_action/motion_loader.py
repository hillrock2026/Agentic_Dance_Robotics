"""Human motion loading + validation for the K1 conversion pipeline.

Thin, explicit layer over Motion Studio's exported clean_motion.npz so the
converter never depends on GVHMR/torch internals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class HumanMotion:
    """Loaded human motion (SMPL, MS_WORLD, meters, radians, axis-angle)."""

    path: Path
    sha256: str
    fps: float
    frame_count: int
    duration: float
    timestamps: np.ndarray                 # (T,)
    joint_positions: np.ndarray            # (T,24,3) world, meters
    global_orient: np.ndarray              # (T,3) axis-angle
    transl: np.ndarray                     # (T,3)
    body_pose: np.ndarray                  # (T,63)
    contact_state: np.ndarray | None       # (T,) int8 estimate or None
    metadata: dict = field(default_factory=dict)

    # segment direction helper used by retargeting
    def segment(self, a: str, b: str, smpl_ids: dict) -> np.ndarray:
        """(T,3) unit directions from joint a -> joint b."""
        ia, ib = smpl_ids[a], smpl_ids[b]
        v = self.joint_positions[:, ib] - self.joint_positions[:, ia]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def load_human_motion(path: str | Path) -> HumanMotion:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"human motion not found: {path}")
    z = np.load(path, allow_pickle=False)
    need = ["joint_positions", "global_orient", "transl", "body_pose",
            "motion_fps", "frame_count", "timestamps", "metadata_json"]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")
    meta = json.loads(str(z["metadata_json"]))
    return HumanMotion(
        path=path,
        sha256=sha256_of(path),
        fps=float(z["motion_fps"]),
        frame_count=int(z["frame_count"]),
        duration=float(z["duration"]),
        timestamps=z["timestamps"].astype(np.float64),
        joint_positions=z["joint_positions"].astype(np.float64),
        global_orient=z["global_orient"].astype(np.float64),
        transl=z["transl"].astype(np.float64),
        body_pose=z["body_pose"].astype(np.float64),
        contact_state=z["contact_state"].astype(np.int8) if "contact_state" in z.files else None,
        metadata=meta,
    )


def validate_human_motion(motion: HumanMotion) -> list[str]:
    """Structural/numeric checks. Returns error list (empty == valid)."""
    errs: list[str] = []
    T = motion.frame_count
    if motion.joint_positions.shape != (T, 24, 3):
        errs.append(f"joint_positions shape {motion.joint_positions.shape} != ({T},24,3)")
    if motion.body_pose.shape != (T, 63):
        errs.append(f"body_pose shape {motion.body_pose.shape} != ({T},63)")
    for name in ("joint_positions", "global_orient", "transl", "body_pose"):
        arr = getattr(motion, name)
        if not np.isfinite(arr).all():
            errs.append(f"{name}: NaN/Inf values")
    if motion.fps <= 0:
        errs.append(f"fps {motion.fps} invalid")
    ts = motion.timestamps
    if len(ts) != T or (T > 1 and abs((ts[1] - ts[0]) - 1 / motion.fps) > 1e-6):
        errs.append("timestamps not uniform at motion_fps")
    if abs(motion.duration - T / motion.fps) > 1e-6:
        errs.append("duration != frame_count/fps")
    return errs
