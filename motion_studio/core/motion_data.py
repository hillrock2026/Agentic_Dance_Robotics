"""MotionData: the unified human motion data container of Motion Studio.

Holds per-frame SMPL pose parameters, root translation, reconstructed joint
positions, estimated foot contacts, and a full provenance/editing metadata
record. All arrays share the same time axis (frame_count T).

Data semantics (canonical):
    body.global_orient   (T,3)   SMPL root orientation, axis-angle, MS_WORLD
    body.body_pose       (T,63)  21 body-joint local rotations, axis-angle
    body.betas           (T,10)  SMPL shape parameters (static shape, per-frame kept)
    body.transl          (T,3)   root (pelvis) translation, MS_WORLD, meters
    body.joint_positions (T,24,3) world joint positions from SMPL FK
    contacts             per-foot contact estimate (NOT force measurements)
    editing              non-destructive edit journal + current edit params
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .motion_schema import (
    COORDINATE_SYSTEM,
    PROCESSING_VERSION,
    ROTATION_REPRESENTATION,
    SCHEMA_VERSION,
    SMPL_JOINT_NAMES,
    validate_motion_dict,
)


@dataclass
class MotionData:
    # ---- provenance ----
    source_video: str | None = None
    source_hash: str | None = None        # sha256 of source video file
    source_fps: float | None = None
    motion_fps: float = 30.0
    body_model: str = "SMPL"
    model_version: str = "SMPL_NEUTRAL_v1.0.0"
    gvhmr_commit: str | None = None
    gvhmr_ckpt: str | None = None

    # ---- body ----
    global_orient: np.ndarray | None = None   # (T,3)
    body_pose: np.ndarray | None = None       # (T,63)
    betas: np.ndarray | None = None           # (T,10)
    transl: np.ndarray | None = None          # (T,3)
    joint_positions: np.ndarray | None = None  # (T,24,3)

    # ---- contacts (estimate) ----
    contacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- editing journal ----
    edit_log: list[dict[str, Any]] = field(default_factory=list)
    time_scale: float = 1.0
    amplitude_scale: float = 1.0

    # ---- extra provenance kept from GVHMR (never dropped) ----
    gvhmr_extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return int(self.global_orient.shape[0])

    @property
    def duration(self) -> float:
        return self.frame_count / self.motion_fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.frame_count) / self.motion_fps

    def metadata(self) -> dict[str, Any]:
        return {
            "source_video": self.source_video,
            "source_hash": self.source_hash,
            "source_fps": self.source_fps,
            "motion_fps": self.motion_fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "body_model": self.body_model,
            "model_version": self.model_version,
            "coordinate_system": COORDINATE_SYSTEM,
            "rotation_representation": ROTATION_REPRESENTATION,
            "units": "meters",
            "joint_names": SMPL_JOINT_NAMES,
            "schema_version": SCHEMA_VERSION,
            "processing_version": PROCESSING_VERSION,
            "gvhmr_commit": self.gvhmr_commit,
            "gvhmr_ckpt": self.gvhmr_ckpt,
            "time_scale": self.time_scale,
            "amplitude_scale": self.amplitude_scale,
            "edit_count": len(self.edit_log),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata(),
            "body": {
                "global_orient": self.global_orient,
                "body_pose": self.body_pose,
                "betas": self.betas,
                "transl": self.transl,
                "joint_positions": self.joint_positions,
            },
            "contacts": copy.deepcopy(self.contacts),
            "editing": {
                "keyframes": self.edit_log,
                "time_scale": self.time_scale,
                "amplitude_scale": self.amplitude_scale,
            },
            "gvhmr_extra": copy.deepcopy(self.gvhmr_extra),
        }

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        data = self.to_dict()
        data["metadata"]["frame_count"] = self.frame_count
        errors = validate_motion_dict(data)
        if self.joint_positions is not None and self.joint_positions.shape[0] != self.frame_count:
            errors.append(
                f"joint_positions frames {self.joint_positions.shape[0]} != pose frames {self.frame_count}"
            )
        # timeline continuity: uniform fps => times are uniform by construction,
        # but verify no NaN in derived times
        if not np.isfinite(self.times).all():
            errors.append("times contains non-finite values")
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("MotionData validation failed:\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------
    def copy(self) -> "MotionData":
        md = MotionData(
            source_video=self.source_video,
            source_hash=self.source_hash,
            source_fps=self.source_fps,
            motion_fps=self.motion_fps,
            body_model=self.body_model,
            model_version=self.model_version,
            gvhmr_commit=self.gvhmr_commit,
            gvhmr_ckpt=self.gvhmr_ckpt,
            global_orient=None if self.global_orient is None else self.global_orient.copy(),
            body_pose=None if self.body_pose is None else self.body_pose.copy(),
            betas=None if self.betas is None else self.betas.copy(),
            transl=None if self.transl is None else self.transl.copy(),
            joint_positions=None if self.joint_positions is None else self.joint_positions.copy(),
            contacts=copy.deepcopy(self.contacts),
            edit_log=copy.deepcopy(self.edit_log),
            time_scale=self.time_scale,
            amplitude_scale=self.amplitude_scale,
            gvhmr_extra=copy.deepcopy(self.gvhmr_extra),
        )
        return md

    def record_edit(self, op: str, params: dict[str, Any], joints=None, frames=None) -> None:
        self.edit_log.append(
            {
                "op": op,
                "params": {k: _jsonable(v) for k, v in params.items()},
                "joints": list(joints) if joints is not None else None,
                "frames": [int(frames[0]), int(frames[1])] if frames is not None else None,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "frame_count_after": self.frame_count,
            }
        )


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_gvhmr_raw(raw_npz_path: str | Path, smpl_fk=None, run_fk: bool = True) -> MotionData:
    """Build a MotionData from the raw NPZ produced by the GVHMR adapter.

    The adapter (run inside the GVHMR torch env) writes:
        smpl_params_global/{body_pose,betas,global_orient,transl}
        smpl_params_incam/...   (kept in gvhmr_extra, never dropped)
        K_fullimg               (kept)
        joints_global           (authoritative reference FK from official smplx)
        model_export_path       (path to smpl_neutral_core.npz)
        + metadata dict
    """
    raw_npz_path = Path(raw_npz_path)
    z = np.load(raw_npz_path, allow_pickle=False)
    required = [
        "global_orient",
        "body_pose",
        "betas",
        "transl",
        "joints_global",
        "meta_json",
    ]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"raw npz missing required fields: {missing} (files={z.files})")

    meta = json.loads(str(z["meta_json"]))
    md = MotionData(
        source_video=meta.get("source_video"),
        source_hash=meta.get("source_hash"),
        source_fps=float(meta["source_fps"]) if meta.get("source_fps") else None,
        motion_fps=float(meta.get("source_fps", 30.0)),  # frames are 1:1 with source
        body_model="SMPL",
        model_version=meta.get("body_model_version", "SMPL_NEUTRAL_v1.0.0"),
        gvhmr_commit=meta.get("gvhmr_commit"),
        gvhmr_ckpt=meta.get("gvhmr_ckpt"),
        global_orient=z["global_orient"].astype(np.float64),
        body_pose=z["body_pose"].astype(np.float64),
        betas=z["betas"].astype(np.float64),
        transl=z["transl"].astype(np.float64),
        joint_positions=z["joints_global"].astype(np.float64),
    )
    md.gvhmr_extra = {
        "video_width": meta.get("video_width"),
        "video_height": meta.get("video_height"),
        "video_frame_count": meta.get("video_frame_count"),
        "gvhmr_processing_fps": meta.get("gvhmr_processing_fps", 30.0),
        "static_cam": meta.get("static_cam"),
        "incam_params_available": True,
        "K_fullimg_shape": list(z["K_fullimg"].shape) if "K_fullimg" in z.files else None,
    }
    # optional recompute + verification of FK in numpy
    if run_fk and smpl_fk is not None:
        joints_np, _ = smpl_fk.joints_from_params(md.global_orient, md.body_pose, md.betas, md.transl)
        max_err = float(np.abs(joints_np - md.joint_positions).max())
        md.gvhmr_extra["numpy_fk_max_abs_err_m"] = max_err
        if max_err > 1e-4:
            raise ValueError(
                f"numpy SMPL FK disagrees with GVHMR reference by {max_err:.6f} m (>1e-4); "
                "coordinate/body-model assumptions are wrong — aborting."
            )
    return md
