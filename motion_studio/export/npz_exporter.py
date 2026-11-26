"""Standardized clean_motion export/load.

Export layout (clean_motion.npz):
    # time
    motion_fps, frame_count, duration, timestamps (T,)
    # body (SMPL, axis-angle, MS_WORLD, meters)
    global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3),
    joint_positions (T,24,3)
    # foot contact ESTIMATE
    contact_left (T,) bool, contact_right (T,) bool,
    contact_conf_left (T,), contact_conf_right (T,),
    contact_state (T,) int8, contact_intervals_json str
    # definitions
    joint_names (24,) str, parents (24,) int, coordinate_json str,
    rotation_representation str, schema_version str
    # provenance / editing
    metadata_json str (full provenance + edit journal)

A human-readable clean_motion.json is written alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.motion_data import MotionData
from ..core.motion_schema import (
    COORDINATE_SYSTEM,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
)


def export_motion(motion_data: MotionData, output_path: str | Path, json_path: str | Path | None = None) -> dict:
    md = motion_data
    md.assert_valid()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")

    meta = md.metadata()
    contacts = md.contacts or {}
    left = contacts.get("left_foot", {})
    right = contacts.get("right_foot", {})
    intervals = {
        "left_foot": [list(map(int, iv)) for iv in left.get("contact_intervals", [])],
        "right_foot": [list(map(int, iv)) for iv in right.get("contact_intervals", [])],
    }
    arrs = dict(
        motion_fps=np.float64(md.motion_fps),
        frame_count=np.int64(md.frame_count),
        duration=np.float64(md.duration),
        timestamps=md.times,
        global_orient=md.global_orient.astype(np.float64),
        body_pose=md.body_pose.astype(np.float64),
        betas=md.betas.astype(np.float64),
        transl=md.transl.astype(np.float64),
        joint_positions=md.joint_positions.astype(np.float64),
        contact_left=np.asarray(left.get("contact", np.zeros(md.frame_count, bool))),
        contact_right=np.asarray(right.get("contact", np.zeros(md.frame_count, bool))),
        contact_conf_left=np.asarray(left.get("confidence", np.zeros(md.frame_count))),
        contact_conf_right=np.asarray(right.get("confidence", np.zeros(md.frame_count))),
        contact_state=np.asarray(contacts.get("state", np.zeros(md.frame_count, np.int8)), dtype=np.int8),
        contact_intervals_json=json.dumps(intervals),
        joint_names=np.array(SMPL_JOINT_NAMES),
        parents=np.array(SMPL_PARENTS, dtype=np.int64),
        coordinate_json=json.dumps(COORDINATE_SYSTEM),
        rotation_representation=np.array("smpl_axis_angle"),
        schema_version=np.array(meta["schema_version"]),
        metadata_json=json.dumps(meta, default=str),
    )
    if md.source_fps is not None:
        arrs["source_fps"] = np.float64(md.source_fps)
    np.savez(output_path, **arrs)

    if json_path is None:
        json_path = output_path.with_suffix(".json")
    structure_doc = {
        "file": str(output_path),
        "metadata": meta,
        "contacts": {
            "note": "ESTIMATE from reconstructed 3D kinematics (height+velocity+hysteresis); NOT force measurements",
            "intervals_frames": intervals,
            "state_legend": {"0": "both feet in air", "1": "left support", "2": "right support", "3": "double support"},
            "method": left.get("method", "n/a"),
        },
        "array_shapes": {k: list(v.shape) for k, v in arrs.items() if hasattr(v, "shape")},
        "units": "meters",
        "coordinate_system": COORDINATE_SYSTEM,
        "edit_journal": md.edit_log,
        "gvhmr_extra": md.gvhmr_extra,
    }
    with open(json_path, "w") as f:
        json.dump(structure_doc, f, indent=2, default=str)
    return {"npz": str(output_path), "json": str(json_path), "frames": md.frame_count}


def load_motion(input_path: str | Path) -> MotionData:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    z = np.load(input_path, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    md = MotionData(
        source_video=meta.get("source_video"),
        source_hash=meta.get("source_hash"),
        source_fps=float(z["source_fps"]) if "source_fps" in z.files else meta.get("source_fps"),
        motion_fps=float(z["motion_fps"]),
        body_model=meta.get("body_model", "SMPL"),
        model_version=meta.get("model_version"),
        gvhmr_commit=meta.get("gvhmr_commit"),
        gvhmr_ckpt=meta.get("gvhmr_ckpt"),
        global_orient=z["global_orient"].astype(np.float64),
        body_pose=z["body_pose"].astype(np.float64),
        betas=z["betas"].astype(np.float64),
        transl=z["transl"].astype(np.float64),
        joint_positions=z["joint_positions"].astype(np.float64),
        time_scale=meta.get("time_scale", 1.0),
        amplitude_scale=meta.get("amplitude_scale", 1.0),
    )
    md.edit_log = structure_doc_edit_journal(input_path)
    contacts = {
        "left_foot": {
            "contact": z["contact_left"].astype(bool),
            "confidence": z["contact_conf_left"].astype(np.float64),
            "contact_intervals": [list(iv) for iv in json.loads(str(z["contact_intervals_json"]))["left_foot"]],
        },
        "right_foot": {
            "contact": z["contact_right"].astype(bool),
            "confidence": z["contact_conf_right"].astype(np.float64),
            "contact_intervals": [list(iv) for iv in json.loads(str(z["contact_intervals_json"]))["right_foot"]],
        },
        "state": z["contact_state"].astype(np.int8),
        "is_estimate": True,
        "fps": float(z["motion_fps"]),
    }
    md.contacts = contacts
    return md


def structure_doc_edit_journal(npz_path: Path) -> list:
    jp = npz_path.with_suffix(".json")
    if jp.exists():
        try:
            with open(jp) as f:
                return json.load(f).get("edit_journal", [])
        except Exception:
            return []
    return []


def validate_motion(motion_data: MotionData | str | Path) -> list[str]:
    """Schema-validate either a MotionData or an exported npz (roundtrip)."""
    if isinstance(motion_data, (str, Path)):
        motion_data = load_motion(motion_data)
    return motion_data.validate()


def verify_roundtrip(motion_data: MotionData, tmp_path: str | Path, atol: float = 1e-12) -> dict:
    """Export -> load -> compare all arrays. Returns per-field max abs errors."""
    import tempfile

    from ..core.motion_data import MotionData as MD

    tmp = Path(tmp_path)
    exp = export_motion(motion_data, tmp / "roundtrip_check.npz")
    loaded = load_motion(exp["npz"])
    report = {}
    for f in ("global_orient", "body_pose", "betas", "transl", "joint_positions"):
        a = np.asarray(getattr(motion_data, f))
        b = np.asarray(getattr(loaded, f))
        report[f] = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
    report["contact_left"] = float(
        (motion_data.contacts["left_foot"]["contact"] != loaded.contacts["left_foot"]["contact"]).mean()
    )
    report["contact_right"] = float(
        (motion_data.contacts["right_foot"]["contact"] != loaded.contacts["right_foot"]["contact"]).mean()
    )
    report["motion_fps"] = abs(motion_data.motion_fps - loaded.motion_fps)
    report["timestamps"] = float(np.abs(motion_data.times - loaded.times).max())
    ok = all(v <= atol for v in report.values())
    report["PASS"] = ok
    return report
