"""K1 action export/load in the internal standard (official-CSV compatible).

Layout per action:
    <dir>/trajectory.csv   frame_index, timestamp_s + official 29 columns
    <dir>/metadata.json    full provenance & conversion record
    <dir>/validation.json  produced by validators / simulation

The 29 official columns keep the official spelling & order (verified equal to
k1_fight_001.csv header), so the trajectory can be consumed by any tool that
reads the official format.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .motion_loader import sha256_of
from .retargeting import K1Trajectory
from .skeleton_mapping import K1_JOINT_ORDER, OFFICIAL_CSV_COLUMNS, TRAJECTORY_COLUMNS
from .trajectory_processing import joint_stats

SOFTWARE_VERSION = "k1_action-1.0.0"


def export_k1_action(traj: K1Trajectory, output_dir: str | Path,
                      action_name: str, action_category: str,
                      source_motion: str, source_sha256: str,
                      source_body_model: str, source_fps: float,
                      validation: dict, overwrite: bool = False) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "trajectory.csv"
    meta_path = out / "metadata.json"
    val_path = out / "validation.json"
    if not overwrite and (csv_path.exists() or meta_path.exists()):
        raise FileExistsError(f"{out} already contains an action (pass overwrite=True)")

    T = traj.frame_count
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRAJECTORY_COLUMNS)
        for t in range(T):
            row = [t, f"{traj.timestamps[t]:.6f}",
                   *[f"{traj.root_pos[t, k]:.9f}" for k in range(3)],
                   *[f"{traj.root_quat[t, k]:.9f}" for k in range(4)],
                   *[f"{traj.joints[jn][t]:.9f}" for jn in K1_JOINT_ORDER]]
            w.writerow(row)

    meta = {
        "action_name": action_name,
        "action_category": action_category,
        "source_motion": source_motion,
        "source_sha256": source_sha256,
        "source_body_model": source_body_model,
        "source_fps": source_fps,
        "output_fps": traj.fps,
        "frame_count": T,
        "duration_s": traj.duration(),
        "joint_names": K1_JOINT_ORDER,
        "joint_order": "official CSV order (head2, left arm4, right arm4, left leg6, right leg6)",
        "angle_unit": "radians (absolute joint angles)",
        "root": {"pos": "world meters", "quat": "xyzw, yaw-only retargeted"},
        "coordinate_convention": "MuJoCo K1_22dof.xml world (+Z up); CSV column names official",
        "retargeting_method": traj.retarget_config.get("method"),
        "retargeting_config": traj.retarget_config,
        "retarget_quality_deg": traj.quality,
        "dropped_human_dof": ["spine (3 joints)", "wrists", "hands", "collar"],
        "software_version": SOFTWARE_VERSION,
        "conversion_status": validation.get("status", "CONVERTED"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)

    return {"dir": str(out),
            "trajectory_csv": str(csv_path),
            "metadata_json": str(meta_path),
            "validation_json": str(val_path),
            "trajectory_sha256": sha256_of(csv_path),
            "frame_count": T}


def load_k1_action(path: str | Path) -> dict:
    """Load an exported action directory (or the trajectory.csv alone)."""
    path = Path(path)
    if path.is_dir():
        csv_path = path / "trajectory.csv"
        meta = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else {}
    else:
        csv_path = path
        meta = {}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"empty trajectory: {csv_path}")
    return {"path": str(csv_path), "metadata": meta, "rows": rows,
            "columns": list(rows[0].keys())}
