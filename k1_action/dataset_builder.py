"""K1 training dataset assembly: manifest + per-action status tracking.

Status ladder (never auto-upgraded):
    CONVERTED -> VALIDATED (validators) -> SIMULATION_VALIDATED (Virtual K1)
    NEEDS_REVIEW / FAILED at any stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from .motion_loader import sha256_of
from .skeleton_mapping import K1_JOINT_ORDER

DATASET_VERSION = "k1_dance_v1"


def build_k1_training_dataset(actions: list[dict], output_path: str | Path,
                              overwrite: bool = False) -> dict:
    """actions: [{name, category, dir, status}] — dirs contain the exported
    trajectory/metadata/validation produced by export_k1_action."""
    out = Path(output_path)
    manifest_path = out / "manifest.json"
    if not overwrite and manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (pass overwrite=True)")
    out.mkdir(parents=True, exist_ok=True)

    entries = []
    for a in actions:
        d = Path(a["dir"])
        traj_csv = d / "trajectory.csv"
        entry = {
            "action_name": a["name"],
            "action_category": a["category"],
            "path": str(d),
            "trajectory_csv": str(traj_csv),
            "trajectory_sha256": sha256_of(traj_csv),
            "metadata_sha256": sha256_of(d / "metadata.json"),
            "status": a["status"],
        }
        entries.append(entry)

    manifest = {
        "dataset": DATASET_VERSION,
        "dataset_sha_note": "per-file sha256 recorded on each entry",
        "action_count": len(entries),
        "categories": sorted({a["category"] for a in actions}),
        "joint_order": K1_JOINT_ORDER,
        "angle_unit": "radians",
        "sampling_rates_hz": sorted({json.loads((Path(a['dir']) / 'metadata.json').read_text())["output_fps"]
                                     for a in actions}),
        "status_definitions": {
            "CONVERTED": "format conversion done",
            "VALIDATED": "passed validators (limits/dynamics/timeline/finite)",
            "SIMULATION_VALIDATED": "passed Virtual K1 replay checks",
            "NEEDS_REVIEW": "flagged issues, manual inspection required",
            "FAILED": "conversion or validation failed",
        },
        "actions": entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
