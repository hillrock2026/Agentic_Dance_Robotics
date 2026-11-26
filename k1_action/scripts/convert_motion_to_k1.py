#!/usr/bin/env python
"""CLI: convert a Motion Studio clean_motion.npz into a K1 action.

  python k1_action/scripts/convert_motion_to_k1.py \
      --npz motion_studio/outputs/motions/<job>/edited_motion.npz \
      --name arm_wave --category arm_wave \
      --out datasets/k1_dance_v1/arm_wave \
      [--upper-body-mode] [--fps 50] [--simulate] [--render]

Pipeline: load -> validate -> retarget -> resample -> limit-check ->
export (trajectory.csv/metadata.json/validation.json) -> optional Virtual K1
replay (A kinematic + B physical) recorded back into validation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from k1_action.action_exporter import export_k1_action, load_k1_action  # noqa: E402
from k1_action.motion_loader import load_human_motion, validate_human_motion  # noqa: E402
from k1_action.retargeting import FullBodyRetargeter  # noqa: E402
from k1_action.trajectory_processing import resample  # noqa: E402
from k1_action.validators import validate_k1_trajectory  # noqa: E402


def convert(npz_path: str, name: str, category: str, out_dir: str,
            fps: float = 50.0, upper_body_mode: bool = False,
            simulate: bool = False, render: bool = False, overwrite: bool = False) -> dict:
    hm = load_human_motion(npz_path)
    errs = validate_human_motion(hm)
    if errs:
        raise ValueError(f"human motion invalid: {errs}")

    retargeter = FullBodyRetargeter()
    traj = retargeter.retarget(hm, upper_body_mode=upper_body_mode)
    traj = resample(traj, fps)

    validation = validate_k1_trajectory(traj)
    validation["human_input_checks"] = {"pass": True, "source": str(hm.path)}

    status = validation["status"]
    exp = export_k1_action(traj, out_dir, action_name=name, action_category=category,
                           source_motion=str(hm.path), source_sha256=hm.sha256,
                           source_body_model="SMPL", source_fps=hm.fps,
                           validation=validation, overwrite=overwrite)

    if simulate:
        import numpy as np

        from k1_action.virtual_k1_replay import kinematic_replay, physical_replay

        contact = None
        if hm.contact_state is not None:
            old_idx = np.arange(len(hm.contact_state)) / hm.fps
            new_idx = np.arange(traj.frame_count) / traj.fps
            contact = np.interp(new_idx, old_idx, hm.contact_state.astype(float)).round().astype(np.int8)
        a = kinematic_replay(traj, contact)
        b = physical_replay(traj, render=render,
                            render_dir=Path(out_dir) / "replay",
                            contact_state=contact)
        validation["simulation"] = {"kinematic": a, "physical": b}
        if a["pass"] and b["pass"] and status == "VALIDATED":
            validation["status"] = "SIMULATION_VALIDATED"
        elif validation["status"] not in ("FAILED",):
            if validation["status"] == "VALIDATED" and not (a["pass"] and b["pass"]):
                validation["status"] = "NEEDS_REVIEW"
        # rewrite validation + metadata status
        val_path = Path(out_dir) / "validation.json"
        val_path.write_text(json.dumps(validation, indent=2, default=str))
        meta_path = Path(out_dir) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["conversion_status"] = validation["status"]
        meta_path.write_text(json.dumps(meta, indent=2))

    exp["status"] = validation["status"]
    exp["quality_deg"] = {k: round(v, 1) for k, v in traj.quality.items()}
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--upper-body-mode", action="store_true",
                    help="legs hold official stance (for upper-body-led motions)")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    r = convert(args.npz, args.name, args.category, args.out, args.fps,
                args.upper_body_mode, args.simulate, args.render, args.overwrite)
    print(json.dumps({k: v for k, v in r.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
