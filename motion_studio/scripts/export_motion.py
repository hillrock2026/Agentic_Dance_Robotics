#!/usr/bin/env python
"""CLI: export a completed GVHMR job to standardized clean_motion.npz.

Usage:
  python motion_studio/scripts/export_motion.py --job <job_id> [--out_dir DIR]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.core.motion_data import load_gvhmr_raw  # noqa: E402
from motion_studio.core.smpl_model import SMPLForwardKinematics  # noqa: E402
from motion_studio.export.mesh_cache import bake_motion_mesh, export_pc2  # noqa: E402
from motion_studio.export.npz_exporter import export_motion, load_motion  # noqa: E402
from motion_studio.gvhmr.processor import GVHMRProcessor  # noqa: E402
from motion_studio.processing.foot_contact import FootContactDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    fk = SMPLForwardKinematics(PROJECT_ROOT / "motion_studio/assets/smpl/smpl_neutral_core.npz")
    proc = GVHMRProcessor()
    job = proc.get_result(args.job)
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "motion_studio/outputs/motions" / args.job
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    md = load_gvhmr_raw(job["raw_npz"], smpl_fk=fk)
    md.contacts = FootContactDetector().detect(md.joint_positions, md.motion_fps)
    exp = export_motion(md, out_dir / "clean_motion.npz")
    export_pc2(bake_motion_mesh(md, fk), out_dir / "clean_motion.pc2", md.motion_fps)

    # independent reload check
    loaded = load_motion(exp["npz"])
    errs = loaded.validate()
    print(f"exported {exp['npz']}")
    print(f"reload validation: {'OK' if not errs else errs}")
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
