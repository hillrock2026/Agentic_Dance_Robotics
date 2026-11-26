#!/usr/bin/env python
"""CLI: build / open the Blender scene for a clean_motion.npz.

  --build : headless build of <npz> -> .blend (+ optional preview renders)
  --open  : launch Blender GUI on the .blend for interactive editing
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_cfg() -> dict:
    import yaml

    with open(PROJECT_ROOT / "motion_studio/config/motion_studio.yaml") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="clean_motion.npz to import")
    ap.add_argument("--blend", help="existing .blend to open")
    ap.add_argument("--render-frames", default=None, help="e.g. 1,40,80")
    ap.add_argument("--mode", default="both", choices=["skeleton", "mesh", "both"])
    args = ap.parse_args()

    cfg = load_cfg()
    blender = cfg["paths"]["blender"]

    if args.npz:
        blend = Path(args.npz).with_suffix(".blend")
        cmd = [blender, "--background", "--python",
               str(PROJECT_ROOT / "motion_studio/blender/import_motion.py"), "--",
               "--npz", args.npz, "--blend", str(blend), "--mode", args.mode]
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        sys.exit(r.returncode)

    if args.blend:
        if args.render_frames:
            prefix = Path(args.blend).parent / "preview"
            cmd = [blender, "--background", args.blend, "--python",
                   str(PROJECT_ROOT / "motion_studio/blender/render_frames.py"), "--",
                   "--frames", args.render_frames, "--out_prefix", str(prefix)]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        cmd = [blender, args.blend]
        subprocess.run(cmd)
        return

    ap.error("provide --npz or --blend")


if __name__ == "__main__":
    main()
