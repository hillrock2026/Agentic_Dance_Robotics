#!/usr/bin/env python
"""CLI: run GVHMR motion recovery on one video (project .venv python).

Usage:
  python motion_studio/scripts/process_video.py --video path/to/dance.mp4 [--moving-cam]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.gvhmr.processor import GVHMRProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--moving-cam", action="store_true", help="do not pass -s (use SimpleVO)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    proc = GVHMRProcessor()
    job = proc.process_video(args.video, static_cam=not args.moving_cam, overwrite=args.overwrite)
    status = job["status"]
    print(f"JOB {job['job_id']} -> {status}")
    if status == "COMPLETED":
        errs = proc.validate_result(job)
        print(f"  frames={job['frames']} fps={job['source_fps']}")
        print(f"  raw_npz={job['raw_npz']}")
        print(f"  validate: {'OK' if not errs else errs}")
    else:
        print(f"  error: {job.get('error')}")
        if job.get("error_log_tail"):
            print("\n".join(job["error_log_tail"][-5:]))
        sys.exit(1)


if __name__ == "__main__":
    main()
