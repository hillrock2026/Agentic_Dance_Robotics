#!/usr/bin/env python
"""CLI: batch GVHMR recovery over a folder (or list) of videos.

Usage:
  python motion_studio/scripts/process_batch.py --videos a.mp4 b.mp4 ...
  python motion_studio/scripts/process_batch.py --folder some/dir
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.gvhmr.processor import GVHMRProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--folder", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos = list(args.videos)
    if args.folder:
        videos += sorted(str(p) for p in Path(args.folder).glob("*.mp4"))
    if not videos:
        ap.error("no videos given")

    proc = GVHMRProcessor()
    jobs = proc.process_batch(videos, overwrite=args.overwrite)
    failed = 0
    for j in jobs:
        print(f"{j['job_id']:45s} {j['status']:10s} frames={j.get('frames') or '-'}")
        if j["status"] != "COMPLETED":
            failed += 1
            print(f"    error: {j.get('error', '')[:200]}")
    print(f"\n{len(jobs) - failed}/{len(jobs)} COMPLETED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
