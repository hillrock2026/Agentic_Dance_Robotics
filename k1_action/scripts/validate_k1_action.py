#!/usr/bin/env python
"""CLI: inspect / re-validate an exported K1 action directory.

  python k1_action/scripts/validate_k1_action.py --action datasets/k1_dance_v1/arm_wave
  python k1_action/scripts/inspect_k1_action.py --csv datasets/.../trajectory.csv --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from k1_action.action_exporter import load_k1_action  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, help="action dir (trajectory.csv + jsons)")
    ap.add_argument("--stats", action="store_true", help="print per-joint statistics")
    args = ap.parse_args()

    act = load_k1_action(args.action)
    meta = act["metadata"]
    print(f"action      : {meta.get('action_name')} ({meta.get('action_category')})")
    print(f"status      : {meta.get('conversion_status')}")
    print(f"frames/fps  : {meta.get('frame_count')} @ {meta.get('output_fps')} Hz "
          f"({meta.get('duration_s'):.2f} s)")
    print(f"source      : {meta.get('source_motion')}")
    print(f"retarget    : {meta.get('retargeting_config', {}).get('mode', '?')} | "
          f"quality {meta.get('retarget_quality_deg', {})}")
    val_path = Path(act["path"]).parent / "validation.json"
    if val_path.exists():
        v = json.loads(val_path.read_text())
        print(f"validation  : {v.get('status')}")
        for k, c in v.get("checks", {}).items():
            print(f"  {'PASS' if c.get('pass') else 'FAIL'}  {k}")
        if v.get("simulation"):
            for kind in ("kinematic", "physical"):
                r = v["simulation"][kind]
                print(f"  {'PASS' if r['pass'] else 'FAIL'}  simulation {kind}")
        for note in v.get("needs_review", []):
            print(f"  REVIEW: {note}")
    if args.stats:
        import numpy as np

        rows = act["rows"]
        for jn in [c for c in act["columns"] if c.endswith("_joint")]:
            a = np.array([float(r[jn]) for r in rows])
            print(f"{jn:34s} {a.min():+.3f}..{a.max():+.3f} (range {a.max()-a.min():.3f})")


if __name__ == "__main__":
    main()
