#!/usr/bin/env python
"""End-to-end Motion Studio pipeline for one GVHMR job.

Steps:
  1. load raw GVHMR npz -> MotionData (with numpy FK cross-check)
  2. foot contact detection (estimate)
  3. export clean_motion.npz + .json (+ exact PC2 mesh cache)
  4. Blender: build animated .blend (skeleton + markers [+ mesh])
  5. Blender: FK validation vs npz joints (headless)
  6. optional edits via --edit operations, producing edited_<name>.npz

Usage (project .venv python):
  python motion_studio/scripts/pipeline.py --job segment_071_343ee79d [--edit edits.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.core.motion_data import MotionData, load_gvhmr_raw
from motion_studio.core.smpl_model import SMPLForwardKinematics
from motion_studio.export.mesh_cache import export_pc2, bake_motion_mesh
from motion_studio.export.npz_exporter import export_motion, load_motion, verify_roundtrip
from motion_studio.gvhmr.processor import GVHMRProcessor
from motion_studio.processing.foot_contact import FootContactDetector


def get_fk() -> SMPLForwardKinematics:
    return SMPLForwardKinematics(PROJECT_ROOT / "motion_studio/assets/smpl/smpl_neutral_core.npz")


def load_cfg() -> dict:
    import yaml

    with open(PROJECT_ROOT / "motion_studio/config/motion_studio.yaml") as f:
        return yaml.safe_load(f)


def run_blender(blender_bin: str, script: str, extra_args: list[str]) -> str:
    cmd = [str(blender_bin), "--background"]
    if script == "validate":
        cmd.append(str(last_blend))  # set by caller
    cmd += ["--python", str(script)] + ["--"] + extra_args
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(PROJECT_ROOT))
    tail = "\n".join(out.stdout.splitlines()[-8:])
    if out.returncode != 0:
        raise RuntimeError(f"blender failed:\n{tail}\n{out.stderr[-2000:]}")
    return out.stdout


last_blend: Path | None = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job_id (dir name under outputs/gvhmr)")
    ap.add_argument("--edit", default=None, help="JSON file with edit operations")
    ap.add_argument("--no-blender", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    blender_bin = cfg["paths"]["blender"]
    fk = get_fk()
    proc = GVHMRProcessor(cfg)
    job = proc.get_result(args.job)
    errs = proc.validate_result(job)
    if errs:
        raise SystemExit(f"GVHMR result invalid: {errs}")

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "motion_studio/outputs/motions" / args.job
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. MotionData + contacts
    md = load_gvhmr_raw(job["raw_npz"], smpl_fk=fk)
    fc = FootContactDetector(cfg["foot_contact"])
    md.contacts = fc.detect(md.joint_positions, md.motion_fps)

    # 3. export (+ exact PC2)
    exp = export_motion(md, out_dir / "clean_motion.npz")
    verts = bake_motion_mesh(md, fk)
    export_pc2(verts, out_dir / "clean_motion.pc2", md.motion_fps)

    # roundtrip check
    rt = verify_roundtrip(md, out_dir, atol=1e-9)
    print(f"[pipeline] roundtrip PASS={rt['PASS']} max_field_err={max(v for k,v in rt.items() if k!='PASS'):.2e}")

    # 4-5. Blender import + validation
    blend_path = out_dir / "clean_motion.blend"
    global last_blend
    last_blend = blend_path
    if not args.no_blender:
        cmd = [blender_bin, "--background", "--python",
               str(PROJECT_ROOT / "motion_studio/blender/import_motion.py"), "--",
               "--npz", str(out_dir / "clean_motion.npz"),
               "--blend", str(blend_path), "--mode", "both"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(PROJECT_ROOT))
        line = [l for l in r.stdout.splitlines() if "BLENDER_IMPORT_OK" in l]
        if r.returncode != 0 or not line:
            raise RuntimeError(f"blender import failed:\n{r.stdout[-1500:]}\n{r.stderr[-2000:]}")
        print("[pipeline]", line[0])

        rep = out_dir / "blender_fk_report.npz"
        cmd = [blender_bin, "--background", str(blend_path), "--python",
               str(PROJECT_ROOT / "motion_studio/blender/validate_fk.py"), "--",
               "--npz", str(out_dir / "clean_motion.npz"), "--out", str(rep), "--step", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(PROJECT_ROOT))
        line = [l for l in r.stdout.splitlines() if "BLENDER_FK_VALIDATE" in l]
        if r.returncode != 0 or not line:
            raise RuntimeError(f"blender validation failed:\n{r.stdout[-1500:]}\n{r.stderr[-2000:]}")
        print("[pipeline]", line[0])

    # 6. edits
    summary = {
        "job": args.job,
        "source_video": job["video"],
        "frames": md.frame_count,
        "fps": md.motion_fps,
        "duration_s": md.duration,
        "gvhmr_fk_max_err": md.gvhmr_extra.get("numpy_fk_max_abs_err_m"),
        "roundtrip": rt,
        "contacts": FootContactDetector.summarize(md.contacts),
        "exports": exp,
    }
    if args.edit:
        edited = apply_edits(md, fk, fc, args.edit)
        eexp = export_motion(edited, out_dir / "edited_motion.npz")
        summary["edited_export"] = eexp
        summary["edited"] = {
            "ops": [e["op"] for e in edited.edit_log],
            "frames": edited.frame_count,
            "duration_s": edited.duration,
            "time_scale": edited.time_scale,
            "amplitude_scale": edited.amplitude_scale,
        }
        everts = bake_motion_mesh(edited, fk)
        export_pc2(everts, out_dir / "edited_motion.pc2", edited.motion_fps)
        if not args.no_blender:
            eblend = out_dir / "edited_motion.blend"
            cmd = [blender_bin, "--background", "--python",
                   str(PROJECT_ROOT / "motion_studio/blender/import_motion.py"), "--",
                   "--npz", str(out_dir / "edited_motion.npz"),
                   "--blend", str(eblend), "--mode", "skeleton", "--name", "edited"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(PROJECT_ROOT))
            if r.returncode != 0:
                raise RuntimeError(f"edited blender import failed:\n{r.stderr[-2000:]}")
            summary["edited_blend"] = str(eblend)

    with open(out_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[pipeline] DONE {out_dir}")


def apply_edits(md: MotionData, fk, fc, edit_json: str) -> MotionData:
    from motion_studio.processing.keyframe_editor import KeyframeEditor
    from motion_studio.processing.motion_filter import smooth
    from motion_studio.processing.motion_retimer import set_speed, trim, resample_fps
    from motion_studio.processing.motion_scaler import scale_amplitude

    with open(edit_json) as f:
        ops = json.load(f)
    out = md.copy()
    for op in ops:
        kind = op["op"]
        params = {k: v for k, v in op.items() if k not in ("op", "comment")}
        if kind == "trim":
            out = trim(out, **params, detector=fc)
        elif kind == "set_speed":
            out = set_speed(out, **params, detector=fc)
        elif kind == "resample_fps":
            out = resample_fps(out, **params, detector=fc)
        elif kind == "scale_amplitude":
            out = scale_amplitude(out, **params, smpl_fk=fk, detector=fc)
        elif kind == "smooth":
            out = smooth(out, **params, smpl_fk=fk, detector=fc)
        elif kind == "keyframes":
            ke = KeyframeEditor(out, smpl_fk=fk, detector=fc)
            for kframe in params.get("add", []):
                ke.add_keyframe(kframe)
            for sp in params.get("set_pose", []):
                ke.set_keyframe_pose(
                    sp["frame"],
                    body_pose=np.asarray(sp["body_pose"], dtype=np.float64) if "body_pose" in sp else None,
                    transl=np.asarray(sp["transl"], dtype=np.float64) if "transl" in sp else None,
                    joints=sp.get("joints"),
                )
            if params.get("interpolate", False):
                ke.interpolate_between_keyframes()
        else:
            raise ValueError(f"unknown edit op {kind}")
    return out


if __name__ == "__main__":
    main()
