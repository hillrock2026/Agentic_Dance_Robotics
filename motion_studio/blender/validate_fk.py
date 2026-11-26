"""Blender FK validation — run inside Blender's Python:

    blender --background <motion.blend> --python motion_studio/blender/validate_fk.py -- \
        --npz <clean_motion.npz> --out <report.npz> [--step N]

Steps through the animation, reads each bone's posed head position in world
space, and compares against the npz joint_positions converted to Blender
coordinates. Writes a report npz with per-frame max error.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.core.coordinate_system import points_ms_to_blender  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args(argv)

    import bpy

    z = np.load(args.npz, allow_pickle=False)
    T = int(z["frame_count"])
    joints_ms = z["joint_positions"].astype(np.float64)
    joints_bl_ref = points_ms_to_blender(joints_ms)  # (T,24,3)

    arm_ob = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm_ob.matrix_world.identity()
    bpy.context.view_layer.update()

    names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
             "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
             "neck", "left_collar", "right_collar", "head", "left_shoulder",
             "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_hand", "right_hand"]

    frames = list(range(1, T + 1, max(1, args.step)))
    max_err_per_frame = []
    per_joint_err = np.zeros(24)
    for f in frames:
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        t = f - 1
        for i, jn in enumerate(names):
            pb = arm_ob.pose.bones.get(f"MS_{jn}")
            head_bl = np.array(arm_ob.matrix_world @ pb.matrix.translation)
            err = float(np.linalg.norm(head_bl - joints_bl_ref[t, i]))
            per_joint_err[i] = max(per_joint_err[i], err)
            max_err_per_frame.append((t, i, err))

    errs = np.array([e for _, _, e in max_err_per_frame])
    report = {
        "frames_checked": np.array([f for f, _, _ in max_err_per_frame]),
        "joint_idx": np.array([j for _, j, _ in max_err_per_frame]),
        "errors_m": errs,
        "max_err_m": errs.max(),
        "mean_err_m": errs.mean(),
        "per_joint_max_err_m": per_joint_err,
    }
    np.savez(args.out, **report)
    print(f"BLENDER_FK_VALIDATE max={errs.max():.6f}m mean={errs.mean():.6f}m frames={len(frames)}")


if __name__ == "__main__":
    main()
