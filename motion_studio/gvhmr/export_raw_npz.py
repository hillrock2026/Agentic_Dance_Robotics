#!/usr/bin/env python
"""GVHMR result adapter — MUST run inside the GVHMR conda environment
(torch + smplx + hmr4d) from the GVHMR repo root.

Converts the official `hmr4d_results.pt` into a portable `gvhmr_raw.npz`:

  - smpl_params_global: body_pose (T,63), betas (T,10), global_orient (T,3),
    transl (T,3)   [SMPL neutral, axis-angle, gravity-aligned world, meters]
  - smpl_params_incam: same fields in camera frame (preserved, never dropped)
  - K_fullimg: (T,3,3) camera intrinsics
  - joints_global: (T,24,3) authoritative joint positions computed with the
    OFFICIAL smplx SMPL (neutral) forward pass — used as FK ground truth by
    Motion Studio's pure-numpy FK for cross-validation.
  - meta_json: provenance (source video hash/fps/size, gvhmr commit, ckpt,
    processing fps assumption, static_cam flag)

Also exports (once) the SMPL neutral model arrays to
`motion_studio/assets/smpl/smpl_neutral_core.npz` so the core environment can
run FK without torch/chumpy.

Usage:
  python export_raw_npz.py --result_dir <dir-with-hmr4d_results.pt> \
      --source_video <original.mp4> [--static_cam] [--model_export <path>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(video: Path) -> dict:
    from hmr4d.utils.video_io_utils import get_video_lwh

    length, width, height = get_video_lwh(video)
    fps = _fps(video)
    return {
        "frame_count": int(length),
        "width": int(width),
        "height": int(height),
        "fps": fps,
    }


def _fps(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 25.0


def export_smpl_core(model_export: Path) -> None:
    """Dump SMPL neutral arrays for pure-numpy FK in the core environment."""
    import pickle

    from hmr4d import PROJ_ROOT

    pkl = PROJ_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    with open(pkl, "rb") as f:
        m = pickle.load(f, encoding="latin1")

    def _a(x):
        return np.asarray(x, dtype=np.float64)

    model_export.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_export,
        v_template=_a(m["v_template"]),                        # (V,3)
        shapedirs=_a(np.asarray(m["shapedirs"]))[:, :, :10],   # (V,3,10)
        posedirs=_a(m["posedirs"]),                            # (V,3,207)
        J_regressor=_a(m["J_regressor"].toarray()),            # (24,V)
        weights=_a(m["weights"]),                              # (V,24)
        parents=np.asarray(m["kintree_table"][0], dtype=np.int64),
        faces=np.asarray(m["f"], dtype=np.int64),
        joint_names=np.array(
            [
                "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
                "neck", "left_collar", "right_collar", "head", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hand", "right_hand",
            ]
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True, type=Path)
    ap.add_argument("--source_video", required=True, type=Path)
    ap.add_argument("--static_cam", action="store_true")
    ap.add_argument("--model_export", type=Path, default=None)
    ap.add_argument("--out_name", default="gvhmr_raw.npz")
    args = ap.parse_args()

    result_pt = args.result_dir / "hmr4d_results.pt"
    assert result_pt.exists(), f"missing {result_pt}"
    out_npz = args.result_dir / args.out_name

    pred = torch.load(result_pt, map_location="cpu")

    # --- authoritative joint positions with the official body model ---
    from hmr4d import PROJ_ROOT
    from hmr4d.utils.smplx_utils import make_smplx

    smpl_model = make_smplx("smpl")
    with torch.no_grad():
        params = {k: v.float() for k, v in pred["smpl_params_global"].items()}
        # GVHMR outputs 21 body joints (63); smplx SMPL expects 23 (69) incl.
        # neutral hand joints — pad with zeros for the official FK reference.
        T = params["body_pose"].shape[0]
        params["body_pose"] = torch.cat(
            [params["body_pose"], torch.zeros(T, 6)], dim=1
        )
        out = smpl_model(return_verts=False, **params)
    joints_global = out.joints[:, :24].numpy()  # (T,24,3)

    # --- video provenance ---
    probe = probe_video(args.source_video)
    T = int(pred["smpl_params_global"]["body_pose"].shape[0])
    if probe["frame_count"] not in (T, T + 1):  # tolerate off-by-one from demuxer
        raise ValueError(f"frame count mismatch: video {probe['frame_count']} vs motion {T}")

    # --- gvhmr repo commit ---
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        commit = None

    meta = {
        "source_video": str(args.source_video),
        "source_hash": sha256_of(args.source_video),
        "source_fps": probe["fps"],
        "source_frame_count": probe["frame_count"],
        "video_width": probe["width"],
        "video_height": probe["height"],
        "gvhmr_commit": commit,
        "gvhmr_ckpt": "gvhmr_siga24_release.ckpt",
        "gvhmr_processing_fps": 30.0,  # GVHMR demo assumes 30fps internally
        "static_cam": bool(args.static_cam),
        "body_model": "SMPL",
        "body_model_version": "SMPL_NEUTRAL.pkl (smplx package)",
        "frame_count": T,
        "joint_order": "smpl24",
    }

    g = pred["smpl_params_global"]
    i = pred["smpl_params_incam"]
    np.savez(
        out_npz,
        global_orient=g["global_orient"].numpy().astype(np.float32),
        body_pose=g["body_pose"].numpy().astype(np.float32),
        betas=g["betas"].numpy().astype(np.float32),
        transl=g["transl"].numpy().astype(np.float32),
        joints_global=joints_global.astype(np.float32),
        incam_global_orient=i["global_orient"].numpy().astype(np.float32),
        incam_body_pose=i["body_pose"].numpy().astype(np.float32),
        incam_betas=i["betas"].numpy().astype(np.float32),
        incam_transl=i["transl"].numpy().astype(np.float32),
        K_fullimg=pred["K_fullimg"].numpy().astype(np.float32),
        meta_json=json.dumps(meta),
    )

    if args.model_export is not None:
        export_smpl_core(args.model_export)

    print(f"RAW_EXPORT_OK {out_npz} frames={T} joints_shape={joints_global.shape}")


if __name__ == "__main__":
    main()
