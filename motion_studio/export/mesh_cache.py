"""PC2 point-cache export of the exact SMPL mesh sequence.

Writes the full LBS-skinned mesh (incl. pose blendshapes) per frame so that
Blender can play back the EXACT GVHMR mesh via the Mesh Cache modifier —
independent of armature approximation.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PC2_SIGNATURE = b"POINTCACHE2\0"


def export_pc2(vertices: np.ndarray, path: str | Path, fps: float, start_frame: int = 1) -> dict:
    """vertices: (T, V, 3) float — mesh per frame (MS_WORLD, meters)."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    T, V, _ = vertices.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(PC2_SIGNATURE)
        f.write(struct.pack("<i", 1))            # file_version
        f.write(struct.pack("<i", V))            # num_points
        f.write(struct.pack("<f", float(start_frame)))
        f.write(struct.pack("<f", float(fps)))   # sample_rate (samples per frame)
        f.write(struct.pack("<i", T))            # num_samples
        f.write(vertices.tobytes())  # little-endian float32 xyz
    return {"path": str(path), "frames": T, "vertices": V, "fps": fps}


def bake_motion_mesh(md, smpl_fk) -> np.ndarray:
    """Exact per-frame SMPL vertices (T,V,3) via numpy LBS."""
    joints, verts = smpl_fk.joints_from_params(
        md.global_orient, md.body_pose, md.betas, md.transl, return_vertices=True
    )
    assert verts is not None and verts.shape[1] > 100, "vertex baking failed"
    return verts
