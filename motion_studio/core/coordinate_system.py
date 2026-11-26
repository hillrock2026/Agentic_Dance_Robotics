"""Coordinate system conversions between Motion Studio (MS_WORLD) and Blender.

MS_WORLD (canonical, from GVHMR gravity-view world):
    +Y up, right-handed, meters. Character faces +Z at identity.

Blender:
    +Z up, right-handed, meters. Default view: -Y toward viewer (character
    facing -Y appears facing the viewer... conventionally imported characters
    face +Y or -Y depending on source). We map:

        blender_pos = R_ms2blender @ ms_pos + t
        R_ms2blender: x_b = x_ms, y_b = -z_ms, z_b = y_ms
        (i.e. rotate -90° about X: Y-up -> Z-up, right-handed preserved)

    This maps MS +Y (up)     -> Blender +Z (up)
                 MS +Z (fwd) -> Blender -Y (into screen in front view)

Rotation of frame: a world rotation R_ms becomes R_bl = R_ms2blender @ R_ms @ R_ms2blender^T.
For SMPL axis-angle arrays we convert per-frame via matrices.

Sanity checks are enforced by tests: feet stay lowest, head highest, and the
character's facing direction is preserved after round trip.
"""

from __future__ import annotations

import numpy as np

from .rotations import aa_to_mat, mat_to_aa

# MS_WORLD -> Blender rotation matrix (also Blender -> MS_WORLD inverse)
_MS2BL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)
_BL2MS = _MS2BL.T

MS2BLENDER = _MS2BL
BLENDER2MS = _BL2MS


def points_ms_to_blender(points: np.ndarray) -> np.ndarray:
    """(N,3) or (T,N,3) points in MS_WORLD -> Blender coordinates."""
    return points @ MS2BLENDER.T


def points_blender_to_ms(points: np.ndarray) -> np.ndarray:
    return points @ BLENDER2MS.T


def rotmat_ms_to_blender(mats: np.ndarray) -> np.ndarray:
    """(N,3,3) world rotation matrices MS -> Blender (similarity transform)."""
    return np.einsum("ab,nbc,dc->nad", MS2BLENDER, mats, MS2BLENDER)


def rotmat_blender_to_ms(mats: np.ndarray) -> np.ndarray:
    return np.einsum("ab,nbc,dc->nad", BLENDER2MS, mats, BLENDER2MS)


def axis_angle_ms_to_blender(aa: np.ndarray) -> np.ndarray:
    """(N,3) or (T,K,3) SMPL world/root axis-angle MS -> Blender (axis rotated)."""
    shape = aa.shape
    flat = aa.reshape(-1, 3)
    out = mat_to_aa(rotmat_ms_to_blender(aa_to_mat(flat))).reshape(shape)
    return out


def axis_angle_blender_to_ms(aa: np.ndarray) -> np.ndarray:
    shape = aa.shape
    flat = aa.reshape(-1, 3)
    return mat_to_aa(rotmat_blender_to_ms(aa_to_mat(flat))).reshape(shape)
