"""Coordinate transforms between the human (MS_WORLD) and K1 (MuJoCo world)
frames, calibrated at runtime from body geometry — never guessed from names.

Frames:
    MS_WORLD   : Motion Studio canonical (SMPL) — +Y up, right-handed, meters,
                 character faces +Z at identity.
    K1 world   : MuJoCo world — +Z up, right-handed, meters.
    K1 torso   : origin at trunk, +X front (verified from foot/toe geometry,
                 ~1.7° from cross(left,up)), +Y left (shoulder line), +Z up.

R_map (3x3, orthonormal): maps human-torso-relative vectors into K1-torso
directions. Human torso frame is built per-motion from the actual shoulder
line and gravity (GVHMR's world is gravity-aligned but arbitrary yaw), so
the mapping is robust to any initial facing of the performer.

Also provides the K1 <--> official-CSV root quaternion conventions
(MuJoCo wxyz vs CSV xyzw).
"""

from __future__ import annotations

import numpy as np

MS_UP = np.array([0.0, 1.0, 0.0])
K1_UP = np.array([0.0, 0.0, 1.0])


def orthonormal_frame(left_vec: np.ndarray, up_vec: np.ndarray) -> np.ndarray:
    """Columns [front, left, up] from a left direction and gravity up."""
    left = np.asarray(left_vec, dtype=np.float64).copy()
    left -= left @ up_vec * up_vec
    left /= max(np.linalg.norm(left), 1e-9)
    front = np.cross(left, up_vec)
    front /= np.linalg.norm(front)
    left2 = np.cross(up_vec, front)
    return np.column_stack([front, left2, up_vec])


def calibrate_torso_map(k1_model, k1_data, human_joints_t0: np.ndarray) -> np.ndarray:
    """R_map = F_k1 @ F_h^T built from shoulder lines of both bodies.

    human_joints_t0: (24,3) first-frame SMPL world joints (MS_WORLD).
    """
    from .skeleton_mapping import SMPL_IDS
    import mujoco

    def k1_xpos(body_name):
        bid = mujoco.mj_name2id(k1_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return k1_data.xpos[bid]

    left_k1 = k1_xpos("aaleft_shoulder_pitch_link") - k1_xpos("aaright_shoulder_pitch_link")
    F_k1 = orthonormal_frame(left_k1, K1_UP)

    jp = human_joints_t0
    left_h = jp[SMPL_IDS["left_shoulder"]] - jp[SMPL_IDS["right_shoulder"]]
    F_h = orthonormal_frame(left_h, MS_UP)
    return F_k1 @ F_h.T


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(q_xyzw, dtype=np.float64), 1)  # xyzw -> wxyz


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(q_wxyz, dtype=np.float64), -1)  # wxyz -> xyzw


def yaw_quat_xyzw(forward_k1: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion (xyzw) heading along forward_k1 (horizontal)."""
    f = np.asarray(forward_k1, dtype=np.float64)
    yaw = np.arctan2(f[1], f[0])
    q = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])  # xyzw
    return q


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Minimal quaternion slerp for xyzw single pairs (small helper; the main
    slerp lives in motion_studio.core.rotations for wxyz batches)."""
    from motion_studio.core.rotations import slerp as _slerp_wxyz

    q0w = quat_xyzw_to_wxyz(q0)[None]
    q1w = quat_xyzw_to_wxyz(q1)[None]
    out = _slerp_wxyz(q0w, q1w, np.array([t]))[0]
    return quat_wxyz_to_xyzw(out)
