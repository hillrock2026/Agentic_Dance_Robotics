"""Motion smoothing filter.

Sliding-window Gaussian smoothing: quaternion channels via weighted
quaternion mean (dominant eigenvector — proper SO(3) averaging, never
component-wise averaging); translation/betas via linear convolution.
"""

from __future__ import annotations

import numpy as np

from ..core.motion_data import MotionData
from ..core.rotations import quat_mean, quat_to_aa, aa_to_quat


def smooth(
    md: MotionData,
    window: int = 5,
    sigma: float = 1.0,
    smpl_fk=None,
    detector=None,
    in_place: bool = False,
) -> MotionData:
    out = md if in_place else md.copy()
    T = md.frame_count
    window = max(1, int(window) | 1)  # force odd
    half = window // 2

    w = np.exp(-0.5 * (np.arange(-half, half + 1) / max(sigma, 1e-6)) ** 2)
    w /= w.sum()

    # rotations via windowed quaternion mean
    q_body = aa_to_quat(md.body_pose.reshape(T, 21, 3))  # (T,21,4)
    q_root = aa_to_quat(md.global_orient)
    Tpad = half
    q_body_p = np.pad(q_body, ((Tpad, Tpad), (0, 0), (0, 0)), mode="edge")
    q_root_p = np.pad(q_root, ((Tpad, Tpad), (0, 0)), mode="edge")

    new_body = np.empty_like(q_body)
    for t in range(T):
        seg = q_body_p[t : t + window]  # (w,21,4)
        new_body[t] = quat_mean(np.transpose(seg, (1, 0, 2)), weights=w)  # (21,4)
    new_root = np.empty_like(q_root)
    for t in range(T):
        seg = q_root_p[t : t + window]
        new_root[t] = quat_mean(seg[:, None, :], weights=w)[0]

    out.body_pose = quat_to_aa(new_body).reshape(T, 63)
    out.global_orient = quat_to_aa(new_root)

    # linear signals
    def conv(x):
        xp = np.pad(x, ((Tpad, Tpad), (0, 0)), mode="edge")
        return np.stack([np.convolve(xp[:, c], w, mode="valid") for c in range(x.shape[1])], axis=1)

    out.transl = conv(md.transl)
    out.betas = conv(md.betas)

    if smpl_fk is not None:
        out.joint_positions, _ = smpl_fk.joints_from_params(
            out.global_orient, out.body_pose, out.betas, out.transl
        )
    if detector is not None and out.joint_positions is not None:
        out.contacts = detector.detect(out.joint_positions, out.motion_fps)

    out.record_edit("smooth", {"window": window, "sigma": sigma})
    return out
