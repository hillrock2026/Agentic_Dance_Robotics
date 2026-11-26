"""Motion amplitude scaling.

Rotations are scaled as quaternion powers relative to a reference pose
(default: SMPL T-pose, i.e. identity local rotations) — never by linear
scaling of angle components or quaternions. A time-window blend avoids
discontinuities at range boundaries. Root translation amplitude is scaled
about its temporal mean (optional).
"""

from __future__ import annotations

import numpy as np

from ..core.motion_data import MotionData
from ..core.motion_schema import BODY_POSE_INDEX, JOINT_GROUPS
from ..core.rotations import aa_to_quat, quat_mul, quat_conj, slerp


def _window_blend(T: int, start: int | None, end: int | None, ramp: int) -> np.ndarray:
    w = np.ones(T)
    if start is None and end is None:
        return w
    s = 0 if start is None else max(0, start)
    e = T if end is None else min(T, end)
    ramp = max(1, int(ramp))
    w[:s] = 0
    w[e:] = 0
    if s + ramp <= T:
        w[s : s + ramp] = np.linspace(0, 1, ramp)
    if e - ramp >= 0:
        w[e - ramp : e] = np.linspace(1, 0, ramp)
    return w


def scale_amplitude(
    md: MotionData,
    factor: float,
    joint_group: str | list[int] = "upper_body",
    frame_range: tuple[int, int] | None = None,
    scale_translation: bool = False,
    ramp_frames: int = 5,
    smpl_fk=None,
    detector=None,
    in_place: bool = False,
) -> MotionData:
    """Scale rotation amplitude of `joint_group` by `factor` (0..inf).

    frame_range: (start, end) frames (half-open); None = whole motion.
    Rotations: q_new = q_ref * slerp(I, q_ref^-1 q, factor) — a proper
    geodesic scaling on SO(3).
    """
    if factor < 0:
        raise ValueError("amplitude factor must be >= 0")
    out = md if in_place else md.copy()
    T = md.frame_count
    joint_ids = JOINT_GROUPS[joint_group] if isinstance(joint_group, str) else list(joint_group)

    w = _window_blend(T, frame_range[0] if frame_range else None,
                      frame_range[1] if frame_range else None, ramp_frames)

    # body_pose scaling (local joints; reference = T-pose identity)
    bp = md.body_pose.reshape(T, 21, 3)
    q = aa_to_quat(bp)  # (T,21,4)
    q_new = q.copy()
    # effective factor blends 1.0 (unchanged) -> factor (full edit) across window
    eff = 1.0 + w * (factor - 1.0)
    for jid in joint_ids:
        if jid == 0:
            continue
        k = BODY_POSE_INDEX.get(jid)
        if k is None:
            continue
        qk = q[:, k].copy()
        # canonicalize to w>=0 hemisphere so the principal axis-angle angle
        # equals the true rotation angle (q and -q are the same rotation)
        neg = qk[:, 0] < 0
        qk[neg] *= -1.0
        # quaternion power q^eff: axis kept, angle scaled (geodesic toward T-pose)
        scaled_aa = quat_to_aa_safe(qk) * eff[:, None]
        q_new[:, k] = aa_to_quat(scaled_aa)
    out.body_pose = quat_to_aa_safe(q_new).reshape(T, 63)

    # global orientation: excluded by default (world heading stays honest)

    if scale_translation:
        center = md.transl.mean(axis=0, keepdims=True)
        out.transl = center + (md.transl - center) * (1.0 + (factor - 1.0) * w)[:, None]

    if smpl_fk is not None:
        out.joint_positions, _ = smpl_fk.joints_from_params(
            out.global_orient, out.body_pose, out.betas, out.transl
        )
    if detector is not None and out.joint_positions is not None:
        out.contacts = detector.detect(out.joint_positions, out.motion_fps)

    out.amplitude_scale = float(md.amplitude_scale * factor)
    out.record_edit(
        "scale_amplitude",
        {"factor": factor, "group": joint_group if isinstance(joint_group, str) else "custom",
         "frame_range": list(frame_range) if frame_range else None},
        joints=joint_ids,
        frames=frame_range,
    )
    return out


def quat_to_aa_safe(q):
    from ..core.rotations import quat_to_aa

    return quat_to_aa(q)
