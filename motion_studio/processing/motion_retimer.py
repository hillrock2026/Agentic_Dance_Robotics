"""Motion re-timing: speed adjustment, trimming, resampling.

Rotation channels are resampled with quaternion SLERP; translation/betas with
linear interpolation. Foot contacts are re-detected on the retimed joints so
contact intervals stay consistent with the new timeline.
"""

from __future__ import annotations

import numpy as np

from ..core.motion_data import MotionData
from ..core.rotations import aa_to_quat, quat_to_aa, slerp


def _sample_times(new_T: int, factor: float, old_T: int) -> np.ndarray:
    """Old-frame coordinates (float) for each new frame."""
    return np.minimum(np.arange(new_T) * factor, old_T - 1.0)


def _interp_rot(aa: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """aa: (T,K,3); coords: (T',) in [0,T-1] -> slerp resampled (T',K,3)."""
    q = aa_to_quat(aa)  # (T,K,4)
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, q.shape[0] - 1)
    t = (coords - i0)[:, None]  # (T',1)
    # batch slerp over K joints
    q_new = slerp(q[i0], q[i1], t)
    return quat_to_aa(q_new)


def _interp_lin(x: np.ndarray, coords: np.ndarray) -> np.ndarray:
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, x.shape[0] - 1)
    t = (coords - i0).reshape(-1, *([1] * (x.ndim - 1)))
    return (1 - t) * x[i0] + t * x[i1]


def _redetect_contacts(md: MotionData, detector) -> None:
    if detector is None:
        return
    md.contacts = detector.detect(md.joint_positions, md.motion_fps)


def set_speed(md: MotionData, factor: float, detector=None, in_place: bool = False) -> MotionData:
    """Speed change: factor>1 => faster (shorter), factor<1 => slower.

    New timeline: t_new = k/fps, old time = t_new*factor. Frame count becomes
    ~T/factor. motion_fps unchanged; time_scale bookkeeping multiplied.
    """
    if factor <= 0:
        raise ValueError("speed factor must be > 0")
    out = md if in_place else md.copy()
    T = md.frame_count
    new_T = max(2, int(round(T / factor)))
    coords = _sample_times(new_T, factor, T)

    out.global_orient = _interp_rot(md.global_orient[:, None, :], coords)[:, 0, :]
    out.body_pose = _interp_rot(md.body_pose.reshape(T, 21, 3), coords).reshape(new_T, 63)
    out.betas = _interp_lin(md.betas, coords)
    out.transl = _interp_lin(md.transl, coords)

    # joints follow the pose (exact FK would re-run; interpolation is
    # sufficient for contact redetection, but keep FK exactness for export)
    out.joint_positions = _interp_lin(md.joint_positions, coords)
    out.time_scale = float(md.time_scale * factor)
    _redetect_contacts(out, detector)
    out.record_edit("set_speed", {"factor": factor})
    return out


def trim(md: MotionData, start_frame: int, end_frame: int, detector=None, in_place: bool = False) -> MotionData:
    """Trim to [start_frame, end_frame) (half-open, frame indices)."""
    T = md.frame_count
    start_frame = max(0, int(start_frame))
    end_frame = min(T, int(end_frame))
    if end_frame - start_frame < 2:
        raise ValueError(f"trim leaves <2 frames [{start_frame},{end_frame})")
    out = md if in_place else md.copy()
    sl = slice(start_frame, end_frame)
    out.global_orient = md.global_orient[sl].copy()
    out.body_pose = md.body_pose[sl].copy()
    out.betas = md.betas[sl].copy()
    out.transl = md.transl[sl].copy()
    out.joint_positions = md.joint_positions[sl].copy()
    # shift contact intervals into the new timeline (keep raw flags sliced)
    if md.contacts:
        for side in ("left_foot", "right_foot"):
            if side in md.contacts:
                c = dict(md.contacts[side])
                c["contact"] = md.contacts[side]["contact"][sl].copy()
                c["confidence"] = md.contacts[side]["confidence"][sl].copy()
                c["contact_intervals"] = _shift_intervals(md.contacts[side]["contact_intervals"], start_frame, end_frame - 1)
                out.contacts[side] = c
        if "state" in md.contacts:
            out.contacts["state"] = md.contacts["state"][sl].copy()
    out.record_edit("trim", {"start": start_frame, "end": end_frame})
    return out


def _shift_intervals(intervals, start, last):
    shifted = []
    for s, e in intervals:
        s2, e2 = s - start, e - start
        if e2 < 0 or s2 > last:
            continue
        shifted.append([max(0, s2), min(last, e2)])
    return shifted


def resample_fps(md: MotionData, new_fps: float, detector=None, in_place: bool = False) -> MotionData:
    """Resample the motion onto a new nominal fps (same real duration)."""
    if new_fps <= 0:
        raise ValueError("fps must be > 0")
    out = md if in_place else md.copy()
    T = md.frame_count
    new_T = max(2, int(round(T * new_fps / md.motion_fps)))
    factor = md.motion_fps / new_fps  # old-frame step per new frame
    coords = _sample_times(new_T, factor, T)

    out.global_orient = _interp_rot(md.global_orient[:, None, :], coords)[:, 0, :]
    out.body_pose = _interp_rot(md.body_pose.reshape(T, 21, 3), coords).reshape(new_T, 63)
    out.betas = _interp_lin(md.betas, coords)
    out.transl = _interp_lin(md.transl, coords)
    out.joint_positions = _interp_lin(md.joint_positions, coords)
    old_fps = md.motion_fps
    out.motion_fps = float(new_fps)
    _redetect_contacts(out, detector)
    out.record_edit("resample_fps", {"from": old_fps, "to": new_fps})
    return out
