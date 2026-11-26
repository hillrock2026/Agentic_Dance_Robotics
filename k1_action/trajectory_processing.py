"""K1 trajectory post-processing: resampling, velocity/acceleration, stats.

Resampling: scalar joint angles interpolate linearly (all K1 joint ranges
are < pi, no wraparound); root quaternion interpolates via SLERP. Duration,
start and end times are preserved exactly.
"""

from __future__ import annotations

import numpy as np

from .coordinate_transform import slerp_quat_xyzw
from .retargeting import K1Trajectory
from .skeleton_mapping import K1_JOINT_ORDER


def resample(traj: K1Trajectory, new_fps: float) -> K1Trajectory:
    if new_fps <= 0:
        raise ValueError("fps must be > 0")
    T = traj.frame_count
    # duration semantics = frame_count / fps (matches Motion Studio): preserved
    # exactly; frame spacing is a strict 1/new_fps grid.
    new_T = max(2, int(round(T * new_fps / traj.fps)))
    new_ts = traj.timestamps[0] + np.arange(new_T) / new_fps
    old_ts = traj.timestamps

    def interp_lin(arr):  # (T,) -> (T',)
        return np.interp(new_ts, old_ts, arr)

    joints = {jn: interp_lin(a) for jn, a in traj.joints.items()}
    joints_raw = {jn: interp_lin(a) for jn, a in traj.joints_raw.items()}
    root_pos = np.stack([np.interp(new_ts, old_ts, traj.root_pos[:, k]) for k in range(3)], axis=1)
    root_quat = np.zeros((new_T, 4))
    idx = np.clip(np.searchsorted(old_ts, new_ts), 1, T - 1)
    for i, t in enumerate(new_ts):
        t0, t1 = old_ts[idx[i] - 1], old_ts[idx[i]]
        u = 0.0 if t1 == t0 else np.clip((t - t0) / (t1 - t0), 0, 1)
        root_quat[i] = slerp_quat_xyzw(traj.root_quat[idx[i] - 1], traj.root_quat[idx[i]], float(u))
    out = K1Trajectory(fps=new_fps, timestamps=new_ts, joints_raw=joints_raw,
                       joints=joints, root_pos=root_pos, root_quat=root_quat,
                       quality=traj.quality, limit_report=traj.limit_report,
                       retarget_config=traj.retarget_config)
    out.retarget_config["resampled_from_fps"] = traj.fps
    return out


def derivatives(traj: K1Trajectory) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-joint velocity and acceleration (rad/s, rad/s^2), central diff."""
    vel, acc = {}, {}
    dt = 1.0 / traj.fps
    for jn, a in traj.joints.items():
        v = np.zeros_like(a)
        v[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
        v[0], v[-1] = (a[1] - a[0]) / dt, (a[-1] - a[-2]) / dt
        c = np.zeros_like(v)
        c[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
        vel[jn], acc[jn] = v, c
    return vel, acc


def joint_stats(traj: K1Trajectory) -> dict:
    """Per-joint min/max/range/max-vel/max-acc + distribution summaries."""
    vel, acc = derivatives(traj)
    stats = {}
    for jn in K1_JOINT_ORDER:
        a, v, c = traj.joints[jn], vel[jn], acc[jn]
        stats[jn] = {
            "min_rad": float(a.min()), "max_rad": float(a.max()),
            "range_rad": float(a.max() - a.min()),
            "max_vel_rad_s": float(np.abs(v).max()),
            "max_acc_rad_s2": float(np.abs(c).max()),
            "vel_p95_rad_s": float(np.percentile(np.abs(v), 95)),
            "acc_p95_rad_s2": float(np.percentile(np.abs(c), 95)),
        }
    return stats


def max_jump(traj: K1Trajectory) -> float:
    """Largest joint-angle step between consecutive frames (rad)."""
    return float(max(np.abs(np.diff(traj.joints[jn])).max() for jn in K1_JOINT_ORDER))
