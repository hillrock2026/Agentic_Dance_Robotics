"""Rotation utilities for Motion Studio (pure numpy).

Conventions
-----------
- Axis-angle: (..., 3) vector, direction = rotation axis (unit), norm = angle in radians.
- Quaternion: (..., 4) as (w, x, y, z), scalar-first, unit norm.
- Matrix: (..., 3, 3) rotation matrices, right-handed, det=+1.

All functions are batched over leading dimensions.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> quaternion (..., 4) wxyz."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.maximum(angle, _EPS)
    half = 0.5 * angle
    w = np.cos(half)
    xyz = np.sin(half) * axis
    q = np.concatenate([w, xyz], axis=-1)
    # Near-zero angle: identity quaternion (guard against imprecise axis)
    small = (angle[..., 0] < 1e-8)
    if np.any(small):
        q[small] = np.array([1.0, 0.0, 0.0, 0.0])
    return q


def quat_to_aa(q: np.ndarray) -> np.ndarray:
    """Quaternion (..., 4) wxyz -> axis-angle (..., 3)."""
    q = _normalize_quat(np.asarray(q, dtype=np.float64))
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = q[..., 1:] / np.maximum(s[..., None], _EPS)
    aa = axis * angle[..., None]
    # Near-identity: angle ~ 0, axis irrelevant
    small = np.abs(angle) < 1e-8
    if np.any(small):
        aa[small] = 0.0
    return aa


def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3) (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)[..., 0]
    axis = aa / np.maximum(np.linalg.norm(aa, axis=-1, keepdims=True), _EPS)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    zeros = np.zeros_like(x)
    m00 = c + x * x * C
    m01 = x * y * C - z * s
    m02 = x * z * C + y * s
    m10 = y * x * C + z * s
    m11 = c + y * y * C
    m12 = y * z * C - x * s
    m20 = z * x * C - y * s
    m21 = z * y * C + x * s
    m22 = c + z * z * C
    mat = np.stack(
        [np.stack([m00, m01, m02], -1), np.stack([m10, m11, m12], -1), np.stack([m20, m21, m22], -1)], -2
    )
    small = angle < 1e-12
    if np.any(small):
        mat[small] = np.eye(3)
    return mat


def mat_to_aa(mat: np.ndarray) -> np.ndarray:
    return quat_to_aa(mat_to_quat(mat))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(np.asarray(q, dtype=np.float64))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m00 = 1 - 2 * (y * y + z * z)
    m01 = 2 * (x * y - z * w)
    m02 = 2 * (x * z + y * w)
    m10 = 2 * (x * y + z * w)
    m11 = 1 - 2 * (x * x + z * z)
    m12 = 2 * (y * z - x * w)
    m20 = 2 * (x * z - y * w)
    m21 = 2 * (y * z + x * w)
    m22 = 1 - 2 * (x * x + y * y)
    return np.stack(
        [np.stack([m00, m01, m02], -1), np.stack([m10, m11, m12], -1), np.stack([m20, m21, m22], -1)], -2
    )


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> quaternion wxyz (Shepperd's method, batched)."""
    mat = np.asarray(mat, dtype=np.float64)
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    tr = m00 + m11 + m22

    t0 = 1.0 + tr
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    q = np.empty(mat.shape[:-2] + (4,), dtype=np.float64)
    # Branch 0
    mask0 = t0 > 0
    if np.any(mask0):
        s = np.sqrt(np.maximum(t0, 0.0)) * 2  # 4w
        q[mask0, 0] = 0.25 * s[mask0]
        q[mask0, 1] = (m21 - m12)[mask0] / s[mask0]
        q[mask0, 2] = (m02 - m20)[mask0] / s[mask0]
        q[mask0, 3] = (m10 - m01)[mask0] / s[mask0]
    mask1 = (~mask0) & (t1 > t2) & (t1 > t3)
    if np.any(mask1):
        s = np.sqrt(np.maximum(t1, 0.0)) * 2  # 4x
        q[mask1, 0] = (m21 - m12)[mask1] / s[mask1]
        q[mask1, 1] = 0.25 * s[mask1]
        q[mask1, 2] = (m01 + m10)[mask1] / s[mask1]
        q[mask1, 3] = (m02 + m20)[mask1] / s[mask1]
    mask2 = (~mask0) & (~mask1) & (t2 > t3)
    if np.any(mask2):
        s = np.sqrt(np.maximum(t2, 0.0)) * 2  # 4y
        q[mask2, 0] = (m02 - m20)[mask2] / s[mask2]
        q[mask2, 1] = (m01 + m10)[mask2] / s[mask2]
        q[mask2, 2] = 0.25 * s[mask2]
        q[mask2, 3] = (m12 + m21)[mask2] / s[mask2]
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if np.any(mask3):
        s = np.sqrt(np.maximum(t3, 0.0)) * 2  # 4z
        q[mask3, 0] = (m10 - m01)[mask3] / s[mask3]
        q[mask3, 1] = (m02 + m20)[mask3] / s[mask3]
        q[mask3, 2] = (m12 + m21)[mask3] / s[mask3]
        q[mask3, 3] = 0.25 * s[mask3]
    return _normalize_quat(q)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (wxyz), batched."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return _normalize_quat(np.stack([w, x, y, z], axis=-1))


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_between(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Relative rotation q_rel such that q_from * q_rel = q_to (in world sense)."""
    return quat_mul(quat_conj(q_from), q_to)


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions.

    q0, q1: (..., 4); t: broadcastable to q0.shape[:-1] or scalar in [0,1].
    Returns shortest-path interpolation (w >= 0 hemisphere).
    """
    q0 = _normalize_quat(np.asarray(q0, dtype=np.float64))
    q1 = _normalize_quat(np.asarray(q1, dtype=np.float64))
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim == 0:
        t_arr = np.full(q0.shape[:-1], float(t_arr))
    # broadcast t to joint shape
    t_b = np.broadcast_to(t_arr, q0.shape[:-1])[..., None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # Flip to shortest path
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # close quaternions -> nlerp fallback
    close = (1.0 - dot) < 1e-6
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = t_b * theta_0
    s0 = np.cos(theta) - dot * np.sin(theta) / np.maximum(sin_theta_0, _EPS)
    s1 = np.sin(theta) / np.maximum(sin_theta_0, _EPS)
    q_slerp = s0 * q0 + s1 * q1
    q_nlerp = (1.0 - t_b) * q0 + t_b * q1
    q = np.where(close, q_nlerp, q_slerp)
    return _normalize_quat(q)


def quat_scale(q: np.ndarray, factor: float) -> np.ndarray:
    """Scale rotation magnitude of q by `factor` keeping axis (power of quaternion)."""
    return quat_to_scale_safe(q, factor)


def quat_to_scale_safe(q: np.ndarray, factor: float) -> np.ndarray:
    aa = quat_to_aa(q)
    return aa_to_quat(aa * factor)


def quat_mean(quats: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted quaternion mean via dominant eigenvector of sum w*q*q^T.

    quats: (..., N, 4) — mean over the N axis.
    """
    quats = _normalize_quat(np.asarray(quats, dtype=np.float64))
    if weights is None:
        weights = np.ones(quats.shape[-2])
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    qpos = np.where(quats[..., :1] < 0, -quats, quats)  # same hemisphere
    T = np.einsum("...nj,n,...ni->...ji", qpos, w, qpos)
    eigvals, eigvecs = np.linalg.eigh(T)
    q_mean = eigvecs[..., -1]  # largest eigenvalue
    q_mean = np.where(q_mean[..., :1] < 0, -q_mean, q_mean)
    return _normalize_quat(q_mean)


def rotate_vectors(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternion q (batched, q (...,4), v (...,3))."""
    return np.einsum("...ij,...j->...i", quat_to_mat(quat), vec)
