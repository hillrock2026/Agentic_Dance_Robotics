"""Unit tests for rotation math (axis-angle / quaternion / matrix / slerp)."""

import numpy as np
import pytest

from motion_studio.core import rotations as R


def test_aa_quat_roundtrip():
    rng = np.random.default_rng(0)
    aa = rng.normal(size=(100, 3)) * 1.5
    q = R.aa_to_quat(aa)
    back = R.quat_to_aa(q)
    assert np.allclose(aa, back, atol=1e-8)


def test_aa_mat_roundtrip():
    rng = np.random.default_rng(1)
    aa = rng.normal(size=(100, 3)) * 2.0
    m = R.aa_to_mat(aa)
    back = R.mat_to_aa(m)
    # compare as rotations: aa is a non-unique chart for angles > pi
    assert np.allclose(R.aa_to_mat(back), m, atol=1e-8)
    small = np.linalg.norm(aa, axis=1) < np.pi - 0.1
    assert np.allclose(aa[small], back[small], atol=1e-6)


def test_mat_quat_roundtrip():
    rng = np.random.default_rng(2)
    aa = rng.normal(size=(100, 3)) * 2.0
    m = R.aa_to_mat(aa)
    q = R.mat_to_quat(m)
    m2 = R.quat_to_mat(q)
    assert np.allclose(m, m2, atol=1e-8)


def test_quat_norms():
    rng = np.random.default_rng(3)
    q = R._normalize_quat(rng.normal(size=(50, 4)))
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_slerp_endpoints():
    rng = np.random.default_rng(4)
    q0 = R._normalize_quat(rng.normal(size=(10, 4)))
    q1 = R._normalize_quat(rng.normal(size=(10, 4)))
    s0, s1 = R.slerp(q0, q1, 0.0), R.slerp(q0, q1, 1.0)
    # per-row sign-insensitive: rows with negative dot legitimately flip to -q
    for s, q in ((s0, q0), (s1, q1)):
        match = np.isclose(np.abs(np.sum(s * q, axis=-1)), 1.0, atol=1e-6)
        assert match.all()


def test_slerp_midpoint_half_angle():
    q0 = np.array([[1.0, 0, 0, 0.0]])
    q1 = R.aa_to_quat(np.array([[0.0, 0.0, np.pi / 2]]))
    mid = R.slerp(q0, q1, 0.5)
    expected = R.aa_to_quat(np.array([[0.0, 0.0, np.pi / 4]]))
    assert np.allclose(mid, expected, atol=1e-8) or np.allclose(mid, -expected, atol=1e-8)


def test_slerp_shortest_path():
    # q and -q represent the same rotation; slerp must not take the long way
    q0 = np.array([[1.0, 0, 0, 0.0]])
    q1 = -np.array([[0.0, 1.0, 0, 0.0]])
    q1 = q1 / np.linalg.norm(q1)
    mid = R.slerp(q0, q1, 0.5)
    # midpoint should have positive w (angle < 90deg from identity)
    assert mid[0, 0] > 0.6


def test_slerp_vs_naive_linear_diverges_but_consistent():
    """Verify slerp lies on the geodesic: constant angular speed."""
    rng = np.random.default_rng(5)
    q0 = R._normalize_quat(rng.normal(size=(1, 4)))
    q1 = R._normalize_quat(rng.normal(size=(1, 4)))
    ts = np.linspace(0, 1, 11)
    qs = np.concatenate([R.slerp(q0, q1, t) for t in ts])  # (11,4)
    angles = 2 * np.arccos(np.clip(np.abs(np.sum(qs * qs[0], axis=1)), -1, 1))
    steps = np.diff(angles)
    assert np.allclose(steps, steps[0], atol=1e-6), "slerp angular speed not constant"


def test_quat_mean_of_identical():
    q = R.aa_to_quat(np.array([[0.3, 0.2, -0.1]]))
    qs = np.tile(q, (7, 1))
    m = R.quat_mean(qs)
    assert np.allclose(m, q, atol=1e-9) or np.allclose(m, -q, atol=1e-9)


def test_quat_scale_power():
    q = R.aa_to_quat(np.array([[0.0, 0.0, 1.2]]))
    half = R.quat_scale(q, 0.5)
    expected = R.aa_to_quat(np.array([[0.0, 0.0, 0.6]]))
    assert np.allclose(half, expected, atol=1e-8)


def test_coordinate_conversion_roundtrip():
    from motion_studio.core.coordinate_system import (
        axis_angle_blender_to_ms, axis_angle_ms_to_blender,
    )

    rng = np.random.default_rng(6)
    aa = rng.normal(size=(20, 3))
    to_bl = axis_angle_ms_to_blender(aa)
    back = axis_angle_blender_to_ms(to_bl)
    # compare as rotations (aa parameterization is non-unique past pi)
    assert np.allclose(R.aa_to_mat(back), R.aa_to_mat(aa), atol=1e-8)


def test_coordinate_points_preserve_up():
    from motion_studio.core.coordinate_system import points_ms_to_blender

    up_ms = np.array([0.0, 1.0, 0.0])
    up_bl = points_ms_to_blender(up_ms[None])[0]
    assert np.allclose(up_bl, [0.0, 0.0, 1.0])  # MS +Y -> Blender +Z
