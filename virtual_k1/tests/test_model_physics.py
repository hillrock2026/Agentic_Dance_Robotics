"""Test 3: 质量 / 惯性 / 碰撞模型物理属性。"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"


@pytest.fixture(scope="module")
def loaded():
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_total_mass_reasonable(loaded):
    m, _ = loaded
    total = float(m.body_mass.sum())
    # K1 为 ~1.3m 级别人形, 官方规格整机约 18~20 kg
    assert 15.0 < total < 25.0, f"总质量异常: {total}"


def test_no_zero_or_negative_mass_on_jointed_bodies(loaded):
    m, _ = loaded
    for bid in range(m.nbody):
        if bid == 0:
            continue  # world
        has_joint_child = any(
            m.jnt_bodyid[j] == bid for j in range(m.njnt))
        if has_joint_child:
            assert m.body_mass[bid] > 0, (
                f"body {bid} 有关节但质量为零")


def test_inertia_positive_definite(loaded):
    m, _ = loaded
    n_bad = 0
    for bid in range(1, m.nbody):
        if m.body_mass[bid] <= 0:
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[bid])
        I = R.reshape(3, 3) @ np.diag(m.body_inertia[bid]) @ R.reshape(3, 3).T
        eig = np.linalg.eigvalsh(I)
        if not np.all(eig > 0):
            n_bad += 1
    assert n_bad == 0


def test_inertia_triangle_inequality(loaded):
    """主惯量需满足 I1 + I2 >= I3 (刚体物理约束)。"""
    m, _ = loaded
    for bid in range(1, m.nbody):
        if m.body_mass[bid] <= 0:
            continue
        d = np.sort(m.body_inertia[bid])
        assert d[0] + d[1] >= d[2] * 0.999, (
            f"body {bid} 惯量三角不等式: {m.body_inertia[bid]}")


def test_symmetry_left_right(loaded):
    m, _ = loaded
    pairs = [("left_hip_pitch_link", "right_hip_pitch_link"),
             ("left_knee_pitch_link", "right_knee_pitch_link"),
             ("left_ankle_roll_link", "right_ankle_roll_link"),
             ("aaleft_shoulder_pitch_link", "aaright_shoulder_pitch_link")]
    for l, r in pairs:
        ml = float(m.body_mass[m.body(l).id])
        mr = float(m.body_mass[m.body(r).id])
        assert ml == pytest.approx(mr, rel=1e-3), f"{l}/{r} 质量不对称"


def test_collision_geometry_coverage(loaded):
    m, _ = loaded
    key_bodies = {"trunk", "left_knee_pitch_link", "right_knee_pitch_link",
                  "left_ankle_roll_link", "right_ankle_roll_link"}
    for bname in key_bodies:
        bid = m.body(bname).id
        has_col = False
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) == bid and (
                    m.geom_contype[gid] > 0 or m.geom_conaffinity[gid] > 0):
                has_col = True
                break
        assert has_col, f"{bname} 没有任何碰撞 geom"


def test_foot_collision_geom_shape(loaded):
    m, _ = loaded
    for bname in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bid = m.body(bname).id
        boxes = [gid for gid in range(m.ngeom)
                 if int(m.geom_bodyid[gid]) == bid
                 and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                 and (m.geom_contype[gid] > 0 or m.geom_conaffinity[gid] > 0)]
        assert len(boxes) == 1, f"{bname} 足底碰撞 box 数量 {len(boxes)}"
        size = m.geom_size[boxes[0]]
        # 足底 box: 约 0.18m 长 × 0.07m 宽 × 0.036m 高 (半尺寸)
        assert 0.06 < size[0] < 0.14, f"足长异常 {size}"
        assert 0.02 < size[1] < 0.06, f"足宽异常 {size}"
        assert 0.005 < size[2] < 0.04, f"足厚异常 {size}"


def test_ground_plane_exists(loaded):
    m, _ = loaded
    gid = m.geom("ground").id
    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
    assert m.geom_friction[gid][0] == pytest.approx(0.4, abs=1e-6)
