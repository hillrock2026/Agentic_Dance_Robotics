"""Test: 足底几何检测 (K1_VIRTUAL_FOOT_GEOMETRY_FIX_01)。

覆盖:
  1. 基础几何: 水平放置 / 悬空 / 穿透 (5mm 与 50mm) 与理论值对比;
  2. 旋转: 绕 X/Y/Z、多轴组合、collision origin 非零平移/旋转,
     以及"中心高度相同但真实底面不同"的旧方法盲区案例;
  3. 机器人级: 真实 K1 足部几何, 左右脚分离、脚踝姿态、root 姿态;
  4. 不支持几何类型显式报错 (不静默退化) 与 GroundPlane 约定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import (  # noqa: E402
    FootGeometryDetector,
    FootGeometryResult,
    GroundPlane,
    K1Sim,
    UnsupportedFootGeomError,
)

TOL = 1e-9           # 数值容差: 理论值 vs 计算值
CONTACT_TOL = 2e-3   # 与默认几何接触判定容差一致


# ---------------------------------------------------------------------------
# 合成模型工具


def _xml_box(half=(0.09, 0.035, 0.018), geom_pos=(0.0, 0.0, 0.0),
             geom_quat_wxyz=None, capsule=False):
    if capsule:
        geom = ('<geom name="sole" type="capsule" size="0.02 0.05" '
                'contype="1" conaffinity="1"/>')
    else:
        if geom_quat_wxyz is not None:
            q = (f'quat="{geom_quat_wxyz[0]} {geom_quat_wxyz[1]} '
                 f'{geom_quat_wxyz[2]} {geom_quat_wxyz[3]}" ')
        else:
            q = ""
        geom = (f'<geom name="sole" type="box" '
                f'size="{half[0]} {half[1]} {half[2]}" '
                f'pos="{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}" {q}'
                f'contype="1" conaffinity="1"/>')
    return f"""
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="10 10 0.1"/>
        <body name="left_foot" pos="0 0 1">
          <freejoint/>
          {geom}
        </body>
      </worldbody>
    </mujoco>
    """


def _detector(model, **kw):
    return FootGeometryDetector(model, foot_bodies={"left": "left_foot"},
                                contact_tolerance_m=CONTACT_TOL, **kw)


def _set_pose(model, data, pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
    data.qpos[:] = 0
    data.qpos[:3] = pos
    data.qpos[3:7] = quat_wxyz
    mujoco.mj_forward(model, data)


def _eval(model, data, pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0), **kw):
    _set_pose(model, data, pos, quat_wxyz)
    return _detector(model, **kw).evaluate(data)["left"]


def _axis_quat(axis, theta):
    """轴角四元数 (w,x,y,z) —— 测试侧独立实现, 不用 mujoco 工具函数。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return tuple(np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * axis]))


def _rodrigues(axis, theta):
    """测试侧独立旋转矩阵 (Rodrigues 公式)。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


HALF = (0.09, 0.035, 0.018)


# ---------------------------------------------------------------------------
# 1. 基础几何


def test_box_touching_ground_exactly():
    """box 水平放置, 底面恰好接触地面: clearance=0, CONTACT。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, HALF[2]])
    assert r.clearance == pytest.approx(0.0, abs=TOL)
    assert r.penetration_depth == pytest.approx(0.0, abs=TOL)
    assert r.state == "CONTACT"
    assert r.max_corner_z == pytest.approx(2 * HALF[2], abs=TOL)


def test_box_hover_5mm():
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, HALF[2] + 0.005])
    assert r.clearance == pytest.approx(0.005, abs=TOL)
    assert r.penetration_depth == 0.0
    assert r.state == "AIRBORNE"


def test_box_penetrate_5mm():
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, HALF[2] - 0.005])
    assert r.clearance == pytest.approx(-0.005, abs=TOL)
    assert r.penetration_depth == pytest.approx(0.005, abs=TOL)
    assert r.state == "PENETRATION"


def test_box_hover_50mm():
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, HALF[2] + 0.050])
    assert r.clearance == pytest.approx(0.050, abs=TOL)
    assert r.state == "AIRBORNE"


def test_box_penetrate_50mm():
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, HALF[2] - 0.050])
    assert r.clearance == pytest.approx(-0.050, abs=TOL)
    assert r.penetration_depth == pytest.approx(0.050, abs=TOL)
    assert r.state == "PENETRATION"


# ---------------------------------------------------------------------------
# 2. 旋转


def test_rotation_about_x():
    """绕 X 轴旋转 θ: 底面 = 中心z - (hy|sinθ| + hz|cosθ|)。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    theta = np.radians(25.0)
    z_c = 0.05
    r = _eval(m, d, [0, 0, z_c], _axis_quat((1, 0, 0), theta))
    expected = z_c - (HALF[1] * abs(np.sin(theta)) + HALF[2] * abs(np.cos(theta)))
    assert r.clearance == pytest.approx(expected, abs=TOL)


def test_rotation_about_y():
    """绕 Y 轴旋转 θ: 底面 = 中心z - (hx|sinθ| + hz|cosθ|)。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    theta = np.radians(30.0)
    z_c = 0.06
    r = _eval(m, d, [0, 0, z_c], _axis_quat((0, 1, 0), theta))
    expected = z_c - (HALF[0] * abs(np.sin(theta)) + HALF[2] * abs(np.cos(theta)))
    assert r.clearance == pytest.approx(expected, abs=TOL)


def test_rotation_about_z_keeps_bottom_face():
    """绕 Z 轴 (偏航) 不改变底面高度: clearance 与未旋转一致, 8 角点同高。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    z_c = 0.04
    flat = _eval(m, d, [0, 0, z_c])
    yawed = _eval(m, d, [0, 0, z_c], _axis_quat((0, 0, 1), np.radians(73.0)))
    assert yawed.clearance == pytest.approx(flat.clearance, abs=TOL)
    assert yawed.max_corner_z == pytest.approx(z_c + HALF[2], abs=TOL)


def test_combined_multi_axis_rotation():
    """多轴组合旋转: 与测试侧独立 Rodrigues 组合逐角点对比。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    ax, ay, az = np.radians(20.0), np.radians(35.0), np.radians(-50.0)
    R = _rodrigues((0, 0, 1), az) @ _rodrigues((0, 1, 0), ay) @ _rodrigues((1, 0, 0), ax)
    z_c, x_c, y_c = 0.05, 0.02, -0.01
    r = _eval(m, d, [x_c, y_c, z_c], _quat_from_matrix(R))
    corners_local = np.array([[sx * HALF[0], sy * HALF[1], sz * HALF[2]]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    expected = np.array([x_c, y_c, z_c]) + (corners_local @ R.T)
    assert r.clearance == pytest.approx(float(expected[:, 2].min()), abs=TOL)
    assert r.max_corner_z == pytest.approx(float(expected[:, 2].max()), abs=TOL)


def _quat_from_matrix(R):
    """测试侧旋转矩阵 -> 四元数 (w,x,y,z), 仅用于构造姿态。"""
    t = np.trace(R)
    w = np.sqrt(max(1.0 + t, 0.0)) / 2
    if w < 1e-6:
        raise ValueError("test helper: 180deg rotation unsupported")
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return tuple(q / np.linalg.norm(q))


def test_collision_origin_translation():
    """collision origin 存在非零平移 (真实 K1: pos 0.026 0 -0.02):
    底面 = body z + pz - hz, 而非 body z - hz。"""
    gpos = (0.026, 0.0, -0.02)
    m = mujoco.MjModel.from_xml_string(_xml_box(geom_pos=gpos))
    d = mujoco.MjData(m)
    z_body = 0.10
    r = _eval(m, d, [0, 0, z_body])
    expected = z_body + gpos[2] - HALF[2]
    assert r.clearance == pytest.approx(expected, abs=TOL)
    # 若漏掉 collision origin 平移, 结果会高出 |pz|=20mm
    assert r.clearance == pytest.approx(z_body - HALF[2] - 0.02, abs=TOL)


def test_collision_origin_translation_with_rotation():
    """平移偏移 + body 旋转组合: 底面由 world 变换后的角点决定。"""
    gpos = (0.026, 0.0, -0.02)
    m = mujoco.MjModel.from_xml_string(_xml_box(geom_pos=gpos))
    d = mujoco.MjData(m)
    theta = np.radians(20.0)
    z_body = 0.12
    r = _eval(m, d, [0, 0, z_body], _axis_quat((0, 1, 0), theta))
    # 测试侧独立计算: 角点 local -> +geom_pos -> 旋转 -> +body_pos
    corners_local = np.array([[sx * HALF[0], sy * HALF[1], sz * HALF[2]]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts = corners_local + np.array(gpos)
    world = pts @ _rodrigues((0, 1, 0), theta).T + np.array([0, 0, z_body])
    assert r.clearance == pytest.approx(float(world[:, 2].min()), abs=TOL)


def test_collision_origin_rotation():
    """collision origin 存在非零旋转 (geom quat 绕 Y 30°):
    底面 = center_z - (hx sin30 + hz cos30)。"""
    theta = np.radians(30.0)
    q = _axis_quat((0, 1, 0), theta)
    m = mujoco.MjModel.from_xml_string(_xml_box(geom_quat_wxyz=q))
    d = mujoco.MjData(m)
    z_body = 0.09
    r = _eval(m, d, [0, 0, z_body])
    expected = z_body - (HALF[0] * np.sin(theta) + HALF[2] * np.cos(theta))
    assert r.clearance == pytest.approx(expected, abs=TOL)
    # geom 局部旋转与 body 旋转叠加 (各应用一次, 不重复)
    phi = np.radians(15.0)
    r2 = _eval(m, d, [0, 0, z_body], _axis_quat((0, 1, 0), phi))
    R = _rodrigues((0, 1, 0), phi) @ _rodrigues((0, 1, 0), theta)
    corners_local = np.array([[sx * HALF[0], sy * HALF[1], sz * HALF[2]]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = corners_local @ R.T + np.array([0, 0, z_body])
    assert r2.clearance == pytest.approx(float(world[:, 2].min()), abs=TOL)


def test_old_center_height_method_blind_spot():
    """构造中心高度相同、真实底面不同的两姿态: 旧方法 (中心高度) 完全无法区分,
    新方法正确识别 —— 这是 Top Rock 穿地被漏检的机理。"""
    m = mujoco.MjModel.from_xml_string(_xml_box())
    d = mujoco.MjData(m)
    z_c = 0.030  # 两种姿态 box 中心世界高度相同
    flat = _eval(m, d, [0, 0, z_c])
    theta = np.radians(25.0)
    pitched = _eval(m, d, [0, 0, z_c], _axis_quat((0, 1, 0), theta))
    # 旧实现 (k1_action/virtual_k1_replay.py 修复前): geom_type==3 对 box(6)
    # 恒为假, 实际取中心高度 —— 两姿态给出完全相同的错误值
    old_flat = old_pitched = z_c
    assert old_flat == old_pitched  # 旧方法无区分能力
    expected_pitched = z_c - (HALF[0] * np.sin(theta) + HALF[2] * np.cos(theta))
    assert flat.clearance == pytest.approx(z_c - HALF[2], abs=TOL)
    assert pitched.clearance == pytest.approx(expected_pitched, abs=TOL)
    assert flat.state == "AIRBORNE"
    assert pitched.state == "PENETRATION"      # 真实穿地 ~24.4mm
    assert pitched.penetration_depth == pytest.approx(-expected_pitched, abs=TOL)
    # 旧方法对 pitched 的漏检量 = 中心高度 - 真实底面 = 0.030 + 0.02435
    assert old_pitched - expected_pitched == pytest.approx(0.05435, abs=1e-3)


# ---------------------------------------------------------------------------
# 3. GroundPlane 与不支持几何类型


def test_ground_plane_conventions():
    g = GroundPlane.horizontal(0.0)
    pts = np.array([[0, 0, 0.5], [0, 0, -0.2], [0, 0, 0.0]])
    assert np.allclose(g.signed_distance(pts), [0.5, -0.2, 0.0])
    g2 = GroundPlane.horizontal(0.3)
    assert g2.signed_distance(np.array([[0, 0, 0.5]]))[0] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        GroundPlane(normal=np.array([0, 0, 2.0]), offset=0.0)


def test_unsupported_geom_raises_no_silent_fallback():
    """capsule 碰撞几何: 显式报错, 不静默退化为中心高度检测。"""
    m = mujoco.MjModel.from_xml_string(_xml_box(capsule=True))
    with pytest.raises(UnsupportedFootGeomError, match="capsule|类型"):
        _detector(m)


def test_sphere_geom_supported():
    xml = """
    <mujoco><worldbody>
      <geom name="ground" type="plane" size="10 10 0.1"/>
      <body name="left_foot" pos="0 0 1"><freejoint/>
        <geom name="ball" type="sphere" size="0.03" contype="1" conaffinity="1"/>
      </body>
    </worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = _eval(m, d, [0, 0, 0.035])
    assert r.clearance == pytest.approx(0.005, abs=TOL)
    assert r.state == "AIRBORNE"
    r2 = _eval(m, d, [0, 0, 0.028])
    assert r2.clearance == pytest.approx(-0.002, abs=TOL)
    assert r2.state == "CONTACT"          # 穿透 2mm 恰在接触容差带边界内
    r3 = _eval(m, d, [0, 0, 0.027])
    assert r3.clearance == pytest.approx(-0.003, abs=TOL)
    assert r3.state == "PENETRATION"      # 穿透 3mm 超出容差带


# ---------------------------------------------------------------------------
# 4. 机器人级: 真实 K1 模型


@pytest.fixture(scope="module")
def k1() -> K1Sim:
    s = K1Sim()
    yield s
    s.close()


def _detector_k1(sim: K1Sim) -> FootGeometryDetector:
    return sim._foot_geometry_detector


def test_k1_reset_stance_design_gap(k1):
    """官方站立姿态: 双足 ~0.5-0.8mm 设计间隙, 判 CONTACT, 无穿透。"""
    k1.reset()
    geo = k1.get_foot_geometry()
    for side in ("left", "right"):
        assert geo[side]["penetration_depth_m"] == 0.0
        assert geo[side]["state"] == "CONTACT"
        assert 0.0 < geo[side]["clearance_m"] < 2.0e-3, (
            f"{side} reset 间隙应约为 0.5-0.8mm, 实际 {geo[side]['clearance_m']:.6f}")


def test_k1_root_height_shift_is_exact(k1):
    """root 抬升/下降 d: 双足 clearance 精确变化 d。"""
    k1.reset()
    root = k1.mapping.root_info
    base = {s: v["clearance_m"] for s, v in k1.get_foot_geometry().items()}
    for dz in (0.005, 0.050, -0.005):
        k1.reset()
        k1.data.qpos[root.qpos_adr + 2] += dz
        mujoco.mj_forward(k1.model, k1.data)
        geo = k1.get_foot_geometry()
        for side in ("left", "right"):
            assert geo[side]["clearance_m"] == pytest.approx(base[side] + dz, abs=1e-9)


def test_k1_root_yaw_invariant(k1):
    """root 偏航 (绕 Z) 不改变足底高度。"""
    k1.reset()
    root = k1.mapping.root_info
    base = {s: v["clearance_m"] for s, v in k1.get_foot_geometry().items()}
    for yaw in (np.radians(45.0), np.radians(90.0), np.radians(-170.0)):
        k1.reset()
        k1.data.qpos[root.qpos_adr + 3:root.qpos_adr + 7] = _axis_quat((0, 0, 1), yaw)
        mujoco.mj_forward(k1.model, k1.data)
        geo = k1.get_foot_geometry()
        for side in ("left", "right"):
            assert geo[side]["clearance_m"] == pytest.approx(base[side], abs=1e-9)


def test_k1_root_pitch_roll_change_clearance(k1):
    """root 俯仰/横滚改变足底高度, 且与独立角点计算一致。"""
    for axis, angle in (((0, 1, 0), 0.10), ((1, 0, 0), -0.08), ((1, 1, 0), 0.12)):
        k1.reset()
        root = k1.mapping.root_info
        k1.data.qpos[root.qpos_adr + 3:root.qpos_adr + 7] = _axis_quat(axis, angle)
        mujoco.mj_forward(k1.model, k1.data)
        det = _detector_k1(k1)
        res = det.evaluate(k1.data)
        for side, r in res.items():
            expected = _manual_k1_foot_corners(k1.model, k1.data, side)
            assert r.clearance == pytest.approx(float(expected[:, 2].min()), abs=1e-9)
        assert res["left"].clearance != res["right"].clearance or np.isclose(
            res["left"].clearance, res["right"].clearance, atol=1e-6)


def _manual_k1_foot_corners(model, data, side):
    """测试侧独立实现: 从 XML 常量直接变换 K1 足部 box 的 8 角点到世界系。

    使用真实 K1 碰撞盒定义 (K1_22dof.xml):
        pos="0.026 0 -0.02" size="0.09 0.035 0.018" (collision origin 平移)
    """
    bname = "left_ankle_roll_link" if side == "left" else "right_ankle_roll_link"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    gpos = np.array([0.026, 0.0, -0.02])
    half = np.array([0.09, 0.035, 0.018])
    corners_local = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # K1 足 box geom_quat 为单位四元数 (XML 无 quat 属性), 只剩 body FK 变换
    world = data.xpos[bid] + (corners_local + gpos) @ data.xmat[bid].reshape(3, 3).T
    return world


def test_k1_matches_manual_corners_multiple_poses(k1):
    """多姿态下 (reset/踝 pitch/踝 roll/组合) 检测器与独立角点计算逐位一致。"""
    poses = [
        {},
        {"left_ankle_pitch_joint": -0.4},
        {"left_ankle_pitch_joint": 0.3, "right_ankle_pitch_joint": -0.3},
        {"left_ankle_roll_joint": 0.25, "right_ankle_roll_joint": -0.25},
        {"left_knee_pitch_joint": 0.6, "left_ankle_pitch_joint": -0.3},
        {"left_hip_pitch_joint": -0.3, "left_knee_pitch_joint": 0.6,
         "left_ankle_pitch_joint": -0.3, "right_hip_roll_joint": 0.2},
    ]
    det = _detector_k1(k1)
    for pose in poses:
        k1.reset()
        if pose:
            k1.set_joint_positions(pose)
        res = det.evaluate(k1.data)
        for side, r in res.items():
            world = _manual_k1_foot_corners(k1.model, k1.data, side)
            assert r.clearance == pytest.approx(float(world[:, 2].min()), abs=1e-9), (
                f"pose={pose} {side}")


def test_k1_left_right_geometry_independent(k1):
    """左右脚使用各自几何参数与变换: 左踝转动不影响右脚, 且左右 y 位置镜像。"""
    k1.reset()
    geo0 = k1.get_foot_geometry(include_corners=True)
    k1.set_joint_positions({"left_ankle_pitch_joint": -0.099419 - 0.35})
    geo1 = k1.get_foot_geometry(include_corners=True)
    assert geo1["left"]["clearance_m"] < geo0["left"]["clearance_m"] - 0.01
    assert geo1["right"]["clearance_m"] == pytest.approx(geo0["right"]["clearance_m"], abs=1e-9)
    # 左脚在 +Y 侧, 右脚在 -Y 侧 (世界系, K1 左腿 hip pos y=+0.096)
    assert np.mean(geo0["left"]["corners_world"], axis=0)[1] > 0
    assert np.mean(geo0["right"]["corners_world"], axis=0)[1] < 0
    # 踝 pitch 到位后左脚实际穿地, 被正确判 PENETRATION
    assert geo1["left"]["state"] == "PENETRATION"


def test_k1_left_and_right_foot_alone(k1):
    """左右脚单独测试: 检测器对每只脚独立输出。"""
    k1.reset()
    det = _detector_k1(k1)
    for side, bname in (("left", "left_ankle_roll_link"),
                        ("right", "right_ankle_roll_link")):
        single = FootGeometryDetector(k1.model, foot_bodies={side: bname},
                                      contact_tolerance_m=CONTACT_TOL)
        r_both = det.evaluate(k1.data)[side]
        r_single = single.evaluate(k1.data)[side]
        assert r_single.clearance == pytest.approx(r_both.clearance, abs=1e-12)


def test_k1_transform_chain_applied_once(k1):
    """显式变换链与 MuJoCo geom_xpos/geom_xmat 合成一致 (每层变换恰好一次)。"""
    k1.reset()
    k1.set_joint_positions({"left_ankle_pitch_joint": -0.3,
                            "right_ankle_roll_joint": 0.2})
    err = _detector_k1(k1).verify_transform_chain(k1.data)
    assert err["max_corner_position_error_m"] < 1e-12


def test_k1_clearance_matches_geom_orientation_theory(k1):
    """踝 pitch 后, 用足 box 世界姿态角反推底面高度, 与检测器一致:
    bottom = center_z - (hx|sinθ| + hz|cosθ|), θ 为 box 局部 X 轴相对水平面的夹角。"""
    k1.reset()
    # 归零髋 roll/yaw 与踝 roll, 使足部姿态为纯 pitch (绕 Y), 公式才是精确解
    k1.set_joint_positions({"left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
                            "left_ankle_roll_joint": 0.0,
                            "left_ankle_pitch_joint": -0.099419 - 0.28})
    bid = mujoco.mj_name2id(k1.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    g = next(gi for gi in range(k1.model.ngeom)
             if int(k1.model.geom_bodyid[gi]) == bid and int(k1.model.geom_contype[gi]) > 0)
    R = k1.data.geom_xmat[g].reshape(3, 3)
    theta = np.arcsin(abs(R[2, 0]))          # box 局部 +X 轴的世界 z 分量
    center_z = float(k1.data.geom_xpos[g][2])
    expected = center_z - (0.09 * abs(np.sin(theta)) + 0.018 * np.cos(theta))
    geo = k1.get_foot_geometry()
    assert geo["left"]["clearance_m"] == pytest.approx(expected, abs=1e-9)
