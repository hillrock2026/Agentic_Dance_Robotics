"""Test 4: 足部地面接触。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402


@pytest.fixture(scope="module")
def sim() -> K1Sim:
    s = K1Sim()
    yield s
    s.close()


@pytest.fixture(scope="module")
def pd(sim: K1Sim) -> JointPDController:
    return JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])


def _hold(sim, pd, targets, seconds):
    for _ in range(int(seconds / sim.timestep)):
        sim.step(pd.compute(sim, targets))


def test_initial_pose_feet_above_ground_no_penetration(sim):
    """reset 后 (未步进): 足底不穿透地面。"""
    sim.reset()
    box_t = int(mujoco.mjtGeom.mjGEOM_BOX)
    for side, bid in sim._foot_body_ids.items():
        for gid in range(sim.model.ngeom):
            if (int(sim.model.geom_bodyid[gid]) == bid
                    and int(sim.model.geom_type[gid]) == box_t
                    and (sim.model.geom_contype[gid] > 0
                         or sim.model.geom_conaffinity[gid] > 0)):
                z = sim.data.geom_xpos[gid][2] - sim.model.geom_size[gid][2]
                assert z > -1e-3, f"{side} 足底穿透: z={z}"


def test_static_standing_double_contact(sim, pd):
    """静态站立 2s: 双脚接触, 法向力合计 ≈ 总重力, 法向朝上。"""
    sim.reset()
    targets = sim.get_joint_positions()
    _hold(sim, pd, targets, 2.0)
    fc = sim.get_foot_contacts()
    for side in ("left", "right"):
        assert fc[side]["in_contact"], f"{side} 脚未接触地面"
        assert fc[side]["n_contacts"] >= 1
        assert fc[side]["total_normal_force"] > 20.0
        for n in fc[side]["normals"]:
            assert n[2] > 0.9, f"{side} 接触法向异常: {n}"
    total_fn = fc["left"]["total_normal_force"] + fc["right"]["total_normal_force"]
    weight = float(sim.model.body_mass.sum()) * 9.81
    assert total_fn == pytest.approx(weight, rel=0.05), (
        f"法向力合计 {total_fn:.1f}N 与重力 {weight:.1f}N 偏差 >5%")


def test_contact_updates_with_posture_change(sim, pd):
    """下蹲 (膝目标 +0.6 rad) 后: 仍双足接触, root 高度下降, 接触点更新。"""
    sim.reset()
    targets = sim.get_joint_positions()
    _hold(sim, pd, targets, 1.5)
    z_before = float(sim.data.qpos[2])
    contact_before = sim.get_foot_contacts()

    for j in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        targets[j] -= 0.3
    for j in ("left_knee_pitch_joint", "right_knee_pitch_joint"):
        targets[j] += 0.6
    for j in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        targets[j] -= 0.3
    _hold(sim, pd, targets, 1.5)
    z_after = float(sim.data.qpos[2])
    contact_after = sim.get_foot_contacts()

    assert z_after < z_before - 0.03, (
        f"下蹲后高度未下降: {z_before:.4f} → {z_after:.4f}")
    for side in ("left", "right"):
        assert contact_after[side]["in_contact"], f"下蹲后 {side} 脚离地"
    # 接触数据必须随姿态变化而更新 (位置/数量/力 至少一项变化)
    changed = []
    for side in ("left", "right"):
        b, a = contact_before[side], contact_after[side]
        pos_moved = (len(b["positions"]) != len(a["positions"])
                     or any(np.linalg.norm(np.asarray(x) - np.asarray(y)) > 1e-4
                            for x, y in zip(a["positions"], b["positions"])))
        force_changed = abs(a["total_normal_force"] - b["total_normal_force"]) > 1.0
        if pos_moved or force_changed:
            changed.append(side)
        print(f"  {side} 脚: 接触点 {len(b['positions'])}→{len(a['positions'])}, "
              f"法向力 {b['total_normal_force']:.1f}→"
              f"{a['total_normal_force']:.1f} N")
    assert changed, "姿态变化后双足接触数据完全未更新"


def test_free_fall_regrounds_after_reset(sim, pd):
    """qpos0 悬空落地: 从 1.0m 释放后应触地产生接触 (验证接触检测动态有效)。"""
    import mujoco
    sim.reset()
    mujoco.mj_resetData(sim.model, sim.data)  # qpos0: root z=1.0 悬空
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.get_foot_contacts()["left"]["n_contacts"] == 0
    targets = {a.joint_name: 0.0 for a in sim.mapping.actuators}
    _hold(sim, pd, targets, 1.5)  # 自由落体 + PD 保持直腿
    fc = sim.get_foot_contacts()
    assert fc["left"]["in_contact"] or fc["right"]["in_contact"], (
        "从 0.45m 落下 1.5s 后仍无足地接触")
