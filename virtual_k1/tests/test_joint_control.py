"""Test 5: 关节控制闭环 (最重要验收项)。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402


@pytest.fixture()
def sim() -> K1Sim:
    s = K1Sim()
    yield s
    s.close()


@pytest.fixture()
def pd(sim: K1Sim) -> JointPDController:
    return JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])


def _hold(sim, pd, targets, seconds):
    for _ in range(int(seconds / sim.timestep)):
        sim.step(pd.compute(sim, targets))


def test_torque_control_moves_joint_directly(sim):
    """裸力矩控制: 恒定力矩使关节产生运动 (验证 torque 语义)。"""
    sim.reset()
    q_before = sim.get_joint_positions()["left_knee_pitch_joint"]
    sim.step({"left_knee_pitch_joint": -5.0}, n_substeps=300)  # 0.3s
    q_after = sim.get_joint_positions()["left_knee_pitch_joint"]
    assert abs(q_after - q_before) > 0.05, (
        f"力矩未使膝关节运动: {q_before} → {q_after}")


def test_pd_tracks_target_change_closed_loop(sim, pd):
    """PD 目标变化 → 实际关节位置跟随变化 (闭环证据; 目标相对初始姿态设定)。"""
    sim.reset()
    targets = sim.get_joint_positions()
    _hold(sim, pd, targets, 1.0)

    # 变化前基准
    q_shoulder_0 = sim.get_joint_positions()["aaleft_shoulder_pitch_joint"]
    q_elbow_0 = sim.get_joint_positions()["left_elbow_pitch_joint"]

    targets["aaleft_shoulder_pitch_joint"] = q_shoulder_0 + 0.6
    targets["left_elbow_pitch_joint"] = q_elbow_0 - 0.8
    _hold(sim, pd, targets, 1.5)

    q_shoulder_1 = sim.get_joint_positions()["aaleft_shoulder_pitch_joint"]
    q_elbow_1 = sim.get_joint_positions()["left_elbow_pitch_joint"]
    assert q_shoulder_1 - q_shoulder_0 > 0.45, (
        f"肩关节未跟随目标: {q_shoulder_0} → {q_shoulder_1}")
    assert q_elbow_1 - q_elbow_0 < -0.6, (
        f"肘关节未跟随目标: {q_elbow_0} → {q_elbow_1}")
    # 稳态误差
    assert abs(q_shoulder_1 - (q_shoulder_0 + 0.6)) < 0.1
    assert abs(q_elbow_1 - (q_elbow_0 - 0.8)) < 0.1


def test_standing_remains_stable_under_pd(sim, pd):
    """PD 站立 3s: root 高度稳定, 双足保持接触, 无发散。

    注: reset 后足底与地面有 0.5 mm 设计间隙, 前 0.1 s 为沉降期不断言接触。
    """
    sim.reset()
    targets = sim.get_joint_positions()
    z_ref = float(sim.data.qpos[2])
    min_z, max_z = z_ref, z_ref
    settle_steps = int(0.1 / sim.timestep)
    for i in range(int(3.0 / sim.timestep)):
        sim.step(pd.compute(sim, targets))
        z = float(sim.data.qpos[2])
        min_z, max_z = min(min_z, z), max(max_z, z)
        if i >= settle_steps:
            fc = sim.get_foot_contacts()
            assert fc["left"]["in_contact"] and fc["right"]["in_contact"], (
                f"t={sim.data.time:.2f} 足部离地")
    assert max_z - min_z < 0.02, f"root 高度波动 {max_z - min_z:.4f} m"
    assert np.isfinite(sim.data.qpos).all()


def test_pd_output_respects_force_limits(sim, pd):
    """PD 输出力矩不超过官方 forcerange。"""
    sim.reset()
    # 极端目标制造大力矩需求
    targets = {a.joint_name: (a.force_range[1] / 10.0) for a in sim.mapping.actuators}
    tau = pd.compute(sim, targets)
    for act in sim.mapping.actuators:
        t = tau[act.actuator_id]
        assert act.force_range[0] - 1e-9 <= t <= act.force_range[1] + 1e-9, (
            f"{act.name} 力矩越界: {t} vs {act.force_range}")
    assert np.abs(tau).max() > 10.0  # 确实产生了大力矩需求并被截断


def test_step_rejects_bad_control(sim):
    sim.reset()
    with pytest.raises(Exception):
        sim.step({"nonexistent_joint": 1.0})
    with pytest.raises(Exception):
        sim.step(np.zeros(sim.model.nu + 1))
    with pytest.raises(Exception):
        sim.step(np.full(sim.model.nu, np.nan))
