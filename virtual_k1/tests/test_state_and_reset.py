"""Test 6: 状态读取与重置。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402
from virtual_k1.robot_state import validate_robot_state  # noqa: E402


@pytest.fixture()
def sim() -> K1Sim:
    s = K1Sim()
    yield s
    s.close()


def test_robot_state_schema_and_units(sim):
    """统一状态字典字段齐全, 22 关节, 数值有限。"""
    sim.reset()
    state = sim.get_robot_state()
    validate_robot_state(state)
    assert len(state["joint_positions"]) == 22
    assert len(state["joint_velocities"]) == 22
    assert len(state["joint_torques"]) == 22
    assert state["time"] == pytest.approx(0.0)
    assert np.asarray(state["base_position"]).shape == (3,)
    assert np.asarray(state["base_orientation"]).shape == (4,)
    quat = np.asarray(state["base_orientation"])
    assert np.linalg.norm(quat) == pytest.approx(1.0)  # 单位四元数 (wxyz)
    assert np.asarray(state["base_linear_velocity"]).shape == (3,)
    assert np.asarray(state["base_angular_velocity"]).shape == (3,)
    assert set(state["foot_contacts"]) == {"left", "right"}


def test_joint_readings_track_motion(sim):
    """运动学写入后读取一致; 步进后速度/力矩非零。"""
    from virtual_k1 import JointPDController
    sim.reset()
    sim.set_joint_positions({"aaleft_shoulder_pitch_joint": 0.3})
    q = sim.get_joint_positions()
    assert q["aaleft_shoulder_pitch_joint"] == pytest.approx(0.3, abs=1e-12)

    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])
    targets = dict(sim.get_joint_positions())
    targets["aaleft_shoulder_pitch_joint"] = -0.4
    moved = False
    for _ in range(800):
        sim.step(pd.compute(sim, targets))
        v = sim.get_joint_velocities()
        if abs(v["aaleft_shoulder_pitch_joint"]) > 0.1:
            moved = True
            break
    assert moved, "关节运动阶段速度读数始终为 0"
    tau = sim.get_joint_torques()
    assert any(abs(t) > 1e-3 for t in tau.values()), "力矩读数全为 0"


def test_actuator_force_readable(sim):
    """data.actuator_force 与 qfrc_actuator 读取通道有效。"""
    sim.reset()
    sim.step({"left_knee_pitch_joint": 2.0}, n_substeps=50)
    af = sim.get_actuator_forces()
    assert len(af) == 22
    assert abs(af["left_knee_pitch_joint"]) > 0.1


def test_base_state_velocity_world_frame(sim):
    """自由落体: base 线速度 z 分量应 ≈ -g*t (世界系验证)。"""
    import mujoco
    mujoco.mj_resetData(sim.model, sim.data)  # qpos0 悬空
    mujoco.mj_forward(sim.model, sim.data)
    t0 = float(sim.data.time)
    for _ in range(100):  # 0.1 s 无控制
        sim.step(None)
    st = sim.get_base_state()
    dt = float(sim.data.time) - t0
    vz = st["linear_velocity"][2]
    assert vz == pytest.approx(-9.81 * dt, rel=0.05), (
        f"自由落体 vz={vz:.3f}, 期望 {-9.81*dt:.3f} m/s")


def test_reset_deterministic_and_recoverable(sim):
    """扰动后 reset 完全恢复初始状态 (逐元素一致)。"""
    sim.reset()
    q0 = sim.data.qpos.copy()
    v0 = sim.data.qvel.copy()

    rng = np.random.default_rng(3)
    sim.set_controls(rng.uniform(-5, 5, sim.model.nu))
    for _ in range(500):
        sim.step()
    assert not np.allclose(sim.data.qpos, q0)

    sim.reset()
    assert np.array_equal(sim.data.qpos, q0), "reset 后 qpos 未恢复"
    assert np.array_equal(sim.data.qvel, v0), "reset 后 qvel 未恢复"
    assert float(sim.data.time) == 0.0
    assert np.all(sim.data.ctrl == 0)
