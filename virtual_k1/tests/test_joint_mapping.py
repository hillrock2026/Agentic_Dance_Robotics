"""Test 2: 关节结构与映射。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim  # noqa: E402


@pytest.fixture(scope="module")
def sim() -> K1Sim:
    s = K1Sim()
    yield s
    s.close()


def test_dimension_bookkeeping(sim):
    """22 驱动关节 + root free joint: nq=29, nv=28, nu=22, njnt=23。"""
    m = sim.mapping.summary()
    assert m["njnt"] == 23
    assert m["nq"] == 29          # 22 hinge + 7 (free joint)
    assert m["nv"] == 28          # 22 hinge + 6 (free joint)
    assert m["nu"] == 22
    assert m["joints_hinge"] == 22
    assert m["joints_free"] == 1
    assert m["actuated_joints"] == 22
    assert m["root_joint"] == "world_joint"


def test_all_actuators_torque_semantics(sim):
    """官方模型 22 个执行器全部为 torque/motor 语义。"""
    for a in sim.mapping.actuators:
        assert a.control_semantics == "torque", (
            f"{a.name}: {a.control_semantics} ({a.bias_note})")
        assert a.transmission_type == "joint"
        assert a.gear == pytest.approx(1.0)
        assert a.force_limited
        lo, hi = a.force_range
        assert lo < 0 < hi


def test_actuator_joint_one_to_one(sim):
    """执行器与关节一一对应, 且执行器名 == 关节名 (官方命名)。"""
    names = [a.name for a in sim.mapping.actuators]
    joints = [a.joint_name for a in sim.mapping.actuators]
    assert len(set(names)) == 22 and len(set(joints)) == 22
    assert names == joints
    # 每个驱动关节恰好被一个执行器驱动
    assert sorted(sim.mapping.actuated_joints) == sorted(joints)


def test_joint_groups_and_ranges(sim):
    """头部 2 + 手臂 8 + 腿 12 = 22, 限位来自实际模型。"""
    groups = {"head": [], "arm": [], "leg": []}
    for j in sim.mapping.joints:
        if j.type != "hinge":
            continue
        n = j.name
        if "head" in n:
            groups["head"].append(j)
        elif "shoulder" in n or "elbow" in n:
            groups["arm"].append(j)
        elif "hip" in n or "knee" in n or "ankle" in n:
            groups["leg"].append(j)
    assert len(groups["head"]) == 2
    assert len(groups["arm"]) == 8
    assert len(groups["leg"]) == 12
    for j in groups["head"] + groups["arm"] + groups["leg"]:
        assert j.limited, f"{j.name} 未声明限位"
        lo, hi = j.range
        assert lo < hi, f"{j.name} 限位异常 {j.range}"
    # 膝关节限位 [0, 2.321] —— 不允许过伸
    knee = sim.mapping.joint("left_knee_pitch_joint")
    assert knee.range[0] == pytest.approx(0.0, abs=1e-6)


def test_qpos_dof_addresses_consistent(sim):
    """驱动关节 qpos/dof 地址互异且覆盖 [7..29)/[6..28)。"""
    qpos_addrs = sorted(sim.mapping.joint(n).qpos_adr
                        for n in sim.mapping.actuated_joints)
    dof_addrs = sorted(sim.mapping.joint(n).dof_adr
                       for n in sim.mapping.actuated_joints)
    assert qpos_addrs == list(range(7, 29))
    assert dof_addrs == list(range(6, 28))


def test_alias_resolution(sim):
    """官方动作 CSV 命名 (肩 pitch 无 aa 前缀) 可解析到同一关节。"""
    mp = sim.mapping
    assert mp.resolve_joint("left_shoulder_pitch_joint") == \
        "aaleft_shoulder_pitch_joint"
    assert mp.resolve_joint("right_shoulder_pitch_joint") == \
        "aaright_shoulder_pitch_joint"
    assert mp.resolve_joint("aahead_yaw_joint") == "aahead_yaw_joint"
    with pytest.raises(KeyError):
        mp.resolve_joint("nonexistent_joint")


def test_named_control_writes_correct_actuator(sim):
    """按关节名写入力矩应落在对应执行器通道上。"""
    sim.reset()
    sim.set_controls({"left_knee_pitch_joint": 3.25})
    aid = sim.mapping.actuator_of_joint("left_knee_pitch_joint").actuator_id
    assert sim.data.ctrl[aid] == pytest.approx(3.25)
    assert sim.data.ctrl.sum() == pytest.approx(3.25)  # 其余为 0
    sim.set_controls(np.zeros(sim.model.nu))
