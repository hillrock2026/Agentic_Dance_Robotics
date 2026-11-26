"""Test 7: 10 秒无图形界面持续仿真。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402

DURATION = 10.0  # s (任务要求 >= 10 s)


def test_sustained_headless_run_10s():
    sim = K1Sim()   # 无 viewer / 无渲染, 纯物理
    try:
        pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])
        sim.reset()
        targets = sim.get_joint_positions()

        n_steps = int(DURATION / sim.timestep)
        t_wall0 = time.perf_counter()
        samples = []
        for i in range(n_steps):
            sim.step(pd.compute(sim, targets))
            # 跳过第 0 步: reset 有 0.5mm 间隙, 接触在数 ms 内建立
            if i % 500 == 0 and i > 0:  # 每 0.5 s 采样一次完整状态
                state = sim.get_robot_state()
                validate_all_finite(state)
                fc = state["foot_contacts"]
                samples.append((state["time"], state["base_position"][2],
                                fc["left"]["in_contact"] and fc["right"]["in_contact"]))
        t_wall = time.perf_counter() - t_wall0

        assert sim.data.time == pytest.approx(DURATION, abs=1e-6)
        assert np.isfinite(sim.data.qpos).all()
        assert np.isfinite(sim.data.qvel).all()
        assert np.isfinite(sim.data.qacc).all()
        assert np.isfinite(sim.data.ctrl).all()
        # 10s 内机器人保持站立 (允许瞬时扰动, 但采样点必须全部双足接触)
        assert all(s[2] for s in samples), f"存在足部离地采样点: {samples}"
        z_last = samples[-1][1]
        assert 0.45 < z_last < 0.60, f"末端 root 高度异常: {z_last}"
        print(f"\n10s 仿真完成: 墙钟 {t_wall:.1f}s, 实时率 {DURATION/t_wall:.1f}x")
        for t, z, both in samples:
            print(f"  t={t:5.1f}s root_z={z:.4f} 双足接触={both}")
    finally:
        sim.close()


def validate_all_finite(state: dict) -> None:
    for key in ("base_position", "base_orientation",
                "base_linear_velocity", "base_angular_velocity"):
        arr = np.asarray(state[key])
        assert np.isfinite(arr).all(), f"{key} 出现 NaN/Inf"
    for key in ("joint_positions", "joint_velocities", "joint_torques"):
        for name, val in state[key].items():
            assert np.isfinite(val), f"{key}[{name}] = {val}"
