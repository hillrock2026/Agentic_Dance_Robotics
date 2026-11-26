#!/usr/bin/env python3
"""重置可重复性验证脚本。

流程: 记录初始状态 → 扰动 (力矩踢 + 步进) → reset → 与初始状态逐项比对,
并重复两次 reset 验证确定性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402


def snapshot(sim: K1Sim) -> dict:
    s = sim.get_robot_state()
    return {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "time": s["time"],
        "base_pos": np.asarray(s["base_position"]),
        "base_quat": np.asarray(s["base_orientation"]),
    }


def main() -> None:
    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])

    s0 = snapshot(sim)
    print("初始状态: time=%.3f root=%s" % (s0["time"], np.round(s0["qpos"][:3], 4)))

    # 扰动: 随机力矩踢 + 自由步进 1.5s (不加控制)
    rng = np.random.default_rng(7)
    kick = {a.joint_name: float(rng.uniform(-8, 8))
            for a in sim.mapping.actuators}
    sim.set_controls(kick)
    for _ in range(300):
        sim.step()
    sim.set_controls(np.zeros(sim.model.nu))
    for _ in range(1200):
        sim.step()
    s1 = snapshot(sim)
    print("扰动后:   time=%.3f root=%s (机器人应已偏离初始状态)" %
          (s1["time"], np.round(s1["qpos"][:3], 4)))
    assert not np.allclose(s1["qpos"], s0["qpos"]), "扰动未产生状态变化"

    # 第一次 reset
    sim.reset()
    s2 = snapshot(sim)
    # 第二次 reset (确定性)
    sim.reset()
    s3 = snapshot(sim)

    for tag, sx in (("第1次", s2), ("第2次", s3)):
        e_qpos = float(np.max(np.abs(sx["qpos"] - s0["qpos"])))
        e_qvel = float(np.max(np.abs(sx["qvel"] - s0["qvel"])))
        e_time = abs(sx["time"] - s0["time"])
        ok = e_qpos == 0.0 and e_qvel == 0.0 and e_time == 0.0
        print(f"{tag} reset: qpos误差={e_qpos:.2e} qvel误差={e_qvel:.2e} "
              f"time误差={e_time:.2e} → {'PASS' if ok else 'FAIL'}")

    sim.close()


if __name__ == "__main__":
    main()
