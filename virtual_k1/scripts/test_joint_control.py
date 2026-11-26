#!/usr/bin/env python3
"""关节控制闭环验证脚本 (Test 5 核心演示)。

流程:
  1. 初始化 → PD 控制器保持初始站立姿态 2 s
  2. 修改左肩 pitch 目标 (+0.5 rad) 与头部 yaw 目标 (+0.5 rad), 运行 2 s
  3. 读取实际关节位置, 验证目标变化引起实际状态变化
  4. 重置, 验证恢复初始状态

同时记录足底接触与 root 高度, 作为站立稳定性证据。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402


def run_hold(sim: K1Sim, pd: JointPDController, targets: dict,
             seconds: float) -> dict:
    n_steps = int(round(seconds / sim.timestep))
    for _ in range(n_steps):
        sim.step(pd.compute(sim, targets))
    state = sim.get_robot_state()
    contact = state["foot_contacts"]
    return {
        "time": state["time"],
        "base_z": float(state["base_position"][2]),
        "left_contact": contact["left"]["in_contact"],
        "right_contact": contact["right"]["in_contact"],
        "left_fn": contact["left"]["total_normal_force"],
        "right_fn": contact["right"]["total_normal_force"],
        "positions": state["joint_positions"],
    }


def main() -> None:
    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"],
                           gravity_compensation=True)
    targets = sim.get_joint_positions()
    q0 = dict(targets)

    print("=" * 72)
    print("K1 关节控制闭环验证")
    print("=" * 72)

    print("\n[1] 初始站立 (PD 保持 2.0 s):")
    r1 = run_hold(sim, pd, targets, 2.0)
    print(f"  time={r1['time']:.2f}s root_z={r1['base_z']:.4f}m "
          f"左脚接触={r1['left_contact']} 右脚接触={r1['right_contact']} "
          f"法向力 L={r1['left_fn']:.1f}N R={r1['right_fn']:.1f}N")
    print(f"  左肩pitch={r1['positions']['aaleft_shoulder_pitch_joint']:+.4f} rad "
          f"头yaw={r1['positions']['aahead_yaw_joint']:+.4f} rad")

    print("\n[2] 目标变化: 左肩 pitch 0 → +0.5 rad, 头 yaw 0 → +0.5 rad (2.0 s):")
    targets["aaleft_shoulder_pitch_joint"] = 0.5
    targets["aahead_yaw_joint"] = 0.5
    r2 = run_hold(sim, pd, targets, 2.0)
    dq_shoulder = r2["positions"]["aaleft_shoulder_pitch_joint"] - \
        r1["positions"]["aaleft_shoulder_pitch_joint"]
    dq_head = r2["positions"]["aahead_yaw_joint"] - r1["positions"]["aahead_yaw_joint"]
    print(f"  time={r2['time']:.2f}s root_z={r2['base_z']:.4f}m "
          f"左脚接触={r2['left_contact']} 右脚接触={r2['right_contact']}")
    print(f"  左肩pitch={r2['positions']['aaleft_shoulder_pitch_joint']:+.4f} rad "
          f"(Δ={dq_shoulder:+.4f}, 目标Δ=+0.5)")
    print(f"  头yaw={r2['positions']['aahead_yaw_joint']:+.4f} rad "
          f"(Δ={dq_head:+.4f}, 目标Δ=+0.5)")

    ok_shoulder = dq_shoulder > 0.3
    ok_head = dq_head > 0.3
    print(f"  闭环判定: 肩 {'PASS' if ok_shoulder else 'FAIL'}, "
          f"头 {'PASS' if ok_head else 'FAIL'}")

    print("\n[3] 重置:")
    sim.reset()
    qr = sim.get_joint_positions()
    err = max(abs(qr[k] - q0[k]) for k in q0)
    print(f"  time={sim.data.time:.3f}s root_z={sim.data.qpos[2]:.4f}m "
          f"最大关节复位误差={err:.2e} rad "
          f"({'PASS' if err < 1e-9 else 'FAIL'})")
    sim.close()

    print("\n结论: 目标变化引起对应仿真状态变化 =",
          "PASS" if (ok_shoulder and ok_head) else "FAIL")


if __name__ == "__main__":
    main()
