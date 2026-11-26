#!/usr/bin/env python3
"""K1 MuJoCo 动作演示。

编排动作序列 (PD 力矩控制, 官方模型):
  站立 → 左手挥手 → 头部环视 → 右手挥手 → 下蹲×2 → 扭摆 groove → 双臂亮相 → 归位

用法:
  MUJOCO_GL=egl python scripts/demo_k1.py                    # 离屏渲染 MP4
  python scripts/demo_k1.py --live                           # 打开实时 MuJoCo Viewer
  MUJOCO_GL=egl python scripts/demo_k1.py --out demo.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim, JointPDController  # noqa: E402


def _smooth(t: float) -> float:
    """0→1 平滑过渡 (余弦缓动)。"""
    return 0.5 * (1 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _sin01(t: float, f: float) -> float:
    """-1..1 正弦。"""
    return np.sin(2 * np.pi * f * np.clip(t, 0.0, 1.0))


class Demo:
    """动作编排: 每个动作是 (名称, 时长, phase→目标覆盖 dict)。"""

    def __init__(self, base: dict[str, float]):
        self.base = dict(base)
        self.moves = self._build()
        self.total = sum(d for _, d, _ in self.moves)

    def _build(self):
        b = self.base
        moves = []

        # 1. 站稳
        moves.append(("stand", 1.2, lambda p, b=b: {}))

        # 2. 左臂举手挥手
        #    关节约定 (FK 实测): shoulder_roll 负→下垂 / 0→平举 / 正→高举;
        #    elbow_yaw 屈肘 (左负=前弯); elbow_pitch 为前臂自旋
        def wave_left(p):
            b_roll, b_yaw = b["left_shoulder_roll_joint"], b["left_elbow_yaw_joint"]
            if p < 0.30:                      # 0.30 抬臂 (0.96s)
                k = _smooth(p / 0.30)
                roll = b_roll + (1.05 - b_roll) * k
                yaw = b_yaw + (-0.75 - b_yaw) * k
            elif p < 0.78:                    # 1.54s 挥动 3 次
                roll = 1.05
                wob = np.sin((p - 0.30) / 0.48 * np.pi * 3)
                yaw = -0.75 + 0.38 * wob
            else:                             # 0.7s 放下
                k = _smooth((p - 0.78) / 0.22)
                roll = 1.05 + (b_roll - 1.05) * k
                yaw = -0.75 + (b_yaw + 0.75) * k
            return {"left_shoulder_roll_joint": roll,
                    "left_elbow_yaw_joint": yaw}

        moves.append(("wave_left", 3.2, wave_left))

        # 3. 头部环视 + 点头
        def head_look(p):
            return {
                "aahead_yaw_joint": 0.55 * _sin01(p, 0.75),
                "aahead_pitch_joint": 0.18 * _sin01(p, 1.5) + 0.05,
            }
        moves.append(("head_look", 3.0, head_look))

        # 4. 右臂举手挥手 (与左臂镜像: roll 正→下垂/负→高举; elbow_yaw 正=前弯)
        def wave_right(p):
            b_roll, b_yaw = b["right_shoulder_roll_joint"], b["right_elbow_yaw_joint"]
            if p < 0.30:
                k = _smooth(p / 0.30)
                roll = b_roll + (-1.05 - b_roll) * k
                yaw = b_yaw + (0.75 - b_yaw) * k
            elif p < 0.78:
                roll = -1.05
                wob = np.sin((p - 0.30) / 0.48 * np.pi * 3)
                yaw = 0.75 - 0.38 * wob
            else:
                k = _smooth((p - 0.78) / 0.22)
                roll = -1.05 + (b_roll + 1.05) * k
                yaw = 0.75 + (b_yaw - 0.75) * k
            return {"right_shoulder_roll_joint": roll,
                    "right_elbow_yaw_joint": yaw}
        moves.append(("wave_right", 3.2, wave_right))

        # 5. 下蹲 ×2 (髋/膝/踝协同; 平足约束: hip+knee+ankle ≈ 0 保持脚掌贴地)
        squat = {
            "left_hip_pitch_joint": -0.55, "right_hip_pitch_joint": -0.55,
            "left_knee_pitch_joint": 1.10, "right_knee_pitch_joint": 1.10,
            "left_ankle_pitch_joint": -0.55, "right_ankle_pitch_joint": -0.55,
        }
        sq = {k: v - b[k] for k, v in squat.items()}

        def do_squat(p):
            # 每次蹲起: 下 45% + 底部保持 10% + 起 45%
            cyc = (p * 2) % 1.0
            if cyc < 0.45:
                k = _smooth(cyc / 0.45)
            elif cyc < 0.55:
                k = 1.0
            else:
                k = 1.0 - _smooth((cyc - 0.55) / 0.45)
            return {j: b[j] + d * k for j, d in sq.items()}

        moves.append(("squat", 4.6, do_squat))

        # 6. 律动 groove (悬垂臂前后反相摆 + 头部律动; 原地无弹跳, 开环稳定)
        def groove(p):
            w = _sin01(p, 0.55)
            w2 = _sin01(p, 1.1)
            return {
                "aaleft_shoulder_pitch_joint": b["aaleft_shoulder_pitch_joint"]
                + 0.35 * w,
                "aaright_shoulder_pitch_joint": b["aaright_shoulder_pitch_joint"]
                - 0.35 * w,
                "left_elbow_yaw_joint": b["left_elbow_yaw_joint"] - 0.15,
                "right_elbow_yaw_joint": b["right_elbow_yaw_joint"] + 0.15,
                "left_knee_pitch_joint": b["left_knee_pitch_joint"] + 0.04
                + 0.03 * w2,
                "right_knee_pitch_joint": b["right_knee_pitch_joint"] + 0.04
                + 0.03 * w2,
                "left_hip_pitch_joint": b["left_hip_pitch_joint"] - 0.04
                - 0.03 * w2,
                "right_hip_pitch_joint": b["right_hip_pitch_joint"] - 0.04
                - 0.03 * w2,
                "aahead_yaw_joint": 0.08 * _sin01(p, 1.1),
                "aahead_pitch_joint": 0.05 * _sin01(p, 2.2) + 0.04,
            }
        moves.append(("groove", 4.4, groove))

        # 7. 双臂 V 字亮相 (roll L=+0.9 / R=-0.9 举至头上, 屈肘)
        def finish(p):
            k = _smooth(p / 0.5)
            pose = {
                "left_shoulder_roll_joint": 0.90,
                "right_shoulder_roll_joint": -0.90,
                "left_elbow_yaw_joint": -0.90,
                "right_elbow_yaw_joint": 0.90,
                "aahead_pitch_joint": 0.12,
            }
            return {j: b[j] + (v - b[j]) * k for j, v in pose.items()}
        moves.append(("finish_pose", 2.6, finish))

        # 8. 归位
        moves.append(("return", 1.6,
                      lambda p: {j: b[j] for j in (
                          "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                          "left_elbow_yaw_joint", "right_elbow_yaw_joint",
                          "aaleft_shoulder_pitch_joint",
                          "aaright_shoulder_pitch_joint",
                          "aahead_pitch_joint")}))

        return moves

    def targets(self, t_abs: float) -> dict[str, float]:
        """给定演示时间, 返回当前目标关节角 (与 base 合并)。"""
        t = min(t_abs, self.total)
        acc = 0.0
        for name, dur, fn in self.moves:
            if t < acc + dur:
                merged = dict(self.base)
                merged.update(fn((t - acc) / dur))
                return merged
            acc += dur
        return dict(self.base)


class TargetSmoother:
    """一阶低通平滑目标轨迹, 消除动作边界处的目标阶跃 (避免 PD 力矩冲击)。"""

    def __init__(self, base: dict[str, float], tau: float = 0.08):
        self.tau = tau
        self.state = dict(base)

    def update(self, raw: dict[str, float], dt: float) -> dict[str, float]:
        alpha = dt / (self.tau + dt)
        for k in self.state:
            self.state[k] += alpha * (raw.get(k, self.state[k]) - self.state[k])
        return dict(self.state)


class BalanceAnkle:
    """CoM 踝策略平衡反馈 (前后向): 保持全身质心 x 在初始参考点。

    原理: 开环关节轨迹下蹲/扭摆时 CoM 会漂出足底支撑域 (前向支撑仅 ~11cm),
    用 subtree_com 与参考点的水平误差微调双侧 ankle_pitch 目标角, 把 CoM 拉回。
    符号经实测确定。侧向 (y) 漂移实测 <1mm, 本演示不含侧向动作, 不做 y 反馈。
    """

    def __init__(self, sim: K1Sim, gain: float = 1.0, limit: float = 0.22):
        self.ref = float(sim.data.subtree_com[0][0])
        self.gain = gain
        self.limit = limit

    def apply(self, sim: K1Sim, targets: dict[str, float]) -> dict[str, float]:
        err = float(sim.data.subtree_com[0][0]) - self.ref
        corr = float(np.clip(self.gain * err, -self.limit, self.limit))
        for j in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            targets[j] = targets.get(j, 0.0) + corr
        return targets


def run_headless(out_path: Path, fps: float = 30.0) -> None:
    import mujoco
    import imageio.v2 as imageio

    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])
    demo = Demo(sim.get_joint_positions())

    # 相机: 3/4 侧视
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.62]
    cam.distance = 2.6
    cam.azimuth = 135.0
    cam.elevation = -6.0

    steps_per_frame = int(round(1.0 / (sim.timestep * fps)))
    real_fps = 1.0 / (sim.timestep * steps_per_frame)
    n_frames = int(demo.total * real_fps) + int(0.8 * real_fps)  # +0.8s 片尾

    sim.model.vis.global_.offwidth = 960
    sim.model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(sim.model, 720, 960)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=real_fps, quality=9, macro_block_size=1)
    smoother = TargetSmoother(demo.base)
    balance = BalanceAnkle(sim)
    z_log = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        for _ in range(steps_per_frame):
            targets = smoother.update(demo.targets(sim.data.time), sim.timestep)
            targets = balance.apply(sim, targets)
            sim.step(pd.compute(sim, targets))
        z_log.append(float(sim.data.qpos[2]))
        renderer.update_scene(sim.data, camera=cam)
        writer.append_data(renderer.render())
        if i % 100 == 0:
            print(f"  frame {i}/{n_frames}  t={sim.data.time:.1f}s "
                  f"root_z={z_log[-1]:.3f} ({time.perf_counter()-t0:.0f}s elapsed)",
                  flush=True)
    writer.close()
    renderer.close()

    fc = sim.get_foot_contacts()
    z_arr = np.array(z_log)
    print(f"\n渲染完成: {out_path} ({n_frames} 帧 @ {real_fps:.2f}fps, "
          f"时长 {n_frames/real_fps:.1f}s)")
    print(f"root_z 范围: [{z_arr.min():.3f}, {z_arr.max():.3f}] m "
          f"(站立段均值 {z_arr[:n_frames//8].mean():.3f})")
    print(f"结束状态: root_z={z_arr[-1]:.4f}m "
          f"双足接触 L={fc['left']['in_contact']} R={fc['right']['in_contact']}")
    if z_arr.min() < 0.30:
        k = int(np.argmin(z_arr))
        print(f"⚠ 检测到疑似摔倒: 最低点 t={k/real_fps:.1f}s z={z_arr.min():.3f}m")
    else:
        print("✓ 全程保持站立")
    sim.close()


def run_live() -> None:
    import mujoco.viewer

    sim = K1Sim()
    pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"])
    demo = Demo(sim.get_joint_positions())

    print(f"实时演示开始, 共 {demo.total:.1f}s + 2s 片尾 (关闭窗口可提前结束)")
    smoother = TargetSmoother(demo.base)
    balance = BalanceAnkle(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        wall0 = time.perf_counter()
        last_sync = 0.0
        while viewer.is_running():
            sim_t = sim.data.time
            if sim_t > demo.total + 2.0:
                break
            for _ in range(10):
                targets = smoother.update(demo.targets(sim.data.time), sim.timestep)
                targets = balance.apply(sim, targets)
                sim.step(pd.compute(sim, targets))
            now = time.perf_counter() - wall0
            if now < sim_t:  # 保持实时
                time.sleep(min(0.01, sim_t - now))
            if sim_t - last_sync > 1 / 60:
                viewer.sync()
                last_sync = sim_t
    print("实时演示结束")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs/k1_demo.mp4"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    if args.live:
        run_live()
    else:
        run_headless(Path(args.out), fps=args.fps)


if __name__ == "__main__":
    main()
