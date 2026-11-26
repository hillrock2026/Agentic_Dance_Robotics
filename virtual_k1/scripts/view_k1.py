#!/usr/bin/env python3
"""渲染 K1 机器人 (离屏 PNG / 交互式 Viewer)。

用法:
  MUJOCO_GL=egl python scripts/view_k1.py                 # 离屏渲染 PNG
  python scripts/view_k1.py --interactive                 # GUI viewer (需要 X11)
  python scripts/view_k1.py --pose standing               # 初始站立姿态
  python scripts/view_k1.py --pose qpos0                  # 官方默认 qpos0 (悬空)

默认渲染 config.initial_pose 站立姿态, 输出 docs/k1_render_<pose>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from virtual_k1 import K1Sim  # noqa: E402

OUT_DIR = PROJECT_ROOT / "docs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=["standing", "qpos0"], default="standing")
    ap.add_argument("--interactive", action="store_true",
                    help="启动 GUI viewer (需要显示环境)")
    ap.add_argument("--hold-seconds", type=float, default=2.0,
                    help="交互模式下 PD 保持站立的时间")
    args = ap.parse_args()

    sim = K1Sim()
    if args.pose == "qpos0":
        import mujoco
        mujoco.mj_resetData(sim.model, sim.data)
        mujoco.mj_forward(sim.model, sim.data)

    print(f"模型: {sim.model_path}")
    print(f"姿态: {args.pose}, root z = {sim.data.qpos[2]:.4f} m")

    if args.interactive:
        import mujoco.viewer
        from virtual_k1 import JointPDController
        pd = JointPDController(sim.mapping, sim.cfg["pd_controller"]["gains"],
                               gravity_compensation=True)
        targets = sim.get_joint_positions()
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running() and sim.data.time < args.hold_seconds:
                sim.step(pd.compute(sim, targets), n_substeps=10)
                viewer.sync()
        print("viewer 已关闭")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = sim.render(width=960, height=720)
    out = OUT_DIR / f"k1_render_{args.pose}.png"

    def _save_png(path: Path, arr) -> None:
        # 纯 NumPy PNG 编码 (避免额外依赖): 使用 zlib + struct 手写 PNG
        import struct
        import zlib
        h, w = arr.shape[:2]
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
        def chunk(tag: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 9))
               + chunk(b"IEND", b""))
        path.write_bytes(png)

    _save_png(out, img)
    n_bright = int((img.sum(axis=2) > 30).sum())
    print(f"渲染完成: {out} ({img.shape[1]}x{img.shape[0]}, "
          f"非暗像素 {n_bright} / {img.shape[0]*img.shape[1]})")
    sim.close()


if __name__ == "__main__":
    main()
