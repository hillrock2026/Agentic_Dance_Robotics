#!/usr/bin/env python3
"""按场景切换点将视频切分为独立片段。

用法:
  python split_video_by_scenes.py <input.mp4> [-o 输出目录] [-t 场景阈值] [--copy]

策略: ffmpeg scene 检测切换点 → 每段重编码精确切割 (默认) 或关键帧快速拷贝。
输出: <outdir>/segment_XXX.mp4 + manifest.csv (起始/结束/时长)。
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()


def detect_cuts(video: str, threshold: float) -> list[float]:
    vf = f"select='gt(scene,{threshold})',showinfo"
    r = subprocess.run([FF, "-i", video, "-vf", vf, "-fps_mode", "vfr",
                        "-f", "null", "-"], capture_output=True, text=True)
    pts = sorted(set(round(float(m), 3) for m in
                     re.findall(r"pts_time:([\d.]+)", r.stderr)))
    # 去除相距 <0.5s 的重复检测
    merged: list[float] = []
    for p in pts:
        if not merged or p - merged[-1] > 0.5:
            merged.append(p)
    return merged


def duration_of(video: str) -> float:
    r = subprocess.run([FF, "-i", video], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("-t", "--threshold", type=float, default=0.20)
    ap.add_argument("--copy", action="store_true",
                    help="关键帧快速拷贝 (边界可能偏差至关键帧间隔)")
    args = ap.parse_args()

    video = args.video
    outdir = Path(args.outdir) if args.outdir else \
        Path(video).with_suffix("") .parent / (Path(video).stem + "_segments")
    outdir.mkdir(parents=True, exist_ok=True)

    total = duration_of(video)
    cuts = detect_cuts(video, args.threshold)
    bounds = [0.0] + cuts + [total]
    print(f"视频总长 {total:.2f}s, 检测到 {len(cuts)} 个切换点 → {len(cuts)+1} 段")

    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        out = outdir / f"segment_{i+1:03d}.mp4"
        cmd = [FF, "-y", "-loglevel", "error",
               "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", video]
        if args.copy:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "128k"]
        cmd += [str(out)]
        subprocess.run(cmd, check=True)
        rows.append({"file": out.name, "start_s": round(t0, 2),
                     "end_s": round(t1, 2), "duration_s": round(t1 - t0, 2)})
        print(f"  [{i+1:03d}] {t0:7.2f} → {t1:7.2f}  ({t1-t0:5.2f}s)  {out.name}")

    with open(outdir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "start_s", "end_s", "duration_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成: {len(rows)} 段 → {outdir}/ (含 manifest.csv)")


if __name__ == "__main__":
    main()
