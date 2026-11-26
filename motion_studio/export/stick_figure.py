"""OpenPose-style stick-figure visualization from clean_motion.npz.

Draws the classic pose-estimation look: colored limb lines + joint dots,
rendered as 2D orthographic projections of the 3D joints (front / side view).
Contact state is drawn as ground markers under the feet.

No mesh, no scary gray bodies — just lines and dots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# SMPL 24-joint id → name
JN = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
      "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
      "neck", "left_collar", "right_collar", "head", "left_shoulder",
      "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
      "left_hand", "right_hand"]
IDX = {n: i for i, n in enumerate(JN)}

# limb segments in the classic 2D-pose style, grouped with stable colors
LIMB_GROUPS = {
    "torso": ([("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
               ("spine3", "neck"), ("neck", "head"),
               ("left_shoulder", "right_shoulder"),
               ("pelvis", "left_hip"), ("pelvis", "right_hip"),
               ("left_shoulder", "spine3"), ("right_shoulder", "spine3"),
               ("left_hip", "right_hip")], "#7f7f7f"),
    "left_arm": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                  ("left_wrist", "left_hand"),
                  ("left_collar", "left_shoulder")], "#1f77b4"),
    "right_arm": ([("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                   ("right_wrist", "right_hand"),
                   ("right_collar", "right_shoulder")], "#d62728"),
    "left_leg": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                  ("left_ankle", "left_foot")], "#2ca02c"),
    "right_leg": ([("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                   ("right_ankle", "right_foot")], "#ff7f0e"),
}

CONTACT_COLORS = {0: "#bbbbbb", 1: "#2ca02c", 2: "#ff7f0e", 3: "#1f77b4"}
CONTACT_NAMES = {0: "airborne", 1: "left support", 2: "right support", 3: "double support"}


def _project(joints: np.ndarray, view: str) -> np.ndarray:
    """joints (N,3) MS_WORLD -> 2D plot coords.

    front: camera at +Z looking -Z -> horizontal = world X, vertical = Y
    side:  camera at +X looking -X -> horizontal = -Z, vertical = Y
    """
    if view == "front":
        return joints[:, [0, 1]]
    if view == "side":
        return np.stack([-joints[:, 2], joints[:, 1]], axis=1)
    raise ValueError(view)


def draw_pose(ax, joints3d: np.ndarray, view: str = "front", contact_state: int | None = None,
              lw: float = 8.0, joint_size: float = 40.0, head_circle: bool = True):
    """Draw one frame of the stick figure onto a matplotlib axis."""
    p = _project(joints3d, view)
    for _name, (segments, color) in LIMB_GROUPS.items():
        for a, b in segments:
            ia, ib = IDX[a], IDX[b]
            ax.plot(p[[ia, ib], 0], p[[ia, ib], 1], color=color, lw=lw,
                    solid_capstyle="round", zorder=2)
    ax.scatter(p[:, 0], p[:, 1], c="white", edgecolors="black", linewidths=0.8,
               s=joint_size, zorder=3)
    if head_circle:
        # head circle centered between neck and head, radius from head joint
        neck, head = p[IDX["neck"]], p[IDX["head"]]
        c = (neck + head) / 2
        r = max(np.linalg.norm(head - neck) * 1.1, 0.055)
        ax.add_patch(plt_circle(c, r))
    if contact_state is not None:
        # ground markers under both feet, colored by support state
        for side in ("left", "right"):
            ankle = joints3d[IDX[f"{side}_ankle"]]
            g = _project(np.array([[ankle[0], 0.0, ankle[2]]]), view)[0]
            ax.plot([g[0]], [g[1]], marker="o", markersize=11,
                    color=CONTACT_COLORS[contact_state], zorder=4,
                    markeredgecolor="black", markeredgewidth=0.6)


def plt_circle(center, radius):
    import matplotlib.patches as mp

    return mp.Circle(center, radius, fill=False, color="black", lw=2.0, zorder=3)


def render_frames(npz_path: str | Path, out_dir: str | Path, view: str = "front",
                  step: int = 4, dpi: int = 96) -> list[Path]:
    """Render stick-figure PNG frames (for GIF assembly)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path, allow_pickle=False)
    joints = z["joint_positions"].astype(np.float64)
    state = z["contact_state"].astype(int) if "contact_state" in z.files else None
    T = joints.shape[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_xy = _project(joints.reshape(-1, 3), view)
    pad = 0.15
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad * 2, all_xy[:, 1].max() + pad
    h = xmax - xmin

    paths = []
    for t in range(0, T, step):
        fig, ax = plt.subplots(figsize=(6.4, 6.4 * (ymax - ymin) / h + 0.5), dpi=dpi)
        cs = int(state[t]) if state is not None else None
        draw_pose(ax, joints[t], view, contact_state=cs)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"frame {t + 1}/{T}", fontsize=10, color="#555555")
        if state is not None:
            ax.text(0.01, 0.01, CONTACT_NAMES[cs], transform=ax.transAxes,
                    fontsize=9, color=CONTACT_COLORS[cs], va="bottom")
        fig.tight_layout(pad=0.3)
        p = out_dir / f"stick_{view}_f{t + 1:04d}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def render_gif(npz_path: str | Path, out_gif: str | Path, view: str = "front",
               step: int = 4, fps: float = 25.0, dpi: int = 96) -> Path:
    """Render frames and assemble an animated GIF at the motion's fps."""
    from PIL import Image

    frames = render_frames(npz_path, Path(out_gif).parent / f"stickframes_{view}", view, step, dpi)
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    for f in frames:
        f.unlink()  # keep only the gif
    (Path(out_gif).parent / f"stickframes_{view}").rmdir()
    return Path(out_gif)


def render_sheet(npz_path: str | Path, out_png: str | Path, view: str = "front",
                 n_frames: int = 12) -> Path:
    """Key-frame contact sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path, allow_pickle=False)
    joints = z["joint_positions"].astype(np.float64)
    state = z["contact_state"].astype(int) if "contact_state" in z.files else None
    T = joints.shape[0]
    idxs = np.linspace(0, T - 1, n_frames).astype(int)

    all_xy = _project(joints.reshape(-1, 3), view)
    xmin, xmax = all_xy[:, 0].min() - 0.1, all_xy[:, 0].max() + 0.1
    ymin, ymax = all_xy[:, 1].min() - 0.2, all_xy[:, 1].max() + 0.1
    cols = 4
    rows = int(np.ceil(n_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2 * (ymax - ymin) / (xmax - xmin)), dpi=110)
    for ax in np.asarray(axes).ravel():
        ax.axis("off")
    for k, t in enumerate(idxs):
        ax = np.asarray(axes).ravel()[k]
        draw_pose(ax, joints[t], view, contact_state=int(state[t]) if state is not None else None,
                  lw=6.0, joint_size=26.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"f{t + 1}", fontsize=9, color="#555555")
    fig.suptitle("Arm Wave — stick figure (pose-estimation style), colored feet markers = contact state",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return Path(out_png)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--view", default="front")
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()
    gif = render_gif(args.npz, args.gif or Path(args.npz).with_name(f"stick_{args.view}.gif"),
                     view=args.view, step=args.step)
    print("GIF", gif)
    if args.sheet:
        print("SHEET", render_sheet(args.npz, args.sheet, view=args.view))
