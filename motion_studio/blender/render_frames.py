"""Render preview frames from a motion .blend — run inside Blender:

    blender --background <motion.blend> --python render_frames.py -- \
        --frames 1,20,60,100 --camera CAM_FRONT --out_prefix <path/prefix>

Renders workbench-style (skeleton visible) stills for visual verification.
"""

import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="1,20,60,100")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--views", default="front,side,top")
    args = ap.parse_args(argv)

    scene = bpy.context.scene
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.image_settings.file_format = "PNG"

    # ambient world light so the figure is clearly visible on headless renders
    if scene.world is None:
        scene.world = bpy.data.worlds.new("MS_world")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        # dark backdrop so colored stick lines / meshes stand out
        bg.inputs[0].default_value = (0.02, 0.02, 0.03, 1.0)
        bg.inputs[1].default_value = 1.0
    for ob in bpy.data.objects:
        if ob.type == "LIGHT" and ob.data.name.startswith("Sun"):
            ob.data.energy = 6.0

    # find camera objects
    cams = {o.name: o for o in bpy.data.objects if o.type == "CAMERA"}
    view_cam = {
        "front": cams.get("CAM_FRONT"),
        "side": cams.get("CAM_SIDE"),
        "top": cams.get("CAM_TOP"),
    }

    frames = [int(f) for f in args.frames.split(",") if f.strip()]
    written = []
    for view in args.views.split(","):
        cam = view_cam.get(view.strip())
        if cam is None:
            continue
        scene.camera = cam
        for f in frames:
            scene.frame_set(f)
            out = f"{args.out_prefix}_{view}_f{f:04d}.png"
            scene.render.filepath = out
            bpy.ops.render.render(write_still=True)
            written.append(out)
    print("RENDER_OK", len(written), "images")


if __name__ == "__main__":
    main()
