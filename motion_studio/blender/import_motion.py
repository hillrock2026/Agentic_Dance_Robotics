"""Blender motion importer — run inside Blender's Python:

    blender --background --python motion_studio/blender/import_motion.py -- \
        --npz <clean_motion.npz> [--blend <out.blend>] [--mode skeleton|mesh|both]

Builds an SMPL-24 armature whose bone transforms EXACTLY follow the motion
(verified against numpy FK; see validate_fk.py), with:
  - quaternion rotation keyframes per bone (rotation_mode QUATERNION)
  - root location keyframes (root translation preserved)
  - scene fps / frame range from the motion
  - ground grid, cameras (Front/Side/Top), foot-contact state markers
  - optional exact mesh via Mesh Cache (PC2) and approximated skinned mesh

Coordinate mapping MS_WORLD (Y-up, face +Z) -> Blender (Z-up):
    (x, y, z)_ms -> (x, -z, y)_bl
"""

import bpy  # noqa: E402
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.core.coordinate_system import (  # noqa: E402
    MS2BLENDER,
    points_ms_to_blender,
    rotmat_ms_to_blender,
)
from motion_studio.core.rotations import aa_to_mat  # noqa: E402
from motion_studio.core.smpl_model import SMPLForwardKinematics  # noqa: E402

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                13, 14, 16, 17, 18, 19, 20, 21]
JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
               "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
               "neck", "left_collar", "right_collar", "head", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hand", "right_hand"]
# bone tail target: first child joint (leaf bones extended artificially)
TAIL_TARGET = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12,
               10: None, 11: None, 12: 15, 13: 16, 14: 17, 15: None, 16: 18,
               17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: None, 23: None}


def load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    import json

    meta = json.loads(str(z["metadata_json"]))
    return z, meta


STICK_GROUPS = {
    "torso": ([("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
               ("spine3", "neck"), ("neck", "head"),
               ("left_shoulder", "right_shoulder"),
               ("pelvis", "left_hip"), ("pelvis", "right_hip"),
               ("left_shoulder", "spine3"), ("right_shoulder", "spine3")], (0.5, 0.5, 0.5, 1.0)),
    "left_arm": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                  ("left_wrist", "left_hand")], (0.12, 0.47, 0.71, 1.0)),
    "right_arm": ([("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                   ("right_wrist", "right_hand")], (0.84, 0.15, 0.16, 1.0)),
    "left_leg": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                  ("left_ankle", "left_foot")], (0.17, 0.63, 0.17, 1.0)),
    "right_leg": ([("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                   ("right_ankle", "right_foot")], (1.0, 0.5, 0.05, 1.0)),
}


def _cylinder_mesh(p0, p1, radius, n_sides, rings, ring_weights):
    """Cylinder from p0 to p1. rings: list of t in [0,1]; ring_weights[i] =
    weight of bone A (rest = weight to p0's bone) for ring i, blended to bone B."""
    import math

    axis = np.asarray(p1) - np.asarray(p0)
    length = float(np.linalg.norm(axis))
    axis = axis / max(length, 1e-8)
    # orthonormal basis
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= max(np.linalg.norm(u), 1e-8)
    v = np.cross(axis, u)
    verts, faces, vweights = [], [], []
    ang = 2 * np.pi / n_sides
    for ri, t in enumerate(rings):
        center = np.asarray(p0) + axis * length * t
        wA = ring_weights[ri]
        for s in range(n_sides):
            d = u * np.cos(ang * s) * radius + v * np.sin(ang * s) * radius
            verts.append(tuple(center + d))
            vweights.append(wA)
    for ri in range(len(rings) - 1):
        for s in range(n_sides):
            a = ri * n_sides + s
            b = ri * n_sides + (s + 1) % n_sides
            c = (ri + 1) * n_sides + (s + 1) % n_sides
            d = (ri + 1) * n_sides + s
            faces.append((a, b, c, d))
    return verts, faces, vweights


def add_stick_figure(arm_ob, rest_bl, line_radius=0.022):
    """Pose-estimation style stick figure driven by the armature:
    colored CYLINDER segments (thick, renderable — endpoint rings skinned to
    the respective bones so they follow the animation exactly) + joint
    spheres parented to bones. No body mesh."""
    # --- lines: one cylinder mesh per color group ---
    for gname, (segments, color) in STICK_GROUPS.items():
        all_v, all_f = [], []
        boneA_of, boneB_of, wA_of = [], [], []
        for a, b in segments:
            ia, ib = JOINT_NAMES.index(a), JOINT_NAMES.index(b)
            # rings at 0, .5, 1; endpoints rigid to their bones, middle blends
            cv, cf, cw = _cylinder_mesh(
                rest_bl[ia], rest_bl[ib], line_radius, 8,
                rings=[0.0, 0.5, 1.0], ring_weights=[1.0, 0.5, 0.0],
            )
            base = len(all_v)
            all_v.extend(cv)
            all_f.extend([tuple(i + base for i in f) for f in cf])
            for k, wA in enumerate(cw):
                boneA_of.append(f"MS_{a}")
                boneB_of.append(f"MS_{b}")
                wA_of.append(wA)
        me = bpy.data.meshes.new(f"stick_{gname}")
        me.from_pydata([tuple(map(float, c)) for c in all_v], [], [tuple(f) for f in all_f])
        me.validate()
        ob = bpy.data.objects.new(f"STICK_{gname.upper()}", me)
        bpy.context.collection.objects.link(ob)
        # vertex groups: weight wA to bone A's group and (1-wA) to bone B's
        allbones = set(boneA_of) | set(boneB_of)
        vgs = {bn: ob.vertex_groups.new(name=bn) for bn in allbones}
        for vi in range(len(all_v)):
            wA = wA_of[vi]
            vgs[boneA_of[vi]].add([vi], max(wA, 1e-4), "REPLACE")
            vgs[boneB_of[vi]].add([vi], max(1.0 - wA, 1e-4), "REPLACE")
        mod = ob.modifiers.new("arm", "ARMATURE")
        mod.object = arm_ob
        mod.use_bone_envelopes = False
        mod.use_vertex_groups = True
        mat = bpy.data.materials.new(f"stick_{gname}")
        mat.diffuse_color = color
        mat.use_nodes = False
        mat.roughness = 0.6
        ob.data.materials.append(mat)
        ob.show_wire = False

    # --- joint spheres: skinned to bones like the cylinders (exact LBS path;
    # a vertex at rest joint j weighted 1.0 to bone MS_j poses to the posed
    # joint — bone parenting was empirically displaced and is not used) ---
    import math as _math

    def _sphere(center, radius, n_seg, n_ring):
        vs = []
        for r in range(n_ring + 1):
            phi = _math.pi * r / n_ring
            for c in range(n_seg):
                th = 2 * _math.pi * c / n_seg
                vs.append((center[0] + radius * _math.sin(phi) * _math.cos(th),
                           center[1] + radius * _math.sin(phi) * _math.sin(th),
                           center[2] + radius * _math.cos(phi)))
        fs = []
        for r in range(n_ring):
            for c in range(n_seg):
                c2 = (c + 1) % n_seg
                a, b, d, e = r * n_seg + c, r * n_seg + c2, (r + 1) * n_seg + c2, (r + 1) * n_seg + c
                fs.append((a, b, d, e))
        return vs, fs

    jverts, jfaces, jbone = [], [], []
    for jn in JOINT_NAMES:
        i = JOINT_NAMES.index(jn)
        radius = 0.042 * (1.5 if jn == "head" else 1.0)
        vs, fs = _sphere(tuple(map(float, rest_bl[i])), radius, 14, 10)
        base = len(jverts)
        jverts.extend(vs)
        jfaces.extend(tuple(k + base for k in f) for f in fs)
        jbone.extend([jn] * len(vs))
    jme = bpy.data.meshes.new("stick_joints")
    jme.from_pydata([tuple(map(float, v)) for v in jverts], [], [tuple(f) for f in jfaces])
    jme.validate()
    job = bpy.data.objects.new("STICK_JOINTS", jme)
    bpy.context.collection.objects.link(job)
    for jn in set(jbone):
        job.vertex_groups.new(name=f"MS_{jn}")
    for vi, jn in enumerate(jbone):
        job.vertex_groups[f"MS_{jn}"].add([vi], 1.0, "REPLACE")
    jmod = job.modifiers.new("arm", "ARMATURE")
    jmod.object = arm_ob
    jmod.use_vertex_groups = True
    jmod.use_bone_envelopes = False
    jmat = bpy.data.materials.new("stick_joint")
    jmat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    jmat.use_nodes = False
    job.data.materials.append(jmat)
    for poly in job.data.polygons:
        poly.use_smooth = True
    return None


def build_armature(name, rest_bl):
    """rest_bl: (24,3) Blender-space rest joints (T-pose, shaped)."""
    arm = bpy.data.armatures.new(name + "_arm")
    ob = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for i, jn in enumerate(JOINT_NAMES):
            b = arm.edit_bones.new(f"MS_{jn}")
            b.head = rest_bl[i]
            parent = SMPL_PARENTS[i]
            tgt = TAIL_TARGET[i]
            if tgt is not None:
                b.tail = rest_bl[tgt]
            else:
                d = rest_bl[i] - rest_bl[parent]
                n = np.linalg.norm(d)
                d = d / max(n, 1e-8)
                b.tail = rest_bl[i] + d * 0.08
            if parent >= 0:
                b.parent = arm.edit_bones[f"MS_{JOINT_NAMES[parent]}"]
                b.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return ob


def desired_world_matrices(z, fk):
    """(T,24) Blender-space 4x4 world transforms from SMPL FK."""
    go = z["global_orient"].astype(np.float64)
    bp = z["body_pose"].astype(np.float64)
    betas = z["betas"].astype(np.float64)
    transl = z["transl"].astype(np.float64)
    R_world = fk.global_rotmats(go, bp)                # (T,24,3,3) MS_WORLD
    joints = z["joint_positions"].astype(np.float64)   # (T,24,3)
    T = go.shape[0]
    A = np.zeros((T, 24, 4, 4))
    A[:, :, 3, 3] = 1.0
    A[:, :, :3, :3] = rotmat_ms_to_blender(R_world.reshape(-1, 3, 3)).reshape(T, 24, 3, 3)
    A[:, :, :3, 3] = points_ms_to_blender(joints.reshape(-1, 3)).reshape(T, 24, 3)
    return A


def keyframe_motion(ob, A):
    """Set matrix_basis per frame, keyframe loc/quat channels.

    Verified relations (Blender 4.5):
      root:  pose_0 = rest_0 @ BASIS_0
      child: pose_i = pose_parent @ (rest_parent^-1 rest_i) @ BASIS_i
    """
    arm = ob.data
    dbones = {b.name: b for b in arm.bones}
    pbones = ob.pose.bones
    for pb in pbones:
        pb.rotation_mode = "QUATERNION"

    T = A.shape[0]
    action = bpy.data.actions.new(ob.name + "_action")
    ad = ob.animation_data_create()
    ad.action = action

    from mathutils import Matrix, Quaternion, Vector

    # precompute rest matrices
    rest = {i: Matrix(dbones[f"MS_{JOINT_NAMES[i]}"].matrix_local) for i in range(24)}

    for t in range(T):
        pose_mats = {}
        for i in range(24):
            A_i = Matrix(A[t, i].tolist())
            pb = pbones[f"MS_{JOINT_NAMES[i]}"]
            parent = SMPL_PARENTS[i]
            if parent < 0:
                basis = rest[i].inverted() @ A_i
            else:
                offs = rest[parent].inverted() @ rest[i]
                basis = offs.inverted() @ pose_mats[parent].inverted() @ A_i
            loc, rot_q, scale = basis.decompose()
            pb.location = loc
            pb.rotation_quaternion = rot_q
            pb.scale = scale
            pb.keyframe_insert("location", frame=t + 1)
            pb.keyframe_insert("rotation_quaternion", frame=t + 1)
            pose_mats[i] = A_i
    return action


def setup_scene(z, meta):
    scene = bpy.context.scene
    fps = float(z["motion_fps"])
    scene.render.fps = int(round(fps))
    scene.render.fps_base = int(round(fps)) / fps
    T = int(z["frame_count"])
    scene.frame_start = 1
    scene.frame_end = T

    # ground grid at estimated ground height
    gz = 0.0
    contacts = z["contact_state"]
    if "ground_y" in z:
        pass
    bpy.ops.object.light_add(type="SUN", location=(3, -4, 8))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.3, 0.6)

    # cameras: front / side / top
    cams = {}
    for cname, rot, loc in [
        ("CAM_FRONT", (np.pi / 2, 0, 0), (0, -4.5, 1.2)),
        ("CAM_SIDE", (np.pi / 2, 0, np.pi / 2), (4.5, 0.5, 1.2)),
        ("CAM_TOP", (0, 0, 0), (0, 0.3, 5.5)),
    ]:
        bpy.ops.object.camera_add(location=loc, rotation=rot)
        cam = bpy.context.object
        cam.name = cname
        cam.data.lens = 50
        cams[cname] = cam
    scene.camera = cams["CAM_FRONT"]
    return scene


def add_contact_markers(z, ob):
    """4 colored spheres keyed visible per frame by contact_state."""
    state = z["contact_state"].astype(int)
    mats = {}
    colors = {1: (0.1, 0.9, 0.2, 1.0), 2: (0.95, 0.3, 0.15, 1.0),
              3: (0.2, 0.5, 1.0, 1.0), 0: (0.6, 0.6, 0.6, 0.4)}
    names = {1: "LEFT_SUPPORT", 2: "RIGHT_SUPPORT", 3: "DOUBLE_SUPPORT", 0: "AIRBORNE"}
    root_head_world = lambda t: np.array(ob.pose.bones["MS_pelvis"].matrix.translation)
    for st in (0, 1, 2, 3):
        mat = bpy.data.materials.new(names[st])
        mat.diffuse_color = colors[st]
        mat.use_nodes = False
        mats[st] = mat
    # parent panel: markers in front of the character, keyed visibility
    for st in (0, 1, 2, 3):
        name = names[st]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09, location=(-0.9, 0.9, 0.15))
        s = bpy.context.object
        s.name = "MARK_" + name
        s.data.materials.append(mats[st])
        s.scale = (1, 1, 1)
        for t in range(len(state)):
            on = 1.0 if state[t] == st else 0.0
            s.scale = (on, on, on)
            s.keyframe_insert("scale", frame=t + 1)
    return mats


def add_mesh(z, fk, arm_ob, rest_joints_ms):
    """Approximate skinned mesh (LBS without pose blendshapes) + optional
    exact PC2 mesh cache if a .pc2 file sits next to the npz."""
    import json

    v_shaped = (
        fk.v_template[None] + np.einsum("vkb,b->vk", fk.shapedirs, z["betas"].astype(np.float64).mean(axis=0))
    ).squeeze(0)
    verts_bl = points_ms_to_blender(v_shaped)
    me = bpy.data.meshes.new("SMPL_mesh")
    me.from_pydata([tuple(map(float, v)) for v in verts_bl], [], [tuple(map(int, f)) for f in fk.faces])
    me.validate()
    mesh_ob = bpy.data.objects.new("SMPL_MESH", me)
    bpy.context.collection.objects.link(mesh_ob)
    # vertex groups from SMPL weights
    for i, jn in enumerate(JOINT_NAMES):
        mesh_ob.vertex_groups.new(name=f"MS_{jn}")
    w = fk.weights
    dominant = np.argmax(w, axis=1)
    vg_add = mesh_ob.vertex_groups
    for v in range(w.shape[0]):
        vg_add[dominant[v]].add([v], 1.0, "REPLACE")
    mod = mesh_ob.modifiers.new("arm", "ARMATURE")
    mod.object = arm_ob
    mesh_ob.parent = arm_ob
    return mesh_ob


def add_pc2_mesh(npz_path, fps):
    """Exact mesh replay via Mesh Cache modifier if PC2 exists."""
    pc2 = Path(npz_path).with_suffix(".pc2")
    if not pc2.exists():
        return None
    # need vertex template: create from smpl core model
    fk = get_fk()
    v0 = fk.v_template
    verts_bl = points_ms_to_blender(v0)
    me = bpy.data.meshes.new("SMPL_exact")
    me.from_pydata([tuple(map(float, v)) for v in verts_bl], [], [tuple(map(int, f)) for f in fk.faces])
    me.validate()
    ob = bpy.data.objects.new("SMPL_EXACT_MESH", me)
    bpy.context.collection.objects.link(ob)
    mod = ob.modifiers.new("meshcache", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = str(pc2)
    mod.forward_axis = "POS_Y"
    mod.up_axis = "POS_Z"
    return ob


_FK = None


def get_fk():
    global _FK
    if _FK is None:
        _FK = SMPLForwardKinematics(
            PROJECT_ROOT / "motion_studio/assets/smpl/smpl_neutral_core.npz"
        )
    return _FK


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--mode", default="skeleton", choices=["skeleton", "mesh", "both"],
                    help="skeleton: pose-estimation stick figure only (default); "
                         "mesh/both: additionally add SMPL body meshes")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    z, meta = load_npz(args.npz)
    fk = get_fk()

    # rest skeleton: shaped T-pose joints (mean betas), Blender coords
    betas_mean = z["betas"].astype(np.float64).mean(axis=0)
    v_shaped = fk.v_template + np.einsum("vkb,b->vk", fk.shapedirs, betas_mean)
    rest_ms = np.einsum("jv,vk->jk", fk.J_regressor, v_shaped)
    rest_bl = points_ms_to_blender(rest_ms)

    name = args.name or Path(args.npz).stem
    arm_ob = build_armature(name, rest_bl)
    A = desired_world_matrices(z, fk)
    keyframe_motion(arm_ob, A)
    setup_scene(z, meta)
    add_contact_markers(z, arm_ob)
    # pose-estimation style stick figure (lines + joint dots), always present
    arm_ob.data.display_type = "STICK"  # armature itself displays as thin sticks
    add_stick_figure(arm_ob, rest_bl)
    if args.mode in ("mesh", "both"):
        add_mesh(z, fk, arm_ob, rest_ms)
        add_pc2_mesh(args.npz, float(z["motion_fps"]))

    blend_path = args.blend or str(Path(args.npz).with_suffix(".blend"))
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"BLENDER_IMPORT_OK {blend_path} frames={int(z['frame_count'])} fps={float(z['motion_fps']):.3f}")


if __name__ == "__main__":
    main()
