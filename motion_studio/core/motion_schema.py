"""Motion Studio unified motion data schema.

Defines the canonical human motion data structure used by every module.
Body model: SMPL (24 joints, 10 betas) — matching GVHMR release output.

Coordinate system (canonical, "MS_WORLD"):
    +Y up (gravity-aligned), right-handed, units in meters.
    Root translation and joint positions are expressed in this world frame.
    SMPL convention: character faces +Z in T-pose.

Rotation representation (canonical):
    SMPL axis-angle, stored as float64 (T, 3) per joint group:
      - global_orient: (T, 3)   root orientation in MS_WORLD
      - body_pose:     (T, 63)  21 body joints x 3 (root excluded)
    Quaternions (w,x,y,z) are used internally for interpolation; exports keep
    axis-angle so downstream consumers see exactly the SMPL convention.

Time:
    Frames are 1:1 with source video frames. `motion_fps` is the nominal
    frame rate of the motion timeline (frames per second). GVHMR internally
    assumes 30 fps during inference; the actual source fps is kept in
    metadata and used for physical (metric) velocity computations.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SMPL 24-joint skeleton definition
# ---------------------------------------------------------------------------

SMPL_JOINT_NAMES: list[str] = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10  (toe)
    "right_foot",      # 11  (toe)
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
    "left_hand",       # 22
    "right_hand",      # 23
]

SMPL_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21,
]

# body_pose (63) covers joints 1..23 except 22,23? No: SMPL body_pose in smplx
# covers 23 joints (1..23). GVHMR outputs 63 = 21 joints (1..21), hands (22,23)
# stay neutral. Mapping of the 21 body_pose entries to joint indices:
BODY_POSE_JOINT_IDS: list[int] = [i for i in range(1, 24) if i not in (22, 23)]

JOINT_INDEX: dict[str, int] = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}

# Convenience joint groups for editing (indices into the 24-joint skeleton)
JOINT_GROUPS: dict[str, list[int]] = {
    "all": list(range(24)),
    "root": [0],
    "upper_body": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_arm": [13, 16, 18, 20, 22],
    "right_arm": [14, 17, 19, 21, 23],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "spine": [3, 6, 9],
    "head_neck": [12, 15],
    "left_foot": [7, 10],
    "right_foot": [8, 11],
}

# joints represented in body_pose (63-dim): ids and their (joint_id -> pose_idx)
BODY_POSE_INDEX: dict[int, int] = {jid: i for i, jid in enumerate(BODY_POSE_JOINT_IDS)}

# ---------------------------------------------------------------------------
# Canonical field specification
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
PROCESSING_VERSION = "motion_studio-1.0.0"

COORDINATE_SYSTEM = {
    "name": "MS_WORLD",
    "up_axis": "+Y",
    "handedness": "right",
    "facing": "+Z at identity (SMPL T-pose)",
    "units": "meters",
    "ground": "XZ plane at the lowest foot contact (gravity along -Y)",
}

ROTATION_REPRESENTATION = "smpl_axis_angle"

# (field, path, shape_suffix, dtype, required)
BODY_FIELDS = [
    ("global_orient", ("body",), (3,), "f8", True),
    ("body_pose", ("body",), (63,), "f8", True),
    ("betas", ("body",), (10,), "f8", True),
    ("transl", ("body",), (3,), "f8", True),
    ("joint_positions", ("body",), (24, 3), "f8", True),
]

FOOT_JOINT_IDS = {"left_foot": (7, 10), "right_foot": (8, 11)}


class MotionValidationError(ValueError):
    """Raised when motion data violates the schema."""


def validate_motion_dict(data: dict, *, strict_timeline: bool = True) -> list[str]:
    """Validate a motion-data nested dict against the schema.

    Returns a list of error strings (empty == valid). Raises nothing.
    """
    errors: list[str] = []
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        errors.append("metadata: missing or not a dict")
        return errors

    for key in ("motion_fps", "frame_count", "duration", "body_model", "coordinate_system"):
        if meta.get(key) is None:
            errors.append(f"metadata.{key}: missing")

    frame_count = meta.get("frame_count")
    fps = meta.get("motion_fps")
    if isinstance(frame_count, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
        dur = frame_count / fps
        if abs(meta.get("duration", dur) - dur) > 1e-6:
            errors.append(f"metadata.duration {meta.get('duration')} != frame_count/fps {dur}")
    else:
        if not isinstance(frame_count, (int, float)) or frame_count <= 0:
            errors.append(f"metadata.frame_count invalid: {frame_count}")
        if not isinstance(fps, (int, float)) or fps <= 0:
            errors.append(f"metadata.motion_fps invalid: {fps}")

    body = data.get("body")
    if not isinstance(body, dict):
        errors.append("body: missing or not a dict")
        return errors

    T = int(frame_count) if isinstance(frame_count, (int, float)) and frame_count > 0 else None
    for field, _, suffix, _dtype, required in BODY_FIELDS:
        arr = body.get(field)
        if arr is None:
            if required:
                errors.append(f"body.{field}: missing")
            continue
        import numpy as np

        arr = np.asarray(arr)
        if T is not None and arr.shape != (T, *suffix):
            errors.append(f"body.{field}: shape {arr.shape} != expected {(T, *suffix)}")
        if np.issubdtype(arr.dtype, np.floating):
            nan = int(np.isnan(arr).sum())
            inf = int(np.isinf(arr).sum())
            if nan:
                errors.append(f"body.{field}: {nan} NaN values")
            if inf:
                errors.append(f"body.{field}: {inf} Inf values")

    # Rotation plausibility: axis-angle magnitudes must be finite & < 2π
    import numpy as np

    for field in ("global_orient", "body_pose"):
        arr = body.get(field)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size and np.issubdtype(arr.dtype, np.floating):
            norms = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=-1)
            if norms.size and np.nanmax(norms) > np.pi * 1.999:
                errors.append(f"body.{field}: max axis-angle magnitude {np.nanmax(norms):.3f} >= 2π (not valid aa)")

    # contacts: optional but must align with timeline if present
    contacts = data.get("contacts")
    if isinstance(contacts, dict):
        for side in ("left_foot", "right_foot"):
            c = contacts.get(side)
            if not isinstance(c, dict):
                continue
            contact = c.get("contact")
            if contact is not None and T is not None:
                import numpy as np

                contact = np.asarray(contact)
                if contact.shape != (T,):
                    errors.append(f"contacts.{side}.contact: shape {contact.shape} != ({T},)")
    if strict_timeline and errors:
        pass
    return errors
