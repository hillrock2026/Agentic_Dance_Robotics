"""Keyframe editing for MotionData.

Keyframes are frame indices marking editable anchor poses. Operations:
  - add/remove/move keyframes
  - overwrite the pose at a keyframe (all joints or a joint subset), given
    as SMPL axis-angle deltas or absolute values
  - re-interpolate all frames between consecutive keyframes using per-joint
    quaternion SLERP (translations linearly)

The edit journal records every operation with affected joints/frames.
"""

from __future__ import annotations

import numpy as np

from ..core.motion_data import MotionData
from ..core.motion_schema import BODY_POSE_INDEX, JOINT_GROUPS
from ..core.rotations import aa_to_quat, quat_to_aa, slerp


class KeyframeEditor:
    def __init__(self, md: MotionData, smpl_fk=None, detector=None):
        self.md = md
        self.smpl_fk = smpl_fk
        self.detector = detector
        self.keyframes: list[int] = []

    # ------------------------------------------------------------------
    def add_keyframe(self, frame: int) -> int:
        frame = int(frame)
        if not (0 <= frame < self.md.frame_count):
            raise IndexError(f"frame {frame} out of range 0..{self.md.frame_count - 1}")
        if frame not in self.keyframes:
            self.keyframes.append(frame)
            self.keyframes.sort()
        self.md.record_edit("keyframe_add", {"frame": frame})
        return frame

    def remove_keyframe(self, frame: int) -> None:
        if frame in self.keyframes:
            self.keyframes.remove(frame)
            self.md.record_edit("keyframe_remove", {"frame": frame})

    def move_keyframe(self, old_frame: int, new_frame: int) -> None:
        self.remove_keyframe(old_frame)
        self.add_keyframe(new_frame)

    def get_pose(self, frame: int) -> dict[str, np.ndarray]:
        f = int(frame)
        return {
            "global_orient": self.md.global_orient[f].copy(),
            "body_pose": self.md.body_pose[f].copy(),
            "transl": self.md.transl[f].copy(),
        }

    # ------------------------------------------------------------------
    def set_keyframe_pose(
        self,
        frame: int,
        body_pose: np.ndarray | None = None,
        global_orient: np.ndarray | None = None,
        transl: np.ndarray | None = None,
        joints: str | list[int] | None = None,
    ) -> None:
        """Overwrite the pose at a keyframe. `joints` limits which body_pose
        entries are replaced (joint ids); None replaces all provided arrays."""
        f = int(frame)
        if f not in self.keyframes:
            self.add_keyframe(f)

        joint_ids = None
        if joints is not None:
            joint_ids = JOINT_GROUPS[joints] if isinstance(joints, str) else list(joints)

        if body_pose is not None:
            body_pose = np.asarray(body_pose, dtype=np.float64).reshape(-1)
            if body_pose.shape == (63,):
                if joint_ids is None:
                    self.md.body_pose[f] = body_pose
                else:
                    for jid in joint_ids:
                        k = BODY_POSE_INDEX.get(jid)
                        if k is not None:
                            self.md.body_pose[f, 3 * k : 3 * k + 3] = body_pose[3 * k : 3 * k + 3]
        if global_orient is not None:
            self.md.global_orient[f] = np.asarray(global_orient, dtype=np.float64).reshape(3)
        if transl is not None:
            self.md.transl[f] = np.asarray(transl, dtype=np.float64).reshape(3)
        self.md.record_edit("keyframe_set_pose", {"frame": f, "joints": joint_ids})

    # ------------------------------------------------------------------
    def interpolate_between_keyframes(self) -> None:
        """Rebuild all frames between consecutive keyframes via SLERP.

        Frames outside [first_kf, last_kf] keep their current values.
        """
        if len(self.keyframes) < 2:
            raise ValueError("need >= 2 keyframes to interpolate")
        md = self.md
        T = md.frame_count
        kfs = [k for k in self.keyframes if 0 <= k < T]
        q_body = aa_to_quat(md.body_pose.reshape(T, 21, 3))  # (T,21,4)
        q_root = aa_to_quat(md.global_orient)  # (T,4)

        for a, b in zip(kfs[:-1], kfs[1:]):
            span = b - a
            for t in range(a + 1, b):
                u = (t - a) / span
                q_body[t] = slerp(q_body[a][None], q_body[b][None], np.array([u]))[0]
                q_root[t] = slerp(q_root[a][None], q_root[b][None], np.array([u]))[0]
                md.transl[t] = (1 - u) * md.transl[a] + u * md.transl[b]
                md.betas[t] = (1 - u) * md.betas[a] + u * md.betas[b]
        md.body_pose = quat_to_aa(q_body).reshape(T, 63)
        md.global_orient = quat_to_aa(q_root)

        if self.smpl_fk is not None:
            md.joint_positions, _ = self.smpl_fk.joints_from_params(
                md.global_orient, md.body_pose, md.betas, md.transl
            )
        if self.detector is not None and md.joint_positions is not None:
            md.contacts = self.detector.detect(md.joint_positions, md.motion_fps)
        md.record_edit("keyframe_interpolate", {"keyframes": list(kfs)})

    # ------------------------------------------------------------------
    def mirror_pose(self, frame: int) -> None:
        """Swap left/right joint rotations at a keyframe (L/R pose mirroring
        across the sagittal plane). Rotations: mirror = R_y(π) R M R_y(π)^T
        with L/R joint index swap; translation x -> -x."""
        f = int(frame)
        md = self.md
        mirror_pairs = [
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17),
            (18, 19), (20, 21), (22, 23),
        ]
        Ry_pi = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]])
        from ..core.rotations import aa_to_mat, mat_to_aa

        bp = md.body_pose[f].reshape(21, 3)
        new_bp = bp.copy()
        for ja, jb in mirror_pairs:
            ka, kb = BODY_POSE_INDEX[ja], BODY_POSE_INDEX[jb]
            Ra, Rb = aa_to_mat(bp[ka]), aa_to_mat(bp[kb])
            new_bp[ka] = mat_to_aa(Ry_pi @ Rb @ Ry_pi.T)
            new_bp[kb] = mat_to_aa(Ry_pi @ Ra @ Ry_pi.T)
        # spine joints mirror in place
        for jid in (3, 6, 9, 12, 15, 0):
            k = BODY_POSE_INDEX.get(jid)
            if k is None:
                continue
            R = aa_to_mat(bp[k])
            new_bp[k] = mat_to_aa(Ry_pi @ R @ Ry_pi.T)
        md.body_pose[f] = new_bp.reshape(63)
        gR = aa_to_mat(md.global_orient[f])
        from ..core.rotations import mat_to_aa as m2a

        md.global_orient[f] = m2a(Ry_pi @ gR @ Ry_pi.T)
        md.transl[f] = md.transl[f] * np.array([-1.0, 1, 1])
        self.md.record_edit("keyframe_mirror_pose", {"frame": f})
