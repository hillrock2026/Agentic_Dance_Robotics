"""Pure-numpy SMPL forward kinematics.

Mirrors the official `smplx.SMPL` forward pass so that joint positions
recomputed in Motion Studio match the GVHMR/PyTorch reference bit-close:

    v_shaped = v_template + shapedirs @ betas            (10475,3)
    J_shaped = J_regressor @ v_shaped                    (24,3)
    R = [global_orient, body_pose(63->zero-pad hands)]   per-joint rotmats
    pose_feature = (R - I).flatten over joints           (207,)
    v_posed = v_shaped + posedirs @ pose_feature
    LBS: A_i = A_parent @ [R_i | (J_i - J_parent)]       global transforms
    joints_i = A_i[:3,:3] @ J_shaped_i + A_i[:3,3]
    joint_positions += transl   (root translation)

The model arrays are exported once from the official SMPL_NEUTRAL.pkl by
`motion_studio/gvhmr/export_raw_npz.py` (run inside the GVHMR environment)
into `assets/smpl/smpl_neutral_core.npz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .motion_schema import BODY_POSE_JOINT_IDS, SMPL_PARENTS
from .rotations import aa_to_mat


class SMPLForwardKinematics:
    def __init__(self, model_npz: str | Path):
        model_npz = Path(model_npz)
        if not model_npz.exists():
            raise FileNotFoundError(
                f"SMPL core model not found at {model_npz}. "
                "Run motion_studio/gvhmr/export_raw_npz.py inside the GVHMR env first."
            )
        m = np.load(model_npz)
        self.v_template = m["v_template"].astype(np.float64)        # (V,3)
        self.shapedirs = m["shapedirs"].astype(np.float64)          # (V,3,10)
        self.posedirs = m["posedirs"].astype(np.float64)            # (V,3,207)
        self.J_regressor = m["J_regressor"].astype(np.float64)      # (24,V)
        self.weights = m["weights"].astype(np.float64)              # (V,24)
        parents = m["parents"].astype(np.int64)
        parents[parents > 2**31] = -1  # root stored as uint32 max in SMPL pkl
        self.parents = parents          # (24,)
        self.faces = m["faces"].astype(np.int64)                    # (F,3)
        self.joint_names = [str(s) for s in m["joint_names"]]
        self.num_betas = self.shapedirs.shape[-1]
        assert list(self.parents) == list(SMPL_PARENTS), "parent array mismatch with schema"
        assert self.joint_names == [str(s) for s in m["joint_names"]]

    # ------------------------------------------------------------------
    def joints_from_params(
        self,
        global_orient: np.ndarray,  # (T,3)
        body_pose: np.ndarray,      # (T,63)
        betas: np.ndarray,          # (T,10)
        transl: np.ndarray,         # (T,3)
        return_vertices: bool = False,
    ):
        """Recompute 24-joint world positions (T,24,3) from SMPL parameters."""
        global_orient = np.atleast_2d(np.asarray(global_orient, dtype=np.float64))
        body_pose = np.atleast_2d(np.asarray(body_pose, dtype=np.float64))
        betas = np.atleast_2d(np.asarray(betas, dtype=np.float64))
        transl = np.atleast_2d(np.asarray(transl, dtype=np.float64))
        T = global_orient.shape[0]

        # --- shape blend ---
        # (T,V,3) = (V,3) + (V,3,10) @ (T,10)^T
        v_shaped = self.v_template[None] + np.einsum("vkb,tb->tvk", self.shapedirs, betas)
        # (T,24,3)
        J_shaped = np.einsum("jv,tvk->tjk", self.J_regressor, v_shaped)

        # --- rotation mats (T,24,3,3); hands (22,23) stay identity ---
        rotmats = np.zeros((T, 24, 3, 3), dtype=np.float64)
        rotmats[:, 0] = aa_to_mat(global_orient)
        body_aa = body_pose.reshape(T, len(BODY_POSE_JOINT_IDS), 3)
        for k, jid in enumerate(BODY_POSE_JOINT_IDS):
            rotmats[:, jid] = aa_to_mat(body_aa[:, k])

        # --- pose blendshapes for vertices (optional) ---
        verts_out = None
        if return_vertices:
            ident = np.eye(3, dtype=np.float64)
            # pose blendshape uses the 23 non-root joints (207 = 23*9)
            pose_feature = (rotmats[:, 1:] - ident).reshape(T, 207)
            v_posed = v_shaped + np.einsum("vpk,tk->tvp", self.posedirs, pose_feature)
            verts_out = v_posed

        # --- LBS skeleton transforms ---
        # A[t,i] = A[t,parent] @ [R_i | J_i - J_parent]
        A = np.zeros((T, 24, 4, 4), dtype=np.float64)
        A[:, :, 3, 3] = 1.0
        for i in range(24):
            p = self.parents[i]
            local = np.zeros((T, 4, 4), dtype=np.float64)
            local[:, 3, 3] = 1.0
            local[:, :3, :3] = rotmats[:, i]
            if p < 0:
                local[:, :3, 3] = J_shaped[:, i]
                A[:, i] = local
            else:
                local[:, :3, 3] = J_shaped[:, i] - J_shaped[:, p]
                A[:, i] = A[:, p] @ local

        # posed world joints = translation column of the global transforms
        # (matches smplx lbs.batch_rigid_transform: posed_joints = transforms[..., :3, 3])
        joints = A[:, :, :3, 3].copy()
        joints = joints + transl[:, None, :]  # root translation in world
        if return_vertices:
            # standard LBS: v_out[t,v] = sum_j W[v,j] * (A[t,j] @ v_h[t,v])
            v_posed_h = np.concatenate([verts_out, np.ones((*verts_out.shape[:-1], 1))], axis=-1)  # (T,V,4)
            v_out_h = np.einsum("vj,tjmn,tvn->tvm", self.weights, A, v_posed_h)
            verts_out = v_out_h[..., :3] / v_out_h[..., 3:]
        return joints, verts_out

    def global_rotmats(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
    ) -> np.ndarray:
        """World-space rotation matrices per joint (T,24,3,3) (no translation)."""
        global_orient = np.atleast_2d(np.asarray(global_orient, dtype=np.float64))
        body_pose = np.atleast_2d(np.asarray(body_pose, dtype=np.float64))
        T = global_orient.shape[0]
        rotmats = np.zeros((T, 24, 3, 3), dtype=np.float64)
        rotmats[:, 0] = aa_to_mat(global_orient)
        body_aa = body_pose.reshape(T, len(BODY_POSE_JOINT_IDS), 3)
        for k, jid in enumerate(BODY_POSE_JOINT_IDS):
            rotmats[:, jid] = aa_to_mat(body_aa[:, k])
        R_world = np.zeros_like(rotmats)
        for i in range(24):
            p = self.parents[i]
            R_world[:, i] = rotmats[:, i] if p < 0 else R_world[:, p] @ rotmats[:, i]
        return R_world
