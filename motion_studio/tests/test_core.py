"""Data-driven tests: numpy SMPL FK vs official smplx reference, schema
validation, foot contact detection."""

import numpy as np
import pytest

from motion_studio.core.motion_data import MotionData, load_gvhmr_raw
from motion_studio.core.motion_schema import validate_motion_dict
from motion_studio.processing.foot_contact import FootContactDetector

EXPECTED_ERR = 1e-4  # meters, float32 reference


def test_numpy_fk_matches_official_smplx(motion_071, fk):
    """The core numeric contract: pure-numpy FK == GVHMR's smplx joints."""
    from conftest import RAW_071

    if not RAW_071.exists():
        pytest.skip("GVHMR output missing")
    joints, _ = fk.joints_from_params(
        motion_071.global_orient, motion_071.body_pose,
        motion_071.betas, motion_071.transl,
    )
    import numpy as np

    raw = np.load(str(RAW_071))
    ref = raw["joints_global"].astype(np.float64)
    assert joints.shape == ref.shape == (motion_071.frame_count, 24, 3)
    assert np.abs(joints - ref).max() < EXPECTED_ERR


def test_motiondata_valid(motion_071):
    errs = motion_071.validate()
    assert errs == [], errs


def test_schema_catches_nan(motion_071):
    d = motion_071.to_dict()
    d["body"]["transl"] = d["body"]["transl"].copy()
    d["body"]["transl"][0, 1] = np.nan
    errs = validate_motion_dict(d)
    assert any("NaN" in e for e in errs)


def test_schema_catches_shape(motion_071):
    d = motion_071.to_dict()
    d["body"]["body_pose"] = d["body"]["body_pose"][:, :60]
    errs = validate_motion_dict(d)
    assert any("shape" in e for e in errs)


def test_schema_catches_missing(motion_071):
    d = motion_071.to_dict()
    d["body"].pop("betas")
    errs = validate_motion_dict(d)
    assert any("betas" in e for e in errs)


def test_schema_catches_bad_aa(motion_071):
    d = motion_071.to_dict()
    d["body"]["global_orient"] = d["body"]["global_orient"].copy()
    d["body"]["global_orient"][0, 0] = 7.0  # > 2π
    errs = validate_motion_dict(d)
    assert any("2π" in e for e in errs)


def test_world_frame_is_y_up(motion_071):
    jp = motion_071.joint_positions
    assert (jp[:, 15, 1] > jp[:, 0, 1]).all()   # head above pelvis
    assert (jp[:, 0, 1] > jp[:, [7, 8], 1].mean()).all()  # pelvis above ankles


# ---------------------------------------------------------------------------
# foot contact
# ---------------------------------------------------------------------------

def test_foot_contact_synthetic_alternating(fk, synthetic_step_motion):
    md = synthetic_step_motion
    md.joint_positions, _ = fk.joints_from_params(
        md.global_orient, md.body_pose, md.betas, md.transl
    )
    det = FootContactDetector()
    c = det.detect(md.joint_positions, md.motion_fps)
    # first 25 frames: both feet down (double support)
    assert (c["state"][:20] == 3).all()
    # 25-49 left lifted -> right support
    assert (c["state"][30:45] == 2).all()
    # 50-74 right lifted -> left support
    assert (c["state"][55:70] == 1).all()
    s = FootContactDetector.summarize(c)
    assert s["double_support_ratio"] > 0.2
    assert s["left_contact_ratio"] > 0.5
    assert s["right_contact_ratio"] > 0.5
    # intervals present and non-overlapping in time for opposite feet


def test_foot_contact_flagged_estimate(fk, synthetic_step_motion):
    md = synthetic_step_motion
    md.joint_positions, _ = fk.joints_from_params(
        md.global_orient, md.body_pose, md.betas, md.transl
    )
    c = FootContactDetector().detect(md.joint_positions, md.motion_fps)
    assert c["is_estimate"] is True
    assert "estimate" in c["left_foot"]["method"].lower()
