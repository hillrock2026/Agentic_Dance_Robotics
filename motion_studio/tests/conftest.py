"""Shared fixtures: real GVHMR-derived MotionData + SMPL FK (data-driven tests).

Skips integration tests when the recovered outputs are not present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from motion_studio.core.motion_data import MotionData, load_gvhmr_raw  # noqa: E402
from motion_studio.core.smpl_model import SMPLForwardKinematics  # noqa: E402

RAW_071 = PROJECT_ROOT / "motion_studio/outputs/gvhmr/segment_071_343ee79d/segment_071/gvhmr_raw.npz"
MODEL = PROJECT_ROOT / "motion_studio/assets/smpl/smpl_neutral_core.npz"


@pytest.fixture(scope="session")
def fk():
    return SMPLForwardKinematics(MODEL)


@pytest.fixture(scope="session")
def motion_071(fk):
    if not RAW_071.exists():
        pytest.skip("GVHMR output for segment_071 not available")
    return load_gvhmr_raw(RAW_071, smpl_fk=fk)


@pytest.fixture()
def synthetic_step_motion() -> MotionData:
    """Synthetic alternating step: pelvis fixed, knees/ankles alternate lift.

    Left foot down frames 0-24, right foot down 25-49, alternating; built so
    foot-contact detector must find alternating single-support intervals.
    """
    T = 100
    t = np.arange(T)
    # body pose zeros except hips (lift legs) and knees (slight flex)
    bp = np.zeros((T, 63))
    # hip joints: body_pose idx 0 = left_hip, 1 = right_hip (BODY_POSE_INDEX)
    lift_l = ((t >= 25) & (t < 50)).astype(float)
    lift_r = ((t >= 50) & (t < 75)).astype(float)
    bp[:, 3 * 0 + 0] = -0.7 * lift_l   # left hip flexion lifts left foot
    bp[:, 3 * 1 + 0] = -0.7 * lift_r   # right hip flexion lifts right foot
    bp[:, 3 * 3 + 0] = 0.9 * lift_l    # left knee flexes the shin
    bp[:, 3 * 4 + 0] = 0.9 * lift_r    # right knee flexes the shin
    md = MotionData(
        motion_fps=25.0,
        global_orient=np.zeros((T, 3)),
        body_pose=bp,
        betas=np.zeros((T, 10)),
        # T-pose toe joint sits ~1.14 m below pelvis: root at 1.15 => toes at ground
        transl=np.tile([0.0, 1.15, 0.0], (T, 1)),
        joint_positions=None,
    )
    return md
