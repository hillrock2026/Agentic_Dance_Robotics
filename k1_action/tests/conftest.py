"""Shared fixtures: real Motion Studio data + cached small retarget result."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "virtual_k1"))

from k1_action.motion_loader import HumanMotion, load_human_motion  # noqa: E402

ARM_WAVE_NPZ = PROJECT_ROOT / "motion_studio/outputs/motions/segment_071_343ee79d/edited_motion.npz"
DATASET = PROJECT_ROOT / "datasets/k1_dance_v1"
OFFICIAL_CSV = PROJECT_ROOT / "virtual_k1/assets/booster_assets/motions/K1/k1_fight_001.csv"


@pytest.fixture(scope="session")
def human_motion():
    if not ARM_WAVE_NPZ.exists():
        pytest.skip("Motion Studio Arm Wave export missing")
    return load_human_motion(ARM_WAVE_NPZ)


@pytest.fixture(scope="session")
def human_motion_short(human_motion):
    """First 15 frames — enough for a faithful retarget test, fast."""
    hm = human_motion
    T = 15
    return HumanMotion(
        path=hm.path, sha256=hm.sha256, fps=hm.fps, frame_count=T,
        duration=T / hm.fps, timestamps=hm.timestamps[:T].copy(),
        joint_positions=hm.joint_positions[:T].copy(),
        global_orient=hm.global_orient[:T].copy(), transl=hm.transl[:T].copy(),
        body_pose=hm.body_pose[:T].copy(),
        contact_state=hm.contact_state[:T].copy() if hm.contact_state is not None else None,
        metadata=hm.metadata,
    )


@pytest.fixture(scope="session")
def arm_wave_traj_full():
    """Retarget of the full Arm Wave (cached for the session, ~15 s)."""
    if not ARM_WAVE_NPZ.exists():
        pytest.skip("Motion Studio Arm Wave export missing")
    from k1_action.retargeting import FullBodyRetargeter

    hm = load_human_motion(ARM_WAVE_NPZ)
    return FullBodyRetargeter().retarget(hm, upper_body_mode=True)
