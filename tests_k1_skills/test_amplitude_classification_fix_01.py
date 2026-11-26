"""Tests for K1_DANCE_SKILL_FRAMEWORK_02_AMPLITUDE_CLASSIFICATION_FIX_01.

Validates the mutually-exclusive tri-state classification of
arm_motion_amplitude when the ACTUAL series is non-finite on a
static-reference joint (the previously miscategorized case).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import k1_skills.evaluation as ev  # noqa: E402

GROUPS = {"arms": slice(2, 10), "legs": slice(10, 22)}


def _z(total=100):
    ref = np.zeros((total, 22))
    ref[:, 2] = np.sin(np.arange(total) * 0.3)     # group-local 0: moving
    act = ref.copy()                               # finite everywhere
    return {"ref_joint": ref.astype(np.float32),
            "act_joint": act.astype(np.float32)}


def _run(z):
    out = ev.arm_motion_amplitude(z, GROUPS)
    json.dumps(out, allow_nan=False)               # strict JSON gate
    return out


class TestAmplitudeClassification:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_A_B_C_static_ref_nonfinite_actual_not_static(self, bad):
        """Static reference + non-finite ACTUAL: invalid, never static."""
        z = _z()
        z["act_joint"][10, 5] = bad                # group-local 3 is static-ref
        out = _run(z)
        assert 3 in out["invalid_actual_joints"]
        assert out["static_arm_joints"] == 6       # 8 - 1 moving - 1 invalid
        assert len(out["static_joint_residual_amplitude_rad"]) == 6
        assert all(v is not None
                   for v in out["static_joint_residual_amplitude_rad"])
        assert out["moving_arm_joints"] == 1

    def test_D_valid_data_classification_unchanged(self):
        out = _run(_z())
        assert out["moving_arm_joints"] == 1
        assert out["static_arm_joints"] == 7
        assert out["invalid_reference_joints"] == []
        assert out["invalid_actual_joints"] == []
        assert out["mean_amplitude_ratio"] == pytest.approx(1.0, rel=1e-3)
        assert all(v is not None
                   for v in out["static_joint_residual_amplitude_rad"])

    def test_E_invalid_ref_valid_actual_excluded_from_classes(self):
        z = _z()
        z["ref_joint"][:, 6] = np.nan              # group-local 4 (static ref)
        out = _run(z)
        assert 4 in out["invalid_reference_joints"]
        assert out["invalid_actual_joints"] == []
        assert out["moving_arm_joints"] == 1
        assert out["static_arm_joints"] == 6       # 8 - 1 moving - 1 invalid
        assert len(out["static_joint_residual_amplitude_rad"]) == 6

    @pytest.mark.parametrize("bad,where", [
        (np.nan, "act_static"), (np.inf, "act_static"), (-np.inf, "act_static"),
        (np.nan, "act_moving"), (np.inf, "ref"), (-np.inf, "ref"),
    ])
    def test_F_classification_mutually_exclusive_and_complete(self, bad, where):
        """Every group joint belongs to EXACTLY one class — no overlap,
        no gap — across all invalidity placements."""
        z = _z()
        if where == "act_static":
            z["act_joint"][10, 5] = bad            # static-ref joint
        elif where == "act_moving":
            z["act_joint"][10, 2] = bad            # the moving joint
        else:
            z["ref_joint"][10, 5] = bad
        out = _run(z)
        n_invalid = len(set(out["invalid_reference_joints"])
                        | set(out["invalid_actual_joints"]))
        assert out["moving_arm_joints"] + out["static_arm_joints"] + n_invalid == 8
        # invalid joints never appear in the static bucket's values
        assert all(v is not None
                   for v in out["static_joint_residual_amplitude_rad"])
        # moving/static/invalid are over disjoint joints by construction:
        # valid = ref_ok & act_ok; moving ∪ static = valid; invalid = ~valid
        if where == "act_moving":
            assert out["moving_arm_joints"] == 0
            assert out["mean_amplitude_ratio"] is None
