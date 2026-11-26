"""Tests for the k1_skills framework (schema v2, synthetic data + real repo).

Run:  python -m pytest tests_k1_skills -q
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

import k1_skills.evaluation as ev
from k1_skills.agent import call_tool
from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.evaluation import evaluate
from k1_skills.spec import SkillDefinition

GROUPS = {"arms": slice(2, 10), "legs": slice(10, 22)}


# ----------------------------------------------------------------------
# evaluation math on synthetic metrics
# ----------------------------------------------------------------------
def _synthetic_metrics(frames: int = 246, cycles: int = 3) -> dict:
    per = frames
    total = per * cycles
    joint = np.zeros((total, 22))
    joint[:, 2] = np.sin(2 * np.pi * np.arange(total) / per)
    ref_joint, act_joint = joint, joint.copy()
    return {
        "t": np.arange(total) * 0.02,
        "abs_frame": np.arange(total), "frame": np.arange(total) % per,
        "cycle": np.arange(total) // per,
        "phase": (np.arange(total) % per) / per,
        "in_transition": np.zeros(total, dtype=bool),
        "ref_joint": ref_joint.astype(np.float32),
        "act_joint": act_joint.astype(np.float32),
        "ref_joint_vel": np.gradient(ref_joint, axis=0).astype(np.float32),
        "act_joint_vel": np.gradient(act_joint, axis=0).astype(np.float32),
        "ref_root_xyz": np.tile([0.0, 0.0, 0.555], (total, 1)),
        "act_root_xyz": np.tile([0.0, 0.0, 0.555], (total, 1)),
        "root_err_xyz": np.zeros((total, 3)),
        "ref_root_quat": np.tile([1.0, 0, 0, 0], (total, 1)),
        "act_root_quat": np.tile([1.0, 0, 0, 0], (total, 1)),
        "trunk_ori_err_deg": np.zeros(total),
        "base_roll_deg": np.zeros(total), "base_pitch_deg": np.zeros(total),
        "base_yaw_deg": np.zeros(total), "base_height": np.full(total, 0.555),
        "left_contact": np.ones(total, dtype=bool),
        "right_contact": np.ones(total, dtype=bool),
        "left_force_n": np.full(total, 200.0),
        "right_force_n": np.full(total, 200.0),
        "left_slide": np.zeros(total), "right_slide": np.zeros(total),
        "fallen": np.zeros(total, dtype=bool),
        "dof_target": joint.astype(np.float32),
        "joint_torque": np.zeros((total, 22)),
        "fps": np.array([50.0]), "period": np.array([per]),
        "n_source_frames": np.array([frames]), "cycles": np.array([cycles]),
        "stopped_early": np.array(False),
    }


def test_perfect_tracking_scores():
    z = _synthetic_metrics()
    r = {
        "joint_tracking_error": ev.joint_tracking_error(z, GROUPS),
        "fall_rate": ev.fall_rate(z, GROUPS),
        "foot_contact": ev.foot_contact(z, GROUPS),
        "root_drift": ev.root_drift(z, GROUPS),
        "arm_motion_amplitude": ev.arm_motion_amplitude(z, GROUPS),
    }
    assert r["joint_tracking_error"]["mae_all_mrad"] == pytest.approx(0.0)
    assert r["fall_rate"]["fall_rate"] == 0.0
    assert r["foot_contact"]["double_support_ratio"] == 1.0
    assert r["root_drift"]["ref_error_mean_mm"] == pytest.approx(0.0)
    assert r["arm_motion_amplitude"]["mean_amplitude_ratio"] == pytest.approx(1.0)


def test_degraded_tracking_detected():
    z = _synthetic_metrics()
    z["act_joint"][:, 2] += 0.1
    z["fallen"][100:110] = True
    jt = ev.joint_tracking_error(z, GROUPS)
    fr = ev.fall_rate(z, GROUPS)
    assert jt["mae_arms_mrad"] == pytest.approx(12.5, abs=0.5)
    assert fr["fallen_steps"] == 10


def test_custom_joint_groups_change_scope():
    """Evaluation groups are configuration, not hardcoded slices."""
    z = _synthetic_metrics()
    z["act_joint"][:, 2] += 0.1           # one "arm" joint biased
    only_that = {"arms": slice(2, 3), "legs": slice(10, 22)}
    jt = ev.joint_tracking_error(z, only_that)
    assert jt["mae_arms_mrad"] == pytest.approx(100.0, abs=0.5)


def test_applicability_oneshot_drops_cycle_consistency():
    z = _synthetic_metrics()
    from pathlib import Path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        np.savez(fh.name, **z)
        path = fh.name
    result = evaluate(path, motion_kind="oneshot")
    assert "cycle_consistency" not in result
    assert any("cycle_consistency" in d for d in result.get("dropped_metrics", []))
    result_c = evaluate(path, motion_kind="cyclic")
    assert "cycle_consistency" in result_c


def test_custom_metric_registration():
    called = {}

    def head_error(z, groups):
        called["yes"] = True
        return {"mae_head_mrad": float(
            np.abs(z["act_joint"] - z["ref_joint"])[:, 0:2].mean() * 1e3)}

    ev.register_metric("head_error_test", head_error)
    z = _synthetic_metrics()
    out = ev.joint_tracking_error  # noqa: F841  (ensure module import path)
    assert ev.METRIC_FUNCS["head_error_test"] is head_error
    ev.METRIC_FUNCS.pop("head_error_test")   # keep global registry clean


def test_compare_flags_improvement_direction():
    def res(arm, root):
        return {"joint_tracking_error": {"mae_arms_mrad": arm, "mae_all_mrad": arm,
                                         "mae_legs_mrad": arm * 0.5},
                "arm_motion_amplitude": {"mean_amplitude_ratio": 0.9},
                "arm_motion_velocity": {"high_speed_mae_mrad_gt10": 200.0},
                "root_drift": {"ref_error_mean_mm": root,
                               "self_drift_per_cycle_end_mm": [10, 10, 10],
                               "trunk_ori_err_mean_deg": 5.0},
                "fall_rate": {"fall_rate": 0.0},
                "foot_contact": {"double_support_ratio": 0.99}}
    cmp = ev.compare(res(100.0, 20.0), res(80.0, 12.0), label_a="old", label_b="new")
    by = {row["metric"]: row for row in cmp["rows"]}
    assert by["arm MAE"]["improved"] is True
    assert by["double support"]["improved"] is False


# ----------------------------------------------------------------------
# schema v2: motion kinds, profile inheritance, v1 upgrade
# ----------------------------------------------------------------------
_MIN_V2 = {
    "schema_version": 2, "skill_id": "t", "version": "1", "description": "d",
    "root_mode": "stationary",
    "motion": {"kind": "cyclic", "fps": 50, "source_frames": 246,
               "transition_frames": 25,
               "resources": {"train_npz": "x.npz", "deploy_npz_cyclic": "y.npz"}},
    "control_profile": {"extends": "k1_default", "physics_dt": 0.005,
                        "decimation": 4, "policy_frequency_hz": 50,
                        "joint_stiffness": [1.0] * 22, "joint_damping": [1.0] * 22,
                        "effort_limit": [1.0] * 22},
    "training_profile": {"framework": "x", "task_id": "T-v0",
                         "play_task_id": "T-v0-Play", "num_envs": 16,
                         "max_iterations": 10, "seed": 1},
    "runtime_profile": {"policies": {"baseline": {"deploy_task": "b"}}},
    "evaluation_profile": {"metrics": ["fall_rate"]},
}


def _write(tmp_path, data):
    p = tmp_path / "skill.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_v2_minimal_loads_and_inherits_profile(tmp_path):
    d = SkillDefinition.from_yaml(_write(tmp_path, _MIN_V2))
    assert d.motion_kind == "cyclic"
    ctrl = d.control_profile
    # inherited robot facts from skills/_profiles/k1_default.yaml
    assert ctrl["model"] == "booster_k1"
    assert ctrl["dof"] == 22
    assert ctrl["_extends"] == "k1_default"
    # skill-level values win
    assert ctrl["physics_dt"] == 0.005
    assert d.policy_roles() == ["baseline"]


def test_v2_oneshot_rejects_cyclic_resources_and_vv(tmp_path):
    bad = yaml.safe_load(yaml.safe_dump(_MIN_V2))
    bad["motion"]["kind"] = "oneshot"
    del bad["motion"]["resources"]["deploy_npz_cyclic"]
    bad["motion"]["resources"]["deploy_npz"] = "z.npz"
    bad["motion"].pop("transition_frames")
    d = SkillDefinition.from_yaml(_write(tmp_path, bad))
    assert d.motion_kind == "oneshot"

    wrong = yaml.safe_load(yaml.safe_dump(_MIN_V2))     # cyclic without cyclic npz
    del wrong["motion"]["resources"]["deploy_npz_cyclic"]
    with pytest.raises(SkillError):
        SkillDefinition.from_yaml(_write(tmp_path, wrong))


def test_v2_stepping_invariant_enforced(tmp_path):
    bad = yaml.safe_load(yaml.safe_dump(_MIN_V2))
    bad["control_profile"]["decimation"] = 5            # 5*5ms != 20ms
    with pytest.raises(SkillError) as ei:
        SkillDefinition.from_yaml(_write(tmp_path, bad))
    assert "physics_dt*decimation" in ei.value.message


def test_v1_definition_upgrades(tmp_path):
    v1 = {
        "skill_id": "legacy", "version": "1", "description": "d",
        "root_mode": "stationary",
        "reference_motion": {
            "fps": 50, "source_frames": 246, "transition_frames": 25,
            "train_npz": "a.npz", "deploy_npz_cyclic": "b.npz"},
        "control_profile": dict(_MIN_V2["control_profile"], extends=None),
        "observation_profile": {}, "reward_profile": {},
        "training_profile": dict(_MIN_V2["training_profile"]),
        "runtime_profile": {"baseline_task": "bt", "trained_task": "tt",
                            "trained_policy_file": "p.jit"},
        "evaluation_profile": {"metrics": ["fall_rate"]},
    }
    d = SkillDefinition.from_yaml(_write(tmp_path, v1))
    assert d.is_cyclic
    roles = d.policy_roles()
    assert set(roles) == {"baseline", "trained"}
    assert d.policy_entry("trained")["policy_file"] == "p.jit"
    assert d.reference_motion["transition_frames"] == 25


def test_policy_role_access_errors(tmp_path):
    d = SkillDefinition.from_yaml(_write(tmp_path, _MIN_V2))
    with pytest.raises(SkillError):
        d.policy_entry("nonexistent")


# ----------------------------------------------------------------------
# agent tool envelope
# ----------------------------------------------------------------------
def test_tool_envelope_error_semantics():
    env = call_tool("skill.validate", {"skill_id": "__nope__"})
    assert env["ok"] is False
    assert env["error"]["code"] == "skill_not_found"
    assert isinstance(env["error"]["retryable"], bool)


def test_unknown_tool_and_bad_json():
    assert call_tool("__nope__", {})["error"]["code"] == "unknown_tool"
    assert call_tool("skill.validate", "{not-json")["error"]["code"] == "invalid_arguments"


def test_rl_status_requires_task_id_or_list_recent():
    env = call_tool("rl.status", {})
    assert env["ok"] is False
    assert env["error"]["code"] == "invalid_arguments"


# ----------------------------------------------------------------------
# real-repo integration
# ----------------------------------------------------------------------
def test_arm_wave_skill_validates_against_real_repo():
    from k1_skills.registry import SkillRegistry
    report = SkillRegistry().validate("arm_wave")
    assert report["valid"], report["problems"]


def test_mj_dance_002_validates_with_lifecycle_warning():
    from k1_skills.registry import SkillRegistry
    report = SkillRegistry().validate("mj_dance_002")
    assert report["valid"], report["problems"]
    assert report["warnings"], "expected not-yet-provisioned warning for trained role"


def test_motion_resource_set_sync_tolerates_one_frame():
    from k1_skills.motion import MotionResourceSet
    from k1_skills.registry import SkillRegistry
    definition = SkillRegistry().get("mj_dance_002")
    mrs = MotionResourceSet.from_spec(definition)
    problems = mrs.verify()
    assert problems == [], problems
    assert mrs.loaded["train_npz"].n_frames == 1261
    assert mrs.loaded["deploy_npz"].n_frames == 1260


def test_evaluate_real_baseline_metrics():
    from k1_skills.paths import OUTPUTS_DIR
    cands = sorted((OUTPUTS_DIR / "arm_wave").glob("*baseline*.metrics.npz"))
    if not cands:
        pytest.skip("no baseline metrics recorded yet")
    result = evaluate(cands[-1], motion_kind="cyclic")
    for family in ("joint_tracking_error", "arm_motion_amplitude",
                   "arm_motion_velocity", "root_drift", "fall_rate",
                   "foot_contact", "cycle_consistency"):
        assert family in result
