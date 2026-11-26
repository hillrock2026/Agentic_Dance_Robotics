"""Targeted tests for K1_DANCE_SKILL_FRAMEWORK_02_FINAL_REMEDIATION_01.

Task B: velocity frame-accounting / amplitude tri-state isolation
Task D: canonical uniqueness + content_sha256 identity binding
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import k1_skills.evaluation as ev  # noqa: E402
from k1_skills.errors import SkillError  # noqa: E402
from k1_skills.motion import MotionResourceSet  # noqa: E402

GROUPS = {"arms": slice(2, 10), "legs": slice(10, 22)}
NAMES = [f"j{i:02d}" for i in range(22)]


def _z(total=200, frames=100):
    t = np.arange(total) * 0.02
    ref = np.zeros((total, 22))
    phase = np.arange(total) % frames
    ref[:, 2] = np.sin(2 * np.pi * phase / frames)          # moving joint
    act = ref + 0.05                                         # 50 mrad err
    return {
        "cycle": np.arange(total) // frames, "phase": np.zeros(total),
        "abs_frame": np.arange(total), "frame": phase,
        "in_transition": np.zeros(total, bool), "t": t,
        "ref_joint": ref.astype(np.float32), "act_joint": act.astype(np.float32),
        "ref_joint_vel": np.gradient(ref, axis=0).astype(np.float32),
        "act_joint_vel": np.gradient(act, axis=0).astype(np.float32),
    }


# ======================================================================
# B — arm_motion_velocity frame accounting
# ======================================================================
class TestVelocityNonFinite:
    def _run(self, z):
        out = ev.arm_motion_velocity(z, GROUPS)
        json.dumps(out, allow_nan=False)
        # hard accounting invariant
        assert (out["binned_frames"] + out["small_bin_frames_omitted"]
                + out["nonfinite_speed_frames"]) == out["total_frames"]
        return out

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_single_invalid_speed_frame_is_counted_not_dropped(self, bad):
        z = _z()
        z["ref_joint_vel"][37, 2] = bad       # speed of frame 37 invalid
        out = self._run(z)
        assert out["total_frames"] == 200
        assert out["nonfinite_speed_frames"] == 1
        assert out["binned_frames"] == 199    # exactly one frame excluded
        # peaks are strict: true peak is non-finite -> None
        assert out["ref_peak_speed_rad_s"] is None

    def test_partial_invalid(self):
        z = _z()
        z["ref_joint_vel"][10:20, 3] = np.nan
        out = self._run(z)
        assert out["nonfinite_speed_frames"] == 10
        assert out["binned_frames"] == 190

    def test_all_invalid(self):
        z = _z()
        z["ref_joint_vel"][:, :] = np.inf
        out = self._run(z)
        assert out["nonfinite_speed_frames"] == 200
        assert out["binned_frames"] == 0
        assert out["mae_by_ref_speed"] == {}

    def test_bin_err_invalid_returns_none_with_count(self):
        z = _z()
        # make fast-moving frames' ERR invalid: inject NaN into act_joint
        # where ref speed is highest (the sine peaks)
        fast = np.abs(z["ref_joint_vel"][:, 2:10]).max(axis=1) > 0.05
        z["act_joint"][fast, 2] = np.nan
        out = self._run(z)
        high = out["mae_by_ref_speed"].get("0-2rad_s")
        assert high is not None and high["mae_mrad"] is None
        assert high["nonfinite_err_samples"] > 0
        assert out["nonfinite_err_samples"] == int(fast.sum())

    def test_normal_finite_unchanged(self):
        out = self._run(_z())
        assert out["nonfinite_speed_frames"] == 0
        assert out["nonfinite_err_samples"] == 0
        for b in out["mae_by_ref_speed"].values():
            assert b["mae_mrad"] is not None
            assert b["nonfinite_err_samples"] == 0
        # historical semantic: the slow bin dominates a sine at 1 Hz
        assert out["mae_by_ref_speed"]["0-2rad_s"]["mae_mrad"] == pytest.approx(
            50.0, abs=2.0)


# ======================================================================
# B — arm_motion_amplitude tri-state
# ======================================================================
class TestAmplitudeNonFinite:
    def _run(self, z):
        out = ev.arm_motion_amplitude(z, GROUPS)
        json.dumps(out, allow_nan=False)
        return out

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_invalid_ref_joint_not_classified_static(self, bad):
        z = _z()
        z["ref_joint"][:, 4] = bad            # arm joint index 4 invalid
        out = self._run(z)
        assert 2 in out["invalid_reference_joints"]          # group-local idx
        assert out["moving_arm_joints"] == 1                 # joint 0 still moving
        # 8 arms = 1 moving + 1 invalid + 6 valid static
        assert out["static_arm_joints"] == 6
        assert len(out["static_joint_residual_amplitude_rad"]) == 6
        assert all(v is not None
                   for v in out["static_joint_residual_amplitude_rad"])
        assert out["reference_amplitude_rad"][2] is None     # invalid -> None

    def test_all_invalid_ref(self):
        z = _z()
        z["ref_joint"][:, 2:10] = np.nan
        out = self._run(z)
        assert out["moving_arm_joints"] == 0
        assert out["static_arm_joints"] == 0                 # none valid-static
        assert out["invalid_reference_joints"] == list(range(8))
        assert out["static_joint_residual_amplitude_rad"] == []
        assert out["mean_amplitude_ratio"] is None

    def test_invalid_act_joint_ratio_none(self):
        z = _z()
        z["act_joint"][:, 2] = np.inf         # the MOVING joint's act invalid
        out = self._run(z)
        assert 0 in out["invalid_actual_joints"]
        assert out["amplitude_ratio_per_joint"][0] is None
        # AMPLITUDE_CLASSIFICATION_FIX_01: a joint with a non-finite ACTUAL
        # series is invalid — no longer counted as moving (reference-only
        # classification was the superseded behaviour)
        assert out["moving_arm_joints"] == 0
        assert out["static_arm_joints"] == 7
        assert out["mean_amplitude_ratio"] is None          # no finite ratio left

    def test_normal_finite_unchanged(self):
        out = self._run(_z())
        assert out["moving_arm_joints"] == 1
        assert out["static_arm_joints"] == 7
        assert out["invalid_reference_joints"] == []
        assert out["invalid_actual_joints"] == []
        assert out["mean_amplitude_ratio"] == pytest.approx(1.0, rel=1e-3)


# ======================================================================
# D — canonical uniqueness
# ======================================================================
class TestCanonicalUniqueness:
    def test_real_canonical_is_22_unique(self):
        from k1_skills.robot import canonical_joint_orders
        o = canonical_joint_orders()
        assert len(o["npz"]) == 22 == len(set(o["npz"]))
        assert len(o["csv"]) == 22 == len(set(o["csv"]))

    def _patched(self, monkeypatch, sim=None, csv=None):
        import booster_deploy.robots.k1 as k1mod
        from k1_skills.robot import canonical_joint_orders
        old_sim = list(k1mod.K1_CFG.sim_joint_names)
        old_csv = list(k1mod.K1_CFG.joint_names)
        if sim is not None:
            monkeypatch.setattr(k1mod.K1_CFG, "sim_joint_names", sim)
        if csv is not None:
            monkeypatch.setattr(k1mod.K1_CFG, "joint_names", csv)
        try:
            return canonical_joint_orders()
        finally:
            k1mod.K1_CFG.sim_joint_names = old_sim
            k1mod.K1_CFG.joint_names = old_csv

    def test_duplicate_in_canonical_rejected(self, monkeypatch):
        names = list(NAMES)
        names[5] = names[4]                    # duplicate compensating a unique
        with pytest.raises(SkillError, match="duplicate"):
            self._patched(monkeypatch, sim=names, csv=list(NAMES))

    def test_wrong_count_rejected(self, monkeypatch):
        with pytest.raises(SkillError, match="22"):
            self._patched(monkeypatch, sim=NAMES[:21], csv=list(NAMES))

    def test_set_mismatch_rejected(self, monkeypatch):
        csv = list(NAMES)
        csv[0] = "other_joint"
        with pytest.raises(SkillError, match="same 22-joint set"):
            self._patched(monkeypatch, csv=csv)


# ======================================================================
# D — content binding for unnamed resources
# ======================================================================
def _write_npz(path, *, joint_names=None, joint_values=None):
    rng = np.random.default_rng(0)
    jp = rng.uniform(-1, 1, (5, 22)).astype(np.float32)
    if joint_values is not None:
        jp = np.asarray(joint_values, np.float32)
    data = {
        "fps": np.array([50.0]), "joint_pos": jp, "joint_vel": np.zeros_like(jp),
        "body_pos_w": np.zeros((5, 23, 3), np.float32),
        "body_quat_w": np.tile([1.0, 0, 0, 0], (5, 23, 1)).astype(np.float32),
        "body_lin_vel_w": np.zeros((5, 23, 3), np.float32),
        "body_ang_vel_w": np.zeros((5, 23, 3), np.float32),
    }
    if joint_names is not None:
        data["joint_names"] = np.array(joint_names)
    np.savez(path, **data)


class _Def:
    def __init__(self, resources):
        self.skill_id = "t"
        self.root_mode = "free"
        self.motion = {"kind": "oneshot", "fps": 50, "source_frames": 5,
                       "transition_frames": 0, "resources": resources}

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else Path("/abs") / p


def _mrs(resources):
    m = MotionResourceSet.from_spec(_Def(resources))
    m.canonical_npz_names = list(NAMES)
    return m


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class TestContentBinding:
    def test_declaration_without_binding_rejected(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(1).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_values=shared)
        problems = _mrs({
            "train_npz": str(a),
            "deploy_npz": {"path": str(b), "joint_order": "train_npz"}}).verify()
        assert any("lacks content_sha256 binding" in p for p in problems)

    def test_content_modification_voids_declaration(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(2).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_values=shared)
        declared = _sha(b)
        mrs = _mrs({"train_npz": str(a),
                    "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                   "content_sha256": declared}})
        assert mrs.verify() == []
        # ANY content modification (even one float) voids the claim
        z = dict(np.load(b))
        z["joint_pos"] = z["joint_pos"] + np.float32(1e-6)
        np.savez(b, **z)
        problems = _mrs({"train_npz": str(a),
                         "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                        "content_sha256": declared}}).verify()
        assert any("no longer matches the declared content_sha256" in p
                   for p in problems)

    def test_identical_column_swap_rejected_by_binding(self, tmp_path):
        """FINAL_REMEDIATION 5.4: two joints with IDENTICAL trajectories,
        legal declaration + content binding established, then the two
        columns are swapped and the ORIGINAL declaration is kept. The
        bytes change -> digest mismatch -> REJECT (identity cannot be
        inferred from identical values; only a fresh explicit
        re-declaration can re-ground it)."""
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        vals = np.random.default_rng(3).uniform(-1, 1, (5, 22))
        vals[:, 3] = vals[:, 4]                       # two identical columns
        _write_npz(a, joint_names=NAMES, joint_values=vals)
        _write_npz(b, joint_values=vals)
        declared = _sha(b)
        mrs = _mrs({"train_npz": str(a),
                    "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                   "content_sha256": declared}})
        assert mrs.verify() == []                     # legal original state
        # swap the two identical columns; keep the ORIGINAL declaration
        swapped = vals[:, [0, 1, 2, 4, 3] + list(range(5, 22))]
        _write_npz(b, joint_values=swapped)
        assert _sha(b) != declared                    # bytes did change
        problems = _mrs({"train_npz": str(a),
                         "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                        "content_sha256": declared}}).verify()
        assert any("no longer matches the declared content_sha256" in p
                   for p in problems)
        # an explicit re-declaration against the swapped bytes re-grounds
        # the claim (human decision, not numeric inference)
        redeclared = _mrs({"train_npz": str(a),
                           "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                          "content_sha256": _sha(b)}})
        assert redeclared.verify() == []

    def test_binding_format_validated_in_schema(self, tmp_path):
        import yaml
        from k1_skills.spec import SkillDefinition
        doc = yaml.safe_load((REPO / "skills" / "mj_dance_002" / "skill.yaml")
                             .read_text(encoding="utf-8"))
        for bad in ("deadbeef", "z" * 64, 12345):
            doc["motion"]["resources"]["deploy_npz"]["content_sha256"] = bad
            p = tmp_path / "skill.yaml"
            p.write_text(yaml.safe_dump(doc), encoding="utf-8")
            with pytest.raises(SkillError, match="content_sha256"):
                SkillDefinition.from_yaml(p)

    def test_real_mj2_binding_verifies(self):
        from k1_skills.registry import SkillRegistry
        definition = SkillRegistry().get("mj_dance_002")
        mrs = MotionResourceSet.from_spec(definition)
        assert mrs.verify() == [], mrs.verify()
