"""Targeted tests for K1_DANCE_SKILL_FRAMEWORK_02_REMEDIATION_01.

Task A: oneshot full-timeline recording (frame 0 .. final, no dupes)
Task B: evaluation non-finite handling + strict JSON (CLI included)
Task C: final effective training config validation (no CLI bypass)
Task D: authoritative 22-DOF joint identity contract
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_argv_backup = sys.argv
sys.argv = ["deploy_mujoco_record_cyclic.py", "--output", "/tmp/_unused.mp4"]
try:
    sys.path.insert(0, str(REPO / "tools" / "booster_deploy" / "scripts"))
    from deploy_mujoco_record_cyclic import record_frame_index  # noqa: E402
finally:
    sys.argv = _argv_backup

import k1_skills.evaluation as ev  # noqa: E402
from k1_skills.errors import SkillError  # noqa: E402
from k1_skills.evaluation import evaluate  # noqa: E402
from k1_skills.learning import (effective_training_config,  # noqa: E402
                                train_command)
from k1_skills.motion import MotionResourceSet  # noqa: E402
from k1_skills.registry import SkillRegistry  # noqa: E402
from k1_skills.spec import validate_effective_training_config  # noqa: E402

GROUPS = {"arms": slice(2, 10), "legs": slice(10, 22)}


# ======================================================================
# Task A — oneshot full timeline
# ======================================================================
class TestOneshotFullTimeline:
    def test_frame_zero_is_a_real_sample(self):
        """The regenerated mj2 record must start with a GENUINE frame-0
        sample: robot initialized exactly at reference frame 0 (zero joint
        difference), t=0, before any policy step."""
        cands = sorted((REPO / "k1_runs" / "outputs" / "mj_dance_002")
                       .glob("*baseline*.metrics.npz"))
        assert cands, "no mj_dance_002 baseline run recorded"
        z = np.load(cands[-1])
        f, af, t = z["frame"], z["abs_frame"], z["t"]
        period = int(np.asarray(z["period"]).ravel()[0])
        assert len(f) == period == 1260
        assert f[0] == 0 and af[0] == 0 and t[0] == 0.0
        init_err = float(np.abs(z["act_joint"][0] - z["ref_joint"][0]).max())
        assert init_err == 0.0, "frame-0 row must be the real init state"
        # every reference frame sampled exactly once, no wrap, no dupes
        assert np.array_equal(f, np.arange(period))
        assert np.allclose(t, np.arange(period) * 0.02)
        assert f[-1] == period - 1

    def test_row_count_vs_timeline_semantics(self):
        """1 pre-step row + (period-1) post-step rows = period rows total;
        documented relationship between sampling count and timeline."""
        period = 1260
        pre_rows, post_rows = 1, period - 1
        assert pre_rows + post_rows == period
        # command coverage note: post-step rows carry the response to
        # commands 0..period-2; the take ends when the robot reaches the
        # final reference frame (no step beyond the take).
        assert record_frame_index(post_rows, period, oneshot=True) == period - 1

    def test_cyclic_unchanged_post_step_only(self):
        cands = sorted((REPO / "k1_runs" / "outputs" / "arm_wave")
                       .glob("*trained*.metrics.npz"))
        assert cands
        z = np.load(cands[-1])
        f, af, t = z["frame"], z["abs_frame"], z["t"]
        period = int(np.asarray(z["period"]).ravel()[0])
        assert len(f) == 3 * period and t[0] == 0.02 and f[0] == 1
        assert np.all(f == af % period)


# ======================================================================
# Task B — non-finite handling + strict JSON
# ======================================================================
def _base_z(cycles=2, frames=100):
    total = cycles * frames
    t = np.arange(total) * 0.02
    ref = np.zeros((total, 22))
    ref[:, 2] = np.sin(2 * np.pi * np.arange(total) / frames)
    act = ref + 0.05
    z = {
        "t": t, "abs_frame": np.arange(total), "frame": np.arange(total) % frames,
        "cycle": np.arange(total) // frames, "phase": np.zeros(total),
        "in_transition": np.zeros(total, bool),
        "ref_joint": ref.astype(np.float32), "act_joint": act.astype(np.float32),
        "ref_joint_vel": np.gradient(ref, axis=0).astype(np.float32),
        "act_joint_vel": np.gradient(act, axis=0).astype(np.float32),
        "ref_root_xyz": np.tile([0, 0, 0.55], (total, 1)),
        "act_root_xyz": np.tile([0, 0, 0.55], (total, 1)),
        "root_err_xyz": np.zeros((total, 3)),
        "ref_root_quat": np.tile([1.0, 0, 0, 0], (total, 1)),
        "act_root_quat": np.tile([1.0, 0, 0, 0], (total, 1)),
        "trunk_ori_err_deg": np.zeros(total),
        "base_roll_deg": np.zeros(total), "base_pitch_deg": np.zeros(total),
        "base_yaw_deg": np.zeros(total), "base_height": np.full(total, 0.55),
        "left_contact": np.ones(total, bool), "right_contact": np.ones(total, bool),
        "left_force_n": np.full(total, 100.0), "right_force_n": np.full(total, 100.0),
        "left_slide": np.zeros(total), "right_slide": np.zeros(total),
        "fallen": np.zeros(total, bool), "dof_target": act.astype(np.float32),
        "joint_torque": np.zeros((total, 22)),
        "fps": np.array([50.0]), "period": np.array([frames]),
        "n_source_frames": np.array([frames]), "cycles": np.array([cycles]),
        "stopped_early": np.array(False),
    }
    return z


def _all_metrics_strict(z):
    """Run every built-in family and enforce strict JSON serializability."""
    result = evaluate.__wrapped__(z) if hasattr(evaluate, "__wrapped__") else None
    # evaluate() needs a file; exercise metric functions directly plus the
    # guard-equivalent check on the assembled dict:
    out = {"motion_kind": "cyclic"}
    for name in ev.METRIC_FUNCS:
        out[name] = ev.METRIC_FUNCS[name](z, GROUPS)
    json.dumps(out, allow_nan=False)   # raises on any NaN/Inf leak
    return out


class TestNonFiniteEvaluation:
    def test_normal_finite_data(self):
        out = _all_metrics_strict(_base_z())
        assert out["joint_tracking_error"]["mae_arms_mrad"] == pytest.approx(50.0)
        assert out["joint_tracking_error"]["nonfinite_samples"] == 0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_single_cycle_poisoned(self, bad):
        z = _base_z()
        z["act_joint"][10, 3] = bad           # inside cycle 0 only
        out = _all_metrics_strict(z)
        jte = out["joint_tracking_error"]
        assert jte["per_cycle_mae_arms_mrad"][0] is None   # invalid cycle -> None
        assert jte["per_cycle_mae_arms_mrad"][1] == pytest.approx(50.0, abs=0.5)
        assert jte["nonfinite_samples"] == 1
        assert jte["mae_arms_mrad"] == pytest.approx(50.0, abs=0.5)  # finite-only
        # cycle_consistency pairs touching the poisoned cycle -> undefined
        cc = out["cycle_consistency"]
        assert cc["cross_cycle_error_correlation"][0] is None
        assert cc["undefined_pairs"] == 1

    def test_all_cycles_invalid(self):
        z = _base_z()
        z["act_joint"][:, 2:10] = np.nan         # the entire arms slice
        out = _all_metrics_strict(z)
        jte = out["joint_tracking_error"]
        assert jte["mae_arms_mrad"] is None       # no finite arm sample at all
        assert jte["per_cycle_mae_arms_mrad"] == [None, None]
        assert jte["mae_all_mrad"] is not None    # legs/head still finite
        assert jte["nonfinite_samples"] == 2 * 100 * 8

    def test_root_and_contact_nonfinite(self):
        z = _base_z()
        z["act_root_xyz"][5] = np.inf             # cycle-0 root data invalid
        z["root_err_xyz"][7, 1] = np.inf          # true max error = infinite
        z["left_slide"][9] = np.nan
        out = _all_metrics_strict(z)
        rd = out["root_drift"]
        assert rd["self_drift_per_cycle_end_mm"][0] is None
        assert rd["self_drift_per_cycle_end_mm"][1] is not None
        assert rd["ref_error_max_mm"] is None     # strict max: inf -> None
        fc = out["foot_contact"]
        assert fc["slide_speed_while_contact_max_m_s"] is None

    def test_constant_error_cycles(self):
        z = _base_z()
        # exactly-constant error (both arrays constant in float32; a
        # varying reference would introduce float32 rounding noise)
        z["ref_joint"] = np.zeros_like(z["ref_joint"])
        z["act_joint"] = np.full_like(z["ref_joint"], 0.1, dtype=np.float32)
        out = _all_metrics_strict(z)
        cc = out["cycle_consistency"]
        assert cc["cross_cycle_error_correlation"] == [None]
        assert cc["mean_correlation"] is None
        assert cc["per_cycle_arm_mae_mrad"] == [pytest.approx(100.0, abs=1.0)] * 2

    def test_insufficient_samples(self):
        z = _base_z(frames=1)                        # 1-frame cycles
        out = _all_metrics_strict(z)
        cc = out["cycle_consistency"]
        assert cc["cross_cycle_error_correlation"][0] is None

    def test_evaluate_guard_on_real_data_strict_json(self, tmp_path):
        cands = sorted((REPO / "k1_runs" / "outputs" / "mj_dance_002")
                       .glob("*baseline*.metrics.npz"))
        assert cands
        result = evaluate(cands[-1], motion_kind="oneshot")
        json.dumps(result, allow_nan=False)          # the real gate

    def test_cli_output_strict_json(self, capsys, tmp_path):
        """The CLI (not just the internal object) must emit strict JSON."""
        from k1_skills import cli
        cands = sorted((REPO / "k1_runs" / "outputs" / "mj_dance_002")
                       .glob("*baseline*.metrics.npz"))
        assert cands
        rc = cli.main(["evaluate", "--skill-id", "mj_dance_002", str(cands[-1])])
        assert rc == 0
        text = capsys.readouterr().out
        json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError(f"non-strict JSON constant {s} leaked: {s}")))


# ======================================================================
# Task C — final effective training config validation
# ======================================================================
class TestEffectiveTrainingConfig:
    @pytest.fixture(scope="class")
    def arm_wave(self):
        return SkillRegistry().get("arm_wave")

    @pytest.fixture(scope="class")
    def mj2(self):
        return SkillRegistry().get("mj_dance_002")

    @pytest.mark.parametrize("kwargs", [
        {"num_envs": 0}, {"num_envs": -4}, {"num_envs": 16.5},
        {"num_envs": True}, {"num_envs": "16"},
        {"max_iterations": 0}, {"max_iterations": -1}, {"max_iterations": 3.5},
        {"max_iterations": True}, {"max_iterations": "12"},
    ])
    def test_illegal_overrides_rejected_before_launch(self, arm_wave, kwargs):
        with pytest.raises(SkillError):
            train_command(arm_wave, **kwargs)      # raises, no subprocess

    def test_falsy_zero_not_silently_dropped(self, arm_wave):
        """num_envs=0 must be REJECTED, not fall back to the YAML value
        (the old `num_envs or yaml` coercion silently bypassed zero)."""
        with pytest.raises(SkillError, match="num_envs"):
            train_command(arm_wave, num_envs=0)

    @pytest.mark.parametrize("patch", [
        {"num_envs": 0}, {"num_envs": float("nan")}, {"num_envs": float("inf")},
        {"max_iterations": 0},
        {"seed": -1}, {"seed": 1.5}, {"seed": True},
    ])
    def test_illegal_effective_fields(self, arm_wave, patch):
        eff = effective_training_config(arm_wave)
        eff.update(patch)
        with pytest.raises(SkillError):
            validate_effective_training_config(eff)

    @pytest.mark.parametrize("ctrl_patch", [
        {"physics_dt": 0.0}, {"physics_dt": -0.005}, {"physics_dt": float("nan")},
        {"decimation": 0}, {"decimation": -1}, {"decimation": 2.5},
        {"decimation": True},
        {"policy_frequency_hz": 0}, {"policy_frequency_hz": -50},
        {"policy_frequency_hz": float("inf")},
    ])
    def test_illegal_control_chain(self, arm_wave, ctrl_patch):
        eff = effective_training_config(arm_wave)
        eff["control"] = dict(eff["control"], **ctrl_patch)
        with pytest.raises(SkillError):
            validate_effective_training_config(eff)

    def test_illegal_gains_in_effective_config(self, arm_wave):
        eff = effective_training_config(arm_wave)
        eff["control"] = dict(eff["control"])
        eff["control"]["joint_stiffness"] = list(eff["control"]["joint_stiffness"])
        eff["control"]["joint_stiffness"][4] = "3.9"      # numeric string
        with pytest.raises(SkillError):
            validate_effective_training_config(eff)
        eff["control"]["joint_stiffness"][4] = float("nan")
        with pytest.raises(SkillError):
            validate_effective_training_config(eff)

    def test_stepping_invariant_on_effective_config(self, arm_wave):
        eff = effective_training_config(arm_wave)
        eff["control"] = dict(eff["control"], decimation=5)   # breaks 4x5ms=20ms
        with pytest.raises(SkillError, match="physics_dt\\*decimation"):
            validate_effective_training_config(eff)

    def test_legal_overrides_work_and_win(self, arm_wave, mj2):
        cmd = train_command(arm_wave, num_envs=1024, max_iterations=12)
        i, j = cmd.index("--num_envs"), cmd.index("--max_iterations")
        assert cmd[i + 1] == "1024" and cmd[j + 1] == "12"   # override wins
        cmd2 = train_command(mj2, max_iterations=12)
        assert cmd2[cmd2.index("--max_iterations") + 1] == "12"
        assert cmd2[cmd2.index("--num_envs") + 1] == str(
            mj2.training_profile["num_envs"])               # yaml default kept

    def test_both_skills_yaml_configs_load(self, arm_wave, mj2):
        for d in (arm_wave, mj2):
            validate_effective_training_config(
                effective_training_config(d))               # must not raise


# ======================================================================
# Task D — authoritative joint identity
# ======================================================================
NAMES = [f"j{i:02d}" for i in range(22)]


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
    m.canonical_npz_names = list(NAMES)      # synthetic authoritative order
    return m


class TestJointIdentity:
    def test_canonical_passes(self, tmp_path):
        a = tmp_path / "a.npz"
        _write_npz(a, joint_names=NAMES)
        assert _mrs({"train_npz": str(a)}).verify() == []

    def test_missing_joint_rejected(self, tmp_path):
        a = tmp_path / "a.npz"
        _write_npz(a, joint_names=NAMES[:21])
        problems = _mrs({"train_npz": str(a)}).verify()
        assert any("missing" in p for p in problems)

    def test_extra_joint_rejected(self, tmp_path):
        a = tmp_path / "a.npz"
        _write_npz(a, joint_names=NAMES + ["j_extra"])
        problems = _mrs({"train_npz": str(a)}).verify()
        assert any("unknown" in p for p in problems)

    def test_duplicate_joint_rejected(self, tmp_path):
        a = tmp_path / "a.npz"
        names = list(NAMES); names[5] = names[4]
        _write_npz(a, joint_names=names)
        problems = _mrs({"train_npz": str(a)}).verify()
        assert any("duplicated" in p and "missing" in p for p in problems)

    def test_swapped_order_rejected(self, tmp_path):
        a = tmp_path / "a.npz"
        names = list(NAMES); names[2], names[9] = names[9], names[2]
        _write_npz(a, joint_names=names)
        problems = _mrs({"train_npz": str(a)}).verify()
        assert any("different ORDER" in p for p in problems)

    def test_unknown_joint_rejected(self, tmp_path):
        a = tmp_path / "a.npz"
        names = list(NAMES); names[7] = "not_a_k1_joint"
        _write_npz(a, joint_names=names)
        problems = _mrs({"train_npz": str(a)}).verify()
        assert any("unknown" in p for p in problems)

    def test_unnamed_without_declaration_rejected(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        _write_npz(a, joint_names=NAMES)
        _write_npz(b)
        problems = _mrs({"train_npz": str(a), "deploy_npz": str(b)}).verify()
        assert any("no joint_names and no joint_order" in p for p in problems)

    def test_declaration_to_unverified_resource_rejected(self, tmp_path):
        """train_npz itself unnamed => the declaration chain has no
        grounded identity => reject (data similarity can't ground it)."""
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(1).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_values=shared)        # unnamed 'train'
        _write_npz(b, joint_values=shared)        # unnamed deploy
        import hashlib
        b_sha = hashlib.sha256(b.read_bytes()).hexdigest()
        problems = _mrs({
            "train_npz": str(a),
            "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                           "content_sha256": b_sha}}).verify()
        assert any("no joint_names and no joint_order" in p for p in problems)
        assert any("not a canonically-verified named resource" in p
                   for p in problems)

    def test_data_similar_swap_is_not_identity(self, tmp_path):
        """Two joints with IDENTICAL trajectories, swapped in the unnamed
        resource: data cross-check passes trivially, but identity is
        grounded by the declaration chain (train_npz canonical), so the
        resource is accepted ONLY through that chain — and a resource
        WITHOUT such a chain is rejected regardless of data similarity."""
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        vals = np.random.default_rng(2).uniform(-1, 1, (5, 22))
        vals[:, 3] = vals[:, 4]                    # two identical columns
        _write_npz(a, joint_names=NAMES, joint_values=vals)
        swapped = vals[:, [0, 1, 2, 4, 3] + list(range(5, 22))]
        _write_npz(b, joint_values=swapped)        # cols 3/4 swapped (identical)
        # with grounded declaration: accepted (swap is numerically
        # undetectable AND inconsequential; identity rides the chain)
        import hashlib
        b_sha = hashlib.sha256(b.read_bytes()).hexdigest()
        mrs = _mrs({"train_npz": str(a),
                    "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                                   "content_sha256": b_sha}})
        assert mrs.verify() == []
        # without any declaration: rejected even though data matches
        _write_npz(b, joint_values=vals)
        problems = _mrs({"train_npz": str(a), "deploy_npz": str(b)}).verify()
        assert any("no joint_names and no joint_order" in p for p in problems)

    def test_real_canonical_source_is_deploy_cfg(self):
        from k1_skills.robot import canonical_joint_orders
        orders = canonical_joint_orders()
        assert len(orders["npz"]) == 22 == len(orders["csv"])
        assert orders["npz"] != orders["csv"]                   # two orders
        assert sorted(orders["npz"]) == sorted(orders["csv"])  # same joints
        # the real skills' named npz match the canonical npz order
        for skill_id in ("arm_wave", "mj_dance_002", "top_rock"):
            definition = SkillRegistry().get(skill_id)
            mrs = MotionResourceSet.from_spec(definition)
            assert mrs.canonical_npz_names == orders["npz"]
            assert mrs.verify() == [], (skill_id, mrs.verify())

    def test_canonical_unavailable_is_rejection_not_silence(self, tmp_path):
        a = tmp_path / "a.npz"
        _write_npz(a, joint_names=NAMES)
        mrs = MotionResourceSet.from_spec(_Def({"train_npz": str(a)}))
        mrs.canonical_npz_names = None            # authority unreachable
        problems = mrs.verify()
        assert any("canonical K1 joint order unavailable" in p
                   for p in problems)
