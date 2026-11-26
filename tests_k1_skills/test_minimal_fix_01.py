"""Targeted tests for K1_DANCE_SKILL_FRAMEWORK_02_MINIMAL_FIX_01.

Task A: oneshot final-frame index (recorder)
Task B: cycle_consistency math (evaluation)
Task C: schema numeric constraints + 22-DOF joint-order contract
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# the recorder parses CLI args at import time (script style); stub argv so
# the import succeeds and only the pure helper is used here
_argv_backup = sys.argv
sys.argv = ["deploy_mujoco_record_cyclic.py", "--output", "/tmp/_unused.mp4"]
try:
    sys.path.insert(0, str(REPO / "tools" / "booster_deploy" / "scripts"))
    from deploy_mujoco_record_cyclic import record_frame_index  # noqa: E402
finally:
    sys.argv = _argv_backup

import k1_skills.evaluation as ev  # noqa: E402
from k1_skills.errors import SkillError  # noqa: E402
from k1_skills.motion import MotionResourceSet  # noqa: E402
from k1_skills.spec import SkillDefinition  # noqa: E402

GROUPS = {"arms": slice(2, 10), "legs": slice(10, 22)}

# ======================================================================
# Task A — oneshot final-frame index
# ======================================================================
class TestRecordFrameIndex:
    def test_oneshot_indices(self):
        period = 1260
        assert record_frame_index(0, period, oneshot=True) == 0
        assert record_frame_index(1, period, oneshot=True) == 1
        assert record_frame_index(630, period, oneshot=True) == 630
        assert record_frame_index(1259, period, oneshot=True) == 1259
        # the historical bug: 1260 % 1260 == 0
        assert 1260 % period == 0
        assert record_frame_index(1260, period, oneshot=True) == 1259
        assert record_frame_index(period + 17, period, oneshot=True) == 1259

    def test_oneshot_1260_scene_full_sequence(self):
        period = 1260
        n = period
        f = [record_frame_index(k + 1, period, oneshot=True) for k in range(n)]
        assert f[0] == 1
        assert f[n - 2] == 1259
        assert f[n - 1] == 1259, "final row must stay on the last frame"
        assert all(a <= b for a, b in zip(f, f[1:])), "must be monotonic"

    def test_cyclic_wrap_preserved(self):
        period = 271
        assert record_frame_index(271, period, oneshot=False) == 0
        assert record_frame_index(272, period, oneshot=False) == 1
        assert record_frame_index(2 * 271 + 7, period, oneshot=False) == 7

    def test_real_oneshot_record_correspondence(self):
        """Rows / timestamps / abs_frame / frame of the REGENERATED mj2 run.

        Updated for the REMEDIATION_01 full-timeline contract: 1 pre-step
        row at (t=0, frame=0) + period-1 post-step rows, i.e. frames
        0..1259 exactly once (the MINIMAL_FIX_01 record this test
        originally pinned — frames 1..1259,1259 — was superseded)."""
        cands = sorted((REPO / "k1_runs" / "outputs" / "mj_dance_002")
                       .glob("*baseline*.metrics.npz"))
        if not cands:
            pytest.skip("no mj_dance_002 baseline run recorded")
        z = np.load(cands[-1])
        f, af, t = z["frame"], z["abs_frame"], z["t"]
        n, period = len(f), int(np.asarray(z["period"]).ravel()[0])
        assert n == period == 1260
        assert np.allclose(t, np.arange(n) * 0.02)             # timestamps
        assert np.all(af == np.arange(n))                        # abs = k
        assert np.all(f == np.minimum(af, period - 1))           # clamped idx
        assert f[0] == 0 and f[-1] == 1259                       # full span
        # the pre-fix record wrapped exactly here; keep it as evidence
        old = np.load(REPO / "k1_runs" / "outputs" / "mj_dance_002" /
                      "k1_mj2_baseline_c1_sim2sim-20261119-134959-899a79.metrics.npz")
        assert old["frame"][-1] == 0

    def test_real_cyclic_record_wraps(self):
        """Cyclic arm_wave recording still wraps (regression)."""
        cands = sorted((REPO / "k1_runs" / "outputs" / "arm_wave")
                       .glob("*trained*.metrics.npz"))
        if not cands:
            pytest.skip("no arm_wave run recorded")
        z = np.load(cands[-1])
        f, af = z["frame"], z["abs_frame"]
        period = int(np.asarray(z["period"]).ravel()[0])
        assert period == 271
        assert np.all(f == af % period)


# ======================================================================
# Task B — cycle_consistency math
# ======================================================================
def _mk_z(error_patterns, frames_per_cycle=120):
    """metrics-like dict whose arm-group error equals the given patterns."""
    cycles = len(error_patterns)
    total = cycles * frames_per_cycle
    ref = np.zeros((total, 22), dtype=np.float32)
    act = np.zeros((total, 22), dtype=np.float32)
    for c, pattern in enumerate(error_patterns):
        sl = slice(c * frames_per_cycle, (c + 1) * frames_per_cycle)
        act[sl, 2:10] = (ref[sl, 2:10] + pattern).astype(np.float32)
    return {
        "cycle": np.repeat(np.arange(cycles), frames_per_cycle),
        "in_transition": np.zeros(total, dtype=bool),
        "act_joint": act, "ref_joint": ref,
    }


def _pattern(seed=0, frames=120, joints=8):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.6, size=(frames, joints))


class TestCycleConsistency:
    def test_identical_patterns_correlation_plus_one(self):
        p = _pattern(seed=1)
        out = ev.cycle_consistency(_mk_z([p, p, p]), GROUPS)
        assert out["cross_cycle_error_correlation"] == pytest.approx([1.0, 1.0, 1.0])
        assert out["mean_correlation"] == pytest.approx(1.0)
        assert out["undefined_pairs"] == 0

    def test_anti_correlated_patterns_minus_one(self):
        p = _pattern(seed=2)
        anti = -p + 2.0                       # affine negative -> corr -1
        out = ev.cycle_consistency(_mk_z([p, anti]), GROUPS)
        assert out["cross_cycle_error_correlation"][0] == pytest.approx(-1.0)

    def test_uncorrelated_patterns_near_zero(self):
        a = _pattern(seed=3)
        b = _pattern(seed=4)                  # independent draw
        out = ev.cycle_consistency(_mk_z([a, b]), GROUPS)
        corr = out["cross_cycle_error_correlation"][0]
        assert abs(corr) < 0.3, corr

    def test_multi_joint_heterogeneous_magnitudes(self):
        """Flattened definition: a pattern with heterogeneous per-joint
        magnitudes, repeated identically across cycles, correlates at 1 —
        the 2-D (time x joint) structure is handled, not collapsed per
        timestep. (Scaling joints *differently between* cycles is a real
        inconsistency under this definition and scores < 1.)"""
        p = _pattern(seed=5) * np.linspace(0.2, 3.0, 8)[None, :]
        out = ev.cycle_consistency(_mk_z([p, p]), GROUPS)
        assert out["cross_cycle_error_correlation"][0] == pytest.approx(1.0, abs=1e-9)

    def test_constant_pattern_is_undefined_not_fake(self):
        """Zero-variance patterns have no Pearson correlation: reported as
        undefined (None), never a fabricated score. Per-cycle MAE carries
        the information for such cases."""
        p = _pattern(seed=6)
        const = np.full_like(p, 0.25)
        out = ev.cycle_consistency(_mk_z([p, const, const]), GROUPS)
        assert out["cross_cycle_error_correlation"] == [None, None, None]
        assert out["mean_correlation"] is None
        assert out["undefined_pairs"] == 3

    def test_nonfinite_pattern_is_undefined(self):
        p = _pattern(seed=7)
        bad = p.copy()
        bad[3, 2] = np.nan
        out = ev.cycle_consistency(_mk_z([p, bad]), GROUPS)
        assert out["cross_cycle_error_correlation"][0] is None
        assert out["undefined_pairs"] == 1

    def test_arm_wave_real_data_recomputed(self):
        cands = sorted((REPO / "k1_runs" / "outputs" / "arm_wave")
                       .glob("*trained*.metrics.npz"))
        if not cands:
            pytest.skip("no arm_wave run recorded")
        z = {k: v for k, v in np.load(cands[-1]).items()}
        out = ev.cycle_consistency(z, GROUPS)
        corrs = [c for c in out["cross_cycle_error_correlation"] if c is not None]
        assert corrs and all(-1.0 <= c <= 1.0 for c in corrs)
        assert out["mean_correlation"] is not None


# ======================================================================
# Task C1 — schema numeric/type constraints
# ======================================================================
def _minimal_v2():
    return {
        "schema_version": 2, "skill_id": "t", "version": "1", "description": "d",
        "root_mode": "stationary",
        "motion": {"kind": "cyclic", "fps": 50, "source_frames": 246,
                   "transition_frames": 25,
                   "resources": {"train_npz": "x.npz",
                                 "deploy_npz_cyclic": "y.npz"}},
        "control_profile": {"extends": "k1_default", "physics_dt": 0.005,
                            "decimation": 4, "policy_frequency_hz": 50,
                            "joint_stiffness": [1.0] * 22,
                            "joint_damping": [1.0] * 22,
                            "effort_limit": [1.0] * 22},
        "training_profile": {"framework": "x", "task_id": "T-v0",
                             "play_task_id": "T-v0-Play", "num_envs": 16,
                             "max_iterations": 10, "seed": 1},
        "runtime_profile": {"policies": {"baseline": {"deploy_task": "b"}}},
        "evaluation_profile": {"metrics": ["fall_rate"]},
    }


def _load(tmp_path, data):
    p = tmp_path / "skill.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SkillDefinition.from_yaml(p)


@pytest.mark.parametrize("field,path_value", [
    ("fps string", "50"), ("fps zero", 0), ("fps negative", -50),
    ("fps nan", float("nan")), ("fps inf", float("inf")), ("fps bool", True),
])
def test_fps_rejected(tmp_path, field, path_value):
    d = _minimal_v2(); d["motion"]["fps"] = path_value
    with pytest.raises(SkillError, match="fps"):
        _load(tmp_path, d)


@pytest.mark.parametrize("value", [0, -5, 246.5, True, "246"])
def test_source_frames_rejected(tmp_path, value):
    d = _minimal_v2(); d["motion"]["source_frames"] = value
    with pytest.raises(SkillError, match="source_frames"):
        _load(tmp_path, d)


@pytest.mark.parametrize("value", [0, -0.005, float("nan"), float("inf"), "0.005", True])
def test_physics_dt_rejected(tmp_path, value):
    d = _minimal_v2(); d["control_profile"]["physics_dt"] = value
    with pytest.raises(SkillError, match="physics_dt"):
        _load(tmp_path, d)


@pytest.mark.parametrize("value", [0, -1, 4.5, True, "4"])
def test_decimation_rejected(tmp_path, value):
    d = _minimal_v2(); d["control_profile"]["decimation"] = value
    with pytest.raises(SkillError, match="decimation"):
        _load(tmp_path, d)


@pytest.mark.parametrize("field", ["num_envs", "max_iterations"])
@pytest.mark.parametrize("value", [0, -1, 16.5, True, "16"])
def test_training_ints_rejected(tmp_path, field, value):
    d = _minimal_v2(); d["training_profile"][field] = value
    with pytest.raises(SkillError, match=field):
        _load(tmp_path, d)


@pytest.mark.parametrize("field", ["joint_stiffness", "joint_damping", "effort_limit"])
def test_gains_type_and_range_rejected(tmp_path, field):
    base = [1.0] * 22
    for bad in ("0.5", None, float("nan"), float("inf"), True):
        d = _minimal_v2(); d["control_profile"][field] = base.copy()
        d["control_profile"][field][7] = bad
        with pytest.raises(SkillError, match=field):
            _load(tmp_path, d)
    # semantic ranges: stiffness/effort must be > 0, damping >= 0
    d = _minimal_v2(); d["control_profile"][field] = base.copy()
    d["control_profile"][field][3] = 0.0 if field != "joint_damping" else -0.1
    if field != "joint_damping":
        d = _minimal_v2(); d["control_profile"][field] = base.copy()
        d["control_profile"][field][3] = -1.0
    with pytest.raises(SkillError, match=field):
        _load(tmp_path, d)
    d = _minimal_v2(); d["control_profile"]["joint_damping"] = base.copy()
    d["control_profile"]["joint_damping"][3] = -0.1
    with pytest.raises(SkillError, match="joint_damping"):
        _load(tmp_path, d)


def test_valid_minimal_definition_still_loads(tmp_path):
    assert _load(tmp_path, _minimal_v2()).motion_kind == "cyclic"


# ======================================================================
# Task C2 — 22-DOF joint-order contract
# ======================================================================
def _write_npz(path: Path, *, frames=5, joint_names=None, joint_values=None,
               permuted=False):
    rng = np.random.default_rng(0)
    jp = rng.uniform(-1, 1, size=(frames, 22)).astype(np.float32)
    if joint_values is not None:
        jp = np.asarray(joint_values, dtype=np.float32)
    if permuted:
        jp = jp[:, [1, 0] + list(range(2, 22))].copy()
    data = {
        "fps": np.array([50.0]),
        "joint_pos": jp,
        "joint_vel": np.zeros_like(jp),
        "body_pos_w": np.zeros((frames, 23, 3), np.float32),
        "body_quat_w": np.tile([1.0, 0, 0, 0], (frames, 23, 1)).astype(np.float32),
        "body_lin_vel_w": np.zeros((frames, 23, 3), np.float32),
        "body_ang_vel_w": np.zeros((frames, 23, 3), np.float32),
    }
    if joint_names is not None:
        data["joint_names"] = np.array(joint_names)
    np.savez(path, **data)


class _Def:
    """Minimal duck-typed definition for MotionResourceSet.from_spec."""
    def __init__(self, resources, root_mode="free", kind="oneshot"):
        import copy
        self.skill_id = "t"
        self.root_mode = root_mode
        self.motion = {"kind": kind, "fps": 50, "source_frames": 5,
                       "transition_frames": 0, "resources": resources}

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else Path("/abs") / p


NAMES = [f"j{i:02d}" for i in range(22)]


class TestJointOrderContract:
    """MINIMAL_FIX_01-era structural checks (REMEDIATION_01 tightened the
    identity contract; see test_remediation_01.py). Synthetic sets inject
    their own canonical order so they don't depend on the deploy env."""

    def _mrs(self, resources):
        mrs = MotionResourceSet.from_spec(_Def(resources))
        mrs.canonical_npz_names = list(NAMES)   # synthetic authority
        return mrs

    def test_named_consistent_passes(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(1).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_names=NAMES, joint_values=shared)
        assert self._mrs({"train_npz": str(a), "deploy_npz": str(b)}).verify() == []

    def test_named_inconsistent_order_rejected(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(2).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_names=NAMES[::-1], joint_values=shared)
        problems = self._mrs({"train_npz": str(a), "deploy_npz": str(b)}).verify()
        assert any("joint identity mismatch" in p for p in problems)

    def test_unnamed_without_declaration_rejected(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(3).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_values=shared)          # no names, no declaration
        problems = self._mrs({"train_npz": str(a), "deploy_npz": str(b)}).verify()
        assert any("no joint_names and no joint_order" in p for p in problems)

    def test_unnamed_with_declaration_and_matching_data_passes(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(4).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_values=shared)          # no names
        import hashlib
        b_sha = hashlib.sha256(b.read_bytes()).hexdigest()
        mrs = self._mrs({
            "train_npz": str(a),
            "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                           "content_sha256": b_sha}})
        assert mrs.verify() == []

    def test_declaration_contradicted_by_data_rejected(self, tmp_path):
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        shared = np.random.default_rng(5).uniform(-1, 1, (5, 22))
        _write_npz(a, joint_names=NAMES, joint_values=shared)
        _write_npz(b, joint_values=shared, permuted=True)   # different order!
        import hashlib
        b_sha = hashlib.sha256(b.read_bytes()).hexdigest()
        problems = self._mrs({
            "train_npz": str(a),
            "deploy_npz": {"path": str(b), "joint_order": "train_npz",
                           "content_sha256": b_sha}}).verify()
        assert any("contradicted by data" in p for p in problems)

    def test_real_skills_joint_order_contract(self):
        from k1_skills.registry import SkillRegistry
        for skill_id in ("arm_wave", "mj_dance_002", "top_rock"):
            definition = SkillRegistry().get(skill_id)
            problems = MotionResourceSet.from_spec(definition).verify()
            assert problems == [], (skill_id, problems)
