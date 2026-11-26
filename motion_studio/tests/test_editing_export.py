"""Editing + export tests: retiming, amplitude, keyframes, roundtrip."""

import numpy as np
import pytest

from motion_studio.core import rotations as R
from motion_studio.export.npz_exporter import export_motion, load_motion, verify_roundtrip
from motion_studio.processing.foot_contact import FootContactDetector
from motion_studio.processing.keyframe_editor import KeyframeEditor
from motion_studio.processing.motion_filter import smooth
from motion_studio.processing.motion_retimer import resample_fps, set_speed, trim
from motion_studio.processing.motion_scaler import scale_amplitude


# ---------------------------------------------------------------------------
# retiming
# ---------------------------------------------------------------------------

def test_speed_double_shortens(motion_071):
    out = set_speed(motion_071, 2.0)
    assert out.frame_count == pytest.approx(motion_071.frame_count / 2, abs=1)
    assert out.duration == pytest.approx(motion_071.duration / 2, rel=0.02)
    assert out.time_scale == pytest.approx(2.0)
    # original untouched
    assert motion_071.frame_count == 123
    assert motion_071.time_scale == 1.0


def test_speed_identity(motion_071):
    out = set_speed(motion_071, 1.0)
    assert np.allclose(out.global_orient, motion_071.global_orient, atol=1e-8)
    assert np.allclose(out.transl, motion_071.transl, atol=1e-8)


def test_speed_slerp_halfway(motion_071):
    """factor 2.0 => new frame k corresponds to old 2k; frame 10 == old 20."""
    out = set_speed(motion_071, 2.0)
    assert np.allclose(out.global_orient[10], motion_071.global_orient[20], atol=1e-6)
    assert np.allclose(out.transl[10], motion_071.transl[20], atol=1e-8)


def test_trim(motion_071):
    out = trim(motion_071, 10, 50)
    assert out.frame_count == 40
    assert np.allclose(out.global_orient[0], motion_071.global_orient[10])


def test_resample_fps(motion_071):
    out = resample_fps(motion_071, 30.0)
    assert out.motion_fps == 30.0
    assert out.frame_count == pytest.approx(123 * 30 / 25, abs=1)
    assert out.duration == pytest.approx(motion_071.duration, rel=0.02)


# ---------------------------------------------------------------------------
# amplitude
# ---------------------------------------------------------------------------

def _rom(md, jid_from, jid_to):
    """Range of motion of a limb: max joint displacement from its mean pos."""
    p = md.joint_positions[:, jid_to]
    return float(np.abs(p - p.mean(axis=0)).max())


def test_amplitude_identity(fk, motion_071):
    out = scale_amplitude(motion_071, 1.0, "arms", smpl_fk=fk)
    # compare as rotations (per-joint sign/representation equivalence)
    m_a = R.aa_to_mat(out.body_pose.reshape(123, 21, 3))
    m_b = R.aa_to_mat(motion_071.body_pose.reshape(123, 21, 3))
    assert np.allclose(m_a, m_b, atol=1e-7)


def _angular_rom(md, jid):
    """Max rotation angle of one body joint over time (radians)."""
    return float(np.linalg.norm(md.body_pose.reshape(-1, 21, 3)[:, jid], axis=1).max())


def test_amplitude_reduce_angular_rom(fk, motion_071):
    full = scale_amplitude(motion_071, 1.0, "arms", smpl_fk=fk)
    half = scale_amplitude(motion_071, 0.5, "arms", smpl_fk=fk)
    # joint angular ROM strictly halves on arm joints (left_shoulder = pose idx 15)
    a_full = _angular_rom(full, 15)
    a_half = _angular_rom(half, 15)
    assert a_half == pytest.approx(a_full * 0.5, rel=1e-6)
    # legs unaffected
    assert np.allclose(half.body_pose.reshape(123, 21, 3)[:, 1],
                       full.body_pose.reshape(123, 21, 3)[:, 1], atol=1e-10)


def test_amplitude_cartesian_arc_shrinks_single_joint(fk):
    """Single-joint swing: halved amplitude must shrink the wrist arc
    (Cartesian monotonicity holds when base pose is at T-pose)."""
    from motion_studio.core.motion_data import MotionData

    T = 50
    t = np.linspace(0, 1, T)
    bp = np.zeros((T, 63))
    bp[:, 3 * 15 + 2] = 0.8 * np.sin(2 * np.pi * t)  # left_shoulder z-swing
    md = MotionData(
        motion_fps=25.0, global_orient=np.zeros((T, 3)), body_pose=bp,
        betas=np.zeros((T, 10)), transl=np.zeros((T, 3)), joint_positions=None,
    )
    full = scale_amplitude(md, 1.0, "arms", smpl_fk=fk)
    half = scale_amplitude(md, 0.5, "arms", smpl_fk=fk)
    assert _rom(half, 16, 20) < _rom(full, 16, 20) * 0.9


def test_amplitude_zero_is_tpose(fk, motion_071):
    out = scale_amplitude(motion_071, 0.0, "all", smpl_fk=fk)
    # all body joints identity => pure T-pose shape
    assert np.abs(out.body_pose).max() < 1e-9


def test_amplitude_window_blend_no_jump(fk, motion_071):
    """Blended window: frames outside the range stay unchanged."""
    out = scale_amplitude(motion_071, 0.3, "arms", frame_range=(40, 80), smpl_fk=fk)
    assert np.allclose(out.body_pose[0], motion_071.body_pose[0], atol=1e-9)
    assert np.allclose(out.body_pose[-1], motion_071.body_pose[-1], atol=1e-9)
    # inside the range the arms differ
    assert not np.allclose(out.body_pose[60], motion_071.body_pose[60])


# ---------------------------------------------------------------------------
# keyframes
# ---------------------------------------------------------------------------

def test_keyframe_interpolate_slerp_midpoint(fk, motion_071):
    ke = KeyframeEditor(motion_071, smpl_fk=fk)
    ke.add_keyframe(0)
    ke.add_keyframe(100)
    # set distinct arm poses at both keyframes
    pose_a = np.zeros(63)
    pose_b = np.zeros(63)
    pose_a[3 * 15 : 3 * 15 + 3] = [0, 0, 0.8]      # left_shoulder
    pose_b[3 * 15 : 3 * 15 + 3] = [0, 0, -0.8]
    ke.set_keyframe_pose(0, body_pose=pose_a)
    ke.set_keyframe_pose(100, body_pose=pose_b)
    ke.interpolate_between_keyframes()
    mid_aa = motion_071.body_pose[50].reshape(21, 3)[15]
    assert np.allclose(mid_aa, [0, 0, 0.0], atol=1e-6)  # slerp midpoint ~ zero
    q_mid = R.aa_to_quat(motion_071.body_pose[50].reshape(21, 3)[15][None])
    expected = R.aa_to_quat(np.array([[0, 0, 0.0]]))
    assert np.allclose(q_mid, expected, atol=1e-6)


def test_keyframe_edit_does_not_touch_original(fk, motion_071):
    orig = motion_071.body_pose.copy()
    ke = KeyframeEditor(motion_071.copy(), smpl_fk=fk)
    ke.add_keyframe(10)
    ke.set_keyframe_pose(10, body_pose=np.full(63, 0.1))
    assert np.allclose(motion_071.body_pose, orig)


def test_smooth_reduces_jitter(fk, motion_071):
    rng = np.random.default_rng(0)
    noisy = motion_071.copy()
    noise = rng.normal(scale=0.03, size=noisy.body_pose.shape)
    noisy.body_pose = noisy.body_pose + noise
    noisy.joint_positions, _ = fk.joints_from_params(
        noisy.global_orient, noisy.body_pose, noisy.betas, noisy.transl
    )
    smoothed = smooth(noisy, window=7, sigma=1.5, smpl_fk=fk)
    jitter = lambda md: np.abs(np.diff(md.joint_positions[:, 20], axis=0)).mean()
    assert jitter(smoothed) < jitter(noisy)


# ---------------------------------------------------------------------------
# export roundtrip
# ---------------------------------------------------------------------------

def test_export_roundtrip_bitexact(tmp_path, fk, motion_071):
    motion_071.contacts = FootContactDetector().detect(
        motion_071.joint_positions, motion_071.motion_fps
    )
    report = verify_roundtrip(motion_071, tmp_path, atol=1e-12)
    assert report["PASS"], report


def test_export_refuses_overwrite(tmp_path, motion_071):
    export_motion(motion_071, tmp_path / "a.npz")
    with pytest.raises(FileExistsError):
        export_motion(motion_071, tmp_path / "a.npz")


def test_loaded_npz_selfcontained(tmp_path, fk, motion_071):
    """Independent reader: load npz WITHOUT Motion Studio classes."""
    import json

    exp = export_motion(motion_071, tmp_path / "clean_motion.npz")
    z = np.load(exp["npz"], allow_pickle=False)
    # key contract fields
    for k in ("motion_fps", "frame_count", "duration", "timestamps",
              "global_orient", "body_pose", "betas", "transl",
              "joint_positions", "contact_left", "contact_right",
              "contact_state", "joint_names", "parents",
              "rotation_representation", "schema_version", "metadata_json"):
        assert k in z.files, f"missing {k}"
    assert z["global_orient"].shape == (123, 3)
    assert z["body_pose"].shape == (123, 63)
    assert z["joint_positions"].shape == (123, 24, 3)
    assert str(z["rotation_representation"]) == "smpl_axis_angle"
    meta = json.loads(str(z["metadata_json"]))
    assert meta["source_hash"], "provenance hash missing"
    assert meta["coordinate_system"]["up_axis"] == "+Y"
    cs = json.loads(str(z["coordinate_json"]))
    assert cs["units"] == "meters"
    # duration consistent with fps/timestamps
    assert abs(float(z["duration"]) - z["timestamps"][-1] - 1 / float(z["motion_fps"])) < 1e-9
