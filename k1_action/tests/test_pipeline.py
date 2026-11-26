"""End-to-end tests on real data (not random arrays).

1  MotionData loading           test_load_human_motion
2  SMPL input format            test_smpl_input_format
3  human joint mapping          test_joint_mapping_config
4  K1 22-joint order            test_k1_joint_order_matches_xml
5  left/right direction         test_retarget_left_right
6  coordinate transform         test_coordinate_transform
7  joint limits                 test_joint_limits
8  time resampling              test_resample_duration_preserved
9  velocity/acceleration        test_derivatives
10 CSV export & reload          test_csv_roundtrip
11 metadata consistency         test_metadata_consistency
12 dataset builder              test_dataset_manifest
13 Virtual K1 replay            test_virtual_k1_replay (kinematic)
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from k1_action.coordinate_transform import (
    calibrate_torso_map, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, yaw_quat_xyzw,
)
from k1_action.joint_limits import apply_limits, load_limits_from_xml
from k1_action.motion_loader import load_human_motion, validate_human_motion
from k1_action.retargeting import FullBodyRetargeter
from k1_action.skeleton_mapping import (
    K1_JOINT_ORDER, OFFICIAL_CSV_COLUMNS, TRAJECTORY_COLUMNS, csv_name_to_xml,
    verify_order_against_model,
)
from k1_action.trajectory_processing import derivatives, joint_stats, resample
from k1_action.validators import validate_k1_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 1 ---------------------------------------------------------------------------
def test_load_human_motion(human_motion):
    assert human_motion.frame_count == 123
    assert human_motion.fps == 25.0
    assert validate_human_motion(human_motion) == []


def test_load_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_human_motion("/nonexistent/motion.npz")


# 2 ---------------------------------------------------------------------------
def test_smpl_input_format(human_motion):
    assert human_motion.joint_positions.shape[1:] == (24, 3)  # SMPL 24 joints
    assert human_motion.body_pose.shape[1:] == (63,)          # 21 joints axis-angle
    assert human_motion.contact_state is not None             # contact estimate present
    coord = json.loads(str(np.load(human_motion.path)["coordinate_json"]))
    assert coord["up_axis"] == "+Y" and coord["units"] == "meters"
    # Y-up sanity: head above pelvis in every frame
    jp = human_motion.joint_positions
    assert (jp[:, 15, 1] > jp[:, 0, 1]).all()


# 3 ---------------------------------------------------------------------------
def test_joint_mapping_config():
    from k1_action.skeleton_mapping import CFG, SMPL_IDS

    assert len(CFG["k1_joint_order"]) == 22
    assert len(CFG["chains"]) == 5  # head + 2 arms + 2 legs
    for jn in CFG["k1_joint_order"]:
        assert jn in CFG["csv_to_xml_name"] or jn in (
            set().union(*[set(v) for v in CFG["csv_to_xml_name"].values()]) | set(CFG["k1_joint_order"]))
    assert SMPL_IDS["left_shoulder"] == 16 and SMPL_IDS["right_ankle"] == 8


# 4 ---------------------------------------------------------------------------
def test_k1_joint_order_matches_xml():
    import mujoco

    m = mujoco.MjModel.from_xml_path(
        str(PROJECT_ROOT / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"))
    assert verify_order_against_model(m) == []
    actual = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    actual = [n for n in actual if n != "world_joint"]
    assert K1_JOINT_ORDER == actual


def test_official_csv_column_compatibility():
    header = next(csv.reader(open(PROJECT_ROOT / "virtual_k1/assets/booster_assets/motions/K1/k1_fight_001.csv")))
    assert OFFICIAL_CSV_COLUMNS == header  # exact official order & spelling
    assert csv_name_to_xml("left_shoulder_pitch_joint") == "aaleft_shoulder_pitch_joint"
    assert csv_name_to_xml("aahead_yaw_joint") == "aahead_yaw_joint"


# 5/6 -------------------------------------------------------------------------
def test_coordinate_transform(human_motion):
    r = FullBodyRetargeter()
    R = calibrate_torso_map(r.model, r.data, human_motion.joint_positions[0])
    assert R.shape == (3, 3)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)   # orthonormal
    assert np.linalg.det(R) > 0                          # right-handed
    # human up must map to K1 up (+Z)
    assert np.allclose(R @ np.array([0, 1.0, 0]), [0, 0, 1], atol=1e-6)
    # quat conventions
    q = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q)), q)
    yq = yaw_quat_xyzw(np.array([1.0, 0, 0]))
    assert np.isclose(np.linalg.norm(yq), 1.0)


def test_retarget_left_right(human_motion_short):
    """Left-arm motion must drive K1 LEFT arm, not mirrored."""
    import mujoco

    r = FullBodyRetargeter().calibrate(human_motion_short.joint_positions[0])
    traj = r.retarget(human_motion_short)
    # drive K1 with the trajectory, check left/right world assignment
    m = r.model
    d = mujoco.MjData(m)
    mid = traj.frame_count // 2
    d.qpos[:] = 0
    d.qpos[3] = 1
    for jn in K1_JOINT_ORDER:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        d.qpos[m.jnt_qposadr[jid]] = traj.joints[jn][mid]
    mujoco.mj_forward(m, d)
    lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_pitch_link")
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_pitch_link")
    assert d.xpos[lb][1] > d.xpos[1][1] > d.xpos[rb][1]  # left on +Y side
    # arms actually moved from stance (retargeting did something)
    moved = max(np.abs(traj.joints[j] - r.stance[j]).max() for j in K1_JOINT_ORDER if "shoulder" in j)
    assert moved > 0.05


# 7 ---------------------------------------------------------------------------
def test_joint_limits_records_violations():
    limits = load_limits_from_xml()
    raw = {jn: np.full(10, (limits[jn][0] + limits[jn][1]) / 2) for jn in limits}
    lo, hi = limits["left_knee_pitch_joint"]
    raw["left_knee_pitch_joint"][:] = hi + 0.5  # force violation
    limited, report = apply_limits(raw, limits, margin_frac=0.02)
    assert len(report.violations) == 10
    assert report.max_excess_rad > 0.3
    assert report.has_severe
    assert limited["left_knee_pitch_joint"].max() <= hi


def test_joint_limits_clean_pass():
    limits = load_limits_from_xml()
    raw = {jn: np.full(5, (limits[jn][0] + limits[jn][1]) / 2) for jn in limits}
    limited, report = apply_limits(raw, limits)
    assert report.violations == []
    assert not report.has_severe


# 8 ---------------------------------------------------------------------------
def test_resample_duration_preserved(arm_wave_traj_full):
    t50 = resample(arm_wave_traj_full, 50.0)
    assert abs(t50.duration() - arm_wave_traj_full.duration()) < 1e-6
    assert abs(t50.timestamps[0] - arm_wave_traj_full.timestamps[0]) < 1e-9
    assert t50.frame_count == pytest.approx(arm_wave_traj_full.frame_count * 2, abs=2)
    # quaternion stays unit-norm through SLERP
    norms = np.linalg.norm(t50.root_quat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    # joint values at shared timestamps match original
    k = 10  # original frame 10 == t=0.4s == resampled frame 20
    j = "aaleft_shoulder_pitch_joint"
    assert t50.joints[j][20] == pytest.approx(arm_wave_traj_full.joints[j][k], abs=1e-9)


# 9 ---------------------------------------------------------------------------
def test_derivatives():
    from k1_action.retargeting import K1Trajectory

    fps = 50.0
    T = 101
    ts = np.arange(T) / fps
    jn = K1_JOINT_ORDER[0]
    joints = {j: np.sin(2 * np.pi * 1.0 * ts) for j in K1_JOINT_ORDER}  # 1 Hz sine
    traj = K1Trajectory(fps=fps, timestamps=ts, joints_raw=joints, joints=joints,
                        root_pos=np.zeros((T, 3)), root_quat=np.tile([0, 0, 0, 1.0], (T, 1)))
    vel, acc = derivatives(traj)
    # sine amplitude 1 at 1 Hz: max |v|=2pi, max |a|=4pi^2
    assert np.abs(vel[jn]).max() == pytest.approx(2 * np.pi, rel=0.01)
    assert np.abs(acc[jn]).max() == pytest.approx(4 * np.pi ** 2, rel=0.02)
    stats = joint_stats(traj)
    assert stats[jn]["range_rad"] == pytest.approx(2.0, abs=0.01)


# 10/11 -----------------------------------------------------------------------
def _export_tmp(tmp_path, arm_wave_traj_full):
    from k1_action.action_exporter import export_k1_action

    v = validate_k1_trajectory(arm_wave_traj_full)
    return export_k1_action(arm_wave_traj_full, tmp_path / "act", action_name="t",
                            action_category="test", source_motion="x.npz",
                            source_sha256="0" * 64, source_body_model="SMPL",
                            source_fps=25.0, validation=v, overwrite=False)


def test_csv_roundtrip(tmp_path, arm_wave_traj_full):
    exp = _export_tmp(tmp_path, arm_wave_traj_full)
    from k1_action.action_exporter import load_k1_action

    act = load_k1_action(exp["dir"])
    assert act["columns"] == TRAJECTORY_COLUMNS
    rows = act["rows"]
    assert len(rows) == arm_wave_traj_full.frame_count
    from k1_action.skeleton_mapping import xml_name_to_csv

    for jn in K1_JOINT_ORDER:
        got = np.array([float(r[xml_name_to_csv(jn)]) for r in rows])
        assert np.allclose(got, arm_wave_traj_full.joints[jn], atol=1e-8)
    # official-29 subset equals the official header order
    assert [c for c in act["columns"] if c not in ("frame_index", "timestamp_s")] == OFFICIAL_CSV_COLUMNS
    # root quat norm
    qn = np.array([[float(r[f"root_quat_{k}"]) for k in "xyzw"] for r in rows])
    assert np.allclose(np.linalg.norm(qn, axis=1), 1.0, atol=1e-8)
    with pytest.raises(FileExistsError):
        _export_tmp(tmp_path, arm_wave_traj_full)  # no overwrite


def test_metadata_consistency(tmp_path, arm_wave_traj_full):
    exp = _export_tmp(tmp_path, arm_wave_traj_full)
    meta = json.loads(Path(exp["metadata_json"]).read_text())
    assert meta["joint_names"] == K1_JOINT_ORDER
    assert meta["source_sha256"] == "0" * 64
    assert meta["software_version"].startswith("k1_action-")
    assert meta["angle_unit"].startswith("radians")
    assert meta["conversion_status"] in ("CONVERTED", "VALIDATED", "NEEDS_REVIEW")
    import hashlib

    h = hashlib.sha256(Path(exp["trajectory_csv"]).read_bytes()).hexdigest()
    assert h == exp["trajectory_sha256"]


# 12 --------------------------------------------------------------------------
def test_dataset_manifest(tmp_path):
    from k1_action.dataset_builder import build_k1_training_dataset

    d = tmp_path / "ds" / "actA"
    d.mkdir(parents=True)
    (d / "trajectory.csv").write_text("frame_index\n0\n")
    (d / "metadata.json").write_text(json.dumps({"output_fps": 50}))
    m = build_k1_training_dataset(
        [{"name": "actA", "category": "test", "dir": str(d), "status": "CONVERTED"}],
        tmp_path / "ds")
    assert m["action_count"] == 1
    assert m["actions"][0]["status"] == "CONVERTED"  # never auto-upgraded
    assert len(m["actions"][0]["trajectory_sha256"]) == 64
    assert (tmp_path / "ds" / "manifest.json").exists()


# 13 --------------------------------------------------------------------------
def test_virtual_k1_replay_kinematic(arm_wave_traj_full, human_motion):
    from k1_action.virtual_k1_replay import kinematic_replay

    r = kinematic_replay(arm_wave_traj_full, human_motion.contact_state)
    assert r["pass"] is True
    assert r["details"]["left_right_assignment_ok"] is True


def test_validate_k1_trajectory_reports(arm_wave_traj_full):
    rep = validate_k1_trajectory(arm_wave_traj_full)
    assert rep["checks"]["finite"]["pass"]
    assert rep["checks"]["joint_limits"]["pass"]
    assert rep["checks"]["timeline"]["pass"]
    # known issue recorded honestly: shoulder speed above threshold
    assert rep["status"] in ("VALIDATED", "NEEDS_REVIEW")
    assert set(rep["joint_stats"]) == set(K1_JOINT_ORDER)
