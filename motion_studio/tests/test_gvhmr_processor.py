"""GVHMRProcessor integration tests (use completed jobs; skip if absent)."""

import json
from pathlib import Path

import pytest

from motion_studio.gvhmr.processor import GVHMRProcessor, probe_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_071 = "segment_071_343ee79d"
SRC_071 = PROJECT_ROOT / "inputs/hiphop87_elements/segment_071.mp4"


@pytest.fixture(scope="module")
def proc():
    return GVHMRProcessor()


def test_probe_video():
    if not SRC_071.exists():
        pytest.skip("source video missing")
    info = probe_video(SRC_071)
    assert info["width"] == 1920 and info["height"] == 1080
    assert info["fps"] == pytest.approx(25.0)
    assert info["duration_s"] == pytest.approx(4.92, abs=0.1)


def test_get_result_completed(proc):
    jobs = {j["job_id"] for j in proc.list_jobs()}
    if JOB_071 not in jobs:
        pytest.skip("job not run yet")
    job = proc.get_result(JOB_071)
    assert job["status"] == "COMPLETED"
    assert Path(job["raw_npz"]).exists()
    assert Path(job["hmr4d_results_pt"]).exists()


def test_validate_result(proc):
    jobs = {j["job_id"] for j in proc.list_jobs()}
    if JOB_071 not in jobs:
        pytest.skip("job not run yet")
    job = proc.get_result(JOB_071)
    assert proc.validate_result(job) == []


def test_validate_result_bad_shape(proc):
    import numpy as np

    jobs = {j["job_id"] for j in proc.list_jobs()}
    if JOB_071 not in jobs:
        pytest.skip("job not run yet")
    job = proc.get_result(JOB_071)
    bad = dict(job)
    bad["raw_npz"] = str(_corrupt_copy(job["raw_npz"]))
    errs = proc.validate_result(bad)
    assert errs != []


def _corrupt_copy(src):
    import shutil
    import tempfile

    z = dict(np.load(src))
    z["body_pose"] = z["body_pose"][:, :30]  # wrong shape
    out = Path(tempfile.mkdtemp()) / "bad.npz"
    np.savez(out, **z)
    return out


import numpy as np  # noqa: E402  (used above)


def test_process_video_missing_raises(proc):
    with pytest.raises(FileNotFoundError):
        proc.process_video("nonexistent/video.mp4")


def test_job_json_has_provenance(proc):
    jobs = {j["job_id"] for j in proc.list_jobs()}
    if JOB_071 not in jobs:
        pytest.skip("job not run yet")
    jf = Path(proc.outputs_root) / JOB_071 / "job.json"
    job = json.loads(jf.read_text())
    assert len(job["video_sha256"]) == 64
    assert job["video_meta"]["fps"] == pytest.approx(25.0)
    assert job["gvhmr_demo_seconds"] > 0
