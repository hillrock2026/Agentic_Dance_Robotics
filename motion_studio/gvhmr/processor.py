"""GVHMR batch processing interface.

Orchestrates the official GVHMR demo pipeline (subprocess, source untouched)
plus Motion Studio's result adapter, with job tracking, provenance, logging
and idempotent (never-overwrite) semantics.

Job states: PENDING -> RUNNING -> COMPLETED | FAILED
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_cfg() -> dict:
    import yaml

    cfg_path = PROJECT_ROOT / "motion_studio/config/motion_studio.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _ffmpeg_bin() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


_FFMPEG_SHIM_DIR = PROJECT_ROOT / "motion_studio/assets/bin"


def ensure_ffmpeg_on_path() -> str:
    """imageio-ffmpeg ships a versioned binary name; expose it as plain
    `ffmpeg` via a stable shim dir and return the PATH prefix to prepend."""
    import os

    src = Path(_ffmpeg_bin())
    _FFMPEG_SHIM_DIR.mkdir(parents=True, exist_ok=True)
    link = _FFMPEG_SHIM_DIR / "ffmpeg"
    if not (link.exists() and os.path.realpath(link) == str(src)):
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
        link.chmod(0o755)
    return str(_FFMPEG_SHIM_DIR)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> dict:
    """Read width/height/fps/duration/frames via the bundled ffmpeg binary."""
    ff = _ffmpeg_bin()
    out = subprocess.run(
        [ff, "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    ).stderr
    info: dict[str, Any] = {}
    for line in out.splitlines():
        line = line.strip()
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            info["duration_s"] = round(float(hh) * 3600 + float(mm) * 60 + float(ss), 3)
        if "Video:" in line and "fps" in line:
            info["fps"] = float(line.split("fps")[0].strip().split()[-1])
            for tok in line.split(","):
                tok = tok.strip()
                if "x" in tok and tok.split("x")[0].strip().isdigit():
                    w, h = tok.split("x")[:2]
                    info["width"], info["height"] = int(w), int(h.split(" ")[0])
    info["frame_count"] = int(round(info["duration_s"] * info["fps"]))
    return info


class GVHMRProcessor:
    """Unified interface over official GVHMR inference + Motion Studio adapter."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or _load_cfg()
        self.paths = self.cfg["paths"]
        self.gvhmr_repo = Path(self.paths["gvhmr_repo"])
        self.gvhmr_python = Path(self.paths["gvhmr_python"])
        self.outputs_root = Path(self.paths["outputs_root"]) / "gvhmr"
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _job_dir(self, video_path: Path) -> Path:
        digest = sha256_of(video_path)[:8]
        return self.outputs_root / f"{video_path.stem}_{digest}"

    def _load_job(self, job_dir: Path) -> dict | None:
        jf = job_dir / "job.json"
        if jf.exists():
            with open(jf) as f:
                return json.load(f)
        return None

    def _save_job(self, job: dict) -> None:
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        with open(job_dir / "job.json", "w") as f:
            json.dump(job, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def process_video(self, video_path: str | Path, static_cam: bool | None = None,
                      overwrite: bool = False) -> dict:
        """Run GVHMR on one video. Idempotent: completed jobs are reused."""
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
        if static_cam is None:
            static_cam = bool(self.cfg["gvhmr"]["static_cam"])

        probe = probe_video(video_path)
        max_sec = self.cfg["gvhmr"].get("max_video_seconds", 60)
        if probe.get("duration_s", 0) > max_sec:
            raise ValueError(
                f"video too long ({probe['duration_s']}s > {max_sec}s); trim it first"
            )

        job_dir = self._job_dir(video_path)
        job_id = job_dir.name
        prev = self._load_job(job_dir)
        if prev and prev.get("status") == "COMPLETED" and not overwrite:
            raw = job_dir / "gvhmr_raw.npz"
            if raw.exists():
                prev["cached"] = True
                self._jobs[job_id] = prev
                return prev

        job: dict[str, Any] = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "video": str(video_path),
            "video_sha256": sha256_of(video_path),
            "video_meta": probe,
            "static_cam": static_cam,
            "status": "PENDING",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gvhmr_repo": str(self.gvhmr_repo),
        }
        self._save_job(job)
        self._jobs[job_id] = job

        try:
            job["status"] = "RUNNING"
            job["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save_job(job)

            t0 = time.time()
            self._run_gvhmr_demo(video_path, job_dir, static_cam)
            t_demo = time.time() - t0
            job["gvhmr_demo_seconds"] = round(t_demo, 1)

            t0 = time.time()
            self._run_adapter(video_path, job_dir, static_cam)
            job["adapter_seconds"] = round(time.time() - t0, 1)

            result = self._collect_result(job_dir)
            job.update(result)
            job["status"] = "COMPLETED"
            job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:  # noqa: BLE001 - record any failure honestly
            job["status"] = "FAILED"
            job["error"] = f"{type(e).__name__}: {e}"
            log = Path(job_dir) / "job.log"
            if log.exists():
                tail = log.read_text(errors="replace").strip().splitlines()[-15:]
                job["error_log_tail"] = tail
        self._save_job(job)
        return job

    # ------------------------------------------------------------------
    def _run_gvhmr_demo(self, video_path: Path, job_dir: Path, static_cam: bool) -> None:
        cmd = [
            str(self.gvhmr_python), "tools/demo/demo.py",
            f"--video={video_path}",
            f"--output_root={job_dir}",
        ]
        if static_cam:
            cmd.append("-s")
        shim_dir = ensure_ffmpeg_on_path()
        import os

        env = os.environ.copy()
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
        env.pop("HF_ENDPOINT", None)
        with open(job_dir / "job.log", "a") as logf:
            logf.write(f"\n===== {time.strftime('%F %T')} RUN: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(
                cmd, cwd=self.gvhmr_repo, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=1800,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"GVHMR demo.py exited with {proc.returncode} (see job.log)")

    def _run_adapter(self, video_path: Path, job_dir: Path, static_cam: bool) -> None:
        # demo.py writes results into <output_root>/<video_stem>/
        result_dir = job_dir / video_path.stem
        if not (result_dir / "hmr4d_results.pt").exists():
            raise FileNotFoundError(f"hmr4d_results.pt not produced under {result_dir}")
        adapter = PROJECT_ROOT / self.cfg["gvhmr"]["adapter_script"]
        model_export = PROJECT_ROOT / "motion_studio/assets/smpl/smpl_neutral_core.npz"
        cmd = [
            str(self.gvhmr_python), str(adapter),
            f"--result_dir={result_dir}",
            f"--source_video={video_path}",
            f"--model_export={model_export}",
        ]
        if static_cam:
            cmd.append("--static_cam")
        import os

        env = os.environ.copy()
        with open(job_dir / "job.log", "a") as logf:
            logf.write(f"\n===== {time.strftime('%F %T')} ADAPTER: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(
                cmd, cwd=self.gvhmr_repo, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=600,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"adapter exited with {proc.returncode} (see job.log)")

    def _collect_result(self, job_dir: Path) -> dict:
        stems = [d for d in job_dir.iterdir() if d.is_dir() and (d / "hmr4d_results.pt").exists()]
        if not stems:
            raise FileNotFoundError("no result dir with hmr4d_results.pt")
        rd = stems[0]
        raw = rd / "gvhmr_raw.npz"
        if not raw.exists():
            raise FileNotFoundError(f"adapter output missing: {raw}")
        z = np.load(raw)
        meta = json.loads(str(z["meta_json"]))
        return {
            "result_dir": str(rd),
            "raw_npz": str(raw),
            "hmr4d_results_pt": str(rd / "hmr4d_results.pt"),
            "preview_incam": str(rd / "1_incam.mp4") if (rd / "1_incam.mp4").exists() else None,
            "preview_global": str(rd / "2_global.mp4") if (rd / "2_global.mp4").exists() else None,
            "frames": int(meta["frame_count"]),
            "gvhmr_commit": meta.get("gvhmr_commit"),
            "source_fps": meta.get("source_fps"),
        }

    # ------------------------------------------------------------------
    def process_batch(self, video_paths: list[str | Path], overwrite: bool = False) -> list[dict]:
        jobs = []
        for vp in video_paths:
            try:
                job = self.process_video(vp, overwrite=overwrite)
            except Exception as e:  # noqa: BLE001
                job = {
                    "job_id": str(vp),
                    "video": str(vp),
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {e}",
                }
            jobs.append(job)
        return jobs

    def get_result(self, job_id: str) -> dict:
        """Reload a job's status/result from disk (survives process restarts)."""
        job_dir = self.outputs_root / job_id
        job = self._load_job(job_dir)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        if job.get("status") == "COMPLETED":
            job.update(self._collect_result(job_dir))
        return job

    def list_jobs(self) -> list[dict]:
        out = []
        if self.outputs_root.exists():
            for d in sorted(self.outputs_root.iterdir()):
                j = self._load_job(d) if d.is_dir() else None
                if j:
                    out.append({k: j.get(k) for k in ("job_id", "status", "video", "frames")})
        return out

    def validate_result(self, result: dict) -> list[str]:
        """Structural validation of a raw GVHMR result (returns error list)."""
        errors = []
        raw = result.get("raw_npz")
        if not raw:
            return ["result has no raw_npz"]
        z = np.load(raw)
        need = ["global_orient", "body_pose", "betas", "transl", "joints_global", "meta_json"]
        for k in need:
            if k not in z.files:
                errors.append(f"raw npz missing {k}")
        if errors:
            return errors
        T = z["global_orient"].shape[0]
        for k, shape in [("body_pose", (T, 63)), ("betas", (T, 10)),
                         ("transl", (T, 3)), ("joints_global", (T, 24, 3))]:
            if z[k].shape != shape:
                errors.append(f"{k} shape {z[k].shape} != {shape}")
        for k in ("global_orient", "body_pose", "transl"):
            if not np.isfinite(z[k]).all():
                errors.append(f"{k} contains NaN/Inf")
        return errors
