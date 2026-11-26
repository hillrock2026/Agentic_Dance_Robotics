"""Learning Engine: drive the existing booster_train infrastructure.

FRAMEWORK_02 generalizations:

- ``train_command`` built from the skill's training_profile with optional
  launch-time overrides (num_envs / max_iterations), per-skill extra CLI
  args (``cli_overrides``), continuation training (``resume``), and a
  derived default ``experiment_name``.
- ``export`` writes a versioned policy artifact: every export lands in
  ``k1_runs/policies/<skill>/<name>-<run>/`` with a manifest (jit+onnx
  copies, checkpoint provenance, sha256) and updates ``latest.json``;
  a stable filename is still staged into the deploy models dir so deploy
  task configs never change, but the manifest makes provenance
  traceable and enables drift detection at runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.paths import (BOOSTER_TRAIN_DIR, POLICIES_DIR, RUNS_DIR,
                             conda_python, deploy_models_dir, ensure_run_dirs)
from k1_skills.spec import SkillDefinition
from k1_skills import tasks as taskmod


def _run(cmd: list[str], cwd: Path, log_file: Path, env_extra: dict | None = None) -> int:
    import os
    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if env_extra:
        env.update(env_extra)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def experiment_name(definition: SkillDefinition) -> str:
    tp = definition.training_profile
    return str(tp.get("experiment_name") or definition.artifact_name("experiment"))


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def prepare(definition: SkillDefinition, *, force: bool = False) -> dict:
    """Generate the training npz from the official CSV (idempotent)."""
    res = definition.motion.get("resources", {})
    csv_path = definition.path(res["booster_csv"]) if res.get("booster_csv") else None
    npz_path = definition.path(res["train_npz"])
    if csv_path is None:
        if npz_path.exists():
            return {"status": "no-booster-csv", "train_npz": str(npz_path),
                    "skipped": True,
                    "note": "skill declares no booster_csv; prepare is a no-op"}
        raise SkillError(SkillErrorCode.MISSING_RESOURCE,
                         f"skill '{definition.skill_id}' declares no booster_csv and "
                         f"train_npz is missing: {npz_path}")
    if not csv_path.exists():
        raise SkillError(SkillErrorCode.MISSING_RESOURCE, f"booster csv missing: {csv_path}")
    if npz_path.exists() and not force:
        return {"status": "exists", "train_npz": str(npz_path), "skipped": True}

    log_file = RUNS_DIR / "logs" / f"{definition.skill_id}_prepare.log"
    cmd = [
        conda_python(), "scripts/csv_to_npz.py", "--headless",
        "--input_file", str(csv_path),
        "--input_fps", str(int(definition.motion["fps"])),
        "--output_file", str(npz_path),
        "--robot", "k1",
    ]
    code = _run(cmd, BOOSTER_TRAIN_DIR, log_file)
    if code != 0 or not npz_path.exists():
        raise SkillError(SkillErrorCode.PREPARE_FAILED,
                         f"csv_to_npz failed (exit {code}); see {log_file}")
    return {"status": "created", "train_npz": str(npz_path), "log": str(log_file)}


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------
def effective_training_config(definition: SkillDefinition, *,
                              num_envs: int | None = None,
                              max_iterations: int | None = None) -> dict:
    """The final effective training config AFTER all layers (YAML, profile
    inheritance, CLI overrides). This — not the raw YAML — is what gets
    validated and executed (REMEDIATION_01 Task C)."""
    tp = definition.training_profile
    return {
        "skill_id": definition.skill_id,
        "num_envs": num_envs if num_envs is not None else tp["num_envs"],
        "max_iterations": (max_iterations if max_iterations is not None
                           else tp["max_iterations"]),
        "seed": tp["seed"],
        "control": {k: definition.control_profile[k]
                    for k in ("physics_dt", "decimation", "policy_frequency_hz",
                              "joint_stiffness", "joint_damping", "effort_limit")
                    if k in definition.control_profile},
    }


def train_command(definition: SkillDefinition, *,
                  num_envs: int | None = None,
                  max_iterations: int | None = None) -> list[str]:
    """Build the booster_train command from the skill's training profile.

    The FINAL effective config (YAML -> inheritance -> overrides) is
    validated before the command is built, so CLI overrides cannot
    bypass the schema. ``cli_overrides`` from the YAML are appended
    verbatim (Isaac/rsl_rl flag passthrough); ``resume`` adds the
    continuation flags.
    """
    from k1_skills.spec import validate_effective_training_config
    effective = effective_training_config(definition, num_envs=num_envs,
                                          max_iterations=max_iterations)
    validate_effective_training_config(effective)   # raises before launch
    tp = definition.training_profile
    cmd = [
        conda_python(), "scripts/rsl_rl/train.py",
        "--task", tp["task_id"],
        "--headless",
        "--num_envs", str(effective["num_envs"]),
        "--max_iterations", str(effective["max_iterations"]),
        "--seed", str(tp["seed"]),
    ]
    resume = tp.get("resume")
    if resume:
        cmd += ["--resume"]
        if resume.get("load_run"):
            cmd += ["--load_run", str(resume["load_run"])]
        if resume.get("checkpoint"):
            cmd += ["--checkpoint", str(resume["checkpoint"])]
    cmd += [str(a) for a in tp.get("cli_overrides", [])]
    return cmd


def train_launch(definition: SkillDefinition, *, background: bool = False,
                 task_id: str | None = None, num_envs: int | None = None,
                 max_iterations: int | None = None) -> dict:
    """Launch RL training. Background mode returns a task id immediately.

    The background process runs under a bash wrapper appending
    ``TRAIN_EXIT_CODE=<n>`` to the log so rl.status can finalize the
    record without a supervisor daemon.
    """
    ensure_run_dirs()
    task_id = task_id or taskmod.new_task_id("rl-train")
    cmd = train_command(definition, num_envs=num_envs, max_iterations=max_iterations)
    log_file = RUNS_DIR / "logs" / f"{task_id}.log"

    if not background:
        code = _run(cmd, BOOSTER_TRAIN_DIR, log_file)
        taskmod.create_task(task_id, "rl.train", cmd, {
            "log": str(log_file), "skill_id": definition.skill_id,
            "max_iterations": int(max_iterations or definition.training_profile["max_iterations"])})
        taskmod.mark_from_exit_code(task_id, code)
        state = taskmod.get_task(task_id)
        if code != 0:
            raise SkillError(SkillErrorCode.TRAIN_LAUNCH_FAILED,
                             f"training exited {code}; see {log_file}")
        return state

    import os
    import shlex
    wrapped = (
        f"{shlex.join(cmd)} ; rc=$? ; "
        f"echo TRAIN_EXIT_CODE=$rc >> {shlex.quote(str(log_file))}"
    )
    launcher = ["/bin/bash", "-c", wrapped]
    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_file.open("a", encoding="utf-8")
    proc = subprocess.Popen(launcher, cwd=str(BOOSTER_TRAIN_DIR), stdout=log_fh,
                            stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    taskmod.create_task(task_id, "rl.train", cmd, {
        "pid": proc.pid, "log": str(log_file), "skill_id": definition.skill_id,
        "max_iterations": int(max_iterations or definition.training_profile["max_iterations"]),
        "experiment_name": experiment_name(definition)})
    return taskmod.get_task(task_id)


_EXIT_RE = re.compile(r"TRAIN_EXIT_CODE=(\d+)")


def _final_exit_code(log_file: str | Path) -> int | None:
    p = Path(log_file)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = _EXIT_RE.findall(text)
    return int(matches[-1]) if matches else None


def _pid_alive(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}").stat()
        return True
    except OSError:
        return False


def train_status(task_id: str | None = None, *, list_recent: bool = False,
                 registry=None) -> dict | list:
    """Training task status/progress; finalizes records of exited runs.

    With ``list_recent`` (or no task_id) returns the recent rl.train
    task records instead.
    """
    from k1_skills import tasks as taskmod
    if list_recent or not task_id:
        recs = [r for r in taskmod.list_tasks(kind="rl.train")]
        recs.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        return recs[:10]

    if registry is None:
        from k1_skills.registry import SkillRegistry
        registry = SkillRegistry()

    state = taskmod.get_task(task_id)
    if state.get("kind") != "rl.train":
        raise SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                         f"task '{task_id}' is not an rl.train task "
                         f"(kind={state.get('kind')})")
    out = dict(state)
    log = state.get("log")
    if state.get("status") == "running":
        it = taskmod.latest_training_iteration(log) if log else None
        if it is not None:
            out["progress"] = {"iteration": it,
                               "max_iterations": state.get("max_iterations")}
        pid = state.get("pid")
        if pid is not None:
            out["process_alive"] = _pid_alive(int(pid))
            if not out["process_alive"]:
                code = _final_exit_code(log) if log else None
                status = ("succeeded" if code == 0 else "failed") if code is not None \
                    else _legacy_infer_status(state) or "failed"
                if code is None:
                    code = 0 if status == "succeeded" else -1
                updates = {"status": status, "exit_code": code}
                if status == "succeeded":
                    try:
                        definition = registry.get(state["skill_id"])
                        run_dir = latest_run_dir(definition)
                        updates["run_dir"] = str(run_dir)
                        updates["artifacts"] = {
                            "run_dir": str(run_dir),
                            "final_checkpoint": str(latest_checkpoint(run_dir))}
                    except Exception:  # noqa: BLE001 - artifacts best-effort
                        pass
                taskmod.update_task(task_id, **updates)
                out.update(updates)
    return out


def _legacy_infer_status(state: dict) -> str | None:
    """Marker-less launches: succeeded iff the final checkpoint exists."""
    max_it = state.get("max_iterations")
    skill_id = state.get("skill_id")
    if not max_it or not skill_id:
        return None
    try:
        from k1_skills.registry import SkillRegistry
        definition = SkillRegistry().get(skill_id)
        run_dir = latest_run_dir(definition)
    except Exception:  # noqa: BLE001
        return None
    if (run_dir / f"model_{int(max_it) - 1}.pt").exists():
        return "succeeded"
    return None


# ----------------------------------------------------------------------
# runs / checkpoints
# ----------------------------------------------------------------------
def latest_run_dir(definition: SkillDefinition) -> Path:
    root = BOOSTER_TRAIN_DIR / "logs" / "rsl_rl" / experiment_name(definition)
    if not root.exists():
        raise SkillError(SkillErrorCode.CHECKPOINT_NOT_FOUND,
                         f"no training runs under {root}")
    runs = sorted(d for d in root.iterdir() if d.is_dir())
    if not runs:
        raise SkillError(SkillErrorCode.CHECKPOINT_NOT_FOUND,
                         f"no training runs under {root}")
    return runs[-1]


def latest_checkpoint(run_dir: Path) -> Path:
    cks = sorted(run_dir.glob("model_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    if not cks:
        raise SkillError(SkillErrorCode.CHECKPOINT_NOT_FOUND,
                         f"no checkpoints in {run_dir}")
    return cks[-1]


# ----------------------------------------------------------------------
# export (versioned policy artifacts + manifest)
# ----------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def staged_policy_name(definition: SkillDefinition, role: str = "trained") -> str:
    entry = {}
    try:
        entry = definition.policy_entry(role)
    except SkillError:
        pass
    name = entry.get("policy_file")
    if not name:
        name = f"{definition.artifact_name('policy')}.jit"
    return Path(name).name


def policy_manifest_path(definition: SkillDefinition, role: str = "trained") -> Path:
    return POLICIES_DIR / definition.skill_id / "latest.json"


def export(definition: SkillDefinition, *, run_dir: Path | None = None,
           task_id: str | None = None, role: str = "trained") -> dict:
    """Export a run to jit+onnx, archive a versioned policy artifact with
    manifest, and stage the jit under a stable filename for sim2sim."""
    ensure_run_dirs()
    task_id = task_id or taskmod.new_task_id("policy-export")
    tp = definition.training_profile
    run_dir = run_dir or latest_run_dir(definition)
    checkpoint = latest_checkpoint(run_dir)
    log_file = RUNS_DIR / "logs" / f"{task_id}.log"

    cmd = [
        conda_python(), "scripts/rsl_rl/play.py",
        "--task", tp["play_task_id"],
        "--headless",
        "--num_envs", "16",
        "--checkpoint", str(checkpoint),
    ]
    taskmod.create_task(task_id, "policy.export", cmd, {
        "skill_id": definition.skill_id, "run_dir": str(run_dir),
        "checkpoint": str(checkpoint)})
    code = _run(cmd, BOOSTER_TRAIN_DIR, log_file)
    if code != 0:
        taskmod.mark_from_exit_code(task_id, code)
        raise SkillError(SkillErrorCode.EXPORT_FAILED,
                         f"export exited {code}; see {log_file}")

    exported = run_dir / "exported"
    jits = sorted(exported.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    onnx = sorted(exported.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if not jits:
        taskmod.mark_from_exit_code(task_id, 1)
        raise SkillError(SkillErrorCode.EXPORT_FAILED,
                         f"no jit produced under {exported}; see {log_file}")

    # -- versioned artifact archive + manifest --------------------------
    version_name = f"{Path(staged_policy_name(definition, role)).stem}-{run_dir.name}"
    version_dir = POLICIES_DIR / definition.skill_id / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    jit_copy = version_dir / Path(staged_policy_name(definition, role)).name
    shutil.copy2(jits[-1], jit_copy)
    onnx_copy = None
    if onnx:
        onnx_copy = version_dir / onnx[-1].name
        shutil.copy2(onnx[-1], onnx_copy)

    staged_name = staged_policy_name(definition, role)
    dst = deploy_models_dir() / staged_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jits[-1], dst)

    manifest = {
        "schema": "k1-policy-manifest/1",
        "skill_id": definition.skill_id,
        "role": role,
        "staged_file": str(dst),
        "staged_sha256": _sha256(dst),
        "version_dir": str(version_dir),
        "jit_archive": str(jit_copy),
        "onnx_archive": str(onnx_copy) if onnx_copy else None,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "training_task": definition.training_profile["task_id"],
        "experiment": experiment_name(definition),
        "control_profile": {
            "extends": definition.control_profile.get("_extends"),
            "physics_dt": definition.control_profile["physics_dt"],
            "decimation": definition.control_profile["decimation"],
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (POLICIES_DIR / definition.skill_id).mkdir(parents=True, exist_ok=True)
    (policy_manifest_path(definition, role)).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    artifacts = {
        "jit_exported": str(jits[-1]),
        "onnx_exported": str(onnx[-1]) if onnx else None,
        "jit_staged": str(dst),
        "staged_sha256": manifest["staged_sha256"],
        "version_dir": str(version_dir),
        "manifest": str(version_dir / "manifest.json"),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
    }
    taskmod.mark_from_exit_code(task_id, 0, artifacts)
    return {**taskmod.get_task(task_id), "artifacts": artifacts}


def check_staged_policy(definition: SkillDefinition, role: str = "trained") -> dict:
    """Runtime contract check: staged jit must match the latest manifest.

    Returns {ok, manifest?, staged?, expected_sha256?, actual_sha256?};
    a missing manifest is reported as ok=None (skill never exported).
    """
    mp = policy_manifest_path(definition, role)
    if not mp.exists():
        return {"ok": None, "note": "no export manifest; baseline-only skill?"}
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    staged = Path(manifest["staged_file"])
    if not staged.exists():
        return {"ok": False, "manifest": str(mp), "staged": str(staged),
                "note": "staged file missing"}
    actual = _sha256(staged)
    return {"ok": actual == manifest["staged_sha256"],
            "manifest": str(mp), "staged": str(staged),
            "expected_sha256": manifest["staged_sha256"],
            "actual_sha256": actual}
