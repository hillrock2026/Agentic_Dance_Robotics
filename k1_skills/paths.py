"""Repository path resolution for the K1 Dance Skill Framework.

All paths are repo-relative so the framework works from any cwd. The two
existing tool trees are consumed as-is (no restructuring):

- ``tools/booster_train``  — Isaac Lab / rsl_rl RL training (conda env)
- ``tools/booster_deploy`` — MuJoCo sim2sim deployment (project venv)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- tool trees (existing, validated) --------------------------------------
BOOSTER_TRAIN_DIR = REPO_ROOT / "tools" / "booster_train"
BOOSTER_DEPLOY_DIR = REPO_ROOT / "tools" / "booster_deploy"
BOOSTER_ASSETS_MOTIONS = REPO_ROOT / "virtual_k1" / "assets" / "booster_assets" / "motions" / "K1"

# --- framework-owned locations ----------------------------------------------
SKILLS_DIR = REPO_ROOT / "skills"
RUNS_DIR = REPO_ROOT / "k1_runs"
TASKS_DIR = RUNS_DIR / "tasks"
OUTPUTS_DIR = RUNS_DIR / "outputs"
POLICIES_DIR = RUNS_DIR / "policies"

# --- interpreters ------------------------------------------------------------
# FRAMEWORK_02 fix #1: interpreter paths are overridable via environment
# variables and fall back through known locations instead of one
# hardcoded absolute path.
_CONDA_BOOSTER_TRAIN = Path("/home/liu/anaconda3/envs/booster_train/bin/python")
_PROJECT_VENV = REPO_ROOT / ".venv" / "bin" / "python"


def _first_existing(*paths: str | Path) -> Path | None:
    for p in paths:
        if p is None or str(p) == "":
            continue          # Path("") == Path(".") would false-positive
        q = Path(p)
        if q.exists():
            return q
    return None


def conda_python() -> str:
    """Interpreter with Isaac Lab (training / FK / deploy stack)."""
    found = _first_existing(
        os.environ.get("K1_SKILLS_TRAIN_PYTHON", ""),
        _CONDA_BOOSTER_TRAIN,
    )
    if found:
        return str(found)
    raise FileNotFoundError(
        "no Isaac Lab interpreter found; set K1_SKILLS_TRAIN_PYTHON "
        f"(default looked for: {_CONDA_BOOSTER_TRAIN})")


def deploy_python() -> str:
    """Interpreter for booster_deploy sim2sim (mujoco + torch + PIL)."""
    found = _first_existing(
        os.environ.get("K1_SKILLS_DEPLOY_PYTHON", ""),
        _CONDA_BOOSTER_TRAIN,
        _PROJECT_VENV,
    )
    if found:
        return str(found)
    return os.environ.get("K1_SKILLS_PYTHON", "python3")


# Backwards-compatible alias used by earlier drafts.
venv_python = deploy_python


def deploy_models_dir() -> Path:
    return BOOSTER_DEPLOY_DIR / "tasks" / "beyond_mimic" / "robots" / "k1" / "models"


def deploy_motions_dir() -> Path:
    return BOOSTER_DEPLOY_DIR / "tasks" / "beyond_mimic" / "robots" / "k1" / "motions"


def ensure_run_dirs() -> None:
    for d in (RUNS_DIR, TASKS_DIR, OUTPUTS_DIR, POLICIES_DIR):
        d.mkdir(parents=True, exist_ok=True)
