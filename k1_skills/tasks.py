"""Long-running task management for the framework and the agent tools.

Task state lives in one JSON file per task under ``k1_runs/tasks/`` so
that (a) the CLI can background training, (b) ``rl.status`` can poll
without the launching process alive, and (c) a future Agent-Kernel
capability provider sees the same task-id / state / artifacts contract
this framework uses internally.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.paths import TASKS_DIR, ensure_run_dirs

TERMINAL_STATES = ("succeeded", "failed", "cancelled")


def new_task_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def create_task(task_id: str, kind: str, command: list[str], extra: dict | None = None) -> dict:
    ensure_run_dirs()
    extra = extra or {}
    state = {
        "task_id": task_id,
        "kind": kind,                     # e.g. rl.train, mujoco.sim2sim
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time(),
        "command": command,
        "pid": extra.get("pid"),
        "log": extra.get("log"),
        "artifacts": extra.get("artifacts", {}),
        "skill_id": extra.get("skill_id"),
    }
    # passthrough of informational fields (e.g. max_iterations, run_dir)
    for key in ("max_iterations", "run_dir", "checkpoint", "policy_role",
                "deploy_task", "cycles"):
        if key in extra:
            state[key] = extra[key]
    state.update(extra.get("fields", {}))
    _write(task_id, state)
    return state


def update_task(task_id: str, **fields) -> dict:
    state = get_task(task_id)
    state.update(fields)
    state["updated_at"] = time.time()
    _write(task_id, state)
    return state


def get_task(task_id: str) -> dict:
    p = task_path(task_id)
    if not p.exists():
        raise SkillError(SkillErrorCode.TASK_NOT_FOUND, f"no task '{task_id}' under {TASKS_DIR}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_tasks(kind: str | None = None) -> list[dict]:
    ensure_run_dirs()
    out = []
    for p in sorted(TASKS_DIR.glob("*.json")):
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if kind is None or state.get("kind") == kind:
            out.append(state)
    return out


def mark_from_exit_code(task_id: str, exit_code: int, artifacts: dict | None = None) -> dict:
    status = "succeeded" if exit_code == 0 else "failed"
    fields = {"status": status, "exit_code": exit_code}
    if artifacts:
        fields["artifacts"] = artifacts
    return update_task(task_id, **fields)


_ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")


def latest_training_iteration(log_file: str | Path) -> int | None:
    """Best-effort progress extraction from a booster_train stdout log."""
    p = Path(log_file)
    if not p.exists():
        return None
    last = None
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _ITER_RE.search(line)
                if m:
                    last = int(m.group(1))
    except OSError:
        return None
    return last


def _write(task_id: str, state: dict) -> None:
    task_path(task_id).write_text(json.dumps(state, indent=2), encoding="utf-8")
