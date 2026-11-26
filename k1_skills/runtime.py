"""Policy Runtime: run a skill's policy against its reference motion in
MuJoCo sim2sim via the existing booster_deploy recorder.

FRAMEWORK_02 generalizations:

- policy roles come from the skill's ``runtime_profile.policies`` mapping
  (any role name; ``baseline`` / ``trained`` are conventions, not laws);
- oneshot motions (``motion.kind: oneshot``) run exactly one pass
  (``--cycles 1``), cyclic motions pass ``--source-frames`` from the
  skill definition;
- the runtime contract is checked: a role whose staged jit drifted from
  its export manifest is reported (hash mismatch) instead of silently
  running an untracked artifact.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.learning import check_staged_policy
from k1_skills.paths import BOOSTER_DEPLOY_DIR, OUTPUTS_DIR, ensure_run_dirs, deploy_python
from k1_skills.spec import SkillDefinition
from k1_skills import tasks as taskmod


def run_sim2sim(definition: SkillDefinition, *, policy: str = "trained",
                cycles: int | None = None, task_id: str | None = None,
                record_video: bool = True) -> dict:
    ensure_run_dirs()
    task_id = task_id or taskmod.new_task_id("sim2sim")

    entry = definition.policy_entry(policy)   # raises for unknown roles
    deploy_task = entry["deploy_task"]

    if definition.is_cyclic:
        cycles = int(cycles or 3)
    else:
        if cycles not in (None, 1):
            raise SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                             f"skill '{definition.skill_id}' is oneshot; "
                             "cycles must be 1 (default)")
        cycles = 1

    # runtime contract: staged policy must match its export manifest
    drift = None
    if entry.get("policy_file"):
        drift = check_staged_policy(definition, policy)
        if drift.get("ok") is False:
            raise SkillError(
                SkillErrorCode.RUN_FAILED,
                f"staged policy for role '{policy}' does not match its export "
                f"manifest ({drift.get('staged')}); re-export or restore the "
                "tracked artifact", details=drift)

    out_dir = OUTPUTS_DIR / definition.skill_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{definition.artifact_name('run')}_{policy}_c{cycles}_{task_id}"
    video = stem.with_suffix(".mp4")
    metrics = stem.with_suffix(".metrics.npz")

    cmd = [
        deploy_python(), "scripts/deploy_mujoco_record_cyclic.py",
        "--task", deploy_task,
        "--output", str(video),
        "--cycles", str(cycles),
    ]
    if definition.is_cyclic and definition.motion.get("source_frames"):
        cmd += ["--source-frames", str(int(definition.motion["source_frames"]))]
    if not record_video:
        cmd.append("--no-video")

    taskmod.create_task(task_id, "mujoco.sim2sim", cmd, {
        "skill_id": definition.skill_id, "policy_role": policy,
        "deploy_task": deploy_task, "cycles": cycles})

    import os
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    log_file = stem.with_suffix(".log")
    with log_file.open("w", encoding="utf-8") as log:
        code = subprocess.Popen(cmd, cwd=str(BOOSTER_DEPLOY_DIR), stdout=log,
                                stderr=subprocess.STDOUT, env=env).wait()

    artifacts = {
        "video": str(video) if video.exists() else None,
        "metrics": str(metrics) if metrics.exists() else None,
        "log": str(log_file),
    }
    taskmod.mark_from_exit_code(task_id, code, artifacts)
    if code != 0 or not metrics.exists():
        raise SkillError(SkillErrorCode.RUN_FAILED,
                         f"sim2sim run failed (exit {code}); see {log_file}")
    return {
        "task_id": task_id,
        "skill_id": definition.skill_id,
        "motion_kind": definition.motion_kind,
        "policy_role": policy,
        "deploy_task": deploy_task,
        "cycles": cycles,
        "video": artifacts["video"],
        "metrics": artifacts["metrics"],
        "log": artifacts["log"],
        "policy_contract": drift,
    }
