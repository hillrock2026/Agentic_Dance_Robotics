"""Robot Interface: the K1 22-DOF control profile as one shared source.

The authoritative numbers live in the skill YAML (declared once per
robot family). ``check_against_deploy`` cross-verifies them against the
booster_deploy task config so training and sim2sim runtime provably use
the same gains/limits/stepping — acceptance requirement #3 of
K1_DANCE_SKILL_FRAMEWORK_01.

CSV joint order (deploy convention, shared with the recorder metrics):

    head yaw, head pitch,                       # 0:2
    left arm (shoulder p/r, elbow p/yaw),       # 2:6
    right arm,                                  # 6:10
    left leg (hip p/r/y, knee, ankle p/r),      # 10:16
    right leg                                   # 16:22
"""

from __future__ import annotations

from pathlib import Path

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.paths import BOOSTER_DEPLOY_DIR

ARM_SLICE = slice(2, 10)
LEG_SLICE = slice(10, 22)
HEAD_SLICE = slice(0, 2)

GROUPS = {
    "head": HEAD_SLICE,
    "arms": ARM_SLICE,
    "legs": LEG_SLICE,
}


def control_summary(control: dict) -> dict:
    """Human/agent-readable summary of the declared control profile."""
    return {
        "model": control["model"],
        "dof": control["dof"],
        "physics_dt": control["physics_dt"],
        "decimation": control["decimation"],
        "policy_frequency_hz": control["policy_frequency_hz"],
        "action_scale_rule": control["action_scale_rule"],
        "pd_gains_source": control["pd_gains_source"],
        "torque_limit_source": control.get("torque_limit_source", "skill.yaml"),
        "policy_dt": control["physics_dt"] * control["decimation"],
    }


def check_step_consistency(control: dict, declared_fps: float) -> list[str]:
    """Training stepping vs motion fps vs policy frequency must agree."""
    problems = []
    policy_dt = control["physics_dt"] * control["decimation"]
    implied_hz = 1.0 / policy_dt
    if abs(implied_hz - control["policy_frequency_hz"]) > 1e-6:
        problems.append(
            f"physics_dt*decimation implies {implied_hz} Hz but policy_frequency "
            f"declared {control['policy_frequency_hz']} Hz")
    if abs(declared_fps - control["policy_frequency_hz"]) > 1e-6:
        problems.append(
            f"motion fps {declared_fps} != policy frequency {control['policy_frequency_hz']} "
            "(one ref frame per policy step is assumed)")
    return problems


def _load_deploy_registry():
    """Import booster_deploy the same way its own scripts do.

    sys.path gets the booster_deploy repo root; ``import tasks`` plus a
    pkgutil walk registers every task config into _TASK_REGISTRY.
    """
    import importlib
    import pkgutil
    import sys

    root = str(BOOSTER_DEPLOY_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    import tasks as tasks_pkg  # noqa: F401  (side-effect: registrations)

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)
    from booster_deploy.utils.registry import _TASK_REGISTRY  # type: ignore
    return _TASK_REGISTRY


def canonical_joint_orders() -> dict[str, list[str]]:
    """The single authoritative source of K1's two legitimate joint orders.

    Both come from the deploy robot config (K1_CFG) — no parallel list is
    maintained anywhere in k1_skills:

    - ``npz``  : sim/tree order — the order inside every motion npz
      (train/deploy/cyclic);
    - ``csv``  : CSV order — deploy commands, recorder metrics and the
      evaluation joint groups.

    Integrity is enforced BEFORE the orders are trusted (FINAL_REMEDIATION_01
    Task D): exactly 22 names in each list, all names UNIQUE within each
    list (a duplicated name with a compensating missing one would pass a
    naive sorted-equality check), and both lists the same joint set.
    Malformed or unloadable canonical -> fail-closed.
    """
    import sys
    if str(BOOSTER_DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(BOOSTER_DEPLOY_DIR))
    from booster_deploy.robots.k1 import K1_CFG
    npz = list(K1_CFG.sim_joint_names)
    csv = list(K1_CFG.joint_names)

    def _check(names: list[str], label: str) -> None:
        if len(names) != 22:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"canonical {label} order must have exactly 22 "
                             f"joints (got {len(names)})")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"canonical {label} order has duplicate names: "
                             f"{dupes}")
        if len(set(names)) != 22:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"canonical {label} order names not unique")

    _check(npz, "npz")
    _check(csv, "csv")
    if sorted(npz) != sorted(csv):
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         "canonical npz and csv orders are not the same "
                         "22-joint set")
    return {"npz": npz, "csv": csv}


def check_against_deploy(control: dict, deploy_task: str) -> list[str]:
    """Compare declared PD/effort/decimation with the deploy task config.

    Runs inside the project venv (mujoco available). Returns a list of
    discrepancies; empty means training/runtime agree.
    """
    try:
        cfg = _load_deploy_registry()[deploy_task]
    except Exception as exc:  # noqa: BLE001 - degrade to explicit error
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"cannot load deploy task '{deploy_task}': {exc}") from exc

    problems: list[str] = []
    robot = getattr(cfg, "robot", None)

    for name, declared in (
        ("joint_stiffness", control["joint_stiffness"]),
        ("joint_damping", control["joint_damping"]),
        ("effort_limit", control["effort_limit"]),
    ):
        got = list(getattr(robot, name, []) or [])
        if len(got) != len(declared):
            problems.append(f"{name}: deploy has {len(got)} values, skill declares {len(declared)}")
            continue
        for i, (a, b) in enumerate(zip(got, declared)):
            if abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(b))):
                problems.append(f"{name}[{i}]: deploy {a} != skill {b}")
                break

    muj = getattr(cfg, "mujoco", None)
    if muj is not None:
        dec = getattr(muj, "decimation", None)
        if dec is not None and int(dec) != int(control["decimation"]):
            problems.append(f"mujoco.decimation: deploy {dec} != skill {control['decimation']}")
        mdt = getattr(muj, "physics_dt", None)
        if mdt is not None and abs(float(mdt) - float(control["physics_dt"])) > 1e-12:
            problems.append(f"mujoco.physics_dt: deploy {mdt} != skill {control['physics_dt']}")
        pdt = getattr(cfg, "policy_dt", None)
        if pdt is not None and mdt is not None:
            implied = float(mdt) * int(dec or 1)
            if abs(implied - float(pdt)) > 1e-9:
                problems.append(
                    f"deploy stepping inconsistent: physics_dt*decimation={implied} != "
                    f"policy_dt={pdt} (time-scale mismatch)")
    return problems


def deploy_task_exists(deploy_task: str) -> bool:
    try:
        return deploy_task in _load_deploy_registry()
    except Exception:  # noqa: BLE001 - best-effort probe
        return False


def motion_path_for_task(deploy_task: str) -> Path | None:
    try:
        cfg = _load_deploy_registry()[deploy_task]
        mp = str(getattr(cfg.policy, "motion_path", ""))
        if mp:
            p = Path(mp)
            return p if p.is_absolute() else BOOSTER_DEPLOY_DIR / "tasks" / "beyond_mimic" / p
    except Exception:  # noqa: BLE001
        pass
    return None
