"""Skill Definition — schema v2 (K1_DANCE_SKILL_FRAMEWORK_02).

v2 generalizes the v1 (arm-wave-shaped) contract along four axes:

1. ``motion.kind``: ``cyclic`` (source + transition loop) or ``oneshot``
   (single pass, no transition frames, no cyclic npz).
2. ``control_profile.extends``: robot-level facts (model/dof/policy_dt/
   action-scale rule/joint groups) are inherited from a shared profile in
   ``skills/_profiles/``; skills declare only the policy-level values they
   were trained under (PD gains, effort limits, stepping).
3. ``runtime_profile.policies``: a role -> {deploy_task, policy_file}
   mapping instead of the fixed baseline/trained pair.
4. ``evaluation_profile``: selectable metric subset, per-skill joint
   groups, and custom metric entry points.

v1 definitions (no ``schema_version``) load through an automatic upgrade;
:func:`load_skill_definition` keeps the same call signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.paths import REPO_ROOT, SKILLS_DIR

SCHEMA_VERSION = 2

MOTION_KINDS = ("cyclic", "oneshot")
ROOT_MODES = ("stationary", "free")

_CYCLIC_METRICS = {"cycle_consistency"}
_BUILTIN_METRICS = (
    "joint_tracking_error", "arm_motion_amplitude", "arm_motion_velocity",
    "root_drift", "fall_rate", "foot_contact", "cycle_consistency",
)


def _require(mapping: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in mapping or mapping[k] in (None, "")]
    if missing:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: missing required fields: {', '.join(missing)}")


# ----------------------------------------------------------------------
# scalar type/range checks (Task C1, MINIMAL_FIX_01)
# NaN/Inf, bools-as-ints and numeric strings are all rejected; YAML
# `!!float .nan` / `.inf` parse to float('nan') / float('inf') and are
# caught by the finite check.
# ----------------------------------------------------------------------
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _isfinite(v) -> bool:
    import math
    return _is_num(v) and math.isfinite(float(v))


def _check_pos_int(mapping: dict, key: str, where: str, *, minimum: int = 1) -> None:
    v = mapping.get(key)
    if not _is_int(v):
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be an integer "
                         f"(got {type(v).__name__}: {v!r})")
    if v < minimum:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be >= {minimum} (got {v})")


def _check_pos_num(mapping: dict, key: str, where: str, *, minimum: float = 0.0,
                   exclusive: bool = True) -> None:
    import math
    v = mapping.get(key)
    if not _is_num(v):
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be a number "
                         f"(got {type(v).__name__}: {v!r})")
    if not math.isfinite(float(v)):
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be finite (got {v!r})")
    if exclusive and float(v) <= minimum:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be > {minimum} (got {v})")
    if not exclusive and float(v) < minimum:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: {key} must be >= {minimum} (got {v})")


# ----------------------------------------------------------------------
# v1 -> v2 upgrade
# ----------------------------------------------------------------------
def _upgrade_v1(raw: dict) -> dict:
    """Upgrade a FRAMEWORK_01 definition in memory (no file rewrite)."""
    ref = raw.get("reference_motion", {})
    cyclic = ref.get("deploy_npz_cyclic")
    kind = "cyclic" if cyclic else "oneshot"
    resources = {k: ref[k] for k in
                 ("dataset_trajectory", "dataset_metadata", "booster_csv",
                  "train_npz", "deploy_npz", "deploy_npz_cyclic") if k in ref}
    motion = {"kind": kind, "fps": ref.get("fps", 50),
              "source_frames": ref.get("source_frames"),
              "resources": resources}
    if kind == "cyclic":
        motion["transition_frames"] = ref.get("transition_frames", 0)

    rt = raw.get("runtime_profile", {})
    policies = {}
    if rt.get("baseline_task"):
        entry = {"deploy_task": rt["baseline_task"]}
        if raw.get("baseline", {}).get("policy"):
            entry["policy_file"] = Path(raw["baseline"]["policy"]).name
        policies["baseline"] = entry
    if rt.get("trained_task"):
        entry = {"deploy_task": rt["trained_task"]}
        if rt.get("trained_policy_file"):
            entry["policy_file"] = rt["trained_policy_file"]
        policies["trained"] = entry

    raw = dict(raw)
    raw["schema_version"] = 1  # marker: upgraded legacy
    raw["motion"] = motion
    raw["runtime_profile"] = {"policies": policies}
    raw.setdefault("evaluation_profile", {"metrics": list(_BUILTIN_METRICS)})
    return raw


@dataclass
class SkillDefinition:
    """Loaded, path-resolved, schema-v2 skill definition."""

    skill_id: str
    version: str
    description: str
    root_mode: str
    raw: dict[str, Any]
    yaml_path: Path
    motion: dict[str, Any] = field(default_factory=dict)
    robot: dict[str, Any] = field(default_factory=dict)
    control_profile: dict[str, Any] = field(default_factory=dict)
    observation_profile: dict[str, Any] = field(default_factory=dict)
    reward_profile: dict[str, Any] = field(default_factory=dict)
    training_profile: dict[str, Any] = field(default_factory=dict)
    runtime_profile: dict[str, Any] = field(default_factory=dict)
    evaluation_profile: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------
    @property
    def motion_kind(self) -> str:
        return str(self.motion["kind"])

    @property
    def is_cyclic(self) -> bool:
        return self.motion_kind == "cyclic"

    @property
    def reference_motion(self) -> dict:
        """v1-compatible flattened view (registry/motion v1 checks)."""
        res = self.motion.get("resources", {})
        out = dict(res)
        out["fps"] = self.motion["fps"]
        out["source_frames"] = self.motion.get("source_frames")
        if self.is_cyclic:
            out["transition_frames"] = self.motion.get("transition_frames", 0)
        return out

    def policy_roles(self) -> list[str]:
        return list(self.runtime_profile.get("policies", {}).keys())

    def policy_entry(self, role: str) -> dict:
        entry = self.runtime_profile.get("policies", {}).get(role)
        if not entry:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"skill '{self.skill_id}' has no policy role '{role}'; "
                             f"known roles: {self.policy_roles()}")
        return entry

    def artifact_name(self, kind: str) -> str:
        return str(self.raw.get("artifact_prefix", self.skill_id))

    def path(self, repo_relative: str | Path) -> Path:
        p = Path(repo_relative)
        return p if p.is_absolute() else REPO_ROOT / p

    # -- loading ---------------------------------------------------------
    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "SkillDefinition":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"skill yaml not found: {yaml_path}")
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"yaml parse error in {yaml_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{yaml_path}: top level must be a mapping")

        version = raw.get("schema_version", 1)
        if version == 1:
            raw = _upgrade_v1(raw)
        elif version != SCHEMA_VERSION:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{yaml_path}: unsupported schema_version {version} "
                             f"(supported: 1, {SCHEMA_VERSION})")
        # merge shared robot profile BEFORE validation so the merged
        # control profile is what gets checked and carried around
        if isinstance(raw.get("control_profile"), dict):
            from k1_skills.profiles import resolve_control_profile
            raw["control_profile"] = resolve_control_profile(raw["control_profile"])
        _validate(raw, yaml_path)

        return SkillDefinition(
            skill_id=str(raw["skill_id"]),
            version=str(raw["version"]),
            description=str(raw["description"]),
            root_mode=str(raw["root_mode"]),
            raw=raw,
            yaml_path=yaml_path,
            motion=dict(raw["motion"]),
            robot=dict(raw.get("robot", {})),
            control_profile=dict(raw["control_profile"]),
            observation_profile=dict(raw.get("observation_profile", {})),
            reward_profile=dict(raw.get("reward_profile", {})),
            training_profile=dict(raw["training_profile"]),
            runtime_profile=dict(raw["runtime_profile"]),
            evaluation_profile=dict(raw["evaluation_profile"]),
            baseline=dict(raw.get("baseline", {})),
        )


def _validate(raw: dict, yaml_path: Path) -> None:
    where = str(yaml_path)
    _require(raw, ["skill_id", "version", "description", "root_mode", "motion",
                   "control_profile", "training_profile", "runtime_profile",
                   "evaluation_profile"], where)
    if raw["root_mode"] not in ROOT_MODES:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: root_mode must be one of {ROOT_MODES}")

    # -- motion ----------------------------------------------------------
    motion = raw["motion"]
    _require(motion, ["kind", "fps", "source_frames", "resources"], f"{where}: motion")
    if motion["kind"] not in MOTION_KINDS:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: motion.kind must be one of {MOTION_KINDS}")
    _check_pos_num(motion, "fps", f"{where}: motion", minimum=0.0)
    _check_pos_int(motion, "source_frames", f"{where}: motion")
    if "transition_frames" in motion and motion["transition_frames"] is not None:
        _check_pos_int(motion, "transition_frames", f"{where}: motion", minimum=0)
    res = motion["resources"]
    _require(res, ["train_npz"], f"{where}: motion.resources")
    # resources entries: "path" string, or {path, joint_order, content_sha256}
    for role, entry in res.items():
        path = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(path, str) or not path:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{where}: motion.resources.{role} must be a path string "
                             f"or {{path, joint_order, content_sha256}} mapping")
        if isinstance(entry, dict):
            ref = entry.get("joint_order")
            if ref is not None and ref not in res:
                raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                 f"{where}: motion.resources.{role}.joint_order "
                                 f"references unknown resource '{ref}'")
            csha = entry.get("content_sha256")
            if csha is not None:
                hexdigits = set("0123456789abcdefABCDEF")
                if (not isinstance(csha, str) or len(csha) != 64
                        or not all(ch in hexdigits for ch in csha)):
                    raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                     f"{where}: motion.resources.{role}."
                                     f"content_sha256 must be a full 64-hex "
                                     f"SHA256 digest (got {csha!r})")
    if motion["kind"] == "cyclic":
        _require(res, ["deploy_npz_cyclic"], f"{where}: motion.resources (cyclic)")
    else:
        _require(res, ["deploy_npz"], f"{where}: motion.resources (oneshot)")

    # -- control (post-merge; merging happens in profiles.resolve) --------
    ctrl = raw["control_profile"]
    _require(ctrl, ["physics_dt", "decimation", "policy_frequency_hz",
                    "joint_stiffness", "joint_damping", "effort_limit"],
             f"{where}: control_profile")
    _check_pos_num(ctrl, "physics_dt", f"{where}: control_profile")
    _check_pos_int(ctrl, "decimation", f"{where}: control_profile")
    _check_pos_num(ctrl, "policy_frequency_hz", f"{where}: control_profile", minimum=0.0)
    for key in ("joint_stiffness", "joint_damping", "effort_limit"):
        values = ctrl[key]
        if not isinstance(values, list):
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{where}: control_profile.{key} must be a list "
                             f"(got {type(values).__name__})")
        if len(values) != 22:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{where}: control_profile.{key} must have 22 entries "
                             f"(got {len(values)})")
        # PD semantics: stiffness>0 (action-scale divisor), damping>=0,
        # effort_limit>0 (torque envelope)
        for i, v in enumerate(values):
            if not _is_num(v) or not _isfinite(v):
                raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                 f"{where}: control_profile.{key}[{i}] must be a "
                                 f"finite number (got {type(v).__name__}: {v!r})")
            lower_ok = (v > 0.0 if key in ("joint_stiffness", "effort_limit")
                        else v >= 0.0)
            if not lower_ok:
                raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                 f"{where}: control_profile.{key}[{i}] out of range "
                                 f"({v})")
    if abs(float(ctrl["physics_dt"]) * int(ctrl["decimation"])
           - 1.0 / float(ctrl["policy_frequency_hz"])) > 1e-9:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: physics_dt*decimation != 1/policy_frequency_hz")

    # -- training ---------------------------------------------------------
    trn = raw["training_profile"]
    _require(trn, ["framework", "task_id", "play_task_id", "num_envs",
                   "max_iterations", "seed"], f"{where}: training_profile")
    _check_pos_int(trn, "num_envs", f"{where}: training_profile")
    _check_pos_int(trn, "max_iterations", f"{where}: training_profile")
    _check_pos_int(trn, "seed", f"{where}: training_profile", minimum=0)

    # -- runtime ------------------------------------------------------------
    policies = raw["runtime_profile"].get("policies")
    if not isinstance(policies, dict) or not policies:
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: runtime_profile.policies must map role -> "
                         "{{deploy_task[, policy_file]}} (at least one role)")
    for role, entry in policies.items():
        if not isinstance(entry, dict) or not entry.get("deploy_task"):
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{where}: runtime_profile.policies.{role} needs deploy_task")

    # -- evaluation -----------------------------------------------------------
    ev = raw["evaluation_profile"]
    metrics = ev.get("metrics")
    if not metrics or not isinstance(metrics, list):
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         f"{where}: evaluation_profile.metrics must be a non-empty list")


def load_skill_definition(skill_dir_or_yaml: str | Path) -> SkillDefinition:
    p = Path(skill_dir_or_yaml)
    if p.is_dir():
        p = p / "skill.yaml"
    return SkillDefinition.from_yaml(p)


def validate_effective_training_config(effective: dict) -> None:
    """Validate the FINAL effective training configuration.

    REMEDIATION_01 Task C: the processing order is
        load YAML -> profile merge -> CLI overrides -> THIS VALIDATION -> train.
    Overrides therefore cannot bypass the schema: the same type/range
    rules that gate the YAML are applied to the merged result.

    Expected keys: num_envs, max_iterations, seed, and (optionally) a
    ``control`` mapping with physics_dt / decimation / policy_frequency_hz /
    gain arrays — the stepping chain training runs with.
    """
    where = "effective training config"
    _require(effective, ["num_envs", "max_iterations", "seed"], where)
    _check_pos_int(effective, "num_envs", where)
    _check_pos_int(effective, "max_iterations", where)
    _check_pos_int(effective, "seed", where, minimum=0)
    ctrl = effective.get("control")
    if ctrl is not None:
        _require(ctrl, ["physics_dt", "decimation", "policy_frequency_hz"],
                 f"{where}: control")
        _check_pos_num(ctrl, "physics_dt", f"{where}: control")
        _check_pos_int(ctrl, "decimation", f"{where}: control")
        _check_pos_num(ctrl, "policy_frequency_hz", f"{where}: control", minimum=0.0)
        for key in ("joint_stiffness", "joint_damping", "effort_limit"):
            if key not in ctrl:
                continue
            values = ctrl[key]
            if not isinstance(values, list) or len(values) != 22:
                raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                 f"{where}: control.{key} must be a 22-entry list")
            for i, v in enumerate(values):
                if not _isfinite(v):
                    raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                     f"{where}: control.{key}[{i}] must be a "
                                     f"finite number (got {v!r})")
        if abs(float(ctrl["physics_dt"]) * int(ctrl["decimation"])
               - 1.0 / float(ctrl["policy_frequency_hz"])) > 1e-9:
            raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                             f"{where}: control physics_dt*decimation != "
                             f"1/policy_frequency_hz")
