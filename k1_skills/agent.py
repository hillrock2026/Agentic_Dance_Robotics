"""Agent Integration Interface: schema-annotated tools over the framework.

This is the Python-side tool adapter the future Agent-Kernel integration
will call (directly now, via a service protocol later — see
VLN_AGENT_KERNEL_INDEPENDENT_REVIEW_01 §11 / AK-M1). Contract:

- every tool has an input JSON Schema, an output JSON Schema and stable
  error codes (k1_skills.errors.SkillErrorCode);
- every call returns an envelope ``{"ok": true, "result": ...}`` or
  ``{"ok": false, "error": {"code", "message", "details", "retryable"}}``;
- long-running work (rl.train) is asynchronous: launch returns a task_id;
  rl.status / rl.result poll the shared task registry.

The tools are callable independently of any agent:

    python -m k1_skills tool skill.validate '{"skill_id": "arm_wave"}'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.evaluation import compare, evaluate
from k1_skills.learning import export as export_policy
from k1_skills.learning import prepare as prepare_motion
from k1_skills.learning import (latest_checkpoint, latest_run_dir,
                                train_launch, train_status)
from k1_skills.registry import SkillRegistry
from k1_skills.runtime import run_sim2sim
from k1_skills import tasks as taskmod


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    handler: Callable[[dict], Any]
    long_running: bool = False
    examples: list[dict] = field(default_factory=list)


def _s(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props,
            "required": required, "additionalProperties": False}


_RESULT_OK = {"type": "object",
              "properties": {"ok": {"const": True}, "result": {"type": "object"}},
              "required": ["ok", "result"]}


# ----------------------------------------------------------------------
# handlers
# ----------------------------------------------------------------------
def _tool_motion_process(args: dict) -> dict:
    registry = SkillRegistry()
    definition = registry.get(args["skill_id"])
    if args.get("action") == "inspect":
        from k1_skills.motion import MotionResourceSet
        try:
            mrs = MotionResourceSet.from_spec(definition)
            info = mrs.describe()
        except SkillError as exc:
            info = {"error": exc.to_dict()}
        return info
    return prepare_motion(definition, force=bool(args.get("force", False)))


def _tool_skill_register(args: dict) -> dict:
    """Register a new skill by writing its skill.yaml (dry-run validated).

    The definition dict must satisfy the SkillDefinition contract; it is
    written to skills/<skill_id>/skill.yaml and immediately validated.
    """
    skill_id = args.get("skill_id", "")
    definition = args.get("definition")
    if not skill_id or not isinstance(definition, dict):
        raise SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                         "skill.register requires skill_id and definition dict")
    import yaml
    from k1_skills.paths import SKILLS_DIR
    target = SKILLS_DIR / skill_id / "skill.yaml"
    if target.exists() and not args.get("overwrite", False):
        raise SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                         f"{target} already exists (pass overwrite=true to replace)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
    report = SkillRegistry().validate(skill_id)
    if not report["valid"]:
        target.unlink(missing_ok=True)
        raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                         "definition failed validation; not registered",
                         {"problems": report["problems"]})
    return {"skill_id": skill_id, "yaml": str(target), "validation": report}


def _tool_skill_validate(args: dict) -> dict:
    return SkillRegistry().validate(args["skill_id"])


def _tool_rl_train(args: dict) -> dict:
    registry = SkillRegistry()
    definition = registry.get(args["skill_id"])
    state = train_launch(
        definition, background=True,
        num_envs=args.get("num_envs"),
        max_iterations=args.get("max_iterations"))
    return {"task_id": state["task_id"], "status": state["status"],
            "log": state["log"], "max_iterations": state.get("max_iterations"),
            "note": "poll rl.status with task_id (or list_recent)"}


def _tool_rl_status(args: dict) -> dict:
    # FRAMEWORK_02 fix #2: list_recent is honoured, and a missing task_id
    # is an invalid_arguments error (was a tool_internal_error KeyError).
    if not args.get("task_id") and not args.get("list_recent"):
        raise SkillError(
            SkillErrorCode.INVALID_ARGUMENTS,
            "rl.status requires task_id, or pass list_recent=true")
    result = train_status(args.get("task_id"),
                          list_recent=bool(args.get("list_recent")))
    return result if not isinstance(result, dict) else result


def _tool_policy_export(args: dict) -> dict:
    registry = SkillRegistry()
    definition = registry.get(args["skill_id"])
    run_dir = Path(args["run_dir"]) if args.get("run_dir") else None
    state = export_policy(definition, run_dir=run_dir)
    return {"task_id": state["task_id"], "status": state["status"],
            "artifacts": state["artifacts"]}


def _tool_mujoco_sim2sim(args: dict) -> dict:
    registry = SkillRegistry()
    definition = registry.get(args["skill_id"])
    return run_sim2sim(definition,
                       policy=args.get("policy", "trained"),
                       cycles=int(args.get("cycles", 3)),
                       record_video=bool(args.get("record_video", True)))


def _tool_eval_analyze(args: dict) -> dict:
    def _ctx(skill_id):
        registry = SkillRegistry()
        definition = registry.get(skill_id)
        from k1_skills.profiles import joint_groups
        groups = joint_groups(definition.evaluation_profile, definition.control_profile)
        return definition.motion_kind, groups

    if args.get("baseline_metrics") and args.get("candidate_metrics"):
        skill_id = args.get("skill_id")
        kind, groups = _ctx(skill_id) if skill_id else ("cyclic", None)
        a = evaluate(args["baseline_metrics"], groups=groups, motion_kind=kind)
        b = evaluate(args["candidate_metrics"], groups=groups, motion_kind=kind)
        return {"baseline": a, "candidate": b,
                "comparison": compare(a, b, label_a="baseline", label_b="candidate")}
    kind, groups = (None, None)
    if args.get("skill_id"):
        kind, groups = _ctx(args["skill_id"])
    result = evaluate(args["metrics_file"], metrics=args.get("metrics"),
                      groups=groups, motion_kind=kind or "cyclic")
    return {"evaluation": result}


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
TOOLS: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


_register(ToolSpec(
    name="motion.process",
    description="Prepare (csv->train npz via Isaac Lab FK) or inspect a skill's reference motion.",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "action": {"type": "string", "enum": ["prepare", "inspect"], "default": "prepare"},
         "force": {"type": "boolean", "default": False}},
        ["skill_id"]),
    output_schema=_RESULT_OK,
    handler=_tool_motion_process,
    examples=[{"skill_id": "arm_wave", "action": "prepare"}],
))

_register(ToolSpec(
    name="skill.register",
    description="Register a new dance skill from a definition dict (written as skills/<id>/skill.yaml after validation).",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "definition": {"type": "object"},
         "overwrite": {"type": "boolean", "default": False}},
        ["skill_id", "definition"]),
    output_schema=_RESULT_OK,
    handler=_tool_skill_register,
))

_register(ToolSpec(
    name="skill.validate",
    description="Validate a skill's definition, motion assets, and control-profile consistency (training vs runtime).",
    input_schema=_s({"skill_id": {"type": "string"}}, ["skill_id"]),
    output_schema=_RESULT_OK,
    handler=_tool_skill_validate,
    examples=[{"skill_id": "arm_wave"}],
))

_register(ToolSpec(
    name="rl.train",
    description="Launch RL training for a skill (async). Returns task_id; poll with rl.status.",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "num_envs": {"type": "integer", "minimum": 1,
                      "description": "launch-time override of training_profile.num_envs"},
         "max_iterations": {"type": "integer", "minimum": 1,
                            "description": "launch-time override (e.g. smoke runs)"}},
        ["skill_id"]),
    output_schema=_RESULT_OK,
    handler=_tool_rl_train,
    long_running=True,
    examples=[{"skill_id": "arm_wave"}],
))

_register(ToolSpec(
    name="rl.status",
    description="Query training task status/progress by task_id, or list recent rl.train tasks.",
    input_schema=_s(
        {"task_id": {"type": "string"},
         "list_recent": {"type": "boolean", "default": False,
                         "description": "return the 10 most recent rl.train tasks"}},
        []),
    output_schema=_RESULT_OK,
    handler=_tool_rl_status,
))

_register(ToolSpec(
    name="policy.export",
    description="Export latest (or given run_dir) checkpoint to jit+onnx and stage jit for sim2sim.",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "run_dir": {"type": "string", "description": "optional booster_train run dir"}},
        ["skill_id"]),
    output_schema=_RESULT_OK,
    handler=_tool_policy_export,
))

_register(ToolSpec(
    name="mujoco.sim2sim",
    description="Run N cycles of the skill in MuJoCo sim2sim with baseline or trained policy; returns metrics path.",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "policy": {"type": "string", "enum": ["baseline", "trained"], "default": "trained"},
         "cycles": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
         "record_video": {"type": "boolean", "default": True}},
        ["skill_id"]),
    output_schema=_RESULT_OK,
    handler=_tool_mujoco_sim2sim,
    examples=[{"skill_id": "arm_wave", "policy": "baseline", "cycles": 3}],
))

_register(ToolSpec(
    name="eval.analyze",
    description="Evaluate a sim2sim metrics file, or compare baseline vs candidate metrics. "
                "Pass skill_id to use the skill's joint groups and motion kind.",
    input_schema=_s(
        {"skill_id": {"type": "string"},
         "metrics_file": {"type": "string"},
         "baseline_metrics": {"type": "string"},
         "candidate_metrics": {"type": "string"},
         "metrics": {"type": "array", "items": {"type": "string"}}},
        []),
    output_schema=_RESULT_OK,
    handler=_tool_eval_analyze,
))


def list_tools() -> list[dict]:
    return [{"name": s.name, "description": s.description,
             "long_running": s.long_running,
             "input_schema": s.input_schema,
             "output_schema": s.output_schema}
            for s in TOOLS.values()]


def call_tool(name: str, arguments: dict | str | None = None) -> dict:
    """Invoke a tool; always returns the envelope (never raises)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return _error(SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                                     f"arguments must be valid JSON: {exc}"))
    arguments = dict(arguments or {})
    spec = TOOLS.get(name)
    if spec is None:
        return _error(SkillError(SkillErrorCode.UNKNOWN_TOOL,
                                 f"unknown tool '{name}'; known: {sorted(TOOLS)}"))
    try:
        _validate_arguments(spec, arguments)
        result = spec.handler(arguments)
        return {"ok": True, "result": result}
    except SkillError as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001 - tool boundary guarantees envelope
        return _error(SkillError(SkillErrorCode.TOOL_INTERNAL_ERROR,
                                 f"{type(exc).__name__}: {exc}"))


def _validate_arguments(spec: ToolSpec, args: dict) -> None:
    required = spec.input_schema.get("required", [])
    missing = [k for k in required if k not in args or args[k] in (None, "")]
    if missing:
        raise SkillError(SkillErrorCode.INVALID_ARGUMENTS,
                         f"missing required argument(s): {', '.join(missing)}",
                         {"tool": spec.name})


def _error(exc: SkillError) -> dict:
    payload = exc.to_dict()
    payload["retryable"] = exc.code.retryable
    return {"ok": False, "error": payload}
