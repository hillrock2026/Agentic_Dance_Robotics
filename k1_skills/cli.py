"""Unified command interface: ``python -m k1_skills <command>``.

FRAMEWORK_02 additions: launch-time training overrides, the ``policies``
command (policy artifact manifests), per-skill evaluation context
(joint groups / motion kind) wired into evaluate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from k1_skills import agent
from k1_skills.errors import SkillError
from k1_skills.evaluation import compare, evaluate
from k1_skills.learning import (check_staged_policy, export as export_policy,
                                prepare as prepare_motion)
from k1_skills.learning import train_launch, train_status
from k1_skills.profiles import joint_groups
from k1_skills.registry import SkillRegistry
from k1_skills.runtime import run_sim2sim


def _print_json(obj) -> None:
    # strict JSON: any NaN/Inf escaping the evaluation layer is a bug and
    # must fail loudly here rather than emit a non-standard literal
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str,
                     allow_nan=False))


def _eval_context(args):
    """Resolve (motion_kind, joint_groups) for a skill when available."""
    if not getattr(args, "skill_id", None):
        return "cyclic", None
    definition = SkillRegistry().get(args.skill_id)
    groups = joint_groups(definition.evaluation_profile, definition.control_profile)
    return definition.motion_kind, groups


# ----------------------------------------------------------------------
def cmd_list(_args) -> int:
    _print_json(SkillRegistry().list())
    return 0


def cmd_validate(args) -> int:
    report = SkillRegistry().validate(args.skill_id)
    _print_json(report)
    return 0 if report["valid"] else 1


def cmd_prepare(args) -> int:
    definition = SkillRegistry().get(args.skill_id)
    _print_json(prepare_motion(definition, force=args.force))
    return 0


def cmd_train(args) -> int:
    definition = SkillRegistry().get(args.skill_id)
    if args.background:
        state = train_launch(definition, background=True,
                             num_envs=args.num_envs,
                             max_iterations=args.max_iterations)
        _print_json(state)
        return 0
    state = train_launch(definition, background=False,
                         num_envs=args.num_envs,
                         max_iterations=args.max_iterations)
    _print_json(state)
    return 0 if state["status"] == "succeeded" else 1


def cmd_train_status(args) -> int:
    state = train_status(args.task_id, list_recent=args.list_recent)
    _print_json(state)
    return 0


def cmd_export(args) -> int:
    definition = SkillRegistry().get(args.skill_id)
    run_dir = Path(args.run_dir) if args.run_dir else None
    state = export_policy(definition, run_dir=run_dir)
    _print_json(state)
    return 0 if state["status"] == "succeeded" else 1


def cmd_policies(args) -> int:
    if args.skill_id:
        definition = SkillRegistry().get(args.skill_id)
        _print_json(check_staged_policy(definition, args.role))
        return 0 if check_staged_policy(definition, args.role).get("ok") is not False else 1
    from k1_skills.paths import POLICIES_DIR
    out = []
    if POLICIES_DIR.exists():
        for skill_dir in sorted(POLICIES_DIR.iterdir()):
            latest = skill_dir / "latest.json"
            if latest.exists():
                manifest = json.loads(latest.read_text(encoding="utf-8"))
                out.append({
                    "skill_id": skill_dir.name,
                    "role": manifest.get("role"),
                    "staged_file": manifest.get("staged_file"),
                    "staged_sha256": (manifest.get("staged_sha256") or "")[:16],
                    "version_dir": manifest.get("version_dir"),
                    "created_at": manifest.get("created_at"),
                })
    _print_json(out)
    return 0


def cmd_run(args) -> int:
    definition = SkillRegistry().get(args.skill_id)
    result = run_sim2sim(definition, policy=args.policy,
                         cycles=args.cycles if args.cycles else None,
                         record_video=not args.no_video)
    _print_json(result)
    return 0


def cmd_evaluate(args) -> int:
    kind, groups = _eval_context(args)
    if args.baseline and args.candidate:
        a = evaluate(args.baseline, groups=groups, motion_kind=kind)
        b = evaluate(args.candidate, groups=groups, motion_kind=kind)
        _print_json({"baseline": a, "candidate": b,
                     "comparison": compare(a, b, label_a="baseline",
                                           label_b="candidate")})
        return 0
    result = evaluate(args.metrics_file, groups=groups, motion_kind=kind)
    _print_json(result)
    return 0


def cmd_tool(args) -> int:
    if args.tool_name == "list" or not args.tool_name:
        _print_json(agent.list_tools())
        return 0
    envelope = agent.call_tool(args.tool_name, args.args_json)
    _print_json(envelope)
    if not envelope.get("ok"):
        return 2
    return 0


def cmd_tasks(args) -> int:
    from k1_skills import tasks as taskmod
    _print_json(taskmod.list_tasks(kind=args.kind))
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="k1_skills",
                                description="K1 Dance Skill Framework")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered skills").set_defaults(func=cmd_list)

    sp = sub.add_parser("validate", help="validate a skill's definition and resources")
    sp.add_argument("skill_id")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("prepare", help="build training npz from the official CSV")
    sp.add_argument("skill_id")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="launch RL training")
    sp.add_argument("skill_id")
    sp.add_argument("--background", action="store_true",
                    help="launch async and return a task id")
    sp.add_argument("--num-envs", type=int, default=None,
                    help="launch-time override of training_profile.num_envs")
    sp.add_argument("--max-iterations", type=int, default=None,
                    help="launch-time override (e.g. smoke runs)")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("train-status", help="query a training task / list recent")
    sp.add_argument("task_id", nargs="?", default=None)
    sp.add_argument("--list-recent", action="store_true")
    sp.set_defaults(func=cmd_train_status)

    sp = sub.add_parser("export", help="export jit+onnx, archive versioned policy, stage for sim2sim")
    sp.add_argument("skill_id")
    sp.add_argument("--run-dir", default=None)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("policies", help="list policy manifests / check staged policy")
    sp.add_argument("skill_id", nargs="?", default=None)
    sp.add_argument("--role", default="trained")
    sp.set_defaults(func=cmd_policies)

    sp = sub.add_parser("run", help="MuJoCo sim2sim run (cyclic: N cycles; oneshot: single pass)")
    sp.add_argument("skill_id")
    sp.add_argument("--policy", default="trained",
                    help="role from runtime_profile.policies (e.g. baseline, trained)")
    sp.add_argument("--cycles", type=int, default=None)
    sp.add_argument("--no-video", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("evaluate", help="evaluate metrics / compare two runs")
    sp.add_argument("metrics_file", nargs="?", default=None)
    sp.add_argument("--skill-id", default=None,
                    help="use the skill's joint groups and motion kind")
    sp.add_argument("--baseline", default=None, help="baseline metrics.npz")
    sp.add_argument("--candidate", default=None, help="candidate metrics.npz")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("tool", help="call an agent tool (or 'list')")
    sp.add_argument("tool_name", nargs="?", default=None)
    sp.add_argument("args_json", nargs="?", default=None)
    sp.set_defaults(func=cmd_tool)

    sp = sub.add_parser("tasks", help="list long-running task records")
    sp.add_argument("--kind", default=None)
    sp.set_defaults(func=cmd_tasks)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SkillError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, indent=2,
                         ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
