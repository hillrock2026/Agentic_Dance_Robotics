"""Skill Registry: discover, load, validate and query skill definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.motion import MotionResourceSet
from k1_skills.paths import SKILLS_DIR
from k1_skills.robot import check_against_deploy, check_step_consistency
from k1_skills.spec import SkillDefinition, load_skill_definition


@dataclass
class SkillRecord:
    definition: SkillDefinition
    dir_name: str


class SkillRegistry:
    """Registry over ``skills/<skill_id>/skill.yaml`` definitions."""

    def __init__(self, skills_dir=SKILLS_DIR):
        self.skills_dir = skills_dir

    # ------------------------------------------------------------------
    def discover(self) -> dict[str, SkillRecord]:
        out: dict[str, SkillRecord] = {}
        if not self.skills_dir.exists():
            return out
        for d in sorted(self.skills_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if not (d / "skill.yaml").exists():
                continue
            definition = load_skill_definition(d)
            out[definition.skill_id] = SkillRecord(definition=definition, dir_name=d.name)
        return out

    def get(self, skill_id: str) -> SkillDefinition:
        records = self.discover()
        if skill_id not in records:
            raise SkillError(SkillErrorCode.SKILL_NOT_FOUND,
                             f"skill '{skill_id}' not found; known: {sorted(records)}")
        return records[skill_id].definition

    def list(self) -> list[dict]:
        out = []
        for skill_id, rec in self.discover().items():
            d = rec.definition
            out.append({
                "skill_id": skill_id,
                "version": d.version,
                "schema": d.raw.get("schema_version", 1),
                "motion_kind": d.motion_kind,
                "root_mode": d.root_mode,
                "policy_roles": d.policy_roles(),
                "description": d.description.strip().splitlines()[0],
                "yaml": str(d.yaml_path),
            })
        return out

    # ------------------------------------------------------------------
    def validate(self, skill_id: str) -> dict:
        """Full resource validation: definition, motion resource set,
        control profile (self + vs deploy), policy artifacts, training
        task registration, evaluation wiring."""
        definition = self.get(skill_id)
        problems: list[str] = []
        warnings: list[str] = []

        # 1. motion resource set: existence + invariants + sync
        try:
            mrs = MotionResourceSet.from_spec(definition)
        except SkillError as exc:
            return self._report(definition, [exc.message], warnings)
        missing = [f"{r}" for r, p in mrs.resources.items() if not p.exists()]
        if missing:
            problems.append(f"motion resources missing: {', '.join(missing)}")
            return self._report(definition, problems, warnings)
        problems += mrs.verify()

        # 2. control profile self-consistency (stepping/fps)
        problems += check_step_consistency(definition.control_profile,
                                           float(definition.motion["fps"]))

        # 3. control profile vs every declared deploy task
        for role, entry in definition.runtime_profile.get("policies", {}).items():
            task = entry.get("deploy_task")
            if not task:
                continue
            try:
                deploy_problems = check_against_deploy(definition.control_profile, task)
                problems += [f"[{role}:{task}] {p}" for p in deploy_problems]
            except SkillError as exc:
                warnings.append(f"[{role}:{task}] {exc.message}")

        # 4. policy artifacts for roles that declare one
        #    lifecycle semantics: not-yet-exported role => warning;
        #    exported-but-missing/drifted => problem (runtime contract)
        from k1_skills.learning import check_staged_policy
        from k1_skills.paths import deploy_models_dir
        import json as _json
        from k1_skills.learning import policy_manifest_path
        for role, entry in definition.runtime_profile.get("policies", {}).items():
            policy_file = entry.get("policy_file")
            if not policy_file:
                continue
            staged = deploy_models_dir() / policy_file
            manifest_exists = policy_manifest_path(definition, role).exists()
            if not staged.exists():
                if manifest_exists:
                    problems.append(
                        f"[{role}] policy '{policy_file}' missing although an export "
                        f"manifest exists — restore from the version archive")
                else:
                    warnings.append(
                        f"[{role}] policy '{policy_file}' not yet provisioned "
                        f"(run train + export)")
                continue
            if manifest_exists:
                drift = check_staged_policy(definition, role)
                if drift.get("ok") is False:
                    problems.append(
                        f"[{role}] staged policy drifted from export manifest "
                        f"({drift.get('actual_sha256', '')[:12]}…); re-export")

        # 5. baseline section (info-only, v1 compat)
        if definition.baseline.get("policy"):
            bp = definition.path(definition.baseline["policy"])
            if not bp.exists() and not (
                    deploy_models_dir() / definition.baseline["policy"].name).exists():
                warnings.append(f"baseline policy path not found: {bp}")

        # 6. training task registered in booster_train source
        task_id = definition.training_profile["task_id"]
        train_root = definition.path(
            "tools/booster_train/source/booster_train/booster_train/tasks")
        if _grep_tree(train_root, task_id) == "":
            problems.append(f"training task id '{task_id}' not found under {train_root}")

        # 7. evaluation wiring: metric names known; custom metrics loadable
        from k1_skills.evaluation import (METRIC_FUNCS, applicable_metrics,
                                          load_custom_metrics)
        custom = definition.evaluation_profile.get("custom_metrics") or []
        if custom:
            try:
                loaded = load_custom_metrics([str(c) for c in custom])
                warnings.append(f"custom metrics loaded: {loaded}")
            except SkillError as exc:
                problems.append(exc.message)
        unknown = [m for m in definition.evaluation_profile["metrics"]
                   if m not in METRIC_FUNCS]
        if unknown:
            problems.append(f"unknown metrics in evaluation_profile: {unknown}")
        _, dropped = applicable_metrics(
            definition.evaluation_profile["metrics"], definition.motion_kind)
        if dropped:
            warnings.append(f"metrics dropped as not applicable "
                            f"({definition.motion_kind}): {dropped}")

        return self._report(definition, problems, warnings)

    @staticmethod
    def _report(definition: SkillDefinition, problems: list[str], warnings: list[str]) -> dict:
        return {
            "skill_id": definition.skill_id,
            "version": definition.version,
            "valid": len(problems) == 0,
            "problems": problems,
            "warnings": warnings,
        }


def _grep_tree(root, needle: str) -> str:
    for p in sorted(Path(root).rglob("*.py")):
        try:
            if needle in p.read_text(encoding="utf-8"):
                return str(p)
        except OSError:
            continue
    return ""
