"""Robot profile resolution: shared facts + per-skill control overrides.

A shared profile (``skills/_profiles/<id>.yaml``) carries robot-level
facts; a skill's ``control_profile.extends`` names the profile it builds
on and adds the policy-level values it was trained under. Merging is
shallow with skill-wins semantics, and the merged result must satisfy
the stepping invariant (checked in spec validation after merge).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.paths import SKILLS_DIR

PROFILE_DIRS = (SKILLS_DIR / "_profiles",)


def load_profile(profile_id: str) -> dict:
    for base in PROFILE_DIRS:
        p = base / f"{profile_id}.yaml"
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                                 f"profile {p} must be a mapping")
            return data
    raise SkillError(SkillErrorCode.INVALID_DEFINITION,
                     f"robot profile '{profile_id}' not found under "
                     f"{[str(b) for b in PROFILE_DIRS]}")


def resolve_control_profile(control: dict) -> dict:
    """Merge ``extends`` chain (skill wins) and return the effective dict.

    The profile contributes robot facts (model/dof/policy_dt/action rule/
    joint groups); the skill contributes gains/limits/stepping. Result
    keeps ``_extends`` for provenance.
    """
    extends = control.get("extends")
    if not extends:
        return dict(control)
    merged = dict(load_profile(str(extends)))
    merged.update({k: v for k, v in control.items() if k != "extends"})
    merged["_extends"] = extends
    # derive policy_frequency_hz from the shared policy_dt when the skill
    # declares only stepping
    if "policy_frequency_hz" not in merged and "policy_dt" in merged:
        merged["policy_frequency_hz"] = round(1.0 / float(merged["policy_dt"]), 6)
    return merged


def joint_groups(evaluation_profile: dict, control_profile: dict) -> dict[str, slice]:
    """Resolve named joint groups -> slices over the 22-dof CSV order.

    Priority: evaluation_profile.joint_groups (skill) over the shared
    profile's groups carried inside control_profile. Values are [start,
    end) index pairs.
    """
    raw = dict(control_profile.get("joint_groups") or {})
    raw.update(evaluation_profile.get("joint_groups") or {})
    out: dict[str, slice] = {}
    for name, pair in raw.items():
        try:
            lo, hi = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        out[name] = slice(lo, hi)
    if "arms" not in out:  # engine fallback so built-ins always work
        out["arms"] = slice(2, 10)
    if "legs" not in out:
        out["legs"] = slice(10, 22)
    return out
