"""K1 Dance Skill Framework.

Layer A of the K1 street-dance project: a Python framework that turns a
YAML skill definition plus motion assets into the full train / export /
sim2sim / evaluate pipeline by *calling* the existing, already-validated
infrastructure (booster_train for RL, booster_deploy for MuJoCo sim2sim).

Layer B (agent integration) lives in :mod:`k1_skills.agent.tools` and
exposes the same capabilities as schema-annotated tools for a future
Agent-Kernel integration.
"""

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.spec import SkillDefinition, load_skill_definition

__version__ = "0.1.0"

__all__ = ["SkillError", "SkillErrorCode", "SkillDefinition", "load_skill_definition"]
