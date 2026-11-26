"""Error semantics shared by the CLI and the agent tool layer.

Every failure carries a stable machine-readable code. The agent tool
envelope is ``{"ok": false, "error": {"code", "message", "details"}}``
so a future Agent-Kernel capability provider can map codes to its own
retryable/terminal classification without parsing messages.
"""

from __future__ import annotations

from enum import Enum


class SkillErrorCode(str, Enum):
    # definition / registry
    SKILL_NOT_FOUND = "skill_not_found"
    INVALID_DEFINITION = "invalid_definition"
    MISSING_RESOURCE = "missing_resource"
    # motion / data
    MOTION_INVALID = "motion_invalid"
    PREPARE_FAILED = "prepare_failed"
    # training / export
    TRAIN_LAUNCH_FAILED = "train_launch_failed"
    TASK_NOT_FOUND = "task_not_found"
    EXPORT_FAILED = "export_failed"
    CHECKPOINT_NOT_FOUND = "checkpoint_not_found"
    # runtime / evaluation
    RUN_FAILED = "run_failed"
    EVAL_FAILED = "eval_failed"
    METRICS_NOT_FOUND = "metrics_not_found"
    # tools
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_INTERNAL_ERROR = "tool_internal_error"

    @property
    def retryable(self) -> bool:
        """Long-horizon agents may retry these; the rest are terminal."""
        return self in (SkillErrorCode.PREPARE_FAILED, SkillErrorCode.RUN_FAILED,
                        SkillErrorCode.TRAIN_LAUNCH_FAILED)


class SkillError(Exception):
    """Framework error with a stable code plus human-readable message."""

    def __init__(self, code: SkillErrorCode, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = SkillErrorCode(code)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message, "details": self.details}
