"""Evaluation Engine (K1_DANCE_SKILL_FRAMEWORK_02 generalization).

Input is a recorder ``.metrics.npz`` (per-frame, CSV joint order).
Built-in metric families:

    joint_tracking_error, arm_motion_amplitude, arm_motion_velocity,
    root_drift, fall_rate, foot_contact, cycle_consistency

Generalizations over FRAMEWORK_01:

- **Joint groups**: every joint-wise metric is parameterized by named
  groups (``arms``/``legs``/...) resolved from the skill's evaluation
  profile / shared robot profile, not hardcoded slices.
- **Applicability**: ``cycle_consistency`` applies to cyclic motions
  only; requesting it for a oneshot run drops it with a note instead of
  fabricating numbers.
- **Extension interface**: :func:`register_metric` adds custom families
  (signature ``fn(z: dict, groups: dict) -> dict``); skills activate
  them via ``evaluation_profile.custom_metrics: ["module:func"]``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from k1_skills.errors import SkillError, SkillErrorCode
from k1_skills.profiles import joint_groups as resolve_joint_groups

SPEED_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0), (10.0, 1e9))


def load_metrics(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SkillError(SkillErrorCode.METRICS_NOT_FOUND, f"metrics file missing: {path}")
    return {k: z for k, z in np.load(path, allow_pickle=False).items()}


# ----------------------------------------------------------------------
# non-finite handling (REMEDIATION_01 Task B)
#
# Rules applied uniformly to every numeric output field:
# - a segment (whole run / a cycle / a joint series / a speed bin) that
#   contains any NaN/Inf is INVALID for aggregate purposes: its scalar
#   becomes None (never 0, never a fabricated value);
# - global statistics are computed over the finite samples only, with
#   `nonfinite_samples` reporting how many were excluded — an invalid
#   cycle therefore never poisons the statistics of valid cycles;
# - the normal-path (all finite) numbers are bit-identical to the
#   previous implementation.
# ----------------------------------------------------------------------
def _mean_or_none(a: np.ndarray) -> tuple[float | None, int]:
    finite = np.isfinite(a)
    n_bad = int((~finite).sum())
    if not finite.any():
        return None, n_bad
    return float(a[finite].mean()), n_bad


def _max_or_none(a: np.ndarray) -> float | None:
    """Finite-only max: None when no finite samples."""
    finite = np.isfinite(a)
    return float(a[finite].max()) if finite.any() else None


def _max_strict(a: np.ndarray) -> float | None:
    """Order-statistic max: None if ANY sample is non-finite.

    A max over data containing +Inf/-Inf/NaN is itself non-finite; the
    honest JSON representation is None (\"the true maximum is not a finite
    number\"), never the max of the surviving finite subset."""
    if a.size == 0 or not np.isfinite(a).all():
        return None
    return float(a.max())


def _min_or_none(a: np.ndarray) -> float | None:
    if a.size == 0 or not np.isfinite(a).all():
        return None
    return float(a.min())


def _series_ok(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def _round_or_none(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(v, nd)


def _idx(groups: dict[str, slice], name: str) -> np.ndarray:
    sl = groups.get(name)
    if sl is None:
        raise SkillError(SkillErrorCode.EVAL_FAILED,
                         f"joint group '{name}' not configured for evaluation")
    return np.arange(sl.start, sl.stop)


def _per_cycle(z: dict) -> list[np.ndarray]:
    cycle = np.asarray(z["cycle"])
    return [np.where(cycle == c)[0] for c in sorted(set(cycle.tolist()))]


# ----------------------------------------------------------------------
# built-in metric families
# ----------------------------------------------------------------------
def joint_tracking_error(z: dict, groups: dict) -> dict:
    err = np.abs(np.asarray(z["act_joint"], dtype=float)
                 - np.asarray(z["ref_joint"], dtype=float))
    in_tr = np.asarray(z["in_transition"]).astype(bool)
    n_bad = int((~np.isfinite(err)).sum())

    def _seg_or_none(a: np.ndarray) -> float | None:
        """Segment rule: any non-finite inside the segment -> None."""
        if a.size == 0 or not _series_ok(a):
            return None
        return float(a.mean()) * 1e3

    mae_all, _ = _mean_or_none(err)
    mae_arms, _ = _mean_or_none(err[:, _idx(groups, "arms")])
    mae_legs, _ = _mean_or_none(err[:, _idx(groups, "legs")])
    out = {
        "nonfinite_samples": n_bad,
        "mae_all_mrad": mae_all * 1e3 if mae_all is not None else None,
        "mae_arms_mrad": mae_arms * 1e3 if mae_arms is not None else None,
        "mae_legs_mrad": mae_legs * 1e3 if mae_legs is not None else None,
        "action_vs_transition_mrad": {
            "action": _seg_or_none(err[~in_tr]),
            "transition": _seg_or_none(err[in_tr]),
        },
        "per_cycle_mae_arms_mrad": [],
        "per_joint_mae_arms_mrad": [],
    }
    arm_idx = _idx(groups, "arms")
    for j in [int(j) for j in arm_idx]:
        out["per_joint_mae_arms_mrad"].append(
            _round_or_none(_seg_or_none(err[:, j])))
    for idx in _per_cycle(z):
        out["per_cycle_mae_arms_mrad"].append(
            _round_or_none(_seg_or_none(err[np.ix_(idx, arm_idx)])))
    return out


def arm_motion_amplitude(z: dict, groups: dict) -> dict:
    a = _idx(groups, "arms")
    ref = np.asarray(z["ref_joint"], dtype=float)[:, a]
    act = np.asarray(z["act_joint"], dtype=float)[:, a]
    n_joints = a.size

    # Tri-state classification (AMPLITUDE_CLASSIFICATION_FIX_01): a joint
    # is valid — eligible for moving/static — only when BOTH its reference
    # and actual series are fully finite. A non-finite ACTUAL series must
    # not land in the static bucket (nor moving) alongside its
    # invalid-actual listing; the three classes are mutually exclusive and
    # exhaustive over the group.
    ref_ok = np.array([_series_ok(ref[:, j]) for j in range(n_joints)])
    act_ok = np.array([_series_ok(act[:, j]) for j in range(n_joints)])
    valid = ref_ok & act_ok
    ref_amp = np.full(n_joints, np.nan)
    act_amp = np.full(n_joints, np.nan)
    for j in range(n_joints):
        if ref_ok[j]:
            ref_amp[j] = ref[:, j].max() - ref[:, j].min()
        if act_ok[j]:
            act_amp[j] = act[:, j].max() - act[:, j].min()
    moving = valid & (ref_amp > 0.1)
    static = valid & (ref_amp <= 0.1)              # valid static only
    ratios = np.where(moving,
                      act_amp / np.where(ref_amp != 0, ref_amp, np.nan), np.nan)
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "reference_amplitude_rad": [None if not np.isfinite(v) else round(float(v), 3)
                                    for v in ref_amp],
        "actual_amplitude_rad": [None if not np.isfinite(v) else round(float(v), 3)
                                 for v in act_amp],
        "moving_arm_joints": int(moving.sum()),
        "static_arm_joints": int(static.sum()),
        "invalid_reference_joints": [int(j) for j in np.where(~ref_ok)[0]],
        "invalid_actual_joints": [int(j) for j in np.where(~act_ok)[0]],
        "mean_amplitude_ratio": (float(finite_ratios.mean())
                                 if finite_ratios.size else None),
        "amplitude_ratio_per_joint": [
            None if not np.isfinite(r) else round(float(r), 3) for r in ratios],
        # residual amplitude is reported ONLY for valid static joints;
        # invalid joints are listed above instead of hiding in this bucket
        "static_joint_residual_amplitude_rad": [
            None if not np.isfinite(v) else round(float(v), 3)
            for v in act_amp[static]],
    }


def arm_motion_velocity(z: dict, groups: dict) -> dict:
    a = _idx(groups, "arms")
    ref_v = np.abs(np.asarray(z["ref_joint_vel"], dtype=float))[:, a]
    act_v = np.abs(np.asarray(z["act_joint_vel"], dtype=float))[:, a]
    err = np.abs(np.asarray(z["act_joint"], dtype=float)
                 - np.asarray(z["ref_joint"], dtype=float))[:, a]
    speed = ref_v.max(axis=1)

    # Frame accounting (FINAL_REMEDIATION_01 Task B): every input frame is
    # accounted for exactly once — non-finite-speed frames are excluded
    # from bins but COUNTED (never silently dropped), small (<5 frame)
    # bins are omitted from the report with their frames counted, and
    # reported + small + nonfinite == total holds as a hard invariant.
    finite_speed = np.isfinite(speed)
    nonfinite_speed_frames = int((~finite_speed).sum())
    total_frames = int(speed.size)

    bins: dict = {}
    small_bin_frames = 0
    for lo, hi in SPEED_BINS:
        sel = (speed >= lo) & (speed < hi)      # non-finite rows match nothing
        n = int(sel.sum())
        if n < 5:
            small_bin_frames += n
            continue
        seg = err[sel]
        seg_bad = int((~np.isfinite(seg)).sum())
        m = float(seg.mean()) * 1e3 if seg_bad == 0 and n > 0 else None
        bins[f"{lo:g}-{hi if hi < 1e9 else 999:g}rad_s"] = {
            "frames": n,
            "mae_mrad": round(m, 1) if m is not None else None,
            "nonfinite_err_samples": seg_bad,   # bin invalid iff > 0 (mae None)
        }
    reported_frames = int(sum(b["frames"] for b in bins.values()))
    if reported_frames + small_bin_frames + nonfinite_speed_frames != total_frames:
        raise SkillError(SkillErrorCode.EVAL_FAILED,
                         "velocity frame accounting invariant violated")

    ref_mean, ref_bad = _mean_or_none(ref_v)
    act_mean, act_bad = _mean_or_none(act_v)
    return {
        "total_frames": total_frames,
        "binned_frames": reported_frames,
        "small_bin_frames_omitted": small_bin_frames,
        "nonfinite_speed_frames": nonfinite_speed_frames,
        "nonfinite_speed_samples": int(ref_bad + act_bad),
        "nonfinite_err_samples": int((~np.isfinite(err)).sum()),
        "ref_peak_speed_rad_s": _max_strict(ref_v),
        "act_peak_speed_rad_s": _max_strict(act_v),
        "mean_speed_tracking_ratio": (act_mean / ref_mean
                                      if ref_mean not in (None, 0)
                                      and act_mean is not None else None),
        "mae_by_ref_speed": bins,
        "high_speed_mae_mrad_gt10": bins.get("10-999rad_s", {}).get("mae_mrad"),
    }


def root_drift(z: dict, groups: dict) -> dict:
    act = np.asarray(z["act_root_xyz"], dtype=float)
    root_err = np.asarray(z["root_err_xyz"], dtype=float)
    ori = np.asarray(z["trunk_ori_err_deg"], dtype=float)
    per_cycle_end: list[float | None] = []
    per_cycle_max: list[float | None] = []
    for idx in _per_cycle(z):
        seg = act[idx]
        if _series_ok(seg):
            rel = np.linalg.norm(seg - seg[0], axis=1)
            per_cycle_end.append(round(float(rel[-1]) * 1e3, 1))
            per_cycle_max.append(round(float(rel.max()) * 1e3, 1))
        else:
            per_cycle_end.append(None)
            per_cycle_max.append(None)
    norms = np.linalg.norm(root_err, axis=1)
    err_mean, _ = _mean_or_none(norms)
    err_max = _max_strict(norms)
    ori_mean, _ = _mean_or_none(ori)
    return {
        "self_drift_per_cycle_end_mm": per_cycle_end,
        "self_drift_per_cycle_max_mm": per_cycle_max,
        "ref_error_mean_mm": err_mean * 1e3 if err_mean is not None else None,
        "ref_error_max_mm": err_max * 1e3 if err_max is not None else None,
        "trunk_ori_err_mean_deg": ori_mean,
        "trunk_ori_err_max_deg": _max_strict(ori),
        "min_base_height_m": _min_or_none(np.asarray(z["base_height"], dtype=float)),
    }


def fall_rate(z: dict, groups: dict) -> dict:
    fallen = np.asarray(z["fallen"]).astype(bool)
    return {
        "fallen_steps": int(fallen.sum()),
        "total_steps": int(fallen.size),
        "fall_rate": float(fallen.mean()),
        "stopped_early": bool(z["stopped_early"]),
    }


def foot_contact(z: dict, groups: dict) -> dict:
    left = np.asarray(z["left_contact"]).astype(bool)
    right = np.asarray(z["right_contact"]).astype(bool)
    both = left & right
    ls = np.asarray(z["left_slide"], dtype=float)
    rs = np.asarray(z["right_slide"], dtype=float)
    lm, _ = _mean_or_none(ls[left]) if left.any() else (None, 0)
    rm, _ = _mean_or_none(rs[right]) if right.any() else (None, 0)
    means = [m for m in (lm, rm) if m is not None]
    return {
        "double_support_ratio": float(both.mean()),
        "left_airborne_steps": int((~left).sum()),
        "right_airborne_steps": int((~right).sum()),
        "slide_speed_while_contact_mean_m_s": (
            float(np.mean(means)) if means else None),
        "slide_speed_while_contact_max_m_s": (
            _max_strict(np.concatenate([ls[left], rs[right]]))
            if (left.any() or right.any()) else None),
    }


def _error_pattern_correlation(sa: np.ndarray, sb: np.ndarray) -> float | None:
    """Pearson correlation between two cycles' error patterns.

    A cycle's error *pattern* is its (action-frames x joints) absolute
    tracking-error matrix, flattened to one vector — the full
    spatiotemporal error signature. Two cycles are consistent (corr=+1)
    iff their signatures match up to positive affine scaling.

    Undefined (returns None, never a fabricated score) when either
    pattern contains non-finite values, has fewer than two samples, or
    is constant (zero variance).
    """
    a = np.asarray(sa, dtype=float).ravel()
    b = np.asarray(sb, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def cycle_consistency(z: dict, groups: dict) -> dict:
    """Cross-cycle consistency of the tracking-error pattern.

    Definition (K1_DANCE_SKILL_FRAMEWORK_02_MINIMAL_FIX_01, Task B):
    for each cycle, take the absolute joint tracking error of the named
    group (rows = action frames — transition frames excluded — columns =
    group joints), truncate all cycles to the shortest action length,
    and compute the Pearson correlation between every pair of cycles'
    flattened error patterns.

    The previous implementation passed the 2-D matrices straight into
    ``np.corrcoef(sa, sb)[0, 1]``: numpy treats rows as variables, so
    that element was the correlation between the first two *timesteps*
    of one cycle — unrelated to cross-cycle consistency (identical
    cycles returned NaN/garbage instead of 1.0).

    Returns:
        n_cycles, per-cycle group MAE (unchanged semantics),
        cross_cycle_error_correlation: list of pairwise correlations
            (None where undefined), mean over defined pairs,
        undefined_pairs: count of pairs where the correlation is
            undefined (non-finite / constant / too short).
    """
    a = _idx(groups, "arms")
    err = np.abs(np.asarray(z["act_joint"]) - np.asarray(z["ref_joint"]))[:, a]
    in_tr = np.asarray(z["in_transition"]).astype(bool)
    per_cycle_err = [err[idx] for idx in _per_cycle(z)]

    # action-frame error patterns per cycle, truncated to common length
    sigs = [err[idx][~in_tr[idx]] for idx in _per_cycle(z)]
    n = min((s.shape[0] for s in sigs), default=0)
    sigs = [s[:n] for s in sigs]

    corrs: list[float | None] = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            corrs.append(_error_pattern_correlation(sigs[i], sigs[j]))
    defined = [c for c in corrs if c is not None]

    out: dict = {
        "n_cycles": len(per_cycle_err),
        "cross_cycle_error_correlation": corrs,
        "mean_correlation": (round(float(np.mean(defined)), 4)
                             if defined else None),
        "undefined_pairs": len(corrs) - len(defined),
        "pattern_samples_per_cycle": int(n) * int(err.shape[1]) if n else 0,
    }
    maes: list[float | None] = [
        round(float(s.mean() * 1e3), 1) if _series_ok(s) else None
        for s in per_cycle_err]
    out["per_cycle_arm_mae_mrad"] = maes
    valid_maes = [m for m in maes if m is not None]
    out["mae_monotonic_increase"] = bool(
        len(valid_maes) >= 2 and all(valid_maes[i + 1] > valid_maes[i]
                                     for i in range(len(valid_maes) - 1)))
    return out


# ----------------------------------------------------------------------
# extension registry + applicability
# ----------------------------------------------------------------------
METRIC_FUNCS: dict[str, callable] = {
    "joint_tracking_error": joint_tracking_error,
    "arm_motion_amplitude": arm_motion_amplitude,
    "arm_motion_velocity": arm_motion_velocity,
    "root_drift": root_drift,
    "fall_rate": fall_rate,
    "foot_contact": foot_contact,
    "cycle_consistency": cycle_consistency,
}

CYCLIC_ONLY = {"cycle_consistency"}


def register_metric(name: str, fn) -> None:
    """Register a custom metric family: fn(z, groups) -> dict."""
    if name in METRIC_FUNCS:
        raise SkillError(SkillErrorCode.EVAL_FAILED,
                         f"metric '{name}' already registered")
    METRIC_FUNCS[name] = fn


def load_custom_metrics(specs: list[str]) -> list[str]:
    """Activate "module:function" custom metrics; returns loaded names."""
    loaded = []
    for spec in specs or []:
        try:
            module_name, fn_name = spec.split(":")
            fn = getattr(importlib.import_module(module_name), fn_name)
        except Exception as exc:  # noqa: BLE001
            raise SkillError(SkillErrorCode.EVAL_FAILED,
                             f"cannot load custom metric '{spec}': {exc}") from exc
        register_metric(fn_name, fn)
        loaded.append(fn_name)
    return loaded


def applicable_metrics(requested: list[str], motion_kind: str) -> tuple[list[str], list[str]]:
    """Split requested metrics into (applicable, dropped-with-reason)."""
    dropped = []
    out = []
    for name in requested:
        if name in CYCLIC_ONLY and motion_kind != "cyclic":
            dropped.append(f"{name} (cyclic-only; motion is {motion_kind})")
        else:
            out.append(name)
    return out, dropped


# ----------------------------------------------------------------------
# evaluate / compare
# ----------------------------------------------------------------------
def evaluate(metrics_path: str | Path, *, metrics: list[str] | None = None,
             groups: dict[str, slice] | None = None,
             motion_kind: str = "cyclic") -> dict:
    z = load_metrics(metrics_path)
    requested = metrics or list(METRIC_FUNCS)
    applicable, dropped = applicable_metrics(requested, motion_kind)
    if groups is None:
        groups = {"arms": slice(2, 10), "legs": slice(10, 22)}
    result = {
        "metrics_file": str(metrics_path),
        "motion_kind": motion_kind,
        "meta": {
            "fps": float(np.asarray(z["fps"]).ravel()[0]),
            "cycles": int(np.asarray(z["cycles"]).ravel()[0]),
            "period_frames": int(np.asarray(z["period"]).ravel()[0]),
            "source_frames": int(np.asarray(z["n_source_frames"]).ravel()[0]),
        },
    }
    if dropped:
        result["dropped_metrics"] = dropped
    for name in applicable:
        fn = METRIC_FUNCS.get(name)
        if fn is None:
            raise SkillError(SkillErrorCode.EVAL_FAILED, f"unknown metric '{name}'")
        result[name] = fn(z, groups)

    # strict-JSON guard: no NaN/Inf may leak into any numeric field
    import json as _json
    try:
        _json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise SkillError(
            SkillErrorCode.EVAL_FAILED,
            f"evaluation result contains non-finite values that escaped "
            f"metric-level handling: {exc}") from exc
    return result


# scalars surfaced in comparisons: (label, getter, unit, lower_is_better)
KEY_SCALARS = [
    ("arm MAE",            lambda r: r["joint_tracking_error"]["mae_arms_mrad"], "mrad", True),
    ("all-joint MAE",      lambda r: r["joint_tracking_error"]["mae_all_mrad"], "mrad", True),
    ("leg MAE",            lambda r: r["joint_tracking_error"]["mae_legs_mrad"], "mrad", True),
    ("amplitude ratio",    lambda r: r["arm_motion_amplitude"]["mean_amplitude_ratio"], "", False),
    (">10rad/s MAE",       lambda r: r["arm_motion_velocity"]["high_speed_mae_mrad_gt10"], "mrad", True),
    ("root err mean",      lambda r: r["root_drift"]["ref_error_mean_mm"], "mm", True),
    ("drift last cycle",   lambda r: r["root_drift"]["self_drift_per_cycle_end_mm"][-1], "mm", True),
    ("trunk ori err",      lambda r: r["root_drift"]["trunk_ori_err_mean_deg"], "deg", True),
    ("fall rate",          lambda r: r["fall_rate"]["fall_rate"], "", True),
    ("double support",     lambda r: r["foot_contact"]["double_support_ratio"], "", False),
]


def compare(result_a: dict, result_b: dict, *, label_a="A", label_b="B") -> dict:
    rows = []
    for label, getter, unit, lower_better in KEY_SCALARS:
        try:
            va, vb = getter(result_a), getter(result_b)
        except (KeyError, IndexError, TypeError):
            continue
        if va is None or vb is None:
            continue
        delta = vb - va
        pct = delta / abs(va) * 100.0 if abs(va) > 1e-12 else None
        improved = (delta < 0) if lower_better else (delta > 0)
        rows.append({
            "metric": label, "unit": unit,
            label_a: round(va, 4), label_b: round(vb, 4),
            "delta": round(delta, 4), "pct": round(pct, 1) if pct is not None else None,
            "improved": bool(improved),
        })
    return {"labels": [label_a, label_b], "rows": rows}
