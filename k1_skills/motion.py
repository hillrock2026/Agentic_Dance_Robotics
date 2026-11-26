"""Motion Resource Interface (K1_DANCE_SKILL_FRAMEWORK_02).

One skill's reference motion is a *resource set* with typed roles:

    dataset_trajectory   ground-truth csv with provenance (31-col)
    booster_csv          official 29-col format (input of csv_to_npz)
    train_npz            Isaac Lab FK body references (training input)
    deploy_npz           MuJoCo FK runtime motion (oneshot skills)
    deploy_npz_cyclic    + Hermite transition loop (cyclic skills)

``MotionResourceSet`` loads the roles that exist, verifies each role's
invariants (fps / frame counts / 22 joints / root mode) and the
cross-role sync (fps equal, train-vs-deploy frame count within +-1 --
the two FK pipelines legitimately differ by one frame).

``stationary`` root skills additionally get the constant-position /
identity-orientation invariant established by the reference-correction
task; ``free`` root skills get no root constraint (full-body motion).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from k1_skills.errors import SkillError, SkillErrorCode


@dataclass
class MotionData:
    kind: str                 # role: train_npz | deploy_npz | deploy_npz_cyclic
    fps: float
    n_frames: int
    joint_names: list[str]
    joint_pos: np.ndarray     # (T, 22)
    root_pos: np.ndarray      # (T, 3)
    root_quat: np.ndarray     # (T, 4)
    path: Path

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "fps": self.fps,
            "frames": self.n_frames,
            "duration_s": round(self.duration_s, 3),
            "joints": len(self.joint_names),
        }


def load_npz(path: str | Path, kind: str) -> MotionData:
    path = Path(path)
    if not path.exists():
        raise SkillError(SkillErrorCode.MISSING_RESOURCE, f"motion npz missing: {path}")
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - map to stable code
        raise SkillError(SkillErrorCode.MOTION_INVALID,
                         f"cannot load npz {path}: {exc}") from exc
    for key in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
        if key not in z:
            raise SkillError(SkillErrorCode.MOTION_INVALID,
                             f"{path}: npz missing key '{key}'")
    fps = float(np.asarray(z["fps"]).ravel()[0])
    joint_pos = np.asarray(z["joint_pos"], dtype=float)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 22:
        raise SkillError(SkillErrorCode.MOTION_INVALID,
                         f"{path}: joint_pos must be (T, 22), got {joint_pos.shape}")
    body_pos = np.asarray(z["body_pos_w"], dtype=float)
    body_quat = np.asarray(z["body_quat_w"], dtype=float)
    # joint_names/body_names are metadata added by newer conversion tools;
    # legacy Booster-released npz may lack them (name checks degrade to no-op)
    joint_names = ([str(n) for n in z["joint_names"]]
                   if "joint_names" in z else [])
    return MotionData(
        kind=kind, fps=fps, n_frames=int(joint_pos.shape[0]),
        joint_names=joint_names,
        joint_pos=joint_pos,
        root_pos=body_pos[:, 0, :],   # body 0 = trunk/root in both pipelines
        root_quat=body_quat[:, 0, :],
        path=path,
    )


# ----------------------------------------------------------------------
# resource set
# ----------------------------------------------------------------------
ROLE_LABELS = {
    "dataset_trajectory": "dataset trajectory (ground truth)",
    "booster_csv": "official booster csv",
    "train_npz": "training npz (Isaac FK)",
    "deploy_npz": "runtime npz (MuJoCo FK)",
    "deploy_npz_cyclic": "cyclic runtime npz (MuJoCo FK + transition)",
}


@dataclass
class MotionResourceSet:
    skill_id: str
    motion_kind: str                         # cyclic | oneshot
    root_mode: str                           # stationary | free
    declared_fps: float
    source_frames: int | None
    transition_frames: int
    resources: dict[str, Path] = field(default_factory=dict)
    loaded: dict[str, MotionData] = field(default_factory=dict)
    joint_order_decls: dict[str, str] = field(default_factory=dict)  # role -> ref role
    content_sha256: dict[str, str] = field(default_factory=dict)     # role -> declared
    canonical_npz_names: list[str] | None = None

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _resolve_entry(entry) -> tuple[Path, str | None, str | None]:
        """Resource entries: path strings or
        {path, joint_order: <role>, content_sha256: <64-hex>}."""
        if isinstance(entry, dict):
            return Path(entry["path"]), entry.get("joint_order"), \
                entry.get("content_sha256")
        return Path(entry), None, None

    @staticmethod
    def from_spec(definition) -> "MotionResourceSet":
        motion = definition.motion
        res: dict[str, Path] = {}
        decls: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for role, entry in motion.get("resources", {}).items():
            path, order_ref, csha = MotionResourceSet._resolve_entry(entry)
            res[role] = definition.path(path) if not path.is_absolute() else path
            if order_ref:
                decls[role] = str(order_ref)
            if csha:
                hashes[role] = str(csha).lower()
        mrs = MotionResourceSet(
            skill_id=definition.skill_id,
            motion_kind=str(motion["kind"]),
            root_mode=definition.root_mode,
            declared_fps=float(motion["fps"]),
            source_frames=(int(motion["source_frames"])
                           if motion.get("source_frames") is not None else None),
            transition_frames=int(motion.get("transition_frames", 0) or 0),
            resources=res, joint_order_decls=decls, content_sha256=hashes,
        )
        try:
            from k1_skills.robot import canonical_joint_orders
            mrs.canonical_npz_names = canonical_joint_orders()["npz"]
        except Exception:  # noqa: BLE001 - strictness handled in verify
            mrs.canonical_npz_names = None
        for role, path in res.items():
            if role.endswith("_npz") or role.endswith("_npz_cyclic"):
                mrs.loaded[role] = load_npz(path, role)
        return mrs

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- invariants ---------------------------------------------------------
    def verify(self) -> list[str]:
        problems: list[str] = []
        loaded = self.loaded

        for role, m in loaded.items():
            if abs(m.fps - self.declared_fps) > 1e-6:
                problems.append(f"{role}: fps {m.fps} != declared {self.declared_fps}")

        problems += self._verify_joint_order()
        problems += self._verify_structure()
        # free-root skills: no root invariants by design
        return problems

    # ------------------------------------------------------------------
    # Joint identity/order contract (FINAL_REMEDIATION_01 Task D)
    #
    # The authoritative 22-joint identity is the deploy robot config's
    # ``sim_joint_names`` (single source; robot.canonical_joint_orders
    # enforces 22 unique names before trusting it). Numeric similarity
    # alone can NEVER prove joint identity — two joints may track similar
    # or even IDENTICAL trajectories — so the contract is:
    #
    #   1. a resource WITH joint_names proves its identity by matching the
    #      canonical list exactly (name, count, order);
    #   2. a resource WITHOUT joint_names must DECLARE an order reference
    #      to a named resource that passed check 1, AND bind that claim to
    #      its exact file content via ``content_sha256``: the digest is
    #      recomputed at every load, so ANY content change — including a
    #      swap of two byte-identical trajectory columns — voids the
    #      declaration and the resource is rejected until re-declared;
    #   3. the joint_pos cross-check vs the referenced resource stays as
    #      auxiliary evidence (contradiction is a problem; agreement
    #      alone proves nothing about identity);
    #   4. no names / no grounded declaration / no content binding /
    #      digest mismatch / canonical unavailable -> fail-closed.
    # ------------------------------------------------------------------
    def _verify_joint_order(self) -> list[str]:
        problems: list[str] = []
        canonical = self.canonical_npz_names
        if canonical is None:
            return [f"{self.skill_id}: joint identity cannot be verified — "
                    "canonical K1 joint order unavailable (deploy robot "
                    "config not loadable); refusing to accept resources"]

        verified_named: set[str] = set()
        for role, m in self.loaded.items():
            if not m.joint_names:
                continue
            if list(m.joint_names) == list(canonical):
                verified_named.add(role)
            else:
                missing = [n for n in canonical if n not in m.joint_names]
                extra = [n for n in m.joint_names if n not in canonical]
                dupes = ({n for n in m.joint_names
                          if list(m.joint_names).count(n) > 1})
                detail = []
                if missing:
                    detail.append(f"missing {missing}")
                if extra:
                    detail.append(f"unknown {extra}")
                if dupes:
                    detail.append(f"duplicated {sorted(dupes)}")
                if not (missing or extra or dupes):
                    detail.append("same 22 names in a different ORDER")
                problems.append(
                    f"{role}: joint identity mismatch vs canonical K1 order "
                    f"({'; '.join(detail)})")

        for role, m in self.loaded.items():
            if m.joint_names or role in self.joint_order_decls:
                continue
            problems.append(
                f"{role}: npz has no joint_names and no joint_order "
                f"declaration; refusing to assume order")

        for role, ref_role in self.joint_order_decls.items():
            if role not in self.loaded:
                continue
            # (a) the identity claim must be strongly bound to the file
            # content: digest recomputed on every load
            declared = self.content_sha256.get(role)
            if not declared:
                problems.append(
                    f"{role}: joint_order declaration lacks content_sha256 "
                    f"binding; an identity claim without content binding "
                    f"cannot detect resource modification — refusing")
                continue
            actual = self._file_sha256(self.resources[role])
            if actual != declared:
                problems.append(
                    f"{role}: resource content no longer matches the declared "
                    f"content_sha256 (expected {declared}, got {actual}); "
                    f"the joint-order identity claim is VOID — re-verify and "
                    f"re-declare, or restore the original resource")
                continue
            # (b) the declaration must ground in a canonically-verified
            # named resource
            if ref_role not in verified_named:
                problems.append(
                    f"{role}: joint_order declaration references '{ref_role}' "
                    f"which is not a canonically-verified named resource; "
                    f"identity would be ungrounded")
                continue
            # (c) auxiliary data cross-check (evidence, not proof)
            unnamed = self.loaded[role]
            ref = self.loaded[ref_role]
            n = min(unnamed.n_frames, ref.n_frames)
            if n > 0:
                diff = float(np.abs(unnamed.joint_pos[:n]
                                    - ref.joint_pos[:n]).max())
                if diff > 1e-3:
                    problems.append(
                        f"{role}: declared joint_order '{ref_role}' "
                        f"contradicted by data (joint_pos max diff "
                        f"{diff:.3g} rad over {n} common frames)")
        return problems

    def _verify_structure(self) -> list[str]:
        problems: list[str] = []
        loaded = self.loaded

        if "train_npz" in loaded:
            t = loaded["train_npz"]
            if self.source_frames is not None and t.n_frames != self.source_frames:
                problems.append(
                    f"train_npz has {t.n_frames} frames, declared source_frames="
                    f"{self.source_frames}")

        if self.motion_kind == "cyclic" and "deploy_npz_cyclic" in loaded:
            c = loaded["deploy_npz_cyclic"]
            if self.source_frames is not None and c.n_frames != (
                    self.source_frames + self.transition_frames):
                problems.append(
                    f"deploy_npz_cyclic has {c.n_frames} frames, expected "
                    f"source+transition = {self.source_frames + self.transition_frames}")
            if "train_npz" in loaded:
                t = loaded["train_npz"]
                head = c.joint_pos[: t.n_frames]
                diff = float(np.abs(head - t.joint_pos).max())
                if diff > 1e-4:
                    problems.append(
                        f"cyclic npz head differs from train npz joints "
                        f"(max {diff:.3g} rad)")
                if self.root_mode == "stationary":
                    pos_span = float(np.abs(c.root_pos - c.root_pos[0]).max())
                    quat_span = float(np.abs(c.root_quat[:, 0] - c.root_quat[0, 0]).max())
                    if pos_span > 1e-6:
                        problems.append(
                            f"stationary-root violation in cyclic npz: {pos_span:.3g} m")
                    if quat_span > 1e-5:
                        problems.append(
                            f"stationary-root violation in cyclic npz: quat w varies "
                            f"{quat_span:.3g}")

        if self.motion_kind == "oneshot":
            if "deploy_npz" in loaded and "train_npz" in loaded:
                t, d = loaded["train_npz"], loaded["deploy_npz"]
                # the two FK pipelines may legitimately differ by one frame
                if abs(t.n_frames - d.n_frames) > 1:
                    problems.append(
                        f"train ({t.n_frames}) vs deploy ({d.n_frames}) frame count "
                        f"differ by more than 1")
                if self.source_frames is not None and not (
                        self.source_frames - 1 <= d.n_frames <= self.source_frames + 1):
                    problems.append(
                        f"deploy_npz frames {d.n_frames} not within +-1 of declared "
                        f"source_frames {self.source_frames}")
            if self.root_mode == "stationary" and "train_npz" in loaded:
                t = loaded["train_npz"]
                pos_span = float(np.abs(t.root_pos - t.root_pos[0]).max())
                if pos_span > 1e-6:
                    problems.append(
                        f"stationary-root violation in train npz: {pos_span:.3g} m")

        # free-root skills: no root invariants by design
        return problems

    # -- description ---------------------------------------------------------
    def describe(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "motion_kind": self.motion_kind,
            "root_mode": self.root_mode,
            "fps": self.declared_fps,
            "source_frames": self.source_frames,
            "transition_frames": self.transition_frames if self.motion_kind == "cyclic" else 0,
            "roles": {role: ({"path": str(p), "exists": p.exists()}
                             if role not in self.loaded
                             else self.loaded[role].describe())
                      for role, p in self.resources.items()},
        }


# ----------------------------------------------------------------------
# legacy single-motion helpers (kept for compatibility)
# ----------------------------------------------------------------------
def verify_reference(spec_motion: dict, motion: MotionData, *, cyclic: bool) -> list[str]:
    """v1-style single-motion check; superseded by MotionResourceSet.verify."""
    problems: list[str] = []
    declared_fps = float(spec_motion["fps"])
    if abs(motion.fps - declared_fps) > 1e-6:
        problems.append(f"fps mismatch: declared {declared_fps}, file {motion.fps}")
    source = spec_motion.get("source_frames")
    transition = int(spec_motion.get("transition_frames", 0) or 0)
    if source is not None:
        expected = int(source) + (transition if cyclic else 0)
        if motion.n_frames != expected:
            problems.append(f"frame count mismatch: expected {expected}, got {motion.n_frames}")
    if spec_motion.get("root_mode_hint") == "stationary" or not cyclic:
        pos_span = float(np.abs(motion.root_pos - motion.root_pos[0]).max())
        quat_span = float(np.abs(motion.root_quat[:, 0] - motion.root_quat[0, 0]).max())
        if pos_span > 1e-6:
            problems.append(f"stationary-root violation: root pos varies by {pos_span:.6g} m")
        if quat_span > 1e-5:
            problems.append(f"stationary-root violation: root quat w varies by {quat_span:.6g}")
    return problems


def load_dataset_csv(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SkillError(SkillErrorCode.MISSING_RESOURCE, f"dataset trajectory missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        n_rows = sum(1 for _ in fh)
    return {"path": str(path), "columns": len(header), "frames": n_rows}
