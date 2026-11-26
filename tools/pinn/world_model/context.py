"""Episode physical context xi_e, extracted from the ACTUAL model file.

All K1_WM episodes were generated with the single unmodified model
virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml, so the physical
context is constant across episodes. Every value below is KNOWN (extracted
from the model file / collector manifest at build time) — nothing is fitted,
inferred from episode IDs, or derived from action labels.

Provenance notes:
  - xml_sha256 / mujoco_version come from stage1_data/index.json (collector
    manifest), cross-checked against the local file at extraction time.
  - effective foot-ground sliding friction is max(ground, foot) = 1.0 by the
    MuJoCo element-wise max rule (verified in K1_PINN_TASK_01 sec 7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
XML_PATH = REPO_ROOT / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_context(data_dir: Path | None = None) -> dict:
    """Return the physical context shared by all episodes (JSON-serializable)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(XML_PATH))
    ctx = {
        "provenance": {
            "xml_path": str(XML_PATH),
            "xml_sha256": _sha256(XML_PATH),
            "mujoco_version": mujoco.__version__,
            "source": "extracted from model file; constant across episodes",
            "per_episode_variation": "none (no physics domain randomization; "
                                     "episode randomization touched initial "
                                     "state and controller gains only)",
        },
        "dimensions": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "na": m.na},
        "gravity": m.opt.gravity.tolist(),                       # m/s^2
        "timestep": float(m.opt.timestep),                       # s
        "integrator": "Euler (semi-implicit: position integrated with the "
                      "updated velocity)",
        "solver": "Newton",
        "solver_tolerance": float(m.opt.tolerance),
        "impratio": float(m.opt.impratio),
        "cone": "pyramidal",
        "total_mass_kg": float(m.body_mass.sum()),
        "dof_armature": m.dof_armature.tolist(),                 # (28,) included in M
        "dof_damping": m.dof_damping.tolist(),                   # (28,) all zero
        "actuator_forcerange": m.actuator_forcerange.tolist(),   # (22,2) Nm
        "actuator_gear": m.actuator_gear[:, 0].tolist(),         # (22,) all 1.0
        "joint_limits_rad": m.jnt_range[1:].tolist(),            # (22,2)
        "ground_friction": m.geom_friction[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")].tolist(),
        "effective_foot_friction": [1.0, 0.005, 0.0001],
        "solref": m.geom_solref[0].tolist(),
        "solimp": m.geom_solimp[0].tolist(),
        "equality_constraints": int(m.neq),                      # 0
        "tendons": int(m.ntendon),                               # 0
    }
    if data_dir is not None:
        manifest = json.loads((Path(data_dir) / "index.json").read_text())
        assert manifest["xml_sha256"] == ctx["provenance"]["xml_sha256"], \
            "episode data was generated with a different model file"
        ctx["provenance"]["matches_data_manifest"] = True
    return ctx


if __name__ == "__main__":
    print(json.dumps(episode_context(), indent=2))
