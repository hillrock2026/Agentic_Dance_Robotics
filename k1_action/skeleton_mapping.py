"""Skeleton mapping: SMPL joints <-> K1 joints, name normalization, orders.

Facts encoded here are verified against K1_22dof.xml (runtime) and the
official action CSVs — see docs/K1_ACTION_FORMAT_ANALYSIS_V1.md.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "human_to_k1_joint_mapping.yaml"


def load_mapping_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


CFG = load_mapping_config()
K1_JOINT_ORDER: list[str] = CFG["k1_joint_order"]
CSV_TO_XML: dict[str, str] = CFG["csv_to_xml_name"]
XML_TO_CSV: dict[str, str] = {v: k for k, v in CSV_TO_XML.items()}
# identity for the 20 identically-named joints
for _j in K1_JOINT_ORDER:
    XML_TO_CSV.setdefault(_j, _j)
SMPL_IDS: dict[str, int] = CFG["smpl_joint_ids"]

assert len(K1_JOINT_ORDER) == 22, "K1 must have 22 actuated joints"

# official CSV column order (root first, then 22 joints in CSV naming)
OFFICIAL_CSV_COLUMNS: list[str] = (
    ["root_pos_x", "root_pos_y", "root_pos_z",
     "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]
    + [XML_TO_CSV[j] for j in K1_JOINT_ORDER]
)

# internal trajectory.csv columns (official 29 + explicit time axis)
TRAJECTORY_COLUMNS: list[str] = ["frame_index", "timestamp_s"] + OFFICIAL_CSV_COLUMNS


def csv_name_to_xml(name: str) -> str:
    return CSV_TO_XML.get(name, name)


def xml_name_to_csv(name: str) -> str:
    return XML_TO_CSV.get(name, name)


def xml_joint_ids(model) -> dict[str, int]:
    """name -> actuator/joint id from a loaded MuJoCo model (sanity helper)."""
    import mujoco

    out = {}
    for name in K1_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert jid >= 0, f"{name} missing in model"
        assert aid >= 0, f"actuator {name} missing in model"
        out[name] = jid
    return out


def verify_order_against_model(model) -> list[str]:
    """Return list of mismatches between configured order and the model's
    actual joint list (empty == consistent)."""
    import mujoco

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    hinges = [n for n in names if n and n != "world_joint"]
    errs = []
    if set(hinges) != set(K1_JOINT_ORDER):
        errs.append(f"joint sets differ: only-in-model={set(hinges)-set(K1_JOINT_ORDER)}")
    return errs
