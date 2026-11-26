"""Test 1: 官方 K1 MuJoCo 模型加载。"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"


def test_model_file_exists():
    assert MODEL_PATH.is_file(), f"官方模型缺失: {MODEL_PATH}"


def test_mesh_assets_exist():
    mesh_dir = MODEL_PATH.parent / "meshes"
    stl = list(mesh_dir.glob("*.STL"))
    assert len(stl) >= 24, f"mesh 文件不足: {len(stl)}"


def test_load_and_forward():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq == 29 and model.nv == 28 and model.nu == 22
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    # 加载后无物理警告
    assert np.all(data.warning.number == 0), f"mj_forward 警告: {data.warning.number}"


def test_no_initial_ground_penetration_at_qpos0():
    """qpos0 机器人悬空 (root z=1.0), 不应有任何接触。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_k1sim_loads_and_resets():
    from virtual_k1 import K1Sim
    sim = K1Sim()
    try:
        assert sim.mapping.summary()["nu"] == 22
        assert np.isfinite(sim.data.qpos).all()
    finally:
        sim.close()
