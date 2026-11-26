"""统一的机器人状态数据结构。

所有数据来自 MuJoCo 仿真 (MjData), 不是实体机器人传感器读数。

单位约定
--------
- joint_positions      : rad       (关节角, 按关节名索引)
- joint_velocities     : rad/s
- joint_torques        : Nm        (qfrc_actuator 在该关节 dof 上的合力矩)
- base_position        : m         (root body 原点, 世界系)
- base_orientation     : 四元数 (w, x, y, z), 无量纲, 世界系
- base_linear_velocity : m/s       (世界系)
- base_angular_velocity: rad/s     (世界系)
- foot_contacts        : bool + 法向力 N + 接触点位置 m / 法向(无量纲)
"""

from __future__ import annotations

from typing import Any


def new_robot_state() -> dict[str, Any]:
    """创建符合 V1 规范的空状态字典 (字段与顺序固定)。"""
    return {
        "time": None,
        "joint_positions": {},
        "joint_velocities": {},
        "joint_torques": {},
        "base_position": None,
        "base_orientation": None,
        "base_linear_velocity": None,
        "base_angular_velocity": None,
        "foot_contacts": {},
    }


def validate_robot_state(state: dict[str, Any]) -> None:
    """检查状态字典包含全部必需字段, 用于测试。"""
    required = set(new_robot_state().keys())
    missing = required - set(state.keys())
    if missing:
        raise ValueError(f"robot_state 缺少字段: {sorted(missing)}")
