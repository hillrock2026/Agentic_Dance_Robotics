"""Virtual K1: Booster K1 MuJoCo 仿真控制接口。"""

from .joint_mapping import JointMapping, JointInfo, ActuatorInfo
from .simulator import K1Sim, K1SimError
from .controllers import JointPDController
from .foot_geometry import (
    FootGeometryDetector,
    FootGeometryError,
    FootGeometryResult,
    GeomFootprint,
    GroundPlane,
    UnsupportedFootGeomError,
)
from . import robot_state

__all__ = [
    "K1Sim", "K1SimError", "JointMapping", "JointInfo", "ActuatorInfo",
    "JointPDController", "robot_state",
    "FootGeometryDetector", "FootGeometryError", "FootGeometryResult",
    "GeomFootprint", "GroundPlane", "UnsupportedFootGeomError",
]

__version__ = "0.1.0"
