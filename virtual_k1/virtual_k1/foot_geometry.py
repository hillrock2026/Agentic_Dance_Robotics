"""足底几何检测 (K1_VIRTUAL_FOOT_GEOMETRY_FIX_01)。

背景
----
旧检测 (k1_action/virtual_k1_replay.py) 用碰撞盒**中心高度**替代足底高度:

    z = data.geom_xpos[g][2] - model.geom_size[g][1]  if geom_type == 3 else ...

存在三处错误:
  1. MuJoCo mjGEOM_BOX = 6, 常量 3 是 mjGEOM_CAPSULE, 条件对 K1 足部 box
     永远为假, 实际退化为中心高度 (连半尺寸都没有减);
  2. 即使减半尺寸, 也只对未旋转的 box 成立, 足踝 pitch/roll 或 root 姿态
     旋转后底面角点不再位于 中心-z·半高;
  3. 左右脚混取全局 min, 无法区分左右足各自的间隙/穿透。

本模块按碰撞几何体的真实空间位置与姿态计算:

    local collision geometry (box 8 角点, ±half-size)
      -> collision frame     (geom pos/quat —— collision origin 偏移)
      -> foot link frame     (data.xpos/xmat, FK 结果)
      -> robot/root frame    (root free joint 位姿已包含在 FK 中)
      -> world frame

每层变换只应用一次; corner_world = foot_xpos + foot_xmat @ (geom_pos + R(geom_quat) @ corner_local)。

地面
----
Virtual K1 官方 K1_22dof.xml 中地面是唯一无限水平 plane "ground" (z=0)。
本模块显式建模为 GroundPlane (单位法向量 n, 有符号距离 n·x + d, 地面上方为正),
当前仅使用水平实例; 倾斜地面场景出现时可直接扩展, 不做超前抽象。

接触状态分类 (几何意义, 与物理接触力相互独立)
----------------------------------------------
PENETRATION : 最低角点已越过地面超过接触容差;
CONTACT     : 最低角点落在接触容差带内;
AIRBORNE    : 最低角点与地面存在明显间隙。

几何接近地面 != 物理仿真中存在接触力; 物理接触证据请使用
K1Sim.get_foot_contacts() (MuJoCo contact manifold + 法向力), 两者不得混用。

不支持静默退化: 若足部碰撞几何不是 box/sphere, 构造时即抛
UnsupportedFootGeomError (K1 官方模型足部为单 box, 见 K1_22dof.xml:
pos="0.026 0 -0.02" size="0.09 0.035 0.018", collision origin 存在非零平移)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 地面平面


@dataclass(frozen=True)
class GroundPlane:
    """地面平面 n·x + d = 0, 单位法向量 n, 地面上方有符号距离为正。"""

    normal: np.ndarray            # (3,) 单位法向量, 指向地面上方
    offset: float                 # 平面方程常数 d

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(n))
        if not np.isclose(norm, 1.0, atol=1e-9):
            raise ValueError(f"GroundPlane 法向量必须单位化, |n|={norm}")
        object.__setattr__(self, "normal", n)

    @staticmethod
    def horizontal(ground_z: float = 0.0) -> "GroundPlane":
        """水平地面 z = ground_z (Virtual K1 当前唯一场景)。"""
        return GroundPlane(normal=np.array([0.0, 0.0, 1.0]), offset=-float(ground_z))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """点到地面的有符号距离 (地面上方为正), points: (N,3) -> (N,)。"""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ self.normal + self.offset


# ---------------------------------------------------------------------------
# 结果结构


@dataclass
class GeomFootprint:
    """单个足部碰撞 geom 的世界坐标足底信息。"""

    geom_id: int
    geom_type: str                      # "box" / "sphere"
    corners_world: np.ndarray           # (N,3) box 8 角点 / 球心 1 点
    signed_distances: np.ndarray        # (N,) 各点对地有符号距离
    radius: float = 0.0                 # sphere 半径 (box 为 0)

    @property
    def min_corner_z(self) -> float:
        """几何体代表点最低高度 (球含半径修正), 只对水平地面等于底面高度。"""
        return float(np.min(self.signed_distances))


@dataclass
class FootGeometryResult:
    """一只脚的足底几何检测结果 (几何证据, 不含物理接触力)。"""

    side: str
    state: str                          # "PENETRATION" / "CONTACT" / "AIRBORNE"
    clearance: float                    # 最低角点有符号间隙 (>0 悬空, <0 穿透)
    penetration_depth: float            # max(0, -clearance)
    max_corner_z: float                 # 最高角点有符号高度
    geoms: list[GeomFootprint] = field(default_factory=list)
    corners_world: np.ndarray | None = None   # 全部角点并集 (N,3), 可选填充

    def as_dict(self, include_corners: bool = False) -> dict:
        out = {
            "side": self.side,
            "state": self.state,
            "clearance_m": self.clearance,
            "penetration_depth_m": self.penetration_depth,
            "max_corner_z_m": self.max_corner_z,
            "n_geoms": len(self.geoms),
        }
        if include_corners and self.corners_world is not None:
            out["corners_world"] = self.corners_world.tolist()
        return out


# ---------------------------------------------------------------------------
# 检测器


class FootGeometryError(RuntimeError):
    """足底几何检测错误。"""


class UnsupportedFootGeomError(FootGeometryError):
    """足部碰撞几何类型不受支持 —— 显式失败, 不静默退化为中心高度检测。"""


_TYPE_NAME = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}


class FootGeometryDetector:
    """按真实碰撞几何 (角点级) 计算双足足底对地间隙/穿透/接触状态。

    Args:
        model: MuJoCo 模型。
        foot_bodies: {side: body_name}, 例如 {"left": "left_ankle_roll_link"}。
        ground: 地面平面; 缺省水平地面 z=0 (K1_22dof.xml 唯一地面形态)。
        contact_tolerance_m: 几何接触判定容差 (|clearance| 在带内判 CONTACT)。
        numeric_tolerance_m: 数值计算容差 (理论值对比用, 不参与状态分类)。
    """

    def __init__(self, model: mujoco.MjModel,
                 foot_bodies: dict[str, str],
                 ground: GroundPlane | None = None,
                 contact_tolerance_m: float = 2e-3,
                 numeric_tolerance_m: float = 1e-6):
        self.model = model
        self.ground = ground or GroundPlane.horizontal(0.0)
        self.contact_tolerance_m = float(contact_tolerance_m)
        self.numeric_tolerance_m = float(numeric_tolerance_m)

        self._foot_body_ids: dict[str, int] = {}
        self._foot_geoms: dict[str, list[int]] = {}
        for side, bname in foot_bodies.items():
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname))
            if bid < 0:
                raise FootGeometryError(f"足部 body 不存在: {bname}")
            self._foot_body_ids[side] = bid
            gids = []
            for g in range(model.ngeom):
                if (int(model.geom_bodyid[g]) == bid
                        and (int(model.geom_contype[g]) > 0
                             or int(model.geom_conaffinity[g]) > 0)):
                    tname = _TYPE_NAME.get(int(model.geom_type[g]))
                    if tname is None:
                        gname = model.geom(g).name or f"geom{g}"
                        raise UnsupportedFootGeomError(
                            f"{side} 足 body '{bname}' 的碰撞 geom '{gname}' 类型 "
                            f"{int(model.geom_type[g])} 不受支持 (仅 box/sphere); "
                            f"拒绝退化为中心高度检测, 请为该几何类型实现精确角点/边界计算")
                    gids.append(g)
            if not gids:
                raise FootGeometryError(f"{side} 足 body '{bname}' 上没有碰撞 geom")
            self._foot_geoms[side] = gids

    # ------------------------------------------------------------ geometry
    def _local_support_points(self, gid: int) -> tuple[np.ndarray, float]:
        """geom 局部坐标系下的底面代表点 (box: 8 角点; sphere: 球心) 与半径。"""
        t = int(self.model.geom_type[gid])
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            hx, hy, hz = self.model.geom_size[gid]
            pts = np.array([[sx * hx, sy * hy, sz * hz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            return pts, 0.0
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return np.zeros((1, 3)), float(self.model.geom_size[gid][0])
        raise UnsupportedFootGeomError(f"geom {gid} 类型 {t} 不受支持")

    def _geom_world_corners(self, data: mujoco.MjData, gid: int,
                            body_id: int) -> tuple[np.ndarray, float]:
        """显式变换链: geom local -> collision frame -> foot link -> world。

        corner_world = foot_xpos + foot_xmat @ (geom_pos + R(geom_quat) @ corner_local)
        每层变换只应用一次; geom_pos/geom_quat 即 collision origin 的平移/旋转偏移。
        """
        pts_local, radius = self._local_support_points(gid)
        r_local = np.zeros(9)
        mujoco.mju_quat2Mat(r_local, self.model.geom_quat[gid])   # collision frame 旋转
        pts_collision = np.asarray(self.model.geom_pos[gid], dtype=float) \
            + pts_local @ r_local.reshape(3, 3).T                 # + collision origin 平移
        foot_xmat = data.xmat[body_id].reshape(3, 3)              # FK: link -> world
        corners = data.xpos[body_id] + pts_collision @ foot_xmat.T
        return corners, radius

    def verify_transform_chain(self, data: mujoco.MjData) -> dict[str, float]:
        """交叉校验: 显式链结果与 MuJoCo geom_xpos/geom_xmat 组合结果的最大偏差。

        MuJoCo 自身按 body→geom 合成 geom_xpos/geom_xmat, 若两者一致则说明
        变换链每一层恰好应用一次、无遗漏无重复。
        """
        max_err = 0.0
        for side, bid in self._foot_body_ids.items():
            for gid in self._foot_geoms[side]:
                corners, _ = self._geom_world_corners(data, gid, bid)
                pts_local, _ = self._local_support_points(gid)
                r = np.zeros(9)
                mujoco.mju_quat2Mat(r, self.model.geom_quat[gid])
                ref = data.geom_xpos[gid] \
                    + (pts_local @ r.reshape(3, 3).T) @ data.geom_xmat[gid].reshape(3, 3).T
                max_err = max(max_err, float(np.max(np.abs(corners - ref))))
        return {"max_corner_position_error_m": max_err}

    # ------------------------------------------------------------ evaluate
    def evaluate(self, data: mujoco.MjData,
                 include_corners: bool = False) -> dict[str, FootGeometryResult]:
        """对当前 data (须已 mj_forward/mj_step) 计算左右足足底几何。"""
        results: dict[str, FootGeometryResult] = {}
        for side, bid in self._foot_body_ids.items():
            geoms: list[GeomFootprint] = []
            all_corners: list[np.ndarray] = []
            min_d = np.inf
            max_d = -np.inf
            for gid in self._foot_geoms[side]:
                corners, radius = self._geom_world_corners(data, gid, bid)
                dist = self.ground.signed_distance(corners) - radius
                geoms.append(GeomFootprint(
                    geom_id=gid,
                    geom_type=_TYPE_NAME[int(self.model.geom_type[gid])],
                    corners_world=corners,
                    signed_distances=dist,
                    radius=radius,
                ))
                all_corners.append(corners)
                min_d = min(min_d, float(np.min(dist)))
                max_d = max(max_d, float(np.max(dist)))
            clearance = float(min_d)
            penetration = max(0.0, -clearance)
            if penetration > self.contact_tolerance_m:
                state = "PENETRATION"
            elif abs(clearance) <= self.contact_tolerance_m:
                state = "CONTACT"
            else:
                state = "AIRBORNE"
            results[side] = FootGeometryResult(
                side=side,
                state=state,
                clearance=clearance,
                penetration_depth=penetration,
                max_corner_z=float(max_d),
                geoms=geoms,
                corners_world=np.vstack(all_corners) if include_corners else None,
            )
        return results
