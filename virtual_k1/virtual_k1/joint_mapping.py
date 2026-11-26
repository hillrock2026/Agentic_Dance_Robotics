"""K1 关节映射构建模块。

从**实际加载的 MuJoCo 模型**中提取关节、执行器及其地址映射,
不做任何硬编码假定。同时处理官方资源中存在的命名差异:

- XML 关节名存在 "aa" 前缀怪癖:
  aahead_yaw_joint / aahead_pitch_joint / aaleft_shoulder_pitch_joint /
  aaright_shoulder_pitch_joint (仅肩 pitch 带 aa 前缀)
- 官方动作 CSV (booster_assets/motions/K1/*.csv) 使用无前缀名:
  head_yaw_joint? 不 —— CSV 对头部保留 aahead_*,
  仅对肩 pitch 去掉 aa (left_shoulder_pitch_joint)。

本模块对每个驱动关节同时接受 XML 原名与 CSV 名作为别名。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


def _obj_name(model: mujoco.MjModel, objtype: int, objid: int) -> str:
    name = mujoco.mj_id2name(model, objtype, objid)
    return name if name else f"<unnamed_{objid}>"


@dataclass
class JointInfo:
    """单个 MuJoCo 关节的实际属性。"""

    joint_id: int
    name: str
    type: str          # free / hinge / slide / ball
    body_id: int
    body_name: str
    qpos_adr: int
    dof_adr: int
    limited: bool
    range: tuple[float, float]
    axis: tuple[float, float, float]
    armature: float
    nq: int            # 该关节在 qpos 中占用的维数
    nv: int            # 该关节在 qvel 中占用的维数


@dataclass
class ActuatorInfo:
    """单个执行器的实际属性与控制语义分类。"""

    actuator_id: int
    name: str
    joint_name: str            # 传动目标关节 (joint transmission)
    transmission_type: str
    control_semantics: str     # torque / position / velocity / unknown
    gear: float
    ctrl_limited: bool
    ctrl_range: tuple[float, float]
    force_limited: bool
    force_range: tuple[float, float]
    gainprm: list[float] = field(default_factory=list)
    biasprm: list[float] = field(default_factory=list)
    bias_note: str = ""


# MuJoCo joint type 枚举值 → 名称
_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


class JointMapping:
    """基于实际加载模型的关节/执行器映射。

    Attributes:
        joints: 全部关节 (含 root free joint) 的 JointInfo 列表。
        actuators: 全部执行器的 ActuatorInfo 列表 (按 actuator 索引序)。
        actuated_joints: 被执行器直接驱动的关节名列表 (按 actuator 顺序)。
        root_joint_name: root free joint 名称 (若存在)。
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.joints: list[JointInfo] = []
        self.actuators: list[ActuatorInfo] = []
        self.actuated_joints: list[str] = []
        self.root_joint_name: str | None = None

        self._joint_by_name: dict[str, JointInfo] = {}
        self._act_by_name: dict[str, ActuatorInfo] = {}
        self._act_id_of_joint: dict[str, int] = {}
        self._alias_to_xml: dict[str, str] = {}

        self._build_joints()
        self._build_actuators()
        self._build_aliases()

    # ------------------------------------------------------------------ build

    def _build_joints(self) -> None:
        for jid in range(self.model.njnt):
            m = self.model
            jt = int(m.jnt_type[jid])
            nq = 7 if jt == int(mujoco.mjtJoint.mjJNT_FREE) else 1
            nv = 6 if jt == int(mujoco.mjtJoint.mjJNT_FREE) else 1
            info = JointInfo(
                joint_id=jid,
                name=_obj_name(m, mujoco.mjtObj.mjOBJ_JOINT, jid),
                type=_JOINT_TYPE_NAMES.get(jt, f"type{jt}"),
                body_id=int(m.jnt_bodyid[jid]),
                body_name=_obj_name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[jid])),
                qpos_adr=int(m.jnt_qposadr[jid]),
                dof_adr=int(m.jnt_dofadr[jid]),
                limited=bool(m.jnt_limited[jid]),
                range=(float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])),
                axis=(float(m.jnt_axis[jid][0]), float(m.jnt_axis[jid][1]),
                      float(m.jnt_axis[jid][2])),
                armature=float(m.dof_armature[int(m.jnt_dofadr[jid])]),
                nq=nq,
                nv=nv,
            )
            self.joints.append(info)
            self._joint_by_name[info.name] = info
            if info.type == "free":
                self.root_joint_name = info.name

    def _build_actuators(self) -> None:
        m = self.model
        for aid in range(m.nu):
            trntype = int(m.actuator_trntype[aid])
            if trntype != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise NotImplementedError(
                    f"actuator {aid} transmission type {trntype} != joint; "
                    "K1 官方模型应为 joint transmission, 请检查模型版本")
            jid = int(m.actuator_trnid[aid][0])
            jname = _obj_name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)

            gainprm = [float(x) for x in m.actuator_gainprm[aid][:3]]
            biasprm = [float(x) for x in m.actuator_biasprm[aid][:3]]
            gaintype = int(m.actuator_gaintype[aid])
            biastype = int(m.actuator_biastype[aid])

            # 依据实际 gain/bias 参数判定控制语义 (MuJoCo actuator 模型):
            #   motor/torque:  force = gain*ctrl + bias, 且 bias ≈ 0
            #   position:      biasprm = (0, -kp, -kv), gainprm[0] = kp
            #   velocity:      biasprm = (0, 0, -kv),   gainprm[0] = kv
            if (biastype == int(mujoco.mjtBias.mjBIAS_AFFINE)
                    and biasprm[1] < 0.0):
                semantics, note = "position", (
                    f"kp={gainprm[0]:.4g}, kv={-biasprm[2]:.4g} (内置位置伺服)")
            elif (biastype == int(mujoco.mjtBias.mjBIAS_AFFINE)
                    and biasprm[2] < 0.0):
                semantics, note = "velocity", (
                    f"kv={gainprm[0]:.4g} (内置速度伺服)")
            else:
                semantics, note = "torque", (
                    f"gaintype={gaintype}, biastype={biastype}, "
                    f"gainprm={gainprm}, biasprm={biasprm} "
                    "(ctrl 直接映射为力矩)")

            info = ActuatorInfo(
                actuator_id=aid,
                name=_obj_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
                joint_name=jname,
                transmission_type="joint",
                control_semantics=semantics,
                gear=float(m.actuator_gear[aid][0]),
                ctrl_limited=bool(m.actuator_ctrllimited[aid]),
                ctrl_range=(float(m.actuator_ctrlrange[aid][0]),
                            float(m.actuator_ctrlrange[aid][1])),
                force_limited=bool(m.actuator_forcelimited[aid]),
                force_range=(float(m.actuator_forcerange[aid][0]),
                             float(m.actuator_forcerange[aid][1])),
                gainprm=gainprm,
                biasprm=biasprm,
                bias_note=note,
            )
            self.actuators.append(info)
            self._act_by_name[info.name] = info
            self._act_id_of_joint[jname] = aid

        self.actuated_joints = [a.joint_name for a in self.actuators]

    def _build_aliases(self) -> None:
        """官方动作 CSV 关节名 → XML 关节名 别名表。

        CSV 命名 (booster_assets/src/booster_assets/motions.py K1_JOINT_NAMES)
        与 XML 的差异仅在两个肩 pitch 关节: CSV 无 "aa" 前缀。
        """
        csv_names = [
            "aahead_yaw_joint", "aahead_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_elbow_pitch_joint", "left_elbow_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_elbow_pitch_joint", "right_elbow_yaw_joint",
            "left_hip_pitch_joint", "left_hip_roll_joint",
            "left_hip_yaw_joint", "left_knee_pitch_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint",
            "right_hip_yaw_joint", "right_knee_pitch_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        xml_set = set(self._joint_by_name)
        for name in csv_names:
            if name in xml_set:
                self._alias_to_xml[name] = name
            elif ("aa" + name) in xml_set:
                self._alias_to_xml[name] = "aa" + name

    # ----------------------------------------------------------------- lookup

    def resolve_joint(self, name: str) -> str:
        """把 XML 名或官方 CSV 别名解析为 XML 关节名。"""
        if name in self._joint_by_name:
            return name
        if name in self._alias_to_xml:
            return self._alias_to_xml[name]
        raise KeyError(
            f"未知关节名 '{name}'。可用: XML 名 {sorted(self._joint_by_name)} "
            f"或别名 {sorted(self._alias_to_xml)}")

    def joint(self, name: str) -> JointInfo:
        return self._joint_by_name[self.resolve_joint(name)]

    def actuator_of_joint(self, joint_name: str) -> ActuatorInfo:
        xml_name = self.resolve_joint(joint_name)
        aid = self._act_id_of_joint.get(xml_name)
        if aid is None:
            raise KeyError(f"关节 '{joint_name}' ({xml_name}) 没有对应执行器")
        return self.actuators[aid]

    def qpos_indices(self, joint_name: str) -> slice:
        info = self.joint(joint_name)
        return slice(info.qpos_adr, info.qpos_adr + info.nq)

    def dof_indices(self, joint_name: str) -> slice:
        info = self.joint(joint_name)
        return slice(info.dof_adr, info.dof_adr + info.nv)

    @property
    def root_info(self) -> JointInfo | None:
        if self.root_joint_name is None:
            return None
        return self._joint_by_name[self.root_joint_name]

    # ------------------------------------------------------------------ check

    def summary(self) -> dict:
        n_hinge = sum(1 for j in self.joints if j.type == "hinge")
        n_free = sum(1 for j in self.joints if j.type == "free")
        semantics = sorted({a.control_semantics for a in self.actuators})
        return {
            "njnt": self.model.njnt,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "joints_hinge": n_hinge,
            "joints_free": n_free,
            "actuated_joints": len(self.actuated_joints),
            "actuator_semantics": semantics,
            "root_joint": self.root_joint_name,
        }
