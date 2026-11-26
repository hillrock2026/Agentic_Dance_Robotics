"""Virtual K1 仿真控制接口 (K1Sim)。

基于 MuJoCo 官方 Python API 与 Booster 官方 K1_22dof.xml 模型。

控制语义 (由实际加载模型确定)
------------------------------
官方模型 22 个执行器均为 torque/motor 类型:
    data.ctrl[i] ∈ forcerange (Nm), 直接作为关节力矩。
因此 step(control) 的 control 是**力矩**; 位置控制请使用
controllers.JointPDController (软件 PD, 不修改官方模型)。

set_joint_positions() 是**运动学写入** (直接改 qpos 后 mj_forward),
用于设置初始姿态 / 状态重置, 不产生动态过程, 与力矩控制互不混淆。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from . import robot_state as rs
from .foot_geometry import (
    FootGeometryDetector,
    FootGeometryError,
    FootGeometryResult,
    GroundPlane,
    UnsupportedFootGeomError,
)
from .joint_mapping import JointMapping

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "k1_sim.yaml"


class K1SimError(RuntimeError):
    """K1Sim 运行错误。"""


class K1Sim:
    """Booster K1 MuJoCo 仿真器封装。"""

    def __init__(self, model_path: str | None = None, config: str | dict | None = None):
        """初始化仿真器。

        Args:
            model_path: 官方 K1 MuJoCo XML 路径; 缺省取 config 中 model.path。
            config: YAML 路径 / dict / None(用默认 config/k1_sim.yaml)。
        """
        if config is None:
            cfg_path = _DEFAULT_CONFIG
            self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        elif isinstance(config, dict):
            self.cfg = config
        else:
            cfg_path = Path(config)
            self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        if model_path is None:
            model_path = self.cfg["model"]["path"]
            if not Path(model_path).is_absolute():
                base = _DEFAULT_CONFIG.parent if config is None else Path(config).resolve().parent
                model_path = str((base / model_path).resolve())
        if not Path(model_path).is_file():
            raise K1SimError(f"模型文件不存在: {model_path}")
        self.model_path = str(model_path)

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.mapping = JointMapping(self.model)

        # 足部 / 地面配置
        ccfg = self.cfg.get("contact", {})
        self._foot_body_ids: dict[str, int] = {}
        for side, bname in ccfg.get("foot_bodies", {}).items():
            self._foot_body_ids[side] = self.model.body(bname).id
        self._foot_geom_sides: dict[int, str] = {}
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            for side, fbid in self._foot_body_ids.items():
                if bid == fbid and (self.model.geom_contype[gid] > 0
                                    or self.model.geom_conaffinity[gid] > 0):
                    self._foot_geom_sides[gid] = side
        self._ground_geom_ids = {
            self.model.geom(n).id for n in ccfg.get("ground_geom_names", ["ground"])
        }
        self._min_normal_force = float(ccfg.get("min_normal_force", 0.5))
        # 足底几何检测器 (FIX_01): 角点级几何间隙/穿透/接触状态, 与上面的
        # 物理接触读取相互独立; 容差可由 contact.foot_geometry 覆盖
        fgcfg = ccfg.get("foot_geometry", {})
        self._foot_geometry_detector = FootGeometryDetector(
            self.model,
            foot_bodies=ccfg.get("foot_bodies", {}),
            ground=GroundPlane.horizontal(0.0),
            contact_tolerance_m=fgcfg.get("contact_tolerance_m", 2e-3),
            numeric_tolerance_m=fgcfg.get("numeric_tolerance_m", 1e-6),
        )
        self._n_substeps = int(self.cfg.get("sim", {}).get("n_substeps", 1))
        if self._n_substeps < 1:
            raise K1SimError("sim.n_substeps 必须 >= 1")

        # 力矩控制语义校验: 官方模型必须为 torque 语义
        semantics = {a.control_semantics for a in self.mapping.actuators}
        self.control_semantics = semantics
        if semantics == {"torque"}:
            self._is_torque_model = True
        elif semantics <= {"position", "velocity"}:
            self._is_torque_model = False
        else:
            raise K1SimError(
                f"混合执行器语义 {semantics}, V1 接口未支持, 请检查模型")

        self._renderer = None
        self.reset()

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """释放渲染资源。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "K1Sim":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- reset

    def reset(self) -> None:
        """重置到 config.initial_pose 指定的初始站立姿态。

        qpos/qvel/ctrl/act 全部清零后写入初始姿态, 再 mj_forward。
        """
        mujoco.mj_resetData(self.model, self.data)
        pose = self.cfg["initial_pose"]
        root = self.mapping.root_info
        if root is None:
            raise K1SimError("模型缺少 root free joint")
        self.data.qpos[root.qpos_adr + 0] = float(pose["root_position"][0])
        self.data.qpos[root.qpos_adr + 1] = float(pose["root_position"][1])
        self.data.qpos[root.qpos_adr + 2] = float(pose["root_position"][2])
        self.data.qpos[root.qpos_adr + 3:root.qpos_adr + 7] = pose["root_quat_wxyz"]
        for name, val in pose["joints"].items():
            jinfo = self.mapping.joint(name)   # 同时接受 XML/CSV 命名
            self.data.qpos[jinfo.qpos_adr] = float(val)
        mujoco.mj_forward(self.model, self.data)
        self._check_finite("reset 后")

    # ----------------------------------------------------------------- step

    def step(self, control: np.ndarray | dict | None = None,
             n_substeps: int | None = None) -> float:
        """推进物理仿真一个控制周期。

        Args:
            control: 力矩指令 (官方模型为 torque 语义):
                - None     : 保持当前 data.ctrl;
                - dict     : {joint_name: torque_Nm}, 未指定的执行器置 0;
                - ndarray  : shape (nu,) 直接写入 data.ctrl。
                (若模型为 position/velocity 执行器, 同样写入 ctrl, 语义随模型)
            n_substeps: 本次调用的物理子步数, 缺省用 config sim.n_substeps。

        Returns:
            仿真时间 data.time (s)。
        """
        if control is not None:
            self.set_controls(control)
        n = self._n_substeps if n_substeps is None else n_substeps
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self._check_finite("step 后")
        return float(self.data.time)

    def set_controls(self, control: np.ndarray | dict) -> None:
        """写入执行器指令 (不推进仿真)。"""
        if isinstance(control, dict):
            ctrl = np.zeros(self.model.nu)
            for name, val in control.items():
                act = self.mapping.actuator_of_joint(name)
                ctrl[act.actuator_id] = float(val)
        else:
            ctrl = np.asarray(control, dtype=float).ravel()
            if ctrl.shape != (self.model.nu,):
                raise K1SimError(
                    f"控制向量长度 {ctrl.shape[0]} != nu {self.model.nu}")
        if not np.all(np.isfinite(ctrl)):
            raise K1SimError("控制指令包含 NaN/Inf")
        # torque 语义下若模型声明 ctrllimited 则截断; K1 官方模型 ctrl 不限幅
        if bool(np.any(self.model.actuator_ctrllimited)):
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            ctrl = np.clip(ctrl, lo, hi)
        self.data.ctrl[:] = ctrl

    # --------------------------------------------------- kinematic pose set

    def set_joint_positions(self, joint_targets: dict,
                            zero_velocities: bool = True) -> None:
        """运动学写入关节角 (直接改 qpos 并 mj_forward, 无动态过程)。

        Args:
            joint_targets: {joint_name: angle_rad}, 接受 XML 名或官方 CSV 别名。
            zero_velocities: 同时将对应关节速度清零。
        """
        for name, val in joint_targets.items():
            jinfo = self.mapping.joint(name)
            self.data.qpos[jinfo.qpos_adr] = float(val)
            if zero_velocities:
                self.data.qvel[jinfo.dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._check_finite("set_joint_positions 后")

    # ------------------------------------------------------------- readings

    def get_joint_positions(self) -> dict[str, float]:
        """驱动关节角 {joint_name: rad} (含 root 除外)。"""
        out = {}
        for act in self.mapping.actuators:
            jinfo = self.mapping.joint(act.joint_name)
            out[act.joint_name] = float(self.data.qpos[jinfo.qpos_adr])
        return out

    def get_joint_velocities(self) -> dict[str, float]:
        """驱动关节角速度 {joint_name: rad/s}。"""
        out = {}
        for act in self.mapping.actuators:
            jinfo = self.mapping.joint(act.joint_name)
            out[act.joint_name] = float(self.data.qvel[jinfo.dof_adr])
        return out

    def get_joint_torques(self) -> dict[str, float]:
        """关节实际合力矩 {joint_name: Nm} (qfrc_actuator, 含执行器输出)。"""
        out = {}
        for act in self.mapping.actuators:
            jinfo = self.mapping.joint(act.joint_name)
            out[act.joint_name] = float(self.data.qfrc_actuator[jinfo.dof_adr])
        return out

    def get_actuator_forces(self) -> dict[str, float]:
        """执行器标量力 {actuator_name: N 或 gear 缩放前力} (data.actuator_force)。"""
        return {a.name: float(self.data.actuator_force[a.actuator_id])
                for a in self.mapping.actuators}

    def get_base_state(self) -> dict[str, Any]:
        """Root (trunk) 位姿与速度, 世界系。

        Returns:
            position: (3,) m
            orientation: (4,) 四元数 (w,x,y,z)
            linear_velocity: (3,) m/s 世界系
            angular_velocity: (3,) rad/s 世界系 (mj_objectVelocity 换算)
        """
        root = self.mapping.root_info
        if root is None:
            raise K1SimError("无 root free joint")
        pos = self.data.qpos[root.qpos_adr:root.qpos_adr + 3].copy()
        quat = self.data.qpos[root.qpos_adr + 3:root.qpos_adr + 7].copy()
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            root.body_id, vel6, 0)  # flg_local=0 → 世界系; [0:3]角速度 [3:6]线速度
        return {
            "position": pos,
            "orientation": quat,
            "linear_velocity": vel6[3:6].copy(),
            "angular_velocity": vel6[0:3].copy(),
        }

    def get_foot_contacts(self) -> dict[str, Any]:
        """读取双足与地面的接触。

        Returns:
            {side: {"in_contact": bool,
                    "n_contacts": int,
                    "total_normal_force": N,
                    "normal_forces": [N...],
                    "positions": [[x,y,z]...m],
                    "normals": [[nx,ny,nz]...],   # 法向从地面指向足部
                    "geoms": [(foot_geom_id, other_geom_id)...]}}
        """
        out: dict[str, Any] = {
            side: {"in_contact": False, "n_contacts": 0,
                   "total_normal_force": 0.0, "normal_forces": [],
                   "positions": [], "normals": [], "geoms": []}
            for side in self._foot_body_ids
        }
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            side = self._foot_geom_sides.get(g1) or self._foot_geom_sides.get(g2)
            other = g2 if side == self._foot_geom_sides.get(g1) else g1
            if side is None:
                continue  # 与足部无关的接触
            if other not in self._ground_geom_ids:
                continue  # 足部与非地面物体的接触 (V1 只统计地面)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            fn = float(wrench[0])  # 接触坐标系法向力
            # MuJoCo contact.frame 法向由 geom1 指向 geom2。
            # 若 geom1 是足部, 法向为 足→地面, 反转为 地面→足部 (即 +z 朝上)。
            normal_world = con.frame[0:3].copy()
            if self._foot_geom_sides.get(g1) is not None:
                normal_world = -normal_world
            rec = out[side]
            rec["n_contacts"] += 1
            rec["normal_forces"].append(fn)
            rec["positions"].append(con.pos.copy())
            rec["normals"].append(normal_world)
            rec["geoms"].append((int(g1) if side == self._foot_geom_sides.get(g1) else int(g2),
                                 int(other)))
            if fn >= self._min_normal_force:
                rec["in_contact"] = True
            rec["total_normal_force"] += max(fn, 0.0)
        return out

    def get_foot_geometry(self, include_corners: bool = False) -> dict[str, Any]:
        """足底几何检测 (FIX_01): 按碰撞几何真实角点计算, 区分左右足。

        返回 {side: {"side","state","clearance_m","penetration_depth_m",
                     "max_corner_z_m","n_geoms"[,"corners_world"]}}:
            clearance_m         最低角点有符号间隙 (>0 悬空, <0 穿透)
            penetration_depth_m max(0, -clearance_m)
            state               PENETRATION / CONTACT / AIRBORNE (几何分类,
                                |clearance| <= contact_tolerance 判 CONTACT)

        注意: 这是几何证据; 物理接触力请用 get_foot_contacts(), 两者不得混用。
        """
        res = self._foot_geometry_detector.evaluate(self.data,
                                                    include_corners=include_corners)
        return {side: r.as_dict(include_corners=include_corners)
                for side, r in res.items()}

    def get_robot_state(self) -> dict[str, Any]:
        """按 V1 规范返回统一状态字典 (单位见 robot_state 模块文档)。"""
        base = self.get_base_state()
        state = rs.new_robot_state()
        state["time"] = float(self.data.time)
        state["joint_positions"] = self.get_joint_positions()
        state["joint_velocities"] = self.get_joint_velocities()
        state["joint_torques"] = self.get_joint_torques()
        state["base_position"] = base["position"]
        state["base_orientation"] = base["orientation"]
        state["base_linear_velocity"] = base["linear_velocity"]
        state["base_angular_velocity"] = base["angular_velocity"]
        state["foot_contacts"] = self.get_foot_contacts()
        return state

    # -------------------------------------------------------------- utility

    def render(self, width: int = 640, height: int = 480,
               camera: str | None = None) -> np.ndarray:
        """离屏渲染一帧 RGB 图像 (用于无界面验证, 需要 MUJOCO_GL=egl/osmesa)。

        注: 仅在内存中调整离屏帧缓冲尺寸 (model.vis.global_.offwidth/height),
        不修改官方 XML 文件。
        """
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(width, 640)
            self.model.vis.global_.offheight = max(height, 480)
            self._renderer = mujoco.Renderer(self.model, height, width)
        cam = -1 if camera is None else self.model.camera(camera).id
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    @property
    def timestep(self) -> float:
        """物理步长 (s), 取自官方模型, 不修改。"""
        return float(self.model.opt.timestep)

    def _check_finite(self, when: str) -> None:
        for name in ("qpos", "qvel", "qacc", "ctrl"):
            arr = getattr(self.data, name)
            if not np.all(np.isfinite(arr)):
                raise K1SimError(f"{when} data.{name} 出现 NaN/Inf")
