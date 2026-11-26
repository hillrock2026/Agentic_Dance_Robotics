"""K1 软件控制器。

官方 K1_22dof.xml 的 22 个执行器全部是 torque/motor 类型
(ctrl 直接映射为关节力矩, 受 forcerange 截断), 官方模型中不存在
位置执行器。因此位置控制在本项目中以软件 PD 控制器实现:
力矩输出同样写入 data.ctrl, 不修改官方模型的任何物理参数。
"""

from __future__ import annotations

import numpy as np

from .joint_mapping import JointMapping


class JointPDController:
    """关节空间 PD 位置控制器 (输出力矩, 按执行器顺序排列)。

    torque = kp * (q_target - q) - kd * dq [+ gravity_comp * qfrc_bias]

    输出被官方 actuator forcerange 截断, 不越界。
    """

    def __init__(self, mapping: JointMapping, gains: dict,
                 gravity_compensation: bool = True):
        """
        Args:
            mapping: 已构建的 JointMapping。
            gains: {joint_name: {kp: float, kd: float}}, 接受 XML 名或 CSV 别名。
            gravity_compensation: 是否叠加重力/科氏补偿项 (qfrc_bias)。
        """
        self.mapping = mapping
        self.gravity_compensation = gravity_compensation
        nu = mapping.model.nu
        self._kp = np.zeros(nu)
        self._kd = np.zeros(nu)
        self._force_lo = np.zeros(nu)
        self._force_hi = np.zeros(nu)
        for act in mapping.actuators:
            i = act.actuator_id
            self._force_lo[i] = act.force_range[0] * abs(act.gear)
            self._force_hi[i] = act.force_range[1] * abs(act.gear)
        for name, g in gains.items():
            act = mapping.actuator_of_joint(name)
            i = act.actuator_id
            self._kp[i] = float(g["kp"])
            self._kd[i] = float(g["kd"])

    def compute(self, sim, targets: dict) -> np.ndarray:
        """计算 PD 力矩。

        Args:
            sim: K1Sim 实例 (读取当前 q, dq 及 qfrc_bias)。
            targets: {joint_name: target_rad}, 未指定的关节目标保持当前角。

        Returns:
            shape (nu,) 力矩数组, 已按 forcerange 截断, 可直接写入 sim.step()。
        """
        q = sim.data.qpos
        dq = sim.data.qvel
        torque = np.zeros(self.mapping.model.nu)
        for act in self.mapping.actuators:
            i = act.actuator_id
            jinfo = self.mapping.joint(act.joint_name)
            qadr, vadr = jinfo.qpos_adr, jinfo.dof_adr
            q_t = float(targets.get(act.joint_name, q[qadr]))
            torque[i] = (self._kp[i] * (q_t - q[qadr])
                         - self._kd[i] * dq[vadr])
            if self.gravity_compensation:
                torque[i] += sim.data.qfrc_bias[vadr]
        return np.clip(torque, self._force_lo, self._force_hi)
