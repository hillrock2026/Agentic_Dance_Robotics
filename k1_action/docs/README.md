# K1 Action Data Conversion — 使用说明

将 Motion Studio 的人体动作（SMPL, clean_motion.npz）转换为 Booster K1 的 22-DOF 关节动作数据，形成可回放、可训练的数据集。

## 1. 快速开始

```bash
cd /home/liu/workspace/Agentic_Program/Agentic_Dance_Robotics

# 单动作转换（含 Virtual K1 运动学+物理验证）
.venv/bin/python k1_action/scripts/convert_motion_to_k1.py \
    --npz motion_studio/outputs/motions/<job>/edited_motion.npz \
    --name arm_wave --category arm_wave \
    --out datasets/k1_dance_v1/arm_wave \
    --upper-body-mode      # 上肢主导动作：腿部保持官方站立位
    --simulate --render    # A 运动学 + B 物理回放 + GIF

# 查看动作详情与验证状态
.venv/bin/python k1_action/scripts/validate_k1_action.py --action datasets/k1_dance_v1/arm_wave
.venv/bin/python k1_action/scripts/inspect_k1_action.py   --action datasets/k1_dance_v1/arm_wave

# 批量数据集
.venv/bin/python - <<'PY'
from k1_action.dataset_builder import build_k1_training_dataset
actions = [{"name": "arm_wave", "category": "arm_wave",
            "dir": "datasets/k1_dance_v1/arm_wave", "status": "NEEDS_REVIEW"}]
build_k1_training_dataset(actions, "datasets/k1_dance_v1", overwrite=True)
PY
```

## 2. Python API

```python
import sys; sys.path.insert(0, '.')
from k1_action.motion_loader import load_human_motion, validate_human_motion
from k1_action.retargeting import FullBodyRetargeter, retarget_to_k1
from k1_action.trajectory_processing import resample, joint_stats
from k1_action.validators import validate_k1_trajectory
from k1_action.action_exporter import export_k1_action, load_k1_action
from k1_action.virtual_k1_replay import kinematic_replay, physical_replay

hm = load_human_motion('motion_studio/outputs/motions/<job>/edited_motion.npz')
assert validate_human_motion(hm) == []
traj = retarget_to_k1(hm)                      # 全身忠实模式
traj = resample(traj, 50.0)                    # 25 -> 50 Hz，时长守恒
report = validate_k1_trajectory(traj)          # 限位/动力学/时间轴/有限性
export_k1_action(traj, 'out/action', action_name='x', action_category='x',
                 source_motion=str(hm.path), source_sha256=hm.sha256,
                 source_body_model='SMPL', source_fps=hm.fps,
                 validation=report)
act = load_k1_action('out/action')             # 独立读取（含官方 CSV 直读）
```

## 3. 输出格式

每个动作目录：

| 文件 | 内容 |
|---|---|
| `trajectory.csv` | `frame_index, timestamp_s` + 官方 29 列（`root_pos_*`, `root_quat_x/y/z/w`, 22 关节角，官方列名/列序，弧度绝对角） |
| `metadata.json` | 溯源（源文件+SHA256、SMPL、源/输出 FPS）、关节顺序、单位、坐标系、重定向方法与质量指标、丢弃 DOF 清单、软件版本、状态 |
| `validation.json` | 限位/速度/加速度/NaN/时间轴检查、重定向质量、仿真结果（A/B 分列）、needs_review 项 |
| `replay/physical_replay.gif` | 物理回放录像（--render 时） |

状态阶梯（不会自动升级）：`CONVERTED → VALIDATED → SIMULATION_VALIDATED`；任何失败/存疑 → `NEEDS_REVIEW` / `FAILED`。

## 4. 重定向方法（要点）

- **分段方向重定向 + MuJoCo 雅可比阻尼最小二乘 IK**（不用 SMPL 欧拉角直拷）：
  上臂方向→肩(2DOF)精确；前臂方向→肘(2DOF)尽力；大腿方向→髋(3DOF)精确；小腿方向→膝(1DOF)；脚方向→踝(2DOF)尽力；颈→头方向→头部(2DOF)
- 坐标系标定运行时完成：人/K1 躯干坐标系由双肩连线+重力构建（对 GVHMR 世界任意朝向鲁棒）；K1 前向已用脚尖几何核实（误差 1.7°）
- Root：yaw-only 四元数（人体朝向）；xy 平移 1:1；z 锁定官方站立高度（upper_body_mode）或限幅 ±0.12m 的缩放起伏
- 后处理：5 帧滑动平均 + 2% 限位裕度；**越限帧逐条记录**（帧号/关节/原始值/修正值/越限量），绝不静默截断
- 已知映射损失（如实记录于 metadata.dropped_human_dof）：脊柱 3 关节、双腕、双手、锁骨

## 5. 转换模式选择

- `--upper-body-mode`：上肢主导动作（如 Arm Wave）。GVHMR 单目恢复的下肢噪声/漂移可达 49°，忠实重定向会破坏 PD 站立；此模式腿部保持官方站立位并在 metadata 声明。
- 默认（full body）：腿部主导动作（如 Top Rock）。忠实重定向全部链；物理回放摔倒风险由平衡控制器决定（本阶段不做），结果如实记录。

## 6. 测试

```bash
.venv/bin/python -m pytest k1_action/tests/ -v   # 17 项，基于真实动作数据
```
