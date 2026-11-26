# K1 World Model — State Transition Dataset (K1_WM_01)

统一状态转移数据接口，覆盖 `tools/pinn/stage1_data/`（30 episodes / 24,578 samples / 1 kHz）。
本包只读 Stage1 数据，不修改任何原始产物，不训练模型。

## 1. 状态转移定义（与 MuJoCo 4.0.1 实测一致）

```
x_t = (q_t, v_t)
q_t ∈ R^29:  root pos (3, 世界系) + quat wxyz (4) + 22 hinge (rad)
v_t ∈ R^28:  root 线速度 (3, 世界系) + root 角速度 (3, root 局部系) + 22 关节速度
ctrl_t ∈ R^22: 力矩指令 (Nm, gear=1 motor)

v_{t+1} = v_t + dt · a_t                  (a_t = 区间起点瞬时前向动力学加速度)
q_{t+1} = mj_integratePos(q_t, v_{t+1}, dt)   [MuJoCo 半隐式 Euler，实测验证]
dt = 1e-3 s
```

## 2. 快速使用

```python
import sys; sys.path.insert(0, "tools/pinn")
from world_model import StateTransitionDataset, RolloutWindowDataset, episode_context

tr = StateTransitionDataset(split="train")          # 16,147 单步转移
s = tr[0]
# s["q_t"](29,), s["v_t"](28,), s["ctrl_t"](22,), s["a_t"](28,),
# s["q_next"](29,), s["v_next"](28,), s["dt"], s["free"], s["support"], s["impact"],
# s["M_upper_t"](406,), s["qfrc_bias_t"], s["qfrc_constraint_t"],   (include_dynamics=True)
# s["episode_id"], s["step"], s["transition_valid"]

rw = RolloutWindowDataset(split="test", horizon=8)  # 3,110 个 8 步窗口（8 ms）
w = rw[0]
# w["ctrl_seq"](8,22), w["q_next_seq"](8,29), w["v_next_seq"](8,28),
# w["a_seq"](8,28), w["time_seq"](9,), w["free_seq"], w["impact_seq"], w["t0"]

ctx = episode_context()                              # ξ_e：模型文件提取的物理上下文
# ctx["total_mass_kg"], ["gravity"], ["dof_armature"](28,), ["actuator_forcerange"](22,2), ...
```

## 3. 字段可用性（模型推理视角）

| 类别 | 字段 |
|---|---|
| 推理时可得 | `q_t, v_t, ctrl_t, dt`，episode context |
| 可推导（= clip(ctrl) 的确定函数，同快照相等已验证 ≤1.9e-6） | `actuator_force_t`、`qfrc_actuator_t[6:28]`（root 6 维恒 0） |
| 模型可计算 | `M_upper_t`、`qfrc_bias_t`（(q,v) 的函数，经 MuJoCo） |
| 仅训练期 | `a_t`、`qfrc_constraint_t`（求解器输出）、`free/support/impact` 标签 |
| 不适用 | `act_t`（na=0，motor 无 activation 状态） |

## 4. 已物化文件（`transitions/`）

| 文件 | 内容 |
|---|---|
| `transitions_{train,val,test}.npz` | 堆叠单步转移：16,147 / 5,249 / 3,152 条 |
| `windows_h8_{split}.npz` | 8 步窗口索引（episode_index, t0）：16,007 / 5,221 / 3,110 条（8 ms/窗） |
| `transition_inventory.json` | 逐 episode 字段/时间/连续性清单 |
| `transition_checks.json` | 连续性与分布检查（issues: NONE） |
| `integration_verification.json` | MuJoCo 积分验证（见 §5） |
| `../episode_context.json` | ξ_e |

窗口保证：不跨 episode、不跨 split、时间连续（每 episode 无丢帧/重复时间戳）、所有样本有效（integrity 零问题）。8 步仅为初始 horizon，`RolloutWindowDataset(horizon=H)` 支持任意 H。

## 5. 积分验证结论（integration_verification.json）

对**全部 24,548 条转移**实测：
- **H1（MuJoCo 半隐式 Euler：`v'=v+dt·a; q'=mj_integratePos(q,v',dt)`）** 复现记录的下一状态至 max |Δq| = **7e-9**（float64 存储精度量级；误差上限由约束求解器二次收敛容差决定，v 更新 max 3.5e-6 × dt ≈ 该量级）。
- H2（显式 Euler，位置用旧速度）与 H3（分量式 `q += v·dt`，四元数处理错误）最大误差均 ~7e-2——**差 7 个数量级**。
- `mj_integratePos` 后四元数模长保持 1.0（12 位小数）。
- 结论：`a_t` 的语义 = 区间起点瞬时加速度，被 `mj_Euler` 以半隐式方式消费。**后续 B/C/D 模型的状态更新必须使用 `mj_integratePos`，禁止分量式更新。**

## 6. Episode Context ξ_e

所有 episode 使用同一未修改模型（xml sha256 与数据清单交叉核验一致），ξ_e 恒定：总质量 19.666 kg、重力 [0,0,−9.81]、dt=1e-3、Euler/Newton/impratio=10/金字塔锥、armature 向量 (28,)、执行器 forcerange (22,2)、关节限位 (22,2)、地面/足底摩擦（有效滑动摩擦 1.0）、solref/solimp、equality=0、tendon=0。全部为**已知参数**（模型文件提取），无逐 episode 变化（随机化只涉及初始状态与控制器增益，未涉及物理参数），无虚构字段。
