# K1_WM_01_REPORT — State Transition Dataset 构建与验证

- 日期: 2026-11-24
- 范围: 只读构建 `tools/pinn/stage1_data/`（30 eps / 24,578 samples / 1 kHz）之上的统一状态转移接口；不训练模型，不改 Stage1 数据，不改 PINN 数学定义
- 判定: **满足 K1_WM_01 验收条件（PASS）**

---

## 1. 实际完成的工作

1. **第一阶段盘点**（以真实文件为准）：Stage1 npz 每 episode 17 个字段（qpos(29,f64), qvel(28,f64), ctrl(22,f64), qacc(28,f64), M_upper(406,f64), qfrc_bias/actuator/constraint/passive/applied(28,f64), actuator_force(22,f64), ncon, nefc, time, foot_l, foot_r, residual_closure）；时间严格 1 kHz 单调、起于 0、无重复/跳变；四元数模长=1（机器精度）；na=0（`act` 字段按设计不存在）；`xfrc_applied≡0` 由采集器断言；xml sha256 与数据清单一致。q/v 同一快照、ctrl→qfrc_actuator 无延迟（同快照相等 ≤1.9e-6 = f64 路径存储一致）。
2. **统一接口设计**（§2 字段可用性表）：q/v/ctrl/dt 推理可得；actuator_force 与 qfrc_actuator 可推导（= clip(ctrl)，gear=1）；M/qfrc_bias 模型可计算；a_t/qfrc_constraint/接触标签仅训练期；act 不适用。**全部按真实存在性标注，无虚构。**
3. **浮动基座状态处理**：不使用 `q+v·dt`；状态更新以 MuJoCo `mj_integratePos` 为准（Task 1 已验证往返为 0），并做了新的实证验证（§4）。
4. **状态转移样本构建**：单步 24,548 条（ep 内相邻行配对）；8 步窗口 24,338 条（每窗恰 8 ms，跨 episode/split/不连续/无效样本均被构造性排除——integrity 零问题故无排除发生，检查代码仍内置守卫）。窗口含 episode id 与原始样本索引（episode_index, t0）。
5. **验证**（§4/§5）：连续性、有限性、四元数模长、快照一致性（passive/applied≡0、actuator_force≡ctrl）、分布统计、free/contact/impact 分布，以及 MuJoCo 积分假设检验。
6. **统一数据 API**（`world_model/wm_data.py`）：`StateTransitionDataset`（支持 include_dynamics 开关）与 `RolloutWindowDataset`（任意 horizon），按 split 加载，附 episode context；冒烟测试通过。

## 2. 修改与新增文件

| 文件 | 性质 |
|---|---|
| `tools/pinn/world_model/__init__.py` | 新增（包导出） |
| `tools/pinn/world_model/context.py` | 新增（ξ_e：从实际 XML 提取，含 sha256 与数据清单交叉核验） |
| `tools/pinn/world_model/wm_data.py` | 新增（两个数据集类） |
| `tools/pinn/world_model/build_transitions.py` | 新增（物化 + 盘点 + 检查） |
| `tools/pinn/world_model/verify_state_integration.py` | 新增（积分验证） |
| `tools/pinn/world_model/README.md` | 新增（使用说明 + 字段可用性表） |
| `tools/pinn/world_model/episode_context.json` | 新增（ξ_e 值） |
| `tools/pinn/world_model/transitions/*`（8 个文件） | 新增（物化数据 + 检查 JSON） |

Stage1 原始数据、Stage0 产物、既有训练/评价代码：**零改动**。

## 3. 数据规模

| split | episodes | 单步转移 | 8 步窗口 | free | impact |
|---|---|---|---|---|---|
| train | 20 | 16,147 | 16,007 | 2,701 | 234 |
| val | 4 | 5,249 | 5,221 | 601 | 55 |
| test | 6 | 3,152 | 3,110 | 1,028 | 54 |
| **合计** | **30** | **24,548** | **24,338** | **4,330** | **343** |

（free 计数为转移起点行标签；support 类别同步提供：double/singleL/singleR/none。）

## 4. 核心验证结果

- **MuJoCo 积分验证（全部 24,548 转移实测）**：H1 半隐式 Euler（`v'=v+dt·a; q'=mj_integratePos(q,v',dt)`）复现记录的下一状态，max |Δq| = **7e-9**（train）/6.5e-9（val）/4.7e-9（test），free/contact/impact 各类一致；显式 Euler 与 naive 分量更新的最大误差 ~7e-2（**差 7 个数量级**）；`mj_integratePos` 后四元数模长保持 1.0（12 位）。**结论：`a_t` = 区间起点瞬时加速度，被 `mj_Euler` 半隐式消费；v 重建残差 3.5e-6 由约束求解器二次收敛容差决定（与 Task 1 观测一致），不是积分规则误差。**
- **连续性**：30 episodes 全部 dt=1e-3 无重复/跳变；issues = NONE。
- **快照一致性**：passive/applied 精确 0；`actuator_force≡ctrl`（≤1.9e-6）；`qfrc_actuator` root 6 维精确 0。
- **Free-Dynamics residual 定义未动**；数据级核验已在 Stage1-01 Oracle 完成（4,333 free 样本，f64 归一化 RMS 8.6e-8），本任务引用不重复。

## 5. 未解决的技术问题

1. `a_t` 的求解器容差级不确定度（~3.5e-6 rad/s²，占 free 样本典型加速度的 <0.1%）：来源是 MuJoCo step 内二次求解与采集快照一次求解的差异。对 B/C/D 模型是标签噪声下限，无法在现有数据上消除；如需消除须改采集协议（step 后即取值），属 Stage2 决策。
2. Episode Context 中**控制器增益的逐 episode 随机化**（PD ×U(0.8,1.2)）未逐 episode 显式存储——可由 seed+采集代码复现（记录在案），但 ξ_e 中不含它；世界模型若需要控制器上下文须从 seed 重放提取。
3. 8 步窗口（8 ms）只检验极短程一致性；长 horizon 能力需后续任务自行调 `horizon` 参数（接口已支持）。

## 6. 对 B 模型（SuperMimic-style 数据驱动）的直接支持

- 输入输出即拿即用：`(q_t, v_t, ctrl_t) → (q_{t+1}, v_{t+1})`，无需任何差分/四元数处理（`q_next/v_next` 已物化）；
- 归一化钩子沿用 `stage1_data/norm_stats.json`（train-only），特征化（6D 旋转等）可复用 `stage0_common.build_features`；
- 按动作来源/接触状态分层评价所需的全部标签已随样本提供（free/support/impact + episode_id）；
- 不依赖任何特定网络结构（返回 numpy dict）。

C/D 模型额外获得：逐样本 `M_upper_t / qfrc_bias_t / qfrc_constraint_t`（PINN residual 与 hybrid 约束项）与 ξ_e（物理参数上下文）。

## 7. 验收条件核对

| 交付要求 | 状态 |
|---|---|
| Stage1 字段盘点报告 | ✅ 本报告 §1.1 + transition_inventory.json |
| 数学与接口设计 | ✅ wm_data.py 模块 docstring + README §1–3 |
| 构建代码 | ✅ build_transitions.py |
| 单步 Transition 数据 | ✅ 24,548 条（3 split npz） |
| 8 步 Window 数据 | ✅ 24,338 条（3 split 索引 npz + API） |
| 完整性验证报告 | ✅ transition_checks.json（issues NONE） |
| MuJoCo 积分验证报告 | ✅ integration_verification.json（H1 精确，7e-9） |
| TRAIN/VAL/TEST 分布 | ✅ 本报告 §3 |
| API 使用示例 | ✅ README §2（已冒烟运行） |
| README/技术文档 | ✅ world_model/README.md |

**K1_WM_01 验收：满足。任务完成，停止；等待 Owner 独立评审后再进入 K1_WM_02。**
