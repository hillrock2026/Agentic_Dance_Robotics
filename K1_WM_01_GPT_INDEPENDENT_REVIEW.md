# K1_WM_01 — GPT Independent Technical Review

**Review date:** 2026-11-24  
**Scope:** K1_WM_01 dataset construction, transition/window API, floating-base integration checks, and compatibility only.

## A. 总体判定

**PASS WITH QUALIFICATIONS**

K1_WM_01 已按 episode 内相邻样本正确构造 24,548 条单步转移和 24,338 个 8 步窗口；API 可实际加载；窗口、split、episode 和标签对齐无误。`qacc` 是在区间起点由 `mj_forward` 得到的独立动力学求解结果，并非从未来速度反算；半隐式 Euler 与 MuJoCo `mj_integratePos` 的使用在数学和工程上正确。

但必须修正报告中的一项实质数值解释：验证中存储的 `dt` 被转换为 float32，造成约 `3.5e-6` 的速度残差和约 `7e-9` 的位置残差。它们不是已证明的“约束求解器二次收敛误差”。使用 `time_next-time_t` 的原始 float64 时间戳后，所有转移的速度/位置重建误差降至约 `1e-13/1e-15`。这不改变数据接口和状态转移定义成立，但必须修改积分验证代码、结果和措辞后再将报告作为最终冻结证据。

## B. 六个方面的评审结论

### 1. 原始数据真实性

**通过。**

- 30 个 episode 的真实字段均为 `qpos(29)`, `qvel(28)`, `ctrl(22)`, `qacc(28)`, `M_upper(406)`；`qfrc_bias/qfrc_actuator/qfrc_constraint/qfrc_passive/qfrc_applied` 均为 28 维，`actuator_force` 为 22 维。
- XML 实际为 `nq=29, nv=28, nu=22, na=0`；模型可没有 `act` 激活状态，但 `actuator_force`、`qfrc_actuator` 仍然真实存在。`act` 缺失并不意味着 actuator 输出不存在。
- 四元数位于 `qpos[3:7]`，全数据模长误差仅机器精度（约 `±1e-15`）。
- 22 个 joint torque motor 的 gear 为 1，transmission 是各自 hinge DOF；因此 `qfrc_actuator[:6]=0`，`qfrc_actuator[6:]=actuator_force`。
- 对本模型，`actuator_force = clip(ctrl, actuator_forcerange)`。全部记录 ctrl 都在 XML 的 force range 内，故记录数据中 `actuator_force` 与 `ctrl` 相等到存储精度；root 6 维 qfrc actuator 为 0。
- “ctrl→qfrc_actuator 同快照、无延迟”仅表示 XML motor 的**静态 transmission/clipping 映射**在同一次 `mj_forward` 快照内成立。它不是对真实电机动力学、延迟或动态执行器响应的证明。
- Free mask 仍是原始定义：`ncon==0 && nefc==0 && ||qfrc_constraint||∞==0`。约束、支持与 impact 标签来自既有 Stage1 labels。
- `a_t` 来自采集时区间起点的 `d.qacc`：`d.ctrl <- ctrl; mj_forward; snapshot; mj_step`。它不是相邻速度差分生成的训练标签。
- Episode context 的 XML SHA256、dimensions、mass、gravity、timestep、armature、forcerange、gear、friction、solver 等可从当前 XML 实际提取，且与 Stage1 manifest 一致。随机化的控制器增益并不在当前 context 中；报告已承认这一点。

### 2. Transition Dataset 接口

**通过，但存在接口一致性细节。**

实际调用 `StateTransitionDataset` 和 `RolloutWindowDataset` 可运行，返回：

- 单步：q/v/ctrl/a、q_next/v_next、dt、原始 `step`、episode ID/index、ncon/nefc、Free/support/impact；
- 窗口：9 个时间戳、8 个 ctrl/acceleration/label、8 个下一状态，跨度准确为 8 个时间间隔；
- split 为 TRAIN/VAL/TEST，长度分别为 16,147/5,249/3,152；8-step 长度分别为 16,007/5,221/3,110。
- `episode_id`、`step` 可追溯到原始 NPZ。

文档中的训练/推理字段表是合理的。需要注意：`include_dynamics=False` 只关闭 `M_upper/qfrc_bias/qfrc_constraint`，仍返回 `a_t`、actuator/constraint/contact 相关观测和标签；它不是严格的“推理输入子集”构造器。`episode_context` 成员也只初始化为 `None`，实际 context 需通过 `episode_context()` 单独获取。

另外，物化 NPZ 使用字母序 episode order，而 Python API 使用 `splits.json` 的 canonical order；两者的 `episode_index` 数值不等价。两条路径各自包含有效的 `episode_names`/episode ID，可追溯，也未造成数据错位，但不宜跨接口直接交换 index。`M_upper_t` 在物化文件与 API 中均被静默转换为 float32，max 差约 `3.66e-7`；`RolloutWindowDataset` 对 horizon 也缺少正整数校验，0 会生成空序列，负 horizon 会形成无效索引，过大 horizon 则静默返回空集。

### 3. 浮动基座状态积分

**通过。**

状态布局正确：

- `qpos`: root position 3 + quaternion wxyz 4 + 22 hinge positions；
- `qvel`: root linear velocity 3（world）+ free-joint angular velocity 3（MuJoCo convention）+ 22 joint velocities；
- 位置更新没有错误地做 `q + dt*v`，而使用 `mj_integratePos(q, v_next, dt)`；
- 半隐式顺序为 `v_next=v_t+dt*a_t` 后积分 q，符合该 XML 的 Euler integrator。

独立重播全部 24,548 条原始转移时，从 `q_t,v_t,ctrl_t` 设置状态，执行 `mj_forward` 后得到的 `qacc` 与记录值完全一致；继续执行 `mj_step` 后的 `qpos/qvel` 与下一记录状态完全一致（双精度存储下最大差为 0）。这验证了字段映射、浮动基座坐标和 episode 采集协议。

### 4. 积分验证独立性

**有条件通过：机制可信，但现有误差归因错误。**

验证输入本身只用了 `q_t,v_t,a_t,dt` 来重建下一状态，没有把 `q_next/v_next` 放入预测器；目标仅用于误差比较。但官方脚本只把记录的 `a_t` 代入 Euler，再检查积分位置，未重新从 `(q_t,v_t,ctrl_t)` 经 `mj_forward` 计算加速度。因此，它对“Euler 积分接口”是直接验证，对“qacc 物理真实性/起点语义”的支持只是依赖采集器代码溯源，后者并非由这个验证脚本独立证明。独立从原始 `(q,v,ctrl)` 全量重放 `mj_forward` 后，`qacc` 最大差为 0，补足了这一独立性缺口。

验证代码从物化 NPZ 读取的 `dt` 已在构建脚本中执行 `np.diff(time).astype(np.float32)`。float32 的 `0.001` 实际为：

`0.0010000000474974513`

该相对误差约 `4.75e-8`。乘以高速 impact/free-frame 加速度（最大重建案例约 `7.38e4`）恰好产生约 `3.50e-6` 的速度误差。其位置误差约 `7.0e-9` 也是该 dt 偏差传播的结果。

使用原始 float64 `time_next-time_t` 后：

- train max v error: `1.812e-13`；q error: `1.332e-15`；
- val max v error: `3.553e-14`；q error: `1.332e-15`；
- test max v error: `4.264e-14`；q error: `1.332e-15`。

因此报告所述“3.5e-6 由约束求解器二次收敛容差决定”没有数值证据支持，是对 float32 dt 量化误差的错误归因。MuJoCo 求解器确实可能存在数值容差，但本验证没有量化出这种贡献。

该验证也只证明：

1. 记录的 `a_t` 与 MuJoCo 动力学求解快照一致；
2. 已记录状态转移满足 Euler + `mj_integratePos`；
3. 接口布局和 dt 在修正精度后可重建记录数据。

它不证明 World Model 的未来预测精度，也不证明连续时间动力学、长 horizon 或真实机器人执行器响应精度。

### 5. 数据连续性与统计正确性

**通过。**

独立复核：

| split | episodes | 单步转移 | 8-step windows | Free（起点标签） | Impact（起点标签） |
|---|---:|---:|---:|---:|---:|
| TRAIN | 20 | 16,147 | 16,007 | 2,701 | 234 |
| VAL | 4 | 5,249 | 5,221 | 601 | 55 |
| TEST | 6 | 3,152 | 3,110 | 1,028 | 54 |
| **Total** | **30** | **24,548** | **24,338** | **4,330** | **343** |

- 原始样本总数 24,578；单步数为 `sum(T_e-1)=24,578-30=24,548`。
- 8-step 数为 `sum(T_e-8)=24,578-30×8=24,338`，与 8 个区间正确对应。
- 所有窗口 key 唯一；每个窗口的 `time_seq` 为 9 个点、8 个约 1 ms 间隔。
- Episode 边界和 split 边界均构造性排除；边界终点索引正确。
- 30 个 episode 均无重复时间戳、>2 ms 跳变或非有限 q/v/qacc/ctrl。
- `Free/support/impact` 是转移起点 `t` 的标签；对窗口而言，`*_seq` 是逐步标签。整个窗口并不保持同一种状态。
- support（起点）分布为 double 15,520、single 4,167、none 4,861，合计 24,548。
- `issues=NONE` 仅涵盖本脚本的 timestamp/nonfinite/passive-applied 检查，不表示动力学 residual 完美闭合；Stage1 记录的最大相对 closure 约 1.94%。此外 `support` 是 2 mm 几何代理，未必等同物理接触真值，`impact` 是启发式标签，`transition_valid=True` 是无条件标记。

### 6. 非侵入性与兼容性

**通过，基于代码路径检查。**

- [build_transitions.py](tools/pinn/world_model/build_transitions.py) 只读取 `stage1_data`，仅写入 `world_model/transitions/` 和 `world_model/episode_context.json`。
- [verify_state_integration.py](tools/pinn/world_model/verify_state_integration.py) 只读取 transition NPZ/XML，仅写 `world_model/integration_verification.json`。
- 未发现对 Stage1 NPZ、`stage0_common.py`、PINN residual、既有训练/评价脚本或 `stage1_runs` 的写入。
- 新 API 直接从 Stage1 raw files 读取，不要求替换既有 PINN loader，因此兼容。
- README 的基本构造和读取示例可实际执行。由于仓库本目录不是 Git 仓库，无法以 VCS diff 证明历史产物字节未变；但所有 K1_WM_01 写路径均被实际审计，未发现覆盖路径。

## C. 关键数值独立复算

### 物化数据

- Transition：**24,548**；Window H=8：**24,338**。
- `dt` 原始时间戳范围：约 `0.00099999999999989`–`0.001000000000000112`。
- 物化 float32 `dt`：恒为 `0.0010000000474974513`。
- API `StateTransitionDataset.dt` 直接返回 float64 timestamp difference，因此保留原始时间戳精度；物化 NPZ 的 `dt` 不保留。

### 官方验证（已复现）

| split | 官方最大 v error | 官方最大 q error | 官方最大四元数测地角误差 |
|---|---:|---:|---:|
| TRAIN | `3.504095943e-6` | `6.998254976e-9` | 约 `4.21e-8 rad` |
| VAL | `1.839936232e-6` | `6.490548543e-9` | 约 `4.21e-8 rad` |
| TEST | `2.435472012e-6` | `4.745885462e-9` | 约 `4.21e-8 rad` |

### 使用原始 float64 dt 独立重算

- TRAIN：v max `1.812e-13`，q max `1.332e-15`；
- VAL：v max `3.553e-14`，q max `1.332e-15`；
- TEST：v max `4.264e-14`，q max `1.332e-15`；
- quaternion norm 与参考目标的测地角误差在上述双精度精度下为 0。

### 代表性最差样本

- TRAIN `ou_o4`, step 221, DOF 10：物化 dt 产生 v error `3.5041e-6`；原始 dt 下 q max error `2.22e-16`。
- VAL `ou_o1`, step 134, DOF 15：物化 dt 产生 v error `1.8399e-6`；原始 dt 下 q max error `1.39e-17`。
- TEST `ou_o0`, step 184, DOF 15：物化 dt 产生 v error `2.4355e-6`；原始 dt 下 q max error `4.44e-16`。

## D. 发现的问题

### MAJOR 1 — 积分残差被错误归因于约束求解器

- **文件/位置：** [build_transitions.py](tools/pinn/world_model/build_transitions.py) 中 `dt_c.append(np.diff(...).astype(np.float32))`；[verify_state_integration.py](tools/pinn/world_model/verify_state_integration.py) 的 `dt` 加载/结论；[K1_WM_01_REPORT.md](K1_WM_01_REPORT.md) §4/§5。
- **证据：** float32 dt 为 `0.0010000000474974513`，而原始 timestamp difference 为约 `0.0010000000000000000`；最大 `a*Δdt` 恰为约 `3.50e-6`。改用原始 float64 dt 后，误差下降约 7 个数量级且 q 误差降至 `1e-15`。
- **影响：** 不影响转移定义和 raw API，但现有 `integration_verification.json` 的解释与报告中的“solver tolerance”结论不可信；物化数据用户也会引入可避免的 dt 量化误差。
- **最小修正：** 将物化 `dt` 保存为 float64，验证器优先用 `time_next-time_t`，更新 JSON/report；删除未被实测证明的 solver attribution，或另做专门求解器实验再提出。

### MINOR 1 — materialized NPZ 与 API 的 episode_index 不一致

- **文件/位置：** [build_transitions.py](tools/pinn/world_model/build_transitions.py) 对 `splits[split]` 执行 `sorted()`；[wm_data.py](tools/pinn/world_model/wm_data.py) 直接使用 `splits.json` 顺序。
- **证据：** TEST API 首个 episode 为 `pd_top_rock`，而 TEST NPZ `episode_names[0]` 为 `drop_d0`；三个 split 均不满足同序。
- **影响：** 单一路径均可通过 episode_names/episode_id 正确追溯，但跨 NPZ 与 API 直接交换 `episode_index` 会解释错误。
- **最小修正：** 两条路径统一使用同一个 split canonical order，或在 README 明示 index 空间不同并提供 episode-ID 映射。

### MINOR 2 — `Episode Context`/训练字段开关的接口语义可更明确

- **文件/位置：** [wm_data.py](tools/pinn/world_model/wm_data.py) 的 `episode_context = None` 和 `include_dynamics` 分支。
- **证据：** context 需单独调用；`include_dynamics=False` 仍返回训练期 `a_t` 和多个动力学/标签字段。
- **影响：** 不导致数据错误，但用户可能把“attached context”和“inference-only sample”理解成已自动生效。
- **最小修正：** 明确属性填充/返回协议，并将字段分组命名为 `state_only`、`dynamics`、`labels` 等；或仅在文档限定现有开关行为。

### MINOR 3 — API 精度与 horizon 校验

- **文件/位置：** [wm_data.py](tools/pinn/world_model/wm_data.py) 的 `M_upper_t` 转换、`RolloutWindowDataset.__init__`；[build_transitions.py](tools/pinn/world_model/build_transitions.py) 的 `M_upper` 转换。
- **证据：** M_upper 从原始 float64 转为 float32 的最大变化为约 `3.66e-7`；horizon 未要求正整数，非法值不会抛错。
- **影响：** 不改变当前 H=8 counts，但会影响高精度动力学消费，并使 API 的“任意 horizon”弱于声明。
- **最小修正：** 保留 `M_upper` 原始 float64 或明确声明精度；在构造器中拒绝 `horizon <= 0` 以及超过最短 episode 长度的配置（或明确返回空集语义）。

### MINOR 4 — actuator mapping 与“无延迟”措辞需限定

- **文件/位置：** [README.md](tools/pinn/world_model/README.md) 和 [K1_WM_01_REPORT.md](K1_WM_01_REPORT.md)。
- **证据：** XML 是 motor/gear=1 静态 transmission，本数据 ctrl 均未越过 forcerange；数据只支持同一 `mj_forward` 快照的静态映射。
- **影响：** 当前结论若被读作“真实执行器动态响应无延迟”会过度外推。
- **最小修正：** 改为“同快照静态 actuator/control mapping，未评估电机动态与硬件延迟”。

## E. 最终验收结论

1. **K1_WM_01 是否满足原始任务要求？**  
   核心数据构造、接口、浮动基座积分、episode 隔离、连续性和可追溯性满足要求；判定为 **PASS WITH QUALIFICATIONS**。

2. **是否存在必须修复的问题？**  
   存在一个必须修正的主要问题：验证用 `dt` 的 float32 量化与错误的 solver 归因，以及相应 JSON/report 文本。它需要很小的代码/文档修复和重跑验证，不需要重采数据或重构数据集。

3. **哪些结论可以直接采信？**  
   - 24,548 单步、24,338 窗口及 split/free/impact/support 统计；
   - episode/split/window 不交叉、8 个区间和 control 对齐正确；
   - qacc 为区间起点 `mj_forward` 求解标签，不是未来速度反推；
   - H1 半隐式 Euler + `mj_integratePos` 是正确状态积分接口；
   - 使用原始 float64 时间戳可重建记录下一状态至机器精度；
   - 新接口对原始 Stage1/PINN 流程是非侵入的。

4. **哪些结论需要附加限定？**  
   - “3.5e-6 是 solver tolerance”应删除；现有证据指向 dt float32 量化；
   - “ctrl 无延迟”只能表示静态 motor mapping，不能表示真实执行器动态响应；
   - 状态重建验证不等于未来状态预测精度或连续动力学模型精度；
   - context 是共享模型常量，不包含逐 episode PD gain；
   - materialized NPZ 与 Python API 的 episode index 空间当前不同。

5. **是否可以关闭 K1_WM_01？**  
   在完成上述 MAJOR 1 的最小修正并更新验证产物后，可以关闭。无需重新采集 Stage1 数据、重做 PINN 实验或调整 State Transition 的数学定义。
