# VLN_AGENT_KERNEL_INDEPENDENT_REVIEW_01
# Agent-Kernel 独立代码评审与 Booster K1 集成决策

- 评审日期：2026-11-19（REV 2，同日修订）
- 评审对象：VLN-AVP 分支 `cap-12-03-05-04-01-ground-plane-e2e`（快照 commit `7c2c4a4`，本地目录与分支 zip 逐字节比对一致）
- 重点模块：`Agent-Kernel/`（C++20，444 个文件；生产代码约 20K 行：src 8,519 行 cpp + include 11,524 行 hpp；测试 99 个 cpp / 18,773 行；`runtime/` C 运行时另有 4,269 行）
- 评审方式：源码逐模块审读 + 全量构建 + 官方测试套件 + 独立编写最小集成验证程序（未修改原仓库，未连接任何真实机器人，未执行任何训练任务）

**REV 2 修订说明**：REV 1 假设 Agent-Kernel 为第三方依赖，据此将"License TBD + 无 git 基线"列为集成阻塞。经作者确认，**Agent-Kernel 与 K1 项目同属一个所有者**，开发任务为双目标：既完成 Booster K1 Agentic Dance Robotics，也以 K1 为驱动场景逐步完善 Agent-Kernel。上述两条从"阻塞"降级为"协同开发前的治理待办"；集成决策由"D 暂不引入"修订为**"双轨协同、分阶段接管"**（见第 8/10/11 节）。第 2-7 节的全部代码证据与技术判定不受修订影响。

---

## 1. 总体结论

**Agent-Kernel 是一个工程纪律非常严格、但"智能体能力"几乎全部未实现的 C++20 架构骨架。它不是一个可用的 Agent Runtime，更不是 Agent Kernel；它甚至还不是"LLM + Tool Calling 封装"——因为其中没有任何 LLM。**

三句话概括：

1. **已实现且质量高**：领域建模（强类型 ID、生命周期状态机）、Tool 注册/发现/准备服务、C Runtime 工作流引擎（Session→Workflow→Task DAG→Scheduler→Resolver→Executor→EventBus）、SQLite Memory Provider、36/36 绿色测试、警告即错误的干净构建。
2. **仅有接口或测试替身**：Model 推理（纯抽象类，项目自述"Rev1 has no production concrete Model"）、Planner（把 goal 字符串原样返回的 ImmediatePlanningEngine）、工具权限（永远返回 kAllowed 且无执行路径调用）、Agent 协调（仅握手应答）。
3. **完全没有**：生产 Agent 循环（FWA 用注入的 `std::function` 测试缝、失败即终止、无重试）、任务/会话持久化与恢复（仅 Memory 对象可持久化）、异步/长时任务执行（CapabilityInvoker 是同步 string→string）、多会话隔离、真实 LLM 输出的结构化解析（协议是纯文本 `tool:<cap>:<args>`）。

**集成决策（REV 2）：双轨协同、分阶段接管 —— K1 项目作为 Agent-Kernel 的驱动场景与验收测试台，Agent-Kernel 按里程碑逐步接管 K1 的智能体运行时职责，直至成为 K1 Agent 的正式 Runtime。** 理由：K1 需要的正是 Agent-Kernel 缺失的部分（LLM 接入、真实规划、带历史的多轮循环、长时异步实验编排、状态持久化）——在双目标语境下，这不再是拒绝理由，而恰恰是 Agent-Kernel 的工作清单（Workstream P/R/C + 若干设计修正），且 K1 的 Arm Wave 任务恰好为每一项提供了真实的验收场景。语言鸿沟（C++ vs Python）通过既有的服务边界模式（kernel_host 的 NDJSON IPC）消化，K1 关键路径在任何阶段都不被 C++ 建设阻塞：Python 先行定标（作为 Workstream C 的可执行规格），C++ 按里程碑接管。Agent-Kernel 的架构文档和边界纪律在过渡期内直接作为 K1 智能体层的设计参照。

**REV 1 结论（已被上述修订取代，保留供对照）**：D——当前不建议引入；理由为 License TBD、无 git 基线、智能能力缺失、语言鸿沟。其中前两条经所有权确认后不再成立，后两条转化为分阶段工作项。

---

## 2. Agent-Kernel 架构分析

### 2.1 分层结构

```
include/agent_kernel/ + src/          领域层（C++20 库）
  foundation/   强类型 ID、生命周期验证器、注册表、诊断
  agent/        Agent 状态机 + 四子系统契约（Memory/Context/Planner/Tool）
  memory/       Memory 聚合根 + SQLite Provider（可选编译，默认 OFF）
  context/      Context 对象存储（纯内存）
  planner/      Planner 边界 + ImmediatePlanningEngine
  prompt/       模板化 Prompt 组装（变量/段落/替换）
  model/        纯抽象 Model 契约（无任何实现）
  tool/         Tool 聚合根 + 8 个服务接口（注册/发现/权限/准备…）
  fwa/          First Working Agent 组合（测试缝驱动的有界循环）
  model_to_runtime/   ModelResult→ModelAction 文本解释器
  runtime_backed_agent/  单步"决策→Runtime 调用→观察→终止"组合
  agent_continuation/    恰好一次续决策组合
runtime/                              C Runtime（scheduler/executor/resolver/eventbus/lifecycle）
VLN-AVP/runtime_host/                 kernel_host.cpp（2,412 行）NDJSON IPC 宿主
VLN-AVP/memory_host/                  SQLite Memory 命令行宿主
```

### 2.2 Agent 本体是什么

`ak::agent::Agent`（`src/agent/agent.cpp`，157 行）是一个**生命周期状态机**（Uninitialized→Ready→Coordinating→Supervising→Suspended/Terminated）加一个 Objective 登记表。`BeginCoordination()` 只调用四个子系统契约的 `AcknowledgeCoordination()` 应答回调——注释原话："Phase 1 acknowledgments only; no subsystem behavior is implemented by the Agent"（`agent.cpp:60-62`）。**Agent 类本身不执行任何任务。**

### 2.3 真正的"循环"在哪里、有多深

- `FirstWorkingAgentComposition`（`src/agent/composition/first_working_agent_composition.cpp`）：StepCycle = 读 Context（固定 ID=1 的单对象）→ 读 Memory（固定 ID=1 的单条目）→ Planner → Prompt 组装 → **注入的 ModelBoundary（std::function 测试双）** → 解释动作 → 工具准备 + **注入的 ToolExecutionBoundary（std::function 测试缝）** → Context 覆盖更新。`Run()` 上限 `max_cycles=8`（`first_working_agent_configuration.hpp:65`）。
- `AgentContinuationComposition`（头注释 23-33 行）自述边界："EXACTLY ONE subsequent decision-generation … No while(true), no retry, no recovery, no persistence, no async/background"——**最深 2 次决策，硬编码**。
- 失败语义：工具未找到、准备失败、执行器抛异常、执行器返回空 → `FailCycle` 直接终止循环，**无重试**（`first_working_agent_composition.cpp:264-305`）。

### 2.4 项目自评（计划书 §2 "Current Baseline" 第 3-6 条）

- "First Working Agent provides a bounded loop, but uses test-scoped Model and Tool execution seams and does not use Runtime."
- "production wiring exists only in tests and continuation is limited to one opportunity."
- "**No production Model provider or production Agent executable exists.**"
- "Runtime is currently one-shot."

README 路线图（103-104 行）明确：Workstream P（生产 Model Provider）、R（Runtime 多次调用）、C（生产组合）在 Workstream B 关闭后**才开始**，即全部未开工。本评审的源码结论与项目自评一致。

---

## 3. 核心模块代码分析

### 3.1 Tool System

| 检查项 | 状态 | 证据 |
|---|---|---|
| 注册机制 | 已实现 | `Tool::Register()/Activate()` + `IToolProvider` 注入（`tool.hpp:80-95`）；Runtime 侧 `Runtime::RegisterCapability(name, provider, invoker)`（`kernel_host.cpp:1948` 用法） |
| 发现机制 | 已实现 | `IToolDiscoveryService::ListCapabilities/FindCapability`；FWA 每轮 `BuildToolSummary()` 把能力 ID 拼进 Prompt |
| 参数 Schema | **仅声明** | `CapabilityInputContract{schema_id, format, description}`——schema 是描述性字符串，**执行路径零校验**，payload 是裸 string 透传 |
| 结果处理 | 部分实现 | Runtime 侧 string→string；FWA 侧要求非空否则判失败 |
| 错误处理 | 已实现（同步路径） | Executor 捕获异常→kFailed；未注册能力→确定性失败（本评审 T3/T4 实测验证） |
| 权限控制 | **仅接口** | `IToolPermissionService::Check` 实现恒返回 `kAllowed`（`tool.cpp:643-665`），且 FWA/Executor/model_to_runtime 无任何调用点 |
| 并发/异步/超时/取消 | **完全没有** | `CapabilityInvoker = std::function<std::string(const std::string&)>`（`capability_binding.h:35-37`），同步进程内调用 |

**K1 六类工具接入判断**：Motion Processing、Skill Registry、RL Training、Policy Export、MuJoCo Runtime、Evaluation Engine 全部是 Python 生态。接入 Agent-Kernel 只有两条路：(a) 为每个工具写 C++ Provider + C++ 实现（不可行）；(b) 像 VLN-AVP 那样写一个 2,400+ 行的 C++ IPC 宿主进程把请求转发回 Python（可行但纯增复杂度——VLN-AVP 的 `parking.move` 默认也只是"echo 回去"的模拟桩）。尤其 RL Training 是小时级任务，同步无超时的 invoker 模型需要先自建进程隔离/异步层，而该层不存在。

### 3.2 Memory 与 Context

- **SQLite Provider：真实实现**（`memory_provider_sqlite.cpp`，686 行，CREATE TABLE/UPSERT/LOAD 齐全；本评审 T7 实测跨实例持久化通过）。但为 opt-in 编译（`AK_BUILD_SQLITE_MEMORY_PROVIDER` 默认 OFF），且仅持久化"记忆对象"，不持久化任务/会话/上下文。
- **Context：纯内存、单对象、覆盖写**。`ContextUpdateService::ApplyUpdate` 是 `object->content_ = std::move(content)`（`context_services.cpp:166`）——**每次工具观察替换先前内容，无历史追加**。3 步以上的工具链（改配置→训练→导出→Sim2Sim→分析）到第 3 步就丢掉第 1 步的观察。这是对"长期 Dance Skill 开发保留工程上下文"目标的直接否定。
- **任务中断恢复**：不存在。kernel_host 的"持久会话"是进程内全局 map（`kernel_host.cpp:967` `g_sessions`），宿主重启即失。
- **多任务上下文隔离**：FWA 级别没有（单组合单目标）；Runtime Session 级别在进程内有隔离。**未验证**跨任务长期隔离（无机制可验）。

### 3.3 Planning 与 Model 链路

- **Planner**：`ImmediatePlanningEngine::Outcome`（`planning_engine.cpp:19-33`）把 `request.goal` 原样 push 为唯一计划步骤并把 goal 串接为 artifact——**是占位符，不是规划器**。VLN-AVP 实际用的"真"规划器 `TaichuPlanningEngine`（`runtime_host/taichu_planning_engine.cpp`）也只是 socket 桥到外部 Python 服务，C++ 侧是传输层。
- **Model**：`ak::model::Model` 纯抽象（`model.hpp:48-79`）；`AgentModelHost` 头注释承认 "Rev1 has no production concrete Model; the deterministic Model … is a test double"。
- **ModelResult→动作**：`InterpretModelResult`（`model_result_interpreter.cpp:41-72`）只认两个纯文本格式：`"complete"` 与 `"tool:<capability>:<arguments>"`。**没有 JSON 函数调用、没有结构化参数、没有多工具调用、没有流式。** 即使未来接入真 LLM，这一层也要推倒重写。
- **"分析 Arm Wave→改配置→RL→导出→Sim2Sim→分析→规划下一轮"链路判断：不可能。** 缺真实 LLM、缺有历史的多轮循环（Context 覆盖写 + max_cycles 硬界 + 失败即终止 + continuation 恰好 1 次）、缺长时异步任务、缺实验持久化。这不是参数调优问题，是能力不存在。

### 3.4 C Runtime（`runtime/`）

这部分是**货真价实的**：Session→Workflow→Task 依赖图→拓扑调度→能力解析→同步执行→事件总线，全链路可跑（本评审 T1/T2 实测：自定义能力注册执行、A→B→C 依赖顺序保证均通过）。但它是"一次性"的（`Runtime::Run` 要求 kCreated 状态，每请求重建整个服务图——kernel_host 注释自认），单线程，无取消/超时。

### 3.5 工程质量

优点：`-Wall -Wextra -Wpedantic -Werror` 干净构建；测试/生产代码比≈0.94；每模块有 boundary-check 测试；宿主代码注释诚实标注"Known limits (do not overstate)"。

问题：
- **License: TBD**（README:110）——REV 2 注：所有权属作者，属治理待办而非不可逾越；但仍应在双轨协同启动前落字，且无论 License 如何，源码复制式集成因分叉问题始终不做（见 §8）。
- **无 git 历史基线**（计划书自评第 7 条"no Git commit baseline; current files are untracked"）——作为依赖无法做可追溯的版本固定。
- 源码注释存在编码损坏（`閳?鎼? 搂` 等 GBK/UTF-8 混杂乱码，遍布 FWA 头/源文件）。
- `tests/integration/`、`tests/regression/` 是空目录；CHANGELOG 停留在 v0.1.0（Phase 0，"No runtime code exists"），后续数千行代码无变更记录。
- 文档驱动痕迹极重：几乎每个设计决定都引用 "TOL-xxx/INV-xxx/Chapter xx" 条款编号，规范文档体量（`docs/architecture/` 16 份 + `standards/`）与实现体量之比罕见地高。

---

## 4. 实际测试结果

环境：Ubuntu 24.04 / GCC 13.3.0 / CMake 3.28.3 / GTest 1.14.0（与 README 基线一致）。

**构建与官方测试**（`/tmp/ak_build`，原仓库零改动）：
- 配置、全量编译通过，无警告（警告即错误）。
- `ctest`：**100% tests passed, 0 tests failed out of 36** —— README 声明属实。

**独立最小集成验证**（自编程序 `/tmp/ak_verify/ak_verify.cpp`，链接分支静态库，14/14 通过）：

| 编号 | 验证项 | 结果 |
|---|---|---|
| T1 | 注册自定义 K1 风格能力并经 Runtime 执行（参数/结果回传） | PASS |
| T2 | A→B→C 依赖工作流，执行顺序与 payload 传递正确 | PASS（order=a:x,b:y,c:z） |
| T3 | 工具抛异常：异常被包含，任务确定性失败，进程不崩 | PASS |
| T4 | 调用未注册能力：确定性失败 | PASS |
| T5a-d | FWA 多轮循环：注入脚本模型，第 1 轮调用工具、第 2 轮完成；恰好消耗 2 次决策；第 2 轮 Prompt 确实携带第 1 轮工具观察 | PASS |
| T6 | 模型输出不可解释：确定性 invalid 终止 | PASS |
| T7a-d | SQLite Memory：跨 Provider 实例（模拟进程重启）记录存活 | PASS |

**明确标注为未验证/不可验证的项**：
- 真实 LLM 推理链路——**仓库中不存在任何实现，无从验证**。
- 长时异步任务（RL 训练级）——无机制。
- 任务/会话状态持久化与中断恢复——代码检查确认不存在（非"未测"，是"没有"）。
- 多会话并发隔离——宿主明确单线程，无机制可验。
- 权限拦截有效性——实现恒为放行，无策略可测。

---

## 5. 主要优点

1. **边界纪律与领域建模是一流的**：强类型 ID 防错、生命周期状态机、每个聚合根的 NON-responsibilities 注释、boundary-check 测试固化架构约束——这套方法论值得 K1 智能体层直接借鉴。
2. **Runtime 工作流引擎真实可用**：DAG 依赖、确定性调度、异常包含、事件发布，36 测试全绿且经本评审独立复测。
3. **SQLite Memory Provider 真实可用**：跨进程持久化有真实证据（memory_host 以进程边界设计验证）。
4. **工程诚实度高**：README/计划书/宿主注释均如实声明未完成项与已知限制，无夸大。
5. **提示词组装子系统**（模板变量/材料类别）结构清晰，概念上可移植。

## 6. 主要缺陷

1. **智能核心全部缺席**：无 LLM Provider、无真规划器、无带历史的多轮循环、无结构化工具调用协议——作为"Agent Kernel"名不副实（阶段定位：架构基线，非可用运行时）。
2. **对长时实验编排零支撑**：同步无超时 invoker、失败即终止无重试、无异步/取消/进度查询——与 RL 训练场景根本冲突。
3. **上下文模型过弱**：单对象覆盖写、无历史、无跨任务工程上下文；除 Memory 对象外无任何持久化。
4. **语言生态错位**：C++20 库 vs K1 全 Python 工具链（k1_action/virtual_k1/IsaacLab/MuJoCo 均为 Python），任何集成方式都要付 IPC 桥接税（VLN-AVP 为此维护 2,400 行宿主 + 协议层 + Python 客户端）。
5. **治理待办（所有权属作者本人，非集成阻塞，但须在双轨协同启动前完成）**：License 待定（作者可直接指定，建议尽早落字以免未来协作歧义）；无 git 提交基线（作者自己的计划书第 7 条已列为待办，init + 打 tag 即可）；CHANGELOG 自 v0.1.0 后失修。
6. 规范文档体量远超实现体量，存在"以文档代实现"的失衡风险（评审纪律第 5 条提醒的反面：复杂 ≠ 优秀，此处是文档复杂）。

---

## 7. K1 项目适配性

逐项对照第 3 节 A/B/C 三类未来场景：

| K1 需要的能力 | Agent-Kernel 现状 | 判定 |
|---|---|---|
| 高层目标理解 | Model 抽象，无实现 | 需自建 |
| 任务规划 | goal 原样返回的占位 Planner | 需自建 |
| 工具调用 | Runtime 能力调用可用，但同步 string 协议、无 schema 校验、无权限 | 部分可借鉴，不可直接用 |
| 实验编排（RL 训练小时级） | 无异步/重试/进度/取消 | 完全缺失 |
| 长期项目状态 | 仅 Memory 对象 SQLite 持久化 | 严重不足 |
| 自动分析与迭代决策 | 循环深度≤8（FWA）/2（continuation），Context 覆盖写 | 不可用 |

**结论（REV 2 视角）：以"拿来即用"衡量，Agent-Kernel 当前能直接复用的代码资产接近于零（SQLite Provider 用 Python 十行可替代）；以"长成内核的地基"衡量，其难替换的底层件（Runtime 工作流引擎、能力契约模型、边界纪律、NDJSON 宿主模式）恰好是值得保留并向上生长的基座——缺失的智能层全部可在 K1 验收场景驱动下增量补齐（见 §11）。**

---

## 8. 集成方式比较（REV 2 修订）

前提变化：Agent-Kernel 与 K1 同属一个所有者，源码级集成的法律/治理障碍消失，剩下的纯粹是工程取舍。双仓库继续独立演进、以服务边界对接仍是技术上的最优解，但理由从"风险规避"变成了"演进自由度"。

| 方案 | 评估（所有权属作者的前提下） | 结论 |
|---|---|---|
| A：直接复制源码进 K1 仓库 | 仍然制造永久分叉：K1 侧副本会随本仓库演进而僵化，双目标（完善 Agent-Kernel）恰好要求单一演进源；复制即背离双目标 | **否决** |
| B：Git Submodule | **变得可行且合理**——但仅限 C++ 侧：未来泛化出的 K1 能力宿主（k1_agent_host）是 C++ 工程，submodule 固定 Agent-Kernel 的里程碑 tag 即可；前提是先建立 git 基线并按里程碑打 tag | **限 C++ 宿主构建采用** |
| C：独立 Python Package | 仍不成立（Agent-Kernel 是 C++ 项目）；若做 pybind11 封装，封装对象应等 Workstream P/C 落地后再评估 | **暂不适用** |
| D：独立 Agent Service | **K1↔Agent-Kernel 的主边界**。kernel_host 已验证的 NDJSON 模式（含协议版本号 protocol/v1）直接泛化复用；进程边界天然解决语言鸿沟、长时任务隔离、崩溃隔离——这三点正是 K1 场景最需要的 | **采用，作为唯一 K1 侧集成形态** |

**最终形态：Agent-Kernel 独立仓库 + 里程碑 tag（治理待办先补齐）→ K1 的 C++ 能力宿主以 submodule 固定版本 → K1 Python 侧只通过 NDJSON 协议与宿主对话，K1 仓库内不出现任何 Agent-Kernel 源码。** 这样 Agent-Kernel 的每次完善（Workstream P/R/C）都能独立发布、被宿主按 tag 消化，K1 侧仅在协议变化时升级。

---

## 9. 推荐架构与分工（问题五回答）

职责边界（Agent 不碰机器人学，Framework 不碰自主决策）：

```
┌─────────────────────────────────────────────────────────────────┐
│  K1 Dance Agent Layer（Python，K1 仓库内自建，Virtual-First）      │
│  - 目标理解 / 任务分解 / LLM 函数调用规划                          │
│  - 实验编排循环（改配置→训练→导出→Sim2Sim→分析→下一步）             │
│  - 实验注册表 + 长期记忆（文件/SQLite：实验记录、成功失败经验）        │
│  - 工具协议：JSON Schema 声明 + 结构化结果（借鉴 AK 的              │
│    CapabilityInputContract/OutputContract 概念与边界纪律）          │
└──────────────┬──────────────────────────────────────────────────┘
               │ 唯一通道：显式工具调用（进程/子进程边界）
┌──────────────▼──────────────────────────────────────────────────┐
│  K1 Dance Skill Framework（Python，独立模块，机器人学归属地）        │
│  - Skill Definition / Skill Registry                             │
│  - Motion Interface（BVH/NPZ→K1 重定向，含 motion_studio/k1_action）│
│  - Robot Interface（K1 模型/关节限位/执行器参数）                    │
│  - RL Learning Engine（训练配置、训练任务、检查点管理）               │
│  - Policy Export / MuJoCo Sim2Sim Runtime                         │
│  - Evaluation Engine（跟踪误差、Root 漂移、稳定性指标）              │
└──────────────┬──────────────────────────────────────────────────┘
               │ Agent 层过渡期先在 Python 实现；其工具 Schema、
               │ Prompt 模板、重试策略、验收标准即 Workstream C 的
               │ 可执行规格，按里程碑（AK-M1..M4）迁入 Agent-Kernel
┌──────────────▼──────────────────────────────────────────────────┐
│  （中远期）Agent-Kernel 能力宿主服务（C++，NDJSON，协议带版本号）     │
│  - 按 AK-M1..M4 接管：结构化工具契约与校验、LLM Provider（P）、      │
│    长时任务与持久化（R）、生产循环（C）                              │
│  - 前提：git 基线 + 里程碑 tag + License 落字（治理待办，见 §8）      │
│  - 机器人交付后，同一宿主即机载执行权威（Virtual-First 的自然终点）    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 最终问题回答

**问题一（技术成熟度）**：介于 1 和 2 之间，但偏移方向特殊——它连"LLM 调用封装"都不是（无 LLM）。准确定性：**一个域建模严谨、运行时工作流引擎可用的"Agent Runtime 骨架"，智能体能力（模型/规划/循环/记忆运用）全部处于接口或测试替身状态**。项目自评与源码一致，无虚假宣传，但离"Kernel"名称所需的能力还差整个 Workstream P/R/C。

**问题二（是否适合作为 K1 智能体核心，REV 2 修订）**：**作为"现成的核心"不适合（现状判定不变）；作为"要长成的核心"适合且是合理选择**——其可用的地基（Runtime 工作流引擎、能力契约模型、SQLite 持久化、边界纪律、NDJSON 宿主模式）恰好是难替换的底层件，缺失的智能层（LLM Provider、函数调用协议、带历史循环、重试/恢复、实验持久化、长时异步）恰好是可在 K1 验收场景驱动下增量补齐的上层件。补齐清单见第 11 节。

**问题三（是否现在引入，REV 2 修订）**：**B 的演进路线——协议与架构对齐先行，能力按里程碑分阶段进入 K1 关键路径，最终以完整 Agent-Kernel 作为 K1 智能体运行时（达成 A 的能力目标，但始终以 D 的服务边界接入）。** 具体地：现在即启动双轨（K1 侧工具协议 + AK 侧治理与 M1），AK-M2 后其 LLM 链路可跑 K1 任务，AK-M3 后接管长时实验，AK-M4 后成为正式 Runtime。源码复制进 K1 仓库（A 的字面形态）自始至终不做。REV 1 的"D"在"K1 只想用现成内核"的前提下成立；在"K1 兼作 AK 完善驱动场景"的双目标下不再成立——缺失清单就是开发清单。

**问题四（集成方式，REV 2 修订）**：**方案 D（独立 Agent Service，NDJSON/HTTP IPC）作为 K1↔AK 的唯一边界形态，配合方案 B（submodule）限定在 C++ 宿主构建内固定 AK 里程碑 tag。** 前置治理待办（作者可即刻完成）：Agent-Kernel 仓库 git init + 基线 tag、指定 License、恢复 CHANGELOG 纪律。K1 Python 仓库不引入任何 AK 源码。

**问题五（分工与架构图）**：见第 9 节。一句话版本：Agent 层管"想和编排"，Skill Framework 管"动作和训练"，两者只通过显式工具协议对话；过渡期 Agent 层在 Python（兼作 AK Workstream C 的可执行规格），Agent-Kernel 按里程碑从执行权威层向智能层上移，机器人交付时同一 C++ 宿主自然成为机载执行权威。

**问题六（下一步，三阶段双轨，避免大规模重构，REV 2 修订）**：
- **阶段 1（并行双轨，1-2 周）**：K1 轨——在 K1 仓库定义 Python 工具协议层（motion.process / skill.register / rl.train / policy.export / mujoco.sim2sim / eval.analyze 六类工具的 JSON Schema 与结果契约，概念对齐 AK 的 CapabilityContract，为日后平移进 AK 能力注册预留同构性）+ 实验记录存储。AK 轨——治理待办清零（git/tag/License/CHANGELOG），启动 AK-M1（见第 11 节：Schema 校验落地 PrepareInvocation、Context 改追加式历史、结构化 ModelResult 定义）。
- **阶段 2（Python 定标 + AK-M2，2-4 周）**：K1 轨——Python Agent 循环达成"Arm Wave 误差改进"端到端验收（分析→改 reward→训练→导出→Sim2Sim→分析→自主决定下一轮），此过程产出的 Schema/Prompt/重试策略/验收标准即 AK Workstream C 的规格输入。AK 轨——完成 AK-M2（生产 LLM Provider + JSON 函数调用解释器），用确定性测试双复跑 K1 的验收场景。
- **阶段 3（AK 接管，AK-M3/M4，机器人交付前完成）**：AK-M3（长时任务 submit/poll/cancel + 进程隔离 + SQLite 任务日志与恢复）以 rl.train 小时级训练为验收；AK-M4（生产组合：单循环替代 FWA 测试缝、可配置重试与失败分类、泛化 k1_agent_host）以"Arm Wave 全链路在 AK 内闭环、Python 侧收缩为工具实现 + 瘦客户端"为验收。

---

## 11. K1 任务驱动的 Agent-Kernel 完善路线（REV 2 新增）

原则：每一项 AK 改进都必须有一个 K1 验收场景作闭环，避免"为完善而完善"；顺序按依赖排列，与第 10 节问题六的三阶段对齐；单项改动保持 AK 既有的规范驱动风格（先契约后实现），不推倒现有边界模型。

### AK-M0：治理与基线（阶段 1，无代码风险）

| 项 | 内容 | 完成标准 |
|---|---|---|
| M0-1 版本基线 | Agent-Kernel 独立 git 仓库 init、首次基线 commit、打 `v0.2.0-baseline` tag；此后每里程碑打 tag | K1 的 C++ 宿主可按 tag submodule 固定 |
| M0-2 License | 指定并落字（作者自有资产，选宽松或内部条款均可，关键是消除 TBD） | README/仓库根出现 LICENSE |
| M0-3 CHANGELOG | 恢复 Keep-a-Changelog 纪律，补记 Phase 1 至今的既有实现 | 变更可追溯 |

### AK-M1：工具契约与上下文模型修正（阶段 1-2，对应评审缺陷 1/2/3 的地基部分）

| 项 | 内容 | K1 验收场景 |
|---|---|---|
| M1-1 Schema 校验落地 | `CapabilityInputContract` 从描述性字符串升级为可执行 JSON Schema；`PrepareInvocation` 真正校验 payload；畸形参数确定性拒绝（kRejected） | agent 调 `rl.train` 传错参数时被拦下并拿到可读错误，而非训练进程崩溃后才知道 |
| M1-2 Context 追加式历史 | `ApplyUpdate` 的替换语义（`context_services.cpp:166`）改为追加式观察日志 + 有界窗口（滚动或摘要），消除"第 3 步丢第 1 步观察" | motion.process → eval.analyze 两步后，第 2 步观察仍在上下文中 |
| M1-3 结构化 ModelResult | 定义 JSON 动作协议（tool_calls 数组 + 参数对象）替代纯文本 `tool:<cap>:<args>`；解释器按 JSON 解析，畸形输出→kInvalidResult | 先用确定性测试双验收，为 M2 的真 LLM 铺路 |

### AK-M2：Workstream P——生产 LLM Provider（阶段 2）

| 项 | 内容 | K1 验收场景 |
|---|---|---|
| M2-1 HTTP Provider | OpenAI 兼容 chat/completions 的 `ak::model::Model` 实现：重试/退避、超时、凭证仅经环境变量（沿用其"禁止凭证泄漏、禁止合成成功"治理条款） | 同一 Agent 循环换上真 LLM 后仍通过 M1 的确定性验收 |
| M2-2 函数调用解释 | 真实 LLM 的 function-calling 输出接入 M1-3 的 JSON 解释器 | LLM 自主选择并排序 K1 工具完成"分析 arm wave 误差"任务 |

### AK-M3：Workstream R——长时任务与持久化（阶段 3 前半，K1 是唯一合理验收场景）

| 项 | 内容 | K1 验收场景 |
|---|---|---|
| M3-1 Runtime 可复用 | Runtime 服务生命周期与请求解耦，消除每请求重建服务图 | 连续多轮工具调用不重建 Runtime |
| M3-2 长时任务语义 | 能力调用增加 submit/poll/cancel、进度事件上 EventBus、执行器进程隔离（派生 Python 工具进程）与超时——当前同步 string→string invoker（`capability_binding.h:35-37`）的替代通道 | `rl.train`（小时级）提交后 agent 可轮询进度、可取消、训练崩溃不影响宿主 |
| M3-3 任务日志与恢复 | Session/Task 落 SQLite（复用 memory_provider_sqlite 模式），宿主重启后恢复 | 宿主在训练中途被重启，agent 恢复现场并继续编排 |

### AK-M4：Workstream C——生产组合（阶段 3 后半）

| 项 | 内容 | K1 验收场景 |
|---|---|---|
| M4-1 生产循环 | 单一组合根替代 FWA 测试缝：M2 Provider + Runtime 执行 + M1-2 历史上下文 + 失败分类（可重试/终局）与重试策略、可配置循环上界 | "Arm Wave 误差改进"全链路（分析→改 reward→训练→导出→Sim2Sim→分析→决策下一轮）在 AK 内闭环 |
| M4-2 泛化宿主 | 从 kernel_host（2,412 行，VLN 域耦合）提炼 k1_agent_host：能力注册表 + protocol/v1+ 会话/长时任务扩展 | Python 侧收缩为工具实现 + 瘦客户端 |
| M4-3 权限落地 | IToolPermissionService 从恒 kAllowed（`tool.cpp:643-665`）改为会话级能力允许表 | 未授权能力（如实机运动）在 Virtual-First 阶段被默认拒绝 |

### 明确不在本路线内的事项

- 不为 K1 重写 AK 已可用部分（Runtime 引擎、SQLite Provider、foundation）；
- 不做 pybind11/独立 Python 包（等 M4 后视需要评估）;
- 不在 AK 内实现任何机器人学（重定向/RL 算法/Sim2Sim 物理均属 K1 Dance Skill Framework，AK 只见能力调用）。

---

## 附：评审纪律自查

- REV 2 基于新确认的事实（作者所有权、双目标开发）修订了决策类章节（§1 决策、§6 缺陷 5、§8、§9 图、§10、新增 §11）；证据类章节（§2-§7、§4 实测结果）未受影响、未回改。REV 1 结论保留于 §1 供对照。
- 所有"已实现/部分/仅接口/未实现"判定均给出文件与行号证据；关键行为（多步工作流、异常包含、上下文传递、持久化、无效输出处理）经独立程序实测，未凭 README 或命名推断。
- 未修改 VLN-AVP 原仓库（构建与验证均在 /tmp）；未连接真实机器人；未执行训练任务。
- README 中可核验的声明（36/36 测试、工具链基线）已核验为真；其"Engineering Baseline 阶段、P/R/C 未开始"的自评与源码一致，本次评审未发现夸大。
- 本评审未扩展为 VLN-AVP 全项目审计（CARLA/感知/learn* 训练脚本仅在理解 Agent-Kernel 使用方式所必需的范围内检视）。
