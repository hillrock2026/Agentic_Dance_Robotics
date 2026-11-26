# K1_DANCE_SKILL_FRAMEWORK_02_REMEDIATION_01_REPORT

日期：2026-11-19。四项补充修复全部完成，双 Skill 回归通过，全量测试 **128 passed / 0 failed / 0 skipped**，12 个受保护资产 SHA256 前后一致。Framework 02 可进入关闭状态。

## 1. 最终结论：PASS

## 2. 四项问题的实际根因

**A. oneshot 缺 frame 0**：录制器只做 post-step 采样（行 k 记录 t_{k+1} 与参考帧 k+1），oneshot 跑满 1260 步时帧 0 永不被采样、末帧 1259 被采两次。上轮修复只解决了"末帧回绕到 0"，未解决覆盖完整性。

**B. Evaluation 非有限值泄漏**：上轮仅对周期相关系数做了 None 语义；`per_cycle_arm_mae_mrad`（`np.mean` 直通）、幅度/速度/root/接触族的 max/mean 均可把 NaN/Inf 带进结果；CLI 的 `json.dumps` 默认 `allow_nan=True`，会输出非标准 `NaN/Infinity` 字面量。

**C. CLI overrides 绕过 Schema**：`train_command` 用 `int(num_envs or tp[...])` 构建命令——校验只发生在 YAML 加载时，覆盖值不经验证；且 falsy-zero（`num_envs=0`）会静默回落到 YAML 值，是双重旁路。

**D. 关节身份验证不充分**：上轮的"数值 allclose 交叉验证"只能证明数值相似，不能证明关节身份（两关节轨迹可能相似）；且名称一致性检查只在资源之间互相比对，没有锚定到权威定义。

## 3. 修改文件与核心内容

| 文件 | 修改 |
|---|---|
| `tools/booster_deploy/scripts/deploy_mujoco_record_cyclic.py` | 行采集提取为 `record_row()` 闭包；oneshot 增加 **pre-step 初始行**（t=0/frame 0，机器人精确初始化于参考帧 0，实测初始关节差=0.0，非伪造索引；该行 dof_target 记录初始化位姿并在注释声明语义）；oneshot 步数 = period−1（1 pre-step + 1259 post-step = 帧 0..1259 恰好各一次）；cyclic 路径逐字未动 |
| `k1_skills/evaluation.py` | 统一非有限值规则：**分段级**（周期/关节/速度箱/漂移段）任一非有限→该条目 `None`；**全局均值**按有限样本计算并报告 `nonfinite_samples`；**max/min 类序统计**改为严格语义（含任一 ±Inf/NaN → `None`，因为真实最大值非有限，取剩余有限最大值会误导）；`evaluate()` 末尾 `json.dumps(..., allow_nan=False)` 兜底守卫 |
| `k1_skills/cli.py` | `_print_json` 改 `allow_nan=False`（CLI 输出层同样严格） |
| `k1_skills/spec.py` | 新增公开 `validate_effective_training_config()`：对**合并后的最终有效配置**（YAML→继承→CLI overrides 之后）复用同套类型/范围规则 |
| `k1_skills/learning.py` | 新增 `effective_training_config()`（override 用 `is not None` 判定，修复 falsy-zero 旁路）；`train_command` 在构建命令**之前**校验最终配置，非法即抛出、不启动训练 |
| `k1_skills/robot.py` | `canonical_joint_orders()`：以 deploy 机器人配置 `K1_CFG.sim_joint_names`（npz 序）与 `joint_names`（CSV 序）为唯一权威来源，k1_skills 不维护任何平行列表 |
| `k1_skills/motion.py` | 身份契约重写：具名资源必须与 canonical **逐项相等**（缺/多/重复/未知/错序一次全捕获）；无名称资源必须声明指向**通过 canonical 验证的具名资源**（身份由声明链接地，不由数据证明）；joint_pos 交叉验证降为辅助证据（矛盾才报错）；canonical 不可加载 → 明确拒绝而非放行 |
| 测试 | `tests_k1_skills/test_remediation_01.py` 新增 56 项；`test_minimal_fix_01.py` 更新（旧顺序测试注入合成 canonical；旧时间轴测试按新契约更新并保留历史 bug 证据断言） |

## 4. 关键修复前后行为对比

**A（mj_dance_002 baseline，三次运行链）**：

| 版本 | 行数 | t 范围 | 首帧→末帧 | 状态 |
|---|---:|---|---|---|
| v1（原始 bug） | 1260 | 0.02–25.20 | 1 → **0**（回绕） | root end 414.3mm / ori end 89.4° |
| v2（上轮修复） | 1260 | 0.02–25.20 | 1 → 1259（末帧重复两次） | root end 218.5mm / ori end 4.45° |
| **v3（本轮）** | 1260 | **0.00–25.18** | **0 → 1259，恰好各一次** | root end 218.5mm / ori end 4.45° |

帧 0 行为真实初始状态（actual-ref 关节差 = 0.0）；`t[k]=k·0.02`、`abs[k]=k`、`frame[k]=k` 全序列公式化验证（自动化测试）。cyclic 回归：arm_wave trained 重跑 813 行、首行 frame=1/t=0.02（post-step 语义未变）、wrap 正确、arms MAE 75.4 mrad 与 F01 一致。

**B**：单周期投毒（NaN/±Inf）→ 该周期条目 `None`、其余周期统计不受污染（`per_cycle=[None, 100.0]` 类场景实测）、全局均值有限样本 + 计数、含 Inf 的 max → `None`；全部指标族对八类输入（正常/单周期 NaN/+Inf/−Inf/全无效/混合/常量/样本不足）通过 `json.dumps(..., allow_nan=False)`；CLI 实际输出经 `json.loads(parse_constant=raise)` 验证。正常数据数值与修复前逐位一致（mae_arms 75.4358… 不变）。

**C**：CLI/工具层非法覆盖（`--max-iterations 0`、`num_envs=-3`、`num_envs=true`、NaN/Inf、数值字符串、浮点整数）全部在**训练启动前**被拒（实测错误码 invalid_definition，未产生任何子进程）；合法覆盖仍生效且优先级高于 YAML（`--num-envs 1024 --max-iterations 12` 实测进入命令行）；两个 Skill 的 YAML 有效配置均通过。

**D**：canonical 匹配一次捕获缺关节/多关节/重复/未知/错序（各一测试）；声明链未接地（指向无名称资源）→ 拒绝；**数据相似不构成身份证明**——两关节轨迹完全相同、无名称资源列被交换的场景：有接地声明链时接受（身份由链承载）、无声明时拒绝（数据匹配不再开绿灯）；canonical 不可用 → 明确报错。真实资源：两 Skill 全部 npz 通过 canonical（`K1_CFG.sim_joint_names`）验证，mj2 声明链 `deploy_npz → train_npz（canonical ✓）` 成立。

## 5. 新增测试与全量结果

- 新增 `test_remediation_01.py`：56 项（A 3 + B 9 含参数化 + C 31 含参数化 + D 13）；
- 全量：`python -m pytest tests_k1_skills -q` → **128 passed, 0 failed, 0 skipped, 2 warnings**（warning 为 numpy 弃用提示，与本任务无关）；
- 未放宽任何既有断言；被更新的两处旧测试均为契约升级（时间轴语义、canonical 注入），且保留了对历史 bug 数据的反向证据断言。

## 6. 两个 Skill 回归（真实资源）

| 项 | arm_wave | mj_dance_002 |
|---|---|---|
| validate | ✓ 全绿 | ✓（trained 未供给警告，预期） |
| 关节身份（canonical） | ✓ 三份 npz 逐项匹配 | ✓ 声明链接地 + 辅助数据一致 |
| Runtime | ✓ 重跑 3 周期 813 步，行为与 F01 一致 | ✓ 重跑全时间轴 1260 行（修复验证运行，baseline 策略） |
| Evaluation（严格 JSON） | ✓ arms 75.4 / fall 0 | ✓ 六族指标 + cycle 降级 |
| 非法覆盖拒绝 | ✓ | ✓ CLI 实测 |

## 7. 受保护资产 SHA256 对比

12 项（arm_wave 两份 npz+循环 npz、两份 JIT、mj2 两份 npz、预训练 PT、model_2999 checkpoint、两个 skill.yaml、共享 profile）修复前后**全部一致**（`k1_runs/remediation_01_{baseline,after}_sha256.txt`）。本轮未修改任何动作资源、策略、checkpoint 与 Skill 配置。

## 8. 尚未解决的问题（均不阻塞 Framework 02 关闭）

1. `mj_dance_002` trained 角色仍无全量训练策略（按任务边界未训练）——后续街舞开发主线自然覆盖。
2. oneshot 全时间轴采样不含"参考末帧之后的保持段"（此前 v2 的第 1261 步行为）；如未来需要评测"末帧保持"行为，可通过 `--cycles` 扩展，属新需求。
3. 仓库仍无版本控制（历史遗留，与 Agent-Kernel 治理待办同源）；本报告及产物清单以 SHA256 快照承担可追溯性。
