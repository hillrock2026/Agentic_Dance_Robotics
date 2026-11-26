# K1_DANCE_SKILL_FRAMEWORK_02 — 执行结果报告

日期：2026-11-19。目标达成：框架从"能跑 Arm Wave 的工程"泛化为
"可标准化接入/训练/导出/运行/评价不同 Dance Skills"的通用框架，并以第二个
Skill（mj_dance_002，oneshot + free-root，与 arm_wave 的 cyclic + stationary
正交）完成多 Skill 复用验证。

## 0. 结论

**八项交付全部完成并以运行结果验证。** 新 Skill 接入成本从"一份 66 数字的
控制配置 + 固定字段"降为"一份 YAML（控制配置继承共享 profile）+ 两侧各一个
<20 行注册文件"；框架引擎、评价、CLI、Agent 工具全部双 Skill 实跑通过；
顺带发现并修复了泛化过程中暴露的一个新 bug（解释器解析 `Path("")→"."`）。

## 1. 交付明细

### ① 通用 Skill Schema（`k1_skills/spec.py`，schema_version 2）

- `motion.kind: cyclic | oneshot` 条件化必填字段（oneshot 无 transition/
  循环 npz；cyclic 必须有循环 npz）；
- `control_profile.extends`：机器人级事实继承自 `skills/_profiles/k1_default.yaml`，
  技能只声明训练所用的增益/力矩/步进；步进不变量
  `physics_dt×decimation == 1/policy_frequency_hz` 在加载期强制（测试覆盖）；
- `runtime_profile.policies`：角色 → {deploy_task, policy_file} 映射，
  任意角色名（baseline/trained 仅为惯例）；
- `evaluation_profile`：可选指标子集、关节组覆盖、自定义指标入口；
- v1 定义自动升级加载（缺省 schema_version=1 → in-memory 升级，测试覆盖）。

### ② 标准 Motion Resource Interface（`k1_skills/motion.py`）

- `MotionResourceSet`：dataset trajectory / booster csv / train npz / deploy
  npz / deploy cyclic npz 五种角色的统一加载与描述（motion.process inspect）；
- 角色不变量：fps 一致、帧数符合声明、22 关节、stationary-root 常位姿/
  恒朝向（free-root 明确不设约束）；
- 跨角色同步：train vs deploy 帧数容忍 ±1（两条 FK 管线的真实差异，
  mj_dance_002 即 1261 vs 1260，测试覆盖）；旧版无名称元数据的 npz 自动降级
  名称校验。

### ③ 通用 Training Configuration（`k1_skills/learning.py`）

- 启动时覆盖：CLI/工具层 `--num-envs` / `--max-iterations`（mj_dance_002 smoke
  以 1024 env × 12 iter 实跑验证）；
- `experiment_name` 缺省自动派生；`cli_overrides` 透传任意训练旗标；
  `resume` 续训配置（load_run/checkpoint）；
- 后台训练以 bash 包装写入 `TRAIN_EXIT_CODE=<n>`，rl.status 在进程退出后
  自动定稿任务（成功时才附 run 目录/checkpoint 工件——失败误附问题已修）。

### ④ 统一 Policy Artifact / Runtime Contract（learning.py + runtime.py）

- 每次 export 归档至 `k1_runs/policies/<skill>/<name>-<run>/`
  （jit+onnx 副本 + manifest.json：sha256、checkpoint、run、控制配置摘要），
  并更新 `latest.json`；deploy 侧暂存文件名保持稳定（任务配置不随训练变化）；
- 漂移检测：`policies <skill>`、`validate`、`run` 三处校验"暂存文件哈希 ==
  manifest"；未导出角色按生命周期语义给警告（"not yet provisioned"）而非误报；
  arm_wave 已有 manifest 实体（`k1_runs/policies/arm_wave/k1_arm_wave_usr_v2-2026-11-19_09-58-43/`）。

### ⑤ 通用 Evaluation Extension Interface（`k1_skills/evaluation.py`）

- 全部关节类指标按命名关节组计算（`skills/_profiles` 声明 + 技能覆盖，
  测试证明换组即换统计范围）；
- 适用性：`cycle_consistency` 仅适用于 cyclic；oneshot 请求时降级并记入
  `dropped_metrics`（mj_dance_002 实跑验证）；
- 扩展点：`register_metric(name, fn)` + `evaluation_profile.custom_metrics:
  ["module:func"]` 动态加载（测试覆盖注册与冲突拒绝）。

### ⑥ 三个已知局部问题修复

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 1 | 解释器路径硬编码（`/home/liu/anaconda3/.../python` 写死，换机即坏） | `K1_SKILLS_TRAIN_PYTHON`/`K1_SKILLS_DEPLOY_PYTHON` 环境变量覆盖 + 已知位置回退；顺带修复空环境变量导致 `Path("")→"."` 被当可执行文件的 bug（该 bug 曾使 smoke 训练以 bash source 方式失败） | 双 Skill 全部命令实跑 |
| 2 | rl.status 工具缺 task_id 时报 `tool_internal_error`（KeyError），schema 声明的 `list_recent` 未实现 | 缺参 → `invalid_arguments`；`list_recent` 返回最近 10 条训练任务 | 工具调用 + 单测 |
| 3 | 策略导出每次覆盖同一 jit 文件，无版本/溯源（F01 报告 §6 已知边界） | ④ 的版本化归档 + manifest + 漂移检测 | `policies` 命令实跑 |

### ⑦ 多 Skill 复用验证（方案 + 实跑）

固定验收序列，逐 Skill 执行：`validate → prepare(按需) → smoke train
（override 小迭代）→ baseline run → evaluate（技能上下文）`。

| 步骤 | arm_wave（cyclic/stationary） | mj_dance_002（oneshot/free） |
|---|---|---|
| validate | ✓ 全绿 | ✓（trained 角色生命周期警告，符合预期） |
| 训练泛化 | F01 已 3000 iter 全量验证 | ✓ smoke 1024 env × 12 iter，TRAIN_EXIT_CODE=0 定稿，run 目录正确归属 |
| run | F01 已 3 周期 ×2 角色 | ✓ 单遍 1260/1260 步、0 摔倒（oneshot 路径 + `--source-frames` 泛化后的录制器） |
| evaluate | ✓ v2 引擎回归：arms 75.4 / all 38.8 mrad 与 F01 报告一致 | ✓ 六族指标 + cycle_consistency 自动降级 |
| 策略契约 | ✓ manifest + 暂存哈希一致 | ✓ 基线无 manifest 正确放行（ok=null） |

新增 mj_dance_002 的全部专属代码：`skills/mj_dance_002/skill.yaml`（1 份）+
`K1MJ2TrainedControllerCfg`（11 行）+ 共享 profile（1 份，两技能共用）。
配套的 deploy 录制器泛化：oneshot 任务允许 `--cycles 1` 单遍录制、
`--source-frames` 参数化（原 246 硬编码，arm_wave 由框架自动传入）。

### ⑧ CLI 与测试

- CLI 新增/更新：`train --num-envs/--max-iterations`、`train-status --list-recent`、
  `policies [skill]`、`run --policy <任意角色>`、`evaluate --skill-id`；
- 测试 8 → **18 项全过**：评价数学/自定义关节组/oneshot 适用性/自定义指标注册/
  v1 升级/继承合并/步进不变量/角色映射/工具错误语义/双 Skill 真仓集成。

## 2. 与 FRAMEWORK_01 的兼容

- F01 的 arm_wave 工作流命令全部不变可用；v1 YAML 可继续加载（自动升级）；
- F01 产物（训练 run、暂存 jit、metrics）无需迁移；arm_wave 的 v2 导出对
  同一 checkpoint 重新归档并建立 manifest。

## 3. 已知边界

- `latest.json` 按 skill 粒度（单角色）；多角色导出并存时以 trained 为准；
- mj_dance_002 的 trained 角色尚未全量训练（smoke 12 iter 不用于策略），
  全量重训属后续独立实验；
- 评价扩展的自定义指标以 Python 模块解析，Agent 侧调用时受同一解释器约束。

## 4. 复现

```bash
python -m k1_skills list
python -m k1_skills validate arm_wave && python -m k1_skills validate mj_dance_002
python -m k1_skills train mj_dance_002 --background --max-iterations 12 --num-envs 1024
python -m k1_skills train-status --list-recent
python -m k1_skills export arm_wave && python -m k1_skills policies
python -m k1_skills run mj_dance_002 --policy baseline
python -m k1_skills evaluate --skill-id mj_dance_002 k1_runs/outputs/mj_dance_002/*baseline*.metrics.npz
python -m pytest tests_k1_skills -q        # 18 passed
```
