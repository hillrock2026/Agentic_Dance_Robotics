# K1_DANCE_SKILL_FRAMEWORK_01 — 执行结果报告

日期：2026-11-19。首版 K1 Dance Skill Framework 交付 + Standing Arm Wave 全链路接入
（定义 → 校验 → 准备 → 训练 → 导出 → Sim2Sim → 评价 → 新旧对比），全部通过框架命令完成。

## 0. 结论

**框架建成并完成首个 Skill 的端到端验收：新增策略经框架训练（3000 迭代）与导出后，
在统一控制配置下三周期连续执行（813 步、零摔倒、无早停），对比零样本 JIT 基线
所有跟踪/稳定性指标全面改善**（手臂 MAE −15%、全身 −43%、腿部 −73%、快速挥臂段 −34%、
root 误差 −50%）。核心训练/执行/评价代码零复制：框架以子进程调用 booster_train 与
booster_deploy 既有代码，每动作只新增 YAML + 两个既有模式的小注册文件。

统一配置过程中发现并修复两个**既有**部署配置缺陷（详见 §3，影响对历史报告数字的解读）。

## 1. 交付物

| 类别 | 位置 |
|---|---|
| 框架包（10 个模块） | `k1_skills/`（spec/registry/motion/robot/learning/runtime/evaluation/tasks/agent/cli） |
| 统一命令接口 | `python -m k1_skills list/validate/prepare/train/train-status/export/run/evaluate/tool/tasks` |
| 首个 Skill 定义 | `skills/arm_wave/skill.yaml`（控制参数单一声明，训练与运行时共用） |
| Agent 工具层 | `k1_skills/agent.py`：motion.process / skill.register / skill.validate / rl.train / rl.status / policy.export / mujoco.sim2sim / eval.analyze，均含输入输出 JSON Schema 与稳定错误码；CLI `python -m k1_skills tool <name> '<json>'` 可独立调用 |
| 单元测试 | `tests_k1_skills/test_framework.py`（8 项，全过：评价数学/对比方向性/定义校验/工具错误语义） |
| 新训练任务 | `tools/booster_train/.../robots/k1/arm_wave_usr/`（复用 mj_dance_002 配方，仅换动作文件） |
| 新 deploy 任务 | `k1_arm_wave_usr_v2`（`tools/booster_deploy/.../robots/k1/__init__.py`，继承 arm_wave 增益） |
| 训练产物 | `tools/booster_train/logs/rsl_rl/k1_arm_wave_usr/2026-11-19_09-58-43/`（model_2999.pt + exported jit/onnx） |
| 部署策略 | `tools/booster_deploy/.../robots/k1/models/k1_arm_wave_usr_v2.jit`（由 export 命令暂存） |
| 运行产物 | `k1_runs/outputs/arm_wave/`：基线与训练策略各 813 帧视频 + metrics.npz + 日志 |
| 任务登记 | `k1_runs/tasks/*.json`（rl-train-20261119-095835-a07100 已定稿 succeeded） |

## 2. 验收清单（十项全部通过）

1. **Skill Definition 加载** ✓ `validate arm_wave` 全绿（定义/资源/npz 不变量/stationary-root 校验）
2. **框架调用训练设施** ✓ `train arm_wave --background` 启动真实 IsaacLab 训练，rl.status 实时报进度
3. **统一控制配置共用** ✓ `validate` 将 skill.yaml 的 PD/力矩/decimation/physics_dt 与 deploy 任务逐值交叉校验；本次修复后三任务（arm_wave/usr/usr_v2）与训练一致
4. **旧 JIT 零样本基线** ✓ `run --policy baseline --cycles 3`：813 步、0 摔倒
5. **新策略经框架训练与导出** ✓ 3000 迭代（奖励 −11.9→73.1，关节误差 1.37→0.43，收敛优于旧训练的 0.60）；`export` 产出 jit+onnx 并暂存至 deploy models
6. **新策略连续三周期** ✓ 813 步 = 3×271 帧、stopped_early=False、fallen=0
7. **Evaluation 自动对比** ✓ `evaluate --baseline --candidate`（下表），eval.analyze 工具同样可跑
8. **无按动作复制核心代码** ✓ 框架零复制；新动作新增仅：YAML + arm_wave_usr 三小文件（复用配方）+ deploy cfg 子类
9. **Agent 工具独立调用** ✓ `python -m k1_skills tool ...` 直接调用（含错误语义演示：unknown_tool / invalid_arguments / skill_not_found）
10. **无 Agent-Kernel 依赖** ✓ 框架无任何 VLN-AVP/Agent-Kernel 引用，独立完整运行

## 3. 统一控制配置过程中发现并修复的两个既有缺陷

1. **基线任务增益继承错误**：`k1_arm_wave_usr`（ROOT_CORRECTION_01 的零样本基线任务）
   原继承 `K1BeyondMimicControllerCfg` 基类增益（腿 80/80/80/80/30/30），并非其报告声称的
   "同 PD/增益"（训练与 k1_arm_wave 实为 30.2/21.4/17.8/60.4/25.0/7.1）。已改为继承
   `K1ArmWaveControllerCfg`。**影响：该报告 Task E 的绝对数值是在错误增益下测得。**
2. **physics_dt 时间尺度缺陷**：`ControllerCfg.__post_init__` 先按默认 decimation=10 算出
   physics_dt=0.002，任务再改 decimation=4 不会重算 → 每策略步物理仅推进 8ms 而参考推进
   20ms（2.5× 时间压缩）。实测证据：旧 k1_arm_wave 录制中实际/参考手臂速度比 2.29
   （usr 任务当时 decimation=10 自洽，比值 0.79）。已在 K1ArmWaveControllerCfg 显式设置
   `physics_dt=0.005`，三个任务均实测 `4×5ms=20ms=50Hz` 与训练一致。
   **影响：CYCLIC_SIM2SIM_01 报告的数值含时间尺度伪影（"快速挥臂滞后 258 mrad"在统一
   时间轴下基线仅 95 mrad）；本框架的 validate 步进一致性检查可自动拦截此类错误。**

## 4. 结果数据（统一配置下，3 周期，均 813 步、0 摔倒、双脚支撑 99.9%）

| 指标 | 零样本基线（model_2999 JIT） | 框架训练策略（v2） | 变化 |
|---|---:|---:|---:|
| 手臂 MAE (mrad) | 89.0 | 75.4 | **−15.3%** |
| 全 22 关节 MAE (mrad) | 68.6 | 38.8 | **−43.4%** |
| 腿部 MAE (mrad) | 56.3 | 15.1 | **−73.1%** |
| 快速挥臂段 >10rad/s MAE (mrad) | 95.1 | 62.7 | **−34.1%** |
| root 误差均值/最大 (mm) | 14.5 / 28.1 | 7.3 / — | **−49.5%** |
| 周期末自身漂移 (mm) | 2.1 | 0.4 | −81% |
| 躯干朝向误差 (°) | 4.31 | 1.84 | −57.2% |
| 挥臂幅度比 | 0.943 | 0.942 | 持平 |
| 周期间误差相关 | 0.87–0.97 | 0.93–0.995 | — |
| 分周期手臂 MAE (mrad) | 87.6/89.8/89.7 | 74.4/76.0/75.9 | 稳定不退化 |

训练配置：4096 env × 3000 iter、seed 42、stationary-root 参考语义自洽（ROOT_CORRECTION_01
遗留建议"新参考已具备重训条件"在本任务兑现）。

## 5. 新增一个 Dance Skill 的路径（框架目标验证）

动作资源 + `skills/<id>/skill.yaml` + booster_train 任务注册（复用配方仅换 npz）+
booster_deploy 任务注册（继承增益仅换 motion/policy）→ validate/prepare/train/export/
run/evaluate 七条命令。专属差异（reward/observation 覆盖）在 YAML profile 声明，不共享不强copy。

## 6. 已知边界

- 评价基于 MuJoCo sim2sim 逐帧录制（50Hz，时间对齐已由既有录制脚本保证）；Isaac→MuJoCo
  域差依旧存在，实机交付前需按 Virtual-First 流程增加实机域随机化。
- `k1_arm_wave_usr_v2.jit` 当前指向本轮训练产物；重训会经 export 覆盖暂存（run 目录保留历史）。
- 长时训练的任务登记在进程退出后由 rl.status 定稿（新启动的训练带 TRAIN_EXIT_CODE 标记，
  本轮为 checkpoint 推断路径）。

## 7. 复现

```bash
python -m k1_skills validate arm_wave
python -m k1_skills run arm_wave --policy baseline --cycles 3
python -m k1_skills train arm_wave --background     # ~2.6h RTX3090
python -m k1_skills train-status <task_id>
python -m k1_skills export arm_wave
python -m k1_skills run arm_wave --policy trained --cycles 3
python -m k1_skills evaluate --baseline <base.metrics.npz> --candidate <trained.metrics.npz>
python -m pytest tests_k1_skills -q
```
