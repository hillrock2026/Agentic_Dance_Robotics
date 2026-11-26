# k1_skills — K1 Dance Skill Framework (schema v2)

一个 Skill = 一份 `skills/<skill_id>/skill.yaml`（schema v2）+ 动作资源 +
两侧各一个小注册文件。框架调用现有已验证基础设施（booster_train 的
IsaacLab/rsl_rl 训练、booster_deploy 的 MuJoCo sim2sim 与录制/分析脚本），
不为任何单个动作复制训练代码。

## 命令

```bash
python -m k1_skills list                          # 技能清单（schema/kind/角色）
python -m k1_skills validate <skill>              # 定义/资源集/控制配置/策略契约校验
python -m k1_skills prepare <skill>               # 官方 CSV -> 训练 npz（幂等）
python -m k1_skills train <skill> --background    # 后台训练（--num-envs/--max-iterations 可覆盖）
python -m k1_skills train-status <task_id>        # 或 --list-recent
python -m k1_skills export <skill>                # 导出 jit+onnx，版本化归档+manifest，暂存部署
python -m k1_skills policies [skill]             # 策略清单 / 暂存文件漂移检查
python -m k1_skills run <skill> --policy baseline|trained [--cycles N]   # oneshot 自动单遍
python -m k1_skills evaluate --skill-id <skill> <metrics.npz>            # 用技能的关节组/动作类型
python -m k1_skills evaluate --skill-id <s> --baseline <a.npz> --candidate <b.npz>
python -m k1_skills tool list                     # Agent 工具（8 个，含 Schema）
python -m k1_skills tasks [--kind rl.train]
```

框架用 booster_train conda 解释器运行（可用 `K1_SKILLS_TRAIN_PYTHON` /
`K1_SKILLS_DEPLOY_PYTHON` 覆盖路径）。

## Schema v2 要点（FRAMEWORK_02）

- `motion.kind: cyclic | oneshot`：oneshot 无过渡帧/循环 npz，运行单遍，
  cycle 类指标自动降级；
- `control_profile.extends: k1_default`：机器人级事实（模型/自由度/policy_dt/
  action scale 规则/关节组）来自 `skills/_profiles/k1_default.yaml`，技能只声明
  其训练所用的增益/力矩/步进；步进不变量 physics_dt×decimation = 1/fps 强制；
- `runtime_profile.policies`：角色 → {deploy_task, policy_file} 映射，任意角色名；
- 策略产物契约：每次 export 归档到 `k1_runs/policies/<skill>/<name>-<run>/`
  （jit+onnx+manifest 含 sha256/checkpoint 溯源），deploy 侧文件名稳定；
  `policies`/`validate`/`run` 均做暂存文件-manifest 漂移检查；
- 评价扩展：指标按命名关节组计算（技能可覆盖组），`register_metric` /
  `evaluation_profile.custom_metrics: ["module:func"]` 接入自定义指标；
- v1 定义自动升级加载（`schema_version` 缺省 = 1）。

## 已接入技能

| skill | motion | root | 基线策略 | 状态 |
|---|---|---|---|---|
| arm_wave | cyclic 246+25 帧 | stationary | model_2999 JIT（零样本） | 框架重训+导出+3 周期验收（F01） |
| mj_dance_002 | oneshot 1261 帧 | free | Booster 预训练 | 接入验收（F02）：smoke 训练/单遍运行/评价 |

## 新增一个舞蹈动作

1. 准备动作资源（数据集 trajectory、官方 CSV、train npz、deploy npz
   [oneshot] / 循环 npz [cyclic]）；
2. booster_train 注册 gym 任务（复制 `robots/k1/arm_wave_usr/` 三小文件改
   motion 路径）；booster_deploy 注册 deploy 任务（继承同增益任务类改
   motion/policy）；
3. 写 `skills/<id>/skill.yaml`（v2；控制配置 extends 共享 profile）；
4. `validate` → `prepare` → `train` → `export` → `policies` → `run` → `evaluate`。

专属差异（Reward/Observation 覆盖）在 skill.yaml 对应 profile 声明。
