# K1_PINN_ROLLOUT_01 — 开环多步状态预测对照实验

Stage1-02 的 10 个 checkpoint（MLP seed0–4、PINN λ=1 seed0–4）在 Stage1 TEST 全部 3,110 个合法 8 步窗口起点上的开环 rollout 对照。

## 运行

```bash
# 解释器: /home/liu/anaconda3/envs/booster_train/bin/python
python rollout/run_rollout.py        # 10 checkpoint × 3110 窗口 × 8 步 + 全部检查
python rollout/aggregate_rollout.py  # 汇总表/配对差值/误差增长图
```

产物：`raw_<run>.npz`（逐窗口逐步长原始误差）、`rollout_manifest.json`（checkpoint/数据/契约记录）、`results/metrics_by_horizon.{csv,json}`、`results/summary_paired.json`、`results/error_growth.png`。

## 契约要点

- 初始状态 = 真实状态（一次性）；此后仅用预测状态自馈 + 记录的 ctrl 序列（开环，非闭环控制）。
- 状态积分：`v_hat += dt·a_hat; q_hat = mj_integratePos(q_hat, v_hat, dt)`；dt 用记录的 f64 时间戳差（**实测：f32 量化 dt 会引入 ~3.5e-6 的速度重建误差，f64 仅 9.2e-14**——WM_01 transitions npz 的 dt 字段为 f32，精确复算请用 episode 时间戳）。
- 输入契约与训练严格一致：81 维 [pos3|rot6|joints22|qvel28|ctrl22]，冻结 train z-score，输出反归一化。
- MuJoCo 仅用于位置积分；无 mj_step/mj_forward，未来状态零泄漏（真值加速度积分检查 9.2e-14/1.3e-15）。
- H=1 rollout 与独立单步实现逐 run 比对差异 = 0；无非有限预测（0 窗口）。
- 分层用的是**起点标签**（free 1024 / contact 2086 / impact 50），不代表整窗状态。
