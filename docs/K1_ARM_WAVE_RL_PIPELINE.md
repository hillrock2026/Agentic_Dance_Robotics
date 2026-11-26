# K1 Arm Wave RL 管线（booster_train → MuJoCo 部署）

日期：2026-11-17。将 `datasets/k1_dance_v1/arm_wave/trajectory.csv`（修复足底几何后的
k1_action 导出轨迹）训练为 BeyondMimic 动作跟踪策略，并在 MuJoCo sim2sim 展示。

## 管线与产物

```
arm_wave trajectory.csv (31 列, 50 Hz, 246 帧)
  → 剔除 frame_index/timestamp_s → k1_arm_wave.csv (官方 29 列格式)
  → booster_train csv_to_npz.py → k1_arm_wave.npz
  → Isaac Sim 5.1 + IsaacLab 2.3.2 + rsl-rl 3.0.1 训练 (Booster-K1-Arm_Wave-v0)
  → play.py 导出 JIT(含归一化) + ONNX
  → booster_deploy (MuJoCo, K1_22dof.xml, 与 virtual_k1 同模型)
  → 离屏录制 MP4 + 跟踪指标
```

| 产物 | 位置 |
|---|---|
| motion csv/npz | `virtual_k1/assets/booster_assets/motions/K1/k1_arm_wave.{csv,npz}` |
| 训练任务注册 | `tools/booster_train/.../beyond_mimic/robots/k1/arm_wave/` |
| 训练日志与 checkpoint | `tools/booster_train/logs/rsl_rl/k1_arm_wave/2026-11-17_13-20-05/`（model_2999.pt 最终） |
| 导出策略 | 同目录 `exported/k1_arm_wave_2026-11-17_13-20-05.{pt,onnx}` |
| 部署任务 | `tools/booster_deploy/tasks/beyond_mimic/robots/k1/`（k1_arm_wave；模型 `models/k1_arm_wave.jit`） |
| MuJoCo 视频 | `virtual_k1/outputs/k1_arm_wave_mujoco.mp4`（16 s，800 帧 @50 Hz，绿色 ghost=参考） |
| 跟踪指标 | `virtual_k1/outputs/k1_arm_wave_mujoco.metrics.npz` |

## 训练配置与结果

- 3000 迭代 × 4096 环境（约 2.6 h，RTX 3090，31.4k steps/s）；官方 K1 配方
  （Rough 地形 + 无状态估计观测 + 延迟随机化），仅替换 motion 文件。
- 训练收尾：mean reward 62.3（起始 -11），训练环境 body 位置误差 29 mm。

## MuJoCo sim2sim 跟踪质量（16 s ≈ 3.25 个动作循环）

| 指标 | 数值 |
|---|---:|
| 平均关节误差（全部 22 关节） | 74 mrad (4.3°) |
| 手臂关节误差（循环 1 / 循环 2-3） | 160 / 71 mrad |
| 腿部关节误差 | 61 mrad |
| 躯干位置误差 mean / max | 222 / 493 mm |
| 站立稳定性 | 16 s 全程未摔倒 |

已知限制（与既有评估一致）：
1. 源动作右肩峰值速度 15.3 rad/s（NEEDS_REVIEW 遗留问题），快速挥动段跟踪滞后最大
   （aaright_shoulder_pitch 平均 226 mrad / 峰值 921 mrad）；
2. 后段躯干相对参考有 ~200 mm 横向漂移（策略为平衡大幅度摆臂动量采用的位移平衡）；
3. reward 在 3000 迭代仍在上升，继续训练可进一步收紧。

## 复现命令

```sh
conda activate booster_train
export OMNI_KIT_ACCEPT_EULA=YES

# 1) 训练（可继续用 --resume 接续 3000 迭代之后的训练）
cd tools/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-Arm_Wave-v0 --headless --num_envs 4096

# 2) 导出 JIT/ONNX
python scripts/rsl_rl/play.py --task=Booster-K1-Arm_Wave-v0-Play --headless --num_envs 1 \
    --checkpoint logs/rsl_rl/k1_arm_wave/<RUN>/model_2999.pt
cp logs/rsl_rl/k1_arm_wave/<RUN>/exported/*.pt ../booster_deploy/tasks/beyond_mimic/robots/k1/models/k1_arm_wave.jit

# 3a) MuJoCo 交互观看（本机 X 显示）
cd ../booster_deploy
python scripts/deploy.py --task k1_arm_wave --mujoco

# 3b) MuJoCo 无头录制 + 指标
MUJOCO_GL=egl python scripts/deploy_mujoco_record.py --task k1_arm_wave \
    --output ../../virtual_k1/outputs/k1_arm_wave_mujoco.mp4 --duration 16
```

## 部署侧一致性说明

部署 PD 增益/力矩上限逐关节复制自训练配置 BOOSTER_K1_CFG（Kp=J_eq·(2πf_n)²，
arms/head 3.948/0.251、hip 30.2/21.4/17.8、knee 60.4/4.81、ankle 25.0/7.14 等），
action scale 由 0.25·effort/Kp 自动等于训练的 K1_ACTION_SCALE；
50 Hz 策略 × decimation 4 × 5 ms 物理步与训练一致；
MuJoCo 模型为 booster_assets 的 K1_22dof.xml（与 virtual_k1 相同）。
