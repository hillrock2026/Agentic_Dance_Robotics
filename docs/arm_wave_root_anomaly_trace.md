# Arm Wave Root 参考异常 — 转换链追溯记录（Task A）

日期：2026-11-18。追溯脚本：`tools/booster_deploy/scripts/trace_root_anomaly.py`。

> 结论先行：原始参考 root 的 xy 游走（路径 1.47 m）与 yaw 净转 +309.6°
> **都是真实传播自源动作**（舞者在源视频中确实走位并整体转身近一圈），
> `upper_body_only` 重定向在钉死腿部的同时 1:1 传播了 pelvis 平移与 heading，
> 共同构成"腿固定站姿 + root 走动转圈"的语义不一致参考。

## 数据链（全部核验）

```
segment_071.mp4 (25fps, 123帧, sha256 343ee79d…)
  → GVHMR gvhmr_raw.npz            [pelvis transl 路径 1.311 m, 净位移 0.230 m]
  → motion_studio 编辑 edited_motion.npz  [SMPL, 与 GVHMR pelvis 完全一致; sha256
                                          eafd6530… == metadata.source_sha256]
  → k1_action retargeting (upper_body_only) → trajectory.csv (246帧@50Hz)
  → 删 2 列 → booster k1_arm_wave.csv
  → csv_to_npz (Isaac FK) → k1_arm_wave.npz  [npz trunk 旋转 == csv root 旋转, <0.09°]
```

## 各环节事实

1. **GVHMR / edited SMPL（源头）**：pelvis 水平位移真实存在——xy 路径 1.311 m、
   净位移 0.230 m、z 波动 ±6 cm；pelvis 朝向含大幅俯仰（ZYX pitch ±88°）与扭转。
   源视频人眼核验：舞者从 45° 侧身开始，片段结束时背对镜头（~150-160°），
   并在画面中移动了位置。**移动和转身是原始动作本身的内容。**
2. **retargeting `_retarget_root()`**（k1_action/retargeting.py:403-433）：
   - yaw：pelvis global_orient 的前向投影 → `yaw_quat_xyzw`，**yaw-only**（roll/pitch 置零）；
   - xy：pelvis 水平位移经 R_map **1:1 传播**（docstring 明示 "xy from human pelvis (1:1)"）；
   - z：upper_body_only 模式下强制常数站姿高度（0.555023 m，官方 CSV 首行）。
   - 坐标变换只应用一次（MS_WORLD Y-up → K1 Z-up，R_map 由首帧肩线标定）；四元数
     存储 xyzw，csv_to_npz 转 wxyz——npz 与 csv 逐帧相对旋转 <0.09°，无重复旋转/平移。
3. **yaw 数值（符号安全 + 正确周期的 unwrap）**：CSV/npz root yaw 起始 −7.13°、
   先降至 −50.2°、随后一路扫至 +302.5°（第 197 帧穿越 ±180° 分支一次）——
   **净转 +309.6°，peak-to-peak 352.7°，峰值速率 429°/s（第 0 帧）**。
   参考语义即"每循环整圈转体"（循环 transition 补 +50.4° 后每周期净转 ≈360°）。
   *分析方法教训*：本轮中途曾得到"净 +31.3°、无整圈转体"的错误读数，根因是
   **`np.unwrap` 默认周期 2π（弧度）被用于以"度"为单位的 yaw 序列**——一切
   >3.14° 的正常帧间变化都被误判为分支穿越。以"度"工作时必须
   `np.unwrap(x, period=360)`。四元数欧拉分解公式各项均为二次型，天然符号
   不变（npz 第 198 帧存在单帧 w<0 的符号翻转，但无影响）。geodesic 首尾差
   50.4° 与循环过渡 slerp 亦符号安全。上一轮报告的"每循环转体约一圈"结论
   经正确重算确认成立。
5. **时间重采样**：25→50 Hz 逐帧复制上采样（metadata: resampled_from_fps 25.0），
   246 帧 = 123×2；root 轨迹形状保持。片段裁剪（segment_071 从完整视频切出）
   造成动作首尾不在同相位——那是循环边界问题（上轮已用过渡段处理），与 root 异常
   无关。
6. **固定腿与 root 运动的语义冲突**：12 个腿关节 246 帧严格常量（官方站姿），
   而 root 沿 pelvis 1:1 平移并 yaw 摆动 ±40°——等效于"脚被焊在站姿几何上、
   躯干却按真人走路/转身平移旋转"的漂浮目标，物理上不自洽。

## 结论

**异常 Root 由两者共同造成：**

- **原始动作本身**带来真实的 pelvis 平移（路径 1.3 m 量级）与朝向大幅变化
  （源视频可见走位与转身）——这部分是"真"的；
- **转换链的设计选择**（`upper_body_only` 模式）在钉死腿部的同时把 pelvis 的
  水平位移与 heading 1:1 传播进 root，未按"站姿上肢动作"语义过滤——这部分把
  真人走动语义错误地注入了站姿参考，是**语义不一致的直接来源**。

链路本身（坐标变换、四元数顺序、无重复应用、无 wrap 错误、重采样、csv↔npz
一致性）未发现 bug；唯一的数据级瑕疵是 npz 第 198 帧四元数符号翻转（无旋转影响）。
