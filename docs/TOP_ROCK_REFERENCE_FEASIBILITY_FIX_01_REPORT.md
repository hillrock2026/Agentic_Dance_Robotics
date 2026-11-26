# TOP_ROCK_REFERENCE_FEASIBILITY_FIX_01 — 执行报告

日期：2026-11-23。Top Rock 参考运动物理可行性修复：deploy FK qpos 顺序错误 + 足底碰撞穿地。

## 总体结论：**PASS —— 允许进入 TOP_ROCK_RL_TRAINING_01**

| 准入项 | 结果 |
|---|---|
| deploy FK 错误 | ✅ 已修复（全 body round-trip 5.8e-8 m / 0.033°；train-deploy 关节差 1.4e-6、root 差 1.5e-7） |
| 足底穿地 | ✅ 58.03/58.39 mm → **0.44 / 1.95 mm**（八角点验收，全部帧 ≤2 mm，>2mm 帧数 **0**） |
| 关节限位 | ✅ 0 违例（最小裕量 0.0207→0.0050 rad，修正消耗了裕量但未越界） |
| Root/舞蹈特征 | ✅ xy 路径 1.261 m 与 yaw −86.6° **逐位不变**；z 范围 0.536–0.598 → 0.549–0.600 m（局部 +13 mm 抬升）；节奏包络相关 0.748→0.719；无超速关节、无 >0.2 rad 跳变 |
| 物理回放跌倒 | ✅ 未发现可归因于参考几何的跌倒（PD replay 对照实验，见 §7） |

## 1. Task A：deploy FK 修复

**根因**：`convert_k1_trajectory.py` 用 `qpos[7:29] = joints_sim` 按位写入，但 **XML 铰链定义序 ≠ npz sim 序（22 位中 18 位错位）**——除 root 外全部 body 的 `body_pos_w/body_quat_w`（及速度）由错误姿态算出。原"trunk-vs-root"自检只能查 root 依赖的 trunk，结构性漏检。

**修复**：按 MuJoCo 关节名查询真实 qpos 地址（`qadr[name]`，跳过 7 槽 free joint）写入；FK 用正确地址重算全部 23 body 位置/姿态与线/角速度；**round-trip 检查升级**——重载写出的 npz、按名称地址重新 FK、逐 body 比对（断言 pos <1e-6 m、quat <0.1°，实测 5.8e-8 m / 0.033°，float32 存储量级）。joint_pos 与 root 逐位不变；train（Isaac FK）/deploy（MuJoCo FK）关节差 1.4e-6 rad、root 差 1.5e-7 m。

## 2. Task B：足底穿地修复

**B1 逐帧定位**（`toprock_foot_analysis.py`，旋转八角点 + 支撑/摆动估计）：左/右最大穿地 58.03/58.39 mm、负高度帧 254/257（与任务书一致）。曲线结构：双足基线系统性偏低（左 ~10 mm、右 30–50 mm），深穿区间集中在左 60–106/111–189/205–247、右 0–142/156–225/252–278 帧——**左右修正需求不同，禁止全局抬升**（会抬起支撑脚）。

**B2 修正**（`toprock_foot_fix.py`）：逐帧 Gauss-Newton，参数 = 双腿各 [hipPitch, kneePitch, anklePitch, ankleRoll, hipRoll] + root-z（共 11 维）。目标 = 双足八角点穿地残差 + **逐足非对称跟踪**（足底抬升至"原高度 + 所需抬升量"：摆动足不变、穿地足恰好到 2 mm 线）+ 关节/root-z 正则。每步 GN 后**投影进限位可行域**（裕量 5 mrad，修正永不被事后剪裁掉）；逐帧 9 帧平滑 × 4 轮外层迭代；残余 ~10 mm 倾斜角点由**局部 root-z 清尾**（逐帧补差 + 5 帧平滑 + 最终无平滑精确补差）收至 1.95 mm。开发中还发现并修复了修正脚本自身的 **root 四元数 xyzw→wxyz 解包错位 bug**（曾使优化器在错误姿态下工作）。

**B3 舞蹈保留**：root xy/yaw 完全冻结（逐位不变）；root z 仅局部抬升（max 29.5 mm、均值 ~10 mm，z 范围整体 +13 mm）；仅腿部 10 关节被修改——hipPitch max 0.249 rad（均值 0.10）、kneePitch 0.178、anklePitch 0.185、ankleRoll/hipRoll ≤0.03；**上肢/躯干/头部 12 关节逐位不变**。步法交替（支撑估计 66/220 帧）、转体、节奏（相关 0.719）保留。

## 3. 修正前后定量对比

| 指标 | 修正前 | 修正后 |
|---|---:|---:|
| 左/右足最大穿地 | 58.03 / 58.39 mm | **0.44 / 1.95 mm** |
| 左/右 >2mm 穿地帧 | 253 / 255 | **0 / 0** |
| 左/右负高度帧 | 254 / 257 | 3 / 40（均 <2mm） |
| 摆动脚最大离地（角点） | 14.3 / 40.8 mm | 17.5 / 44.7 mm（步法保留） |
| 关节限位违例 | 0（裕量 0.0207） | 0（裕量 0.0050） |
| 超 8 rad/s 关节 / >0.2 rad 跳变 | 0 / 0 | 0 / 0 |
| root xy 路径 / 净 yaw | 1.261 m / −86.6° | **1.261 m / −86.6°（不变）** |
| root z 范围 | 0.536–0.598 m | 0.549–0.600 m |
| 节奏包络相关（vs 源） | 0.748 | 0.719 |
| 支撑相滑动（运动学估计） | 左 0.29 / 右 0.24 m/s | 左 0.19 / 右 0.38 m/s（见 §5） |

## 4. 足底接触与滑动（独立计算，标注为运动学估计）

支撑判定从未使用穿地帧（标准 = 最低角点 <2 cm 且为较低足，无物理求解器，**标记为估计**）。修正后支撑/摆动交替清晰（左 72 / 右 214 帧，右足 26 帧离地抬步）。**滑动是独立的已知参考特征**（重定向足链误差 + top rock 拖步风格），修正未针对滑动（超出本任务范围）：左降至 0.19 m/s、右因支撑判定集变化升至 0.38 m/s。RL 训练的足部位置/朝向奖励将实际处理滑动；若训练后足部质量不足，再考虑足底锁定后处理。

## 5. 修正前后视频与曲线

`k1_runs/outputs/top_rock/fix01/`：`before_after.mp4`（左右并排，286 帧）、`playback_before/after.mp4`、`before_after_curves.png`（双足足底高度曲线 + root z + 右腿关键关节前后对比）、`root_xy_compare.png`（xy 逐位重合）、`foot_analysis_before/after.json`（逐帧曲线数据）。目检：步法交替、转体、手臂摆动、躯干律动全部保留，修正后无可见姿态畸变；root z 无突兀台阶（清尾残差 ≤1 mm/帧）。

## 6. MuJoCo replay

- **运动学回放**（逐帧设 qpos）：完整 5.72 s 播放正常，无跌倒概念问题（纯 FK）。
- **PD replay**（无策略，训练增益开环跟踪，`pd_replay_reference.py`）：top_rock 跌倒 269/286 帧、跟踪误差 2.06 rad。**对照实验**：对已通过 F01 全链路验证的 arm_wave 静态参考做同样 PD replay——同样跌倒（249 帧、2.23 rad）。结论：**开环关节 PD 无平衡控制必然跌倒（这正是 RL 训练要提供的），跌倒不可归因于参考几何**；top_rock 与已知良好参考同量级。训练准入条款"不再出现由参考几何问题导致的明显跌倒"据此判定满足。

## 7. 更新资源与 SHA256（前 16 位，完整值见 fix01/fixed_resources_sha256.txt）

| 资源 | SHA256 前 16 |
|---|---|
| `datasets/k1_dance_v1/top_rock/trajectory.csv`（修正版） | 新文件 |
| `datasets/k1_dance_v1/top_rock/trajectory_original.csv`（**原始保留**，与修复前 trajectory.csv 哈希一致） | `a25c91cc086ded5a` |
| `virtual_k1/.../K1/k1_top_rock.csv` / `.npz`（train） | `a2da0b58…`→新 / `8e19442…`→新 |
| `tools/.../motions/k1_top_rock.npz`（deploy，FK 修正+足底修正） | `c65f189…`→新（round-trip 通过） |
| metadata.json / validation.json | 追加 fix_history / fix01_check 块 |
| 视频×3 + 曲线 | 见 fix01 目录 |

原始资产修复前快照 `fix01/original_sha256.txt`（6 文件）；GVHMR 产物、源视频、Skill 框架、其他 Skill 资产零改动；`validate top_rock` 全绿（关节身份 canonical 匹配、oneshot/free-root 配置不变、训练/部署引用同一版参考——train 与 deploy npz 同源 CSV 重生）。

## 8. RL Training 准入结论

**PASS**。两项确认缺陷均已修复并有定量验收；限位/速度/身份/一致性全绿；舞蹈特征保留（xy/yaw 逐位不变、上肢零改动、节奏相关 0.719）；PD 跌倒经对照实验排除参考几何归因。已知不阻塞项：支撑相滑动（§4）、右足限位裕量收紧至 5 mrad（训练动作空间在限内）。可直接进入 `TOP_ROCK_RL_TRAINING_01`：`python -m k1_skills train top_rock --background`。
