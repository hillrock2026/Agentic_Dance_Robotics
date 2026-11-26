# K1_DANCE_SKILL_003_MOTION_RETARGETING_01 — 执行报告

日期：2026-11-20。第三个街舞 Skill：Top Rock 真实动作资源 → K1 22-DOF 重定向 →
运动学验证 → 框架接入。

## 1. 选择了什么动作，来源是否可靠

**B-boy Top Rock**（全身步法 + 手臂摆动 + 躯干转体），来源链完整且全部为既有项目资产：

| 环节 | 资产 | 说明 |
|---|---|---|
| 原始视频 | `inputs/hiphop87_elements/segment_082.mp4`（manifest 行：411.44–416.36s，4.92s） | HIPHOP 87 元素库的真实街舞片段 |
| 人体重建 | `motion_studio/outputs/gvhmr/segment_082_cropped_72c3b3ef/` | GVHMR（SMPL，30fps） |
| 编辑 | `motion_studio/outputs/edits/toprock_edits.json`：trim 10–110 帧 + **0.7× 减速**（步法检查用） | 有意为之，非数据缺陷 |
| K1 重定向 | `datasets/k1_dance_v1/top_rock/`（trajectory.csv 286 帧 @50Hz，官方 CSV 序，root 世界坐标/yaw-only 四元数） | k1_action 逐链方向 IK（阻尼最小二乘，full_body），2026-11-16 产物，本任务**复用未改动** |

**可靠性**：重定向器自校验（`validation.json`）finite/joint_limits/timeline/lr_assignment 全过、dynamics 全过（无超速关节、无 needs_review 项）。证据限制：hiphop87 库未随仓库提供授权文本——来源记录为项目本地资产，许可状态待项目侧确认；GVHMR 重建为单目视频估计，绝对尺度/深度存在固有不确定性（与 arm_wave 同源同类）。

**为何选 top_rock 而非 popping**：popping 的肩 pitch 速度 15.8/17.9 rad/s 超出 8 rad/s 阈值且帧间跳变 0.36 rad（NEEDS_REVIEW）；top_rock 全项干净，且步法/转体的全身协调性正是本任务"非站立摆臂"的验收目标。popping 保留为后续候选（减速能解决其速度问题）。

## 2. 转换产物（K1 22-DOF 轨迹 + Root + 足底）

| 产物 | 路径 | SHA256（前 16） |
|---|---|---|
| 官方 29 列 CSV | `virtual_k1/assets/booster_assets/motions/K1/k1_top_rock.csv` | `a2da0b58b97c1847` |
| 训练 npz（Isaac FK，286×22，23 body） | `.../motions/K1/k1_top_rock.npz` | `8e19442607b55cea` |
| 部署 npz（MuJoCo FK，自由 root 原样保留） | `tools/.../robots/k1/motions/k1_top_rock.npz` | `c65f18974e5a2b8f` |
| 参考回放视频（286 帧 @50Hz，5.72s） | `k1_runs/outputs/top_rock/reference_playback.mp4` | `962f880bd7ff0fd6` |
| 运动学检查报告（JSON） | `k1_runs/outputs/top_rock/kinematic_checks.json` | `4f8cbfc0768ba315` |

转换器：新增通用脚本 `tools/booster_deploy/scripts/convert_k1_trajectory.py`（31 列 CSV → 29 列 + MuJoCo FK 部署 npz；root/朝向逐帧原样写入，`trunk` FK 自检与重定向 root 偏差 0.0000m）；训练 npz 走既有 `csv_to_npz.py`（与 arm_wave 完全同路径）。

## 3. 运动学检查结果（独立复算，不引用重定向器自评）

- **关节限位**：对照 K1_22dof.xml 逐关节范围 **零违例**，最小裕量 0.0207 rad（left_ankle_roll，与重定向器 0.02 限位裕量一致）；
- **动力学**：最大关节速度 3.24 rad/s（踝 roll），**无关节超 8 rad/s**；最大加速度 81 rad/s²；最大帧间跳变 0.065 rad，>0.2 rad 跳变 **0 处**——时间轴平滑连续；
- **Root 行为**：xy 路径 1.261 m、净位移 0.216 m（步法移动 ✓）；高度 0.536–0.598 m（站姿 0.555 上下 ±3cm 律动，无塌陷）；yaw 净转 −86.6°、跨度 100.2°（top rock 转体特征 ✓）、峰值角速度 101°/s；
- **足底行为**：左/右触地占比 63%/50%（步法单脚支撑交替 ✓）；右脚 14 帧离地（抬步 ✓）、最高抬至 0.11 m；**已知瑕疵**：左/右足底最低 −10.5/−3.3 mm（轻微穿地）、支撑相水平滑动均值 0.25/0.20 m/s（峰值 ~1 m/s）——与重定向器足链 42.5° 中位误差一致，属脚部重定向精度限制；
- **节奏**：与 SMPL 源编辑的能量包络相关 **0.748**，能量峰 6→5 个（段落结构保留；0.7× 减速下节拍间隔同比拉伸）；
- **姿态/协调性目检**：回放视频 8 关键帧抽检——全程直立高度稳定、手臂摆动与步法交替连贯、无破碎/翻转姿态（视频可复核）。

**证据限制**：足底穿地与支撑滑动来自 FK 层面参考（非物理仿真），其绝对数值依赖脚底几何近似（box geom 底面）；"动作像 top rock" 的风格判断基于视频目检与步法/转体统计特征，未做与源视频的逐帧姿态比对（SMPL→K1 无公共关节空间可直接逐帧对比，以链向误差中位数代理）。

## 4. 框架接入（复用，无平行结构）

- `skills/top_rock/skill.yaml`（schema v2，oneshot / free-root / 控制配置继承 k1_default + arm_wave 训练配方增益）；
- booster_train 任务 `Booster-K1-Top_Rock-v0`（3 小文件，复用 mj_dance_002 配方仅换动作）；
- deploy 任务 `k1_top_rock_trained`（继承 arm_wave 增益，oneshot 不循环）；
- **`validate top_rock` 全绿**（含关节身份 canonical 匹配、步进一致性、任务注册、±1 帧同步）；`list` 显示三号 Skill；`motion.process inspect` 正常；
- 测试回归：真仓关节身份契约测试扩展至三 Skill，全量 **161 passed / 0 failed**。

## 5. 是否具备 RL Training 条件

**具备。** 训练任务已注册、训练 npz 已就位（与 arm_wave 完全同管线）、参考动力学干净（无超速/跳变，RL 无需奖励 shaping 补偿）、控制配置经 validate 与两侧一致。训练命令即 `python -m k1_skills train top_rock --background`（≈2.6h @4096 env×3000 iter，与 arm_wave 同量级）。

建议训练前知悉的两项参考瑕疵（不阻塞）：足底轻微穿地（≤10.5mm）与支撑相滑动——BeyondMimic 奖励按身体位置/姿态跟踪，不含穿地惩罚，策略大概率学到略高的抬脚以物理可行；若训练后评价发现足部质量不足，优先做参考后处理（足底高度提升/滑动抑制），而非先改奖励。

## 6. 新增/修改文件清单

新增：`tools/booster_deploy/scripts/convert_k1_trajectory.py`、`.../play_reference_motion.py`、`.../check_reference_quality.py`（三个通用脚本，后续 Skill 直接复用）；`skills/top_rock/skill.yaml`；booster_train `robots/k1/top_rock/` 三文件；上表产物。
修改：`tools/booster_deploy/.../robots/k1/__init__.py`（+K1TopRockTrainedControllerCfg，追加式）；三个测试文件的真仓 Skill 列表加入 top_rock。
未改动：top_rock 重定向原始产物、既有 Skill 资产、框架代码（本任务零框架改动，仅新增接入）。
