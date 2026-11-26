# K1_PINN_PAPER_METHODS_01 — 论文方法章节（正式草稿）

> 用途：可直接纳入论文的方法章节。编号说明：任务书建议该章为第 3 章；按完整初稿章节体系（引言为第 1 章）编为**第 2 章**，全部公式编号以本文件为准并在初稿中沿用。所有实现细节取自实际代码（`tools/pinn/stage0_common.py`、`train_stage0.py`、`contact_loss_ablation_02/train_ablation2.py`、`rollout/run_rollout.py`），未作猜测。
> 图编号引用对应《K1_PINN_PAPER_FIGURE_PLAN_01.md》（图 1 模型与建模框架，图 2 网络与损失结构）。

---

# 2 浮动基座机器人动力学模型与 PINN 建模方法

## 2.1 浮动基座机器人动力学模型

研究对象 Booster K1 为浮动基座全身机器人，含 1 个六自由度浮动基座与 22 个转动关节，广义坐标维数 nq=29（浮动基座位姿 7 维，含单位四元数）、广义速度维数 nv=28、控制维数 nu=22。其浮动基座动力学在广义坐标下可写为

$$
\boldsymbol{M}(\boldsymbol{q})\ddot{\boldsymbol{q}} + \boldsymbol{C}(\boldsymbol{q},\dot{\boldsymbol{q}}) + \boldsymbol{G}(\boldsymbol{q}) = \boldsymbol{qfrc}_{\mathrm{actuator}} + \boldsymbol{qfrc}_{\mathrm{constraint}},
\tag{1}
$$

其中 $\boldsymbol{q}\in\mathbb{R}^{29}$、$\dot{\boldsymbol{q}},\ddot{\boldsymbol{q}}\in\mathbb{R}^{28}$ 分别为广义位置、速度与加速度；$\boldsymbol{M}(\boldsymbol{q})\in\mathbb{R}^{28\times28}$ 为对称正定的广义质量矩阵；$\boldsymbol{C}(\boldsymbol{q},\dot{\boldsymbol{q}})$ 与 $\boldsymbol{G}(\boldsymbol{q})$ 分别为科氏力、离心力项与重力项（在 MuJoCo 中二者合成为偏置力 $\boldsymbol{qfrc}_{\mathrm{bias}}=\boldsymbol{C}+\boldsymbol{G}$）；$\boldsymbol{qfrc}_{\mathrm{actuator}}$ 为执行器产生的广义驱动力；$\boldsymbol{qfrc}_{\mathrm{constraint}}$ 为求解器输出的**广义约束力**，由接触、关节限位与等式约束等共同产生，不应等同于足底接触力。

式(1)将系统的运动状态划分为两类性质不同的区域。**自由运动区**：当样本满足无接触点（$n_{\mathrm{con}}=0$）、无等式约束行（$n_{\mathrm{efc}}=0$）且广义约束力为零（$\lVert\boldsymbol{qfrc}_{\mathrm{constraint}}\rVert_\infty=0$）时，式(1)右端约束力项消失，动力学完全由已知的刚体方程决定——这一区域是物理先验可信、残差形式无歧义的。**接触运动区**：约束力由接触几何、摩擦与求解路径共同决定，其解析形式难以可靠写出，式(1)对模型而言是不完备的。本文方法设计的出发点即利用这一划分：在自由运动区施加已知动力学的残差约束，在接触区考察不同约束形式的有效性边界。

## 2.2 数据驱动动力学模型

**输入输出定义。** 学习目标是加速度映射 $f_{\boldsymbol{\theta}}:\boldsymbol{x}\mapsto\hat{\ddot{\boldsymbol{q}}}$。输入 $\boldsymbol{x}\in\mathbb{R}^{81}$ 由广义状态与控制构成（图 1）：

$$
\boldsymbol{x} = \big[\,\underbrace{\boldsymbol{p}_{\mathrm{root}}}_{3};\ \underbrace{R_6(\boldsymbol{Q}_{\mathrm{root}})}_{6};\ \underbrace{\boldsymbol{q}_{\mathrm{joint}}}_{22};\ \underbrace{\dot{\boldsymbol{q}}}_{28};\ \underbrace{\boldsymbol{u}}_{22}\,\big],
\tag{2}
$$

其中根部位置 3 维、根部姿态由四元数转换为 6 维连续旋转表示 $R_6$（避免四元数双覆盖与符号歧义）、关节位置 22 维（前三项即 qpos 的 29 维编码为 31 维）、广义速度 28 维、控制量 22 维；输出 $\hat{\ddot{\boldsymbol{q}}}\in\mathbb{R}^{28}$ 为广义加速度。

**网络结构。** 全部模型组共享同一主干网络：81→256→256→256→28 的全连接网络，隐藏层激活函数为 SiLU，共 159,772 个参数（图 2）。选择小容量同构主干的目的是使对照实验中唯一的差异来源为损失函数，避免容量与结构差异混淆物理约束的作用。

**数据损失。** 训练目标采用逐自由度 z-score 化的加速度均方误差：

$$
L_{\mathrm{acc}}(\boldsymbol{\theta}) = \frac{1}{28}\sum_{i=1}^{28}\Big(\frac{\hat a_i - a_i^{*}}{s_i}\Big)^{2},
\tag{3}
$$

其中 $a_i^{*}$ 为仿真真值加速度，$s_i$ 为训练集第 $i$ 维加速度标准差。输入输出归一化统计量均仅由训练集计算并冻结。

**数据预处理与训练配置。** 训练与验证集采用冻结的 10σ 冲击帧过滤：剔除加速度幅值超过训练集 10 倍标准差的极端冲击帧（训练集由 16,167 个样本保留 15,978 个，剔除 189 个），避免个别冲击样本主导梯度；测试集不作过滤。所有模型组统一使用 AdamW 优化器（学习率 $10^{-3}$，权重衰减 $10^{-5}$），批大小 256，最多 2,000 个 epoch，以验证集数据损失早停（patience 200），梯度裁剪阈值 1.0。每个配置训练 5 个随机种子（0–4）；同一种子下各组共享网络初始化与数据顺序，构成配对比较。

## 2.3 Free-Dynamics PINN

Free-Dynamics PINN 将 2.1 节自由运动区的已知动力学作为归纳偏置注入训练。对自由运动区样本，定义物理残差（由式(1)移项，约束力为零）：

$$
\boldsymbol{r}_{\mathrm{free}} = \boldsymbol{M}(\boldsymbol{q})\,\hat{\ddot{\boldsymbol{q}}}_{\mathrm{phys}} + \boldsymbol{qfrc}_{\mathrm{bias}} - \boldsymbol{qfrc}_{\mathrm{actuator}},
\tag{4}
$$

其中 $\hat{\ddot{\boldsymbol{q}}}_{\mathrm{phys}}$ 为网络输出反归一化后的物理单位加速度；$\boldsymbol{M}$、$\boldsymbol{qfrc}_{\mathrm{bias}}$、$\boldsymbol{qfrc}_{\mathrm{actuator}}$ 取自数据采集时与真值加速度同帧的 MuJoCo 快照，不引入额外近似。训练损失为

$$
L = L_{\mathrm{acc}} + \lambda\, L_{\mathrm{free}},\qquad
L_{\mathrm{free}} = \frac{1}{|\mathcal{B}_{\mathrm{free}}|}\sum_{j\in\mathcal{B}_{\mathrm{free}}}\ \frac{1}{28}\sum_{i=1}^{28}\Big(\frac{r_{j,i}}{s^{r}_{i}}\Big)^{2},
\tag{5}
$$

其中 $\mathcal{B}_{\mathrm{free}}$ 为批内自由运动样本集合——物理损失**仅作用于其物理假设成立的样本**，接触区样本不承担残差惩罚；$s^{r}_{i}$ 为训练集自由域残差的逐维尺度（仅由训练集计算），用于平衡 28 个广义坐标间残差量级的差异。$\lambda\in\{0.1, 1, 10\}$ 为唯一研究变量。

两点实现说明。其一，残差定义经数据闭环校验：将真值加速度代入式(4)，残差归一化 RMS 为 $10^{-7}$ 量级，证明 $\boldsymbol{M}$、偏置力与驱动力三个快照量与加速度自洽。其二，式(5)中的逐维归一化使 $\lambda=1$ 并不表示数据项与物理项贡献相等：训练诊断显示该配置下物理项梯度约为数据项的 23–48 倍，$\lambda$ 只是相对标量系数。

## 2.4 接触动力学约束扩展

为检验自由域残差思想向接触区域的可用性，在基线 $L_{\mathrm{acc}}$ 上引入三种接触约束变体（记加速度误差 $\boldsymbol{e}=\hat{\ddot{\boldsymbol{q}}}_{\mathrm{phys}}-\ddot{\boldsymbol{q}}$；$\lambda=1$，其余条件与 2.2 节完全一致）。

**G3：质量矩阵加权误差约束。**

$$
L_{G3} = L_{\mathrm{acc}} + \lambda\cdot \frac{1}{|\mathcal{B}|}\sum_{j\in\mathcal{B}}\boldsymbol{e}_j^{\top}\boldsymbol{M}(\boldsymbol{q}_j)\,\boldsymbol{e}_j \Big/ s_3^{2},
\tag{6}
$$

以广义动能的二次型对 28 维误差加权，隐式提高大惯量自由度的监督权重；$\boldsymbol{M}$ 使用由上三角元素重建的完整 $28\times28$ 对称矩阵。

**G4：质量矩阵作用后误差平方范数约束。**

$$
L_{G4} = L_{\mathrm{acc}} + \lambda\cdot \frac{1}{|\mathcal{B}|}\sum_{j\in\mathcal{B}}\big\lVert\boldsymbol{M}(\boldsymbol{q}_j)\,\boldsymbol{e}_j\big\rVert_2^{2} \Big/ s_4^{2}.
\tag{7}
$$

G4 将误差先经质量矩阵投影（$\boldsymbol{M}\boldsymbol{e}$，量纲对应广义动量误差）再施加平方惩罚。**需强调式(6)与式(7)不是同一损失**：$\boldsymbol{e}^{\top}\boldsymbol{M}\boldsymbol{e}$ 与 $\lVert\boldsymbol{M}\boldsymbol{e}\rVert^2=\boldsymbol{e}^{\top}\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{e}$ 对误差各分量的加权结构不同，本文分别实现并比较。

**尺度归一化。** 式(6)(7)中的 $s_3^{2}$、$s_4^{2}$ 为**仅由训练集（保留样本）计算的标量尺度**（$s_3^{2}=9{,}694.8$，即 $\boldsymbol{e}^{\top}\boldsymbol{M}\boldsymbol{e}$ 在训练集上的均值；$s_4^{2}=31{,}805.1$，为 $\lVert\boldsymbol{M}\boldsymbol{e}\rVert^2$ 的对应值）。尺度归一化不改变各物理项单独的最优解，但改变其相对数据项的有效权重——实验实际训练目标是式(6)(7)所示归一化后的形式，而非未归一化的裸公式；训练诊断显示归一化后物理项梯度仍明显大于数据项（G3 约 10.6–12.0 倍，G4 约 22.4–46.9 倍）。因此本章消融结论限定于当前归一化方式与 $\lambda=1$ 设置。

**G5：广义约束力辅助预测头。** 在 G1 主干上并联一个 28 维辅助头预测广义约束力 $\boldsymbol{g}=\boldsymbol{qfrc}_{\mathrm{constraint}}$（共享主干参数，辅助头新增 7,196 个参数）：

$$
L_{G5} = L_{\mathrm{acc}} + \lambda\cdot\big\lVert\,\hat{\boldsymbol{g}}_{\mathrm{norm}}-\boldsymbol{g}_{\mathrm{norm}}\big\rVert_2^{2},
\tag{8}
$$

辅助标签经 z-score 标准化，统计量 $\boldsymbol{\mu}_g,\boldsymbol{\sigma}_g$ 仅由训练集计算。辅助头**只在训练阶段承担辅助监督**（以及其自身的泛化评价）；单步推理与多步状态预测只使用主任务的 $\hat{\ddot{\boldsymbol{q}}}$ 输出，辅助头在推理阶段丢弃，不向输入引入测试期 privileged 信息。G5 的动机在于：约束力是接触动力学的核心未知量，若网络能从 $(\boldsymbol{q},\dot{\boldsymbol{q}},\boldsymbol{u})$ 预测它，则接触区的动力学预测应受益。

## 2.5 多步状态预测方法

为检验单步动力学模型的实际预测价值，本文采用开环状态预测评价。每个窗口仅以真实状态 $(\boldsymbol{q}_{t_0},\dot{\boldsymbol{q}}_{t_0})$ 初始化一次，此后完全由模型自身预测递推，无 teacher forcing、不注入任何未来真值：

$$
\hat{\ddot{\boldsymbol{q}}}_t = f_{\boldsymbol{\theta}}(\hat{\boldsymbol{q}}_t,\hat{\dot{\boldsymbol{q}}}_t,\boldsymbol{u}_t),
\qquad
\hat{\dot{\boldsymbol{q}}}_{t+1} = \hat{\dot{\boldsymbol{q}}}_t + \Delta t\,\hat{\ddot{\boldsymbol{q}}}_t,
\qquad
\hat{\boldsymbol{q}}_{t+1} = \operatorname{mj\_integratePos}(\hat{\boldsymbol{q}}_t,\hat{\dot{\boldsymbol{q}}}_{t+1},\Delta t).
\tag{9}
$$

积分采用半隐式欧拉格式；由于浮动基座的位置分量含四元数，广义位置更新必须调用 MuJoCo 的 `mj_integratePos`，不能使用逐分量 $\boldsymbol{q}_{t+1}=\boldsymbol{q}_t+\dot{\boldsymbol{q}}_t\Delta t$；步长 $\Delta t$ 取数据记录的 float64 时间戳差。递推实现的正确性经真值加速度积分重建校验（速度重建误差上限 $9.2\times10^{-14}$）。

评价协议：在测试集的全部 3,110 个合法 8 步窗口上，每窗口连续递推 8 步后读取第 1/2/4/8 步（H=1/2/4/8）的预测状态误差；配对模型共享完全相同的窗口起点与记录控制序列；窗口不跨 episode。窗口按起点状态划入 2.1 节定义的互斥区域（自由运动 / 冲击 / 接触专属）；区域标签取自窗口起点的真实状态，不使用预测状态，不引入未来信息。

需要明确：本评价为**记录控制序列下的开环状态预测**，用于考察动力学模型在无外部校正条件下的短时预测品质，不是闭环控制实验，其结果不代表闭环控制性能。
