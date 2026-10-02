# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FERPO: Forward Entropy-Regularized Policy Optimization](https://arxiv.org/abs/2610.02198) | FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。 |
| [^2] | [Muon meets Tamed Langevin: Momentum Preconditioning beyond Convex and gradient-Lipschitz Potentials](https://arxiv.org/abs/2610.02158) | 该论文提出一族非二次动能以构建带动量预条件化的欠阻尼朗之万系统，在势能既非凸也非全局梯度Lipschitz的宽松条件下证明了其保持目标吉布斯测度不变并指数收敛到平衡态，且相应的 Euler-Maruyama 离散化无需修改势能梯度即具有时间一致的矩界，保证了采样算法的稳定性。 |
| [^3] | [Sample complexity bounds for categorical Markov random fields via Discrete Diffusions](https://arxiv.org/abs/2610.02128) | 本文针对低阶马尔可夫随机场建模的局部依赖类别分布，为均匀加噪离散扩散模型建立了端到端的样本复杂度保证，其核心创新在于发现离散得分函数的“固定分解”结构——时间依赖与目标依赖可乘法分离，这与连续扩散的情形有本质不同。 |
| [^4] | [Wasserstein Gradient Flows and Forward-Only Diffusion Are Not Enough for Multimodal Sampling](https://arxiv.org/abs/2610.02081) | 本文证明Wasserstein梯度流与前向扩散过程具有相同的密度演化，因而继承了非平衡统计物理中已知的亚稳态和慢混合现象，其指数收敛保证并不能说明它们能高效采样多模态分布。 |
| [^5] | [The Curvature of Regret in Contextual Linear Optimization](https://arxiv.org/abs/2610.01980) | 本文证明了上下文线性优化中的遗憾在数据分布平均后呈局部二次曲率，给出了其闭式表达（支撑在法扇墙面上的矩阵值测度）以及仅需一次投影即可计算的弱收敛近似，并将其应用于决策感知的场景生成，实现30.8%的遗憾降低。 |
| [^6] | [Sharp Non-Asymptotic Analysis of the Penalized Challenger in $\beta$-EB-TCI for Bernoulli Bandits](https://arxiv.org/abs/2610.01951) | 本文首次对β-EB-TCI算法的惩罚挑战者给出精确非渐近分析，证明经验领导者成为真实最优臂后停止时间达到T_β*(μ)log(1/δ)（直至低阶项），从而为所有唯一最优臂的伯努利实例建立了非渐近高概率界。 |
| [^7] | [Pragmatic DML with AI-Learned Representations](https://arxiv.org/abs/2610.01935) | 该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。 |
| [^8] | [Error-Corrected Inference-Time Scaling for Imperfect Diffusion Models](https://arxiv.org/abs/2610.01933) | 提出基于能量的Feynman-Kac校正器框架，能在推理时实时纠正不完美扩散模型的路径追踪误差和终点失配问题。 |
| [^9] | [Generalized Engression Models](https://arxiv.org/abs/2610.01823) | 本文提出广义engression模型，一个统一的非参数分布回归框架，可处理任意类型（连续、二元、类别、有序、排序）且相互条件依赖的多变量结果，通过数据类型特定的连接函数和随机扰动实现基于梯度的训练。 |
| [^10] | [Inferring Multi-Timescale Neural Dynamics with Switching Linear Dynamical Systems](https://arxiv.org/abs/2610.01786) | 提出了多时间尺度切换线性动力学系统（MTS-SLDS）框架，能够从高维神经群体记录中识别随行为状态变化的多时间尺度潜在神经动力学。 |
| [^11] | [In-context Learning of Single-index Targets: Comparing Kernel and Feature Learners](https://arxiv.org/abs/2610.01712) | 本文通过复本方法比较了核学习器与特征学习器两种单层注意力架构在非线性单指标任务上的上下文学习能力，推导出记忆与泛化误差的理论预测和相图，揭示了两类架构各自占优的条件。 |
| [^12] | [Lower Bounds for Stochastic First-Order Algorithms with Variance Reduction in Nonconvex--Concave Minimax Optimization](https://arxiv.org/abs/2610.01662) | 该论文首次为允许方差缩减技术的随机一阶算法在非凸-凹极小极大优化问题中建立了复杂度下界，突破了现有下界结果对算法类型的限制。 |
| [^13] | [The hidden advantage of mask resampling: a theory of masked autoencoders](https://arxiv.org/abs/2610.01578) | 该论文首次从理论上证明掩码自编码器的掩码线性重建能在PCA失效的情形下以线性样本复杂度恢复潜在特征，并量化了掩码重采样中更大掩码多样性降低样本复杂度的统计优势。 |
| [^14] | [Exact Distinguishability in Non-Markovian Decision Processes](https://arxiv.org/abs/2610.01527) | 本文首次精确刻画了非马尔可夫决策过程（RDP）中固定行为策略数据能否区分两个候选模型的条件，证明观测等价候选的后验几率在任何样本量下都恒等于先验几率，提出了线性时间判定算法PEC并在Lean 4中完成形式化验证，同时发现已有工作的可区分性假设在多数测试环境中并不成立。 |
| [^15] | [Langevin-Informed Transfer Learning: Replacing Target Samples by Black-Box Feedback](https://arxiv.org/abs/2610.01522) | 提出了朗之万信息驱动的迁移学习（LITL）框架，仅依靠黑盒反馈即可从有偏源样本中恢复目标朗之万动力学，实现谱形式的动力学重构与慢流形梯度场估计。 |
| [^16] | [No Model Required: Text Entropy Rate Filtering Mitigates Iterative Fine-Tuning Collapse](https://arxiv.org/abs/2610.01493) | 该论文提出了一种完全无需模型的非参数化Kontoyiannis熵率估计器，仅通过原始文本的匹配长度统计来过滤训练数据，能够有效缓解合成数据迭代微调导致的模型崩溃，其效果显著优于依赖模型对数概率的传统过滤方法。 |
| [^17] | [Zero Flux: Flow-Based Comparison of High-Dimensional Discrete Distributions](https://arxiv.org/abs/2610.01472) | 提出“零通量”差异准则，将基于流匹配的分布比较方法从连续分布扩展到高维离散分布，证明在独立耦合下当且仅当两个分布相同时所有局部概率通量在中点处消失，且该度量可分解为局部贡献并从样本中高效估计。 |
| [^18] | [Tight Transition Time Bounds for Separable Logistic Regression at the Edge of Stability](https://arxiv.org/abs/2610.01459) | 本文推翻了此前关于稳定性边缘现象中过渡时间与步长无关的猜想，证明了最坏情况下过渡时间随步长按 $\Theta((\log\eta)^{\min\{n-2,d-2\}})$ 增长，且该界在任意维度下均紧致。 |
| [^19] | [On skew-symmetric distributions and their use in Monte Carlo sampling algorithms: coordinate-free, Gibbs-style and manifold versions of the Barker proposal](https://arxiv.org/abs/2610.01448) | 本文回顾了基于斜对称分布的 Barker 提议并提出三种扩展（无坐标变体、Gibbs 风格变体和简化流形版本），其中 Gibbs 风格变体在每次部分坐标更新时重新评估梯度，从而在相关目标分布上提升了采样效率。 |
| [^20] | [Optimal Transport Meets Reinforcement Learning: A Survey](https://arxiv.org/abs/2610.01413) | 本综述系统梳理了最优传输（OT）在强化学习中的应用，指出当待比较的概率分布重叠较弱时，OT比传统散度度量更有效，并从OT的角色、被比较的分布、OT形式化方法及时间结构处理四个维度对现有方法进行了分类。 |
| [^21] | [Clifford Sheaf Neural Networks](https://arxiv.org/abs/2610.01322) | 本文提出克利福德层神经网络（CSNN），通过用K项夹逼替代代数同态作为限制映射，构造出天然半正定、无需旋量约束且能混合不同阶的层拉普拉斯算子，并从阶耦合、自同态空间覆盖范围和条件数三个维度系统刻画了该限制映射族。 |
| [^22] | [IQS-BO: In-Context Query Selection for Bayesian Optimisation](https://arxiv.org/abs/2610.01269) | 本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。 |
| [^23] | [Counterfactual Generation via Flow Matching: Coupling-Sensitive End-to-End Rates](https://arxiv.org/abs/2610.01193) | 本文提出一种基于流匹配的反事实生成方法，将双重稳健训练目标与学习到的源-结果耦合相结合，并证明其常数步欧拉离散化的KL误差界由耦合下的位移矩控制、与维度近线性相关，而非依赖速度场的全局正则性。 |
| [^24] | [Gradient-Guided Density Peak Clustering](https://arxiv.org/abs/2610.01050) | 本文提出梯度引导的密度峰值聚类（GGDPC），通过在每次最近邻上坡搜索前执行梯度上升步骤来稳定低密度区域的不规则路径，并在局部模态恢复、调整兰德指数等五个互补准则下建立了该方法的统计一致性理论。 |
| [^25] | [Posterior sampling by source-space MCMC via prior-based few-step transport maps](https://arxiv.org/abs/2610.01034) | 该论文提出了一种源空间广义贝叶斯推断框架，利用一步或少步的改进MeanFlow（iMF）映射表示隐式先验，在高斯源空间中通过MCMC进行后验采样，并给出了精确后验与学习后验之间基于训练次优性和模型类近似误差的Wasserstein误差理论保证。 |
| [^26] | [Tolerance-Based Fairness Auditing: Violation Certification and Sensitivity Screening](https://arxiv.org/abs/2610.01005) | 该论文提出了一个统一的基于容差的公平性审计框架，针对违规认证和敏感性筛查两个互补目标，分别通过约束经验似然检验（结合最不利点校准）控制虚假违规声明风险，并降低漏检违规的概率。 |
| [^27] | [The Price of Correlated Tests: How Strict Should a Model Release Gate Be?](https://arxiv.org/abs/2610.00993) | 该论文将模型发布门槛的设定视为一个统计设计问题，证明当测试之间存在相关性时，更严格的门槛总能提升可靠性，因此最优策略是选择仍能满足可靠性目标的最为宽松的门槛，而“全部通过”式门控会随着测试数量的增加把保留的好模型比例推向零。 |
| [^28] | [Joint Branch-Space Transform Coding for Diffusion Activation Quantization with Classifier-Free Guidance](https://arxiv.org/abs/2610.00930) | 该论文提出分支空间变换编码方法（含GCBT），利用离线推导的2x2正交矩阵联合旋转CFG的条件与无条件激活以挖掘其强相关结构，并结合引导方向和跨分支二阶矩，在固定比特预算下显著提升扩散模型激活量化的保真度。 |
| [^29] | [Platonic Task Arithmetic](https://arxiv.org/abs/2610.00929) | 本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。 |
| [^30] | [Block Optimism for Nonstationary Bandits with Latent Linear Dynamics](https://arxiv.org/abs/2610.00911) | 该论文提出基于自适应块级乐观的UCB算法，借助循环近似截断无限记忆奖励过程，将具有潜线性动态的非平稳赌博机的遗憾值从 $\tilde{O}(T^{2/3})$ 提升到更优速率。 |
| [^31] | [Learning Multiple Timescales for Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2610.00849) | 提出GITA方法，将单一价值函数以时间抽象尺度k为条件，并通过在多个k值上聚合优势加权监督信号来训练一个策略，从而解决离线目标条件强化学习在长时程任务中价值信号消失的问题。 |
| [^32] | [Inference for stochastic differential equations driven by weighted sub-fractional Brownian motion using neural networks and the Euler approximation](https://arxiv.org/abs/2610.00793) | 本文提出将神经网络与Euler近似相结合，对由加权亚分数布朗运动这类高斯过程驱动的随机微分方程进行统计推断，从离散观测数据中估计漂移系数、扩散系数与噪声协方差。 |
| [^33] | [Scalable Multi-Task Inverse Reinforcement Learning](https://arxiv.org/abs/2610.00758) | 该论文提出一种基于低秩假设的多任务逆强化学习方法，通过汇集多个智能体的数据缓解覆盖度要求，并使规划计算量随秩而非任务数扩展，从而实现对多任务在新环境中的可扩展高效评估。 |
| [^34] | [Learning to Price Electricity for Optimal Demand Response](https://arxiv.org/abs/2610.00755) | 本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。 |
| [^35] | [Signal-Noise Factorization Isolates Nuisance Variation into Removable Subspaces](https://arxiv.org/abs/2610.00751) | 该论文提出在训练中强化信号-噪声分解（SNF）与信号-信号分解（SSF）的正则化方法，将干扰变异隔离到可移除的子空间中，实验表明增强SNF能显著提升模型在CIFAR-100及医学图像腐蚀数据集上的性能。 |
| [^36] | [Sequential Functional Structured Tucker Compression for Large Language Model Attentions](https://arxiv.org/abs/2610.00717) | 提出FTC序列化结构化压缩框架，在固定存储预算下联合利用Q/K/V头原生结构并适应先前压缩引起的表示偏移，无需微调即可在6B至32B的大语言模型上实现最先进的注意力压缩效果。 |
| [^37] | [Multifidelity Formulations for Triangular Transport](https://arxiv.org/abs/2610.00698) | 该论文提出两种多保真度策略——相邻保真度层级间映射组合的分层方法与保单调性参数化修正的非分层方法，利用丰富的低保真度数据在仅有少量高保真度样本时更准确地构建三角传输映射。 |
| [^38] | [How Divergence Becomes Decision Flips in Compressed Language Models](https://arxiv.org/abs/2610.00694) | 该研究证明全变差而非KL散度能直接预测压缩语言模型的决策翻转率，二者比例的中位数为1.05且无需拟合常数，而KL散度因先对词元取平均而无法可靠比较不同模型和语料库下的压缩效果。 |
| [^39] | [Grand Canonical Generators](https://arxiv.org/abs/2610.00683) | 提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。 |
| [^40] | [Adaptive Conformal Prediction for Image Regression Models with Application to an Inertial Confinement Fusion Emulator](https://arxiv.org/abs/2610.00535) | 本文提出了基于最近邻的自适应保形预测方法（ACPNN），为图像回归模型提供依赖于输入的局部自适应不确定性量化，并将其应用于惯性约束聚变仿真器。 |
| [^41] | [Fractional Laplace Neural Operators: Exact Architectures, an Expressivity Frontier at Criticality, and Certified Stability for Memory-Driven Network Dynamics](https://arxiv.org/abs/2610.00515) | 本文提出分数拉普拉斯神经算子（fLNO），证明单层图-谱架构可精确表示线性Volterra记忆算子，揭示了有限有理实现无法再现分数记忆非整数临界渐近行为的表达能力边界，并给出构造上即保证稳定性裕度的可训练参数化方法。 |
| [^42] | [Heteroskedastic Canonical Polyadic Tensor Decomposition](https://arxiv.org/abs/2610.00498) | 本文提出异方差CP分解（HCP），通过引入低秩精度张量来建模张量条目的异方差性，并采用交替分块坐标上升方法从含噪观测中同时估计低秩均值与精度张量，其计算复杂度与CP-ALS相当。 |
| [^43] | [Exact information accounting for SGD methods](https://arxiv.org/abs/2610.00446) | 该论文提出了SGD的精确信息论分析框架，证明预条件SGD步骤是高斯贝叶斯模型的后验均值更新，并给出一个信息核算恒等式，将凸收敛、鞍点逃逸、平坦性与泛化关系、学习率调度及各类SGD变体统一起来。 |
| [^44] | [ChainLoRA: Geometry-Preserving Task Vector Merging for Continual Learning in LLMs](https://arxiv.org/abs/2610.00431) | ChainLoRA 提出了一种免回放的持续学习合并框架，通过链式更新训练与自适应 SVD 合并来保持任务向量的几何结构，在恒定的历史状态与正则化开销下，平衡大语言模型的知识保留与新任务适应。 |
| [^45] | [IrekoGPT: Turning Structured Pruning into Post-Hoc Slimmable LLMs](https://arxiv.org/abs/2610.00426) | IrekoGPT提出一种事后方法，通过保留SliceGPT投影矩阵、多压缩率逐层校准和无梯度岭回归修正，将预训练大语言模型转换为推理时可调节宽度的可瘦身模型，并在高压缩率下显著优于基于PCA的朴素瘦身方法。 |
| [^46] | [Target-Dependent Limits of Causal Repair: A Leading-Log Frontier in a Gaussian Model](https://arxiv.org/abs/2610.00424) | 该论文在高斯因果模型中量化了因果预测器潜在改进与实际学到的修复增益之间的差距，证明在 1/k 学习尺度下所有可行学习器都面临 k^{-2} 的评估下界，并在幅度充裕情形下刻画出尖锐的前导对数评估指数前沿 min{ℓ_k, 2kη_k/U}。 |
| [^47] | [Learning to Cover Locally: Graph Neural Combinatorial Optimization under a Hard Information Horizon](https://arxiv.org/abs/2610.00422) | 本文形式化了每个节点只能看到k跳邻域的“硬信息视界”下组合优化问题（以OLSRv2协议的NP难MPR选择为实例），并证明L层GNN严格等价于L跳选择器，即模型容量无法弥补信息半径的不足。 |
| [^48] | [Transferable Graph Metanetworks](https://arxiv.org/abs/2610.00420) | 提出可迁移图元网络，基于表示不变性与连续性两项设计原则，使元网络的性能能够跨不同宽度的输入神经网络迁移，从而实现“在小网络上训练、在大网络上评估”的效率提升。 |
| [^49] | [VANDAM: Viewing a nucleotide sequence with DNA molecular priors](https://arxiv.org/abs/2610.00411) | VANDAM框架通过在自监督训练中预测区域分子性质并在有标签时于输入端注入局部特征，将DNA分子先验融入基因组基础模型，从而在多种架构和下游任务上持续提升性能。 |
| [^50] | [RACE: Residual-Aware Test-Time Adaptation for Neighbor-Rich Time-Series Foundation Model Forecasting](https://arxiv.org/abs/2610.00405) | 该论文提出RACE，一种残差感知的测试时自适应方法，通过处理邻居序列间相互矛盾的残差证据与跨场景变化的残差模式，在无需逐域微调的情况下提升时间序列基础模型在邻居丰富预测场景中的预测性能。 |
| [^51] | [MatrixReward: Reward from Rubric Matrix for Open-Ended Generation](https://arxiv.org/abs/2610.00389) | MatrixReward通过在每条评分标准下对采样回答进行两两比较构建胜率矩阵，利用列离散度衡量评分标准的区分能力、列相关性检测评分标准的重复性，从而生成数据自适应的评分标准权重，为缺乏标准答案的开放式生成构造更有效的奖励信号。 |
| [^52] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^53] | [STCFormer: Adaptive Spatio-Temporal Modeling with Dynamic Cluster Transformer for Station-based Weather Forecasting](https://arxiv.org/abs/2610.00377) | 提出STCFormer，一种根据每个时间补丁内站点局部演化动态分组站点的自适应时空Transformer，通过融合簇内细粒度局部注意力与区域级全局注意力来提升站点天气预报的准确性。 |
| [^54] | [M$^2$Weather: A Benchmark for Joint Multi-Station and Multi-Variable Weather Forecasting](https://arxiv.org/abs/2610.00370) | 该论文提出了 M$^2$Weather 基准，通过收集覆盖法国、欧洲和全球三个空间尺度的 2,809 个高质量站点与 5 个物理耦合天气变量，并提供统一的训练与评估协议，首次实现了对多站点空间依赖与多变量物理耦合的联合系统性评估。 |
| [^55] | [Partial AUC Maximization from Positive-unlabeled Data](https://arxiv.org/abs/2610.00284) | 本文提出了一种无需负例数据、仅利用正例和未标注数据即可最大化部分AUC（pAUC）的方法，解决了实际中负例数据因隐私或标注专业性要求而难以收集的问题。 |
| [^56] | [Weighted Data Selection: Sharp Upper-Half and Five-Dimensional Laws](https://arxiv.org/abs/2610.00101) | 该论文为加权最小二乘的最小范数学习器证明了在 ⌈3d/2⌉ ≤ n ≤ 2d−1 范围内精确的风险定律 Γ_d(n)=3−n/d，并在 (d,n)=(5,6) 时进一步证明 Γ_5(6)=11/5，其完整的数据集层面之上界与尖锐性构造均已在 Lean 4 中形式化验证。 |
| [^57] | [Dynamic Spatial Bayesian Machine Learning Model: Applications to Intergenerational Economic Mobility and Geographic Income Inequality in the United States](https://arxiv.org/abs/2610.00072) | 该论文提出了一种结合马蹄铁收缩的动态空间面板贝叶斯可加回归树模型（DSP-BART-HS），在九种模拟情景中均达到最优或与最优统计上无差异的性能，尤其显著优于传统区域-时间聚合方法，能有效处理个体层面的非线性效应。 |
| [^58] | [Bandits with Multiple Optimal Arms: Minimax Regret and Non-Adaptivit](https://arxiv.org/abs/2609.38659) | 该论文针对具有多个最优臂的多臂老虎机问题，通过对子采样算法的更精细分析建立了近乎极小极大最优的遗憾界 $\tilde{O}(\frac{K-A}{\sqrt{KA}}\sqrt{T})$，给出匹配下界，并证明了解最优臂数量对于达到近乎最优遗憾是必要的。 |
| [^59] | [Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems](https://arxiv.org/abs/2609.37944) | 本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。 |
| [^60] | [Unbiased Top-$k$ Estimation for On-Policy Distillation](https://arxiv.org/abs/2609.34447) | 该论文提出了用于在线策略蒸馏中反向KL散度梯度估计的无偏Top-k估计方法，在有限计算成本下实现比采样token更丰富的分布监督。 |
| [^61] | [Ensembles of Exactly Solved Subsamples for Clusterwise Regression: Trimming Without a Trimming Level](https://arxiv.org/abs/2609.31019) | 该论文提出一种基于精确求解随机子样本的聚类回归集成方法，可自动估计截尾水平而无需预先设定，在响应变量含高达20%粗大离群值时实现0.89的最坏情况准确率，优于传统的截尾交替法。 |
| [^62] | [Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing](https://arxiv.org/abs/2609.28611) | 该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。 |
| [^63] | [Rank and computation of the pathlifting Jacobian of a DAG ReLU network](https://arxiv.org/abs/2609.18682) | 本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。 |
| [^64] | [A distribution-free certification framework for trustworthy crash-severity prediction](https://arxiv.org/abs/2609.11592) | 该论文提出了一个无需分布假设的认证框架，可无需修改地包裹任何交通事故严重程度预测模型，利用KABCO序数标签的结构特性，提供序数预测集、逐类有效性、向未观测真实严重程度的覆盖率传递以及部署偏移下的单侧保证。 |
| [^65] | [Tensor-Train Weak SINDy: Identifying High-Dimensional Nonlinear Dynamics](https://arxiv.org/abs/2609.09434) | 本文提出TT-WSINDy方法，通过张量列车格式结合MANDy和WSINDy技术，实现对指数增长的候选函数空间的高效搜索，从而避免维数灾难并完成高维非线性动力学的数据驱动识别。 |
| [^66] | [Credal Large Language Models for Semantic Commitment under Uncertainty](https://arxiv.org/abs/2608.23244) | 通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。 |
| [^67] | [The Exceedance Design Effect: Effective Sample Size for Thresholds under Clustering](https://arxiv.org/abs/2608.21262) | 本文提出在聚类相关数据下，设置阈值时需采用不同于平均值的有效样本量计算方法，以准确评估阈值的可靠性。 |
| [^68] | [ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum](https://arxiv.org/abs/2607.04535) | ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。 |
| [^69] | [Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts](https://arxiv.org/abs/2606.28228) | 该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。 |
| [^70] | [INDEQS: Informed Neural controlled Differential EQuationS](https://arxiv.org/abs/2606.19138) | 该论文提出INDEQS方法，将预先已知的有向图结构以不同架构位置融入基于图的神经控制微分方程（NCDE）时间序列预测模型，通过分离节点间隐藏状态内层混合与向量场-控制外层混合，并提供轻量级图约束变体和基于自适应图卷积的更具表现力的变体来有效利用图先验知识。 |
| [^71] | [Flow-Transformed Implicit Processes for Function-Space Variational Inference](https://arxiv.org/abs/2606.01954) | 提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。 |
| [^72] | [CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](https://arxiv.org/abs/2605.20468) | 提出了CASCADE共形预测框架，通过将筛查分类器的认知不确定性传播到下游回归任务中，动态缩放帕金森病药物剂量预测的预测区间，为两阶段临床决策支持提供不确定性自适应的可靠性量化。 |
| [^73] | [Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits](https://arxiv.org/abs/2604.14908) | 本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。 |
| [^74] | [Optimal Centered Active Excitation in Linear System Identification](https://arxiv.org/abs/2604.05518) | 该论文提出了一种基于普通最小二乘法和半定规划的线性系统辨识主动学习算法，通过最优中心化噪声激励实现了理论上最优的样本复杂度，其上界与任意算法的下界在常数因子内匹配，并明确了样本复杂度对状态维度等系统参数的依赖关系。 |
| [^75] | [Casewise and Cellwise Robust Tensor-on-Tensor Regression](https://arxiv.org/abs/2603.25911) | 本文提出了一种名为ROTOT的稳健张量对张量回归新方法，能够同时应对样本级与单元格级异常值并处理缺失值。 |
| [^76] | [Notes on Forr\'e's Notion of Conditional Independence and Causal Calculus for Continuous Variables](https://arxiv.org/abs/2603.24333) | 本札记进一步阐释了Forré的转移条件独立性框架的动机与文献联系，揭示了测度论因果演算中的微妙之处，并将ID算法的“单行”表述推广到一般测度论设定。 |
| [^77] | [Extending SSMs with the Exponentially Weighted Signature](https://arxiv.org/abs/2603.19198) | 该论文提出指数加权签名（EWS），通过可学习矩阵生成元、将步长推广为输入因果泛函的时钟以及更高截断深度，将状态空间模型（含Mamba）统一并扩展为签名理论下的连续时间模型，在长时间序列分类任务上取得了最优表现。 |
| [^78] | [Hierarchy of discriminative power and complexity in learning quantum ensembles](https://arxiv.org/abs/2601.22005) | 本文提出了量子系综距离度量的层级结构MMD-$k$，揭示了判别力与统计效率之间的严格权衡——估计MMD-$k$需要$\Theta(N^{1-1/k})$个样本，而任何具有完全判别能力的稳定距离度量至少需要$\Omega(N)$个样本。 |
| [^79] | [A Unified Kantorovich Duality for Multimarginal Optimal Transport](https://arxiv.org/abs/2601.17171) | 该论文建立了多边际最优传输的统一Kantorovich对偶理论，在紧与非紧（波兰空间）情形下均证明了对偶最优解的存在性，并刻画了最优对偶势的结构。 |
| [^80] | [BalLOT: Balanced $k$-means clustering with optimal transport](https://arxiv.org/abs/2512.05926) | 提出 BalLOT 算法，将最优传输融入交替最小化框架以求解平衡 $k$-means 聚类，并从理论上证明了其整值耦合性质与植入聚类恢复保证，数值实验验证了其快速有效性。 |
| [^81] | [Provable FDR Control for Deep Feature Selection: Deep MLPs and Beyond](https://arxiv.org/abs/2512.04696) | 首个在通用深度学习设置下为特征选择提供错误发现率（FDR）控制理论保证的框架，可覆盖多层感知机、卷积/循环网络、注意力机制等广泛架构。 |
| [^82] | [SSLfmm: An R Package for Semi-Supervised Learning with Mixed Missingness](https://arxiv.org/abs/2512.03322) | SSLfmm是一个R包，通过联合建模标签缺失机制与类别分布，支持混合缺失机制下的半监督学习，并提供了统一的R接口。 |
| [^83] | [Efficient Solvers for SLOPE in R, Python, Julia, and C++](https://arxiv.org/abs/2511.02430) | 该论文提出了 R、Python、Julia 和 C++ 中高效求解 SLOPE 问题的软件包套件，采用高效的混合坐标下降算法支持多种损失函数和数据结构，并在速度上超越了现有的 SLOPE 实现。 |
| [^84] | [The Benchmarking Epistemology: Validity Theory for Evaluating Machine Learning Models](https://arxiv.org/abs/2510.23191) | 本文借鉴心理学有效性理论，提出使基准测试科学推断所需假设显式化的有效性条件，并通过ImageNet和脆弱家庭挑战赛两个案例，将预测性基准测试确立为机器学习中一种独特的认知实践。 |
| [^85] | [Meta-reinforcement learning with minimum attention](https://arxiv.org/abs/2505.16741) | 该论文将Brockett的最小注意力（最小作用原理）作为奖励项引入强化学习，与元学习相结合，在高维非线性动力学中显著提升了少样本快速适应能力并降低了扰动方差，且可无缝集成到DreamerV3、MAMBA等现代世界模型中。 |
| [^86] | [Stochastic Optimal Control for Continuous-Time fMRI Representation Learning](https://arxiv.org/abs/2502.04892) | 该论文提出将自监督学习重构为随机最优控制问题的新框架，把大脑活动建模为连续时间潜在动力学，并统一掩码自编码（MAE）与联合嵌入预测（JEPA），从而学习到对时间不规则性和噪声鲁棒的fMRI表征。 |
| [^87] | [Convergence Analysis of the Wasserstein Proximal Algorithm beyond Geodesic Convexity](https://arxiv.org/abs/2501.14993) | 本文在不假设测地凸性的条件下，借助Wasserstein版本的Polyak-Łojasiewicz不等式证明了Wasserstein近端算法的无偏线性收敛速率，并改进了强测地凸性下已有的收敛率结果。 |
| [^88] | [Exploiting Exogenous Structure for Sample-Efficient Reinforcement Learning](https://arxiv.org/abs/2409.14557) | 该论文针对外生马尔可夫决策过程，建立了离散MDP、Exo-MDP与离散线性混合MDP之间的表征等价性，并在外生状态不可观测时证明了 $\Theta(Hr\sqrt{K})$ 的匹配极小极大遗憾界，为样本高效的强化学习提供了理论与算法基础。 |
| [^89] | [Roto-translated Local Coordinate Frames For Interacting Dynamical Systems](https://arxiv.org/abs/2110.14961) | 本研究提出了为每个节点-对象引入局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。 |
| [^90] | [Geometry-Aware Adaptation for Pretrained Models.](http://arxiv.org/abs/2307.12226) | 本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。 |

# 详细

[^1]: FERPO：前向熵正则化策略优化

    FERPO: Forward Entropy-Regularized Policy Optimization

    [https://arxiv.org/abs/2610.02198](https://arxiv.org/abs/2610.02198)

    FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。

    

    在连续控制的在线强化学习中，若干最先进的方法利用学习到的评论家对动作的梯度来改进策略。然而，评论家通常被训练用于预测回报，准确的价值预测并不一定能产生准确的动作导数，这可能导致不可靠的策略更新。我们提出前向熵正则化策略优化（FERPO），这是一种在策略的最大熵强化学习算法，它利用评论家的价值进行策略改进，而无需对评论家关于动作进行求导。FERPO从由熵和Kullback-Leibler（KL）散度正则化的策略改进目标中推导出最优的目标动作分布。随后，我们通过最小化前向KL目标将Actor拟合到该目标分布上，该目标使用自归一化重要性采样（SNIS）进行估计，动作从rollout策略中抽取。通过限制目标分布……

    arXiv:2610.02198v1 Announce Type: cross  Abstract: Several state-of-the-art methods for online reinforcement learning in continuous control improve policies using action gradients of a learned critic. However, critics are typically trained to predict returns, and accurate value predictions do not necessarily yield accurate action derivatives, potentially leading to unreliable policy updates. We propose Forward Entropy-Regularized Policy Optimization (FERPO), an on-policy maximum entropy reinforcement learning algorithm that performs policy improvement using critic values without differentiating the critic with respect to actions. FERPO derives an optimal target action distribution from a policy-improvement objective regularized by entropy and Kullback-Leibler (KL) divergence. We then fit the actor to this target by minimizing a forward-KL objective, estimated using self-normalized importance sampling (SNIS) with actions drawn from the rollout policy. By limiting the target distribution
    
[^2]: Muon 遇见驯服朗之万：超越凸性与梯度Lipschitz势能的动量预条件化

    Muon meets Tamed Langevin: Momentum Preconditioning beyond Convex and gradient-Lipschitz Potentials

    [https://arxiv.org/abs/2610.02158](https://arxiv.org/abs/2610.02158)

    该论文提出一族非二次动能以构建带动量预条件化的欠阻尼朗之万系统，在势能既非凸也非全局梯度Lipschitz的宽松条件下证明了其保持目标吉布斯测度不变并指数收敛到平衡态，且相应的 Euler-Maruyama 离散化无需修改势能梯度即具有时间一致的矩界，保证了采样算法的稳定性。

    

    我们考虑在矩阵空间上从吉布斯分布进行采样的研究问题，其中势能既非凸函数也不满足全局梯度Lipschitz条件。我们引入了一族非二次动能，从而得到一种新的具有动量预条件化的欠阻尼朗之万系统，其中动能的梯度起到对动量进行光滑谱驯服的作用。我们证明，在对势能的这些宽松假设下，所得动力学保持目标吉布斯测度不变，并且我们在加权全变差距离下建立了向平衡态的指数收敛性。最后，我们证明相应的 Euler-Maruyama 离散化在不对势能梯度做任何修改的情况下具有时间一致的矩界，从而保证了所得采样算法的稳定性。

    arXiv:2610.02158v1 Announce Type: cross  Abstract: We consider the problem of sampling from Gibbs distributions on matrix spaces whose potential energies are neither convex nor globally gradient-Lipschitz. We introduce a family of non-quadratic kinetic energies that lead to a new underdamped Langevin system with momentum preconditioning, in which the gradient of the kinetic energy acts as a smooth spectral taming of the momentum. We prove that, under these relaxed assumptions on the potential, the resulting dynamics leaves the target Gibbs measure invariant, and we establish exponential convergence to equilibrium in a weighted total variation distance. Finally, we show that the corresponding Euler-Maruyama discretization admits moment bounds that are uniform in time, without any modification of the potential gradient, which ensures the stability of the resulting sampling algorithm.
    
[^3]: 基于离散扩散的类别马尔可夫随机场样本复杂度界

    Sample complexity bounds for categorical Markov random fields via Discrete Diffusions

    [https://arxiv.org/abs/2610.02128](https://arxiv.org/abs/2610.02128)

    本文针对低阶马尔可夫随机场建模的局部依赖类别分布，为均匀加噪离散扩散模型建立了端到端的样本复杂度保证，其核心创新在于发现离散得分函数的“固定分解”结构——时间依赖与目标依赖可乘法分离，这与连续扩散的情形有本质不同。

    

    统计学、经济学和物理学中的许多应用需要从具有局部依赖结构的高维类别分布中进行采样。例子包括有限记忆语言模型、统计物理和蛋白质折叠中的Ising与Potts系统等。在现代机器学习中，离散扩散已成为采样此类数据的一种灵活方法，并展现出强大的实证性能。受此启发，我们针对由低阶马尔可夫随机场（MRF）建模的局部依赖结构，为采用均匀加噪的离散扩散模型开发了具有端到端样本复杂度保证的学习方法。我们的主要技术洞察是离散得分函数的一种全新“固定分解”方法。它表明，与连续扩散不同，得分函数可分解为多个分量，其中对时间的依赖性与对目标的依赖性呈乘法分离。基于这一分解，我们提出了一种……（摘要在此处被截断）

    arXiv:2610.02128v1 Announce Type: cross  Abstract: Many applications in statistics, economics, and physics require sampling from high-dimensional categorical distributions with local dependence structures. Examples include finite memory language models, Ising and Potts systems in statistical physics and protein folding, etc. In modern machine learning, discrete diffusions have emerged as a flexible approach for sampling such data, with strong empirical performance. Motivated by this, we develop learning methods with end-to-end sample complexity bounds for discrete diffusion with uniform noising under local dependence, which we model through low order Markov random fields (MRFs). Our main technical insight is a new \emph{pinning decomposition} of the discrete score. It shows that unlike in continuous diffusions, the score decomposes into components where the dependence on time separates multiplicatively from the dependence on the target. Building on this decomposition, we propose a \emp
    
[^4]: Wasserstein梯度流与前向扩散不足以实现多模态采样

    Wasserstein Gradient Flows and Forward-Only Diffusion Are Not Enough for Multimodal Sampling

    [https://arxiv.org/abs/2610.02081](https://arxiv.org/abs/2610.02081)

    本文证明Wasserstein梯度流与前向扩散过程具有相同的密度演化，因而继承了非平衡统计物理中已知的亚稳态和慢混合现象，其指数收敛保证并不能说明它们能高效采样多模态分布。

    

    基于Wasserstein梯度流（WGF）和前向扩散过程（FODP）的采样算法近年来大量涌现，并且通常伴随着指数级快速收敛到目标分布的理论保证。这些保证经常被解读为这类方法能够高效采样复杂多模态分布的证据，且往往有实证结果支持。在这项工作中，我们认为这种解读从根本上具有误导性。通过运用Jordan-Kinderlehrer-Otto（JKO）格式和Otto演算，我们证明了经典的WGF采样动力学与过阻尼前向扩散具有相同的密度演化过程，因此继承了非平衡统计物理学中早已被充分认识的亚稳态性和慢混合现象。我们使用两种互补的分析工具——谱分析和平均首达时间（MFPT）分析——来研究这一类采样器……（原文摘要在此处截断）

    arXiv:2610.02081v1 Announce Type: new  Abstract: There has been a proliferation of sampling algorithms based on Wasserstein gradient flows (WGF) and forward-only diffusion processes (FODP), often accompanied by theoretical guarantees of exponentially fast convergence to the target distribution. These guarantees are frequently interpreted as evidence that such methods can efficiently sample complex multimodal distributions, often supported by empirical results. In this work, we argue that this interpretation is fundamentally misleading. By invoking the Jordan-Kinderlehrer-Otto (JKO) scheme and Otto calculus, we establish that the canonical WGF sampling dynamics and overdamped forward diffusion share the same density evolution and therefore inherit the same metastability and slow-mixing phenomena long understood in nonequilibrium statistical physics. We analyze this family of samplers using two complementary tools -- spectral analysis and mean first-passage time (MFPT) analysis -- and sh
    
[^5]: 上下文线性优化中遗憾的曲率

    The Curvature of Regret in Contextual Linear Optimization

    [https://arxiv.org/abs/2610.01980](https://arxiv.org/abs/2610.01980)

    本文证明了上下文线性优化中的遗憾在数据分布平均后呈局部二次曲率，给出了其闭式表达（支撑在法扇墙面上的矩阵值测度）以及仅需一次投影即可计算的弱收敛近似，并将其应用于决策感知的场景生成，实现30.8%的遗憾降低。

    

    面向决策的线性优化学习因优化器的不连续性而变得复杂：微小的成本误差可能使决策保持不变，也可能使其移动到另一个顶点。我们证明，这种逐点非光滑的行为在对数据分布取平均后变为局部二次型，并以闭式形式推导出该曲率——具体而言，是一个支撑在法扇（normal fan）墙面上的矩阵值测度。该测度仅依赖于可行集，数据分布仅以权重形式出现。随后，我们为该曲率提供了一种易于计算的近似，只需一次向可行集的投影即可求得。我们证明该近似弱收敛于真实的总体曲率。我们给出了这一发现的一个应用：面向期望成本线性优化的决策感知场景生成方法。我们的实验验证了二次规律与弱收敛规律，并展示了30.8%的遗憾降低（原文在此处截断）。

    arXiv:2610.01980v1 Announce Type: cross  Abstract: Decision-focused learning for linear optimization is complicated by the discontinuity of the optimizer, where small cost errors may leave the decision unchanged or move it to a different vertex. We show that this non-smooth pointwise behavior becomes locally quadratic after averaging over the data distribution, and we derive the curvature in closed form, specifically, a matrix-valued measure supported on the walls of the normal fan. This measure depends only on the feasible set, with the data distribution entering only as a weight. We then offer a tractable approximation for this curvature, computable with just one projection to the feasible set. We prove that the approximation weakly converges to the true population curvature. We offer one application of our findings, a decision-aware scenario generation method for expected-cost linear optimization. Our experiments test the quadratic and weak convergence laws and show a 30.8% regret i
    
[^6]: 伯努利老虎机中β-EB-TCI惩罚挑战者的精确非渐近分析

    Sharp Non-Asymptotic Analysis of the Penalized Challenger in $\beta$-EB-TCI for Bernoulli Bandits

    [https://arxiv.org/abs/2610.01951](https://arxiv.org/abs/2610.01951)

    本文首次对β-EB-TCI算法的惩罚挑战者给出精确非渐近分析，证明经验领导者成为真实最优臂后停止时间达到T_β*(μ)log(1/δ)（直至低阶项），从而为所有唯一最优臂的伯努利实例建立了非渐近高概率界。

    

    Top-two算法是固定置信度最优臂识别中简单而有效的方法，但其精确的非渐近行为仍不被充分理解。我们通过β-EB-TCI——Jourdan等人提出的经验最优top-two规则——来研究伯努利老虎机的这一问题，该规则使用带对数计数惩罚的伯努利传输代价来选择挑战者。我们证明，在经验领导者成为真实最优臂且其采样比例保持在接近β之后，停止时间在低阶集中项范围内为T_β*(μ)log(1/δ)。我们还证明，在此区域内，每个挑战者都会被线性次数地采样。因此，对于不含强制探索的原始算法，剩下的主要难点是控制经验领导者何时永久性地变为正确。这些结果为所有具有唯一最优臂的伯努利实例给出了非渐近高概率界。

    arXiv:2610.01951v1 Announce Type: cross  Abstract: Top-two algorithms are simple and effective for fixed-confidence best-arm identification, but their sharp non-asymptotic behavior is still not well understood. We study this problem for Bernoulli bandits through $\beta$-EB-TCI, the empirical-best top-two rule of Jourdan et al., whose challenger is chosen using a Bernoulli transportation cost with a logarithmic count penalty. We prove that, after the empirical leader has become the true best arm and its sampling fraction stays close to $\beta$, the stopping time is $T_{\beta}^{\star}(\mu)\log(1/\delta)$ up to lower-order concentration terms. We also show that, in this regime, every challenger is sampled linearly often. Thus, for the original algorithm without forced exploration, the main remaining difficulty is to control when the empirical leader becomes permanently correct. These results imply a non-asymptotic high-probability bound for all Bernoulli instances with a unique best arm. 
    
[^7]: 实用的双机器学习方法与AI学习表示

    Pragmatic DML with AI-Learned Representations

    [https://arxiv.org/abs/2610.01935](https://arxiv.org/abs/2610.01935)

    该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。

    

    文本、图像及其他丰富的协变量正日益被压缩为AI学习得到的表示，并被用作因果分析中的控制变量。我们研究了这一方法在何种情况下是有效的，并针对基于学习表示的因果推断开发了一个实用框架。对于广泛一类估计对象，不完美的表示会通过两个表示误差的乘积来扭曲目标因果参数：一个来自结果回归，另一个来自平衡权重（或Riesz表示元）。这带来了三项建设性成果。第一，交叉拟合的双机器学习（DML）为依赖表示的目标提供了有效的Wald推断。当表示误差较小时，同一置信区间能够覆盖因果参数，甚至可以达到半参数有效界。第二，按折进行的表示学习（或微调）与针对因果参数的DML推断是兼容的。为此，我们开发了凸-（原文在此处截断）

    arXiv:2610.01935v1 Announce Type: new  Abstract: Text, images, and other rich covariates are increasingly compressed into AI-learned representations and then used as controls in causal analysis. We study when this approach is valid and develop a practical framework for causal inference with learned representations. For a broad class of estimands, an imperfect representation distorts the target causal parameter by the product of two representation errors: one in the outcome regression and one in the balancing weight (or Riesz representer). This yields three constructive results. First, cross-fitted double machine learning (DML) provides valid Wald inference for the representation-dependent target. When representation errors are small, the same interval covers the causal parameter, and it can even attain the semiparametric efficiency bound. Second, fold-wise representation learning (or fine-tuning) is compatible with DML inference for the causal parameter. To this end, we develop convex-
    
[^8]: 面向不完美扩散模型的纠错推理时扩展方法

    Error-Corrected Inference-Time Scaling for Imperfect Diffusion Models

    [https://arxiv.org/abs/2610.01933](https://arxiv.org/abs/2610.01933)

    提出基于能量的Feynman-Kac校正器框架，能在推理时实时纠正不完美扩散模型的路径追踪误差和终点失配问题。

    

    推理时扩展无需额外训练即可将预训练扩散模型适配到新的采样任务。现有方法主要依赖增加粒子数的蒙特卡洛采样，但其前提是预训练模型完全精确。在实践中，数据和训练的局限性使模型并不完美，而这些方法会继承模型的误差。更多的粒子可以减少蒙特卡洛误差，但无法消除采样终点与期望目标之间的失配，也无法消除追踪预定概率路径时产生的误差。我们提出了基于能量的Feynman-Kac校正器，这是一个针对基于能量的扩散模型的框架，能够在给定参考能量的情况下，实时纠正这些误差。我们首先推导出即使在模型不完美的情况下，也能在连续时间总体极限下精确追踪预定路径的Feynman-Kac动力学，并使用带方差控制引导的序贯蒙特卡洛方法来近似这些动力学。

    arXiv:2610.01933v1 Announce Type: new  Abstract: Inference-time scaling adapts pretrained diffusion models to new sampling tasks without additional training. Existing methods rely primarily on Monte Carlo sampling with more particles, yet are premised on the pretrained model being exact. In practice, data and training limitations make the model imperfect, and these methods inherit its error. More particles reduce Monte Carlo error but cannot remove the mismatch between the endpoint and the desired target or the error in tracking the prescribed probability path. We introduce the Energy-based Feynman-Kac Corrector (EBFKC), a framework for energy-based diffusion models that corrects these errors on the fly given a reference energy. We first derive Feynman-Kac dynamics that track a prescribed path exactly in the continuous-time population limit even when the model is imperfect, and approximate these dynamics using sequential Monte Carlo with variance-controlling guidance. To remove the end
    
[^9]: 广义Engression模型

    Generalized Engression Models

    [https://arxiv.org/abs/2610.01823](https://arxiv.org/abs/2610.01823)

    本文提出广义engression模型，一个统一的非参数分布回归框架，可处理任意类型（连续、二元、类别、有序、排序）且相互条件依赖的多变量结果，通过数据类型特定的连接函数和随机扰动实现基于梯度的训练。

    

    我们考虑在给定协变量的情况下估计多变量结果的条件分布，其中结果的各个坐标可以是连续型、二元型、类别型、有序型或排序型，并且它们之间存在条件依赖关系。针对每种结果类型，学界已经发展出不同的统计方法，但大多数方法针对的是条件分布的某种摘要统计量，例如每个坐标的均值，而非结果向量的联合分布。我们提出了广义engression模型，这是一个适用于任何类型结果的统一非参数分布回归框架。所提出的方法建立在engression（一种基于评分规则的深度生成模型）之上，并引入了针对数据类型的特定连接函数以及一种平滑损失的随机扰动，使得即使连接函数不连续也能进行基于梯度的训练。我们为连续型、离散型和混合型结果建立了通用表示性结果。在模拟实验中……

    arXiv:2610.01823v1 Announce Type: cross  Abstract: We consider estimating the conditional distribution of a multivariate outcome given covariates when its coordinates may be continuous, binary, categorical, ordinal or rankings, and are conditionally dependent on one another. Different statistical methods have been developed for each outcome type, and most of them target a summary of the conditional distribution, such as the mean of each coordinate, rather than the joint distribution of the outcome vector. We develop generalized engression models, a unified nonparametric distributional regression framework for outcomes of any type. The proposed method builds upon engression, a scoring-rule-based deep generative model, and introduces a data-type-specific link function and a stochastic perturbation that smooths the loss, enabling gradient-based training even with discontinuous links. We establish universal representation results for continuous, discrete and mixed outcomes. In simulations 
    
[^10]: 基于切换线性动力学系统的多时间尺度神经动力学推断

    Inferring Multi-Timescale Neural Dynamics with Switching Linear Dynamical Systems

    [https://arxiv.org/abs/2610.01786](https://arxiv.org/abs/2610.01786)

    提出了多时间尺度切换线性动力学系统（MTS-SLDS）框架，能够从高维神经群体记录中识别随行为状态变化的多时间尺度潜在神经动力学。

    

    神经活动通常表现出多种时间尺度，这些时间尺度会随行为状态和任务条件而变化。从神经记录中识别这些时间尺度，对于更好地理解神经计算与功能非常重要。然而，基于自相关拟合的传统方法难以扩展到高维群体记录，并且当神经动力学随行为变化时会变得不可靠。状态空间模型一直是通过潜在动力学系统建模高维神经群体活动的强大框架，但标准的建模形式和推断方法并未显式考虑多个时间尺度，因此无法保证对底层时间结构的准确恢复。基于这些问题，我们提出了多时间尺度切换线性动力学系统（MTS-SLDS），这是一个用于从连续或尖峰神经数据中识别不同状态下特有潜在时间尺度的框架。

    arXiv:2610.01786v1 Announce Type: cross  Abstract: Neural activity often exhibits multiple timescales that can vary with behavioral states and task conditions. Identifying these timescales from neural recordings is important for better understanding neural computation and function. However, traditional approaches based on autocorrelation fitting are difficult to scale to high-dimensional population recordings and can become unreliable when neural dynamics change with behavior. State-space models have been a powerful framework for modeling high-dimensional neural population activity through latent dynamical systems, but standard formulations and inference methods do not explicitly account for multiple timescales and therefore do not guarantee accurate recovery of the underlying temporal structure. Motivated by these questions, we introduce the Multi-Timescale Switching Linear Dynamical System (MTS-SLDS), a framework for identifying regime-specific latent timescales from continuous or sp
    
[^11]: 单指标目标的上下文学习：核学习器与特征学习器的比较

    In-context Learning of Single-index Targets: Comparing Kernel and Feature Learners

    [https://arxiv.org/abs/2610.01712](https://arxiv.org/abs/2610.01712)

    本文通过复本方法比较了核学习器与特征学习器两种单层注意力架构在非线性单指标任务上的上下文学习能力，推导出记忆与泛化误差的理论预测和相图，揭示了两类架构各自占优的条件。

    

    上下文学习（ICL）使预训练模型能够从演示样本中推断任务，而无需更新其参数。虽然现有理论大多集中于线性目标函数，但本文通过在同一族单指标任务上比较两种单层注意力架构来研究非线性情形。核学习器首先将输入通过固定的非线性特征映射进行变换，然后应用线性注意力；而特征学习器则对原始输入直接应用注意力，随后接一个可学习的非线性读出层。我们利用复本方法推导了这两种架构的记忆误差和泛化误差的预测结果，并保留了预训练规模、任务池多样性以及训练和推理上下文长度的影响。所得预测在广泛的参数范围内与数值实验高度吻合。我们的分析给出了相图，刻画了随着……每种架构在何种条件下更具优势（注：原文摘要在此处不完整）。

    arXiv:2610.01712v1 Announce Type: cross  Abstract: In-context learning (ICL) enables a pretrained model to infer a task from demonstrations without updating its parameters. While much of the existing theory focuses on linear target functions, in this paper we study nonlinear cases by comparing two one-layer attention architectures on the same family of single-index tasks. A kernel learner first maps inputs through a fixed nonlinear feature map and then applies linear attention, whereas a feature learner applies attention to the original input, followed by a learned nonlinear readout. We derive predictions for their memorization and generalization errors using the replica method, retaining the effects of pretraining size, task-pool diversity, and training and inference context lengths. The resulting predictions closely match numerical experiments across a broad range of regimes. Our analysis yields phase diagrams that characterize when each architecture is advantageous as the amount of 
    
[^12]: 非凸-凹极小极大优化中带方差缩减的随机一阶算法的复杂度下界

    Lower Bounds for Stochastic First-Order Algorithms with Variance Reduction in Nonconvex--Concave Minimax Optimization

    [https://arxiv.org/abs/2610.01662](https://arxiv.org/abs/2610.01662)

    该论文首次为允许方差缩减技术的随机一阶算法在非凸-凹极小极大优化问题中建立了复杂度下界，突破了现有下界结果对算法类型的限制。

    

    我们建立了非凸-凹极小极大优化中随机一阶算法的复杂度下界，并允许算法使用方差缩减技术。我们的主要贡献是针对一类允许方差缩减的“零尊重”算法建立了下界，这超越了现有某些下界结果所施加的算法限制。我们考虑的目标函数具有 $L$-Lipschitz 连续的联合梯度，对偶域为欧氏半径至多 $D_Y$ 的紧凸集，原始值函数（通过对偶变量最大化目标函数定义）的初始次优性至多为 $\Delta$。目标精度 $\varepsilon$ 通过参数为 $1/(2L)$ 的约束原始值函数的 Moreau 包络的梯度范数来衡量。在方差至多为 $\sigma^2$ 且满足均方光滑性的无偏随机一阶预言机模型下，我们证明了复杂度下界为 $\Omega(L^2 D_Y \Delta / \varepsilon^2)$（摘要在此处截断）。

    arXiv:2610.01662v1 Announce Type: cross  Abstract: We establish complexity lower bounds for stochastic first-order algorithms in nonconvex--concave minimax optimization, allowing algorithms to use variance reduction. Our main contribution is a lower bound for a zero-respecting algorithm class that permits variance reduction, extending beyond the algorithmic restrictions imposed by some existing lower bounds. We consider objectives with an $L$-Lipschitz continuous joint gradient, a compact convex dual domain of Euclidean radius at most $D_Y$, and a primal value function, defined by maximizing the objective over the dual variable, with initial suboptimality at most $\Delta$. The target accuracy $\varepsilon$ is measured by the gradient norm of the Moreau envelope of the constrained primal value function with parameter $1/(2L)$. Under an unbiased stochastic first-order oracle with variance at most $\sigma^2$ and mean-square smoothness, we prove the lower bound $\Omega\!\left(L^2D_Y\Delta\
    
[^13]: 掩码重采样的隐藏优势：掩码自编码器的理论

    The hidden advantage of mask resampling: a theory of masked autoencoders

    [https://arxiv.org/abs/2610.01578](https://arxiv.org/abs/2610.01578)

    该论文首次从理论上证明掩码自编码器的掩码线性重建能在PCA失效的情形下以线性样本复杂度恢复潜在特征，并量化了掩码重采样中更大掩码多样性降低样本复杂度的统计优势。

    

    为什么掩码预测能够学习到未掩码重建所遗漏的有用表示？我们在一个高维掩码自编码器（MAE）模型中研究这一问题，该模型在具有共享潜在结构和异质噪声的数据上进行训练。我们证明，在未掩码线性重建（等价于PCA）会失败的情形下，掩码线性重建仍能以线性样本复杂度恢复潜在特征。该分析还量化了掩码重采样（掩码预训练中的一个既有要素）的统计优势。通过为每个样本引入固定的K个掩码集合，我们刻画了其对特征恢复和下游性能的影响，并识别出更大的掩码多样性能够降低样本复杂度的情形。在这一理论预测的指导下，我们发现标准图像训练流程中的随机裁剪和翻转可能会通过不断更新预测任务而掩盖掩码重采样的优势（原文在此处截断）。

    arXiv:2610.01578v1 Announce Type: new  Abstract: Why can masked prediction learn useful representations that unmasked reconstruction misses? We study this question in a high-dimensional model of a masked autoencoder (MAE) trained on data with shared latent structure and heterogeneous noise. We prove that masked linear reconstruction can recover the latent feature at linear sample complexity in regimes where unmasked linear reconstruction, equivalent to PCA, fails. The analysis also quantifies the statistical advantage of mask resampling, an established ingredient of masked pretraining. By introducing a fixed collection of $K$ masks per sample, we characterize its effect on feature recovery and downstream performance, identifying regimes where greater mask diversity lowers sample complexity. Guided by this prediction, we find that random cropping and flipping in standard image-training pipelines can obscure the advantage of mask resampling by renewing the prediction task even when the p
    
[^14]: 非马尔可夫决策过程中的精确可区分性

    Exact Distinguishability in Non-Markovian Decision Processes

    [https://arxiv.org/abs/2610.01527](https://arxiv.org/abs/2610.01527)

    本文首次精确刻画了非马尔可夫决策过程（RDP）中固定行为策略数据能否区分两个候选模型的条件，证明观测等价候选的后验几率在任何样本量下都恒等于先验几率，提出了线性时间判定算法PEC并在Lean 4中完成形式化验证，同时发现已有工作的可区分性假设在多数测试环境中并不成立。

    

    非马尔可夫环境通常被建模为正则决策过程（RDP），其动态通过有限自动机依赖于交互历史。现有的RDP离线保证依赖于对行为策略的可区分性假设，但没有提供验证该假设的方法。当该假设被违反时，不同的模型可能同样好地解释数据。我们研究在固定行为策略下收集的数据何时能够区分两个候选RDP。我们证明了在观测等价的候选之间，后验几率在每个样本量下始终等于先验几率，即使策略访问了自动机的每个状态也是如此，并且我们在Lean 4中对这两个结果进行了形式化验证。随后我们精确刻画了这种等价性，并由此推导出PEC算法，该算法可在与乘积自动机规模呈线性关系的时间内判定这种等价性。先前工作中所要求的可区分性假设在我们的四个测试环境中有三个都无法成立。

    arXiv:2610.01527v1 Announce Type: cross  Abstract: Non-Markovian environments are often modeled as Regular Decision Processes (RDPs), where dynamics depend on the interaction history through a finite automaton. Existing offline guarantees for RDPs rely on a distinguishability assumption on the behaviour policy but provide no means of verifying it. When the assumption is violated, distinct models may explain the data equally well. We study when data collected under a fixed behaviour policy can distinguish two candidate RDPs. We prove that the posterior odds between observationally equivalent candidates remain equal to the prior odds at every sample size, even when the policy visits every automaton state, and verify both results formally in Lean 4. We then characterize this equivalence exactly and derive PEC, an algorithm that decides it in time linear in the size of the product automaton. The distinguishability assumption of prior work fails on three of our four test environments, and t
    
[^15]: 朗之万信息驱动的迁移学习：用黑盒反馈替代目标样本

    Langevin-Informed Transfer Learning: Replacing Target Samples by Black-Box Feedback

    [https://arxiv.org/abs/2610.01522](https://arxiv.org/abs/2610.01522)

    提出了朗之万信息驱动的迁移学习（LITL）框架，仅依靠黑盒反馈即可从有偏源样本中恢复目标朗之万动力学，实现谱形式的动力学重构与慢流形梯度场估计。

    

    从分子动力学到扩散模型等许多科学和机器学习系统，都由具有低维结构的随机动力学所支配，并在缓慢的时间尺度上演化。然而，用于识别和解释此类动力学的目标轨迹往往是不可获取的：仅能获得探索底层流形的有偏样本或静态样本。我们提出了朗之万信息驱动的迁移学习（LITL），这是一个仅使用黑盒反馈、从有偏源样本中恢复目标朗之万动力学的框架。LITL 通过狄利克雷表示学习来学习目标无穷小生成元的主导谱结构和投影漂移，从而实现谱形式的动力学重构以及慢流形梯度场估计。我们进一步引入了一种球面变体，非常适合将学习系统中常用的归一化潜在表示引导至期望的目标。

    arXiv:2610.01522v1 Announce Type: cross  Abstract: Many scientific and machine learning systems, from molecular dynamics to diffusion models and beyond, are governed by stochastic dynamics with low-dimensional structure, evolving on slow timescales. However, target trajectories, used to identify and interpret such dynamics, are often inaccessible: only biased or static samples that explore the underlying manifold are available. We introduce Langevin-Informed Transfer Learning (LITL), a framework for recovering target Langevin dynamics from biased source samples using only black-box feedback. LITL learns the leading spectral structure of the target infinitesimal generator and the projected drift through Dirichlet representation learning, enabling kinetic reconstruction in spectral form and slow-manifold gradient field estimation. We further introduce a spherical variant well suited to steering normalized latent representations commonly used in learning systems toward desired objectives.
    
[^16]: 无需模型：文本熵率过滤缓解迭代微调崩溃

    No Model Required: Text Entropy Rate Filtering Mitigates Iterative Fine-Tuning Collapse

    [https://arxiv.org/abs/2610.01493](https://arxiv.org/abs/2610.01493)

    该论文提出了一种完全无需模型的非参数化Kontoyiannis熵率估计器，仅通过原始文本的匹配长度统计来过滤训练数据，能够有效缓解合成数据迭代微调导致的模型崩溃，其效果显著优于依赖模型对数概率的传统过滤方法。

    

    在合成数据上进行迭代微调会导致“模型崩溃”：随着稀有模式逐渐丢失，输出多样性不断收窄，其最显著的特征表现为短语级别的重复。现有的缓解方法要么需要模型的对数概率、外部预言机，要么需要持续获取真实的人类数据。本文提出了一种基于数学信息论的新方法：非参数化的Kontoyiannis熵率估计器 $h_k$，它完全通过匹配长度统计从原始文本中计算得出，不需要任何模型。我们证明，在一个完全合成、单谱系的微调设置中，这实际上是一种在文本多样性指标上表现更优的训练数据过滤器。在针对Llama-3.1-8B的六代QLoRA崩溃实验中，基于对数概率的过滤（最成熟的、需要模型访问权限的基线方法）在所有指标上均未带来显著的文本多样性提升（$p > 0.23$），而 $h_k$ 过滤则带来了 +42% 的唯一性提升。

    arXiv:2610.01493v1 Announce Type: cross  Abstract: Iterative fine-tuning on synthetic data causes \emph{model collapse}: output diversity narrows as rare patterns are progressively lost, a signature most visible as phrase-level repetition. Existing mitigations either require model log-probabilities, an external oracle, or continued access to real human data. Here we develop a new approach grounded in mathematical information theory: the non-parametric Kontoyiannis entropy rate estimator $h_k$, computed entirely from raw text via match-length statistics, with no model of any kind. We show that this is in fact a \emph{superior} training-data filter on text-diversity metrics in a fully-synthetic, single-lineage fine-tuning setting. In a six-generation QLoRA collapse experiment on Llama-3.1-8B, logprob-based filtering (the most established model-access-requiring baseline) provides no significant text-diversity benefit on any metric ($p > 0.23$), whereas $h_k$-filtering yields $+42\%$ uniqu
    
[^17]: 零通量：基于流的高维离散分布比较

    Zero Flux: Flow-Based Comparison of High-Dimensional Discrete Distributions

    [https://arxiv.org/abs/2610.01472](https://arxiv.org/abs/2610.01472)

    提出“零通量”差异准则，将基于流匹配的分布比较方法从连续分布扩展到高维离散分布，证明在独立耦合下当且仅当两个分布相同时所有局部概率通量在中点处消失，且该度量可分解为局部贡献并从样本中高效估计。

    

    由于状态空间呈指数级增长以及相互作用之间的复杂变化，比较两个高维离散分布一直是一项具有挑战性的任务。最近的一项工作提出通过流匹配训练两个连续分布之间的向量场来比较分布。当且仅当两个分布相同时，所得到的向量场在中点处消失（为零）。然而，这种基于流的判据并不能自然地应用于离散分布。我们将这一原理扩展到离散领域，提出了“零通量”（Zero Flux）准则，这是一种基于局部概率通量的差异度量。在独立耦合下，我们证明当且仅当两个分布相同时，所有局部概率通量在中点处消失。该差异度量将联合分布的差异分解为更小的局部贡献，并且可以从样本中高效估计。我们建立了有限样本误差界……（原文摘要在此处截断）

    arXiv:2610.01472v1 Announce Type: new  Abstract: Comparing two high-dimensional discrete distributions has always been a challenging task due to the exponentially growing state space and complex changes in interactions. A recent work suggests comparing distributions through a vector field trained using flow matching between two continuous distributions. The resulting vector field at mid-point vanishes if and only if two distributions identical. However, such a flow-based criterion does not naturally apply to discrete distributions. We extend this principle to the discrete domain and introduce the \emph{Zero Flux} criterion, a discrepancy based on local probability fluxes. Under independent coupling, we show that all local probability fluxes vanish at the midpoint if and only if two distributions are the same. This discrepancy decomposes the joint distributional difference into smaller, local contributions and can be efficiently estimated from samples. We establish finite sample error b
    
[^18]: 线性可分逻辑回归在稳定性边缘处的紧致过渡时间界

    Tight Transition Time Bounds for Separable Logistic Regression at the Edge of Stability

    [https://arxiv.org/abs/2610.01459](https://arxiv.org/abs/2610.01459)

    本文推翻了此前关于稳定性边缘现象中过渡时间与步长无关的猜想，证明了最坏情况下过渡时间随步长按 $\Theta((\log\eta)^{\min\{n-2,d-2\}})$ 增长，且该界在任意维度下均紧致。

    

    我们研究了在线性可分数据上采用大常数步长 $\eta$ 进行梯度下降的逻辑回归问题。此类动力学可能表现出一种典型的“稳定性边缘”现象，即损失最初发生振荡，随后过渡到单调下降的稳定阶段。现有工作在维度 $d=2$、$\eta \to \infty$ 的情形下给出了紧致的 $\Theta(1)$ 界，并猜测在任意维度 $d\geq 2$ 下存在与 $\eta$ 无关的界。本文推翻了这一猜测，证明对于每个固定的样本量 $n\geq 2$ 和足够小的间隔 $\gamma$，最坏情况下的过渡时间为 $\Theta((\log\eta)^{\min\{n-2,d-2\}})$，且该结果对所有 $d\geq 2$ 一致成立。建立紧致界的关键挑战在于，对梯度贡献最大的样本可能在迭代过程中反复变化。为解决这一问题，我们通过对维度和样本量的归纳来控制这种变化……

    arXiv:2610.01459v1 Announce Type: cross  Abstract: We study logistic regression on linearly separable data under gradient descent with a large constant stepsize $\eta$. Such dynamics may exhibit a characteristic Edge of Stability phenomenon, in which the loss initially oscillates before transitioning to a stable phase of monotone decrease. Existing work provides a tight $\Theta(1)$ bound in dimension $d=2$ as $\eta \to \infty$ and conjectures a bound independent of $\eta$ in arbitrary dimensions $d\geq 2$. In this paper, we disprove this conjecture by showing that, for every fixed sample size $n\geq 2$ and sufficiently small margin $\gamma$, the worst-case transition time is $$\Theta\!\left((\log\eta)^{\min\{n-2,d-2\}}\right)$$ uniformly over $d\geq2$. The key challenge in establishing a tight bound is that the sample contributing most strongly to the gradient can change repeatedly across iterations. To address this issue, we control such changes by induction on dimension and sample si
    
[^19]: 关于斜对称分布及其在蒙特卡洛采样算法中的应用：Barker 提议的无坐标、Gibbs 风格与流形版本

    On skew-symmetric distributions and their use in Monte Carlo sampling algorithms: coordinate-free, Gibbs-style and manifold versions of the Barker proposal

    [https://arxiv.org/abs/2610.01448](https://arxiv.org/abs/2610.01448)

    本文回顾了基于斜对称分布的 Barker 提议并提出三种扩展（无坐标变体、Gibbs 风格变体和简化流形版本），其中 Gibbs 风格变体在每次部分坐标更新时重新评估梯度，从而在相关目标分布上提升了采样效率。

    

    斜对称概率分布为将梯度信息融入马尔可夫链蒙特卡洛算法提供了一种有原则的机制。本文回顾了（预条件化的）Barker 提议——一种基于斜对称分布构建的 Metropolis–Hastings 算法，并阐述了其设计动机。随后我们引入三种自然的扩展：首先，我们提出了 Barker 算法的无坐标变体；其次，我们引入了一种 Gibbs 风格的 Barker 算法，该算法在每次部分更新坐标时重新评估梯度；第三，我们推导出一种简化的流形 Barker 算法，所得到的流形采样器相比自然对比方法具有更强的鲁棒性。数值实验表明，Gibbs 风格变体在相关目标分布上提升了原始采样效率，而在考虑计算成本后，无坐标变体相比标准 Barker 提议的实际优势有限……

    arXiv:2610.01448v1 Announce Type: cross  Abstract: Skew-symmetric probability distributions provide a principled mechanism for incorporating gradient information into Markov chain Monte Carlo algorithms. Here we review the (preconditioned) Barker proposal, a Metropolis--Hastings algorithm built on skew-symmetric distributions, and motivate its design. We then introduce three natural extensions. First, we propose coordinate-free variants of the Barker algorithm. Second, we introduce a Gibbs-style Barker algorithm that re-evaluates the gradient at each partially updated coordinate. Third, we derive a simplified manifold Barker algorithm, producing a manifold sampler with enhanced robustness compared to natural comparators. Numerical experiments demonstrate that the Gibbs-style variant improves raw sampling efficiency on correlated targets, that the coordinate-free variants offer limited practical advantage over the standard Barker proposal once computational costs are accounted for, and 
    
[^20]: 最优传输遇见强化学习：综述

    Optimal Transport Meets Reinforcement Learning: A Survey

    [https://arxiv.org/abs/2610.01413](https://arxiv.org/abs/2610.01413)

    本综述系统梳理了最优传输（OT）在强化学习中的应用，指出当待比较的概率分布重叠较弱时，OT比传统散度度量更有效，并从OT的角色、被比较的分布、OT形式化方法及时间结构处理四个维度对现有方法进行了分类。

    

    强化学习（RL）算法经常需要比较概率分布，例如由策略和专家产生的状态访问分布、学习到的策略与离线数据集的动作分布，或者学习到的模型与环境之间的转移分布。然而，当这些分布重叠较弱时，常用的散度度量可能会失效，这种情况在模仿学习、离线强化学习以及分布偏移下的部署中经常出现。最优传输通过在编码任务几何结构的基础代价下，度量将概率质量从一个分布“搬运”到另一个分布的代价，提供了一种替代方案。本综述涵盖了OT在强化学习目标和算法中的使用方式。对于每种方法，我们识别出：OT所扮演的角色、被比较的分布、所使用的OT形式化方法，以及对时间结构的处理方式。除了对现有方法进行分类之外，我们还讨论了……（摘要原文在此处截断）

    arXiv:2610.01413v1 Announce Type: cross  Abstract: Reinforcement learning (RL) algorithms frequently compare probability distributions, such as state visitation distributions induced by policies and experts, action distributions from learned policies and offline datasets, or transition distributions from learned models and environments. However, commonly used divergences may become ineffective when these distributions overlap weakly, which is frequently encountered in imitation learning, offline RL, and deployment under distribution shift. Optimal transport (OT) offers an alternative by measuring the cost of \emph{moving} probability mass from one distribution to another under a ground cost that encodes task geometry. This survey covers how OT is used inside RL objectives and algorithms. For each method, we identify: the role OT plays, the distributions compared, the OT formulation used, and the treatment of temporal structure. Beyond categorising existing methods, we discuss the motiv
    
[^21]: 克利福德层神经网络

    Clifford Sheaf Neural Networks

    [https://arxiv.org/abs/2610.01322](https://arxiv.org/abs/2610.01322)

    本文提出克利福德层神经网络（CSNN），通过用K项夹逼替代代数同态作为限制映射，构造出天然半正定、无需旋量约束且能混合不同阶的层拉普拉斯算子，并从阶耦合、自同态空间覆盖范围和条件数三个维度系统刻画了该限制映射族。

    

    我们提出了克利福德层神经网络（CSNN），这是一种面向几何图的等变层神经网络，它在胞腔层的每个茎上放置克利福德代数，并沿边传输多重向量特征。对于茎取值为代数的层，限制映射的经典选择是代数同态。在加入等变性约束后，这一朴素选择变为旋量共轭。然而，旋量共轭的表达能力较弱，因此我们放弃代数同态约束，得到了K项夹逼。由此得到的层拉普拉斯算子在构造上即为半正定，无需旋量约束，且仍能混合不同阶。我们的主要贡献从三个维度刻画了所得的限制映射族：映射耦合了哪些阶、它能覆盖自同态空间的多大比例，以及它的条件数优劣。K项夹逼张成了自同态空间的一半，并且在Cl(3, 0, 0)中它对应于

    arXiv:2610.01322v1 Announce Type: cross  Abstract: We introduce the Clifford Sheaf Neural Network (CSNN), an equivariant sheaf neural network for geometric graphs that places a Clifford algebra on each stalk of a cellular sheaf and transports multivector features along edges. The canonical choice of restriction map for sheaves with algebra-valued stalks is algebra homomorphism. Adding the constraint of equivariance, the naive choice becomes versor conjugation. However, versor conjugation is expressively weak, so we drop algebra homomorphism and arrive at the K-term sandwich. The resulting sheaf Laplacian is positive semidefinite by construction, needs no versor constraint, and still mixes grades. Our main contribution characterizes the resulting family of restriction maps along three axes: which grades a map couples, how much of the endomorphism space it reaches, and how well it is conditioned. The K-term sandwich spans half of the endomorphism space, and in Cl(3, 0, 0) it corresponds 
    
[^22]: IQS-BO：面向贝叶斯优化的上下文内查询选择

    IQS-BO: In-Context Query Selection for Bayesian Optimisation

    [https://arxiv.org/abs/2610.01269](https://arxiv.org/abs/2610.01269)

    本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。

    

    贝叶斯优化（BO）是优化昂贵黑盒函数的强大框架，但通常需要在每次评估步骤中重新拟合代理模型并最大化采集函数。基于先验数据拟合网络（PFNs）的上下文内方法通过在从合成先验中抽取的函数上预训练Transformer来分摊部分成本。PFNs4BO分摊了代理模型的计算成本，但仍依赖于数值最大化的采集函数；而FIBO通过从学习到的密度中采样优化器位置，完全在上下文内执行BO，这固定了决策规则且不使用代理模型。学习型采集函数通过训练好的网络对有限候选集进行评分，但由于缺乏查询的标签，它们只能在先前解决的任务上通过强化学习来学习评分。我们提出了IQS-BO，一个通过在合成先验上进行监督学习来学习查询决策的PFN。在单次前向传递中，IQS-BO……

    arXiv:2610.01269v1 Announce Type: cross  Abstract: Bayesian Optimisation (BO) is a powerful framework for the optimisation of expensive black-box functions, but typically requires refitting a surrogate and maximising an acquisition function at every evaluation step. In-context approaches based on Prior-data Fitted Networks (PFNs) amortise part of this cost by pre-training transformers on functions drawn from synthetic priors. PFNs4BO amortises the surrogate but still relies on a numerically maximised acquisition function, while FIBO performs BO fully in-context by sampling optimiser locations from a learned density, which fixes the decision rule and admits no surrogate. Learned acquisition functions score a finite candidate set with a trained network, but, lacking a label for the query, learn the score by reinforcement learning on previously solved tasks. We propose IQS-BO, a PFN that learns the query decision by supervised learning on synthetic priors. In a single forward pass, IQS-BO
    
[^23]: 基于流匹配的反事实生成：耦合敏感的端到端误差率

    Counterfactual Generation via Flow Matching: Coupling-Sensitive End-to-End Rates

    [https://arxiv.org/abs/2610.01193](https://arxiv.org/abs/2610.01193)

    本文提出一种基于流匹配的反事实生成方法，将双重稳健训练目标与学习到的源-结果耦合相结合，并证明其常数步欧拉离散化的KL误差界由耦合下的位移矩控制、与维度近线性相关，而非依赖速度场的全局正则性。

    

    反事实生成旨在利用事实分配机制下收集的观测数据，对假设性干预或决策下的结果进行采样。我们提出了一种流匹配方法，该方法将样本拆分、双重稳健的训练目标与在观测源结果和从拟合的条件结果模型中抽取的目标结果之间学习到的耦合相结合。为了实现有限步生成，我们利用了基于高斯平滑插值的分数校正随机采样器。我们的主要理论贡献是针对常数步欧拉离散化的耦合敏感KL散度界：该误差由所选耦合下源-目标位移的矩来控制，而非由速度场的全局一致正则性控制，且与环境维度呈近线性依赖关系。我们还为学习到的速度场和分数场建立了有限样本非参数保证……

    arXiv:2610.01193v1 Announce Type: cross  Abstract: Counterfactual generation seeks to sample outcomes under a hypothetical intervention or decision using observational data collected under the factual assignment mechanism. We develop a flow-matching approach that combines a sample-split, doubly robust training objective with a learned coupling between observed source outcomes and target outcomes drawn from a fitted conditional outcome model. To enable finite-step generation, we leverage a score-corrected stochastic sampler based on a Gaussian-smoothed interpolation. Our main theoretical contribution is a coupling-sensitive KL bound for constant-step Euler discretization: the error is controlled by moments of the source--target displacement under the chosen coupling, rather than by global uniform regularity of the velocity field, and has near-linear dependence on the ambient dimension. We also establish finite-sample non-parametric guarantees for the learned velocity and score fields wh
    
[^24]: 梯度引导的密度峰值聚类

    Gradient-Guided Density Peak Clustering

    [https://arxiv.org/abs/2610.01050](https://arxiv.org/abs/2610.01050)

    本文提出梯度引导的密度峰值聚类（GGDPC），通过在每次最近邻上坡搜索前执行梯度上升步骤来稳定低密度区域的不规则路径，并在局部模态恢复、调整兰德指数等五个互补准则下建立了该方法的统计一致性理论。

    

    密度峰值聚类（DPC）将每个观测点连接到其密度更高的最近邻，并将聚类中心识别为具有异常大的最近邻上坡偏移的高密度观测点。然而，从观测点到聚类中心所产生的上坡路径在低密度区域可能是不规则且不稳定的，这使得聚类分配对局部扰动敏感，并模糊了DPC图的总体几何结构。在本文中，我们提出了梯度引导的密度峰值聚类（GGDPC），它在每次最近邻上坡搜索之前执行一个梯度上升步骤。我们发展了一种稳定性理论，将GGDPC图与总体密度的梯度上升流联系起来。特别地，我们在五个互补的准则下建立了GGDPC的一致性：局部模态的恢复、调整兰德指数、树状图（聚类树）、路径长度和瀑布度量。

    arXiv:2610.01050v1 Announce Type: cross  Abstract: Density peak clustering (DPC) connects each observation to its nearest neighbor of higher density and identifies cluster centers as high-density observations with unusually large nearest neighbor uphill shifts. The resulting uphill paths from observations to cluster centers, however, can be irregular and unstable in low-density regions, making the clustering assignments sensitive to local perturbations and obscuring the population geometry of the DPC graph. In this paper, we introduce \emph{gradient-guided density peak clustering} (GGDPC), which performs a gradient ascent step before each nearest neighbor uphill search. We develop a stability theory that relates the GGDPC graph to the gradient ascent flow of the population density. In particular, we establish consistency of GGDPC under five complementary criteria: recovery of local modes, adjusted Rand index, dendrogram (cluster tree), path length, and waterfall measure. Together, thes
    
[^25]: 通过基于先验的少步传输映射在源空间进行MCMC后验采样

    Posterior sampling by source-space MCMC via prior-based few-step transport maps

    [https://arxiv.org/abs/2610.01034](https://arxiv.org/abs/2610.01034)

    该论文提出了一种源空间广义贝叶斯推断框架，利用一步或少步的改进MeanFlow（iMF）映射表示隐式先验，在高斯源空间中通过MCMC进行后验采样，并给出了精确后验与学习后验之间基于训练次优性和模型类近似误差的Wasserstein误差理论保证。

    

    贝叶斯推断越来越多地使用仅由样本表示的信息丰富但隐式的先验，例如历史集合、模拟器输出和预训练生成模型。同样的计算问题也出现在测试时引导任务（广义贝叶斯）中，其中显式的正权重（例如指数化的奖励）会对隐式先验进行倾斜。我们开发了一个源空间广义贝叶斯推断框架，该框架将低成本的少步先验传输与后验稳定性保证相结合。具体而言，我们使用一步或少步的改进MeanFlow（iMF）映射来表示先验，并在其高斯源空间中执行后验采样。我们从联合总体iMF损失和辅助速度损失的角度建立了精确后验与学习后验之间的Wasserstein误差界，该损失被分解为训练次优性和模型类近似误差两部分。在iMF源空间中，我们采用并行回火……（摘要在此处被截断）

    arXiv:2610.01034v1 Announce Type: cross  Abstract: Bayesian inference increasingly uses informative but implicit priors represented only by samples, such as historical ensembles, simulator outputs, and pretrained generative models. The same computational problem appears in the test-time guidance task (generalized Bayes), where an explicit positive weight, e.g., an exponentiated reward, tilts an implicit prior. We develop a framework for source-space generalized Bayesian inference that combines inexpensive few-step prior transports with posterior stability guarantees. Specifically, we represent the prior using a one- or few-step improved MeanFlow (iMF) map and perform posterior sampling in its Gaussian source space. We establish Wasserstein error bounds between the exact and learned posteriors in terms of the joint population iMF and auxiliary-velocity loss, decomposed into training suboptimality and model-class approximation error. In the iMF source space, we adopt parallel tempering w
    
[^26]: 基于容差的公平性审计：违规认证与敏感性筛查

    Tolerance-Based Fairness Auditing: Violation Certification and Sensitivity Screening

    [https://arxiv.org/abs/2610.01005](https://arxiv.org/abs/2610.01005)

    该论文提出了一个统一的基于容差的公平性审计框架，针对违规认证和敏感性筛查两个互补目标，分别通过约束经验似然检验（结合最不利点校准）控制虚假违规声明风险，并降低漏检违规的概率。

    

    随着人工智能日益广泛部署，算法不公平性引发了越来越多的关注，也加剧了对透明公平性审计的需求。在实践中，可容忍的算法不公平程度取决于具体的法律、伦理或应用场景。在给定预先设定的容差阈值的情况下，一个重要的统计问题是如何在不同的审计目标下判断群体间差异是否超出允许的容差范围。为解决这一问题，我们针对两个互补的审计目标开发了统一的基于容差的公平性审计框架：其一是违规认证，优先控制虚假违规声明的概率；其二是敏感性筛查，优先减少漏检违规的情况。针对第一个目标，我们开发了一种约束经验似然检验方法，适用于正式的审计场景，该方法采用最不利点校准，并可与虚假标记率控制相结合。

    arXiv:2610.01005v1 Announce Type: new  Abstract: As artificial intelligence is increasingly deployed, algorithmic unfairness has raised growing concerns and intensified demands for transparent fairness auditing. In practice, the tolerable degree of algorithmic unfairness depends on the specific legal, ethical, or application context. Given a prespecified tolerance threshold, an important statistical question is how to determine whether a group disparity exceeds the allowable tolerance across different auditing objectives. To address this problem, we develop a unified tolerance-based fairness auditing framework for two complementary auditing objectives: violation certification, which prioritizes control of false violation declarations, and sensitivity screening, which prioritizes reducing missed violations. For the first objective, we develop a constrained empirical likelihood test for formal settings that uses least-favorable-point calibration and can be combined with false flagging ra
    
[^27]: 相关测试的代价：模型发布门槛应该设得多严格？

    The Price of Correlated Tests: How Strict Should a Model Release Gate Be?

    [https://arxiv.org/abs/2610.00993](https://arxiv.org/abs/2610.00993)

    该论文将模型发布门槛的设定视为一个统计设计问题，证明当测试之间存在相关性时，更严格的门槛总能提升可靠性，因此最优策略是选择仍能满足可靠性目标的最为宽松的门槛，而“全部通过”式门控会随着测试数量的增加把保留的好模型比例推向零。

    

    在机器学习模型发布之前，它通常需要通过一整套自动化测试。要求每项测试都通过看似安全，但这可能会拒绝许多本可以为用户提供良好服务的模型，而且它也无法说明一个通过测试的模型实际上有多可信。我们将发布门槛视为一个设计问题：选择模型必须通过多少项测试，使得通过审查的模型满足既定的可靠性目标，同时尽可能保留更多的好模型。一个两类的潜在因子模型使这两种代价变得明确，并将每次计算简化为一维积分。我们证明，当两个类别共享相同的潜在相关性时，更严格的门槛总能提高可靠性，因此能保留最多好模型的门槛是仍能满足目标的最为宽松的那个。在“全部通过”式门控下，任何低于完美的可靠性目标在模型内都是可以实现的，但随着测试套件规模的增长，被保留的好模型比例却趋于零。

    arXiv:2610.00993v1 Announce Type: new  Abstract: Before a machine learning model ships, it often has to pass a suite of automated tests. Requiring every test to pass looks safe, yet it can reject many models that would have served users well, and it does not say how trustworthy a passing model actually is. We treat the release gate as a design problem: choose how many tests a model must pass so that cleared models meet a stated reliability target, while keeping as many good models as possible. A two-class latent-factor model makes both costs explicit and reduces each calculation to a one-dimensional integral. We prove that when both classes share the same latent correlation, a stricter gate always raises reliability, so the gate that keeps the most good models is the most lenient one that still meets the target. Under pass-all gating, any reliability target short of perfection is attainable within the model, but the share of good models kept tends to zero as the suite grows. Correlatio
    
[^28]: 面向带无分类器引导的扩散模型激活量化的联合分支空间变换编码

    Joint Branch-Space Transform Coding for Diffusion Activation Quantization with Classifier-Free Guidance

    [https://arxiv.org/abs/2610.00930](https://arxiv.org/abs/2610.00930)

    该论文提出分支空间变换编码方法（含GCBT），利用离线推导的2x2正交矩阵联合旋转CFG的条件与无条件激活以挖掘其强相关结构，并结合引导方向和跨分支二阶矩，在固定比特预算下显著提升扩散模型激活量化的保真度。

    

    扩散模型的后训练量化日益利用时间步、特征和层结构。尽管最近的工作已开始将无分类器引导（CFG）结构纳入扩散量化，但激活量化仍在条件坐标和无条件坐标上独立进行，跨激活的结构未被利用。我们证明，匹配的CFG激活构成了一个强相关的二维信源，并且在固定比特预算下，分支编码基的选择对量化保真度有实质性影响。基于这一观察，我们引入分支空间变换编码，通过一个离线推导的2x2正交矩阵旋转匹配的CFG分支，仅需对模型参数或量化流程做极小的修改。我们进一步推导出引导相关分支变换（GCBT），它联合融合了CFG引导方向与跨分支二阶矩。在相等的比特预算下……（摘要原文在此处截断）

    arXiv:2610.00930v1 Announce Type: cross  Abstract: Post-training quantization for diffusion models increasingly exploits timestep, feature, and layer structure. While recent work has begun incorporating CFG structure into diffusion quantization, activation quantization still operates independently across conditional and unconditional coordinates, leaving cross-activation structure unexploited. We show that matched CFG activations form a strongly correlated two-dimensional source and that, under a fixed bit budget, the choice of branch coding basis materially affects quantization fidelity. Motivated by this observation, we introduce branch-space transform coding, which rotates matched CFG branches via an offline derived 2x2 orthogonal matrix, requiring minimal modifications to model parameters or the quantization pipeline. We further derive the Guidance-Correlation Branch Transform (GCBT), which jointly incorporates the CFG guidance direction and cross-branch second moments. Under an eq
    
[^29]: 柏拉图式任务算术

    Platonic Task Arithmetic

    [https://arxiv.org/abs/2610.00929](https://arxiv.org/abs/2610.00929)

    本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。

    

    针对同一任务进行专门化训练的模型会收敛到相似的行为，然而产生这种行为的参数更新却缺乏共同的坐标系，因此权重空间中的任务算术仍然局限于单一模型，在没有结构对应关系的情况下无法跨越不同架构。借鉴柏拉图的洞穴寓言，我们假设这些特定于模型的更新是某个共享的、与模型无关的对象的投影，我们将其称为“柏拉图任务向量”。为了使这一概念对将图像或音频编码器与文本编码器配对的模型具有可操作性，我们引入了通用任务描述符：一种形状独立于架构和嵌入维度的矩阵，它记录任务的功能效果，并支持将加法和取反作为矩阵运算。将描述符迁移到目标模型中意味着对目标模型进行编辑，直到它能在任务的无标签探测图像和类别名称提示上重现该描述符，无需逐图像标注。

    arXiv:2610.00929v1 Announce Type: cross  Abstract: Models specialized for the same task converge to similar behavior, yet the parameter updates that produce it share no common coordinate system, so weight-space task arithmetic stays confined to a single model and cannot cross architectures without a structural correspondence. Drawing on Plato's allegory of the cave, we hypothesize that these model-specific updates are shadows of one shared, model-agnostic object, which we call the platonic task vector. To make it operational for models that pair an image or audio encoder with a text encoder, we introduce Universal Task Descriptors: matrices whose shape is independent of architecture and embedding dimension, which record a task's functional effect and support addition and negation as matrix operations. Transferring a descriptor into a target means editing the target until it reproduces the descriptor on the task's unlabeled probe images and class-name prompts, requiring no per-image lab
    
[^30]: 潜在线性动态下非平稳赌博机的块级乐观算法

    Block Optimism for Nonstationary Bandits with Latent Linear Dynamics

    [https://arxiv.org/abs/2610.00911](https://arxiv.org/abs/2610.00911)

    该论文提出基于自适应块级乐观的UCB算法，借助循环近似截断无限记忆奖励过程，将具有潜线性动态的非平稳赌博机的遗憾值从 $\tilde{O}(T^{2/3})$ 提升到更优速率。

    

    我们研究了一类具有潜在线性动态的内生非平稳随机赌博机问题，其中动作既影响即时奖励，也影响未观测的潜在状态的未来演化。奖励是当前动作与潜在状态的双线性函数，这导致了依赖历史的奖励和一个非平凡的长时域规划问题。现有的“先探索后承诺”方法通过均匀探索来估计潜在动态，然后承诺执行一个优化的开环动作序列，实现了 $\tilde{O}(T^{2/3})$ 的遗憾值。我们证明通过自适应的块级乐观策略可以改进这一速率。我们的关键步骤是循环近似：在稳定动态下，无限记忆的奖励过程可以被截断，而开环基准可以通过优化有限记忆的块级代理来近似。基于这一归约，我们提出了一种基于UCB的块算法，该算法为截断后的动态参数维护置信集……

    arXiv:2610.00911v1 Announce Type: new  Abstract: We study an endogenous nonstationary stochastic bandit problem with latent linear dynamics, where actions affect both immediate rewards and the future evolution of an unobserved latent state. Rewards are bilinear in the current action and latent state, inducing history-dependent rewards and a nontrivial long-horizon planning problem. The existing explore-then-commit approach achieves $\tilde{O}(T^{2/3})$ regret by uniformly exploring to estimate the latent dynamics and then committing to an optimized open-loop action sequence. We show that this rate can be improved via adaptive block-level optimism. Our key step is a cyclic approximation: under stable dynamics, the infinite-memory reward process can be truncated, and the open-loop benchmark can be approximated by optimizing a finite-memory block-level proxy. Building on this reduction, we propose a UCB-based block algorithm that maintains confidence sets for the truncated dynamics parame
    
[^31]: 为目标条件强化学习学习多种时间尺度

    Learning Multiple Timescales for Goal-Conditioned Reinforcement Learning

    [https://arxiv.org/abs/2610.00849](https://arxiv.org/abs/2610.00849)

    提出GITA方法，将单一价值函数以时间抽象尺度k为条件，并通过在多个k值上聚合优势加权监督信号来训练一个策略，从而解决离线目标条件强化学习在长时程任务中价值信号消失的问题。

    

    现有的离线目标条件强化学习（GCRL）方法在长时程任务上表现不佳。折扣机制会缩小远距离状态之间的价值差异，直到其低于函数逼近误差，导致智能体失去用于对状态进行排序的信号。时间抽象方法将 k 个环境步视为一次状态转移，能够在长距离上恢复这一信号，但没有任何单一的固定 k 值适用于所有状态-目标距离：较大的 k 能保留长时程距离上的价值差异，却会抹平邻近状态之间的区分；较小的 k 则恰好相反。我们明确阐述了这一权衡，并提出了广义隐式时间抽象（Generalized Implicit Temporal Abstraction，GITA），它使单一的价值函数以 k 为条件。GITA 通过聚合多个 k 值上的优势加权监督信号来训练一个策略，因此对某个状态-目标对赋予更大正优势的时间尺度将在该状态-目标对的更新中贡献更大的权重。

    arXiv:2610.00849v1 Announce Type: new  Abstract: Existing approaches to offline goal-conditioned reinforcement learning (GCRL) struggle with long-horizon tasks. Discounting shrinks value differences between distant states until they fall below the function approximation error, leaving the agent with no signal for ranking states. Temporal abstraction, which treats k environment steps as a single transition, restores this signal at long range, but no single fixed k suits all state-goal distances: large k preserves value differences across long temporal distances while collapsing distinctions between nearby states, and small k does the reverse. We make this trade-off explicit and introduce Generalized Implicit Temporal Abstraction (GITA), which conditions a single value function on k. GITA trains one policy by aggregating advantage-weighted supervision across multiple k values, so scales assigning larger positive advantages to a state-goal pair contribute more strongly to its update. GITA
    
[^32]: 基于神经网络与Euler近似的加权亚分数布朗运动驱动的随机微分方程的统计推断

    Inference for stochastic differential equations driven by weighted sub-fractional Brownian motion using neural networks and the Euler approximation

    [https://arxiv.org/abs/2610.00793](https://arxiv.org/abs/2610.00793)

    本文提出将神经网络与Euler近似相结合，对由加权亚分数布朗运动这类高斯过程驱动的随机微分方程进行统计推断，从离散观测数据中估计漂移系数、扩散系数与噪声协方差。

    

    我们考虑从高斯过程驱动的随机微分方程的离散观测中对漂移系数、扩散系数和噪声协方差进行估计。对于固定的观测时间区间 T>0 和已知的初始状态 x₀∈ℝ，我们研究方程 dX_t = a(X_t)dt + σ(X_t)dZ_t^{β,f}，其中 X₀ = x₀，0 ≤ t ≤ T。这里 a:ℝ→ℝ 是漂移系数，σ:ℝ→(0,∞) 是扩散系数，而 Z^{β,f} 是属于加权亚分数布朗运动族的中心化高斯过程，其协方差为 Cov(Z_s^{β,f}, Z_t^{β,f}) = ∫₀^{s∧t} f(r)q_β(s−r, t−r)dr，其中 0 ≤ s, t ≤ T，s∧t = min{s,t}。时间权重 f:[0,T]→[0,∞) 是可测、有界且几乎处处为正的函数，β∈(0,…（原文摘要在此处截断）

    arXiv:2610.00793v1 Announce Type: new  Abstract: We consider the estimation of drift, diffusion, and noise covariance from discrete observations of stochastic differential equations driven by Gaussian processes. For a fixed observation horizon $T>0$ and a known initial state $x_0\in\mathbb R$, we study \begin{equation*}   dX_t=a(X_t)\,dt+\sigma(X_t)\,dZ_t^{\beta,f},   \qquad X_0=x_0,\quad 0\leq t\leq T. \end{equation*} \smallskip\noindent Here $a:\mathbb R\to\mathbb R$ is the drift coefficient, $\sigma:\mathbb R\to(0,\infty)$ is the diffusion coefficient, and $Z^{\beta,f}$ is a centered Gaussian process from the weighted sub-fractional Brownian family, with covariance \begin{equation*}   \operatorname{Cov}(Z_s^{\beta,f},Z_t^{\beta,f})   =\int_0^{s\wedge t} f(r)q_\beta(s-r,t-r)\,dr,   \qquad 0\leq s,t\leq T. \end{equation*} \smallskip\noindent Here $s\wedge t=\min\{s,t\}$. The temporal weight $f:[0,T]\to[0,\infty)$ is measurable, bounded, and positive almost everywhere, and $\beta\in(0,
    
[^33]: 可扩展的多任务逆强化学习

    Scalable Multi-Task Inverse Reinforcement Learning

    [https://arxiv.org/abs/2610.00758](https://arxiv.org/abs/2610.00758)

    该论文提出一种基于低秩假设的多任务逆强化学习方法，通过汇集多个智能体的数据缓解覆盖度要求，并使规划计算量随秩而非任务数扩展，从而实现对多任务在新环境中的可扩展高效评估。

    

    通过学习可迁移的奖励函数，逆强化学习（IRL）能够对智能体在修改后的环境中进行反事实评估。这种迁移对覆盖度提出了严格要求，因为目标环境会影响智能体的状态占据分布。我们提出了一种多任务IRL方法，该方法在低秩假设下，汇集在同一环境中具有不同奖励的多个智能体的数据。除了缓解覆盖度要求外（即每个任务无需访问每一个状态，只要其他任务访问过即可），该方法还支持在新环境下对多任务进行可扩展的评估，因为计算密集型的规划随秩而非任务数量进行扩展。我们为奖励恢复以及新环境中的策略学习提供了有限样本理论保证。实验表明，我们的方法对有限覆盖度具有鲁棒性，能够恢复每个任务支撑集之内及之外的奖励，并以低于基线方法的遗憾值迁移到目标环境。

    arXiv:2610.00758v1 Announce Type: cross  Abstract: By learning transferable rewards, inverse reinforcement learning (IRL) enables counterfactual evaluation of agents under modified environments. Such transfer places strict requirements on coverage since target environments affect agents' state occupancy. We propose a multi-task IRL method that pools data across multiple agents with different rewards in the same environment under a low-rank assumption. In addition to alleviating coverage requirements, so each task need not visit every state as long as others do, the method offers scalable evaluation of multiple tasks under new environments as computationally intensive planning scales with rank rather than the number of tasks. We provide finite sample guarantees on reward recovery and on policy learning in new environments. Experiments show our method is robust to limited coverage, recovers rewards on and off of each task's support, transfers to target environments at lower regret than b
    
[^34]: 学习电价定价以实现最优需求响应

    Learning to Price Electricity for Optimal Demand Response

    [https://arxiv.org/abs/2610.00755](https://arxiv.org/abs/2610.00755)

    本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。

    

    利用随时间变化的电价来引导消费者需求响应，并更好地使能源需求与可再生能源生产相匹配，这一点引起了广泛关注。然而，最优电价通常会随时间变化，以响应诸如天气预报、日出/日落时间和星期规律等复杂信号；而现有方法无法有效利用如此丰富的上下文信息。在此，我们提出了一种基于神经网络的上下文能量定价算法，将定价问题建模为Stackelberg博弈，并利用了Mehrabi等人（2024）提出的均场解表示方法。该方法学习从上下文特征到可行价格信号的受限映射。我们通过模拟美国多个城市的电网验证了我们的方法，结果表明，融入上下文信息可以显著提升需求响应计划的价值。

    arXiv:2610.00755v1 Announce Type: new  Abstract: There is considerable interest in using time-varying electricity prices to shape consumer demand response, and better align energy demand with renewable production. However, optimal prices generally vary over time in response to complex signals such as weather forecasts, sunrise/sunset times, and day-of-week patterns; and existing methods are not able to make efficient use of such rich contextual information. Here, we propose a neural-network-based algorithm for contextual energy pricing, modeling pricing as a Stackelberg game and leveraging a mean-field solution representation from Mehrabi et al.~(2024). The approach learns constrained mappings from contextual features to feasible price signals. We validate our approach by simulating the energy grid in several US cities, and show that incorporating contextual information can considerably increase the value of the demand response programs.
    
[^35]: 信号-噪声分解将干扰变异隔离到可移除的子空间中

    Signal-Noise Factorization Isolates Nuisance Variation into Removable Subspaces

    [https://arxiv.org/abs/2610.00751](https://arxiv.org/abs/2610.00751)

    该论文提出在训练中强化信号-噪声分解（SNF）与信号-信号分解（SSF）的正则化方法，将干扰变异隔离到可移除的子空间中，实验表明增强SNF能显著提升模型在CIFAR-100及医学图像腐蚀数据集上的性能。

    

    近期的理论工作识别出了表征几何的若干基本性质，这些性质塑造了深度神经网络的推理能力。其中包括信号-噪声分解（SNF），即把信号与噪声分离的能力；以及信号-信号分解（SSF），即把任务相关信号与任务无关信号分离的能力。在此，我们构建了在训练过程中强化这两种性质的正则化器。我们在CIFAR-100分类任务上将使用这些正则化器训练的网络与使用$L_2$正则化的基线网络进行比较，以了解我们的正则化器如何塑造表征几何，并影响模型在这一著名计算机视觉基线任务上的性能。通过正则化增强SNF提升了模型性能，而增强SSF则没有提升效果。出于生物医学应用的动机，我们研究了我们的正则化器如何在经五种严重程度MedMNIST-C腐蚀处理的BloodMNIST数据集上影响模型性能……

    arXiv:2610.00751v1 Announce Type: cross  Abstract: Recent theoretical work identified fundamental properties of representation geometry that shape inference ability of deep neural networks. These include signal-noise factorization (SNF), the ability to segregate signal from noise, and signal-signal factorization (SSF), the ability to segregate task-specific and task-irrelevant signals. Here, we built regularizers that reinforce these two properties during training. We compared networks trained with these regularizers to $L_2$-regularized baseline networks on the CIFAR-100 classification task to understand how our regularizers shape representation geometry and impact performance on a well-known computer vision baseline. Enhancing SNF via regularization improved model performance but enhancing SSF did not. Motivated by biomedical applications, we investigated how our regularizers affected performance on the BloodMNIST dataset treated with MedMNIST-C corruptions at five severity levels, a
    
[^36]: 面向大语言模型注意力的序列化函数结构化Tucker压缩

    Sequential Functional Structured Tucker Compression for Large Language Model Attentions

    [https://arxiv.org/abs/2610.00717](https://arxiv.org/abs/2610.00717)

    提出FTC序列化结构化压缩框架，在固定存储预算下联合利用Q/K/V头原生结构并适应先前压缩引起的表示偏移，无需微调即可在6B至32B的大语言模型上实现最先进的注意力压缩效果。

    

    LLM注意力的训练后压缩通常被表述为独立的矩阵近似问题，这既忽略了注意力投影之间的共享结构，也忽略了先前压缩所引入的表示偏移。我们提出FTC，一个序列化结构化压缩框架，它在固定存储预算下使近似适应于当前已压缩的模型，同时联合利用Q/K/V头原生的结构。输出投影则被单独处理，以应对注意力后表示的变化。FTC既不需要微调，也不需要基于梯度的恢复。在从6B到32B参数的七个仅解码器大语言模型上，FTC在五个现代GQA模型的每个测试保留率下都取得了对比方法中最低的WikiText-2困惑度，且在激进压缩下收益最大。这些改进能够迁移到下游任务，并在32B规模上依然保持显著优势。

    arXiv:2610.00717v1 Announce Type: cross  Abstract: Post-training compression of LLM attention is often formulated as independent matrix approximation, ignoring both the shared structure among attention projections and the representation shift introduced by earlier compression. We propose FTC, a sequential structured compression framework that adapts the approximation to the current compressed model while jointly exploiting the native Q/K/V head structure under a fixed storage budget. The output projection is handled separately to account for the changed post-attention representation. FTC requires neither fine-tuning nor gradient-based recovery. Across seven decoder-only LLMs from 6B to 32B parameters, FTC achieves the lowest WikiText-2 perplexity among the compared methods at every tested keep ratio on five modern GQA models, with the largest gains under aggressive compression. The improvements transfer to downstream tasks and remain substantial at the 32B scale.
    
[^37]: 三角传输映射的多保真度方法

    Multifidelity Formulations for Triangular Transport

    [https://arxiv.org/abs/2610.00698](https://arxiv.org/abs/2610.00698)

    该论文提出两种多保真度策略——相邻保真度层级间映射组合的分层方法与保单调性参数化修正的非分层方法，利用丰富的低保真度数据在仅有少量高保真度样本时更准确地构建三角传输映射。

    

    我们开发了从样本构建三角传输映射的多保真度方法，适用于高保真度数据稀缺而低保真度数据较为丰富的情形。利用这组多保真度数据，我们逼近一个三角传输映射，该映射在易于处理的参考密度与高保真度目标分布之间建立双射映射。我们提出了两种利用低保真度数据的策略：一种是在相邻保真度层级之间组合映射的分层方法；另一种是通过保持单调性的映射参数化修正来融入低保真度信息的非分层方法。数值实验将这些策略与单保真度传输方法进行比较，结果表明所提出的多保真度方法能够在有限的高保真度数据下改进映射估计。为了展示所学映射的更广泛用途，我们还将其应用于下游的摊销式基于仿真推断任务中。

    arXiv:2610.00698v1 Announce Type: cross  Abstract: We develop multifidelity methods for constructing triangular transport maps from samples, when high-fidelity data are scarce but lower-fidelity data are more abundant. Using this set of multifidelity data, we approximate a triangular transport map that bijectively maps between a tractable reference density and the high-fidelity target distribution. We introduce two strategies to leverage low-fidelity data: a hierarchical approach that composes maps between adjacent fidelity levels, and a non-hierarchical method that incorporates low-fidelity information through monotonicity-preserving corrections to the map parameterization. Numerical experiments compare these strategies with single-fidelity transport and demonstrate how the proposed multifidelity approaches can improve map estimation from limited high-fidelity data. To illustrate the broader utility of the learned maps, we also deploy them in a downstream amortized simulation-based in
    
[^38]: 压缩语言模型中散度如何演变为决策翻转

    How Divergence Becomes Decision Flips in Compressed Language Models

    [https://arxiv.org/abs/2610.00694](https://arxiv.org/abs/2610.00694)

    该研究证明全变差而非KL散度能直接预测压缩语言模型的决策翻转率，二者比例的中位数为1.05且无需拟合常数，而KL散度因先对词元取平均而无法可靠比较不同模型和语料库下的压缩效果。

    

    压缩报告通常通过KL散度来总结压缩后的语言模型与原始密集模型之间的偏离程度；而依赖密集模型输出的部署场景需要知道其有多少决策发生了改变。我们证明全变差能够直接回答这一问题。在五个语料库上对19个开源模型的802个压缩及扰动副本、涵盖九种机制上互不相关的扰动族的实验中，arg-max词元发生改变的速率（即“翻转率”）以中位数为1.05的比例跟踪全变差，且无需拟合常数。KL散度只能通过其平方根以及一个在不同模型和语料库间变化达四倍的因子来转化为翻转，这是因为KL散度在取平方根之前就先对词元进行了平均；按词元平均的一阶统计量（如Hellinger距离）可以避免这一问题，但压缩报告中很少提供这些统计量。因此，对于两个分别在不同模型和语料库上报告、翻转率相差至少……（原文截断）的压缩器，无法比较其优劣。

    arXiv:2610.00694v1 Announce Type: cross  Abstract: Compression reports summarize how far a compressed language model moved from the dense one, usually by a KL divergence; a deployment that relies on the dense model's outputs needs to know how many of its decisions changed. We show that total variation, not KL, answers this directly. Across 802 compressed and perturbed copies of 19 open models on five corpora and nine mechanically unrelated perturbation families, the rate at which the arg-max token changes (the \emph{flip rate}) tracks total variation at a ratio with median $1.05$, with no fitted constant. KL converts into flips only through its square root and a factor that varies fourfold across models and corpora, because KL averages over tokens before the root is taken; first-order statistics averaged per token, such as Hellinger distance, avoid this, but reports rarely give them. As a result, of two compressors reported on different models and corpora whose flip rates differ by at 
    
[^39]: 巨正则生成器

    Grand Canonical Generators

    [https://arxiv.org/abs/2610.00683](https://arxiv.org/abs/2610.00683)

    提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。

    

    我们提出了巨正则生成器，这是一种将玻尔兹曼生成器扩展到巨正则系综的生成式框架。我们提出了两种设计方案：第一种以化学势为条件对可变尺寸的生成模型进行条件化，从而联合采样粒子数和构型；第二种将巨正则分布分解为粒子数分布和相应的正则玻尔兹曼密度。这种分解式设计可以对正则分量复用任何现有的玻尔兹曼生成器，以解析方式编码已知的化学势线性依赖关系，并产生易于处理的似然，从而支持自归一化重要性采样（SNIS）。实验结果表明，GCG在Lennard-Jones流体和沸石中甲烷吸附问题上准确再现了巨正则观测量，展示了跨化学势的泛化能力，并可通过SNIS和巨正则蒙特卡洛进行校正。

    arXiv:2610.00683v1 Announce Type: cross  Abstract: We introduce Grand Canonical Generators (GCG), a generative framework that extends Boltzmann generators to the grand canonical ensemble. We present two designs. The first conditions a variable-size generative model on the chemical potential, sampling particle number and configuration jointly. The second factorizes the grand canonical distribution into a particle-number distribution and the corresponding canonical Boltzmann density. This factorized formulation can use any existing Boltzmann generator for the canonical component, encodes the known linear chemical-potential dependence analytically, and yields a tractable likelihood that supports self-normalized importance sampling (SNIS). Empirically, GCG accurately reproduces grand canonical observables on a Lennard--Jones fluid and methane adsorption in a zeolite, demonstrating generalization across chemical potentials and correction via SNIS and grand canonical Monte Carlo.
    
[^40]: 面向图像回归模型的自适应保形预测及其在惯性约束聚变仿真器中的应用

    Adaptive Conformal Prediction for Image Regression Models with Application to an Inertial Confinement Fusion Emulator

    [https://arxiv.org/abs/2610.00535](https://arxiv.org/abs/2610.00535)

    本文提出了基于最近邻的自适应保形预测方法（ACPNN），为图像回归模型提供依赖于输入的局部自适应不确定性量化，并将其应用于惯性约束聚变仿真器。

    

    不确定性量化在科学机器学习中至关重要，因为基于图像的黑盒模型越来越多地被部署在高风险场景中。在许多此类应用中，模型输出会为高成本决策提供依据，然而大多数方法仅提供点估计而未量化预测不确定性。模型内部的可访问性和可解释性有限，使得难以评估模型在输入空间不同区域上的可靠性，这进一步加剧了这一挑战。因此，人们越来越需要能够提供依赖于输入的不确定性估计的方法，以指导模型开发和下游实验。为了满足这一需求，我们提出了基于最近邻的自适应保形预测（ACPNN），这是一种用于图像回归的输入自适应保形框架。ACPNN 利用来自邻近样本的信息来产生局部自适应的不确定性估计，同时保持较低的……（原文摘要截断）

    arXiv:2610.00535v1 Announce Type: new  Abstract: Uncertainty quantification is critical in scientific machine learning, where black-box, image-based models are increasingly deployed in high-stakes settings. In many such applications, model outputs inform costly decisions, yet most methods provide only point estimates without quantifying predictive uncertainty. This challenge is compounded by the limited accessibility and interpretability of model internals, making it difficult to assess reliability across different regions of the input space. As a result, there is a growing need for methods that can provide input-dependent uncertainty estimates to guide both model development and downstream experimentation. To address this need, we propose Adaptive Conformal Prediction using Nearest Neighbors (ACPNN), an input-adaptive conformal framework for image regression. ACPNN leverages information from neighboring samples to produce locally adaptive uncertainty estimates while maintaining low co
    
[^41]: 分数拉普拉斯神经算子：精确架构、临界状态下的表达能力边界，以及记忆驱动网络动力学的可认证稳定性

    Fractional Laplace Neural Operators: Exact Architectures, an Expressivity Frontier at Criticality, and Certified Stability for Memory-Driven Network Dynamics

    [https://arxiv.org/abs/2610.00515](https://arxiv.org/abs/2610.00515)

    本文提出分数拉普拉斯神经算子（fLNO），证明单层图-谱架构可精确表示线性Volterra记忆算子，揭示了有限有理实现无法再现分数记忆非整数临界渐近行为的表达能力边界，并给出构造上即保证稳定性裕度的可训练参数化方法。

    

    神经算子学习函数空间之间的映射，而遗传性网络动力学则由具有非有理拉普拉斯符号的Volterra预解式来描述。我们提出了一种分数拉普拉斯神经算子（fLNO），将这种结构嵌入到所学习的映射之中。对于可交换的激励—拉普拉斯算子对，单个块图-谱层即可精确表示完整的线性Volterra解算子。我们建立了有限有理实现的表达能力边界：它们在紧频率窗口上以几何速率逼近分数记忆，但无法再现由分支点产生的非整数临界渐近行为，且在半直线上最佳有理逼近速率仅为根指数级。同一理论还给出了通过构造即可强制满足指定稳定性裕度的可训练参数化方法，以及一个graphon迁移定理，将真正的算子一致性与参数共享区分开来。在一个共同数据基准测试中，正有理……（原文摘要在此处截断）

    arXiv:2610.00515v1 Announce Type: new  Abstract: Neural operators learn maps between function spaces, while hereditary network dynamics are described by Volterra resolvents with non-rational Laplace symbols. We introduce a fractional Laplace neural operator (fLNO) that embeds this structure in the learned map. For commuting excitation--Laplacian pairs, one block graph-spectral layer represents the full linear Volterra solution operator exactly. We establish an expressivity frontier for finite rational realizations: they approximate fractional memory geometrically on compact frequency windows, but cannot reproduce the non-integer critical asymptotics generated by a branch point, and on the half-line the best rational rate is root-exponential. The same theory yields trainable parametrizations that enforce a prescribed stability margin by construction, and a graphon-transfer theorem separates genuine operator consistency from parameter sharing. In a common-data benchmark, positive rationa
    
[^42]: 异方差规范多面张量分解

    Heteroskedastic Canonical Polyadic Tensor Decomposition

    [https://arxiv.org/abs/2610.00498](https://arxiv.org/abs/2610.00498)

    本文提出异方差CP分解（HCP），通过引入低秩精度张量来建模张量条目的异方差性，并采用交替分块坐标上升方法从含噪观测中同时估计低秩均值与精度张量，其计算复杂度与CP-ALS相当。

    

    当最小化平方误差损失时，流行的CP分解可以被解释为高斯模型中的参数推断，该模型具有低秩均值张量，且张量各条目的方差恒定。我们提出了异方差CP（HCP），它用一个非常数、低秩的精度张量来建模逐条目的变异性，并开发了一种交替分块坐标上升方法，从含噪观测中同时恢复低秩均值张量和精度张量。我们的方法在计算上具有竞争力，其因子更新的首阶复杂度与CP-ALS相同。我们在合成实验和一个脑电图（EEG）应用中展示了HCP的效果。

    arXiv:2610.00498v1 Announce Type: cross  Abstract: When minimizing the squared-error loss, the popular CP decomposition can be interpreted as parameter inference in a Gaussian model with a low-rank mean tensor and constant variance across the tensor entries. We introduce heteroskedastic-CP (HCP), which models entrywise variability with a non-constant, low-rank precision tensor, and develop an alternating block-coordinate ascent method to recover both the low-rank mean and precision tensors from noisy observations. Our procedure is computationally competitive, with the same leading-order factor-update complexity as CP-ALS. We demonstrate HCP on synthetic experiments and an EEG application.
    
[^43]: SGD方法的精确信息核算

    Exact information accounting for SGD methods

    [https://arxiv.org/abs/2610.00446](https://arxiv.org/abs/2610.00446)

    该论文提出了SGD的精确信息论分析框架，证明预条件SGD步骤是高斯贝叶斯模型的后验均值更新，并给出一个信息核算恒等式，将凸收敛、鞍点逃逸、平坦性与泛化关系、学习率调度及各类SGD变体统一起来。

    

    作为标准几何分析方法的一种替代，我们对随机梯度下降（SGD）及其变体给出了精确的信息论分析。我们证明预条件化的SGD步骤等价于一个高斯贝叶斯模型的后验均值更新，并且其单步遗憾可以分解为内在时间成本与比较器信息变化两部分。这一分解还可扩展为关于目标函数本身的恒等式。凸收敛性、严格鞍点逃逸、平坦性与泛化之间的联系、标准学习率调度、自适应优化器，以及SGD的噪声、动量、重尾和无梯度变体，均对应于该恒等式中的某一项或某种特例。我们在合成数据和真实训练过程中测量了该恒等式的各项。在真实网络上，该恒等式将经典收敛界的松弛归因于其推导过程中所忽略的项，并且能够区分那些达到相同训练损失的不同优化器。这种区分……

    arXiv:2610.00446v1 Announce Type: cross  Abstract: As an alternative to the standard geometric analyses, we give an exact, information-theoretic analysis of stochastic gradient descent (SGD) and its variants. We show that a preconditioned SGD step is the posterior-mean update of a Gaussian Bayes model, and that its one-step regret splits into an intrinsic-time cost and a change in comparator information. The split extends to an identity for the objective itself. Convex convergence, strict-saddle-point escape, the link between flatness and generalization, the standard learning-rate schedules, adaptive optimizers, and the noisy, momentum, heavy-tailed, and gradient-free variants of SGD each correspond to a term or a special case of this identity. We measure its terms on synthetic and real training runs. On real networks it attributes the slack of classical convergence bounds to the terms their derivations drop and separates optimizers that reach the same training loss. That separation fo
    
[^44]: ChainLoRA：面向大语言模型持续学习的几何保持任务向量合并

    ChainLoRA: Geometry-Preserving Task Vector Merging for Continual Learning in LLMs

    [https://arxiv.org/abs/2610.00431](https://arxiv.org/abs/2610.00431)

    ChainLoRA 提出了一种免回放的持续学习合并框架，通过链式更新训练与自适应 SVD 合并来保持任务向量的几何结构，在恒定的历史状态与正则化开销下，平衡大语言模型的知识保留与新任务适应。

    

    面向大语言模型（LLMs）的持续参数高效微调必须在保留先前已获得的知识、适应新任务以及严格的参数预算之间取得平衡。我们提出了 ChainLoRA，一个建立在链式更新任务向量几何之上的免回放持续合并框架。从参数合并的视角出发，我们通过任务更新之间可度量的交互作用，对遗忘给出了几何化的表述，将方向重叠与系数耦合区分开来。基于这一观点，ChainLoRA 将链式更新训练与流后自适应 SVD 合并相结合。在训练期间，初始化和单边正交性代理仅使用最后一个载体，使得随着任务流的增长，其历史状态占用和正则化开销保持恒定。在合并阶段，自适应 SVD 提取一个共享载体，并通过 Procrustes 适配将其与最新任务对齐。我们的理论分析表明……（原文在此处截断）

    arXiv:2610.00431v1 Announce Type: new  Abstract: Continual parameter-efficient fine-tuning for large language models (LLMs) must balance retention of previously acquired knowledge, adaptation to new tasks, and strict parameter budgets. We present \textbf{ChainLoRA}, a replay-free continual merging framework built on chain-updated task-vector geometry. From a parameter-merging perspective, we formulate a geometric view of forgetting through a measurable interaction between task updates, separating directional overlap from coefficient coupling. Building on this view, ChainLoRA combines chain-updated training with post-stream adaptive SVD merging. During training, initialization and a one-sided orthogonality proxy use only the last carrier, keeping their historical-state footprint and regularization overhead constant as the task stream grows. At merging time, Adaptive SVD extracts a shared carrier and aligns it to the latest task through Procrustes adaptation. Our theoretical analysis sho
    
[^45]: IrekoGPT：将结构化剪枝转化为事后可调节宽度的可瘦身大语言模型

    IrekoGPT: Turning Structured Pruning into Post-Hoc Slimmable LLMs

    [https://arxiv.org/abs/2610.00426](https://arxiv.org/abs/2610.00426)

    IrekoGPT提出一种事后方法，通过保留SliceGPT投影矩阵、多压缩率逐层校准和无梯度岭回归修正，将预训练大语言模型转换为推理时可调节宽度的可瘦身模型，并在高压缩率下显著优于基于PCA的朴素瘦身方法。

    

    我们提出IrekoGPT，一种事后（post-hoc）方法，可将预训练大语言模型转换为可瘦身（slimmable）模型，其宽度可在推理时进行调节。该方法基于SliceGPT，保留其投影矩阵而不进行剪枝，使单一模型能够暴露出不同宽度的嵌套子网络。我们通过在多个压缩率下校准每一层来提升鲁棒性，并通过无梯度的岭回归对下游线性层进行修正。在Llama和Qwen模型上的初步结果显示，该方法优于朴素的基于PCA的瘦身方法，且在高压缩率下收益最大。代码可在 https://github.com/aimagelab/IrekoGPT 获取。

    arXiv:2610.00426v1 Announce Type: cross  Abstract: We introduce IrekoGPT, a post-hoc method for converting pretrained LLMs into slimmable models whose width can be adjusted at inference time. Building on SliceGPT, we retain its projection matrices without pruning them, allowing a single model to expose nested subnetworks at different widths. We improve robustness by calibrating each layer across multiple compression ratios, and correct downstream linear layers through gradient-free ridge regression. Across Llama and Qwen models, preliminary results show improvements over naive PCA-based slimming, with the largest gains at high compression. Code is available at https://github.com/aimagelab/IrekoGPT
    
[^46]: 目标依赖的因果修复极限：高斯模型中的前导对数前沿

    Target-Dependent Limits of Causal Repair: A Leading-Log Frontier in a Gaussian Model

    [https://arxiv.org/abs/2610.00424](https://arxiv.org/abs/2610.00424)

    该论文在高斯因果模型中量化了因果预测器潜在改进与实际学到的修复增益之间的差距，证明在 1/k 学习尺度下所有可行学习器都面临 k^{-2} 的评估下界，并在幅度充裕情形下刻画出尖锐的前导对数评估指数前沿 min{ℓ_k, 2kη_k/U}。

    

    知道一个因果预测器能够改进多少，并不必然揭示实际学到的修复所获得的增益。我们在一个具有已知干预几何结构的标量高斯因果实验中量化了这一差距：辅助数据可在有界污染条件下识别效应大小，而诊断数据识别方向。研究目标是被实际训练出的修复相对于拟合参考模型的平方损失增益。在学习均方误差 η 一致约束下联合优化学习器与评估器，可以避免“不做任何修复”的平凡解。在通常的 1/k 学习尺度下，即使预言机潜势能够以更快的速率被估计，每个可行的学习器仍会面临一个 k^{-2} 的评估下界。在幅度充裕的情形下，我们刻画了一个尖锐的前导对数前沿：评估指数在一阶相对精度下为 min{ℓ_k, 2kη_k/U}，其中 ℓ_k = log(1/(k²E_k))，E_k 为辅助数据精度。一个诊断-弃权规则可以达到这一……

    arXiv:2610.00424v1 Announce Type: cross  Abstract: Knowing how much a causal predictor could improve need not reveal the gain of the repair actually learned. We quantify this gap in a scalar Gaussian causal experiment with known intervention geometry: auxiliary data identify effect magnitude up to bounded contamination, while diagnostics identify direction. The target is the squared-loss gain of the realized trained repair relative to a fitted reference. Jointly optimizing the learner and assessor under uniform learning MSE $\eta$ avoids the trivial solution of making no repair. At the usual $1/k$ learning scale, every feasible learner incurs a $k^{-2}$ assessment floor, even when oracle potential is estimable at a faster rate. In the magnitude-rich regime, we characterize a sharp leading-log frontier: the assessment exponent is $\min{\ell_k,2k\eta_k/U}$ to first relative order, where $\ell_k=\log(1/(k^2E_k))$ and $E_k$ is auxiliary precision. A diagnostic-abstention rule attains this 
    
[^47]: 学习局部覆盖：硬信息视界下的图神经网络组合优化

    Learning to Cover Locally: Graph Neural Combinatorial Optimization under a Hard Information Horizon

    [https://arxiv.org/abs/2610.00422](https://arxiv.org/abs/2610.00422)

    本文形式化了每个节点只能看到k跳邻域的“硬信息视界”下组合优化问题（以OLSRv2协议的NP难MPR选择为实例），并证明L层GNN严格等价于L跳选择器，即模型容量无法弥补信息半径的不足。

    

    神经组合优化通常假设存在一个能读取完整实例的集中式求解器。我们研究相反的情形：硬信息视界下的组合优化，即每个节点仅凭其 $k$ 跳邻域内的局部信息，就要对全局解中属于自己的部分做出承诺，且这些承诺必须能够拼合成一个全局可行的解。我们将该问题形式化为局部集合覆盖，并以加权多点中继（MPR）选择为实例进行验证——这是优化链路状态路由协议第2版（OLSRv2，RFC 7181）中的NP难的2跳覆盖问题，其信息视界由协议本身强加，而非建模者自行选择。我们证明了两项结果：任何视界少一跳的确定性选择器要么无法完成覆盖，要么与最优解相差 $\Delta$ 因子；而在决策节点处读取输出的 $L$ 层图神经网络（GNN）恰好等价于一个 $L$ 跳选择器，因此提升模型容量无法弥补信息半径的不足。反之……

    arXiv:2610.00422v1 Announce Type: cross  Abstract: Neural combinatorial optimization typically assumes a centralized solver that reads the whole instance. We study the opposite: combinatorial optimization under a hard information horizon, where every node commits to its share of a global solution seeing only its $k$-hop neighborhood, and those commitments must compose into a globally feasible solution. We formalize this as local set cover and instantiate it on weighted multipoint relay (MPR) selection, the NP-hard 2-hop covering problem of the Optimized Link State Routing Protocol version 2 (OLSRv2) routing protocol (RFC~7181), whose horizon is imposed by the protocol, not chosen by the modeler. We prove two results. Any deterministic selector whose horizon is one hop short must either fail coverage or land a factor $\Delta$ from optimal, and an $L$-layer graph neural network (GNN) read out at the deciding node is exactly an $L$-hop selector, so capacity cannot buy back radius. Convers
    
[^48]: 可迁移图元网络

    Transferable Graph Metanetworks

    [https://arxiv.org/abs/2610.00420](https://arxiv.org/abs/2610.00420)

    提出可迁移图元网络，基于表示不变性与连续性两项设计原则，使元网络的性能能够跨不同宽度的输入神经网络迁移，从而实现“在小网络上训练、在大网络上评估”的效率提升。

    

    权重空间网络（或称元网络）以另一个神经网络的权重作为输入，并预测该网络的性质。大多数先前的工作仅在一种或几种固定大小的输入网络上训练此类模型，并在分布内进行评估。少数针对分布外尺寸泛化的尝试范围有限，且仅取得了较为有限的成功。因此，在小网络上训练并在大得多的网络上进行评估所带来的潜在效率提升在很大程度上尚未实现。我们提出可迁移图元网络，它通过对图元网络范式的一系列修改，使性能能够在不同宽度的输入网络之间迁移。这些修改遵循两个原则：一是对不同宽度的网络表示同一函数的方式保持不变性；二是连续性，即表示相似函数的权重会获得相似的预测。我们进一步研究了尺寸泛化是否……（原文摘要在此处截断）

    arXiv:2610.00420v1 Announce Type: new  Abstract: A weight space network (or metanetwork) takes the weights of another neural network as input and predicts properties of it. Most prior work trains such models on input networks of one or a few fixed sizes and evaluates them in-distribution. The few attempts at out-of-distribution size generalization remain limited in scope and have achieved only modest success. Consequently, the potential efficiency gains of training on small networks and evaluating on much larger ones remain largely unrealized. We propose Transferable Graph Metanetworks, which extend the graph metanetwork paradigm with a set of modifications that make performance transferable across input networks of different widths. The modifications follow two principles: invariance to the ways in which networks of different widths represent the same function, and continuity, such that weights representing similar functions receive similar predictions. We further study whether size g
    
[^49]: VANDAM：利用DNA分子先验来观察核苷酸序列

    VANDAM: Viewing a nucleotide sequence with DNA molecular priors

    [https://arxiv.org/abs/2610.00411](https://arxiv.org/abs/2610.00411)

    VANDAM框架通过在自监督训练中预测区域分子性质并在有标签时于输入端注入局部特征，将DNA分子先验融入基因组基础模型，从而在多种架构和下游任务上持续提升性能。

    

    当代基因组基础模型（GFMs）依赖于“DNA作为字符串”的范式，采用掩码token预测目标进行预训练。然而，这种抽象并未显式建模对生物功能至关重要的生化、结构和物理性质。许多分子性质可以通过成熟的生物物理模型从序列中估计出来，因此它们的价值不在于提供一种独立的模态，而在于引入训练目标能够显式利用的先验。我们提出了VANDAM，一个用DNA分子先验来扩展基因组基础模型训练的框架。在自监督训练中，VANDAM从池化表示中预测区域分子性质。当功能标签可用且能够奖励对分子先验的保留时，局部特征还会在输入端被注入。VANDAM在四种架构家族和九个（下游任务上）持续提升下游性能。

    arXiv:2610.00411v1 Announce Type: cross  Abstract: Contemporary Genomic Foundation Models (GFMs) rely on a DNA-as-a-string paradigm that employs masked token prediction objectives for pretraining. However, this abstraction does not explicitly model the biochemical, structural, and physical properties essential to biological function. Many molecular properties can be estimated from sequence using established biophysical models, so their utility lies not in providing an independent modality, but in introducing priors that training objectives can explicitly exploit. We introduce VANDAM, a framework that extends the training of GFMs with DNA molecular priors. In self-supervised training, VANDAM predicts regional molecular properties from pooled representations. When functional labels are available and can reward retaining molecular priors, local features are additionally injected at the input. VANDAM consistently improves downstream performance across four architecture families and nine he
    
[^50]: RACE：面向邻居丰富场景的时间序列基础模型预测的残差感知测试时自适应

    RACE: Residual-Aware Test-Time Adaptation for Neighbor-Rich Time-Series Foundation Model Forecasting

    [https://arxiv.org/abs/2610.00405](https://arxiv.org/abs/2610.00405)

    该论文提出RACE，一种残差感知的测试时自适应方法，通过处理邻居序列间相互矛盾的残差证据与跨场景变化的残差模式，在无需逐域微调的情况下提升时间序列基础模型在邻居丰富预测场景中的预测性能。

    

    时间序列基础模型（TSFMs）在各类预测任务中表现优异，但其逐序列的推理方式并不适用于“邻居丰富”的预测场景——即每个查询都能访问相关但不完全相同的历史序列。连续血糖监测（CGM）和Web/云工作负载正是此类场景的典型代表：CGM轨迹虽共享生理模式，却在个体、设备和条件上各有差异；Web/云工作负载则将常见运行模式与非平稳性、重尾分布和突发流量相结合。这些历史序列虽共享有用的结构，但各邻居序列的相关性并不相同。现有方法要么针对每个目标域对TSFMs进行微调，导致额外成本且跨骨干模型的可迁移性有限；要么直接拼接检索到的序列，而不验证其是否真正有助于当前预测。关键挑战在于来自邻居序列的相互矛盾的残差证据，以及跨（原文在此处截断）变化的残差模式。

    arXiv:2610.00405v1 Announce Type: cross  Abstract: Time-series foundation models (TSFMs) perform strongly across forecasting tasks, but their per-series inference is ill-suited to neighbor-rich forecasting, where each query has access to related but nonidentical historical series. Continuous glucose monitoring (CGM) and Web/cloud workloads exemplify this setting: CGM trajectories share physiological patterns but vary across individuals, devices, and conditions, while Web/cloud workloads combine common operating regimes with non-stationarity, heavy tails, and bursts. These histories share useful structure, yet neighbors are not equally relevant. Existing methods either fine-tune TSFMs for each target domain, incurring additional costs and offering limited transferability across backbones, or append retrieved series without verifying whether they support the current forecast. The key challenges are conflicting residual evidence from neighboring series and residual patterns that vary acro
    
[^51]: MatrixReward：基于评分标准矩阵的开放式生成奖励方法

    MatrixReward: Reward from Rubric Matrix for Open-Ended Generation

    [https://arxiv.org/abs/2610.00389](https://arxiv.org/abs/2610.00389)

    MatrixReward通过在每条评分标准下对采样回答进行两两比较构建胜率矩阵，利用列离散度衡量评分标准的区分能力、列相关性检测评分标准的重复性，从而生成数据自适应的评分标准权重，为缺乏标准答案的开放式生成构造更有效的奖励信号。

    

    开放式查询生成缺乏标准答案，因此需要一种有效的奖励机制。逐点式评分标准（rubrics）对于同一提示下样本答案之间的相对质量所能提供的信息有限；而将多个评分标准的判断合并为单一分数，也可能掩盖这些答案之间的差异。我们提出MatrixReward，它通过在每条评分标准下对所有采样回答进行两两比较，构建一个“回答×评分标准”的胜率矩阵，并据此构造奖励。矩阵每一列的离散程度刻画了该评分标准对当前采样回答的区分能力，而列与列之间的相关性则揭示评分标准的重复性；这些统计量共同生成依赖于数据的评分标准权重。我们将这些权重与评分标准的先验权重相结合。在列归一化和加权之后，各评分标准下观测到的最大值和最小值定义了正理想轮廓与负理想轮廓，每个采样回答到这两个轮廓的距离……

    arXiv:2610.00389v1 Announce Type: cross  Abstract: Open-ended query generation lacks standard answers, thus necessitating an effective reward mechanism. Pointwise scoring rubrics provide limited information about the relative quality of sample answers under the same prompt; merging multiple rubric judgments into a single score may also mask the differences between these answers. We propose MatrixReward, which constructs rewards from a rollout-by-rubric win-rate matrix obtained by comparing every pair of sampled responses under each rubric. The spread of each matrix column captures how strongly that rubric distinguishes the current rollouts, while correlations between columns reveal rubric repetition; together, these statistics yield data-dependent rubric weights. We combine these weights with the prior weights of rubrics. After column normalization and weighting, the observed per-rubric maxima and minima define positive and negative ideal profiles. Each rollout's distances to these two
    
[^52]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^53]: STCFormer：面向站点天气预报的基于动态聚类Transformer的自适应时空建模

    STCFormer: Adaptive Spatio-Temporal Modeling with Dynamic Cluster Transformer for Station-based Weather Forecasting

    [https://arxiv.org/abs/2610.00377](https://arxiv.org/abs/2610.00377)

    提出STCFormer，一种根据每个时间补丁内站点局部演化动态分组站点的自适应时空Transformer，通过融合簇内细粒度局部注意力与区域级全局注意力来提升站点天气预报的准确性。

    

    基于站点的天气预报支撑着日常生活与经济活动，然而准确的预报需要对站点之间复杂的空间依赖关系进行建模。近期基于聚类的选择性建模为稠密的站点间交互提供了一种有前景的替代方案。然而，在整个观测窗口内共享的分组方式可能掩盖站点关系的局部变化，而仅依靠簇内交互也可能遗漏重要的全局上下文信息。此外，选择性交互相对稠密连接的理论优势也尚未得到充分理解。因此，我们提出了STCFormer，这是一种自适应的时空Transformer，它根据每个时间补丁内站点的局部演化动态地对站点进行分组。其聚类引导注意力块结合了簇内的细粒度局部注意力与基于区域状态摘要的全局注意力，使每个站点都能获取自身簇之外的信息……

    arXiv:2610.00377v1 Announce Type: cross  Abstract: Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective modeling offers a promising alternative to dense inter-station interactions. However, a grouping shared across an observation window may obscure local changes in station relationships, while intra-cluster interactions alone may miss important global context. The theoretical advantages of selective interactions over dense connectivity also remain insufficiently understood. We therefore propose STCFormer, an adaptive spatio-temporal Transformer that dynamically groups stations according to their local evolution within each temporal patch. Its Cluster-Guided Attention Block combines fine-grained local attention within clusters and global attention over regional state summaries, allowing each station to access information beyond its own cluster. W
    
[^54]: M$^2$Weather：一个联合多站点与多变量天气预报的基准测试

    M$^2$Weather: A Benchmark for Joint Multi-Station and Multi-Variable Weather Forecasting

    [https://arxiv.org/abs/2610.00370](https://arxiv.org/abs/2610.00370)

    该论文提出了 M$^2$Weather 基准，通过收集覆盖法国、欧洲和全球三个空间尺度的 2,809 个高质量站点与 5 个物理耦合天气变量，并提供统一的训练与评估协议，首次实现了对多站点空间依赖与多变量物理耦合的联合系统性评估。

    

    站点天气预报从根本上受到两方面因素的共同影响：站点之间复杂的空间依赖关系，以及天气变量之间强烈的物理耦合。然而，现有研究通常将这两类关系分开考虑，并使用不同的数据集和实验设置，这阻碍了对它们各自贡献及联合贡献的系统性评估。在本文中，我们提出了 M$^2$Weather，一个面向联合多站点、多变量天气预报的基准数据集。通过多标准质量控制与站点分层，我们收集了覆盖法国、欧洲和全球三个空间尺度的 2,809 个高质量站点，并包含 5 个物理耦合的天气变量。这种多尺度设计使我们能够检验相关结论能否从国家级站点网络推广到全球站点网络。我们还引入了统一的训练与评估协议，以便对不同站点-变量建模范式进行公平比较。为了进一步考察……（原文摘要在此处被截断）

    arXiv:2610.00370v1 Announce Type: cross  Abstract: Station weather forecasting is fundamentally shaped by both complex spatial dependencies across stations and strong physical coupling among weather variables. However, existing studies often consider these relationships separately and use different datasets and experimental settings, hindering systematic assessment of their individual and joint contributions. In this paper, we introduce $M^2$Weather, a benchmark for joint multi-station and multi-variable weather forecasting. Through multi-criteria quality control and station stratification, we collect 2,809 high-quality stations with 5 physically coupled weather variables across three spatial scales: France, Europe, and Global. This multi-scale design lets us examine whether conclusions persist from national to global station networks. We also introduce unified training and evaluation protocols to enable fair comparison of different station-variable modeling paradigms. To further exami
    
[^55]: 基于正例-未标注数据的部分AUC最大化

    Partial AUC Maximization from Positive-unlabeled Data

    [https://arxiv.org/abs/2610.00284](https://arxiv.org/abs/2610.00284)

    本文提出了一种无需负例数据、仅利用正例和未标注数据即可最大化部分AUC（pAUC）的方法，解决了实际中负例数据因隐私或标注专业性要求而难以收集的问题。

    

    接收者操作特征曲线的部分下面积是二分类的一项重要性能指标，它概括了在特定假阳性率范围内的真阳性率。在网络安全、医疗保健和广告等许多现实应用中，需要获得具有高pAUC的分类器。尽管已经提出了许多最大化pAUC的方法，但它们通常需要已标注的正例数据和负例数据进行训练。然而，在实际中，由于隐私问题或标注需要高度专业知识，已标注的负例数据往往难以收集。在本文中，我们提出了一种无需负例数据、仅从正例和未标注数据中最大化pAUC的方法。在经验风险最小化框架内，我们证明了pAUC（包括其依赖于FPR的阈值）可以仅用正例数据和边际分布来表示。

    arXiv:2610.00284v1 Announce Type: cross  Abstract: The partial area under the receiver operating characteristic curve (pAUC) is an important performance metric for binary classification that summarizes true positive rates within a specific range of false positive rates (FPRs). Classifiers that achieve high pAUC need to be obtained in many real-world applications such as cybersecurity, medical care, and advertising. Although many methods for maximizing the pAUC have been proposed, they typically require both labeled positive and negative data for training. However, in practice, labeled negative data are often difficult to collect due to privacy concerns or the need for high expertise to annotate them. In this paper, we propose a method for maximizing the pAUC from positive and unlabeled (PU) data without negative data. Within an empirical risk minimization framework, we show that the pAUC, including its FPR-dependent thresholds, can be represented using only the positive and marginal de
    
[^56]: 加权数据选择：上半区与五维的尖锐定律

    Weighted Data Selection: Sharp Upper-Half and Five-Dimensional Laws

    [https://arxiv.org/abs/2610.00101](https://arxiv.org/abs/2610.00101)

    该论文为加权最小二乘的最小范数学习器证明了在 ⌈3d/2⌉ ≤ n ≤ 2d−1 范围内精确的风险定律 Γ_d(n)=3−n/d，并在 (d,n)=(5,6) 时进一步证明 Γ_5(6)=11/5，其完整的数据集层面之上界与尖锐性构造均已在 Lean 4 中形式化验证。

    

    一个规模很小的重新加权的训练支撑集会保留多少风险？对于采用最小范数学习器的有限加权最小二乘问题，我们在整个区间 ⌈3d/2⌉ ≤ n ≤ 2d−1 上证明了精确的定律 Γ_d(n) = 3 − n/d。该保证覆盖每一种观测到的特征秩，并且所使用的选取方法保持了完整的特征张成空间。平衡单纯形锚点用于降低维度；正权提升与独立线压缩补齐了风险界的论证；平移坐标对则达到了与之匹配的下界。完整的数据集层面的上界与尖锐性构造已在 Lean 4 中得到验证。在更小的预算 (d, n) = (5, 6) 下，我们还证明了 Γ_5(6) = 11/5，它与由 5 = 3 + 2 导出的单纯形块预测相匹配，且该预测对任意交互构型均成立。圈覆盖、比较二阶矩与圈平面概率给出了尖锐的超出量 6/5，而极面几何解决了共享三秩圈的问题。一般的单纯形块……（原文摘要在此处截断）

    arXiv:2610.00101v1 Announce Type: new  Abstract: How much risk does a small reweighted training support retain? For finite weighted least squares with the minimum-norm learner, we prove the exact law $\Gamma_d(n)=3-n/d$ throughout $\lceil3d/2\rceil\leq n\leq2d-1$. The guarantee covers every observed feature rank and uses selections that preserve the full feature span. Balanced simplex anchors reduce dimension; positive-weight lifting and independent-line compression close the risk bound. Shifted coordinate pairs attain the matching lower bound. The complete dataset-level upper bound and sharpness construction are verified in Lean 4. At the smaller budget $(d,n)=(5,6)$, we also prove $\Gamma_5(6)=11/5$, matching the simplex-block prediction from $5=3+2$ over arbitrary interacting configurations. Circuit covers, comparison second moments, and circuit-plane probabilities give the sharp excess $6/5$, while polar-face geometry resolves shared rank-three circuits. The general simplex-block f
    
[^57]: 动态空间贝叶斯机器学习模型：在美国代际经济流动性与地理收入不平等中的应用

    Dynamic Spatial Bayesian Machine Learning Model: Applications to Intergenerational Economic Mobility and Geographic Income Inequality in the United States

    [https://arxiv.org/abs/2610.00072](https://arxiv.org/abs/2610.00072)

    该论文提出了一种结合马蹄铁收缩的动态空间面板贝叶斯可加回归树模型（DSP-BART-HS），在九种模拟情景中均达到最优或与最优统计上无差异的性能，尤其显著优于传统区域-时间聚合方法，能有效处理个体层面的非线性效应。

    

    我们开发了一种结合马蹄铁收缩先验的动态空间面板贝叶斯可加回归树模型（DSP-BART-HS），用于高维时空面板数据分析。我们在九种数据生成情景（每种情景包含200次重复实验）下，将该模型与结构空间计量经济学方法、非参数机器学习方法以及小区域估计器的完整体系进行联合评估，这些情景涵盖不规则空间拓扑、密集政策效应以及非线性个体层面交互作用。结果显示，DSP-BART-HS在每种情景中都是最优估计器，或与最优估计器在统计上无显著差异。当个体层面的非线性驱动结果方差时，传统的区域-时间聚合对比方法会出现严重的性能下降——落后三倍以上——而本框架的树集成设计直接解决了这一局限。该模型在零训练区域的条件下也保持了较强的预测准确性。

    arXiv:2610.00072v1 Announce Type: cross  Abstract: We develop a Dynamic Spatial Panel Bayesian Additive Regression Trees model with Horseshoe shrinkage (DSP-BART-HS) for high-dimensional spatio-temporal panel data. We jointly evaluate the model against a comprehensive suite of structural spatial econometrics, non-parametric machine learning methods, and small-area estimators across nine data-generating scenarios spanning 200 replicates each, including irregular spatial topologies, dense policy effects, and non-linear individual-level interactions. DSP-BART-HS is the best or statistically indistinguishable from the best estimator in every scenario. Conventional region-time-aggregate comparators suffer severe performance degradation -- trailing by a factor of three or more -- whenever individual-level non-linearity drives outcome variance, a limitation this framework's tree-ensemble design directly addresses. The model also maintains strong predictive accuracy under a zero-training-regio
    
[^58]: 具有多个最优臂的多臂老虎机：极小极大遗憾与非自适应性

    Bandits with Multiple Optimal Arms: Minimax Regret and Non-Adaptivit

    [https://arxiv.org/abs/2609.38659](https://arxiv.org/abs/2609.38659)

    该论文针对具有多个最优臂的多臂老虎机问题，通过对子采样算法的更精细分析建立了近乎极小极大最优的遗憾界 $\tilde{O}(\frac{K-A}{\sqrt{KA}}\sqrt{T})$，给出匹配下界，并证明了解最优臂数量对于达到近乎最优遗憾是必要的。

    

    我们研究具有多个最优臂的多臂老虎机问题，其动机在于许多实际的决策问题往往允许多个正确答案。对于具有 $A$ 个最优臂的 $K$ 臂老虎机，我们首先对先前的子采样算法（De Heide et al., 2021; Zhu and Nowak, 2020）进行了更精细的分析，建立了 $\tilde{O}\Big(\frac{K-A}{\sqrt{KA}}\sqrt{T}\Big)$ 的极小极大遗憾，其中 $T$ 是总交互次数，$\tilde{O}(\cdot)$ 省略了所有常数和对数因子，改进了此前 $\tilde{O}(\sqrt{KT/A})$ 的遗憾界。随后我们给出了在对数因子意义下与之匹配的下界，表明我们所建立的速率几乎达到了极小极大最优。我们进一步证明，在 $\tilde{O}(1)$ 因子意义下掌握最优臂的数量 $A$ 是实现近乎最优遗憾的必要条件，因为针对某一最优臂数量设计的近乎最优算法，在最优臂数量较少时必然会产生远大于最优遗憾的损失。

    arXiv:2609.38659v1 Announce Type: cross  Abstract: We study multi-armed bandits (MAB) with multiple optimal arms, motivated by the fact that many practical decision making problems admit multiple correct answers. For $K$-armed bandits with $A$ optimal arms, we first provide a sharper analysis of previous sub-sampling algorithms (De Heide et al., 2021; Zhu and Nowak, 2020), establishing a $\tilde{O}\Big(\frac{K-A}{\sqrt{KA}}\sqrt{T} \Big)$ minimax regret, where $T$ is the total number of interactions and $\tilde O(\cdot)$ drops all constant and logarithmic factors, improving the previous $\tilde{O}(\sqrt{KT/A})$ regret. We then provide a matching lower bound up to logarithmic factors, indicating that our established rate is nearly minimax-optimal. We further show that the knowledge of $A$ up to $\tilde{O}(1)$ factors is necessary to achieve near-optimal regret, as near-optimal algorithms for one number of optimal arms must incur substantially larger regret than optimal regret for a smal
    
[^59]: 时滞物理系统驱动项与动力学的可辨识性保证

    Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems

    [https://arxiv.org/abs/2609.37944](https://arxiv.org/abs/2609.37944)

    本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。

    

    目前已有大量方法被提出，包括物理信息神经网络（功能强大但不保证动力学的可辨识性）、符号回归（需要一组预先计算好的操作）以及因果发现（更具原则性，但通常依赖于物理系统可能违反的强假设）。在本工作中，我们开发了一种有理论支撑的方法，并证明在一组宽松的假设下，随机时滞微分方程的结构驱动项和漂移项是可辨识的。我们的方法在驱动项可辨识性基准测试中优于其他方法，并在第二个用于评估所学动力学物理一致性的基准测试中也表现更佳。

    arXiv:2609.37944v1 Announce Type: cross  Abstract: A wide range of methods have been proposed, including physics-informed neural networks, which are powerful but do not guarantee identifiability of the dynamics, symbolic regression, which requires a set of precomputed operations, and causal discovery, which is more principled but usually relies on strong assumptions that physical systems may violate. In this work, we develop a theory-grounded method and prove that under a set of permissive assumptions, the structural drivers and drift of stochastic delayed differential equations are identifiable. Our method outperforms others on a benchmark for driver identifiability, and on a second benchmark to evaluate physical consistency of the learned dynamics.
    
[^60]: 面向在线策略蒸馏的无偏Top-k估计

    Unbiased Top-$k$ Estimation for On-Policy Distillation

    [https://arxiv.org/abs/2609.34447](https://arxiv.org/abs/2609.34447)

    该论文提出了用于在线策略蒸馏中反向KL散度梯度估计的无偏Top-k估计方法，在有限计算成本下实现比采样token更丰富的分布监督。

    

    在线策略蒸馏（OPD）正日益成为大语言模型（LLM）后训练中的重要组成部分，用于将强大的教师大语言模型的推理能力迁移到较弱的学生大语言模型上。OPD通过学生策略生成的采样序列，最小化教师与学生之间的反向KL散度来训练学生模型。然而，在OPD中估计反向KL散度的梯度仍然是一个挑战。仅使用学生生成序列中采样得到的token在计算上成本低廉，但提供的分布监督有限，从而会降低准确性。此外，使用完整词表可以提供完整的分布监督，但计算代价高昂。因此，近期的工作提出了Top-k OPD（TK-OPD），它使用筛选出的top-k个token，在计算成本远低于完整词表估计的同时，提供比采样token估计更丰富的分布监督……（摘要在此处截断）

    arXiv:2609.34447v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) is becoming an important component of large language model (LLM) post-training for transferring the reasoning capability of a strong teacher LLM to a weaker student LLM. OPD trains the student by minimizing the reverse KL divergence between the teacher and the student via rollouts generated by the student's policy. However, estimating the gradient of the reverse KL divergence in OPD remains a challenge. Using only the sampled token from the student-generated rollout is computationally cheap but provides limited distributional supervision, which will degrade accuracy. In addition, using the full vocabulary provides complete distributional supervision but is computationally expensive. Therefore, recent works propose Top-$k$ OPD (TK-OPD) that use selected top-$k$ tokens, which provides richer distributional supervision than sampled-token estimation at substantially lower computational cost than full-vo
    
[^61]: 面向聚类回归的精确求解子样本集成方法：无需截尾水平的截尾技术

    Ensembles of Exactly Solved Subsamples for Clusterwise Regression: Trimming Without a Trimming Level

    [https://arxiv.org/abs/2609.31019](https://arxiv.org/abs/2609.31019)

    该论文提出一种基于精确求解随机子样本的聚类回归集成方法，可自动估计截尾水平而无需预先设定，在响应变量含高达20%粗大离群值时实现0.89的最坏情况准确率，优于传统的截尾交替法。

    

    聚类最小二乘法将回归数据划分为K个组，并为每组拟合独立的线性模型。我们研究了一种基学习器为精确求解的集成方法：在B个大小为m << n的随机子样本上，将问题求解至全局最优，然后通过最近曲面分配扩展每个解，对齐标签，并通过投票或选择方式组合各次重复结果。每个重复结果都是m个点上的经验K-量化器，因此该集成方法可以进行精确分析。只要“干净子样本的概率”与“干净数据上重复结果准确率”的乘积超过二分之一，投票在某个单元上就是正确的，无论污染值如何；在集成方法尚未标记的单元上进行迭代，可得到一种能够估计截尾水平的变体，而无需事先给定截尾水平。当响应变量中存在高达20%的粗大离群值时，该方法的最坏情况准确率达到0.89，而真实污染比例下的截尾交替法仅为0.80，且在任何固定截尾水平下表现都更低。

    arXiv:2609.31019v1 Announce Type: cross  Abstract: Clusterwise least squares partitions regression data into K groups with separate linear fits. We study an ensemble whose base learner is exact: solve the problem to global optimality on each of B random subsamples of size m << n, extend each solution by nearest-surface assignment, align the labels, and combine the replicates by vote or by selection. Each replicate is then an empirical K-quantizer on m points, and the ensemble admits an exact analysis. A vote is correct at a unit once the probability of a clean subsample times the clean-data replicate accuracy exceeds one half, whatever the contaminating values; iterating the ensemble on the units it has not flagged gives a variant that estimates the trimming level rather than requiring it. With up to 20% of gross outliers in the response its worst-case accuracy was 0.89, against 0.80 for trimmed alternation at the true contamination fraction and less at every fixed level tried. Conditi
    
[^62]: 基于模拟退火的三阶朗之万动力学在非凸优化中的全局收敛性

    Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing

    [https://arxiv.org/abs/2609.28611](https://arxiv.org/abs/2609.28611)

    该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。

    

    我们研究了在固定摩擦力与递减噪声的模拟退火框架下，三阶朗之万动力学用于非凸优化的全局收敛性保证。一个显式的三块扭曲熵结构将耗散从含噪的辅助变量传递到整个状态空间。在耗散性、正则性以及低温泛函不等式假设下，对数冷却调度使目标值以势垒控制的动力学速率依概率收敛到全局最小值。对于精确力积分和中点三阶段离散化格式，多项式递减的步长可在物理时间尺度上保持该收敛速率。三次局部端点估计给出了比现有冻结力动力学结果更宽松的充分步长条件。与单梯度UBU积分器的比较表明，在相同的强耦合分析框架下，其中心化随机局部误差会带来更小的充分（步长条件要求）。

    arXiv:2609.28611v1 Announce Type: cross  Abstract: We study global convergence guarantees of third-order Langevin dynamics for non-convex optimization via simulated annealing with fixed friction and decreasing noise. An explicit three-block distorted entropy transfers dissipation from the noisy auxiliary variable to the full state. Under dissipativity, regularity, and low-temperature functional-inequality assumptions, logarithmic cooling drives the objective values to the global minimum in probability at the barrier-controlled kinetic rate. For the exact-force-integral and midpoint three-stage discretizations, polynomially decreasing steps preserve this rate on the physical time scale. The cubic local endpoint estimate gives a less restrictive sufficient step-size condition than the available frozen-force kinetic result. A comparison with the one-gradient UBU integrator shows how its centered stochastic local error leads, under the same strong-coupling analysis, to a smaller sufficient
    
[^63]: DAG ReLU网络路径提升雅可比矩阵的秩与计算

    Rank and computation of the pathlifting Jacobian of a DAG ReLU network

    [https://arxiv.org/abs/2609.18682](https://arxiv.org/abs/2609.18682)

    本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。

    

    本文通过对网络的隐藏节点数量进行归纳，为DAG ReLU网络的路径提升雅可比矩阵的秩提供了一个自包含的证明。实际上，这种归纳是初等的，关键方法在于考虑网络的骨架矩阵（一个编码网络路径的稀疏矩阵），并将其中的一个隐藏神经元的表示转换为输出节点。该证明依赖于一些中间命题，这些命题将路径提升、其雅可比矩阵、网络参数及其骨架矩阵联系起来，除了能够得出路径提升雅可比矩阵秩的结论外，还提供了一种无需反向传播即可计算该矩阵的方法，其计算成本在实践中比常规的反向传播高效得多。本文附带一个Python模块，该模块实现了论文中针对前馈网络的各个命题，并用于实验性地量化计算……

    arXiv:2609.18682v1 Announce Type: cross  Abstract: This paper provides a self-contained proof of the rank of the pathlifting Jacobian of a DAG ReLU network by performing an induction on the network's number of hidden nodes. In fact, the induction is elementary, and the key recipe is to consider the skeleton matrix of the network, a sparse matrix encoding the network paths, and transform the representation of one of its hidden neurons into an output node. The proof relies on intermediate propositions which link the pathlifting, its Jacobian, the network parameters, and its skeleton matrix, which, on top of permitting to conclude on the rank of the pathlifting Jacobian, also provide a way to compute it without backpropagation and whose computation cost is super efficient in practice compare to usual backpropagation. The paper is provided with a Python module that implements the different propositions of the paper for feed forward networks and is used to experimentally quantifies the comp
    
[^64]: 一种面向可信交通事故严重程度预测的无分布假设认证框架

    A distribution-free certification framework for trustworthy crash-severity prediction

    [https://arxiv.org/abs/2609.11592](https://arxiv.org/abs/2609.11592)

    该论文提出了一个无需分布假设的认证框架，可无需修改地包裹任何交通事故严重程度预测模型，利用KABCO序数标签的结构特性，提供序数预测集、逐类有效性、向未观测真实严重程度的覆盖率传递以及部署偏移下的单侧保证。

    

    交通事故严重程度模型为伤情筛查、调度和路段优先级排序提供依据，然而在部署时却缺乏对单次预测含义的有限样本保证声明。现成的保证方法在此失效，因为交通事故严重程度的独特特征使其难以适用：KABCO结果是序数型的，记录的标签是现场评估，其与真实医学严重程度仅约一半时间一致，且以结构化的方式出错；部署场景跨越了校准时从未见过的不同司法管辖区和年份。我们开发了一个认证层，可以不加修改地包裹任何严重程度模型，并利用上述结构提供无分布假设的保证：连续的序数预测集合，可解读为“B级或更差”；对于任何预先声明的划分实现逐类有效性，并给出预言机效率刻画；通过声明的报告误差区间将覆盖率传递到未观测的真实严重程度，并给出最坏情况下的紧致性结果；部署分布偏移下的单侧证书；以及严重程度加权的（摘要在此处截断）

    arXiv:2609.11592v1 Announce Type: cross  Abstract: Crash-severity models inform screening, dispatch and site prioritization, yet are deployed without a finite-sample statement of what one prediction means. Off-the-shelf guarantees fail here, because the features that make crash severity distinctive defeat them: the KABCO outcome is ordinal, the recorded label is a field assessment agreeing with medical severity about half the time, erring in a structured way, and deployment crosses jurisdictions and years calibration never saw. We develop a certification layer that wraps any severity model unmodified, with distribution-free guarantees using this structure: contiguous ordinal sets that read as "B or worse"; per-class validity for any pre-declared partition, with an oracle efficiency characterization; transfer of coverage to unobserved true severity through a declared reporting band, with a worst-case sharpness result; a one-sided certificate under deployment shift; and severity-weighted
    
[^65]: 张量列车弱形式SINDy：识别高维非线性动力学

    Tensor-Train Weak SINDy: Identifying High-Dimensional Nonlinear Dynamics

    [https://arxiv.org/abs/2609.09434](https://arxiv.org/abs/2609.09434)

    本文提出TT-WSINDy方法，通过张量列车格式结合MANDy和WSINDy技术，实现对指数增长的候选函数空间的高效搜索，从而避免维数灾难并完成高维非线性动力学的数据驱动识别。

    

    近年来，弱形式方法在数据驱动的动力系统发现领域取得了重大进展。然而，在高维设置下，现有技术在计算和内存方面的代价可能十分高昂。在这项工作中，我们提出了TT-WSINDy方法，该方法结合了非线性动力学多维近似（MANDy）和非线性动力学弱稀疏识别（WSINDy）两种方法的技术，并以张量列车（TT）格式实现所需的计算。我们证明了该方法能够在指数增长的候选函数空间中进行搜索——执行弱形式变换、回归和稀疏化——而不会遭受维数灾难的影响。

    arXiv:2609.09434v1 Announce Type: cross  Abstract: In recent years, weak-form methods have made significant advances in data-driven discovery of dynamical systems. However, in high-dimensional settings, current techniques can prove expensive in both computation and memory. In this work, we introduce TT-WSINDy, which combines techniques of the Multidimensional Approximation of Nonlinear Dynamics (MANDy) and Weak Sparse Identification of Nonlinear Dynamics (WSINDy) methods, implementing requisite computations in the tensor-train (TT) format. We demonstrate that this method is able to search an exponentially-growing space of candidate functions -- performing weak-form transformation, regression, and sparsification -- without suffering from the curse of dimensionality.
    
[^66]: 不确定性下的语义承诺可信大语言模型

    Credal Large Language Models for Semantic Commitment under Uncertainty

    [https://arxiv.org/abs/2608.23244](https://arxiv.org/abs/2608.23244)

    通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。

    

    大型语言模型（LLMs）通常会产生流畅但错误的答案，并带有过度的自信。一个核心限制是，标准LLMs通过单一预测分布表示不确定性，将认知上的无知与真正的模糊性混为一谈。我们引入了可信大语言模型（CLLMs）：通过一组LoRA适配器的集成诱导出一个可信集，其下界和上界概率暴露了合理预测分布的扩散范围，而不是坍缩为单一的softmax输出。从这一表示中，我们推导出两个互补的承诺分数。可信令牌承诺（CTC）是一个令牌空间分数，结合了下界支持、可信宽度和交集熵，无需额外生成即可计算。语义承诺一致性（SCC）通过采样补全将承诺扩展到语义空间，其中SCC-Gap衡量令牌级和语义级支持之间的不匹配。我们评估了幻觉情况。

    arXiv:2608.23244v1 Announce Type: cross  Abstract: Large language models (LLMs) often produce fluent but incorrect answers with unwarranted confidence. A central limitation is that standard LLMs represent uncertainty through a single predictive distribution, conflating epistemic ignorance with genuine ambiguity. We introduce Credal Large Language Models (CLLMs): an ensemble of LoRA adapters induces a credal set whose lower and upper probabilities expose the spread of plausible predictive distributions rather than collapsing to a single softmax output. From this representation we derive two complementary commitment scores. Credal Token Commitment (CTC) is a token-space score that combines lower-bound support, credal width, and intersection entropy, computed without additional generation. Semantic Commitment Consistency (SCC) extends commitment to semantic space using sampled completions, with SCC-Gap measuring the mismatch between token-level and semantic-level support. We evaluate hall
    
[^67]: 超越设计效应：聚类下阈值的有效样本量

    The Exceedance Design Effect: Effective Sample Size for Thresholds under Clustering

    [https://arxiv.org/abs/2608.21262](https://arxiv.org/abs/2608.21262)

    本文提出在聚类相关数据下，设置阈值时需采用不同于平均值的有效样本量计算方法，以准确评估阈值的可靠性。

    

    arXiv:2608.21262v1 公告类型：交叉 摘要：许多机器学习系统在校准集的分位数处设置阈值：例如，共形预测器通过将截止点设在校准集的第90百分位数来承诺90%的覆盖率；弃权门在模型得分低于校准集第10百分位数时拒绝回答；安全过滤器阻止任何得分超过参考集第99百分位数的输出。所有这些系统都承诺阈值在新数据上以规定比率保持。该承诺假设校准样本是独立的，但在现代流程中通常并非如此：它们共享提示、文档或推理轨迹。调查统计学自1965年以来就知道如何对相关数据进行折扣处理，通过计算样本相当于多少个独立观测值，但这仅适用于平均值。我们表明，阈值需要不同的计数方式。该计数取决于聚类得分落在阈值同一侧的频率，而这进一步影响...

    arXiv:2608.21262v1 Announce Type: cross  Abstract: Many machine-learning systems set a threshold at a quantile of a calibration set: conformal predictors that promise 90% coverage by drawing their cutoff at the calibration set's 90th percentile, abstention gates that decline to answer when a model's score falls below the calibration set's tenth percentile, safety filters that block any output scoring above the 99th percentile of a reference set. All of them promise that the threshold will hold at the stated rate on new data. The promise assumes the calibration examples are independent, and in modern pipelines they usually are not: they share a prompt, a document, a reasoning trace. Survey statistics has known how to discount correlated data since 1965, by counting how many independent observations a sample is worth, but only for averages. We show that a threshold needs a different count. The count depends on how often clustered scores land on the same side of the threshold, and that ch
    
[^68]: ManifoldFlow：具有可学习奇异谱的SPD松弛Stiefel层

    ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

    [https://arxiv.org/abs/2607.04535](https://arxiv.org/abs/2607.04535)

    ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。

    

    正交层和Stiefel层为神经网络权重提供了精确的谱控制，但它们也带来了很强的建模约束：所有被表示的奇异值都被固定为1。许多受益于正交归一基的场景仍然需要方向相关的衰减或放大。我们提出了ManifoldFlow，这是对固定谱Stiefel层的一种最小松弛方法，它在保持基向量位于Stiefel流形上的同时，通过 W = Q S^{1/2}（其中 Q^T Q = I 且 S 为正定矩阵）学习一个有界的正谱。由于 W^T W = S，S 的特征值恰好是所实现权重的奇异值的平方，这使得特征值裁剪成为一种直接的奇异值控制机制。在配对的序列、表格和图像实验中，可学习的SPD谱在Stiefel先验有效的报告设置中优于固定谱的Stiefel对应方法，其中在循环语言模型投影任务中增益最大（原文摘要在此处截断）。

    arXiv:2607.04535v2 Announce Type: replace-cross  Abstract: Orthogonal and Stiefel layers give neural weights exact spectral control, but they also impose a strong modeling constraint: all represented singular values are fixed at one. Many settings that benefit from an orthonormal basis still need direction-dependent attenuation or amplification. We introduce ManifoldFlow, a minimal relaxation of a fixed-spectrum Stiefel layer that keeps the basis on the Stiefel manifold while learning a bounded positive spectrum through W = Q S^{1/2}, with Q^T Q = I and S positive definite. Since W^T W = S, the eigenvalues of S are exactly the squared singular values of the realized weight, making eigenvalue clipping a direct singular-value control mechanism. Across paired sequence, tabular, and image experiments, the learnable SPD spectrum improves the fixed-spectrum Stiefel counterpart in the reported settings where the Stiefel prior is useful, with the largest gains in recurrent language-model proje
    
[^69]: 解耦连续时间潜在动力学：通过扩散系数变化实现潜在随机微分方程的可辨识性

    Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts

    [https://arxiv.org/abs/2606.28228](https://arxiv.org/abs/2606.28228)

    该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。

    

    面向时间序列的因果表征学习在离散时间潜在因果模型中已建立了较强的可辨识性结果，但连续时间潜在随机微分方程（SDE）模型中的可辨识性问题在很大程度上仍悬而未决。我们利用环境引起的扩散协方差变化来填补这一空白。我们研究了通过未知非线性微分同胚观测到的加性噪声潜在SDE，其漂移项在不同环境间共享，而扩散协方差则因环境而异。我们证明：两个具有逐坐标成对不同方差比的对角扩散机制，可以在无需对漂移项施加任何稀疏性假设的条件下，将潜在坐标识别至置换、逐坐标缩放以及可能的常数平移的等价类内。我们首先针对线性Ornstein-Uhlenbeck系统证明了这一结果，随后将其推广至一般的加性噪声潜在SDE。在温和的光滑性条件下，瞬时漂移-雅可比因果图是可辨识的……

    arXiv:2606.28228v2 Announce Type: replace-cross  Abstract: Causal representation learning for time series has developed strong identifiability results in discrete-time latent causal models, but identifiability in continuous-time latent stochastic differential equation (SDE) models remains largely open. We address this gap using environment-induced shifts in diffusion covariance. We study additive-noise latent SDEs observed through an unknown nonlinear diffeomorphism, with shared drift but environment-specific diffusion covariance. We show that two diagonal diffusion regimes with pairwise distinct coordinate-wise variance ratios identify the latent coordinates up to permutation, coordinate-wise scaling, and a possible constant shift, without any sparsity assumption on the drift. We first prove this result for linear Ornstein-Uhlenbeck systems and then extend it to general additive-noise latent SDEs. Under mild smoothness, the instantaneous drift-Jacobian causal graph is identifiable up 
    
[^70]: INDEQS：先验信息引导的神经控制微分方程

    INDEQS: Informed Neural controlled Differential EQuationS

    [https://arxiv.org/abs/2606.19138](https://arxiv.org/abs/2606.19138)

    该论文提出INDEQS方法，将预先已知的有向图结构以不同架构位置融入基于图的神经控制微分方程（NCDE）时间序列预测模型，通过分离节点间隐藏状态内层混合与向量场-控制外层混合，并提供轻量级图约束变体和基于自适应图卷积的更具表现力的变体来有效利用图先验知识。

    

    神经控制微分方程（NCDE）为时间序列预测提供了一个强大的连续时间框架，但标准的基于图的扩展通常纯粹从数据中学习空间结构，即使在有向图结构可以预先得知的情况下也是如此。我们提出了先验信息引导的神经控制微分方程（INDEQS），这是对基于图的NCDE预测方法的一种改进，它在不同的架构位置融入了有向图的先验知识。INDEQS将图节点间隐藏状态的内层混合与向量场和控制信号之间的外层混合分离开来，并提供了一个轻量级的图约束变体和一个更具表现力的变体，后者通过自适应图卷积从数据中学习额外的图连接。为了系统地研究图的先验信息在何时对预测有益，我们设计了一个有向图上的连续平流模拟，生成合成的……（原文摘要在此处截断）

    arXiv:2606.19138v2 Announce Type: replace-cross  Abstract: Neural Controlled Differential Equations (NCDE) provide a powerful continuous-time framework for forecasting time series, but standard graph-based extensions typically learn spatial structure purely from data, even in settings where a directed graph structure is known a priori. We introduce Informed Neural controlled Differential EQuationS (INDEQS), a modification to graph-based NCDE forecasting methods that incorporates prior knowledge of a directed graph at distinct architectural positions. INDEQS separates inner mixing of hidden states across graph nodes from outer mixing between vector field and control, and offers both a lightweight graph-constrained variant and a more expressive variant, learning additional graph connections from data via adaptive graph convolutions. To systematically study when graph informedness is beneficial in forecasting, we devise a continuous advection simulation on directed graphs, yielding synthe
    
[^71]: 面向函数空间变分推断的流变换隐过程

    Flow-Transformed Implicit Processes for Function-Space Variational Inference

    [https://arxiv.org/abs/2606.01954](https://arxiv.org/abs/2606.01954)

    提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。

    

    隐过程先验通过灵活的生成机制来定义函数上的分布，这使其在贝叶斯函数空间建模中颇具吸引力。然而，使用此类先验进行后验推断具有挑战性，因为其诱导的函数空间分布通常不具备闭式形式。一种实用的策略是使用有限个采样函数的集合来近似先验，然后将后验函数表示为这些样本的学习组合。现有方法通常在组合权重上放置高斯变分分布。尽管这种方法易于处理，但它限制了所能表示的后验不确定性的形状，尤其是当真实后验呈现非对称、重尾或多峰特性时。我们提出了流变换隐过程（FTIP），这是一种变分推断方法，使这种有限维函数空间近似更加灵活。

    arXiv:2606.01954v2 Announce Type: replace-cross  Abstract: Implicit-process priors define distributions over functions through flexible generative mechanisms, making them attractive for Bayesian function-space modelling. However, performing posterior inference with such priors is challenging because their induced function-space distributions are typically not available in closed form. One practical strategy is to approximate the prior using a finite collection of sampled functions, and then represent posterior functions as learned combinations of these samples. Existing approaches commonly place a Gaussian variational distribution over the combination weights. While tractable, this choice limits the shapes of posterior uncertainty that can be represented, especially when the true posterior is asymmetric, heavy-tailed, or multimodal. We propose Flow-Transformed Implicit Processes (FTIP), a variational inference method that makes this finite-dimensional function-space approximation more 
    
[^72]: CASCADE共形预测：面向两阶段临床决策支持的不确定性自适应预测区间

    CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support

    [https://arxiv.org/abs/2605.20468](https://arxiv.org/abs/2605.20468)

    提出了CASCADE共形预测框架，通过将筛查分类器的认知不确定性传播到下游回归任务中，动态缩放帕金森病药物剂量预测的预测区间，为两阶段临床决策支持提供不确定性自适应的可靠性量化。

    

    帕金森病（PD）的有效药物管理面临挑战，原因在于疾病进展的异质性、患者反应的差异以及药物副作用。虽然人工智能模型可以预测左旋多巴等效日剂量（LEDD）作为药物需求的衡量指标，但标准的不确定性量化方法往往无法传达这些预测的可靠性，对高置信度和低置信度的临床决策一视同仁地处理。我们提出了CASCADE（通过共形和分布估计的校准自适应缩放），这是一种新颖的共形预测框架，它将来自筛查分类器的认知不确定性传播到下游预测中，以自适应地调整预测。与依赖辅助残差回归的标准共形方法不同，我们利用来自主要分类任务（识别是否需要改变药物）的认知不确定性，来动态缩放次要回归任务的预测区间……

    arXiv:2605.20468v3 Announce Type: replace-cross  Abstract: Effective medication management in Parkinson's Disease (PD) is challenging due to heterogeneous disease progression, variable patient response, and medication side effects. While AI models can forecast levodopa equivalent daily dose (LEDD) as a measure of medication needs, standard uncertainty quantification often fails to communicate the reliability of these predictions, treating high and low confidence clinical decisions identically. We introduce CASCADE (Calibrated Adaptive Scaling via Conformal And Distributional Estimation), a novel conformal prediction framework that propagates epistemic uncertainty from a screening classifier to adapt downstream predictions. Unlike standard conformal methods that rely on auxiliary residual regression, we leverage epistemic uncertainty from a primary classification task (identifying whether a medication change is needed) to dynamically scale the prediction intervals of a secondary regress
    
[^73]: 基于组合满意化老虎机的多用户毫米波波束与速率自适应

    Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits

    [https://arxiv.org/abs/2604.14908](https://arxiv.org/abs/2604.14908)

    本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。

    

    我们研究了多用户毫米波MISO系统中的下行链路波束与速率自适应问题，其中多个基站（BS）各自使用有限码本中的模拟波束赋形，以每个用户设备（UE）唯一的波束和离散的数据传输速率来服务多个单天线用户设备。基站基于ACK/NACK反馈学习传输是否成功。为了刻画服务目标，我们引入了满意化吞吐量阈值 $\tau_r$，并将波束与速率的联合自适应建模为波束-速率元组上的组合半老虎机问题。在该框架下，我们提出了SAT-CTS，这是一种轻量级、阈值感知的策略，它将保守的置信估计与后验采样相结合，引导学习朝着满足 $\tau_r$ 的方向发展，而非仅仅追求最大化。我们的主要理论贡献在于为具有满意化目标的组合半老虎机提供了首个有限时间遗憾界：当 $\tau_r$ 可实现时，我们对累积遗憾（摘要在此处截断）……

    arXiv:2604.14908v2 Announce Type: replace-cross  Abstract: We study downlink beam and rate adaptation in a multi-user mmWave MISO system where multiple base stations (BSs), each using analog beamforming from finite codebooks, serve multiple single-antenna user equipments (UEs) with a unique beam per UE and discrete data transmission rates. BSs learn about transmission success based on ACK/NACK feedback. To encode service goals, we introduce a satisficing throughput threshold $\tau_r$ and cast joint beam and rate adaptation as a combinatorial semi-bandit over beam-rate tuples. Within this framework, we propose SAT-CTS, a lightweight, threshold-aware policy that blends conservative confidence estimates with posterior sampling, steering learning toward meeting $\tau_r$ rather than merely maximizing. Our main theoretical contribution provides the first finite-time regret bounds for combinatorial semi-bandits with satisficing objective: when $\tau_r$ is realizable, we upper bound the cumula
    
[^74]: 线性系统辨识中的最优中心化主动激励

    Optimal Centered Active Excitation in Linear System Identification

    [https://arxiv.org/abs/2604.05518](https://arxiv.org/abs/2604.05518)

    该论文提出了一种基于普通最小二乘法和半定规划的线性系统辨识主动学习算法，通过最优中心化噪声激励实现了理论上最优的样本复杂度，其上界与任意算法的下界在常数因子内匹配，并明确了样本复杂度对状态维度等系统参数的依赖关系。

    

    我们提出了一种用于线性系统辨识的主动学习算法，该算法采用最优的中心化噪声激励。值得注意的是，我们的算法基于普通最小二乘法和半定规划，在达到最小样本复杂度的同时，还能高效地计算系统矩阵的估计。更具体地说，我们首先为任意主动学习算法达到给定精度和置信水平所需的样本复杂度建立了下界。接着，我们推导出所提算法的样本复杂度上界，该上界在常数因子范围内与任意算法的下界相匹配。我们得到的紧致界易于解释，并明确展示了其对状态维度等系统参数的依赖性。

    arXiv:2604.05518v2 Announce Type: replace-cross  Abstract: We propose an active learning algorithm for linear system identification with optimal centered noise excitation. Notably, our algorithm, based on ordinary least squares and semidefinite programming, attains the minimal sample complexity while allowing for efficient computation of an estimate of a system matrix. More specifically, we first establish lower bounds of the sample complexity for any active learning algorithm to attain the prescribed accuracy and confidence levels. Next, we derive a sample complexity upper bound of the proposed algorithm, which matches the lower bound for any algorithm up to universal factors. Our tight bounds are easy to interpret and explicitly show their dependence on the system parameters such as the state dimension.
    
[^75]: 针对样本级与单元格级异常值的稳健张量对张量回归

    Casewise and Cellwise Robust Tensor-on-Tensor Regression

    [https://arxiv.org/abs/2603.25911](https://arxiv.org/abs/2603.25911)

    本文提出了一种名为ROTOT的稳健张量对张量回归新方法，能够同时应对样本级与单元格级异常值并处理缺失值。

    

    张量对张量回归是分析张量数据的重要工具，旨在从一组对应的预测张量中预测一组响应张量。然而，标准的张量对张量回归对异常值十分敏感，这些异常值可能同时存在于响应和预测变量中。回归结果可能受到样本级异常值（即偏离数据主体的观测值）的影响，也可能受到单元格级异常值（即张量中个别异常单元格）的影响。由于张量数据通常包含大量单元格，后者尤为常见。本文提出了一种新颖的稳健张量对张量回归方法，称为ROTOT，该方法能够同时处理这两种类型的异常值，并且还可以应对缺失值。该方法使用单一损失函数来降低响应中样本级和单元格级异常值的影响。预测变量中的异常值则通过……（原文此处被截断）

    arXiv:2603.25911v2 Announce Type: replace-cross  Abstract: Tensor-on-tensor regression is an important tool for the analysis of tensor data, aiming to predict a set of response tensors from a corresponding set of predictor tensors. However, standard tensor-on-tensor regression is sensitive to outliers, which may be present in both the response and the predictor. It can be affected by casewise outliers, which are observations that deviate from the bulk of the data, as well as by cellwise outliers, which are individual anomalous cells within the tensors. The latter are particularly common due to the typically large number of cells in tensor data. This paper introduces a novel robust tensor-on-tensor regression method, named ROTOT, that can handle both types of outliers simultaneously, and can cope with missing values as well. This method uses a single loss function to reduce the influence of both casewise and cellwise outliers in the response. The outliers in the predictor are handled us
    
[^76]: 关于Forré的条件独立性概念与连续变量因果演算的札记

    Notes on Forr\'e's Notion of Conditional Independence and Causal Calculus for Continuous Variables

    [https://arxiv.org/abs/2603.24333](https://arxiv.org/abs/2603.24333)

    本札记进一步阐释了Forré的转移条件独立性框架的动机与文献联系，揭示了测度论因果演算中的微妙之处，并将ID算法的“单行”表述推广到一般测度论设定。

    

    近日，Forré（arXiv:2104.11547，2021）提出了“转移条件独立性”的概念，这是一种为随机变量和非随机变量提供统一框架的条件独立性概念。其原始论文建立了一个强全局马尔可夫性质，将转移条件独立性与带输入节点的有向混合图（iDMG）的相应图分离准则联系起来，并在一般测度论设定下给出了适用于iDMG的因果演算的一个版本。本札记旨在进一步阐明该框架背后的动机及其与相关文献的联系，指出一般测度论因果演算中的若干微妙之处，并将Richardson等人（Ann. Statist. 51(1):334--361，2023）的ID算法的“单行”表述推广到一般测度论设定。

    arXiv:2603.24333v2 Announce Type: replace-cross  Abstract: Recently, Forr\'e (arXiv:2104.11547, 2021) introduced transitional conditional independence, a notion of conditional independence that provides a unified framework for both random and non-stochastic variables. The original paper establishes a strong global Markov property connecting transitional conditional independencies with suitable graphical separation criteria for directed mixed graphs with input nodes (iDMGs), together with a version of causal calculus for iDMGs in a general measure-theoretic setting. These notes aim to further illustrate the motivations behind this framework and its connections to the literature, highlight certain subtlies in the general measure-theoretic causal calculus, and extend the "one-line" formulation of the ID algorithm of Richardson et al. (Ann. Statist. 51(1):334--361, 2023) to the general measure-theoretic setting.
    
[^77]: 用指数加权签名扩展状态空间模型

    Extending SSMs with the Exponentially Weighted Signature

    [https://arxiv.org/abs/2603.19198](https://arxiv.org/abs/2603.19198)

    该论文提出指数加权签名（EWS），通过可学习矩阵生成元、将步长推广为输入因果泛函的时钟以及更高截断深度，将状态空间模型（含Mamba）统一并扩展为签名理论下的连续时间模型，在长时间序列分类任务上取得了最优表现。

    

    我们提出了指数加权签名（EWS），这是一种连续时间模型，用于计算路径的迭代积分，其中每个增量由一个可学习生成元在流逝时钟时间上的矩阵指数进行加权。我们证明它求解一个线性受控微分方程，保持了签名的群状结构与普适性，并满足一个修正的Chen恒等式，从而可以采用并行扫描。在深度为1时，EWS即为一个状态空间模型（SSM），我们以闭式形式将线性时不变SSM、Mamba通道以及Mamba-2头映射到该模型中。EWS通过任意矩阵生成元、将步长推广为输入因果泛函的时钟，以及单层内对路径呈非线性的更高截断深度来扩展SSM。实验表明，EWS在六个长时间序列分类数据集上取得了最高的平均准确率与排名，且增加深度通常能带来提升。

    arXiv:2603.19198v3 Announce Type: replace  Abstract: We introduce the exponentially weighted signature (EWS), a continuous-time model that computes iterated integrals of a path, where each increment is weighted by the matrix exponential of a learnable generator over elapsed clock time. We prove that it solves a linear controlled differential equation, keeps the group-like structure and the universality of the signature, and satisfies a modified Chen identity, enabling a parallel scan. At depth one the EWS is a state-space model (SSM), and we map linear time-invariant SSMs, Mamba channels and Mamba-$2$ heads to it in closed form. The EWS extends SSMs through an arbitrary matrix generator, a clock that generalises the step size to causal functionals of the input, and higher truncation depths that are non-linear in the path within a single layer. Empirically, the EWS achieves the highest average accuracy and rank on six long time-series classification datasets, where depth generally helps
    
[^78]: 学习量子系综中的判别力与复杂度层级

    Hierarchy of discriminative power and complexity in learning quantum ensembles

    [https://arxiv.org/abs/2601.22005](https://arxiv.org/abs/2601.22005)

    本文提出了量子系综距离度量的层级结构MMD-$k$，揭示了判别力与统计效率之间的严格权衡——估计MMD-$k$需要$\Theta(N^{1-1/k})$个样本，而任何具有完全判别能力的稳定距离度量至少需要$\Omega(N)$个样本。

    

    距离度量是机器学习的核心，然而由于量子测量的基本约束，量子态系综之间的距离仍然鲜为人知。我们引入了一类积分概率度量的层级结构，称为MMD-$k$，它将最大均值差异（maximum mean discrepancy）推广到量子系综，并且随着矩阶数$k$的增加，展现出判别力与统计效率之间的严格权衡。对于规模为$N$的纯态系综，在常数$k$下，使用任意测量方案估计MMD-$k$需要$\Theta(N^{1-1/k})$个样本。与此同时，我们证明任何具有完全判别能力的稳定距离度量都存在$O(N\log N)$的样本复杂度上界以及$\Omega(N)$的实例间隔下界。对于量子Wasserstein距离，在状态维度足够大且固定的情况下，我们在恒定加性精度下建立了关于系综规模的近线性下界，并……

    arXiv:2601.22005v2 Announce Type: replace-cross  Abstract: Distance metrics are central to machine learning, yet distances between ensembles of quantum states remain poorly understood due to fundamental quantum measurement constraints. We introduce a hierarchy of integral probability metrics, termed MMD-$k$, which generalizes the maximum mean discrepancy to quantum ensembles and exhibits a strict trade-off between discriminative power and statistical efficiency as the moment order $k$ increases. For pure-state ensembles of size $N$, estimating MMD-$k$ with arbitrary measurement schemes requires $\Theta(N^{1-1/k})$ samples for constant $k$. At the same time, we prove that any stable distance metric with full discriminative power admits an $O(N\log N)$ upper bound and an $\Omega(N)$ instance-gap lower bound. For quantum Wasserstein distance, with sufficiently large fixed state dimension, we establish a nearly linear lower bound in the ensemble size at constant additive accuracy, together
    
[^79]: 多边际最优传输的统一Kantorovich对偶理论

    A Unified Kantorovich Duality for Multimarginal Optimal Transport

    [https://arxiv.org/abs/2601.17171](https://arxiv.org/abs/2601.17171)

    该论文建立了多边际最优传输的统一Kantorovich对偶理论，在紧与非紧（波兰空间）情形下均证明了对偶最优解的存在性，并刻画了最优对偶势的结构。

    

    我们研究具有有界连续成本函数的多边际最优传输（MOT）的Kantorovich对偶。主要关注点不仅在于原始问题与对偶问题取值的相等性，还在于最优对偶势的结构。对于紧度量空间，我们证明对偶问题在相互 $c$-共轭族类中存在最优解。证明结合了Fenchel--Rockafellar对偶论证、等度连续性估计、对偶势的规范化处理以及Arzelà--Ascoli定理。随后我们考虑非紧情形，即边际空间为波兰空间的情况。我们首先通过截断与紧性论证恢复了Kantorovich对偶恒等式，其中利用了成本函数的有界性和Prokhorov紧性。对于对偶最优解的存在性，我们依赖于最优支撑集的几何性质。在自然的支撑分裂条件下，我们在规范类中获得了一个有界的Borel可测对偶最优解。

    arXiv:2601.17171v2 Announce Type: replace-cross  Abstract: We study Kantorovich duality for multimarginal optimal transport (MOT) with bounded continuous cost functions. The main focus is not only the equality between the primal and dual values, but also the structure of optimal dual potentials. For compact metric spaces, we prove that the dual problem admits an optimizer in the class of mutually $c$-conjugate families. The proof combines a Fenchel--Rockafellar duality argument with equicontinuity estimates, a normalization of the dual potentials, and the Arzel\`a--Ascoli theorem. We then consider the non-compact case, where the marginal spaces are Polish. We first recover the Kantorovich duality identity by a truncation and tightness argument, using the boundedness of the cost and Prokhorov compactness. For dual attainment, we rely on the geometry of optimal supports. Under a natural support-splitting condition, we obtain a bounded Borel measurable dual optimizer in the canonical clas
    
[^80]: BalLOT：基于最优传输的平衡 $k$-means 聚类

    BalLOT: Balanced $k$-means clustering with optimal transport

    [https://arxiv.org/abs/2512.05926](https://arxiv.org/abs/2512.05926)

    提出 BalLOT 算法，将最优传输融入交替最小化框架以求解平衡 $k$-means 聚类，并从理论上证明了其整值耦合性质与植入聚类恢复保证，数值实验验证了其快速有效性。

    

    我们研究平衡 $k$-means 聚类这一基本问题。特别地，我们提出了一种名为 BalLOT 的方法，它将最优传输引入交替最小化框架，并证明该方法能为该问题提供快速而有效的解决方案。我们通过若干理论保证和多种数值实验来确立这一点。在理论方面，我们首先证明对于一般数据，BalLOT 在每一步都会产生整值耦合。接着，我们进行了损失景观分析，为随机球模型下植入聚类的精确恢复和部分恢复提供理论保证。我们还提出了能够在一步内实现植入聚类恢复的初始化方案。最后，我们给出了验证上述理论结果的数值实验。

    arXiv:2512.05926v2 Announce Type: replace  Abstract: We consider the fundamental problem of balanced $k$-means clustering. In particular, we introduce an optimal transport approach to alternating minimization called BalLOT, and we show that it delivers a fast and effective solution to this problem. We establish this with several theoretical guarantees and a variety of numerical experiments. On the theory front, we first prove that for generic data, BalLOT produces integral couplings at each step. Next, we perform a landscape analysis to provide theoretical guarantees for both exact and partial recoveries of planted clusters under the stochastic ball model. We also propose initialization schemes that achieve one-step recovery of planted clusters. To conclude, we present numerical experiments that corroborate our theoretical results.
    
[^81]: 深度特征选择的可证FDR控制：深度多层感知机及更广泛的架构

    Provable FDR Control for Deep Feature Selection: Deep MLPs and Beyond

    [https://arxiv.org/abs/2512.04696](https://arxiv.org/abs/2512.04696)

    首个在通用深度学习设置下为特征选择提供错误发现率（FDR）控制理论保证的框架，可覆盖多层感知机、卷积/循环网络、注意力机制等广泛架构。

    

    我们开发了一个基于深度神经网络的灵活特征选择框架，该框架能够近似控制错误发现率（FDR），即一类I型错误的度量。该方法适用于第一层为全连接层的网络架构。从第二层开始，它可兼容任意宽度和深度的多层感知机（MLP）、卷积神经网络与循环神经网络、注意力机制、残差连接以及dropout。该流程还兼容采用与数据无关的初始化和学习率的随机梯度下降。据我们所知，这是首个在如此通用的深度学习设置下为特征选择提供FDR控制理论保证的工作。我们的分析建立在多指标（multi-index）数据生成模型以及一个渐近机制之上，其中特征维度 $n$ 的发散速度快于潜在维度 $q^{*}$，同时样本量、训练迭代次数……（摘要在此处被截断）

    arXiv:2512.04696v3 Announce Type: replace  Abstract: We develop a flexible feature selection framework based on deep neural networks that approximately controls the false discovery rate (FDR), a measure of Type-I error. The method applies to architectures whose first layer is fully connected. From the second layer onward, it accommodates multilayer perceptrons (MLPs) of arbitrary width and depth, convolutional and recurrent networks, attention mechanisms, residual connections, and dropout. The procedure also accommodates stochastic gradient descent with data-independent initializations and learning rates. To the best of our knowledge, this is the first work to provide a theoretical guarantee of FDR control for feature selection within such a general deep learning setting.   Our analysis is built upon a multi-index data-generating model and an asymptotic regime in which the feature dimension $n$ diverges faster than the latent dimension $q^{*}$, while the sample size, the number of trai
    
[^82]: SSLfmm：一个用于混合缺失机制半监督学习的R包

    SSLfmm: An R Package for Semi-Supervised Learning with Mixed Missingness

    [https://arxiv.org/abs/2512.03322](https://arxiv.org/abs/2512.03322)

    SSLfmm是一个R包，通过联合建模标签缺失机制与类别分布，支持混合缺失机制下的半监督学习，并提供了统一的R接口。

    

    arXiv:2512.03322v3 公告类型：替换-交叉 摘要：当所有数据的特征都被观测到，但仅有部分样本的类别标签可用时，就会出现部分标记样本。在这种情况下，控制标签可用性的机制本身可能包含与分类相关的信息，但在标准的半监督学习流程中通常未被建模。SSLfmm包实现了基于似然的高斯有限混合分类，其中标签缺失过程与类别分布联合建模。它支持完全案例、完全随机缺失（MCAR）、基于熵的随机缺失（MAR）以及混合MCAR/MAR分析。对于混合机制，缺失标签的来源可能是可观测的或潜在的，允许同一建模框架适应关于标签可用性的不同形式信息。该包提供了一个统一的R接口，用于模型拟合、预测、性能评估、模拟和基于熵的分析。

    arXiv:2512.03322v3 Announce Type: replace-cross  Abstract: Partially labelled samples arise when features are observed for all data, but class labels are available for only a subset. In such settings, the mechanism governing label availability may itself contain information relevant to classification, yet it is typically left unmodelled in standard semi-supervised learning procedures. The SSLfmm package implements likelihood-based Gaussian finite-mixture classification in which the label-missingness process is modelled jointly with the class distribution. It supports complete-case, missing completely at random (MCAR), entropy-based missing at random (MAR), and mixed MCAR/MAR analyses. For the mixed mechanism, the source of a missing label may be observed or latent, allowing the same modelling framework to accommodate different forms of information about label availability. A common R interface is provided for model fitting, prediction, performance assessment, simulation, and entropy-ba
    
[^83]: R、Python、Julia 和 C++ 中的高效 SLOPE 求解器

    Efficient Solvers for SLOPE in R, Python, Julia, and C++

    [https://arxiv.org/abs/2511.02430](https://arxiv.org/abs/2511.02430)

    该论文提出了 R、Python、Julia 和 C++ 中高效求解 SLOPE 问题的软件包套件，采用高效的混合坐标下降算法支持多种损失函数和数据结构，并在速度上超越了现有的 SLOPE 实现。

    

    我们提出了一套在 R、Python、Julia 和 C++ 中的软件包，用于高效求解排序 L1 正则化估计问题。这些软件包采用高效的混合坐标下降算法，可以拟合广义线性模型（GLM），并支持多种损失函数，包括高斯损失、二项损失、泊松损失和多项逻辑回归损失。我们的实现设计追求快速、节省内存且灵活。这些软件包支持多种数据结构（稠密矩阵、稀疏矩阵和内存外矩阵），能够高效地拟合完整的 SLOPE 路径，并处理 SLOPE 模型的交叉验证，包括松弛 SLOPE（relaxed SLOPE）。我们展示了如何使用这些软件包的示例，并通过基准测试证明了这些软件包在真实数据和模拟数据上的性能，结果表明我们的软件包在速度方面优于现有的 SLOPE 实现。

    arXiv:2511.02430v4 Announce Type: replace-cross  Abstract: We present a suite of packages in R, Python, Julia, and C++ that efficiently solve the Sorted L-One Penalized Estimation (SLOPE) problem. The packages feature a highly efficient hybrid coordinate descent algorithm that fits generalized linear models (GLMs) and supports a variety of loss functions, including Gaussian, binomial, Poisson, and multinomial logistic regression. Our implementation is designed to be fast, memory-efficient, and flexible. The packages support a variety of data structures (dense, sparse, and out-of-memory matrices) and are designed to efficiently fit the full SLOPE path as well as handle cross-validation of SLOPE models, including the relaxed SLOPE. We present examples of how to use the packages and benchmarks that demonstrate the performance of the packages on both real and simulated data and show that our packages outperform existing implementations of SLOPE in terms of speed.
    
[^84]: 基准测试认识论：用于评估机器学习模型的有效性理论

    The Benchmarking Epistemology: Validity Theory for Evaluating Machine Learning Models

    [https://arxiv.org/abs/2510.23191](https://arxiv.org/abs/2510.23191)

    本文借鉴心理学有效性理论，提出使基准测试科学推断所需假设显式化的有效性条件，并通过ImageNet和脆弱家庭挑战赛两个案例，将预测性基准测试确立为机器学习中一种独特的认知实践。

    

    预测性基准测试，即基于预测性能和竞争排名来评估机器学习模型，是机器学习研究和科学探究的核心。然而，基准测试分数充其量只能衡量相对于特定数据集和学习问题的性能。要从中得出实质性的科学推断需要额外的假设。本文借鉴心理学有效性理论的思想，提出了使这些假设显式化的有效性条件。通过两个案例研究——ImageNet和脆弱家庭挑战赛，我们展示了基准测试结果如何能够支持关于研究进展和可预测性极限的推断，从而将预测性基准测试定位为机器学习中一种独特的认知实践。

    arXiv:2510.23191v2 Announce Type: replace-cross  Abstract: Predictive benchmarking, evaluating machine learning models based on predictive performance and competitive ranking, is central to machine learning research and scientific inquiry. However, benchmark scores at best measure performance relative to a specific dataset and learning problem. Drawing substantial scientific inferences requires additional assumptions. Adapting ideas from psychological validity theory, we propose validity conditions that make these assumptions explicit. In two case studies---ImageNet and the Fragile Families Challenge---we show how benchmark results can support inferences about research progress and limits of predictability, situating predictive benchmarking as a distinct epistemic practice in machine learning.
    
[^85]: 基于最小注意力的元强化学习

    Meta-reinforcement learning with minimum attention

    [https://arxiv.org/abs/2505.16741](https://arxiv.org/abs/2505.16741)

    该论文将Brockett的最小注意力（最小作用原理）作为奖励项引入强化学习，与元学习相结合，在高维非线性动力学中显著提升了少样本快速适应能力并降低了扰动方差，且可无缝集成到DreamerV3、MAMBA等现代世界模型中。

    

    最小注意力将最小作用原理应用于控制量随状态和时间的变化，该思想由Brockett首次提出。其涉及的正则化在模拟生物控制（如运动学习）方面具有重要意义。我们将最小注意力作为奖励的一部分引入强化学习（RL），并研究其与元学习和稳定化之间的联系。具体而言，我们在高维非线性动力学中探索了带有最小注意力的基于模型的元学习方法，交替执行基于集成（ensemble）的模型学习和基于梯度的元策略学习。实验结果表明，与无模型和基于模型的RL基线相比，最小注意力提升了少样本下的快速适应能力，并降低了模型与环境扰动带来的方差；此外，将其集成到现代世界模型（DreamerV3、MAMBA）中也能获得一致的增益。最小注意力还在能……方面展现出改进。

    arXiv:2505.16741v5 Announce Type: replace-cross  Abstract: Minimum attention applies the least action principle in changes of control concerning state and time, first proposed by Brockett. The involved regularization is highly relevant in emulating biological control, such as motor learning. We apply minimum attention in reinforcement learning (RL) as part of the rewards and investigate its connection to meta-learning and stabilization. Specifically, model-based meta-learning with minimum attention is explored in high-dimensional nonlinear dynamics. Ensemble-based model learning and gradient-based meta-policy learning are alternately performed. Empirically, minimum attention improves fast adaptation in few shots and reduces variance from perturbations of the model and environment, compared to model-free and model-based RL baseline, and yields consistent gain when integrated into modern world models (DreamerV3, MAMBA). Furthermore, the minimum attention demonstrates an improvement in en
    
[^86]: 面向连续时间fMRI表征学习的随机最优控制

    Stochastic Optimal Control for Continuous-Time fMRI Representation Learning

    [https://arxiv.org/abs/2502.04892](https://arxiv.org/abs/2502.04892)

    该论文提出将自监督学习重构为随机最优控制问题的新框架，把大脑活动建模为连续时间潜在动力学，并统一掩码自编码（MAE）与联合嵌入预测（JEPA），从而学习到对时间不规则性和噪声鲁棒的fMRI表征。

    

    从功能磁共振成像（fMRI）中学习鲁棒的表征，从根本上受到来自异构数据源所固有之时间不规则性和噪声的挑战。现有的自监督学习（SSL）方法通常通过对fMRI信号进行离散化或平均化处理而丢弃关键的时间信息。为解决这一问题，我们引入了一种新颖的框架，将自监督学习重新构建为随机最优控制（SOC）问题。我们的方法将大脑活动建模为连续时间的潜在动力学，通过优化一种对时间不规则性不敏感的控制策略，来学习大脑动力学的鲁棒表征。该SOC框架自然地统一了掩码自编码（MAE）与联合嵌入预测（JEPA），以提取紧凑的、由控制导出的表征。此外，一种无需仿真的推理策略确保了计算效率，并使其可扩展至大规模fMRI数据集。我们的模型……（原文摘要在此处被截断）

    arXiv:2502.04892v2 Announce Type: replace-cross  Abstract: Learning robust representations from functional magnetic resonance imaging (fMRI) is fundamentally challenged by the temporal irregularity and noise inherent in data from heterogeneous sources. Existing self-supervised learning (SSL) methods often discard critical temporal information by discretizing or averaging fMRI signals. To address this, we introduce a novel framework that reframes SSL as a Stochastic Optimal Control (SOC) problem. Our approach models brain activity as continuous-time latent dynamics, learning a robust representation of brain dynamics by optimizing a control policy that is agnostic to the temporal irregularity. This SOC framework naturally unifies masked autoencoding (MAE) and joint-embedding prediction (JEPA) to extract compact, control-derived representations. Furthermore, a simulation-free inference strategy ensures computational efficiency and scalability for large-scale fMRI datasets. Our model demon
    
[^87]: 超越测地凸性的Wasserstein近端算法收敛性分析

    Convergence Analysis of the Wasserstein Proximal Algorithm beyond Geodesic Convexity

    [https://arxiv.org/abs/2501.14993](https://arxiv.org/abs/2501.14993)

    本文在不假设测地凸性的条件下，借助Wasserstein版本的Polyak-Łojasiewicz不等式证明了Wasserstein近端算法的无偏线性收敛速率，并改进了强测地凸性下已有的收敛率结果。

    

    近端算法是在一般度量空间中最小化非线性、非光滑泛函的有力工具。受近年来研究均值场机制下两层神经网络中噪声梯度下降训练动力学的最新进展启发，本文在不假设目标泛函具有测地凸性的条件下，为通用的Wasserstein近端算法的收敛性提供了一个简单且自洽的分析。在欧氏Polyak-Łojasiewicz不等式的自然Wasserstein类似条件下，我们证明了近端算法能够达到无偏的线性收敛速率。我们的收敛速率改进了在强测地凸性条件下求解Wasserstein梯度流的近端算法的现有收敛速率。我们还将分析扩展到测地半凸目标情形下的不精确近端算法。在数值实验中，近端算法……（摘要在此处被截断）

    arXiv:2501.14993v4 Announce Type: replace-cross  Abstract: The proximal algorithm is a powerful tool to minimize nonlinear and nonsmooth functionals in a general metric space. Motivated by the recent progress in studying the training dynamics of the noisy gradient descent algorithm on two-layer neural networks in the mean-field regime, we provide in this paper a simple and self-contained analysis for the convergence of the general-purpose Wasserstein proximal algorithm without assuming geodesic convexity of the objective functional. Under a natural Wasserstein analog of the Euclidean Polyak-{\L}ojasiewicz inequality, we establish that the proximal algorithm achieves an unbiased and linear convergence rate. Our convergence rate improves upon existing rates of the proximal algorithm for solving Wasserstein gradient flows under strong geodesic convexity. We also extend our analysis to the inexact proximal algorithm for geodesically semiconvex objectives. In our numerical experiments, prox
    
[^88]: 利用外生结构实现样本高效的强化学习

    Exploiting Exogenous Structure for Sample-Efficient Reinforcement Learning

    [https://arxiv.org/abs/2409.14557](https://arxiv.org/abs/2409.14557)

    该论文针对外生马尔可夫决策过程，建立了离散MDP、Exo-MDP与离散线性混合MDP之间的表征等价性，并在外生状态不可观测时证明了 $\Theta(Hr\sqrt{K})$ 的匹配极小极大遗憾界，为样本高效的强化学习提供了理论与算法基础。

    

    我们研究了一类结构化的马尔可夫决策过程，称为外生MDP（Exo-MDP），其状态空间被划分为外生和内生两个组成部分。外生状态以随机方式演化，独立于智能体的动作；而内生状态则基于这两部分状态和动作确定性地演化。Exo-MDP涵盖了运筹学中的许多应用场景，包括库存控制、资源管理和网约车调度。我们的第一个贡献是结构层面的：我们建立了离散MDP、Exo-MDP和离散线性混合MDP之间的表征等价性。我们的第二个贡献是统计层面的：当有效维度r相对于内生状态空间和动作空间较小时，我们刻画了在Exo-MDP中学习的极小极大遗憾。当外生状态不可观测时，我们证明了在K个回合、时域为H的情况下，遗憾的匹配上下界为 $\Theta(Hr \sqrt{K})$ 阶……

    arXiv:2409.14557v5 Announce Type: replace  Abstract: We study a structured class of Markov Decision Processes, known as Exo-MDPs, in which the state space is partitioned into exogenous and endogenous components. Exogenous states evolve stochastically, independent of the agent's actions, while endogenous states evolve deterministically based on both state components and actions. Exo-MDPs capture many operations research settings, including inventory control, resource management, and ride-sharing. Our first contribution is structural: we establish a representational equivalence between discrete MDPs, Exo-MDPs, and discrete linear mixture MDPs. Our second contribution is statistical. We characterize the minimax regret of learning in Exo-MDPs when the effective dimension r is small relative to the endogenous state and action spaces. When the exogenous states are unobserved, we prove matching upper and lower regret bounds of order $\Theta(Hr \sqrt{K})$ over $K$ episodes of horizon $H$, wher
    
[^89]: 用于相互作用动态系统的Roto-translated局部坐标系

    Roto-translated Local Coordinate Frames For Interacting Dynamical Systems

    [https://arxiv.org/abs/2110.14961](https://arxiv.org/abs/2110.14961)

    本研究提出了为每个节点-对象引入局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。

    

    建模相互作用在学习复杂动态系统中是至关重要的，即相互作用对象具有高度非线性和时变行为的系统。在$\textit{几何图}$，$\textit{即}$，节点在欧几里得空间中放置的图形中，即使是在$\textit{任意}$选择的全局坐标系中，可以形式化地表示大类这样的系统，例如交通场景中的车辆。尽管全局坐标系是任意选择的，但各自动态系统的控制动力学不变于旋转和平移，也被称为$\textit{伽利略不变性}$。忽略这些不变性会导致更差的泛化能力，因此在这项工作中，我们提出每个节点对象的局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。此外，局部坐标系允许自然定义各向异性滤波器

    arXiv:2110.14961v3 Announce Type: replace  Abstract: Modelling interactions is critical in learning complex dynamical systems, namely systems of interacting objects with highly non-linear and time-dependent behaviour. A large class of such systems can be formalized as $\textit{geometric graphs}$, $\textit{i.e.}$, graphs with nodes positioned in the Euclidean space given an $\textit{arbitrarily}$ chosen global coordinate system, for instance vehicles in a traffic scene. Notwithstanding the arbitrary global coordinate system, the governing dynamics of the respective dynamical systems are invariant to rotations and translations, also known as $\textit{Galilean invariance}$. As ignoring these invariances leads to worse generalization, in this work we propose local coordinate frames per node-object to induce roto-translation invariance to the geometric graph of the interacting dynamical system. Further, the local coordinate frames allow for a natural definition of anisotropic filtering in g
    
[^90]: 面向预训练模型的几何感知自适应技术

    Geometry-Aware Adaptation for Pretrained Models. (arXiv:2307.12226v1 [cs.LG])

    [http://arxiv.org/abs/2307.12226](http://arxiv.org/abs/2307.12226)

    本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。

    

    机器学习模型，包括著名的零样本模型，通常在仅具有较小比例标签空间的数据集上进行训练。这些标签空间通常使用度量来衡量标签之间的距离关系。我们提出了一种简单的方法来利用这些信息，将已训练的模型调整以可靠地预测新类别，或者在零样本预测的情况下改善性能，而无需额外的训练。我们的技术是标准预测规则的替代方案，在其中将argmax替换为Fréchet平均值。我们为这种方法提供了全面的理论分析，研究了（i）学习理论结果，权衡标签空间直径、样本复杂性和模型维度，（ii）表征可能预测任何未观察到的类别的所有情景的特征，（iii）一种最优的主动学习式下一类别选择过程，以获取最佳的训练类别。

    Machine learning models -- including prominent zero-shot models -- are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes -- or, in the case of zero-shot prediction, to improve its performance -- without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping argmax with the Fr\'echet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes f
    

