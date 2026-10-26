# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Singular Geometry of Low-Rank Adaptation](https://arxiv.org/abs/2610.32854) | 本文借助奇异学习理论中的实对数典型阈值（RLCT）刻画了LoRA作为奇异统计模型的学习复杂度，证明该复杂度由适应层前向激活与反向传播敏感度构成的协方差决定，并在非退化情形下恢复降秩回归的经典复杂度。 |
| [^2] | [Mixed Engression: Distributional Regression for Multivariate Mixed-Type Responses](https://arxiv.org/abs/2610.32850) | 提出混合engression方法，利用有界差异诱导的能量损失拟合连续生成器并对软二元输出进行阈值化，从而在无需响应矩假设的条件下估计多变量混合类型（连续与二元）响应的条件分布。 |
| [^3] | [Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning](https://arxiv.org/abs/2610.32849) | 该论文提出带反向耦合的神经低度滤波谱理论，从反向传播梯度推导出跨层局部耦合动力学，利用上层敏感性为早期谱步骤生成有效标签，从而在逐层学习框架中实现跨层特征交互。 |
| [^4] | [Generalization properties of the token perceptron: the role of embedding and vocabulary](https://arxiv.org/abs/2610.32788) | 该论文提出“词元感知机”这一最小可解线性模型，并用复制方法解析计算其泛化误差，揭示了嵌入空间的几何结构与有限词表大小如何影响大语言模型中词元预测的学习与泛化。 |
| [^5] | [Structure and Exact Certification of Nonnegative Tensor Decompositions](https://arxiv.org/abs/2610.32787) | 本文为非负张量分解建立了结构性理论并给出精确认证算法，能在 G(R)L^{O(R²)} 时间内计算非负秩、求出最小长度分解、判定最小性与本质唯一性，并在性质失效时返回精确见证。 |
| [^6] | [Sparse Bayesian Neural Networks via Sticky Piecewise Deterministic Markov Processes](https://arxiv.org/abs/2610.32753) | 该论文提出利用粘性分段确定性马尔可夫过程结合尖峰-平板先验，在贝叶斯神经网络的采样过程中将权重精确置零，实现渐近精确的稀疏贝叶斯神经网络采样与剪枝。 |
| [^7] | [Strongly Adaptive Online Learning with Time-Varying Movement Costs](https://arxiv.org/abs/2610.32740) | 本文提出了一种针对时变移动代价在线凸优化的强自适应算法，通过新颖地归约到此前未被探索的带移动代价的睡眠专家问题，在任意时间区间上实现了数据依赖的遗憾界。 |
| [^8] | [Score-Preserving Continual Lasso under Task Shift](https://arxiv.org/abs/2610.32708) | 提出保得分持续Lasso（SP-CL），通过保留历史平方损失得分将稀疏估计与状态传递分离，避免了递归复用 ℓ₁ 估计量导致的正则化偏差跨任务累积，并在平方损失下精确恢复全历史Lasso解与稀疏率。 |
| [^9] | [Acquisition for Spherical-Linear Bayesian Optimization on the Cube](https://arxiv.org/abs/2610.32628) | 本文分析了单位立方体上各类贝叶斯优化采集子问题全局求解的复杂度，并提出了一种新的采集函数，其全局优化耗时远少于现有多起点拟牛顿方法，同时保持或提升了优化性能。 |
| [^10] | [Contextual Decision Making: Predictive or Generative](https://arxiv.org/abs/2610.32584) | 该论文从统计精度与优化可解性两个角度系统比较了先预测后优化与先生成后优化两种情境决策范式，证明了平滑性差异可以抵消生成式方法的维度负担，并揭示了凸性保持与学习复杂度之间的基本权衡。 |
| [^11] | [Batched NN 4D-Var: a performant hybrid model framework for the Integrated Forecasting System](https://arxiv.org/abs/2610.32582) | 本文提出了批处理神经网络四维变分框架，通过在综合预报系统中嵌入神经网络表示模式误差，并以随机顺序处理多个同化窗口、对参数增量取平均实现在线学习，构建了高性能的机器学习与数值预报混合模式。 |
| [^12] | [Exact Global Minimax Risk near Nonidentification](https://arxiv.org/abs/2610.32546) | 该论文给出了在冗余参数非识别点附近，最大临界路径极小极大风险等于全参数空间精确渐近极小极大风险的充分条件，并构造了能一致达到该精确首项常数的估计量，且在含未知方差的logistic误差变量回归中确定出五阶精确风险常数。 |
| [^13] | [Random Jump Forests](https://arxiv.org/abs/2610.32498) | 本文提出随机跳跃森林这一非参数方法，利用单棵树同时划分协变量空间并在节点处结合Nelson-Aalen估计，无需马尔可夫假设即可估计多状态模型中的条件累积转移率与占据概率，并建立了一致性和泛函中心极限定理等理论保证。 |
| [^14] | [Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning](https://arxiv.org/abs/2610.32438) | 该研究提出一种C++17处理流水线，通过对长期LoRaWAN数据集进行系统性变量覆盖审计，发现CO2数据完全缺失、气压覆盖率极低，从而将分析重新聚焦于覆盖率高达95.76%的温度-湿度变量对，并结合异常检测、主成分分析、聚类与机器学习方法开展双变量分析。 |
| [^15] | [Optimal Multiclass Online Regret with Arbitrary Label Spaces](https://arxiv.org/abs/2610.32339) | 该论文确定了具有任意（可能无限）标签空间的不可知多类在线学习的最优遗憾上界 $O(\min\{T,\sqrt{dT}\})$，解决了 Hanneke 等人（2023）提出的开放问题。 |
| [^16] | [Empirical concentration for chains with infinite memory](https://arxiv.org/abs/2610.32265) | 本文将经验集中不等式从马尔可夫链推广到具有无限记忆的平稳链，在Wasserstein压缩条件下建立了有界差分不等式，并仅凭单条轨迹估计依赖系数，从而得到可直接使用的经验集中界。 |
| [^17] | [Federated Nonparametric Regression with Heterogeneous Clients](https://arxiv.org/abs/2610.32178) | 该论文提出一种联邦Nadaraya-Watson非参数回归方法，客户端仅传输本地核摘要即可构建协变量密度加权的联邦估计器，并通过精确恒等式与两阶段分析刻画异构客户端从联邦预测中获益的条件及统计推断。 |
| [^18] | [Covariance-Informed Ensemble Score Filtering: Embedding Localized Covariance Geometry into Score-Based Transport](https://arxiv.org/abs/2610.32148) | 该论文提出协方差感知的集合分数滤波器（CI-EnSF），通过将局部化预报误差协方差几何直接嵌入分数模型的前向扩散与反向输运过程，为大气数据同化中似然信息从观测分量到未观测分量的传播建立了内在的协方差介导通路。 |
| [^19] | [Implicit Neural Representations for Variational Problems on Graphons](https://arxiv.org/abs/2610.32147) | 该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。 |
| [^20] | [Conditional Flow Models for Offline Maximum Entropy Reinforcement Learning](https://arxiv.org/abs/2610.32125) | 该论文提出一个分离式框架，利用基于概率流ODE的条件扩散模型分别估计奖励分布和转移核，从而构建软最优贝尔曼算子的数据驱动近似来估计最优软动作价值函数，并提供了非渐近理论保证。 |
| [^21] | [DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting](https://arxiv.org/abs/2610.32071) | 提出双分支校准模块DBCM，通过共享适配器利用跨通道预测残差的共享结构，并结合低秩通道分支保留有限的通道特定灵活性，实现时间序列预测中更高效的测试时适应。 |
| [^22] | [Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data](https://arxiv.org/abs/2610.32039) | 本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。 |
| [^23] | [Compatible Likelihoods for Flow Matching on Manifolds](https://arxiv.org/abs/2610.31981) | 该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。 |
| [^24] | [Bayesian Propensity Score Predictive Inference for Extending Trial Findings to Target Populations](https://arxiv.org/abs/2610.31915) | 本文提出贝叶斯倾向得分预测推断（PSPI）框架，结合贝叶斯加性回归树与倾向得分自然三次样条，解决了当治疗效果存在效应修饰因素异质性时随机试验结果向目标人群推广的难题。 |
| [^25] | [Combinatorial Characterizations of the Learnability of Conditional Distributions](https://arxiv.org/abs/2610.31890) | 本文引入fat_γ Natarajan维数和fat_γ Graph维数两个组合维度，证明前者在每个正尺度上的有限性是条件分布函数类PAC可学习性的充要条件，并证明分布值函数类可学习当且仅当其每个坐标类均可学习。 |
| [^26] | [Early learning in large residual networks](https://arxiv.org/abs/2610.31783) | 该论文通过图表计算方法分析了大型残差网络在早期训练阶段的动态，发现在大输入维度极限下，网络在训练初期只能学习到输入的线性函数，无法学习更高阶的单项式。 |
| [^27] | [Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability](https://arxiv.org/abs/2610.31733) | 该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。 |
| [^28] | [Operationalising Weaker Faithfulness: Recovery, Ambiguity and Computational Limits](https://arxiv.org/abs/2610.31724) | 本文提出两种操作化弱忠实性的策略——邻接学习中的选择性边保护与显式保留备选有向图解释（解释家族），使结构学习器能够恢复单独独立但联合预测结果的变量（如XOR）之间的联系，将完整父对恢复率从0/20提升至20/20，且家族估计器在48/50的样本中保留了全部备选解释。 |
| [^29] | [Autoregressive Physics-Informed Neural Operator Surrogates and Joint 4-D Seismic-Production Bayesian History Matching](https://arxiv.org/abs/2610.31677) | 该论文提出了一种以自回归物理信息傅里叶神经算子作为代理模型加速的贝叶斯历史拟合框架，通过两阶段 aREKI 与 ES-MDA 策略联合同化生产数据与四维地震数据，显著降低了大规模油藏反演的计算成本与病态性。 |
| [^30] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^31] | [Differentially Private Synthetic Data via Generalized Moment Matching](https://arxiv.org/abs/2610.30196) | 提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。 |
| [^32] | [The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits](https://arxiv.org/abs/2610.29982) | 该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。 |
| [^33] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^34] | [Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals](https://arxiv.org/abs/2610.26944) | 该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。 |
| [^35] | [SpectrumQuant: Spectrum Alignment for LLM Quantization](https://arxiv.org/abs/2610.20303) | 提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。 |
| [^36] | [Latent-Space Importance Weighting under Covariate Shift](https://arxiv.org/abs/2610.15571) | 该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。 |
| [^37] | [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332) | 提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。 |
| [^38] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^39] | [Bandits with Probing: Optimal Regret and the Limits of Winner Feedback](https://arxiv.org/abs/2609.15248) | 本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。 |
| [^40] | [Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency](https://arxiv.org/abs/2609.02728) | 该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。 |
| [^41] | [Verdict Instability of OOD Scores under Reference Resampling](https://arxiv.org/abs/2609.00691) | 本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。 |
| [^42] | [Discretization and Statistical Consistency of Functional Flow Matching](https://arxiv.org/abs/2608.04531) | 本文证明了在有限观测细化（允许非嵌套）且一致有界的条件下，泛函流匹配的速度目标依均方收敛，并量化了观测、学习与数值误差对生成分布的影响，为其函数空间模型的统计一致性提供了理论保证。 |
| [^43] | [Learning Bidirectional Causal Interactions with Heteroscedastic Neural Networks](https://arxiv.org/abs/2607.22313) | 提出无需外部工具变量的异方差神经联立方程估计器 SEM-DNN，通过条件协方差对角化实现观测数据中双向同期因果交互的唯一识别与非线性联合估计。 |
| [^44] | [Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability](https://arxiv.org/abs/2606.30930) | 本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。 |
| [^45] | [Online Distributional Prediction via Latent Cluster Geometry Under Drift and Corruption](https://arxiv.org/abs/2606.18778) | 该论文提出一种在漂移与对抗污染环境下基于潜在聚类几何配置的在线分布预测方法，利用Gibbs拟后验平均与可逆跳转MCMC进行预测，并给出了将PAC-贝叶斯复杂度、污染敏感后验扰动与漂移效应分离的累积Wasserstein-1遗憾界。 |
| [^46] | [Transient Amplification in Cascaded Gradient Methods: Exact Minimax Laws and Tracking without Resonance](https://arxiv.org/abs/2606.04031) | 该论文针对级联梯度方法首次给出精确且与维度无关的最坏瞬态放大极小极大公式，证明最坏瞬态为 1 + c/(1-γ) 量级、仅凭特征值无法预测有限时间增益，并在鞅噪声下确立了三次逆间隙缩放的跟踪方差定律。 |
| [^47] | [Instance-dependent Stochastic Lipschitz bandit](https://arxiv.org/abs/2605.29748) | 该论文提出了一种新的分析与算法，通过Lipschitz函数在其水平集上的次优性间隙积分来刻画遗憾，从而得到了真正依赖实例的遗憾界，能够适应水平集的局部增长结构而不仅仅是其渐近行为。 |
| [^48] | [Causal Representation Learning for Generalisable Recommendation](https://arxiv.org/abs/2605.27043) | 该论文提出一种基于因果表征学习的推荐方法，通过信息论解耦准则和可优化的变分下界，从处理表征中剔除受混淆因子影响的非因果信息，从而解决训练分布与服务分布不一致导致的推荐模型泛化难题。 |
| [^49] | [Efficient Benchmarking Is Just Feature Selection and Multiple Regression](https://arxiv.org/abs/2605.25773) | 该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。 |
| [^50] | [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757) | 该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。 |
| [^51] | [POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles](https://arxiv.org/abs/2605.07775) | POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。 |
| [^52] | [Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching](https://arxiv.org/abs/2605.06474) | 提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。 |
| [^53] | [Frame Theoretical Derivation of Three Factor Learning Rule for Oja's Subspace Rule](https://arxiv.org/abs/2604.02849) | 本文利用框架理论从Oja子空间规则中系统地推导出生物学上合理的误差门控Hebbian PCA学习规则（EGHR-PCA），并证明其全局第三因子恰好是对称矩阵空间自然框架下的框架系数。 |
| [^54] | [On the Complexity of Best-Arm Identification in Non-Stationary Linear Bandits](https://arxiv.org/abs/2603.10346) | 本文研究了非平稳线性老虎机中固定预算最优臂识别的复杂度，指出基于G-最优设计的经典复杂度度量过于悲观，因为其下界仅建立在标准基向量构成的臂集合上，掩盖了更丰富的臂集合结构可能带来的优势。 |
| [^55] | [Softly Symbolifying Kolmogorov-Arnold Networks](https://arxiv.org/abs/2512.07875) | 提出软符号化的Kolmogorov-Arnold网络（S2KAN），通过将符号基元直接融入训练，利用可微分的门控稀疏化机制与最小描述长度目标，在保留稠密样条灵活逼近能力的同时发现可解释的激活函数。 |
| [^56] | [Conformalized Polynomial Chaos Expansion for Uncertainty-aware Surrogate Modeling](https://arxiv.org/abs/2510.22375) | 该论文将折刀法保形预测与多项式混沌展开代理模型相结合，利用回归模型的线性特性以解析闭式形式计算留一法残差，从而在不需留出校准数据集的情况下为代理模型提供预测不确定性区间。 |
| [^57] | [From Dueling Bandits to Episodic MDPs: Learning with Adversarial Preference Feedback](https://arxiv.org/abs/2507.11706) | 本文将决斗老虎机扩展到偏好反馈会影响状态转移的情景MDP设定，在对抗性偏好变化下建立了极小极大遗憾界 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，并给出了达到该界的算法。 |
| [^58] | [Uniform ergodicity and spectral gap of importance-tempered MCMC with heavy-tailed targets](https://arxiv.org/abs/2506.19186) | 该论文针对重尾目标分布，给出了重要性加权退火RWM算法一致遍历性的充要条件（d/γ < β < (γ-2)/γ），并证明了适当选择退火参数β可显著降低对初始化的敏感性、大幅提升相对标准RWM的采样效率。 |
| [^59] | [Stochastic Inverse Problem: stability, regularization and Wasserstein gradient flow](https://arxiv.org/abs/2410.00229) | 本文将逆问题推广到随机情形，在概率空间中借助测度输运理论研究了直接反演、带正则化的变分方法以及Wasserstein梯度流优化，并揭示了度量选择对稳定性和解的性质的关键影响。 |
| [^60] | [A Randomized Permutation Whole-Model Test Heuristic for Self-Validated Ensemble Models (SVEM)](https://arxiv.org/abs/2405.11156) | 本文提出了一种基于随机置换的全模型检验启发式方法，通过在多个因子空间点上构建参考分布并利用马氏距离，来检验自验证集成模型（SVEM）拟合相对于常数响应原假设的统计显著性。 |

# 详细

[^1]: 低秩适应的奇异几何

    Singular Geometry of Low-Rank Adaptation

    [https://arxiv.org/abs/2610.32854](https://arxiv.org/abs/2610.32854)

    本文借助奇异学习理论中的实对数典型阈值（RLCT）刻画了LoRA作为奇异统计模型的学习复杂度，证明该复杂度由适应层前向激活与反向传播敏感度构成的协方差决定，并在非退化情形下恢复降秩回归的经典复杂度。

    

    低秩适应如今已成为对大型神经网络进行参数高效微调的标准方法。其低秩分解本质上不可辨识，这使得LoRA成为一个奇异统计模型，其学习复杂度无法仅凭参数维度来刻画。我们利用奇异学习理论中的实对数典型阈值（RLCT）来研究这种复杂度，并分析它如何同时依赖于低秩参数化和预训练网络。在局部线性化下，LoRA诱导出一个低秩迹回归模型，其Kullback–Leibler几何由一个前向-后向协方差所控制，该协方差由被适应层的激活值和反向传播敏感度构造而成。这一协方差决定了哪些低秩扰动在统计上是可观测的。当它在相关低秩方向上非退化时，我们恢复了降秩回归的经典学习复杂度；当它……（原文摘要在此处截断）

    arXiv:2610.32854v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is now a standard approach to parameter-efficient fine-tuning of large neural networks. Its low-rank factorization is inherently non-identifiable, making LoRA a singular statistical model whose learning complexity cannot be described by parameter dimension alone. We study this complexity using the real log canonical threshold (RLCT) from singular learning theory and how it depends on both the low-rank parameterization and the pretrained network. Under a local linearization, LoRA induces a low-rank trace regression model whose Kullback--Leibler geometry is controlled by a forward-backward covariance, constructed from activations and backpropagated sensitivities at the adapted layer. This covariance determines which low-rank perturbations are statistically observable. When it is nondegenerate along the relevant low-rank directions, we recover the classical learning complexity of reduced-rank regression; when its 
    
[^2]: 混合Engression：面向多变量混合类型响应的分布回归

    Mixed Engression: Distributional Regression for Multivariate Mixed-Type Responses

    [https://arxiv.org/abs/2610.32850](https://arxiv.org/abs/2610.32850)

    提出混合engression方法，利用有界差异诱导的能量损失拟合连续生成器并对软二元输出进行阈值化，从而在无需响应矩假设的条件下估计多变量混合类型（连续与二元）响应的条件分布。

    

    估计同时包含连续分量和二元分量的多变量响应的条件分布，需要捕捉它们之间的依赖关系、尊重它们不同的支撑集，并适应可能具有重尾特征的连续结果。我们提出了混合engression（mixed engression）方法：首先使用由有界差异诱导的能量损失拟合一个连续生成器，然后对其软二元输出进行阈值化处理，从而产生混合响应样本。由于差异是有界的，即使不对响应矩做任何假设，总体目标函数仍然保持良定义。然而，我们证明了完整响应向量的全响应能量距离可能在阈值化之前趋于零，但随着响应维度的增长，阈值化之后仍可能有界地远离零。因此，对整个二元向量施加一个单独的能量损失，可以捕捉被全响应比较所掩盖的差异，并控制阈值化的影响。生成器因此通过……（摘要原文在此处截断）

    arXiv:2610.32850v1 Announce Type: cross  Abstract: Estimating the conditional distribution of multivariate responses with continuous and binary components requires capturing their dependence, respecting their different supports, and accommodating potentially heavy-tailed continuous outcomes. We propose mixed engression, which fits a continuous generator using energy losses induced by bounded discrepancies and then thresholds its soft binary outputs to produce mixed-response samples. Because the discrepancies are bounded, the population objective remains well defined without response-moment assumptions. However, we show that the full-response energy distance for the complete response vector can tend to zero before thresholding yet remain bounded away from zero afterward as response dimensions grow. A separate energy loss on the entire binary vector captures discrepancies obscured by the full-response comparison and controls the effect of thresholding. The generator is therefore fitted u
    
[^3]: 带反向耦合的神经低度滤波：跨层特征学习的谱理论

    Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning

    [https://arxiv.org/abs/2610.32849](https://arxiv.org/abs/2610.32849)

    该论文提出带反向耦合的神经低度滤波谱理论，从反向传播梯度推导出跨层局部耦合动力学，利用上层敏感性为早期谱步骤生成有效标签，从而在逐层学习框架中实现跨层特征交互。

    

    在联合训练中，反向传播允许后面的层重塑前面的表示，这种反馈机制在纯逐层学习中是不存在的。我们扩展了神经低度滤波（Neural LoFi）——一种最近提出的用于特征学习的逐层谱替代方法——以捕捉这种交互作用。从精确的反向传播梯度出发，我们推导出神经元的输入权重与其对应的下一层列之间的局部耦合。在具有平滑激活函数和固定周围环境的受控小初始化条件下，这些动力学会放大跨层相关算子的奇异模式。每个增长的模式通过共享的神经元系数将较早层的特征与上层响应配对。这启发了一种反向耦合算法：上层敏感性为较早的谱步骤生成依赖于样本的有效标签，同时经过修正的特征则细化向上传输的信息。

    arXiv:2610.32849v1 Announce Type: cross  Abstract: During joint training, backpropagation allows later layers to reshape earlier representations, a feedback mechanism absent from purely layer-wise learning. We extend Neural Low-Degree Filtering (Neural LoFi), a recently proposed layer-wise spectral surrogate for feature learning, to capture this interaction. Starting from the exact backpropagation gradients, we derive a local coupling between a neuron's incoming weights and its corresponding next-layer column. In a controlled small-initialization regime with smooth activations and fixed surroundings, these dynamics amplify singular modes of a cross-layer correlation operator. Each growing mode pairs an earlier feature with an upper-layer response through shared neuron coefficients. This motivates a backward-coupling algorithm in which upper-layer sensitivities generate sample-dependent effective labels for earlier spectral steps, while corrected features refine upward transmission. A f
    
[^4]: 词元感知机的泛化性质：嵌入与词表的作用

    Generalization properties of the token perceptron: the role of embedding and vocabulary

    [https://arxiv.org/abs/2610.32788](https://arxiv.org/abs/2610.32788)

    该论文提出“词元感知机”这一最小可解线性模型，并用复制方法解析计算其泛化误差，揭示了嵌入空间的几何结构与有限词表大小如何影响大语言模型中词元预测的学习与泛化。

    

    大语言模型被训练用于从有限词表中预测词元，每个词元由 d 维嵌入空间中的一个向量表示。因此，词元预测本质上是一个分类问题，其类别的几何结构由嵌入决定。我们研究这种几何结构如何影响一个最小可解模型中的学习：该模型是一个线性预测器，通过每个位置上一个 d×d 的矩阵，将 N 个嵌入词元的序列映射为 d 维输出，并在教师-学生框架下进行训练。我们将其称为“词元感知机”（token perceptron）。我们利用复制方法计算其泛化误差，在每位置样本数 α=P/N 固定的情况下取 N→∞，同时保持嵌入维度 d 和词表大小 D 有限。作为第一个参考情形，我们考虑球面上的连续嵌入，以分离出嵌入维度所起的作用。此时自然负载为 α/d，且当 d>1 时，归一化标签固定了 d……

    arXiv:2610.32788v1 Announce Type: cross  Abstract: Large language models are trained to predict tokens from a finite vocabulary, each represented by a vector in a $d$-dimensional embedding space. Therefore, token prediction is a classification problem whose classes have a geometry determined by the embedding. We study how this geometry affects learning in a minimal solvable model: a linear predictor that maps a sequence of $N$ embedded tokens to a $d$-dimensional output through one $d\times d$ matrix per position, trained in the teacher--student setting. We call this the \textit{token perceptron}. We compute its generalization error with the replica method, taking $N\to\infty$ at fixed number of examples per position $\alpha=P/N$ and keeping $d$ and the vocabulary size $D$ finite. As a first reference case, we consider continuous embeddings on the sphere, which isolate the role of the embedding dimension. The natural load is then $\alpha/d$, and for $d>1$ a normalized label fixes the d
    
[^5]: 非负张量分解的结构与精确认证

    Structure and Exact Certification of Nonnegative Tensor Decompositions

    [https://arxiv.org/abs/2610.32787](https://arxiv.org/abs/2610.32787)

    本文为非负张量分解建立了结构性理论并给出精确认证算法，能在 G(R)L^{O(R²)} 时间内计算非负秩、求出最小长度分解、判定最小性与本质唯一性，并在性质失效时返回精确见证。

    

    我们研究了通过有理因子矩阵给出的非负张量分解的结构与精确认证问题。最小性或本质唯一性的每一次失效都包含一个不可约的正交换，即发生在 p 个给定项与 q 个竞争项之间的交换。基于 Lovitz--Petrov 分裂定理的维数计数将这种交换的有效阶数（即各模式上联合因子张成维数之和减一）界定为 p+q-2，并且在每个给定长度 R≥2 时该界均可达到。在利用 Farkas 引理消去环境行变量之后，所得的半代数系统的实变量个数少于 2R² 个。对于输入比特长度 L，这给出了一个确定性的、运行时间为 G(R)L^{O(R²)} 的算法，该算法能够计算非负秩并给出精确的最小长度分解，判定最小性与本质唯一性，并且当任一性质失效时返回精确的见证。

    arXiv:2610.32787v1 Announce Type: cross  Abstract: We study the structure and exact certification of nonnegative tensor decompositions supplied through rational factor matrices. Every failure of minimality or essential uniqueness contains an irreducible positive exchange between $p$ supplied terms and $q$ competing terms. A dimension count based on the Lovitz--Petrov splitting theorem bounds the effective order of such an exchange, the sum over the modes of the joint factor-span dimension minus one, by $p+q-2$, and this bound is attained at every supplied length $R\ge2$. After eliminating ambient-row variables by the Farkas lemma, the resulting semialgebraic systems have fewer than $2R^2$ real variables. For input bit length $L$, this gives a deterministic $G(R)L^{O(R^2)}$-time algorithm that computes the nonnegative rank and an exact minimum-length decomposition, decides minimality and essential uniqueness, and returns exact witnesses when either property fails. The exponent is indepe
    
[^6]: 基于粘性分段确定性马尔可夫过程的稀疏贝叶斯神经网络

    Sparse Bayesian Neural Networks via Sticky Piecewise Deterministic Markov Processes

    [https://arxiv.org/abs/2610.32753](https://arxiv.org/abs/2610.32753)

    该论文提出利用粘性分段确定性马尔可夫过程结合尖峰-平板先验，在贝叶斯神经网络的采样过程中将权重精确置零，实现渐近精确的稀疏贝叶斯神经网络采样与剪枝。

    

    越来越多的证据表明，神经网络的性能在很大程度上取决于其权重中的一小部分，这些权重通常通过训练一个密集网络然后进行剪枝来识别。在贝叶斯框架内，剪枝可以通过尖峰-平板先验直接纳入，从而对所涉及的不确定性进行恰当处理，包括剪枝不确定性和参数不确定性。在贝叶斯神经网络（BNN）中，通过MCMC进行采样是出了名的困难。我们证明了基于分段确定性马尔可夫过程（PDMP）的现代蒙特卡洛方法，只要数值构造的速率界成立，就可以渐近精确地采样BNN的后验分布。尖峰-平板先验可以通过粘性版本的PDMP纳入，在采样过程中将权重精确地设置为零。我们检验了这些采样器的预测性能和收敛性，并表明以这种方式采样的稀疏BNN…

    arXiv:2610.32753v1 Announce Type: new  Abstract: There is growing evidence that the performance of a neural network depends largely on a small subset of its weights, which are typically identified by training a dense network and pruning it afterwards. Within a Bayesian framework, pruning can be directly incorporated through a spike-and-slab prior, providing a proper treatment of the uncertainties involved, both with respect to pruning and to parameter uncertainty. In Bayesian neural networks (BNNs), sampling by MCMC is notoriously hard. We demonstrate that modern Monte Carlo methods based on piecewise deterministic Markov processes (PDMPs) can be used to sample posteriors of BNNs asymptotically exact whenever numerically constructed rate bounds hold. Spike-and-slab priors can be incorporated through sticky versions of the PDMPs, setting weights exactly to zero during sampling. We examine the predictive performance and convergence of the samplers, and show that sparse BNNs sampled this 
    
[^7]: 具有时变移动代价的强自适应在线学习

    Strongly Adaptive Online Learning with Time-Varying Movement Costs

    [https://arxiv.org/abs/2610.32740](https://arxiv.org/abs/2610.32740)

    本文提出了一种针对时变移动代价在线凸优化的强自适应算法，通过新颖地归约到此前未被探索的带移动代价的睡眠专家问题，在任意时间区间上实现了数据依赖的遗憾界。

    

    arXiv:2610.32740v1 公告类型：cross 摘要：我们研究了具有时变移动代价的在线凸优化（OCO）的强自适应遗憾，该设定用于评估学习者在任意子区间上的性能。这一框架刻画了复杂的序列决策问题，其中环境与切换决策的代价均随时间波动。我们提出了一种新颖的算法，对于任意区间 $[a,b] \subseteq [T]$，可实现数据依赖的遗憾界 $\widetilde{\mathcal{O}}\left( \sqrt{(D^2+DP_{[a,b]})\sum_{t \in [a,b]}(\|g_t\|^2+\lambda_t\|g_t\|)} \right)$，其中 $D$ 为决策域的直径，$P_{[a,b]}$ 为比较序列在 $[a,b]$ 上的路径长度，$g_t$ 为损失的梯度，$\lambda_t$ 为第 $t$ 轮的移动代价系数。我们方法的关键在于一种新颖的归约，将问题转化为带移动代价的睡眠专家问题，这是一个此前未被探索过、且具有独立研究价值的变体。作为……

    arXiv:2610.32740v1 Announce Type: cross  Abstract: We study strongly adaptive regret for Online Convex Optimization (OCO) with time-varying movement costs, a setting that evaluates a learner's performance over arbitrary sub-intervals. This framework captures complex sequential decision-making problems where both the environment and the cost of switching decisions fluctuate over time. We propose a novel algorithm which, for any interval $[a,b] \subseteq [T]$, achieves a data-dependent regret bound of $ \widetilde{\mathcal{O}}\left( \sqrt{(D^2+DP_{[a,b]})\sum_{t \in [a,b]}(\|g_t\|^2+\lambda_t\|g_t\|)} \right) $, where $D$ is the diameter of the decision domain, $P_{[a,b]}$ is the path length of the comparator sequence over $[a,b]$, $g_t$ is the gradient of the loss, and $\lambda_t$ is the movement-cost coefficient at round $t$. Key to our approach is a novel reduction to sleeping experts with movement costs, a previously unexplored variant that is of independent interest. As a result of 
    
[^8]: 任务偏移下的保得分持续Lasso

    Score-Preserving Continual Lasso under Task Shift

    [https://arxiv.org/abs/2610.32708](https://arxiv.org/abs/2610.32708)

    提出保得分持续Lasso（SP-CL），通过保留历史平方损失得分将稀疏估计与状态传递分离，避免了递归复用 ℓ₁ 估计量导致的正则化偏差跨任务累积，并在平方损失下精确恢复全历史Lasso解与稀疏率。

    

    持续学习旨在从序列任务中进行学习，同时在历史数据不再可访问的情况下缓解灾难性遗忘。我们研究高维稀疏线性回归中的持续学习问题，其中 ℓ₁ 正则化带来了持续岭回归中所不存在的困难：递归地复用 ℓ₁ 惩罚估计量会传播早期的收缩效应，并在任务间累积正则化偏差。我们提出保得分持续Lasso（SP-CL），该方法通过保留历史平方损失得分，将稀疏估计与状态传递分离开来。在平方损失下，其累积估计量能够精确恢复全历史Lasso的极小值解。当任务系数一致时，它继承了全历史的稀疏率。对于相关的任务特定系数，SP-CL 采用一种正则化的当前任务修正，该修正不会改变所传递的状态，并达到离线源辅助（截断）

    arXiv:2610.32708v1 Announce Type: new  Abstract: Continual learning (CL) aims to learn from sequential tasks while mitigating catastrophic forgetting when historical data are no longer accessible. We study continual learning for high-dimensional sparse linear regression, where $\ell_1$ regularization creates a difficulty absent from continual ridge regression: that recursively reusing an $\ell_1$-penalized estimator can propagate earlier shrinkage and accumulate regularization bias across tasks. We propose score-preserving continual Lasso (SP-CL), which separates sparse estimation from state transmission by preserving the historical squared-loss score. Under squared loss, its cumulative estimator exactly recovers a full-history Lasso minimizer. When task coefficients coincide, it inherits the full-history sparse rate. For related task-specific coefficients, SP-CL uses a regularized current-task correction that leaves the transmitted state unchanged and attains the offline source-assist
    
[^9]: 立方体上球面-线性贝叶斯优化的采集函数

    Acquisition for Spherical-Linear Bayesian Optimization on the Cube

    [https://arxiv.org/abs/2610.32628](https://arxiv.org/abs/2610.32628)

    本文分析了单位立方体上各类贝叶斯优化采集子问题全局求解的复杂度，并提出了一种新的采集函数，其全局优化耗时远少于现有多起点拟牛顿方法，同时保持或提升了优化性能。

    

    近期研究表明，仅使用线性代理模型并配合巧妙的变换来执行贝叶斯优化（BO），就足以在高维测试函数上逼近基于标准高斯过程（GP）方法的性能。当决策区域为球体时，该策略可以实现快速精确的采集。然而，在贝叶斯优化文献中最常遇到的决策区域是单位立方体，在这种情形下，通常采用标准方法（即多起点拟牛顿法）来求解采集子问题。在本文中，我们将讨论关于单位立方体上各种采集子问题全局求解复杂度的一系列正反两方面的结果。在此讨论的基础上，我们将提出一种新的采集函数，其全局优化所需的执行时间显著少于现有方法，同时能够保持或提高性能。

    arXiv:2610.32628v1 Announce Type: new  Abstract: Recent work demonstrated that executing Bayesian Optimization (BO) using simply a linear surrogate with a clever transformation is enough to approximate the performance of the standard Gaussian-Process (GP) based version on high-dimensional test functions. When the decision region is a sphere, this strategy admits rapid exact acquisition. However, it is the unit cube we encounter most often in the BO literature, and in this setting a standard approach, namely a multi-start quasi-Newton method, is employed for solving the acquisition subproblem. In this article, we will discuss a series of results, both positive and negative, regarding the complexity of global resolution of various acquisition subproblems over the unit cube. On the basis of this discussion, we will propose a new acquisition function which is globally optimizable using significantly less execution time than the incumbent approach while maintaining or improving performance.
    
[^10]: 情境决策：预测式还是生成式

    Contextual Decision Making: Predictive or Generative

    [https://arxiv.org/abs/2610.32584](https://arxiv.org/abs/2610.32584)

    该论文从统计精度与优化可解性两个角度系统比较了先预测后优化与先生成后优化两种情境决策范式，证明了平滑性差异可以抵消生成式方法的维度负担，并揭示了凸性保持与学习复杂度之间的基本权衡。

    

    情境决策利用观测到的协变量在不确定性条件下指导决策。两种自然的方法是“先预测后优化”，它学习条件期望目标；以及“先生成后优化”，它学习条件不确定性并利用生成的场景进行优化。我们通过考察学习目标的选择如何影响统计精度与优化的可解性，来研究何时哪种方法更优。尽管PtO学习的目标范围更窄，我们的遗憾界表明，平滑性方面的差异可以抵消GtO所承担的额外维度负担。利用平滑性同时引发了一个优化上的权衡：在本文所研究的非负权重PtO方法类中，一个下界限制了额外平滑性所能带来的收益；而更灵活的方法可能引入带符号的权重，从而失去保持凸性的保证。GtO能够保持逐场景凸性，但需要学习更丰富（的不确定性模型）。

    arXiv:2610.32584v1 Announce Type: cross  Abstract: Contextual decision making uses observed covariates to guide decisions under uncertainty. Two natural approaches are predict-then-optimize (PtO), which learns the conditional expected objective, and generate-then-optimize (GtO), which learns conditional uncertainty and optimizes using generated scenarios. We study when each approach is preferable by examining how the choice of learning target affects statistical accuracy and optimization tractability. Although PtO learns a narrower target, our regret bounds show how differences in smoothness can offset GtO's additional dimensional burden. Exploiting smoothness also raises an optimization tradeoff: within the class of nonnegative-weight PtO methods studied, a lower bound limits the gains from additional smoothness, while more flexible methods may introduce signed weights and lose the guarantee of preserving convexity. GtO preserves scenario-wise convexity, but requires learning a richer
    
[^11]: 批处理神经网络四维变分（Batched NN 4D-Var）：面向综合预报系统的高性能混合模式框架

    Batched NN 4D-Var: a performant hybrid model framework for the Integrated Forecasting System

    [https://arxiv.org/abs/2610.32582](https://arxiv.org/abs/2610.32582)

    本文提出了批处理神经网络四维变分框架，通过在综合预报系统中嵌入神经网络表示模式误差，并以随机顺序处理多个同化窗口、对参数增量取平均实现在线学习，构建了高性能的机器学习与数值预报混合模式。

    

    结合机器学习与数值天气预报的混合预报系统，为在保持物理一致性的同时提升预报技巧提供了一条有前景的途径。在本研究中，我们利用四维变分（4D-Var）方法研究了在综合预报系统（Integrated Forecasting System, IFS）内对依赖于状态的模式误差订正进行在线学习。在以往工作的基础上，我们使用一个嵌入预报模式内部的神经网络来表示模式误差，并通过资料同化循环利用观测对其进行训练。首先，我们引入了一种针对性的神经网络四维变分变体，其中仅估计神经网络参数，而大气状态由预先计算的分析场提供。随后，我们提出了一种批处理扩展方法，该方法以随机顺序处理多个同化窗口，并对多个窗口上的参数增量取平均，其灵感来自小批量随机优化。实验在综合预报系统上开展（原文摘要至此截断）。

    arXiv:2610.32582v1 Announce Type: new  Abstract: Hybrid forecasting systems combining machine learning and numerical weather prediction offer a promising route toward improved forecast skill while preserving physical consistency. In this study, we investigate online learning of state-dependent model-error corrections within the Integrated Forecasting System using 4D-Var. Building on previous work, we represent model error using a neural network embedded within the forecast model and trained from observations through the data assimilation cycle. First, we introduce a targeted variant of neural-network 4D-Var in which only the neural-network parameters are estimated, while the atmospheric state is provided by a precomputed analysis. We then propose a batched extension that processes assimilation windows in random order and averages parameter increments over multiple windows, drawing inspiration from mini-batch stochastic optimization. Experiments are conducted with the Integrated Forecas
    
[^12]: 非识别点附近的精确全局极小极大风险

    Exact Global Minimax Risk near Nonidentification

    [https://arxiv.org/abs/2610.32546](https://arxiv.org/abs/2610.32546)

    该论文给出了在冗余参数非识别点附近，最大临界路径极小极大风险等于全参数空间精确渐近极小极大风险的充分条件，并构造了能一致达到该精确首项常数的估计量，且在含未知方差的logistic误差变量回归中确定出五阶精确风险常数。

    

    我们研究当某个冗余参数在一个子模型上不可识别、但其会影响一个在该处趋于消失的目标量时的平方误差估计问题。对于具有高阶剖面展开的模型，我们给出了充分条件，使得最大的临界路径极小极大风险恰为整个参数空间上的精确渐近极小极大风险。一个高效的可见特征、可忽略的校准误差以及稳定的恢复机制，使得单个估计量能够在剖面坐标未知的情况下一致地达到首项常数。该估计量利用了校准估计、局部特征均值以及一个经验恢复特征。该常数通过一个有界正态决策问题来刻画。我们在两个模型中验证了这些条件。在具有一个代理变量和两个未知方差的中心化高斯logistic误差变量回归模型中，我们确定了五阶全局系数风险及其精确常数。一个稀疏扩展将方差分解……（原文摘要在此处截断）

    arXiv:2610.32546v1 Announce Type: new  Abstract: We study squared-error estimation when a nuisance parameter is unidentified on a submodel but affects a target that vanishes there. For models with higher-order profile expansions, we give sufficient conditions under which the largest critical-path minimax risk is the exact asymptotic minimax risk over the full parameter space. An efficient observable feature, negligible calibration error and stable recovery yield one estimator attaining the leading constant uniformly, with the profile coordinates unknown. The estimator uses calibration estimates, a local feature mean and an empirical recovery feature. The constant is expressed through a bounded-normal decision problem. We verify the conditions in two models. In centered Gaussian logistic errors-in-variables regression with one surrogate and two unknown variances, we determine the fifth-order global coefficient risk and its exact constant. A sparse extension separates variance-decomposit
    
[^13]: 随机跳跃森林

    Random Jump Forests

    [https://arxiv.org/abs/2610.32498](https://arxiv.org/abs/2610.32498)

    本文提出随机跳跃森林这一非参数方法，利用单棵树同时划分协变量空间并在节点处结合Nelson-Aalen估计，无需马尔可夫假设即可估计多状态模型中的条件累积转移率与占据概率，并建立了一致性和泛函中心极限定理等理论保证。

    

    本文引入随机跳跃森林，作为一种非参数工具，用于估计多状态模型中的条件累积转移率和占据概率。其主要思想是：利用部分观测到的轨迹，让单棵树同时为所有转移划分协变量空间，并在终端节点处使用Nelson-Aalen估计量。占据概率通过乘积积分恢复，而无需马尔可夫假设。在协变量向量取有限多个值的假设以及适当的分裂条件下，可以建立累积率估计量的一致性和泛函中心极限定理。特别地，在马尔可夫情形下可以给出逐点置信区间和同时置信带。模拟研究表明，该方法的预测精度接近于正确设定的基准模型，而当基准模型被……时则有显著改进。

    arXiv:2610.32498v1 Announce Type: cross  Abstract: Random jump forests are introduced as a nonparametric tool for estimating conditional cumulative transition rates and occupation probabilities in multi-state models. The main idea is that a single tree partitions the covariate space for all transitions simultaneously, using partially observed trajectories, and with Nelson--Aalen estimators at the terminal nodes. Occupation probabilities are recovered through the product integral without requiring the Markov assumption. Uniform consistency and a functional central limit theorem for the estimators of cumulative rates can be established under the assumption that the covariate vector takes finitely many values, and under suitable splitting conditions. In particular, pointwise confidence intervals and simultaneous bands are available in the Markov case. Simulation studies show a predictive accuracy close to the correctly specified benchmark models and marked improvements when benchmarks are
    
[^14]: 长期LoRaWAN网络的数据质量审计与温度-湿度双变量分析：异常检测、主成分分析、聚类与机器学习

    Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning

    [https://arxiv.org/abs/2610.32438](https://arxiv.org/abs/2610.32438)

    该研究提出一种C++17处理流水线，通过对长期LoRaWAN数据集进行系统性变量覆盖审计，发现CO2数据完全缺失、气压覆盖率极低，从而将分析重新聚焦于覆盖率高达95.76%的温度-湿度变量对，并结合异常检测、主成分分析、聚类与机器学习方法开展双变量分析。

    

    长期环境物联网/LPWAN部署通常会聚合来自多种设备的异构数据流，但在分析之前并未对各变量的实际数据覆盖情况进行系统性审计——当多种传感器类型共存于同一网络时，这种风险会进一步加大。本工作提出了一种C++17处理流水线，并应用于Tour Perret LoRaWAN数据集（421,937帧数据，2021年6月至2023年6月，5个设备）。集成于数据提取阶段的系统性覆盖审计显示，CO2完全缺失（覆盖率0%），气压数据仅由一个孤立的基础模块（Wyres，占语料库的4.24%）上报，这使最初进行四变量联合分析的假设不成立。因此，分析重新聚焦于温度-湿度变量对，二者在来自4个Elsys EMS传感器的205,005帧数据（95.76%）中联合可用。该流水线依次执行异常检测（修正Z分数法、孤立森林算法）、双变量主成分分析（PCA）、聚类以及机器学习分析。

    arXiv:2610.32438v1 Announce Type: cross  Abstract: Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an increased risk when multiple sensor types coexist within the same network. This work proposes a C++17 processing pipeline applied to the Tour Perret LoRaWAN dataset (421 937 frames, June 2021--June 2023, 5 devices). A systematic coverage audit integrated into the extraction phase reveals that CO2 is completely missing (0% coverage) and pressure is reported only by an isolated base module (Wyres, 4.24% of the corpus), invalidating the initial hypothesis of a joint four-variable analysis. The analysis is refocused on the Temperature--Humidity pair, jointly available across 205{,}005 frames (95.76%) from 4 Elsys EMS sensors. The pipeline sequentially performs anomaly detection (modified Z-score, Isolation Forest), bivariate Principal Component Anal
    
[^15]: 具有任意标签空间的多类在线学习最优遗憾

    Optimal Multiclass Online Regret with Arbitrary Label Spaces

    [https://arxiv.org/abs/2610.32339](https://arxiv.org/abs/2610.32339)

    该论文确定了具有任意（可能无限）标签空间的不可知多类在线学习的最优遗憾上界 $O(\min\{T,\sqrt{dT}\})$，解决了 Hanneke 等人（2023）提出的开放问题。

    

    我们确定了在具有任意（可能是无限的）标签空间的不可知多类在线学习中的最优遗憾。对于每个多类 Littlestone 维度为 $d<\infty$ 的假设类，我们证明了在 $T$ 轮中 $O(\min\{T,\sqrt{dT}\})$ 的遗憾上界，该上界与已知的下界相匹配。这解决了 Hanneke 等人（2023）提出的一个问题，并将 Alon 等人（2021）的最优二分类遗憾界推广到了多类情形。我们的证明将 Hanneke 等人（2023）的单边专家覆盖加强为分数覆盖，并将该覆盖与到有限专家博弈的极小极大归约相结合，其最佳响应奖励提供了链式论证所需的分歧界。

    arXiv:2610.32339v1 Announce Type: cross  Abstract: We determine the optimal regret in agnostic multiclass online learning with arbitrary, possibly infinite, label spaces. For every hypothesis class of multiclass Littlestone dimension $d<\infty$, we prove a regret upper bound of $O(\min\{T,\sqrt{dT}\})$ over $T$ rounds, matching the known lower bound. This resolves a question of Hanneke et al. (2023) and extends the optimal binary regret bound of Alon et al. (2021). Our proof strengthens the one-sided expert cover of Hanneke et al. (2023) to a fractional cover. We combine this cover with a minimax reduction to a finite expert game, whose best-response rewards provide the disagreement bounds needed for chaining.
    
[^16]: 具有无限记忆的链的经验集中不等式

    Empirical concentration for chains with infinite memory

    [https://arxiv.org/abs/2610.32265](https://arxiv.org/abs/2610.32265)

    本文将经验集中不等式从马尔可夫链推广到具有无限记忆的平稳链，在Wasserstein压缩条件下建立了有界差分不等式，并仅凭单条轨迹估计依赖系数，从而得到可直接使用的经验集中界。

    

    时间序列的集中不等式通常涉及衡量依赖性的未知量，这妨碍了它们直接用作置信界或学习算法的准则。近期的工作针对平稳马尔可夫链证明了经验集中不等式，其中未知量被估计出来。我们将这一方法扩展到马尔可夫情形之外，考虑具有无限记忆的平稳链。我们首先在可求和的Wasserstein压缩条件下，于有界Polish度量空间上建立了有界差分不等式。随后，我们从单条轨迹估计其依赖系数：当状态空间有限时使用转移计数，当状态空间连续时使用核条件密度估计的导数。我们给出了一般的误差界，并说明这些误差界对记忆尾部的依赖关系。最后，通过一个固定的乘法安全因子，将这些估计量转化为经验集中不等式。

    arXiv:2610.32265v1 Announce Type: new  Abstract: Concentration inequalities for time series usually involve unknown quantities measuring dependence. This prevents their direct use as confidence bounds, or as criteria for learning algorithms. Recent works proved empirical concentration inequalities for stationary Markov chains, where the unknown quantities are estimated. We extend this approach beyond the Markov case, by considering stationary chains with infinite memory. We first establish a bounded-differences inequality on bounded Polish metric spaces under a summable Wasserstein contraction condition. We then estimate its dependence coefficients from a single trajectory, using either transition counts when the state space is finite, or derivatives of kernel conditional-density estimates for a continuous state space. We give general error bounds and illustrate their dependence on the memory tail. Finally, a fixed multiplicative safety factor turns these estimators into empirical conc
    
[^17]: 具有异构客户端的联邦非参数回归

    Federated Nonparametric Regression with Heterogeneous Clients

    [https://arxiv.org/abs/2610.32178](https://arxiv.org/abs/2610.32178)

    该论文提出一种联邦Nadaraya-Watson非参数回归方法，客户端仅传输本地核摘要即可构建协变量密度加权的联邦估计器，并通过精确恒等式与两阶段分析刻画异构客户端从联邦预测中获益的条件及统计推断。

    

    联邦预测器是否能使单个客户端受益，取决于总体异质性和估计不确定性两个方面。我们在三角阵列设置下针对Nadaraya-Watson回归研究这一问题，其中拟合分布与评估分布可能不同。每个客户端仅传输本地核摘要，使服务器能够在不集中个体观测数据的情况下构建一个以协变量密度加权的联邦估计器。客户端通过预设权重将其拟合的本地预测器与联邦预测器相结合，增益定义为在独立审计观测上配对的平方预测损失的减少量。我们给出了精确恒等式，用以刻画已实现拟合配对的增益及其重复拟合均值。我们还为联邦估计器建立了逐点推断。对客户端增益的两阶段分析将拟合变异性与审计变异性分离，并保留了当客户端（原文在此处截断）

    arXiv:2610.32178v1 Announce Type: new  Abstract: Whether a federated predictor benefits an individual client depends on both population heterogeneity and estimation uncertainty. We study this question for Nadaraya--Watson regression in a triangular array with potentially different fitting and evaluation distributions. Each client transmits local kernel summaries, allowing the server to construct a covariate-density-weighted federated estimator without centralizing individual observations. A client combines its fitted local predictor with the federated predictor through a prescribed weight, and gain is the paired reduction in squared prediction loss on an independent audit observation. Exact identities characterize both the gain of the realized fitted pair and its repeated-fitting mean. We also establish pointwise inference for the federated estimator. A two-stage analysis of client gains separates fitting variability from audit variability, retains the covariance created when a client'
    
[^18]: 协方差感知的集合分数滤波：将局部化协方差几何嵌入基于分数的输运

    Covariance-Informed Ensemble Score Filtering: Embedding Localized Covariance Geometry into Score-Based Transport

    [https://arxiv.org/abs/2610.32148](https://arxiv.org/abs/2610.32148)

    该论文提出协方差感知的集合分数滤波器（CI-EnSF），通过将局部化预报误差协方差几何直接嵌入分数模型的前向扩散与反向输运过程，为大气数据同化中似然信息从观测分量到未观测分量的传播建立了内在的协方差介导通路。

    

    背景误差协方差模型对大气数据同化至关重要，因为它们能将部分观测信息在空间、变量以及动态耦合的尺度上进行传播。集合方法和变分方法能够高效利用这种协方差几何，但其标准的分析公式主要建立在高斯近似或局部二次近似的基础上。基于分数的集合滤波为非线性分布输运提供了一个框架，但现有的集合分数滤波器缺乏一种内在的协方差介导通路，无法将似然信息从已观测的状态分量传播到未观测的状态分量。我们提出了一种协方差感知的集合分数滤波器（CI-EnSF），它将局部化的预报误差协方差几何直接嵌入到前向扩散和反向输运过程中。所提出的前向扩散在标准化状态空间中呈协方差形状，并在特定条件下退化为各向同性扩散（原文在此处截断）。

    arXiv:2610.32148v1 Announce Type: cross  Abstract: Background-error covariance models are central to atmospheric data assimilation because they spread partial observational information across space, variables, and dynamically coupled scales. Ensemble and variational methods exploit this covariance geometry efficiently, but their standard analysis formulations are largely built around Gaussian or locally quadratic approximations. Score-based ensemble filtering provides a framework for nonlinear distributional transport, but existing Ensemble Score Filter (EnSF) formulations lack an intrinsic covariance-mediated pathway for propagating likelihood information from observed to unobserved state components. We introduce a covariance-informed ensemble score filter (CI-EnSF) that embeds localized forecast-error covariance geometry directly into the forward diffusion and reverse transport. The proposed forward diffusion is covariance-shaped in standardized state space and becomes isotropic unde
    
[^19]: 面向Graphon上变分问题的隐式神经表示

    Implicit Neural Representations for Variational Problems on Graphons

    [https://arxiv.org/abs/2610.32147](https://arxiv.org/abs/2610.32147)

    该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。

    

    我们提出了一个用于求解Graphon上变分问题的神经探索框架，其中Graphon是指作为稠密图序列极限而出现的对称可测函数 $[0,1]^2 \to [0,1]$。此类问题在数学文献的两个领域中处于核心地位：其一是极值图论，它研究在给定约束下的图参数优化问题，尤其针对顶点数量巨大的图；其二是稠密随机图的大偏差理论，它通过受约束的Graphon优化来刻画稀有事件的结构。我们采用隐式神经网络来表示Graphon，并通过梯度下降来优化以Graphon为变量的目标函数。我们的设计结合了三个要素：一种偏向于尖锐、阶跃状Graphon的多尺度正弦残差架构；一个内嵌的求解器，用于强制施加单一经验密度约束，并通过隐式微分进行求导；以及对对称性敏感的蒙特卡洛估计器。

    arXiv:2610.32147v1 Announce Type: cross  Abstract: We propose a neural exploratory framework for variational problems over graphons, i.e., symmetric measurable functions $[0,1]^2 \to [0,1]$, that arise as limits of dense graph sequences. Such problems are central to two parts of the mathematical literature: extremal graph theory, which studies graph parameter optimisation under given restrictions, often for those graphs with a large number of vertices, and the large deviation theory of dense random graphs, which characterises the structure of rare events through constrained graphon optimisation. We represent graphons by implicit neural networks and optimise graphon objectives by gradient descent. Our design combines three ingredients: a multi-scale sinusoidal residual architecture biased toward sharp, step-like graphons; an embedded solver that enforces a single empirical density constraint and is differentiated by implicit differentiation; and symmetry-aware Monte Carlo estimators. On
    
[^20]: 离线最大熵强化学习的条件流模型

    Conditional Flow Models for Offline Maximum Entropy Reinforcement Learning

    [https://arxiv.org/abs/2610.32125](https://arxiv.org/abs/2610.32125)

    该论文提出一个分离式框架，利用基于概率流ODE的条件扩散模型分别估计奖励分布和转移核，从而构建软最优贝尔曼算子的数据驱动近似来估计最优软动作价值函数，并提供了非渐近理论保证。

    

    离线最大熵强化学习旨在从固定数据集中估计最优软动作价值函数 Q*，而无需与环境进行进一步的交互。尽管 Q* 由软最优贝尔曼方程刻画，但由于奖励分布和转移核均未知，相应的贝尔曼算子无法直接计算。因此，估计 Q* 需要同时对环境建模和价值函数学习进行联合处理。我们提出了一个将这两项任务分开处理的框架。基于概率流常微分方程（ODE）的条件扩散模型用于估计条件奖励分布和转移核。由此得到的分布估计器定义了软最优贝尔曼算子的数据驱动近似，通过在一类深度神经网络中最小化经验贝尔曼残差来估计 Q*。我们建立了非渐近的理论保证……

    arXiv:2610.32125v1 Announce Type: new  Abstract: Offline maximum entropy reinforcement learning seeks to estimate the optimal soft action-value function \(Q^*\) from a fixed dataset without further interaction with the environment. Although \(Q^*\) is characterized by the soft optimal Bellman equation, the corresponding Bellman operator cannot be evaluated directly because both the reward distribution and the transition kernel are unknown. Consequently, estimating \(Q^*\) requires the joint treatment of environment modeling and value function learning. We propose a framework that addresses these tasks separately. Conditional diffusion models based on probability flow ODEs estimate the conditional reward distribution and transition kernel. The resulting distributional estimators define a data driven approximation of the soft optimal Bellman operator, whose empirical Bellman residual is minimized over a class of deep neural networks to estimate \(Q^*\). We establish nonasymptotic guarant
    
[^21]: DBCM：用于时间序列预测测试时适应的双分支校准模块

    DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting

    [https://arxiv.org/abs/2610.32071](https://arxiv.org/abs/2610.32071)

    提出双分支校准模块DBCM，通过共享适配器利用跨通道预测残差的共享结构，并结合低秩通道分支保留有限的通道特定灵活性，实现时间序列预测中更高效的测试时适应。

    

    非平稳性会在时间序列预测（TSF）中引发测试时的分布偏移，导致预测结果与相应真实值之间产生预测残差。近期关于时间序列预测测试时适应的工作通过在冻结的预测器周围引入轻量级校准模块来解决这一问题，这些模块利用测试时逐步获得的观测数据进行更新。现有方法通常采用逐通道适配器，隐含地假设预测残差本质上是通道特定的。然而，我们观察到预测残差在通道之间往往表现出大量的共享结构，这表明完全独立的逐通道参数化可能是冗余的。受此观察启发，我们提出了双分支校准模块（DBCM），它通过共享适配器在通道之间共享适应能力，同时通过低秩通道分支保留有限的通道特定灵活性。

    arXiv:2610.32071v1 Announce Type: cross  Abstract: Non-stationarity induces test-time distribution shifts in time series forecasting (TSF), resulting in forecast residuals between predictions and the corresponding ground truth. Recent works on test-time adaptation for TSF address this by introducing lightweight calibration modules around frozen forecasters, updated using observations that become incrementally available at test time. Existing approaches typically employ channel-wise adapters, implicitly assuming that forecast residuals are inherently channel-specific. However, we observe that forecast residuals often exhibit substantial shared structure across channels, suggesting that fully independent channel-wise parameterization may be redundant. Motivated by this observation, we propose the Dual-Branch Calibration Module (DBCM), which shares adaptation capacity across channels through a shared adapter while retaining limited channel-specific flexibility through a low-rank channel-w
    
[^22]: 重新思考长尾数据下监督对比学习中的表示几何

    Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data

    [https://arxiv.org/abs/2610.32039](https://arxiv.org/abs/2610.32039)

    本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。

    

    监督对比学习及其平衡变体已在长尾分布下的表示学习中展现了其有效性。然而，尽管先前的分析表明其最优表示可能呈现正则单纯形几何结构，但这种几何结构是否确实存在仍然缺乏充分的理解。在这项工作中，我们提出了一个理论框架，用于刻画这些目标函数在最优解处是否形成正则单纯形。通过将该框架应用于代表性的目标函数，我们表明这些目标函数在最优解处无法形成正则单纯形，因为它们仍然依赖于每个类别的样本数量。受这一分析的启发，我们引入了可应用于对比目标的结构平衡方法，并提出了一种完全平衡的对比目标，该目标可证明地在其最优解处形成正则单纯形几何结构。理论和实证结果均表……（摘要在此处截断）

    arXiv:2610.32039v1 Announce Type: cross  Abstract: Supervised contrastive learning and its balanced variants have demonstrated their effectiveness for representation learning under long-tailed distributions. However, although prior analyses suggest that their optimal representations may exhibit a regular simplex geometry, whether such a geometry is indeed exhibited remains insufficiently understood. In this work, we present a theoretical framework for these objectives that characterizes whether a regular simplex is formed at the optimum. By applying this to representative objectives, we show that these objectives fail to form a regular simplex at the optimum because they remain dependent on the number of samples per class. Motivated by this analysis, we introduce structural balancing that can be applied to contrastive objectives, and propose a fully balanced contrastive objective that provably forms a regular simplex geometry at its optimum. Both theoretical and empirical results demon
    
[^23]: 流形上流匹配的相容似然

    Compatible Likelihoods for Flow Matching on Manifolds

    [https://arxiv.org/abs/2610.31981](https://arxiv.org/abs/2610.31981)

    该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。

    

    在最简单的非欧几里得流形——一维平坦环面上，我们展示了一个分布，其中最近提出的黎曼流匹配扩展方法被一个将样本推离数据分布的速度场全局优化。我们将这种失败模式追溯到一个我们称之为“相容性”的性质：如果一个基于似然的流匹配目标的总体最小化器能够恢复由数据和耦合所诱导的边际速度场，则称该目标是相容的。我们给出了相容性的一般条件，将其应用于现有方法，并通过在相容性条件下消失的残差分量来界定最坏情况误差。此外，我们注意到现有的相容目标将概率密度置于紧流形上不可行的速度上。为此，我们提出了一个基于端点上指数族构建的目标，以条件速度作为充分统计量，该目标既满足相容性，又仅支持可行的速度。

    arXiv:2610.31981v1 Announce Type: cross  Abstract: On the simplest non-Euclidean manifold, the 1D flat torus, we exhibit a distribution where a recently proposed extension of Riemannian flow matching is globally optimized by a velocity field that pushes samples away from the data distribution. We trace this failure mode to a property we call compatibility; a likelihood-based flow matching objective is compatible if its population minimizer recovers the marginal velocity field induced by the data and coupling. We give a general condition for compatibility, apply it to existing methods, and bound the worst-case error in terms of a residual component vanishing under compatibility. Next, we note that existing compatible objectives put density on velocities that are infeasible on compact manifolds. We propose an objective built from an exponential family over endpoints, with the conditional velocity as sufficient statistic, which is both compatible and supported only on feasible velocities.
    
[^24]: 贝叶斯倾向得分预测推断：将试验结果推广至目标人群

    Bayesian Propensity Score Predictive Inference for Extending Trial Findings to Target Populations

    [https://arxiv.org/abs/2610.31915](https://arxiv.org/abs/2610.31915)

    本文提出贝叶斯倾向得分预测推断（PSPI）框架，结合贝叶斯加性回归树与倾向得分自然三次样条，解决了当治疗效果存在效应修饰因素异质性时随机试验结果向目标人群推广的难题。

    

    arXiv:2610.31915v1 公告类型：交叉 摘要：CODIACS Vanguard随机对照试验比较了抑郁护理干预与常规护理对急性冠脉综合征后持续存在抑郁症状患者的疗效。我们评估其研究结果能否推广到由All of Us电子健康记录构建的两个队列所代表的更广泛患者群体。当治疗效果随协变量变化，而这些协变量在试验人群与目标人群中的分布存在差异时，这种推广具有挑战性。我们提出了贝叶斯倾向得分预测推断（PSPI），这是一种基于交互作用的结果建模框架，将用于潜在高维协变量的贝叶斯加性回归树与抽样倾向得分的自然三次样条相结合。该方法包含两种变体：一种仅在治疗效果成分中单独使用样条，另一种则在预后成分和治疗效果成分中均使用样条。样条成分允许对所选……

    arXiv:2610.31915v1 Announce Type: cross  Abstract: The CODIACS Vanguard randomized trial compared a depression care intervention with usual care for patients with persistent depressive symptoms following acute coronary syndrome. We assess whether its findings can be extended to broader patient populations represented by two cohorts constructed from the All of Us electronic health records. This extension is challenging when treatment effects vary across covariates whose distributions differ between trial and target populations. We develop Bayesian Propensity Score Predictive Inference (PSPI), an interaction-based outcome modeling framework that combines Bayesian additive regression trees for potentially high-dimensional covariates with natural cubic splines of the sampling propensity score. Two variants include a separate spline in the treatment effect component alone or in both the prognostic and treatment effect components. The spline components allow separate regularization of select
    
[^25]: 条件分布可学习性的组合刻画

    Combinatorial Characterizations of the Learnability of Conditional Distributions

    [https://arxiv.org/abs/2610.31890](https://arxiv.org/abs/2610.31890)

    本文引入fat_γ Natarajan维数和fat_γ Graph维数两个组合维度，证明前者在每个正尺度上的有限性是条件分布函数类PAC可学习性的充要条件，并证明分布值函数类可学习当且仅当其每个坐标类均可学习。

    

    我们研究将输入映射到有限空间上概率分布的函数类的PAC可学习性。学习者观察独立同分布的输入-标签对，其中每个标签从该类中未知目标假设的条件分布中采样得到。我们通过在满足两个温和假设的有界正常损失下的超额风险来衡量学习效果。我们引入了fat_γ Natarajan维数，它同时推广了fat_γ维数和Natarajan维数，并证明其在每个正尺度上的有限性是可学习性的充要条件。我们还引入了fat_γ Graph维数，它是经典Graph维数的推广，并建立了在不同尺度上比较这两个维数的不等式，从而得到可学习性的等价刻画。最后，我们证明一个取值为分布的函数类是可学习的，当且仅当它的每个坐标类都是可学习的。

    arXiv:2610.31890v1 Announce Type: new  Abstract: We study the PAC-learnability of function classes that map inputs to probability distributions over a finite space. The learner observes i.i.d. input-label pairs, with each label sampled from the conditional distribution of an unknown target hypothesis in the class. We measure learning through excess risk under a bounded proper loss that satisfies two mild assumptions. We introduce the \textbf{fat$_\gamma$Natarajan dimension}, which generalizes both fat$_\gamma$ and Natarajan dimension, and prove that its finiteness at every positive scale is necessary and sufficient for learnability. We also introduce a \textbf{fat$_\gamma$Graph} dimension extending the classical \gra dimension and establish inequalities comparing the two dimensions at different scales, leading to an equivalent characterization of learnability. Finally, we show that a distribution-valued function class is learnable if and only if each of its coordinate classes is learna
    
[^26]: 大型残差网络中的早期学习

    Early learning in large residual networks

    [https://arxiv.org/abs/2610.31783](https://arxiv.org/abs/2610.31783)

    该论文通过图表计算方法分析了大型残差网络在早期训练阶段的动态，发现在大输入维度极限下，网络在训练初期只能学习到输入的线性函数，无法学习更高阶的单项式。

    

    我们研究了由零隐藏层块构成的非线性残差网络的早期训练动态，并将其作为更一般残差网络的模型。我们在一个顺序极限下分析网络：首先让宽度和深度发散，然后让训练数据点数量发散，最后让输入维度发散。与典型的平均场理论分析（其中深度和宽度发散，但训练数据点数量和输入维度保持固定）相比，这是一个更精细的设置。我们的主要技术成果是一种图表计算方法，用训练集中输入内积的统计量来刻画梯度下降几步之后的网络输出和隐藏表示。我们的一个主要概念性发现是一个负面结果：在大输入维度极限下，网络输出在训练早期只能学习其输入的线性函数，因为更高阶的学习单项式……（摘要原文在此处被截断）

    arXiv:2610.31783v1 Announce Type: new  Abstract: We study the early training dynamics of nonlinear residual networks with blocks of zero hidden layers as a model of more general residual networks. We analyze networks in a sequential limit where first width and depth diverge, then the number of training data points diverges, and finally the input dimension diverges. This is a more refined setup compared to the typical mean-field theory analysis in which depth and width diverge but the number of training datapoints and input dimension are fixed. Our main technical result is a diagrammatic computation of both the network output and the hidden representations after the few steps of gradient descent in terms of the statistics of inner products of inputs from the training set. One of our principle conceptual findings is a negative result: in the limit of large input dimension the network output can learn only a linear function of its inputs early in training since higher learned monomials ar
    
[^27]: 学会被监测：面向思维链可监测性的因果界与干预感知的后训练方案

    Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2610.31733](https://arxiv.org/abs/2610.31733)

    该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。

    

    思维链（CoT）监测旨在从大型语言模型（LLM）的思维链中检测其潜在的有害行为。基于干预的评估方法通过检验“当目标行为发生时，监测器能否从思维链中识别出干预的作用”来衡量这一属性，通常采用 Guan 等人（2025）提出的最低标准敏感度（$\mathsf{TPR_{mc}}$）指标。然而，在干预下观察到的目标行为并不一定是由干预引起的，我们证明该指标可能会高估监测器对“实际由干预引起的目标行为”的检出频率。具体而言，我们将 $\mathsf{TPR_{mc}}$ 确立为该因果检测率的上界，并推导出一个互补的下界，从而为 CoT 可监测性提供了更严格的评估框架。借助该框架，我们研究了能否通过训练使 LLM 的思维链变得更易于监测。我们首先构建了一个训练（后文截断）

    arXiv:2610.31733v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring aims to detect potentially harmful behaviors in large language models (LLMs) from their CoTs. Intervention-based evaluations measure this property by asking whether a monitor can detect the role of an intervention from the CoT when the targeted behavior occurs, typically using minimal-criterion sensitivity ($\mathsf{TPR_{mc}}$) proposed by~\citep{guan2025monitoring}. However, a target behavior observed under intervention is not necessarily caused by it, and we show that this metric can overestimate how often a monitor detects target behaviors actually caused by the intervention. Specifically, we establish $\mathsf{TPR_{mc}}$ as an upper bound on this causal detection rate and derive a complementary lower bound, which provides a tighter evaluation framework for CoT monitorability. With this framework, we study whether LLMs can be trained to make their CoTs easier to monitor. We first construct a trainin
    
[^28]: 操作化较弱忠实性：恢复、歧义与计算极限

    Operationalising Weaker Faithfulness: Recovery, Ambiguity and Computational Limits

    [https://arxiv.org/abs/2610.31724](https://arxiv.org/abs/2610.31724)

    本文提出两种操作化弱忠实性的策略——邻接学习中的选择性边保护与显式保留备选有向图解释（解释家族），使结构学习器能够恢复单独独立但联合预测结果的变量（如XOR）之间的联系，将完整父对恢复率从0/20提升至20/20，且家族估计器在48/50的样本中保留了全部备选解释。

    

    变量可以联合预测某个结果，而每个变量单独却与该结果相互独立，这会导致结构学习器丢弃含有信息量的关联。我们研究如何通过两种途径来操作化较弱忠实性：一是在邻接学习器中进行选择性保护，二是显式保留备选的有向图解释。匹配实验表明，早期候选移除阶段与后期剪枝阶段两处的改动都可能是必要的。在带后代节点的噪声XOR实验中（N=10,000，20个匹配样本），将重新激活与保护相结合，使完整父节点对的恢复率从单独采用任一改动时的0/20提升至20/20，但同时会引入假边。一项验证性研究显示，家族估计器能够保留因对受保护学习器所选骨架进行定向而丢失的解释：在一个较弱信号的XOR条件下（N=10,000），它在48/50个样本中保留了完整的参考解释家族，而骨架自适应[估计器]仅为0/50。

    arXiv:2610.31724v1 Announce Type: cross  Abstract: Variables can predict an outcome jointly while each is individually independent of it, causing structure learners to discard informative links. We investigate how to operationalise weaker faithfulness through two approaches: selective protection in an adjacency learner and explicit retention of alternative directed graph explanations. Matched experiments show that changes at both an early candidate-removal stage and later pruning can be necessary. In noisy XOR with a descendant (N = 10,000, 20 matched samples), combining reactivation and protection raises complete parent-pair recovery from 0/20 for either change alone to 20/20, while adding false edges. A confirmation study shows that the family estimator can preserve explanations lost by orienting the protected learner's selected skeleton: in one weaker signal XOR condition (N = 10,000) it retains the entire reference family in 48/50 samples, compared with 0/50 for the skeleton adapte
    
[^29]: 自回归物理信息神经算子代理模型与四维地震-生产数据联合贝叶斯历史拟合

    Autoregressive Physics-Informed Neural Operator Surrogates and Joint 4-D Seismic-Production Bayesian History Matching

    [https://arxiv.org/abs/2610.31677](https://arxiv.org/abs/2610.31677)

    该论文提出了一种以自回归物理信息傅里叶神经算子作为代理模型加速的贝叶斯历史拟合框架，通过两阶段 aREKI 与 ES-MDA 策略联合同化生产数据与四维地震数据，显著降低了大规模油藏反演的计算成本与病态性。

    

    我们提出了一种由代理模型加速的贝叶斯历史拟合框架，并在一个真实感的合成模型（N=100,188个网格单元）上进行了验证。该框架将正演映射 G 替换为在闭环中训练的傅里叶神经算子 G_θ，即 x̂_{n+1} = G_θ(u, x̂_n)，并以离散黑油方程残差 R(x_{n+1}, x_n) 作为物理信息神经算子（PINO）惩罚项，同时采用截断反向传播与噪声注入技术。我们利用自适应正则化集合卡尔曼反演（aREKI）同化生产数据与四维地震数据，并通过代理模型的闭式 Gassmann 变换在滚动预测过程中合成 ΔV_p，采用两阶段调度方案（先对生产数据执行 aREKI，再对两类数据同时使用多数据同化集合平滑器 ES-MDA）。为缓解问题的病态性，我们并非在 R^{2N} 中直接反演 (log K, φ)，因为当 N_e << 2N 时会导致协方差矩阵秩亏并产生零空间……

    arXiv:2610.31677v1 Announce Type: new  Abstract: We present a surrogate-accelerated Bayesian history-matching framework, demonstrated on a realistic synthetic model ($N=100{,}188$ cells). The forward map $\mathcal{G}$ is replaced by a Fourier neural operator $\mathcal{G}_\theta$ trained in a closed loop, $\hat{x}_{n+1}=\mathcal{G}_\theta(u,\hat{x}_n)$, with the discrete black-oil residual $R(x_{n+1},x_n)$ as a physics-informed neural operator (PINO) penalty, truncated backpropagation and noise injection. We assimilate production data and 4-D seismic using adaptive regularised ensemble Kalman inversion (aREKI), and synthesise $\Delta V_p$ inside the rollout via a closed-form Gassmann transform of the surrogate, under a two-stage schedule (aREKI on production, then an ensemble smoother with multiple data assimilation, ES-MDA, on both). To temper the ill-posedness, we invert $(\log K,\phi)$ not in $\mathbb{R}^{2N}$, where $N_e\ll 2N$ leaves a rank-deficient covariance and a null space the
    
[^30]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^31]: 基于广义矩匹配的差分隐私合成数据

    Differentially Private Synthetic Data via Generalized Moment Matching

    [https://arxiv.org/abs/2610.30196](https://arxiv.org/abs/2610.30196)

    提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。

    

    我们研究了在指定的积分概率度量（IPM）下，针对[-1,1]^d中包含n条记录的任意数据库的(ε,δ)-差分隐私合成数据生成问题。我们提出了广义矩匹配方法，该方法私有地估计矩、拟合概率测度并采样合成记录。当矩误差能够控制目标IPM时，可应用一个一般性误差定理。该框架扩展了针对1-Wasserstein距离和光滑查询的切比雪夫矩匹配方法。我们将该框架应用于采用Matern核、Laplace核以及两层ReLU神经正切核的最大均值差异（MMD），还获得了切片Wasserstein-1以及由范数受限ReLU网络诱导的IPM的保证。在合成记录数量足够多的情况下，对于固定维度，这些误差率在相差对数因子的意义下是极小极大最优的。

    arXiv:2610.30196v1 Announce Type: cross  Abstract: We study $(\varepsilon,\delta)$-differentially private synthetic data for arbitrary databases of $n$ records in $[-1,1]^d$ under a specified integral probability metric (IPM). We introduce generalized moment matching, which estimates moments privately, fits a probability measure, and samples synthetic records. A general error theorem applies when moment error controls the target IPM. This framework extends Chebyshev moment matching for $1$-Wasserstein distance and smooth queries. We apply the framework to maximum mean discrepancy with Matern and Laplace kernels and a two-layer ReLU neural tangent kernel.We also obtain guarantees for sliced Wasserstein-$1$ and the IPM induced by norm-bounded ReLU networks. With sufficiently many synthetic records,these rates are minimax optimal up to logarithmic factors for fixed dimension.
    
[^32]: 非平稳多臂老虎机中现时效应推断的代价

    The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits

    [https://arxiv.org/abs/2610.29982](https://arxiv.org/abs/2610.29982)

    该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。

    

    自适应实验利用老虎机算法来偏好具有更高期望收益的干预措施。然而，决策者还需要对现时处理效应进行可靠的推断：即在当前时刻干预措施与参照臂平均收益之间的差异，而非历史平均。这两个目标相互冲突：收益最大化倾向于选择更优的干预，而现时效应推断则要求随着结果随时间变化，持续探索可能次优的干预。我们针对平均收益具有指数 β∈(0,1] 和常数 L 的 Hölder 平滑性的 K 臂老虎机刻画了这一权衡。在要求置信区间以不超过 w 的宽度、以高概率同时覆盖所有现时效应的条件下，我们在期望动态遗憾上建立了（在对数因子范围内）匹配的极小极大界。在时间跨度 T 内，我们的算法 TRACE（Tracking Rewards and Current Effects）实现了遗憾……（原文摘要至此截断）

    arXiv:2610.29982v1 Announce Type: cross  Abstract: Adaptive experiments use bandit algorithms to favor interventions with higher expected rewards. However, decision-makers also require reliable inference on current treatment effects: differences between intervention and reference-arm mean rewards at the present time, rather than historical averages. These objectives conflict: reward maximization favors better interventions, whereas current-effect inference requires continued exploration of potentially suboptimal interventions as outcomes change over time. We characterize this tradeoff for $K$-arm bandits with H\"older-smooth mean rewards of exponent $\beta\in(0,1]$ and constant $L$. Requiring high-probability simultaneous coverage of all current effects by confidence intervals of width at most $w$, we establish matching minimax bounds, up to logarithmic factors, on expected dynamic regret. Over horizon $T$, our algorithm TRACE (Tracking Rewards and Current Effects) achieves regret of o
    
[^33]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^34]: 未知边缘分布下严格可复制可实现学习的二次样本复杂度

    Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals

    [https://arxiv.org/abs/2610.26944](https://arxiv.org/abs/2610.26944)

    该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。

    

    在量词∃A∀D的框架下，即可实现分布及其边缘分布均未知的情况下，我们确定了严格可复制性（即两个独立样本和一个共享随机种子必须产生完全相同的函数）的有限类样本复杂度指数。当ε=δ=ρ=10⁻⁴时，在|H|≤M上的最坏情况复杂度为Θ((log M)²)，且不含迭代对数项。对于d≥1、0<ε≤1/16以及0<δ,ρ≤10⁻³，任何在整个显式锚定可实现类上有效的随机学习器（即使允许非proper学习）都需要Ω(d²/(ερ²))个样本。对于完整布尔类P_d（d≥2），在相同的参数范围内，我们借助一个proper的泊松化球采样算法证明了匹配的刻画结果Θ(d²/(ερ²)+ε⁻¹log(1/δ))。此外，我们还为每个有限可测布尔类给出了显式的proper Gibbs算法上界。

    arXiv:2610.26944v1 Announce Type: cross  Abstract: Under the quantifier $\exists A\,\forall D$, where the realizable distribution and its marginal are unknown, we determine the finite-class sample-complexity exponent for strict replicability: two independent samples and one shared seed must produce identical functions. For $\varepsilon=\delta=\rho=10^{-4}$, the worst case over $|H|\le M$ is $\Theta((\log M)^2)$, without iterated logarithms. For $d\ge1$, $0<\varepsilon\le1/16$, and $0<\delta,\rho\le10^{-3}$, every randomized, possibly improper learner valid throughout an explicit anchored realizable family requires $\Omega(d^2/(\varepsilon\rho^2))$ samples. For the full Boolean class $P_d$, $d\ge2$, in the same parameter range, we prove the matching characterization $\Theta(d^2/(\varepsilon\rho^2)+\varepsilon^{-1}\log(1/\delta))$ using a proper Poissonized ball-sampling algorithm. We also give an explicit proper Gibbs upper bound for every finite measurable Boolean class. Our lower boun
    
[^35]: SpectrumQuant：面向大语言模型量化的频谱对齐方法

    SpectrumQuant: Spectrum Alignment for LLM Quantization

    [https://arxiv.org/abs/2610.20303](https://arxiv.org/abs/2610.20303)

    提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。

    

    正交变换已成为大语言模型（LLM）低比特量化的有效工具，因为它能够在不改变底层模型函数的情况下重塑激活分布。现有方法通常依赖于固定的、与数据无关的变换（如Hadamard旋转），或者依赖于在校准阶段对可学习旋转的优化。我们从第一性原理出发研究旋转选择问题，并证明最大化协方差可以带来频谱对齐，从而为构建量化友好的正交变换提供了一个有理论依据的准则。基于这一洞察，我们提出了SpectrumQuant，这是一种闭式解的旋转方法，它从校准数据中估计激活协方差，并利用它构建协方差感知的正交变换以用于训练后量化。我们的公式化将量化友好的旋转刻画为对齐分量与混合分量的组合，同时将Hadamard旋转作为特例恢复出来（此句原文被截断）。

    arXiv:2610.20303v1 Announce Type: new  Abstract: Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rotations, or on calibration-time optimization of learned rotations. We study rotation selection from first principles and show that maximizing covariance leads to spectral alignment and provides a principled criterion for constructing quantization-friendly orthogonal transforms. Based on this insight, we propose SpectrumQuant, a closed-form rotation method that estimates activation covariance from calibration data and uses it to construct a covariance-aware orthogonal transform for post-training quantization. Our formulation characterizes quantization-friendly rotations as compositions of alignment and mixing components, while recovering Hadamar
    
[^36]: 协变量偏移下的潜在空间重要性加权

    Latent-Space Importance Weighting under Covariate Shift

    [https://arxiv.org/abs/2610.15571](https://arxiv.org/abs/2610.15571)

    该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。

    

    在本工作中，我们提出了一种针对协变量偏移的迁移学习算法，将样本重加权与表示学习相结合。我们方法的一个关键特征是使用低维潜在空间进行建模，这与直接在高维环境空间中操作的传统方法形成对比。我们的方法通过估计学习到的潜在特征的密度比来进行下游的加权预测。对于具有正则学习空间预测分数和密度比的严格充分学习表示，我们利用独立的数据划分建立了明确的目标域超额风险率。我们在给定的平滑性条件下给出了维度比较，并对真实权重和固定预测器进行了方差比较。一般的有限样本不等式保留了表示质量误差。在模拟数据集和真实世界数据集上的实验表明，结合学习到的表示与潜在加权能够带来性能提升。

    arXiv:2610.15571v1 Announce Type: new  Abstract: In this work, we propose a transfer learning algorithm for covariate shift, integrating sample reweighting with representation learning. A key feature of our approach is the use of a lower-dimensional latent space for modeling, in contrast to conventional methods that operate directly in the high-dimensional ambient space. Our method estimates the density ratio of learned latent features for downstream weighted prediction. We establish explicit target-domain excess-risk rates with independent data folds for strictly sufficient learned representations with regular learned-space prediction scores and density ratios. We give dimension comparisons under stated smoothness conditions and a variance comparison for true weights and a fixed predictor. General finite-sample inequalities retain representation-quality errors. Experiments on simulated and real-world datasets demonstrate gains from combining learned representations and latent weightin
    
[^37]: 面向处理效应估计的预测驱动数据融合方法

    Prediction-Powered Data Fusion for Treatment Effect Estimation

    [https://arxiv.org/abs/2610.12332](https://arxiv.org/abs/2610.12332)

    提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。

    

    随机对照试验（RCT）能够在无混杂的情况下识别处理效应，但样本量通常较小；而观察性研究（OBS）样本量大，但可能存在混杂。目前已有许多将小型RCT与大型OBS相结合的估计方法用于估计平均处理效应（ATE）和条件平均处理效应（CATE）。然而，现有的ATE估计方法要么对OBS施加额外假设，要么未能从OBS中充分借力。相比ATE，CATE的研究相对较少，现有的CATE方法要么假设OBS无混杂，要么依赖于混杂函数的模型，要么以引入偏差为代价来换取更低的方差。为此，我们提出了一个框架，在对OBS不做任何特殊假设的前提下融合OBS与RCT：既保持基于RCT估计的无偏性，又从大型OBS中借力以提升估计精度。基于这一原则，我们构建了具有闭式权重的ATE估计器AIPW-Fusion……

    arXiv:2610.12332v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) identify treatment effects without confounding but are often small, whereas observational studies (OBS) are large but may be confounded. Many estimators combining a small RCT with a large OBS have been developed for the average treatment effect (ATE) and the conditional ATE (CATE). However, existing ATE estimators either make assumptions on the OBS or do not borrow enough power from them. The CATE has been studied less than the ATE. Existing CATE methods either assume the OBS are unconfounded, rely on a model of the confounding function, or accept bias in exchange for lower variance. We therefore propose a framework that, without special assumptions on the OBS, fuses the OBS and the RCT by preserving the unbiasedness of RCT-based estimation while borrowing power from the large OBS to boost precision. Applying this principle, we build an ATE estimator, AIPW-Fusion, with closed-form weights and confide
    
[^38]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^39]: 带探测的多臂老虎机：最优遗憾与赢家反馈的极限

    Bandits with Probing: Optimal Regret and the Limits of Winner Feedback

    [https://arxiv.org/abs/2609.15248](https://arxiv.org/abs/2609.15248)

    本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。

    

    学习者在每轮最多探测 n 个臂中的 k 个，接收它们在 [0,1] 区间内奖励的最大值，并与最优固定臂进行竞争。那么，探测的优势何时能为学习“买单”？我们确定了两个极小极大定律。在具有赢家反馈（最大奖励值及获胜臂标签）的独立随机奖励下，或者在给定块最大值之间单个带符号对比的任意固定序列下，极小极大遗憾的阶为 Φ_{n,k}(T)=min{(n-k)T/n, (n-k)/k}，其中 2≤k<n。在赢家反馈下，任意联合独立同分布奖励和固定序列的极小极大遗憾阶均为 R_{n,k}(T)=(n-k)/n · min{T, (n+T)/k, √(nT/k)}。这两个定律均包含普适常数和任意时刻有效上界。第一个定律将遗憾归结为纯粹的覆盖成本：同轮对比吸收了稳定性成本，而独立性允许精确重采样，其收益用于推动样本的进一步获取。第二个定律则额外引入了一个学习成本……

    arXiv:2609.15248v1 Announce Type: new  Abstract: A learner probes at most $k$ of $n$ arms each round, receives the maximum of their rewards in $[0,1]$, and competes with the best fixed arm. When does the probing advantage pay for learning? We determine two minimax laws. Under independent stochastic rewards with winner feedback (the maximum and a winning label), or on arbitrary fixed sequences given a single signed contrast between block maxima, the minimax regret has order $\Phi_{n,k}(T)=\min\{\frac{n-k}{n}T,\frac{n-k}{k}\}$, $2\le k<n$. Under winner feedback, both arbitrary joint i.i.d. rewards and fixed sequences have minimax regret of order $R_{n,k}(T)=\frac{n-k}{n}\min\{T,\frac{n+T}{k},\sqrt{\frac{nT}{k}}\}$. Both laws have universal constants and anytime upper bounds. The first reduces regret to a pure coverage cost: same-round contrasts absorb the stability cost, and independence permits exact resampling whose gains fund sample advancement. The second adds a learning cost that be
    
[^40]: 大批量训练中的动量：Polyak 动量扩大临界批大小，Nesterov 动量提升数据效率

    Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency

    [https://arxiv.org/abs/2609.02728](https://arxiv.org/abs/2609.02728)

    该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。

    

    我们在单遍（one-pass）训练机制下研究动量何时以及如何改善大批量训练，并以幂律核回归作为一个易于解析的设定。我们首先通过临界学习率刻画风险的稳定性，临界学习率定义为保证训练稳定的最大学习率，并得到 $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$、$\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$ 以及 $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$，其中 $B$ 为批大小，$\rho$ 为动量因子，$\beta>1$ 为容量指数。在该允许区域内，我们推导出完整风险动力学的标度律，刻画了训练从早期瞬态阶段、经幂律衰减、直至噪声底限的完整演化过程。随后，在固定数据预算下，我们在允许的学习率与动量因子范围内最小化最后一步风险，得到了一个包含三个区制的批大小相图，揭示了（原文摘要在此处截断）

    arXiv:2609.02728v1 Announce Type: cross  Abstract: We study when and how momentum improves large-batch training in the one-pass regime, using power-law kernel regression as a tractable setting. We first characterize risk stability through the critical learning rate, defined as the largest learning rate for stable training, and obtain $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$, $\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$, and $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$, where $B$ is the batch size, $\rho$ is the momentum factor, and $\beta>1$ is the capacity exponent. Within this admissible region, we derive scaling laws for the full risk dynamics, capturing the progression from an early transient, through power-law decay, to a noise floor. We then minimize the final-step risk over the admissible learning rates and momentum factors under a fixed data budget, yielding a three-regime batch-size phase diagram that reveals how the rol
    
[^41]: 参考集重采样下OOD分数的判定不稳定性

    Verdict Instability of OOD Scores under Reference Resampling

    [https://arxiv.org/abs/2609.00691](https://arxiv.org/abs/2609.00691)

    本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。

    

    事后（post-hoc）分布外（OOD）检测器是在有限的参考集上拟合的，因此它们产生的每一个分数都只是一个估计值。如果我们选择了另一个参考集，某些判定结果就会发生改变。我们通过重采样参考集并记录分数的自助法（bootstrap）标准差来度量这种变动，并将其称为“判定不稳定性”。该量具有无需拟合任何参数的闭式解。一个判定的不稳定性等于所分配类别沿查询方向的类内离散度除以该类别参考样本数的平方根。正是这一样本数将判定不稳定性与分数分布的几何结构区分开来，且只有在类别不平衡的情况下才能将其识别出来。不稳定性随局部离散度的增大而增长。远分布外（Far-OOD）查询位于各向异性嵌入的低方差方向上，因此在我们测试的所有基于距离的分数中，最高分数值都被分配给了那些最可复现的判定。只有es…（摘要不完整）

    arXiv:2609.00691v1 Announce Type: new  Abstract: Post-hoc out-of-distribution detectors are fitted on a finite reference set, so every score they produce is an estimate. If we had chosen a different set, some verdicts would have moved. We measure that movement by resampling the reference set and recording the bootstrap standard deviation of the score, which we call verdict instability. It admits a closed form with no fitted parameters. The instability of a verdict is the within-class dispersion of the assigned class along the query's direction, divided by the square root of that class's reference count. That count is what separates verdict instability from the geometry of the score distribution, and it is identifiable only under class imbalance. Instability grows with the local dispersion. Far-OOD queries lie along the low-variance directions of an anisotropic embedding, so every distance-based score we test assigns its highest values to the verdicts that are most reproducible. Only es
    
[^42]: 泛函流匹配的离散化与统计一致性

    Discretization and Statistical Consistency of Functional Flow Matching

    [https://arxiv.org/abs/2608.04531](https://arxiv.org/abs/2608.04531)

    本文证明了在有限观测细化（允许非嵌套）且一致有界的条件下，泛函流匹配的速度目标依均方收敛，并量化了观测、学习与数值误差对生成分布的影响，为其函数空间模型的统计一致性提供了理论保证。

    

    泛函流匹配通过将一个简单的概率分布连续变换为目标分布，学习生成随机函数，例如速度场或偏微分方程的解。然而在实践中，训练和采样仅使用有限个网格值或系数。那么，细化这些观测能否恢复预期的函数空间模型？观测误差、学习误差和数值误差又如何影响生成的分布？我们通过以下方式回答这些问题：首先证明，当有限重构收敛于底层函数并保持一致有界时，用于训练的速度目标依均方收敛。这些观测不必是嵌套的：每次细化都可以替换先前的传感器位置。我们还给出了正交投影的定量界，并在适当的正则性与重构条件下将结果推广到点观测情形。

    arXiv:2608.04531v2 Announce Type: replace-cross  Abstract: Functional flow matching learns to generate random functions, such as velocity fields or solutions of partial differential equations, by continuously transforming a simple probability distribution into a target distribution. In practice, however, training and sampling use only finitely many grid values or coefficients. Does refining these observations recover the intended function-space model, and how do observation, learning, and numerical errors affect the generated distribution? We address these questions by first proving that the velocity targets used for training converge in mean square when the finite reconstructions converge to the underlying functions and remain uniformly bounded. The observations need not be nested: each refinement may replace earlier sensor locations. We also give quantitative bounds for orthogonal projections and extend the result to point observations under suitable regularity and reconstruction ass
    
[^43]: 基于异方差神经网络学习双向因果交互

    Learning Bidirectional Causal Interactions with Heteroscedastic Neural Networks

    [https://arxiv.org/abs/2607.22313](https://arxiv.org/abs/2607.22313)

    提出无需外部工具变量的异方差神经联立方程估计器 SEM-DNN，通过条件协方差对角化实现观测数据中双向同期因果交互的唯一识别与非线性联合估计。

    

    从观测数据中估计同期双向交互作用是困难的，因为每个结果变量相对于另一方都是内生的，而灵活的回归可能只能捕捉到简化形式的依赖关系。本文提出了 SEM-DNN，一种异方差神经联立方程估计器，能够在没有外部工具变量的情况下学习互惠的结构性交互作用。其识别策略利用了条件协方差对角化：当结构冲击具有零条件均值、在给定前定协变量的条件下互不相关、且表现出非成比例的条件方差时，只有真实的交互系数才能在整个特征空间上使条件残差协方差对角化。该方法使用对角高斯拟似然来联合逼近非线性结构均值函数和依赖于特征的方差，该拟似然中融入了联立方程系统的雅可比矩阵。我们建立了唯一识别性……

    arXiv:2607.22313v2 Announce Type: replace  Abstract: Estimating contemporaneous bidirectional interactions from observational data is difficult because each outcome is endogenous to the other, while flexible regressions may capture only reduced-form dependence. This paper proposes SEM-DNN, a heteroscedastic neural simultaneous-equation estimator that learns reciprocal structural interactions without external instruments. Identification exploits conditional covariance diagonalization: when structural shocks have zero conditional means, are conditionally uncorrelated given predetermined covariates, and exhibit nonproportional conditional variances, only the true interaction coefficients diagonalize the conditional residual covariance across the feature space. The method jointly approximates nonlinear structural mean functions and feature-dependent variances using a diagonal Gaussian quasi-likelihood that incorporates the simultaneous-system Jacobian. We establish unique identification an
    
[^44]: 通过随机李雅普诺夫稳定性分析大步长随机梯度下降在交叉熵损失下的收敛性

    Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability

    [https://arxiv.org/abs/2606.30930](https://arxiv.org/abs/2606.30930)

    本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。

    

    arXiv:2606.30930v2 公告类型：替换 摘要：现代深度学习已被证明能够以远超经典优化理论所能解释的学习率运行。以往对这一现象的大多数分析都集中在确定性梯度下降上，而随机设置在很大程度上尚未被探索。在这项工作中，我们为应用于多分类交叉熵损失的随机梯度下降（SGD）提供了精确的收敛保证，涵盖了线性分类器和两层神经网络。我们表明，SGD的随机性可能导致其动力学在由曲率驱动的振荡所主导的“稳定边缘”区域与期望损失以可控速率下降的稳定区域之间交替。尽管如此，我们证明了SGD能够稳定动力学，确保迭代点在固定的迭代次数内回归稳定，从而即使在学习率很大的情况下也能实现最佳迭代意义上的收敛。实验验证了我们的理论结果。

    arXiv:2606.30930v2 Announce Type: replace  Abstract: Modern deep learning has been shown to operate using learning rates far larger than those justified by classical optimization theory. Most prior analyses of this phenomenon focus on deterministic gradient descent, leaving the stochastic setting largely unexplored. In this work, we provide sharp convergence guarantees for Stochastic Gradient Descent (SGD) applied to the multiclass cross-entropy loss, for both linear classifiers and two-layer neural networks. We show that the stochasticity of SGD may cause the dynamics to alternate between an edge-of-stability regime that is dominated by curvature-driven oscillations, and a stable regime in which the expected loss decreases at a controlled rate. Despite that, we prove that SGD stabilizes the dynamics, ensuring that the iterates return to stability in a fixed number of iterations and allowing convergence in the best-iterate sense even with large learning rates. Experiments validate our 
    
[^45]: 漂移与污染环境下基于潜在聚类几何的在线分布预测

    Online Distributional Prediction via Latent Cluster Geometry Under Drift and Corruption

    [https://arxiv.org/abs/2606.18778](https://arxiv.org/abs/2606.18778)

    该论文提出一种在漂移与对抗污染环境下基于潜在聚类几何配置的在线分布预测方法，利用Gibbs拟后验平均与可逆跳转MCMC进行预测，并给出了将PAC-贝叶斯复杂度、污染敏感后验扰动与漂移效应分离的累积Wasserstein-1遗憾界。

    

    非平稳数据流中的在线学习通常被表述为跟踪一个点估计，然而许多应用需要预测完整的数据生成分布。我们研究了在漂移和有界因果对抗污染下的在线分布预测问题。候选分布由可变大小的潜在聚类配置来表示，这些配置诱导出预测分布。通过在配置空间上构建Gibbs拟后验并进行后验平均，可以得到一个在线预测器，并可使用可逆跳转MCMC进行采样。该方法不依赖数据流的似然函数，而是指定了一个核混合预测族。我们针对时变的真实分布评估累积Wasserstein-1遗憾，并研究了周期性重启、滑动窗口和指数折扣记忆与全历史记忆的对比。所给出的界通过 $A_T^{\mathrm{OT}}=\sum_{t=2}^T W_2^...$ 将PAC-贝叶斯复杂度、污染敏感的后验扰动和漂移这三部分效应分离开来。

    arXiv:2606.18778v2 Announce Type: replace-cross  Abstract: Online learning in non-stationary streams is often formulated as tracking a point estimate, although many applications require predicting the full data-generating distribution. We study online distributional prediction under drift and bounded causal adversarial corruption. Candidate laws are represented by variable-size latent cluster configurations that induce predictive distributions. A Gibbs quasi-posterior over configurations yields an online predictor by posterior averaging and can be sampled using reversible-jump MCMC. The method uses no stream likelihood but specifies a kernel-mixture predictive family. We evaluate cumulative Wasserstein-1 regret against the time-varying true law and study periodic restart, sliding-window, and exponential-discounting memory against full history. The bounds separate PAC-Bayesian complexity, corruption-sensitive posterior perturbation, and drift through $A_T^{\mathrm{OT}}=\sum_{t=2}^T W_2^
    
[^46]: 级联梯度方法中的瞬态放大：精确极小极大定律与无共振跟踪

    Transient Amplification in Cascaded Gradient Methods: Exact Minimax Laws and Tracking without Resonance

    [https://arxiv.org/abs/2606.04031](https://arxiv.org/abs/2606.04031)

    该论文针对级联梯度方法首次给出精确且与维度无关的最坏瞬态放大极小极大公式，证明最坏瞬态为 1 + c/(1-γ) 量级、仅凭特征值无法预测有限时间增益，并在鞅噪声下确立了三次逆间隙缩放的跟踪方差定律。

    

    主从梯度迭代（包括双层优化和双时间尺度方法）尽管具有渐近稳定性，但仍可能出现较大的有限时间偏移。对于具有分块矩阵 J = [[A,0],[C,D]] 的级联迭代，其中对称对角块的范数至多为 γ < 1，耦合范数至多为 c，我们证明了一个精确的、与维度无关的极小极大公式：在任意时间范围 t，最坏情况的放大倍数等于矩阵 [[γ^t, 0], [c·t·γ^(t-1), γ^t]] 的谱范数，且该值可由一个对齐的二维二次问题达到。因此，最坏瞬态的量级为 1 + c/(1-γ)，且仅凭特征值无法预测有限时间增益。在鞅噪声下，对齐系统的平稳跟随者方差为 σ²c²(1+γ²)/(1-γ²)³，确立了三次逆间隙缩放规律。对于二次型跟随者，耦合与跟随者敏感度 R 通过 C = (I-D)R 相关联。当领导者为各向同性且跟随者映射为正……（原文摘要在此处截断）

    arXiv:2606.04031v3 Announce Type: replace-cross  Abstract: Leader-follower gradient iterations, including bilevel and two-time-scale methods, can exhibit large finite-time excursions despite asymptotic stability. For cascaded iterations with block matrix J = [[A,0],[C,D]], symmetric diagonal blocks of norm at most gamma < 1, and coupling norm at most c, we prove an exact, dimension-free minimax formula: at every horizon t, the worst-case amplification equals the spectral norm of [[gamma^t,0],[c t gamma^(t-1),gamma^t]], attained by an aligned two-dimensional quadratic. Consequently, the worst transient is on the order of 1 + c/(1-gamma), and eigenvalues alone cannot predict finite-time gain. Under martingale noise, the aligned system has stationary follower variance sigma^2 c^2 (1+gamma^2)/(1-gamma^2)^3, establishing cubic inverse-gap scaling. For quadratic followers, the coupling is linked to follower sensitivity R by C = (I-D)R. When the leader is isotropic and the follower map is pos
    
[^47]: 依赖实例的随机Lipschitz老虎机问题

    Instance-dependent Stochastic Lipschitz bandit

    [https://arxiv.org/abs/2605.29748](https://arxiv.org/abs/2605.29748)

    该论文提出了一种新的分析与算法，通过Lipschitz函数在其水平集上的次优性间隙积分来刻画遗憾，从而得到了真正依赖实例的遗憾界，能够适应水平集的局部增长结构而不仅仅是其渐近行为。

    

    我们研究Lipschitz老虎机问题，其中学习者通过带噪的点评估，在域 $\mathcal{X} \subset [0,1]^d$ 上顺序地最大化一个未知的Lipschitz函数 $f$。现有的遗憾界要么是最坏情况下的，其量级为 $\tilde{\Theta}(T^{(d+1)/(d+2)})$，要么是通过zooming维度 $d_z$ 实现自适应的，得到 $\tilde{\Theta}(T^{(d_z+1)/(d_z+2)})$。然而，这类基于zooming的保证只是部分依赖实例的，因为它们仅仅依赖于近最优水平集的渐近增长，无法捕捉 $f$ 更精细的结构性质。我们提供了一种分析和算法，通过 $f$ 在其水平集上的次优性间隙的积分来刻画遗憾。这产生了能够适应水平集局部增长的遗憾界，而不仅仅是其渐近行为。作为推论，当最大化子集合的维度为 $d^\star>0$ 时，我们得到了改进的遗憾界。

    arXiv:2605.29748v2 Announce Type: replace  Abstract: We study the Lipschitz bandit problem, where a learner sequentially maximizes an unknown Lipschitz function $f$ over a domain $\mathcal{X} \subset [0,1]^d$ using noisy pointwise evaluations. Existing regret bounds are either worst-case, scaling as $\tilde{\Theta} \left ( T^{d+1/d+2}\right )$, or adaptive via the zooming dimension $d_z$, yielding $\tilde{\Theta} \left ( T^{d_z+1/d_z+2}\right )$. However, such zooming-based guarantees are only partially instance-dependent, as they depend solely on the asymptotic growth of near-optimal level sets and fail to capture finer structural properties of $f$. We provide an analysis and an algorithm that characterizes the regret through integrals of the suboptimality gap of $f$ over its level sets. This yields regret bounds that adapt to the local growth of level sets, rather than only their asymptotic behavior. As a corollary, when the set of maximizers has dimension $d^\star>0$, we obtain impr
    
[^48]: 面向可泛化推荐的因果表征学习

    Causal Representation Learning for Generalisable Recommendation

    [https://arxiv.org/abs/2605.27043](https://arxiv.org/abs/2605.27043)

    该论文提出一种基于因果表征学习的推荐方法，通过信息论解耦准则和可优化的变分下界，从处理表征中剔除受混淆因子影响的非因果信息，从而解决训练分布与服务分布不一致导致的推荐模型泛化难题。

    

    在观测数据上训练的预测模型在部署时往往难以泛化，尤其是当训练数据本身是被优化系统运行所产生的结果时。推荐系统就是一个典型的例子：它们是在交互日志上进行训练的，而这些日志受到已部署策略、用户历史行为以及平台过滤机制的混淆。其结果是，训练分布与服务阶段所评分的候选分布并不一致，这一差距使得离线指标无法可靠地预测线上表现。我们提出一种受因果表征学习（CRL）启发的方法来解决该分布偏移问题。我们提出了一种信息论解耦准则，并证明其最优解能够从处理（treatment）表征中排除依赖于混淆因子的非因果信息。我们进一步推导出一个可计算的变分下界，使该准则能够基于有限的观测数据进行优化。我们方法的适用范围比……

    arXiv:2605.27043v2 Announce Type: replace  Abstract: Predictive models trained on observational data often fail to generalise when deployed, especially when the training data is a product of the system being optimised. Recommender systems are a canonical example: they are trained on interaction logs confounded by the deployed policy, past user behaviour, and platform filtering. As a result, the training distribution differs from the candidate distribution scored at serving time, a gap that makes offline metrics unreliable predictors of online performance. We address the distribution shift problem with a method motivated by causal representation learning (CRL). We propose an information-theoretic disentanglement criterion and prove that its optimum excludes confounder-dependent non-causal information from the treatment representation. We derive a tractable variational lower bound making the criterion optimisable from finite observational data. The scope of our method is narrower than th
    
[^49]: 高效基准测试本质上就是特征选择与多元回归

    Efficient Benchmarking Is Just Feature Selection and Multiple Regression

    [https://arxiv.org/abs/2605.25773](https://arxiv.org/abs/2605.25773)

    该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。

    

    arXiv:2605.25773v3 公告类型：replace-cross 摘要：高效基准测试技术旨在通过仅使用基准测试题目中的一个子集来预测完整基准测试分数，从而降低评估大语言模型（LLM）的计算成本。通过将这一问题重新构建为带特征选择的多元回归的一个实例，我们发现现有高效基准测试方法只需在预测阶段采用核岭回归（kernel ridge regression）即可得到大幅改进。此外，利用一种称为最小冗余最大相关性（mRMR）的信息论特征选择算法，我们可以通过选择对预测最有价值的题目子集来进一步提升这些方法的效果。除在数据极度匮乏的情形外，这些方法在涵盖二值与连续分数的多种基准测试上，均始终取得更小的预测误差（在MAE和RMSE两方面）以及预测分数与真实分数之间更高的排序相关性（在Spearman ρ和Kendall τ两方面）。

    arXiv:2605.25773v3 Announce Type: replace-cross  Abstract: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous
    
[^50]: NeuroMAS：将多智能体系统视为神经网络并结合联合强化学习

    NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning

    [https://arxiv.org/abs/2605.16757](https://arxiv.org/abs/2605.16757)

    该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。

    

    多智能体语言系统通常被构建为手工设计的工作流，其中智能体被预先分配语义角色，通信协议也事先指定。我们提出 NeuroMAS，这是一种首次将多智能体语言系统视为可训练、可扩展的类神经网络架构的方法，其中 LLM 智能体作为节点，中间文本信号作为边。在 NeuroMAS 中，智能体节点是角色无关但结构感知的：拓扑结构仅决定信息大体上如何流动，而强化学习训练决定节点如何通信、专门化与协调。这一表述将多智能体设计从工作流工程转向架构设计，使深度、宽度、连接性和增长协议成为可扩展的能力来源。此外，我们从理论上说明了当任务可进行层次化分解时，为什么这种模块化的文本计算更具参数效率。

    arXiv:2605.16757v2 Announce Type: replace  Abstract: Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like architecture with LLM agents as nodes and intermediate textual signals as edges. In NeuroMAS, agent nodes are role-free but structure-aware: the topology only determines how information can flow in general, while reinforcement learning training determines how nodes communicate, specialize, and coordinate. This formulation shifts multi-agent design from workflow engineering toward architecture design, where depth, width, connectivity, and growth protocol become scalable sources of capability. Further, we provide a theoretical perspective showing why such modular textual computation is more parameter-efficient when tasks admit hierarchical decompo
    
[^51]: POETS：通过计算高效的策略集成实现不确定性感知的LLM优化

    POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles

    [https://arxiv.org/abs/2605.07775](https://arxiv.org/abs/2605.07775)

    POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。

    

    在黑盒优化和强化学习中，平衡探索与利用是一个核心挑战。我们提出了POETS（用于汤普森采样的策略集成，Policy Ensembles for Thompson Sampling），这是一个连接不确定性量化与策略优化的新颖框架。我们的方法基于这样一个洞察：使用Kullback-Leibler (KL)正则化训练的策略会隐式地编码潜在的奖励函数。基于这一洞察，POETS绕过了训练不确定性感知奖励模型并单独将策略拟合到该模型这一复杂且嵌套的流程。相反，我们直接训练一个策略集成，通过将隐式编码的奖励函数与在线的自举数据相匹配来捕获认知不确定性。为了克服集成大型语言模型（LLM）所带来的高昂计算与内存开销，POETS采用了一种高效的架构：该集成共享一个预训练……

    arXiv:2605.07775v2 Announce Type: replace-cross  Abstract: Balancing exploration with exploitation is a core challenge in black-box optimization and reinforcement learning. We introduce POETS ($\textbf{Po}$licy $\textbf{E}$nsembles for $\textbf{T}$hompson $\textbf{S}$ampling), a novel framework that bridges uncertainty quantification and policy optimization. Our approach is grounded in the insight that policies trained with Kullback-Leibler (KL) regularization implicitly encode an underlying reward function. Building on this, POETS bypasses the complex, nested process of training an uncertainty-aware reward model and separately fitting a policy to this model. Instead, we directly train a policy ensemble to capture epistemic uncertainty by matching implicitly encoded reward functions to online, bootstrapped data. To overcome the prohibitive compute and memory constraints of ensembling Large Language Models (LLMs), POETS utilizes an efficient architecture: the ensemble shares a pre-train
    
[^52]: Q-MMR：通过递归重加权与矩匹配进行离策略评估

    Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching

    [https://arxiv.org/abs/2605.06474](https://arxiv.org/abs/2605.06474)

    提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。

    

    我们提出了一种新颖的理论框架 Q-MMR，用于有限时域马尔可夫决策过程（MDP）中的离策略评估。Q-MMR 学习一组标量权重，每个数据点对应一个权重，使得重加权后的奖励能够近似目标策略下的期望回报。这些权重通过与价值函数判别器类进行矩匹配的目标，以自上而下的方式归纳地学习。值得注意的是，或许令人惊讶的是，在仅需 $Q^\pi$ 可实现性的条件下，即可为一般函数逼近建立数据相关的有限样本保证，且该边界与维度无关——也就是说，误差不依赖于函数类的统计复杂度。我们还与若干现有方法建立了联系，例如重要性采样和线性 FQE。进一步的理论分析为覆盖性这一离线强化学习中至关重要的概念的本质提供了新的认识。

    arXiv:2605.06474v3 Announce Type: replace-cross  Abstract: We present a novel theoretical framework, Q-MMR, for off-policy evaluation in finite-horizon MDPs. Q-MMR learns a set of scalar weights, one for each data point, such that the reweighted rewards approximate the expected return under the target policy. The weights are learned inductively in a top-down manner via a moment matching objective against a value-function discriminator class. Notably, and perhaps surprisingly, a data-dependent finite-sample guarantee for general function approximation can be established under only the realizability of $Q^\pi$, with a dimension-free bound -- that is, the error does not depend on the statistical complexity of the function class. We also establish connections to several existing methods, such as importance sampling and linear FQE. Further theoretical analyses shed new light on the nature of coverage, a concept of fundamental importance to offline RL.
    
[^53]: 基于框架理论的Oja子空间规则三因子学习规则推导

    Frame Theoretical Derivation of Three Factor Learning Rule for Oja's Subspace Rule

    [https://arxiv.org/abs/2604.02849](https://arxiv.org/abs/2604.02849)

    本文利用框架理论从Oja子空间规则中系统地推导出生物学上合理的误差门控Hebbian PCA学习规则（EGHR-PCA），并证明其全局第三因子恰好是对称矩阵空间自然框架下的框架系数。

    

    我们证明了用于主成分分析（PCA）的误差门控Hebbian规则（EGHR-PCA）——一种在高斯输入下与Oja子空间规则等价的三因子学习规则——可以利用框架理论从Oja子空间规则中系统地推导出来。当学习规则相对于对称矩阵空间上的一个自然框架进行展开时，EGHR-PCA中的全局第三因子恰好作为框架系数出现。这为从数学上规范的对应规则推导出生物学上合理的学习规则提供了一种有原则的、非启发式的推导方法。

    arXiv:2604.02849v2 Announce Type: replace-cross  Abstract: We show that the error-gated Hebbian rule for PCA (EGHR-PCA), a three-factor learning rule equivalent to Oja's subspace rule under Gaussian inputs, can be systematically derived from Oja's subspace rule using frame theory. The global third factor in EGHR-PCA arises exactly as a frame coefficient when the learning rule is expanded with respect to a natural frame on the space of symmetric matrices. This provides a principled, non-heuristic derivation of a biologically plausible learning rule from its mathematically canonical counterpart.
    
[^54]: 非平稳线性老虎机中最优臂识别问题的复杂度研究

    On the Complexity of Best-Arm Identification in Non-Stationary Linear Bandits

    [https://arxiv.org/abs/2603.10346](https://arxiv.org/abs/2603.10346)

    本文研究了非平稳线性老虎机中固定预算最优臂识别的复杂度，指出基于G-最优设计的经典复杂度度量过于悲观，因为其下界仅建立在标准基向量构成的臂集合上，掩盖了更丰富的臂集合结构可能带来的优势。

    

    我们研究非平稳线性老虎机中的固定预算最优臂识别（BAI）问题。具体而言，给定一个固定的时间预算 $T\in \mathbb{N}$、一个有限的臂集合 $\mathcal{X} \subset \mathbb{R}^d$，以及一个可能具有对抗性的未知参数序列 $\lbrace \theta_t\rbrace_{t=1}^{T}$（因此是非平稳的），学习者旨在以高概率识别出具有最大累积奖励的臂 $x_* = \arg\max_{x \in \mathcal{X}} x^\top\sum_{t=1}^T \theta_t$。在该设定下，众所周知，从G-最优设计中独立同分布地采样各臂可以得到 $\exp\left(-\Theta\left(T / H_{G}\right)\right)$ 的极小极大最优错误概率，其中 $H_{G}$ 与维度 $d$ 成正比。然而，这种复杂度度量过于悲观，因为它源自一个下界，而该下界仅考虑了由标准基向量构成的臂集合，从而掩盖了臂集合本身可能带来的任何潜在优势

    arXiv:2603.10346v2 Announce Type: replace  Abstract: We study the fixed-budget best-arm identification (BAI) problem in non-stationary linear bandits. Concretely, given a fixed time budget $T\in \mathbb{N}$, finite arm set $\mathcal{X} \subset \mathbb{R}^d$, and a potentially adversarial sequence of unknown parameters $\lbrace \theta_t\rbrace_{t=1}^{T}$ (hence non-stationary), a learner aims to identify the arm with the largest cumulative reward $x_* = \arg\max_{x \in \mathcal{X}} x^\top\sum_{t=1}^T \theta_t$ with high probability. In this setting, it is well-known that i.i.d. sampling arms from the G-optimal design yields a minimax-optimal error probability of $\exp\left(-\Theta\left(T / H_{G}\right)\right)$, where $H_{G}$ scales proportionally with the dimension $d$. However, this notion of complexity is overly pessimistic, as it is derived from a lower bound in which the arm set consists only of the standard basis vectors, thus masking any potential advantages arising from arm sets 
    
[^55]: 软符号化的Kolmogorov-Arnold网络

    Softly Symbolifying Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2512.07875](https://arxiv.org/abs/2512.07875)

    提出软符号化的Kolmogorov-Arnold网络（S2KAN），通过将符号基元直接融入训练，利用可微分的门控稀疏化机制与最小描述长度目标，在保留稠密样条灵活逼近能力的同时发现可解释的激活函数。

    

    Kolmogorov-Arnold网络（KANs）为可解释机器学习提供了一条有前景的路径：其可学习的激活函数可以被单独研究，同时能够共同准确地拟合复杂数据。然而在实践中，训练得到的激活函数往往缺乏*符号保真度*，会学习到与可解释形式没有任何有意义对应关系的病态分解。我们提出软符号化的Kolmogorov-Arnold网络（S2KAN），它将符号基元直接集成到训练过程中。每个激活函数从符号项与稠密项组成的字典中提取，并通过可学习的门控机制对表示进行稀疏化。至关重要的是，这种稀疏化是可微分的，能够实现端到端优化，并采用了受最小描述长度启发的优化目标。S2KAN能够在保留稠密样条以实现灵活逼近的同时，发现可解释的激活函数。我们引入了一种对符号……（原文摘要在此处截断）

    arXiv:2512.07875v2 Announce Type: replace-cross  Abstract: Kolmogorov--Arnold Networks (KANs) offer a promising path toward interpretable machine learning: their learnable activations can be studied individually while collectively fitting complex data accurately. In practice, however, trained activations often lack *symbolic fidelity*, learning pathological decompositions with no meaningful correspondence to interpretable forms. We propose Softly Symbolified Kolmogorov--Arnold Networks (S2KAN), which integrate symbolic primitives directly into training. Each activation draws from a dictionary of symbolic and dense terms, with learnable gates that sparsify the representation. Crucially, this sparsification is differentiable, enabling end-to-end optimization, and uses an optimization objective inspired by minimum description length. S2KAN can discover interpretable activation functions while retaining dense splines for flexible approximation. We introduce a quantitative measure of symbol
    
[^56]: 面向不确定性感知代理模型的保形化多项式混沌展开

    Conformalized Polynomial Chaos Expansion for Uncertainty-aware Surrogate Modeling

    [https://arxiv.org/abs/2510.22375](https://arxiv.org/abs/2510.22375)

    该论文将折刀法保形预测与多项式混沌展开代理模型相结合，利用回归模型的线性特性以解析闭式形式计算留一法残差，从而在不需留出校准数据集的情况下为代理模型提供预测不确定性区间。

    

    本工作提出了一种方法，为数据驱动的多项式混沌展开代理模型配备能够量化代理模型预测不确定性的区间。为此，将基于折刀法的保形预测集成到基于回归的多项式混沌展开中。折刀算法使用留一法残差在多项式混沌代理模型的预测周围生成预测区间。折刀+扩展方法则额外需要留一法模型预测。这两种方法都可以使用整个数据集进行模型训练，且无需留出数据集来校准预测区间。高效实现的关键在于利用多项式混沌回归模型的线性特性，使得留一法残差以及（如有必要的）留一法模型预测能够通过解析的闭式表达式计算得出，从而消除了重复模型重新评估的需要。

    arXiv:2510.22375v4 Announce Type: replace-cross  Abstract: This work introduces a method to equip data-driven polynomial chaos expansion surrogate models with intervals that quantify the predictive uncertainty of the surrogate. To that end, jackknife-based conformal prediction is integrated into regression-based polynomial chaos expansions. The jackknife algorithm uses leave-one-out residuals to generate predictive intervals around the predictions of the polynomial chaos surrogate. The jackknife+ extension additionally requires leave-one-out model predictions. Both methods allow to use the entire dataset for model training and do not require a hold-out dataset for prediction interval calibration. The key to efficient implementation is to leverage the linearity of the polynomial chaos regression model, so that leave-one-out residuals and, if necessary, leave-one-out model predictions can be computed with analytical, closed-form expressions. This eliminates the need for repeated model re
    
[^57]: 从决斗老虎机到情景马尔可夫决策过程：对抗性偏好反馈下的学习

    From Dueling Bandits to Episodic MDPs: Learning with Adversarial Preference Feedback

    [https://arxiv.org/abs/2507.11706](https://arxiv.org/abs/2507.11706)

    本文将决斗老虎机扩展到偏好反馈会影响状态转移的情景MDP设定，在对抗性偏好变化下建立了极小极大遗憾界 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，并给出了达到该界的算法。

    

    在决斗老虎机问题中，学习者在每一轮比较两个臂（选项），并且只能观察到哪一个更受偏好。本文探讨当被比较的选项对还会影响下一个状态时问题会发生什么变化，例如在推荐界面中，展示给用户的选项对会影响用户的下一个情境。我们将该问题形式化为一个基于状态的偏好MDP（马尔可夫决策过程），即一个情景式的表格型MDP，其中学习者在每个访问的状态选择一对臂，只观察到两者之间带有噪声的比较结果，并转移到可能依赖于该选项对的下一个状态。偏好可能在不同的情景之间以对抗方式变化，选项对的损失通过Borda分数定义。我们证明了在转移概率已知的情况下，该问题的极小极大遗憾为 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，其中 $H$、$S$、$K$ 和 $T$ 分别是每个情景的步数、状态数、臂数和情景数，并且该速率可以通过基于在线学习的算法达到。

    arXiv:2507.11706v2 Announce Type: replace-cross  Abstract: In dueling bandits, the learner compares two arms in each round and observes only which one is preferred. This paper asks what changes when the compared pair also affects the next state, as in a recommendation interface where the pair shown to a user affects the user's next context. We formulate this problem as a state-wise preference-based MDP, an episodic tabular MDP in which the learner selects a pair of arms at each visited state, observes only a noisy comparison between them, and moves to a next state that may depend on the pair. The preferences may change adversarially across episodes, and the loss of a pair is defined through Borda scores. We show that the minimax regret of this problem under known transitions is $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$, where $H$, $S$, $K$, and $T$ are the numbers of steps per episode, states, arms, and episodes, and that this rate is achieved by an algorithm based on online li
    
[^58]: 重尾目标分布下重要性加权退火MCMC的一致遍历性与谱隙

    Uniform ergodicity and spectral gap of importance-tempered MCMC with heavy-tailed targets

    [https://arxiv.org/abs/2506.19186](https://arxiv.org/abs/2506.19186)

    该论文针对重尾目标分布，给出了重要性加权退火RWM算法一致遍历性的充要条件（d/γ < β < (γ-2)/γ），并证明了适当选择退火参数β可显著降低对初始化的敏感性、大幅提升相对标准RWM的采样效率。

    

    我们研究了在 $\mathbb{R}^d$ 上的重要性加权退火随机游走Metropolis（RWM）算法，该算法以退火密度 $\pi_\beta(x) \propto \pi(x)^\beta$（其中 $\beta \in (0,1)$）为目标分布，并通过重要性加权来校正由此产生的偏差。该采样器的动力学可以用一个连续时间马尔可夫链来描述，其嵌入链为退火RWM。我们证明，当 $\pi$ 的尾部以 $\|x\|^{-\gamma}$ 的速率衰减时，该连续时间链一致遍历当且仅当 $d/\gamma < \beta < (\gamma - 2)/\gamma$。此外，该链的全局谱隙可以通过限制在半径随 $d$ 多项式增长的球内的链的谱隙来控制。我们的理论和仿真研究表明，通过适当选择 $\beta$，重要性退火可以大幅降低对初始化的敏感性，并显著提高相对于标准RWM的采样效率。

    arXiv:2506.19186v2 Announce Type: replace-cross  Abstract: We study the importance-tempered random-walk Metropolis (RWM) algorithm on $\mathbb{R}^d$, which targets the tempered density $\pi_\beta(x) \propto \pi(x)^\beta$ with $\beta \in (0,1)$ and corrects for the resulting bias by importance weighting. The dynamics of this sampler can be described by a continuous-time Markov chain whose embedded chain is the tempered RWM. We prove that when the tails of $\pi$ decay at rate $\|x\|^{-\gamma}$, this continuous-time chain is uniformly ergodic if and only if $d/\gamma < \beta < (\gamma - 2)/\gamma$. Further, the global spectral gap of this chain can be controlled via the spectral gap of the chain restricted to a ball with radius growing polynomially in $d$. Our theory and simulation studies show that, with a suitable choice of $\beta$, importance tempering can substantially reduce sensitivity to initialization and significantly improve efficiency over the standard RWM.
    
[^59]: 随机逆问题：稳定性、正则化与Wasserstein梯度流

    Stochastic Inverse Problem: stability, regularization and Wasserstein gradient flow

    [https://arxiv.org/abs/2410.00229](https://arxiv.org/abs/2410.00229)

    本文将逆问题推广到随机情形，在概率空间中借助测度输运理论研究了直接反演、带正则化的变分方法以及Wasserstein梯度流优化，并揭示了度量选择对稳定性和解的性质的关键影响。

    

    物理或生物科学中的逆问题通常涉及恢复一个随机的未知参数。所寻求的目标是未知参数的概率分布，该分布产生的数据与测量结果相一致。因此，这些问题自然地被构建为随机逆问题。本文从三个方面探讨这一问题：直接反演、带正则化的变分形式，以及通过梯度流进行优化，并与确定性逆问题进行类比。与确定性情形的一个关键区别在于我们所工作的空间。在这里，我们在概率空间而非欧几里得空间或Sobolev空间中进行研究，这使得测度输运理论的工具成为研究的必要手段。我们的研究结果表明，度量的选择——无论是在损失函数的设计中还是在优化过程中——对稳定性和问题的性质都有显著影响。

    arXiv:2410.00229v2 Announce Type: replace  Abstract: Inverse problems in physical or biological sciences often involve recovering an unknown parameter that is random. The sought-after quantity is a probability distribution of the unknown parameter, that produces data that aligns with measurements. Consequently, these problems are naturally framed as stochastic inverse problems. In this paper, we explore three aspects of this problem: direct inversion, variational formulation with regularization, and optimization via gradient flows, drawing parallels with deterministic inverse problems. A key difference from the deterministic case is the space in which we operate. Here, we work within a probability space rather than Euclidean or Sobolev spaces, making tools from measure transport theory necessary for the study. Our findings reveal that the choice of metric---both in the design of the loss function and in the optimization process---significantly impacts the stability and properties of th
    
[^60]: 自验证集成模型（SVEM）的随机置换全模型检验启发式方法

    A Randomized Permutation Whole-Model Test Heuristic for Self-Validated Ensemble Models (SVEM)

    [https://arxiv.org/abs/2405.11156](https://arxiv.org/abs/2405.11156)

    本文提出了一种基于随机置换的全模型检验启发式方法，通过在多个因子空间点上构建参考分布并利用马氏距离，来检验自验证集成模型（SVEM）拟合相对于常数响应原假设的统计显著性。

    

    我们引入了一种启发式方法，用于检验自验证集成模型（Self-Validated Ensemble Models, SVEM）拟合的显著性，其原假设为响应为常数。SVEM模型对目标数据集的分式加权自助样本进行nBoot次拟合并对其预测取平均。每次拟合都在训练数据的一个验证副本上进行调优，训练权重和验证权重呈反相关。所提出的检验方法在整个因子空间中均匀分布的nPoint个点上，计算以响应列均值为中心、并按集成变异性归一化的SVEM预测值。参考分布的构建方式是：将SVEM模型重新拟合到响应列的nPerm个随机置换上，并记录nPoint个点上相应的标准化预测值。对经过中心化和缩放处理的nPerm × nPoint参考矩阵应用降秩奇异值分解，以计算马氏距离

    arXiv:2405.11156v2 Announce Type: replace-cross  Abstract: We introduce a heuristic to test the significance of fit of Self-Validated Ensemble Models (SVEM) against the null hypothesis of a constant response. A SVEM model averages predictions from nBoot fits of a model, applied to fractionally weighted bootstraps of the target dataset. It tunes each fit on a validation copy of the training data, utilizing anti-correlated weights for training and validation. The proposed test computes SVEM predictions centered by the response column mean and normalized by the ensemble variability at each of nPoint points spaced throughout the factor space. A reference distribution is constructed by refitting the SVEM model to nPerm randomized permutations of the response column and recording the corresponding standardized predictions at the nPoint points. A reduced-rank singular value decomposition applied to the centered and scaled nPerm x nPoint reference matrix is used to calculate the Mahalanobis di
    

