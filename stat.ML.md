# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-Crossing Deep Quantile Regression for Distributional Survival Prediction](https://arxiv.org/abs/2608.16864) | 本文提出了一种基于深度学习的非交叉分位数回归框架，用于生存分析中的分布预测，通过构造保证分位数排序一致性，并在多种复杂条件下优于现有方法。 |
| [^2] | [Hide&Seek: Learning to Explain in an End-to-End Differentiable Network](https://arxiv.org/abs/2608.16689) | 本文提出Hide&Seek，一种端到端可微分模型，通过将特征移除转化为可微操作，在无信息泄漏的情况下联合学习特征选择与预测，实现高效训练和卓越性能。 |
| [^3] | [Density-Reweighted Entropic Optimal Transport: Decoupling Geometry from Sampling Density](https://arxiv.org/abs/2608.16506) | 提出一种密度重加权熵最优传输框架，可调节采样密度影响，实现从标准EOT到纯几何驱动的对齐，并证明其收敛性。 |
| [^4] | [Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization](https://arxiv.org/abs/2608.16492) | 本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。 |
| [^5] | [Deep adaptive design with an evidential bias criterion](https://arxiv.org/abs/2608.16466) | 本文提出了一种基于“反偏置”证据准则的深度自适应实验设计方法，以更好地控制实验产生误导性证据的风险，弥补传统期望信息增益准则的不足。 |
| [^6] | [Convergence Analysis of Statistical Inverse Problems on Reproducing Kernel Banach Spaces](https://arxiv.org/abs/2608.16404) | 本文在再生核巴拿赫空间框架下，利用Tikhonov正则化方法对统计逆问题中的解进行稳定估计，并首次给出了随数据量增加时的收敛性证明及收敛速率。 |
| [^7] | [LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models](https://arxiv.org/abs/2608.16340) | 提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。 |
| [^8] | [Conditional Evaluation of Language Models with Cheap Auxiliary Signals](https://arxiv.org/abs/2608.16210) | 本文提出LACE方法，利用廉价辅助信号通过局部中心化技术实现条件语言模型评估的半监督估计，在保证无偏性的同时显著降低评估成本。 |
| [^9] | [Coded Hankel Polynomial Chaos: Spectral Identification of Dominant Polynomial-Chaos Modes](https://arxiv.org/abs/2608.16126) | 本文提出编码汉克尔多项式混沌（CH-PC）方法，通过谱分析和低秩汉克尔矩阵从多项式混沌展开中高效识别主导模态，并利用相位编码冗余提升鲁棒性。 |
| [^10] | [EMS Coreset: An Efficient Expectation-Maximization Algorithm for Sinkhorn Coreset](https://arxiv.org/abs/2608.16101) | 本文提出了一种高效的Sinkhorn核心集算法，通过非均匀权重实现闭式更新，大幅降低计算成本，同时保证近似质量和稳定性。 |
| [^11] | [Generalized Linear Bandits with Memory](https://arxiv.org/abs/2608.15848) | 本文通过改进分析并设计基于收缩置信区间的分块算法，在带记忆的广义线性老虎机中实现了$\sqrt{T}$遗憾率，超越了先前$\tilde{O}(T^{3/4})$的界限。 |
| [^12] | [Self-Supervised Auxiliary Task Discovery for Stable Reinforcement Learning in Stock Trading](https://arxiv.org/abs/2608.15841) | 本文提出一种自监督框架，通过自动发现通用价值函数形式的辅助任务来增强状态表示，从而提升股票交易强化学习的稳定性和性能。 |
| [^13] | [How Many Samples Are Needed to Determine Causal Direction? Sharp Minimax Bounds for Bivariate LiNGAM](https://arxiv.org/abs/2608.15840) | 本文首次给出了双变量LiNGAM中确定因果方向所需样本数的尖锐极小极大下界，揭示了因果效应强度、非高斯性和扰动尺度差异如何共同决定样本复杂度。 |
| [^14] | [Cross-Entropy Risk Estimation for Language Models: Inconsistency Must Be Dense, and the Holdout Method Is No Exception](https://arxiv.org/abs/2608.15798) | 本文证明语言模型的每令牌交叉熵风险无法被一致估计，无论使用何种估计器（包括保留法），这是由有限与无限风险在状态空间中的拓扑密集性决定的根本性限制。 |
| [^15] | [Inferential Evaluation of Surrogate-Derived Models under Covariate Shift](https://arxiv.org/abs/2608.15783) | 本文提出了一种在协变量偏移下评估代理衍生模型目标性能的方法，利用三样本设置和交叉拟合估计器，结合密度比与核修正来处理标签稀缺和分布差异问题。 |
| [^16] | [Learning Stock Trading Policies via Barycenter-Based Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/2608.15770) | BRaG通过重心聚合多专家策略并利用对抗模仿学习预训练交易模型，结合控制屏障函数实现风险约束的股票交易策略学习。 |
| [^17] | [FirstDiff: One-Step Diffusion-Based Anomaly Detection for Multivariate Time Series via Initial Noise Prediction](https://arxiv.org/abs/2608.15727) | 本文提出FirstDiff，通过仅使用反向扩散初始步骤的预测噪声进行异常检测，大幅降低计算成本并利用中间扩散信息，实现多变量时间序列的高效一步检测。 |
| [^18] | [On Stopping Rules and Spatial Adaptation for CART](https://arxiv.org/abs/2608.15649) | 本文证明了CART算法结合最小杂质减少停止规则和适当阈值，能在空间异质和各向异性平滑条件下实现逐点极小极大最优速率，并指出常用停止规则无法实现空间适应性。 |
| [^19] | [Sparse Prototype Code Underlies Classification and Prediction Across Modalities](https://arxiv.org/abs/2608.15632) | 本文发现分类任务中神经表征存在跨模态的普遍几何结构，并提出一个解析平均场理论，通过类质心坐标的变异性和类半径重整化准确预测分类性能。 |
| [^20] | [Benchmarking Quantum Machine Learning for Power-System Attack Detection: Evaluation Choices Decide the Outcome Before the Models Do](https://arxiv.org/abs/2608.15617) | 该论文通过系统性的基准测试揭示，电力系统攻击检测中量子机器学习与经典模型的比较结果高度依赖于评估协议中的八项关键选择，这些选择在模型运行前就决定了结论走向。 |
| [^21] | [PERO: Efficient Robust Post-Training Foundation Models for Encrypted Traffic Classification](https://arxiv.org/abs/2608.15504) | 本文提出PERO框架，通过轻量级代理预评估高损失样本，实现加密流量基础模型的高效鲁棒后训练，以应对风险敏感场景中的罕见高损失错误。 |
| [^22] | [Generalized Hierarchical Conformal Prediction](https://arxiv.org/abs/2608.15500) | 提出广义分层保形预测（GHCP），通过随机分配参考组大小恢复对称性，使分层保形预测能在仅有少量目标组观测时有效利用数据。 |
| [^23] | [Optimal Lower Bounds for Networked Information Aggregation](https://arxiv.org/abs/2608.15472) | 本文解决了网络化信息聚合中均方误差下界的开放问题，确定了最优下界，填补了先前上界与下界之间的差距。 |
| [^24] | [Prediction Inference of Time Series with Standard ReLU Deep Neural Networks](https://arxiv.org/abs/2608.15362) | 本文提出一种基于标准ReLU深度神经网络的时间序列预测方法，通过构建相关预测区间（PPI）来同时量化未来变异性和估计变异性，并证明了估计器在β混合数据下的一致性。 |
| [^25] | [GFCM: A Tail-Sensitive Mixed-Type Conditional Independence Test for Causal Discovery](https://arxiv.org/abs/2608.15332) | 本文提出了一种新的混合类型条件独立性检验方法GFCM，通过中心矩和条件分位数指示器结合柯西规则，有效检测尾部依赖性，并解决了尺度特征的偏差问题，适用于因果发现。 |
| [^26] | [Shape Operator PCA: Curvature-Aware Projections for Geometric Machine Learning](https://arxiv.org/abs/2608.15313) | 本文提出SHOPCA方法，通过将形状算子正则化融入PCA，在无监督降维中同时捕捉方差和曲率信息，并利用谱特征间隙自动选择正则化参数。 |
| [^27] | [A Unified Geometric Framework for Developmental Analysis of Spatial Transcriptomic Data](https://arxiv.org/abs/2608.15306) | 本文提出了一个统一几何框架，利用Gromov-Wasserstein空间嵌入分析空间转录组数据中基因表达网络的时空演变，克服了现有方法忽略关系结构的局限。 |
| [^28] | [Convolution Smoothed Quantile Regression for XGBoost](https://arxiv.org/abs/2608.15290) | 该论文提出了QXGB框架，通过引入卷积平滑损失到分位数梯度提升中，在保持XGBoost计算效率的同时恢复Hessian信息，实现对极端结果和超越概率的可解释预测。 |
| [^29] | [Learning reshapes power-law anisotropy in internal representations](https://arxiv.org/abs/2608.15239) | 该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。 |
| [^30] | [The Distributional View of Knowledge Distillation](https://arxiv.org/abs/2608.15215) | 本文提出了一种分布视角的知识蒸馏方法，通过多温度视图的几何聚合替代逐点比较，并证明了特定池化与单温度的等价性，为蒸馏提供了更丰富的理论框架。 |
| [^31] | [Identifying parameter couplings and uncertainties of mixed-noise stochastic systems via full-covariance Gaussian mixture network](https://arxiv.org/abs/2608.15198) | 本文提出了一种基于全协方差高斯混合网络的参数估计方法，能够有效处理混合噪声随机系统，并显式揭示参数间的耦合关系和多模态不确定性。 |
| [^32] | [Beyond Effective Sample Size: Effective Number of Proposals for Adaptive Importance Sampling](https://arxiv.org/abs/2608.15154) | 本文提出了一种新的提案级诊断指标——有效提案数量（ENP），用于更准确地评估基于种群的适应性重要性采样中提案组件的多样性，弥补了传统有效样本量（ESS）仅关注权重集中度而忽略提案空间分布的不足。 |
| [^33] | [Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems](https://arxiv.org/abs/2608.15144) | 本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。 |
| [^34] | [Sufficient Dimesion Reduction via Generalized Stein's Lemma](https://arxiv.org/abs/2608.15121) | 本文提出一种基于广义斯坦引理的新充分降维框架，通过构建交叉矩矩阵，有效克服了现有方法在多元响应和有限样本下的分布假设、稀疏性和计算成本等限制。 |
| [^35] | [Do Geometry-Aware Positional Encodings Help Transformers in Spatial Imperfect-Information Games?](https://arxiv.org/abs/2608.14982) | 本文通过多级基准测试证明，几何感知位置编码HexRoPE在空间不完美信息游戏中显著提升了Transformer的隐藏目标追踪与策略学习性能。 |
| [^36] | [A Deep Learning Model for Spatially Clustered Data via Differentiable Cluster Assignment](https://arxiv.org/abs/2608.14968) | 该论文提出了一种联合学习空间划分和聚类特定回归函数的深度学习模型，通过可微聚类分配和惩罚机制实现高效的非参数回归估计。 |
| [^37] | [PathFinder: Joint Decompositions of Linked Multimodal Datasets](https://arxiv.org/abs/2608.14951) | 提出一种新方法PathFinder，允许不共享维度的多模态数据集通过路径连接实现联合低秩分解，从而发现跨模态的公共模式。 |
| [^38] | [Optimal Watermark Localization in Mixed-Source Large Language Model Texts](https://arxiv.org/abs/2608.14906) | 本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。 |
| [^39] | [ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics](https://arxiv.org/abs/2608.14866) | ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。 |
| [^40] | [Generative Learning of Separatrices](https://arxiv.org/abs/2608.14743) | 该论文提出了一种结合监督分类与生成建模的新框架，用于高效重建高维多稳态动力系统中的分离面，克服了传统方法的计算限制和采样不足问题。 |
| [^41] | [AccretionLink: On-Device Auditing of Exposure-Control Attacks on Attribute Inference](https://arxiv.org/abs/2608.14735) | AccretionLink提出了一种设备端审计框架，通过定义机密性和完整性博弈及依赖感知的e过程，有效检测并量化暴露控制攻击对属性推断的增强效应。 |
| [^42] | [Rethinking Reverse KL as Adaptive Entropy Distillation](https://arxiv.org/abs/2608.14685) | 本文提出自适应熵蒸馏（AED），通过重新分解反向KL目标为教师拟合和学生熵项，利用教师熵动态调整蒸馏权重，实现更优的模仿与生成平衡。 |
| [^43] | [RouteTS: Frequency-Time Routing for Time Series Forecasting](https://arxiv.org/abs/2608.14682) | RouteTS通过振幅路由动态分配频率分量到最优计算域（频域或时域），解决了时间序列预测中周期性与局部变化的处理冲突。 |
| [^44] | [Context Is Not Authority: Structured Runtime Governance for Financial Market Agents](https://arxiv.org/abs/2608.09025) | SAGE-Fin通过结构化运行时治理框架，确保金融代理的每个拟议效果都受到类型化适配器绑定、覆盖债务记录和状态变化后重新授权检查的严格管控，从而防止上下文被滥用为未经授权的行为。 |
| [^45] | [The Spectral Neuron](https://arxiv.org/abs/2608.08003) | 本文提出了“谱神经元”模型，通过读取仿射矩阵函数的特征值实现非线性预测，在保持系数透明性的同时增强表达力，填补了线性模型与神经网络之间的空白。 |
| [^46] | [Estimating and Testing Kinks in Panel Data Models](https://arxiv.org/abs/2608.07162) | 该论文首次在固定效应面板数据模型中提出一种惩罚最小二乘方法，能够自适应估计未知数量的共同断点日期，并给出断点位置和斜率收敛速率的理论保证。 |
| [^47] | [Priors learned from legacy reconstructions inherit undetectable overconfidence](https://arxiv.org/abs/2607.21721) | 论文指出，从旧重建档案学习的先验在盲子空间上继承不可检测的过度自信，导致误差被低估且无法通过部署测试发现。 |
| [^48] | [Sum-of-Squares Degree Barriers for the Reweighted-Hinge Method in Robust Halfspace Learning: A Christoffel-Function Characterization](https://arxiv.org/abs/2606.17215) | 本文通过克里斯托弗函数精确刻画了平方和证书在鲁棒半空间学习中的度数限制，揭示了隐藏污染的最大质量与证书度数之间的关系。 |
| [^49] | [A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast](https://arxiv.org/abs/2606.14403) | 本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。 |
| [^50] | [Computationally tractable robust differentially private mean estimation](https://arxiv.org/abs/2606.12654) | 提出了一种计算上易处理且鲁棒的差分隐私均值估计方法“气球均值”，通过扩展Mahalanobis球上的迭代裁剪实现，在重尾和污染数据下优于现有方法。 |
| [^51] | [A Quantitative Approximation Framework for Flow Distillation in Diffusion Models](https://arxiv.org/abs/2606.03820) | 本文提出了一个定量框架，揭示低噪声多模态下分数逼近与动态稳定性的分离，并给出了指数增长稳定性因子的可计算界，从而识别直接蒸馏的适用条件。 |
| [^52] | [Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](https://arxiv.org/abs/2606.00984) | 本文提出两种仅需$O(\log\log T)$次参数更新的线性上下文赌博机算法，在静态调度下同时在小规模和大规模动作集下实现极小极大最优遗憾，并澄清了批处理与稀有更新的实际区别。 |
| [^53] | [Period spacings and global seismic parameters for K2 red giants using deep learning](https://arxiv.org/abs/2605.08051) | 本文首次利用深度学习从K2短基线数据中自动测量红巨星的引力模周期间距，无需背景拟合，显著扩展了恒星核心探测的样本范围。 |
| [^54] | [The Optimal Sample Complexity of Multiclass and List Learning](https://arxiv.org/abs/2604.24749) | 本文通过证明最大超图密度受DS维上界限制，解决了长期猜想，从而确定了多类与列表学习的最优样本复杂度。 |
| [^55] | [Informative Perturbation Selection for Uncertainty-Aware Post-hoc Explanations](https://arxiv.org/abs/2603.14894) | 本文提出EAGLE框架，将事后模型无关解释中的扰动选择视为信息论主动学习问题，通过自适应采样最大化期望信息增益的扰动，生成更可靠且不确定性感知的局部解释。 |
| [^56] | [Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling](https://arxiv.org/abs/2602.18518) | 本文提出了一种基于设计的测量系统，结合机器学习辅助抽样和多模态LLM标注，实现了对平台内容违规普遍性的高效、无偏估计，并支持多维度细分。 |
| [^57] | [On Fibonacci Ensembles: An Alternative Approach to Ensemble Learning Inspired by the Timeless Architecture of the Golden Ratio](https://arxiv.org/abs/2512.22284) | 本文提出斐波那契集成框架，利用基于黄金比例的归一化权重和二阶递归结构，通过正交化与Rao-Blackwell优化实现基础学习器间的方差减少，从而补充和扩展经典集成方法。 |
| [^58] | [Inference for Similarity and Alignability between Noisy High-Dimensional Datasets](https://arxiv.org/abs/2511.21074) | 本文提出了一种基于流形信号加噪声模型和谱特性的统计推断框架，用于在异质噪声下度量高维数据集的相似性与可对齐性，该度量具有尺度与旋转不变性。 |
| [^59] | [Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](https://arxiv.org/abs/2509.16586) | 本文提出了生成模型下受限平均奖励MDP的基于模型算法，在松弛和严格可行性设置下分别实现了近最优的样本复杂度界。 |
| [^60] | [Enhancing Differentially Private Linear Regression via Public Second-Moment](https://arxiv.org/abs/2508.18037) | 本文提出一种利用公共二阶矩矩阵转换私有数据的新方法，以改善差分隐私线性回归中充分统计量扰动估计器的条件数，从而提升其准确性和鲁棒性。 |
| [^61] | [Projection-based multifidelity linear regression for data-scarce applications](https://arxiv.org/abs/2508.08517) | 本文提出了两种基于投影的多保真线性回归方法，通过主成分降维和两种数据增强策略，有效利用低保真数据提升高保真模型在数据稀缺和高维输出场景下的预测精度。 |
| [^62] | [ROC-n-reroll: How verifier imperfection affects test-time scaling](https://arxiv.org/abs/2507.12399) | 本文通过理论证明和实验验证，指出验证器的ROC曲线几何决定测试时扩展方法的精度，并发现RS在固定计算下优于BoN，但无法从低计算量表现预测高计算量性能。 |
| [^63] | [Leveraging Generative Artificial Intelligence for Causal Inference with Unstructured Data](https://arxiv.org/abs/2507.03897) | 本文提出GPI框架，利用开源生成式AI模型提取非结构化数据的低维表示，无需微调即可进行因果推断，兼顾计算效率与不确定性量化。 |
| [^64] | [Multilook Coherent Imaging: Theoretical Guarantees and Algorithms](https://arxiv.org/abs/2505.23594) | 本文首次在深度图像先验假设下为多视相干成像的最大似然估计器建立了均方误差的理论上界，并提供了相应的算法框架，填补了该领域理论基础研究的空白。 |
| [^65] | [One-shot Robust Federated Learning of Independent Component Analysis](https://arxiv.org/abs/2505.20532) | 提出了一种基于谱聚类和几何中位数的联邦ICA一次性聚合方法，有效解决了符号置换和异构质量问题，并在大量低质量数据下保持鲁棒性。 |
| [^66] | [Weak Physics Informed Neural Networks for Geometry Compatible Hyperbolic Conservation Laws on Manifolds](https://arxiv.org/abs/2505.19036) | 本文提出了一种弱物理信息神经网络（wPINN）框架，通过建立局部$L_1$-稳定性估计和收敛性分析，首次为流形上低正则性双曲守恒律的熵解提供了严格的近似保证。 |
| [^67] | [WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales](https://arxiv.org/abs/2505.04608) | 本文提出加权共形测试鞅（WCTMs），以支持AI部署后的在线自适应监控，克服了现有方法在假设类别限制、缺乏适应性和诊断能力上的不足。 |
| [^68] | [On Stopping Times of Power-one Sequential Tests: Tight Lower and Upper Bounds](https://arxiv.org/abs/2504.19952) | 本文提出了两个适用于任意复合假设检验的通用停止时间下界，覆盖Wald和Farrell两种设定，无需主导测度，显著推广了现有理论。 |
| [^69] | [Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching](https://arxiv.org/abs/2502.14424) | 本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。 |
| [^70] | [Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data](https://arxiv.org/abs/2410.13341) | 当评委模型不比被评估模型更准确时，任何去偏方法最多只能将所需的地面真实标签减少一半，这暴露了LLM作为评委范式的根本局限。 |
| [^71] | [Score Attack: A Lower Bound Technique for Optimal Differentially Private Learning](https://arxiv.org/abs/2303.07152) | 本文提出“分数攻击”方法，为差分隐私下的参数估计极小极大风险提供最优下界，适用于任意具有分数统计量的统计模型。 |
| [^72] | [Sequential Batch Learning in Finite-Action Linear Contextual Bandits](https://arxiv.org/abs/2004.06321) | 本文提出了有限动作线性上下文赌博机中的序贯批量学习问题，并针对任意生成和共同高斯分布两种上下文设置进行了理论分析。 |
| [^73] | [ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription.](http://arxiv.org/abs/2307.15691) | ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。 |
| [^74] | [Spectral clustering in the Gaussian mixture block model.](http://arxiv.org/abs/2305.00979) | 本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。 |
| [^75] | [Doubly robust nearest neighbors in factor models.](http://arxiv.org/abs/2211.14297) | 该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。 |

# 详细

[^1]: 非交叉深度分位数回归用于分布生存预测

    Non-Crossing Deep Quantile Regression for Distributional Survival Prediction

    [https://arxiv.org/abs/2608.16864](https://arxiv.org/abs/2608.16864)

    本文提出了一种基于深度学习的非交叉分位数回归框架，用于生存分析中的分布预测，通过构造保证分位数排序一致性，并在多种复杂条件下优于现有方法。

    

    在生存分析中，协变量对事件风险的影响方式往往在早期和晚期失败时间之间有所不同，然而基于风险和均值的摘要将这些变化压缩为一个单一数字。基于分位数的建模则描述了原始时间尺度上的完整条件分布，但现有的删失数据方法要么不够灵活，要么产生逻辑上不一致的交叉分位数曲线。我们提出了一种用于右删失数据的删失非交叉分位数（CNQ）框架，该框架联合估计多个条件生存分位数，并通过构造保证有效的排序，其灵活性由Kolmogorov-Arnold和Transformer骨干网络提供，我们还建立了一个在所有拟合分位数水平上联合成立的有限样本超额风险界。在27种模拟设置和六个队列中，当条件分布复杂时，该框架的弹球损失低于基于分位数、风险和树的竞争方法。

    arXiv:2608.16864v1 Announce Type: cross  Abstract: In survival analysis the way covariates act on the risk of an event often differs between early and late failure times, yet hazard- and mean-based summaries collapse this variation into a single number. Quantile-based modeling instead describes the full conditional distribution on the original time scale, but existing censored-data methods are either inflexible or produce logically inconsistent crossing quantile curves. We propose a Censored Non-crossing Quantile (CNQ) framework for right-censored data that jointly estimates several conditional survival quantiles and guarantees valid ordering by construction, with flexibility supplied by Kolmogorov-Arnold and Transformer backbones, and we establish a finite-sample excess-risk bound holding jointly across all fitted quantile levels. Across 27 simulation settings and six cohorts the framework attains lower pinball loss than quantile-, hazard- and tree-based competitors whenever the condi
    
[^2]: 隐藏与寻找：在端到端可微分网络中学习解释

    Hide&Seek: Learning to Explain in an End-to-End Differentiable Network

    [https://arxiv.org/abs/2608.16689](https://arxiv.org/abs/2608.16689)

    本文提出Hide&Seek，一种端到端可微分模型，通过将特征移除转化为可微操作，在无信息泄漏的情况下联合学习特征选择与预测，实现高效训练和卓越性能。

    

    实例级特征选择是解释标注数据和黑盒模型预测的有价值工具。与全局特征选择技术相比，实例级方法动态地为每个实例识别重要特征。越来越多的方法学习一个选择器（识别重要特征）和一个预测器（利用这些特征进行预测）。然而，这些开创性方法面临信息泄漏和缺乏可微性等挑战，这可能减缓训练过程。在本文中，我们提出Hide&Seek，一种用于实例级特征选择的端到端可微分模型。我们在单一目标下联合学习特征选择和预测，且无信息泄漏。Hide&Seek在多种实验中优于现有最先进模型，并且训练速度快。我们通过将特征移除重新表述为可微分操作来实现这一目标，而不是丢弃特征。

    arXiv:2608.16689v1 Announce Type: cross  Abstract: Instance-wise feature selection is a valuable tool for interpreting labeled data and the predictions of black-box models. In contrast to global feature selection techniques, instance-wise methods dynamically identify important features for each instance. A growing number of methods learn a selector, which identifies important features, and a predictor, which uses these to make predictions. However, these pioneering methods face challenges including information leakage and lack of differentiability, which can slow training. In this paper, we present Hide&Seek, an end-to-end differentiable model for instance-wise feature selection. We jointly learn feature selection and prediction under a single objective without information leakage. Hide&Seek outperforms existing state-of-the-art models across a range of experiments and is fast to train. We achieve this by reformulating feature removal as a differentiable operation where instead of disc
    
[^3]: 密度重加权熵最优传输：解耦几何与采样密度

    Density-Reweighted Entropic Optimal Transport: Decoupling Geometry from Sampling Density

    [https://arxiv.org/abs/2608.16506](https://arxiv.org/abs/2608.16506)

    提出一种密度重加权熵最优传输框架，可调节采样密度影响，实现从标准EOT到纯几何驱动的对齐，并证明其收敛性。

    

    数据集对齐是科学和工程数据分析中的一个核心步骤，其目标是匹配不同数据集之间的观测值。熵最优传输（EOT）通过将跨数据集的亲和性编码在传输计划中，为这一任务提供了计算上可行的框架。然而，当两个数据集从几何相似但采样密度显著不同的低维结构中采样时，EOT计划可能根据相对采样密度而非几何邻近性匹配点，从而导致几何上误导性的对应关系。为解决此问题，我们提出了一种密度重加权EOT框架，其中采样密度对传输计划的影响可以根据需求被折减到期望程度，范围从标准EOT到完全由底层几何驱动的对齐。在适当的正则性条件下，我们建立了重加权EOT计划收敛到一族总体层面的结果。

    arXiv:2608.16506v1 Announce Type: cross  Abstract: Dataset alignment is a central step in data analysis across science and engineering, where the goal is to match observations between datasets. Entropic Optimal Transport (EOT) offers a computationally tractable framework for this task by encoding cross-dataset affinities in a transport plan. However, when two datasets are sampled from geometrically similar low-dimensional structures with substantially different sampling densities, the EOT plan may match points by relative sampling density rather than geometric proximity, yielding geometrically misleading correspondences. To address this issue, we propose a density-reweighted EOT framework in which the influence of sampling density on the transport plan can be discounted to a desired degree, ranging from standard EOT to alignment driven purely by underlying geometry. Under suitable regularity conditions, we establish convergence of the reweighted EOT plan to a family of population-level
    
[^4]: 并行高斯过程强盗优化的改进遗憾分析

    Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization

    [https://arxiv.org/abs/2608.16492](https://arxiv.org/abs/2608.16492)

    本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。

    

    本文研究了并行高斯过程（GP）强盗优化的遗憾分析。广泛使用的GP批量上置信界和GP批量汤普森采样（GP-BTS）的已知遗憾上界，在批量大小$Q$上存在一个乘性因子。为避免这种性能退化，现有分析需要在优化开始时对$Q$进行多项式数量的不确定性采样（US）。然而，这种初始US阶段在实践中往往效果不佳。本文以GP-BTS为例，表明无需初始US阶段即可实现无$Q$乘性因子的遗憾上界。此外，我们展示了在无噪声设置下，遗憾上界远优于有噪声设置，这与顺序GP强盗设置中的情况一致。

    arXiv:2608.16492v1 Announce Type: cross  Abstract: This paper studies the regret analysis for parallel Gaussian process (GP) bandit optimization. The known regret upper bounds for the widely used GP batched upper confidence bound and GP batched Thompson sampling (GP-BTS) suffer from a multiplicative factor with respect to the batch size $Q$. To avoid this degradation, existing analyses require a polynomial number of uncertainty sampling (US) for $Q$ at the beginning of optimization. However, this initial US phase is often ineffective in practice. This paper shows that the regret upper bound without the multiplicative factor on $Q$ can be achieved without the initial US phase, using GP-BTS as an example. Furthermore, we show much better regret upper bounds in the noiseless setting than in the noisy setting, as in the sequential GP bandit setting.
    
[^5]: 基于证据偏差准则的深度自适应设计

    Deep adaptive design with an evidential bias criterion

    [https://arxiv.org/abs/2608.16466](https://arxiv.org/abs/2608.16466)

    本文提出了一种基于“反偏置”证据准则的深度自适应实验设计方法，以更好地控制实验产生误导性证据的风险，弥补传统期望信息增益准则的不足。

    

    贝叶斯最优实验设计（BOED）旨在通过优化反映实验目标的期望效用函数来收集信息丰富的数据。然而，对于常见的效用函数和复杂模型，这种优化在计算上具有挑战性，尤其是对于序列或自适应设计，其中设计和数据收集交替进行，因此必须考虑已观测数据的反馈。大多数现有BOED研究采用信息增益作为效用，导致期望信息增益（EIG）准则。虽然EIG广泛有用，但它可能并不总能充分反映实验目标。EIG可视为奖励平均上对真相产生大量正证据的实验，但它并不直接控制实验产生误导性证据的风险。在此，我们考虑一种替代准则，称为“反偏置”（BA），该准则优先关注这种控制。为解决此问题，我们提出了深度自适应设计方法。

    arXiv:2608.16466v1 Announce Type: cross  Abstract: Bayesian optimal experimental design (BOED) aims to collect informative data by optimizing an expected utility reflecting the goals of an experiment. However, this optimization is computationally challenging for common utilities and complex models. This is especially so for sequential or adaptive designs, where design and data collection alternate, so that feedback from already observed data must be taken into account. Most existing BOED research employs information gain as the utility, leading to the expected information gain (EIG) criterion. While EIG is widely useful, it may not always adequately reflect experimental goals. EIG can be viewed as rewarding experiments that produce large positive evidence for the truth on average, but it does not directly control the risk of an experiment producing misleading evidence. Here we consider an alternative criterion, which we call bias against (BA), that prioritizes such control. To address 
    
[^6]: 再生核巴拿赫空间上统计逆问题的收敛性分析

    Convergence Analysis of Statistical Inverse Problems on Reproducing Kernel Banach Spaces

    [https://arxiv.org/abs/2608.16404](https://arxiv.org/abs/2608.16404)

    本文在再生核巴拿赫空间框架下，利用Tikhonov正则化方法对统计逆问题中的解进行稳定估计，并首次给出了随数据量增加时的收敛性证明及收敛速率。

    

    arXiv:2608.16404v1 公告类型：交叉 摘要：近年来，由于统计学习理论和函数分析方法在机器学习和人工智能领域中的重要性日益增长，统计逆问题引起了广泛关注。本文研究了满足方程 $Au = g$ 的元素 $u^{\dagger}$ 的稳定逼近问题，其中 $A$ 是将巴拿赫空间映射到适当函数空间的线性算子。函数 $g$ 仅通过独立同分布的数据点观测得到，这些数据点受到噪声污染，并假设服从未知分布 $\rho$。我们采用Tikhonov正则化方案，利用统计学习技术和再生核巴拿赫空间的框架来估计解。我们建立了估计解相对于真实解的收敛性，并随着数据点数量的增加推导出其收敛速率。

    arXiv:2608.16404v1 Announce Type: cross  Abstract: Statistical inverse problems have garnered significant attention in recent years due to the growing importance of statistical learning theory and functional analytic approaches in the fields of machine learning and artificial intelligence. In this paper, we investigate the stable approximation of the element $u^{\dagger}$ that satisfies the equation $Au = g$, where $A$ is a linear operator that maps a Banach space into an appropriate function space. The function $g$ is observed only through independently and identically distributed data points that are corrupted by noise and assumed to follow an unknown distribution $\rho$. We employ the Tikhonov regularization scheme, leveraging statistical learning techniques and the framework of reproducing kernel Banach spaces to estimate the solution. We establish convergence and derive the convergence rate of the estimated solution with respect to the true solution as the number of data points in
    
[^7]: LiD-GLM：利普希茨约束的深度广义线性模型

    LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models

    [https://arxiv.org/abs/2608.16340](https://arxiv.org/abs/2608.16340)

    提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。

    

    摘要：arXiv:2608.16340v1 公告类型：交叉 摘要：将传统统计模型与神经网络（NN）组件结合成半结构化混合模型，是一种引人入胜的方法，旨在构建理想情况下兼具传统可解释性与神经网络前所未有的灵活性的模型。为了保持可解释性，通常需要限制神经网络组件，以防止它们主导模型。然而，现有对神经网络组件施加结构约束的方法严重限制了模型的灵活性；相反，仅施加弱且间接约束的方法则失去了有意义的可解释性。因此，我们提出的方法利用可逆残差神经网络（i-ResNets）为广义线性模型配备非线性参数估计和对其分布假设的灵活校正，同时始终保留所建模分布在（原线性）变量上的随机单调性。

    arXiv:2608.16340v1 Announce Type: cross  Abstract: The combination of traditional statistical models and neural network (NN) components into semi-structured hybrid models is an intriguing approach to construct models that, ideally, combine traditional interpretability with the unprecedented flexibility of NNs. In order to preserve interpretability, it is usually necessary to restrict the NN components to prevent them from dominating the model. However, existing methods that enforce structural constraints on their NN components severely limit their models' flexibility; in contrast, methods that only enforce weak, indirect constraints lose meaningful interpretability. The method we propose therefore leverages invertible residual neural networks (i-ResNets) to equip generalized linear models with both nonlinear parameter estimation and a flexible correction of their distributional assumptions while always retaining stochastic monotonicity of the modeled distribution in the (formerly linea
    
[^8]: 基于廉价辅助信号的条件语言模型评估

    Conditional Evaluation of Language Models with Cheap Auxiliary Signals

    [https://arxiv.org/abs/2608.16210](https://arxiv.org/abs/2608.16210)

    本文提出LACE方法，利用廉价辅助信号通过局部中心化技术实现条件语言模型评估的半监督估计，在保证无偏性的同时显著降低评估成本。

    

    arXiv:2608.16210v1 公告类型：新 摘要：总体准确率掩盖了模型在何处成功或失败。仅从金标准标签估计条件性能概况成本高昂，而诸如LLM评判分数、成对比较、置信度分数和评判分歧特征等廉价辅助信号虽可为每个基准项目收集，但往往存在偏差或校准不足。我们提出LACE（局部增强控制变量评估），一种用于条件LLM评估的半监督估计器。关键步骤是局部中心化：在目标概况区域内减去廉价信号的条件均值后，任何线性增强具有零条件均值，因此无法改变估计目标。增强系数仅用于提高效率，局部岭控制变量将标记子集的金标准残差均值与完整项目池的廉价信号均值相结合。我们证明了无校准识别性、组的无偏性。

    arXiv:2608.16210v1 Announce Type: new  Abstract: Aggregate accuracy hides where models succeed and fail. Estimating conditional performance profiles from gold labels alone is expensive, while cheap auxiliary signals such as LLM-judge scores, pairwise comparisons, confidence scores, and judge-disagreement features can be collected for every benchmark item but are often biased or miscalibrated. We propose LACE (Local Augmented Control-Variate Evaluation), a semi-supervised estimator for conditional LLM evaluation. The key step is local centering: after subtracting the conditional mean of a cheap signal within the target profile region, any linear augmentation has zero conditional mean and therefore cannot change the estimand. The augmentation coefficient is used only for efficiency, and a local ridge control variate combines a gold-label residual mean from the labeled subset with a cheap-signal mean from the full item pool. We prove calibration-free identification, unbiasedness for group
    
[^9]: 编码汉克尔多项式混沌：主导多项式混沌模态的谱识别

    Coded Hankel Polynomial Chaos: Spectral Identification of Dominant Polynomial-Chaos Modes

    [https://arxiv.org/abs/2608.16126](https://arxiv.org/abs/2608.16126)

    本文提出编码汉克尔多项式混沌（CH-PC）方法，通过谱分析和低秩汉克尔矩阵从多项式混沌展开中高效识别主导模态，并利用相位编码冗余提升鲁棒性。

    

    摘要：主导多项式混沌模态的识别通常被表述为在采样多元多项式字典上的稀疏回归问题。我们开发了编码汉克尔多项式混沌（CH-PC），这是一种用于主导模态识别的互补谱方法。有限生成变换将多项式混沌展开系数转换为系数生成多项式，并沿几何相位轨道评估产生有限指数和。其模型阶数和谱节点由低秩汉克尔矩阵编码，而坐标相位移动附加单位根标签，从中恢复完整的多项式多指标。坐标移动探针被组合为公共节点快照，并且当单一谱编码条件不佳时，独立相位编码提供冗余表示。对于有限观测，总体、有限数据和观测探针保持区分：采样或求积误差...

    arXiv:2608.16126v1 Announce Type: cross  Abstract: Identification of dominant polynomial-chaos modes is usually formulated as a sparse-regression problem on a sampled multivariate polynomial dictionary. We develop coded Hankel polynomial chaos (CH-PC), a complementary spectral formulation for dominant-mode identification. A finite generating transform converts PCE coefficients into a coefficient-generating polynomial, and evaluation along a geometric phase orbit produces a finite exponential sum. Its model order and spectral nodes are encoded by low-rank Hankel matrices, while coordinate phase shifts attach root-of-unity labels from which the full polynomial multi-indices are recovered. Coordinate-shifted probes are combined as common-node snapshots, and independent phase encodings provide redundant representations when a single spectral encoding is poorly conditioned. For finite observations, population, finite-data, and observed probes are kept distinct: sampling or quadrature error 
    
[^10]: EMS核心集：一种高效的Sinkhorn核心集期望最大化算法

    EMS Coreset: An Efficient Expectation-Maximization Algorithm for Sinkhorn Coreset

    [https://arxiv.org/abs/2608.16101](https://arxiv.org/abs/2608.16101)

    本文提出了一种高效的Sinkhorn核心集算法，通过非均匀权重实现闭式更新，大幅降低计算成本，同时保证近似质量和稳定性。

    

    arXiv:2608.16101v1 公告类型：交叉 摘要：核心集将大型数据集蒸馏为小型、具有代表性的子集，以支持高效的下游学习。然而，基于最优传输（OT）的选择通常需要密集计算传输计划，限制了其可扩展性。我们引入了一种可扩展的Sinkhorn核心集方法，通过允许非均匀核心集权重，实现了熵正则化OT耦合的闭式更新。这产生了通过软分配泛化k均值的质心。我们建立了所选度量的渐近一致性以及对数据扰动的Lipschitz稳定性，提供了准确性和鲁棒性保证。在合成和真实世界基准测试中，与基于Wasserstein和标准Sinkhorn的核心集选择相比，所提出的方法在实现竞争性或改进的近似质量的同时，显著减少了运行时间，尤其是在大规模场景下。

    arXiv:2608.16101v1 Announce Type: cross  Abstract: Coresets distill large datasets into small, representative subsets for efficient downstream learning. Yet Optimal Transport (OT)-based selection typically requires intensive computation of transport plans, limiting scalability. We introduce a scalable Sinkhorn coreset method that permits closed-form updates of the entropically regularized OT coupling by allowing non-uniform coreset weights. This produces centroids that generalize k-means via soft assignments. We establish asymptotic consistency of the selected measure and Lipschitz stability to data perturbations, providing accuracy and robustness guarantees. Across synthetic and real-world benchmarks, the proposed method achieves competitive or improved approximation quality while substantially reducing runtime compared to Wasserstein- and standard Sinkhorn-based coreset selection, especially at large scale.
    
[^11]: 带记忆的广义线性老虎机

    Generalized Linear Bandits with Memory

    [https://arxiv.org/abs/2608.15848](https://arxiv.org/abs/2608.15848)

    本文通过改进分析并设计基于收缩置信区间的分块算法，在带记忆的广义线性老虎机中实现了$\sqrt{T}$遗憾率，超越了先前$\tilde{O}(T^{3/4})$的界限。

    

    arXiv:2608.15848v1 公告类型：交叉 摘要：我们研究了带记忆的广义线性老虎机，这是一种内生非平稳设置，其中奖励通过一个有限记忆矩阵依赖于过去的动作。基于先前针对线性模型的工作（Clerici等人，2024），我们证明了先前已知的$\tilde{O}(T^{3/4})$遗憾界源于松散的分析，并提供了一种改进的分析，在线性情况下恢复了$\tilde{O}(\sqrt{T})$的遗憾率。然后，我们将这一改进扩展到广义线性模型，并提出了一种基于收缩置信区间的分块算法。我们的算法实现了$\tilde{O}\left(\sqrt{mT} + d\sqrt{T} + \sqrt{\kappa}\, d^{2} m^{1/4} T^{1/4} + \kappa d^{2} \right)$的遗憾界，其中$d$表示特征维度，$m$表示记忆长度，$\kappa$表示链接函数的曲率参数。尽管存在非线性奖励和记忆效应，该算法仍达到了$\sqrt{T}$类型的速率。据我们所知，这一分析...

    arXiv:2608.15848v1 Announce Type: cross  Abstract: We study generalized linear bandits with memory, an endogenous non-stationary setting in which rewards depend on past actions through a finite memory matrix. Building on prior work for linear models (Clerici et al., 2024), we show that the previously known $\tilde{O}(T^{3/4})$ regret bound stems from a loose analysis, and we provide a sharpened analysis that recovers a $\tilde{O}(\sqrt{T})$ regret rate in the linear case. We then extend this improvement to generalized linear models and propose a block-wise algorithm based on shrunken confidence bounds. Our algorithm achieves a regret bound of $\tilde{O}\left(\sqrt{mT} + d\sqrt{T} + \sqrt{\kappa}\, d^{2} m^{1/4} T^{1/4} + \kappa d^{2} \right)$, where $d$ denotes the feature dimension, $m$ the memory length, and $\kappa$ a curvature parameter of the link function. This attains a $\sqrt{T}$-type rate despite nonlinear rewards and memory effects. To the best of our knowledge, this analysis
    
[^12]: 面向股票交易中稳定强化学习的自监督辅助任务发现

    Self-Supervised Auxiliary Task Discovery for Stable Reinforcement Learning in Stock Trading

    [https://arxiv.org/abs/2608.15841](https://arxiv.org/abs/2608.15841)

    本文提出一种自监督框架，通过自动发现通用价值函数形式的辅助任务来增强状态表示，从而提升股票交易强化学习的稳定性和性能。

    

    arXiv:2608.15841v1 公告类型：新 摘要：强化学习作为一种数据驱动的股票交易方法日益受到关注。然而，由于市场行为的非平稳性和噪声奖励信号，学习一个既盈利又稳定的策略仍然具有挑战性。辅助任务常用于改善表示学习并稳定训练，但它们通常需要手动设计，并且高度依赖于对目标和预测视界的先验假设。这种固定设计可能无法在变化的市场环境中保持适用。在本工作中，我们提出了一种自监督框架，能够自动发现辅助任务以支持股票交易的强化学习。这些辅助任务被构建为通用价值函数，使其预测能够丰富学习到的状态表示并协助策略优化。该框架由两个网络组成。主网络学习交易策略以及辅助预测。

    arXiv:2608.15841v1 Announce Type: new  Abstract: Reinforcement learning has gained increasing attention as a data-driven approach for stock trading. However, learning a policy that is both profitable and stable remains challenging due to non-stationary market behaviour and noisy reward signals. Auxiliary tasks are often used to improve representation learning and stabilize training, yet they are usually designed manually and depend heavily on prior assumptions about targets and prediction horizons. Such fixed designs may not remain suitable across changing market regimes. In this work, we propose a self-supervised framework that automatically discovers auxiliary tasks to support reinforcement learning for stock trading. The auxiliary tasks are formulated as General Value Functions so that their predictions enrich the learned state representation and assist policy optimization. The framework consists of two networks. The main network learns the trading policy along with the auxiliary pr
    
[^13]: 确定因果方向需要多少样本？双变量LiNGAM的尖锐极小极大界

    How Many Samples Are Needed to Determine Causal Direction? Sharp Minimax Bounds for Bivariate LiNGAM

    [https://arxiv.org/abs/2608.15840](https://arxiv.org/abs/2608.15840)

    本文首次给出了双变量LiNGAM中确定因果方向所需样本数的尖锐极小极大下界，揭示了因果效应强度、非高斯性和扰动尺度差异如何共同决定样本复杂度。

    

    arXiv:2608.15840v1 公告类型：交叉 摘要：我们研究了确定两个线性相关变量之间的因果方向需要多少观测值。经典LiNGAM理论表明，独立的非高斯扰动可以识别方向，但当因果效应较弱或扰动接近高斯分布时，并未量化难度。设$\beta$为结构系数绝对值的下界，$\nu$衡量每个标准化扰动与高斯分布的距离，并设扰动尺度位于$[\underline\sigma,\overline\sigma]$范围内。我们证明了尖锐的局部极小极大律 \[   N_2^\star(\beta,\nu,\delta)   \asymp   \frac{\log(1/\delta)}   {d_\beta^2+\beta^2\nu^2},   \qquad   d_\beta=   \left[\beta^2  \left(1-\frac{\underline\sigma^2}{\overline\sigma^2}\right)\right]_+. \] 以往理论建立了总体可识别性或假设两个方向之间存在固定分离。相比之下，我们建立了尖锐的样本复杂度。

    arXiv:2608.15840v1 Announce Type: cross  Abstract: We study how many observations are needed to determine the causal direction between two linearly related variables. Classical LiNGAM theory shows that independent non-Gaussian disturbances identify the direction, but does not quantify the difficulty when the causal effect is weak or the disturbances are nearly Gaussian. Let $\beta$ bound the absolute structural coefficient from below, let $\nu$ measure each standardized disturbance's distance from Gaussianity, and let the disturbance scales lie in $[\underline\sigma,\overline\sigma]$. We prove the sharp local minimax law \[   N_2^\star(\beta,\nu,\delta)   \asymp   \frac{\log(1/\delta)}   {d_\beta^2+\beta^2\nu^2},   \qquad   d_\beta=   \left[\beta^2  \left(1-\frac{\underline\sigma^2}{\overline\sigma^2}\right)\right]_+. \] Previous theory established population identifiability or assumed a fixed separation between the two directions. By contrast, we establish the sharp sample complexit
    
[^14]: 语言模型的交叉熵风险估计：不一致性必然密集，且保留法也不例外

    Cross-Entropy Risk Estimation for Language Models: Inconsistency Must Be Dense, and the Holdout Method Is No Exception

    [https://arxiv.org/abs/2608.15798](https://arxiv.org/abs/2608.15798)

    本文证明语言模型的每令牌交叉熵风险无法被一致估计，无论使用何种估计器（包括保留法），这是由有限与无限风险在状态空间中的拓扑密集性决定的根本性限制。

    

    arXiv:2608.15798v1 公告类型：新 摘要：语言模型通过其保留的每令牌交叉熵风险进行比较——这是拟合缩放定律所依据的量。我们证明该风险无法被一致地估计。一致性，即收敛到估计目标，是相对于一个“可能的世界状态”定义的：该状态由数据生成分布和我们最终训练的模型组成的一对构成。对模型以及数据生成机制进行量化至关重要，因为决定模型风险是否可估计的是其权重诱导分布的尾部性质，而任何样本都无法揭示这一点。每令牌交叉熵风险难以估计，源于一个拓扑事实：在可能的状态中，有限风险和无限风险各自任意接近另一类的每个实例。因此，没有任何估计器——不仅仅是保留平均值——能在风险定义的所有状态下保持一致。更糟的是，不一致估计在两种情况下仍然存在。

    arXiv:2608.15798v1 Announce Type: new  Abstract: Language models are compared by their held-out per-token cross-entropy risk---the quantity scaling laws are fitted to. We show that it cannot be consistently estimated. Consistency, or convergence to the estimand, is defined relative to a \emph{possible state of the world}: a pair consisting of a data-generating distribution and a model we turn out to train. Quantifying over models as well as data-generating mechanisms is essential, because what decides whether a model's risk is estimable is a tail property of the distribution its weights induce, which no sample reveals. The per-token cross-entropy risk is hard to estimate because of a topological fact: among the possible states, finite risk and infinite risk each lie arbitrarily close to every instance of the other. Consequently no estimator---not merely the holdout average---is consistent at every state at which the risk is defined. Worse, inconsistent estimation persists under both bo
    
[^15]: 协变量偏移下代理衍生模型的推断评估

    Inferential Evaluation of Surrogate-Derived Models under Covariate Shift

    [https://arxiv.org/abs/2608.15783](https://arxiv.org/abs/2608.15783)

    本文提出了一种在协变量偏移下评估代理衍生模型目标性能的方法，利用三样本设置和交叉拟合估计器，结合密度比与核修正来处理标签稀缺和分布差异问题。

    

    arXiv:2608.15783v1 公告类型：交叉 摘要：在迁移学习场景中，从丰富的代理标签中衍生出的模型可能被部署到目标群体中，而该群体中金标准结果未被观测到。评估其在目标环境中的性能对于确定基于该模型的决策是否仍然可靠至关重要，但当金标准标签稀缺且不同数据源间的协变量分布存在差异时，这一评估变得困难。我们研究了一个三样本设置，包含一个小的金标准标记源、一个较大的代理标记源和一个未标记的目标。在条件可迁移性假设下，我们针对目标群体中的潜在金标准结果评估代理衍生模型。我们提出了交叉拟合估计器，通过源特定的密度比从两个标记源传递信息。我们还结合了结果回归增强与核修正，用于在阈值附近估计模型，并考虑了来自所有三个样本的不确定性。

    arXiv:2608.15783v1 Announce Type: cross  Abstract: In transfer-learning settings, a model derived from abundant surrogate labels may be deployed in a target population where gold-standard outcomes are unobserved. Evaluating its target performance is essential for determining whether decisions based on the model remain reliable, yet it is difficult when gold labels are scarce, and covariate distributions differ across data sources. We study a three-sample setting with a small gold-labeled source, a larger surrogate-labeled source, and an unlabeled target. Under conditional transportability, we evaluate the surrogate-derived model against the latent gold-standard outcome in the target population. We propose cross-fitted estimators that transport information from the two labeled sources through source-specific density ratios. We also combine outcome-regression augmentation with a kernel correction for estimating the model near a threshold, accounting for uncertainty from all three samples
    
[^16]: 基于重心对抗逆向强化学习的股票交易策略学习

    Learning Stock Trading Policies via Barycenter-Based Adversarial Inverse Reinforcement Learning

    [https://arxiv.org/abs/2608.15770](https://arxiv.org/abs/2608.15770)

    BRaG通过重心聚合多专家策略并利用对抗模仿学习预训练交易模型，结合控制屏障函数实现风险约束的股票交易策略学习。

    

    摘要：使用强化学习设计有效的交易策略仍面临挑战，原因包括奖励的延迟性和噪声、探索能力差，以及难以强制执行明确的风险约束。在这项工作中，我们提出了BRaG，一种基于重心的对抗逆向强化学习框架，用于股票交易，能够从多个异构专家策略中学习交易行为。BRaG使用性能加权的Wasserstein重心聚合专家演示，生成一个稳定的伪专家表示，捕捉不同交易风格间的共享结构。该表示用于通过对抗模仿学习预训练交易策略，缓解强化学习中的不稳定探索问题。预训练策略随后使用真实市场奖励通过强化学习进行优化。为确保风险感知的决策，BRaG引入了控制屏障函数，以约束...

    arXiv:2608.15770v1 Announce Type: new  Abstract: Designing effective trading strategies using reinforcement learning remains challenging due to delayed and noisy rewards, poor exploration, and the difficulty of enforcing explicit risk constraints. In this work, we propose BRaG, a barycenter-based adversarial inverse reinforcement learning framework for stock trading that learns trading behavior from multiple heterogeneous expert strategies. BRaG aggregates expert demonstrations using a performance-weighted Wasserstein barycenter, yielding a stable pseudo-expert representation that captures shared structure across diverse trading styles. This representation is used to pretrain a trading policy via adversarial imitation learning, which alleviates unstable exploration during reinforcement learning. The pretrained policy is subsequently refined using reinforcement learning with true market rewards. To ensure risk-aware decision-making, BRaG incorporates control barrier functions that const
    
[^17]: FirstDiff：基于初始噪声预测的一步扩散多变量时间序列异常检测

    FirstDiff: One-Step Diffusion-Based Anomaly Detection for Multivariate Time Series via Initial Noise Prediction

    [https://arxiv.org/abs/2608.15727](https://arxiv.org/abs/2608.15727)

    本文提出FirstDiff，通过仅使用反向扩散初始步骤的预测噪声进行异常检测，大幅降低计算成本并利用中间扩散信息，实现多变量时间序列的高效一步检测。

    

    摘要：扩散模型近来通过迭代去噪学习正常数据的分布，在多变量时间序列异常检测中展现出强大潜力。然而，现有的基于扩散的方法通常在完成反向扩散过程后才进行异常检测，主要依赖最终重建信号，忽视了去噪过程中产生的信息性表示。这种设计带来了高昂的计算成本，并限制了中间扩散信息在异常检测中的应用。在本文中，我们提出FirstDiff，一种基于扩散的异常检测框架，其核心观察是：在初始反向扩散评估时预测的扩散噪声已包含足够信息，可用于准确的异常检测。FirstDiff利用验证数据对正常行为下预测扩散噪声的统计分布进行建模，从而能够实现...

    arXiv:2608.15727v1 Announce Type: cross  Abstract: Diffusion models have recently shown strong potential for multivariate time-series anomaly detection by learning the distribution of normal data through iterative denoising. Existing diffusion-based approaches, however, typically perform anomaly detection after completing the reverse diffusion process, relying primarily on the final reconstructed signal and overlooking informative representations produced during denoising. This design incurs substantial computational cost and limits the use of intermediate diffusion information for anomaly detection.   In this paper, we propose FirstDiff, a diffusion-based anomaly detection framework based on the observation that the predicted diffusion noise at the initial reverse-diffusion evaluation already contains sufficient information for accurate anomaly detection. FirstDiff models the statistical distribution of predicted diffusion noise under normal behavior using validation data, enabling an
    
[^18]: 关于CART的停止规则与空间适应性研究

    On Stopping Rules and Spatial Adaptation for CART

    [https://arxiv.org/abs/2608.15649](https://arxiv.org/abs/2608.15649)

    本文证明了CART算法结合最小杂质减少停止规则和适当阈值，能在空间异质和各向异性平滑条件下实现逐点极小极大最优速率，并指出常用停止规则无法实现空间适应性。

    

    arXiv:2608.15649v1 公告类型：交叉 摘要：流行的CART算法用于回归树时，结合了贪心分裂规则与停止规则，但尽管分裂规则已被广泛研究，停止规则的统计作用仍不太清楚。同时，尽管通过贝叶斯方法或经验风险最小化（ERM）拟合的回归树已被证明对局部平滑性和各向异性具有空间适应性，但尚不清楚CART是否能实现同样的适应性。我们通过证明，在空间异质和各向异性的平滑性以及回归函数和协变量分布的适当结构假设下，使用最小杂质减少（MID）停止规则和适当阈值的CART，其逐点速率在双对数因子内达到极小极大最优。这些速率在域内所有点上同时成立。此外，我们证明，在广泛使用的停止规则下无法实现空间适应性。

    arXiv:2608.15649v1 Announce Type: cross  Abstract: The popular CART algorithm for regression trees combines a greedy splitting rule with a stopping rule, but while the splitting rule has been well studied, the statistical role of stopping rules is less well understood. Meanwhile, although regression trees fit using Bayesian methods or via empirical risk minimization (ERM) have been shown to be spatially adaptive to local smoothness and anisotropy, it is unknown whether CART can achieve the same adaptation. We address these gaps by proving that, under spatially heterogeneous and anisotropic smoothness and appropriate structural assumptions on the regression function and covariate distribution, CART with the minimum impurity decrease (MID) stopping rule and a suitable threshold achieves pointwise rates that are minimax up to logarithmic factors. These rates hold simultaneously over all points in the domain. Moreover, we prove that spatial adaptation cannot be achieved under the widely us
    
[^19]: 稀疏原型编码支撑跨模态分类与预测

    Sparse Prototype Code Underlies Classification and Prediction Across Modalities

    [https://arxiv.org/abs/2608.15632](https://arxiv.org/abs/2608.15632)

    本文发现分类任务中神经表征存在跨模态的普遍几何结构，并提出一个解析平均场理论，通过类质心坐标的变异性和类半径重整化准确预测分类性能。

    

    神经表征已成为研究现代AI模型内部机制的核心工具，但其复杂的高维结构使其难以解释。我们表明，分类任务产生了一种普遍的表征几何结构，在视觉、音频和语言处理领域的最先进模型中共享。关键结构在于，类内变异性在表征空间中并非随机。相反，其与分类器相关的成分与该类自身的质心及其竞争类别的质心具有强烈且结构化的相关性。基于这一观察，我们推导了一个解析平均场理论，该理论主要由真实类和竞争类质心坐标上的变异性以及一个全局的类半径重整化（用于补偿真实表征的非高斯统计）所主导。该理论准确预测了分类准确性。

    arXiv:2608.15632v1 Announce Type: cross  Abstract: Neural representations have become a central tool for studying the internal mechanisms of modern AI models, yet their complex high-dimensional structure makes them difficult to interpret. We show that classification tasks give rise to a universal representational geometry, shared across state-of-the-art models in vision, audio, and language processing. The key structure is that within-class variability is not random in representation space. Instead, its classifier-relevant component has strong and structured correlations with the class's own centroid and with the centroids of its competing classes. Building on this observation, we derive an analytical mean-field theory governed mainly by the variability along true-class and rival-class centroid coordinates, together with a global renormalization of the class radius that compensates for the non-Gaussian statistics of real representations. The theory accurately predicts classification ac
    
[^20]: 量子机器学习用于电力系统攻击检测的基准测试：评估选择在模型之前就决定了结果

    Benchmarking Quantum Machine Learning for Power-System Attack Detection: Evaluation Choices Decide the Outcome Before the Models Do

    [https://arxiv.org/abs/2608.15617](https://arxiv.org/abs/2608.15617)

    该论文通过系统性的基准测试揭示，电力系统攻击检测中量子机器学习与经典模型的比较结果高度依赖于评估协议中的八项关键选择，这些选择在模型运行前就决定了结论走向。

    

    arXiv:2608.15617v1 公告类型：新 摘要：针对电力系统网络攻击的机器学习检测器本身也是攻击面，而量子机器学习已被提议用于此目的。我们在公共电力系统攻击数据（密西西比州立大学/ORNL）上，将保真度核支持向量机和变分分类器与六种调优后的经典模型进行基准测试，涵盖白盒、迁移、基于决策的黑盒和投毒攻击。我们的主要发现是方法论层面的：基准测试的答案在模型之前就由评估者的选择决定了。评估协议中的六项选择和基准测试本身调优中的两项选择，每一项都在固定模型下逆转或移动了一个结论。最大的是数据分割：行级协议得分为0.905宏F1，而保留整个源文件时得分为0.594；在封顶匹配维度设置下，量子部分在随机噪声范围内，经典部分高出0.024。保真度核在受到攻击前看起来最稳健。

    arXiv:2608.15617v1 Announce Type: new  Abstract: Machine-learning detectors for power-system cyberattacks are themselves attack surfaces, and quantum machine learning has been proposed for them. We benchmark fidelity-kernel SVMs and variational classifiers against six tuned classical models on public power-system attack data (Mississippi State/ORNL), across white-box, transfer, decision-based black-box, and poisoning attacks. Our headline finding is methodological: the benchmark's answers are set by the evaluator's choices before the models. Eight choices -- six in the evaluation protocol, two in the tuning the benchmark itself runs -- each reversed or moved a conclusion at fixed models. The largest is the split: the row-level protocol scores 0.905 macro-F1 where holding whole source files out leaves 0.594, and in the capped matched-dimensionality regime the quantum arm sits within noise of chance with the classical arm 0.024 above it. A fidelity kernel looks most robust until attacked
    
[^21]: PERO：面向加密流量分类的高效鲁棒基础模型后训练方法

    PERO: Efficient Robust Post-Training Foundation Models for Encrypted Traffic Classification

    [https://arxiv.org/abs/2608.15504](https://arxiv.org/abs/2608.15504)

    本文提出PERO框架，通过轻量级代理预评估高损失样本，实现加密流量基础模型的高效鲁棒后训练，以应对风险敏感场景中的罕见高损失错误。

    

    加密流量分类对网络安全至关重要，然而实际部署中天然对罕见但高损失的错误（如恶意流量误分类）敏感。加密流量基础模型作为一种有前景的通用技术，能够实现出色的整体性能。然而，采用经验风险最小化等标准目标往往忽视了高风险尾部事件，且常用的性能指标难以反映风险敏感场景下的鲁棒性局限。直接应用条件风险价值等鲁棒优化目标对大型模型进行后训练在计算上不可行，因为识别高损失样本会消耗大量计算资源。为此，我们提出预评估鲁棒优化（PERO），一种用于加密流量基础模型的高效鲁棒后训练框架。PERO利用轻量级代理来估计高损失样本，从而避免昂贵计算。

    arXiv:2608.15504v1 Announce Type: new  Abstract: Encrypted traffic classification is vital for network security, yet real-world deployments are inherently sensitive to rare but high-loss errors such as misclassification of malicious traffic. The encrypted traffic foundation model, as a promising general-purpose technique, can achieve impressive overall performance. However, employing standard objectives such as empirical risk minimization often overlooks high-risk tail events, and commonly used performance metrics hardly reflect robustness limitations in risk-sensitive scenarios. Directly applying robust optimization objectives, such as conditional value-at-risk, to post-training is computationally prohibitive for large models, as identifying high-loss samples exhausts substantial computation. To this end, we propose Pre-Evaluation Robust Optimization (PERO), an efficient robust post-training framework for encrypted traffic foundation models. PERO employs a lightweight proxy to estimat
    
[^22]: 广义分层保形预测

    Generalized Hierarchical Conformal Prediction

    [https://arxiv.org/abs/2608.15500](https://arxiv.org/abs/2608.15500)

    提出广义分层保形预测（GHCP），通过随机分配参考组大小恢复对称性，使分层保形预测能在仅有少量目标组观测时有效利用数据。

    

    arXiv:2608.15500v1 公告类型：交叉 摘要：许多预测问题在数据按组收集的情况下出现。在此设定下，分层保形预测（HCP）（Lee 等，2026）在分层可交换性条件下，为来自先前未见组的新观测提供了无分布预测集。然而，在许多应用中，预测仅在收集了目标组的少量观测后才进行。标准 HCP 无法利用这些观测，因为其所需的对称条件在此设定下不成立。同时，初始样本可能仍然太小，以至于在测试组内应用标准保形预测无法提供有用信息。我们为这一设定开发了预测推断方法。我们提出的方法，广义 HCP（GHCP），通过为测试组分配一个随机“捐赠”的参考组大小，恢复了保形推断所需的对称性。GHCP 进一步利用了初始测试组的观测。

    arXiv:2608.15500v1 Announce Type: cross  Abstract: Many prediction problems arise with data collected in groups. In this setting, hierarchical conformal prediction (HCP) (Lee et al., 2026) provides distribution-free prediction sets for a new observation from a previously unseen group under hierarchical exchangeability. In many applications, however, prediction is conducted only after a few observations from the group of interest have already been collected. Standard HCP cannot leverage these observations, as its required symmetry conditions do not hold in this setting. At the same time, the initial sample may still be too small for standard conformal prediction applied within the test group to be informative.   We develop predictive inference methods for this setting. Our proposed method, Generalized HCP (GHCP), restores the relevant symmetry needed for conformal inference by assigning the test group a randomly "donated" reference group size. GHCP further leverages the initial test gro
    
[^23]: 网络化信息聚合的最优下界

    Optimal Lower Bounds for Networked Information Aggregation

    [https://arxiv.org/abs/2608.15472](https://arxiv.org/abs/2608.15472)

    本文解决了网络化信息聚合中均方误差下界的开放问题，确定了最优下界，填补了先前上界与下界之间的差距。

    

    arXiv:2608.15472v1 公告类型：交叉 摘要：Kearns等人（2026年）研究的网络化信息聚合问题涉及一组位于有向无环图$G$顶点上的学习者，每个学习者根据局部特征以及其父节点学习到的预测器，学习一个固定随机变量$Y$的线性预测器$\widehat Y$。学习过程迭代进行，学习者按照$G$的拓扑排序排列。主要关注的数量是当前学习者在这种信息流约束下，相对于使用所有已见特征的最佳线性预测器所产生的误差。当研究的误差为均方误差（MSE），即$\mathbb{E} (\widehat Y - Y)^2$时，Kearns等人（2026年）表明沿长度为$D$的路径，误差最多为$O(1/\sqrt{D})$。他们还构造了一个硬实例，其中MSE的下界为$\Omega(1/D)$，留下了正确阶数的开放问题。在这项工作中，我们解决了这一核心开放问题。

    arXiv:2608.15472v1 Announce Type: cross  Abstract: The problem of networked information aggregation, studied in Kearns et al. (2026), involves a group of learners situated on the vertices of a directed acyclic graph $G$, each learning a linear predictor $\widehat Y$ for a fixed random variable $Y$ given access to a local feature, as well as the predictors learnt by its parents. Learning proceeds iteratively, with learners ordered according to a topological sort of $G$. The main quantity of interest is the error incurred by the current learner, constrained to this flow of information, with respect to the best linear predictor using all the features seen so far. When the studied error is the MSE, i.e., $\mathbb{E} (\widehat Y - Y)^2$, Kearns et al. (2026) show that the error is at most $O(1/\sqrt{D})$ along a path of length $D$. They also obtain a hard instance where the MSE is lower bounded by $\Omega(1/D)$, leaving the correct order open. In this work, we resolve this central open prob
    
[^24]: 基于标准ReLU深度神经网络的时间序列预测推断

    Prediction Inference of Time Series with Standard ReLU Deep Neural Networks

    [https://arxiv.org/abs/2608.15362](https://arxiv.org/abs/2608.15362)

    本文提出一种基于标准ReLU深度神经网络的时间序列预测方法，通过构建相关预测区间（PPI）来同时量化未来变异性和估计变异性，并证明了估计器在β混合数据下的一致性。

    

    我们提出了一种基于标准ReLU深度神经网络（DNN）的方法来进行预测并量化其不确定性。传统上，人们依赖线性、非线性或非参数核方法来拟合和预测时间序列。随着DNN通用逼近能力的揭示，其应用在多个科学领域的预测任务中越来越普遍。然而，相应的不确定性量化尚未得到充分研究。特别是，预测中的不确定性由两部分组成：（1）未来变异性；（2）训练数据内的估计变异性。为捕捉这两种变异性，我们利用DNN模型估计器构建了所谓的相关预测区间（PPI）。我们首先探讨了DNN估计器在β混合依赖数据下的一致性性质。随后，我们证明隐含的前向自举序列仍具有β混合性。

    arXiv:2608.15362v1 Announce Type: cross  Abstract: We propose a methodology based on the standard ReLU Deep Neural Networks (DNN) to make predictions and quantify their uncertainty. Classically, people rely on linear, non-linear, or non-parametric kernel methods to fit and then predict the time series. As the universal approximation ability was revealed for DNN, its application has become more and more popular for prediction tasks in various scientific areas. However, the corresponding uncertainty quantification has not been studied thoroughly. Particularly, the uncertainty in prediction will consist of two parts: (1) the future variability; (2) the estimation variability within training data. To capture both variabilities, we build the so-called pertinent prediction interval (PPI) with the DNN model estimator. We first explore the consistency property of the DNN estimator with beta-mixing dependent data. Subsequently, we show that the implied forward bootstrap series is still beta-mix
    
[^25]: GFCM：一种面向尾部敏感性的混合类型条件独立性检验，用于因果发现

    GFCM: A Tail-Sensitive Mixed-Type Conditional Independence Test for Causal Discovery

    [https://arxiv.org/abs/2608.15332](https://arxiv.org/abs/2608.15332)

    本文提出了一种新的混合类型条件独立性检验方法GFCM，通过中心矩和条件分位数指示器结合柯西规则，有效检测尾部依赖性，并解决了尺度特征的偏差问题，适用于因果发现。

    

    arXiv:2608.15332v1 公告类型：交叉 摘要：基于约束的因果发现方法（如PC和FCI）依赖于其条件独立性检验。偏相关和广义协方差测量（GCM）仅检测残差的条件协方差，因此它们会遗漏均值非线性部分、尺度以及尾部的依赖性。能检测更多信息的检验在PC内部存在偏差、不可扩展、仅适用于连续数据，或未针对尾部进行优化。我们提出的广义特征协方差测量（GFCM）是有效的，能检测超出协方差范围的敏感性，在PC内部具有稳健性，并适用于混合类型数据。它在可配置的残差特征集上运行GCM模板，这些特征具有条件均值为零（中心矩和条件分位数指示器），按块汇集并通过柯西规则组合，使用增长结点样条作为回归成本的干扰项。我们贡献了（i）一个中心化结果，使尺度特征满足内曼正交性，而未中心化版本存在偏差；（ii）方向不对称性t...

    arXiv:2608.15332v1 Announce Type: cross  Abstract: Constraint-based causal discovery like PC and FCI depends on its conditional independence test. Partial correlation and the Generalised Covariance Measure (GCM) detect only the conditional covariance of residuals, so they miss dependence in the mean's nonlinear part, the scale, and the tails. Tests that detect more are biased inside PC, not scalable, only continuous, or not aimed at the tails. Our Generalised Feature Covariance Measure (GFCM) is valid, sensitive beyond covariance, robust inside PC, and applicable to mixed-type data. It runs the GCM template on a configurable set of residual features with conditional mean zero (centered moments and conditional quantile indicators), pooled in blocks and combined by the Cauchy rule, with a growing-knot spline nuisance at regression cost. We contribute (i) a centering result making the scale feature Neyman orthogonal, where the uncentered version is biased; (ii) the orientation asymmetry t
    
[^26]: 形状算子PCA：面向几何机器学习曲率感知投影方法

    Shape Operator PCA: Curvature-Aware Projections for Geometric Machine Learning

    [https://arxiv.org/abs/2608.15313](https://arxiv.org/abs/2608.15313)

    本文提出SHOPCA方法，通过将形状算子正则化融入PCA，在无监督降维中同时捕捉方差和曲率信息，并利用谱特征间隙自动选择正则化参数。

    

    本文提出SHOPCA（基于形状算子的主成分分析），一种新颖的无监督度量学习和降维方法，它将微分几何信息融入经典PCA的协方差结构中。SHOPCA通过数据流形估计的绝对局部形状算子平均值，即平均形状算子，对全局协方差矩阵进行正则化，使主成分朝向最大方差和信息丰富曲率的方向。一个迹归一化的混合系数α控制正则化强度，当α=0时恢复标准PCA，当α→∞时产生曲率驱动的嵌入。我们还引入了一种完全无监督的α选择准则，基于正则化协方差矩阵的谱特征间隙，在不需标签的情况下最大化前d个特征值与剩余特征值之间的相对分离。

    arXiv:2608.15313v1 Announce Type: cross  Abstract: In this paper, we propose SHOPCA (Shape Operator-based Principal Component Analysis), a novel method for unsupervised metric learning and dimensionality reduction that incorporates differential geometric information into the covariance structure of classical PCA. SHOPCA regularizes the global covariance matrix using the mean shape operator, defined as the average of the absolute local shape operators estimated from the data manifold, steering principal components toward directions of both maximum variance and informative curvature. A single trace-normalized mixing coefficient $\alpha$ controls the regularization, recovering standard PCA at $\alpha = 0$ and a curvature-driven embedding as $\alpha \to \infty$. We further introduce a fully unsupervised criterion for selecting $\alpha$ based on the spectral eigengap of the regularized covariance matrix, maximizing the relative separation between the top-$d$ and remaining eigenvalues withou
    
[^27]: 空间转录组数据发育分析的统一几何框架

    A Unified Geometric Framework for Developmental Analysis of Spatial Transcriptomic Data

    [https://arxiv.org/abs/2608.15306](https://arxiv.org/abs/2608.15306)

    本文提出了一个统一几何框架，利用Gromov-Wasserstein空间嵌入分析空间转录组数据中基因表达网络的时空演变，克服了现有方法忽略关系结构的局限。

    

    高通量单细胞和空间转录组技术提供了异质细胞状态的高分辨率快照，但其破坏性特性阻止了对同一细胞随时间的重复测量。因此，必须从独立采样、未对齐的细胞群体中推断时间和空间动态，这使得重建发育轨迹变得具有挑战性。最优传输（OT）为对齐细胞群体和推断发育轨迹提供了几何框架，但许多现有方法侧重于建模基因表达空间中细胞分布的演变，而非基因表达网络编码的关系结构。为解决这一局限性，我们引入了一个几何框架，通过将每个发育阶段表示为Gromov-Wasserstein（GW）空间中的嵌入，来分析基因表达网络的时空演变。

    arXiv:2608.15306v1 Announce Type: cross  Abstract: High-throughput single-cell and spatial transcriptomic technologies provide high-resolution snapshots of heterogeneous cellular states, but their destructive nature prevents repeated measurements of the same cells over time. Consequently, temporal and spatial dynamics must be inferred from independently sampled, unaligned cell populations, making it challenging to reconstruct developmental trajectories. Optimal transport (OT) offers a geometric framework for aligning cell populations and inferring developmental trajectories, but many existing approaches focus on modeling the evolution of distributions of cells in gene expression space rather than the relational structure encoded by gene expression networks. To address this limitation, we introduce a geometric framework for analyzing the spatiotemporal evolution of gene expression networks through embeddings in Gromov--Wasserstein (GW) space. By representing each developmental stage as 
    
[^28]: 卷积平滑分位数回归在XGBoost中的应用

    Convolution Smoothed Quantile Regression for XGBoost

    [https://arxiv.org/abs/2608.15290](https://arxiv.org/abs/2608.15290)

    该论文提出了QXGB框架，通过引入卷积平滑损失到分位数梯度提升中，在保持XGBoost计算效率的同时恢复Hessian信息，实现对极端结果和超越概率的可解释预测。

    

    摘要：arXiv:2608.15290v1 公告类型：交叉 摘要：随着许多科学领域中大型复杂数据集的日益普及，机器学习（ML）已被广泛用于预测。然而，大多数机器学习算法侧重于点估计，对预测不确定性或响应的条件分布提供的信息有限，这限制了它们刻画稀有或极端结果的能力。我们开发了QXGB，一种基于分位数的梯度提升框架，并在其中引入卷积平滑损失，该损失估计条件分位数，用于构建密集累积分布函数（CDFs）、超越概率以及与极端结果相关的尾部行为。这种方法保留了极端梯度提升的计算效率，同时恢复了XGBoost在树分裂中所依赖的Hessian信息，进而为极端值和超越概率预测提供可解释的度量。

    arXiv:2608.15290v1 Announce Type: cross  Abstract: The increasing availability of large and complex datasets across many scientific disciplines has led to widespread adoption of machine learning (ML) for prediction. However, most ML algorithms focus on point estimation and provide limited information about predictive uncertainty or the conditional distribution of the response, restricting their ability to characterize rare or extreme outcomes. We develop QXGB, a quantile-based gradient boosting framework, and introduce a convolution smoothed loss within it that estimates conditional quantiles for constructing dense cumulative distribution functions (CDFs), exceedance probabilities, and tail behaviour relevant to extreme outcomes. This approach preserves the computational efficiency of extreme gradient boosting while restoring the Hessian information XGBoost relies on for tree splitting, in turn providing interpretable measures of extreme value and exceedance probability predictions. We
    
[^29]: 学习重塑内部表示中的幂律各向异性

    Learning reshapes power-law anisotropy in internal representations

    [https://arxiv.org/abs/2608.15239](https://arxiv.org/abs/2608.15239)

    该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。

    

    摘要：在从最先进的语言模型到小鼠大脑皮层等广泛的生物和人工神经系统中，内部表示中的幂律各向异性已被观察到。这种各向异性是高维信息处理的关键几何特性，并支撑着多种理论分析。然而，其从输入结构和任务驱动学习中涌现的机制仍不清楚。在此，我们通过精确求解一个宽两层线性神经网络在教师-学生设置下，具有幂律输入和教师结构的学习动态，来刻画这一形成过程。我们表明，在特征学习机制中，内部表示谱的局部幂律指数在训练过程中非单调演化，并在模式和训练时间上表现出多达四个不同的渐近区域。相比之下，在惰性机制中，该指数重新...

    arXiv:2608.15239v1 Announce Type: new  Abstract: Power-law anisotropy in internal representations has been observed across a wide range of biological and artificial neural systems, from state-of-the-art language models to the mouse cerebral cortex. This anisotropy is a key geometric property of high-dimensional information processing and underlies a variety of theoretical analyses. However, the mechanism by which it emerges from input structure and task-driven learning has remained unclear. Here, we characterize this formation process by exactly solving the learning dynamics of a wide two-layer linear neural network in a teacher--student setting with power-law input and teacher structures. We show that, in the feature-learning regime, the local power-law exponent of the internal-representation spectrum evolves nonmonotonically over the course of training and exhibits up to four distinct asymptotic regimes across modes and training times. By contrast, in the lazy regime, the exponent re
    
[^30]: 知识蒸馏的分布视角

    The Distributional View of Knowledge Distillation

    [https://arxiv.org/abs/2608.15215](https://arxiv.org/abs/2608.15215)

    本文提出了一种分布视角的知识蒸馏方法，通过多温度视图的几何聚合替代逐点比较，并证明了特定池化与单温度的等价性，为蒸馏提供了更丰富的理论框架。

    

    arXiv:2608.15215v1 公告类型：交叉 摘要：令牌级知识蒸馏（KD）匹配每个位置的两种条件分布，然而标准目标函数逐点比较它们：Kullback-Leibler梯度无法感知哪个错误令牌获得了概率质量。我们发展了一种分布视角，其中教师不是由单个软化输出表示，而是由一系列多温度视图——其logits退火路径的边际分布——表示，学生则在这种几何感知的视图聚合下训练，该聚合基于嵌入基础的地面成本。我们形式化了由此产生的设计空间（混合、对数线性池化、熵Wasserstein重心，以及集线器和路径形式的去偏Sinkhorn散度旗舰），证明了一个精确的坍缩结果，表明温度视图的对数线性池化等价于单一温度，并给出了一个多边际Schrödinger桥解读，产生可证伪的预测。在指令调优的Pythia对上进行了验证。

    arXiv:2608.15215v1 Announce Type: cross  Abstract: Token-level knowledge distillation (KD) matches two conditional distributions per position, yet the standard objectives compare them pointwise: a Kullback-Leibler gradient is blind to which wrong token receives probability mass. We develop a distributional view in which the teacher is represented not by a single softened output but by a family of multi-temperature views - marginals of the annealing path of its logits - and the student is trained against a geometry-aware aggregate of these views under an embedding-based ground cost. We formalize the resulting design space (mixtures, log-linear pooling, entropic Wasserstein barycenters, and a debiased Sinkhorn-divergence flagship in hub and path forms), prove an exact collapse result showing log-linear pooling of tempered views is equivalent to a single temperature, and give a multi-marginal Schrodinger-bridge reading that yields falsifiable predictions. On instruction-tuned Pythia pairs
    
[^31]: 通过全协方差高斯混合网络识别混合噪声随机系统的参数耦合与不确定性

    Identifying parameter couplings and uncertainties of mixed-noise stochastic systems via full-covariance Gaussian mixture network

    [https://arxiv.org/abs/2608.15198](https://arxiv.org/abs/2608.15198)

    本文提出了一种基于全协方差高斯混合网络的参数估计方法，能够有效处理混合噪声随机系统，并显式揭示参数间的耦合关系和多模态不确定性。

    

    由混合噪声驱动的随机动力系统的参数识别因似然函数难以处理而具有挑战性。我们提出了PENN-GMD，一种参数估计神经网络，它将部分观测轨迹映射到系统参数上的高斯混合分布（GMD）。与传统的不确定性估计不同，GMD采用全协方差矩阵，以显式揭示参数耦合和多模态似然结构。该网络通过最小化负对数似然进行训练，并使用一种满射参数化方法硬编码所有GMD约束，从而近似真实似然。我们在五个复杂度递增的数值示例上验证了该方法，包括由分数高斯和Lévy噪声驱动的系统、有色噪声下的振荡器、不同可观测性下的耦合神经元，以及具有不可辨识随机扰动的气动弹性翼型。

    arXiv:2608.15198v1 Announce Type: cross  Abstract: Parameter identification of stochastic dynamical systems driven by mixed noises is challenging due to intractable likelihood functions. We propose PENN-GMD, a parameter estimation neural network that maps partially observed trajectories to a Gaussian mixture distribution (GMD) over the system parameters. Unlike conventional uncertainty estimates, the GMD employs full covariance matrices to explicitly reveal parameter couplings and multi-modal likelihood structures. The network is trained by minimizing the negative log-likelihood via a surjective parameterization that hard-encodes all GMD constraints, thereby approximating the true likelihood. We validate the method on five numerical examples with increasing complexity, including systems driven by fractional Gaussian and L\'evy noises, oscillators with colored noise, coupled neurons under different observability, and an aeroelastic airfoil with unidentifiable stochastic disturbances. Re
    
[^32]: 超越有效样本量：自适应重要性采样中提案的有效数量

    Beyond Effective Sample Size: Effective Number of Proposals for Adaptive Importance Sampling

    [https://arxiv.org/abs/2608.15154](https://arxiv.org/abs/2608.15154)

    本文提出了一种新的提案级诊断指标——有效提案数量（ENP），用于更准确地评估基于种群的适应性重要性采样中提案组件的多样性，弥补了传统有效样本量（ESS）仅关注权重集中度而忽略提案空间分布的不足。

    

    arXiv:2608.15154v1 公告类型：交叉 摘要：基于种群的 adaptive importance sampling（AIS）方法使用一组提案密度来近似复杂的目标分布。它们的性能通常通过有效样本量（ESS）和相关的基于权重的诊断指标来评估，这些指标衡量归一化重要性权重的集中程度。然而，较大的 ESS 仅表明归一化样本权重并非高度集中；它并未描述提案组件在采样空间中的排列方式。在基于种群的 AIS 中，多个提案组件可能在同一目标区域生成样本，因此即使不同提案组件的有效数量很小，样本权重也可能看起来平衡良好。本文引入了提案的有效数量（ENP），这是一种用于基于种群的 AIS 的相似性感知的提案级诊断指标。ENP 结合了分配给每个提案组件的总归一化权重。

    arXiv:2608.15154v1 Announce Type: cross  Abstract: Population-based adaptive importance sampling (AIS) methods use a set of   proposal densities to approximate complex target distributions. Their   performance is commonly assessed through effective sample size (ESS) and related   weight-based diagnostics, which measure the concentration of normalized   importance weights. However, a large ESS only indicates that the normalized   sample weights are not strongly concentrated; it does not describe how the   proposal components are arranged in the sampling space. In population-based AIS,   several proposal components may generate samples in the same region of the   target, so the sample weights can appear well balanced even though the effective   number of distinct proposal components is small. This letter introduces the   effective number of proposals (ENP), a similarity-aware proposal-level diagnostic   for population-based AIS. ENP combines the total normalized weight assigned to   each
    
[^33]: 扩散逆问题的尺度一致后验动力学

    Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems

    [https://arxiv.org/abs/2608.15144](https://arxiv.org/abs/2608.15144)

    本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。

    

    arXiv:2608.15144v1 公告类型：交叉 摘要：使用预训练扩散先验进行后验采样，受条件分数控制，其中间似然分量通常难以处理。我们从理想的一参数后验SDE族出发，其中随机性参数控制概率流传输和随机探索，而不改变后验边缘分布。为了获得可处理的模型，我们在重标定的干净图像坐标中表达似然，并使用对数信噪比来组织所得的后验代理。通过前向算子投影扩散不确定性，得到噪声条件协方差路径，其目标接近干净后验。由于这些目标的端点一致性不能确保代理传输遵循它们，我们将传输与冻结目标的Langevin校正器交错，生成连续代理SDE。我们使用外部Lie--Trotter分裂和方差减少对此模型进行离散化。

    arXiv:2608.15144v1 Announce Type: cross  Abstract: Posterior sampling with a pretrained diffusion prior is governed by a conditional score whose intermediate likelihood component is generally intractable. We begin from an ideal one-parameter posterior SDE family in which a stochasticity parameter controls probability-flow transport and stochastic exploration without changing the posterior marginals. To obtain a tractable model, we express the likelihood in a rescaled clean-image coordinate and use log-SNR to organize the resulting posterior proxies. Projecting the diffusion uncertainty through the forward operator then yields a noise-conditioned covariance path whose targets approach the clean posterior. Because endpoint consistency of these targets does not ensure that a surrogate transport follows them, we interleave the transport with a frozen-target Langevin corrector, producing a continuous surrogate SDE. We discretize this model with an outer Lie--Trotter splitting and a variance
    
[^34]: 基于广义斯坦引理的充分降维

    Sufficient Dimesion Reduction via Generalized Stein's Lemma

    [https://arxiv.org/abs/2608.15121](https://arxiv.org/abs/2608.15121)

    本文提出一种基于广义斯坦引理的新充分降维框架，通过构建交叉矩矩阵，有效克服了现有方法在多元响应和有限样本下的分布假设、稀疏性和计算成本等限制。

    

    充分降维（SDR）旨在寻找预测变量的最小子空间，以捕捉响应的完整条件分布，该子空间被称为中心子空间（CS）。当响应为多元时，问题变得更具挑战性，尤其是在样本量有限的情况下。现有方法面临不同的局限性：逆回归方法依赖强分布假设和矩阵求逆，且其多响应扩展存在严重的切片稀疏问题；前向回归方法依赖于计算密集的迭代平滑，其成本随响应维度增长；而基于深度学习的方法需要大量标注数据。为规避这些不足，我们提出了一种基于广义斯坦引理的SDR框架。我们的方法构建了多元响应与边际得分函数之间的交叉矩矩阵。

    arXiv:2608.15121v1 Announce Type: cross  Abstract: Sufficient dimension reduction (SDR) seeks the minimal subspace of the predictors that captures the full conditional distribution of the response, which is known as the central subspace (CS). When the response is multivariate, the problem becomes considerably more challenging, particularly when the sample size is limited. Existing methods face different limitations:inverse regression approaches rely on strong distributional assumptions and matrix inversion, and their multi-response extensions suffer from severe slice sparsity; forward regression methods depend on computationally intensive iterative smoothing whose cost grows with the response dimension; and deep learning-based approaches demand large amounts of labeled data. To circumvent these shortcomings, we propose an SDR framework based on the generalized Stein's lemma. Our method constructs a cross-moment matrix between the multivariate response and the marginal score function of
    
[^35]: 几何感知位置编码是否有助于空间不完美信息游戏中的Transformer？

    Do Geometry-Aware Positional Encodings Help Transformers in Spatial Imperfect-Information Games?

    [https://arxiv.org/abs/2608.14982](https://arxiv.org/abs/2608.14982)

    本文通过多级基准测试证明，几何感知位置编码HexRoPE在空间不完美信息游戏中显著提升了Transformer的隐藏目标追踪与策略学习性能。

    

    摘要：应用于空间不完美信息游戏的Transformer必须在表示地图几何结构的同时，随时间追踪隐藏实体。我们探讨几何感知位置编码是否能提升这些能力，而不声称提出新的位置编码方法。我们在一个六边形海军追逐游戏中构建了四级基准：受控的几何与拓扑探测、精确贝叶斯隐藏目标追踪任务、在1k和10k游戏上的离线策略模仿，以及与三个传统对手进行的7,200个固定种子游戏。在匹配的Transformer主干网络中，HexRoPE相对于无位置编码，在D6变换的测试轨道上将精确信念后验交叉熵降低了0.278，在更大的地图上降低了0.329；两个层次自助置信区间均排除零，且两个经Holm校正的p值均低于0.001。在1k游戏时，HexRoPE相对于无编码将策略动作准确率提高了4.63个百分点，相对于矩形编码提高了2.05个百分点。

    arXiv:2608.14982v1 Announce Type: cross  Abstract: Transformers applied to spatial imperfect-information games must represent map geometry while tracking hidden entities through time. We ask whether geometry-aware positional encodings improve these capabilities, without claiming a new positional encoding. We construct a four-level benchmark on a hexagonal naval pursuit game: controlled geometry and topology probes, an exact-Bayes hidden-target tracking task, offline policy imitation at 1k and 10k games, and 7,200 fixed-seed games against three legacy opponents. Across matched Transformer backbones, HexRoPE reduces exact-belief posterior cross-entropy relative to no positional encoding by 0.278 on D6-transformed test orbits and 0.329 on a larger map; both hierarchical-bootstrap confidence intervals exclude zero, and both Holm-adjusted p-values are below 0.001. At 1k games, HexRoPE improves policy action accuracy by 4.63 percentage points over no encoding and 2.05 points over rectangular
    
[^36]: 一种通过可微聚类分配处理空间聚类数据的深度学习模型

    A Deep Learning Model for Spatially Clustered Data via Differentiable Cluster Assignment

    [https://arxiv.org/abs/2608.14968](https://arxiv.org/abs/2608.14968)

    该论文提出了一种联合学习空间划分和聚类特定回归函数的深度学习模型，通过可微聚类分配和惩罚机制实现高效的非参数回归估计。

    

    我们考虑了当响应变量与其协变量之间的关系在空间域的一个未知划分上发生变化时的非参数回归问题。所提出的估计器联合学习划分和聚类特定的回归函数。一个仅依赖于位置的神经网络确定聚类成员关系，而单独的神经网络描述聚类内部的协变量-响应关系。退火的softmax松弛允许对原本离散的分配进行基于梯度的估计。图拉普拉斯和占用惩罚被用来防止碎片化区域和退化解。我们建立了在标签置换下的可识别性，在边际条件下界定了划分误差，并将预测风险分解为回归和分配组件。当划分被足够准确地估计时，所得速率与一个神谕估计器的速率一致。模拟表明...

    arXiv:2608.14968v1 Announce Type: cross  Abstract: We consider nonparametric regression when the association between a response and its covariates changes across an unknown partition of a spatial domain. The proposed estimator learns the partition and the cluster-specific regression functions jointly. A neural network depending only on location determines cluster membership, while separate neural networks describe the covariate--response relationship within the clusters. An annealed softmax relaxation permits gradient-based estimation of the otherwise discrete assignments. Graph-Laplacian and occupancy penalties are used to discourage fragmented regions and degenerate solutions. We establish identifiability up to label permutation, bound partition error under a margin condition, and decompose prediction risk into regression and assignment components. The resulting rate agrees with that of an oracle estimator when the partition is estimated sufficiently accurately. Simulations show that
    
[^37]: 路径发现者：关联多模态数据集的联合分解

    PathFinder: Joint Decompositions of Linked Multimodal Datasets

    [https://arxiv.org/abs/2608.14951](https://arxiv.org/abs/2608.14951)

    提出一种新方法PathFinder，允许不共享维度的多模态数据集通过路径连接实现联合低秩分解，从而发现跨模态的公共模式。

    

    arXiv:2608.14951v1 公告类型：新 摘要：低秩矩阵分解可以揭示数据中的模式和结构，并在许多学科中具有多种应用。为关联不同模态的数据集，已提出了“联合”低秩分解的扩展方法。虽然这些方法能够发现跨模态的常见模式，但它们要求所有多模态数据共享一个或多个维度。我们提出了一种新的分析方法，路径发现者（PathFinder），它能够对不一定共享维度的数据集进行共同分析。关键洞察在于，只要矩阵对或子组确实共享某个维度，并且存在一条或多条路径连接各数据矩阵，就可以寻求全局联合分解。这使得能够在不同模态、物种或尺度之间联合估计常见模式，即使并非所有数据沿某个维度都有一对一的映射。我们展示了该方法的有效性。

    arXiv:2608.14951v1 Announce Type: new  Abstract: Low-rank matrix decompositions can uncover patterns and structure in data and have a number of different applications across many disciplines. Extensions to "joint" low-rank decompositions have been proposed to link datasets from different modalities. While these methods enable the discovery of common patterns across modalities, they require that all the multimodal data share one or more dimensions. We propose a new analysis method, PathFinder, that enables co-analysis of datasets that do not necessarily all share a dimension. The key insight is that as long as pairs or subgroups of matrices do share some dimension, and that there are one or more paths that link across the data matrices, a global joint decomposition can be sought out. This enables the joint estimation of common patterns across different modalities, species, or scales, where a one-to-one mapping across all data along some dimension is not necessarily available. We show th
    
[^38]: 混合来源大型语言模型文本中的最优水印定位

    Optimal Watermark Localization in Mixed-Source Large Language Model Texts

    [https://arxiv.org/abs/2608.14906](https://arxiv.org/abs/2608.14906)

    本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。

    

    水印提供了一种有原则的方式来认证由大型语言模型（LLMs）生成的文本。然而，在实践中，最终文本可能是混合来源的，经过改写、插入、删除或释义后，水印证据仅存留在部分标记位置。尽管先前的研究已经探讨了水印信号的全局检测，但何时能对这些信号进行定位仍不清楚。我们将水印定位问题表述为基于关键统计量的标记级多重检验问题，其中包含一个潜在指示符，记录每个位置的水印依赖是否存活。在由信号稀疏性、下一标记浓度和有效词汇增长指数所索引的渐近框架下，我们推导出全局检测的尖锐边界，以及在坐标级基于关键统计的定位规则类别内的发现和分类相变。我们表明，发现严格比分类更难，并提供了最优规则。

    arXiv:2608.14906v1 Announce Type: cross  Abstract: Watermarking provides a principled way to authenticate text generated by large language models (LLMs). In practice, however, the final text may be mixed-source, with watermark evidence surviving at only a subset of token positions after rewriting, insertion, deletion, or paraphrasing. Although prior work has studied global detection of watermark signals, when such signals can be localized remains unclear. We formulate watermark localization as a token-level multiple-testing problem based on pivotal statistics, with a latent indicator recording whether watermark dependence survives at each position. Under an asymptotic regime indexed by exponents for signal sparsity, next-token concentration, and effective-vocabulary growth, we derive a sharp boundary for global detection and phase transitions for discovery and classification within the class of coordinatewise pivot-based localization rules. We show that discovery is strictly harder tha
    
[^39]: ARISE：一种自适应残差信息稳定性集成方法，用于小样本生物医学组学中的特征选择

    ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics

    [https://arxiv.org/abs/2608.14866](https://arxiv.org/abs/2608.14866)

    ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。

    

    摘要：目标：小样本分子分类需要能够为二元和多类结果识别预测性、稳定性和非冗余子集的特征选择器。我们提出了ARISE（自适应残差信息稳定性集成），它整合了互补的相关性信号、类平衡的稳定性评估、残差信息冗余控制以及多类配对覆盖。方法：ARISE通过15个预定义配置文件组合七个百分位归一化的相关性成分，并通过嵌套内部交叉验证自适应加权。它在五个分子数据集、八个特征集大小、三个固定分类器（k近邻、支持向量机和随机森林）以及六个过滤比较器上进行了评估。泛化性能通过五次外部交叉验证重复50次，使用平衡准确率、宏观F1分数和Cohen's kappa进行估计。结果：在210,000次留出评估中，ARISE排名第一。

    arXiv:2608.14866v1 Announce Type: cross  Abstract: Objective: Small-sample molecular classification requires feature selectors that identify predictive, stable, and nonredundant subsets for binary and multiclass outcomes. We propose ARISE (Adaptive Residual-Informed Stability Ensemble), which integrates complementary relevance signals, class-balanced stability assessment, residual-informed redundancy control, and multiclass pairwise coverage.   Methods: ARISE combines seven percentile-normalized relevance components through 15 predefined profiles, adaptively weighted by nested inner cross-validation. It was evaluated on five molecular datasets, eight feature-set sizes, three fixed classifiers (k-nearest neighbours, support vector machine, and random forest), and six filter comparators. Generalization was estimated by five-fold outer cross-validation repeated 50 times using balanced accuracy, macro-F1, and Cohen's kappa.   Results: Across 210,000 held-out assessments, ARISE ranked first
    
[^40]: 分离面的生成学习

    Generative Learning of Separatrices

    [https://arxiv.org/abs/2608.14743](https://arxiv.org/abs/2608.14743)

    该论文提出了一种结合监督分类与生成建模的新框架，用于高效重建高维多稳态动力系统中的分离面，克服了传统方法的计算限制和采样不足问题。

    

    arXiv:2608.14743v1 公告类型：新 摘要：在多稳态、多维动力系统中，识别和重建吸引盆地的边界（即分离面）是计算动力学中的一个基本挑战。这些结构控制着转变路径和其他重要的大时间尺度行为，但由于直接模拟过程中其邻域通常不会被常规访问，它们往往采样不足。传统计算方法在高维系统中面临计算限制，并且需要预先知道动力系统及其方程。诸如随机或均匀采样相空间之类的简单采样方法，通常无法定量近似分离面及其整体结构。我们引入并实现了一个结合监督分类和生成建模的框架来解决这一挑战。我们的方法首先在均匀分布的数据上训练神经网络分类器。

    arXiv:2608.14743v1 Announce Type: new  Abstract: The identification and reconstruction of the boundaries separating basins of attraction in multistable, multidimensional dynamical systems presents a fundamental challenge in computational dynamics. These structures govern transition pathways and other important large timescale behavior, yet they remain typically under-sampled since their neighborhood does not get routinely visited during direct simulations. Traditional computational approaches face computational limitations in high-dimensional systems and require a priori knowledge of the dynamical system and its equations. Simplistic sampling methods such as random or uniform sampling of the phase space typically fail to quantitatively approximate separatrices and their structure altogether.   We introduce and implement a framework that combines supervised classification with generative modeling to address this challenge. Our approach first trains neural network classifiers on uniforml
    
[^41]: AccretionLink：针对属性推断中暴露控制攻击的设备端审计

    AccretionLink: On-Device Auditing of Exposure-Control Attacks on Attribute Inference

    [https://arxiv.org/abs/2608.14735](https://arxiv.org/abs/2608.14735)

    AccretionLink提出了一种设备端审计框架，通过定义机密性和完整性博弈及依赖感知的e过程，有效检测并量化暴露控制攻击对属性推断的增强效应。

    

    arXiv:2608.14735v1 公告类型：交叉 摘要：暴露控制允许攻击者对真实的公开帖子进行排序，以加强私有属性推断，而无需修改内容。AccretionLink为该攻击定义了机密性和完整性博弈，通过部分识别对受限选择几率进行建模，并构建了依赖感知的时间均匀e过程。在52个保留的合成档案上，几率四的选择在每个时间范围都降低了聚合负对数似然。在八个帖子时，优势为0.01595纳特（95%置信区间[0.00890, 0.02336]），四个目标效应中的三个通过了霍尔姆调整，且标签盲模型引导选择导致了6/109个高置信度错误反转。在142个PAN15测试档案上，探索性选择产生了0.01227纳特的优势，但没有反转。一个单独的TF-IDF选择器对未改变的G5目标保持了0.01470纳特的优势，而匹配的身份洗牌未能复现该优势。Pixel 10一次性编码了所有1,622个保留帖子。

    arXiv:2608.14735v1 Announce Type: cross  Abstract: Exposure control lets an adversary rank authentic public posts to strengthen private-attribute inference without altering content. AccretionLink defines confidentiality and integrity games for this attack, models bounded selection odds through partial identification, and constructs dependence-aware time-uniform e-processes. On 52 held-out synthetic profiles, odds-four selection reduced aggregate negative log likelihood at every horizon. At eight posts the advantage was 0.01595 nats (95% CI [0.00890, 0.02336]), three of four target effects survived Holm adjustment, and label-blind model-guided selection caused 6/109 high-confidence false reversals. On 142 PAN15 test profiles, exploratory selection produced a 0.01227-nat advantage but no reversal. A separate TF-IDF selector retained a 0.01470-nat advantage against the unchanged G5 target, while matched identity shuffling did not reproduce it. Pixel 10 encoded all 1,622 held-out posts onc
    
[^42]: 重新思考反向KL作为自适应熵蒸馏

    Rethinking Reverse KL as Adaptive Entropy Distillation

    [https://arxiv.org/abs/2608.14685](https://arxiv.org/abs/2608.14685)

    本文提出自适应熵蒸馏（AED），通过重新分解反向KL目标为教师拟合和学生熵项，利用教师熵动态调整蒸馏权重，实现更优的模仿与生成平衡。

    

    arXiv:2608.14685v1 公告类型：新论文 摘要：知识蒸馏（KD）广泛用于将大型语言模型（LLMs）的能力转移到较小的学生模型上，但现有目标函数常常难以平衡忠实模仿和稳健生成。特别是，现有方法主要结合前向KL（FKL）和反向KL（RKL），却忽视了RKL本身提供了一种调整学生模仿强度的机制。基于此，我们重新审视了策略上的反向KL（RKL）蒸馏，并将其目标函数分解为教师拟合项和学生熵项，无需引入显式的FKL分支。我们从理论上证明，令牌级的最优学生分布对应于教师分布的温和变体，其中自适应权重控制着模式寻求和不确定性保留之间的权衡。受此洞察启发，我们提出了\textbf{自适应熵蒸馏（AED）}，它利用教师的熵来动态调整蒸馏过程。

    arXiv:2608.14685v1 Announce Type: new  Abstract: Knowledge distillation (KD) is widely used to transfer the capabilities of large language models (LLMs) to smaller students, but existing objectives often struggle to balance faithful imitation and robust generation. In particular, existing methods mainly combine FKL and RKL, overlooking that RKL itself provides a mechanism for adjusting the student's imitation strength. Motivated by this, we revisit on-policy Reverse Kullback-Leibler (RKL) distillation and decompose its objective into a teacher-fitting term and a student-entropy term, without introducing an explicit FKL branch. We show theoretically that the token-level optimal student distribution corresponds to a tempered variant of the teacher distribution, where the adaptive weight controls the trade-off between mode-seeking and uncertainty preservation. Guided by this insight, we propose \textbf{Adaptive Entropy Distillation (AED)}, which uses the teacher's entropy to dynamically c
    
[^43]: RouteTS：时间序列预测的频率-时间路由

    RouteTS: Frequency-Time Routing for Time Series Forecasting

    [https://arxiv.org/abs/2608.14682](https://arxiv.org/abs/2608.14682)

    RouteTS通过振幅路由动态分配频率分量到最优计算域（频域或时域），解决了时间序列预测中周期性与局部变化的处理冲突。

    

    摘要：arXiv:2608.14682v1 公告类型：新  摘要：现实世界的时间序列本质上将全局周期结构与局部非平稳变化交织在一起。现有方法在单一计算域中处理这些异质动态，导致根本性限制：时域模型在长视野内遭受周期性错位，而频域模型则过度平滑瞬态尖峰。我们认为，最优计算域不是模型的属性，而是数据本身的属性。基于这一原则，我们提出RouteTS，一个统一的预测框架，通过振幅路由划分频谱，并将组件分配给其数学上最优的域。主导频率由频域中的复值线性预测器处理，以保持周期性结构，而残差谱能量则恢复到时域，由轻量级MLP建模局部变化。大量实验表明...

    arXiv:2608.14682v1 Announce Type: new  Abstract: Real-world time series inherently intertwine global periodic structures with localized non-stationary variations. Existing approaches process these heterogeneous dynamics within a single computational domain, incurring fundamental limitations: time-domain models suffer from periodic misalignment over long horizons, while frequency-domain models over-smooth transient spikes. We argue that the optimal computational domain is not a property of the model, but of the data itself. Based on this principle, we propose RouteTS, a unified forecasting framework that partitions the frequency spectrum via amplitude routing and delegates components to their mathematically optimal domains. Dominant frequencies are processed by a complex-valued linear predictor in the frequency domain to preserve periodic structure, while residual spectral energy is reverted to the time domain and modeled by a lightweight MLP for local variations. Extensive experiments 
    
[^44]: 上下文并非权威：金融市场代理的结构化运行时治理

    Context Is Not Authority: Structured Runtime Governance for Financial Market Agents

    [https://arxiv.org/abs/2608.09025](https://arxiv.org/abs/2608.09025)

    SAGE-Fin通过结构化运行时治理框架，确保金融代理的每个拟议效果都受到类型化适配器绑定、覆盖债务记录和状态变化后重新授权检查的严格管控，从而防止上下文被滥用为未经授权的行为。

    

    摘要：arXiv:2608.09025v2 公告类型：替换版 摘要：金融代理可能将正确的上下文转化为未经授权的行为：面向客户的承诺、交易或部署的政策。我们提出了SAGE-Fin，一种金融专用的权威交接合约，它将拟议的效果（而非仅仅是文本）作为运行时控制的对象。SAGE-Fin将提案编译为类型化、适配器绑定的候选对象；将缺失或过时的机构义务记录为覆盖债务；在当前市场、账户、政策和对话状态下收缩权威；并要求提供精确工件收据，其名义类型与消费响应、执行或政策适配器匹配。证据和工作流进展不能替代效果权威，且先前授权在状态变化后会被重新检查。在自建的616个案例目录中，五个确定性规范生成3,080个输出；一个标签隔离的测试平台获得616/616的二元参考原型一致性，包括3/3个命名响应门修复。

    arXiv:2608.09025v2 Announce Type: replace  Abstract: Financial agents can turn correct context into an unauthorized effect: a customer-facing commitment, trade, or deployed policy. We present SAGE-Fin, a finance-specific authority-handoff contract that makes the proposed effect, not merely its text, the object of runtime control. SAGE-Fin compiles proposals into typed, adapter-bound candidates; records missing or stale institutional obligations as coverage debt; contracts authority under current market, account, policy, and dialogue state; and requires an exact-artifact receipt whose nominal type matches the consuming response, execution, or policy adapter. Evidence and workflow progress cannot substitute for effect authority, and prior authorization is rechecked after state changes. Across an authored 616-case catalog, five deterministic specifications yield 3,080 outputs; a label-isolated harness obtains 616/616 binary reference-prototype parity, including 3/3 named response-gate fix
    
[^45]: 谱神经元

    The Spectral Neuron

    [https://arxiv.org/abs/2608.08003](https://arxiv.org/abs/2608.08003)

    本文提出了“谱神经元”模型，通过读取仿射矩阵函数的特征值实现非线性预测，在保持系数透明性的同时增强表达力，填补了线性模型与神经网络之间的空白。

    

    摘要：arXiv:2608.08003v2 公告类型：替换-交叉 摘要：随着机器学习模型在复杂性和表达能力上的增加，更简单模型的特征，如内在系数透明性和对模型函数形状的控制，逐渐丧失。在频谱的一端，我们有简单的线性模型，它们拥有系数透明性，但表达能力有限。在频谱的另一端，我们有神经网络，其表达能力随规模扩大而增强，但大多不透明。在这项工作中，我们开发了“谱神经元”概念：一个标量模型，由$f(x)=\lambda_k (A_0 + A_1 x + ... + A_n x_n)$给出，其中学习到实对称矩阵$A_0, ..., A_n$。输入通过仿射矩阵函数进入模型，但预测通过读取其一个特征值获得。因此，该模型是非线性的，但非线性的来源在数学上仍然明确。这为我们提供了一个有用的中间地带：模型可以变得更具表达力。

    arXiv:2608.08003v2 Announce Type: replace-cross  Abstract: As machine learned models increase in complexity and expressive power, features of simpler models, such as intrinsic coefficient transparency and control over the shape of the modeled function are lost. On the one edge of the spectrum we have simple linear models that possess coefficient transparency, but have a limited expressive power. On the other edge we have neural networks, that have expressive power that improves with scaling, but are mostly opaque. In this work we develop the \emph{spectral neuron} concept: a scalar model given by $f(x)=\lambda_k (A_0 + A_1 x + ... + A_n x_n)$, with learned real symmetric matrices $A_0, ..., A_n$. The input enters the model through an affine matrix function, but the prediction is obtained by reading one of its eigenvalues. Thus, the model is nonlinear, but the source of nonlinearity is still mathematically explicit. This gives us a useful middle ground: the model can become more express
    
[^46]: 面板数据模型中的断点估计与检验

    Estimating and Testing Kinks in Panel Data Models

    [https://arxiv.org/abs/2608.07162](https://arxiv.org/abs/2608.07162)

    该论文首次在固定效应面板数据模型中提出一种惩罚最小二乘方法，能够自适应估计未知数量的共同断点日期，并给出断点位置和斜率收敛速率的理论保证。

    

    摘要：arXiv:2608.07162v2 公告类型：替换 摘要：许多经济和金融关系可能逐渐变化而非突然变化。我们研究面板数据模型，其中系数向量在日历时间上是连续且分段线性的，具有有限个未知的断点日期，在这些日期其斜率发生变化。我们提出了一种惩罚最小二乘估计器，该估计器对系数路径的二阶差分应用自适应加权组惩罚，并发展了渐近理论，表明该估计器能以接近一的概率恢复断点的数量和位置。据我们所知，这是首个在固定效应下估计时变系数路径中未知数量共同断点日期的面板框架。我们建立了端点斜率以通常的三次区间长度速率收敛，而内部斜率以由其自身和相邻区间长度决定的速率收敛。我们还开发了一种逐系数扩展，允许个体r

    arXiv:2608.07162v2 Announce Type: replace  Abstract: Many economic and financial relationships may change gradually rather than abruptly. We study panel data models in which the coefficient vector is continuous and piecewise linear in calendar time, with a finite number of unknown kink dates at which its slope changes. We propose a penalised least squares estimator that applies adaptive weighted group penalties to the second differences of the coefficient path, and develop asymptotic theory showing that it recovers both the number and the locations of the kinks with probability approaching one. To our knowledge, this is the first panel framework to estimate an unknown number of common kink dates in a time-varying coefficient path under fixed effects. We establish that endpoint slopes converge at the usual cubic regime-length rate and interior slopes at rates determined by their own and adjacent regime lengths. We also develop a coefficient-by-coefficient extension allowing individual r
    
[^47]: 从旧重建中学习的先验继承了不可检测的过度自信

    Priors learned from legacy reconstructions inherit undetectable overconfidence

    [https://arxiv.org/abs/2607.21721](https://arxiv.org/abs/2607.21721)

    论文指出，从旧重建档案学习的先验在盲子空间上继承不可检测的过度自信，导致误差被低估且无法通过部署测试发现。

    

    arXiv:2607.21721v3 公告类型：替换交叉 摘要：在真相稀缺的领域（例如地震和医学成像），针对不适定逆问题的先验是在旧重建档案——即旧方法的输出——上训练的，其不确定性被视为数据驱动。在总体极限下，后验样本档案是产生它的正则化器，向真相推进了一步期望最大化。在算子解决的方向上，它改进了假设；在其盲子空间上，该步骤是恒等变换，因此无论重建多少次，假设都保持不变。每个调查一个的最佳单一重建档案在那里没有扩散：盲区间无论惩罚是什么都会坍缩，因此误差变成过度自信。假设作为档案进入，以报告的不确定性离开，部署中没有任何测试能检验它。仅在这些方面不同的两个真相共享数据规律，且仅使用调查和档案的程序无法区分它们。

    arXiv:2607.21721v3 Announce Type: replace-cross  Abstract: Where truths are scarce (e.g., seismic and medical imaging), a prior for an ill-posed inverse problem is trained on an archive of legacy reconstructions---an older method's outputs---and its uncertainty is treated as data-driven. In the population limit, an archive of posterior samples is the regularizer that produced it, advanced one expectation-maximization step toward the truth. On directions the operator resolves, it improves the assumption; on its blind subspace, the step is the identity, so the assumption survives unchanged however often it is rebuilt. An archive of single-best reconstructions, one per survey, keeps no spread there: the blind interval collapses whatever the penalty was, so error becomes overconfidence. The assumption enters as the archive and leaves as a reported spread, and nothing in deployment tests it. Two truths differing only there share the data law, and no procedure using survey and archive alone 
    
[^48]: 鲁棒半空间学习中重加权铰链方法的平方和度数障碍：克里斯托弗函数刻画

    Sum-of-Squares Degree Barriers for the Reweighted-Hinge Method in Robust Halfspace Learning: A Christoffel-Function Characterization

    [https://arxiv.org/abs/2606.17215](https://arxiv.org/abs/2606.17215)

    本文通过克里斯托弗函数精确刻画了平方和证书在鲁棒半空间学习中的度数限制，揭示了隐藏污染的最大质量与证书度数之间的关系。

    

    arXiv:2606.17215v2 公告类型：替换  摘要：一个去除异常值的证书仅通过其低阶矩观察数据，而对手恰好利用这一点，将污染隐藏在干净数据看起来已经典型的地方，即任何有界度数测试都无法解决的盲区。该盲区具有精确大小：干净边际分布的克里斯托弗函数，这是数据分析用于检测异常值的阈值量，这里从对手的角度解读为证书无法去除的污染量。我们将这种反转转化为重加权铰链方法在恶意噪声下鲁棒学习γ-间隔半空间（Shen 2025; Zeng-Shen 2025）的组织原则：主导资源是证书的平方和度数，分辨率原则指出，在中心c处，可从度数-2t证书中隐藏的最大污染质量恰好是克里斯托弗函数λ_{t+1}(c)。三个推论随之而来，均针对中心...

    arXiv:2606.17215v2 Announce Type: replace  Abstract: A certificate that removes outliers sees the data only through its low-degree moments, and an adversary exploits exactly this, hiding corruption where the clean data already looks typical, in the blind spot no bounded-degree test resolves. That blind spot has an exact size: the Christoffel function of the clean marginal, the quantity data analysis thresholds to detect outliers, here read from the adversary's side as the corruption a certificate cannot remove. We turn this inversion into the organizing principle of the reweighted-hinge approach to robustly learning $\gamma$-margin halfspaces under malicious noise (Shen 2025; Zeng-Shen 2025): the governing resource is the Sum-of-Squares degree of the certificate, and the resolution principle states that the maximal corruption mass hideable at a center $c$ from a degree-$2t$ certificate is exactly the Christoffel function $\lambda_{t+1}(c)$. Three consequences follow, all against the ce
    
[^49]: 一种深度零膨胀模型用于美国东海岸北大西洋露脊鲸存在性建模以支持蓝色经济管理

    A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast

    [https://arxiv.org/abs/2606.14403](https://arxiv.org/abs/2606.14403)

    本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。

    

    arXiv:2606.14403v2 公告类型：替换-交叉  摘要：对濒危海洋哺乳动物物种（如北大西洋露脊鲸）的有效建模，对于在日益增长的蓝色经济中平衡海洋保护至关重要。由自主水下航行器收集的被动声学监测数据为局部海洋物种检测和海洋学感知提供了新机会，但也引入了复杂的统计挑战，如零膨胀、不完全检测和复杂依赖结构。为此，我们提出了深度零膨胀伯努利（DeepZIB）模型——一种深度统计方法，该方法联合建模潜在物种存在性和条件检测概率，同时从异构协变量信息中学习复杂的栖息地关系。我们建立了模型结构性质的理论结果，并进行了模拟实验以证明其恢复底层参数和潜在存在场的能力。应用...

    arXiv:2606.14403v2 Announce Type: replace-cross  Abstract: Effective modeling of endangered marine mammal species, such as the North Atlantic Right Whale, is critical for balancing marine conservation with the growing blue economy. Passive acoustic monitoring data collected by autonomous underwater vehicles provide new opportunities for localized marine species detection and oceanographic sensing, but introduce complex statistical challenges such as zero inflation, imperfect detection, and intricate dependence structures. In response, we propose the Deep Zero-Inflated Bernoulli (DeepZIB) model--a deep statistical method which jointly models latent species presence and conditional detection probabilities while learning complex habitat relationships from heterogeneous covariate information. We establish theoretical results on the model's structural properties and conduct simulation experiments to demonstrate its ability to recover underlying parameters and latent presence fields. Applica
    
[^50]: 计算上易处理的鲁棒差分隐私均值估计

    Computationally tractable robust differentially private mean estimation

    [https://arxiv.org/abs/2606.12654](https://arxiv.org/abs/2606.12654)

    提出了一种计算上易处理且鲁棒的差分隐私均值估计方法“气球均值”，通过扩展Mahalanobis球上的迭代裁剪实现，在重尾和污染数据下优于现有方法。

    

    我们开发了一种新的差分隐私均值估计器，称为气球均值。气球均值的主要特点是计算上易处理，并且对异常观测值具有鲁棒性。它基于在扩展的Mahalanobis球（或“气球”）上的迭代裁剪过程。该方法满足零集中差分隐私，并依赖于少量可解释的调优参数。我们在重尾和污染的椭圆模型下提供了理论保证，表征了其统计性能和异常值鲁棒性。大量模拟表明，气球均值对重尾和污染数据具有鲁棒性，并且在污染场景下优于现有的差分隐私均值估计器。

    arXiv:2606.12654v2 Announce Type: replace-cross  Abstract: We develop a new, differentially private mean estimator called the balloon mean. The main features of the balloon mean are that it is computationally tractable and enjoys robustness to outlying observations. It is based on an iterative clipping procedure over expanding Mahalanobis balls, or ``balloons.'' The method satisfies zero-concentrated differential privacy and depends on a small number of interpretable tuning parameters. We provide theoretical guarantees under heavy-tailed and contaminated elliptical models, characterizing its statistical performance and robustness to outliers. Extensive simulations demonstrate that the balloon mean is robust to heavy-tailed and contaminated data, and outperforms existing differentially private mean estimators in contaminated settings.
    
[^51]: 扩散模型中流蒸馏的定量逼近框架

    A Quantitative Approximation Framework for Flow Distillation in Diffusion Models

    [https://arxiv.org/abs/2606.03820](https://arxiv.org/abs/2606.03820)

    本文提出了一个定量框架，揭示低噪声多模态下分数逼近与动态稳定性的分离，并给出了指数增长稳定性因子的可计算界，从而识别直接蒸馏的适用条件。

    

    我们通过将少步采样视为学习流映射组合的逼近，为扩散蒸馏开发了一个定量框架。对于概率流ODE的轨迹蒸馏，我们表明低噪声多模态区域将分数可逼近性与动态稳定性分离：分数仍然高效可逼近，而小的局部误差可能被刚性流动力学强烈放大。在高斯混合Ornstein--Uhlenbeck模型中，我们证明了ReLU和ReQU网络对\(L^p(p_t)\)分数的均匀时间逼近，具有显式的多对数复杂度，并推导了流速度的可计算Lipschitz界\(L(t)\)。稳定性因子\(\exp\bigl(\int_s^t L(u)\mathrm du\bigr)\)在噪声降低和混合分离增加时可能指数增长。将此证书与单步学生的认证局部Lipschitz预算进行比较，识别出直接蒸馏的适用区域。

    arXiv:2606.03820v2 Announce Type: replace-cross  Abstract: We develop a quantitative framework for diffusion distillation by viewing few step sampling as approximation through compositions of learned flow maps. For trajectory distillation of the probability flow ODE, we show that low noise multimodal regimes separate score approximability from dynamical stability: the score remains efficiently approximable, while small local errors may be strongly amplified by stiff flow dynamics. In a Gaussian mixture Ornstein--Uhlenbeck model, we prove time uniform \(L^p(p_t)\) score approximation by ReLU and ReQU networks with explicit polylogarithmic complexity, and derive a computable Lipschitz bound \(L(t)\) for the flow velocity. The stability factor \(\exp\bigl(\int_s^t L(u)\mathrm du\bigr)\) can grow exponentially as noise decreases and mixture separation increases. Comparing this certificate with a certified local Lipschitz budget for one step students identifies regimes of direct distillatio
    
[^52]: 线性上下文赌博机在稀有参数更新下的实用最优算法

    Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates

    [https://arxiv.org/abs/2606.00984](https://arxiv.org/abs/2606.00984)

    本文提出两种仅需$O(\log\log T)$次参数更新的线性上下文赌博机算法，在静态调度下同时在小规模和大规模动作集下实现极小极大最优遗憾，并澄清了批处理与稀有更新的实际区别。

    

    arXiv:2606.00984v2 公告类型：替换-交叉 摘要：我们研究在稀有参数更新下的线性上下文赌博机：学习器只能在少量更新时间点将奖励反馈纳入其参数估计，同时仍需在线观察上下文并按顺序选择动作。这一视角澄清了文献中常被模糊的一个实际区别：许多“严格批处理”方法额外限制了区间内的上下文自适应性，即区间内的动作规则不能依赖于该区间内已实现的上下文/动作序列（除当前轮次的上下文外）。对于线性上下文赌博机，我们提出了两种仅需$O(\log\log T)$次参数更新的实用算法。我们的第一种算法BLCE-G在静态调度下，同时在小$K$和大$K$区域中达到极小极大最优遗憾（在$T$的多对数因子范围内）。我们的第二种算法BLCE移除了近G-最优设计。

    arXiv:2606.00984v2 Announce Type: replace-cross  Abstract: We study linear contextual bandits under rare parameter updates: the learner may incorporate reward feedback into its parameter estimate only at a small number of update times, while still observing contexts online and selecting actions sequentially. This viewpoint clarifies a practical distinction that is often blurred in the literature: many "strictly batched" methods additionally restrict within-interval context adaptivity, meaning that the action rule inside an interval cannot depend on the sequence of realized contexts/actions in that interval (beyond the current round's context). For linear contextual bandits, we propose two practical algorithms with only $O(\log\log T)$ parameter updates. Our first algorithm BLCE-G attains minimax-optimal regret (up to polylogarithmic factors in $T$) simultaneously in both the small-$K$ and large-$K$ regimes under a static schedule. Our second algorithm BLCE removes the near G-optimal de
    
[^53]: 利用深度学习测量K2红巨星的周期间距与全球地震学参数

    Period spacings and global seismic parameters for K2 red giants using deep learning

    [https://arxiv.org/abs/2605.08051](https://arxiv.org/abs/2605.08051)

    本文首次利用深度学习从K2短基线数据中自动测量红巨星的引力模周期间距，无需背景拟合，显著扩展了恒星核心探测的样本范围。

    

    arXiv:2605.08051v2 公告类型：替换-交叉 摘要：红巨星的引力模周期间距（DPi_1）直接探测恒星核心，约束其结构、质量和演化状态。其测量需要解析狭窄且密集分布的混合模，迄今依赖于开普勒四年的基线数据。从K2典型的约80天短基线中恢复DPi_1，在群体尺度上尚未充分探索。我们开发了一种自动化机器学习技术，用于从单次K2巡天光度测量中测量红巨星的全球地震学参数和引力模周期间距。两个深度残差神经网络以约80天光变曲线的全分辨率功率谱作为输入，无需背景拟合或模式识别，直接输出每个参数的概率分布，从而提供点估计和不对称不确定性。它们基于约800万个合成光谱进行训练，并在保留的合成数据及开普勒数据上进行评估。

    arXiv:2605.08051v2 Announce Type: replace-cross  Abstract: Gravity-mode period spacings (DPi_1) of red giants probe the stellar core directly, constraining its structure, mass and evolutionary state. Their measurement requires resolving narrow, densely spaced mixed modes and has so far relied on the four-year baseline of Kepler. Recovering DPi_1 from the much shorter (~80-day) baselines typical of K2 remains largely unexplored at the ensemble scale. We develop an automated machine-learning technique to measure global asteroseismic parameters and gravity-mode period spacings for red giants from single-campaign K2 photometry. Two deep residual neural networks take the full-resolution power spectrum of an ~80-day light curve as input, without background fitting or mode identification, and return a probability distribution for each parameter, yielding a point estimate and asymmetric uncertainties. They are trained on ~8 million synthetic spectra and evaluated on held-out synthetics, on Kep
    
[^54]: 多类与列表学习的最优样本复杂度

    The Optimal Sample Complexity of Multiclass and List Learning

    [https://arxiv.org/abs/2604.24749](https://arxiv.org/abs/2604.24749)

    本文通过证明最大超图密度受DS维上界限制，解决了长期猜想，从而确定了多类与列表学习的最优样本复杂度。

    

    arXiv:2604.24749v2 公告类型：替换 摘要：虽然基于VC维的二元分类最优样本复杂度已得到充分确立，但确定多类分类的最优样本复杂度仍然悬而未决。多类分类的适当复杂度参数是DS维，尽管付出了大量努力，样本复杂度的上下界之间仍存在$\sqrt{\text{DS}}$的差距。Hanneke等人（2026）的最新工作展示了多类假设类在DS维方面的新颖代数特征。在此基础上，我们证明任何多类假设类的最大超图密度均受其DS维的上界限制。这证明了Daniely和Shalev-Shwartz（2014）的一个长期猜想。因此，我们确定了多类及列表学习样本复杂度对DS维的最优依赖关系。

    arXiv:2604.24749v2 Announce Type: replace  Abstract: While the optimal sample complexity of binary classification in terms of the VC dimension is well-established, determining the optimal sample complexity of multiclass classification has remained open. The appropriate complexity parameter for multiclass classification is the DS dimension, and despite significant efforts, a gap of $\sqrt{\text{DS}}$ has persisted between the upper and lower bounds on sample complexity.   Recent work by Hanneke et al. (2026) shows a novel algebraic characterization of multiclass hypothesis classes in terms of their DS dimension. Building up on this, we show that the maximum hypergraph density of any multiclass hypothesis class is upper-bounded by its DS dimension. This proves a longstanding conjecture of Daniely and Shalev-Shwartz (2014). As a consequence, we determine the optimal dependence of the sample complexity on the DS dimension for multiclass as well as list learning.
    
[^55]: 面向不确定性感知事后解释的信息性扰动选择

    Informative Perturbation Selection for Uncertainty-Aware Post-hoc Explanations

    [https://arxiv.org/abs/2603.14894](https://arxiv.org/abs/2603.14894)

    本文提出EAGLE框架，将事后模型无关解释中的扰动选择视为信息论主动学习问题，通过自适应采样最大化期望信息增益的扰动，生成更可靠且不确定性感知的局部解释。

    

    arXiv:2603.14894v3 公告类型：交叉替换。摘要：由于不透明机器学习（ML）模型的广泛部署，信任和伦理问题促使了对可靠模型解释的需求。事后模型无关解释方法通过学习一个替代模型来解决这一挑战，该模型在感兴趣样本的局部区域近似部署的黑盒ML模型的行为。在事后场景中，底层模型参数和训练数据均不可用，因此，必须通过在感兴趣样本的邻域生成扰动输入及其对应的模型预测来构建局部邻域。我们提出了\texttt{EAGLE}（局部解释的期望主动增益），这是一种事后模型无关解释框架，将扰动选择形式化为信息论主动学习问题。通过自适应采样最大化期望信息增益的扰动，\texttt{EAGLE}能够高效地构建信息丰富的局部邻域，从而生成更可靠且不确定性感知的解释。

    arXiv:2603.14894v3 Announce Type: replace-cross  Abstract: Trust and ethical concerns due to the widespread deployment of opaque machine learning (ML) models motivating the need for reliable model explanations. Post-hoc model-agnostic explanation methods addresses this challenge by learning a surrogate model that approximates the behavior of the deployed black-box ML model in the locality of a sample of interest. In post-hoc scenarios, neither the underlying model parameters nor the training are available, and hence, this local neighborhood must be constructed by generating perturbed inputs in the neighborhood of the sample of interest, and its corresponding model predictions. We propose \emph{Expected Active Gain for Local Explanations} (\texttt{EAGLE}), a post-hoc model-agnostic explanation framework that formulates perturbation selection as an information-theoretic active learning problem. By adaptively sampling perturbations that maximize the expected information gain, \texttt{EAGL
    
[^56]: 利用机器学习辅助抽样与大型语言模型标注测量违规政策内容的普遍性

    Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling

    [https://arxiv.org/abs/2602.18518](https://arxiv.org/abs/2602.18518)

    本文提出了一种基于设计的测量系统，结合机器学习辅助抽样和多模态LLM标注，实现了对平台内容违规普遍性的高效、无偏估计，并支持多维度细分。

    

    arXiv:2602.18518v2 公告类型：替换  摘要：内容安全团队需要反映用户实际体验的指标，而不仅仅是报告的内容。我们研究普遍性：在给定日期，用户观看（印象）中流向违反特定政策内容的比例。准确的普遍性测量具有挑战性，因为违规行为往往罕见，且人工标注成本高昂，使得频繁、平台代表性的研究进展缓慢。我们提出了一种基于设计的测量系统，该系统（i）使用机器学习辅助权重从印象流中每日抽取概率样本，以集中标注预算于高曝光和高风险内容，同时保持无偏性，（ii）使用受政策提示和黄金集验证约束的多模态大型语言模型对样本项进行标注，以及（iii）生成设计一致的普遍性估计，包括置信区间和仪表板下钻分析。一个关键设计目标是采用单一全局样本并支持多种枢轴：同一每日样本支持按不同维度（如政策类型、内容类别等）的普遍性细分。

    arXiv:2602.18518v2 Announce Type: replace  Abstract: Content safety teams need metrics that reflect what users actually experience, not only what is reported. We study prevalence: the fraction of user views (impressions) that went to content violating a given policy on a given day. Accurate prevalence measurement is challenging because violations are often rare and human labeling is costly, making frequent, platform-representative studies slow. We present a design-based measurement system that (i) draws daily probability samples from the impression stream using ML-assisted weights to concentrate label budget on high-exposure and high-risk content while preserving unbiasedness, (ii) labels sampled items with a multimodal LLM governed by policy prompts and gold-set validation, and (iii) produces design-consistent prevalence estimates with confidence intervals and dashboard drilldowns. A key design goal is one global sample with many pivots: the same daily sample supports prevalence by su
    
[^57]: 论斐波那契集成：受黄金比例永恒架构启发的集成学习替代方法

    On Fibonacci Ensembles: An Alternative Approach to Ensemble Learning Inspired by the Timeless Architecture of the Golden Ratio

    [https://arxiv.org/abs/2512.22284](https://arxiv.org/abs/2512.22284)

    本文提出斐波那契集成框架，利用基于黄金比例的归一化权重和二阶递归结构，通过正交化与Rao-Blackwell优化实现基础学习器间的方差减少，从而补充和扩展经典集成方法。

    

    arXiv:2512.22284v2 公告类型：替换-交叉 摘要：自然很少直白地揭示其秘密，但在斐波那契序列中，她让我们得以一窥其生长、和谐与递归稳定性的静谧架构\citep{Koshy2001Fibonacci, Livio2002GoldenRatio}。从螺旋星系到叶片的展开，这一谦逊的序列反映了一种普遍的平衡语法。在这项工作中，我们引入了\emph{斐波那契集成}，这是一种数学上有原则且受哲学启发的集成学习框架，它补充并扩展了诸如装袋、提升和随机森林等经典聚合方案\citep{Breiman1996Bagging, Breiman2001RandomForests, Friedman2001GBM, Zhou2012Ensemble, HastieTibshiraniFriedman2009ESL}。两个相互交织的公式展开：（1）使用归一化的斐波那契权重——通过正交化和Rao-Blackwell优化进行调和——以实现基础学习器间系统性的方差减少，以及（2）一个二阶递归...

    arXiv:2512.22284v2 Announce Type: replace-cross  Abstract: Nature rarely reveals her secrets bluntly, yet in the Fibonacci sequence she grants us a glimpse of her quiet architecture of growth, harmony, and recursive stability \citep{Koshy2001Fibonacci, Livio2002GoldenRatio}. From spiral galaxies to the unfolding of leaves, this humble sequence reflects a universal grammar of balance. In this work, we introduce \emph{Fibonacci Ensembles}, a mathematically principled yet philosophically inspired framework for ensemble learning that complements and extends classical aggregation schemes such as bagging, boosting, and random forests \citep{Breiman1996Bagging, Breiman2001RandomForests, Friedman2001GBM, Zhou2012Ensemble, HastieTibshiraniFriedman2009ESL}. Two intertwined formulations unfold: (1) the use of normalized Fibonacci weights -- tempered through orthogonalization and Rao--Blackwell optimization -- to achieve systematic variance reduction among base learners, and (2) a second-order rec
    
[^58]: 噪声高维数据集相似性与可对齐性的推断

    Inference for Similarity and Alignability between Noisy High-Dimensional Datasets

    [https://arxiv.org/abs/2511.21074](https://arxiv.org/abs/2511.21074)

    本文提出了一种基于流形信号加噪声模型和谱特性的统计推断框架，用于在异质噪声下度量高维数据集的相似性与可对齐性，该度量具有尺度与旋转不变性。

    

    arXiv:2511.21074v2 公告类型：替换-交叉 摘要：高维数据集在广泛科学领域的快速增长，迫切需要新的统计方法来比较具有潜在低维结构的分布。评估观测值集中在低维流形附近的高维数据集之间的相似性，由于高维中噪声的非平凡效应而特别具有挑战性。我们提出了一个原则性框架，用于在异质噪声下对具有低维平滑信号的高维数据集的相似性和可对齐性进行统计推断。关键思想是将观测数据矩阵的谱特性与其底层信号分布的几何结构联系起来。在流形信号加噪声模型下，我们基于底层信号相关的主方差，开发了一种尺度不变和旋转不变的两个数据集之间的相异度度量。

    arXiv:2511.21074v2 Announce Type: replace-cross  Abstract: The rapid growth of high-dimensional datasets across a wide range of scientific domains has created an urgent need for new statistical methods to compare distributions with underlying low-dimensional structure. Assessing similarity between high-dimensional datasets whose observations concentrate near low-dimensional manifolds is particularly challenging due to the nontrivial effects of noise in high dimensions. We propose a principled framework for statistical inference on the similarity and alignability of high-dimensional datasets with low-dimensional smooth signals under heterogeneous noise. The key idea is to link the spectral properties of the observed data matrices to the geometry of their underlying signal distributions. Under a manifold signal-plus-noise model, we build on the principal variances associated to the underlying signals and develop a scale- and rotation-invariant dissimilarity measure between two datasets t
    
[^59]: 受限平均奖励马尔可夫决策过程的近最优样本复杂度界

    Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs

    [https://arxiv.org/abs/2509.16586](https://arxiv.org/abs/2509.16586)

    本文提出了生成模型下受限平均奖励MDP的基于模型算法，在松弛和严格可行性设置下分别实现了近最优的样本复杂度界。

    

    arXiv:2509.16586v2 公告类型：替换 摘要：近期进展显著提升了我们对生成模型下平均奖励马尔可夫决策过程（AMDPs）学习样本复杂度的理解。然而，对于受限平均奖励MDP（CAMDP），其中策略必须满足长期平均约束，已知结果较少。在本工作中，我们通过研究生成模型下CAMDPs中学习ε-最优策略的样本复杂度来填补这一空白。我们提出了一种基于模型的算法，该算法在两种设置下运行：（i）松弛可行性，允许较小的约束违反；（ii）严格可行性，其中输出策略满足约束。我们证明，在松弛和严格可行性设置下，我们的算法分别实现了样本复杂度为$\tilde{O}\left(\frac{S A (B+H)}{ \epsilon^2}\right)$和$\tilde{O} \left(\frac{S A (B+H)}{\epsilon^2 \zeta^2} \right)$。这里，$\zeta$是Slater常数。

    arXiv:2509.16586v2 Announce Type: replace  Abstract: Recent advances have significantly improved our understanding of the sample complexity of learning in average-reward Markov decision processes (AMDPs) under the generative model. However, much less is known about the constrained average-reward MDP (CAMDP), where policies must satisfy long-run average constraints. In this work, we address this gap by studying the sample complexity of learning an $\epsilon$-optimal policy in CAMDPs under a generative model. We propose a model-based algorithm that operates under two settings: (i) relaxed feasibility, which allows small constraint violations, and (ii) strict feasibility, where the output policy satisfies the constraint. We show that our algorithm achieves sample complexities of $\tilde{O}\left(\frac{S A (B+H)}{ \epsilon^2}\right)$ and $\tilde{O} \left(\frac{S A (B+H)}{\epsilon^2 \zeta^2} \right)$ under the relaxed and strict feasibility settings, respectively. Here, $\zeta$ is the Slater
    
[^60]: 通过公共二阶矩增强差分隐私线性回归

    Enhancing Differentially Private Linear Regression via Public Second-Moment

    [https://arxiv.org/abs/2508.18037](https://arxiv.org/abs/2508.18037)

    本文提出一种利用公共二阶矩矩阵转换私有数据的新方法，以改善差分隐私线性回归中充分统计量扰动估计器的条件数，从而提升其准确性和鲁棒性。

    

    arXiv:2508.18037v2 公告类型：替换 摘要：利用公共数据的信息已成为提升差分隐私（DP）方法效用的关键。传统的DP方法通常仅基于私有数据添加噪声，这可能会显著降低效用。在本文中，我们针对无界数据假设下基于充分统计量扰动（SSP）的线性回归普通最小二乘估计器（OLSE）的背景下解决了这一局限性。我们提出了一种新方法，涉及使用公共二阶矩矩阵转换私有数据，以计算转换后的SSP-OLSE，其二阶矩矩阵具有更好的条件数，从而提高了OLSE的准确性和鲁棒性。我们推导了关于我们的方法和标准SSP-OLSE相对于非DP OLSE的理论误差界，这揭示了我们的方法所实现的改进鲁棒性和准确性。在合成和真实世界数据集上的实验证明了...

    arXiv:2508.18037v2 Announce Type: replace  Abstract: Leveraging information from public data has become increasingly crucial in enhancing the utility of differentially private (DP) methods. Traditional DP approaches often require adding noise based solely on private data, which can significantly degrade utility. In this paper, we address this limitation in the context of the ordinary least squares estimator (OLSE) of linear regression based on sufficient statistics perturbation (SSP) under the unbounded data assumption. We propose a novel method that involves transforming private data using the public second-moment matrix to compute a transformed SSP-OLSE, whose second-moment matrix yields a better condition number and improves the OLSE accuracy and robustness. We derive theoretical error bounds about our method and the standard SSP-OLSE to the non-DP OLSE, which reveal the improved robustness and accuracy achieved by our approach. Experiments on synthetic and real-world datasets demon
    
[^61]: 基于投影的多保真线性回归用于数据稀缺应用

    Projection-based multifidelity linear regression for data-scarce applications

    [https://arxiv.org/abs/2508.08517](https://arxiv.org/abs/2508.08517)

    本文提出了两种基于投影的多保真线性回归方法，通过主成分降维和两种数据增强策略，有效利用低保真数据提升高保真模型在数据稀缺和高维输出场景下的预测精度。

    

    摘要：针对高维感兴趣量的系统进行代理建模仍然具有挑战性，尤其是在训练数据获取成本高昂的情况下。本文开发了多输入多输出线性回归的多保真方法，针对数据有限且输出高维的应用。多保真方法将大量廉价的低保真模型评估与有限且昂贵的高保真评估相结合。我们引入了两种基于投影的多保真线性回归方法，分别采用线性和非线性特征，利用主成分基向量进行降维，并通过以下方式组合多保真数据：（i）使用低保真数据进行直接数据增强，以及（ii）在低保真和高保真数据之间引入显式线性修正的数据增强。这些数据增强方法将高保真和低保真数据合并为一个统一的训练集。

    arXiv:2508.08517v2 Announce Type: replace-cross  Abstract: Surrogate modeling for systems with high-dimensional quantities of interest remains challenging, particularly when training data are costly to acquire. This work develops multifidelity methods for multiple-input multiple-output linear regression targeting data-limited applications with high-dimensional outputs. Multifidelity methods integrate many inexpensive low-fidelity model evaluations with limited, costly high-fidelity evaluations. We introduce two projection-based multifidelity linear regression approaches with linear and nonlinear features that leverage principal component basis vectors for dimensionality reduction and combine multifidelity data through: (i) a direct data augmentation using low-fidelity data, and (ii) a data augmentation incorporating explicit linear corrections between low-fidelity and high-fidelity data. The data augmentation approaches combine high-fidelity and low-fidelity data into a unified trainin
    
[^62]: ROC-n-reroll：验证器不完善性如何影响测试时扩展

    ROC-n-reroll: How verifier imperfection affects test-time scaling

    [https://arxiv.org/abs/2507.12399](https://arxiv.org/abs/2507.12399)

    本文通过理论证明和实验验证，指出验证器的ROC曲线几何决定测试时扩展方法的精度，并发现RS在固定计算下优于BoN，但无法从低计算量表现预测高计算量性能。

    

    arXiv:2507.12399v3 公告类型：替换 摘要：测试时扩展旨在通过推理期间利用额外计算来提升语言模型性能。许多工作实证研究了诸如Best-of-N（BoN）和拒绝采样（RS）等技术，这些技术利用验证器来实现测试时扩展。然而，迄今为止，对于验证器不完善性如何影响性能的理论理解甚少——我们在此工作中填补了这一空白。具体而言，我们证明了这些方法的实例级精度恰好由验证器的ROC曲线几何特性决定。我们的理论有两个重要结论，这些结论通过使用Qwen和LLama模型在GSM8K和MATH500上的实验得到证实。首先，在固定计算量下，RS优于BoN，而两种方法在无限计算极限下收敛到相同的精度。其次，通常无法基于低计算量下的观测来预测任一方法在高计算量下的性能。

    arXiv:2507.12399v3 Announce Type: replace  Abstract: Test-time scaling aims to improve language model performance by leveraging additional compute during inference. Many works have empirically studied techniques such as Best-of-N (BoN) and Rejection Sampling (RS) that make use of a verifier to enable test-time scaling. However, to date there is little theoretical understanding of how verifier imperfection affects performance -- a gap we address in this work. Specifically, we prove that the instance-level accuracy of these methods is precisely characterized by the geometry of the verifier's ROC curve. Our theory has two important takeaways, confirmed by experiments with Qwen and LLama models on GSM8K and MATH500. First, RS outperforms BoN for fixed compute, while both methods converge to the same accuracy in the infinite-compute limit. Second, it is generally impossible to predict the high-compute performance of either method based on observations in the low-compute regime.
    
[^63]: 利用生成式人工智能对非结构化数据进行因果推断

    Leveraging Generative Artificial Intelligence for Causal Inference with Unstructured Data

    [https://arxiv.org/abs/2507.03897](https://arxiv.org/abs/2507.03897)

    本文提出GPI框架，利用开源生成式AI模型提取非结构化数据的低维表示，无需微调即可进行因果推断，兼顾计算效率与不确定性量化。

    

    arXiv:2507.03897v4 公告类型：替换 摘要：我们引入了生成式人工智能驱动的推断（GPI），这是一个用于基于非结构化数据（包括文本和图像）进行因果和预测推断的统计框架。GPI利用开源生成式人工智能（GenAI）模型——如大型语言模型和扩散模型——不仅大规模生成非结构化数据，还提取能够保证捕获其底层结构的低维表示。通过将机器学习应用于这些表示，GPI能够估计因果效应，同时量化相关的估计不确定性。与现有的表示学习方法不同，GPI不需要对生成模型进行微调，因此计算效率高且易于广泛使用。我们通过三个应用展示了GPI框架的多功能性：（1）在调整文本混杂因素的同时，估计中国社交媒体审查的影响。

    arXiv:2507.03897v4 Announce Type: replace  Abstract: We introduce GenAI-Powered Inference (GPI), a statistical framework for both causal and predictive inference using unstructured data, including text and images. GPI leverages open-source Generative Artificial Intelligence (GenAI) models---such as large language models and diffusion models---not only to generate unstructured data at scale but also to extract low-dimensional representations that are guaranteed to capture their underlying structure. Applying machine learning to these representations, GPI enables estimation of causal effects while quantifying associated estimation uncertainty. Unlike existing approaches to representation learning, GPI does not require fine-tuning of generative models, making it computationally efficient and broadly accessible. We illustrate the versatility of the GPI framework through three applications: (1) estimating the effects of Chinese social media censorship while adjusting for textual confounders
    
[^64]: 多视相干成像：理论保证与算法

    Multilook Coherent Imaging: Theoretical Guarantees and Algorithms

    [https://arxiv.org/abs/2505.23594](https://arxiv.org/abs/2505.23594)

    本文首次在深度图像先验假设下为多视相干成像的最大似然估计器建立了均方误差的理论上界，并提供了相应的算法框架，填补了该领域理论基础研究的空白。

    

    arXiv:2505.23594v2 公告类型：替换交叉  摘要：多视相干成像是一种广泛应用于数字全息、超声成像和合成孔径雷达等领域的技术。这些系统中的一个核心挑战是乘性噪声（通常称为散斑）的存在，它会降低图像质量。尽管相干成像系统被广泛使用，但其理论基础仍相对未被充分探索。在本文中，我们研究了基于似然方法的多视相干成像的理论和算法方面，为分析和方法开发提供了严格的框架。我们的理论贡献包括在深度图像先验假设下，首次建立了最大似然估计器均方误差（MSE）的理论上界。我们的结果捕捉了MSE对深度图像先验中参数数量、视数、信号维度和n的依赖性。

    arXiv:2505.23594v2 Announce Type: replace-cross  Abstract: Multilook coherent imaging is a widely used technique in applications such as digital holography, ultrasound imaging, and synthetic aperture radar. A central challenge in these systems is the presence of multiplicative noise, commonly known as speckle, which degrades image quality. Despite the widespread use of coherent imaging systems, their theoretical foundations remain relatively underexplored. In this paper, we study both the theoretical and algorithmic aspects of likelihood-based approaches for multilook coherent imaging, providing a rigorous framework for analysis and method development. Our theoretical contributions include establishing the first theoretical upper bound on the Mean Squared Error (MSE) of the maximum likelihood estimator under the deep image prior hypothesis. Our results capture the dependence of MSE on the number of parameters in the deep image prior, the number of looks, the signal dimension, and the n
    
[^65]: 一次性鲁棒联邦独立成分分析

    One-shot Robust Federated Learning of Independent Component Analysis

    [https://arxiv.org/abs/2505.20532](https://arxiv.org/abs/2505.20532)

    提出了一种基于谱聚类和几何中位数的联邦ICA一次性聚合方法，有效解决了符号置换和异构质量问题，并在大量低质量数据下保持鲁棒性。

    

    arXiv:2505.20532v2 公告类型：替换 摘要：本文研究了分布式和联邦独立成分分析（ICA）中的鲁棒一次性聚合问题。在该场景下，每个客户端计算一个局部ICA估计器，而服务器旨在在不访问原始数据的情况下恢复一个共同的全局混合矩阵。主要难点在于局部ICA估计器仅能通过符号置换进行识别，且其估计质量可能高度异构。我们提出了谱鲁棒联邦ICA（SRF-ICA），一种一次性聚合方法，该方法从所有局部原子构建符号不变亲和矩阵，执行谱k均值以解决置换模糊性，在每个估计簇内对齐符号，然后应用几何中位数进行鲁棒聚合。我们证明了谱聚类步骤控制了簇级误聚类率，并且即使当大量局部原子来自低质量数据时，最终估计器仍能保持准确性。

    arXiv:2505.20532v2 Announce Type: replace  Abstract: This paper studies robust one-shot aggregation for distributed and federated Independent Component Analysis (ICA). In this setting, each client computes a local ICA estimator, while the server aims to recover a common global mixing matrix without accessing raw data. The main difficulty is that local ICA estimators are identifiable only up to signed permutations and may have highly heterogeneous estimation quality. We propose Spectral-Robust-Federated ICA (SRF-ICA), a one-shot aggregation method that constructs a sign-invariant affinity matrix from all local atoms, performs spectral k-means to resolve the permutation ambiguity, aligns signs within each estimated cluster, and then applies the geometric median for robust aggregation. We prove that the spectral clustering step controls the cluster-wise misclustering rate, and that the final estimator remains accurate even when a substantial fraction of local atoms are produced from low-q
    
[^66]: 弱物理信息神经网络用于流形上几何兼容的双曲守恒律

    Weak Physics Informed Neural Networks for Geometry Compatible Hyperbolic Conservation Laws on Manifolds

    [https://arxiv.org/abs/2505.19036](https://arxiv.org/abs/2505.19036)

    本文提出了一种弱物理信息神经网络（wPINN）框架，通过建立局部$L_1$-稳定性估计和收敛性分析，首次为流形上低正则性双曲守恒律的熵解提供了严格的近似保证。

    

    arXiv:2505.19036v3 公告类型：替换-交叉 摘要：物理信息神经网络（PINNs）提供了一种无网格方法，用于在复杂几何上求解高维偏微分方程，但其在流形上的理论基础仍然有限。此外，传统的PINN分析通常依赖于解的平滑性，而PINNs对于由非线性双曲方程产生的低正则性解可能表现不佳。在本文中，我们开发了一个弱PINN（wPINN）框架，用于近似黎曼流形$\mathcal{M}^d$上几何兼容的双曲守恒律的熵解。基于适定性理论，我们建立了一个局部$L_1$-稳定性估计，将局部熵残差转化为终端误差界，并引出了所提方法的严格收敛性分析。然后，我们推导了流形上时间依赖熵解的近似保证，揭示了近似误差如何在长时间范围内累积。

    arXiv:2505.19036v3 Announce Type: replace-cross  Abstract: Physics-informed neural networks (PINNs) provide a mesh-free approach to solving high-dimensional PDEs on complex geometries, but their theoretical foundations on manifolds remain limited. Moreover, conventional PINN analyses typically rely on solution smoothness, while PINNs may perform poorly for low-regularity solutions arising from nonlinear hyperbolic equations. In this paper, we develop a weak PINN (wPINN) framework for approximating entropy solutions of geometry-compatible hyperbolic conservation laws on Riemannian manifolds $\mathcal{M}^d$. Building on the well-posedness theory, we establish a localized $L_1$-stability estimate that converts localized entropy residuals into terminal error bounds and leads to a rigorous convergence analysis of the proposed method. We then derive approximation guarantees for time-dependent entropy solutions on manifolds, revealing how approximation errors accumulate over long time horizon
    
[^67]: WATCH：基于加权共形鞅的AI部署自适应监控

    WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales

    [https://arxiv.org/abs/2505.04608](https://arxiv.org/abs/2505.04608)

    本文提出加权共形测试鞅（WCTMs），以支持AI部署后的在线自适应监控，克服了现有方法在假设类别限制、缺乏适应性和诊断能力上的不足。

    

    arXiv:2505.04608v5 公告类型：交叉替换 摘要：在高风险环境中负责任地部署人工智能（AI）/机器学习（ML）系统，不仅需要证明系统的可靠性，还需要持续的后部署监控，以快速检测并处理任何不安全行为。非参数序贯测试方法——尤其是共形测试鞅（CTMs）和随时有效推断——为这一监控任务提供了有前景的工具。然而，现有方法局限于监控有限的假设类别或“警报标准”（例如，检测违反特定交换性或独立同分布假设的数据偏移），不允许在响应偏移时进行在线自适应，和/或无法诊断退化或警报的原因。在本文中，我们通过提出加权共形测试鞅（WCTMs）的泛化形式来解决这些局限性，这为任何意外事件的在线监控奠定了理论基础。

    arXiv:2505.04608v5 Announce Type: replace-cross  Abstract: Responsibly deploying artificial intelligence (AI) / machine learning (ML) systems in high-stakes settings arguably requires not only proof of system reliability, but also continual, post-deployment monitoring to quickly detect and address any unsafe behavior. Methods for nonparametric sequential testing -- especially conformal test martingales (CTMs) and anytime-valid inference -- offer promising tools for this monitoring task. However, existing approaches are restricted to monitoring limited hypothesis classes or ``alarm criteria'' (e.g., detecting data shifts that violate certain exchangeability or IID assumptions), do not allow for online adaptation in response to shifts, and/or cannot diagnose the cause of degradation or alarm. In this paper, we address these limitations by proposing a weighted generalization of conformal test martingales (WCTMs), which lay a theoretical foundation for online monitoring for any unexpected 
    
[^68]: 关于幂一次序贯检验停止时间的紧致下界与上界

    On Stopping Times of Power-one Sequential Tests: Tight Lower and Upper Bounds

    [https://arxiv.org/abs/2504.19952](https://arxiv.org/abs/2504.19952)

    本文提出了两个适用于任意复合假设检验的通用停止时间下界，覆盖Wald和Farrell两种设定，无需主导测度，显著推广了现有理论。

    

    arXiv:2504.19952v2 公告类型：替换-交叉 摘要：我们提出了两个关于任意复合零假设$\mathcal P$与备择假设$\mathcal Q$之间序贯检验停止时间的通用下界。第一个下界适用于“Wald设定”，即当固定备择假设$Q \in \mathcal Q$时，第一类错误水平$\alpha$趋近于零，该下界等于$\log(1/\alpha)$除以$\mathcal P$与$Q$之间的某个特定infimum KL散度，记为$\operatorname{KL_{inf}}$。第二个下界适用于“Farrell设定”，即当$\alpha$固定且$\operatorname{KL_{inf}}$沿着一系列备择假设趋近于零时，该序列所需的期望样本量至少为$\operatorname{KL^{-1}_{inf}} \log \log \operatorname{KL^{-1}_{inf}}$量级。我们的主要贡献在于这些下界的通用性，它们适用于非参数、复合设定，无需主导参考测度，从而显著推广了现有结果。

    arXiv:2504.19952v2 Announce Type: replace-cross  Abstract: We present two general lower bounds for stopping times of sequential tests between arbitrary composite nulls $\mathcal P$ and alternatives $\mathcal Q$. The first lower bound is for the ``Wald setting'' where the type-1 error level $\alpha$ approaches zero for a fixed alternative $Q \in \mathcal Q$, and equals $\log(1/\alpha)$ divided by a certain infimum KL divergence between $\mathcal P$ and $Q$, termed $\operatorname{KL_{inf}}$. The second lower bound applies to the ``Farrell setting'', where $\alpha$ is fixed and $\operatorname{KL_{inf}}$ approaches $0$ along a sequence of alternatives such that the required expected sample size along that sequence is of order at least $\operatorname{KL^{-1}_{inf}} \log \log \operatorname{KL^{-1}_{inf}}$. Our main contribution is the generality of these bounds, which hold in non-parametric, composite settings, without requiring a dominating reference measure, substantially generalizing the 
    
[^69]: 将生成式学习引入表示学习：作为分布匹配的自监督迁移学习

    Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching

    [https://arxiv.org/abs/2502.14424](https://arxiv.org/abs/2502.14424)

    本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。

    

    arXiv:2502.14424v3 公告类型：替换交叉 摘要：大多数自监督学习目标旨在防止表示坍缩，但未明确目标表示规律。我们将表示学习形式化为分布匹配（DM），学习一个增强不变的编码器，其诱导的分布规律与一个明确的几何参考匹配。参考规律指定了学习到的表示分布应呈现的形式，而单独选择的差异度量则用于衡量与该目标的偏差；此处我们使用马氏距离。DM框架揭示了一个方向性反转：生成式学习将可处理的参考映射到数据，而表示学习则将数据映射到设计的参考规律。我们将总体目标与类中心分离和分类误差联系起来，并证明了非渐近神经筛保证。模拟和图像基准测试显示了流形校正、细粒度结构和跨标签空间的迁移能力。

    arXiv:2502.14424v3 Announce Type: replace-cross  Abstract: Most self-supervised learning objectives defend against collapse but leave the target representation law unspecified. We formulate representation learning as Distribution Matching (DM), learning an augmentation-invariant encoder whose induced law matches an explicit geometric reference. The reference law specifies what the learned representation distribution should look like, whereas a separately chosen discrepancy determines how deviations from this target are measured; here we use Mallows distance. The DM framework reveals a directional inverse: generative learning maps a tractable reference to data, whereas representation learning maps data to a designed reference law. We connect the population objective to class-centre separation and classification error and prove a non-asymptotic neural-sieve guarantee. Simulations and image benchmarks show manifold rectification, fine-grained structure and transfer across label spaces.
    
[^70]: 前沿可扩展评估的极限：作为评委的LLM无法胜过两倍数据

    Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data

    [https://arxiv.org/abs/2410.13341](https://arxiv.org/abs/2410.13341)

    当评委模型不比被评估模型更准确时，任何去偏方法最多只能将所需的地面真实标签减少一半，这暴露了LLM作为评委范式的根本局限。

    

    arXiv:2410.13341v4 公告类型：替换 摘要：在爆炸性增长的机器学习生态系统中，高质量标注日益成为瓶颈。因此，避免昂贵标注的可扩展评估方法已成为重要的研究目标。许多人希望利用现有的强大模型替代昂贵标签，以提供廉价的模型评估。不幸的是，这种使用模型作为评委的方法引入了偏见，如自我偏好，这可能扭曲模型比较。新兴的去偏工具系列承诺通过使用少量高质量标签来纠正大量模型判断中的这些问题。在本文中，我们研究了这些去偏方法原则上能走多远。我们的主要结果表明，当评委的准确性不高于被评估模型时，任何去偏方法都无法将所需的地面真实标签数量减少超过一半。我们的结果揭示了LLM作为评委范式在前沿的严重局限性。

    arXiv:2410.13341v4 Announce Type: replace  Abstract: High quality annotations are increasingly a bottleneck in the explosively growing machine learning ecosystem. Scalable evaluation methods that avoid costly annotation have therefore become an important research ambition. Many hope to use strong existing models in lieu of costly labels to provide cheap model evaluations. Unfortunately, this method of using models as judges introduces biases, such as self-preferencing, that can distort model comparisons. An emerging family of debiasing tools promises to fix these issues by using a few high quality labels to debias a large number of model judgments. In this paper, we study how far such debiasing methods, in principle, can go. Our main result shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half. Our result speaks to the severe limitations of the LLM-as-a-judge paradigm at the 
    
[^71]: 分数攻击：最优差分隐私学习的一种下界技术

    Score Attack: A Lower Bound Technique for Optimal Differentially Private Learning

    [https://arxiv.org/abs/2303.07152](https://arxiv.org/abs/2303.07152)

    本文提出“分数攻击”方法，为差分隐私下的参数估计极小极大风险提供最优下界，适用于任意具有分数统计量的统计模型。

    

    在确保个人数据隐私的同时实现最优统计性能，是现代数据分析中一个具有挑战性但至关重要的目标。然而，在隐私约束下刻画最优性，特别是极小极大下界，在技术上十分困难。为解决这一问题，我们提出了一种名为“分数攻击”的新方法，它为差分隐私约束下的参数估计极小极大风险提供了一个下界。分数攻击方法基于差分隐私中的追踪攻击概念，可应用于任何具有良好定义分数统计量的统计模型。它能够以对数因子为精度，在确保差分隐私的前提下，为一系列统计问题最优地给出未知模型参数估计极小极大风险的下界。我们通过多个示例（如广义线性模型）展示了这一通用方法的有效性和最优性。

    arXiv:2303.07152v3 Announce Type: replace-cross  Abstract: Achieving optimal statistical performance while ensuring the privacy of personal data is a challenging yet crucial objective in modern data analysis. However, characterizing the optimality, particularly the minimax lower bound, under privacy constraints is technically difficult.   To address this issue, we propose a novel approach called the score attack, which provides a lower bound on the differential-privacy-constrained minimax risk of parameter estimation. The score attack method is based on the tracing attack concept in differential privacy and can be applied to any statistical model with a well-defined score statistic. It can optimally lower bound the minimax risk of estimating unknown model parameters, up to a logarithmic factor, while ensuring differential privacy for a range of statistical problems. We demonstrate the effectiveness and optimality of this general method in various examples, such as the generalized linea
    
[^72]: 有限动作线性上下文赌博机中的序贯批量学习

    Sequential Batch Learning in Finite-Action Linear Contextual Bandits

    [https://arxiv.org/abs/2004.06321](https://arxiv.org/abs/2004.06321)

    本文提出了有限动作线性上下文赌博机中的序贯批量学习问题，并针对任意生成和共同高斯分布两种上下文设置进行了理论分析。

    

    arXiv:2004.06321v2 公告类型：替换 摘要：我们研究了具有有限动作集的线性上下文赌博机中的序贯批量学习问题，其中决策者被限制将新到的个体分成（最多）固定数量的批次，并且只能在每批结束时观察到该批次内个体的结果。与标准在线上下文赌博机学习和上下文赌博机中的离线策略学习相比，这一序贯批量学习问题为许多实际应用中的个性化序贯决策问题提供了更细粒度的表述，包括临床试验中的医疗治疗、电子商务中的产品推荐以及众包中的自适应实验设计。我们研究了该问题的两种设置：一种是上下文任意生成的设置，另一种是上下文向量在不同动作和时间上相互独立且遵循共同高斯分布的设置。在每种设置中，我们建立了...

    arXiv:2004.06321v2 Announce Type: replace  Abstract: We study the sequential batch learning problem in linear contextual bandits with finite action sets, where the decision maker is constrained to split incoming individuals into (at most) a fixed number of batches and can only observe outcomes for the individuals within a batch at the batch's end. Compared with both standard online contextual-bandit learning and offline policy learning in contextual bandits, this sequential batch learning problem provides a finer-grained formulation of many personalized sequential decision making problems in practical applications, including medical treatment in clinical trials, product recommendation in e-commerce and adaptive experiment design in crowdsourcing.   We study two settings of the problem: one where the contexts are arbitrarily generated and the other where the context vectors are mutually independent across actions and time and follow a common Gaussian distribution. In each setting, we es
    
[^73]: ODTlearn: 一个用于学习预测和处方的最优决策树的包

    ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription. (arXiv:2307.15691v1 [stat.ML])

    [http://arxiv.org/abs/2307.15691](http://arxiv.org/abs/2307.15691)

    ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。

    

    ODTLearn是一个开源的Python包，提供了基于混合整数优化(MIO)框架的高风险预测和处方任务的最优决策树学习方法。该包的当前版本提供了学习最优分类树、公平最优分类树、鲁棒最优分类树和从观测数据学习最优处方树的实现。我们设计了该包以便于维护和扩展，当引入新的最优决策树问题类、重构策略和解决算法时，可以轻松更新。为此，该包遵循面向对象的设计原则，并支持商业(Gurobi)和开源(COIN-OR branch and cut)求解器。包的文档和详细用户指南可以在https://d3m-research-group.github.io/odtlearn/找到。

    ODTLearn is an open-source Python package that provides methods for learning optimal decision trees for high-stakes predictive and prescriptive tasks based on the mixed-integer optimization (MIO) framework proposed in Aghaei et al. (2019) and several of its extensions. The current version of the package provides implementations for learning optimal classification trees, optimal fair classification trees, optimal classification trees robust to distribution shifts, and optimal prescriptive trees from observational data. We have designed the package to be easy to maintain and extend as new optimal decision tree problem classes, reformulation strategies, and solution algorithms are introduced. To this end, the package follows object-oriented design principles and supports both commercial (Gurobi) and open source (COIN-OR branch and cut) solvers. The package documentation and an extensive user guide can be found at https://d3m-research-group.github.io/odtlearn/. Additionally, users can view
    
[^74]: 高斯混合块模型中的谱聚类

    Spectral clustering in the Gaussian mixture block model. (arXiv:2305.00979v1 [stat.ML])

    [http://arxiv.org/abs/2305.00979](http://arxiv.org/abs/2305.00979)

    本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。

    

    高斯混合块模型是用于模拟现代网络的图分布：对于这样的模型生成一个图，我们将每个顶点 $i$ 与一个从高斯混合中抽样到的潜在特征向量 $u_i \in \mathbb{R}^d$ 相关联，当且仅当特征向量足够相似，即 $\langle u_i,u_j \rangle \ge \tau$ 时，我们才会添加边 $(i,j)$。高斯混合的不同组成部分表示可能具有不同特征分布的不同类型的节点，例如在社交网络中，每个组成部分都表示独特社区的不同属性。这些网络涉及到的自然算法任务有嵌入（恢复潜在的特征向量）和聚类（通过其混合组分将节点分组）。本文开启了对从高维高斯混合块模型抽样的图进行聚类和嵌入研究。

    Gaussian mixture block models are distributions over graphs that strive to model modern networks: to generate a graph from such a model, we associate each vertex $i$ with a latent feature vector $u_i \in \mathbb{R}^d$ sampled from a mixture of Gaussians, and we add edge $(i,j)$ if and only if the feature vectors are sufficiently similar, in that $\langle u_i,u_j \rangle \ge \tau$ for a pre-specified threshold $\tau$. The different components of the Gaussian mixture represent the fact that there may be different types of nodes with different distributions over features -- for example, in a social network each component represents the different attributes of a distinct community. Natural algorithmic tasks associated with these networks are embedding (recovering the latent feature vectors) and clustering (grouping nodes by their mixture component).  In this paper we initiate the study of clustering and embedding graphs sampled from high-dimensional Gaussian mixture block models, where the
    
[^75]: 因子模型中的双重稳健最近邻方法

    Doubly robust nearest neighbors in factor models. (arXiv:2211.14297v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14297](http://arxiv.org/abs/2211.14297)

    该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。

    

    我们介绍并分析了在潜在因子模型中处理缺失数据的改进最近邻（NN）方法。我们考虑一个带有缺失数据的矩阵补全问题，其中当被观察到时，第$(i, t)$个条目由其均值$f(u_i, v_t)$加上均值为零的噪声给出，其中$f$为未知函数，$u_i$和$v_t$为潜在因子。之前的NN策略，如单元-单元NN，用于估计均值$f(u_i, v_t)$，依赖于存在其他行$j$使得$u_j \approx u_i$。类似地，时间-时间NN策略依赖于存在列$t'$使得$v_{t'} \approx v_t$。当相似行或相似列不可用时，这些策略的性能较差。我们的估计在两个方面对这种不足是双重稳健的：(1) 只要存在良好的行或列邻居，我们的估计提供一致的估计。 (2) 此外，如果存在良好的行和列邻居，它提供了（近似）二次改进非渐近性能。

    We introduce and analyze an improved variant of nearest neighbors (NN) for estimation with missing data in latent factor models. We consider a matrix completion problem with missing data, where the $(i, t)$-th entry, when observed, is given by its mean $f(u_i, v_t)$ plus mean-zero noise for an unknown function $f$ and latent factors $u_i$ and $v_t$. Prior NN strategies, like unit-unit NN, for estimating the mean $f(u_i, v_t)$ relies on existence of other rows $j$ with $u_j \approx u_i$. Similarly, time-time NN strategy relies on existence of columns $t'$ with $v_{t'} \approx v_t$. These strategies provide poor performance respectively when similar rows or similar columns are not available. Our estimate is doubly robust to this deficit in two ways: (1) As long as there exist either good row or good column neighbors, our estimate provides a consistent estimate. (2) Furthermore, if both good row and good column neighbors exist, it provides a (near-)quadratic improvement in the non-asympto
    

