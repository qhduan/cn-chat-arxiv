# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TorchCurves: Differentiable Parametric Curves](https://arxiv.org/abs/2611.14516) | TorchCurves是一个将B样条和勒让德多项式参数曲线作为可学习基元的可微分PyTorch库，支持端到端优化，可广泛应用于连续特征嵌入、KAN网络和函数回归等任务。 |
| [^2] | [A Theory of Policy Optimization in Post-Training: Coverage Collapse and Reference Resets](https://arxiv.org/abs/2611.14467) | 该论文为语言模型后训练中的策略优化建立了理论框架，证明REINFORCE类算法（PPO和GRPO的基础）即使奖励提升也会发生“覆盖坍缩”、使最优响应概率趋近于零并导致优化停滞，而通过重置KL参考等手段可实现由基础模型响应覆盖度所约束的保覆盖近优学习。 |
| [^3] | [One Curve, Different Weights: AUC, the Brier Score, Decision-Weighted AUC, and the Policy Youden Index](https://arxiv.org/abs/2611.14465) | 论文揭示AUC、Brier分数与提出的决策加权AUC（AUC_π）本质上是同一条Youden指数曲线在不同权重下的加权平均，其中AUC_π以决策者的行动概率为权重，从而将模型评估与实际决策需求对齐。 |
| [^4] | [A Theory of Cross-Validation Variance](https://arxiv.org/abs/2611.14463) | 该论文证明了交叉验证估计方差的精确有限样本恒等式，将方差表示为每个观测值在训练与评估两种角色中权重的函数，从而为应选择何种数据划分方案提供了理论指导。 |
| [^5] | [Nonparametric Maximum Likelihood Estimation for Mixtures of Logistic Regressions](https://arxiv.org/abs/2611.14445) | 本文针对混合分布未指定的逻辑回归混合模型，建立了筛法非参数极大似然估计的非渐近误差界与收敛速度理论，并提出了高效的有限粒子计算算法。 |
| [^6] | [Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation](https://arxiv.org/abs/2611.14328) | 该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。 |
| [^7] | [Compositional random forest](https://arxiv.org/abs/2611.14275) | 该论文提出了一种面向成分数据响应的随机森林框架，以对数似然增益作为分裂准则，并结合定义在不同表示空间上的分布，统一兼容了单纯形、对数比坐标和单位球面方向数据三种成分数据分析视角。 |
| [^8] | [On the Spectral Estimation of VARMA Processes](https://arxiv.org/abs/2611.13973) | 本文提出了一种基于传输熵概念的新矩问题，能够在不固定移动平均分量或不将其限制为标量的情况下，学习VARMA过程的功率谱密度。 |
| [^9] | [MEOW: Medical Open-World Interactive Segmentation](https://arxiv.org/abs/2611.13935) | MEOW是首个无需更新模型参数即可将非交互式分割骨干网络转变为交互式开放世界学习器的框架，通过结合无参数传导推理与记忆巩固，实现对未见类别的持续性能改进以及零样本跨模态迁移。 |
| [^10] | [Outliers and the Hard-Margin Linear SVM in High-Dimension, Low-Sample-Size Settings](https://arxiv.org/abs/2611.13869) | 该论文建立了高维小样本下离群点对硬间隔线性SVM影响的精确渐近理论：单侧离群点的全部效应可归结为一个有效距离并由此导出内点支持丧失、间隔收缩、截距位移和误分类的显式阈值，而两类对称污染仅收缩间隔、不改变极限判别函数。 |
| [^11] | [Beyond Interaction Scores: Selecting Feature Interactions via Independent Replication](https://arxiv.org/abs/2611.13765) | LENS 提出了一个模型无关的特征交互选择框架，通过在不相交数据上训练两个模型进行独立复现来验证交互模式的真实性，无论特征数量多少仅需两次模型拟合和一次多重检验校正。 |
| [^12] | [CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks](https://arxiv.org/abs/2611.13647) | 该论文提出CP-MLPs张量秩框架，将门控单元分解为检测、读取、写入三种角色，并严格证明：解绑门控块（如ReGLU/SwiGLU）在拟合不定二次交互和角色非对称目标时比绑定门控块（如ReLU²）具有严格的宽度效率优势，而绑定结构更匹配对称纯幂次目标，即解绑的收益取决于目标函数的结构。 |
| [^13] | [First-order asymptotic expansions for spectral convergence of normalized graph Laplacians constructed from samples on manifolds](https://arxiv.org/abs/2611.13623) | 该论文在不对特征值施加单重性假设的前提下，推导了高斯核构造的对称归一化图拉普拉斯算子特征值与特征向量的一阶渐近展开及余项集中不等式，从而将谱收敛理论推广到特征值可能具有非平凡重数的一般流形。 |
| [^14] | [Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design](https://arxiv.org/abs/2611.13565) | 本文系统研究了利用可观测中间响应的复合建模在多目标贝叶斯优化中对科学设计的收益，并探究了影响其效果差异的因素。 |
| [^15] | [Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding](https://arxiv.org/abs/2611.13545) | 该论文发现RL后训练中BF16因“权重粘滞”问题收敛慢于FP16，并提出在将FP32影子权重量化为BF16时采用随机舍入，从而在不增加与参考模型KL散度的前提下消除收敛速度差距。 |
| [^16] | [Simultaneous Near-Optimal Region Identification in Continuous Simulation Optimization: Minimax Complexity and Adaptive Sampling](https://arxiv.org/abs/2611.13489) | 本文证明在连续设计空间上同时识别所有容差水平下的近优区域并非优化的副产品，给出了极小极大采样复杂度下界并提出了达到该下界的自适应采样方法。 |
| [^17] | [Differentially Private Statistical Inference for Rankings from Pairwise Comparisons](https://arxiv.org/abs/2611.13485) | 该论文首次在Bradley-Terry模型框架下实现了边差分隐私下的排名统计推断，通过留一分析给出有限样本误差界、渐近有效置信区间以及结合岭偏差校正与高斯-拉普拉斯校准的偏差修正方法，适用于保护隐私的大语言模型成对比较评估。 |
| [^18] | [Sufficient Dimension Reduction with Multimodal Predictor Objects](https://arxiv.org/abs/2611.13460) | 本文提出了一种针对欧氏预测变量与一般度量空间中随机预测对象的联合充分降维新框架，通过序贯条件独立性、核嵌入与正则化包络模型实现两阶段降维，并建立了第一阶段子空间 root-$n$ 收敛速度与第二阶段子空间大样本一致性的理论保证。 |
| [^19] | [Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression](https://arxiv.org/abs/2611.13383) | 该论文通过建立依赖初始化信息的匹配上下风险界，解决了尾平均SGD分析中头部特征空间初始误差方差贡献的“预热间隔”问题，并证明现有上界和下界均不够紧致。 |
| [^20] | [VC-Dimension and Generalization in Geometric Machine Learning with Applications to Active Learning](https://arxiv.org/abs/2611.13337) | 该论文计算了黎曼几何模型空间中度量球分类器的VC维并给出了多种情形下的上下界，进而在数据分布假设下利用覆盖论证证明了泛化界，并将其应用于主动学习中的基于覆盖的算法。 |
| [^21] | [Time Series Change Point Detection with Gaussian Processes](https://arxiv.org/abs/2611.13295) | 该论文提出了一种基于高斯过程的通用变点检测框架，将变化定义为高斯过程先验超参数的改变，具有理论性能保证，且无需预先指定变化前后的时间序列函数形式，超参数可从数据中自动学习。 |
| [^22] | [Continuous Horizon Forecasting](https://arxiv.org/abs/2611.13228) | 本文提出连续视界（CH）预测策略，将预测重新定义为连续算子、把预测视界作为查询变量而非索引，在理论上证明了其相对于离散策略的估计方差优势，并通过大规模实验验证了其具有竞争力的精度，同时独特地支持并行推理和对未见视界的插值与外推。 |
| [^23] | [MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis](https://arxiv.org/abs/2611.13219) | MediPFN通过将预训练表格基础模型适配于干扰函数预测，并结合交叉拟合影响函数估计器与独立校准机制，为因果中介分析提供具有覆盖保证的置信区间。 |
| [^24] | [Prediction Inference for Hybrid Models on Realized Volatility Forecasting](https://arxiv.org/abs/2611.13212) | 该论文通过在 GARCH-LSTM 混合模型中引入 ARMA 条件均值结构，并提出一种结合经验残差自助抽样与共形预测校正的向前自助算法，为已实现波动率构建了覆盖率更高的多步预测区间，弥补了现有方法忽视点估计不确定性的缺陷。 |
| [^25] | [Near-Optimal Sample Complexity of Deterministic Agnostic PAC Learning](https://arxiv.org/abs/2611.13210) | 本文证明了VC维为 $d$、直径为 $K$ 的概念类在确定性标签下的不可知PAC学习样本复杂度为 $\widetilde{O}(K^{1/3} d^{2/3}/\varepsilon)$，并构造了对数因子内紧的下界，填补了此前上下界之间的差距。 |
| [^26] | [Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series](https://arxiv.org/abs/2611.12955) | ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。 |
| [^27] | [Generative Learning on Manifolds for Probabilistic Seismic Performance Assessment and Structural Identifiability](https://arxiv.org/abs/2611.12908) | 该研究提出一种结合扩散映射、分数扩散模型与几何调和的流形生成学习框架，可从有限的非线性仿真中学习结构参数与地震响应的联合分布，同时实现地震性能的正向概率预测与结构特性的反向辨识。 |
| [^28] | [Geometry and Stochastic Analysis of Flow and Diffusion Models: A Survey](https://arxiv.org/abs/2611.12107) | 这篇综述从三种几何（状态空间、概率测度空间、路径空间）与随机分析的统一视角阐明流与扩散模型的数学基础，指出所有得分恒等式本质上都是分部积分，并将Malliavin演算等工具贯穿于归一化流、流匹配与少步映射等方法之中。 |
| [^29] | [On Stochastic Continuous-Time Linear System Identification for Adaptive Stabilization](https://arxiv.org/abs/2611.02340) | 本文提出了基于系统辨识的自适应镇定的失败概率与轨迹长度和数量、维度、可控性Gramian矩阵及信噪比等所有相关量之间简单、实用且可靠的定量关系。 |
| [^30] | [Where Has the Foreign Signal Gone?](https://arxiv.org/abs/2610.28175) | 该研究发现Twitter情绪噪声对下一季度股票损失的预测关系在本国股票上比外国股票更加稳定，并用“生产者-知识错配假说”解释：外国股票的英语Twitter贡献者池缺乏拥有信息的母国市场投资者，导致其信息流中非基本面噪声成分更大。 |
| [^31] | [Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking](https://arxiv.org/abs/2610.09919) | 本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。 |
| [^32] | [A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning](https://arxiv.org/abs/2608.08743) | 本文提出一种基于分位数分布映射的数据预处理算法，通过序贯估计反事实状态和奖励，在强化学习中实现反事实公平性，并将常用的可加性假设作为特例涵盖其中。 |
| [^33] | [Total Variation Distance Estimation in Autoregressive Models](https://arxiv.org/abs/2607.19510) | 本文研究了在样本访问、logit 访问和含噪 logit 访问三种模型下估计两个长度为 $n$ 的自回归分布之间全变差距离的问题，将样本访问下的查询复杂度从已有的 $\widetilde{O}(n^3 m/\varepsilon^5)$ 改进为 $\widetilde{O}(n^2 K/\varepsilon^2)$，并给出了 logit 访问下紧的 $O(n/\varepsilon^2)$ 查询界。 |
| [^34] | [Disentangling Forced and Internal Climate Variability in Single Realizations using Dynamic Mode Decomposition with Control](https://arxiv.org/abs/2607.18298) | 该研究提出PullbackDMDc方法，基于非自治动力系统理论、带控制的动态模态分解与拉回吸引子估计，将外部强迫视为动力学驱动因子，从而实现从单一气候观测记录中分离强迫趋势与内部气候变率。 |
| [^35] | [Uniform-in-Time Weak Propagation-of-Chaos in Shallow Neural Networks](https://arxiv.org/abs/2605.22010) | 该论文通过利用平均场确定性 Wasserstein 梯度流动的收敛速率，而非依赖强凸性或对数 Sobolev 不等式，首次在浅层神经网络的梯度下降训练中建立了随时间一致成立的非渐近弱混沌传播结果。 |
| [^36] | [Variance Reduction for Gradients from Diffusion Teachers](https://arxiv.org/abs/2605.21489) | 提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。 |
| [^37] | [Density-Ratio Losses for Post-Hoc Learning to Defer](https://arxiv.org/abs/2605.19557) | 该论文提出用模型与专家理想分布之间的密度比来定义事后学习延后，推导出两种 DR CPE 损失用于训练延后评分器，其中联合变体在 KL 散度情形下与 Chow 规则及专家倾斜的贝叶斯后验相联系且计算便利。 |
| [^38] | [Factor Augmented High-Dimensional SGD](https://arxiv.org/abs/2605.19291) | 提出了一种完全基于流式数据的因子增强随机梯度下降方法（FSGD），并首次建立了将潜在因子估计误差纳入SGD分析的理论框架，为高维机器学习提供了可靠且可扩展的优化基础。 |
| [^39] | [Minimax Rates and Spectral Distillation for Tree Ensembles](https://arxiv.org/abs/2605.11841) | 本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。 |
| [^40] | [Posterior Contraction Rates for Sparse Kolmogorov-Arnold Networks in Anisotropic Besov Spaces](https://arxiv.org/abs/2605.11652) | 本文证明了配备尖峰-厚板稀疏先验的稀疏贝叶斯Kolmogorov-Arnold网络在各向异性Besov空间上可达到接近极小极大的后验收缩速率，能通过超先验自适应未知的光滑度，且网络深度可保持固定。 |
| [^41] | [Affine Tracing: A New Paradigm for Probabilistic Linear Solvers](https://arxiv.org/abs/2605.10566) | 论文证明贝叶斯概率线性求解器只是非平稳仿射概率迭代方法的特例，并提出“仿射追踪”框架，能够从仿射迭代方法的标准实现中自动构建概率迭代方法，从而推动其广泛实际应用。 |
| [^42] | [Minimax Optimal Estimation of Transport-Growth Pairs in Unbalanced Optimal Transport](https://arxiv.org/abs/2605.08705) | 本文针对具有二次代价和KL边际惩罚的非平衡最优传输，提出估计传输-增长对的最近邻估计器和基于核的估计器，并证明了核估计器在对数因子内达到最小极大最优速率。 |
| [^43] | [Gaussian mixture models in Hilbert spaces via kernel methods](https://arxiv.org/abs/2605.05996) | 该论文提出了一种基于核均值嵌入的希尔伯特空间数据高斯混合模型框架，用于对动态函数数据等无限维随机对象进行聚类分析，并从理论和实验上证明了其良定义性、稠密近似能力以及在多种数据几何形态上的有效性。 |
| [^44] | [Allocating Human Oversight in AI-Enabled Analytics](https://arxiv.org/abs/2604.12497) | 该论文提出了一种在任务难度未知情形下分配共享人类验证预算的上置信界策略，能够从人类标签与AI预测的配对数据中联合学习预测校正系数和任务难度，并具有分配遗憾理论保证，在实验中将性能提升至接近最优方案的水平。 |
| [^45] | [A unifying view of contrastive learning, importance sampling, and bridge sampling for energy-based models](https://arxiv.org/abs/2604.08116) | 本文提出了一个统一框架，将噪声对比估计、反向逻辑回归、多重重要性采样和桥接采样在基于能量的模型中联系起来，揭示了它们之间的等价关系，并促进了更高效估计量的开发。 |
| [^46] | [Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation](https://arxiv.org/abs/2603.23398) | 提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。 |
| [^47] | [The Sampling Complexity of Condorcet Winner Identification in Dueling Bandits](https://arxiv.org/abs/2603.15189) | 提出一种利用完整间隔矩阵的孔多塞胜者识别算法，其样本复杂度在所有情形下达到最优（上下界匹配），并揭示了采样复杂度由定位有信息比较的成本所刻画的权衡结构。 |
| [^48] | [Decentralized Ranking Aggregation via Gossip: Convergence and Robustness](https://arxiv.org/abs/2602.22847) | 本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。 |
| [^49] | [LoBoost: Model-Native Local Conformal Prediction for Gradient-Boosted Trees](https://arxiv.org/abs/2602.22432) | LoBoost通过重用梯度提升树集成模型自身的叶结构构建多尺度嵌套分组，实现模型原生的局部保形预测，无需辅助划分、重训练或额外数据分割即可局部估计残差分位数并适应异方差性。 |
| [^50] | [Low-degree lower bounds via almost orthonormal bases](https://arxiv.org/abs/2509.09353) | 本文提出了一种基于近似正交多项式基的更直接证明策略，用于在具有植入结构的随机图模型中建立低阶多项式下界，克服了传统规范正交多项式族方法在估计任务和复杂检验问题中的局限性。 |
| [^51] | [Do we really need the Rademacher complexities?](https://arxiv.org/abs/2502.15118) | 该论文证明只需利用函数类协方差结构的粗略信息，即可在重尾情形下推导出与高斯情形最先进速率相匹配的样本复杂度上界，从而表明学习问题无需依赖难以控制的局部化Rademacher复杂度。 |
| [^52] | [The Star Geometry of Critic-Based Regularizer Learning](https://arxiv.org/abs/2408.16852) | 该论文开创性地研究了在星形体的规范函数这一正则化器族上优化基于评论器的损失函数，为学习到的任务相关正则化器的结构及其与测量数据和真实数据分布之间的几何联系提供了首个理论视角。 |
| [^53] | [Scalable Bayesian inference for the generalized linear mixed model](https://arxiv.org/abs/2403.03007) | 该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。 |

# 详细

[^1]: TorchCurves：可微分参数曲线

    TorchCurves: Differentiable Parametric Curves

    [https://arxiv.org/abs/2611.14516](https://arxiv.org/abs/2611.14516)

    TorchCurves是一个将B样条和勒让德多项式参数曲线作为可学习基元的可微分PyTorch库，支持端到端优化，可广泛应用于连续特征嵌入、KAN网络和函数回归等任务。

    

    我们提出了TorchCurves，这是一个将向量值参数曲线作为可学习基元的PyTorch库。其核心操作将批量标量输入映射到以B样条或勒让德多项式基表示的曲线上的点，并支持对输入和系数进行微分。曲线系数可以与周围模型联合优化，或由另一个网络生成。该库提供了可训练模块、可复用的基函数模块以及函数式接口，支持从连续特征嵌入到Kolmogorov-Arnold网络（KAN）和函数回归等多种应用。其实现将经典曲线求值算法与向量化PyTorch操作相结合，在无需自定义硬件内核的情况下强调代码的可读性和可维护性。计时实验在不同问题规模下，将曲线求值与NumPy和SciPy进行了对比，并将KAN训练与EfficientKAN进行了对比。结果表明……（摘要原文截断）

    arXiv:2611.14516v1 Announce Type: new  Abstract: We present TorchCurves, a PyTorch library that treats vector-valued parametric curves as learnable primitives. Its central operation maps batches of scalar inputs to points on curves represented in B-spline or Legendre polynomial bases, with differentiation through both inputs and coefficients. Curve coefficients can be optimized jointly with the surrounding model or produced by another network. The library provides trainable modules, reusable basis modules, and a functional interface, supporting applications ranging from continuous feature embeddings to Kolmogorov-Arnold networks and functional regression. Its implementation combines classical evaluation algorithms with vectorized PyTorch operations, emphasizing readability and maintainability without custom hardware kernels. Timing experiments compare curve evaluation against NumPy and SciPy and KAN training against EfficientKAN across a range of problem sizes. The results illustrate w
    
[^2]: 后训练中策略优化的理论：覆盖坍缩与参考重置

    A Theory of Policy Optimization in Post-Training: Coverage Collapse and Reference Resets

    [https://arxiv.org/abs/2611.14467](https://arxiv.org/abs/2611.14467)

    该论文为语言模型后训练中的策略优化建立了理论框架，证明REINFORCE类算法（PPO和GRPO的基础）即使奖励提升也会发生“覆盖坍缩”、使最优响应概率趋近于零并导致优化停滞，而通过重置KL参考等手段可实现由基础模型响应覆盖度所约束的保覆盖近优学习。

    

    策略优化是语言模型后训练的核心，然而其原则性的理论指导仍然有限。我们重新审视其理论基础，旨在识别一类能够学习到近似最优策略的算法，其所需的总样本数和梯度步数由基础模型的响应覆盖度决定——即基础模型生成最优响应的容易程度。我们称这类算法为“保覆盖的”。这一保证刻画了通过从基础模型采样来放大其可及能力的目标。(1) 覆盖坍缩。作为PPO和GRPO基础的REINFORCE算法，即使在奖励提升的同时，也可能将最优响应的概率驱使到接近于零。这种覆盖坍缩会使优化陷入停滞，即便初始覆盖度良好，收敛也可能需要指数级多的步数。即使使用精确梯度，这种失败也会出现，并且在加入对基础模型的KL正则化后依然存在。(2) 参考重置。我们证明，通过反复重置KL参考……（摘要在此处被截断）

    arXiv:2611.14467v1 Announce Type: new  Abstract: Policy optimization is central to language model post-training, yet principled guidance remains limited. We revisit its theoretical foundations to identify algorithms that learn near-optimal policies using total samples and gradient steps governed by the base model's response coverage, a measure of how readily it generates optimal responses. We call such algorithms coverage-preserving. This guarantee captures the goal of amplifying capabilities accessible through sampling from the base model.   (1) Coverage collapse. REINFORCE, the basis for PPO and GRPO, can drive the probability of optimal responses to near zero, even as reward improves. This coverage collapse stalls optimization, and convergence can require exponentially many steps despite good initial coverage. The failure arises even with exact gradients and persists under KL regularization to the base model.   (2) Reference resets. We show that repeatedly resetting the KL reference
    
[^3]: 一条曲线，不同权重：AUC、Brier分数、决策加权AUC与策略Youden指数

    One Curve, Different Weights: AUC, the Brier Score, Decision-Weighted AUC, and the Policy Youden Index

    [https://arxiv.org/abs/2611.14465](https://arxiv.org/abs/2611.14465)

    论文揭示AUC、Brier分数与提出的决策加权AUC（AUC_π）本质上是同一条Youden指数曲线在不同权重下的加权平均，其中AUC_π以决策者的行动概率为权重，从而将模型评估与实际决策需求对齐。

    

    AUC和Brier分数是二元风险评分的标准汇总指标，但它们对模型的排序可能不一致。我们提出了一种加权AUC，记为AUC_π，它根据具有行动概率π的决策者对待每对观测值的差异程度来为各观测对加权。我们证明，这三者都建立在同一条曲线的加权平均之上——即跨截止点t的Youden指数 J(t) = TPR(t) − FPR(t)——而各种平均的区别仅在于它们赋予截止点的权重不同。每个指标都依据某个量跨越该截止点上升的幅度来为其加权：对AUC而言，是所有评分的分布；对Brier分数而言，是真实概率 E[Y|S]；对AUC_π而言，则是决策者的行动概率。在归一化意义下，AUC_π等价于理想预测的Brier技巧分数，只是其权重被转移到了决策者所关心的截止点上。我们还证明，用 |π(s) − π(s′)| 对观测对加权，在尺度变换意义下是唯一的成对加权方式。

    arXiv:2611.14465v1 Announce Type: cross  Abstract: AUC and the Brier score are standard summaries of a binary risk score, yet they can rank models differently. We introduce a weighted AUC, AUC$_\pi$, which weights each pair of observations by how differently a decision maker with probability of acting $\pi$ treats them. We show that all three are built on weighted averages of one curve, the Youden index $J(t) = \mathrm{TPR}(t) - \mathrm{FPR}(t)$ across cutoffs $t$, and that the averages differ in the weights they put on the cutoffs. Each weights a cutoff by how much something rises across it: for AUC, the distribution of all scores; for the Brier score, the true probability $\mathbb{E}[Y \mid S]$; for AUC$_\pi$, the decision maker's probability of acting. Up to normalization, AUC$_\pi$ is the Brier skill score of the ideal forecast with its weight moved to the cutoffs the decision maker cares about. We also show that weighting pairs by $|\pi(s) - \pi(s')|$ is, up to scale, the only pai
    
[^4]: 交叉验证方差理论

    A Theory of Cross-Validation Variance

    [https://arxiv.org/abs/2611.14463](https://arxiv.org/abs/2611.14463)

    该论文证明了交叉验证估计方差的精确有限样本恒等式，将方差表示为每个观测值在训练与评估两种角色中权重的函数，从而为应选择何种数据划分方案提供了理论指导。

    

    交叉验证通过反复将数据划分为训练集和验证集来估计预测性能，在训练集上拟合模型并在验证集上进行评估。每个观测值以两种角色影响估计结果：当它被留出时通过其自身的评估产生影响，当它作为训练数据时通过其帮助拟合的模型产生影响。随机划分决定了它在每种角色中的权重。基于这一表示，我们证明了交叉验证估计方差的精确有限样本恒等式。对于具有相同训练集大小的给定划分方案，方差是这些权重的函数，并附带取决于模型、度量和总体的常数。该理论涵盖了在观测值上取平均的度量（如均方误差），以及在病例-对照对上取平均的度量（如AUC）。该恒等式指明了应采用哪些划分方案。方差可分解为仅取决于训练集大小的抽样方差，以及由（摘要在此处截断）……

    arXiv:2611.14463v1 Announce Type: cross  Abstract: Cross-validation estimates predictive performance by repeatedly splitting the data into training and validation sets, fitting the model on the first and evaluating it on the second. Every observation influences the estimate in two roles, through its own evaluation when it is held out and through the models it helps fit when it trains. The random splits determine its weight in each role. From this representation, we prove an exact finite-sample identity for the variance of the cross-validated estimate. For a given splitting scheme with a common training set size, the variance is a function of these weights, with constants of the model, the metric, and the population. It covers metrics averaged over observations, such as mean squared error, and over case--control pairs, such as the AUC. The identity shows which schemes to use. The variance decomposes into sampling variance, which depends only on the training size, and variance due to the
    
[^5]: 逻辑回归混合模型的非参数极大似然估计

    Nonparametric Maximum Likelihood Estimation for Mixtures of Logistic Regressions

    [https://arxiv.org/abs/2611.14445](https://arxiv.org/abs/2611.14445)

    本文针对混合分布未指定的逻辑回归混合模型，建立了筛法非参数极大似然估计的非渐近误差界与收敛速度理论，并提出了高效的有限粒子计算算法。

    

    逻辑回归混合模型为二元响应建模提供了一个灵活的框架，它在保留可解释回归结构的同时，扩展了标准逻辑回归以适应未观测到的异质性。本文研究了混合分布未指定的此类模型，我们分析了筛法非参数极大似然估计量，并在固定设计和随机设计两种情形下推导了所得混合回归函数估计误差的非渐近界。我们的结果揭示了统计估计误差与筛半径引起的近似误差之间的明确权衡，并由此在混合分布的若干尾部条件下推导出收敛速度。在计算方面，我们提出了一种有限粒子算法，通过优化有限支撑混合分布的位置和权重来近似筛法NPMLE，该方法在数值上具有优势。

    arXiv:2611.14445v1 Announce Type: cross  Abstract: Mixtures of logistic regressions provide a flexible framework for modeling binary responses, extending standard logistic regression to accommodate unobserved heterogeneity while retaining an interpretable regression structure. In this paper, we study such models with an unspecified mixing distribution. We analyze a sieve nonparametric maximum likelihood estimator (NPMLE) and derive nonasymptotic bounds on the estimation error of the resulting mixture regression function under both fixed and random designs. Our results reveal an explicit tradeoff between statistical estimation error and the approximation error induced by the sieve radius, from which we derive convergence rates under several tail conditions on the mixing distribution. For computation, we propose a finite-particle algorithm that approximates the sieve NPMLE by optimizing the locations and weights of a finitely supported mixing distribution. The method offers numerical adv
    
[^6]: 前向值离散化：无需高阶近似的更快随机扩散采样

    Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation

    [https://arxiv.org/abs/2611.14328](https://arxiv.org/abs/2611.14328)

    该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。

    

    随机扩散采样器通常需要比确定性采样器多得多的神经函数评估次数（NFE）才能达到相当的样本质量。尽管高阶离散化能够加速确定性采样，但其对随机采样的益处尚不明确。我们建立了匹配的上界与最坏情况下的下界，表明即使使用精确的分数函数，采用高阶近似的随机采样器仍然只具有一阶收敛性。因此，提高近似阶数通常并不能改善收敛阶数。受此局限性的启发，我们提出了前向值离散化，这是一种免训练的框架，它在由先前预测构建的前瞻估计处评估数据/噪声预测模型。对于确定性采样，由此得到的 Forward DPMSolver（F-DPMSolver）通过其基于 DDIM 的变体，在低 NFE 预算下提升了样本质量，尽管仍保持一阶……

    arXiv:2611.14328v1 Announce Type: cross  Abstract: Stochastic diffusion samplers often require substantially more neural function evaluations (NFEs) than deterministic samplers to achieve comparable sample quality. Although higher-order discretization accelerates deterministic sampling, its benefits for stochastic sampling remains unclear. We establish matching upper and worst-case lower bounds showing that stochastic samplers with higher-order approximations remain first-order convergent, even with exact score functions. Thus, increasing the approximation order does not generally improve the convergence order. Motivated by this limitation, we introduce forward-value discretization, a training-free framework that evaluates the data/noise prediction model at a lookahead estimate constructed from previous predictions. For deterministic sampling, the resulting Forward DPMSolver (F-DPMSolver), improves sample quality at low NFE budgets through its DDIM-based variant, despite remaining firs
    
[^7]: 成分数据随机森林

    Compositional random forest

    [https://arxiv.org/abs/2611.14275](https://arxiv.org/abs/2611.14275)

    该论文提出了一种面向成分数据响应的随机森林框架，以对数似然增益作为分裂准则，并结合定义在不同表示空间上的分布，统一兼容了单纯形、对数比坐标和单位球面方向数据三种成分数据分析视角。

    

    成分数据在文献中主要从三个角度进行分析：直接在单纯形上分析、在对数比坐标下分析，以及作为单位球面上的方向数据进行分析。本文构建了一个适用于成分响应的随机森林框架，能够兼容上述全部三种表示方式。随机森林是决策树的集成，分裂准则决定了每棵树的生长方式。我们采用对数似然增益作为分裂准则，并考虑定义在成分数据可表示的各种空间上的分布。Dirichlet似然在文献中已有研究，但其协方差在仅相差一个标量的意义上由均值决定，且其最大似然估计没有闭式解。为了获得更灵活的准则，我们使用了两种著名的变换。首先，平方根变换将成分映射到单位球面上，而现有的方向分布需要数值估计……

    arXiv:2611.14275v1 Announce Type: cross  Abstract: Compositional data have been analysed in the literature from three main points of view: directly on the simplex, in log-ratio coordinates, and as directional data on the unit sphere. In this paper, we construct a random forest framework for compositional responses that accommodates all three. Random forests are ensembles of decision trees, and the split criterion determines how each tree grows. We use the log-likelihood gain as the split criterion and consider distributions defined on the different spaces in which compositions can be represented. The Dirichlet likelihood has been studied in the literature, but its covariance is fixed by its mean up to a single scalar, and its maximum likelihood estimator has no closed form. To obtain more flexible criteria, we use two well-known transformations. First, the square-root transformation maps compositions onto the unit sphere, where existing directional distributions require numerical estim
    
[^8]: 论VARMA过程的谱估计

    On the Spectral Estimation of VARMA Processes

    [https://arxiv.org/abs/2611.13973](https://arxiv.org/abs/2611.13973)

    本文提出了一种基于传输熵概念的新矩问题，能够在不固定移动平均分量或不将其限制为标量的情况下，学习VARMA过程的功率谱密度。

    

    我们考虑估计向量自回归移动平均（VARMA）过程的功率谱密度的问题。完成这一任务的一种成熟方法是构建矩问题，在匹配底层过程的某些矩的同时，选择使熵类指标最大化的谱密度。然而，现有方法假设移动平均（MA）分量是固定的或标量的，从而缩小了寻求最优谱密度的ARMA模型类别。在本文中，我们基于传输熵的概念，提出了一种新的矩问题，用于学习VARMA过程的谱密度，而无需固定MA分量或将其限制为标量。

    arXiv:2611.13973v1 Announce Type: cross  Abstract: We consider the problem of estimating the power spectral density of a Vector-valued Autoregressive Moving-Average (VARMA) process. A well-established approach to accomplishing this task consists of formulating a moment problem that selects the spectral density maximizing an entropy-like index while matching certain moments of the underlying process. However, existing methods assume that the moving-average (MA) component is either fixed or scalar, thereby reducing the class of ARMA models over which the optimal spectral density is sought. In this paper, we propose a new moment problem, based on the notion of transportation entropy, for learning the spectral density of a VARMA process without fixing the MA component or restricting it to be scalar.
    
[^9]: MEOW：医学开放世界交互式分割

    MEOW: Medical Open-World Interactive Segmentation

    [https://arxiv.org/abs/2611.13935](https://arxiv.org/abs/2611.13935)

    MEOW是首个无需更新模型参数即可将非交互式分割骨干网络转变为交互式开放世界学习器的框架，通过结合无参数传导推理与记忆巩固，实现对未见类别的持续性能改进以及零样本跨模态迁移。

    

    在开放世界识别中，机器学习模型不仅需要对未知结构进行预测（“广义开放集”），还要随着时间推移从这些预测中持续学习并不断改进。交互式分割尽管能够获取用户反馈，但迄今为止仍局限于开放集设定。在这项工作中，我们提出了MEOW，这是首个能够在不更新可学习模型参数的情况下，将非交互式分割骨干网络转变为交互式开放世界学习器的框架。MEOW将无参数的传导推理与记忆巩固相结合，以积累知识并从而提升未来性能。聚焦于医学数据，我们展示了MEOW在训练中从未见过的类别上能够持续改进自身的预测，并可零样本迁移到陌生领域，包括micro-CT数据和其他成像模态。

    arXiv:2611.13935v1 Announce Type: cross  Abstract: In open-world recognition, machine learning models not only make predictions on unknown structures ("generalized open-set"), but also continually learn and improve from those predictions over time. Interactive segmentation, despite its access to user feedback, has so far remained confined to the open-set setting. In this work we introduce MEOW, the first framework that turns non-interactive segmentation backbones into interactive open-world learners without updating learnable model parameters. MEOW combines parameter-free transductive reasoning with memory consolidation to accumulate knowledge and thereby improve future performance. Focusing on medical data, we show that MEOW consistently improves its own predictions on classes never seen during training, and transfers zero-shot to unfamiliar domains, including micro-CT data and other imaging modalities.
    
[^10]: 高维小样本设置中的离群点与硬间隔线性支持向量机

    Outliers and the Hard-Margin Linear SVM in High-Dimension, Low-Sample-Size Settings

    [https://arxiv.org/abs/2611.13869](https://arxiv.org/abs/2611.13869)

    该论文建立了高维小样本下离群点对硬间隔线性SVM影响的精确渐近理论：单侧离群点的全部效应可归结为一个有效距离并由此导出内点支持丧失、间隔收缩、截距位移和误分类的显式阈值，而两类对称污染仅收缩间隔、不改变极限判别函数。

    

    我们研究了当维度增长而样本量保持固定时，离群点如何影响硬间隔线性支持向量机。对于单侧污染情形，中心化Gram矩阵收敛为一个常数内点块加上一个离群点对角分量，极限对偶问题具有闭式解，所有离群点均通过一个有效距离发挥作用。这一表示给出了内点支持丧失、间隔收缩、截距位移以及渐近误分类的显式阈值。随后我们研究两类数据均受污染的情形：极限截距是类别平衡约束的乘子，两个类别会将其向相反方向拉动。因此，至多只有一个类别会发生崩溃或渐近误分类；而对称污染虽然会使间隔收缩，却使极限判别函数保持不变。七个数值实验验证了对偶分配、相位……（原文摘要在此处截断）

    arXiv:2611.13869v1 Announce Type: cross  Abstract: We study how outliers affect the hard-margin linear support vector machine when the dimension grows while the sample sizes remain fixed. For one-sided contamination, the centered Gram matrix converges to a constant inlier block plus an outlier-diagonal component, and the limiting dual problem has a closed-form solution. All outliers act through one effective distance. This representation yields explicit thresholds for loss of inlier support, margin shrinkage. intercept displacement, and asymptotic misclassification. We then treat contamination in both classes. The limiting intercept is the multiplier of the class-balance constraint, and the two classes pull it in opposite directions. Consequently, at most one class can collapse or be asymptotically misclassified, whereas symmetric contamination leaves the limiting discriminant function unchanged even though the margin shrinks. Seven numerical experiments verify the dual allocation, pha
    
[^11]: 超越交互分数：通过独立复现选择特征交互

    Beyond Interaction Scores: Selecting Feature Interactions via Independent Replication

    [https://arxiv.org/abs/2611.13765](https://arxiv.org/abs/2611.13765)

    LENS 提出了一个模型无关的特征交互选择框架，通过在不相交数据上训练两个模型进行独立复现来验证交互模式的真实性，无论特征数量多少仅需两次模型拟合和一次多重检验校正。

    

    学习特征交互对于准确预测和可解释建模至关重要，然而高的交互分数可能反映的是对噪声的拟合，而非可复现的模式。排序方法对特征对进行打分，但将阈值的选择留给用户；检验方法虽然设定了阈值，但依赖于反复的模型重拟合或对联合特征分布建模。我们提出了 LENS，一个通过独立复现来选择特征交互的模型无关框架。LENS 使用相同的学习算法在不相交的数据上训练两个模型，并且仅通过预测来访问它们。对于每个特征对，LENS 从联合效应面中移除主效应以分离出交互模式，按该模式的强度对特征对进行排序，并检验第二个模型是否能以有意义的强度复现该模式。只需一次多重检验校正即可确定保留哪些特征对，且无论特征数量多少，仅需两次模型拟合。

    arXiv:2611.13765v1 Announce Type: new  Abstract: Learning feature interactions is essential for accurate prediction and interpretable modeling, yet a high interaction score may reflect fitting noise rather than a reproducible pattern. Ranking methods score feature pairs but leave the cutoff to the user; testing methods set the cutoff but rely on repeated model refits or a model of the joint feature distribution. We present LENS, a model-agnostic framework for selecting feature interactions through independent replication. LENS trains two models with the same learning algorithm on disjoint data and accesses them only through predictions. For each feature pair, it removes main effects from the joint-effect surface to isolate an interaction pattern, ranks pairs by its strength, and tests whether the second model reproduces it at a meaningful strength. A single multiple-testing correction determines which pairs to retain, using only two model fits regardless of the number of features. We p
    
[^12]: CP-MLPs：绑定与解绑MLP块的张量秩理论

    CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks

    [https://arxiv.org/abs/2611.13647](https://arxiv.org/abs/2611.13647)

    该论文提出CP-MLPs张量秩框架，将门控单元分解为检测、读取、写入三种角色，并严格证明：解绑门控块（如ReGLU/SwiGLU）在拟合不定二次交互和角色非对称目标时比绑定门控块（如ReLU²）具有严格的宽度效率优势，而绑定结构更匹配对称纯幂次目标，即解绑的收益取决于目标函数的结构。

    

    Transformer前馈块结合了多种设计选择：平方激活、门控机制，以及绑定与解绑的参数共享方式，而这些选择对表达能力的各自贡献仍知之甚少。我们提出了CP-MLPs，这是一个张量框架，其中每个乘法单元贡献一个秩-1交互，并承担三种角色：检测输入特征、读取数值以及写入输出。绑定的门控块（如ReLU$^2$）迫使检测器角色与数值角色共享同一方向；解绑的门控块（如ReGLU和SwiGLU）则允许二者不同。我们证明这一区分带来了精确的、按次数划分的张量秩分离：对于不定二次交互和角色非对称的单项目标，解绑的检测-数值单元在宽度效率上严格更优；而绑定单元在结构上则与对称纯幂次目标相匹配。因此，解绑所带来的宽度优势取决于目标函数的结构。实验表明损失……

    arXiv:2611.13647v1 Announce Type: new  Abstract: Transformer feed-forward blocks combine several design choices: squared activations, gating, and tied versus untied parameter sharing, whose separate contributions to expressivity remain poorly understood. We introduce CP-MLPs, a tensor framework in which each multiplicative unit contributes a rank-1 interaction with three roles: detecting an input feature, reading a value, and writing to the output. Tied gated blocks such as ReLU$^2$ force the detector and value roles to share the same direction; untied gated blocks such as ReGLU and SwiGLU let them differ. We show that this distinction yields exact, degree-wise tensor-rank separations: untied detector-value units are strictly more width-efficient for indefinite quadratic interactions and role-asymmetric monomials, while tied units are structurally matched to symmetric pure-power targets. The width advantage of untying therefore depends on the target structure. Experiments show loss pla
    
[^13]: 流形上采样构造的归一化图拉普拉斯算子谱收敛的一阶渐近展开

    First-order asymptotic expansions for spectral convergence of normalized graph Laplacians constructed from samples on manifolds

    [https://arxiv.org/abs/2611.13623](https://arxiv.org/abs/2611.13623)

    该论文在不对特征值施加单重性假设的前提下，推导了高斯核构造的对称归一化图拉普拉斯算子特征值与特征向量的一阶渐近展开及余项集中不等式，从而将谱收敛理论推广到特征值可能具有非平凡重数的一般流形。

    

    在从嵌入欧氏空间的闭黎曼流形上进行独立同分布均匀采样的条件下，我们推导了通过高斯核构造的对称归一化图拉普拉斯算子的特征值和特征向量的一阶渐近展开，并给出了余项的集中不等式。作为直接推论，我们建立了相应的弱收敛结果。与大多数相关工作不同，我们的分析不对特征值施加单重性假设，从而将该框架的适用性扩展到具有潜在非平凡特征值重数的一般流形。

    arXiv:2611.13623v1 Announce Type: cross  Abstract: Under i.i.d. uniform sampling from a closed Riemannian manifold embedded in Euclidean space, we derive first-order asymptotic expansions for the eigenvalues and eigenvectors of symmetrically normalized graph Laplacians constructed via Gaussian kernels and provide concentration inequalities for the remainder terms. As a direct consequence, we establish the corresponding weak convergence results. Unlike most related works, our analysis does not impose a multiplicity-one assumption on the eigenvalues, thereby extending the applicability of the framework to general manifolds with potentially nontrivial eigenvalue multiplicities.
    
[^14]: 面向科学设计的多目标贝叶斯优化中复合建模的研究

    Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design

    [https://arxiv.org/abs/2611.13565](https://arxiv.org/abs/2611.13565)

    本文系统研究了利用可观测中间响应的复合建模在多目标贝叶斯优化中对科学设计的收益，并探究了影响其效果差异的因素。

    

    科学设计通常需要在相互冲突的目标之间进行权衡，而这些目标通过实验或仿真进行评估的代价十分高昂。多目标贝叶斯优化（MOBO）通过使用概率模型来指导有前景设计的选择，从而应对这一挑战。在许多应用中，可调输入会映射到可观测的中间响应，随后通过已知函数将这些中间响应组合起来以计算最终目标。先前关于复合函数贝叶斯优化的研究表明，从这些中间响应中学习（我们称之为复合建模），与仅依赖目标函数黑盒观测的方法相比，能够显著提高单目标优化中的采样效率。然而，复合建模在MOBO中的收益仍缺乏充分研究。在本工作中，我们研究了复合建模在MOBO中的收益，并探究了导致其表现变化的因素。

    arXiv:2611.13565v1 Announce Type: cross  Abstract: Scientific design often requires balancing competing objectives that are expensive to evaluate through experiments or simulations. Multi-objective Bayesian optimization (MOBO) addresses this challenge by using probabilistic models to guide the selection of promising designs. In many applications, tunable inputs map to observable intermediate responses, which are then combined through known functions to compute the final objectives. Prior work on Bayesian optimization of composite functions shows that learning from these intermediate responses, which we call Composite modeling, can substantially improve sampling efficiency in single-objective optimization, compared to methods that condition only on black-box observations of the objective function. The benefits of Composite modeling in MOBO, however, remain understudied. In this work, we study the benefits of Composite modeling in MOBO and investigate the factors driving variation in its
    
[^15]: 用随机舍入在强化学习后训练中缩小BF16与FP16的收敛速度差距

    Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding

    [https://arxiv.org/abs/2611.13545](https://arxiv.org/abs/2611.13545)

    该论文发现RL后训练中BF16因“权重粘滞”问题收敛慢于FP16，并提出在将FP32影子权重量化为BF16时采用随机舍入，从而在不增加与参考模型KL散度的前提下消除收敛速度差距。

    

    BF16因具有与FP32相同的动态范围、且收敛效果与FP32训练相近，被广泛用于大语言模型的预训练。然而，在采用可验证奖励的强化学习（RL）中，学习率通常较低，我们发现标准混合精度BF16训练在相同学习率下的收敛速度比FP16更慢。我们将这一差距归因于“权重粘滞”（weight sticking）现象：当使用就近舍入（round-to-nearest）方式将FP32影子权重量化为BF16时，较小的更新往往不会改变用于rollout生成和训练的BF16权重。因此，可能需要多次优化器迭代后，影子权重的更新才能体现在BF16权重中。提高BF16学习率可以减少权重粘滞并达到与FP16相当的收敛速度，但同时也会增加与参考模型之间的KL散度。我们转而在将FP32影子权重量化为BF16时应用随机舍入（stochastic rounding）……

    arXiv:2611.13545v1 Announce Type: new  Abstract: BF16 is widely used to pretrain large language models because it has the same dynamic range as FP32 and converges similarly to FP32 training. However, in reinforcement learning with verifiable rewards, where learning rates are typically low, we find that standard mixed-precision BF16 training converges more slowly than FP16 training at the same learning rate. We attribute this gap to weight sticking: when FP32 shadow weights are quantized to BF16 using round-to-nearest, small updates often leave the BF16 weights used for rollout generation and training unchanged. Consequently, several optimizer steps may be needed before updates to the shadow weights are reflected in the BF16 weights. Raising the BF16 learning rate reduces weight sticking and matches FP16's convergence speed, but it also increases the KL divergence from the reference model. We instead apply stochastic rounding when quantizing the FP32 shadow weights to BF16. This approac
    
[^16]: 连续仿真优化中的同时近优区域识别：极小极大复杂度与自适应采样

    Simultaneous Near-Optimal Region Identification in Continuous Simulation Optimization: Minimax Complexity and Adaptive Sampling

    [https://arxiv.org/abs/2611.13489](https://arxiv.org/abs/2611.13489)

    本文证明在连续设计空间上同时识别所有容差水平下的近优区域并非优化的副产品，给出了极小极大采样复杂度下界并提出了达到该下界的自适应采样方法。

    

    仿真优化通常只返回单一估计的最优点。当决策还依赖于独立于仿真进行评估的其他准则（如资源成本）时，决策者需要的是其期望性能落在最优值容差范围内的设计集合，而且往往只有在看到权衡关系之后才能确定容差。我们研究如何在连续设计空间上识别这些近优区域，对所有不超过规定最大值 $T$ 的容差同时进行识别，并将区域估计误差控制在 $\eta$ 以内。我们证明，近优区域识别并非优化的副产品。具体而言，当 $\alpha>\nu$ 时，任何算法，即使事先已知最优设计，也至少需要 $\Omega\bigl(\eta^{-(2+d/\nu)}\min\{1,KT^{d/\alpha}\}\bigr)$ 次仿真评估，其中 $d$ 为维度，$\nu$ 为光滑度，$(K,\alpha)$ 为刻画近优区域几何形状的覆盖参数。

    arXiv:2611.13489v1 Announce Type: cross  Abstract: Simulation optimization typically returns a single estimated optimal point. When decisions also depend on criteria evaluated independently of simulation, such as resource cost, the decision maker instead needs the set of designs whose expected performance lies within a tolerance of the optimum, often choosing the tolerance only after seeing the tradeoffs. We study how to identify these near-optimal regions on a continuous design space, simultaneously for all tolerances up to a prescribed maximum $T$, with region estimation error at most $\eta$. We show that region identification is not a by-product of optimization. Specifically, when $\alpha>\nu$, any procedure, even one given the optimal design in advance, needs at least $\Omega\bigl(\eta^{-(2+d/\nu)}\min\{1,KT^{d/\alpha}\}\bigr)$ simulation evaluations, where $d$ is the dimension, $\nu$ the smoothness, and $(K,\alpha)$ the covering parameters that describe the geometry of the near-op
    
[^17]: 基于成对比较的排名的差分隐私统计推断

    Differentially Private Statistical Inference for Rankings from Pairwise Comparisons

    [https://arxiv.org/abs/2611.13485](https://arxiv.org/abs/2611.13485)

    该论文首次在Bradley-Terry模型框架下实现了边差分隐私下的排名统计推断，通过留一分析给出有限样本误差界、渐近有效置信区间以及结合岭偏差校正与高斯-拉普拉斯校准的偏差修正方法，适用于保护隐私的大语言模型成对比较评估。

    

    基于人类成对比较的大语言模型排名的可靠推断必须保护个体偏好记录的隐私。为满足这一需求，我们在Bradley-Terry模型下开发了具有边差分隐私的分数与排名推断方法。针对目标扰动估计器，一种新颖的留一分析建立了欧几里得范数和逐元素估计误差的有限样本界。对于足够稠密的比较图，这些误差界可延伸至更小的隐私预算，包括隐私项主导统计项的情形。一致渐近展开为个体分数以及成对分数差异提供了渐近有效的置信区间。为改善有限样本性能，我们通过将即插即用的岭回归偏差校正与高斯-拉普拉斯校准相结合，同时考虑了正则化偏差和隐私引入的随机性。我们进一步建立了自助法（摘要在此处被截断）。

    arXiv:2611.13485v1 Announce Type: cross  Abstract: Reliable inference for rankings of large language models based on human pairwise comparisons must preserve the privacy of individual preference records. To meet this need, we develop methods for score and rank inference under the Bradley-Terry model with edge differential privacy. For an objective-perturbed estimator, a new leave-one-out analysis establishes finite-sample bounds on Euclidean and entrywise estimation errors. For sufficiently dense comparison graphs, the error bounds extend to smaller privacy budgets, including regimes in which its privacy term dominates its statistical term. A uniform asymptotic expansion yields asymptotically valid confidence intervals for individual scores and pairwise score differences. To improve finite-sample performance, we account for both regularization bias and privacy-induced randomness by combining a plug-in ridge-bias correction with Gaussian-Laplace calibration. We further establish a boots
    
[^18]: 基于多模态预测对象的充分降维

    Sufficient Dimension Reduction with Multimodal Predictor Objects

    [https://arxiv.org/abs/2611.13460](https://arxiv.org/abs/2611.13460)

    本文提出了一种针对欧氏预测变量与一般度量空间中随机预测对象的联合充分降维新框架，通过序贯条件独立性、核嵌入与正则化包络模型实现两阶段降维，并建立了第一阶段子空间 root-$n$ 收敛速度与第二阶段子空间大样本一致性的理论保证。

    

    我们研究了标量响应变量 $Y$ 对欧氏预测变量 $\mathbf{X} \in \mathbb{R}^p$ 以及处于一般度量空间中的随机预测对象 $\mathcal{Z}$ 在任意结构依赖下的回归问题。我们通过序贯条件独立性框架引入了联合充分降维 $(\mathbf{B}^\top\mathbf{X}, \mathcal{G}_Z)$，这与 $Y \mid \mathbf{X}$ 的普通中心子空间存在本质区别。在方法论上，$\mathcal{Z}$ 被嵌入到再生核希尔伯特空间中，并通过正则化核-响应包络模型提取特征映射的有限维降维。随后，部分自适应估计在给定第一阶段坐标的条件下恢复欧氏降维。我们建立了在收缩正则化参数下第一阶段包络子空间的 root-$n$ 收敛速度，以及第二阶段欧氏子空间的大样本一致性。

    arXiv:2611.13460v1 Announce Type: cross  Abstract: We study the regression of a scalar response $Y$ on a Euclidean predictor $\mathbf{X} \in \mathbb{R}^p$ and a random predictor object $\mathcal{Z}$ in a general metric space under arbitrary structural dependence. We introduce a joint sufficient dimension reduction $(\mathbf{B}^\top\mathbf{X}, \mathcal{G}_Z)$ constructed via a sequential conditional independence framework, which differs fundamentally from the ordinary central subspace of $Y \mid \mathbf{X}$. Methodologically, $\mathcal{Z}$ is embedded into a reproducing kernel Hilbert space, and a regularized kernel-response envelope model extracts a finite-dimensional reduction of the feature map. Partial adaptive estimation then recovers the Euclidean reduction conditional on the first-stage coordinates. We establish root-$n$ consistency rates for the first-stage envelope subspace under a shrinking regularizer, alongside large-sample consistency for the second-stage Euclidean subspace
    
[^19]: 解决预热间隔问题：线性回归中尾平均SGD的尖锐风险界

    Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression

    [https://arxiv.org/abs/2611.13383](https://arxiv.org/abs/2611.13383)

    该论文通过建立依赖初始化信息的匹配上下风险界，解决了尾平均SGD分析中头部特征空间初始误差方差贡献的“预热间隔”问题，并证明现有上界和下界均不够紧致。

    

    我们研究了恒定步长在线随机梯度下降（SGD）结合尾平均方法在高维线性回归中的超额风险。基于Zou等人（2021）提出的实例相关谱框架，当轨迹谱反馈条件成立时，我们建立了尾平均SGD总超额风险的匹配的、初始化显式的上下界，这些界由总体协方差谱 $\mathbf{H}$ 和初始化误差 $\mathbf{w}_0-\mathbf{w}^*$ 刻画。现有分析在有效方差方面存在一个根本性的差距：Zou等人（2021）的上界在头部特征空间 $\mathbf{I}_{0:k^\dagger}$ 中保留了来自初始误差的未衰减贡献，而他们的下界却完全忽略了头部初始化所诱导的方差。我们证明现有的上界和下界都是宽松的：来自头部初始化的单跳快模式方差贡献……

    arXiv:2611.13383v1 Announce Type: new  Abstract: We study the excess risk of constant-stepsize online stochastic gradient descent (SGD) with tail-averaging for high-dimensional linear regression. Building on the instance-dependent spectral framework of Zou et al. (2021), we establish matching initialization-explicit upper and lower bounds on the total excess risk of tail-averaged SGD in terms of the population covariance spectrum $\mathbf{H}$ and initialization $\mathbf{w}_0-\mathbf{w}^*$ whenever a trajectory spectral feedback condition holds. Existing analyses exhibit a fundamental gap in the effective variance: the upper bound of Zou et al. (2021) retains an undamped contribution from the initial error in the head eigenspace $\mathbf{I}_{0:k^\dagger}$, whereas their lower bound omits the variance induced by the head initialization altogether. We show that both the existing upper and lower bounds are loose: the single-hop fast-mode variance contribution from the head initialization i
    
[^20]: 几何机器学习中的VC维与泛化及其在主动学习中的应用

    VC-Dimension and Generalization in Geometric Machine Learning with Applications to Active Learning

    [https://arxiv.org/abs/2611.13337](https://arxiv.org/abs/2611.13337)

    该论文计算了黎曼几何模型空间中度量球分类器的VC维并给出了多种情形下的上下界，进而在数据分布假设下利用覆盖论证证明了泛化界，并将其应用于主动学习中的基于覆盖的算法。

    

    我们计算了黎曼几何模型空间（欧几里得空间、球面、双曲空间）中度量球分类器的VC维，并在多种其他情形下给出了度量球VC维的上界与下界，包括一般的赋范向量空间（VC维可能为无穷大）、具有 $\ell^p$ 范数的向量空间（VC维至少随维度线性增长，当 $p=1$ 时增长速度快于欧几里得情形）、平坦环面，以及一般的黎曼流形（在共形变换后VC维可以任意大）。在VC维很大甚至为无穷的若干情形下，我们利用覆盖论证，在附加的数据分布假设下证明了泛化界。无论是在VC维还是覆盖的设定中，我们都考虑了位于高维欧几里得空间低维子集上的数据。作为应用，我们研究了主动学习中基于覆盖的算法。

    arXiv:2611.13337v1 Announce Type: cross  Abstract: We compute the VC-dimension of metric ball classifiers in the model spaces of Riemannian geometry (Euclidean, spherical, hyperbolic) and present upper and lower bounds on the VC-dimension of metric balls in several other situations, including general normed vector spaces (potentially infinite VC-dimension), vector spaces with $\ell^p$-norms (VC-dimension grows at least linearly in the dimension, faster than Euclidean for $p=1$), flat tori, and general Riemannian manifolds (arbitrarily large VC-dimension after a conformal change). In several situations where the VC-dimension is large or even infinite, we prove generalization bounds under additional assumptions on the data distribution using covering arguments. Both in the VC-dimension and covering setting, we consider data on low-dimensional subsets of high-dimensional Euclidean spaces. As an application, we consider coverage-based algorithms in active learning.
    
[^21]: 基于高斯过程的时间序列变点检测

    Time Series Change Point Detection with Gaussian Processes

    [https://arxiv.org/abs/2611.13295](https://arxiv.org/abs/2611.13295)

    该论文提出了一种基于高斯过程的通用变点检测框架，将变化定义为高斯过程先验超参数的改变，具有理论性能保证，且无需预先指定变化前后的时间序列函数形式，超参数可从数据中自动学习。

    

    本文提出了一种新颖的通用高斯过程框架，用于时间序列中的变点检测。在该框架中，变化被定义为高斯过程先验超参数的改变。我们将该问题置于统计假设检验的背景下进行建模，并展示了该方法在离线和在线多变点检测任务中的性能。该方法为其性能提供了理论保证。所开发框架的一个重要优势在于，它无需在变化发生前对时间序列函数进行详细设定，超参数的值可以从数据中学习得到。本文在合成数据和真实数据上对该框架进行了全面评估，并验证了其有效性。

    arXiv:2611.13295v1 Announce Type: cross  Abstract: This paper proposes a novel general Gaussian process framework for change point detection in time series. A change is defined as an alteration of the hyperparameters of the Gaussian process prior. We formulate the problem in the context of statistical hypothesis testing and demonstrate its performance over both offline and online multiple change points detection. The approach provides theoretical guaranties of its performance. An important advantage of the developed framework is that it does not need a detailed specification of the time series function before and after a change. The hyperparameter values are learned from the data. Comprehensive evaluation of the proposed framework is provided both on synthetic and real data and its efficiency is demonstrated.
    
[^22]: 连续视界预测

    Continuous Horizon Forecasting

    [https://arxiv.org/abs/2611.13228](https://arxiv.org/abs/2611.13228)

    本文提出连续视界（CH）预测策略，将预测重新定义为连续算子、把预测视界作为查询变量而非索引，在理论上证明了其相对于离散策略的估计方差优势，并通过大规模实验验证了其具有竞争力的精度，同时独特地支持并行推理和对未见视界的插值与外推。

    

    多视界预测一直由离散预测策略主导，这些策略将模型输出视为固定向量或通过循环方式生成。在本文中，我们引入了连续视界（Continuous Horizon, CH）策略，它通过将预测重新定义为连续算子，并将预测视界定义为查询变量而非索引，从而对递归、直接和多步策略进行了泛化。我们为CH策略提供了理论基础，包括一个通用算子逼近定理，以及连续视界的偏差-方差分解，该分解从形式上证明了CH相对于离散策略在估计方差上的改进。我们通过仿真研究以及在M系列基准的16个大规模数据集上对CH策略进行了实证验证，结果表明CH实现了有竞争力的RMSE精度，同时独特地支持并行推理、针对性视界查询，以及对未见视界的插值和外推。

    arXiv:2611.13228v1 Announce Type: cross  Abstract: Multi-horizon forecasting has been dominated by discrete forecast strategies that treat model outputs as fixed-vectors or recurrently create them. In this paper, we introduce the Continuous Horizon (CH) strategy that generalizes upon recursive, direct, and multi-step strategies by redefining forecasts as a continuous operator and the forecast horizon as a query variable rather than an index. We present the theoretical foundations for the CH strategy, a Universal Operator Approximation Theorem, and a continuous-horizon bias-variance decomposition that formally proves the estimation variance improvements of CH over the discrete strategies. We validate the CH strategy empirically through a simulation study, and on 16 large-scale datasets from the M-series benchmark, showing how CH achieves competitive RMSE accuracy, while uniquely enabling parallel inference, targeted horizon querying, and interpolation and extrapolation across unseen hor
    
[^23]: MediPFN：面向因果中介分析的摊销式上下文干扰学习与推断

    MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis

    [https://arxiv.org/abs/2611.13219](https://arxiv.org/abs/2611.13219)

    MediPFN通过将预训练表格基础模型适配于干扰函数预测，并结合交叉拟合影响函数估计器与独立校准机制，为因果中介分析提供具有覆盖保证的置信区间。

    

    因果中介分析将处理效应分解为通过中介变量的间接路径和通过其他过程的直接路径。表格基础模型通过上下文学习执行分类和回归，为因果中介分析提供条件均值和倾向得分的估计。基于这些估计构建置信区间时，需要考虑拟合和组合多个干扰函数所引入的不确定性。我们提出MediPFN，这是一个摊销推断框架，它将预训练的表格基础模型适配于干扰预测任务，并通过交叉拟合的影响函数估计器组合其预测结果。一个单独的输出头从具有已知间接效应的模拟表格中学习估计器标准化误差的分位数。随后，通过监督微调、对称化以及独立校准来生成置信区间。我们建立了覆盖保证……（原文摘要在此处截断）

    arXiv:2611.13219v1 Announce Type: cross  Abstract: Causal mediation analysis decomposes a treatment effect into indirect pathways through mediators and direct pathways through other processes. Tabular foundation models perform classification and regression through in-context learning, providing estimates of conditional means and propensity scores used in causal mediation analysis. Constructing confidence intervals from these estimates requires accounting for uncertainty introduced by fitting and combining several nuisance functions. We introduce MediPFN, an amortized inference framework that adapts a pretrained tabular foundation model to nuisance prediction and combines its predictions through a cross-fitted influence-function estimator. A separate head learns quantiles of the estimator's standardized error from simulated tables with known indirect effects. Supervised fine-tuning, symmetrization, and independent calibration then produce a confidence interval. We establish coverage bou
    
[^24]: 已实现波动率预测中混合模型的预测推断

    Prediction Inference for Hybrid Models on Realized Volatility Forecasting

    [https://arxiv.org/abs/2611.13212](https://arxiv.org/abs/2611.13212)

    该论文通过在 GARCH-LSTM 混合模型中引入 ARMA 条件均值结构，并提出一种结合经验残差自助抽样与共形预测校正的向前自助算法，为已实现波动率构建了覆盖率更高的多步预测区间，弥补了现有方法忽视点估计不确定性的缺陷。

    

    arXiv:2611.13212v1 公告类型：cross 摘要：GARCH-LSTM 混合模型在波动率预测文献中日益流行。迄今为止，这些模型几乎完全专注于点估计以及诸如 RMSE 和 MAPE 之类的精度度量，而未能评估这些点估计的不确定性。我们从两个方面来填补这一空白。首先，我们通过用 ARMA 结构对条件均值进行建模，对现有的 GARCH-LSTM 模型进行了扩展。其次，我们开发了一种向前自助（forward bootstrap）算法，为未来的已实现波动率构建多步预测区间。为了解决预测区间普遍存在的覆盖率不足问题，我们实施了两项解决方案：一是从经验残差分布中抽取自助创新项（bootstrap innovations），以模仿金融序列的厚尾特性；二是利用共形预测（conformal prediction）的原理，系统地扩展预测区间的长度。我们在具有厚尾（的模拟环境中评估了这些方法）

    arXiv:2611.13212v1 Announce Type: cross  Abstract: GARCH-LSTM hybrid models have become increasingly popular in the volatility forecasting literature. To date, these models have almost exclusively focused on point estimation and accuracy measures such as RMSE and MAPE, while failing to assess the uncertainty of these point estimates. We approach this gap in two ways. First, we extend the existing GARCH-LSTM model by modeling the conditional mean with an ARMA structure. Second, we develop a forward bootstrap algorithm to build multi-step-ahead prediction intervals for future realized volatility. In order to address the notorious undercoverage of prediction intervals, we implement two solutions. One draws bootstrap innovations from the empirical residual distribution to mimic the fat-tail property of financial series. The other utilizes principles from conformal prediction to systematically extend the prediction interval length. We evaluate the methods in simulations with heavy-tailed an
    
[^25]: 确定性不可知PAC学习的近最优样本复杂度

    Near-Optimal Sample Complexity of Deterministic Agnostic PAC Learning

    [https://arxiv.org/abs/2611.13210](https://arxiv.org/abs/2611.13210)

    本文证明了VC维为 $d$、直径为 $K$ 的概念类在确定性标签下的不可知PAC学习样本复杂度为 $\widetilde{O}(K^{1/3} d^{2/3}/\varepsilon)$，并构造了对数因子内紧的下界，填补了此前上下界之间的差距。

    

    我们研究确定性标签下的不可知PAC学习问题，其中样本由任意确定性函数标注，该函数不必属于概念类 $C$。该模型刻画了不含随机标签噪声的模型误设情形，介于可实现设定与完全不可知设定之间。先前的工作表明，当概念类 $C$ 具有有限直径 $K$ 和VC维 $d$ 时，确定性标签可以将样本复杂度上界从经典的不可知保证 $O(d/\varepsilon^2)$ 改进到 $\widetilde{O}(K/\varepsilon)$，但上下界之间存在较大差距。我们的主要结果使这一图景更加清晰：我们证明，每一个VC维为 $d$ 且直径有限为 $K$ 的概念类都可以以样本复杂度 $\widetilde{O}\left(\frac{K^{1/3} d^{2/3}}{\varepsilon}\right)$ 进行学习，并且我们构造了使该界在对数因子范围内紧的概念类。我们还研究了其在线版本的对应问题。（原文摘要在此处截断）

    arXiv:2611.13210v1 Announce Type: cross  Abstract: We study the problem of agnostic PAC learning under deterministic labels, where the examples are labeled by an arbitrary deterministic function, which need not belong to the concept class $C$. This model captures misspecification without stochastic label noise and lies between the realizable and fully agnostic settings. Prior work showed that, when $C$ has finite diameter $K$ and VC dimension $d$, deterministic labels can improve the sample complexity upper bound from the classical agnostic guarantee $O(d/\varepsilon^2)$ to $\widetilde{O}(K/\varepsilon)$, but left a large gap between upper and lower bounds. Our main result sharpens this picture. We prove that every class of VC dimension $d$ and finite diameter $K$ is learnable with sample complexity $\widetilde{O}\left( \frac{K^{1/3} \, d^{2/3}}{\varepsilon} \right)$, and we construct classes for which this bound is tight up to logarithmic factors. We also study the online analogue of 
    
[^26]: 预测性惊奇作为可解释时间序列的自锚定概念瓶颈

    Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

    [https://arxiv.org/abs/2611.12955](https://arxiv.org/abs/2611.12955)

    ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。

    

    部署在风险敏感领域的时间序列分类器必须做到：(1) 准确，(2) 可解释，(3) 在证据不足时能够弃权（拒绝预测）。现有方法无法同时满足这三点，且这一障碍是结构性的，而非操作层面的。若不知道分段边界，就无法定义有意义的时间模式；而若不知道期待何种模式，又无法划分有意义的边界。现有方法通过使用固定窗口或专家标注的词汇表来打破这种循环，从而至少牺牲了上述三个属性中的一个。我们提出 ConceptTime，它通过单一的自监督信号闭环解决这一问题。一个冻结的概率预测器在序列的每个时间步预测下一个观测值的分布。我们将其预测与实际信号之间的不匹配称为预测性惊奇。该信号既能定位分段边界，又能刻画由此产生的每个概念……

    arXiv:2611.12955v1 Announce Type: cross  Abstract: Time-series classifiers deployed in risk-sensitive domains must be (1) accurate, (2) interpretable, and (3) able to abstain when evidence is weak. Existing methods do not deliver all 3 at once, and the obstacle is structural, rather than logistical. Meaningful temporal patterns cannot be defined without knowing segment boundaries, yet meaningful boundaries cannot be placed without knowing which patterns to expect. Existing approaches break this circularity by using fixed windows or expert-annotated vocabularies, thereby sacrificing at least one of the 3 properties. We introduce ConceptTime, which closes the loop through a single self-supervised signal. A frozen probabilistic forecaster predicts a distribution over the next observation at every timestep of the sequence. We call the mismatch between its prediction and the realized signal the predictive surprise. This signal both locates segment boundaries and characterizes each resulting
    
[^27]: 基于流形生成学习的概率地震性能评估与结构可辨识性

    Generative Learning on Manifolds for Probabilistic Seismic Performance Assessment and Structural Identifiability

    [https://arxiv.org/abs/2611.12908](https://arxiv.org/abs/2611.12908)

    该研究提出一种结合扩散映射、分数扩散模型与几何调和的流形生成学习框架，可从有限的非线性仿真中学习结构参数与地震响应的联合分布，同时实现地震性能的正向概率预测与结构特性的反向辨识。

    

    地震性能评估依赖于计算成本高昂的非线性仿真，这限制了不确定性量化以及从观测响应中推断结构特性的能力。我们开发了一种基于流形的生成框架，能够从有限的高保真仿真中学习不确定结构参数与地震响应轨迹的联合分布。该框架结合了扩散映射、基于分数的扩散建模和几何调和方法，可实现正向概率预测与反向推断。将该框架应用于一座九层钢抗弯框架，结果揭示了哪些结构特性可从地震性能中辨识出来，以及这些特性的不确定性如何影响结构响应。特别是，学习到的表示比单个强度参数更有效地捕捉了峰值后退化特性。反向推断可识别出不同的结构特性（原文在此处截断）。

    arXiv:2611.12908v1 Announce Type: cross  Abstract: Seismic performance assessment relies on computationally expensive nonlinear simulations, limiting uncertainty quantification and inference of structural properties from observed responses. We develop a manifold-based generative framework that learns the joint distribution of uncertain structural parameters and seismic-response trajectories from limited high-fidelity simulations. Combining Diffusion Maps, score-based Diffusion Modeling, and Geometric Harmonics, the framework enables both forward probabilistic prediction and inverse inference. Applied to a nine-story steel moment-resisting frame, the results reveal which structural properties are identifiable from seismic performance and how uncertainty in these properties influences response. In particular, the learned representation captures post-capping degradation characteristics more effectively than individual strength parameters. Inverse inference identifies distinct structural c
    
[^28]: 流与扩散模型的几何与随机分析：综述

    Geometry and Stochastic Analysis of Flow and Diffusion Models: A Survey

    [https://arxiv.org/abs/2611.12107](https://arxiv.org/abs/2611.12107)

    这篇综述从三种几何（状态空间、概率测度空间、路径空间）与随机分析的统一视角阐明流与扩散模型的数学基础，指出所有得分恒等式本质上都是分部积分，并将Malliavin演算等工具贯穿于归一化流、流匹配与少步映射等方法之中。

    

    基于流的模型和基于扩散的模型已成为学习与采样高维概率分布的主要手段，其应用涵盖图像、视频、蛋白质、分子和物理场，并日益扩展到语言领域。每种模型都沿着一条概率测度曲线将一个简单的规律传递到数据的规律之中，而在其多样的名称背后隐藏着三种几何：数据所栖居的状态空间的几何、曲线所穿越的概率测度空间的几何，以及生成过程所描绘的路径空间的几何。一个统一的解析思想贯穿整个领域：用于训练扩散模型的每一个得分恒等式本质上都是一次分部积分——在噪声上、在数据上或在维纳空间上进行——而Malliavin演算将这一思想推广到非线性与无穷维扩散过程。本文以完整的定义系统发展了这些理论基础，并将其贯穿应用于归一化流、流匹配和少步映射等方法之中。

    arXiv:2611.12107v1 Announce Type: cross  Abstract: Flow-based and diffusion-based models have become the principal means of learning and sampling high-dimensional probability distributions, from images and video to proteins, molecules and physical fields, and increasingly language. Each carries a simple law to the law of the data along a curve of probability measures, and behind the variety of their names lie three geometries, those of the state space on which the data live, of the space of probability measures through which the curve passes, and of the space of paths traced by the generating process. A single analytic idea runs through the subject. Every score identity used to train a diffusion model is an integration by parts, on the noise, on the data or on Wiener space, and Malliavin calculus extends this to nonlinear and infinite-dimensional diffusions. We develop these foundations with full definitions and follow them through normalising flows, flow matching and few-step maps, di
    
[^29]: 关于用于自适应镇定的随机连续时间线性系统辨识

    On Stochastic Continuous-Time Linear System Identification for Adaptive Stabilization

    [https://arxiv.org/abs/2611.02340](https://arxiv.org/abs/2611.02340)

    本文提出了基于系统辨识的自适应镇定的失败概率与轨迹长度和数量、维度、可控性Gramian矩阵及信噪比等所有相关量之间简单、实用且可靠的定量关系。

    

    系统辨识研究的是从状态和输入轨迹中可靠地学习系统动态这一经典问题。其重要应用之一是估计能够镇定未知系统的反馈增益矩阵。本文聚焦于对多变量线性时不变状态空间模型的自适应镇定方案的分析。系统的不稳定性实际上使得轨迹较短，即观测到的数据相当有限。因此，紧密考虑所有相关量的影响至关重要，包括轨迹的长度和数量、维度、可控性Gramian矩阵以及信噪比。我们提出了一个简单、实用且可靠的关系，用以刻画基于系统辨识的自适应镇定未能镇定系统的概率与上述所有量之间的联系。我们的发现表明，尽管存在指数级的影响……

    arXiv:2611.02340v2 Announce Type: replace-cross  Abstract: System identification studies the canonical problem of reliably learning system-dynamics from state and input trajectories. One of its important applications consists of estimating a feedback gain matrix that can stabilize an unknown system. This paper focuses on the analysis of such adaptive stabilization schemes for multivariate linear time-invariant state-space models. The system instability practically renders trajectories short, i.e., the observed data is notably limited. So, it is crucial to tightly consider effects of \emph{all} involved quantities including length and number of trajectories, dimension, controllability Gramian, and signal-to-noise ratio. We put forward a simple, practical, and reliable relationship between the chance that adaptive stabilization (based on system identification) fails to stabilize the system, in terms of all the above-mentioned quantities. Our findings show that despite exponential effects
    
[^30]: 外国信号去哪儿了？

    Where Has the Foreign Signal Gone?

    [https://arxiv.org/abs/2610.28175](https://arxiv.org/abs/2610.28175)

    该研究发现Twitter情绪噪声对下一季度股票损失的预测关系在本国股票上比外国股票更加稳定，并用“生产者-知识错配假说”解释：外国股票的英语Twitter贡献者池缺乏拥有信息的母国市场投资者，导致其信息流中非基本面噪声成分更大。

    

    我们比较了Twitter信息流中的情绪噪声与本国公司及外国公司股票损失之间的关系。下一季度损失对所测噪声的斜率的季度均值“国内-国外”差异为-1.40（Newey-West t = -2.99）。对于本国股票和外国股票，该斜率分别在78%和48%的季度中为负（块自助法 p = 0.016）。因此，这一关系在时间上对本国股票更加稳定。我们运用一个噪声理性预期模型，通过贡献者池的构成来解释这些模式。与本国公司相比，外国公司股票的信息流具有更高的重复作者集中度，以及更大的模型隐含的非基本面成分。我们通过“生产者-知识错配假说”来解读这些差异：对于外国公司，英语贡献者池低估了往往拥有信息优势的母国市场投资者……（原文摘要至此截断）

    arXiv:2610.28175v1 Announce Type: new  Abstract: We compare the relation between sentiment noise in the Twitter feed and losses for stocks of domestic and foreign firms. The mean quarterly domestic-foreign difference in the slope of next-quarter losses on measured noise is -1.40 (Newey-West t = -2.99). This slope is negative for 78 and 48 percent of the quarters for domestic and foreign stocks, respectively (block-bootstrap p = 0.016). Thus, this relationship is temporally more stable for domestic stocks. We use a noisy rational expectations model to explain these patterns through the composition of the contributor pool. The feeds for stocks of foreign firms have higher repeat-author concentration and a larger model-implied nonfundamental component than those for domestic firms. We interpret these differences through the producer-knowledge mismatch hypothesis. For foreign firms, the English-language contributor pool underrepresents the home-market investors who tend to possess informat
    
[^31]: 深度学习中Hessian矩阵的特征值：对称性的起源及其破缺

    Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking

    [https://arxiv.org/abs/2610.09919](https://arxiv.org/abs/2610.09919)

    本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。

    

    深度学习中训练后模型的Hessian谱呈现出一种持续存在的模式：特征值组织成不同的簇，包括一个位于零附近的大块以及少数孤立的离群值。本文表明，当将原始设定理解为对附近一个原本隐藏的高度对称参考构型的偏离时，这些谱现象便获得了一个自然的解释。通过对架构、数据分布或参数度量等进行修改，可以揭示出一个邻近的参考构型，其Hessian展现出丰富的、无法由权重对称性所解释的不变性。在该参考构型中，对称性使得谱能够被精确描述，并迫使出现高维的核空间以及大重数的特征值。而回到原始构型则破坏了Hessian的对称性，从而产生了所观测到的簇与离群值的层级结构。该框架在相当一般的情形下被建立，并对……（摘要原文在此处截断）

    arXiv:2610.09919v1 Announce Type: new  Abstract: Hessian spectra at trained models in deep learning exhibit a persistent pattern: eigenvalues organize into distinct clusters, including a large bulk near zero and a few isolated outliers. This paper shows that a natural account of these spectral phenomena emerges when the original setting is understood as a departure from a nearby, otherwise hidden, highly symmetric reference.   Modifications, including changes to the architecture, data distribution, or parameter metric, expose a nearby reference configuration whose Hessian exhibits rich invariances-ones not accounted for by weight symmetries. There, symmetry enables a precise description of the spectra, forcing high-dimensional kernels and eigenvalues of large multiplicity. Returning to the original configuration breaks the Hessian symmetry and thereby produces the observed hierarchy of clusters and outliers.   The framework is developed in some generality, with a detailed analysis of t
    
[^32]: 反事实公平强化学习的分布映射方法

    A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning

    [https://arxiv.org/abs/2608.08743](https://arxiv.org/abs/2608.08743)

    本文提出一种基于分位数分布映射的数据预处理算法，通过序贯估计反事实状态和奖励，在强化学习中实现反事实公平性，并将常用的可加性假设作为特例涵盖其中。

    

    强化学习（RL）旨在通过优化序贯决策来最大化长期的群体收益。然而，在医疗保健等高风险场景中部署时，强化学习的决策可能系统性地限制某些亚群体获取有价值服务的机会，这与利益相关者的价值观和目标相违背。反事实公平性（CF）提供了一个基于因果推理来解决这一问题的有前景的框架。本文提出了一种数据预处理算法，将其与策略学习结合使用时，能够在强化学习中实现反事实公平性。我们的算法依赖于一种新颖的分位数分布映射方法，在数据预处理步骤中序贯地估计反事实状态和奖励，并将反事实预测中常用的可加性假设作为特例纳入其中。我们从理论上证明了每步的反事实不公平程度和无限时域的次优差距……

    arXiv:2608.08743v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) seeks to optimize sequential decisions to maximize population-level benefits over time. However, when deployed in high-stakes settings such as healthcare, RL decisions might systematically restrict some subpopulation's access to valuable services in a manner contrary to the values and goals of stakeholders. Counterfactual fairness (CF) offers a promising framework to address this problem based on causal reasoning. This paper develops a data preprocessing algorithm that, when used in tandem with policy learning, enables CF in RL. Our algorithm relies on a novel quantile distribution mapping method for sequentially estimating the counterfactual states and rewards in the data preprocessing step, subsuming common additivity assumptions used for counterfactual prediction as a special case. We theoretically prove that the per-step level of counterfactual unfairness and infinite-horizon suboptimality gap ca
    
[^33]: 自回归模型中的全变差距离估计

    Total Variation Distance Estimation in Autoregressive Models

    [https://arxiv.org/abs/2607.19510](https://arxiv.org/abs/2607.19510)

    本文研究了在样本访问、logit 访问和含噪 logit 访问三种模型下估计两个长度为 $n$ 的自回归分布之间全变差距离的问题，将样本访问下的查询复杂度从已有的 $\widetilde{O}(n^3 m/\varepsilon^5)$ 改进为 $\widetilde{O}(n^2 K/\varepsilon^2)$，并给出了 logit 访问下紧的 $O(n/\varepsilon^2)$ 查询界。

    

    现代大语言模型的部署在固定权重之上使用了多种实现选择和推理优化手段（如批处理、自定义内核和量化），因此两个提供“同一模型”服务的引擎可能产生具有实质性差异的分布。我们研究了在三种访问模型下，估计两个长度为 $n$ 的自回归分布之间的全变差（TV）距离至加性误差 $\varepsilon$ 的问题。（1）在样本访问下，我们使用 $\widetilde{O}(n^2 K/\varepsilon^2)$ 次查询，其中 $K$ 是下一个标记（next-token）分布的最大支撑集大小，这改进了 Meel 等人（2025）需要 $\widetilde{O}(n^3 m/\varepsilon^5)$ 次查询的估计器，其中 $m \geq K$ 是标记字母表的总大小。（2）在 logit 访问下，我们使用 $O(n/\varepsilon^2)$ 次查询，且该界是紧的。（3）在含噪 logit 访问下，我们在上述两种保证之间实现了平滑插值：如果概率值以相对……（摘要原文在此处截断）

    arXiv:2607.19510v2 Announce Type: replace  Abstract: Modern LLM deployments use a number of implementation choices and inference optimizations (e.g., batching, custom kernels, and quantization) on top of fixed weights, so two engines serving "the same model" can produce meaningfully different distributions. We study the problem of estimating the total variation (TV) distance between two length-$n$ autoregressive distributions to additive error $\varepsilon$, under three access models. (1) Under sample access, we use $\widetilde{O}(n^2 K/\varepsilon^2)$ queries, where $K$ is the maximum support of the next-token distribution. This improves upon the $\widetilde{O}(n^3 m/\varepsilon^5)$-query estimator of Meel et al. (2025), where $m \geq K$ is the total size of the token alphabet. (2) Under logit access, we use $O(n/\varepsilon^2)$ queries, and this is tight. (3) Under noisy logit access, we smoothly interpolate between the above two guarantees: if probability values are given to relativ
    
[^34]: 利用带控制的动态模态分解在单一实现中解耦强迫气候变率与内部气候变率

    Disentangling Forced and Internal Climate Variability in Single Realizations using Dynamic Mode Decomposition with Control

    [https://arxiv.org/abs/2607.18298](https://arxiv.org/abs/2607.18298)

    该研究提出PullbackDMDc方法，基于非自治动力系统理论、带控制的动态模态分解与拉回吸引子估计，将外部强迫视为动力学驱动因子，从而实现从单一气候观测记录中分离强迫趋势与内部气候变率。

    

    将外部强迫趋势与内部气候变率分离是气候科学中的一项根本性挑战，尤其是在分析仅以单一实现形式存在的观测记录时。我们证明，通过将外部强迫视为线性随机系统中的一个动力学驱动因子，可以将单一气候实现分解为强迫分量和内部分量，这一思想植根于拉回吸引子理论。这解决了气候预估以及强迫响应检测与归因中的一个核心难题。现有的单一实现方法依赖于忽略强迫作用的线性逆模型（LIM），或忽略动力过程的线性回归。在此，我们提出PullbackDMDc方法，该方法基于非自治动力系统理论和带控制的动态模态分解（DMDc），并结合拉回吸引子估计，将单一气候实现分解为空间模态……（原文摘要在此处截断）

    arXiv:2607.18298v2 Announce Type: replace-cross  Abstract: Separating externally forced trends from internal climate variability is a fundamental challenge in climate science, particularly when analyzing observational records that exist only as a single realization. We show that a single climate realization can be decomposed into forced and internal components by treating external forcing as a dynamical driver within a linear stochastic system, an idea grounded in pullback attractor theory. This addresses a central challenge for climate projection and for detecting and attributing the forced response. Existing single-realization methods rely on linear inverse models (LIMs), which ignore forcing, or linear regression, which omits dynamics. Here we introduce PullbackDMDc, a method grounded in non-autonomous dynamical systems theory and dynamic mode decomposition with control (DMDc), incorporating pullback attractor estimation to decompose a single climate realization into spatial modes a
    
[^35]: 浅层神经网络中随时间一致的弱混沌传播

    Uniform-in-Time Weak Propagation-of-Chaos in Shallow Neural Networks

    [https://arxiv.org/abs/2605.22010](https://arxiv.org/abs/2605.22010)

    该论文通过利用平均场确定性 Wasserstein 梯度流动的收敛速率，而非依赖强凸性或对数 Sobolev 不等式，首次在浅层神经网络的梯度下降训练中建立了随时间一致成立的非渐近弱混沌传播结果。

    

    我们考虑在特征学习机制下使用梯度下降训练的单隐层神经网络，并将有限宽度网络的输出 $f_{\hat{\rho}_t^m}$ 与其无穷宽度对应物 $f_{\rho_t^{MF}}$（按平均场动力学演化）建立联系。虽然可以通过标准的 Grönwall 估计得到 $\|f_{\rho_t^{MF}} - f_{\hat{\rho}_t^m}\|$ 在常数时间范围内的界，但波动的长时间行为是一个更为微妙的问题。随时间一致的界通常依赖于损失景观中的（局部）强凸性，或带噪梯度动力学中存在的对数 Sobolev 不等式。在这项工作中，我们建立了随时间一致成立的非渐近弱混沌传播，这是通过利用平均场确定性 Wasserstein 梯度流动动力学的收敛速率而实现的。具体而言，记 $L_t$ 为时刻 $t$ 的平均场超额 MSE 损失，$m$ 为神经元数量……（摘要在此处被截断）

    arXiv:2605.22010v2 Announce Type: replace-cross  Abstract: We consider one-hidden layer neural networks trained in the feature-learning regime using gradient descent, and relate the output of the finite-width network $f_{\hat{\rho}_t^m}$ to its infinite-width counterpart $f_{\rho_t^{MF}}$, which evolves in the mean-field dynamics.   While constant-time horizon bounds for $\|f_{\rho_t^{MF}} - f_{\hat{\rho}_t^m}\|$ may be obtained via standard Gr\"onwall estimates, the long-time behavior of the fluctuation is a more delicate matter. Uniform-in-time bounds often rely on (local) strong convexity in the landscape or Logarithmic Sobolev inequalities present in noisy gradient dynamics.   In this work, we establish non-asymptotic weak propagation-of-chaos that holds uniformly in time, obtained by exploiting instead the convergence rate of the mean-field deterministic Wasserstein-gradient-flow dynamics. Specifically, denoting by $L_t$ the mean-field excess MSE loss at time $t$ and $m$ the numbe
    
[^36]: 扩散教师模型梯度的方差削减

    Variance Reduction for Gradients from Diffusion Teachers

    [https://arxiv.org/abs/2605.21489](https://arxiv.org/abs/2605.21489)

    提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。

    

    预训练扩散模型可作为冻结的教师模型，服务于文本到3D生成、单步蒸馏和数据归因等流程。这些流程所消耗的教师梯度是关于噪声水平和高斯噪声的期望，通过蒙特卡洛（MC）方法估计。每次MC采样都需要昂贵的上游计算（如渲染、模拟、编码），因此估计器的方差主导了计算成本。我们提出了CARV，一个计算感知的方差核算框架，它将这些梯度建模为一个具有不对称每层成本的经典两阶段采样问题。这启发了一种分层蒙特卡洛估计器，通过将每次昂贵的上游计算在多个廉价的扩散采样中复用来摊销其成本，并通过时间步重要性采样（IS）和分层逆CDF采样进一步降低单元内方差。CARV在不改变目标函数的情况下，为文本到3D和数据归因任务带来了2至3倍的有效计算乘数。

    arXiv:2605.21489v3 Announce Type: replace-cross  Abstract: Pretrained diffusion models serve as frozen teachers for pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are expectations over noise levels and Gaussian noise, estimated by Monte Carlo (MC). Each MC draw requires expensive upstream work (rendering, simulation, encoding), so estimator variance dominates compute cost. We introduce CARV, a compute-aware variance-accounting framework that casts these gradients as a classical two-stage sampling problem with asymmetric per-level costs. This motivates a hierarchical MC estimator that amortizes each expensive upstream computation by reusing it across cheap diffusion draws, with within-unit variance further reduced by timestep importance sampling (IS) and stratified inverse-CDF sampling. CARV delivers 2 to 3x effective compute multipliers on text-to-3D and data attribution without changing the objective. In text
    
[^37]: 用于事后学习延后的密度比损失

    Density-Ratio Losses for Post-Hoc Learning to Defer

    [https://arxiv.org/abs/2605.19557](https://arxiv.org/abs/2605.19557)

    该论文提出用模型与专家理想分布之间的密度比来定义事后学习延后，推导出两种 DR CPE 损失用于训练延后评分器，其中联合变体在 KL 散度情形下与 Chow 规则及专家倾斜的贝叶斯后验相联系且计算便利。

    

    我们通过理想分布的视角研究事后学习延后：理想分布是指数据分布在发散度正则化下的重加权，使得模型在该分布下能取得较低的任务损失。我们通过模型理想与专家理想之间的密度比来定义延后。利用从密度比估计到类概率估计的归约，我们推导出适用于事后 L2D 评分器的 DR CPE 损失。延后决策随后通过对评分器进行阈值化来做出。对于基于 KL 散度的理想分布，我们的延后规则在原始分布下可恢复 Chow 规则，并且（取决于理想分布是边缘分布还是联合分布）与专家倾斜的贝叶斯后验——其中纳入了专家的表现——建立起联系。这导出了 DR CPE 损失的两种变体，其中联合变体在实践中计算上更为便利。当……时，我们建立了联合 DR CPE 损失的一致性性质。

    arXiv:2605.19557v2 Announce Type: replace-cross  Abstract: We study post-hoc Learning to Defer (L2D) through the lens of ideal distributions: divergence-regularized reweightings of the data distribution under which a model attains low task loss. We define deferral via the density-ratio between a model's and an expert's ideals. Using the reduction from density-ratio estimation to class-probability estimation, we derive the DR CPE losses for post-hoc L2D scorers. Deferral decisions are then made by thresholding the scorer. For KL-based ideal distributions, our deferral rule recovers Chow's rule under the original distribution and a connection to an expert-tilted Bayes posterior---which incorporates the expert's performance---depending on if the ideal distributions are marginal or joint distributions. This leads to two variants of the DR CPE losses, with the joint variant being computationally convenient in practice. We establish consistency properties of the joint DR CPE losses when cons
    
[^38]: 因子增强的高维随机梯度下降

    Factor Augmented High-Dimensional SGD

    [https://arxiv.org/abs/2605.19291](https://arxiv.org/abs/2605.19291)

    提出了一种完全基于流式数据的因子增强随机梯度下降方法（FSGD），并首次建立了将潜在因子估计误差纳入SGD分析的理论框架，为高维机器学习提供了可靠且可扩展的优化基础。

    

    随机梯度下降（SGD）是一种在现代机器学习中广泛应用的基础优化算法。本文提出了因子增强随机梯度下降（FSGD），这是一种新的优化方法，能够在高维学习任务中利用潜在因子表示。与依赖离线表示学习和完整数据存储的标准两阶段降维方法不同，FSGD的一个关键创新在于它完全基于流式数据运行，使其能够扩展到大规模和高维问题。此外，我们建立了首个将潜在因子估计误差显式纳入SGD分析的理论框架，并在衰减步长和小批量更新条件下给出了$\ell^s$范数的矩收敛性结果。我们的成果为在高维机器学习系统中可靠且可扩展地应用SGD提供了新的理论基础。

    arXiv:2605.19291v2 Announce Type: replace-cross  Abstract: Stochastic gradient descent (SGD) is a fundamental optimization algorithm widely used in modern machine learning. In this paper, we propose Factor-Augmented SGD (FSGD), a new optimization method that leverages latent factor representations in high-dimensional learning tasks. Unlike standard two-stage dimension reduction approaches that rely on offline representation learning and full data storage, a key novelty of FSGD is that it operates purely on streaming data, making it scalable to large-scale and high-dimensional problems. Furthermore, we establish the first theoretical framework that explicitly incorporates latent factor estimation error into the analysis of SGD, and provide moment convergence in $\ell^s$ norm under decaying step sizes and mini-batch updates. Our results provide a new foundation for employing SGD reliably and scalably in high-dimensional machine learning systems.
    
[^39]: 树集成模型的极小化极大速率与谱蒸馏

    Minimax Rates and Spectral Distillation for Tree Ensembles

    [https://arxiv.org/abs/2605.11841](https://arxiv.org/abs/2605.11841)

    本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。

    

    树集成模型，如随机森林（RF）和梯度提升机（GBM），是最广泛使用的监督学习方法之一，但其理论性质仍未被完全理解。我们对这些算法采用谱视角进行分析，包含两个主要贡献。首先，我们推导了随机森林回归的极小化极大最优收敛性，证明在树生长的温和正则性条件下，诱导核算子的特征值衰减决定了统计收敛速率。其次，我们利用这一谱视角为树集成模型开发了压缩方案。对于随机森林，核算子的主特征函数能够捕捉主导的预测方向；对于梯度提升机，平滑矩阵的主奇异向量则起到类似的作用。通过为这些谱表示学习非线性映射，可以得到比原始模型小数个数量级、同时保持有竞争力的预测性能的蒸馏模型。

    arXiv:2605.11841v3 Announce Type: replace-cross  Abstract: Tree ensembles such as random forests (RFs) and gradient boosting machines (GBMs) are among the most widely used supervised learners, yet their theoretical properties remain incompletely understood. We adopt a spectral perspective on these algorithms, with two main contributions. First, we derive minimax-optimal convergence for RF regression, showing that, under mild regularity conditions on tree growth, the eigenvalue decay of the induced kernel operator governs the statistical rate. Second, we exploit this spectral viewpoint to develop compression schemes for tree ensembles. For RFs, leading eigenfunctions of the kernel operator capture the dominant predictive directions; for GBMs, leading singular vectors of the smoother matrix play an analogous role. Learning nonlinear maps for these spectral representations yields distilled models that are orders of magnitude smaller than the originals while maintaining competitive predict
    
[^40]: 各向异性Besov空间中稀疏Kolmogorov-Arnold网络的后验收缩速率

    Posterior Contraction Rates for Sparse Kolmogorov-Arnold Networks in Anisotropic Besov Spaces

    [https://arxiv.org/abs/2605.11652](https://arxiv.org/abs/2605.11652)

    本文证明了配备尖峰-厚板稀疏先验的稀疏贝叶斯Kolmogorov-Arnold网络在各向异性Besov空间上可达到接近极小极大的后验收缩速率，能通过超先验自适应未知的光滑度，且网络深度可保持固定。

    

    我们研究了各向异性Besov空间上稀疏贝叶斯Kolmogorov-Arnold网络（KANs）的后验收缩速率，从贝叶斯的视角为KANs提供了统计学基础。我们证明，配备尖峰-厚板型稀疏先验的稀疏贝叶斯KANs能够达到接近极小极大的后验收缩速率。特别地，该收缩速率取决于目标函数本身的内在各向异性光滑度。此外，通过对单一模型规模参数设置超先验，所得后验能够自适应未知的各向异性光滑度，并仍然达到相应的接近极小极大的速率。与标准多层感知机的相关结果相比，我们结果的一个显著特点是KAN的深度可以保持固定：得益于可学习样条边函数的表达能力，所需的近似复杂度通过网络宽度以及稀疏性的范围和规模来控制。

    arXiv:2605.11652v2 Announce Type: replace-cross  Abstract: We study posterior contraction rates for sparse Bayesian Kolmogorov-Arnold networks (KANs) over anisotropic Besov spaces, providing a statistical foundation for KANs from a Bayesian point of view. We show that sparse Bayesian KANs equipped with spike-and-slab-type sparsity priors attain a near-minimax posterior contraction rate. In particular, the contraction rate depends on the intrinsic anisotropic smoothness of the underlying function. Moreover, by placing a hyperprior on a single model-size parameter, the resulting posterior adapts to unknown anisotropic smoothness and still achieves the corresponding near-minimax rate. A distinctive feature of our results, compared with those for standard multilayer perceptrons, is that the KAN depth can be kept fixed: owing to the expressive power of learnable spline edge functions, the required approximation complexity is controlled through the network width, the range and size of the sp
    
[^41]: 仿射追踪：概率线性求解器的新范式

    Affine Tracing: A New Paradigm for Probabilistic Linear Solvers

    [https://arxiv.org/abs/2605.10566](https://arxiv.org/abs/2605.10566)

    论文证明贝叶斯概率线性求解器只是非平稳仿射概率迭代方法的特例，并提出“仿射追踪”框架，能够从仿射迭代方法的标准实现中自动构建概率迭代方法，从而推动其广泛实际应用。

    

    概率线性求解器（PLSs）返回概率分布，用于量化求解线性系统时因计算资源有限而产生的不确定性。文献中传统上将贝叶斯PLS（其基于从线性系统投影中获得的信息对先验进行条件化）与概率迭代方法进行区分，后者将经典迭代求解器提升到概率空间。在本工作中，我们证明这种二分法是不成立的：贝叶斯PLS是非平稳仿射PIM的一种特例。此外，我们证明了任何现实的仿射PIM都是经过校准的。这些结果促使人们关注（非平稳）仿射PIM，但其实际应用一直受到实现它们所需的大量手工工作的限制。为解决这一问题，我们提出了仿射追踪，这是一种算法框架，能够通过仿射迭代方法的标准实现自动构建PIM……

    arXiv:2605.10566v2 Announce Type: replace-cross  Abstract: Probabilistic linear solvers (PLSs) return probability distributions that quantify uncertainty due to limited computation in the solution of linear systems. The literature has traditionally distinguished between Bayesian PLSs, which condition a prior on information obtained from projections of the linear system, and probabilistic iterative methods (PIMs), which lift classical iterative solvers to probability space. In this work we show this dichotomy to be false: Bayesian PLSs are a special case of non-stationary affine PIMs. In addition, we prove that any realistic affine PIM is calibrated. These results motivate a focus on (non-stationary) affine PIMs, but their practical adoption has been limited by the significant manual effort required to implement them. To address this, we introduce affine tracing, an algorithmic framework that automatically constructs a PIM from a standard implementation of an affine iterative method by 
    
[^42]: 非平衡最优传输中传输-增长对的最小极大最优估计

    Minimax Optimal Estimation of Transport-Growth Pairs in Unbalanced Optimal Transport

    [https://arxiv.org/abs/2605.08705](https://arxiv.org/abs/2605.08705)

    本文针对具有二次代价和KL边际惩罚的非平衡最优传输，提出估计传输-增长对的最近邻估计器和基于核的估计器，并证明了核估计器在对数因子内达到最小极大最优速率。

    

    非平衡最优传输将经典最优传输扩展到了总质量不同的测度情形，但针对传输映射和增长映射估计的统计保证仍然有限。我们研究了具有二次代价和库尔巴克-莱布勒（KL）边际惩罚的非平衡最优传输，并论证了估计的自然目标是传输-增长对。因此，我们在两种设定下为传输-增长对开发了两种估计器：一种是在一般密度之间进行传输的最近邻估计器，另一种是在光滑密度之间进行传输的基于核的估计器。我们通过推导最小极大风险的匹配下界，证明了基于核的估计器在对数因子范围内达到了最小极大最优速率。我们的主要技术贡献是将估计问题归约为非平衡最优传输值的稳定性界，该界控制了传输映射和增长映射（摘要在此处被截断）。

    arXiv:2605.08705v2 Announce Type: replace-cross  Abstract: Unbalanced optimal transport extends classical optimal transport to measures with different total masses, but statistical guarantees for estimation of the transport map and the growth map remain limited. We study unbalanced optimal transport with quadratic cost and Kullback-Leibler marginal penalties and argue that the natural target of estimation is a transport-growth pair. Consequently, we develop two estimators for the transport-growth pair under two setups: a nearest-neighbor estimator for transporting between general densities, and a kernel-based estimator for transporting between smooth densities. We prove that the kernel-based estimator achieves the minimax optimal rate up to a logarithmic factor by deriving a matching lower bound on the minimax risk. Our main technical contribution is a reduction of estimation to a stability bound on the unbalanced optimal transport value. This bound controls the transport map and growt
    
[^43]: 基于核方法的希尔伯特空间中的高斯混合模型

    Gaussian mixture models in Hilbert spaces via kernel methods

    [https://arxiv.org/abs/2605.05996](https://arxiv.org/abs/2605.05996)

    该论文提出了一种基于核均值嵌入的希尔伯特空间数据高斯混合模型框架，用于对动态函数数据等无限维随机对象进行聚类分析，并从理论和实验上证明了其良定义性、稠密近似能力以及在多种数据几何形态上的有效性。

    

    现代许多学科的数据集日益由随时间演化、潜在无限维的随机对象组成，例如动态函数数据，这些数据自然地在希尔伯特空间中进行建模。在这些场景下，对概率测度进行刻画（例如通过密度函数）可能是病态定义的或在技术上具有挑战性。受聚类应用的启发，我们提出了一种基于核均值嵌入的希尔伯特空间取值数据的高斯混合框架，并开发了高效的优化算法用于估计。我们建立了理论保证，证明所提出的算法是良定义的，且该模型在无限维空间中产生稠密的近似类。我们通过对多种数据几何形态的广泛实验评估了该框架，包括 $L^2$ 函数数据以及现代医疗应用中出现的图信号。

    arXiv:2605.05996v2 Announce Type: replace-cross  Abstract: Modern datasets across many disciplines increasingly consist of time-evolving, potentially infinite-dimensional random objects, such as dynamic functional data, which are naturally modeled in Hilbert spaces. In these settings, characterizing probability measures, for example, through densities, can be ill-defined or technically challenging. Motivated by clustering applications, we propose a Gaussian mixture framework for Hilbert-space-valued data based on kernel mean embeddings and develop efficient optimization algorithms for estimation. We establish theoretical guarantees showing that the proposed algorithm is well-defined and that the model yields a dense class of approximations in infinite-dimensional spaces. We evaluate the framework through extensive experiments on diverse data geometries, including $L^2$-functional data and graph signals arising in modern medical applications.
    
[^44]: AI赋能分析中的人类监督分配

    Allocating Human Oversight in AI-Enabled Analytics

    [https://arxiv.org/abs/2604.12497](https://arxiv.org/abs/2604.12497)

    该论文提出了一种在任务难度未知情形下分配共享人类验证预算的上置信界策略，能够从人类标签与AI预测的配对数据中联合学习预测校正系数和任务难度，并具有分配遗憾理论保证，在实验中将性能提升至接近最优方案的水平。

    

    人工智能为许多分析任务提供了低成本的预测，但其可靠性以及额外获取一个人类标签的价值因任务而异。我们研究了在任务级不确定性未知的情况下，如何分配共享的人类验证预算，其中每个获取的标签同时服务于估计和后续的分配决策。对于难度已知的情形，基准方法遵循一种成本调整后的Neyman规则，该规则同时考虑任务重要性和标签成本。在我们的“调优纠偏-难度”框架内，我们开发了一种上置信界策略，从配对的人类标签和预测中联合学习调优预测驱动推断的校正系数以及由此产生的任务难度，并为置信半径策略建立了分配遗憾保证。在合成实验和一项包含68个任务、超过2000名受访者的真实数字孪生调查中，我们的方法弥补了与最优方案之间的大部分差距。

    arXiv:2604.12497v3 Announce Type: replace  Abstract: Artificial intelligence provides low-cost predictions for many analytics tasks, but their reliability and the value of another human label vary across tasks. We study how to allocate a shared human-validation budget when task-level uncertainty is unknown and each acquired label informs both estimation and subsequent allocation. For known difficulty levels, the benchmark follows a cost-adjusted Neyman rule that accounts for task importance and label cost. Within our Tuned Rectification-Difficulty framework, we develop an upper-confidence-bound policy that jointly learns the correction coefficients of tuned prediction-powered inference and the resulting difficulties from paired human labels and predictions, and establish an allocation-regret guarantee for the confidence-radius policy. In synthetic experiments and a real digital-twin survey with 68 tasks and over 2,000 respondents, our method closes most of the gap to the oracle on the 
    
[^45]: 对比学习、重要性采样和桥接采样在基于能量的模型中的统一视角

    A unifying view of contrastive learning, importance sampling, and bridge sampling for energy-based models

    [https://arxiv.org/abs/2604.08116](https://arxiv.org/abs/2604.08116)

    本文提出了一个统一框架，将噪声对比估计、反向逻辑回归、多重重要性采样和桥接采样在基于能量的模型中联系起来，揭示了它们之间的等价关系，并促进了更高效估计量的开发。

    

    在过去几十年中，基于能量的模型（EBMs）已成为一类重要的概率模型，其似然函数的一部分是难处理的，因此无法显式计算。因此，对于传统推断方法而言，EBMs中的参数估计具有挑战性。在这项工作中，我们提供了一个统一框架，将噪声对比估计（NCE）、反向逻辑回归（RLR）、多重重要性采样（MIS）和桥接采样在EBMs的背景下联系起来。我们进一步证明，这些方法在特定条件下是等价的。这种统一视角澄清了现有方法之间的关系，并使得新估计量的开发成为可能，有望提高统计和计算效率。此外，本研究有助于阐明NCE在灵活性和鲁棒性方面的成功，同时识别出其性能可能下降的场景。

    arXiv:2604.08116v2 Announce Type: replace-cross  Abstract: In the last decades, energy-based models (EBMs) have become an important class of probabilistic models in which a component of the likelihood is intractable and therefore cannot be evaluated explicitly. Consequently, parameter estimation in EBMs is challenging for conventional inference methods. In this work, we provide a unified framework that connects noise contrastive estimation (NCE), reverse logistic regression (RLR), multiple importance sampling (MIS), and bridge sampling within the context of EBMs. We further show that these methods are equivalent under specific conditions. This unified perspective clarifies relationships among existing methods and enables the development of new estimators, with the potential to improve statistical and computational efficiency. Furthermore, this study helps elucidate the success of NCE in terms of its flexibility and robustness, while also identifying scenarios in which its performance c
    
[^46]: 图能量匹配：面向图生成的传输对齐能量建模方法

    Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation

    [https://arxiv.org/abs/2603.23398](https://arxiv.org/abs/2603.23398)

    提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。

    

    arXiv:2603.23398v4 公告类型：replace-cross。摘要：离散数据（如图）的生成建模是众多科学与工业应用的基础，包括分子发现与材料设计。在这些领域中，概率推断尤为宝贵，因为它支持可组合的生成，并能有原则性地融入期望的约束，如结构或功能性质。基于能量的模型通过捕捉相对似然，并允许在推断时直接施加约束，天然契合这一目标。然而，离散能量模型通常难以实现高效且高质量的采样：支持集之外的区域往往存在虚假的局部极小值，会使采样器陷入其中并导致训练不稳定，从而使其与离散扩散模型相比存在保真度差距。为弥合这一差距，我们提出了图能量匹配，一种受 Jordan-Kinderlehrer-Otto 输运机制启发的离散生成框架。（注：原文摘要在此处被截断）

    arXiv:2603.23398v4 Announce Type: replace-cross  Abstract: Generative modeling of discrete data, such as graphs, underpins many scientific and industrial applications, including molecular discovery and materials design. In these domains, probabilistic inference is particularly valuable, as it enables composable generation and principled incorporation of desired constraints, such as structural or functional properties. Energy-based models naturally support this goal by capturing relative likelihoods and enabling composable inference by directly enforcing constraints during inference. However, discrete energy-based models typically struggle with efficient and high-quality sampling, as off-support regions often contain spurious local minima, trapping samplers and causing training instabilities, resulting in a fidelity gap compared to discrete diffusion models. To address this gap, we introduce Graph Energy Matching (GEM), a discrete generative framework inspired by the Jordan-Kinderlehrer
    
[^47]: 对决老虎机中孔多塞胜者识别的采样复杂度

    The Sampling Complexity of Condorcet Winner Identification in Dueling Bandits

    [https://arxiv.org/abs/2603.15189](https://arxiv.org/abs/2603.15189)

    提出一种利用完整间隔矩阵的孔多塞胜者识别算法，其样本复杂度在所有情形下达到最优（上下界匹配），并揭示了采样复杂度由定位有信息比较的成本所刻画的权衡结构。

    

    我们在大规模随机对决老虎机中研究最优臂识别问题，仅假设存在孔多塞胜者，即以至少 1/2 的概率赢得每次含噪两两比较的臂。我们提出了一种新的识别方法，该方法利用完整的间隔矩阵 $\Delta_{i,j}=q_{i,j}-\tfrac{1}{2}$（其中 $q_{i,j}$ 是臂 $i$ 击败臂 $j$ 的概率），而不仅仅利用孔多塞胜者与其他臂之间的间隔。我们推导出了高概率、依赖于实例的样本复杂度保证，通过利用涉及胜者之外的有信息比较，在（对数因子上）改进了已知最好的结果。我们用匹配的下界补充了这些结果，证明了我们的方法在所有情形下的最优性。总体而言，我们的结果揭示了采样复杂度的一般形式，其特征在于定位有信息比较的成本与其他代价之间的权衡。

    arXiv:2603.15189v2 Announce Type: replace-cross  Abstract: We study best-arm identification in large-scale stochastic dueling bandits under the sole assumption that a Condorcet winner exists, i.e., an arm that wins each noisy pairwise comparison with probability at least $1/2$. We introduce a new identification procedure that exploits the full gap matrix $\Delta_{i,j}=q_{i,j}-\tfrac12$ (where $q_{i,j}$ is the probability that arm $i$ beats arm $j$), rather than only the gaps between the Condorcet winner and the other arms. We derive high-probability, instance-dependent sample-complexity guarantees that (up to logarithmic factors) improve the best known ones by leveraging informative comparisons beyond those involving the winner. We complement these results with matching lower bounds that establish the optimality of our procedures in all regimes. Overall, our results reveal the general form of the sampling complexity, characterized by a trade-off between the cost of locating informative
    
[^48]: 基于Gossip算法的去中心化排名聚合：收敛性与鲁棒性

    Decentralized Ranking Aggregation via Gossip: Convergence and Robustness

    [https://arxiv.org/abs/2602.22847](https://arxiv.org/abs/2602.22847)

    本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。

    

    排名聚合的概念在偏好分析中扮演着核心角色。文献中已记录了大量用于计算中位数排名的算法，这些算法通常源于社会选择理论，并在集中式设置下（即所有待聚合的排名数据能够汇集到单一计算单元时）提供理论保证。对于许多技术（例如点对点网络、物联网、多智能体系统）而言，在去中心化环境下——即偏好数据最初分布在通信网络中时——扩展计算共识排名的能力，同时保证收敛性以及对潜在污染数据的鲁棒性，仍然是一个重大的方法论挑战。事实上，近年来关于去中心化计算的文献主要集中于使用gossip算法来计算或优化诸如算术平均值之类的统计量。本文的目的……（原文摘要在此处截断）

    arXiv:2602.22847v3 Announce Type: replace-cross  Abstract: The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, \textit{i.e.}, when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (\textit{e.g.} peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees of convergence and resilience to potential contamination in a decentralized setting, when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of 
    
[^49]: LoBoost：面向梯度提升树的模型原生局部保形预测

    LoBoost: Model-Native Local Conformal Prediction for Gradient-Boosted Trees

    [https://arxiv.org/abs/2602.22432](https://arxiv.org/abs/2602.22432)

    LoBoost通过重用梯度提升树集成模型自身的叶结构构建多尺度嵌套分组，实现模型原生的局部保形预测，无需辅助划分、重训练或额外数据分割即可局部估计残差分位数并适应异方差性。

    

    梯度提升决策树是表格回归中最强大的现成预测器之一，但仅靠点预测无法量化不确定性。保形预测可以提供无分布的边际覆盖保证，然而标准的分割保形方法只使用单一的全局残差分位数，难以适应异方差性。我们提出LoBoost，一种模型原生的局部保形方法，它重用已拟合集成模型的叶结构来定义特征空间的多尺度划分。每个输入由其沿提升路径所经过的叶节点序列来表示，匹配的叶前缀定义了嵌套分组，并在这些分组内局部估计残差分位数。通过重用模型已经学习到的预测结构，LoBoost无需辅助划分或干扰模型，无需重新训练，也无需在标准保形校准之外进行额外的数据分割。我们的理论将拟合的稳定性……（摘要在此处截断）

    arXiv:2602.22432v3 Announce Type: replace-cross  Abstract: Gradient-boosted decision trees are among the strongest off-the-shelf predictors for tabular regression, but point predictions alone do not quantify uncertainty. Conformal prediction provides distribution-free marginal coverage, yet standard split conformal uses a single global residual quantile and can adapt poorly to heteroscedasticity. We propose LoBoost, a model-native local conformal method that reuses the fitted ensemble's leaf structure to define a multiscale partition of the feature space. Each input is represented by the sequence of leaves it visits along the boosting path, and matching leaf prefixes define nested groups in which residual quantiles are estimated locally. By reusing the predictive structure already learned by the model, LoBoost requires no auxiliary partition or nuisance model, no retraining, and no additional data split beyond standard conformal calibration. Our theory connects the stability of the fit
    
[^50]: 基于近似正交基的低阶多项式下界

    Low-degree lower bounds via almost orthonormal bases

    [https://arxiv.org/abs/2509.09353](https://arxiv.org/abs/2509.09353)

    本文提出了一种基于近似正交多项式基的更直接证明策略，用于在具有植入结构的随机图模型中建立低阶多项式下界，克服了传统规范正交多项式族方法在估计任务和复杂检验问题中的局限性。

    

    低阶多项式已成为在多种高维统计模型中为统计-计算差距提供证据的有力范式[Wein25]。对于检测问题——即目标是将植根分布 $\mathbb{P}'$ 与具有独立成分的零假设分布 $\mathbb{P}$ 进行比较检验——标准方法是使用 $\mathbb{L}^2(\mathbb{P})$-规范正交多项式族来界定优势。然而，对于估计任务或更复杂的检验问题（其中 $\mathbb{P}$ 具有某种植入结构，因此不存在简单的 $\mathbb{L}^2(\mathbb{P})$-正交多项式族），这种方法会失效。为应对这一挑战，已有若干技术性变通方案被提出[SW22,SW25]，但其实现较为繁琐。在本工作中，我们提出了一种更直接的证明策略。聚焦于随机图模型，我们构造了一个多项式基

    arXiv:2509.09353v3 Announce Type: replace-cross  Abstract: Low-degree polynomials have emerged as a powerful paradigm for providing evidence of statistical--computational gaps across a variety of high-dimensional statistical models [Wein25]. For detection problems---where the goal is to test a planted distribution $\mathbb{P}'$ against a null distribution $\mathbb{P}$ with independent components---the standard approach is to bound the advantage using an $\mathbb{L}^2(\mathbb{P})$-orthonormal family of polynomials. However, this method breaks down for estimation tasks or more complex testing problems where $\mathbb{P}$ has some planted structures, so that no simple $\mathbb{L}^2(\mathbb{P})$-orthogonal polynomial family is available. To address this challenge, several technical workarounds have been proposed [SW22,SW25], though their implementation can be delicate. In this work, we propose a more direct proof strategy. Focusing on random graph models, we construct a basis of polynomials
    
[^51]: 我们真的需要Rademacher复杂度吗？

    Do we really need the Rademacher complexities?

    [https://arxiv.org/abs/2502.15118](https://arxiv.org/abs/2502.15118)

    该论文证明只需利用函数类协方差结构的粗略信息，即可在重尾情形下推导出与高斯情形最先进速率相匹配的样本复杂度上界，从而表明学习问题无需依赖难以控制的局部化Rademacher复杂度。

    

    我们研究了在凸函数类中关于平方损失学习这一基本问题。在次高斯设定下，已知最尖锐的样本复杂度界是由与极限高斯过程相关联的不动点所刻画的。而在更一般的情形下（例如重尾场景），最先进的界依赖于局部化Rademacher复杂度，这类复杂度通常难以控制，且可能远大于其对应的极限高斯复杂度。我们证明，只要能够获得关于底层函数类协方差结构的粗略信息，就可以推导出样本复杂度的上界，该上界在高斯情形下能达到最先进的速率——即使是在重尾情形下同样成立。因此，在相当弱的最小假设下，具有相同L2结构的学习问题——即便是涉及重尾分布的问题——也拥有相同的上界。

    arXiv:2502.15118v2 Announce Type: replace-cross  Abstract: We study the fundamental problem of learning with respect to the squared loss in a convex class. In the subgaussian setting, the sharpest known sample complexity bounds are characterised via fixed points associated with the limiting gaussian process. In more general settings (for example, heavier-tailed scenarios) the state-of-the-art bounds rely on localised Rademacher complexities, which are often difficult to control and can be considerably larger than their limiting gaussian counterparts.   We show that if one has access to coarse information on the covariance structure of the underlying class, it is possible to derive an upper bound on the sample complexity that matches the state-of-the-art rates in the gaussian setting---even in heavy-tailed regimes. As a consequence, under rather minimal assumptions, learning problems that share the same $L_2$-structure---even those involving heavy-tailed distributions---admit the same u
    
[^52]: 基于评论器的正则化器学习中的星形几何

    The Star Geometry of Critic-Based Regularizer Learning

    [https://arxiv.org/abs/2408.16852](https://arxiv.org/abs/2408.16852)

    该论文开创性地研究了在星形体的规范函数这一正则化器族上优化基于评论器的损失函数，为学习到的任务相关正则化器的结构及其与测量数据和真实数据分布之间的几何联系提供了首个理论视角。

    

    变分正则化是解决统计推断任务和逆问题的一种经典技术，而现代数据驱动方法通过深度神经网络对正则化器进行参数化，展现出令人瞩目的实证性能。沿此思路的最新研究学习依赖于具体任务的正则化器：其做法是在一个无监督的、基于评论器的损失函数中融合测量数据与真实数据的信息，其中正则化器为可能的数据赋予较低的值，为不太可能的数据赋予较高的值。然而，关于通过这一过程学习到的正则化器具有何种结构，以及它与两种数据分布之间的关系，目前几乎没有相关理论。为了在这一挑战上取得进展，我们开创性地研究了在特定正则化器族——星形体的规范函数（gauge，即 Minkowski 泛函）——上优化基于评论器的损失函数以学习正则化器。该正则化器族包含了常见的……

    arXiv:2408.16852v3 Announce Type: replace  Abstract: Variational regularization is a classical technique to solve statistical inference tasks and inverse problems, with modern data-driven approaches parameterizing regularizers via deep neural networks showcasing impressive empirical performance. Recent works along these lines learn task-dependent regularizers. This is done by integrating information about the measurements and ground-truth data in an unsupervised, critic-based loss function, where the regularizer attributes low values to likely data and high values to unlikely data. However, there is little theory about the structure of regularizers learned via this process and how it relates to the two data distributions. To make progress on this challenge, we initiate a study of optimizing critic-based loss functions to learn regularizers over a particular family of regularizers: gauges (or Minkowski functionals) of star-shaped bodies. This family contains regularizers that are common
    
[^53]: 通用线性混合模型的可扩展贝叶斯推断

    Scalable Bayesian inference for the generalized linear mixed model

    [https://arxiv.org/abs/2403.03007](https://arxiv.org/abs/2403.03007)

    该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。

    

    通用线性混合模型（GLMM）是处理相关数据的一种流行统计方法，在包括生物医学数据等大数据常见的应用领域被广泛使用。本文的重点是针对GLMM的可扩展统计推断，我们将统计推断定义为：（i）对总体参数的估计以及（ii）在存在不确定性的情况下评估科学假设。人工智能（AI）学习算法擅长可扩展的统计估计，但很少包括不确定性量化。相比之下，贝叶斯推断提供完整的统计推断，因为不确定性量化自动来自后验分布。不幸的是，包括马尔可夫链蒙特卡洛（MCMC）在内的贝叶斯推断算法在大数据环境中变得难以计算。在本文中，我们介绍了一个统计推断算法

    arXiv:2403.03007v1 Announce Type: cross  Abstract: The generalized linear mixed model (GLMM) is a popular statistical approach for handling correlated data, and is used extensively in applications areas where big data is common, including biomedical data settings. The focus of this paper is scalable statistical inference for the GLMM, where we define statistical inference as: (i) estimation of population parameters, and (ii) evaluation of scientific hypotheses in the presence of uncertainty. Artificial intelligence (AI) learning algorithms excel at scalable statistical estimation, but rarely include uncertainty quantification. In contrast, Bayesian inference provides full statistical inference, since uncertainty quantification results automatically from the posterior distribution. Unfortunately, Bayesian inference algorithms, including Markov Chain Monte Carlo (MCMC), become computationally intractable in big data settings. In this paper, we introduce a statistical inference algorithm 
    

