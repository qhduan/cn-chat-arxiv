# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provably adaptive sampling with uniform and remasking discrete diffusion models](https://arxiv.org/abs/2608.23554) | 本文证明了均匀和重掩码离散扩散模型的采样器可实现自适应采样，其复杂度仅依赖于数据的内在复杂度而非环境维度，并能并行纠正去噪错误。 |
| [^2] | [ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings](https://arxiv.org/abs/2608.23551) | 本文提出ConvergeFlow，一种基于嵌入空间的流式语言模型，通过约束数据预测器到词元嵌入凸包并仅用均方误差训练，证明了流可收敛到有效词元嵌入，从而消除了对交叉熵解码器的需求。 |
| [^3] | [Interpretable AI with Local Distillation](https://arxiv.org/abs/2608.23538) | 本文提出局部蒸馏方法，利用黑盒教师模型在每个查询点指导正则化线性学生模型，通过定义局部性和锚定预测来实现高精度与可解释性的兼顾。 |
| [^4] | [Primal--Dual Alternating Neural Learning for Timely Classification with Performance Guarantees](https://arxiv.org/abs/2608.23480) | 本文提出一种原对偶交替神经学习方法，在满足敏感性和监测成本约束下最大化特异性，为及时临床分类提供性能保证的决策规则。 |
| [^5] | [Strong Averaging Principle and Long-Time Dynamics for Fast-Slow SDEs with Increasing Time-Scale Separation and Degenerate Noise](https://arxiv.org/abs/2608.23462) | 本文提出一种基于耗散性的强平均原理方法，适用于时间尺度分离递增且噪声退化的快速-慢速随机微分方程，获得慢变量与平均ODE的强收敛率，并建立长期动态的渐近伪轨迹性质。 |
| [^6] | [Traceable Spectral Inference via Influence Functions: Efficient Data Attribution and Error Proxies for the Ariel Mission](https://arxiv.org/abs/2608.23458) | 本工作提出了一种基于影响函数的数据归因方法，通过预测影响、高效计算和误差代理三项创新，为Ariel任务提供了无标签可追溯的光谱推断与误差评估方案。 |
| [^7] | [Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction](https://arxiv.org/abs/2608.23356) | 本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。 |
| [^8] | [Credal Large Language Models for Semantic Commitment under Uncertainty](https://arxiv.org/abs/2608.23244) | 通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。 |
| [^9] | [One Inverse Step is a Convex Program: Bayes-Limit Calibration of Diffusion Inversion](https://arxiv.org/abs/2608.23094) | 本文证明了一步DDIM逆过程在贝叶斯极限下等价于一个强凸优化问题，其唯一性、误差界和收缩性均不依赖数据分布假设，但求解器稳定性受势函数曲率限制。 |
| [^10] | [Neural Boltzmann Equations](https://arxiv.org/abs/2608.23022) | 本文提出了神经玻尔兹曼方程（NBEs），通过结合神经分布函数、蒙特卡洛重要性采样和自然梯度法，克服了传统玻尔兹曼方程求解中高维积分和参数扫描的瓶颈，实现了高效且精确的宇宙学动力学模拟。 |
| [^11] | [Stochastic gradient descent with initial regularization](https://arxiv.org/abs/2608.22953) | 本文分析了带初始正则化的随机梯度下降法，在无噪声和有噪声情况下推导了期望超额风险的上界和下界，并与岭回归进行了比较。 |
| [^12] | [A Commutator Framework for Selective Spectral Alignment in Deep Neural Networks](https://arxiv.org/abs/2608.22910) | 该论文提出了一个交换子框架，精确量化深度神经网络中特征几何对齐的机制，并揭示了内部几何与特征侧对齐之间的传输关系。 |
| [^13] | [Change Detection in Probability Flow ODE: Online Testing in Diffusion Latent Spaces](https://arxiv.org/abs/2608.22807) | 本文提出一种基于条件扩散模型和概率流常微分方程的在线变化点检测方法，通过将观测映射到标准高斯潜在空间并利用最大均值差异检测分布变化，适用于分布无闭合形式的情况。 |
| [^14] | [Generative Neural Networks for Sinkhorn Distributionally Robust Hypothesis Testing](https://arxiv.org/abs/2608.22746) | 本文提出了一种基于生成式神经网络的框架，通过推导Sinkhorn差异模糊集的等价条件KL散度表示并利用强对偶性，将SDRHT问题转化为可扩展的凸势最大化问题，实现了高效训练和端到端采样。 |
| [^15] | [Q-Learning with Stable Infinite-Dimensional Linear Function Approximation](https://arxiv.org/abs/2608.22636) | 该论文提出了一种基于非扩张重建和压缩算子的无限维线性函数近似框架，确保Q学习在马尔可夫轨迹下的稳定收敛，并给出了两种随机逼近算法的超范数收敛界。 |
| [^16] | [Scale-invariant Optimal Sampling for Rare-events Data with Sparse Models](https://arxiv.org/abs/2608.22597) | 提出了一种尺度不变的最优子采样方法，通过最小化预测误差并利用自适应lasso，有效解决了稀有事件数据中非活跃特征受缩放变换影响的问题。 |
| [^17] | [Sparse Additive Off-Policy Evaluation for Reinforcement Learning with Potentially Limited Number of Trajectories](https://arxiv.org/abs/2608.22595) | 本文提出了一种稀疏加性离策略评估方法，在轨迹数量或时间范围有限的情况下，通过非线性稀疏建模和组稀疏特征筛选，实现了误差界仅对数依赖维度的高效价值估计。 |
| [^18] | [Recovering Weighted Tangent Geometry from a Single-Scale Score Field](https://arxiv.org/abs/2608.22334) | 本文提出一种方法，在未知分支中心和齐次度的情况下，仅利用单一噪声水平的分数场，通过弱形式的Ornstein-Uhlenbeck方程和球面分数积分，恢复加权切向几何及其标量高斯-锥变换。 |
| [^19] | [Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN](https://arxiv.org/abs/2608.22223) | 本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。 |
| [^20] | [Joint Causal Structure and Cluster Discovery Using Variational Inference](https://arxiv.org/abs/2608.22212) | 本文提出了一种基于变分推断的新方法，能够同时推断潜在变量聚类及其因果结构，无需预先知道聚类信息。 |
| [^21] | [Token-Level Likelihood-Array Regression for Membership Inference and AI-Generated Text Detection](https://arxiv.org/abs/2608.22179) | 提出似然数组回归（LAR）方法，通过嵌套上下文窗口评估词元似然并组织成结构化数组，显著提升成员推断和AI生成文本检测的准确性。 |
| [^22] | [Symbolic Neural ODEs: Learning interpretable models from time-series data](https://arxiv.org/abs/2608.22112) | 本文提出一种符号神经ODE框架，通过多步预测和稀疏正则化，从时间序列数据中学习稳定且可解释的动力学模型。 |
| [^23] | [Structured Learning on Mapper Representations](https://arxiv.org/abs/2608.22044) | 本文提出了一种将Mapper构造作为完整表示的一部分进行学习的框架，并研究了其数学性质，如重标号不变性和距离泛函，以捕捉数据的多尺度结构组织。 |
| [^24] | [Barycentric Fused Gromov-Wasserstein Balancing for Causal Inference under Multiple Treatments](https://arxiv.org/abs/2608.22024) | 提出CIHSI-Net框架，利用重心融合Gromov-Wasserstein平衡目标实现全局对齐，降低计算复杂度，提升多重处理下因果推断的准确性和效率。 |
| [^25] | [Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments](https://arxiv.org/abs/2608.21995) | 我们提出了一种方差驱动的探索方法论，通过最小化最终决策不确定性来分配采样资源，在高随机环境中显著提升了纯探索任务的效率与理论保证。 |
| [^26] | [Improved denoising diffusion probabilistic models with efficient non-diagonal covariance modeling](https://arxiv.org/abs/2608.21972) | 本文提出了一种名为K-DCT的协方差模型，通过Kronecker分解和离散余弦变换高效建模非对角协方差，从而在更少采样步骤中加速DDPM生成，同时保持样本质量。 |
| [^27] | [Guidance for Prior Change via Density Ratio Estimation](https://arxiv.org/abs/2608.21729) | 提出一种基于密度比估计的无偏测试时引导框架，通过学习得分引导项有效解耦推断过程与先验训练，避免了先验依赖和系统性偏差。 |
| [^28] | [GeoQ: Geometry-Aware Conditional Quantile Error Estimation for Scientific Surrogate Models](https://arxiv.org/abs/2608.21652) | GeoQ提出了一种几何感知的非侵入式校准方法，通过锚点平均误差加条件分位数修正，实现了代理模型在查询点的准确误差估计。 |
| [^29] | [Subzero matrix completion for sparse data analysis: large-scale learning of latent low-rank structure](https://arxiv.org/abs/2608.21607) | 本文提出一种随机交替最小二乘算法，通过处理稠密矩阵的小块并利用稀疏优化和CUDA加速，实现了大规模稀疏数据中潜在低秩结构的有效恢复。 |
| [^30] | [Random Hazard Forests](https://arxiv.org/abs/2608.21597) | 随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。 |
| [^31] | [Sparse Separable Factor Analysis in the Complex Domain with an Application to Local Field Potential Data](https://arxiv.org/abs/2608.21551) | 本文提出了一种复域中的稀疏可分离因子分析模型，通过复数软阈值和期望最大化算法，有效利用了复值数组的幅度和相位信息，实现了可解释的协方差估计。 |
| [^32] | [The geometry of AI validation: Exact certification limits for iid best-of-N search](https://arxiv.org/abs/2608.21496) | 本文通过核几何方法，精确推导出独立同分布最佳N次搜索中验证的模糊宽度公式，并揭示其主导尺度为$m^2/N$，为AI验证提供了理论极限。 |
| [^33] | [A Data-Driven Approach to State Construction in Markov Models](https://arxiv.org/abs/2608.21480) | 本文提出了一种数据驱动的状态构建方法，通过结合监督特征选择与无监督学习技术，无需先验假设即可识别马尔可夫模型中的同质状态，从而提高模型的效度和预测能力。 |
| [^34] | [Gauss--Hermite Quadrature for Gaussian-Mixture Entropy with an Action-Space Hermite Surrogate](https://arxiv.org/abs/2608.21467) | 本文提出了一种利用高斯-埃尔米特求积法高效计算高斯混合微分熵的方法，并引入动作空间的埃尔米特多项式代理以提升连续优化性能，显著优于传统泰勒近似。 |
| [^35] | [Spectral partitioning for $k$-block averaging kernels of finite Markov chains](https://arxiv.org/abs/2608.21466) | 本文提出基于谱划分的算法来选择状态空间划分，通过加权k-means舍入特征函数构造平均核，并证明目标函数等价于Pearson卡方互信息，从而加速有限可逆马尔可夫链的收敛。 |
| [^36] | [Sobolev Regularized Score Difference Estimation in Diffusion Models](https://arxiv.org/abs/2608.18237) | 本文提出了一种基于Sobolev正则化的得分差异估计方法，既保证了统计一致性又支持高维扩展，并给出了明确的收敛速率和极小极大下界。 |
| [^37] | [Deep adaptive design with an evidential bias criterion](https://arxiv.org/abs/2608.16466) | 本文提出了一种基于“反偏置”证据准则的深度自适应实验设计方法，以更好地控制实验产生误导性证据的风险，弥补传统期望信息增益准则的不足。 |
| [^38] | [Rethinking Reverse KL as Adaptive Entropy Distillation](https://arxiv.org/abs/2608.14685) | 本文提出自适应熵蒸馏（AED），通过重新分解反向KL目标为教师拟合和学生熵项，利用教师熵动态调整蒸馏权重，实现更优的模仿与生成平衡。 |
| [^39] | [Tensor-normal maximum likelihood estimation at the operator-norm sample threshold](https://arxiv.org/abs/2608.10488) | 本文证明了张量正态最大似然估计在样本量条件可降低至$d_{\max}$的二次依赖，并提供了相应的误差界。 |
| [^40] | [On Non-Stationary Dynamic Pricing: Adaptivity and Optimality](https://arxiv.org/abs/2607.24115) | 本文提出了一种无需预先知道分段数或变化预算的自适应多尺度变点检测算法，在非平稳上下文动态定价中实现了接近最优的遗憾界，并引入了新的设计调整变化预算概念。 |
| [^41] | [CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference](https://arxiv.org/abs/2607.22511) | CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。 |
| [^42] | [How Fast Do Signatures Learn? Statistical Theory and Applications for Path Regression](https://arxiv.org/abs/2607.17865) | 本文首次量化了基于路径签名的回归模型在截断级别增加时的逼近误差收敛速度，并证明了其最小最大最优性及三种统计学习过程的一致性。 |
| [^43] | [What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs](https://arxiv.org/abs/2606.28615) | 本文提出了一种基于信息论的指标SCSuff，利用LLM自身生成替代输入来评估自由文本解释的充分性，无需预设偏见，并证明解释充分性依赖于输入分布。 |
| [^44] | [A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development](https://arxiv.org/abs/2606.19230) | 本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。 |
| [^45] | [Relational Structural Causal Models](https://arxiv.org/abs/2606.14892) | 本文提出关系结构因果模型，通过定义关系因果图和符号识别标准，解决了在对象和关系变化场景下对未观测混杂因素进行因果与观测查询识别的难题，并验证了关系神经因果模型的有效性。 |
| [^46] | [Conformal Prediction for Dyadic Regression Under Complex Missingness](https://arxiv.org/abs/2606.11136) | 本文提出了一个在复杂缺失机制下用于二元回归的共形预测框架，通过新颖的双射论证和多种程序（如行列方法和选择性共形）实现了有限样本有效性和掩码条件有效性。 |
| [^47] | [Asymptotic Optimality of Thompson Sampling for Risk-Averse Bandits with Sub-Gaussian Rewards](https://arxiv.org/abs/2606.09191) | 本文证明了非参数汤普森采样算法在仅需连续风险泛函条件下，对次高斯奖励的风险厌恶多臂老虎机达到渐近最优遗憾，首次为非Lipschitz风险度量（如夏普比率）提供实例最优保证。 |
| [^48] | [Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions](https://arxiv.org/abs/2606.07399) | ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。 |
| [^49] | [Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](https://arxiv.org/abs/2606.00984) | 本文提出两种仅需$O(\log\log T)$次参数更新的线性上下文赌博机算法，在静态调度下同时在小规模和大规模动作集下实现极小极大最优遗憾，并澄清了批处理与稀有更新的实际区别。 |
| [^50] | [GraphSVR: A Graph Convolutional Support Vector Regression Framework for Robust Spatiotemporal Air Pollution Forecasting](https://arxiv.org/abs/2605.03795) | 本文提出了一个结合图卷积和支持向量回归的GraphSVR框架，用于鲁棒的城市空气污染时空预测，有效处理非线性动态和异常值。 |
| [^51] | [Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence](https://arxiv.org/abs/2602.11333) | 本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。 |
| [^52] | [You Need Better Attention Priors](https://arxiv.org/abs/2601.15380) | 该论文通过熵最优传输统一了注意力机制，提出GOAT，用可学习先验替代均匀先验，兼容FlashAttention，解决注意力汇问题，并实现长度泛化。 |
| [^53] | [Stability and Accuracy Trade-offs in Statistical Estimation](https://arxiv.org/abs/2601.11701) | 本文从统计决策论角度，将稳定性视为估计约束，探讨了最坏情况和平均情况稳定性与准确性之间的权衡，揭示了稳定性带来的统计成本。 |
| [^54] | [Radial Compensation: The Inverse Base-Distribution Problem for Chart-Based Generative Models on Riemannian Manifolds](https://arxiv.org/abs/2511.14056) | 本文揭示了球面和双曲空间中潜在变量模型的标准构造会无意中强制固定距离分布，并提出了一种闭式径向补偿方法，以实现任意预期距离分布，同时确保图表无关似然性，为变分自编码器提供了理论下界和实验验证。 |
| [^55] | [DIGing--SGLD: Decentralized and Scalable Langevin Sampling over Time--Varying Networks](https://arxiv.org/abs/2511.12836) | 本文提出DIGing-SGLD算法，首次将梯度跟踪机制与朗之万采样结合，在时变网络上实现了无偏且可扩展的去中心化贝叶斯采样，解决了现有方法仅适用于静态网络且存在网络效应偏差的问题。 |
| [^56] | [Learning discrete Bayesian networks with hierarchical Dirichlet shrinkage](https://arxiv.org/abs/2509.13267) | 本文提出了一种层次狄利克雷收缩方法，通过后验收缩到低维潜在参数来减少离散贝叶斯网络中的参数数量，并利用对数凹性实现高效采样，同时保持DAG结构。 |
| [^57] | [Prob-GParareal: A Probabilistic Numerical Parallel-in-Time Solver for Differential Equations](https://arxiv.org/abs/2509.03945) | 本文提出了Prob-GParareal，一种概率数值并行时间求解器，通过高斯过程建模Parareal校正函数，实现微分方程求解中的不确定性量化和概率预测，并支持概率初始条件及与现有框架的无缝集成。 |
| [^58] | [Distributional Sensitivity Analysis: Enabling Differentiability in Sample-Based Inference](https://arxiv.org/abs/2508.09347) | 本文提出了一种用于估计随机样本对分布参数敏感性的数学框架，并提供了两种解析公式和四种数值算法，以在基于样本的推断中实现可微性。 |
| [^59] | [One-shot Robust Federated Learning of Independent Component Analysis](https://arxiv.org/abs/2505.20532) | 提出了一种基于谱聚类和几何中位数的联邦ICA一次性聚合方法，有效解决了符号置换和异构质量问题，并在大量低质量数据下保持鲁棒性。 |
| [^60] | [Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching](https://arxiv.org/abs/2502.14424) | 本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。 |
| [^61] | [Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference](https://arxiv.org/abs/2501.06926) | 本文提出一种半参数双重强化学习方法，通过直接对Q函数施加工作性半参数限制，以提升长期因果推断中的估计效率与稳定性，特别是在时间重叠较弱的情况下。 |
| [^62] | [Conditional regression for the Nonlinear Single-Variable Model](https://arxiv.org/abs/2411.09686) | 本文提出了一种针对非线性单变量组合模型的条件回归估计方法，通过响应切片和局部主成分分析克服了高维预测变量下的维度灾难。 |
| [^63] | [Cross-validating causal discovery via Leave-One-Variable-Out](https://arxiv.org/abs/2411.05625) | 本文提出一种无需真实因果图的留一变量法（LOVO）预测方法，通过分别对排除一个变量的数据集进行因果发现，实现对因果发现算法的交叉验证，并能估计条件期望而不需联合观测。 |
| [^64] | [Robust performance metrics for imbalanced classification problems](https://arxiv.org/abs/2404.07661) | 本文提出了一种通过引入调整参数来鲁棒化MCC、科恩κ和F分数等性能指标的方法，以确保在不平衡分类问题中分类器不会忽略少数类别。 |
| [^65] | [Deep Clustering Evaluation: How to Validate Internal Clustering Validation Measures](https://arxiv.org/abs/2403.14830) | 本文解决了深度聚类方法在评估聚类质量时面临的挑战，提出了一种系统方法来应用聚类有效性指标。 |
| [^66] | [Physically-based dimensionless features for pluvial flood mapping with machine learning](https://arxiv.org/abs/2211.00636) | 本文提出一种基于无量纲多尺度特征和逻辑回归的机器学习框架，通过捕捉洪水过程的跨区域相似性，显著提升暴雨洪水绘图的泛化能力和预测效率。 |
| [^67] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |
| [^68] | [Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback.](http://arxiv.org/abs/2307.09295) | 嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。 |
| [^69] | [Neuro-Causal Factor Analysis.](http://arxiv.org/abs/2305.19802) | 该论文提出了一种名为神经因果因素分析（NCFA）的新方法，它通过学习到的图形匹配马尔可夫因式分解的分布来识别因素，并使用变分自编码器（VAE）对数据进行重建任务。与标准VAE相比，NCFA具有更稀疏的架构和低模型复杂度，具有因果解释性。 |
| [^70] | [Signal identification without signal formulation.](http://arxiv.org/abs/2304.06522) | 该研究提出了一种无需信号建模即可识别信号的方法，该方法基于样本和其邻居之间相对距离，可以在小样本和高维数据中识别“类似于信号”的变量。 |

# 详细

[^1]: 可证明自适应的均匀与重掩码离散扩散模型采样方法

    Provably adaptive sampling with uniform and remasking discrete diffusion models

    [https://arxiv.org/abs/2608.23554](https://arxiv.org/abs/2608.23554)

    本文证明了均匀和重掩码离散扩散模型的采样器可实现自适应采样，其复杂度仅依赖于数据的内在复杂度而非环境维度，并能并行纠正去噪错误。

    

    arXiv:2608.23554v1 公告类型：交叉 摘要：离散扩散模型通过支持并行更新，为自回归生成提供了一种有前景的替代方案，但其采样效率可能强烈依赖于前向过程和采样器的选择。对于均匀前向过程，现有标准τ跳跃采样器的下界随环境维度d线性增长，这引发了该依赖性是否前向过程固有的问题。我们对此问题给出了否定回答。我们考虑了一种基于留一法去噪器的一阶采样器，适用于均匀和重掩码过程，其坐标更新可并行执行。在两种情况下，该采样器都能在采样过程中纠正去噪错误，这在许多坐标同时更新时变得必要。我们的主要结果建立了一个自适应采样保证：在忽略对数因子下，需要N = O(\mathrm{DTC}(X_0) / \varepsilon)个离散化步骤。

    arXiv:2608.23554v1 Announce Type: cross  Abstract: Discrete diffusion models offer a promising alternative to autoregressive generation by enabling parallel updates, but their sampling efficiency can depend strongly on the choice of the forward process and the sampler. For the uniform forward process, existing lower bounds for the standard $\tau$-leaping sampler scale linearly with the ambient dimension $d$, raising the question of whether this dependence is intrinsic to the forward process. We answer this question in the negative. We consider a first-order sampler based on the leave-one-out denoiser for uniform and remasking processes whose coordinate updates can be performed in parallel. In both cases, the sampler can correct denoising mistakes during the sampling process, which becomes necessary when many coordinates are updated together. Our main result establishes an adaptive sampling guarantee: up to logarithmic factors, $N = O(\mathrm{DTC}(X_0) / \varepsilon)$ discretization ste
    
[^2]: 汇聚流：具有可证明收敛到词元嵌入的语言流

    ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings

    [https://arxiv.org/abs/2608.23551](https://arxiv.org/abs/2608.23551)

    本文提出ConvergeFlow，一种基于嵌入空间的流式语言模型，通过约束数据预测器到词元嵌入凸包并仅用均方误差训练，证明了流可收敛到有效词元嵌入，从而消除了对交叉熵解码器的需求。

    

    arXiv:2608.23551v1 公告类型：交叉 摘要：近期在连续扩散和基于流的语言模型（LMs）方面取得的进展，已达到与离散LMs竞争的性能。然而，现有的连续框架仍依赖于通过交叉熵（CE）监督的解码器，因为流轨迹不保证终止于有效的词元嵌入。受此局限性启发，我们引入了\textbf{ConvergeFlow}，一种基于嵌入空间的流式语言模型，它将数据预测器约束在词元嵌入的凸包内，并仅使用由流匹配引起的均方误差目标进行训练。在适当的正则性条件下，我们证明尽管数据预测器存在误差，所得到的流仍会收敛到有效的词元嵌入，从而无需CE监督的解码器即可实现直接词元预测。我们进一步开发了三种采样机制，用于控制生成困惑度与熵之间的权衡。在OpenWebText上的实验表明...

    arXiv:2608.23551v1 Announce Type: cross  Abstract: Recent advances in continuous diffusion and flow-based language models (LMs) have achieved performance competitive with discrete LMs. However, existing continuous frameworks still rely on decoders supervised with cross entropy (CE) because the flow trajectories are not guaranteed to terminate at valid token embeddings. Motivated by this limitation, we introduce \textbf{ConvergeFlow}, an embedding-space flow-based LM, which constrains the data predictor to the convex hull of token embeddings and trains it solely with the mean squared error objective induced by flow matching. Under suitable regularity conditions, we prove that the resulting flow converges to valid token embeddings despite errors in the data predictor, enabling direct token prediction without a CE-supervised decoder. We further develop three sampling mechanisms for controlling the trade-off between the generative perplexity and entropy. Experiments on OpenWebText demonstr
    
[^3]: 可解释人工智能的局部蒸馏方法

    Interpretable AI with Local Distillation

    [https://arxiv.org/abs/2608.23538](https://arxiv.org/abs/2608.23538)

    本文提出局部蒸馏方法，利用黑盒教师模型在每个查询点指导正则化线性学生模型，通过定义局部性和锚定预测来实现高精度与可解释性的兼顾。

    

    现代AI模型，如表格基础模型和梯度提升集成模型，在预测性能上优于经典方法，但对其预测的推理依据提供甚少。高风险决策要求模型既准确又具备内在可解释性。局部线性建模提供了一条前进之路：平滑回归函数在局部可由线性函数良好近似，使得在每个查询点附近的线性拟合能够在保持透明性的同时实现高精度。挑战在于学习什么是“局部”以及开发用于解释的统计工具。在此，我们提出局部蒸馏方法，其中黑盒“教师”模型在每个查询点指导一个正则化的线性“学生”模型。教师模型通过增加预测结果相似的训练观测的权重来定义局部性，并将其在查询点的预测作为伪观测包含进来以锚定拟合，该伪观测的权重会被估计。

    arXiv:2608.23538v1 Announce Type: cross  Abstract: Modern AI models such as tabular foundation models and gradient-boosted ensembles can outpredict classical methods, but provide little basis for reasoning about their predictions. High-stakes decisions call for models that are both accurate and interpretable as built. Local linear modeling offers a path forward: a smooth regression function is locally well approximated by a linear one, allowing a linear fit near each query point to achieve high accuracy without sacrificing transparency. The challenges lie in learning what is "local" and developing statistical tools for interpretation.   Here, we propose local distillation, in which a black-box "teacher" guides a regularized linear "student" model at each query point. The teacher (1) defines locality by upweighting training observations with similar predicted outcomes, and (2) anchors the fit with its prediction at the query point, included as a pseudo-observation whose weight is estima
    
[^4]: 具有性能保证的及时分类的原对偶交替神经学习

    Primal--Dual Alternating Neural Learning for Timely Classification with Performance Guarantees

    [https://arxiv.org/abs/2608.23480](https://arxiv.org/abs/2608.23480)

    本文提出一种原对偶交替神经学习方法，在满足敏感性和监测成本约束下最大化特异性，为及时临床分类提供性能保证的决策规则。

    

    及时的风险分类在许多临床监测环境中至关重要，其中决策必须平衡早期对患者进行分类以便后续干预的益处与观察更多数据的价值。然而，大多数现有的统计和机器学习方法是为完全观察到的轨迹设计的，并且对关键操作特征（如敏感性、特异性和监测成本）的控制有限。我们将序列分类问题置于一个针对这三个标准的多目标优化框架中。我们通过一个价值递归来刻画最优决策规则，该递归在每个时间点量化了立即分类与继续监测之间的权衡。为了从数据中估计该规则，我们制定了一个受约束的优化问题，在强制执行预设的敏感性和监测成本约束的同时最大化特异性。然后，我们开发了一种估计方法（原文截断，此处保留）。

    arXiv:2608.23480v1 Announce Type: new  Abstract: Timely risk classification is essential in many clinical monitoring settings, where decisions must balance the benefit of classifying patients early for subsequent intervention against the value of observing additional data. Yet most existing statistical and machine-learning methods are designed for fully observed trajectories and offer limited control over key operating characteristics such as sensitivity, specificity, and monitoring cost. We cast the sequential classification problem within a multi-objective optimization framework targeting these three criteria. We characterize the optimal decision rule through a value recursion that quantifies, at each time point, the trade-off between immediate classification and continued monitoring. To estimate the rule from data, we formulate a constrained optimization problem that maximizes specificity while enforcing prespecified sensitivity and monitoring-cost constraints. We then develop an es
    
[^5]: 快速-慢速随机微分方程中随时间尺度分离增强和退化噪声的强平均原理与长期动力学

    Strong Averaging Principle and Long-Time Dynamics for Fast-Slow SDEs with Increasing Time-Scale Separation and Degenerate Noise

    [https://arxiv.org/abs/2608.23462](https://arxiv.org/abs/2608.23462)

    本文提出一种基于耗散性的强平均原理方法，适用于时间尺度分离递增且噪声退化的快速-慢速随机微分方程，获得慢变量与平均ODE的强收敛率，并建立长期动态的渐近伪轨迹性质。

    

    摘要：arXiv:2608.23462v1 公告类型：交叉 摘要：我们为具有时间依赖尺度分离参数$(\varepsilon_t)_{t \geq 0}$（满足$\varepsilon_t \to 0$当$t \to \infty$）的快速-慢速随机微分方程建立了强平均原理。与基于噪声诱导平滑或椭圆正则性的方法不同，我们的方法依赖于冻结快速动态的耗散性，因此允许退化扩散系数。我们证明了在后期时间慢变量与平均常微分方程之间的最大$L^p$-估计，具有经典强收敛率阶数$1/2$。在$(\varepsilon_t)_{t \ge 0}$的额外衰减条件下，该估计意味着慢变量几乎必然成为平均常微分方程的渐近伪轨迹。因此，通过分析动态系统，我们获得了识别慢变量可能极限点以及收敛到渐近稳定平衡点的准则。

    arXiv:2608.23462v1 Announce Type: cross  Abstract: We establish a strong averaging principle for fast-slow stochastic differential equations with a time-dependent scale-separation parameter $(\varepsilon_t)_{t \geq 0}$ satisfying $\varepsilon_t \to 0$ as $t \to \infty$. In contrast to approaches based on noise-induced smoothing or elliptic regularity, our approach relies on dissipativity of the frozen fast dynamics and therefore permits degenerate diffusion coefficients. We prove a maximal $L^p$-estimate between the slow variable and the averaged ODE at late times, with the classical strong convergence rate of order $1/2$. Under an additional decay condition on $(\varepsilon_t)_{t \ge 0}$, this estimate implies that the slow variable is almost surely an asymptotic pseudo-trajectory of the averaged ODE. As a consequence, we obtain criteria for the identification of possible limit points and for convergence toward asymptotically stable equilibria for the slow variable by analyzing the dy
    
[^6]: 通过影响函数实现可追溯光谱推断：为Ariel任务提供高效数据归因与误差代理

    Traceable Spectral Inference via Influence Functions: Efficient Data Attribution and Error Proxies for the Ariel Mission

    [https://arxiv.org/abs/2608.23458](https://arxiv.org/abs/2608.23458)

    本工作提出了一种基于影响函数的数据归因方法，通过预测影响、高效计算和误差代理三项创新，为Ariel任务提供了无标签可追溯的光谱推断与误差评估方案。

    

    可解释性对于部署在科学太空任务（如欧洲空间局的Ariel任务）中的机器学习模型至关重要，因为在运行期间没有真实数据可用，必须评估物理合理性。虽然大多数可解释人工智能方法侧重于特征归因，但本工作通过影响函数研究训练数据归因，并为操作光谱处理流程引入了三项关键贡献。首先，将影响重新表述为基于预测而非损失的形式，从而实现无标签部署。其次，通过利用极限学习机的闭式岭解，高效计算无穷小预测影响。第三，通过将训练残差传播通过影响敏感性，推导出基于影响的保守误差代理。针对模拟光谱的评估表明，所提出的代理与基于尺度和形状的光谱误差强相关。此外，该代理还展示了……

    arXiv:2608.23458v1 Announce Type: cross  Abstract: Interpretability is critical for machine learning models deployed in scientific space missions such as ESA's Ariel, where ground truth is unavailable during operations and physical plausibility must be assessed. While most explainable AI methods focus on feature attribution, this work investigates training data attribution through influence functions and introduces three key contributions for operational spectroscopy pipelines. First, influence is reformulated in terms of prediction rather than loss, enabling label-free deployment. Second, by leveraging the closed-form ridge solution of an Extreme Learning Machine, infinitesimal prediction influence is efficiently computed. Third, an influence-based conservative error proxy is derived by propagating training residuals through the influence sensitivities. Evaluated against simulated spectra, the proposed proxy correlates strongly with scale and shape-based spectral errors. Furthermore, 
    
[^7]: 分层指数-高斯混合模型用于观看时间分布预测

    Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction

    [https://arxiv.org/abs/2608.23356](https://arxiv.org/abs/2608.23356)

    本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。

    

    准确的观看时间（WT）预测是短视频推荐的重要需求。然而，WT分布具有接近零膨胀、长尾和多模态的特点。最近的指数-高斯混合网络（EGMN）对完整的条件WT分布进行建模，而非单一的点估计，并达到了最先进的性能。我们的大规模复现研究表明，EGMN容易遭受方差崩溃、组件冗余和无效组件的问题。我们提出了一种分层指数-高斯混合（HEGM）模型，通过分层跳过观看分解、基于KL的方差正则化、结构化初始化、去除强制高斯偏移和熵正则化来解决这些失败模式。在公开和大规模工业数据集上，HEGM提高了排序准确性和阈值事件预测，同时保持了竞争力的点估计准确性，并显著提升了整体性能。

    arXiv:2608.23356v1 Announce Type: new  Abstract: Accurate watch-time (WT) prediction is an important requirement for short-video recommendations. Yet WT distributions are near-zero-inflated, long-tailed and multimodal. The recent Exponential-Gaussian Mixture Network (EGMN) models the full conditional WT distribution rather than a single point estimate and achieves state-of-the-art performance. Our large-scale reproduction study reveals that EGMN is vulnerable to variance collapse, component redundancy, and inactive components. We propose a Hierarchical Exponential-Gaussian Mixture (HEGM) model that addresses these failure modes through a hierarchical skip-watch decomposition, KL-based variance regularization, structured initialization, removing the forced Gaussian shift and the entropy regularizer. Across public and large-scale industrial datasets, HEGM improves ranking accuracy and threshold-event prediction, while maintaining competitive point-estimation accuracy and substantially im
    
[^8]: 不确定性下的语义承诺可信大语言模型

    Credal Large Language Models for Semantic Commitment under Uncertainty

    [https://arxiv.org/abs/2608.23244](https://arxiv.org/abs/2608.23244)

    通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。

    

    大型语言模型（LLMs）通常会产生流畅但错误的答案，并带有过度的自信。一个核心限制是，标准LLMs通过单一预测分布表示不确定性，将认知上的无知与真正的模糊性混为一谈。我们引入了可信大语言模型（CLLMs）：通过一组LoRA适配器的集成诱导出一个可信集，其下界和上界概率暴露了合理预测分布的扩散范围，而不是坍缩为单一的softmax输出。从这一表示中，我们推导出两个互补的承诺分数。可信令牌承诺（CTC）是一个令牌空间分数，结合了下界支持、可信宽度和交集熵，无需额外生成即可计算。语义承诺一致性（SCC）通过采样补全将承诺扩展到语义空间，其中SCC-Gap衡量令牌级和语义级支持之间的不匹配。我们评估了幻觉情况。

    arXiv:2608.23244v1 Announce Type: cross  Abstract: Large language models (LLMs) often produce fluent but incorrect answers with unwarranted confidence. A central limitation is that standard LLMs represent uncertainty through a single predictive distribution, conflating epistemic ignorance with genuine ambiguity. We introduce Credal Large Language Models (CLLMs): an ensemble of LoRA adapters induces a credal set whose lower and upper probabilities expose the spread of plausible predictive distributions rather than collapsing to a single softmax output. From this representation we derive two complementary commitment scores. Credal Token Commitment (CTC) is a token-space score that combines lower-bound support, credal width, and intersection entropy, computed without additional generation. Semantic Commitment Consistency (SCC) extends commitment to semantic space using sampled completions, with SCC-Gap measuring the mismatch between token-level and semantic-level support. We evaluate hall
    
[^9]: 一步逆过程是一个凸规划：扩散反演的贝叶斯极限校准

    One Inverse Step is a Convex Program: Bayes-Limit Calibration of Diffusion Inversion

    [https://arxiv.org/abs/2608.23094](https://arxiv.org/abs/2608.23094)

    本文证明了一步DDIM逆过程在贝叶斯极限下等价于一个强凸优化问题，其唯一性、误差界和收缩性均不依赖数据分布假设，但求解器稳定性受势函数曲率限制。

    

    arXiv:2608.23094v1 公告类型：新  摘要：一步隐式DDIM逆过程是探测预训练扩散模型是否编码局部流形几何的最廉价方法。它是显式势函数的平稳性条件，$x-G(x)=\nabla\Psi_t(x)$，在贝叶斯极限下强凸，其模量恰好为$e^{-h_t}$，其中$h_t$是步骤的对数信噪比间隙——这适用于任意数据分布、调度和点，无需流形、可达性或单峰性假设。必须区分三个后果。(i) 在贝叶斯极限下解是唯一的；第二个解要求训练得分违反后验协方差界，超出因子$1/(1-e^{-h_t})$，这是模型错误的无假设证书；同一界限使收缩成为调度常数，$\rho_g^{\star}=1-e^{-h_t}<0.326$，贯穿标准DDPM调度。(ii) 求解器仍可能失败：Picard迭代是$\Psi_t$上的单位步长梯度下降，在$\lambda_{\max}(\nabla^2\Psi_t)>2$处不稳定，因此振荡不证明任何事；d

    arXiv:2608.23094v1 Announce Type: new  Abstract: One implicit DDIM inversion step is the cheapest probe of whether a pretrained diffusion model encodes local manifold geometry. It is the stationarity condition of an explicit potential, $x-G(x)=\nabla\Psi_t(x)$, strongly convex at the Bayes limit with modulus exactly $e^{-h_t}$ for the step's log-SNR gap $h_t$ $-$ for every data law, schedule and point, with no manifold, reach or unimodality hypothesis. Three consequences must be kept apart. (i) The solution is unique at the Bayes limit; a second one requires the trained score to violate the posterior-covariance bound by $1/(1-e^{-h_t})$, a hypothesis-free certificate of model error; the same bound makes contraction a schedule constant, $\rho_g^{\star}=1-e^{-h_t}<0.326$ throughout the standard DDPM schedule. (ii) The solver can still fail: Picard iteration is unit-step gradient descent on $\Psi_t$, unstable wherever $\lambda_{\max}(\nabla^2\Psi_t)>2$, so oscillation certifies nothing; d
    
[^10]: 神经玻尔兹曼方程

    Neural Boltzmann Equations

    [https://arxiv.org/abs/2608.23022](https://arxiv.org/abs/2608.23022)

    本文提出了神经玻尔兹曼方程（NBEs），通过结合神经分布函数、蒙特卡洛重要性采样和自然梯度法，克服了传统玻尔兹曼方程求解中高维积分和参数扫描的瓶颈，实现了高效且精确的宇宙学动力学模拟。

    

    早期宇宙中粒子的动力学由玻尔兹曼方程描述，这些方程涉及高维相空间积分。经典方法使用求积积分法，并在固定动量网格上演化系统，这种方法难以扩展到复杂系统和参数扫描，严重限制了可研究过程的复杂性。我们引入了神经玻尔兹曼方程（NBEs），它结合了三个耦合概念来克服这些限制。首先，粒子属性被编码在受物理启发的神经分布函数中，其参数可通过神经网络预测，从而实现高效的参数扫描。其次，相空间积分使用蒙特卡洛方法评估，并采用来自对撞机物理的重要性采样工具。第三，我们使用自然梯度法来演化系统。在展示了NBEs的各自优势后，我们使用该框架进行了精确计算。

    arXiv:2608.23022v1 Announce Type: cross  Abstract: The dynamics of particles in the early universe are described by Boltzmann equations, which involve high-dimensional phase-space integrals. Classical approaches use quadrature integration and evolve the system on a fixed momentum grid, which scales poorly to complicated systems and parameter scans, severely limiting the complexity of processes that can be studied. We introduce Neural Boltzmann Equations (NBEs), which combine three coupled concepts to overcome these limitations. First, particle properties are encoded in physics-inspired neural distribution functions, with parameters that can be predicted using neural networks, enabling efficient parameter scans. Second, phase-space integrals are evaluated with Monte Carlo, using importance sampling tools from collider physics. Third, we use the natural gradient method to evolve the system. After demonstrating the individual benefits of NBEs, we use the framework to perform a precision c
    
[^11]: 带初始正则化的随机梯度下降法

    Stochastic gradient descent with initial regularization

    [https://arxiv.org/abs/2608.22953](https://arxiv.org/abs/2608.22953)

    本文分析了带初始正则化的随机梯度下降法，在无噪声和有噪声情况下推导了期望超额风险的上界和下界，并与岭回归进行了比较。

    

    arXiv:2608.22953v1 公告类型：交叉 摘要：我们分析了一种带初始正则化的随机梯度下降法（SGDIR）变体，并针对平方损失推导了其期望超额风险的无维度上界。在无噪声情况下，我们在矩、源和容量假设下，为平均和非平均SGDIR获得了新的界。对于源参数的特定值，这些界的阶数为$m^{-2}\log^{2}m$，其中训练样本数量为$m$阶。对于源参数的另一值，我们获得，对任意$\epsilon>0$，当容量参数超过$\epsilon^{-1}$时，界的阶数为$m^{-3+\epsilon}$。我们还建立了一个下界，在某些情况下与我们的上界匹配，直到多对数因子。在有噪声情况下，我们提供了SGDIR与岭回归之间的基于实例的比较。在一般假设和正则化参数的温和下界下，我们展示了期望超额风险的性质。

    arXiv:2608.22953v1 Announce Type: cross  Abstract: We analyze a variant of stochastic gradient descent with initial regularization (SGDIR) and derive dimension-free upper bounds on its expected excess risk for the squared loss. In the noiseless case, we obtain new bounds for both averaged and non-averaged SGDIR under moment, source, and capacity assumptions. For a particular value of the source parameter, these bounds are of order $m^{-2}\log^{2}m$, where the number of training samples is of order $m$. For another value of the source parameter, we obtain, for any $\epsilon>0$, bounds of order $m^{-3+\epsilon}$, provided that the capacity parameter exceeds $\epsilon^{-1}$. We also establish a lower bound that matches our upper bounds in certain regimes up to a polylogarithmic factor. In the noisy case, we provide an instance-based comparison between SGDIR and ridge regression. Under general assumptions and a mild lower bound on the regularization parameter, we show that the expected exc
    
[^12]: 一种用于深度神经网络中选择性谱对齐的交换子框架

    A Commutator Framework for Selective Spectral Alignment in Deep Neural Networks

    [https://arxiv.org/abs/2608.22910](https://arxiv.org/abs/2608.22910)

    该论文提出了一个交换子框架，精确量化深度神经网络中特征几何对齐的机制，并揭示了内部几何与特征侧对齐之间的传输关系。

    

    arXiv:2608.22910v1 公告类型：新 摘要：我们开发了一个有限宽度几何框架，描述了深度神经网络中学习到的特征几何如何被组织、传输和选择性对齐。由权重生成的协方差、门控和反向敏感性之间的不兼容性通过三族交换子进行量化：门控与协方差之间、敏感性与协方差之间，以及平均梯度外积（AGOPs）与神经特征矩阵（NFMs）之间。一个精确的逐层恒等式将敏感性-协方差交换子分解为四个来源：下游传输、相邻层不平衡、逐点敏感性波动和非线性门控-协方差相互作用。AGOP-NFM交换子是内部交换子的奇异值加权传输，解释了为何观察到的特征侧对齐本身并不决定其产生的内部几何。缓冲局部能量解决了分离与混合之间的区分问题。

    arXiv:2608.22910v1 Announce Type: new  Abstract: We develop a finite-width geometric framework describing how learned feature geometries are organized, transported, and selectively aligned in deep neural networks. Incompatibility among weight-generated covariance, gates, and backward sensitivities is quantified through three families of commutators: between gates and covariance, between sensitivities and covariance, and between average gradient outer products (AGOPs) and neural feature matrices (NFMs).   An exact layerwise identity decomposes the sensitivity-covariance commutator into four sources: downstream transport, adjacent-layer imbalance, pointwise sensitivity fluctuations, and nonlinear gate-covariance interactions. The AGOP-NFM commutator is a singular-value-weighted transport of the internal commutator, explaining why observed feature-side alignment alone does not determine the internal geometry from which it emerges.   Buffered localized energies resolve mixing between separ
    
[^13]: 概率流常微分方程中的变化检测：扩散潜在空间中的在线检验

    Change Detection in Probability Flow ODE: Online Testing in Diffusion Latent Spaces

    [https://arxiv.org/abs/2608.22807](https://arxiv.org/abs/2608.22807)

    本文提出一种基于条件扩散模型和概率流常微分方程的在线变化点检测方法，通过将观测映射到标准高斯潜在空间并利用最大均值差异检测分布变化，适用于分布无闭合形式的情况。

    

    arXiv:2608.22807v1 公告类型：交叉 摘要：快速增长的顺序数据任务范围，例如识别金融市场中的趋势反转、自动分割视频和音频记录、从运动传感器检测移动方向的变化，如果不对时间顺序数据中的分布变化进行检测，就无法完全解决。我们考虑一个序列变化点检测问题，其中条件密度在未知时间发生切换，但变化前和变化后的分布都没有闭合形式。经典似然比统计在此情况下不适用。一个在变化前数据上训练并带有冻结上下文编码器的条件扩散模型，通过概率流常微分方程定义了一个确定性双射。变化前的观测被映射到标准高斯潜在变量上。变化后的观测，通过相同的冻结映射处理，会偏离这一参考。我们采用最大均值差异作为检验统计量。

    arXiv:2608.22807v1 Announce Type: cross  Abstract: A rapidly growing range of sequential data tasks, such as identifying trend reversals in financial markets, auto-segmenting video and audio recordings, detecting changes in movement direction from motion sensors cannot be fully addressed without detection of distributional shifts in time-ordered data. We consider a sequential change-point detection problem where the conditional density switches at an unknown time, yet neither the pre- nor post-change distribution admits a closed-form. Classical likelihood-ratio statistics are inapplicable in this settings.   A conditional diffusion model, trained on pre-change-point data with a frozen context encoder, defines a deterministic bijection via the probability flow ODE. Pre-change observations are mapped onto standard Gaussian latent variables. Post-change observations, processed through the same frozen map, deviate from this reference. We employ the Maximum Mean Discrepancy as the test stat
    
[^14]: 用于Sinkhorn分布鲁棒假设检验的生成式神经网络

    Generative Neural Networks for Sinkhorn Distributionally Robust Hypothesis Testing

    [https://arxiv.org/abs/2608.22746](https://arxiv.org/abs/2608.22746)

    本文提出了一种基于生成式神经网络的框架，通过推导Sinkhorn差异模糊集的等价条件KL散度表示并利用强对偶性，将SDRHT问题转化为可扩展的凸势最大化问题，实现了高效训练和端到端采样。

    

    本文研究了Sinkhorn分布鲁棒假设检验（SDRHT）问题，旨在针对以经验分布为中心的Sinkhorn差异模糊集中的最不利分布，寻求稳健的检测器。现有方法通过求解大规模锥规划问题来处理该问题，但这些方法不可扩展。为克服这一局限，我们提出了一种生成式框架，该框架学习最不利分布，并支持高效训练和端到端采样。对于基于Sinkhorn差异的模糊集，我们首先推导出相对于核平滑参考分布的等价条件KL散度表示。这一性质使我们能够证明约束和无约束极小极大SDRHT公式的强对偶性。基于闭式最优检测器和Brenier定理，我们将极大极小对偶公式重新表述为关于凸势函数的最大化问题，其解通过生成网络参数化。

    arXiv:2608.22746v1 Announce Type: new  Abstract: This paper studies the Sinkhorn distributionally robust hypothesis testing (SDRHT) problem, seeking a robust detector against least-favorable distributions in Sinkhorn discrepancy-based ambiguity sets centered at the empirical distributions. Existing approaches solve this problem by solving large-scale conic programs, which are not scalable. To overcome this, we propose a generative framework that learns least-favorable distributions and supports efficient training and end-to-end sampling. For the Sinkhorn discrepancy-based ambiguity sets, we first derive an equivalent conditional-KL-divergence representation with respect to kernel-smoothed reference distributions. This property allows us to prove strong duality for both constrained and unconstrained minimax SDRHT formulations. Based on the closed-form optimal detector and Brenier's theorem, we reformulate the max-min dual formulation as a maximization problem over convex potentials whos
    
[^15]: 具有稳定无限维线性函数近似的Q学习

    Q-Learning with Stable Infinite-Dimensional Linear Function Approximation

    [https://arxiv.org/abs/2608.22636](https://arxiv.org/abs/2608.22636)

    该论文提出了一种基于非扩张重建和压缩算子的无限维线性函数近似框架，确保Q学习在马尔可夫轨迹下的稳定收敛，并给出了两种随机逼近算法的超范数收敛界。

    

    arXiv:2608.22636v1 公告类型：交叉 摘要：具有线性函数近似的Q学习可能不稳定，因为任意近似架构不必保留贝尔曼收缩性。我们开发了一个稳定的无限维线性函数近似框架，用于从单一马尔可夫行为策略轨迹中进行Q学习。学习变量是在紧致潜在度量空间$(\mathbb L,\rho)$上的系数场$\theta\in C(\mathbb L)$。该框架使用一个重建算子将$\theta$映射到连续的Q函数，以及一个压缩算子将贝尔曼更新映射回潜在坐标。两个算子的非扩张性在$C(\mathbb L)$上诱导出一个收缩的潜在贝尔曼映射，其唯一不动点$\theta^*$的重建在表示误差范围内近似最优Q函数。我们提出了两种随机逼近（SA）算法，并建立了其超范数收敛界，主导项阶数为$\w$。

    arXiv:2608.22636v1 Announce Type: cross  Abstract: Q-learning with linear function approximation can be unstable because an arbitrary approximation architecture need not preserve the Bellman contraction. We develop a stable infinite-dimensional linear function approximation framework for Q-learning from a single Markovian behavior-policy trajectory. The learning variable is a coefficient field $\theta\in C(\mathbb L)$ on a compact latent metric space $(\mathbb L,\rho)$. The framework uses a reconstruction operator that maps $\theta$ to a continuous Q-function and a compression operator that maps Bellman updates back to latent coordinates. Nonexpansiveness of both operators induces a contractive latent Bellman map on $C(\mathbb L)$, with a unique fixed point $\theta^*$ whose reconstruction approximates the optimal Q-function up to representation error. We propose two stochastic approximation (SA) algorithms and establish their sup-norm convergence bounds with a leading term of order $\w
    
[^16]: 稀疏模型下稀有事件数据的尺度不变最优子采样

    Scale-invariant Optimal Sampling for Rare-events Data with Sparse Models

    [https://arxiv.org/abs/2608.22597](https://arxiv.org/abs/2608.22597)

    提出了一种尺度不变的最优子采样方法，通过最小化预测误差并利用自适应lasso，有效解决了稀有事件数据中非活跃特征受缩放变换影响的问题。

    

    arXiv:2608.22597v1 公告类型：新 摘要：子采样在应对稀有事件大规模数据的计算挑战方面非常有效。过度激进的子采样可能会对估计效率产生不利影响，因此最优子采样对于缓解信息损失至关重要。然而，现有的最优子采样概率依赖于数据尺度，某些缩放变换可能导致低效的子样本。当存在非活跃特征时，这一问题更为显著，因为不恰当的缩放变换可能会任意放大它们对子采样概率的影响。我们针对这一挑战，在稀疏模型的背景下引入了一种尺度不变的最优子采样函数，其中通常假设存在非活跃特征。我们不再专注于估计模型参数，而是定义一个最优子采样函数以最小化预测误差，使用自适应lasso来概述估计过程，并研究其理论性质。

    arXiv:2608.22597v1 Announce Type: new  Abstract: Subsampling is effective in tackling computational challenges for massive data with rare events. Overly aggressive subsampling may adversely affect estimation efficiency, and optimal subsampling is essential to mitigate the information loss. However, existing optimal subsampling probabilities depend on data scales, and some scaling transformations may result in inefficient subsamples. This problem is more significant when there are inactive features, because their influence on the subsampling probabilities can be arbitrarily magnified by inappropriate scaling transformations. We tackle this challenge and introduce a scale-invariant optimal subsampling function in the context of sparse models, where inactive features are commonly assumed. Instead of focusing on estimating model parameters, we define an optimal subsampling function to minimize the prediction error, using adaptive lasso to outline the estimation procedure and study its theo
    
[^17]: 稀疏加性强化学习离策略评估：应对潜在有限轨迹数量

    Sparse Additive Off-Policy Evaluation for Reinforcement Learning with Potentially Limited Number of Trajectories

    [https://arxiv.org/abs/2608.22595](https://arxiv.org/abs/2608.22595)

    本文提出了一种稀疏加性离策略评估方法，在轨迹数量或时间范围有限的情况下，通过非线性稀疏建模和组稀疏特征筛选，实现了误差界仅对数依赖维度的高效价值估计。

    

    arXiv:2608.22595v1 公告类型：新 摘要：我们为无限时域强化学习开发了一个灵活、非线性且可解释的离策略评估新框架。为处理大型状态空间并支持透明决策，我们使用具有稀疏加性结构的非线性函数类对Q函数进行建模。我们推导了估计目标策略价值函数的高概率有限样本误差界，并表明这些界仅依赖于环境维度$d$的对数，从而缓解了维度灾难问题。与大多数现有离策略评估理论通常假设可访问大量轨迹不同，我们的分析保证了当轨迹数量或时间范围足够大时，价值估计的准确性。此外，我们提出了一种基于组稀疏性的特征筛选程序，该程序能以高概率识别出包含所有相关协变量的简化特征集。

    arXiv:2608.22595v1 Announce Type: new  Abstract: We develop a new framework for flexible, nonlinear, and interpretable off-policy evaluation for infinite-horizon reinforcement learning. To handle large state spaces and support transparent decision-making, we model the Q-function using a nonlinear function class with a sparse additive structure. We derive high-probability finite-sample error bounds for estimating the value function of a target policy and show that the bounds depend only logarithmically on the ambient dimension $d$, thereby alleviating the curse of dimensionality. In contrast to most existing theory for off-policy evaluation, which typically assumes access to many trajectories, our analysis guarantees accurate value estimation when either the number of trajectories or the time horizon is sufficiently large. In addition, we propose a group-sparsity-based feature screening procedure that identifies, with high probability, a reduced feature set containing all relevant covar
    
[^18]: 从单尺度分数场恢复加权切向几何

    Recovering Weighted Tangent Geometry from a Single-Scale Score Field

    [https://arxiv.org/abs/2608.22334](https://arxiv.org/abs/2608.22334)

    本文提出一种方法，在未知分支中心和齐次度的情况下，仅利用单一噪声水平的分数场，通过弱形式的Ornstein-Uhlenbeck方程和球面分数积分，恢复加权切向几何及其标量高斯-锥变换。

    

    arXiv:2608.22334v1 公告类型：新 摘要：在光滑数据流形附近，一个切空间概括了局部几何。在分支点处，相应的一阶对象则是切线方向上的测度，其归一化质量记录了在所选数据测度下每个分支的局部份额。我们探究当分支中心和齐次度 $d$ 未知时，单一噪声水平下的分数场是否能确定这种加权切向几何。在该切向测度模型中，$d$ 是局部测度维度。齐次切向测度的高斯平滑满足Ornstein--Uhlenbeck特征函数方程。其弱形式将分数值（无需分数导数）转化为关于中心和齐次度的线性系统，并带有显式秩条件和扰动界限。在校准后，单一球面上的切向分数是标量高斯-锥变换的球面对数梯度。积分可在尺度上恢复该变换。

    arXiv:2608.22334v1 Announce Type: new  Abstract: Near a smooth data manifold, one tangent space summarizes local geometry. At a branch point, the corresponding first-order object is instead a measure over tangent directions, whose normalized masses record the local share of each branch under the chosen data measure. We ask whether a score field at one noise level determines this weighted tangent geometry when the branch center and homogeneity degree $d$ are unknown. In this tangent-measure model, $d$ is the local measure dimension. Gaussian smoothing of a homogeneous tangent measure satisfies an Ornstein--Uhlenbeck eigenfunction equation. Its weak form turns score values---without score derivatives---into a linear system for the center and homogeneity degree, with an explicit rank condition and perturbation bound. After this calibration, the tangential score on one sphere is the spherical log-gradient of a scalar Gaussian--cone transform. Integration recovers that transform up to scale
    
[^19]: 稀缺确认性PET测量的合理分配：A4/LEARN中的目标对齐验证

    Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN

    [https://arxiv.org/abs/2608.22223](https://arxiv.org/abs/2608.22223)

    本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。

    

    抗淀粉样蛋白疗法和血液生物标志物正在将阿尔茨海默病的诊疗流程转变为两阶段测量工作流：首先使用成本较低的信息进行广泛筛查，然后在能支持最终报告决策的关键环节使用稀缺的确认性淀粉样蛋白测量。淀粉样蛋白正电子发射断层扫描（PET）仍是此类用于评估淀粉样蛋白负担的协议测量手段之一，但PET机位、试验预算和面向支付方的证据包都是有限的。本文提出了一个明确的操作性问题：何时简单的透明PET验证足够，何时拟合残差不确定性评分值得增加复杂度？对于加权协议目标，验证受试者i的一阶价值是目标影响与残差协议不确定性的乘积。通用不确定性采样仅使用第二个因素，可能将PET测量分配给那些难以预测但对科学、临床或商业目标影响较弱的受试者。

    arXiv:2608.22223v1 Announce Type: cross  Abstract: Anti-amyloid therapies and blood-based biomarkers are changing Alzheimer disease workups into a two-stage measurement workflow: screen broadly with cheaper information, then spend scarce confirmatory amyloid measurements where they support the decision that will be reported. Amyloid positron-emission tomography (PET) remains one such protocol measurement for amyloid burden, but PET slots, trial budgets, and payer-facing evidence packages are finite. This paper asks a deliberately operational question: when is simple transparent PET validation enough, and when is a fitted residual-uncertainty score worth the added complexity? For a weighted protocol target, the first-order value of validating subject i is the product of target influence and residual protocol uncertainty. Generic uncertainty sampling uses only the second factor and can spend PET measurements on subjects that are hard to predict but weak for the scientific, clinical, or c
    
[^20]: 基于变分推断的联合因果结构与聚类发现

    Joint Causal Structure and Cluster Discovery Using Variational Inference

    [https://arxiv.org/abs/2608.22212](https://arxiv.org/abs/2608.22212)

    本文提出了一种基于变分推断的新方法，能够同时推断潜在变量聚类及其因果结构，无需预先知道聚类信息。

    

    因果发现旨在理解个体随机变量之间的关系。在许多应用中，如脑成像和气候建模，考虑变量组间的相互作用更具意义。现有方法在建模交互时假设这些组或聚类的知识是明确可用的。然而，在实践中，这些聚类以及它们之间的因果关系是潜在的。在本文中，我们提出了一种基于变分推断的新方法，以同时推断潜在的聚类和因果结构。我们通过分别考虑基于类别和伯努利模型的变分分布，学习了聚类和图结构的近似后验。我们推导了变分下界和估计技术来学习变分和模型参数。我们提出的聚类和因果发现方法的有效性得到了证明。

    arXiv:2608.22212v1 Announce Type: cross  Abstract: Causal discovery aims to understand the relationships between individual random variables. In many applications, such as brain imaging and climate modeling, it is more meaningful to consider interactions among groups of variables. Existing methods assume that knowledge of such groups or clusters is explicitly available when modeling interactions. However, in practice, these clusters as well as the causal relationships among them, are latent. In this paper, we present a novel approach based on variational inference to simultaneously infer both the latent clusters and causal structures. We learn an approximate posterior over clusters and graph-structure by considering variational distributions based on categorical and Bernoulli models respectively. We derive variational lower bounds and estimation techniques to learn variational and model parameters. The effectiveness of our proposed methods for cluster and causal discovery are demonstra
    
[^21]: 基于词元级似然数组回归的成员推断与AI生成文本检测

    Token-Level Likelihood-Array Regression for Membership Inference and AI-Generated Text Detection

    [https://arxiv.org/abs/2608.22179](https://arxiv.org/abs/2608.22179)

    提出似然数组回归（LAR）方法，通过嵌套上下文窗口评估词元似然并组织成结构化数组，显著提升成员推断和AI生成文本检测的准确性。

    

    成员推断旨在判断一段文本是否用于训练语言模型，而AI生成文本检测则判断文本是由语言模型生成还是人类撰写。现有的基于似然的方法通常将词元级概率压缩为少数预设分数，且大多仅使用基于完整前文上下文的条件概率。我们提出了似然数组回归（LAR），该方法在嵌套的左上下文窗口下评估每个目标词元，并将得到的似然衍生特征组织成结构化数组。在将不同长度文本的数组对齐后，LAR学习检测信息如何随上下文尺度、词元位置和似然特征变化。LAR-1聚合单个对齐单元格的学习贡献，而LAR-2增加了由同一目标词元在不同上下文长度下成对评估形成的二阶特征。对于路径内查询...

    arXiv:2608.22179v1 Announce Type: new  Abstract: Membership inference asks whether a text was used to train a language model, whereas AI-generated text detection asks whether it was generated by a language model rather than written by a human. Existing likelihood-based methods typically compress token-level probabilities into a few prespecified scores, most often using only probabilities conditioned on the full preceding context. We propose likelihood-array regression (LAR), which evaluates each target token under nested left-context windows and organizes the resulting likelihood-derived features into a structured array. After aligning arrays across texts of different lengths, LAR learns how detection information varies with context scale, token position, and likelihood features. LAR-1 aggregates learned contributions from individual aligned cells, while LAR-2 adds second-order features formed from pairs of evaluations of the same target token across context lengths. For within-path qu
    
[^22]: 符号神经常微分方程：从时间序列数据中学习可解释模型

    Symbolic Neural ODEs: Learning interpretable models from time-series data

    [https://arxiv.org/abs/2608.22112](https://arxiv.org/abs/2608.22112)

    本文提出一种符号神经ODE框架，通过多步预测和稀疏正则化，从时间序列数据中学习稳定且可解释的动力学模型。

    

    我们提出了一种机器学习框架，用于直接从时间序列数据中识别动态系统的稀疏、可解释模型。我们的方法使用神经架构参数化底层向量场，并通过在有限时间范围内最小化多步预测损失来训练该模型。为确保数值可处理性，我们优化了跨预测步骤平均的绝对误差目标，并在训练过程中逐步增加预测范围。该公式的一个关键特性是，它在学习到的动态的重复组合下强制一致性。因此，与基于向量场一步回归的方法相比，所识别的模型表现出显著改善的稳定性。当与促进稀疏性的正则化结合时，这产生了能够超越训练数据泛化的简约模型。我们展示了在多种行为系统上的准确恢复能力。

    arXiv:2608.22112v1 Announce Type: cross  Abstract: We present a machine learning framework for identifying sparse, interpretable models of dynamical systems directly from time-series data. Our approach parameterizes the underlying vector field using a neural architecture and trains it by minimizing a multi-step prediction loss over a finite horizon. To ensure numerical tractability, we optimize a mean absolute error objective averaged across prediction steps, and progressively increase the horizon during training. A key feature of this formulation is that it enforces consistency under repeated composition of the learned dynamics. As a result, the identified models exhibit significantly improved stability compared with approaches based on one-step regression of the vector field. When combined with sparsity-promoting regularization, this leads to parsimonious models that generalize beyond the training data. We demonstrate accurate recovery of systems exhibiting a wide range of behaviors,
    
[^23]: Mapper表示上的结构化学习

    Structured Learning on Mapper Representations

    [https://arxiv.org/abs/2608.22044](https://arxiv.org/abs/2608.22044)

    本文提出了一种将Mapper构造作为完整表示的一部分进行学习的框架，并研究了其数学性质，如重标号不变性和距离泛函，以捕捉数据的多尺度结构组织。

    

    现代机器学习（ML）方法在预测任务中非常有效，但许多常用的表示方法将复杂数据降维为固定维度的嵌入，这可能抑制多尺度结构组织。拓扑数据分析（TDA）中的Mapper算法提供了不同的视角，通过神经构造将数据分解为重叠的局部区域，产生一种结构化表示，同时捕捉几何组织、局部统计行为和关系连通性。在这项工作中，我们开发了一个在Mapper诱导的结构化表示上进行学习的框架。我们不将Mapper视为预处理步骤，仅生成图供下游学习使用，而是将完整的Mapper构造视为表示本身的一部分。我们研究了这些表示的数学性质，包括重标号下的不变性、距离泛函等。

    arXiv:2608.22044v1 Announce Type: new  Abstract: Modern machine learning (ML) methods are highly effective for prediction tasks, but many commonly used representations reduce complex data to fixed dimensional embeddings that may suppress multiscale structural organization. The Mapper algorithm from topological data analysis (TDA) provides a different perspective by decomposing data into overlapping local regions connected through a nerve construction, producing a structured representation that captures geometric organization, local statistical behavior, and relational connectivity simultaneously. In this work, we develop a framework for learning over Mapper induced structured representations. Rather than treating Mapper as a preprocessing step that produces a graph for downstream learning, we treat the full Mapper construction as part of the representation itself. We study mathematical properties of these representations, including invariance under relabeling, a distance functional on 
    
[^24]: 重心融合Gromov-Wasserstein平衡用于多重处理下的因果推断

    Barycentric Fused Gromov-Wasserstein Balancing for Causal Inference under Multiple Treatments

    [https://arxiv.org/abs/2608.22024](https://arxiv.org/abs/2608.22024)

    提出CIHSI-Net框架，利用重心融合Gromov-Wasserstein平衡目标实现全局对齐，降低计算复杂度，提升多重处理下因果推断的准确性和效率。

    

    arXiv:2608.22024v1 公告类型：交叉 摘要：在多重同时处理下，从观测数据中估计异质性单一和交互处理效应对于决策至关重要。为减少估计方差，以往研究在每对处理模式之间平衡表示分布。然而，这种成对平衡随处理模式数量呈二次方扩展，并且无法跨模式保持一致的局部邻近结构，从而降低了反事实估计的质量。为解决这些挑战，我们提出了用于异质性单一和交互处理效应因果推断网络（CIHSI-Net），这是一个基于新型重心融合Gromov-Wasserstein平衡（BFG-WB）目标的深度学习框架。BFG-WB将每个处理模式的表示分布与共享的Wasserstein重心对齐，实现全局对齐，同时将计算复杂度从二次方降低。

    arXiv:2608.22024v1 Announce Type: cross  Abstract: Estimating heterogeneous single and interaction treatment effects from observational data under multiple simultaneous treatments is crucial for decision-making. To mitigate estimation variance, previous studies balance representation distributions between every pair of treatment patterns. However, such pairwise balancing scales quadratically with the number of treatment patterns and fails to preserve consistent local proximity structures across patterns, which degrades counterfactual estimation. To address these challenges, we propose the Causal Inference for Heterogeneous Single and Interaction Treatment Effects Network (CIHSI-Net), a deep learning framework built on a novel Barycentric Fused Gromov-Wasserstein Balancing (BFG-WB) objective. BFG-WB aligns the representation distribution of each treatment pattern with a shared Wasserstein barycenter, achieving global alignment while reducing the computational complexity from quadratic t
    
[^25]: 方差驱动探索：高随机环境中纯探索的一种可证明且高效的方法论

    Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments

    [https://arxiv.org/abs/2608.21995](https://arxiv.org/abs/2608.21995)

    我们提出了一种方差驱动的探索方法论，通过最小化最终决策不确定性来分配采样资源，在高随机环境中显著提升了纯探索任务的效率与理论保证。

    

    我们提出方差驱动探索（VarDE），这是一种在高随机环境中进行纯探索的原则性方法，其中探索过程受随机方差主导。VarDE基于一个基本原则：采样努力应被分配以最小化最终决策的不确定性。我们通过一个平滑的决策函数形式化最终决策的不确定性，并推导出分配规则，这些规则明确捕捉了各个组件中的随机噪声如何影响最终输出的可靠性。我们将此方法应用于纯探索的三个核心问题——最佳臂识别（BAI）、蒙特卡洛树搜索（MCTS）和最佳策略识别（BPI）——并提供了关于方差衰减和简单遗憾的理论保证。在实证中，我们展示了VarDE相比现有方法的一致且显著的改进，尤其是在高随机环境中取得了强劲的提升。

    arXiv:2608.21995v1 Announce Type: cross  Abstract: We propose Variance Driven Exploration (VarDE), a principled approach for pure exploration in highly stochastic environments, where the exploration process is dominated by stochastic variance. VarDE is built on a fundamental principle: sampling effort should be allocated to minimize the uncertainty of the final decision. We formalize the uncertainty of the final decision through a smooth decision function and derive allocation rules that explicitly capture how stochastic noise in individual components affects the reliability of the final output. We apply this methodology to three core problems of pure exploration -- Best Arm Identification (BAI), Monte Carlo Tree Search (MCTS), and Best-Policy Identification (BPI) -- with theoretical guarantees on variance decay and simple regret. Empirically, we demonstrate consistent and significant improvements of VarDE over existing methods, with especially strong gains in highly stochastic environ
    
[^26]: 改进的去噪扩散概率模型：高效非对角协方差建模

    Improved denoising diffusion probabilistic models with efficient non-diagonal covariance modeling

    [https://arxiv.org/abs/2608.21972](https://arxiv.org/abs/2608.21972)

    本文提出了一种名为K-DCT的协方差模型，通过Kronecker分解和离散余弦变换高效建模非对角协方差，从而在更少采样步骤中加速DDPM生成，同时保持样本质量。

    

    arXiv:2608.21972v1 公告类型：交叉 摘要：去噪扩散概率模型（DDPM）的采样过程可以通过利用去噪后验协方差的近似形式（即二阶信息）来加速，从而允许在更少但更大的采样步骤中生成质量可接受的样本。先前尝试使用此类信息的方法对协方差进行了剧烈简化（例如对角化），但这未能充分反映自然图像的独特统计结构，该结构在像素和颜色通道之间表现出强烈的非对角相关性，以及慢衰减的幂律频率谱。在此，我们开发了一种新颖的协方差模型，以捕捉这些特征。我们的Kronecker-DCT（K-DCT）模型使用Kronecker分解来建模颜色间协方差，并在频域中使用离散余弦变换（DCT）建模空间协方差。DCT的使用将计算复杂度从...

    arXiv:2608.21972v1 Announce Type: cross  Abstract: The sampling process of Denoising Diffusion Probabilistic Models (DDPMs) can be accelerated by leveraging second-order information in the form of approximations to the denoising posterior covariance -- allowing samples of acceptable quality to be produced in fewer but larger sampling steps. Previous attempts at using such information have used drastic (e.g.\ diagonal) simplifications of the covariance. These do not do justice to the peculiar statistical structure of natural images, which exhibit strong non-diagonal correlations between pixels and color channels, and a slow-decaying power-law frequency spectrum. Here, we develop a novel covariance model that captures these features. Our Kronecker-DCT (K-DCT) model uses a Kronecker-factored decomposition of inter-color covariances and spatial covariances modeled in the frequency domain using the Discrete Cosine Transform (DCT). The use of the DCT reduces the computational complexity from
    
[^27]: 通过密度比估计进行先验变化引导

    Guidance for Prior Change via Density Ratio Estimation

    [https://arxiv.org/abs/2608.21729](https://arxiv.org/abs/2608.21729)

    提出一种基于密度比估计的无偏测试时引导框架，通过学习得分引导项有效解耦推断过程与先验训练，避免了先验依赖和系统性偏差。

    

    模拟推演推断（SBI）是在模拟器涉及难以处理的似然函数的科学领域中，用于参数推断的重要框架。然而，虽然摊销生成模型能够快速进行后验估计，但它们通常受限于训练期间使用的特定先验，从而随着先验知识的发展限制了其灵活性。为解决这一先验依赖问题，引入了PriorGuide作为一种推理时引导方法，但由于其难以处理的公式形式，它依赖于反向转移核的高斯近似和先验比的高斯混合模型拟合，这两者都会引入系统性偏差。受这些局限性的启发，我们提出了一种无偏的测试时引导框架，利用密度比估计（DRE）学习一个得分引导项，有效将推断过程与先验训练解耦。此外，我们的框架对具体模型规格保持不可知性。

    arXiv:2608.21729v1 Announce Type: new  Abstract: Simulation-Based Inference (SBI) serves as a vital framework for parameter inference in scientific fields where simulators involve intractable likelihoods, yet while amortized generative models offer rapid posterior estimation, they are often restricted by the specific priors used during training, thereby limiting their flexibility as prior knowledge evolves. To address this prior dependency, PriorGuide was introduced as an inference-time guidance method, but due to its intractable formulation, it relies on Gaussian approximations of the reverse transition kernel and Gaussian mixture model fitting for the prior ratio, both of which introduce systematic bias. Motivated by these limitations, we propose an unbiased test-time guidance framework that leverages Density Ratio Estimation (DRE) to learn a score guidance term, effectively decoupling the inference process from the prior training. Moreover, our framework remains agnostic to the spec
    
[^28]: GeoQ：面向科学代理模型的几何感知条件分位数误差估计

    GeoQ: Geometry-Aware Conditional Quantile Error Estimation for Scientific Surrogate Models

    [https://arxiv.org/abs/2608.21652](https://arxiv.org/abs/2608.21652)

    GeoQ提出了一种几何感知的非侵入式校准方法，通过锚点平均误差加条件分位数修正，实现了代理模型在查询点的准确误差估计。

    

    摘要：神经网络代理模型越来越多地用于加速科学模拟，但在外推和自回归设置中的部署需要输入相关的预测误差估计。在这项工作中，我们引入了GeoQ（几何感知条件分位数误差估计），一种非侵入式校准框架，用于在单个查询点估计代理误差。GeoQ将查询点的误差表示为锚点平均校准误差加上一个学习到的非负修正。该修正被建模为锚点相对误差增量的上条件分位数，使用基于几何的特征，这些特征编码了表示空间位移和局部支持密度。交叉拟合过程生成近似样本外的校准元组，而特征空间中的k最近邻支持分数识别出学习误差模型受校准数据支持的区域。

    arXiv:2608.21652v1 Announce Type: cross  Abstract: Neural-network surrogate models are increasingly used to accelerate scientific simulations, but their deployment in extrapolative and autoregressive settings requires input-dependent estimates of prediction error. In this work, we introduce GeoQ (Geometry-Aware Conditional Quantile Error Estimation), a non-intrusive calibration framework for estimating surrogate error at individual query points. GeoQ represents the error at a query point as an anchor-averaged calibration error plus a learned nonnegative correction. This correction is modeled as an upper conditional quantile of the anchor-relative error increment, using geometry-based features that encode representation-space displacement and local support density. A cross-fitting procedure generates approximately out-of-sample calibration tuples, while a feature-space k-nearest-neighbor support score identifies regions \textcolor{black}{where the learned error model is supported by cal
    
[^29]: 子零矩阵补全用于稀疏数据分析：潜在低秩结构的大规模学习

    Subzero matrix completion for sparse data analysis: large-scale learning of latent low-rank structure

    [https://arxiv.org/abs/2608.21607](https://arxiv.org/abs/2608.21607)

    本文提出一种随机交替最小二乘算法，通过处理稠密矩阵的小块并利用稀疏优化和CUDA加速，实现了大规模稀疏数据中潜在低秩结构的有效恢复。

    

    arXiv:2608.21607v1 公告类型：交叉 摘要：我们研究何时可以通过将实值低秩矩阵的负元素置零来恢复稀疏非负矩阵。这种分解的可能性暗示了稀疏性与秩之间的数学联系；我们分析了多个具有这种潜在低秩结构的稀疏矩阵，并用它们来说明这种联系的几何起源。之前的算法通过低秩矩阵因子的交替最小化发现了这些分解，但这样做，它们还需要计算并存储另一个矩阵，该矩阵既不稀疏也不低秩，其大小等于它们的乘积。我们开发了一种随机交替最小二乘算法，该算法作用于这个稠密矩阵的较小块，并因此扩展到更大的问题。我们还展示了如何通过稀疏优化和定制的CUDA内核进一步加速该算法。作为示例，我们使用...

    arXiv:2608.21607v1 Announce Type: cross  Abstract: We investigate when a sparse nonnegative matrix can be recovered from a real-valued matrix of much lower rank by zeroing out its negative elements. The potential for such decompositions suggests a mathematical connection between sparsity and rank; we analyze a number of sparse matrices with this latent low-rank structure and use them to illustrate the geometric origins of this connection. Previous algorithms have discovered these decompositions via an alternating minimization over the factors of a low-rank matrix, but to do so, they have also needed to compute and store another matrix, neither sparse nor low-rank, that is the size of their product. We develop a stochastic, alternating least-squares algorithm that operates on smaller blocks of this dense matrix and scales as a result to much larger problems. We also show how to further accelerate this algorithm with sparse optimizations and customized CUDA kernels. As one example, we us
    
[^30]: 随机风险森林

    Random Hazard Forests

    [https://arxiv.org/abs/2608.21597](https://arxiv.org/abs/2608.21597)

    随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。

    

    arXiv:2608.21597v1 公告类型：新 摘要：临床数据源，如电子健康记录和可穿戴传感器，会在随访期间反复记录患者状态，通常时间不规律且不同测量有不同的时间表。这些数据为持续更新、个体化的风险预测创造了机会。然而，现有方法在建模前往往简化了时间结构。我们引入了随机风险森林（RHF），这是一种生存树集成方法，它学习当新测量值可用时，患者风险如何在连续时间内变化。RHF通过可预测协变量过程的非参数风险似然直接公式化估计问题。一个高效的工作模型指导树的构建，之后为每个终端节点估计灵活的时间变化风险。给定任何可预测的协变量路径，每棵树沿其终端节点随时间跟踪路径，并组装相应的节点级风险估计。

    arXiv:2608.21597v1 Announce Type: new  Abstract: Clinical data sources such as electronic health records and wearable sensors record patient status repeatedly over follow-up, often at irregular times and on different schedules for different measurements. These data create opportunities for continuously updated, individualized risk prediction. Existing approaches, however, often simplify the temporal structure before modeling it. We introduce Random Hazard Forests (RHF), a survival tree ensemble that learns how a patient's hazard changes in continuous time as new measurements become available. RHF formulates the estimation problem directly through a nonparametric hazard likelihood for predictable covariate processes. An efficient working model guides tree construction, after which flexible time-varying hazards are estimated for each terminal node. Given any predictable covariate path, each tree follows the path through its terminal nodes over time and assembles the corresponding node-le
    
[^31]: 复域中的稀疏可分离因子分析及其在局部场电位数据中的应用

    Sparse Separable Factor Analysis in the Complex Domain with an Application to Local Field Potential Data

    [https://arxiv.org/abs/2608.21551](https://arxiv.org/abs/2608.21551)

    本文提出了一种复域中的稀疏可分离因子分析模型，通过复数软阈值和期望最大化算法，有效利用了复值数组的幅度和相位信息，实现了可解释的协方差估计。

    

    arXiv:2608.21551v1 公告类型：新 摘要：复值数组在信号处理中经常出现，其中科学解释依赖于保留幅度和相位信息。现有的协方差估计方法要么忽略此类数据的多路组织，要么依赖于无法直接利用其复数结构的实域嵌入。我们开发了稀疏可分离因子分析（SSFA），这是一种针对复值数组的潜在因子模型，在各模式间具有可分离的协方差结构。每个模式特定的协方差矩阵通过低秩厄米特因子结构和对角残差协方差矩阵建模。为了获得可解释的估计，我们在复加载矩阵上施加逐元素套索惩罚，并使用模式-wise参数扩展期望最大化程序来估计SSFA参数。所得的加载更新允许闭式复数软阈值解，该解在保留相位信息的同时收缩每个加载的模。

    arXiv:2608.21551v1 Announce Type: new  Abstract: Complex-valued arrays arise in signal processing, where scientific interpretation depends on retaining amplitude and phase information. Existing covariance estimation methods either ignore the multiway organization of such data or rely on real-domain embeddings that do not directly exploit their complex structure. We develop sparse separable factor analysis (SSFA), a latent factor model for complex-valued arrays with a separable covariance structure across modes. Each mode-specific covariance matrix is modeled through a low-rank Hermitian factor structure and a diagonal residual covariance matrix. To obtain interpretable estimates, we impose elementwise lasso penalties on the complex loading matrices and estimate the SSFA parameters using a mode-wise parameter-expanded expectation-maximization procedure. The resulting loading updates admit closed-form complex soft-thresholding solutions, which shrink the modulus of each loading while pre
    
[^32]: AI验证的几何学：独立同分布最佳N次搜索的精确认证极限

    The geometry of AI validation: Exact certification limits for iid best-of-N search

    [https://arxiv.org/abs/2608.21496](https://arxiv.org/abs/2608.21496)

    本文通过核几何方法，精确推导出独立同分布最佳N次搜索中验证的模糊宽度公式，并揭示其主导尺度为$m^2/N$，为AI验证提供了理论极限。

    

    摘要：人工智能系统越来越多地生成备选方案、检查证据并部署选定的输出。因此，验证是相对于目标的：证据仅在被产生它的干预所解决的方向上认证部署。我们将验证和部署规则表示为可靠性表面上的核。它们的跨度几何将复制（减少采样噪声）与新的干预方向（减少结构性盲区）区分开来。我们使这一原理在独立同分布的最佳N次搜索中精确成立。在标量排序、随机平局、最大选择、有界二元真值以及稳定的排序-真值关系下，通过$n=m$了解最佳$n$可靠性，留下精确的模糊宽度$B_{m,N}=1+2\sum_{r=1}^{m}(-1)^r\cos^{2N}{r\pi/[2(m+1)]}$。显式有界世界达到整个区间，完整前缀在限于$n\le m$的可靠性均值审计中是信息最大化的。主导尺度是$m^2/N$：这……

    arXiv:2608.21496v1 Announce Type: cross  Abstract: AI systems increasingly generate alternatives, inspect evidence, and deploy a selected output. Validation is therefore target-relative: evidence certifies deployment only in directions resolved by the interventions that produced it. We represent validation and deployment rules as kernels over a reliability surface. Their span geometry separates replication, which reduces sampling noise, from new intervention directions, which reduce structural blindness. We make this principle exact for iid best-of-$N$ search. Under scalar ranking, randomized ties, maximum selection, bounded binary truth, and a stable rank-truth relation, knowing best-of-$n$ reliability through $n=m$ leaves exact ambiguity width $B_{m,N}=1+2\sum_{r=1}^{m}(-1)^r\cos^{2N}{r\pi/[2(m+1)]}$. Explicit bounded worlds attain the entire interval, and the complete prefix is information-maximal among reliability-mean audits confined to $n\le m$. The governing scale is $m^2/N$: wh
    
[^33]: 马尔可夫模型中状态构建的数据驱动方法

    A Data-Driven Approach to State Construction in Markov Models

    [https://arxiv.org/abs/2608.21480](https://arxiv.org/abs/2608.21480)

    本文提出了一种数据驱动的状态构建方法，通过结合监督特征选择与无监督学习技术，无需先验假设即可识别马尔可夫模型中的同质状态，从而提高模型的效度和预测能力。

    

    摘要：马尔可夫链是一种广泛使用的随机过程，用于模拟随时间发生的随机事件。这些模型建立在整个数据集的子集（称为状态）之上，这些状态在转移概率方面被认为是同质的。然而，这些状态的创建常常被忽视或基于先验假设，这可能违反同质性要求，从而降低模型的有效性和预测能力。为了填补这一空白，本文结合监督特征选择与无监督学习技术，实现数据驱动的状态构建。研究了基于密度的聚类、谱聚类和Kohonen自组织映射在无先验假设下识别潜在分组的能力。本研究的贡献有两方面。首先，本文提出了一种结合适当无监督学习技术的状态构建方法论框架，并辅以适当的...

    arXiv:2608.21480v1 Announce Type: new  Abstract: A Markov chain is a widely used stochastic process modelling random events over time. These models are built on subsets of the entire dataset, referred to as states, which are considered to be homogeneous regarding transition probabilities. However, the creation of these states is often disregarded or based on prior assumption, potentially violating the homogeneity requirement and thus decreasing the validity and predictive power of the model. In order to fill this gap, this paper combines supervised feature selection with unsupervised learning techniques for data-driven state construction. Density-based clustering, spectral clustering, and Kohonen self-organizing maps are examined for their ability to identify latent groups without prior assumptions. The contribution of this study is twofold. First, the paper presents a methodological framework for state construction incorporating suitable unsupervised learning techniques, with appropri
    
[^34]: 高斯-埃尔米特求积法用于高斯混合熵计算及其在动作空间中的埃尔米特代理

    Gauss--Hermite Quadrature for Gaussian-Mixture Entropy with an Action-Space Hermite Surrogate

    [https://arxiv.org/abs/2608.21467](https://arxiv.org/abs/2608.21467)

    本文提出了一种利用高斯-埃尔米特求积法高效计算高斯混合微分熵的方法，并引入动作空间的埃尔米特多项式代理以提升连续优化性能，显著优于传统泰勒近似。

    

    arXiv:2608.21467v1 公告类型：新 摘要：高斯分布常用于对信号和状态中的不确定性进行建模，而当底层分布呈多模态时，通常使用高斯混合模型。与单一高斯分布不同，高斯混合模型的微分熵通常没有闭式表达式，因此需要数值近似。我们提出了一种基于高斯-埃尔米特求积法的方法来评估高斯混合微分熵。求积阶数控制着近似的数值分辨率。该方法在一维和二维高斯混合基准测试中，与泰勒近似、解析熵界以及数值积分参考进行了评估。对于连续动作上的重复优化，我们还提出了一种在动作空间中使用埃尔米特多项式代理的方法。在雷达指向基准测试中，其二阶形式相比基于二阶泰勒代理的方法，实现了显著更低的代理误差和优化器遗憾值。

    arXiv:2608.21467v1 Announce Type: new  Abstract: Gaussian distributions are used to model uncertainty in signals and states, and Gaussian mixtures are often used when the underlying distribution is multimodal. Unlike a single Gaussian, a Gaussian mixture generally has no closed-form expression for differential entropy and therefore requires numerical approximation. We propose a Gauss--Hermite quadrature method for evaluating Gaussian mixture differential entropy. The quadrature order controls the numerical resolution of the approximation. The method is evaluated on one- and two-dimensional Gaussian mixture benchmarks against Taylor approximations, analytic entropy bounds, and numerical integration references.   For repeated optimization over continuous actions, we also propose a Hermite polynomial surrogate in action space. In a radar pointing benchmark, its second-order form achieves substantially lower surrogate error and optimizer regret than a second-order Taylor surrogate based on
    
[^35]: 有限马尔可夫链的$k$-块平均核的谱划分

    Spectral partitioning for $k$-block averaging kernels of finite Markov chains

    [https://arxiv.org/abs/2608.21466](https://arxiv.org/abs/2608.21466)

    本文提出基于谱划分的算法来选择状态空间划分，通过加权k-means舍入特征函数构造平均核，并证明目标函数等价于Pearson卡方互信息，从而加速有限可逆马尔可夫链的收敛。

    

    arXiv:2608.21466v1 公告类型：新 摘要：我们开发了谱算法来选择状态空间划分，这些划分定义了有限、遍历且可逆马尔可夫链的平均核。对于划分$\mathcal O$，吉布斯核$G_{\mathcal O}$在平稳条件分布下在当前块内重新采样；当这种更新可操作时，将其与基线核$P$组合或混合可加速收敛。我们通过使用加权$k$-均值对$P^2$的底部非常值特征函数（或对加性混合使用$P$的代数最小特征函数）进行舍入来选择$\mathcal O$。对于$F(\mathcal O)=\|G_{\mathcal O}P-\Pi\|_{F,\pi}^2$，我们推导了精确的迹和归一化割表示，并表明$F$等于初始块标签与一次转移后状态之间的皮尔逊$\chi^2$-互信息，这赋予该矩阵目标自然的概率解释。在双块情形下，阈值扫描精确地...

    arXiv:2608.21466v1 Announce Type: new  Abstract: We develop spectral algorithms for selecting state-space partitions that define averaging kernels for finite, ergodic and reversible Markov chains. For a partition $\mathcal O$, the Gibbs kernel $G_{\mathcal O}$ resamples within the current block from the stationary conditional distribution; when this update is tractable, composing or mixing it with a baseline kernel $P$ can accelerate convergence. We select $\mathcal O$ by rounding the bottom nonconstant eigenfunctions of $P^2$, or the algebraically smallest eigenfunctions of $P$ for additive mixtures, using weighted $k$-means. For $F(\mathcal O)=\|G_{\mathcal O}P-\Pi\|_{F,\pi}^2$, we derive exact trace and normalized-cut representations and show that $F$ equals the Pearson $\chi^2$-mutual information between the initial block label and the state after one transition, giving this matrix objective a natural probabilistic interpretation. In the two-block case, a threshold sweep exactly so
    
[^36]: 扩散模型中基于Sobolev正则化的得分差异估计

    Sobolev Regularized Score Difference Estimation in Diffusion Models

    [https://arxiv.org/abs/2608.18237](https://arxiv.org/abs/2608.18237)

    本文提出了一种基于Sobolev正则化的得分差异估计方法，既保证了统计一致性又支持高维扩展，并给出了明确的收敛速率和极小极大下界。

    

    摘要：估计两个Stein得分函数之间的差异是生成建模中的一个基本问题。特别是，得分差异在迁移学习中自然出现，其中得分差异提供了将预训练模型适应到新目标分布的机制，并且在基于扩散模型的后期训练方法（如判别器引导）中也扮演关键角色。现有用于这些场景的得分差异估计器要么缺乏统计一致性，要么难以在高维中扩展。我们提出了一种基于Sobolev正则化的统计一致且可扩展的得分差异估计器，该正则化在确保一致性和稳定小样本训练中起关键作用。数学上，我们建立了收敛速率$O(n^{-\frac{s-1}{d+2s-2}})$，其中$d$是维度，$s$表示底层密度的平滑度，并提供了极小极大下界。

    arXiv:2608.18237v1 Announce Type: cross  Abstract: Estimating the difference of two Stein's score functions is a fundamental problem in generative modeling. In particular, score differences arise naturally in transfer learning, where the score difference provides the mechanism for adapting a pre-trained model to a new target distribution, and in diffusion model-based post-training methods such as discriminator guidance. Existing estimators for score differences in these settings either lack of statistical consistency or are difficult to scale up in high-dimensions. We propose a statistically consistent and scalable estimator for score differences based on Sobolev regularization, which plays a crucial role in ensuring consistency and stablizing the training in the small-sample regime. Mathematically, we establish a convergence rate of $O(n^{-\frac{s-1}{d+2s-2}})$ where $d$ is the dimension and $s$ denotes the smoothness of the underlying densities, and provide a minimax lower bound of $
    
[^37]: 基于证据偏差准则的深度自适应设计

    Deep adaptive design with an evidential bias criterion

    [https://arxiv.org/abs/2608.16466](https://arxiv.org/abs/2608.16466)

    本文提出了一种基于“反偏置”证据准则的深度自适应实验设计方法，以更好地控制实验产生误导性证据的风险，弥补传统期望信息增益准则的不足。

    

    贝叶斯最优实验设计（BOED）旨在通过优化反映实验目标的期望效用函数来收集信息丰富的数据。然而，对于常见的效用函数和复杂模型，这种优化在计算上具有挑战性，尤其是对于序列或自适应设计，其中设计和数据收集交替进行，因此必须考虑已观测数据的反馈。大多数现有BOED研究采用信息增益作为效用，导致期望信息增益（EIG）准则。虽然EIG广泛有用，但它可能并不总能充分反映实验目标。EIG可视为奖励平均上对真相产生大量正证据的实验，但它并不直接控制实验产生误导性证据的风险。在此，我们考虑一种替代准则，称为“反偏置”（BA），该准则优先关注这种控制。为解决此问题，我们提出了深度自适应设计方法。

    arXiv:2608.16466v1 Announce Type: cross  Abstract: Bayesian optimal experimental design (BOED) aims to collect informative data by optimizing an expected utility reflecting the goals of an experiment. However, this optimization is computationally challenging for common utilities and complex models. This is especially so for sequential or adaptive designs, where design and data collection alternate, so that feedback from already observed data must be taken into account. Most existing BOED research employs information gain as the utility, leading to the expected information gain (EIG) criterion. While EIG is widely useful, it may not always adequately reflect experimental goals. EIG can be viewed as rewarding experiments that produce large positive evidence for the truth on average, but it does not directly control the risk of an experiment producing misleading evidence. Here we consider an alternative criterion, which we call bias against (BA), that prioritizes such control. To address 
    
[^38]: 重新思考反向KL作为自适应熵蒸馏

    Rethinking Reverse KL as Adaptive Entropy Distillation

    [https://arxiv.org/abs/2608.14685](https://arxiv.org/abs/2608.14685)

    本文提出自适应熵蒸馏（AED），通过重新分解反向KL目标为教师拟合和学生熵项，利用教师熵动态调整蒸馏权重，实现更优的模仿与生成平衡。

    

    arXiv:2608.14685v1 公告类型：新论文 摘要：知识蒸馏（KD）广泛用于将大型语言模型（LLMs）的能力转移到较小的学生模型上，但现有目标函数常常难以平衡忠实模仿和稳健生成。特别是，现有方法主要结合前向KL（FKL）和反向KL（RKL），却忽视了RKL本身提供了一种调整学生模仿强度的机制。基于此，我们重新审视了策略上的反向KL（RKL）蒸馏，并将其目标函数分解为教师拟合项和学生熵项，无需引入显式的FKL分支。我们从理论上证明，令牌级的最优学生分布对应于教师分布的温和变体，其中自适应权重控制着模式寻求和不确定性保留之间的权衡。受此洞察启发，我们提出了\textbf{自适应熵蒸馏（AED）}，它利用教师的熵来动态调整蒸馏过程。

    arXiv:2608.14685v1 Announce Type: new  Abstract: Knowledge distillation (KD) is widely used to transfer the capabilities of large language models (LLMs) to smaller students, but existing objectives often struggle to balance faithful imitation and robust generation. In particular, existing methods mainly combine FKL and RKL, overlooking that RKL itself provides a mechanism for adjusting the student's imitation strength. Motivated by this, we revisit on-policy Reverse Kullback-Leibler (RKL) distillation and decompose its objective into a teacher-fitting term and a student-entropy term, without introducing an explicit FKL branch. We show theoretically that the token-level optimal student distribution corresponds to a tempered variant of the teacher distribution, where the adaptive weight controls the trade-off between mode-seeking and uncertainty preservation. Guided by this insight, we propose \textbf{Adaptive Entropy Distillation (AED)}, which uses the teacher's entropy to dynamically c
    
[^39]: 在算子范数样本阈值下的张量正态最大似然估计

    Tensor-normal maximum likelihood estimation at the operator-norm sample threshold

    [https://arxiv.org/abs/2608.10488](https://arxiv.org/abs/2608.10488)

    本文证明了张量正态最大似然估计在样本量条件可降低至$d_{\max}$的二次依赖，并提供了相应的误差界。

    

    arXiv:2608.10488v2 公告类型：替换交叉 摘要：设$X_1,\ldots,X_n$是$\mathbb{R}^{d_1}\otimes\cdots\otimes\mathbb{R}^{d_k}$中的独立高斯张量，其共同协方差矩阵由$k$个未知正定因子的Kronecker乘积给出，并设$D=\prod_{a=1}^k d_a$和$d_{\max}=\max_a d_a$。Franks等人（2026）在样本量条件$nD\gtrsim k^2 d_{\max}^3$下建立了张量正态最大似然估计器的无条件数保证，并询问$d_{\max}$的三次依赖是否可以降低为二次依赖。我们肯定地回答了这个问题。对于$t\geq 1$，如果$nD\geq C k^2 d_{\max}^2 t^2$，那么以高概率，最大似然估计器存在、唯一，并满足$d_{\rm FR}(\widehat\Theta,\Theta)\leq C t \sqrt{k} d_{\max}/\sqrt{n}$和对于每个模态$a$，$d_{\rm FR}(\widehat\Theta_a,\Theta_a)\leq C t\sqrt{k d_a} d_{\max}/\sqrt{nD}$。对于每个$d_a=d_{\max}$的模态$a$，我们进一步...

    arXiv:2608.10488v2 Announce Type: replace-cross  Abstract: Let $X_1,\ldots,X_n$ be independent Gaussian tensors in $\mathbb{R}^{d_1}\otimes\cdots\otimes\mathbb{R}^{d_k}$ with a common covariance matrix given by the Kronecker product of $k$ unknown positive-definite factors, and let $D=\prod_{a=1}^k d_a$ and $d_{\max}=\max_a d_a$. Franks et al. (2026) established condition-number-free guarantees for the tensor-normal maximum likelihood estimator under the sample-size condition $nD\gtrsim k^2 d_{\max}^3$ and asked whether the cubic dependence on $d_{\max}$ could be reduced to a quadratic one. We answer this question affirmatively. For $t\geq 1$, if $nD\geq C k^2 d_{\max}^2 t^2$, then with high probability the maximum likelihood estimator exists, is unique, and satisfies $d_{\rm FR}(\widehat\Theta,\Theta)\leq C t \sqrt{k} d_{\max}/\sqrt{n}$ and $d_{\rm FR}(\widehat\Theta_a,\Theta_a)\leq C t\sqrt{k d_a} d_{\max}/\sqrt{nD}$ for every mode $a$. For every mode $a$ with $d_a=d_{\max}$, we furt
    
[^40]: 非平稳动态定价：适应性与最优性

    On Non-Stationary Dynamic Pricing: Adaptivity and Optimality

    [https://arxiv.org/abs/2607.24115](https://arxiv.org/abs/2607.24115)

    本文提出了一种无需预先知道分段数或变化预算的自适应多尺度变点检测算法，在非平稳上下文动态定价中实现了接近最优的遗憾界，并引入了新的设计调整变化预算概念。

    

    我们研究了非平稳环境下的上下文动态定价问题，其中企业向T个顺序到达的消费者销售产品，这些消费者的行为遵循一个可能随时间变化的未知需求模型。需求模型被假设为广义线性模型（GLM），允许使用\mathbb{R}^d中的特征向量来编码产品和消费者信息。为了实现最优收入（即最小遗憾），企业需要在监控潜在变化的同时学习和利用未知的GLM。我们提出了一种基于多尺度变点检测的算法，其遗憾达到阶数\widetilde{O}(\sqrt{s_TdT}\wedge\{V_T^{1/3}d^{1/3}T^{2/3}+\sqrt{dT}\})，其中s_T是分段平稳段的数量，V_T是我们新定义的模型参数设计调整变化预算概念。我们的算法具有适应性，无需知道s_T或V_T。此外，据我们所知，这是首次实现此结果。

    arXiv:2607.24115v2 Announce Type: replace  Abstract: We study the contextual dynamic pricing problem under non-stationarity, where a firm sells products to $T$ sequentially arriving consumers that behave according to an unknown demand model that can change over time. The demand model is assumed to be a generalized linear model (GLM), allowing for a feature vector in $\mathbb{R}^d$ that encodes products and consumer information. To achieve optimal revenue (i.e., least regret), the firm needs to learn and exploit the unknown GLMs while monitoring for potential changes. We propose a multiscale change-point detection based algorithm that achieves a regret of order $\widetilde{O}(\sqrt{s_TdT}\wedge\{V_T^{1/3}d^{1/3}T^{2/3}+\sqrt{dT}\})$, where $s_T$ is the number of piecewise stationary segments and $V_T$ is a newly defined notion of design-adjusted variation budget of model parameters. Our algorithm is adaptive and does not require knowing $s_T$ or $V_T$. Moreover, to our knowledge, this i
    
[^41]: CausalSmith：一个形式化基础、自我改进的自动化因果推断研究代理框架

    CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference

    [https://arxiv.org/abs/2607.22511](https://arxiv.org/abs/2607.22511)

    CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。

    

    自动化理论研究不仅受限于候选结果的生成，还受限于其可靠评估。一种常见方法是使用大型语言模型（LLM）评审员来闭环研究过程。然而，此类评审员在经验上仍不可靠：他们可能接受伪造论文，并以接近随机水平的概率检测出这些论文（Bad Scientist，2025）。我们提出了CausalSmith，一个基于Lean证明助手的因果推断自动化理论研究框架。CausalSmith结合了Causalean（一个基础性的因果推断Lean库，包含7,035条机器检查的声明，在人类设计与审查下借助语言模型辅助开发）以及CausalSmith（一个自我改进的代理管道，用于选择研究主题、提出结果、形式化陈述、构造证明，并呈现最终产物供人类检查）。由于机器检查的证明……

    arXiv:2607.22511v3 Announce Type: replace-cross  Abstract: Automating theoretical research is constrained not only by the generation of candidate results, but also by their reliable evaluation. A common approach is to close the research loop with a large language model (LLM) reviewer. However, such reviewers remain empirically unreliable: they may accept fabricated papers and detect them at rates close to chance (Bad Scientist, 2025). We present CausalSmith, a framework for automated theoretical research in causal inference grounded in the Lean proof assistant. CausalSmith combines Causalean, a foundational Lean library for causal inference containing 7,035 machine-checked declarations developed with language-model assistance under human design and review, with CausalSmith, a self-improving agentic pipeline that selects research topics, proposes results, formalizes statements, constructs proofs, and presents the resulting artifacts for human inspection. Because a machine-checked proof 
    
[^42]: 签名学习的速度有多快？路径回归的统计理论与应用

    How Fast Do Signatures Learn? Statistical Theory and Applications for Path Regression

    [https://arxiv.org/abs/2607.17865](https://arxiv.org/abs/2607.17865)

    本文首次量化了基于路径签名的回归模型在截断级别增加时的逼近误差收敛速度，并证明了其最小最大最优性及三种统计学习过程的一致性。

    

    摘要：arXiv:2607.17865v2 公告类型：替换-交叉 摘要：运筹学中的许多预测和决策问题涉及路径值协变量——即随时间演变的数据——对于这些数据，路径签名已成为一种典型的特征表示。其使用由普遍逼近定理证明合理，但这只是一个存在性结果：它保证有限级别的签名可以近似任何连续路径泛函，但没有量化近似误差随截断级别增加而减少的速度。本文开发了基于签名的路径回归的逼近和统计理论。我们为伊藤扩散的平滑泛函建立了\(L^2\)逼近率，并证明其是最小最大最优的。然后，我们将截断误差传播到三种统计学习过程中——签名-OLS、签名-LASSO和签名-逻辑回归——并建立它们的一致性。三个真实数据应用表明，签名提供了

    arXiv:2607.17865v2 Announce Type: replace-cross  Abstract: Many prediction and decision-making problems in operations research involve path-valued covariates -- data that evolve over time -- for which path signatures have become a canonical feature representation. Their use is justified by a universal approximation theorem, but this is an existence result: it guarantees that a finite-level signature can approximate any continuous path functional, without quantifying how fast the approximation error decreases as the truncation level grows. This paper develops approximation and statistical theory for signature-based path regression. We establish an \(L^2\) approximation rate for smooth functionals of It\^{o} diffusions and show that it is minimax optimal. We then propagate the truncation error through three statistical learning procedures -- Signature-OLS, Signature-LASSO, and Signature-Logistic -- and establish their consistency. Three real-data applications show that signatures provide
    
[^43]: 大型语言模型所解释的并非其真实信念：在模型自身输入信念下评估解释充分性

    What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs

    [https://arxiv.org/abs/2606.28615](https://arxiv.org/abs/2606.28615)

    本文提出了一种基于信息论的指标SCSuff，利用LLM自身生成替代输入来评估自由文本解释的充分性，无需预设偏见，并证明解释充分性依赖于输入分布。

    

    arXiv:2606.28615v2 公告类型：替换-交叉 摘要：大型语言模型（LLMs）越来越多地被部署在高风险领域，在这些领域中，自由文本解释（如思维链和事后理由）被用于证明模型输出的合理性。然而，这些解释是否充分仍不清楚，即它们是否包含足够的信息来解释模型的输出生成过程。我们将经典充分性从特征归因推广到任意解释，并证明解释充分性可能随输入分布而变化，这必须为LLM解释明确定义。我们提出利用LLM自身生成基于解释的替代输入，以捕捉其对可能输入的信念。我们将自洽充分性形式化为自由文本解释的目标，并引入一种信息论指标SCSuff，该指标能够在无需依赖预定义偏见或假设的情况下评估自由文本解释。

    arXiv:2606.28615v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in high-stakes domains, where free-text explanations such as chain-of-thought and post-hoc rationales are used to justify model outputs. Yet it remains unclear whether these explanations are sufficient, i.e., if they contain enough information to explain the model's output-generating process. We generalize classical sufficiency from feature attributions to arbitrary explanations and prove that explanation sufficiency can change depending on the input distribution, which must be explicitly defined for LLM explanations. We propose using the LLM itself to generate alternative inputs conditioned on an explanation, capturing its beliefs about possible inputs. We formalize self-consistent sufficiency as a goal for free-text explanations and introduce an information-theoretic metric, SCSuff, that enables evaluation of free-text explanations without relying on predefined biases or 
    
[^44]: 一种面向约束感知生物过程开发的人机协同贝叶斯优化框架

    A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development

    [https://arxiv.org/abs/2606.19230](https://arxiv.org/abs/2606.19230)

    本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。

    

    本文介绍了对帕累托前沿引导采样（PFGS）的扩展，这是一种人机协同（HitL）贝叶斯优化（BO）框架，其中高斯过程（GP）代理导出的量被重新表述为多目标优化问题的目标，所得帕累托前沿暴露给领域专家进行交互式候选选择，而非返回单一自动推荐。该框架在两个方向上进行了扩展：约束优化通过将满足输出规格限制的后验概率作为显式帕累托目标来处理，该概率从GP后验分布解析计算；鲁棒优化通过蒙特卡洛采样策略来处理，该策略估计在用户定义的输入扰动变异性下的期望下置信性能，捕捉在可能实现偏差下的性能退化。

    arXiv:2606.19230v2 Announce Type: replace-cross  Abstract: This work presents an extension to Pareto Front Guided Sampling (PFGS), a Human-in-the-Loop (HitL) Bayesian Optimization (BO) framework in which Gaussian process (GP) surrogate-derived quantities are reformulated as objectives of a multi-objective optimization problem, and the resulting Pareto front is exposed to a domain expert for interactive candidate selection rather than returning a single automated recommendation. The framework is extended in two directions: constrained optimization is addressed by incorporating the posterior probability of satisfying output specification limits as an explicit Pareto objective, computed analytically from the GP posterior distribution; robust optimization is addressed by a Monte Carlo sampling strategy that estimates expected lower-confidence performance over a user-defined variability of input perturbations, capturing performance degradation under likely implementation deviations. The res
    
[^45]: 关系结构因果模型

    Relational Structural Causal Models

    [https://arxiv.org/abs/2606.14892](https://arxiv.org/abs/2606.14892)

    本文提出关系结构因果模型，通过定义关系因果图和符号识别标准，解决了在对象和关系变化场景下对未观测混杂因素进行因果与观测查询识别的难题，并验证了关系神经因果模型的有效性。

    

    arXiv:2606.14892v2 公告类型：替换 摘要：人工智能必须对其环境有一个因果模型，支持关于干预和反事实的推理，同时也需要具备组合性，以支持对未见过的对象组合进行泛化。在这项工作中，我们正式研究这种模型何时以及如何可以被学习。我们发展了关系结构因果模型，将结构因果模型（Pearl 2009）扩展到对象及其关系变化的情境中。首先，我们展示了在没有进一步假设的情况下，不仅是对因果查询，而且是对未见过对象组合的观测查询的答案都无法被识别。为了实现这种识别——包括在存在未观测混杂因素的情况下——我们定义了关系因果图，并推导出符号识别标准。最后，我们提出了关系神经因果模型，这是一种可证明正确的方法，在具有不同汽车的模拟交通场景中优于非关系基线。

    arXiv:2606.14892v2 Announce Type: replace  Abstract: An artificial intelligence must have a model of its environment that is causal, supporting reasoning about interventions and counterfactuals, and also combinatorial, supporting generalization to unseen combinations of objects. In this work, we formally study when and how such a model can be learned. We develop relational structural causal models, extending structural causal models (Pearl 2009) to settings where objects and their relations vary. First, we show how answers to not only causal but also observational queries about unseen combinations of objects can not be identified without further assumptions. To enable such identification--including in the presence of unobserved confounding--we define relational causal graphs and derive symbolic identification criteria. Finally, we propose relational neural causal models, a provably correct approach that outperforms non-relational baselines on simulated traffic scenes with varying cars,
    
[^46]: 复杂缺失机制下二元回归的共形预测

    Conformal Prediction for Dyadic Regression Under Complex Missingness

    [https://arxiv.org/abs/2606.11136](https://arxiv.org/abs/2606.11136)

    本文提出了一个在复杂缺失机制下用于二元回归的共形预测框架，通过新颖的双射论证和多种程序（如行列方法和选择性共形）实现了有限样本有效性和掩码条件有效性。

    

    arXiv:2606.11136v3 公告类型：替换交叉 摘要：我们开发了一个在复杂缺失机制下用于二元回归问题的共形预测框架。在理论层面，我们建立了通用技术工具，用于在比可交换性更弱的分布不变性条件下证明共形预测的有限样本有效性。一个关键结果处理了样本本身是指标集随机子集的情况，这一场景未被现有理论覆盖，通过一种新颖的双射论证，构造了事件之间显式的保测对应关系。此外，我们提出了针对联合可交换数组的共形预测程序，包括全共形、分裂共形、利用行内和列内相似性的行列方法，以及实现掩码条件有效性的选择性共形程序。对于缺失元素，我们在非参数条件下建立了加权共形程序的渐近有效性。

    arXiv:2606.11136v3 Announce Type: replace-cross  Abstract: We develop a framework for conformal prediction in dyadic regression problems under complex missingness mechanisms. At the theoretical level, we develop general technical tools for establishing finite-sample validity of conformal prediction under distributional invariance conditions weaker than exchangeability. A key result handles the case where the sample itself is a random subset of the index set, a setting not covered by existing theory, via a novel bijection argument that constructs an explicit measure-preserving correspondence between events. In addition, we propose conformal prediction procedures for jointly exchangeable arrays, including full conformal, split conformal, a row-column approach exploiting similarities within rows and columns, and a selective conformal procedure achieving mask-conditional validity. For missing elements, we establish asymptotic validity of a weighted conformal procedure under a nonparametric
    
[^47]: 汤普森采样在次高斯奖励下风险厌恶多臂老虎机中的渐近最优性

    Asymptotic Optimality of Thompson Sampling for Risk-Averse Bandits with Sub-Gaussian Rewards

    [https://arxiv.org/abs/2606.09191](https://arxiv.org/abs/2606.09191)

    本文证明了非参数汤普森采样算法在仅需连续风险泛函条件下，对次高斯奖励的风险厌恶多臂老虎机达到渐近最优遗憾，首次为非Lipschitz风险度量（如夏普比率）提供实例最优保证。

    

    我们证明了 $\rho\text{-}\mathrm{NPTS}_{\mathrm{SG}}$，一种用于风险厌恶多臂老虎机的无锚非参数汤普森采样算法，其遗憾值在 $\log n$ 的主导阶上匹配实例相关下界，从而在具有有界密度和次高斯尾部（包括高斯臂）的分布类别上，对任何连续风险泛函 $\rho$（如CVaR、均值方差、夏普比率、失真风险度量等）建立其渐近最优性。该结果及其有界支撑版本仅需 $\rho$ 的连续性：这严格弱于先前参数化汤普森采样结果的支配条件，也严格弱于UCB型算法的Lipschitz条件，从而在没有参数奖励假设的情况下，首次为非Lipschitz泛函（如夏普比率）提供了实例最优保证。有界支撑情况首先被开发为垫脚石。

    arXiv:2606.09191v2 Announce Type: replace-cross  Abstract: We prove that $\rho\text{-}\mathrm{NPTS}_{\mathrm{SG}}$, an anchor-free nonparametric Thompson Sampling algorithm for risk-averse bandits, achieves regret matching the instance-dependent lower bound to leading order in $\log n$, establishing it as asymptotically optimal for any continuous risk functional $\rho$ (CVaR, mean-variance, Sharpe ratio, distortion risk measures, and more) on the class of distributions with bounded density and sub-Gaussian tails, including Gaussian arms. Both this result and its bounded-support counterpart require only continuity of $\rho$: strictly weaker than the dominance condition of prior parametric Thompson Sampling results, and strictly weaker than the Lipschitz condition of UCB-type algorithms, yielding the first instance-optimal guarantees for non-Lipschitz functionals such as the Sharpe ratio without parametric reward assumptions. The bounded-support case is developed first as a stepping ston
    
[^48]: 通用干预下自动、去偏且不变的反事实生成

    Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions

    [https://arxiv.org/abs/2606.07399](https://arxiv.org/abs/2606.07399)

    ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。

    

    反事实结果的生成模型在复杂干预下的决策支持方面具有巨大潜力，但现有方法受限于估计不稳定、跨环境泛化能力差以及因干扰模型误设而产生的偏差。我们提出了ADIGen框架，用于在通用干预（包括高维干预和结果）下实现自动、去偏且不变的反事实生成。ADIGen结合了Riesz回归以避免不稳定的密度比估计、因果不变性以改善分布偏移下的泛化能力，以及正交统计学习以获得针对干扰模型误设的双重稳健保证。我们提供了超额风险界，表明ADIGen在通用干预下控制反事实风险，具有乘积偏差干扰余项和跨环境的不变风险界。然后，我们将该框架扩展到多...

    arXiv:2606.07399v2 Announce Type: replace  Abstract: Generative models for counterfactual outcomes have great potential to support decision-making under complex interventions, but existing approaches are limited by unstable estimation, poor generalization across environments, and bias from nuisance model misspecification. We introduce ADIGen, a framework for automatic, debiased, and invariant counterfactual generation under general interventions, including high-dimensional interventions and outcomes. ADIGen combines Riesz regression to avoid unstable density-ratio estimation, causal invariance to improve generalization under distribution shift, and orthogonal statistical learning to obtain doubly robust guarantees against nuisance model misspecification. We provide excess-risk bounds showing that ADIGen controls counterfactual risk under general interventions, with a product-bias nuisance remainder and an invariant risk bound across environments. We then extend this framework to multip
    
[^49]: 线性上下文赌博机在稀有参数更新下的实用最优算法

    Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates

    [https://arxiv.org/abs/2606.00984](https://arxiv.org/abs/2606.00984)

    本文提出两种仅需$O(\log\log T)$次参数更新的线性上下文赌博机算法，在静态调度下同时在小规模和大规模动作集下实现极小极大最优遗憾，并澄清了批处理与稀有更新的实际区别。

    

    arXiv:2606.00984v2 公告类型：替换-交叉 摘要：我们研究在稀有参数更新下的线性上下文赌博机：学习器只能在少量更新时间点将奖励反馈纳入其参数估计，同时仍需在线观察上下文并按顺序选择动作。这一视角澄清了文献中常被模糊的一个实际区别：许多“严格批处理”方法额外限制了区间内的上下文自适应性，即区间内的动作规则不能依赖于该区间内已实现的上下文/动作序列（除当前轮次的上下文外）。对于线性上下文赌博机，我们提出了两种仅需$O(\log\log T)$次参数更新的实用算法。我们的第一种算法BLCE-G在静态调度下，同时在小$K$和大$K$区域中达到极小极大最优遗憾（在$T$的多对数因子范围内）。我们的第二种算法BLCE移除了近G-最优设计。

    arXiv:2606.00984v2 Announce Type: replace-cross  Abstract: We study linear contextual bandits under rare parameter updates: the learner may incorporate reward feedback into its parameter estimate only at a small number of update times, while still observing contexts online and selecting actions sequentially. This viewpoint clarifies a practical distinction that is often blurred in the literature: many "strictly batched" methods additionally restrict within-interval context adaptivity, meaning that the action rule inside an interval cannot depend on the sequence of realized contexts/actions in that interval (beyond the current round's context). For linear contextual bandits, we propose two practical algorithms with only $O(\log\log T)$ parameter updates. Our first algorithm BLCE-G attains minimax-optimal regret (up to polylogarithmic factors in $T$) simultaneously in both the small-$K$ and large-$K$ regimes under a static schedule. Our second algorithm BLCE removes the near G-optimal de
    
[^50]: GraphSVR：一种用于鲁棒时空空气污染预测的图卷积支持向量回归框架

    GraphSVR: A Graph Convolutional Support Vector Regression Framework for Robust Spatiotemporal Air Pollution Forecasting

    [https://arxiv.org/abs/2605.03795](https://arxiv.org/abs/2605.03795)

    本文提出了一个结合图卷积和支持向量回归的GraphSVR框架，用于鲁棒的城市空气污染时空预测，有效处理非线性动态和异常值。

    

    城市空气质量预测具有挑战性，因为污染物浓度是非线性、非平稳、时空依赖的，并且常常受到交通拥堵、工业排放和季节性气象变化等异常观测的影响。本研究提出了一种图卷积支持向量回归（GraphSVR）框架，用于城市空气污染的鲁棒时空预测。该模型结合了图卷积学习以捕捉站点间的空间依赖性，以及支持向量回归以建模非线性时间动态，同时降低对异常观测值的敏感性。该框架使用印度德里37个监测站和孟买18个监测站的空气质量记录进行评估，分别代表内陆和沿海大都市环境。预测性能在多个时间跨度上进行评估，并与已建立的时空和时序方法进行比较。

    arXiv:2605.03795v3 Announce Type: replace-cross  Abstract: Urban air quality forecasting is challenging because pollutant concentrations are nonlinear, nonstationary, spatiotemporally dependent, and often affected by anomalous observations caused by traffic congestion, industrial emissions, and seasonal meteorological variability. This study proposes a Graph Convolutional Support Vector Regression (GraphSVR) framework for robust spatiotemporal forecasting of urban air pollution. The model combines graph convolutional learning to capture inter-station spatial dependence with support vector regression to model nonlinear temporal dynamics while reducing sensitivity to outlier observations. The proposed framework is evaluated using air quality records from 37 monitoring stations in Delhi and 18 stations in Mumbai, representing inland and coastal metropolitan environments in India. Forecasting performance is assessed across multiple horizons and compared with established temporal and spatio
    
[^51]: 无需交叉拟合的多重依赖去偏机器学习方法

    Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence

    [https://arxiv.org/abs/2602.11333](https://arxiv.org/abs/2602.11333)

    本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。

    

    arXiv:2602.11333v3 公告类型：替换 摘要：本文针对广义矩估计（GMM）模型中具有一般多重聚类依赖的两步去偏机器学习（DML）估计量，开发了一种渐近理论，且不依赖交叉拟合。虽然交叉拟合被广泛使用，但当第一阶段学习器复杂且有效样本量由独立聚类数量决定时，它在统计上可能低效且计算负担沉重。我们证明，通过结合Neyman正交矩条件和基于局部化的经验过程方法，可以在不进行样本分割的情况下实现有效推断，并允许任意数量的聚类维度。结果表明，在多重聚类依赖下，所得的去偏GMM估计量具有渐近线性和渐近正态性。本文的一个核心技术贡献是为一般类别推导出新的全局和局部极大不等式。

    arXiv:2602.11333v3 Announce Type: replace  Abstract: This paper develops an asymptotic theory for two-step debiased machine learning (DML) estimators in generalised method of moments (GMM) models with general multiway clustered dependence, without relying on cross-fitting. While cross-fitting is commonly employed, it can be statistically inefficient and computationally burdensome when first-stage learners are complex and the effective sample size is governed by the number of independent clusters. We show that valid inference can be achieved without sample splitting by combining Neyman-orthogonal moment conditions with a localisation-based empirical process approach, allowing for an arbitrary number of clustering dimensions. The resulting debiased GMM estimators are shown to be asymptotically linear and asymptotically normal under multiway clustered dependence. A central technical contribution of the paper is the derivation of novel global and local maximal inequalities for general clas
    
[^52]: 你需要更好的注意力先验

    You Need Better Attention Priors

    [https://arxiv.org/abs/2601.15380](https://arxiv.org/abs/2601.15380)

    该论文通过熵最优传输统一了注意力机制，提出GOAT，用可学习先验替代均匀先验，兼容FlashAttention，解决注意力汇问题，并实现长度泛化。

    

    arXiv:2601.15380v2 公告类型：交叉替换 摘要：我们通过熵最优传输的视角来泛化注意力机制，揭示了标准注意力对应于一个由隐式均匀先验正则化的传输问题。我们引入了具有可训练先验的广义最优传输注意力（GOAT），这是一种新的注意力机制，用可学习的连续先验替代了这种朴素假设。该先验与诸如FlashAttention等优化内核保持完全兼容。GOAT还提供了基于EOT的注意力汇解释，并为其实现了具体解决方案，避免了标准注意力的表征权衡。最后，通过将空间信息吸收到核心注意力计算中，GOAT学习了一个可外推的先验，该先验结合了学习位置嵌入的灵活性与固定编码的长度泛化能力。

    arXiv:2601.15380v2 Announce Type: replace-cross  Abstract: We generalize the attention mechanism by viewing it through the lens of Entropic Optimal Transport, revealing that standard attention corresponds to a transport problem regularized by an implicit uniform prior. We introduce Generalized Optimal transport Attention with Trainable priors (GOAT), a new attention mechanism that replaces this naive assumption with a learnable, continuous prior. This prior maintains full compatibility with optimized kernels such as FlashAttention. GOAT also provides an EOT-based explanation of attention sinks and materializes a solution for them, avoiding the representational trade-offs of standard attention. Finally, by absorbing spatial information into the core attention computation, GOAT learns an extrapolatable prior that combines the flexibility of learned positional embeddings with the length generalization of fixed encodings.
    
[^53]: 统计估计中的稳定性与准确性权衡

    Stability and Accuracy Trade-offs in Statistical Estimation

    [https://arxiv.org/abs/2601.11701](https://arxiv.org/abs/2601.11701)

    本文从统计决策论角度，将稳定性视为估计约束，探讨了最坏情况和平均情况稳定性与准确性之间的权衡，揭示了稳定性带来的统计成本。

    

    arXiv:2601.11701v2 公告类型：交叉替换 摘要：算法稳定性是统计学和学习理论中的一个核心概念，它衡量算法输出对训练数据微小变化的敏感程度。稳定性在理解泛化、鲁棒性和可复制性方面起着关键作用，并且在不同的学习环境中已提出了多种稳定性概念。然而，尽管稳定性带来了理想的特性，但它通常本身不足以用于统计学习——事实上，它可能与准确性相冲突，因为一个总是输出恒定函数的算法是完全稳定的，但在统计上毫无意义。因此，理解稳定性的潜在统计成本至关重要。在本工作中，我们通过采用统计决策论视角来回答这个问题，将稳定性视为估计中的一个约束。我们聚焦于两种代表性概念——最坏情况稳定性和平均情况稳定性。

    arXiv:2601.11701v2 Announce Type: replace-cross  Abstract: Algorithmic stability is a central concept in statistics and learning theory that measures how sensitive an algorithm's output is to small changes in the training data. Stability plays a crucial role in understanding generalization, robustness, and replicability, and a variety of stability notions have been proposed in different learning settings. However, while stability entails desirable properties, it is typically not sufficient on its own for statistical learning -- and indeed, it may be at odds with accuracy, since an algorithm that always outputs a constant function is perfectly stable but statistically meaningless. Thus, it is essential to understand the potential statistical cost of stability. In this work, we address this question by adopting a statistical decision-theoretic perspective, treating stability as a constraint in estimation. Focusing on two representative notions-worst-case stability and average-case stabil
    
[^54]: 径向补偿：黎曼流形上基于图表的生成模型中的逆基础分布问题

    Radial Compensation: The Inverse Base-Distribution Problem for Chart-Based Generative Models on Riemannian Manifolds

    [https://arxiv.org/abs/2511.14056](https://arxiv.org/abs/2511.14056)

    本文揭示了球面和双曲空间中潜在变量模型的标准构造会无意中强制固定距离分布，并提出了一种闭式径向补偿方法，以实现任意预期距离分布，同时确保图表无关似然性，为变分自编码器提供了理论下界和实验验证。

    

    arXiv:2511.14056v3 公告类型：替换交叉 摘要：球面和双曲空间上的潜在变量模型通常在高斯分布下于基点处的切空间中采样，并将其推送到流形上。在这些空间中，到基点的距离是承载意义的坐标：层级中的深度、旋转的角度、蛋白质框架与参考的偏差。我们表明，标准构建会静默地将建模者意图的任何距离分布替换为一个固定的分布，即缩放卡方分布，其形状无法通过任何缩放设置改变。然后我们解决了逆问题。给定预期的距离分布，我们以闭式形式推导出实现该分布的切空间密度，证明它是广泛图表类别中唯一具有图表无关似然性的各向同性选择，并证明忽略此问题对变分自编码器造成成本的下界（带显式常数）。实验得到了精确的每次运行归一化审计的支持。

    arXiv:2511.14056v3 Announce Type: replace-cross  Abstract: Latent-variable models on spheres and hyperbolic spaces usually draw a Gaussian in the tangent space at a base point and push it onto the manifold. On these spaces the distance from the base point is the coordinate that carries meaning: depth in a hierarchy, the angle of a rotation, the deviation of a protein frame from a reference. We show that the standard construction silently replaces whatever distance distribution the modeler intended with a fixed one, a scaled chi law whose shape no setting of the scale can change. We then solve the reverse problem. Given the intended distance distribution, we derive in closed form the tangent density that realizes it, prove it is the only isotropic choice with chart-independent likelihoods for a broad class of charts, and prove a lower bound with explicit constants on what ignoring the problem costs a variational autoencoder. Experiments backed by an exact per-run normalization audit con
    
[^55]: DIGing--SGLD：时变网络上的去中心化与可扩展朗之万采样

    DIGing--SGLD: Decentralized and Scalable Langevin Sampling over Time--Varying Networks

    [https://arxiv.org/abs/2511.12836](https://arxiv.org/abs/2511.12836)

    本文提出DIGing-SGLD算法，首次将梯度跟踪机制与朗之万采样结合，在时变网络上实现了无偏且可扩展的去中心化贝叶斯采样，解决了现有方法仅适用于静态网络且存在网络效应偏差的问题。

    

    摘要：从训练数据诱导的目标分布中进行采样是贝叶斯学习的核心，其中随机梯度朗之万动力学（SGLD）是可扩展后验采样的关键工具，而去中心化变体则使得当数据分布在一个代理网络中时能够进行学习。本文介绍了DIGing-SGLD，这是一种去中心化的SGLD算法，专为在时变网络上运行的多代理系统中进行可扩展贝叶斯学习而设计。现有的去中心化SGLD方法局限于静态网络拓扑，并且许多方法即使在使用全批次时也会因网络效应而产生稳态采样偏差。DIGing-SGLD通过将基于朗之万的采样与DIGing算法的梯度跟踪机制相结合，克服了这些限制，该机制最初是为时变网络上的去中心化优化而开发的，从而无需中央协调器即可实现高效且无偏的采样。

    arXiv:2511.12836v2 Announce Type: replace-cross  Abstract: Sampling from a target distribution induced by training data is central to Bayesian learning, with Stochastic Gradient Langevin Dynamics (SGLD) serving as a key tool for scalable posterior sampling and decentralized variants enabling learning when data are distributed across a network of agents. This paper introduces DIGing-SGLD, a decentralized SGLD algorithm designed for scalable Bayesian learning in multi-agent systems operating over time-varying networks. Existing decentralized SGLD methods are restricted to static network topologies, and many exhibit steady-state sampling bias caused by network effects, even when full batches are used. DIGing-SGLD overcomes these limitations by integrating Langevin-based sampling with the gradient-tracking mechanism of the DIGing algorithm, originally developed for decentralized optimization over time-varying networks, thereby enabling efficient and bias-free sampling without a central coo
    
[^56]: 基于层次狄利克雷收缩的离散贝叶斯网络学习

    Learning discrete Bayesian networks with hierarchical Dirichlet shrinkage

    [https://arxiv.org/abs/2509.13267](https://arxiv.org/abs/2509.13267)

    本文提出了一种层次狄利克雷收缩方法，通过后验收缩到低维潜在参数来减少离散贝叶斯网络中的参数数量，并利用对数凹性实现高效采样，同时保持DAG结构。

    

    离散贝叶斯网络是由分类变量组成的有向无环图（DAG）。DBN建模的两种流行方法包括分类方法和非参数方法。然而，这两种方法通常需要大量参数，前者需要高阶交互项，后者需要单元概率。在本文中，我们提出了一种用于节点-父节点条件概率的层次模型，通过后验诱导对低维潜在参数的收缩。我们使用吉布斯采样器中的Metropolis调整Langevin算法从这些潜在变量的后验分布生成样本。此外，我们验证了在温和条件下完全条件分布是对数凹的，从而促进了高效采样。然后，我们详细介绍了多种结构学习算法，这些算法结合了我们的层次先验并保持了DAG性质。通过模拟，我们进行了评估。

    arXiv:2509.13267v3 Announce Type: replace-cross  Abstract: A discrete Bayesian network is a directed acyclic graph (DAG) consisting of categorical variables. Two popular approaches for DBN modeling include classification and nonparametric methods. However, both methods often require a large number of parameters, such as high-order interactions in the former and cell probabilities in the latter. In this article, we propose a hierarchical model for node-parent conditional probabilities, inducing shrinkage to low-dimensional latent parameters aposteriori. We generate samples from the posterior distribution of these latent variables using the Metropolis-adjusted Langevin algorithm within a Gibbs sampler. Moreover, we verify that the full conditional distribution is log-concave under mild conditions, facilitating efficient sampling. We then detail several algorithms for structure learning that incorporate our hierarchical prior and preserve the DAG property. Through simulations, we evaluate
    
[^57]: Prob-GParareal：一种用于微分方程的概率数值并行时间求解器

    Prob-GParareal: A Probabilistic Numerical Parallel-in-Time Solver for Differential Equations

    [https://arxiv.org/abs/2509.03945](https://arxiv.org/abs/2509.03945)

    本文提出了Prob-GParareal，一种概率数值并行时间求解器，通过高斯过程建模Parareal校正函数，实现微分方程求解中的不确定性量化和概率预测，并支持概率初始条件及与现有框架的无缝集成。

    

    arXiv:2509.03945v3 公告类型：替换-交叉 摘要：我们引入了Prob-GParareal，这是GParareal算法的一种概率扩展，旨在为（常微分和偏微分）方程（ODEs, PDEs）的并行时间（PinT）求解提供不确定性量化。该方法采用高斯过程（GPs）来建模Parareal校正函数，与GParareal一致，进一步实现了数值不确定性在时间上的传播，并生成系统演化的概率预测。此外，Prob-GParareal支持概率初始条件，并与经典数值求解器保持兼容，确保其能轻松集成到现有Parareal框架中。在此，我们首先对Prob-GParareal的计算复杂度进行理论分析，并推导其误差界。然后，我们在五个基准ODE系统上数值演示了所提算法的准确性和鲁棒性，包括ch（截断部分）。

    arXiv:2509.03945v3 Announce Type: replace-cross  Abstract: We introduce Prob-GParareal, a probabilistic extension of the GParareal algorithm designed to provide uncertainty quantification for the Parallel-in-Time (PinT) solution of (ordinary and partial) differential equations (ODEs, PDEs). The method employs Gaussian processes (GPs) to model the Parareal correction function, in line with GParareal, further enabling the propagation of numerical uncertainty across time and yielding probabilistic forecasts of the system's evolution. Furthermore, Prob-GParareal accommodates probabilistic initial conditions and maintains compatibility with classical numerical solvers, ensuring its straightforward integration into existing Parareal frameworks. Here, we first conduct a theoretical analysis of the computational complexity and derive error bounds of Prob-GParareal. Then, we numerically demonstrate the accuracy and robustness of the proposed algorithm on five benchmark ODE systems, including ch
    
[^58]: 分布敏感性分析：在基于样本的推断中实现可微性

    Distributional Sensitivity Analysis: Enabling Differentiability in Sample-Based Inference

    [https://arxiv.org/abs/2508.09347](https://arxiv.org/abs/2508.09347)

    本文提出了一种用于估计随机样本对分布参数敏感性的数学框架，并提供了两种解析公式和四种数值算法，以在基于样本的推断中实现可微性。

    

    摘要：本文介绍了一种数学框架，用于估计任意维度下随机样本的空间参数敏感性。这种敏感性有效地充当了随机样本相对于分布参数的梯度，这对于核物理中基于样本的反问题（例如推断量子关联函数）至关重要。我们提出了两种用于敏感性和梯度估计的解析公式。第一种将敏感性解释为一维条件分布逆映射的偏导数。第二种适用于容忍不精确梯度的优化方法，采用对角近似，以最小的精度损失降低计算成本。当封闭形式不可用时，提供了四种二阶数值算法来近似这两种表达式。验证和确认研究证实了这些算法的正确性及其有效性。

    arXiv:2508.09347v2 Announce Type: replace  Abstract: This work introduces a mathematical framework for estimating the space-parameter sensitivity of random samples in arbitrary dimensions. Such sensitivity effectively acts as gradients of random samples with respect to distributional parameters, which are essential in sample-based inverse problems in nuclear physics, such as inferring quantum correlation functions. We present two analytical formulae for sensitivity and gradient estimation. The first interprets sensitivity as the partial derivatives of the inverse mapping of 1-D conditional distributions. The second, suited for optimization methods that tolerate inexact gradients, applies a diagonal approximation that reduces computational cost with minimal accuracy loss. When closed forms are unavailable, four second-order numerical algorithms are provided to approximate both expressions. Verification and validation studies confirm the correctness of these algorithms and the effectiven
    
[^59]: 一次性鲁棒联邦独立成分分析

    One-shot Robust Federated Learning of Independent Component Analysis

    [https://arxiv.org/abs/2505.20532](https://arxiv.org/abs/2505.20532)

    提出了一种基于谱聚类和几何中位数的联邦ICA一次性聚合方法，有效解决了符号置换和异构质量问题，并在大量低质量数据下保持鲁棒性。

    

    arXiv:2505.20532v2 公告类型：替换 摘要：本文研究了分布式和联邦独立成分分析（ICA）中的鲁棒一次性聚合问题。在该场景下，每个客户端计算一个局部ICA估计器，而服务器旨在在不访问原始数据的情况下恢复一个共同的全局混合矩阵。主要难点在于局部ICA估计器仅能通过符号置换进行识别，且其估计质量可能高度异构。我们提出了谱鲁棒联邦ICA（SRF-ICA），一种一次性聚合方法，该方法从所有局部原子构建符号不变亲和矩阵，执行谱k均值以解决置换模糊性，在每个估计簇内对齐符号，然后应用几何中位数进行鲁棒聚合。我们证明了谱聚类步骤控制了簇级误聚类率，并且即使当大量局部原子来自低质量数据时，最终估计器仍能保持准确性。

    arXiv:2505.20532v2 Announce Type: replace  Abstract: This paper studies robust one-shot aggregation for distributed and federated Independent Component Analysis (ICA). In this setting, each client computes a local ICA estimator, while the server aims to recover a common global mixing matrix without accessing raw data. The main difficulty is that local ICA estimators are identifiable only up to signed permutations and may have highly heterogeneous estimation quality. We propose Spectral-Robust-Federated ICA (SRF-ICA), a one-shot aggregation method that constructs a sign-invariant affinity matrix from all local atoms, performs spectral k-means to resolve the permutation ambiguity, aligns signs within each estimated cluster, and then applies the geometric median for robust aggregation. We prove that the spectral clustering step controls the cluster-wise misclustering rate, and that the final estimator remains accurate even when a substantial fraction of local atoms are produced from low-q
    
[^60]: 将生成式学习引入表示学习：作为分布匹配的自监督迁移学习

    Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching

    [https://arxiv.org/abs/2502.14424](https://arxiv.org/abs/2502.14424)

    本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。

    

    arXiv:2502.14424v3 公告类型：替换交叉 摘要：大多数自监督学习目标旨在防止表示坍缩，但未明确目标表示规律。我们将表示学习形式化为分布匹配（DM），学习一个增强不变的编码器，其诱导的分布规律与一个明确的几何参考匹配。参考规律指定了学习到的表示分布应呈现的形式，而单独选择的差异度量则用于衡量与该目标的偏差；此处我们使用马氏距离。DM框架揭示了一个方向性反转：生成式学习将可处理的参考映射到数据，而表示学习则将数据映射到设计的参考规律。我们将总体目标与类中心分离和分类误差联系起来，并证明了非渐近神经筛保证。模拟和图像基准测试显示了流形校正、细粒度结构和跨标签空间的迁移能力。

    arXiv:2502.14424v3 Announce Type: replace-cross  Abstract: Most self-supervised learning objectives defend against collapse but leave the target representation law unspecified. We formulate representation learning as Distribution Matching (DM), learning an augmentation-invariant encoder whose induced law matches an explicit geometric reference. The reference law specifies what the learned representation distribution should look like, whereas a separately chosen discrepancy determines how deviations from this target are measured; here we use Mallows distance. The DM framework reveals a directional inverse: generative learning maps a tractable reference to data, whereas representation learning maps data to a designed reference law. We connect the population objective to class-centre separation and classification error and prove a non-asymptotic neural-sieve guarantee. Simulations and image benchmarks show manifold rectification, fine-grained structure and transfer across label spaces.
    
[^61]: 半参数双重强化学习及其在长期因果推断中的应用

    Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference

    [https://arxiv.org/abs/2501.06926](https://arxiv.org/abs/2501.06926)

    本文提出一种半参数双重强化学习方法，通过直接对Q函数施加工作性半参数限制，以提升长期因果推断中的估计效率与稳定性，特别是在时间重叠较弱的情况下。

    

    arXiv:2501.06926v5 公告类型：替换 摘要：双重强化学习（DRL）为非参数马尔可夫决策过程（MDP）中的策略值提供了高效的在策略外推断，但完全非参数估计器在时间重叠较弱且占用率比高维时可能不稳定。这一局限性在随机实验的长期因果推断中尤为相关：随机化确保了治疗分配的平衡，但并不能保证持续干预使用所导致的未来状态轨迹的平衡。我们针对无限时域$Q$-函数的连续线性泛函开发了半参数DRL。我们不是对奖励和转移规律施加线性MDP结构，而是对$Q$-函数本身（即折扣贝尔曼方程的解）施加工作性半参数限制。当这些限制正确时，相对于无限制DRL，它们可以提高效率，同时允许丰富、可能无限维的模型。

    arXiv:2501.06926v5 Announce Type: replace  Abstract: Double reinforcement learning (DRL) provides efficient off-policy inference for policy values in nonparametric Markov decision processes (MDPs), but fully nonparametric estimators can be unstable when intertemporal overlap is weak and occupancy ratios are high-dimensional. This limitation is especially relevant for long-term causal inference from randomized experiments: randomization ensures overlap in treatment assignment, but not over future state trajectories induced by continued intervention use. We develop semiparametric DRL for continuous linear functionals of the infinite-horizon $Q$-function. Rather than impose linear MDP structure on the reward and transition laws, we place working semiparametric restrictions on the $Q$-function itself, the solution of the discounted Bellman equation. When correct, these restrictions can improve efficiency relative to unrestricted DRL while allowing rich, possibly infinite-dimensional models
    
[^62]: 非线性单变量模型的条件回归

    Conditional regression for the Nonlinear Single-Variable Model

    [https://arxiv.org/abs/2411.09686](https://arxiv.org/abs/2411.09686)

    本文提出了一种针对非线性单变量组合模型的条件回归估计方法，通过响应切片和局部主成分分析克服了高维预测变量下的维度灾难。

    

    arXiv:2411.09686v4 公告类型：替换 摘要：在不遭受统计和计算维度灾难的情况下对定义在 $\mathbb{R}^d$ 上的函数 $F$ 进行回归，需要可利用的结构。组合模型 $F=f\circ g$ 中，$g$ 具有低维值域，这包括经典的单指数和多指数模型以及某些神经网络；虽然线性 $g$ 的情形已被充分理解，但对于非线性 $g$ 的认知则显著不足。我们研究模型 $F(X)=f(\Pi_\gamma X)$，其中 $\Pi_\gamma$ 是与未知正则曲线 $\gamma$ 相关联的最近点坐标，而 $f$ 是未知的一维链接函数。预测变量 $X$ 不必是内在低维的，其变异性可以在曲线的管状邻域内具有全维度变化。我们构建了一种基于响应切片、局部主成分分析、数据自适应切片分配和一维局部多项式回归的非参数估计器。在粗（coarse）条件下...

    arXiv:2411.09686v4 Announce Type: replace  Abstract: Regressing a function $F$ on $\mathbb{R}^d$ without incurring the statistical and computational curse of dimensionality requires exploitable structure. Compositional models $F=f\circ g$ in which $g$ has a low-dimensional range include classical single- and multi-index models as well as certain neural networks; while the case of linear $g$ is well understood, substantially less is known for nonlinear $g$. We study the model $F(X)=f(\Pi_\gamma X)$, where $\Pi_\gamma$ is the closest-point coordinate associated with an unknown regular curve $\gamma$, and $f$ is an unknown one-dimensional link function. The predictor $X$ need not be intrinsically low-dimensional and may have full-dimensional variation throughout a tubular neighborhood of the curve. We construct a nonparametric estimator based on response slicing, local principal component analysis, data-adaptive slice assignment, and one-dimensional local polynomial regression. Under coar
    
[^63]: 通过留一变量法交叉验证因果发现

    Cross-validating causal discovery via Leave-One-Variable-Out

    [https://arxiv.org/abs/2411.05625](https://arxiv.org/abs/2411.05625)

    本文提出一种无需真实因果图的留一变量法（LOVO）预测方法，通过分别对排除一个变量的数据集进行因果发现，实现对因果发现算法的交叉验证，并能估计条件期望而不需联合观测。

    

    我们提出了一种新方法，无需真实因果图即可对因果发现算法进行证伪，该方法基于在因果模型学习过程中排除一个变量对来测试模型。具体而言，给定数据$X, Y, \boldsymbol{Z}=X, Y, Z_1,\dots,Z_k$，我们分别将因果发现算法应用于“留一”数据集$X, \boldsymbol{Z}$和$Y, \boldsymbol{Z}$。我们证明，这两个结果因果模型（以DAG、ADMG、CPDAG或PAG形式表示）通常能推断出$X$和$Y$之间的依赖关系结论，并允许仅基于留一数据集估计$\mathbb{E}(Y\mid X=x)$，而无需$X$和$Y$的联合观测。这种估计称为“留一变量法（LOVO）”预测。其误差可以被估计，因为联合分布$P(X, Y)$是可用的，且$X$和$Y$仅被省略用于证伪目的。我们提出了LOVO预测的两种变体。

    arXiv:2411.05625v2 Announce Type: replace  Abstract: We propose a new approach to falsify causal discovery algorithms without ground truth, which is based on testing the causal model on a variable pair excluded during learning the causal model. Specifically, given data on $X, Y, \boldsymbol{Z}=X, Y, Z_1,\dots,Z_k$, we apply the causal discovery algorithm separately to the 'leave-one-out' data sets $X, \boldsymbol{Z}$ and $Y, \boldsymbol{Z}$. We demonstrate that the two resulting causal models, in the form DAGs, ADMGs, CPDAGs or PAGs, often entail conclusions on the dependencies between $X$ and $Y$ and allow to estimate $\mathbb{E}(Y\mid X=x)$ without any joint observations of $X$ and $Y$, given only the leave-one-out datasets. This estimation is called "Leave-One-Variable-Out (LOVO)" prediction. Its error can be estimated since the joint distribution $P(X, Y)$ is available, and $X$ and $Y$ have only been omitted for the purpose of falsification.   We present two variants of LOVO predic
    
[^64]: 不平衡分类问题的鲁棒性能指标

    Robust performance metrics for imbalanced classification problems

    [https://arxiv.org/abs/2404.07661](https://arxiv.org/abs/2404.07661)

    本文提出了一种通过引入调整参数来鲁棒化MCC、科恩κ和F分数等性能指标的方法，以确保在不平衡分类问题中分类器不会忽略少数类别。

    

    arXiv:2404.07661v2 公告类型：替换 摘要：我们表明，在二分类中已建立的性能指标，如马修斯相关系数（MCC）、科恩的κ系数、F分数或杰卡德相似系数，对于类别不平衡并不鲁棒，即如果少数类别的比例趋近于0，贝叶斯分类器在这些指标下的真正例率（TPR）也趋近于0。因此，在不平衡分类问题中，这些指标偏向于忽略少数类别的分类器。为解决此问题，我们引入了MCC、科恩的κ系数和F分数的鲁棒化修改，并添加了一个额外的调整参数，该参数允许调整对类别不平衡的鲁棒性程度。作为理论保证，我们表明，对于这些鲁棒化性能指标，当以类别条件密度f_i的密度比f_1/f_0表示时，贝叶斯最优分类器具有一个阈值参数。

    arXiv:2404.07661v2 Announce Type: replace  Abstract: We show that established performance metrics in binary classification, such as Matthews' correlation coefficient (MCC), Cohen's $\kappa$, the F-score or the Jaccard similarity coefficient are not robust to class imbalance in the sense that if the proportion of the minority class tends to $0$, the true positive rate (TPR) of the Bayes classifier under these metrics tends to $0$ as well. Thus, in imbalanced classification problems, these metrics favour classifiers which ignore the minority class. To alleviate this issue we introduce robustified modifications of the MCC, of Cohen's $\kappa$ and of the F-score with an additional tuning parameter which allows to adapt the amount of robustness against class imbalance. As theoretical guarantee we show that the Bayes-optimal classifier for these robustified performance metrics, when expressed in terms of the density ratio $f_1/f_0$ of the class-conditional densities $f_i$, has a threshold pa
    
[^65]: 深度聚类评估：如何验证内部聚类有效性测量方法

    Deep Clustering Evaluation: How to Validate Internal Clustering Validation Measures

    [https://arxiv.org/abs/2403.14830](https://arxiv.org/abs/2403.14830)

    本文解决了深度聚类方法在评估聚类质量时面临的挑战，提出了一种系统方法来应用聚类有效性指标。

    

    arXiv:2403.14830v1 通告类型：跨领域 摘要：深度聚类是一种使用深度神经网络对复杂、高维数据进行划分的方法，它面临着独特的评估挑战。传统的聚类验证方法，设计用于低维空间，对于涉及将数据投影到较低维嵌入空间后再进行划分的深度聚类来说是有问题的。论文确定了两个关键问题：1）在将这些方法应用于原始数据时的维度灾难，2）由于不同聚类模型的训练过程和参数设置的变化而导致不同嵌入空间中的聚类结果无法可靠比较。本文解决了在深度学习中评估聚类质量所面临的挑战。我们提出了一个理论框架来强调在原始数据和嵌入数据上使用内部验证方法可能出现的无效性，并提出了一种系统方法来应用深度聚类有效性指标。

    arXiv:2403.14830v1 Announce Type: cross  Abstract: Deep clustering, a method for partitioning complex, high-dimensional data using deep neural networks, presents unique evaluation challenges. Traditional clustering validation measures, designed for low-dimensional spaces, are problematic for deep clustering, which involves projecting data into lower-dimensional embeddings before partitioning. Two key issues are identified: 1) the curse of dimensionality when applying these measures to raw data, and 2) the unreliable comparison of clustering results across different embedding spaces stemming from variations in training procedures and parameter settings in different clustering models. This paper addresses these challenges in evaluating clustering quality in deep learning. We present a theoretical framework to highlight ineffectiveness arising from using internal validation measures on raw and embedded data and propose a systematic approach to applying clustering validity indices in deep 
    
[^66]: 基于物理的无量纲特征用于机器学习驱动的暴雨洪水绘图

    Physically-based dimensionless features for pluvial flood mapping with machine learning

    [https://arxiv.org/abs/2211.00636](https://arxiv.org/abs/2211.00636)

    本文提出一种基于无量纲多尺度特征和逻辑回归的机器学习框架，通过捕捉洪水过程的跨区域相似性，显著提升暴雨洪水绘图的泛化能力和预测效率。

    

    快速划定骤发洪水范围对于调动应急资源和组织疏散至关重要，从而保护生命和财产。机器学习方法相较于传统的高分辨率二维洪水模型，能够以更低的计算需求实现快速洪水划定。然而，现有的机器学习方法受限于对未见过的条件的泛化能力不足。在此，我们提出一个框架，基于无量纲、多尺度特征来改进机器学习模型的泛化能力，这些特征捕捉了跨区域洪水过程的相似性。无量纲特征通过白金汉Π定理约束，并与逻辑回归模型结合，用于洪水风险的概率性判定。通过改变河流划定的累积阈值，在不同尺度上计算这些特征。模拟的洪水地图与二维水动力模型的结果吻合良好。

    arXiv:2211.00636v4 Announce Type: replace-cross  Abstract: Rapid delineation of flash flood extents is critical to mobilize emergency resources and to manage evacuations, thereby saving lives and property. Machine learning (ML) approaches enable rapid flood delineation with reduced computational demand compared to conventional high-resolution, 2D flood models. However, existing ML approaches are limited by a lack of generalization to never-before-seen conditions. Here, we propose a framework to improve ML model generalization based on dimensionless, multi-scale features that capture the similarity of the flooding process across regions. The dimensionless features are constrained with the Buckingham $\Pi$ theorem and used with a logistic regression model for a probabilistic determination of flood risk. The features were calculated at different scales by varying accumulation thresholds for stream delineation. The modeled flood maps compared well with the results of 2D hydraulic models th
    
[^67]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    
[^68]: 嵌套消除：一种从基于选择的反馈中识别最佳项目的简单算法

    Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback. (arXiv:2307.09295v1 [cs.LG])

    [http://arxiv.org/abs/2307.09295](http://arxiv.org/abs/2307.09295)

    嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。

    

    我们研究了基于选择的反馈中识别最佳项目的问题。在这个问题中，公司依次向一群顾客展示显示集，并收集他们的选择。目标是以最少的样本数量和高置信水平识别出最受欢迎的项目。我们提出了一种基于消除的算法，即嵌套消除(Nested Elimination，NE)，它受到信息理论下界所暗示的嵌套结构的启发。NE的结构简单，易于实施，具有对样本复杂度的强大理论保证。具体而言，NE利用了一种创新的消除准则，并避免了解决任何复杂的组合优化问题的需要。我们提供了NE的特定实例和非渐近性的样本复杂度的上界。我们还展示了NE实现了高阶最坏情况渐近最优性。最后，来自合成和真实数据的数值实验验证了我们的理论。

    We study the problem of best-item identification from choice-based feedback. In this problem, a company sequentially and adaptively shows display sets to a population of customers and collects their choices. The objective is to identify the most preferred item with the least number of samples and at a high confidence level. We propose an elimination-based algorithm, namely Nested Elimination (NE), which is inspired by the nested structure implied by the information-theoretic lower bound. NE is simple in structure, easy to implement, and has a strong theoretical guarantee for sample complexity. Specifically, NE utilizes an innovative elimination criterion and circumvents the need to solve any complex combinatorial optimization problem. We provide an instance-specific and non-asymptotic bound on the expected sample complexity of NE. We also show NE achieves high-order worst-case asymptotic optimality. Finally, numerical experiments from both synthetic and real data corroborate our theore
    
[^69]: 神经因果因素分析

    Neuro-Causal Factor Analysis. (arXiv:2305.19802v1 [stat.ML])

    [http://arxiv.org/abs/2305.19802](http://arxiv.org/abs/2305.19802)

    该论文提出了一种名为神经因果因素分析（NCFA）的新方法，它通过学习到的图形匹配马尔可夫因式分解的分布来识别因素，并使用变分自编码器（VAE）对数据进行重建任务。与标准VAE相比，NCFA具有更稀疏的架构和低模型复杂度，具有因果解释性。

    

    因素分析是一种通过研究带有一些相互依赖关系的观察变量可以如何表示为相互独立的未观察因素的函数的统计工具，并广泛应用于心理学、生物学和物理科学领域。我们从因果发现和深度学习的新视角重新审视这种经典方法，引入了神经因果因素分析（NCFA）的框架。我们的方法是完全非参数的：它通过潜在的因果发现方法识别因素，然后使用变分自编码器（VAE），该VAE受到与学习图的关于马尔可夫因式分解的分布相符的限制。我们评估了NCFA在真实的和合成的数据集上，发现它在数据重建任务上的表现与标准VAE相当，但具有更稀疏的架构、更低的模型复杂度和因果可解释性。与传统的FA方法不同，我们提出的NCFA方法可以通过学习到的图形表示因素之间的因果关系，从而具有因果解释性。

    Factor analysis (FA) is a statistical tool for studying how observed variables with some mutual dependences can be expressed as functions of mutually independent unobserved factors, and it is widely applied throughout the psychological, biological, and physical sciences. We revisit this classic method from the comparatively new perspective given by advancements in causal discovery and deep learning, introducing a framework for Neuro-Causal Factor Analysis (NCFA). Our approach is fully nonparametric: it identifies factors via latent causal discovery methods and then uses a variational autoencoder (VAE) that is constrained to abide by the Markov factorization of the distribution with respect to the learned graph. We evaluate NCFA on real and synthetic data sets, finding that it performs comparably to standard VAEs on data reconstruction tasks but with the advantages of sparser architecture, lower model complexity, and causal interpretability. Unlike traditional FA methods, our proposed N
    
[^70]: 无需信号建模的信号识别方法

    Signal identification without signal formulation. (arXiv:2304.06522v1 [physics.data-an])

    [http://arxiv.org/abs/2304.06522](http://arxiv.org/abs/2304.06522)

    该研究提出了一种无需信号建模即可识别信号的方法，该方法基于样本和其邻居之间相对距离，可以在小样本和高维数据中识别“类似于信号”的变量。

    

    当信号和噪声混合时，物理学家通常通过信号建模来识别信号，而统计学家则相反，他们试图对噪声进行建模来识别信号。在本研究中，我们应用了统计学家的信号检测概念，对具有小样本和高维数据的物理数据进行了处理，而不对信号进行建模。自然界中的大部分数据，无论是噪声还是信号，都被假定为是由动态系统生成的；因此，在这些生成过程之间基本上没有区别。我们提出了动态系统的相关长度和样本数对于在这样的系统中生成的信号变量中区分噪声变量的实际定义至关重要。由于具有短期相关性的变量随着样本数的减少会更快地达到正态分布，因此它们被认为是“类似于噪声”的变量，而具有相反特性的变量则是“类似于信号”的变量。正态性检验不适用于小样本和高维数据，因此我们提出了一种基于样本和其邻居之间相对距离的新方法来识别“类似于噪声”的变量。实验证明，所提出的方法可以在不进行任何信号建模的情况下识别“类似于信号”的变量。

    When there are signals and noises, physicists try to identify signals by modeling them, whereas statisticians oppositely try to model noise to identify signals. In this study, we applied the statisticians' concept of signal detection of physics data with small-size samples and high dimensions without modeling the signals. Most of the data in nature, whether noises or signals, are assumed to be generated by dynamical systems; thus, there is essentially no distinction between these generating processes. We propose that the correlation length of a dynamical system and the number of samples are crucial for the practical definition of noise variables among the signal variables generated by such a system. Since variables with short-term correlations reach normal distributions faster as the number of samples decreases, they are regarded to be ``noise-like'' variables, whereas variables with opposite properties are ``signal-like'' variables. Normality tests are not effective for data of small-
    

