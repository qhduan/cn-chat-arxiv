# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pointwise Majorization for sub-Weibull and Mixed Tail Processes with Applications in Quadratic Chaos and Ergodic Diffusions](https://arxiv.org/abs/2609.01576) | 本文建立了首个针对次Weibull与混合尾Banach值随机过程的同时逐点控制理论，其尾界由逐点Fernique-Talagrand泛函刻画的逐点复杂度决定而非全局最坏情况界，并应用于二次混沌与遍历扩散。 |
| [^2] | [Variable Selection for Feature-Based Newsvendor](https://arxiv.org/abs/2609.01544) | 本文针对基于特征的报童问题，在特征数量的硬基数约束下提出变量选择方法，通过带ℓ₂正则化的ℓ₀约束建模、强化的混合整数二阶锥规划重构以及随机舍入和贪心等可扩展算法，同时给出了稀疏库存策略估计量的理论保证。 |
| [^3] | [On the Reliability of Generative Augmentation: A Wasserstein-Based Theoretical and Empirical Study](https://arxiv.org/abs/2609.01410) | 该论文建立了条件生成式数据增强的统计理论框架，证明分类风险的失真由增强强度和真实与生成分布之间的类条件Wasserstein距离共同控制，并推导出基于Rademacher复杂度的泛化界，揭示了假设复杂度、增强强度与生成保真度之间的权衡，实证表明CWGAN-GP在不平衡分类任务上表现更优。 |
| [^4] | [Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity](https://arxiv.org/abs/2609.01397) | 该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。 |
| [^5] | [Matched Queries for Curvature and Density at Branching Junctions](https://arxiv.org/abs/2609.01319) | 本文提出在两个噪声尺度下进行匹配得分查询并相减的方法，消去切向贡献后线性地暴露出包含分支曲率与对数密度斜率信息的项 G，并证明在已知各射线的切向方向与权重时，交汇点处全部 sD 个分支参数可被唯一识别。 |
| [^6] | [One-Layer Transformer Provably Learns Multiclass One-Nearest Neighbor in Context](https://arxiv.org/abs/2609.01311) | 本文证明了带argmax分类头的单层Transformer在多分类的上下文学习中行为与单最近邻分类器完全一致，填补了此前工作依赖非标准舍入方法所留下的理论空白。 |
| [^7] | [Multi-Head Self Attention is a Parameter Identification Mechanism](https://arxiv.org/abs/2609.01231) | 该论文证明多头缩放点积注意力本质上是一种参数辨识机制——头数越多，未辨识参数比例从 1/2 降至 1/(2H)，但注意力永远无法被完全辨识，并且这一视角还能解释 RoPE 和 GQA 等现代改进为何能提升“有意义”参数的占比。 |
| [^8] | [Nonparametric inference for density-dependent McKean--Vlasov diffusions](https://arxiv.org/abs/2609.01166) | 该论文基于稀疏ReQU神经网络构造筛最大似然估计量，实现了多元McKean--Vlasov扩散中密度依赖漂移系数与平稳密度的非参数估计，并给出了匹配的Assouad下界证明其收敛速度的理论最优性。 |
| [^9] | [Artificial Rosetta Stone: Constrained Maximum A Posteriori (MAP) Reconstruction of Symbolic Raga Sequences via Order-k Markov Models](https://arxiv.org/abs/2609.01064) | 本文提出“人工罗塞塔石碑”框架，将印度拉格音乐中缺失音符的重建形式化为带约束的最大后验估计问题，并利用k阶马尔可夫模型和动态规划给出精确可解的方案。 |
| [^10] | [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion](https://arxiv.org/abs/2609.01043) | 提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。 |
| [^11] | [The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow](https://arxiv.org/abs/2609.01034) | 本文通过将损失函数分解为 $f = g + \varepsilon h$ 的微扰分析，首次为深度学习稳定边缘处梯度下降的中心流提供了系统性推导，并揭示出其中存在快速振荡、中间自稳定与缓慢中心流演化三个时间尺度。 |
| [^12] | [Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches](https://arxiv.org/abs/2609.00946) | 本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。 |
| [^13] | [When Metropolis and Hastings Meet Bradley and Terry: Exact MCMC From Preference Voting](https://arxiv.org/abs/2609.00905) | 提出Pref-MH，一种仅依靠Bradley-Terry裁判的随机二元成对偏好比较即可实现精确Metropolis-Hastings条件采样的通用算法，并设计了可证明收敛的接受/拒绝规则。 |
| [^14] | [Semi-Supervised Classification with Informative Missing Labels in Weibull Mixture Models](https://arxiv.org/abs/2609.00774) | 该论文提出在两分量威布尔混合模型的半监督分类中，将标签缺失概率建模为分类不确定性的函数，从而证明缺失标签指示变量本身携带分类器信息，并据此刻画了贝叶斯决策边界的结构并推导了相应的Fisher信息量。 |
| [^15] | [Deep Skew-t Mixture Models](https://arxiv.org/abs/2609.00773) | 提出深度偏斜t混合模型（DStMM），通过沿潜变量路径传播共享的逆伽马混合变量，联合建模高维聚类中的厚尾性与方向性偏斜，并采用随机/蒙特卡洛EM算法进行估计。 |
| [^16] | [Verdict Instability of OOD Scores under Reference Resampling](https://arxiv.org/abs/2609.00691) | 本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。 |
| [^17] | [An efficient EM algorithm for both element-wise and structural missingness in matrix-variate normal mixture models](https://arxiv.org/abs/2609.00616) | 本文提出一种高效的部分EM算法，通过坐标级逼近处理矩阵正态混合模型中元素级与子矩阵级缺失数据，保持Kronecker结构并避免昂贵的协方差矩阵求逆，从而显著降低计算成本。 |
| [^18] | [A convolutional framework for detecting event-driven dynamics in energy price series](https://arxiv.org/abs/2609.00402) | 本文提出了一个通用卷积神经网络框架，可精确表示并逼近多种事件检测统计量（如极差、最大回撤、波动率等），并在能源价格序列中有效识别出以地缘政治为主的事件驱动动态。 |
| [^19] | [Neural means and kernel corrections for operator learning](https://arxiv.org/abs/2609.00389) | 该论文提出将神经网络均值与Matérn核回归修正相结合的方法，在结构力学和OCO-2辐射传输两个基准问题上达到了或超越了已发表最佳结果，并从理论上证明和量化了核修正之所以有效的机制。 |
| [^20] | [Exact Global MCMC with Denoising Diffusion](https://arxiv.org/abs/2609.00279) | 该论文提出去噪扩散蒙特卡洛方法，通过在MALA局部采样样本上训练去噪扩散模型并施加Metropolis-Hastings精确校正步骤，为复杂高维目标分布提供高接受率的全局MCMC提议。 |
| [^21] | [Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost](https://arxiv.org/abs/2609.00193) | 提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。 |
| [^22] | [Stochastic complexity of vectors containing cluster structure](https://arxiv.org/abs/2609.00084) | 本文提出一种递归公式来高效计算NML模型的归一化常数，将计算包含聚类结构向量最短编码长度的时间复杂度从多项式时间降低到线性时间。 |
| [^23] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^24] | [Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess](https://arxiv.org/abs/2608.27757) | 该论文提出用朝向网络自身MCTS先验的前向质量覆盖KL散度（先验引导探索）替代传统熵奖励，并结合由价值头不确定性驱动的熵自适应采样温度，通过自我对弈强化学习将无搜索国际象棋网络的谜题准确率从93.9%提升至94.9%。 |
| [^25] | [Common-Center Geometry and Certified Radial Reconstruction for Energy-Form Full Conformal Regions](https://arxiv.org/abs/2608.24964) | 本文证明了在对称性和凸性条件下，能量形式全共形预测区域呈星形，且对于幂距离在β≥1时具有确定性几何性质，同时指出候选评分凸性不足以保证连通性。 |
| [^26] | [Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining](https://arxiv.org/abs/2608.24814) | 本文发现语言模型预训练中，有效学习率（LR与参数范数的比值）是损失动态的核心控制变量，匹配ELR可使不同配置的损失轨迹坍缩一致，并据此提出了可跨方法迁移的缩放定律。 |
| [^27] | [Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements](https://arxiv.org/abs/2608.18294) | 本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。 |
| [^28] | [Logarithmic-Free Moment and Generalization Bounds for Uniformly Stable Algorithms](https://arxiv.org/abs/2608.09870) | 该论文去除了一致稳定算法泛化界中多余的对数因子 $\log n$，证明了无对数的矩不等式，从而肯定地回答了Bousquet等人（2020）提出的公开问题。 |
| [^29] | [Seeing the Forest for the Trees: The Gaussian Process Limit of BART](https://arxiv.org/abs/2607.28844) | 本文证明当树的数量趋于无穷时，BART收敛于一个具有特定核函数的高斯过程，并引入随机树特征作为其近似，实现了仅以对数方式依赖维度的极小化极大最优学习率，从而解释了BART优异性能的来源。 |
| [^30] | [DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking](https://arxiv.org/abs/2605.26087) | 提出了交互式基准测试DiscoverPhysics，通过让大语言模型在物理规律刻意偏离现实的22个模拟世界中设计实验、观察轨迹数据并归纳未知的运动定律，从而将模型真正的科学推理能力与对既有物理知识的记忆区分开来。 |
| [^31] | [FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet](https://arxiv.org/abs/2604.22494) | 提出了FedSPDnet框架，通过ProjAvg和RLAvg两种保持Stiefel流形几何结构的聚合策略，实现了基于SPD矩阵的联邦深度学习，在EEG运动想象基准上以更少的通信参数和更强的鲁棒性超越了联邦EEGnet。 |
| [^32] | [Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence](https://arxiv.org/abs/2602.11333) | 本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。 |
| [^33] | [Persistent Entropy as a Detector of Phase Transitions](https://arxiv.org/abs/2602.09058) | 本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。 |
| [^34] | [Modeling Information Blackouts in Missing Not-At-Random Time Series Data](https://arxiv.org/abs/2601.01480) | 该论文提出了一种感知非随机缺失（MNAR）的潜在状态空间模型，用于建模交通传感器网络中的连续信息中断，证明当缺失机制依赖于潜在交通状态时，考虑这种依赖关系可显著提升数据插补精度与缺失检测性能。 |
| [^35] | [Model Predictive Control is almost Optimal for Heterogeneous Restless Multi-armed Bandits](https://arxiv.org/abs/2511.08097) | 本文针对每个臂参数各不相同的异质无限时域不休息多臂老虎机，证明通过反复求解有限线性规划的模型预测控制策略（LP-update）在一致遍历性假设下具有 O(√(1/N)) 的次优性差距，即该经典算法几乎是最优的。 |
| [^36] | [If you can distinguish, you can express: Galois theory, Stone--Weierstrass, machine learning, and linguistics](https://arxiv.org/abs/2510.09902) | 本文揭示了伽罗瓦理论基本定理与Stone–Weierstrass定理的共同本质——区分能力决定表达能力，并将这一原理延伸至机器学习、数据科学与语言学领域。 |
| [^37] | [Performance-Efficiency Tradeoffs in Transformers: An Approximation Theory Perspective](https://arxiv.org/abs/2510.03784) | 本文从逼近理论视角刻画了Transformer中注意力头数量与头维度在固定参数预算下的权衡，发现并证明了softmax激活的饱和行为，表明较深的层可以用更小的头维度实现高效运行。 |
| [^38] | [AL-SPCE - Reliability analysis for nondeterministic models using stochastic polynomial chaos expansions and active learning](https://arxiv.org/abs/2507.04553) | 提出了一种结合随机多项式混沌展开与主动学习的方法AL-SPCE，能够以显著更少的训练样本对具有随机性的非确定性模型进行高精度、低成本的可靠性分析。 |
| [^39] | [Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture](https://arxiv.org/abs/2506.19935) | 本研究将掩码扩散模型置于仅解码器架构框架中，与自回归模型进行公平比较，发现其通过温度退火等技术可实现约25倍的推理加速且困惑度相当，为降低大语言模型推理计算成本提供了新路径。 |
| [^40] | [On the Existence of Consistent Adversarial Attacks in High-Dimensional Linear Classification](https://arxiv.org/abs/2506.12454) | 本文提出了一种新的误差度量来区分真正的一致性对抗攻击（即保持真实标签不变的扰动）与因数据有限或模型能力不足导致的普通误分类，并通过精确的渐近理论分析证明，随着模型过参数化程度的提高，其对标签保持扰动的脆弱性会不断增大。 |
| [^41] | [Online simultaneous inference for quantiles via smoothed stochastic gradient descent](https://arxiv.org/abs/2505.13299) | 本文提出一种平滑随机梯度下降方法用于流数据的在线分位数估计，其估计量在每次迭代中关于分位数水平单调，并借助一致Bahadur表示与布朗桥最大值的高斯近似，实现了维度随样本量指数增长时跨坐标与分位数水平的在线同时统计推断。 |
| [^42] | [Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms](https://arxiv.org/abs/2502.20115) | 本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。 |
| [^43] | [Generalization Bounds for Markov Algorithms through Entropy Flow Computations](https://arxiv.org/abs/2502.07584) | 该论文提出新的技术工具，将熵流方法的适用范围从特定的噪声和算法结构（如朗之万动力学）扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法，从而为这一广泛类别的算法建立泛化界。 |
| [^44] | [QABBA: Error-Guaranteed Symbolic Time-Series Compression via Integer-Quantized Aggregation](https://arxiv.org/abs/2411.15209) | 提出QABBA，通过量化符号中心实现ABBA的整数化压缩，在保证重建质量的同时提供严格的误差界限。 |
| [^45] | [Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies](https://arxiv.org/abs/2408.12845) | 针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。 |
| [^46] | [Deep learning based numerical approximation algorithms for stochastic partial differential equations](https://arxiv.org/abs/2012.01194) | 本文提出一种基于深度学习的随机偏微分方程逼近算法，通过神经网络沿噪声轨迹逼近SPDE解并估计其经验分布，在随机热方程、Black-Scholes方程和Zakai方程等测试中实现了高达100维空间下的快速精确求解。 |

# 详细

[^1]: 次Weibull过程与混合尾过程的逐点控制理论及其在二次混沌与遍历扩散中的应用

    Pointwise Majorization for sub-Weibull and Mixed Tail Processes with Applications in Quadratic Chaos and Ergodic Diffusions

    [https://arxiv.org/abs/2609.01576](https://arxiv.org/abs/2609.01576)

    本文建立了首个针对次Weibull与混合尾Banach值随机过程的同时逐点控制理论，其尾界由逐点Fernique-Talagrand泛函刻画的逐点复杂度决定而非全局最坏情况界，并应用于二次混沌与遍历扩散。

    

    经典链式方法通过单一的最坏情况界来控制由指标索引的随机过程，这可能掩盖指标集上各点之间的显著差异。我们建立了首个针对具有次Weibull增量或双度量混合尾增量的Banach值过程的同时逐点控制理论。对于可分指标空间上的锚定次Weibull过程，记 $v(t):=d(t,t_0)$。给定参考测度 $\mu$，点 $t$ 处的包络由 $\alpha$ 阶逐点Fernique-Talagrand泛函 $\Phi_{\mu,d}^{(\alpha)}(t):=\int_0^{4v(t)}(\log\frac{1}{\mu(B_d(t,r))})^{1/\alpha}dr$ 控制。对任意 $\delta\in(0,1)$，我们得到 $\mathbb{P}(\|Z_t\|\lesssim\{\Phi_{\mu,d}^{(\alpha)}(t)+v(t)(\log(e/\delta))^{1/\alpha}\},\forall t)\ge 1-\delta$。我们的界由逐点复杂度 $\Phi_{\mu,d}^{(\alpha)}$ 而非全局量决定。该结果对任意 $\alpha>0$ 均成立，且不涉及二进对数因子（摘要原文在此处截断）。

    arXiv:2609.01576v1 Announce Type: cross  Abstract: Classical chaining controls an indexed stochastic process through a single worst-case bound, which can obscure substantial variation across the index set. We establish the first simultaneous pointwise majorization theory for Banach-valued processes with sub-Weibull or two-metric mixed-tail increments. For an anchored sub-Weibull process on a separable index space, write $v(t):=d(t,t_0)$. Given a reference measure $\mu$, the envelope at $t$ is governed by the pointwise Fernique-Talagrand functional of order $\alpha$, $\Phi_{\mu,d}^{(\alpha)}(t):=\int_0^{4v(t)}(\log\frac{1}{\mu(B_d(t,r))})^{1/\alpha}dr$. $\forall \delta\in(0,1)$, we obtain that $$ \mathbb{P}(\|Z_t\|\lesssim\{\Phi_{\mu,d}^{(\alpha)}(t)+v(t)(\log(e/\delta))^{1/\alpha}\},\forall t)\ge 1-\delta. $$ Our bound is determined by the pointwise complexity $\Phi_{\mu,d}^{(\alpha)}$ rather than a global quantity. The result holds for every $\alpha>0$ and does not involve dyadic loga
    
[^2]: 基于特征的报童问题中的变量选择

    Variable Selection for Feature-Based Newsvendor

    [https://arxiv.org/abs/2609.01544](https://arxiv.org/abs/2609.01544)

    本文针对基于特征的报童问题，在特征数量的硬基数约束下提出变量选择方法，通过带ℓ₂正则化的ℓ₀约束建模、强化的混合整数二阶锥规划重构以及随机舍入和贪心等可扩展算法，同时给出了稀疏库存策略估计量的理论保证。

    

    基于特征的报童模型利用可观测的协变量来定制库存决策，旨在需求不确定性下平衡持有成本与缺货成本。然而，高维特征集合往往损害模型的可解释性，并增加数据收集与实施成本。本文研究了在所选特征数量受硬基数约束下，基于特征的报童问题的变量选择。我们对由此产生的带ℓ₂正则化的ℓ₀约束经验报童问题进行建模，证明了其计算难度，并开发了一种混合整数二阶锥规划重构方法，该方法强化了标准的Big-M公式。为了在精确优化之外实现可扩展性，我们开发了一种具有双准则保证的随机舍入算法以及一种贪心启发式算法。在统计方面，我们对所得的稀疏策略估计量提供了理论分析，包括有限样本（摘要在此处截断）

    arXiv:2609.01544v1 Announce Type: new  Abstract: Feature-based newsvendor models use observable covariates to tailor inventory decisions, aiming to balance holding and shortage costs under demand uncertainty. However, high-dimensional feature sets often hinder interpretability and inflate data collection and implementation costs. This paper studies variable selection for the feature-based newsvendor problem under a hard cardinality constraint on the number of selected features. We formulate the resulting $\ell_0$-constrained empirical newsvendor problem with $\ell_2$-regularization, establish its computational hardness, and develop a mixed-integer second-order cone programming reformulation that strengthens the standard Big-$M$ formulation. To enable scalability beyond exact optimization, we develop a randomized-rounding algorithm with a bi-criteria guarantee and a greedy heuristic. Statistically, we provide theoretical analysis of the resulting sparse policy estimator, including finit
    
[^3]: 论生成式数据增强的可靠性：基于Wasserstein距离的理论与实证研究

    On the Reliability of Generative Augmentation: A Wasserstein-Based Theoretical and Empirical Study

    [https://arxiv.org/abs/2609.01410](https://arxiv.org/abs/2609.01410)

    该论文建立了条件生成式数据增强的统计理论框架，证明分类风险的失真由增强强度和真实与生成分布之间的类条件Wasserstein距离共同控制，并推导出基于Rademacher复杂度的泛化界，揭示了假设复杂度、增强强度与生成保真度之间的权衡，实证表明CWGAN-GP在不平衡分类任务上表现更优。

    

    生成式数据增强被广泛用于缓解类别不平衡问题，但其对下游泛化性能的理论影响仍知之甚少。在本工作中，我们为条件生成式数据增强建立了一个统计框架，并分析其对分类风险的影响。我们将数据增强形式化为一个分布混合过程，并证明由此产生的风险失真同时受增强强度以及真实分布与生成分布之间类条件Wasserstein差异的控制。我们进一步基于Rademacher复杂度推导出一个依赖于模型容量的泛化界，揭示了假设复杂度、增强强度与生成保真度之间的明确权衡。在实证方面，我们在二分类和多分类不平衡分类任务上，采用条件GAN和条件WGAN-GP增强对该框架进行评估。在所有数据集上，CWGAN-GP始终取得较低的Wa……（原文摘要此处被截断）

    arXiv:2609.01410v1 Announce Type: new  Abstract: Generative data augmentation is widely used to mitigate class imbalance, yet its theoretical effect on downstream generalization remains poorly understood. In this work, we develop a statistical framework for conditional generative augmentation and analyze its impact on classification risk. We formalize augmentation as a distribution-mixing process and show that the resulting risk distortion is controlled by both the augmentation strength and the class-conditional Wasserstein discrepancy between real and generated distributions. We further derive a capacity-dependent generalization bound based on Rademacher complexity, revealing an explicit trade-off between hypothesis complexity, augmentation intensity, and generative fidelity. Empirically, we evaluate the framework on binary and multiclass imbalanced classification tasks using Conditional GAN and Conditional WGAN-GP augmentation. Across datasets, CWGAN-GP consistently achieves lower Wa
    
[^4]: 通过集成边界与局部预测变异性衡量一致性：在预测多样性存在下审计决策系统

    Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity

    [https://arxiv.org/abs/2609.01397](https://arxiv.org/abs/2609.01397)

    该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。

    

    罗生门效应是机器学习中的一种现象，即准确度相同的模型会对相同的输入产生不同的预测（预测多样性）。现有工作主要关注单个模型内部的多样性，但在更复杂的决策系统中，罗生门效应的影响尚不十分清楚。在本研究中，我们从审计错误集成预测的角度研究多样性问题，其中将某个实例转移给人工审查的决策基于一个一致性准则，该准则将集成边界与每个组成模型的局部预测变异性度量相结合。在关于稳定性和平滑性的温和假设下，我们证明随着集成规模以及用于测量局部预测变异性的样本数量的增加，有限集成的一致性分数收敛于来自罗生门集合的期望模型的相应一致性分数。为了演示……

    arXiv:2609.01397v1 Announce Type: cross  Abstract: The Rashomon effect is a machine learning phenomenon where equally accurate models produce different predictions for the same inputs (predictive multiplicity). Existing work primarily focuses on multiplicity within individual models, but in more complex decision systems, the impact of the Rashomon effect is less well understood. In this work, we study multiplicity from the perspective of auditing incorrect ensemble predictions, where the decision to divert an instance for human review is based on a consistency criterion that combines the ensemble margin with a measure of local prediction variability for each constituent model. With mild assumptions about stability and smoothness, we show that the consistency scores of finite ensembles converge to the corresponding consistency score of the expected model from the Rashomon set as the ensemble size and the number of samples used to measure local prediction variability increase. To demonst
    
[^5]: 分支交汇点处用于曲率与密度恢复的匹配查询

    Matched Queries for Curvature and Density at Branching Junctions

    [https://arxiv.org/abs/2609.01319](https://arxiv.org/abs/2609.01319)

    本文提出在两个噪声尺度下进行匹配得分查询并相减的方法，消去切向贡献后线性地暴露出包含分支曲率与对数密度斜率信息的项 G，并证明在已知各射线的切向方向与权重时，交汇点处全部 sD 个分支参数可被唯一识别。

    

    在一个交汇点处，得分场可以揭示出带权重的切向射线，然而这些一阶量并不能确定各条分支如何弯曲，也不能确定其密度在远离中心处如何变化。恢复这些缺失信息对于描述超越单点的局部延续是必要的，但有限的观测必须在允许估计中心存在误差的情况下，分离出各分支的二阶效应。我们利用在噪声尺度 σ 和 λσ 下的匹配得分查询来解决这一逆问题。对于 ℝ^D 中由 C^{2,α} 半分支构成的有限并集，归一化得分具有展开式 F_σ = F_0 + σG + O(σ^{1+α})。通过匹配相减可消去切向贡献并暴露出 G，它线性地依赖于各分支的曲率和对数密度斜率。在给定不同射线上的切向方向与权重的前提下，G 能唯一确定全部 sD 个分支参数，并且 sD 个标量分量观测足以实现这一恢复（摘要在此处被截断）。

    arXiv:2609.01319v1 Announce Type: new  Abstract: At a junction, a score field can reveal weighted tangent rays, yet these first-order quantities do not determine how individual branches bend or how their densities change away from the center. Recovering this missing information is necessary for describing local continuation beyond a single point, but finite observations must separate branchwise second-order effects while allowing error in the estimated center. We address this inverse problem using matched score queries at noise scales $\sigma$ and $\lambda\sigma$. For a finite union of $C^{2,\alpha}$ half-branches in $\mathbb{R}^D$, the normalized score has the expansion $F_\sigma=F_0+\sigma G+O(\sigma^{1+\alpha})$. Matched subtraction cancels the tangent contribution and exposes $G$, which depends linearly on branchwise curvature and log-density slope. Given tangent directions and weights on distinct rays, $G$ uniquely identifies all $sD$ branch parameters, and $sD$ scalar component o
    
[^6]: 单层Transformer被证明能够以上下文方式学习多分类单最近邻

    One-Layer Transformer Provably Learns Multiclass One-Nearest Neighbor in Context

    [https://arxiv.org/abs/2609.01311](https://arxiv.org/abs/2609.01311)

    本文证明了带argmax分类头的单层Transformer在多分类的上下文学习中行为与单最近邻分类器完全一致，填补了此前工作依赖非标准舍入方法所留下的理论空白。

    

    我们将近期一项在二分类设定下建立了单层Transformer与最近邻分类器之间等价性的工作，扩展到多分类情形。通过利用单纯形编码，我们证明了带argmax分类头的单层Transformer在多分类设定下的行为与单最近邻分类器完全一致。这填补了先前工作留下的空白——先前工作的多分类结果依赖于基于舍入的非标准方法，而非实践中常用的argmax分类头。

    arXiv:2609.01311v1 Announce Type: new  Abstract: We extend recent work establishing an equivalence between one-layer transformers and nearest-neighbor classifiers in the binary setting to the multiclass case. By leveraging the simplex encoding, we show that one-layer transformers with an argmax classification head behave identically to a one-nearest-neighbor classifier in the multiclass setting. This closes a gap left by prior work, whose multiclass result relied on a non-standard rounding-based approach rather than the typical argmax head used in practice.
    
[^7]: 多头自注意力是一种参数辨识机制

    Multi-Head Self Attention is a Parameter Identification Mechanism

    [https://arxiv.org/abs/2609.01231](https://arxiv.org/abs/2609.01231)

    该论文证明多头缩放点积注意力本质上是一种参数辨识机制——头数越多，未辨识参数比例从 1/2 降至 1/(2H)，但注意力永远无法被完全辨识，并且这一视角还能解释 RoPE 和 GQA 等现代改进为何能提升“有意义”参数的占比。

    

    我们证明，多头缩放点积注意力可以被看作是一种参数辨识策略。未辨识参数与总参数数量之比随头数呈倒数关系缩放（从 1/2 降为 1/(2H)），这意味着头数更多的模型在结构上具有更强的参数可辨识性。这一数学观察还揭示了一个微妙的副作用：注意力永远无法被完全辨识。类似地，我们还证明了在单头和多头设置下，某些偏置项对基于 softmax 的注意力层没有任何影响，尽管这主要是一个奇特现象，其对模型规模以及模型训练/预测效率的影响应是边际性的。我们还从这一视角审视了 Transformer 的现代改进方法，包括 RoPE 和 GQA，说明这些改进同样能够提高“有意义”参数占全部参数的比例。简单的数值示例表明，训练确实能够……

    arXiv:2609.01231v1 Announce Type: cross  Abstract: We prove that a multi-head scaled dot product attention can be viewed as a parameter identification strategy. The ratio of unidentified parameters to the total number of parameters scales like the reciprocal of the number of heads ($1/2 \to 1/(2H)$), meaning models with more heads are structurally more identified. A subtle side effect of the mathematics observation that attention can never be fully identified. Similarly we also show that some bias terms can have no effect on softmax-based attention layers in both the single- and multiple-head settings, though this is mostly a curiosity that should have a marginal effect on model size and model training/prediction efficiency. We also touch on modern improvements to transformers including RoPE and GQA from this perspective, illustrating how those as well can improve the ratio of ``meaningful'' parameters to all parameters. Simple numerical examples demonstrate that training can indeed in
    
[^8]: 密度依赖McKean--Vlasov扩散过程的非参数推断

    Nonparametric inference for density-dependent McKean--Vlasov diffusions

    [https://arxiv.org/abs/2609.01166](https://arxiv.org/abs/2609.01166)

    该论文基于稀疏ReQU神经网络构造筛最大似然估计量，实现了多元McKean--Vlasov扩散中密度依赖漂移系数与平稳密度的非参数估计，并给出了匹配的Assouad下界证明其收敛速度的理论最优性。

    

    本研究致力于从同一时刻的独立观测出发，对多元McKean--Vlasov扩散过程中密度依赖的漂移系数以及平稳密度进行非参数估计。在（已知的）势函数满足一定假设的条件下，我们将该问题化简为一维情形，并基于满足结构约束与Hölder约束的稀疏ReQU神经网络构造了筛最大似然估计量。借助端点自适应的分级逼近方法，我们在真实平稳密度与估计平稳密度之间的Kullback-Leibler散度上达到了 $\left(b_n\log n/n\right)^{2(\beta+1)/(2\beta+3)}$ 的收敛速度，其中 $b_n$ 至多是一个对数因子。类似地，文中证明了所构造的漂移系数估计量在 $L^2$ 度量下以 $\left(b_n\log n/n\right)^{\beta/(2\beta+3)}$ 的速度收敛于真实漂移系数。此外还给出了与之匹配的Assouad下界。

    arXiv:2609.01166v1 Announce Type: cross  Abstract: The present research is devoted to the nonparametric estimation of a density-dependent drift coefficient in a multivariate McKean--Vlasov diffusion from independent observations at a common time, as well as the stationary density. Under certain assumptions on the (known) potential, we reduce the problem to the one-dimensional one and construct a sieve maximum-likelihood estimator based on sparse ReQU neural networks subject to structural and H\"{o}lder constraints. Using the endpoint-adapted graded approximation, we achieve the rate of $\left(b_n\log n/n\right)^{2(\beta+1)/(2\beta+3)}$ for the Kullback-Leibler divergence between the true and estimated stationary densities, with $b_n$ being at most a logarithmic factor. Similarly, it is shown that the constructed estimator for the drift coefficient converges to the true one at the rate of $\left(b_n\log n/n\right)^{\beta/(2\beta+3)}$ in the $L^2$-metric. A matching Assouad lower bound p
    
[^9]: 人工罗塞塔石碑：基于k阶马尔可夫模型的符号化拉格序列约束最大后验（MAP）重建

    Artificial Rosetta Stone: Constrained Maximum A Posteriori (MAP) Reconstruction of Symbolic Raga Sequences via Order-k Markov Models

    [https://arxiv.org/abs/2609.01064](https://arxiv.org/abs/2609.01064)

    本文提出“人工罗塞塔石碑”框架，将印度拉格音乐中缺失音符的重建形式化为带约束的最大后验估计问题，并利用k阶马尔可夫模型和动态规划给出精确可解的方案。

    

    重建受损的音乐片段是一个逆问题：观测到的序列仅包含部分信息，而拉格编码了限制允许补全方式的约束。本文为此形式化了一个数学框架，提出了“人工罗塞塔石碑”（ARS）。我们区分了三个经常被混淆的命题：符号序列可以被概率性地重建；序列可以与显式语法保持一致；以及历史演奏可以被认证。我们仅支持前两个命题。我们通过有限字母表和约束系统对拉格进行建模，并使用k阶马尔可夫模型来描述旋律概率。对称狄利克雷先验使得后验分布易于处理。我们将缺失音符的重建表述为一个约束MAP问题。对于定长序列和有限阶约束，该优化问题存在精确的动态规划解，其最坏情况时间复杂度为 $O(TN^{k+1})$。我们推导出参数……

    arXiv:2609.01064v1 Announce Type: cross  Abstract: Reconstructing a damaged musical fragment is an inverse problem: the observed sequence contains partial information, while a raga encodes constraints limiting allowable completions. This paper formalizes a mathematical framework for this, proposing the Artificial Rosetta Stone (ARS). We separate three claims often conflated: a symbolic sequence can be reconstructed probabilistically; a sequence can be consistent with an explicit grammar; and a historical performance can be authenticated. We only support the first two. We model a raga via a finite alphabet and constraint system, using an order-k Markov model for melodic probabilities. A symmetric Dirichlet prior yields a tractable posterior. We pose missing-note reconstruction as a constrained MAP problem. For fixed-length sequences and finite-order constraints, optimization admits an exact dynamic-programming solution with worst-case time complexity $O(TN^{k+1})$. We derive the paramet
    
[^10]: 从截断到承诺：均匀离散扩散中的持久上下文

    From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion

    [https://arxiv.org/abs/2609.01043](https://arxiv.org/abs/2609.01043)

    提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。

    

    均匀状态离散扩散模型并行更新所有词元，同时保持每个位置都可被修改。即使常用的 top-p 规则在一个位置只留下一个候选，该选择也仅影响当前的反向步骤，并可在下一个采样步骤中被修改。我们探讨当被选中的假设转而成为后续预测的持久上下文时会发生什么变化。为此，我们提出了承诺式揭示采样，这是一种无需训练的采样器，它存储被选中的 argmax 词元，并将其插入后续的模型输入中。我们的分析为“更晚做出选择”和“保持被选词元可见”提供了理论依据：在精确的前向过程下，随着噪声降低，选择干净词元的贝叶斯误差不会增加；而在一个简单的潜变量模式模型中，保持被选词元可见有助于后续的并行预测在相同的序列级选择上达成一致。实证上，在 Duo-distilled 模型上的成对实验（摘要在此处截断）……

    arXiv:2609.01043v1 Announce Type: cross  Abstract: Uniform-state discrete diffusion models update all tokens in parallel while keeping every position revisable. Even when the commonly used top-$p$ rule leaves only one candidate at a position, that choice affects only the current reverse step and can be revised at the next sampling step. We ask what changes when selected hypotheses instead become persistent context for later predictions. We therefore propose committed reveal sampling (CRS), a training-free sampler that stores selected argmax tokens and inserts them into subsequent model inputs. Our analysis gives a rationale for selecting later and for keeping selected tokens visible. Under the exact forward process, the Bayes error of selecting a clean token cannot increase as noise decreases, while in a simple latent-mode model, keeping the selected token visible helps later parallel predictions agree on the same sequence-level choice. Empirically, paired experiments on Duo-distilled 
    
[^11]: 稳定边缘上梯度下降的多重时间尺度：中心流的微扰推导

    The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow

    [https://arxiv.org/abs/2609.01034](https://arxiv.org/abs/2609.01034)

    本文通过将损失函数分解为 $f = g + \varepsilon h$ 的微扰分析，首次为深度学习稳定边缘处梯度下降的中心流提供了系统性推导，并揭示出其中存在快速振荡、中间自稳定与缓慢中心流演化三个时间尺度。

    

    Cohen等人（2025）提出的中心流是深度学习中稳定边缘处梯度下降的一个在经验上准确的连续时间模型，然而其推导是启发式的。我们提出了一种微扰机制，在该机制下中心流是梯度下降的极限：我们假设损失函数分解为 $f = g + \varepsilon h$；在 $\varepsilon \to 0$ 的极限下，学习率为 $\eta$ 的梯度下降动力学收敛到 $h$ 的梯度流，且该梯度流被约束在锐度至多为 $2/\eta$ 的 $g$ 的极小值点上。我们的方法是形式化的而非严格证明的；它将梯度下降视为关于 $\varepsilon$ 的奇异摄动动力系统。由此涌现出三个时间尺度：沿最锐利方向的快速振荡时间尺度、自稳定机制的中间时间尺度，以及沿 $g$ 极小值点动力学的慢时间尺度——即中心流。利用多尺度方法，一（原文摘要在此处被截断）

    arXiv:2609.01034v1 Announce Type: new  Abstract: The central flow of Cohen et al. (2025) is an empirically accurate continuous-time model of gradient descent at the edge of stability in deep learning, However, its derivation is heuristic. We propose a perturbative regime in which the central flow is the limit of gradient descent: we assume that the loss decomposes as $f = g + \varepsilon h$; in the limit $\varepsilon \to 0$, the dynamics of gradient descent with learning rate $\eta$ converge to the gradient flow of $h$ constrained to the minimizers of $g$ of sharpness at most $2/\eta$. Our approach is formal rather than rigorous; it treats gradient descent as a singularly perturbed dynamical system in $\varepsilon$. Three timescales emerge: a fast timescale of oscillations along the sharpest direction, an intermediate timescale of the self-stabilization mechanism, and a slow timescale of the dynamics along the minimizers of $g$-the central flow. Using the method of multiple scales, a c
    
[^12]: 面向大语言模型生成文本的嵌入式条件独立性检验及其在德国联邦议院演讲中的应用

    Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches

    [https://arxiv.org/abs/2609.00946](https://arxiv.org/abs/2609.00946)

    本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。

    

    条件独立性检验（CITs）用于检验在给定第三个随机对象 Z 的条件下，两个随机对象 X 和 Y 之间是否存在条件依赖关系。现有的 CITs 对高维数据的适用性有限，尤其是像文本这样的多模态数据。然而，我们表明此类检验对大语言模型（LLM）的输出具有重要意义：即检验从源文本 Z 生成的输出 X 是否携带超出 Z 本身所含信息之外的属性 Y 的信息。为此，我们提出了嵌入式条件独立性检验（eCITs），该方法对 X 和 Z 进行嵌入，并将现有的 CIT 应用于所得的表示以及 Y。我们证明，只要 Z 的嵌入是充分的，即保留了 Z 所携带的关于 Y 或 X 的表示的信息，原假设就会从 X 和 Z 转移到它们的表示上，因此对嵌入后假设有效的 CIT 对原始假设同样有效。我们进一步给出了等变性的相关条件……

    arXiv:2609.00946v1 Announce Type: cross  Abstract: Conditional independence tests (CITs) test for conditional dependence between two random objects $X$ and $Y$ given a third random object $Z$. Existing CITs have limited applicability to high-dimensional data, especially multimodal data like text. However, we show that such tests are of interest for large language model (LLM) outputs, where we test whether an output $X$ generated from a source text $Z$ carries information about an attribute $Y$ beyond $Z$ itself. For this purpose, we propose embedded CITs (eCITs), which embed $X$ and $Z$ and apply an existing CIT to the resulting representations and to $Y$. We show that, provided the embedding of $Z$ is sufficient, i.e. retains the information $Z$ carries about either $Y$ or the representation of $X$, the null hypothesis transfers from $X$ and $Z$ to their representations, so that a CIT valid for the embedded hypothesis is valid for the original one. We further give conditions for equiv
    
[^13]: 当Metropolis与Hastings遇见Bradley与Terry：从偏好投票实现精确MCMC

    When Metropolis and Hastings Meet Bradley and Terry: Exact MCMC From Preference Voting

    [https://arxiv.org/abs/2609.00905](https://arxiv.org/abs/2609.00905)

    提出Pref-MH，一种仅依靠Bradley-Terry裁判的随机二元成对偏好比较即可实现精确Metropolis-Hastings条件采样的通用算法，并设计了可证明收敛的接受/拒绝规则。

    

    从以期望语义属性为条件的分布中进行采样，是现代生成式建模中一个新兴的挑战。Metropolis-Hastings（MH）为条件采样提供了一种有原则的途径，但它需要对目标密度进行精确的逐点评估，而这在生成式场景中是无法获得的。与此同时，由人类或模型“裁判”给出的成对比较非常容易获得，并已在多种应用中被证明具有价值。我们提出了Pref-MH，一种仅使用随机二元成对比较、即可从裁判诱导的条件分布中进行采样的通用精确MH采样器。我们的关键观察是：MH的非归一化密度比恰好与Bradley-Terry（BT）选择模型的偏好几率相匹配。核心挑战在于，MH要求精确的比率计算，而BT裁判只能提供采样的二元反馈。为此，我们开发了一种有效的接受/拒绝规则，其产生的马尔可夫链可被证明收敛（摘要在此处截断）。

    arXiv:2609.00905v1 Announce Type: cross  Abstract: Sampling from distributions conditioned on desired semantic properties is an emerging challenge in modern generative modeling. Metropolis-Hastings (MH) provides a principled route to conditional sampling, but requires access to exact pointwise target-density evaluations, which are not available in generative settings. Meanwhile, pairwise comparisons by humans or model "judge" are highly accessible and have proved valuable across diverse applications. We introduce Pref-MH, a general exact MH sampler for judge-induced conditional distributions using only stochastic binary pairwise comparisons. Our key observation is that the MH unnormalized density ratio matches the preference odds of the Bradley-Terry (BT) choice model. The central challenge is that while MH requires precise ratio computation, BT judges provide only sampled binary feedback. To this end, we develop a valid accept/reject rule whose resulting Markov chain provably converge
    
[^14]: 威布尔混合模型中含信息性缺失标签的半监督分类

    Semi-Supervised Classification with Informative Missing Labels in Weibull Mixture Models

    [https://arxiv.org/abs/2609.00774](https://arxiv.org/abs/2609.00774)

    该论文提出在两分量威布尔混合模型的半监督分类中，将标签缺失概率建模为分类不确定性的函数，从而证明缺失标签指示变量本身携带分类器信息，并据此刻画了贝叶斯决策边界的结构并推导了相应的Fisher信息量。

    

    我们考虑从来自两分量威布尔混合模型的部分已分类样本中进行半监督分类。所有数据的特征均可观测，而部分类别标签存在缺失。我们将标签缺失的概率建模为分类不确定性的函数，从而得到一种依赖于特征的随机缺失（MAR）机制，该机制与威布尔混合分类器共享参数。因此，除观测到的特征和已有的类别标签之外，缺失标签的指示变量本身也能提供关于分类器的信息。在威布尔形状参数相同的情形下，贝叶斯决策规则至多有一个正的决策边界，且当规则非常数时该边界唯一；在形状参数不同的情形下，则可能出现两个决策边界。我们刻画了这些决策区域，推导出在对缺失机制中的冗余参数进行调整之后的分类器Fisher信息量，并得到了期望……的决策边界展开式（原文摘要在此处截断）。

    arXiv:2609.00774v1 Announce Type: cross  Abstract: We consider semi-supervised classification from a partially classified sample arising from a two-component Weibull mixture. The feature is observed for all data, whereas some class labels are missing. The probability of a missing label is modelled as a function of classification uncertainty, giving a feature-dependent missing-at-random (MAR) mechanism that shares parameters with the Weibull-mixture classifier. The missing-label indicators can therefore provide information about the classifier in addition to the observed features and available class labels. Under a common Weibull shape, a Bayes' rule has at most one positive decision boundary, which is unique when the rule is nonconstant; under unequal shapes, it can have two. We characterise these decision regions, derive the Fisher information for the classifier after adjustment for nuisance parameters in the missingness model, and obtain a decision-boundary expansion of the expected 
    
[^15]: 深度偏斜t混合模型

    Deep Skew-t Mixture Models

    [https://arxiv.org/abs/2609.00773](https://arxiv.org/abs/2609.00773)

    提出深度偏斜t混合模型（DStMM），通过沿潜变量路径传播共享的逆伽马混合变量，联合建模高维聚类中的厚尾性与方向性偏斜，并采用随机/蒙特卡洛EM算法进行估计。

    

    当分量分布同时呈现厚尾性与方向性偏斜时，高维聚类是一项极具挑战性的任务。我们提出了一种深度偏斜t混合模型（DStMM），这是一种基于广义双曲偏斜t正态均值-方差表示的层次化因子分析混合模型。一个共享的逆伽马混合变量沿每条完整的潜变量路径传播，使得厚尾性与方向性偏斜能够被联合建模，同时保持条件高斯性。因此，每条完整路径都具备精确的GHST（广义双曲偏斜t）边际表示。我们形式化了向对称深度t模型、高斯深度混合模型以及单层GHST因子分析模型的简化过程，讨论了局部不可识别性及实现层面的参数计数约定，并推导了用于估计的条件广义逆高斯分布。估计通过随机/蒙特卡洛EM算法进行，并包含一个……

    arXiv:2609.00773v1 Announce Type: cross  Abstract: High-dimensional clustering is challenging when component distributions are both heavy-tailed and directionally asymmetric. We propose a deep skew-$t$ mixture model (DStMM), a hierarchical factor-analytic mixture based on the generalised-hyperbolic skew-$t$ normal mean--variance representation. A shared inverse-gamma mixing variable is propagated along each complete latent pathway, allowing heavy tails and directional asymmetry to be modelled jointly while preserving conditional Gaussianity. Each complete pathway therefore admits an exact GHST marginal representation. We formalise the reductions to symmetric deep $t$, Gaussian deep-mixture, and single-layer GHST factor-analytic models, discuss local non-identifiability and the implementation-level parameter-counting convention, and derive the conditional generalised inverse Gaussian law used for estimation. Estimation is carried out by a stochastic/Monte Carlo EM algorithm, with an exp
    
[^16]: 参考集重采样下OOD分数的判定不稳定性

    Verdict Instability of OOD Scores under Reference Resampling

    [https://arxiv.org/abs/2609.00691](https://arxiv.org/abs/2609.00691)

    本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。

    

    事后（post-hoc）分布外（OOD）检测器是在有限的参考集上拟合的，因此它们产生的每一个分数都只是一个估计值。如果我们选择了另一个参考集，某些判定结果就会发生改变。我们通过重采样参考集并记录分数的自助法（bootstrap）标准差来度量这种变动，并将其称为“判定不稳定性”。该量具有无需拟合任何参数的闭式解。一个判定的不稳定性等于所分配类别沿查询方向的类内离散度除以该类别参考样本数的平方根。正是这一样本数将判定不稳定性与分数分布的几何结构区分开来，且只有在类别不平衡的情况下才能将其识别出来。不稳定性随局部离散度的增大而增长。远分布外（Far-OOD）查询位于各向异性嵌入的低方差方向上，因此在我们测试的所有基于距离的分数中，最高分数值都被分配给了那些最可复现的判定。只有es…（摘要不完整）

    arXiv:2609.00691v1 Announce Type: new  Abstract: Post-hoc out-of-distribution detectors are fitted on a finite reference set, so every score they produce is an estimate. If we had chosen a different set, some verdicts would have moved. We measure that movement by resampling the reference set and recording the bootstrap standard deviation of the score, which we call verdict instability. It admits a closed form with no fitted parameters. The instability of a verdict is the within-class dispersion of the assigned class along the query's direction, divided by the square root of that class's reference count. That count is what separates verdict instability from the geometry of the score distribution, and it is identifiable only under class imbalance. Instability grows with the local dispersion. Far-OOD queries lie along the low-variance directions of an anisotropic embedding, so every distance-based score we test assigns its highest values to the verdicts that are most reproducible. Only es
    
[^17]: 面向矩阵正态混合模型中元素级与结构性缺失的高效EM算法

    An efficient EM algorithm for both element-wise and structural missingness in matrix-variate normal mixture models

    [https://arxiv.org/abs/2609.00616](https://arxiv.org/abs/2609.00616)

    本文提出一种高效的部分EM算法，通过坐标级逼近处理矩阵正态混合模型中元素级与子矩阵级缺失数据，保持Kronecker结构并避免昂贵的协方差矩阵求逆，从而显著降低计算成本。

    

    在观测数据天然组织为二维数组的应用中，含缺失元素的矩阵型数据经常出现。尽管矩阵正态分布通过其Kronecker协方差结构提供了简约的模型，但由于任意缺失模式通常会在E步中破坏这种可分性，标准EM估计的计算代价可能非常高昂。本文提出了一种针对含缺失元素的矩阵正态数据的高效部分EM算法。该方法通过坐标级逼近来更新缺失分量的条件均值与协方差，避免了对特定缺失模式协方差矩阵的重复求逆，也避免了构造完整的向量化协方差矩阵。此外，我们针对子矩阵缺失情形开发了一种专门的更新方法，其中缺失块的精度矩阵保持Kronecker乘积结构，协方差的更新也随之得到简化。

    arXiv:2609.00616v1 Announce Type: cross  Abstract: Matrix-variate data with missing entries arise frequently in applications where observations are naturally organized as two-dimensional arrays. Although the matrix normal distribution provides a parsimonious model through its Kronecker covariance structure, standard EM estimation can be computationally expensive because arbitrary missingness patterns typically destroy this separability in the E-step. In this paper, we propose an efficient partial EM algorithm for matrix-variate normal data with missing entries. The proposed method updates the conditional mean and covariance of the missing component through coordinate-wise approximations, avoiding repeated inversion of pattern-specific covariance matrices and avoiding construction of the full vectorized covariance matrix. We further develop a specialized update for submatrix missingness, where the missing-block precision retains a Kronecker product structure, and the covariance update c
    
[^18]: 一种用于检测能源价格序列中事件驱动动态的卷积框架

    A convolutional framework for detecting event-driven dynamics in energy price series

    [https://arxiv.org/abs/2609.00402](https://arxiv.org/abs/2609.00402)

    本文提出了一个通用卷积神经网络框架，可精确表示并逼近多种事件检测统计量（如极差、最大回撤、波动率等），并在能源价格序列中有效识别出以地缘政治为主的事件驱动动态。

    

    本文开发了一个通用的卷积神经网络（CNN）框架，用于检测单变量时间序列窗口中异质的事件驱动动态。我们证明该CNN类能够精确表示基于极差、最大上涨、最大回撤和斜率变化的分类器，并在紧集上一致逼近已实现波动率和自回归爆炸性。我们进一步为有限样本中的代表性规则建立了误差界，并为跨规则学习建立了预言机不等式。模拟结果表明，随着训练样本的增长，所提出的模型能够匹配或超越基于单一统计量的分类器。在应用于六个日度能源价格序列时，层次化CNN能够区分事件窗口和事件族。在无需重新训练的情况下，将该模型应用于2026年2月20日之后保留的观测数据，其成功在多种石油和成品油价格中识别出以地缘政治为主的事件驱动动态。

    arXiv:2609.00402v1 Announce Type: new  Abstract: This paper develops a general convolutional neural network (CNN) framework for detecting heterogeneous event-driven dynamics in univariate time series windows. We show that the induced CNN class exactly represents classifiers based on range, maximum drawup, maximum drawdown and slope change, and uniformly approximates realised volatility and autoregressive explosiveness on compact domains. We further establish error bounds for representative rules in finite samples and an oracle inequality for learning across them. Simulations show that the proposed model can match or outperform classifiers based on individual statistics as the training sample grows. In an application to six daily energy price series, a hierarchical CNN distinguishes event windows and event families. Applied without retraining to observations withheld after 20 February 2026, the fitted model identifies predominantly geopolitical dynamics in several oil and refined produc
    
[^19]: 用于算子学习的神经均值与核修正

    Neural means and kernel corrections for operator learning

    [https://arxiv.org/abs/2609.00389](https://arxiv.org/abs/2609.00389)

    该论文提出将神经网络均值与Matérn核回归修正相结合的方法，在结构力学和OCO-2辐射传输两个基准问题上达到了或超越了已发表最佳结果，并从理论上证明和量化了核修正之所以有效的机制。

    

    我们将神经网络均值与其残差以及学习到的特征的精确Matérn核回归相结合，并在两个有公开基线的公共仿真问题上评估了这种组合：de Hoop等人的结构力学基准和Lamminpää等人的OCO-2辐射传输仿真器。在结构力学问题上，该组合达到了4.55%的测试误差，与已发表的最佳架构相当；在低数据量情形下达到5.38%，优于已发表的6.49%。在OCO-2问题上，该组合在该问题自身的测试点上改进了已发表的高斯过程仿真器，在三个光谱波段中的两个上完全超越；同一个核在原始状态上落后于网络十倍，却在网络的特征上反超网络，并且我们测量了原因（在固定有效维度的前提下，目标在原生空间中的平方范数下降了约四十倍）并证明了这一机制。在两个方法家族打平的地方，我们所评估的每种架构的残差……

    arXiv:2609.00389v1 Announce Type: new  Abstract: We combine neural network means with exact Mat\'ern kernel regressions of their residuals and of their learned features, and evaluate the pairing on two public emulation problems with published baselines: the structural-mechanics benchmark of de Hoop et al. and the OCO-2 radiative-transfer emulator of Lamminp\"a\"a et al. On structural mechanics the combination reaches 4.55% test error, matching the best published architecture, and 5.38% against a published 6.49% in the low-data regime. On OCO-2 it improves on the published Gaussian-process emulator on that problem's own test points, outright on two of the three spectral bands; the same kernel that trails the network tenfold on the raw state overtakes it on the network's features, and we measure why (the target's squared native-space norm drops about fortyfold at fixed effective dimension) and prove the mechanism. Where the two families tie instead, the residuals of every architecture we
    
[^20]: 基于去噪扩散的精确全局MCMC方法

    Exact Global MCMC with Denoising Diffusion

    [https://arxiv.org/abs/2609.00279](https://arxiv.org/abs/2609.00279)

    该论文提出去噪扩散蒙特卡洛方法，通过在MALA局部采样样本上训练去噪扩散模型并施加Metropolis-Hastings精确校正步骤，为复杂高维目标分布提供高接受率的全局MCMC提议。

    

    这项工作表明，通过标准去噪损失训练的扩散模型能够为复杂的高维目标分布提供有效的全局MCMC提议。该方法的动机源于一个观察：依次应用前向和反向扩散过程可以定义一个马尔可夫链，当去噪器是在目标分布样本上训练的理想去噪器时，该链具有目标平稳分布。通过应用一个Metropolis-Hastings步骤——其接受率包含离散时间SDE近似的前向和反向路径密度——这一观察结果对任意去噪器都可以被精确化。因此，我们提出在局部收敛的MALA样本上训练去噪扩散模型，以学习全局MCMC提议。我们将基于全局去噪器的路径采样器与局部MALA采样器的组合称为去噪扩散蒙特卡洛。实验表明，DDMC能够在各种……（原文在此处截断）

    arXiv:2609.00279v1 Announce Type: new  Abstract: This work shows that diffusion models learned with standard denoising loss can provide effective global MCMC proposals for complex high-dimensional target densities. The method is motivated by the observation that sequentially applying a forward and reverse diffusion process defines a Markov chain with a target stationary distribution for an ideal denoiser trained on samples of the target distribution. This observation can be made exact for any denoiser by applying a Metropolis-Hastings step whose acceptance ratio includes the density of the forward and reverse paths of a discrete time SDE approximation. We therefore propose to train denoising diffusion models on locally convergent MALA samples to learn global MCMC proposals. We call the composition of the global denoiser-based path sampler and a local MALA sampler Denoising Diffusion Monte Carlo (DDMC). Experiments show that DDMC can provide global proposals with high acceptance across 
    
[^21]: 具有线性函数逼近和对数级通信成本的可证明高效的联邦强化学习

    Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost

    [https://arxiv.org/abs/2609.00193](https://arxiv.org/abs/2609.00193)

    提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。

    

    我们研究了具有线性函数逼近的联邦在线强化学习。尽管近期的多智能体强化学习算法实现了很强的遗憾保证，但它们通常需要共享原始轨迹。这种依赖性导致通信成本随回合数线性增长，并违反了联邦设置中的隐私约束。为了解决这些局限性，我们提出了Fed-LSVI，这是首个针对分段马尔可夫决策过程中具有线性函数逼近的在线强化学习的可证明高效的联邦算法。通过将基于行列式的事件触发同步机制与逐步反向更新机制相结合，Fed-LSVI使智能体能够通过仅交换压缩的充分统计量来协作学习最优策略。我们证明Fed-LSVI实现了$\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$的遗憾界，其中$d$是特征维度，$H$是……

    arXiv:2609.00193v1 Announce Type: cross  Abstract: We study federated online reinforcement learning with linear function approximation. While recent multi-agent reinforcement learning algorithms achieve strong regret guarantees, they typically require sharing raw trajectories. This reliance incurs a communication cost that scales linearly with the number of episodes and violates the privacy constraints of federated settings. To address these limitations, we propose Fed-LSVI, the first provably efficient federated algorithm for online reinforcement learning with linear function approximation in episodic Markov decision processes. By integrating a determinant-based event-triggered synchronization with a stepwise backward update mechanism, Fed-LSVI enables agents to collaboratively learn an optimal policy by exchanging only compressed sufficient statistics. We prove that Fed-LSVI achieves a regret bound of $\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$, where $d$ is the feature dimension, $H$ 
    
[^22]: 包含聚类结构的向量的随机复杂度

    Stochastic complexity of vectors containing cluster structure

    [https://arxiv.org/abs/2609.00084](https://arxiv.org/abs/2609.00084)

    本文提出一种递归公式来高效计算NML模型的归一化常数，将计算包含聚类结构向量最短编码长度的时间复杂度从多项式时间降低到线性时间。

    

    本文研究了使用归一化最大似然（NML）模型计算包含聚类结构的编码向量的随机概率（最短编码长度）的问题。这对于基于最小描述长度（MDL）原理的数据聚类具有重要的理论和实践意义，例如用于估计数据的最佳聚类数目和最佳聚类结构。基于NML模型直接计算包含聚类结构的向量的最短编码长度，需要相对于向量大小和聚类数目的多项式时间。我们通过引入一个递归公式来高效计算NML模型的归一化常数，证明了这是一个可解的问题。新公式的时间复杂度是线性的，相比于之前关于向量大小和聚类数目的多项式时间有了显著改进。

    arXiv:2609.00084v1 Announce Type: new  Abstract: This paper studies the problem of computing the stochastic probability (shortest code length) of the encoded vectors containing cluster structure using Normalized Maximum Likelihood (NML) model. This is of great theoretical and practical importance in data clustering based on Minimum Description Length (MDL) principle, such as for estimating the best number of clusters and best cluster structure for the data. Straightforward computation of the shortest code length of the vector containing cluster structure based on the NML model requires polynomial time with respect to the size of the vector and number of clusters. We show that this is a tractable problem by introducing a recursion formula for the efficient computation of normalizing constant from the NML model. The time complexity of the new formula is linear opposed to previous polynomial time with respect to the size of the vector and number of clusters.
    
[^23]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^24]: 超越搜索模仿：面向无搜索国际象棋的先验引导探索

    Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess

    [https://arxiv.org/abs/2608.27757](https://arxiv.org/abs/2608.27757)

    该论文提出用朝向网络自身MCTS先验的前向质量覆盖KL散度（先验引导探索）替代传统熵奖励，并结合由价值头不确定性驱动的熵自适应采样温度，通过自我对弈强化学习将无搜索国际象棋网络的谜题准确率从93.9%提升至94.9%。

    

    无搜索国际象棋网络通过单次前向传播即可达到人类大师水平，其方法是模仿一个更强的教师——即Leela Chess Zero（Lc0）发布的最强网络Chessformer，它蒸馏了AlphaZero风格蒙特卡洛树搜索（MCTS）的访问计数。然而，模仿搜索对于无搜索下棋来说是一个糟糕的替代目标，因此我们采用自我对弈强化学习（RL）进行微调以提升单次前向传播的棋力。这类方法的探索机制通常由熵奖励（即到均匀分布的反向KL散度）提供。我们将其替换为朝向网络自身MCTS先验的前向、质量覆盖KL散度（先验引导探索），使探索能够覆盖先验判断为有希望的着法，并将其与熵自适应采样温度相结合——该温度由价值头的结果不确定性设定，在局面胜负已定后会收紧分布。在大约两千步训练内，该方法将10万个谜题测试集上的谜题准确率从93.9%提升至94.9%，并提升了四步杀（摘要在此处被截断）

    arXiv:2608.27757v1 Announce Type: new  Abstract: Searchless chess networks reach human master strength from a single forward pass by imitating a stronger teacher: the strongest, Leela Chess Zero's (Lc0) released Chessformer, distills the visit counts of an AlphaZero-style Monte Carlo Tree Search (MCTS). Imitating a search is a poor proxy for playing without one, so we fine-tune for single-pass strength with self-play reinforcement learning (RL). Its exploration is usually supplied by an entropy bonus, the reverse Kullback-Leibler (KL) divergence to uniform. We replace it with a forward, mass-covering KL toward the network's own MCTS prior (prior-directed exploration), so exploration covers the moves the prior judges promising, and pair it with an entropy-adaptive sampling temperature, set by the value head's outcome uncertainty, that sharpens once a position is decided. In about two thousand steps it raises puzzle accuracy from 93.9% to 94.9% on a 100,000-puzzle suite and mate-in-four 
    
[^25]: 公共中心几何与能量形式全共形区域的认证径向重建

    Common-Center Geometry and Certified Radial Reconstruction for Energy-Form Full Conformal Regions

    [https://arxiv.org/abs/2608.24964](https://arxiv.org/abs/2608.24964)

    本文证明了在对称性和凸性条件下，能量形式全共形预测区域呈星形，且对于幂距离在β≥1时具有确定性几何性质，同时指出候选评分凸性不足以保证连通性。

    

    本文研究了由经验能量形式成对评分生成的全共形预测（FullCP）区域的几何性质。仅凭候选评分的凸性并不能保证FullCP区域的连通性，即使候选评分是损失函数在其第一个参数上凸的经验平均值。通过直接展开留一评分，发现能量形式评分的每个训练点比较恰好是一个成对不相似性子水平条件。在对称性、常数对角线、对角线下界以及相关Fr\'echet型目标达到的条件下，每个比较区域都包含一个公共最小化器；当比较区域为凸时，非平凡精确共形区域因此关于该点呈星形。对于幂距离$\rho_\beta(x,y)=\|x-y\|^\beta$，当$\beta\ge1$时，这种确定性几何成立，而传统能量评分在$0<\beta<2$时是严格适当的。

    arXiv:2608.24964v1 Announce Type: cross  Abstract: This note studies the geometry of full conformal prediction (FullCP) regions generated by an empirical energy-form pairwise score. Candidate-score convexity alone does not guarantee connected FullCP regions, even when the candidate score is an empirical average of a loss convex in its first argument. Direct expansion of the leave-one-out scores shows that each training-point comparison for the energy-form score is exactly a pairwise-dissimilarity sublevel condition. Under symmetry, a constant diagonal, a diagonal lower bound, and attainment of the associated Fr\'echet-type objective, every comparison region contains a common minimizer; when the comparison regions are convex, the nontrivial exact conformal region is therefore star-shaped about that same point. For power distances $\rho_\beta(x,y)=\|x-y\|^\beta$, this deterministic geometry holds for $\beta\ge1$, while the conventional energy score is strictly proper for $0<\beta<2$. In 
    
[^26]: 有效学习率主导语言模型预训练中的损失动态

    Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining

    [https://arxiv.org/abs/2608.24814](https://arxiv.org/abs/2608.24814)

    本文发现语言模型预训练中，有效学习率（LR与参数范数的比值）是损失动态的核心控制变量，匹配ELR可使不同配置的损失轨迹坍缩一致，并据此提出了可跨方法迁移的缩放定律。

    

    arXiv:2608.24814v1 公告类型：新 摘要：我们在语言模型预训练中发现了ELR坍缩现象：学习率（LR）和参数范数主要通过它们的比值——有效学习率（ELR）——来主导损失动态。当不同运行中的ELR匹配时，尽管学习率和参数范数显著不同，它们的损失轨迹在整个训练过程中会坍缩一致。跨优化器、架构、数据集和模型规模，平均坍缩误差通常为几×10^-3，低于代表性配置中测得的种子间变异。系统消融实验识别出归一化设计和LR-范数变化的时间尺度是坍缩精度的关键决定因素。受控干预进一步表明，权重衰减和超球形状主要通过它们诱导的ELR调度来影响损失动态。用ELR替换LR使得拟合的函数缩放定律（FSL）能够在不同的范数控制方法间迁移。基于ELR的FSL还解释了延迟加速现象。

    arXiv:2608.24814v1 Announce Type: new  Abstract: We uncover ELR collapse in language model pretraining: learning rate (LR) and parameter norm govern loss dynamics primarily through their ratio, the effective learning rate (ELR). When ELR is matched across runs, their loss trajectories collapse throughout training despite substantially different LRs and parameter norms. Across optimizers, architectures, datasets, and model scales, mean collapse errors are typically a few x 10^-3, below the seed-to-seed variation measured in a representative configuration. Systematic ablations identify normalization design and the timescale of LR-norm variation as key determinants of collapse precision. Controlled interventions further show that weight decay and Hyperball shape loss dynamics primarily through the ELR schedules they induce. Replacing LR with ELR enables a fitted functional scaling law (FSL) to transfer across norm-control methods. The resulting ELR-based FSL also explains delayed accelera
    
[^27]: 无金标准标签下AI生成数据的去偏推断：通过多重不完美测量进行识别

    Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements

    [https://arxiv.org/abs/2608.18294](https://arxiv.org/abs/2608.18294)

    本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。

    

    越来越多的学者使用AI来测量变量，并将其纳入后续的下游分析。尽管AI测量的变量通常被视为无误差观测，但忽略自动化测量中的预测误差会导致下游分析中的显著偏差和无效置信区间，即使AI测量准确度很高（例如超过90%）。现有的解决方案，如基于设计的有监督学习和预测支持推断，将基于AI的易错测量与金标准标签相结合，但在某些应用领域中，获取金标准标签可能成本高昂且困难。在本文中，我们提出了多重不完美测量的去偏推断（DMM），这是一个结合多个易错AI测量以实现无需金标准标签的有效下游推断的框架。基于CP分解的既有成果，DMM假设这些测量是独立的。

    arXiv:2608.18294v1 Announce Type: cross  Abstract: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.   In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent 
    
[^28]: 无对数因子的矩不等式与一致稳定算法的泛化界

    Logarithmic-Free Moment and Generalization Bounds for Uniformly Stable Algorithms

    [https://arxiv.org/abs/2608.09870](https://arxiv.org/abs/2608.09870)

    该论文去除了一致稳定算法泛化界中多余的对数因子 $\log n$，证明了无对数的矩不等式，从而肯定地回答了Bousquet等人（2020）提出的公开问题。

    

    一致稳定性是控制学习算法泛化误差的经典工具。Bousquet、Klochkov和Zhivotovskiy（2020）证明了该问题可以归约为关于独立随机变量的弱相互作用函数之和的矩不等式。他们的界包含一个额外的因子 $\log n$，并提出能否去除该因子的疑问。我们对这个上界问题给出了肯定的回答。更具体地，设 $Z=(Z_1,\ldots,Z_n)$ 的各坐标相互独立，且 $g_i(Z)$ 满足 $\mathbb{E}[g_i(Z)\mid Z_{-i}]=0$，$|\mathbb{E}[g_i(Z)\mid Z_i]|\le M$，对每个 $i=1,\dots,n$ 成立，其中 $Z_{-i}$ 表示除 $Z_i$ 之外的所有坐标。进一步假设改变任意坐标 $Z_j$（$j\neq i$）至多使 $g_i$ 改变 $\beta$，我们证明，对每个 $p\ge2$，有 $\left\|\sum_{i=1}^n g_i(Z)\right\|_p \le 16pn\beta+M\sqrt{2pn}$。

    arXiv:2608.09870v2 Announce Type: replace-cross  Abstract: Uniform stability is a classical tool for controlling the generalization error of a learning algorithm. Bousquet, Klochkov, and Zhivotovskiy (2020) showed that the problem can be reduced to a moment inequality for a sum of weakly interacting functions of independent random variables. Their bound contains an additional factor $\log n$, and they asked whether this factor can be removed. We answer this upper-bound question affirmatively. More specifically, let $Z=(Z_1,\ldots,Z_n)$ have independent coordinates and let $g_i(Z)$ satisfy $\mathbb E[g_i(Z)\mid Z_{-i}]=0, \ \left| \mathbb E[g_i(Z)\mid Z_i]\right|\le M, \ \text{for every } i = 1, \dots, n, $ where $Z_{-i}$ denotes all coordinates except $Z_i$. Assume additionally that changing any coordinate $Z_j$, $j\neq i$, changes $g_i$ by at most $\beta$, we prove that, for every $p\ge2$, for every $p\ge2$, $$ \left\| \sum_{i=1}^n g_i(Z)\right\|_p \le 16pn\beta+M\sqrt{2pn}. $$ This r
    
[^29]: 见树亦见林：BART的高斯过程极限

    Seeing the Forest for the Trees: The Gaussian Process Limit of BART

    [https://arxiv.org/abs/2607.28844](https://arxiv.org/abs/2607.28844)

    本文证明当树的数量趋于无穷时，BART收敛于一个具有特定核函数的高斯过程，并引入随机树特征作为其近似，实现了仅以对数方式依赖维度的极小化极大最优学习率，从而解释了BART优异性能的来源。

    

    贝叶斯加性回归树（BART）在预测和因果推断问题中均展现出最先进的性能。以往的理论工作试图通过为标准BART模型建立后验收缩率来解释BART的卓越性能，但这些收缩率强烈依赖于协变量的数量。在本文中，我们采取了一种不同的方法，研究当树的数量增长至无穷大时BART的行为。我们证明在这种极限情形下，BART收敛于一个具有特定核函数的高斯过程（GP）。该核函数及其对应的再生核希尔伯特空间（RKHS）具有良好的推断性质，这有助于解释BART的出色表现。我们引入随机树特征作为该极限高斯过程的近似，并为基于这些随机特征的岭回归建立了极小化极大最优学习率，该学习率仅以对数方式依赖于维度。除了……（摘要在此处被截断）

    arXiv:2607.28844v2 Announce Type: replace-cross  Abstract: Bayesian Additive Regression Trees (BART) have shown state-of-the-art performance in both prediction and causal inference problems. Previous theoretical work has attempted to explain BART's superior performance by establishing posterior contraction rates for standard BART models, but these rates depend strongly on the number of covariates. Here, we take a different approach and study the behavior of BART as the number of trees grows towards infinity. We show that in this regime, BART converges to a Gaussian process (GP) with a particular kernel. The kernel and its corresponding reproducing kernel Hilbert space (RKHS) have favorable inferential properties that help explain BART's excellent performance. We introduce random tree features as an approximation to this limiting GP, and establish minimax-optimal learning rates for ridge regression on these random features that depend only logarithmically on dimension. In addition to pr
    
[^30]: DiscoverPhysics：评估大语言模型开箱即用科学思维能力的基准测试

    DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking

    [https://arxiv.org/abs/2605.26087](https://arxiv.org/abs/2605.26087)

    提出了交互式基准测试DiscoverPhysics，通过让大语言模型在物理规律刻意偏离现实的22个模拟世界中设计实验、观察轨迹数据并归纳未知的运动定律，从而将模型真正的科学推理能力与对既有物理知识的记忆区分开来。

    

    前沿大语言模型如今在各类物理评测中表现优异，但很难将其真正的推理能力与对既有科学知识的记忆区分开来。我们提出了DiscoverPhysics，这是一个交互式基准测试，要求大语言模型智能体去发现一个模拟世界的运动定律，而该世界的物理规律被刻意设置为偏离我们的现实世界。我们构建了22个这样的世界，其物理规律包括屏蔽引力、分数幂引力、多物种耦合、隐藏的类暗物质粒子、非坐标无关的物理以及随时间变化的相互作用等。每个世界由N体模拟器按需生成，智能体需要提出多轮实验方案、观察原始轨迹数据，并最终提交对该世界物理规律的自然语言解释以及所推断定律的Python代码实现。由于解决一个世界的问题需要智能体设计具有信息量的实验并不断修正其假设，该基准……

    arXiv:2605.26087v2 Announce Type: replace-cross  Abstract: Frontier LLMs now perform strongly across a wide range of physics evaluations, but it is hard to disentangle genuine reasoning from recall of established science. We introduce DiscoverPhysics, an interactive benchmark that asks a LLM agent to discover the laws of motion of a simulated world whose physics deliberately deviates from our own. We construct 22 worlds governed by, among others, screened and fractional-power gravity, multi-species couplings, hidden dark-matter-like particles, non-coordinate-free physics, and time-varying interactions. Each world is generated on demand by an N-body simulator, for which the agent proposes several rounds of experiments, observes raw trajectory data, and ultimately submits both a natural-language explanation of the world's physics and a Python implementation of the inferred law. Because solving a world requires the agent to design informative experiments and revise its hypotheses, the ben
    
[^31]: FedSPDnet：基于SPDnet的几何感知联邦深度学习

    FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet

    [https://arxiv.org/abs/2604.22494](https://arxiv.org/abs/2604.22494)

    提出了FedSPDnet框架，通过ProjAvg和RLAvg两种保持Stiefel流形几何结构的聚合策略，实现了基于SPD矩阵的联邦深度学习，在EEG运动想象基准上以更少的通信参数和更强的鲁棒性超越了联邦EEGnet。

    

    我们为经典的SPDnet模型提出了两个联邦学习框架，该模型处理对称正定（SPD）矩阵并带有Stiefel约束参数。与违反正交性的标准欧几里得平均不同，我们的方法通过两种高效的聚合策略保持几何结构：ProjAvg（将算术平均投影到Stiefel流形上）和RLAvg（通过回缩和提升近似切空间平均）。这两种方法计算高效、与优化器无关，并能为特征为SPD矩阵的信号处理应用实现可扩展的联邦学习。在EEG运动想象基准上的仿真表明，FedSPDnet在F1分数以及对联邦和部分参与场景的鲁棒性方面优于联邦EEGnet，同时每轮通信使用的参数更少。

    arXiv:2604.22494v2 Announce Type: replace-cross  Abstract: We introduce two federated learning frameworks for the classical SPDnet model operating on symmetric positive definite (SPD) matrices with Stiefel-constrained parameters. Unlike standard Euclidean averaging, which violates orthogonality, our approach preserves geometric structure through two efficient aggregation strategies: ProjAvg, projecting arithmetic means onto the Stiefel manifold, and RLAvg, approximating tangent-space averaging via retractions and liftings. Both methods are computationally efficient, independent of the optimizer, and enable scalable federated learning for signal processing applications whose features are SPD matrices. Simulations on EEG motor imagery benchmarks show that FedSPDnet outperforms federated EEGnet in F1 score and robustness to federation and partial participation, while using fewer parameters per communication round.
    
[^32]: 无需交叉拟合的多重依赖去偏机器学习方法

    Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence

    [https://arxiv.org/abs/2602.11333](https://arxiv.org/abs/2602.11333)

    本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。

    

    arXiv:2602.11333v3 公告类型：替换 摘要：本文针对广义矩估计（GMM）模型中具有一般多重聚类依赖的两步去偏机器学习（DML）估计量，开发了一种渐近理论，且不依赖交叉拟合。虽然交叉拟合被广泛使用，但当第一阶段学习器复杂且有效样本量由独立聚类数量决定时，它在统计上可能低效且计算负担沉重。我们证明，通过结合Neyman正交矩条件和基于局部化的经验过程方法，可以在不进行样本分割的情况下实现有效推断，并允许任意数量的聚类维度。结果表明，在多重聚类依赖下，所得的去偏GMM估计量具有渐近线性和渐近正态性。本文的一个核心技术贡献是为一般类别推导出新的全局和局部极大不等式。

    arXiv:2602.11333v3 Announce Type: replace  Abstract: This paper develops an asymptotic theory for two-step debiased machine learning (DML) estimators in generalised method of moments (GMM) models with general multiway clustered dependence, without relying on cross-fitting. While cross-fitting is commonly employed, it can be statistically inefficient and computationally burdensome when first-stage learners are complex and the effective sample size is governed by the number of independent clusters. We show that valid inference can be achieved without sample splitting by combining Neyman-orthogonal moment conditions with a localisation-based empirical process approach, allowing for an arbitrary number of clustering dimensions. The resulting debiased GMM estimators are shown to be asymptotically linear and asymptotically normal under multiway clustered dependence. A central technical contribution of the paper is the derivation of novel global and local maximal inequalities for general clas
    
[^33]: 持续熵作为相变的探测器

    Persistent Entropy as a Detector of Phase Transitions

    [https://arxiv.org/abs/2602.09058](https://arxiv.org/abs/2602.09058)

    本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。

    

    持续熵是持续性条形码的一种标量摘要，被广泛用于检测状态变化，然而目前尚无理论阐明条形码中的结构性变化何时必然会导致可检测的熵变化。我们建立了一个与模型无关的定理来提供此类条件。通过将持久图视为由控制参数索引的随机对象，我们在归一化持久权重中识别出一种“分散-凝聚”机制，并推导出两种状态之间熵差的显式下界，该下界在有限样本量下以高概率成立，且对条形寿命的绝对尺度不敏感。我们还给出了一套在经验条形码上验证这些假设的程序。应用于卷积网络时，该准则表明 Gabrielsson 和 Carlsson 所报告的学习滤波器的环形组织是通过一次尖锐的拓扑相变而产生的，并定位了该相变的发生起点。

    arXiv:2602.09058v2 Announce Type: replace-cross  Abstract: Persistent entropy is a scalar summary of persistence barcodes widely used to detect regime changes, yet there is no account of when a structural change in a barcode must produce a detectable change in entropy. We establish a model-agnostic theorem supplying such conditions. Treating persistence diagrams as random objects indexed by a control parameter, we identify a dispersion-condensation mechanism in the normalized persistence weights and derive an explicit lower bound on the entropy difference between the two regimes, valid with high probability at finite sample size and insensitive to the absolute scale of bar lifetimes. We also give a procedure for verifying the hypotheses on empirical barcodes. Applied to convolutional networks, the criterion shows that the circular organization of learned filters reported by Gabrielsson and Carlsson emerges through a sharp topological phase transition, and locates its onset: within a fe
    
[^34]: 建模非随机缺失时间序列数据中的信息中断

    Modeling Information Blackouts in Missing Not-At-Random Time Series Data

    [https://arxiv.org/abs/2601.01480](https://arxiv.org/abs/2601.01480)

    该论文提出了一种感知非随机缺失（MNAR）的潜在状态空间模型，用于建模交通传感器网络中的连续信息中断，证明当缺失机制依赖于潜在交通状态时，考虑这种依赖关系可显著提升数据插补精度与缺失检测性能。

    

    交通预测系统依赖于固定传感器网络，而这些网络经常出现连续性的数据中断。此类中断通常被当作可忽略的缺失数据处理，尽管数据丢失实际上可能取决于未观测到的交通状况。我们通过一个感知非随机缺失（MNAR）的潜在状态空间模型来研究这种可能性，该模型将线性交通动力学与伯努利缺失通道相结合，其缺失概率取决于潜在状态。推断采用扩展卡尔曼滤波器（EKF）以及随后的Rauch-Tung-Striebel（RTS）平滑，参数通过近似EM算法学习。我们使用一套无数据泄漏、月份平衡的300个独特的全视界对齐中断窗口数据集，对西雅图的交通数据进行评估。在该基准测试中，MAR-LDS达到4.264英里/小时的合并插补RMSE，而MNAR-LDS将其改进至4.177（差异为-0.086）；基于检测器聚类的自助法95%置信区间为[-0.182, -0.002]。因果性单步预测潜在表示将缺失检测的ROC-AUC从……

    arXiv:2601.01480v3 Announce Type: replace-cross  Abstract: Traffic forecasting systems rely on fixed sensor networks that frequently exhibit contiguous blackouts. Such outages are usually treated as ignorable missingness, although dropout can depend on unobserved traffic conditions. We study this possibility with an MNAR-aware latent state-space model that combines linear traffic dynamics with a Bernoulli missingness channel whose probability depends on the latent state. Inference uses an Extended Kalman Filter (EKF) followed by Rauch-Tung-Striebel (RTS) smoothing, and parameters are learned by approximate EM. We evaluate Seattle using a leakage-free, month-balanced set of 300 unique all-horizon-aligned blackout windows. On this benchmark, MAR-LDS attains 4.264 mph pooled imputation RMSE and MNAR-LDS improves it to 4.177 (difference -0.086); the detector-cluster bootstrap 95% interval is [-0.182,-0.002]. A causal one-step predicted latent representation raises missingness ROC-AUC from 
    
[^35]: 模型预测控制对于异质不休息多臂老虎机几乎是最优的

    Model Predictive Control is almost Optimal for Heterogeneous Restless Multi-armed Bandits

    [https://arxiv.org/abs/2511.08097](https://arxiv.org/abs/2511.08097)

    本文针对每个臂参数各不相同的异质无限时域不休息多臂老虎机，证明通过反复求解有限线性规划的模型预测控制策略（LP-update）在一致遍历性假设下具有 O(√(1/N)) 的次优性差距，即该经典算法几乎是最优的。

    

    我们考虑了一般性的无限时域异质不休息多臂老虎机（RMAB）问题。异质性是许多现实系统中的一个根本性难题，主要是因为它使得许多集中性论证难以适用。在本文中，我们假设 $N$ 个臂中的每一个都可以具有不同的模型参数。模型预测控制是一种著名的控制策略，它通过反复求解长度为 $\tau$ 的有限时域优化问题来产生可应用于无限时域环境的策略。在本文中，我们采用这一方法，通过反复求解有限线性规划，得到我们称之为无限时域问题的LP-update策略。在一致遍历性这一温和假设下，我们证明了这一在实践中表现非常好的著名算法具有 $\mathcal{O}\left(\sqrt{1/N}\right)$ 的次优性差距。除了LP-update策略之外，我们还能够推导出一种有限时域策略（LP-update w……

    arXiv:2511.08097v2 Announce Type: replace-cross  Abstract: We consider a general infinite horizon Heterogeneous Restless multi-armed Bandit (RMAB). Heterogeneity is a fundamental problem for many real-world systems largely because it resists many concentration arguments. In this paper, we assume that each of the $N$ arms can have different model parameters. Model predictive control is a well-known control strategy that repeatedly solves a finite-horizon optimization problem of length $\tau$ to produce a policy that can be applied to an infinite-horizon setting. In this paper, we adopt this approach by repeatedly solving a finite linear program, yielding what we call the LP-update policy for the infinite-horizon problem. Under a mild assumption of uniform ergodicity, we show an $\mathcal{O}\left(\sqrt{1/N}\right)$ suboptimality gap on this well-known algorithm that works very well in practice. In addition to the LP-update policy we are able to derive a finite-horizon policy (LP-update w
    
[^36]: 如果你能区分，你就能表达：伽罗瓦理论、Stone–Weierstrass定理、机器学习与语言学

    If you can distinguish, you can express: Galois theory, Stone--Weierstrass, machine learning, and linguistics

    [https://arxiv.org/abs/2510.09902](https://arxiv.org/abs/2510.09902)

    本文揭示了伽罗瓦理论基本定理与Stone–Weierstrass定理的共同本质——区分能力决定表达能力，并将这一原理延伸至机器学习、数据科学与语言学领域。

    

    本文探讨了伽罗瓦理论基本定理与Stone–Weierstrass定理之间的平行关系：两者都可以被视为将一类对象的区分能力与其表达能力联系起来的断言。我们提供了一个连接相关“区分能力”概念的初等定理。我们还讨论了机器学习和数据科学领域中这些定理出现的相关情境，以及更广泛意义上区分能力与表达能力之间联系这一主题。最后，我们在语言学语境中讨论了同一主题，它在语言学中作为一种基础性原则出现，并用几个例子加以说明。

    arXiv:2510.09902v3 Announce Type: replace-cross  Abstract: This essay develops a parallel between the Fundamental Theorem of Galois Theory and the Stone--Weierstrass theorem: both can be viewed as assertions that tie the distinguishing power of a class of objects to their expressive power. We provide an elementary theorem connecting the relevant notions of "distinguishing power". We also discuss machine learning and data science contexts in which these theorems, and more generally the theme of links between distinguishing power and expressive power, appear. Finally, we discuss the same theme in the context of linguistics, where it appears as a foundational principle, and illustrate it with several examples.
    
[^37]: Transformer中性能与效率的权衡：基于逼近理论的视角

    Performance-Efficiency Tradeoffs in Transformers: An Approximation Theory Perspective

    [https://arxiv.org/abs/2510.03784](https://arxiv.org/abs/2510.03784)

    本文从逼近理论视角刻画了Transformer中注意力头数量与头维度在固定参数预算下的权衡，发现并证明了softmax激活的饱和行为，表明较深的层可以用更小的头维度实现高效运行。

    

    Transformer在各类应用中取得了显著的成功，但其模型效率的理论基础仍未得到充分探索。在这项工作中，我们研究了模型参数——主要是注意力头数量和头的维度——应如何在不同层之间分配，以平衡表达能力与效率。我们首先从逼近理论的角度对早期层在信息提取中的作用进行了数学分析，并在固定参数预算下对注意力头数量与头维度之间的权衡进行了理论刻画。此外，我们发现并证明了softmax激活的饱和行为：持续增加头维度可能导致学习误差的收益递减，特别是在长序列情况下。在理论和实验的双重支持下，这种饱和模式表明后面的层可以通过减少头维度以更高效的方式运行。

    arXiv:2510.03784v2 Announce Type: replace  Abstract: Transformers have achieved remarkable successes across a wide range of applications, yet the theoretical foundation of their model efficiency remains underexplored. In this work, we investigate how the model parameters -- mainly attention heads and head dimensions -- should be allocated across layers to balance expressivity and efficiency. We first provide mathematical analysis on the role of early layers in information extraction from an approximation perspective, with a theoretical characterization on the trade-off between the number of heads and head dimension under a fixed parameter budget. In addition, we uncover and prove the \emph{saturation} behavior of softmax activations: Continuously increasing head dimensions can lead to diminishing returns in learning errors, particularly for long sequences. Supported by both theory and experiments, this saturation pattern suggests that later layers can operate more efficiently with redu
    
[^38]: AL-SPCE——基于随机多项式混沌展开与主动学习的非确定性模型可靠性分析

    AL-SPCE - Reliability analysis for nondeterministic models using stochastic polynomial chaos expansions and active learning

    [https://arxiv.org/abs/2507.04553](https://arxiv.org/abs/2507.04553)

    提出了一种结合随机多项式混沌展开与主动学习的方法AL-SPCE，能够以显著更少的训练样本对具有随机性的非确定性模型进行高精度、低成本的可靠性分析。

    

    arXiv:2507.04553v2 公告类型：replace-cross 摘要：可靠性分析传统上依赖于确定性模拟器，即相同的输入会产生相同的输出。然而，许多现实世界的系统表现出随机行为，即使在相同条件下也会产生不可重复的结果。随机模拟器通过将响应表示为随机变量来刻画这种行为，其内在的变异性必须在可靠性分析中加以考虑。虽然蒙特卡洛模拟可以解决这一问题，但其计算成本往往过于高昂。因此，随机代理模型（emulator）被引入，作为能够在更低成本下重现随机模拟器响应的替代模型。近期研究已展示了其在可靠性分析方面的潜力，但准确的估计仍可能需要相对较大的训练集，这对于计算代价高昂的模型而言并不现实。在这项工作中，我们提出了一个主动学习框架，以进一步降低计算成本。

    arXiv:2507.04553v2 Announce Type: replace-cross  Abstract: Reliability analysis traditionally relies on deterministic simulators, where identical inputs yield identical outputs. However, many real-world systems exhibit stochastic behavior, producing non-repeatable outcomes even under identical conditions. Stochastic simulators account for this behavior by representing the response as a random variable, whose intrinsic variability must be considered in reliability analysis. While Monte Carlo simulation can address this problem, its computational cost is often prohibitive. Stochastic emulators have therefore been introduced as surrogate models capable of reproducing the random simulator response at reduced cost. Recent studies have shown their potential for reliability analysis, but accurate estimates may still require relatively large training sets, which can be impractical for expensive models. In this work, we propose an active learning framework to further reduce the computational ef
    
[^39]: 任意顺序GPT作为掩码扩散模型：解耦建模公式与架构

    Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture

    [https://arxiv.org/abs/2506.19935](https://arxiv.org/abs/2506.19935)

    本研究将掩码扩散模型置于仅解码器架构框架中，与自回归模型进行公平比较，发现其通过温度退火等技术可实现约25倍的推理加速且困惑度相当，为降低大语言模型推理计算成本提供了新路径。

    

    高效扩展大语言模型（LLM）需要探索自回归（AR）主导方法的替代方案，掩码扩散模型（MDM）正成为有力的候选。然而，比较AR（通常为仅解码器架构）与MDM（通常为仅编码器架构）这两种范式时，架构差异会造成混淆，掩盖了真正的算法与效率权衡。本研究通过在仅解码器框架内评估MDM来解耦这些因素，以实现两个目标：(1) 通过对生成顺序差异的分析，公平比较MDM（作为任意顺序自回归模型）与标准AR范式；(2) 研究MDM架构对计算效率的影响。我们证明，尽管建模空间更大，仅解码器MDM借助温度退火等技术，仍可实现显著的推理加速（约25倍），同时保持与AR模型相当的困惑度，为降低推理计算成本提供了一条可行路径。这项工作为开发计算效率更高的模型提供了重要见解。

    arXiv:2506.19935v2 Announce Type: replace-cross  Abstract: Efficiently scaling Large Language Models (LLMs) necessitates exploring alternatives to dominant autoregressive (AR) methods, with Masked Diffusion Models (MDMs) emerging as candidates. However, comparing AR (typically decoder-only) and MDM (often encoder-only) paradigms is confounded by differing architectures, obscuring true algorithmic and efficiency trade-offs. This research decouples these factors by evaluating MDMs within a decoder-only framework to: (1) Equitably compare MDM (as Any-Order AR) and standard AR paradigms through discrepancies on orders. (2) Investigate MDM architectural impacts on computational efficiency. We show decoder-only MDMs, despite a larger modeling space, can achieve significant inference speedups ($\sim25\times$) and comparable perplexity with techniques like temperature annealing, offering a path to reduced inference compute. This work provides insights for developing more computationally effici
    
[^40]: 关于高维线性分类中一致性对抗攻击的存在性

    On the Existence of Consistent Adversarial Attacks in High-Dimensional Linear Classification

    [https://arxiv.org/abs/2506.12454](https://arxiv.org/abs/2506.12454)

    本文提出了一种新的误差度量来区分真正的一致性对抗攻击（即保持真实标签不变的扰动）与因数据有限或模型能力不足导致的普通误分类，并通过精确的渐近理论分析证明，随着模型过参数化程度的提高，其对标签保持扰动的脆弱性会不断增大。

    

    对抗攻击与因模型表达能力有限或数据有限而导致的错误分类，其根本区别究竟是什么？在本工作中，我们在高维二分类的设定下研究这一问题，其中数据有限所带来的统计效应起着核心作用。我们引入了一种新的误差度量，能够精确捕捉这一区别，量化模型对一致性对抗攻击的脆弱性——即那些保持真实标签不变的扰动。我们的主要技术贡献在于对良好指定模型和潜在空间模型中的这些度量给出了精确且严格的渐近刻画，揭示了与标准鲁棒误差度量不同的脆弱性模式。理论结果表明，随着模型变得更加过参数化，其对抗保持标签扰动的脆弱性也随之增长，为理解这一机制提供了理论洞见。

    arXiv:2506.12454v2 Announce Type: replace-cross  Abstract: What fundamentally distinguishes an adversarial attack from a misclassification due to limited model expressivity or finite data? In this work, we investigate this question in the setting of high-dimensional binary classification, where statistical effects due to limited data availability play a central role. We introduce a new error metric that precisely capture this distinction, quantifying model vulnerability to consistent adversarial attacks -- perturbations that preserve the ground-truth labels. Our main technical contribution is an exact and rigorous asymptotic characterization of these metrics in both well-specified models and latent space models, revealing different vulnerability patterns compared to standard robust error measures. The theoretical results demonstrate that as models become more overparameterized, their vulnerability to label-preserving perturbations grows, offering theoretical insight into the mechanisms
    
[^41]: 基于平滑随机梯度下降的分位数在线同时推断

    Online simultaneous inference for quantiles via smoothed stochastic gradient descent

    [https://arxiv.org/abs/2505.13299](https://arxiv.org/abs/2505.13299)

    本文提出一种平滑随机梯度下降方法用于流数据的在线分位数估计，其估计量在每次迭代中关于分位数水平单调，并借助一致Bahadur表示与布朗桥最大值的高斯近似，实现了维度随样本量指数增长时跨坐标与分位数水平的在线同时统计推断。

    

    本文考虑通过随机梯度下降（SGD）算法的平滑版本来估计分位数。通过使用与学习率相关联的带宽对得分函数进行平滑，我们得到的估计量在每次迭代中都关于分位数水平保持单调，同时保留了流数据处理所需的内存和计算效率。我们建立了平滑估计量在使用与不使用Polyak-Ruppert平均两种情况下的非渐近尾概率界，这些界是具有多区域结构的亚指数型。对于平均估计量，我们进一步推导出关于分位数水平和各坐标一致成立的Bahadur表示，以及由布朗桥最大值给出的高斯近似，其中维度 $p$ 允许随样本量呈指数级增长。由此实现了跨坐标与分位数水平的同时推断。作为一种避免估计……的替代方法（摘要在此处被截断）

    arXiv:2505.13299v2 Announce Type: replace-cross  Abstract: This paper considers the estimation of quantiles via a smoothed version of the stochastic gradient descent (SGD) algorithm. By smoothing the score function with a bandwidth tied to the learning rate, we obtain estimates that are monotone in the quantile level at every iteration, while retaining the memory and computational efficiency required for streaming data. We establish non-asymptotic tail probability bounds for the smoothed estimate with and without Polyak-Ruppert averaging, which are sub-exponential with a multi-regime structure. For the averaged estimate we further derive a Bahadur representation that is uniform in the quantile level and across coordinates, and a resulting Gaussian approximation by the maximum of Brownian bridges, with the dimension $p$ allowed to grow exponentially in the sample size. This yields simultaneous inference across coordinates and quantile levels. As an alternative that avoids estimating the
    
[^42]: 无需非高斯性假设的多视图因果发现：可辨识性与算法

    Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms

    [https://arxiv.org/abs/2502.20115](https://arxiv.org/abs/2502.20115)

    本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。

    

    因果发现是一个困难的问题，通常依赖于对数据生成模型的强假设，例如非高斯性。在实践中，许多现代应用提供了同一系统的多个相关视图，而这一点在因果发现领域很少被考虑。在此，我们利用这种多视图结构，在弱假设条件下实现因果发现。我们提出了一个多视图线性结构方程模型（SEM），该模型通过交替利用视图间的相关性，扩展了著名的非高斯扰动框架。我们证明了该模型在无环SEM情形下的可辨识性。随后，受单视图算法（DirectLiNGAM、PairwiseLiNGAM和ICA-LiNGAM）的启发，我们提出了几种多视图因果发现算法。新方法通过仿真实验和神经影像数据应用得到了验证，在这些应用中，它们能够估计脑区之间的因果图。

    arXiv:2502.20115v4 Announce Type: replace  Abstract: Causal discovery is a difficult problem that typically relies on strong assumptions on the data-generating model, such as non-Gaussianity. In practice, many modern applications provide multiple related views of the same system, which has rarely been considered for causal discovery. Here, we leverage this multi-view structure to achieve causal discovery with weak assumptions. We propose a multi-view linear Structural Equation Model (SEM) that extends the well-known framework of non-Gaussian disturbances by alternatively leveraging correlation over views. We prove the identifiability of the model for acyclic SEMs. Subsequently, we propose several multi-view causal discovery algorithms, inspired by single-view algorithms (DirectLiNGAM, PairwiseLiNGAM, and ICA-LiNGAM). The new methods are validated through simulations and applications on neuroimaging data, where they enable the estimation of causal graphs between brain regions.
    
[^43]: 通过熵流计算为马尔可夫算法建立泛化界

    Generalization Bounds for Markov Algorithms through Entropy Flow Computations

    [https://arxiv.org/abs/2502.07584](https://arxiv.org/abs/2502.07584)

    该论文提出新的技术工具，将熵流方法的适用范围从特定的噪声和算法结构（如朗之万动力学）扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法，从而为这一广泛类别的算法建立泛化界。

    

    许多学习算法可以表示为马尔可夫过程，理解它们的泛化误差是学习理论中的核心课题。对于特定的连续时间含噪算法，一种突出的分析技术依赖于信息论工具和所谓的“熵流”方法。该技术与广泛的假设条件兼容，并利用学习动力学的收敛性质来产生有意义的泛化界，这些界也可以具有信息量或扩展到离散时间设置。尽管取得了成功，现有的熵流公式仅限于特定的噪声和算法结构（例如，朗之万动力学）。在这项工作中，我们利用新的技术工具将其适用性扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法。我们的方法基于对马尔可夫算法的原理性连续时间近似……

    arXiv:2502.07584v3 Announce Type: replace-cross  Abstract: Many learning algorithms can be represented as Markov processes, and understanding their generalization error is a central topic in learning theory. For specific continuous-time noisy algorithms, a prominent analysis technique relies on information-theoretic tools and the so-called ``entropy flow'' method. This technique is compatible with a broad range of assumptions and leverages the convergence properties of learning dynamics to produce meaningful generalization bounds, which can also be informative or extend to discrete-time settings. Despite their success, existing entropy flow formulations are limited to specific noise and algorithm structures (\eg, Langevin dynamics). In this work, we exploit new technical tools to extend its applicability to all learning algorithms whose iterative dynamics is governed by a time-homogeneous Markov process. Our approach builds on a principled continuous-time approximation of Markov algori
    
[^44]: QABBA：通过整数量化聚合实现带误差保证的符号时间序列压缩

    QABBA: Error-Guaranteed Symbolic Time-Series Compression via Integer-Quantized Aggregation

    [https://arxiv.org/abs/2411.15209](https://arxiv.org/abs/2411.15209)

    提出QABBA，通过量化符号中心实现ABBA的整数化压缩，在保证重建质量的同时提供严格的误差界限。

    

    来自传感器和监控系统的时间序列数据的扩张使得紧凑表示变得越来越重要。这种表示应在削减存储、传输和计算成本的同时保留信号结构。自适应布朗桥聚合（ABBA）通过将长数值序列转换为短符号序列来满足这一需求，但参数存储和计算精度的降低仍然值得追求。我们提出了量化ABBA（QABBA），即ABBA的量化版本。通过量化符号中心，QABBA减少了参数占用，并启用整数运算，同时保持高重建质量。我们为量化引入的额外近似建立了多个误差界：每个段超额误差的无维度界、时域重建误差界、符号分配的稳定性条件，以及分配位的规则。

    arXiv:2411.15209v3 Announce Type: replace  Abstract: The expansion of time-series data from sensors and monitoring systems has made compact representations increasingly important. Such representations should retain signal structure while cutting storage, transmission and computation costs. Adaptive Brownian Bridge-based Aggregation (ABBA) addresses this need by converting long numerical series into short symbolic sequences, but reductions in parameter storage and computational precision remain desirable.   We propose Quantized ABBA (QABBA), a quantized version of ABBA. By quantizing the symbolic centers, QABBA reduces the parameter footprint and enables integer arithmetic while maintaining high reconstruction quality. We establish several error bounds for the additional approximation introduced by quantization: a dimension-free bound on the excess error of each segment, a time-domain reconstruction-error bound, a stability condition for symbolic assignment, and a rule for allocating bi
    
[^45]: 让每个人都满意：少量副本下大量物品的在线公平分配

    Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies

    [https://arxiv.org/abs/2408.12845](https://arxiv.org/abs/2408.12845)

    针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。

    

    本文研究了在线公平分配问题的一种新变体，该问题涉及多个智能体，学习者按顺序观察到不可分割的物品，必须将其不可撤销地分配给其中一个智能体，以在公平性和效率之间实现理想的平衡。现有算法假设物品数量少且副本数量足够大，这保证了能够从带噪声的观测效用中对所有物品-智能体对进行良好的效用估计。然而，这一假设在许多现实应用中可能不成立，例如，一个在线平台拥有大量用户（物品），这些用户仅使用平台的服务提供商（智能体）少数几次（即物品只有少量副本），这使得难以准确估计所有物品-智能体对的效用。为了解决这一局限性，我们假设效用是物品-智能体特征的未知函数，并提出将在线公平分配建模为上下文老虎机问题的算法。

    arXiv:2408.12845v3 Announce Type: replace-cross  Abstract: This paper considers a novel variant of the online fair division problem involving multiple agents in which a learner sequentially observes an indivisible item that must be irrevocably allocated to one of the agents to achieve a desired balance between fairness and efficiency. Existing algorithms assume a small number of items with a sufficiently large number of copies, which ensures a good utility estimation for all item-agent pairs from noisy observed utilities. However, this assumption may not hold in many real-life applications, e.g., an online platform with a large number of users (items) who use the platform's service providers (agents) only a few times (a few copies of items), making it difficult to accurately estimate utilities for all item-agent pairs. To address this limitation, we assume utility is an unknown function of item-agent features. We propose algorithms that model online fair division as a contextual bandit
    
[^46]: 基于深度学习的随机偏微分方程数值逼近算法

    Deep learning based numerical approximation algorithms for stochastic partial differential equations

    [https://arxiv.org/abs/2012.01194](https://arxiv.org/abs/2012.01194)

    本文提出一种基于深度学习的随机偏微分方程逼近算法，通过神经网络沿噪声轨迹逼近SPDE解并估计其经验分布，在随机热方程、Black-Scholes方程和Zakai方程等测试中实现了高达100维空间下的快速精确求解。

    

    在这篇文章中，我们介绍了一种基于深度学习的随机偏微分方程（SPDEs）逼近算法。我们的方法采用神经网络来逼近SPDEs在给定驱动噪声过程实现下的解。当应用于一组模拟的噪声轨迹时，该方法可以产生SPDE解的经验分布，从中能够估计诸如均值和方差等泛函。我们在具有加性和乘性噪声的随机热方程、具有乘性噪声的随机Black-Scholes方程以及来自非线性滤波理论的Zakai方程上测试了该方法的性能。在所有情况下，所提出的算法在高达100个空间维度上都能产生准确的结果，且运行时间短。

    arXiv:2012.01194v3 Announce Type: replace-cross  Abstract: In this article, we introduce a deep learning based approximation algorithm for SPDEs. Our approach employs neural networks to approximate the solutions of SPDEs along given realizations of the driving noise process. If applied to a set of simulated noise trajectories, it yields empirical distributions of SPDE solutions, from which functionals like the mean and variance can be estimated. We test the performance of the method on stochastic heat equations with additive and multiplicative noise as well as stochastic Black-Scholes equations with multiplicative noise and Zakai equations from nonlinear filtering theory. In all cases, the proposed algorithm yields accurate results with short runtimes in up to 100 space dimensions.
    

