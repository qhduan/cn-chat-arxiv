# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Memory-Conditioned Diffusion Model for Generalized Langevin Dynamics](https://arxiv.org/abs/2609.28371) | 该论文提出一种记忆条件扩散模型，利用递归更新的指数滤波器组从观测轨迹中学习广义朗之万动力学的随机流映射，无需识别记忆核或重建未解析变量，且记忆成本与历史长度无关。 |
| [^2] | [Local Geometric Mixing via Dobrushin Contraction with Applications to Diffusion Path Monte Carlo and the Proximal Sampler](https://arxiv.org/abs/2609.28338) | 本文提出基于Dobrushin收缩的局部几何混合分析框架，并将其应用于扩散路径蒙特卡洛和近端采样器，在最少假设下为理想方法及Metropolis校正版本提供了混合时间保证。 |
| [^3] | [How Sensitive Are LLM Leaderboard Claims to Hidden Model Selection?](https://arxiv.org/abs/2609.28177) | 该论文提出一种敏感性分析方法，量化一个排行榜领先幅度背后可能隐藏的私下挑选的模型变体数量，并对 Open LLM Leaderboard 上 394 个相邻排名声明进行审计，发现其中 391 个即使不考虑选择效应也缺乏统计支持。 |
| [^4] | [Rank-One Signal Recovery in Sparse Wishart Noise](https://arxiv.org/abs/2609.28163) | 本文利用复制方法解析计算了秩一信号形变的稀疏 Wishart 噪声矩阵的最大特征值对统计特性，通过递归分布方程组与种群动力学算法高效求解，刻画了信号恢复性能对信号强度、矩阵矩形比和噪声连通度的依赖关系。 |
| [^5] | [NPBoost: Neural Processes with Gradient-Boosted Fixed Effects](https://arxiv.org/abs/2609.28122) | NPBoost将结构化响应变异性分解为跨任务共享的梯度提升树固定效应与捕捉任务间随机变化的神经过程随机效应，并通过提升算法联合训练二者，在共享结构包含不连续性等不规则特征的表格元学习任务上优于标准神经过程。 |
| [^6] | [Improving Ensemble Filters with Flow Matching](https://arxiv.org/abs/2609.28015) | 提出流集合滤波器，利用条件流匹配学习非线性更新，将经典基线滤波器的预报集合传输为分析集合，在稀疏观测的动力学系统中性能超越所有经典集合滤波器及最先进的生成式方法。 |
| [^7] | [Conformal Bayes under Continuous Label Shift: Sensitivity Analysis and the Limits of Exact Validity](https://arxiv.org/abs/2609.27976) | 该论文提出联合倾斜敏感性共形贝叶斯（JTS-CB/JTS-SCB），通过对预设的合理倾斜集合进行联合敏感性分析来应对连续标签偏移，同时揭示了精确有限样本有效性取决于密度比尾部行为的根本局限。 |
| [^8] | [Fourth-Moment Strong Universality for Finite-Type Transpose-Correlated Random Matrices](https://arxiv.org/abs/2609.27971) | 本文在精确的四阶矩阈值处证明了有限型转置相关的非厄米随机矩阵族在矩阵放大和非交换 *-多项式下强收敛于显式协方差匹配的自由高斯族的强普适性定理。 |
| [^9] | [Dirichlet Process Mixtures of Trees with Gaussian Process Splits: A Bayesian Nonparametric Framework with Posterior Contraction Rate](https://arxiv.org/abs/2609.27930) | 该论文提出了一种基于狄利克雷过程先验和由高斯过程后验预测驱动分裂规则的贝叶斯非参数回归树混合框架，统一了CART、BART、随机森林和提升方法，并证明了在真实回归函数仅连续的宽松条件下Hellinger距离上 $n^{-1/4}$ 的后验收缩速率。 |
| [^10] | [Theoretical Study on the Evidential Learning-based Variational Autoencoder](https://arxiv.org/abs/2609.27853) | 该论文从理论上证明，证据学习变分自编码器中正态-逆伽马潜在层级的四个参数仅有三维商空间 (γ, α, c) 是可辨识的，并且通过对前向KL散度的精确偏最小化，可以实现从四参数到三坐标的精确约简，同时保持最优值不变。 |
| [^11] | [Optimal State-Space Order for Spectral Gaps of Sliding-Window Occupation Counts](https://arxiv.org/abs/2609.27836) | 本文证明了滑动窗口占据计数所定义的投影计数核的谱隙与原马尔可夫核谱隙之比的最优常数 $c_m^\star$ 满足 $\Theta(1/m)$ 的阶，给出了普适下界 $1/(1080m)$ 和形如 $(m+\log m+O(1))^{-1}$ 的上界，但精确值仍待确定。 |
| [^12] | [Financial Tail Risk Beyond Lipschitz Continuity via Semi-Discrete Optimal Transport](https://arxiv.org/abs/2609.27785) | 该论文证明了Lipschitz连续性约束使基于神经生成器的采样方法在数学上无法精确匹配厚尾的金融收益分布，并提出用半离散最优传输（SDOT）放松映射正则性来突破这一限制，从而实现更准确的金融尾部风险估计。 |
| [^13] | [Type-II Error Bounds for Test Supermartingales from Lower-Tail Hypotheses](https://arxiv.org/abs/2609.27766) | 本文针对检验上鞅方法中的第二类错误问题，研究了对对数增量下尾概率的不同假设所导出的第二类错误界，并将所有结果统一为一个主不等式，即通过 e 变量（逆）矩生成函数的单侧勒让德变换来刻画序贯检验在固定时域和任意时刻的第二类错误上界。 |
| [^14] | [The Type-II Error of Test Supermartingales: e-Power versus the Chernoff-Stein Exponent](https://arxiv.org/abs/2609.27765) | 该论文证明了 e-幂（对数增长率）本身无法为检验上鞅的第二类错误提供任何有限时间保证，而真正控制第二类错误的是 e-变量的 Chernoff-Stein 指数。 |
| [^15] | [FedIncome: Federated Learning for Income Estimation in Digital Lending Under Data Sovereignty Constraints](https://arxiv.org/abs/2609.27654) | FedIncome提出了一种联邦学习框架，使放贷机构无需共享原始借款人数据即可协同训练收入估计模型，在保障数据主权的同时，为小样本机构带来了显著的预测性能提升。 |
| [^16] | [Robustness of Diffusion Models under Distribution Shift](https://arxiv.org/abs/2609.27546) | 本文首次从理论上刻画了分布偏移下扩散模型的鲁棒分数估计，证明其可分解为学习参考分布的统计代价与随Wasserstein半径二次增长且极小极大最优的偏移代价，并构造了无需知晓偏移半径即可达到最优鲁棒速率的有限样本估计器。 |
| [^17] | [Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following](https://arxiv.org/abs/2609.27421) | 提出CC-OPD方法，颠覆传统蒸馏的监督方向，通过从教师模型条件中依次消融各约束并利用逐词元概率差分构建每约束的监督信号，从而提升大语言模型的多约束指令遵循能力。 |
| [^18] | [Discrete Diffusion Models via Evolving Variational Autoregressive Networks](https://arxiv.org/abs/2609.27306) | 提出一种利用变分自回归网络参数化归一化概率分布的离散扩散模型，通过显式马尔可夫跳跃算子控制加噪与去噪动力学，将归一化离散扩散模型成功扩展至高维晶格上的自旋系统，并准确计算了二维和三维伊辛模型的自由能、能量、磁化强度等热力学量。 |
| [^19] | [Multitask Regression with Pairwise Fusion](https://arxiv.org/abs/2609.27280) | 该论文提出一种通过对跨任务所有成对系数差异进行惩罚来估计多任务回归系数矩阵的方法，能够灵活刻画不同预测变量上任务间系数的共享与差异结构，并在活跃预测变量数和异常系数数这两个结构量上实现了匹配的上下界。 |
| [^20] | [Functional Causal Discovery via Conditional Covariance Ordering](https://arxiv.org/abs/2609.27256) | 该论文通过比较条件协方差算子的范数提出新的拓扑排序可识别性条件，摆脱了传统因果发现对结构（线性/非线性）和分布（高斯/非高斯）假设的依赖，并结合混合回归模型与变量选择实现了函数型变量因果DAG的估计及其渐近一致性。 |
| [^21] | [On the Sample Complexity of Active Learning with Membership Queries](https://arxiv.org/abs/2609.27241) | 本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。 |
| [^22] | [Prediction with Expert Advice: Anytime Regret with Many Experts Matches the Fixed-Time Constant](https://arxiv.org/abs/2609.27206) | 本文提出一种无需预知时间视界的专家建议预测算法，使任意时刻的累积遗憾达到 $(1+O(\sqrt{\ln\ln n/\ln n}))\sqrt{t\ln n/2}$，消除了此前的 $\sqrt{2}$ 因子差距，从而在多专家情形下将任意时刻遗憾匹配到固定时限的最优常数。 |
| [^23] | [Artificial intelligence surrogates for treatment effect estimation with before-and-after data](https://arxiv.org/abs/2609.27180) | 该论文提出一个新框架，利用预训练AI模型对每位患者治疗前后的测量数据进行结局预测并比较个体内差异，从而以AI预测作为低成本替代指标来估计治疗的因果效应。 |
| [^24] | [Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods](https://arxiv.org/abs/2609.27179) | 本文建立了非可交换数据下共形p值行为的系统理论，证明现有共形鞅变点检测方法在PFA和ARL控制下是次优的（检测延迟可达$\Omega(T)$和$\Omega(\sqrt{\text{ARL}})$），并提出了可证明极小化极大最优的新型共形e-过程和e-检测器。 |
| [^25] | [The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems](https://arxiv.org/abs/2609.27155) | 该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。 |
| [^26] | [WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps](https://arxiv.org/abs/2609.27033) | 提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。 |
| [^27] | [Tight Regret Bound for Online Inverse Linear Optimization via Multiscale Matrix Weights](https://arxiv.org/abs/2609.26978) | 该论文提出了一种基于多尺度矩阵乘法权重的随机化算法，实现了在线逆线性优化中O(√d)的期望遗憾界，达到了理论最优水平。 |
| [^28] | [Rolling Conformal Prediction in Sequential Model Training](https://arxiv.org/abs/2609.26951) | 本文提出滚动保形预测，一种无需数据划分的无分布预测推断方法，能够为序贯模型训练过程中的预测提供边际覆盖率保证。 |
| [^29] | [Additive Nonparametric Regression with Spatial and Network Objects](https://arxiv.org/abs/2609.26867) | 本文提出了一种将任务态和结构MRI图像视为函数型数据的新型可加非参数回归框架，通过高斯过程联合先验同时捕捉空间与网络预测变量的结构及其相互联系，从而从结构MRI和静息态fMRI预测脑激活图并量化预测不确定性。 |
| [^30] | [Gaussian-process surrogate indicators for residual-based adaptive GMsFEM](https://arxiv.org/abs/2609.26843) | 本文提出一种非侵入式的高斯过程代理模型来加速基于残差的自适应GMsFEM中反复的指标评估，在不改变多尺度求解和基函数加密的前提下用KRR等价形式预测指标分数，并通过扰动标记理论量化了分数误差对指标质量捕获的影响。 |
| [^31] | [On Basis Function Selection for Sparse Gaussian Process Regression](https://arxiv.org/abs/2609.26624) | 本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。 |
| [^32] | [Locally Private Inference for Riemannian Stochastic Optimization](https://arxiv.org/abs/2609.22642) | 该论文提出了一种在局部差分隐私下对流形值总体极小值点进行统计推断的方法，通过条件中心化的随机切梯度保持一阶方程，并引入对称对回归（SPR）从相同私有消息中估计渐近方差，进而证明了中心极限定理及基于交互记录的三明治协方差和内在Wald区域的一致性。 |
| [^33] | [Riemannian Simultaneous Inference for Tangent Vector Field Regression](https://arxiv.org/abs/2609.21910) | 该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。 |
| [^34] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^35] | [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895) | TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。 |
| [^36] | [Energy-guided Recursive Model](https://arxiv.org/abs/2607.10128) | 提出能量引导递归模型（ERM），利用Hopfield型记忆为候选轨迹分配内在能量，从而有原则地指导轨迹选择与递归深度的确定，在数独、铅笔谜题和迷宫等推理任务上取得递归建模领域的最佳准确率，并降低了语言建模的困惑度。 |
| [^37] | [Simultaneous Latent Budget Trees for Stratified Classification](https://arxiv.org/abs/2606.13295) | 本文提出同时潜在预算树，一种面向含时间、空间或人口等分层因素的场景的分类树概率机器学习框架，通过将子节点解释为同时混合模型的潜在成分来构建基于模型的条件分裂规则。 |
| [^38] | [A lift for input-convex neural net training](https://arxiv.org/abs/2605.24274) | 针对输入凸神经网络训练中softplus参数化导致负权重区域梯度指数衰减、逃逸缓慢的问题，提出用可学习松弛量加无约束网络（以批次置换不变摘要为输入）替换自由潜在权重的“提升”方法。 |
| [^39] | [ProteinJEPA: Latent prediction improves protein language model pretraining](https://arxiv.org/abs/2605.07554) | ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。 |
| [^40] | [The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices](https://arxiv.org/abs/2603.18482) | 该论文提出“截断盲区”概念，揭示 top-k 和核采样等解码策略因截断低概率词元而系统性地排除了 8–18% 的人类典型选词，从而为机器生成文本为何始终可被检测提供了机制性解释。 |
| [^41] | [Regular Fourier Features for Nonstationary Gaussian Processes](https://arxiv.org/abs/2602.23006) | 该论文提出正则傅里叶特征方法，通过直接离散化可调和非平稳高斯过程的谱表示，摆脱了谱密度必须为概率测度的限制性假设，实现了无需概率假设、结构上半正定的高效低秩近似。 |
| [^42] | [Learning to Approximate Uniform Facility Location via Graph Neural Networks](https://arxiv.org/abs/2602.13155) | 提出了一种融合近似算法原理的完全可微分消息传递神经网络，用于求解均匀设施选址问题，既具有可证明的近似保证，又在实证中优于标准近似算法并缩小了与整数线性规划的差距。 |
| [^43] | [Advances in Diffusion-Based Generative Compression](https://arxiv.org/abs/2601.18932) | 本文系统综述了基于扩散模型的生成式有损压缩最新进展，重点介绍了图像压缩中通过嵌入表示编码并利用扩散模型迭代细化、从而在极低码率下实现逼真重建的方法。 |
| [^44] | [BOCO: Bayesian Online Contextual Optimization for Decision-Focused Online Learning](https://arxiv.org/abs/2511.20413) | 提出了BOCO框架，通过维护决策聚焦的贝叶斯后验分布并聚合多个预测来考虑参数不确定性，同时开发了基于粒子的在线推断算法，使在线决策聚焦学习在有限数据下更稳定且更易推广到异构优化问题。 |
| [^45] | [InsurTech innovation using natural language processing](https://arxiv.org/abs/2507.21112) | 本文展示了如何运用自然语言处理技术将非结构化文本转化为结构化数据，通过特征去偏、特征压缩和行业分类来丰富商业保险定价的费率因子，并为评估潜在风险提供新视角。 |
| [^46] | [Role of scrambling and noise in temporal information processing with quantum systems](https://arxiv.org/abs/2505.10080) | 本文揭示了基于高阶幺正设计的加扰量子储层在时间信息处理中的关键特性：无噪声时测量读出集中度不随迭代恶化、小储层可反复复用，但扩大规模需指数级测量开销否则损害泛化，且早期输入记忆随储层规模与迭代次数均呈指数衰减。 |
| [^47] | [Localized Diffusion Models](https://arxiv.org/abs/2505.04417) | 提出局部化扩散模型，通过利用目标分布中的局部性结构（稀疏条件依赖），以局部化神经网络估计得分函数，从而规避维数灾难并显著降低样本复杂度。 |
| [^48] | [Statistical Properties of Deep Neural Networks with Dependent Data](https://arxiv.org/abs/2410.11113) | 该论文为非平稳β-混合依赖数据下的深度神经网络估计量建立了非渐近误差理论，覆盖全连接与卷积网络且无需对权重施加有界或稀疏约束，并推广至非参数回归、逻辑回归和分位数回归等场景。 |
| [^49] | [FastManly: An EM-Gradient Algorithm for Manly Mixture Models](https://arxiv.org/abs/2410.00848) | FastManly方法通过在EM梯度算法中采用牛顿法（并推导出梯度和完整Hessian矩阵）替代传统EM中的Nelder-Mead优化，显著加快了Manly变换混合模型的计算速度。 |
| [^50] | [Outlier Detection for Multi-Network Data](https://arxiv.org/abs/2205.06398) | 该论文针对节点共享而边各异的多网络数据（如神经影像大脑网络）提出了离群点检测方法，用于识别因数据质量差而产生的异常网络，防止其作为影响点污染后续统计分析。 |
| [^51] | [Random Polytope Descriptors](https://arxiv.org/abs/2009.13987) | 该论文提出了一类既通用又计算友好的随机多面体描述符，可用于数据分析中的分类与聚类任务，并允许用户在数据描述的紧致性与计算速度之间灵活权衡。 |

# 详细

[^1]: 面向广义朗之万动力学的记忆条件扩散模型

    Memory-Conditioned Diffusion Model for Generalized Langevin Dynamics

    [https://arxiv.org/abs/2609.28371](https://arxiv.org/abs/2609.28371)

    该论文提出一种记忆条件扩散模型，利用递归更新的指数滤波器组从观测轨迹中学习广义朗之万动力学的随机流映射，无需识别记忆核或重建未解析变量，且记忆成本与历史长度无关。

    

    广义朗之万方程描述了一类非马尔可夫动力学，其中已解析变量的演化依赖于其历史信息。我们提出了一种记忆条件扩散方法，用于从观测轨迹中学习这类动力学的随机流映射，而无需识别记忆核或重建未解析变量。一个紧凑的、递归更新的指数滤波器组使流映射能够在多个时间尺度上保留预测历史，而无需以长的观测窗口作为条件。下一步的分布以当前观测值和该记忆状态为条件，在滤波器组规模固定的情况下，其存储与更新成本与历史长度无关。预测准则用于指导记忆预算的分配，在向量基准测试中采用了参考辅助选择方法，可选的线性投影进一步降低了条件化维度。基于核的分数估计器生成条件样本……

    arXiv:2609.28371v1 Announce Type: new  Abstract: Generalized Langevin equations describe non-Markovian dynamics in which the evolution of resolved variables depends on their past. We propose a memory-conditioned diffusion method for learning stochastic flow maps of these dynamics from observed trajectories, without identifying a memory kernel or reconstructing unresolved variables. A compact, recursively updated bank of exponential filters enables the flow map to retain predictive history over multiple time scales without conditioning on long observation windows. The next-step distribution is conditioned on the current observation and this memory state, whose storage and update costs are independent of the history length for a fixed bank size. Predictive criteria guide the memory budget, with reference-assisted selection in the vector benchmark, and an optional linear projection further reduces the conditioning dimension. A kernel-based score estimator generates conditional samples wit
    
[^2]: 通过Dobrushin收缩实现局部几何混合及其在扩散路径蒙特卡洛与近端采样器中的应用

    Local Geometric Mixing via Dobrushin Contraction with Applications to Diffusion Path Monte Carlo and the Proximal Sampler

    [https://arxiv.org/abs/2609.28338](https://arxiv.org/abs/2609.28338)

    本文提出基于Dobrushin收缩的局部几何混合分析框架，并将其应用于扩散路径蒙特卡洛和近端采样器，在最少假设下为理想方法及Metropolis校正版本提供了混合时间保证。

    

    局部几何混合通过仅在有限多次转移中要求在总变差距离下几何收敛到平衡态，从而将几何混合局部化。它能够容纳局部收敛速率，并刻画快速的局部均衡化现象，即使全局混合慢得多。我们通过Dobrushin收缩建立并讨论了局部几何混合的界。随后，我们将该方法应用于扩散路径蒙特卡洛——这是一种最近提出的马尔可夫链蒙特卡洛方法，旨在利用基于分数（score-based）建模的最新进展，其理想转移与近端采样器的转移相一致。我们的分析同时覆盖了理想方法及其可实现的经过Metropolis校正的对应版本，并在最少的假设下提供了混合保证。对于理想方法，这些保证补充了最近的谱隙估计结果，我们进一步将其发展为混合时间界。

    arXiv:2609.28338v1 Announce Type: cross  Abstract: Local geometric mixing localizes geometric mixing by requiring geometric convergence to equilibrium in total variation only over finitely many transitions. It accommodates local convergence rates and captures rapid local equilibration, even when global mixing is much slower. We establish and discuss local geometric mixing bounds through Dobrushin contraction. We then apply this approach to Diffusion Path Monte Carlo, a recently proposed Markov chain Monte Carlo method, aimed at leveraging advances in score-based modeling, whose ideal transitions coincide with those of the Proximal Sampler. Our analysis covers both the ideal method and its implementable Metropolis-adjusted counterpart, providing mixing guarantees under minimal assumptions. For the ideal method, these guarantees complement recent spectral gap estimates, which we develop into mixing time bounds.
    
[^3]: LLM 排行榜声明对隐藏模型选择有多敏感？

    How Sensitive Are LLM Leaderboard Claims to Hidden Model Selection?

    [https://arxiv.org/abs/2609.28177](https://arxiv.org/abs/2609.28177)

    该论文提出一种敏感性分析方法，量化一个排行榜领先幅度背后可能隐藏的私下挑选的模型变体数量，并对 Open LLM Leaderboard 上 394 个相邻排名声明进行审计，发现其中 391 个即使不考虑选择效应也缺乏统计支持。

    

    LLM 排行榜上的领先可能反映了在私下评估的多个模型变体之间进行挑选的结果，然而变体的数量及其相关性均未公开。我们探究的问题是：在保持对某个固定比较对象具有统计优势证据的前提下，一个已公布的领先幅度最多能支持多少个隐藏变体。对于一个固定候选模型族，在高斯边际模型下，我们推导出一条敏感性曲线，将该最大数量表示为族内相关性下界的函数。相关的相关性必须与用于排名的分数和抽样模型相匹配：在一个受控模型族中，合并题目层面的相关性为 0.90，而在题目重抽样下综合分数的相关性为 0.46，在 MMLU 学科重抽样下则为 0.92。对 Open LLM Leaderboard 上 394 个相邻排名声明的基于题目的审计发现，即使在未考虑选择效应之前，其中 391 个声明就已缺乏统计支持。在通过未校正检验的声明中，经认证（原文此处截断）……

    arXiv:2609.28177v1 Announce Type: cross  Abstract: LLM leaderboard gains can reflect selection among privately evaluated model variants, yet neither the number of variants nor their dependence is public. We ask how many hidden variants a published margin can support while retaining statistical evidence of a provider's advantage over a fixed comparator. For a fixed candidate family under a Gaussian margin model, we derive a sensitivity curve that reports this maximum count as a function of a lower bound on within-family correlation. The relevant correlation must match the score used for ranking and the sampling model: in a controlled family, pooled item correlation is 0.90, whereas composite-score correlation is 0.46 under item resampling and 0.92 when MMLU subjects are resampled. An item-based audit of 394 adjacent-rank claims on the Open LLM Leaderboard finds that 391 lack statistical support even before accounting for selection. Among claims that pass the uncorrected test, certificat
    
[^4]: 稀疏 Wishart 噪声中的秩一信号恢复

    Rank-One Signal Recovery in Sparse Wishart Noise

    [https://arxiv.org/abs/2609.28163](https://arxiv.org/abs/2609.28163)

    本文利用复制方法解析计算了秩一信号形变的稀疏 Wishart 噪声矩阵的最大特征值对统计特性，通过递归分布方程组与种群动力学算法高效求解，刻画了信号恢复性能对信号强度、矩阵矩形比和噪声连通度的依赖关系。

    

    我们研究了在稀疏类 Wishart 噪声存在下信号向量 $\mathbf{x}$ 的高维恢复问题。我们定义了一个 $N \times N$ 矩阵 $A = J+(\theta/N)\mathbf{xx}^{\top}$，其中 $\mathbf{xx}^{\top}$ 是对随机噪声矩阵 $J$ 的秩一形变。我们考虑类 Wishart 矩阵 $J={X}^{\top} X$，其中 $X$ 是一个稀疏的 $M \times N$ 随机矩阵，其元素为 $X_{ij} = c_{ij}W_{ij}$，其中 $c_{ij}$ 调节非零元素的密度，$W_{ij}$ 为键权重。利用复制方法，我们解析地计算了 $A$ 的最大特征值对的统计量，及其对信号强度 $\theta$、矩形比 $\alpha=\sqrt{M/N}$ 以及噪声平均连通度的依赖关系。这些谱观测量由一组递归分布方程表示，可通过种群动力学算法高效求解。它们使我们能够计算平均最大特征值（摘要在此处截断）。

    arXiv:2609.28163v1 Announce Type: cross  Abstract: We study the high-dimensional recovery of a signal vector $\mathbf{x}$ in the presence of sparse Wishart-like noise. We define an $N \times N$ matrix $A = J+(\theta/N)\mathbf{xx}^{\top}$, where $\mathbf{xx}^{\top}$ is the rank-one deformation of the random noise matrix $J$. We consider a Wishart-like matrix $J={X}^{\top} X$, where $X$ is a sparse $M \times N$ random matrix with entries $X_{ij} = c_{ij}W_{ij}$, with $c_{ij}$ regulating the density of non-zero elements, and $W_{ij}$ the bond weights. Using the replica method, we compute analytically the top eigenpair statistics of $A$, and their dependence on the signal strength $\theta$, the rectangularity ratio $\alpha=\sqrt{M/N}$, and the average connectivity of the noise. The spectral observables are expressed in terms of a system of Recursive Distributional Equations, which are efficiently solved via a Population Dynamics algorithm. They allow us to compute the average largest eigen
    
[^5]: NPBoost：基于梯度提升固定效应的神经过程

    NPBoost: Neural Processes with Gradient-Boosted Fixed Effects

    [https://arxiv.org/abs/2609.28122](https://arxiv.org/abs/2609.28122)

    NPBoost将结构化响应变异性分解为跨任务共享的梯度提升树固定效应与捕捉任务间随机变化的神经过程随机效应，并通过提升算法联合训练二者，在共享结构包含不连续性等不规则特征的表格元学习任务上优于标准神经过程。

    

    神经过程（Neural Processes, NPs）是一类基于模型的元学习器，能够隐式地学习一个随机过程，并从小型上下文集出发适应新任务。大多数神经过程的扩展工作都聚焦于改进神经网络架构。与此不同，我们从元学习与混合效应模型的共同层次化解释出发，开发了一种新的扩展方法。具体而言，我们提出了神经过程提升（Neural Process Boosting, NPBoost），它将结构化的响应变异性分解为跨任务共享的树提升固定效应，以及捕捉任务间随机变化的神经过程随机效应。我们提出使用一种提升算法联合训练这两个组件，其中神经过程学习残差的任务特定结构，树集成模型则估计跨任务的共同模式。在合成数据和真实世界的表格元学习问题上，实验表明当共享结构包含不连续性或其他不规则特征时，这种分解方法优于标准神经过程。

    arXiv:2609.28122v1 Announce Type: cross  Abstract: Neural Processes (NPs) are model-based meta-learners that implicitly learn a stochastic process and adapt to a new task from a small context set. Most extensions of NPs focus on improving the neural network architecture. We instead develop an extension motivated by the shared hierarchical interpretation of meta-learning and mixed-effects models. Specifically, we introduce Neural Process Boosting (NPBoost), which decomposes structured response variability into tree-boosted fixed effects shared across tasks and NP random effects that capture stochastic task-to-task variation. We propose to train the two components jointly using a boosting algorithm in which an NP learns residual task-specific structure and a tree ensemble estimates common patterns across tasks. Across synthetic and real-world tabular meta-learning problems, this decomposition improves over a standard NP when the shared structure contains discontinuities or other irregula
    
[^6]: 基于流匹配改进的集合滤波器

    Improving Ensemble Filters with Flow Matching

    [https://arxiv.org/abs/2609.28015](https://arxiv.org/abs/2609.28015)

    提出流集合滤波器，利用条件流匹配学习非线性更新，将经典基线滤波器的预报集合传输为分析集合，在稀疏观测的动力学系统中性能超越所有经典集合滤波器及最先进的生成式方法。

    

    数据同化旨在从部分且含噪声的观测中估计动力学状态。经典的集合滤波器虽然高效，但其分析更新受限于有限样本协方差和仿射高斯分布。我们提出了流集合滤波器，它利用条件流匹配将预报集合从经典基线滤波器传输到分析集合。FlowEF 在训练时使用局部化高斯源，在部署时从基线滤波器传输预报集合成员，并将其速度场条件于该基线滤波器的集合以及观测之上。因此，所提出的模型在学习非线性更新的同时，能够独立地映射每个基线集合成员。对于稀疏观测的动力学系统，FlowEF 在确定性和概率性指标上均优于全部四种经典集合滤波器，并且在最先进的生成式模型中取得了最佳性能。

    arXiv:2609.28015v1 Announce Type: cross  Abstract: Data assimilation estimates a dynamical state from partial and noisy observations. Classical ensemble filters are efficient but restrict analysis updates through finite sample covariance and affine Gaussian distribution. We introduce the Flow Ensemble Filter (FlowEF), which uses conditional flow matching to transport the forecast ensemble from a classical baseline filter to an analysis ensemble. FlowEF uses a localized Gaussian source during training, transports forecast ensemble members from a baseline filter at deployment, and conditions its velocity field on ensembles from that baseline filter and the observation. The proposed model therefore learns a nonlinear update while mapping each baseline ensemble independently. For sparsely observed dynamical systems, FlowEF improves both deterministic and probabilistic metrics over all four classical ensemble filters. It also achieves the best performance among the state-of-the-art generati
    
[^7]: 连续标签偏移下的共形贝叶斯：敏感性分析与精确有效性的局限

    Conformal Bayes under Continuous Label Shift: Sensitivity Analysis and the Limits of Exact Validity

    [https://arxiv.org/abs/2609.27976](https://arxiv.org/abs/2609.27976)

    该论文提出联合倾斜敏感性共形贝叶斯（JTS-CB/JTS-SCB），通过对预设的合理倾斜集合进行联合敏感性分析来应对连续标签偏移，同时揭示了精确有限样本有效性取决于密度比尾部行为的根本局限。

    

    共形贝叶斯将贝叶斯后验预测分数与共形校准相结合，但在连续标签偏移下，分数和校准权重都依赖于未知的响应边缘密度比。现有方法通常从伪标签或预测样本中估计单个偏移参数并将其代入校准。与之不同，我们提出了联合倾斜敏感性共形贝叶斯，它在预先指定的合理倾斜集合上进行敏感性分析；其分割共形实现为 JTS-SCB。每个倾斜共同决定贝叶斯共形分数和共形重要性权重。JTS-SCB 在候选倾斜上形成一个有界的敏感性包络，但其仅进行校准的构造并不能继承精确的有限样本加权共形保证。因此，我们研究了一个单独的候选加权精确对应方法，并证明其有用性在很大程度上取决于尾部行为。对于标量线性……（原文摘要在此处不完整）

    arXiv:2609.27976v1 Announce Type: cross  Abstract: Conformal Bayes combines Bayesian posterior predictive scores with conformal calibration, but under continuous label shift both the score and calibration weight depend on the unknown response-marginal density ratio. Existing methods typically estimate one shift parameter from pseudo-labels or predictive samples and plug it into calibration. We instead propose Joint Tilt-Sensitivity Conformal Bayes (JTS-CB), which performs sensitivity analysis over a prespecified set of plausible tilts; its split-conformal realization is JTS-SCB. Each tilt jointly determines the Bayesian conformal score and conformal importance weight. JTS-SCB forms a bounded sensitivity envelope over candidate tilts, but its calibration-only construction does not inherit the exact finite-sample weighted-conformal guarantee. We therefore study a separate candidate-weighted exact counterpart and show that its usefulness depends sharply on tail behavior. For scalar linear
    
[^8]: 有限型转置相关随机矩阵的四阶矩强普适性

    Fourth-Moment Strong Universality for Finite-Type Transpose-Correlated Random Matrices

    [https://arxiv.org/abs/2609.27971](https://arxiv.org/abs/2609.27971)

    本文在精确的四阶矩阈值处证明了有限型转置相关的非厄米随机矩阵族在矩阵放大和非交换 *-多项式下强收敛于显式协方差匹配的自由高斯族的强普适性定理。

    

    我们为有限个由独立无序对向量原子构成的非厄米随机矩阵族，在精确的四阶矩阈值处证明了一个强普适性定理。在单个原子内部，矩阵颜色与两个端点方向可以具有任意的联合实协方差，只需满足有序类型对之间的反转一致性以及同类型块中的端点可交换性；其分布还可以依赖于有限多个端点类型。这些矩阵可以与任意一个与类型投影强联合收敛的确定性元组相连接。对于每个固定的矩阵放大以及固定的非交换 *-多项式，所得元组强收敛到一个显式的协方差匹配自由高斯族。跨类型块由掩蔽圆变量描述，而同类型块则分裂为相互独立的端点对称和端点反对称半圆扇区。只需有限……

    arXiv:2609.27971v1 Announce Type: cross  Abstract: We prove a strong-universality theorem at the exact fourth-moment threshold for finite families of non-Hermitian random matrices assembled from independent unordered-pair vector atoms. Within one atom, the matrix colors and the two endpoint orientations may have arbitrary joint real covariance, subject to reversal consistency across ordered type pairs and endpoint exchangeability in same-type blocks; the law may also depend on finitely many endpoint types. The matrices may be adjoined to an arbitrary deterministic tuple that converges jointly strongly with the type projections. For every fixed matrix amplification and fixed noncommutative \(*\)-polynomial, the resulting tuple converges strongly to an explicit covariance-matched free Gaussian family. Cross-type blocks are described by masked circular variables, whereas same-type blocks split into independent endpoint-symmetric and endpoint-antisymmetric semicircular sectors. Only a fini
    
[^9]: 具有高斯过程分裂的狄利克雷过程树混合：一个具有后验收缩速率的贝叶斯非参数框架

    Dirichlet Process Mixtures of Trees with Gaussian Process Splits: A Bayesian Nonparametric Framework with Posterior Contraction Rate

    [https://arxiv.org/abs/2609.27930](https://arxiv.org/abs/2609.27930)

    该论文提出了一种基于狄利克雷过程先验和由高斯过程后验预测驱动分裂规则的贝叶斯非参数回归树混合框架，统一了CART、BART、随机森林和提升方法，并证明了在真实回归函数仅连续的宽松条件下Hellinger距离上 $n^{-1/4}$ 的后验收缩速率。

    

    我们提出了一种贝叶斯非参数回归树混合模型，对树-参数对施加狄利克雷过程先验，从而实现集成规模的数据驱动选择，并统一了CART、BART、随机森林和提升方法。一种新颖的分裂规则由每个终端节点内高斯过程的后验预测驱动，可生成灵活、平滑的决策边界；值得注意的是，高斯过程密度在GROW/PRUNE移动的Metropolis-Hastings比率中恰好完全相消，从而保证了计算可行性。我们提供了一个用于后验预测推断的精确吉布斯采样器，通过随机树遍历来传播不确定性。一个并行MPI实现将独立的树更新分配到多个处理器上，实现了可观的速度提升。在仅要求真实回归函数连续的条件下（允许模型误设），我们通过恒等式 $h(\Theta)=0$ 证明了在Hellinger距离下以 $n^{-1/4}$ 速率的后验一致性。在Friedman数据集上的模拟实验……

    arXiv:2609.27930v1 Announce Type: cross  Abstract: We propose a Bayesian nonparametric mixture of regression trees with a Dirichlet process prior over tree-parameter pairs, enabling data-driven selection of ensemble size and unifying CART, BART, random forests, and boosting. A novel splitting rule driven by the posterior predictive of a Gaussian process within each terminal node generates flexible, smooth decision boundaries; remarkably, the GP density cancels exactly in the Metropolis--Hastings ratio for GROW/PRUNE moves, ensuring computational feasibility. An exact Gibbs sampler for posterior predictive inference propagates uncertainty through random tree traversal. A parallel MPI implementation distributes independent tree updates across processors, achieving adequate speedups. We prove posterior consistency at rate $n^{-1/4}$ in Hellinger distance under only continuity of the true regression function, allowing misspecification, via the identity $h(\Theta)=0$. Simulations on Friedma
    
[^10]: 基于证据学习的变分自编码器的理论研究

    Theoretical Study on the Evidential Learning-based Variational Autoencoder

    [https://arxiv.org/abs/2609.27853](https://arxiv.org/abs/2609.27853)

    该论文从理论上证明，证据学习变分自编码器中正态-逆伽马潜在层级的四个参数仅有三维商空间 (γ, α, c) 是可辨识的，并且通过对前向KL散度的精确偏最小化，可以实现从四参数到三坐标的精确约简，同时保持最优值不变。

    

    正态-逆伽马（NIG）潜在层级结构包含四个参数，但其诱导的潜在分布并不能唯一确定全部四个参数。对于 σ²~InvGamma(α,β)，μ|σ²~N(γ, σ²/ν)，z|μ,σ²~N(μ, σ²)，z 的边际分布仅通过 c=β(1+1/ν) 依赖于 (ν, β)。因此，重构可见的参数空间是三维商空间 (γ, α, c)，并带有一维的纤维自由度。对于固定的层级变分目标，对前向KL散度到一个完整NIG先验的精确偏最小化，会在每条纤维上选择出一个唯一的先验相对代表元，从而实现精确的三坐标约简，且其最优值与四坐标目标完全相同。记 ρ₀=2β₀/ν₀，T=c/{α[(γ-γ₀)²+ρ₀]}，我们证明逆规范分配 1/ν_can……（原摘要至此截断）

    arXiv:2609.27853v1 Announce Type: cross  Abstract: A normal--inverse-gamma (NIG) latent hierarchy has four parameters, but its induced latent law does not identify all four. For $\sigma^2\sim\mathrm{InvGamma}(\alpha,\beta)$, $\mu\mid\sigma^2\sim\mathcal{N}(\gamma,\sigma^2/\nu)$, and $z\mid\mu,\sigma^2\sim\mathcal{N}(\mu,\sigma^2)$, the marginal law of $z$ depends on $(\nu,\beta)$ only through $c=\beta(1+1/\nu)$. Hence the reconstruction-visible parameter space is the three-dimensional quotient $(\gamma,\alpha,c)$, with a one-dimensional fiber degree of freedom. For a fixed hierarchical variational objective, exact partial minimization of the forward KL divergence to a complete NIG prior selects a unique prior-relative representative on each fiber, yielding an exact three-coordinate reduction with the same optimum as the four-coordinate objective. Writing $\rho_0=2\beta_0/\nu_0$ and $T=c/\{\alpha[(\gamma-\gamma_0)^2+\rho_0]\}$, we show that inverse canonical allocation $1/\nu_{\rm can}$
    
[^11]: 滑动窗口占据计数谱隙的最优状态空间阶

    Optimal State-Space Order for Spectral Gaps of Sliding-Window Occupation Counts

    [https://arxiv.org/abs/2609.27836](https://arxiv.org/abs/2609.27836)

    本文证明了滑动窗口占据计数所定义的投影计数核的谱隙与原马尔可夫核谱隙之比的最优常数 $c_m^\star$ 满足 $\Theta(1/m)$ 的阶，给出了普适下界 $1/(1080m)$ 和形如 $(m+\log m+O(1))^{-1}$ 的上界，但精确值仍待确定。

    

    设 $P$ 是定义在 $m$ 个状态的空间 $\Omega$ 上的不可约可逆马尔可夫核，其右谱隙记为 $\gamma=1-\lambda_2(P)$。基于平稳轨迹，令 $K_t$ 为从时刻 $t$ 开始的长度为 $n$ 的窗口所对应的占据计数向量。平稳对 $(K_0,K_1)$ 定义了一个可逆的投影计数核 $\widetilde P_n$。对每个 $m\ge2$，定义 $c_m^\star=\inf_{P,\;n\ge2}\frac{n\Gap(\widetilde P_n)}{\Gap(P)}$。我们证明 $\frac{1}{1080m}\le c_m^\star\le q_{m-2}$，其中 $q_0=\frac14$，$q_{r+1}=q_r(1-q_r)$。我们还证明 $q_{m-2}=(m+\log m+O(1))^{-1}$，从而推出 $c_m^\star=\Theta(m^{-1})$。因此，最优比较系数的阶为 $m$，但其精确值仍是未决问题。该下界在 $n=1$ 时同样成立，且对 $P$ 一致成立，包括稀疏核与周期核。其证明将短窗口去相关与平均锚定-游走分解相结合……

    arXiv:2609.27836v1 Announce Type: cross  Abstract: Let $P$ be an irreducible reversible Markov kernel on a $m$-state space $\Omega$, and denote its right spectral gap $\gamma=1-\lambda_2(P)$. From a stationary trajectory, let $K_t$ be the occupation-count vector of the length-$n$ window beginning at time $t$. The stationary pair $(K_0,K_1)$ defines a reversible projected count kernel $\widetilde P_n$. For every $m\ge2$, let \[   c_m^\star=   \inf_{\substack{ P,\; n \ge 2}}   \frac{n\Gap(\widetilde P_n)}{\Gap(P)}. \] We prove \[   \frac1{1080m}\le c_m^\star\le q_{m-2},   \qquad   q_0=\frac14,\quad q_{r+1}=q_r(1-q_r). \] We also show $q_{m-2}=(m+\log m+O(1))^{-1}$, which implies that $c_m^\star=\Theta(m^{-1})$. Thus, the optimal comparison coefficient has order $m$, although its exact value remains open. The lower bound also holds for $n=1$ and is uniform in $P$, including sparse and periodic kernels. Its proof combines short-window decorrelation with an averaged anchor-excursion decompo
    
[^12]: 基于半离散最优传输的超越Lipschitz连续性约束的金融尾部风险

    Financial Tail Risk Beyond Lipschitz Continuity via Semi-Discrete Optimal Transport

    [https://arxiv.org/abs/2609.27785](https://arxiv.org/abs/2609.27785)

    该论文证明了Lipschitz连续性约束使基于神经生成器的采样方法在数学上无法精确匹配厚尾的金融收益分布，并提出用半离散最优传输（SDOT）放松映射正则性来突破这一限制，从而实现更准确的金融尾部风险估计。

    

    金融收益呈厚尾分布，准确的尾部风险估计是投资组合风险管理的核心。现代神经生成器通过将一个简单的基准分布经过学习到的映射来生成样本，而为了保证训练稳定性，该映射通常由Lipschitz组件构建。这正是关键约束所在：高斯分布经过Lipschitz映射后仍是亚高斯分布，因此对于更厚尾的目标分布，任何有限的Lipschitz常数都无法实现精确匹配。Monge–Ampère方程将Brenier映射的局部畸变与密度比 f/(g∘T) 联系起来，因此目标密度中更深的低谷需要更高增益的映射，从而产生更高方差的估计器。这一论证仅需要有界畸变，因此同样适用于归一化流、流匹配、生成对抗网络和扩散采样器。半离散最优传输（SDOT）放松的是映射的正则性，而非源分布的尾部类别。其幂图划分为每个训练观测分配一个单元（摘要在此处被截断）。

    arXiv:2609.27785v1 Announce Type: new  Abstract: Financial returns are heavy-tailed, and accurate tail risk estimation is central to portfolio risk management. Modern neural generators sample by pushing a simple base distribution through a learned map, and for training stability that map is built from Lipschitz components. This is the binding constraint: a Lipschitz map of a Gaussian is sub-Gaussian, so heavier-tailed targets admit no exact match at any finite Lipschitz constant. The Monge--Amp\`ere equation ties the Brenier map's local distortion to the density ratio $f/(g\circ T)$, so a deeper trough in the target density requires a higher-gain map and yields a higher-variance estimator. The argument needs only bounded distortion, so it covers normalizing flows, flow matching, GANs, and diffusion samplers alike.   Semi-Discrete Optimal Transport (SDOT) relaxes the map's regularity rather than the source's tail class. Its power diagram gives every training observation a cell holding e
    
[^13]: 基于下尾假设的检验上鞅的第二类错误界

    Type-II Error Bounds for Test Supermartingales from Lower-Tail Hypotheses

    [https://arxiv.org/abs/2609.27766](https://arxiv.org/abs/2609.27766)

    本文针对检验上鞅方法中的第二类错误问题，研究了对对数增量下尾概率的不同假设所导出的第二类错误界，并将所有结果统一为一个主不等式，即通过 e 变量（逆）矩生成函数的单侧勒让德变换来刻画序贯检验在固定时域和任意时刻的第二类错误上界。

    

    在使用检验上鞅的安全假设检验中，若当财富过程首次超过 1/α 时即拒绝原假设，则 Ville 不等式可以为每个显著性水平 α∈(0,1] 提供随时有效的第一类错误保证。由于固有的不对称性，第二类错误却没有这样的保证：概率在对数增量下尾上的高度集中可能导致一次灾难性的下注，从而抵消已积累的任何证据。本文研究了对这些下尾概率施加的不同假设如何导出序贯检验第二类错误的不同界。这些结果都可归结为一个主不等式，它在固定时域和序贯两种情形下，利用 e 变量的（逆）矩生成函数的单侧勒让德变换在某一个数值处的取值来界定水平 α 下的第二类错误，该数值即为下界超出……

    arXiv:2609.27766v1 Announce Type: cross  Abstract: In safe hypothesis testing with test supermartingals, Ville's inequality provides anytime-valid type-I error guarantees for every significance level $\alpha\in(0,1]$, if one rejects the null hypothesis whenever the wealth process first exceeds $\frac{1}{\alpha}$. Due to an inherent asymmetry, the type-II error does not have such guarantees: a heavy concentration of the probability on the lower tail of the log-increments can lead to one catastrophic bet that undoes any amount of accumulated evidence. This paper studies how different hypotheses on those lower-tail probabilities lead to different bounds on the type-II error of the sequential test. They all reduce to one master inequality, which bounds the type-II error at level $\alpha$, at a fixed horizon and sequentially, in terms of a one-sided Legendre transform of the (inverse-)moment generating function of the e-variables, evaluated at one number: the amount by which the lower bound
    
[^14]: 检验上鞅的第二类错误：e-幂与 Chernoff-Stein 指数

    The Type-II Error of Test Supermartingales: e-Power versus the Chernoff-Stein Exponent

    [https://arxiv.org/abs/2609.27765](https://arxiv.org/abs/2609.27765)

    该论文证明了 e-幂（对数增长率）本身无法为检验上鞅的第二类错误提供任何有限时间保证，而真正控制第二类错误的是 e-变量的 Chernoff-Stein 指数。

    

    在基于检验上鞅的安全假设检验中，只要当财富过程首次超过 1/α 时拒绝原假设，Ville 不等式就能为每个显著性水平 α∈(0,1] 提供任意时刻有效的第一类错误保证。由于一种内在的不对称性，第二类错误的表现有所不同。针对简单原假设与简单备择假设的情形，我们关于第二类错误证明了两点。第一，均值增长率 𝔼_{P₁}[log E]（即 e-幂，也就是 Kelly 赌注策略和增长率最优 e-变量所最大化的量）本身无法约束任何东西：对于每个水平 c>0、每个 α 和时间范围 t，我们都构造出条件 e-幂恰好为 c 的 e-变量，其在时刻 t 之前仍未拒绝原假设的概率可以任意接近于 1。这虽然能迫使最终拒绝，但无法由此得到任何有限时间范围的保证。第二，真正控制第二类错误的量是 e-变量的 Chernoff-Stein 指数，即 Λ(E)=sup_{s≥0}{…（摘要原文在此处截断）

    arXiv:2609.27765v1 Announce Type: cross  Abstract: In safe hypothesis testing with test supermartingales, Ville's inequality provides anytime-valid type-I error guarantees for every significance level $\alpha\in(0,1]$, if one rejects the null hypothesis whenever the wealth process first exceeds $1/\alpha$. Due to an inherent asymmetry, the type-II error behaves differently. We prove two things about the latter, for a simple null and alternative. First, the mean growth rate $\mathbb{E}_{P_1}[\log E]$, the e-power, that Kelly betting and growth-rate-optimal e-variables maximise, bounds nothing on its own. For every level $c>0$, every $\alpha$ and horizon $t$ we construct e-variables of conditional e-power exactly $c$ whose probability of not rejecting by $t$ is arbitrarily close to one. It forces eventual rejection, but no finite-horizon guarantee follows. Second, the quantity that does control the type-II error is the Chernoff-Stein exponent of an e-variable, $\Lambda(E)=\sup_{s\ge0}\{-
    
[^15]: FedIncome：数据主权约束下面向数字借贷收入估计的联邦学习

    FedIncome: Federated Learning for Income Estimation in Digital Lending Under Data Sovereignty Constraints

    [https://arxiv.org/abs/2609.27654](https://arxiv.org/abs/2609.27654)

    FedIncome提出了一种联邦学习框架，使放贷机构无需共享原始借款人数据即可协同训练收入估计模型，在保障数据主权的同时，为小样本机构带来了显著的预测性能提升。

    

    在数字贷款申请中，经过核实的收入信息往往无法获得，迫使放贷机构依赖借款人自行申报的收入，这可能导致过度放贷、贷款方案过于保守，或拒绝具备还款能力的申请人。跨机构数据共享的限制使得这一问题对训练数据有限的小型放贷机构尤为棘手。我们提出了FedIncome，一个用于收入估计的联邦学习框架，使各机构无需汇集原始借款人记录即可协同训练共享模型。我们使用超过一百万笔LendingClub贷款数据，将其划分为50个州级客户端，模拟了一个异构的放贷联盟。最优的联邦模型在时间外测试中达到R²=0.608，而集中式数据池基准为0.619。与集中式数据池基准相比，小样本客户端的时间外R²平均提升了3.8个百分点，且客户端层面的拟合……

    arXiv:2609.27654v1 Announce Type: cross  Abstract: Verified income is often unavailable in digital loan applications, forcing lenders to rely on reported income and potentially leading to over-lending, overly conservative offers, or rejection of creditworthy applicants. Cross-institutional data-sharing constraints make this problem especially difficult for smaller lenders with limited training data. We introduce FedIncome, a federated learning framework for income estimation that enables institutions to train a shared model without pooling raw borrower records. Using more than one million LendingClub loans partitioned into $50$ state-level clients, we simulate a heterogeneous lending consortium. The best federated model achieves out-of-time $R^2=0.608$, compared with $0.619$ for a pooled centralised benchmark. Small-sample clients obtain an average out-of-time $R^2$ improvement of $3.8$ percentage points relative to the pooled centralised benchmark, while the fitted client-level relati
    
[^16]: 扩散模型在分布偏移下的鲁棒性

    Robustness of Diffusion Models under Distribution Shift

    [https://arxiv.org/abs/2609.27546](https://arxiv.org/abs/2609.27546)

    本文首次从理论上刻画了分布偏移下扩散模型的鲁棒分数估计，证明其可分解为学习参考分布的统计代价与随Wasserstein半径二次增长且极小极大最优的偏移代价，并构造了无需知晓偏移半径即可达到最优鲁棒速率的有限样本估计器。

    

    基于分数的扩散模型越来越多地被应用于底层数据分布可能与训练分布不一致的场景，然而现有的理论保证大多集中在无分布偏移的设定下。在本工作中，我们研究了参考分布在 Wasserstein 扰动下的鲁棒分数估计问题。针对 Ornstein-Uhlenbeck 扩散，我们证明鲁棒估计可以分解为两个基本组成部分：学习参考分布的统计代价与分布偏移的内在代价。后者随 Wasserstein 半径呈二次方增长，且这种依赖关系是极小极大最优的。我们构造了一个显式的有限样本估计器，在不知道偏移半径的情况下即可达到相应的鲁棒极小极大速率。当参考分布位于未知的低维子空间上时，统计项能够自适应于内在维度，而偏移代价保持不变。

    arXiv:2609.27546v1 Announce Type: cross  Abstract: Score-based diffusion models are increasingly considered in settings where the underlying data distribution may differ from the training distribution, yet existing theoretical guarantees largely focus on the no-shift setting. In this work, we study robust score estimation under Wasserstein perturbations of a reference distribution. For the Ornstein--Uhlenbeck diffusion, we show that robust estimation decomposes into two fundamental components: the statistical cost of learning the reference distribution and the intrinsic cost of distribution shift. The latter scales quadratically with the Wasserstein radius, and this dependence is minimax optimal. We construct an explicit finite-sample estimator achieving the resulting robust minimax rate without knowing the shift radius. When the reference distribution lies on an unknown low-dimensional subspace, the statistical term adapts to the intrinsic dimension while the shift cost remains unchan
    
[^17]: 面向多约束指令遵循的反事实约束条件化在线策略蒸馏

    Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following

    [https://arxiv.org/abs/2609.27421](https://arxiv.org/abs/2609.27421)

    提出CC-OPD方法，颠覆传统蒸馏的监督方向，通过从教师模型条件中依次消融各约束并利用逐词元概率差分构建每约束的监督信号，从而提升大语言模型的多约束指令遵循能力。

    

    多约束指令遵循要求模型在多个同时生效的约束条件下对查询作出回应。即使是强大的指令微调模型，也经常违反其中一些约束。现有方法要么利用来自外部验证器或学习型评估器的序列级或词元级强化学习奖励来增强监督，要么使用针对单一全上下文教师模型的在线策略蒸馏（OPD），但随着同时生效的约束数量增多，该教师模型的概率质量会被稀释。我们提出了CC-OPD（反事实约束条件化在线策略蒸馏），该方法颠覆了蒸馏中标准的监督-生成方向。CC-OPD并非用学生模型看不到的信息来丰富教师模型，而是依次从教师模型的条件中消融各个约束，并从由此产生的逐词元概率差分中构建针对每个约束的信号。由此产生的逐词元留一法对数似然……

    arXiv:2609.27421v1 Announce Type: new  Abstract: Multi-constraint instruction following requires a model to respond to a query under many simultaneously active constraints. Even strong instruction-tuned models still routinely violate some of them. Existing approaches either augment supervision with sequence- or token-level RL rewards from external verifiers or learned graders, or use on-policy distillation (OPD) against a single full-context teacher whose probability mass becomes diluted as more constraints become simultaneously active. We propose CC-OPD (Counterfactual Constraint-Conditioned On-Policy Distillation), which inverts the standard supervision-generation direction in distillation. Rather than enriching the teacher with information beyond what the student sees, CC-OPD ablates each constraint from the teacher's conditioning in turn, and constructs the per-constraint signal from the resulting per-token probability differentials. The resulting per-token leave-one-out log-likeli
    
[^18]: 基于演化变分自回归网络的离散扩散模型

    Discrete Diffusion Models via Evolving Variational Autoregressive Networks

    [https://arxiv.org/abs/2609.27306](https://arxiv.org/abs/2609.27306)

    提出一种利用变分自回归网络参数化归一化概率分布的离散扩散模型，通过显式马尔可夫跳跃算子控制加噪与去噪动力学，将归一化离散扩散模型成功扩展至高维晶格上的自旋系统，并准确计算了二维和三维伊辛模型的自由能、能量、磁化强度等热力学量。

    

    arXiv:2609.27306v1 公告类型：新论文 摘要：传统的基于分数的扩散模型在学习分数函数时并不表示归一化密度，而易于处理的归一化模型则能同时支持采样和直接似然评估。最近的一种张量网络方法提供了这种表示，但在很大程度上仅限于低维晶格。本文提出了一种离散扩散模型，该模型使用变分自回归网络对归一化概率分布进行参数化。显式的马尔可夫跳跃算子控制前向加噪和反向去噪动力学，将具有归一化分布的离散扩散模型扩展到更高维晶格上的自旋系统。我们将该框架应用于有序、临界和无序相区中的二维和三维伊辛模型，准确计算了包括自由能、能量和磁化强度在内的热力学量。我们进一步将该框架与蒙特卡罗采样相结合，使用自适应（方法）……

    arXiv:2609.27306v1 Announce Type: new  Abstract: Conventional score-based diffusion models learn scores without representing normalized densities, whereas tractable normalized models support both sampling and direct likelihood evaluation. A recent tensor-network approach provides such a representation but is largely restricted to low-dimensional lattices. Here we introduce a discrete diffusion model that parameterizes normalized probability distributions using variational autoregressive networks. Explicit Markov jump operators govern the forward noising and reverse denoising dynamics, extending discrete diffusion models with normalized distributions to spin systems on higher-dimensional lattices. We apply this framework to the two- and three-dimensional Ising models across ordered, critical, and disordered regimes, accurately computing thermodynamic quantities including free energy, energy, and magnetization. We further integrate the framework with Monte Carlo sampling, using adaptive 
    
[^19]: 具有成对融合的多任务回归

    Multitask Regression with Pairwise Fusion

    [https://arxiv.org/abs/2609.27280](https://arxiv.org/abs/2609.27280)

    该论文提出一种通过对跨任务所有成对系数差异进行惩罚来估计多任务回归系数矩阵的方法，能够灵活刻画不同预测变量上任务间系数的共享与差异结构，并在活跃预测变量数和异常系数数这两个结构量上实现了匹配的上下界。

    

    我们研究了系数共享情况可因预测变量而异的多任务回归问题。对于某个给定的预测变量，许多任务可能具有相同的系数，而少数任务有所不同，且对于另一个预测变量，例外的任务不必相同。我们用两个量来刻画这种结构：活跃预测变量的数量，以及与其对应预测变量最常见取值不同的任务系数总数。我们通过惩罚跨任务的所有成对系数差异来估计系数矩阵，在需要进行预测变量选择时，还会附加一个组惩罚。所得到的上界和下界对这两个量具有相同的依赖关系。我们还考虑了更强的设定，即一大组任务共享同一个完整的系数向量。在明确的样本量条件下，同一个成对估计器可以将这组任务完全合并，同时允许其余任务有所不同。模拟实验和家庭能源数据的分析验证了该方法的有效性。

    arXiv:2609.27280v1 Announce Type: cross  Abstract: We study multitask regression when coefficient sharing can differ by predictor. For a given predictor, many tasks may have the same coefficient while a few differ, and the exceptional tasks need not be the same for another predictor. We describe this structure by two quantities: the number of active predictors and the total number of task coefficients that differ from the most common value for their predictor. We estimate the coefficient matrix by penalizing all pairwise coefficient differences across tasks, with an additional group penalty when predictor selection is needed. The resulting upper and lower bounds have the same dependence on these two quantities. We also consider the stronger setting in which a large set of tasks shares one entire coefficient vector. Under explicit sample-size conditions, the same pairwise estimator pools those tasks exactly, while allowing the remaining tasks to differ. Simulations and household energy 
    
[^20]: 基于条件协方差排序的函数型因果发现

    Functional Causal Discovery via Conditional Covariance Ordering

    [https://arxiv.org/abs/2609.27256](https://arxiv.org/abs/2609.27256)

    该论文通过比较条件协方差算子的范数提出新的拓扑排序可识别性条件，摆脱了传统因果发现对结构（线性/非线性）和分布（高斯/非高斯）假设的依赖，并结合混合回归模型与变量选择实现了函数型变量因果DAG的估计及其渐近一致性。

    

    我们研究每个节点均为随机函数的因果发现问题。以往关于该主题的研究依赖于结构性假设（如线性或非线性）以及分布性假设（如高斯或非高斯）。相比之下，我们利用协方差算子来避免这些假设。在函数型加性噪声模型下，我们提出了一种新的充分条件，通过比较条件协方差算子的范数来识别有效的拓扑排序。利用这一可识别性条件，我们开发了一种新的混合回归模型，该模型同时涵盖了线性和非线性模型。结合变量选择，我们的方法实现了对函数型变量的因果有向无环图（DAG）的估计。在理论方面，我们建立了该回归模型的最小二乘型理论，并推导了排序确定、稀疏回归以及DAG识别的渐近一致性。

    arXiv:2609.27256v1 Announce Type: cross  Abstract: We study causal discovery where each node is a random function. Previous studies on this topic rely on structural assumptions, e.g., linearity or non-linearity, and distributional assumptions, e.g., Gaussianity or non-Gaussianity. In contrast, we make use of covariance operators to avoid these assumptions. Under functional additive noise models, we propose a new sufficient condition to identify a valid topological ordering based on comparing norms of conditional covariance operators. Taking advantage of this identifiability condition, we develop a new mixed regression model that subsumes linear and non-linear models. Together with variable selection, our procedure yields an estimation of the causal directed acyclic graph (DAG) for functional variables. In theory, we develop the least-squares-type theory of this regression model, and derive asymptotic consistency of order determination, sparse regression, as well as identifying the DAG.
    
[^21]: 关于带成员查询的主动学习的样本复杂度

    On the Sample Complexity of Active Learning with Membership Queries

    [https://arxiv.org/abs/2609.27241](https://arxiv.org/abs/2609.27241)

    本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。

    

    本工作重新审视了主动学习中的一个根本问题：合成任意查询的能力究竟有多强大？与基于池的主动学习相比——即学习者只能从给定的未标注数据池中选择查询——我们发现这种看似微小的查询能力变化可能会极大地改变统计学习的难度。特别地，某些在基于池的设置下本质上学习缓慢、其误差随样本数量仅呈多项式衰减的假设类，一旦允许合成查询，便变得可以指数级快速学习。这一显著的差距表明，成员查询的合成引发了一种根本不同的学习模式，这种模式未能被现有主动学习理论充分刻画，需要新的分析工具来表征其复杂度。受这一现象启发，我们提出了若干充分条件，展示了有趣的例子，并提出……

    arXiv:2609.27241v1 Announce Type: cross  Abstract: This work revisits a fundamental question in active learning: how powerful is the ability to synthesize arbitrary queries? Compared to pool-based active learning, where the learner only selects queries from a given unlabeled pool, we find that this seemingly mild change in query ability may dramatically alter the difficulty of statistical learning. In particular, some hypothesis classes that are inherently slow to learn in the pool-based setting, achieving only polynomial error decay in the number of samples, become exponentially learnable once synthesized queries are allowed. This striking gap suggests that membership query synthesis induces a fundamentally different mode of learning, one that is not adequately captured by existing active learning theory and calls for new analytical tools to characterize its complexity. Motivated by this phenomenon, we develop several sufficient conditions, present intriguing examples, and propose a c
    
[^22]: 专家建议预测：多专家情形下任意时刻遗憾匹配固定时限最优常数

    Prediction with Expert Advice: Anytime Regret with Many Experts Matches the Fixed-Time Constant

    [https://arxiv.org/abs/2609.27206](https://arxiv.org/abs/2609.27206)

    本文提出一种无需预知时间视界的专家建议预测算法，使任意时刻的累积遗憾达到 $(1+O(\sqrt{\ln\ln n/\ln n}))\sqrt{t\ln n/2}$，消除了此前的 $\sqrt{2}$ 因子差距，从而在多专家情形下将任意时刻遗憾匹配到固定时限的最优常数。

    

    专家建议预测是在线学习中的一个基本问题。当时间视界 $T$ 事先已知时，$n$ 个专家下的极小化极大累积遗憾渐近为 $\sqrt{\frac{T \ln n}{2}}$，这可以通过乘性权重更新算法并以针对 $T$ 调整的学习率来实现，且已知该界是紧的。然而，如果要求遗憾界在每个时刻 $t$ 都同时成立，此前已知的最优保证为 $\sqrt{t \ln n}$——比前者差一个 $\sqrt{2}$ 的因子——而这个 $\sqrt{2}$ 因子是否必要一直悬而未决。本文证明该因子并不必要：我们给出一种无需知道时间视界的算法，其累积遗憾对所有 $t \ge 1$ 同时满足 $R_t \le \bigl(1 + O(\sqrt{\ln \ln n / \ln n})\bigr)\sqrt{t \ln n / 2}$。

    arXiv:2609.27206v1 Announce Type: cross  Abstract: Prediction with expert advice is a fundamental problem in online learning. When the time horizon $T$ is known in advance, the minimax cumulative regret over $n$ experts is asymptotically $\sqrt{\frac{T \ln n}{2}}$. This is achieved by the Multiplicative Weights Update algorithm with a learning rate tuned to $T$, and is known to be tight. If instead the regret bound is required to hold simultaneously at every time $t$, the best known guarantee has been $\sqrt{t \ln n}$---a factor of $\sqrt{2}$ worse---and it has remained unknown whether this factor of $\sqrt{2}$ is necessary. We show that it is not. We give an algorithm, requiring no knowledge of the horizon, whose cumulative regret satisfies $R_t \le \bigl(1 + O(\sqrt{\ln \ln n / \ln n})\bigr)\sqrt{t \ln n / 2}$ simultaneously for every $t \ge 1$.
    
[^23]: 基于治疗前-后数据的人工智能替代指标用于治疗效应估计

    Artificial intelligence surrogates for treatment effect estimation with before-and-after data

    [https://arxiv.org/abs/2609.27180](https://arxiv.org/abs/2609.27180)

    该论文提出一个新框架，利用预训练AI模型对每位患者治疗前后的测量数据进行结局预测并比较个体内差异，从而以AI预测作为低成本替代指标来估计治疗的因果效应。

    

    当临床上重要的结局指标测量成本高昂或需要长期随访时，估计医学治疗的因果效应十分困难。短期或低成本的替代结局指标提供了一种潜在的替代方案，但替代生物标志物可能不可用或难以识别。人工智能（AI）的进步使得从廉价的高维测量数据中预测临床结局越来越准确，这为将AI预测本身用作替代指标创造了机会。为此，我们开发了一个框架，用于从每位接受治疗个体的治疗前和治疗后配对测量数据中估计治疗效应。该方法将预训练的AI模型应用于治疗前后的测量数据，我们的估计器比较所得的结局预测。我们刻画了这种个体内对比能够识别平均治疗效应所需满足的技术假设条件。

    arXiv:2609.27180v1 Announce Type: cross  Abstract: Estimating the causal effects of medical treatments is difficult when clinically important outcomes are costly to measure or require long follow-up. Short-term or inexpensive surrogate outcomes offer a potential alternative, but surrogate biomarkers may be unavailable or difficult to identify. Advances in artificial intelligence (AI) have enabled increasingly accurate prediction of clinical outcomes from inexpensive, high-dimensional measurements, which creates an opportunity to use AI predictions themselves as surrogates. To this end, we develop a framework for estimating treatment effects from paired measurements obtained before and after treatment for each treated individual. A pretrained AI model is applied to the before and after measurements, and our estimator compares the resulting outcome predictions. We characterize the technical assumptions under which this within-person contrast identifies the average treatment effect on the
    
[^24]: 基于共形鞅的变点检测：新的最优构造与现有方法的次优性

    Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods

    [https://arxiv.org/abs/2609.27179](https://arxiv.org/abs/2609.27179)

    本文建立了非可交换数据下共形p值行为的系统理论，证明现有共形鞅变点检测方法在PFA和ARL控制下是次优的（检测延迟可达$\Omega(T)$和$\Omega(\sqrt{\text{ARL}})$），并提出了可证明极小化极大最优的新型共形e-过程和e-检测器。

    

    我们研究针对独立观测的免分布序贯变点检测问题，其中变化前后的分布未知且不受任何限制。我们以Vovk(2021)提出的共形检验鞅及相关的e-检测器为基础，二者分别用于控制误报概率（PFA）和平均运行长度（ARL）。现有工作大多只关注方法的有效性，而统计效率通常仅留待数值模拟来检验。我们针对变点位于未知时刻$T$的非可交换数据，建立了共形p值行为的全面理论。我们利用该理论分析了共形鞅方法在变化后统计量的增长及由此导致的检测延迟，并证明现有的标准方法在PFA和ARL控制下是次优的，其检测延迟可分别达到$\Omega(T)$和$\Omega(\sqrt{\text{ARL}})$。我们提出了多种不同的共形e-过程和e-检测器，并证明它们具有极小化极大最优性。

    arXiv:2609.27179v1 Announce Type: cross  Abstract: We study distribution-free sequential changepoint detection for independent observations with unknown and unrestricted pre- and post-change laws. We build on the conformal test martingales and associated e-detectors of Vovk(2021), which control the probability of false alarm (PFA) and the average run length (ARL) respectively. The majority of these works focus on validity, with statistical efficiency usually left for simulations. We develop a comprehensive theory of how conformal p-values behave under non-exchangeable data with a changepoint at an unknown time $T$. We use this to analyze the post-change growth and resulting detection delay of conformal martingale methods, and prove that the standard existing methods are suboptimal for PFA and ARL control, and can lead to delays that are $\Omega(T)$ and $\Omega(\sqrt{\text{ARL}})$ respectively. We propose different conformal e-processes and e-detectors that are provably minimax optimal,
    
[^25]: 点赞陷阱：针对基于相似度的推荐系统中智能体的多阶段投毒攻击

    The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems

    [https://arxiv.org/abs/2609.27155](https://arxiv.org/abs/2609.27155)

    该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。

    

    随着大语言模型（LLMs）及基于LLM的智能体的最新发展，这些智能体正变得日益自主，并能够更广泛地代表用户在互联网上执行操作。然而，部署在社交媒体平台上的自动化智能体（例如用于管理用户个人账户的智能体）的脆弱性仍未得到充分探索。现有关于智能体投毒的研究通常假设攻击者能够将投毒内容直接暴露给智能体。尽管这种攻击方式直接且有效，但更容易被检测和缓解。在社交媒体平台的背景下，这留下了一个悬而未决的问题：推荐系统本身是否会以更隐蔽的方式将此类内容推送给智能体。通过理论分析，我们证明了OASIS系统中使用的点赞评分机制可以被利用，并刻画了多阶段投毒帖子链能够操纵智能体信息流的条件。基于这些……

    arXiv:2609.27155v1 Announce Type: cross  Abstract: With recent advancements in large language models (LLMs) and LLM-based agents, these agents are becoming increasingly autonomous and gaining broader access to act on users' behalf on the internet. However, the vulnerability of automated agents deployed on social media platforms (e.g., for managing a user's personal account) remains underexplored. Existing studies on agent poisoning typically assume that the adversary can expose poisoned content to the agent. Although such an attack is direct and effective, it is more easily detected and mitigated. In the context of social media platforms, this leaves open whether the recommendation system itself would surface such content to the agent in a more subtle manner. Through theoretical analysis, we show that the like-score mechanism used in OASIS can be exploited, and we characterize the conditions under which a multi-stage chain of poisoned posts can steer the agent's feed. Based on these in
    
[^26]: WTF?! 基于Wasserstein倾斜流映射的无模拟强化学习

    WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps

    [https://arxiv.org/abs/2609.27033](https://arxiv.org/abs/2609.27033)

    提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。

    

    奖励微调旨在更新预训练的基于流的生成模型，以提升其生成样本的下游奖励。现有方法通常将该问题表述为从奖励倾斜分布中采样，即KL正则化奖励最大化问题的解。本文引入了一种直接基于预训练漂移构建的最优传输正则化器。与KL奖励倾斜不同，所得目标是使个体样本向更高奖励方向传输，而非对基础分布进行重新加权。我们证明了该问题等价于流上的一个确定性最优控制问题。给定预训练的流映射，这种等价性催生了一种用于微调生成流的无模拟强化学习算法。我们将所得框架称为Wasserstein倾斜流映射，这是首个原生于流映射的端到端微调方案。其输出是一个微调后的流映射……

    arXiv:2609.27033v1 Announce Type: new  Abstract: Reward fine-tuning aims to update a pre-trained flow-based generative model to improve the downstream reward of its generated samples. Existing methods typically formulate this problem as sampling from a reward-tilted distribution, the solution to a KL-regularized reward-maximization problem. Here, we introduce an optimal transport regularizer built directly from the pre-trained drift. Unlike KL reward tilting, the resulting objective transports individual samples toward higher reward rather than reweighting the base distribution. We show that the resulting problem is equivalent to a deterministic optimal control problem on the flow. Given a pre-trained flow map, this equivalence yields a simulation-free reinforcement learning algorithm for fine-tuning generative flows. We call the resulting framework Wasserstein-Tilted Flow Maps (WTF), the first end-to-end fine-tuning recipe native to flow maps. The output is a fine-tuned flow map that 
    
[^27]: 基于多尺度矩阵权重的在线逆线性优化的紧致遗憾界

    Tight Regret Bound for Online Inverse Linear Optimization via Multiscale Matrix Weights

    [https://arxiv.org/abs/2609.26978](https://arxiv.org/abs/2609.26978)

    该论文提出了一种基于多尺度矩阵乘法权重的随机化算法，实现了在线逆线性优化中O(√d)的期望遗憾界，达到了理论最优水平。

    

    我们研究了具有固定未知线性效用函数的在线逆线性优化问题：在每一轮中，环境给出一个紧致的动作集合，学习者从中推荐一个动作，环境则返回在同一集合上最大化效用的动作。当效用向量和动作位于d维欧几里得单位球内时，我们提出了一个随机化算法，其遗憾值（相对于最优动作的累计效用损失）在期望意义下为O(√d)，且适用于任意时间范围，无需预知时间范围。根据已知的当T≥d时Ω(√d)的下界，该算法对d的依赖性在常数因子内是最优的。我们的算法在按几何间距分布的多项式特征空间上维护矩阵乘法权重，通过求解线性规划来选择推荐分布，并通过比较可用动作与反馈动作来更新评分矩阵。

    arXiv:2609.26978v1 Announce Type: cross  Abstract: We study online inverse linear optimization with a fixed unknown linear utility: in each round, an environment presents a compact action set, the learner recommends an action from it, and the environment returns an action that maximizes the utility over the same set. When the utility vector and the actions lie in the $d$-dimensional Euclidean unit ball, we give a randomized algorithm whose regret---the cumulative utility shortfall relative to optimal actions---is $O(\sqrt d)$ in expectation for every time horizon, without knowledge of the horizon. The dependence on $d$ is optimal up to a constant factor by the known $\Omega(\sqrt d)$ lower bound for horizons $T\ge d$. Our algorithm maintains matrix multiplicative weights on polynomial feature spaces at geometrically spaced scales. It selects a recommendation distribution by solving a linear program and updates its score matrices by comparing the available actions with the feedback acti
    
[^28]: 序贯模型训练中的滚动保形预测

    Rolling Conformal Prediction in Sequential Model Training

    [https://arxiv.org/abs/2609.26951](https://arxiv.org/abs/2609.26951)

    本文提出滚动保形预测，一种无需数据划分的无分布预测推断方法，能够为序贯模型训练过程中的预测提供边际覆盖率保证。

    

    我们提出了滚动保形预测，这是一种面向序贯模型训练场景的无分布预测推断方法。具体而言，给定数据流 $(X_1,Y_1),(X_2,Y_2),\dots$，在每个时刻 $n$，训练得到的模型可能依赖于已观测的历史数据 $\{(X_i,Y_i)\}_{i<n}$。这一设定在现代序贯训练中自然出现，包括对海量数据集的单遍训练，以及在部署过程中对语言模型进行持续微调或测试时自适应。Rolling-CP 首先针对当前预测器对每个新到的观测进行校准，然后将其滚动纳入后续训练中。通过这种方式，我们避免了对数据进行划分的需要。值得注意的是，尽管时刻 $n=1,2,\dots$ 的模型可能具有完全不同的性质和精度水平，但对于可交换数据，仍然可以建立边际覆盖率的保证，并带有常见的普适二倍因子保证（最坏情况……

    arXiv:2609.26951v1 Announce Type: cross  Abstract: We introduce Rolling Conformal Prediction (rolling-CP), a distribution-free predictive inference method for the setting of sequential model training. Specifically, given a data stream $(X_1,Y_1),(X_2,Y_2),\dots$, at each time $n$ the trained model may depend on the observed history $\{(X_i,Y_i)\}_{i<n}$. This setting arises naturally in modern sequential training, including one-pass training over massive datasets and continual fine-tuning or test-time adaptation of language models during deployment.   Rolling-CP first calibrates each incoming observation against the current predictor and then rolls it into future training. In this way, we avoid the need for data splitting. Remarkably, although the models at times $n=1,2,\dots$ may have entirely different properties and accuracy levels, for exchangeable data it is nonetheless possible to establish a guarantee of marginal coverage, with a familiar universal factor-two guarantee (a worst 
    
[^29]: 带有空间与网络对象的可加非参数回归

    Additive Nonparametric Regression with Spatial and Network Objects

    [https://arxiv.org/abs/2609.26867](https://arxiv.org/abs/2609.26867)

    本文提出了一种将任务态和结构MRI图像视为函数型数据的新型可加非参数回归框架，通过高斯过程联合先验同时捕捉空间与网络预测变量的结构及其相互联系，从而从结构MRI和静息态fMRI预测脑激活图并量化预测不确定性。

    

    本文受青少年大脑认知发展（ABCD）研究中的一个成像应用启发，旨在利用结构磁共振成像（s-MRI）的皮层指标和静息态功能磁共振成像（rs-fMRI）的脑连接数据，通过任务态功能磁共振成像（t-fMRI）预测基于任务的脑激活图。层次贝叶斯模型非常适合整合多种成像数据并量化预测的不确定性。然而，由于在设计能够捕捉不同成像模态之间结构和相互联系的联合先验方面存在挑战，加上计算复杂性和缺乏理论保证，该领域的进展受到限制。为应对这些挑战，本文引入了一种新颖的回归框架，将t-fMRI和s-MRI图像视为函数型数据，纳入网络预测变量和函数型预测变量对函数型响应的可加非线性效应。具体而言，我们采用高斯过程

    arXiv:2609.26867v1 Announce Type: cross  Abstract: This article is motivated by an imaging application from the Adolescent Brain Cognitive Development (ABCD) study, aiming to predict task-based brain activation maps from t-fMRI using cortical metrics from structural MRI (s-MRI) and brain connectivity data from resting-state fMRI (rs-fMRI). Hierarchical Bayesian modeling is well-suited for integrating diverse imaging data and quantifying prediction uncertainty. However, progress in this field is limited due to challenges in designing joint priors that capture the structures and interconnections between different imaging modalities, along with computational complexity and lack of theoretical assurances. To address these challenges, the article introduces a novel regression framework that treats t-fMRI and s-MRI images as functional data, incorporating additive non-linear effects of both network and functional predictors on the functional response. Specifically, we employ Gaussian process
    
[^30]: 用于基于残差的自适应GMsFEM的高斯过程代理指标

    Gaussian-process surrogate indicators for residual-based adaptive GMsFEM

    [https://arxiv.org/abs/2609.26843](https://arxiv.org/abs/2609.26843)

    本文提出一种非侵入式的高斯过程代理模型来加速基于残差的自适应GMsFEM中反复的指标评估，在不改变多尺度求解和基函数加密的前提下用KRR等价形式预测指标分数，并通过扰动标记理论量化了分数误差对指标质量捕获的影响。

    

    针对高对比度椭圆问题的基于残差的自适应广义多尺度有限元方法（GMsFEM）需要在每个粗网格邻域上反复计算局部加权 $H^{-1}$ 指标，这使得指标评估成为重复查询场景中反复出现的计算开销。我们为 Dörfler 标记中所使用的指标分数引入了一种非侵入式的高斯过程（GP）代理模型。该操作性预测器为GP后验均值，在所述约定下与核岭回归（KRR）估计器在代数上等价；它利用压缩后的局部解和谱特征，而无需改变多尺度求解、局部谱构造或基函数加密过程。一个非均匀扰动标记结果量化了逐点分数误差如何影响由代理选择的邻域所捕获的精确指标质量，而一个条件有界差异的KRR途径为此类分数界确定了充分假设。在受控的留出分布内测试中……

    arXiv:2609.26843v1 Announce Type: cross  Abstract: Residual-based adaptive GMsFEM for high-contrast elliptic problems repeatedly evaluates local weighted $H^{-1}$ indicators on every coarse neighborhood, making indicator evaluation a recurring cost in repeated-query settings. We introduce a non-intrusive Gaussian-process (GP) surrogate for the indicator scores used in D\"orfler marking. The operational predictor is the GP posterior mean, algebraically equivalent to a kernel ridge regression (KRR) estimator under the stated convention; it uses compressed local solution and spectral features without changing the multiscale solve, local spectral construction, or basis enrichment. A nonuniform perturbed-marking result quantifies how pointwise score errors affect the exact indicator mass captured by surrogate-selected neighborhoods, while a conditional bounded-discrepancy KRR pathway identifies sufficient assumptions for such score bounds. In controlled held-out in-distribution tests, the s
    
[^31]: 关于稀疏高斯过程回归中基函数选择的研究

    On Basis Function Selection for Sparse Gaussian Process Regression

    [https://arxiv.org/abs/2609.26624](https://arxiv.org/abs/2609.26624)

    本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。

    

    稀疏高斯过程通过在输入空间上用固定基函数集 {φ_j} 的适当展开来替代核函数，从而实现 O(N) 的推断。在给定计算预算 M ≪ N 的情况下，从业者通常习惯性地将基截断为前 M 个基函数。然而，从形式上看，并没有任何限制阻止人们只选择那些对当前数据真正重要的 M 个基函数。这样做可以避免将计算预算浪费在没有信号的基函数上，但这需要一个能够对候选基函数进行排序的准则。我们从基函数选择问题的信息论视角出发，提出了三种这样的准则。每种准则分别对应于选择时所处的不同知识状态：无数据状态、无先验状态以及介于两者之间的状态。随后，我们在六个 UCI 回归基准数据集上，针对三种基函数族（包括希尔伯特空间高斯过程 HSGP 等），研究了截断策略与选择策略的性能表现。

    arXiv:2609.26624v1 Announce Type: cross  Abstract: Sparse Gaussian processes achieve $O(N)$ inference by replacing the kernel with an appropriate expansion in a fixed basis $\{\phi_j\}$ on the input space. Given a compute budget $M \ll N$, practitioners conventionally truncate the basis to its first $M$ entries. Nothing in the formalism, however, prevents one from selecting only those $M$ basis functions that matter for the data at hand. This would avoid spending budget on basis functions where there is no signal, but it requires a criterion for ranking the candidates. We propose three such criteria derived from an information-theoretic view of the basis-function selection problem. Each criterion matches a different state of knowledge at selection time: a no-data state, a no-prior state, and an in-between state. We then study the performance of truncation versus selection strategies on six UCI regression benchmarks across three basis families: Hilbert-space Gaussian processes (HSGP), v
    
[^32]: 黎曼随机优化的局部私有推断

    Locally Private Inference for Riemannian Stochastic Optimization

    [https://arxiv.org/abs/2609.22642](https://arxiv.org/abs/2609.22642)

    该论文提出了一种在局部差分隐私下对流形值总体极小值点进行统计推断的方法，通过条件中心化的随机切梯度保持一阶方程，并引入对称对回归（SPR）从相同私有消息中估计渐近方差，进而证明了中心极限定理及基于交互记录的三明治协方差和内在Wald区域的一致性。

    

    我们针对流形值的总体极小值点发展了统计推断方法，适用于每个观测属于不同参与者、且分析师只能接收到局部私有消息的场景。该方法释放随机化的切空间梯度，并通过黎曼随机逼近和Polyak-Ruppert平均将其组合。将私有数据替代量直接插入非线性损失可能会移动其总体目标，而对释放梯度进行条件中心化则可以保持一阶方程。我们引入对称对回归（SPR），从用于点估计的相同私有消息中估计渐近方差，无需保留部分参与者或请求第二次数据释放。我们在局部差分隐私下证明了中心极限定理，以及完全基于交互记录的三明治协方差和内在Wald区域的一致性。在多种统计问题和流形上的模拟支持了该方法的预测性能。

    arXiv:2609.22642v1 Announce Type: cross  Abstract: We develop inference for manifold-valued population minimizers when each observation belongs to a different participant and only locally private messages reach the analyst. The method releases randomized tangent gradients and combines them through Riemannian stochastic approximation and Polyak-Ruppert averaging. Directly inserting a private data surrogate into a nonlinear loss can shift its population target, whereas conditional centring of the released gradient preserves the first-order equation. We introduce symmetric-pair regression (SPR) to estimate the asymptotic variance from the same private messages used for point estimation, without holding out participants or requesting a second release. We prove the central limit theorem and consistency of the fully transcript-based sandwich covariance and intrinsic Wald region under local differential privacy. Simulations across various statistical problems and manifolds support the predict
    
[^33]: 黎曼流形上切向量场回归的同时推断

    Riemannian Simultaneous Inference for Tangent Vector Field Regression

    [https://arxiv.org/abs/2609.21910](https://arxiv.org/abs/2609.21910)

    该论文针对无边黎曼流形上的切向量场回归提出了一种基于平行输运与体积校正的核估计方法，并通过单位切丛上的上确界表示与 Gumbel 极限理论，构造了回归场的可行同时置信管。

    

    我们考虑无边黎曼流形上的非参数切向量场回归。由于不同点处的响应位于不同的切空间中，所提出的核估计量首先将近邻的响应平行输运到目标切空间，然后形成经过体积校正的局部平均。我们首先推导了该估计量的一致二阶偏差、有限带宽协方差以及随机收敛率。对于同时推断，我们将切范数表示为单位切丛上的上确界。精确的协方差白化给出了一个单位方差的高斯场，其相关长度沿底流形为 $h$ 阶，沿纤维为一阶。其局部协方差几何导致了一个带有显式内蕴常数的 Gumbel 极限。将该极限与高斯近似和交叉拟合协方差估计相结合，得到了回归场的可行同时置信管。我们进一步讨论……（摘要被截断）

    arXiv:2609.21910v1 Announce Type: cross  Abstract: We consider nonparametric tangent vector field regression on a Riemannian manifold without boundary. Because responses at different points lie in different tangent spaces, the proposed kernel estimator first parallel transports nearby responses to the target tangent space and then forms a volume-corrected local average. We first derive its uniform second-order bias, finite-bandwidth covariance, and stochastic rate. For simultaneous inference, the tangent norm is written as a supremum over the unit tangent bundle. Exact covariance whitening gives a unit-variance Gaussian field whose correlation length is of order $h$ along the base manifold and of order one along the fibre. Its local covariance geometry leads to a Gumbel limit with an explicit intrinsic constant. Combining this limit with Gaussian approximation and cross-fitted covariance estimation yields a feasible simultaneous confidence tube for the regression field. We further disc
    
[^34]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^35]: TabPFN-3.5：技术报告

    TabPFN-3.5: Technical Report

    [https://arxiv.org/abs/2609.17895](https://arxiv.org/abs/2609.17895)

    TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。

    

    我们推出 TabPFN-3.5，这是我们的全新旗舰表格基础模型。它在广泛的表格任务上显著超越了其前代模型 TabPFN-3 以及所有现有基线。TabPFN-3.5 在 TabArena 的标准表格预测任务上创造了新的最先进水平，并将其扩展到实际从业者会遇到的数据场景：具有时间或分组划分的非独立同分布数据、包含字符串、文本和图像的表格、高基数类别特征，以及具有众多特征的宽表。这些优势延续到我们的任务专用框架中：在关系型数据上达到最先进水平，并具备更强的时间序列预测能力。为了实现更快的推理，我们的变体 TabPFN-3.5-Fast 运行速度最高可达 TabPFN-3 的 3 倍，同时保留了大部分精度提升。此外，我们升级了 TabPFN-3.5-Plus，通过先进的文本和日期处理以及专有推理优化扩展了多模态能力。最后，我们发布了一个新的版本。

    arXiv:2609.17895v1 Announce Type: new  Abstract: We introduce TabPFN-3.5, our new flagship Tabular Foundation Model. It significantly outperforms its predecessor, TabPFN-3, and all existing baselines across a broad range of tabular problems. TabPFN-3.5 sets a new state of the art on standard tabular prediction in TabArena, and extends it to the data practitioners encounter in practice: non-i.i.d. data with temporal or grouped splits, tables with strings, text and images, high-cardinality categorical features, and wide tables with many features. These gains carry over to our task-specific harnesses: state of the art on relational data and stronger time-series forecasting. For faster inference, our variant TabPFN-3.5-Fast runs up to 3x faster than TabPFN-3 while keeping most of the accuracy gains. In addition, we upgrade TabPFN-3.5-Plus, expanding our multimodal capabilities with advanced text and date handling alongside proprietary inference optimizations. Finally, we release a new vers
    
[^36]: 能量引导递归模型

    Energy-guided Recursive Model

    [https://arxiv.org/abs/2607.10128](https://arxiv.org/abs/2607.10128)

    提出能量引导递归模型（ERM），利用Hopfield型记忆为候选轨迹分配内在能量，从而有原则地指导轨迹选择与递归深度的确定，在数独、铅笔谜题和迷宫等推理任务上取得递归建模领域的最佳准确率，并降低了语言建模的困惑度。

    

    递归模型在推理和语言任务上展现出巨大潜力，但其在测试时的扩展缺乏一种有原则性的准则来选择轨迹或确定递归深度。我们提出了能量引导递归模型，该模型利用Hopfield型记忆存储有效的局部和全局结构，为候选轨迹分配内在能量。这些能量可以指导候选轨迹的选择，并提示递归深度的有效范围，这意味着更深的递归并不一定能提高推理准确率。这些能量还使得并行回火等采样方法能够改善探索效果。在推理任务上，ERM在数独（98.97%）、铅笔谜题基准（Pencil Puzzle Bench，PPBench，88.04%）和迷宫（99.30%）上达到了最优解，取得了递归建模中的最佳准确率。在语言建模方面，ERM以极小的推理开销将RedPajama-V2的困惑度降低了1.74%。这些结果支持了能量引导的方法……

    arXiv:2607.10128v3 Announce Type: replace  Abstract: Recursive models show promise on reasoning and language tasks, yet their test-time scaling lacks a principled criterion for selecting trajectories or determining recurrent depth. We introduce \textbf{Energy-guided Recursive Model (ERM)}, which uses Hopfield-type memories of valid local and global structures to assign intrinsic energies to candidate trajectories. These energies guide candidate selection and suggest an effective range of recurrent depths, implying that deeper recurrence does not necessarily improve reasoning accuracy. They also enable sampling methods such as parallel tempering to improve exploration. For reasoning tasks, ERM achieves optimal solutions on Sudoku ($98.97\%$), Pencil Puzzle Bench (PPBench, $88.04\%$) and Maze ($99.30\%$), reaching the best accuracy in recursive modeling. On language modeling, ERM reduces RedPajama-V2 perplexity by $1.74\%$ with marginal inference overhead. The results support energy guid
    
[^37]: 用于分层分类的同时潜在预算树

    Simultaneous Latent Budget Trees for Stratified Classification

    [https://arxiv.org/abs/2606.13295](https://arxiv.org/abs/2606.13295)

    本文提出同时潜在预算树，一种面向含时间、空间或人口等分层因素的场景的分类树概率机器学习框架，通过将子节点解释为同时混合模型的潜在成分来构建基于模型的条件分裂规则。

    

    在可解释人工智能时代，单棵决策树因其易于解释而重新受到关注。本文提出了同时潜在预算树，这是一种在存在分层因素（如时间、空间或人口统计学变量，其可作为控制变量或潜在混杂因素）情况下的分类树的概率机器学习框架。标准的树生长程序并非为优化条件分裂规则而设计。本文提出了一种基于模型的分裂规则，其中子节点被解释为拟合于父节点的同时混合模型（如同时潜在预算模型及其约束版本）的潜在成分。混合参数针对每个组以不同方式将观测值引导至子节点，而潜在预算参数则更新控制变量各水平下的响应类别轮廓。参数估计

    arXiv:2606.13295v3 Announce Type: replace-cross  Abstract: In the era of Explainable Artificial Intelligence, there is a renewed focus on single trees for their ease of interpretation. This paper introduces Simultaneous Latent Budget Trees, a probabilistic machine learning framework for classification trees in the presence of a stratification factor such as a temporal, spatial, or demographic variable, acting as a control variable or potential confounder. Standard tree growth procedures are not designed to optimize a conditional split rule. A model-based split rule is proposed in which child nodes are interpreted as latent components of a simultaneous mixture model, such as the Simultaneous Latent Budget Model and its constrained versions, fitted to the parent node. Mixing parameters drive the observations, differently for each group, to the child nodes whereas latent budgets parameters update the response classes profile of each level of the control variable. Parameters are estimated 
    
[^38]: 一种用于输入凸神经网络训练的提升方法

    A lift for input-convex neural net training

    [https://arxiv.org/abs/2605.24274](https://arxiv.org/abs/2605.24274)

    针对输入凸神经网络训练中softplus参数化导致负权重区域梯度指数衰减、逃逸缓慢的问题，提出用可学习松弛量加无约束网络（以批次置换不变摘要为输入）替换自由潜在权重的“提升”方法。

    

    输入凸神经网络为密度模型和传输映射的凸势能进行参数化，其凸性要求层间权重必须非负。投影梯度下降通过在每步之后进行投影来强制满足该约束，但由于小批量噪声的存在，边界会被无限次地重新穿越，导致投影永远无法正确识别出一个活跃集。可微的替代方案——直接的softplus参数化——通过softplus正性映射来优化一个自由的潜在权重，其导数在权重为负的区域（即“肩部”）会以指数方式衰减梯度，因此一旦某个坐标到达该区域，就会在指数长的时间内停留在那里。为了在保留这种无约束参数化优点的同时避免其缓慢逃逸的问题，我们提出了提升方法，它用一个可学习的松弛量加上一个无约束网络（称为“主体”）来替换自由的潜在权重，该网络以训练批次的置换不变摘要作为输入。由此，潜在权重

    arXiv:2605.24274v2 Announce Type: replace  Abstract: Input-convex neural nets parametrize the convex potentials of density models and transport maps, and their convexity requires the inter-layer weights to be non-negative. Projected gradient descent enforces this by projecting after each step, and due to mini-batch noise the boundary is re-crossed indefinitely, which leads to an active set the projection never identifies. The differentiable alternative, direct softplus, optimizes a free latent weight through a softplus positivity map whose derivative attenuates the gradient exponentially where the weight is negative---the shoulder---so a coordinate that reaches it stays for an exponentially long time. To keep this unconstrained parametrization without its slow escape, we propose the lift, which replaces the free latent weight by a learnable slack plus an unconstrained network---the body---that takes a permutation-invariant summary of the training batch as input. The latent weight thus 
    
[^39]: ProteinJEPA：潜在预测改进蛋白质语言模型预训练

    ProteinJEPA: Latent prediction improves protein language model pretraining

    [https://arxiv.org/abs/2605.07554](https://arxiv.org/abs/2605.07554)

    ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。

    

    蛋白质语言模型主要以掩码语言建模（MLM）进行训练，即预测被掩码的氨基酸身份。联合嵌入预测架构（JEPA）则改为预测潜在表示，但尚未被应用于蛋白质领域。ProteinJEPA在MLM的基础上增加了一个余弦损失，用于在给定未掩码序列的条件下预测教师模型的半深度隐藏状态。在19个任务上，采用3500万和1.5亿参数的ESM2模型以及三个预训练随机种子，MLM+JEPA在114次比较中分别有78次和76次优于计算量匹配和训练步数匹配的仅MLM持续训练（14次落后，22次平局）。在结构与同源性敏感的任务上，计算量匹配的中位数增益为+0.0106，而其他任务上仅为+0.0041，其中以SCOPe-40检索和远程同源性任务提升最为显著，分别实现了Recall@1提高6.1个百分点和准确率提高2.7个百分点。这些任务上的增益随模型规模从800万增加到1.5亿而持续增大。

    arXiv:2605.07554v2 Announce Type: replace-cross  Abstract: Protein language models are trained primarily with masked language modeling (MLM), which predicts masked amino-acid identities. Joint-embedding predictive architectures (JEPA) instead predict latent representations, but have not been applied to proteins.   ProteinJEPA supplements MLM with a cosine loss for predicting the half-depth hidden states of a teacher given the unmasked sequence. On 19 tasks, with ESM2 at 35M and 150M parameters and three pretraining seeds, MLM+JEPA outperforms compute-matched and step-matched MLM-only continued training in 78 and 76 of 114 comparisons (14 losses, 22 ties). The median compute-matched gain is $+0.0106$ on structure- and homology-sensitive tasks versus $+0.0041$ elsewhere, led by SCOPe-40 retrieval and remote homology with improvements of 6.1 percentage points in Recall@1 and 2.7 points in accuracy, respectively. Gains on these tasks increase with model size from 8M to 150M. Against the of
    
[^40]: 截断盲区：解码策略如何系统性地排除人类式的词元选择

    The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices

    [https://arxiv.org/abs/2603.18482](https://arxiv.org/abs/2603.18482)

    该论文提出“截断盲区”概念，揭示 top-k 和核采样等解码策略因截断低概率词元而系统性地排除了 8–18% 的人类典型选词，从而为机器生成文本为何始终可被检测提供了机制性解释。

    

    为什么机器生成的文本依然能够被检测出来？我们在解码阶段研究了一种机制性解释：诸如 top-k 和核采样（nucleus sampling）等标准策略将生成过程限制在高概率词元上，而人类作者通常会选择模型概率分布中更深层次、但在语境中合适的词语。截断使得这些人类选择中可测量的部分变得无法触及；我们将其称为“截断盲区”。在五个开源模型和三个领域的实验中，8–18% 的人类选择词元落在了常见截断边界之外。语言分析进一步揭示，实义词元被不成比例地排除在外。在一个包含 180 万条机器生成文本的基准测试中，仅使用可预测性和词汇多样性特征的分类器即达到接近 0.97 的平均 AUC-ROC，且在不同解码设置下存在显著差异，并在不同生成器之间展现出很强的迁移能力。概率下限采样器能够在很大程度上缩……（原文摘要在此处截断）

    arXiv:2603.18482v4 Announce Type: replace  Abstract: Why does machine-generated text remain detectable? We investigate a mechanistic explanation at the decoding stage: standard strategies such as top-$k$ and nucleus sampling restrict generation to high-probability tokens, while human writers routinely choose contextually appropriate words from deeper in the model's probability distribution. Truncation makes a measurable share of these choices unreachable; we call this the \emph{truncation blind spot}. Across five open models and three domains, 8--18\% of human-selected tokens fall outside common truncation boundaries. Linguistic analysis further reveals disproportionate exclusion of content-word tokens. In a benchmark comprising 1.8 million machine generations, classifiers using only predictability and lexical diversity achieve mean AUC-ROC near 0.97, with substantial variation across decoding settings and strong transfer across generators. Probability-floor samplers substantially narr
    
[^41]: 非平稳高斯过程的正则傅里叶特征

    Regular Fourier Features for Nonstationary Gaussian Processes

    [https://arxiv.org/abs/2602.23006](https://arxiv.org/abs/2602.23006)

    该论文提出正则傅里叶特征方法，通过直接离散化可调和非平稳高斯过程的谱表示，摆脱了谱密度必须为概率测度的限制性假设，实现了无需概率假设、结构上半正定的高效低秩近似。

    

    模拟高斯过程需要从高维高斯分布中采样，其计算复杂度随采样位置数量呈三次方增长。谱方法通过利用傅里叶表示，并将谱密度视为适合蒙特卡洛近似的概率分布来解决这一挑战。尽管这种概率解释对平稳过程是有效的，但对于非平稳情况而言则过于受限，因为非平稳过程的谱密度通常并非概率测度。为了避免这一限制，我们提出了一种针对具有一维输入的可调和过程的正则傅里叶特征方法。我们的方法直接对谱表示进行离散化，在不需要概率假设的情况下保留了谱权重之间的相关结构。在假设谱支撑有限的前提下，该方法可产生一种结构上即为半正定的高效低秩近似。

    arXiv:2602.23006v3 Announce Type: replace-cross  Abstract: Simulating a Gaussian process requires sampling from a high-dimensional Gaussian distribution, which scales cubically with the number of sample locations. Spectral methods address this challenge by exploiting the Fourier representation and treating the spectral density as a probability distribution suitable for Monte Carlo approximation. Although this probabilistic interpretation is valid for stationary processes, it is overly restrictive for the nonstationary case, where spectral densities are generally not probability measures. To avoid this limitation, we propose regular Fourier features for harmonizable processes with one-dimensional inputs. Our method discretizes the spectral representation directly, preserving the correlation structure among spectral weights without requiring probability assumptions. Assuming finite spectral support, this yields an efficient low-rank approximation that is positive semi-definite by constru
    
[^42]: 基于图神经网络学习近似求解均匀设施选址问题

    Learning to Approximate Uniform Facility Location via Graph Neural Networks

    [https://arxiv.org/abs/2602.13155](https://arxiv.org/abs/2602.13155)

    提出了一种融合近似算法原理的完全可微分消息传递神经网络，用于求解均匀设施选址问题，既具有可证明的近似保证，又在实证中优于标准近似算法并缩小了与整数线性规划的差距。

    

    神经网络，特别是消息传递神经网络（MPNN），正日益被用作求解困难组合优化问题的启发式方法。然而，许多基于学习的方法依赖于监督、强化学习或梯度估计器，导致计算成本高、训练不稳定或保证有限。经典近似算法虽然提供最坏情况下的理论保证，但不可微分，且无法适应自然输入分布中的结构。我们通过均匀设施选址问题（UniFL）来研究这种权衡，该问题在聚类、摘要、物流和供应链等领域具有广泛应用。我们提出了一种完全可微分的MPNN，它融合了近似算法的原理，无需求解器监督或离散松弛。该模型具有可证明的近似保证，并在实证中优于标准近似算法，缩小了与整数线性规划之间的差距。

    arXiv:2602.13155v3 Announce Type: replace  Abstract: Neural networks, particularly message-passing neural networks (MPNNs), are increasingly used as heuristics for hard combinatorial optimization problems. Yet many learning-based methods rely on supervision, reinforcement learning, or gradient estimators, causing high computational cost, unstable training, or limited guarantees. Classical approximation algorithms provide worst-case guarantees but are non-differentiable and cannot adapt to structure in natural input distributions. We study this tradeoff through Uniform Facility Location (UniFL), a problem with applications in clustering, summarization, logistics, and supply chains. We propose a fully differentiable MPNN that incorporates approximation-algorithmic principles without solver supervision or discrete relaxations. The model has provable approximation guarantees and empirically improves on standard approximation algorithms, narrowing the gap to integer linear programming.
    
[^43]: 基于扩散模型的生成式压缩研究进展

    Advances in Diffusion-Based Generative Compression

    [https://arxiv.org/abs/2601.18932](https://arxiv.org/abs/2601.18932)

    本文系统综述了基于扩散模型的生成式有损压缩最新进展，重点介绍了图像压缩中通过嵌入表示编码并利用扩散模型迭代细化、从而在极低码率下实现逼真重建的方法。

    

    扩散模型及其相关的生成建模方法凭借强大的图像生成能力而广受欢迎，并在视觉媒体应用中取得了广泛成功。特别是，扩散方法为数据压缩开辟了新的途径，能够在极低码率下生成逼真的重建结果。本文对近期基于扩散的生成式有损压缩方法进行了统一综述，重点聚焦于图像压缩。这些方法通常将信源编码为嵌入表示，并在解码过程中利用扩散模型对其进行迭代细化，使重建结果近似服从真实数据分布。嵌入表示可以采取多种形式，通常通过辅助熵模型进行传输；近期的方法还探索了利用扩散模型本身通过信道模拟来进行信息传输。我们综述了代表性方法…

    arXiv:2601.18932v2 Announce Type: replace-cross  Abstract: Popularized by their strong image generation performance, diffusion and related methods for generative modeling have found widespread success in visual media applications. In particular, diffusion methods have enabled new approaches to data compression, where realistic reconstructions can be generated at extremely low bit-rates. This article provides a unifying review of recent diffusion-based methods for generative lossy compression, with a focus on image compression. These methods generally encode the source into an embedding and use a diffusion model to iteratively refine it during decoding, so that the reconstruction approximately follows the true data distribution. The embedding can take various forms and is typically transmitted via an auxiliary entropy model, and recent methods also explore the use of diffusion models themselves for information transmission via channel simulation. We review representative approaches thro
    
[^44]: BOCO：面向决策聚焦在线学习的贝叶斯在线上下文优化

    BOCO: Bayesian Online Contextual Optimization for Decision-Focused Online Learning

    [https://arxiv.org/abs/2511.20413](https://arxiv.org/abs/2511.20413)

    提出了BOCO框架，通过维护决策聚焦的贝叶斯后验分布并聚合多个预测来考虑参数不确定性，同时开发了基于粒子的在线推断算法，使在线决策聚焦学习在有限数据下更稳定且更易推广到异构优化问题。

    

    决策聚焦学习训练预测模型以优化下游决策，而非仅追求预测精度。尽管近期研究已将该范式扩展到基于流式数据的在线场景，但现有的在线决策聚焦学习方法通常仅维护点估计，且其基于梯度的更新要么需要可微的优化层，要么需要针对特定问题的替代损失函数。因此，这些方法在数据有限时可能不稳定，且难以推广到异构的优化问题。我们提出了贝叶斯在线上下文优化，这是一个在模型参数上维护决策聚焦后验分布的框架。BOCO 在制定决策时聚合由此产生的多个预测，从而将参数不确定性纳入考量。为了在动态演化的环境中跟踪该后验分布，我们开发了两种基于粒子的推断算法：序贯蒙特卡罗采样……（摘要原文在此处截断）

    arXiv:2511.20413v2 Announce Type: replace-cross  Abstract: \emph{Decision-focused learning} (DFL) trains predictive models to optimize downstream decisions rather than prediction accuracy alone. While recent studies have extended this paradigm to online settings with streaming data, existing online DFL methods generally maintain a point estimate, while their gradient-based updates require either a differentiable optimization layer or a problem-specific surrogate loss. Consequently, they can be unstable under limited data and difficult to apply across heterogeneous optimization problems. We introduce Bayesian Online Contextual Optimization (\texttt{BOCO}), a framework that maintains a decision-focused posterior over model parameters. \texttt{BOCO} aggregates the resulting predictions when prescribing decisions, thereby accounting for parameter uncertainty. To track this posterior in evolving environments, we develop two particle-based inference algorithms: a sequential Monte Carlo sampl
    
[^45]: 利用自然语言处理技术进行保险科技创新

    InsurTech innovation using natural language processing

    [https://arxiv.org/abs/2507.21112](https://arxiv.org/abs/2507.21112)

    本文展示了如何运用自然语言处理技术将非结构化文本转化为结构化数据，通过特征去偏、特征压缩和行业分类来丰富商业保险定价的费率因子，并为评估潜在风险提供新视角。

    

    随着保险科技（InsurTech）的迅速崛起，传统保险公司日益探索替代数据源和先进技术以保持其竞争优势。本文对自然语言处理（NLP）及其在保险运营中的新兴应用提供了概念性概述和实际案例研究，重点是将原始的非结构化文本转化为适合精算分析和决策的结构化数据。利用由保险科技行业合作伙伴提供的、能够丰富传统保险数据源的真实世界替代数据，我们应用多种NLP技术在商业保险场景中展示了特征去偏、特征压缩和行业分类。这些丰富的、源自文本的洞察不仅补充和优化了商业保险定价的传统费率因子，还为评估潜在风险提供了新颖的视角。

    arXiv:2507.21112v4 Announce Type: replace  Abstract: With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations, focusing on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate feature de-biasing, feature compression, and industry classification in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by 
    
[^46]: 加扰与噪声在量子系统时间信息处理中的作用

    Role of scrambling and noise in temporal information processing with quantum systems

    [https://arxiv.org/abs/2505.10080](https://arxiv.org/abs/2505.10080)

    本文揭示了基于高阶幺正设计的加扰量子储层在时间信息处理中的关键特性：无噪声时测量读出集中度不随迭代恶化、小储层可反复复用，但扩大规模需指数级测量开销否则损害泛化，且早期输入记忆随储层规模与迭代次数均呈指数衰减。

    

    加扰量子系统作为时间信息处理的有效基底引起了广泛关注。本文考虑了一个量子储层计算框架，该框架涵盖了利用量子系统的广泛物理计算模型。我们研究了以高阶幺正设计建模的加扰储层在无噪声和有噪声两种设置下模型的可扩展性与记忆保持能力。在无噪声情形下，我们证明测量读出会随储层规模增大而呈指数级集中，但令人惊讶的是，其并不会随储层迭代次数的增加而恶化。因此，尽管对量子数据反复复用小型加扰储层可能是可行的，但扩大问题规模会使泛化能力退化，除非能够承担指数级的测量次数开销。相比之下，早期输入和初始态的记忆随储层规模和储层迭代次数均呈指数衰减。在有噪声……

    arXiv:2505.10080v3 Announce Type: replace-cross  Abstract: Scrambling quantum systems have attracted attention as effective substrates for temporal information processing. Here we consider a quantum reservoir processing framework that captures a broad range of physical computing models with quantum systems. We examine the scalability and memory retention of the model with scrambling reservoirs modelled by high-order unitary designs in both noiseless and noisy settings. In the former regime, we show that measurement readouts become exponentially concentrated with increasing reservoir size, yet strikingly do not worsen with the reservoir iterations. Thus, while repeatedly reusing a small scrambling reservoir with quantum data might be viable, scaling up the problem size deteriorates generalization unless one can afford an exponential shot overhead. In contrast, the memory of early inputs and initial states decays exponentially in both reservoir size and reservoir iterations. In the noisy
    
[^47]: 局部化扩散模型

    Localized Diffusion Models

    [https://arxiv.org/abs/2505.04417](https://arxiv.org/abs/2505.04417)

    提出局部化扩散模型，通过利用目标分布中的局部性结构（稀疏条件依赖），以局部化神经网络估计得分函数，从而规避维数灾难并显著降低样本复杂度。

    

    扩散模型是各种生成任务中最先进的工具。然而，训练这些模型需要估计高维得分函数，这一任务在原则上会受到维数灾难的影响。因此，理解如何在这些模型中利用目标分布的低维结构十分重要。本文考虑局部性结构，它描述了目标随机变量之间的某些稀疏条件依赖关系。给定某种局部性结构，得分函数实际上是低维的，因此可以通过局部化的神经网络进行估计，从而显著降低样本复杂度。这一观察启发了局部化扩散模型，即使用局部化得分匹配损失在局部化假设空间内训练得分函数。我们证明这种局部化使扩散模型能够以与维度无关的方式规避维数灾难。

    arXiv:2505.04417v3 Announce Type: replace  Abstract: Diffusion models are state-of-the-art tools for various generative tasks. Yet training these models involves estimating high-dimensional score functions, a task that in principle suffers from the curse of dimensionality. It is therefore important to understand how low-dimensional structure in the target distribution can be exploited in these models. Here we consider locality structure, which describes certain sparse conditional dependencies among the target random variables. Given some locality structure, the score function is effectively low-dimensional, so that it can be estimated by a localized neural network with significantly reduced sample complexity. This observation motivates the localized diffusion model, where a localized score matching loss is used to train the score function within a localized hypothesis space. We prove that such localization enables diffusion models to circumvent the curse of dimensionality with dimensio
    
[^48]: 依赖数据下深度神经网络的统计性质

    Statistical Properties of Deep Neural Networks with Dependent Data

    [https://arxiv.org/abs/2410.11113](https://arxiv.org/abs/2410.11113)

    该论文为非平稳β-混合依赖数据下的深度神经网络估计量建立了非渐近误差理论，覆盖全连接与卷积网络且无需对权重施加有界或稀疏约束，并推广至非参数回归、逻辑回归和分位数回归等场景。

    

    本文为依赖数据下的深度神经网络（DNN）估计量建立了理论。为了提供适用于各类基于DNN的估计量的理论，我首先在可能非平稳的、取值于无界集合的β-混合数据条件下，针对一类一般性的估计问题，建立了非参数筛（sieve）估计量的理论误差和经验L²误差的非渐近概率界。随后，我将该理论应用于全连接和卷积DNN估计量，且无需对DNN权重施加有界性或稀疏性限制。对于这两类DNN，当待估计函数为Hölder光滑、数据为非平稳、次高斯且具有指数或多项式衰减的β-混合时，我推导了一般性结果。接着，我将这些结果具体应用于非参数回归、逻辑回归和分位数回归的设定。在指数β-混合条件下，所得估计量能够达到（原文在此处截断）……

    arXiv:2410.11113v4 Announce Type: replace-cross  Abstract: This paper develops theory for deep neural network (DNN) estimators under dependent data. To provide theory applicable to a variety of DNN-based estimators, I first establish nonasymptotic probability bounds on the theoretical and empirical $\mathcal{L}^{2}$-errors of nonparametric sieve estimators for a general class of estimation problems under possibly nonstationary $\beta$-mixing data taking values in unbounded sets. I then apply the theory to fully connected and convolutional DNN estimators without bounds or sparsity restrictions on the DNN weights. For both DNN classes, I derive general results when the function to be estimated is H\"older smooth and the data are nonstationary, subgaussian, and $\beta$-mixing with either exponential or polynomial decay. I then specialize these to nonparametric regression, logistic regression, and quantile regression settings. Under exponential $\beta$-mixing, the resulting estimators atta
    
[^49]: FastManly：用于Manly混合模型的EM梯度算法

    FastManly: An EM-Gradient Algorithm for Manly Mixture Models

    [https://arxiv.org/abs/2410.00848](https://arxiv.org/abs/2410.00848)

    FastManly方法通过在EM梯度算法中采用牛顿法（并推导出梯度和完整Hessian矩阵）替代传统EM中的Nelder-Mead优化，显著加快了Manly变换混合模型的计算速度。

    

    本文提出了一种更快速的Manly变换混合模型实现方法。该方法称为FastManly，在EM梯度算法中使用牛顿法进行优化，取代了传统EM算法中的Nelder-Mead方法。文中推导出了梯度以及完整的Hessian矩阵。仿真结果表明该方法性能更优，且速度提升显著。

    arXiv:2410.00848v2 Announce Type: replace-cross  Abstract: A faster implementation of mixtures of Manly transformations is proposed. This method, called FastManly, uses Newton's method for optimization in an EM gradient algorithm instead of Nelder-Mead in a traditional EM. A gradient and full Hessian are derived. Simulations show improved performance with noticeable speedups.
    
[^50]: 多网络数据的离群点检测

    Outlier Detection for Multi-Network Data

    [https://arxiv.org/abs/2205.06398](https://arxiv.org/abs/2205.06398)

    该论文针对节点共享而边各异的多网络数据（如神经影像大脑网络）提出了离群点检测方法，用于识别因数据质量差而产生的异常网络，防止其作为影响点污染后续统计分析。

    

    在神经科学研究领域，使用神经影像技术测量不同个体的大脑网络已成为常规做法。这些网络通常表示为邻接矩阵，矩阵中的每个单元格包含一对大脑区域之间连接性的概要。目前已出现一批统计文献，描述了分析此类多网络数据的方法——其中节点在各网络间是共同的，但边则各不相同。然而，对于离群点检测这一重要问题，几乎没有任何相关研究。特别是，某些受试者的神经影像数据质量非常差，以至于无法可靠地重建其网络。对于这些受试者，所得的邻接矩阵可能大部分为零，或呈现出与正常运作的大脑不一致的奇异模式。这些离群网络可能作为影响点，污染后续的统计分析。我们提出了一种简单的多网络数据离群点检测方法……

    arXiv:2205.06398v2 Announce Type: cross  Abstract: It has become routine in neuroscience studies to measure brain networks for different individuals using neuroimaging. These networks are typically expressed as adjacency matrices, with each cell containing a summary of connectivity between a pair of brain regions. There is an emerging statistical literature describing methods for the analysis of such multi-network data in which nodes are common across networks but the edges vary. However, there has been essentially no consideration of the important problem of outlier detection. In particular, for certain subjects, the neuroimaging data are so poor quality that the network cannot be reliably reconstructed. For such subjects, the resulting adjacency matrix may be mostly zero or exhibit a bizarre pattern not consistent with a functioning brain. These outlying networks may serve as influential points, contaminating subsequent statistical analyses. We propose a simple Outlier DetectIon for 
    
[^51]: 随机多面体描述符

    Random Polytope Descriptors

    [https://arxiv.org/abs/2009.13987](https://arxiv.org/abs/2009.13987)

    该论文提出了一类既通用又计算友好的随机多面体描述符，可用于数据分析中的分类与聚类任务，并允许用户在数据描述的紧致性与计算速度之间灵活权衡。

    

    我们引入了一类随机多面体，它同时推广了多种已知的构造方法。这类多面体不仅相当通用，而且在计算上也异常友好。我们说明了如何利用这些性质来完成数据分析中的分类与聚类任务。至关重要的是，我们的构造让用户能够在更紧凑的数据描述与更快的计算之间平滑地权衡。

    arXiv:2009.13987v3 Announce Type: replace  Abstract: We introduce a class of random polytopes which simultaneously generalizes several known constructions. While being fairly general, these polytopes are also computationally exceptionally benign. We indicate how these properties can be exploited for classification and clustering tasks in data analysis. Crucially, our construction lets users smoothly trade off between a tighter description of the data and faster computation.
    

