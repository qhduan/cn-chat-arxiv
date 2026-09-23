# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Decentralized Partially Observable Team Decision Methodology with Delayed Information Sharing](https://arxiv.org/abs/2609.26783) | 提出一种完全去中心化的团队决策方法，各成员仅利用本地私有信息和延迟共享的公共信息，通过低秩近似建模与最小二乘值迭代学习策略，无需集中式协调或训练即可近似达到集中式团队最优解。 |
| [^2] | [Automatic depth-based local center clustering via $\beta$-integrated local depth and adaptive grouping](https://arxiv.org/abs/2609.26748) | 提出了一种完全数据驱动的聚类方法A-DLCC，利用β-积分局部深度识别局部中心，并结合基于图论瓶颈路径思想的自适应合并准则，无需任何数值参数调优即可自动完成聚类。 |
| [^3] | [Optimal Sequential Annotations for Off-Policy Evaluation](https://arxiv.org/abs/2609.26707) | 该论文在真值标注预算受限的场景下，通过带缺失奖励的双鲁棒离线策略评估，刻画了序贯前向单调标注协议下方差最优的标注概率，并给出可行的批量自适应实现方法。 |
| [^4] | [Context-Adaptive Thresholding for Conditionally Representative Monitoring and Classification](https://arxiv.org/abs/2609.26652) | 本文提出一种将分类与监测阈值自适应于上下文协变量的方法，在保持误报率的同时实现条件代表性的标签预测，并能通过阈值规则近似推断未知的报警事件。 |
| [^5] | [On Basis Function Selection for Sparse Gaussian Process Regression](https://arxiv.org/abs/2609.26624) | 本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。 |
| [^6] | [Gap-Free Streaming PCA Beyond Rank-One Updates: Near-Optimal Rates and Applications to Differential Privacy](https://arxiv.org/abs/2609.26508) | 该论文对Oja算法在无特征间隙假设、超越秩一更新的最一般流式PCA问题上给出了近最优速率分析与近乎匹配的下界，并将其应用于次高斯数据下的差分隐私PCA。 |
| [^7] | [A Practical Guide on Graphical Model Validation](https://arxiv.org/abs/2609.26445) | 本文系统化整理了保险精算建模中常用的图形化与统计模型验证工具，并强调保单加权与风险暴露加权度量之间的区别是确保保费方案在正确尺度上校准的关键。 |
| [^8] | [SuperPCA: subspace analysis and an efficient algorithm for high-dimensional PCA](https://arxiv.org/abs/2609.26406) | 该论文证明样本协方差矩阵前若干特征向量张成的子空间远早于单个特征向量收敛时便已包含主要信号的关键信息，并据此提出了高维PCA的高效算法SuperPCA。 |
| [^9] | [Error Bounds for Statistical Estimators in BTL Model with Parametric Multivariate Utility Functions](https://arxiv.org/abs/2609.26326) | 该论文证明在BTL模型下，只要满足联合可识别性条件，标准极大似然估计量无需紧性约束或正则化即可保持有限，并达到由Fisher信息几何结构决定的极小极大最优误差界。 |
| [^10] | [Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining](https://arxiv.org/abs/2609.26290) | 提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。 |
| [^11] | [Improved Multiplayer Bandit Algorithm for Bernoulli Rewards](https://arxiv.org/abs/2609.26213) | 该论文用KL散度置信界取代Hoeffding式置信区间，提出三种多人老虎机算法（mKL-UCB、mKL-UCB-Intervals、mKL-DSEE），在伯努利奖励的信息不对称场景下获得了至少两倍、且在极端均值时更大的严格遗憾改进。 |
| [^12] | [Three Routes to One Answer: Reconciling AIPW, TMLE, and Double Machine Learning for Applied Researchers](https://arxiv.org/abs/2609.26142) | 该教程在 NHEFS 真实数据上，用同一套 Super Learner 库和相同的交叉拟合折，从单一影响函数手工实现 AIPW、TMLE 和 DML 三种估计量，阐明了应用研究者需对齐哪些要素，才能使不同方法与软件包对平均处理效应给出仅相差约 0.1 kg 的一致双重稳健估计。 |
| [^13] | [xWhyL: Causal Interactive Learning](https://arxiv.org/abs/2609.26037) | 该论文提出xWhyL框架，首次通过从解释中学习因果模型来连接因果性与可解释AI，将解释转化为与观测数据互补的学习信号，从而突破观测因果发现的局限。 |
| [^14] | [Conditional Tensor Diffusion: Distributional Counterfactual Learning and Inference](https://arxiv.org/abs/2609.25924) | 该论文提出“反事实Tucker扩散”（CFTDiff）方法，将处理掩码与潜在Tucker结构融入条件扩散模型，通过在低维核心上进行非线性得分学习来恢复高维张量数据中缺失对照结果的联合条件分布，并给出依赖Tucker秩等因素的高概率误差界，从而实现分布层面的反事实学习与推断。 |
| [^15] | [Beyond Scalar Sensitivity: Activation-Aware Mixed-Precision LLM Quantization with Cross-Layer Refinement](https://arxiv.org/abs/2609.25916) | 提出CASA两阶段方法，用基于Kronecker分解Hessian的激活感知敏感度度量替代不可靠的标量代理（其失真可高达10¹³），并结合跨层优化实现更精确的混合精度大语言模型量化。 |
| [^16] | [Statistical Gains from Looped Estimation under Parameter Budgets](https://arxiv.org/abs/2609.25778) | 该论文证明，在相同参数预算下，通过迭代间共享参数反复应用同一算子的循环估计器能够提升统计精度，揭示了参数—迭代—精度之间的权衡，并证明循环残差前馈网络与后层归一化 Transformer 可达到极小极大最优速率（至多相差对数因子）。 |
| [^17] | [Beyond Class Marginals: Bounding Rehearsal Gaps without Freezing Class Co-occurrence](https://arxiv.org/abs/2609.25735) | 该论文提出随机遍次回放（RPR）调度器，在不使用未来类别信息、不增加回放样本或前向传播且不冻结类共现的前提下，将类回放间隔严格限制在2*ceil(C/b)-1以内，从而缓解因类长时间缺席回放而导致的分类器偏置。 |
| [^18] | [Optimal Tradeoffs Between Network Size and Parameter Magnitude in Neural Approximation and Minimax Regression](https://arxiv.org/abs/2609.25710) | 本文建立了固定深度神经网络中网络宽度与参数幅值之间的最优权衡，证明使用有界1-Lipschitz二分-三角激活函数逼近β-Hölder类的最优误差率为[N²log(eNT)]^(-β/d)，且该速率对所有固定全局Hölder激活函数均匹配下界。 |
| [^19] | [On the Gradient Heterogeneity Dynamics of Adversarially Robust Federated Regression](https://arxiv.org/abs/2609.25705) | 该论文从每轮使用新鲜数据的线性和非线性回归统计模型出发推导联邦学习中的梯度异质性，将其分解为诚实客户端真实模型参数差异、有限样本标签噪声和初始化三个来源，并据此为系数为 O(f/n) 的 稳健聚合器建立理论保证。 |
| [^20] | [Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices](https://arxiv.org/abs/2609.25645) | 提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。 |
| [^21] | [Generalized Deep Regression for Repeated Measurements](https://arxiv.org/abs/2609.25605) | 该论文提出使用ReLU深度神经网络对具有重复测量的数据进行广义边际回归估计，证明了在β-Hölder光滑类上可达到n^{-1}+(nm)^{-2β/(2β+d)}的最优收敛速率，并提供了预言不等式及逐点推断的理论保证。 |
| [^22] | [Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis](https://arxiv.org/abs/2609.25576) | 提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。 |
| [^23] | [Direct Optimization of Generators for Search in Automated Theorem Proving](https://arxiv.org/abs/2609.25575) | 该论文通过对策略引导搜索的抽象，将计算对齐训练（CAT）扩展到树搜索场景，推导出可处理的搜索感知损失，并提出与搜索无关的均匀分配（UA）损失，从而直接优化用于自动定理证明搜索的生成器。 |
| [^24] | [Continuous Optimization for p-adic Models](https://arxiv.org/abs/2609.25501) | 该论文首次实现了 p-进参数机器学习模型的原生连续梯度下降，其核心创新是通过 Berkovich 仿射线这一路径连通的度量树扩张，使连续优化、反向传播以及动量和 Adam 等优化器在 p-进设定下成为可能。 |
| [^25] | [A Practical Recipe for Semi-Supervised Federated ASR: Online Pseudo-Labels with Server Update Stabilization](https://arxiv.org/abs/2609.25471) | 本文提出一种半监督联邦ASR的实用方案，通过服务器端标注数据更新来稳定客户端在线教师模型以生成可靠伪标签，从而显著缩小与完全监督联邦学习之间的性能差距。 |
| [^26] | [PICPIs: Prediction-Interval-Conditional Prediction Intervals](https://arxiv.org/abs/2609.25388) | 该论文提出了预测区间条件化预测区间这一新框架，通过自洽条件使预测区间同时定义预测值分层并保证该层内的平均结果落在同一区间内，从而在共形预测中填补了边际有效性分辨率不足与完全条件保证不可实现之间的空白。 |
| [^27] | [Penalized Nonreversible Langevin for Constrained Sampling](https://arxiv.org/abs/2609.25381) | 提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。 |
| [^28] | [Empirical Auditing of Edge-Private Graph Generators](https://arxiv.org/abs/2609.25155) | 该论文提出了一个针对边隐私图生成器的实证审计框架，通过统计有效的隐私损失下界来比较直接边、局部结构和GNN三类攻击，发现隐私泄露程度依赖于生成机制和网络本身，且GNN学习到的表示能揭示传统局部统计无法捕获的隐私信息。 |
| [^29] | [Variational objectives for amortized Bayesian inference in inverse problems: The role of posterior conditioning](https://arxiv.org/abs/2609.25145) | 该研究比较了逆问题摊销贝叶斯推断中三种VAE变分目标（反向KL、非对称JS和JS-Wasserstein），并通过广义Fisher基下的局部线性–高斯分析，揭示了后验条件化在弱可辨识参数方向上对后验精度和梯度行为的关键作用。 |
| [^30] | [The Informational Content in Lepto-Variance and Its Relation to Higher Moments](https://arxiv.org/abs/2609.25144) | 本文通过正态样本模拟研究1比特Lepto方差与样本方差、偏度和超额峰度的关系，发现正态分布的Lepto比率收敛于36.3%，且美国历史股票收益58%的变异性是无法被任何金融因子解释的1比特Lepto方差。 |
| [^31] | [The Probabilistic Structure of Large Language Models](https://arxiv.org/abs/2609.25134) | 本文以统一的概率论框架阐述大语言模型——将其视为令牌序列上的概率测度，把训练归结为最大似然估计、把生成归结为随机过程的顺序模拟，并揭示了KL散度的非对称性与幻觉现象及“统计合理性与真实性之别”之间的内在联系。 |
| [^32] | [FREESIA: Covariance-Aware Posterior Transport for Expressive and Scalable Data Assimilation](https://arxiv.org/abs/2609.25085) | 提出了一种无需训练、渐近精确的协方差感知后验传输方法 FREESIA，通过将预测交叉协方差嵌入基于流的传输，在高维稀疏观测下准确恢复未观测状态并保持非高斯多峰后验结构。 |
| [^33] | [What Does Chain-of-Thought Entropy Measure? A Channel Audit of Scaffolding, Routing, and Content](https://arxiv.org/abs/2609.25039) | 该论文提出通过指定脚手架词表子集，将思维链token熵精确分解为脚手架通道与内容通道，证明两种约定在判断最大分叉位置上可能相互矛盾，并发现脚手架最多可占原始高熵token集合的41%。 |
| [^34] | [Density-Ratio Rescoring for Imbalanced Classification](https://arxiv.org/abs/2609.23926) | 提出密度比重新评分（DRR）方法，通过调查整权法构造对偶分数并与基础分类器分数以固定权重融合，无需重采样或重新拟合即可在24个不平衡表格数据基准上一致提升平均精度。 |
| [^35] | [Interpretable AI with Local Distillation](https://arxiv.org/abs/2608.23538) | 本文提出局部蒸馏方法，利用黑盒教师模型在每个查询点指导正则化线性学生模型，通过定义局部性和锚定预测来实现高精度与可解释性的兼顾。 |
| [^36] | [A Quantum/Classical Example Oracle Separation for Making Things Up](https://arxiv.org/abs/2608.11648) | 本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。 |
| [^37] | [Sharp Characterization of Bias in Post-Bandit Inference](https://arxiv.org/abs/2608.01069) | 本文通过引入“有效探索率”这一关键量，精确刻画了老虎机算法后推断中样本均值偏差的算法来源，发现UCB1下偏差以极慢的1/√(log T)速率衰减，并揭示了探索带来的遗憾-偏差权衡。 |
| [^38] | [Chaos Is a LADDER: Domain Generalization Beyond Invariance via Reweighting](https://arxiv.org/abs/2607.26458) | 提出LADDER方法，将多域风格的“混沌”转化为定位未见目标域的阶梯，通过潜在域解耦与环境重加权，超越传统不变性原则实现更灵活的域泛化。 |
| [^39] | [Not All Objectives Are Born Equal: Priority-Constrained Descent for Hierarchical Multi-Objective Optimization](https://arxiv.org/abs/2606.29521) | 提出优先级约束下降（PCD）框架，通过单一参数控制最小失真，在保持主要目标下降方向的同时保证次要目标取得进展，且对目标缩放不变并给出两、三目标问题的精确闭式解。 |
| [^40] | [Unbiased Gradients, Moving Stability Boundaries: Exact Mini-Batch Geometry in Linear Self-Attention](https://arxiv.org/abs/2605.21292) | 该论文在线性自注意力的上下文回归中证明，无偏小批量随机梯度虽在期望上等价于全梯度，却会因采样曲率与目标相关性导致稳定性边界严格向外漂移，并给出了精确的边界穿越准则及几乎必然逃逸的证明。 |
| [^41] | [Tight Sample Complexity Bounds for Entropic Best Policy Identification](https://arxiv.org/abs/2605.13717) | 该论文通过基于KL探索奖励的前向模型算法改进了对指数效用的集中性控制，将熵风险敏感强化学习中最优策略识别的样本复杂度上界从 O(e^{2|β|H}) 降至与下界匹配的 O(e^{|β|H})，从而弥合了长期存在的指数级差距。 |
| [^42] | [Flow Matching for Count Data](https://arxiv.org/abs/2605.07746) | 提出了count-FM，一种基于连续时间生灭过程的计数数据流匹配框架，通过免模拟训练条件转移速率，在计数空间中高效实现任意计数分布间的分布传输，适用于单细胞RNA测序等高维计数数据场景。 |
| [^43] | [SPLICE: Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting](https://arxiv.org/abs/2605.00126) | SPLICE将JEPA潜空间生成式插补与自适应保形推断相结合，为电力负荷时间序列修复同时提供高质量重建与有限样本覆盖率保证的预测区间，且流匹配变体实现5-10倍加速。 |
| [^44] | [Conditional Distributional Treatment Effects: Doubly Robust Estimation and Testing](https://arxiv.org/abs/2603.16829) | 本文提出了捕捉条件分布处理效应的新估计对象及其双重稳健的极小极大最优估计方法，并开发了首个具有有效第一类错误控制和一致性的条件潜在结果分布同质性检验。 |
| [^45] | [Communication-Efficient Byzantine-Robust Federated Conformal Prediction via Partial Sharing](https://arxiv.org/abs/2602.18396) | 提出PRISM-FCP框架，通过部分模型共享以M/D比例衰减训练阶段拜占庭投毒攻击的扰动能量，并结合基于直方图的过滤抵御对抗性校准提交，实现了兼顾两个阶段安全性与通信效率的联邦保形预测。 |
| [^46] | [Efficient and scalable clustering of survival curves](https://arxiv.org/abs/2512.16481) | 该论文提出一种结合k-means聚类与log-rank检验的新方法，无需计算昂贵的自助法重抽样，即可高效且可扩展地对生存曲线进行聚类分析。 |
| [^47] | [Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits](https://arxiv.org/abs/2510.10730) | 该论文提出了一个统一的集成采样算法框架，为广义线性和神经两类非线性上下文老虎机提供了可证明的后悔界，其中GLM-ES达到了与最先进随机探索算法相匹配的性能。 |
| [^48] | [A variational approach to dimension-free self-normalized concentration](https://arxiv.org/abs/2508.06483) | 本文提出一种变分方法，为向量值随机过程的"sub-ψ"类过程建立无维数依赖的自归一化集中界，推广了经典结果，并首次给出无维数依赖的自归一化经验 Bernstein 不等式。 |
| [^49] | [Likelihood Based Inference in Fully and Partially Observed Exponential Family Graphical Models with Intractable Normalizing Constants](https://arxiv.org/abs/2404.17763) | 本文针对具有难解归一化常数的完全观测与部分观测（含潜变量）指数族图模型，提出了基于似然的统计推断方法。 |

# 详细

[^1]: 具有延迟信息共享的去中心化部分可观测团队决策方法

    A Decentralized Partially Observable Team Decision Methodology with Delayed Information Sharing

    [https://arxiv.org/abs/2609.26783](https://arxiv.org/abs/2609.26783)

    提出一种完全去中心化的团队决策方法，各成员仅利用本地私有信息和延迟共享的公共信息，通过低秩近似建模与最小二乘值迭代学习策略，无需集中式协调或训练即可近似达到集中式团队最优解。

    

    我们研究了具有低秩潜在动力学和未知系统模型的去中心化部分可观测团队决策问题。所提出的框架将团队理论等价性与低秩模型表示相结合，在不需要转移模型先验知识的情况下，解决部分可观测马尔可夫决策过程中的协作决策问题。每个团队成员基于本地私有信息和团队内共享的延迟公共信息进行决策。仅利用这些可用信息，每个成员学习一个近似的低秩马尔可夫决策过程，并应用最小二乘值迭代来计算其策略。由此得到一种完全去中心化的学习与规划算法，既不需要集中式协调器，也不需要集中式训练。我们证明了所得的成员侧解能够近似集中式团队解：尽管存在部分可观测性、未知动力学和延迟通信等挑战。

    arXiv:2609.26783v1 Announce Type: cross  Abstract: We study decentralized partially observable team decision problems with low-rank latent dynamics and unknown system models. The proposed framework combines team-theoretic equivalence with low-rank model representations to address cooperative decision-making in partially observable Markov decision processes without prior knowledge of the transition model. Each team member makes decisions based on local private information and delayed common information shared across the team. Using only this available information, each member learns an approximate low-rank Markov decision process and applies least-squares value iteration to compute its policy. This yields a fully decentralized learning and planning algorithm that requires neither a centralized coordinator nor centralized training. We show that the resulting member-side solutions approximate the centralized team solution: despite partial observability, unknown dynamics, and delayed commo
    
[^2]: 基于β-积分局部深度与自适应分组的自动深度局部中心聚类

    Automatic depth-based local center clustering via $\beta$-integrated local depth and adaptive grouping

    [https://arxiv.org/abs/2609.26748](https://arxiv.org/abs/2609.26748)

    提出了一种完全数据驱动的聚类方法A-DLCC，利用β-积分局部深度识别局部中心，并结合基于图论瓶颈路径思想的自适应合并准则，无需任何数值参数调优即可自动完成聚类。

    

    聚类是一种将无标签数据划分为若干组的无监督学习技术。大多数现有方法需要用户指定参数，例如聚类数量或邻域大小。与此不同，我们提出了基于深度的自动局部中心聚类（A-DLCC），这是一种完全数据驱动的方法，无需数值参数调优。A-DLCC使用β-积分局部深度来识别稳定的代表点，即在多个局部性层级上始终处于中心位置的点，称为局部中心，并按其代表性进行排序。每个局部中心会诱导出一组相似点，组级相似性通过我们提出的一种称为组级局部相似性的非参数度量来衡量。为指导合并过程，我们引入了图论中的瓶颈路径思想，这构成了我们自适应合并准则的基础。基于该准则，我们设计了一种单一的凝聚规则，其中一个组要么被……

    arXiv:2609.26748v1 Announce Type: cross  Abstract: Clustering is an unsupervised learning technique that partitions unlabeled data into groups. Most existing methods require user-specified parameters, such as the number of clusters or neighborhood size. Conversely, we propose automatic depth-based local center clustering (A-DLCC), a fully data-driven method that eliminates numerical parameter tuning. A-DLCC uses the $\beta$-integrated local depth to identify stable exemplars, points consistently central across multiple locality levels, termed local centers, which are ranked by their representativeness. Each local center induces a group of similar points, with group-level similarity measured by a proposed nonparametric metric called group-level local similarity. To guide merging, we incorporate the bottleneck path idea from graph theory, which forms the basis of our adaptive merging criterion. Based on this criterion, we design a single agglomeration rule in which a group is either abso
    
[^3]: 离线策略评估的最优序贯标注

    Optimal Sequential Annotations for Off-Policy Evaluation

    [https://arxiv.org/abs/2609.26707](https://arxiv.org/abs/2609.26707)

    该论文在真值标注预算受限的场景下，通过带缺失奖励的双鲁棒离线策略评估，刻画了序贯前向单调标注协议下方差最优的标注概率，并给出可行的批量自适应实现方法。

    

    离线强化学习与离线策略评估旨在部署之前，基于回顾性收集的数据评估动态治疗规则。在近期的AI应用中，状态和奖励信息以复杂的文本或图像形式记录，诸如“LLM作为评判者”等最新AI技术可以对其进行标注，但偏差未知。专家标注虽然可用，但成本更高。例如，通过廉价但不完美的分类器进行安全分类，与昂贵的专家审核相对比。我们展示了如何通过带缺失奖励的双鲁棒离线策略评估来利用有限的真值数据标注预算，并为序贯离线策略评估优化方差最优的标注概率，其中目标策略价值由标注数据进行估计。我们刻画了序贯前向单调标注协议的最优标注概率，并提供了一种可行的批量自适应实现方案。

    arXiv:2609.26707v1 Announce Type: cross  Abstract: Offline reinforcement learning and off-policy evaluation evaluates dynamic treatment rules based on retrospectively collected data prior to deployment. In recent AI applications, state and reward information is recorded as complex text or image, which recent AI advancements such as LLM-as-a-judge can label with unknown bias. Expert annotation may be available but at a higher cost. For example, safety classification via cheap but imperfect classifiers vs. expensive expert review. We show how a limited budget for ground-truth data-annotation can be used via doubly-robust OPE with missing rewards, and we optimize variance-optimal annotation probabilities for sequential off-policy evaluation, where the target policy value is estimated from annotated data. We characterize the optimal annotation probabilities for sequential forward-monotone annotation protocols, and provide a feasible batch-adaptive implementation. Our work is motivated by a
    
[^4]: 用于条件代表性监测与分类的上下文自适应阈值方法

    Context-Adaptive Thresholding for Conditionally Representative Monitoring and Classification

    [https://arxiv.org/abs/2609.26652](https://arxiv.org/abs/2609.26652)

    本文提出一种将分类与监测阈值自适应于上下文协变量的方法，在保持误报率的同时实现条件代表性的标签预测，并能通过阈值规则近似推断未知的报警事件。

    

    通常，分类器和监测程序是通过优化诸如误分类率之类的目标函数从有标签数据中训练得到的。这可能导致在给定重要外部变量的条件下，结果（标签）的条件分布不具代表性，与总体中的条件分布不同。我们展示了如何修改任何给定的阈值型分类器或监测规则，通过将阈值自适应于协变量 Z（即上下文）来分配灵敏度，同时保持误报率，从而实现具有代表性的条件标签预测。在报警事件未知的情况下，该方法还允许通过阈值规则（近似地）推断该事件。该方法通过计算成本低廉的非参数估计程序实现，其性质通过非渐近误差界以及包括经验过程理论在内的渐近分布理论进行了研究。

    arXiv:2609.26652v1 Announce Type: cross  Abstract: Commonly, classifiers and monitoring procedures are trained from labeled data by optimizing an objective such as the misclassification rate. This may lead to unrepresentative conditional distributions of the outcome (the labels) given important external variables, different from the conditional laws in the population. We show how to modify any given threshold-type classifier resp. monitoring rule to achieve representative conditional label prediction by using adapting the threshold to a covariate $Z$ (the context) to distribute sensitivity while maintaining the false alarm rate. In case that the alarm event is unknown, this approach also allows to (approximately) infer the event in terms of a thresholding rule. The approach is implemented by a computationally cheap nonparametric estimation procedure, and its properties are studied in terms of nonasymptotic error bounds and asymptotic distribution theory including empirical process theo
    
[^5]: 关于稀疏高斯过程回归中基函数选择的研究

    On Basis Function Selection for Sparse Gaussian Process Regression

    [https://arxiv.org/abs/2609.26624](https://arxiv.org/abs/2609.26624)

    本文从信息论视角提出三种基函数选择准则，用于在稀疏高斯过程回归中依据数据挑选最相关的基函数，以替代传统的固定截断策略，从而更高效地利用有限的计算预算。

    

    稀疏高斯过程通过在输入空间上用固定基函数集 {φ_j} 的适当展开来替代核函数，从而实现 O(N) 的推断。在给定计算预算 M ≪ N 的情况下，从业者通常习惯性地将基截断为前 M 个基函数。然而，从形式上看，并没有任何限制阻止人们只选择那些对当前数据真正重要的 M 个基函数。这样做可以避免将计算预算浪费在没有信号的基函数上，但这需要一个能够对候选基函数进行排序的准则。我们从基函数选择问题的信息论视角出发，提出了三种这样的准则。每种准则分别对应于选择时所处的不同知识状态：无数据状态、无先验状态以及介于两者之间的状态。随后，我们在六个 UCI 回归基准数据集上，针对三种基函数族（包括希尔伯特空间高斯过程 HSGP 等），研究了截断策略与选择策略的性能表现。

    arXiv:2609.26624v1 Announce Type: cross  Abstract: Sparse Gaussian processes achieve $O(N)$ inference by replacing the kernel with an appropriate expansion in a fixed basis $\{\phi_j\}$ on the input space. Given a compute budget $M \ll N$, practitioners conventionally truncate the basis to its first $M$ entries. Nothing in the formalism, however, prevents one from selecting only those $M$ basis functions that matter for the data at hand. This would avoid spending budget on basis functions where there is no signal, but it requires a criterion for ranking the candidates. We propose three such criteria derived from an information-theoretic view of the basis-function selection problem. Each criterion matches a different state of knowledge at selection time: a no-data state, a no-prior state, and an in-between state. We then study the performance of truncation versus selection strategies on six UCI regression benchmarks across three basis families: Hilbert-space Gaussian processes (HSGP), v
    
[^6]: 超越秩一更新的无间隙流式主成分分析：近最优速率及差分隐私应用

    Gap-Free Streaming PCA Beyond Rank-One Updates: Near-Optimal Rates and Applications to Differential Privacy

    [https://arxiv.org/abs/2609.26508](https://arxiv.org/abs/2609.26508)

    该论文对Oja算法在无特征间隙假设、超越秩一更新的最一般流式PCA问题上给出了近最优速率分析与近乎匹配的下界，并将其应用于次高斯数据下的差分隐私PCA。

    

    流式主成分分析（PCA）旨在通过对数据流进行单次遍历来恢复主导谱子空间。我们对广泛使用的Oja算法 [Oja82] 在该问题最一般的无间隙变体上给出了新的分析，即不对底层均值矩阵做任何特征间隙假设，并辅以近乎匹配的下界。先前在流式PCA上实现近最优速率的工作要么需要间隙假设 [JJK+16, HNWW21]，要么仅限于秩一更新 [AZL17, Lia23]。我们的证明仅使用了单个随机更新的二阶矩界，绕过了先前近最优分析所需的几乎必然界以及类似的离线矩阵Bernstein界。我们还将结果扩展到基于瑞利商的近似PCA概念，解决了 [JJK+16] 提出的一个开放问题。作为主要应用，我们为次高斯数据给出了无间隙的差分隐私PCA保证，设定……（摘要原文在此处截断）

    arXiv:2609.26508v1 Announce Type: new  Abstract: Streaming principal component analysis (PCA) seeks to recover a leading spectral subspace in a single pass over a data stream. We give a new analysis of the ubiquitous Oja's algorithm [Oja82] for the most general, gap-free variant of this problem, where no eigengap assumptions are made on the underlying mean matrix, complemented by a nearly-matching lower bound. Prior works achieving near-optimal rates for streaming PCA either required gap assumptions [JJK+16, HNWW21], or were limited to rank-one updates [AZL17, Lia23]. Our proof only uses a second moment bound on the individual stochastic updates, bypassing the almost sure bounds needed by prior near-optimal analyses, and the analogous offline matrix Bernstein bound. We also extend our result to a Rayleigh quotient notion of approximate PCA, addressing an open question of [JJK+16]. As our main application, we give gap-free differentially private PCA guarantees for sub-Gaussian data, set
    
[^7]: 图形化模型验证实用指南

    A Practical Guide on Graphical Model Validation

    [https://arxiv.org/abs/2609.26445](https://arxiv.org/abs/2609.26445)

    本文系统化整理了保险精算建模中常用的图形化与统计模型验证工具，并强调保单加权与风险暴露加权度量之间的区别是确保保费方案在正确尺度上校准的关键。

    

    本手稿对一般保险精算建模中最常用的模型验证工具进行了形式化。这些工具包括图形化工具，如校准图、实际与预期对比图、提升图、Murphy图，以及经典统计工具，如Bregman损失、偏差损失、基本损失、Murphy分解和Gini分数。文中特别强调了在校准能力和区分能力的研究中，应采用保单加权还是风险暴露加权的总体度量。这一区别对于确保保费方案在正确的尺度上进行校准至关重要。

    arXiv:2609.26445v1 Announce Type: cross  Abstract: This manuscript formalizes the most popular model validation tools used in general insurance actuarial modeling. These include graphical tools like calibration plots, actual-vs-expected plots, lift charts, Murphy diagrams, as well as classical statistical tools such as Bregman losses, deviance losses, elementary losses, Murphy's decomposition and Gini scores. Particular emphasis is placed on whether calibration and discrimination are studied under a policy-weighted or an exposure-weighted population measure. This distinction is crucial in ensuring that premium schemes are calibrated on the correct scale.
    
[^8]: SuperPCA：子空间分析与高维主成分分析的高效算法

    SuperPCA: subspace analysis and an efficient algorithm for high-dimensional PCA

    [https://arxiv.org/abs/2609.26406](https://arxiv.org/abs/2609.26406)

    该论文证明样本协方差矩阵前若干特征向量张成的子空间远早于单个特征向量收敛时便已包含主要信号的关键信息，并据此提出了高维PCA的高效算法SuperPCA。

    

    主成分分析（PCA）是许多应用中用于降低数据维度的基本工具。PCA通过计算样本协方差矩阵的特征向量，找到包含数据大部分变异性的少数信号方向。在这项工作中，我们专注于尖峰协方差模型，其中数据向量由少数正交信号加上各向同性高斯噪声定义，我们的目标是估计其中一个或多个主要信号。我们的主要理论发现是：样本协方差矩阵的前若干个特征向量所张成的子空间，早在各个特征向量 individually 收敛到总体主成分之前，就已包含关于目标信号的重要信息。为了证明这一点，我们利用扰动理论推导了目标总体信号张成的子空间与从样本中获得的子空间之间夹角的后验上界。

    arXiv:2609.26406v1 Announce Type: cross  Abstract: Principal component analysis (PCA) is a fundamental tool to reduce the dimensionality of the data in many applications. PCA finds a few signal directions that contain most of the variability of the data by computing the eigenvectors of the sample covariance matrix. In this work, we focus on the spiked covariance model, in which the data vectors are defined by a few orthogonal signals plus an isotropic Gaussian noise, and our goal is to estimate one or more of the leading signals. Our main theoretical finding is that the subspace spanned by several leading eigenvectors of the sample covariance matrix contains significant information about the desired signals long before the individual eigenvectors converge to the population principal components. To prove this, we derive a posteriori bounds for the angle between the subspace spanned by the desired population signals and the subspace obtained from the sample using perturbation theory for 
    
[^9]: 具有参数化多元效用函数的BTL模型中统计估计量的误差界

    Error Bounds for Statistical Estimators in BTL Model with Parametric Multivariate Utility Functions

    [https://arxiv.org/abs/2609.26326](https://arxiv.org/abs/2609.26326)

    该论文证明在BTL模型下，只要满足联合可识别性条件，标准极大似然估计量无需紧性约束或正则化即可保持有限，并达到由Fisher信息几何结构决定的极小极大最优误差界。

    

    我们研究Bradley-Terry-Luce（BTL）模型下的偏好引出问题，其中真实的部分价值向量是未知的，需要作为参数利用所引出的偏好信息进行估计。所选成对查询的集合在备选方案集合上是非均匀的、确定性的且任意的，只要其满足联合可识别性条件即可。我们重点研究在什么情况下标准的极大似然估计量（MLE）是有限的，并且在不对可行域施加显式紧性约束或不使用外部正则化项的情况下具有尖锐的误差界。为此，我们在标准的有界动态范围条件下推导了极小极大下界，并发现经典Cramér-Rao下界中相同的Fisher信息几何结构支撑着该估计问题的有限样本难度。通过将似然得分方程的非渐近展开与不动点局部化论证相结合，我们……（摘要在此处截断）

    arXiv:2609.26326v1 Announce Type: cross  Abstract: We study preference elicitation under the Bradley-Terry-Luce (BTL) model where the true partworth vector is unknown and has to be estimated as a parameter with elicited preference information. The set of selected pairwise queries is non-uniform, deterministic, and arbitrary over a collection of alternatives, provided that it satisfies a joint identifiability condition. We focus on understanding when the canonical maximum likelihood estimator (MLE) is finite and admits sharp error bounds without explicit compactness constraints on the feasible set or external regularizers. To this end, we derive minimax lower bounds under the standard bounded dynamic range condition, and find that the same Fisher-information geometry in the classic Cram\'er-Rao lower bounds underpins the finite-sample difficulty of the estimation problem. By combining a non-asymptotic expansion of the likelihood score equation with a fixed-point localization argument, w
    
[^10]: 学习波动：因果表格预训练的统计基础

    Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining

    [https://arxiv.org/abs/2609.26290](https://arxiv.org/abs/2609.26290)

    提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。

    

    因果表格基础模型在多个合成机制之间摊销效应估计，但潜在效应监督只奖励后验收缩，而非直接编码固定部署总体中所需的重复样本响应。我们提出波动监督预训练（FSP）：每个合成表格由其平均处理效应加上其有效影响函数的波动来标注，而部署时仍只需一次冻结的前向传播。沿着路径 $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$，我们证明了一个端点相变：每个固定的 $\lambda<1$ 都会保留 $(1-\lambda)^2/n$ 阶的标签模糊性，而完全波动使高斯标签变得可观测，并将最优有限层因果标签预测风险降至 $n^{-2}$ 阶。一个有限预训练界综合了标签误差、网络误差、回合采样误差与优化误差；其产生的采样缺陷控制了固定机制偏差、均方……（摘要截断）

    arXiv:2609.26290v1 Announce Type: cross  Abstract: Causal tabular foundation models amortize effect estimation across synthetic mechanisms, but latent-effect supervision rewards posterior shrinkage instead of directly encoding the repeated-sample response needed in a fixed deployment population. We introduce fluctuation-supervised pretraining (FSP): each synthetic table is labeled by its average treatment effect plus its efficient influence-function fluctuation, while deployment remains a single frozen forward pass. Along the path $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$, we prove an endpoint transition: every fixed $\lambda<1$ retains label ambiguity of order $(1-\lambda)^2/n$, whereas full fluctuation makes the Gaussian label observable and reduces optimal finite-stratum causal label-prediction risk to order $n^{-2}$. One finite-pretraining bound combines label, network, episode-sampling, and optimization errors; its resulting sampling defect controls fixed-mechanism bias, mean sq
    
[^11]: 针对伯努利奖励的改进型多人老虎机算法

    Improved Multiplayer Bandit Algorithm for Bernoulli Rewards

    [https://arxiv.org/abs/2609.26213](https://arxiv.org/abs/2609.26213)

    该论文用KL散度置信界取代Hoeffding式置信区间，提出三种多人老虎机算法（mKL-UCB、mKL-UCB-Intervals、mKL-DSEE），在伯努利奖励的信息不对称场景下获得了至少两倍、且在极端均值时更大的严格遗憾改进。

    

    我们研究了伯努利奖励下具有信息不对称的多人多臂老虎机问题，涵盖三种信息结构：动作不对称、奖励不对称以及两者兼有的不对称。通过用基于Kullback–Leibler（KL）散度的置信界取代先前工作中的Hoeffding式置信区间，我们在每种情况下都获得了严格更紧的遗憾保证。我们提出了mKL-UCB、mKL-UCB-Intervals和mKL-DSEE三种算法，并由Pinsker不等式证明改进因子至少为2，而当奖励均值接近0或1时改进幅度远大于此。对于奖励不对称的情形，我们证明两条臂的KL置信区间在确定数量的采样后必然分离，且M个独立玩家可进一步加速淘汰过程。

    arXiv:2609.26213v1 Announce Type: cross  Abstract: We study the multiplayer multi-armed bandit problem with information asymmetry under Bernoulli rewards, for three information structures: asymmetry in actions, in rewards, and in both. Replacing the Hoeffding-style confidence intervals of prior work with Kullback--Leibler (KL) divergence-based bounds gives strictly tighter regret guarantees in each case. We propose \texttt{mKL-UCB}, \texttt{mKL-UCB-Intervals} and \texttt{mKL-DSEE}, and show that the improvement factor is at least two by Pinsker's inequality and far larger when reward means are near zero or one. For asymmetry in rewards we prove that two arms' KL intervals separate after a deterministic number of samples, and that $M$ independent players accelerate elimination further.
    
[^12]: 三条路径，一个答案：为应用研究者调和 AIPW、TMLE 与双重机器学习

    Three Routes to One Answer: Reconciling AIPW, TMLE, and Double Machine Learning for Applied Researchers

    [https://arxiv.org/abs/2609.26142](https://arxiv.org/abs/2609.26142)

    该教程在 NHEFS 真实数据上，用同一套 Super Learner 库和相同的交叉拟合折，从单一影响函数手工实现 AIPW、TMLE 和 DML 三种估计量，阐明了应用研究者需对齐哪些要素，才能使不同方法与软件包对平均处理效应给出仅相差约 0.1 kg 的一致双重稳健估计。

    

    增广逆概率加权（AIPW）、目标最大似然估计（TMLE）以及双重/去偏机器学习（DML）是通向平均处理效应同一有效影响函数的三条路径——这一已成定论的理论我们仅作为背景。本教程的贡献在于其在真实数据上开展的、共享干扰参数的实操性调和：实践者究竟需要对齐哪些要素，才能让这三条路径及其软件包得出一致的结果。我们在公开的 NHEFS 数据（n=1566）上研究戒烟对体重变化的影响，使用同一套 Super Learner 库和完全相同的交叉拟合折，从同一个影响函数出发手工构建全部三个估计量，涵盖全样本、交叉拟合与双重交叉拟合三种变体；由此得到的六个双重稳健估计仅落在 3.32–3.42 kg 的狭窄区间内，与既有基准一致。随后，我们将同一估计对象在我们的计算引擎与 tmle、AIPW、DoubleML 以及 tmle3 等软件包之间进行调和……

    arXiv:2609.26142v1 Announce Type: cross  Abstract: Augmented inverse-probability weighting (AIPW), targeted maximum likelihood estimation (TMLE), and double/debiased machine learning (DML) are three routes to the same efficient influence function for the average treatment effect --- settled theory we treat as background. This tutorial's contribution is its worked, shared-nuisance reconciliation on real data: what a practitioner must actually match for the routes, and the software packages, to agree. Working the effect of smoking cessation on weight change in the open NHEFS data (n=1566) with one shared Super Learner library and identical cross-fitting folds, we build all three estimators by hand from one influence function, in full-sample, cross-fit, and double-cross-fit variants; the six resulting doubly-robust estimates span only 3.32--3.42 kg, consistent with the established benchmark. We then reconcile the same estimand across our engine and the tmle, AIPW, DoubleML, and tmle3 pack
    
[^13]: xWhyL：因果交互学习

    xWhyL: Causal Interactive Learning

    [https://arxiv.org/abs/2609.26037](https://arxiv.org/abs/2609.26037)

    该论文提出xWhyL框架，首次通过从解释中学习因果模型来连接因果性与可解释AI，将解释转化为与观测数据互补的学习信号，从而突破观测因果发现的局限。

    

    解释是因果推理的核心，认知科学早已确立人类解释的驱动力本身就是学习因果关系的一种机制。尽管如此，从这些溯因信号中学习在人工智能领域基本被忽视。虽然可解释人工智能日益依赖因果模型来生成解释，但解释能为因果性做什么这一相反方向仍在很大程度上未被探索。为填补这一空白，我们提出了xWhyL，一个通过从解释中学习因果模型来连接因果性与可解释人工智能的形式化框架。我们发展了一套数学理论，将解释转化为与观测数据互补的学习信号，并展示它如何能够克服观测因果发现的局限。由于解释可能源自错误信念并与数据冲突——我们将这种张力称为"因果拔河"（Causal Tug-of-War），我们证明了该框架能够……

    arXiv:2609.26037v1 Announce Type: new  Abstract: Explanations are central to causal reasoning, and cognitive science has long established that the human drive to explain is itself a mechanism for learning about causality. Despite this, learning from those abductive signals is largely ignored in artificial intelligence. While explainable AI (XAI) increasingly draws on causal models to generate explanations, the converse direction about what explanations can do for causality remains largely unexplored. To fill this gap, we propose xWhyL, a formal framework connecting causality and XAI by learning causal models from explanations. We develop a mathematical theory that translates explanations into a learning signal complementary to observational data, and demonstrate how it enables overcoming the limits of observational causal discovery. As explanations can be derived from incorrect beliefs and clash with data, a tension we call the Causal Tug-of-War, we prove conditions under which our fra
    
[^14]: 条件张量扩散：分布型反事实学习与推断

    Conditional Tensor Diffusion: Distributional Counterfactual Learning and Inference

    [https://arxiv.org/abs/2609.25924](https://arxiv.org/abs/2609.25924)

    该论文提出“反事实Tucker扩散”（CFTDiff）方法，将处理掩码与潜在Tucker结构融入条件扩散模型，通过在低维核心上进行非线性得分学习来恢复高维张量数据中缺失对照结果的联合条件分布，并给出依赖Tucker秩等因素的高概率误差界，从而实现分布层面的反事实学习与推断。

    

    因果推断可以指导运营和管理决策，但在高维面板或张量数据环境中仍然充满挑战，因为此时决策可能依赖于缺失的对照结果的联合条件分布。我们提出了“反事实Tucker扩散”（CFTDiff），该方法将处理掩码和潜在Tucker结构整合进条件扩散模型中，通过对低维核心进行高效的非线性得分学习，在给定已观测对照结果的条件下恢复这一分布。掩码Tucker得分在保留张量各模式之间依赖关系的同时，将非线性得分学习的维度从各模式维度的乘积降低到规模小得多的Tucker秩的乘积。我们为条件得分估计建立了高概率误差界，该误差界取决于Tucker秩、最大模式维度以及经因子强度调整的缺失结果数量，并展示了这些误差界如何转化为...

    arXiv:2609.25924v1 Announce Type: cross  Abstract: Causal inference guides operational and managerial decisions but remains challenging in high-dimensional panel or tensor settings, where decisions may depend on the joint conditional distribution of missing control outcomes. We develop \emph{Counterfactual Tucker Diffusion} (\CFTDiff), which integrates the treatment mask and latent Tucker structure into conditional diffusion to recover this distribution given observed control outcomes through efficient nonlinear score learning in a low-dimensional core. The masked Tucker score preserves dependence across tensor modes while reducing the dimension of nonlinear score learning from the product of mode dimensions to the much smaller product of Tucker ranks. We establish high-probability error bounds for conditional score estimation that depend on the Tucker ranks, largest mode dimension, and the factor-strength-adjusted number of missing outcomes, and show how these bounds translate into re
    
[^15]: 超越标量敏感度：基于跨层优化的激活感知混合精度大语言模型量化

    Beyond Scalar Sensitivity: Activation-Aware Mixed-Precision LLM Quantization with Cross-Layer Refinement

    [https://arxiv.org/abs/2609.25916](https://arxiv.org/abs/2609.25916)

    提出CASA两阶段方法，用基于Kronecker分解Hessian的激活感知敏感度度量替代不可靠的标量代理（其失真可高达10¹³），并结合跨层优化实现更精确的混合精度大语言模型量化。

    

    混合精度权重量化通常被表述为多选择背包问题（MCKP），然而现有的求解器依赖于标量敏感度代理，这些代理将每个权重矩阵的Hessian矩阵压缩为单个数值，并将每个模块独立对待。我们证明，即使是最优的标量代理，相对于完整的激活感知二次形式，也会产生高达 √(κ(A)κ(B)) 的乘性失真，其中 κ(A) 和 κ(B) 分别表示输入侧和输出侧Hessian因子的条件数。对于典型的大语言模型模块，该界限在 10¹ 到 10¹³ 之间变化，使得模块间的敏感度排序不可靠。为了解决这些局限性，我们提出了跨层激活感知敏感度分配（CASA），这是一种两阶段方法。在第一阶段，标量代理被由Kronecker分解的Hessian导出的激活感知度量所取代，从而将MCKP简化为一个……

    arXiv:2609.25916v1 Announce Type: new  Abstract: Mixed-precision weight quantization is commonly formulated as a Multiple-Choice Knapsack Problem (MCKP), yet existing solvers rely on scalar sensitivity proxies that collapse each weight matrix's Hessian into a single number and treat every module independently. We prove that even the optimal scalar proxy incurs multiplicative distortion up to $\sqrt{\kappa(\mathbf{A})\kappa(\mathbf{B})}$ relative to the full activation-aware quadratic, where $\kappa(\mathbf{A})$ and $\kappa(\mathbf{B})$ denote the condition numbers of the input- and output-side Hessian factors. This bound varies from $10^1$ to $10^{13}$ for typical LLM modules, making inter-module sensitivity ranking unreliable. To address these limitations, we propose Cross-layer Activation-aware Sensitivity Allocation (CASA), a two-phase method. In Stage 1, the scalar proxy is replaced by an activation-aware metric derived from the Kronecker-factored Hessian, reducing the MCKP to a fo
    
[^16]: 参数预算下循环估计的统计增益

    Statistical Gains from Looped Estimation under Parameter Budgets

    [https://arxiv.org/abs/2609.25778](https://arxiv.org/abs/2609.25778)

    该论文证明，在相同参数预算下，通过迭代间共享参数反复应用同一算子的循环估计器能够提升统计精度，揭示了参数—迭代—精度之间的权衡，并证明循环残差前馈网络与后层归一化 Transformer 可达到极小极大最优速率（至多相差对数因子）。

    

    arXiv:2609.25778v1 公告类型：交叉发布。摘要：人工智能中不断增长的内存需求促使人们探索使用更少可训练参数的学习方法。我们提出这样一个问题：循环估计器——即反复应用同一个已拟合的算子并在各迭代之间共享参数——能否在相同的参数预算下提升统计精度。其传统的非绑定对应方法则在每次迭代中使用各自独立的参数。对于一般的似然模型，我们建立了循环筛网极大似然估计的平方 Hellinger 风险上界，以及在调优的非绑定族上的极小极大下界。这些界揭示了一种参数—迭代—精度之间的权衡：重复计算可以在不增加参数的情况下改善逼近能力，但会增加计算成本和拟合类的复杂度。对于具有已知 Hölder 光滑度的目标，循环残差前馈网络和一种特定的后层归一化 Transformer 在固定参数预算下能够达到极小极大多项式速率（至多相差对数因子）。

    arXiv:2609.25778v1 Announce Type: cross  Abstract: Growing memory demands in artificial intelligence motivate learning with fewer trainable parameters. We ask whether a looped estimator, which repeatedly applies one fitted operator with parameters shared across iterations, can improve statistical accuracy under a common parameter budget. Its conventional untied counterpart uses separate parameters at each iteration. For general likelihood models, we establish an upper bound on squared Hellinger risk for looped sieve maximum likelihood and a minimax lower bound over the tuned untied family. These bounds reveal a parameter--iteration--accuracy tradeoff: repeated computation can improve approximation without adding parameters, while increasing computational cost and fitted-class complexity. For targets of known H\"older smoothness, looped residual feedforward networks and a specified post-layer-normalized Transformer attain the minimax polynomial rate up to logarithmic factors with a fixe
    
[^17]: 超越类边缘分布：在不冻结类共现的前提下界定重演间隔

    Beyond Class Marginals: Bounding Rehearsal Gaps without Freezing Class Co-occurrence

    [https://arxiv.org/abs/2609.25735](https://arxiv.org/abs/2609.25735)

    该论文提出随机遍次回放（RPR）调度器，在不使用未来类别信息、不增加回放样本或前向传播且不冻结类共现的前提下，将类回放间隔严格限制在2*ceil(C/b)-1以内，从而缓解因类长时间缺席回放而导致的分类器偏置。

    

    类平衡回放控制了各类的出现频率，但并不能决定一个类在连续两次回放出现之间的时间间隔。我们将这一间隔（即重演间隔）与类边缘分布和类共现分离开来单独研究，并提出随机遍次回放（RPR）方法，该方法在每次打乱顺序的遍次中对每个驻留类访问一次。对于固定的C个驻留类集合以及小于等于C的回放批大小b，RPR在保持平衡的时间平均类边缘分布的同时，将每个重演间隔限制在2*ceil(C/b)-1以内；当驻留类集合发生变动时，则适用一个以变动为条件的界。该调度器不使用任何未来类别信息，也不增加额外的回放样本或前向传播。在一个采用线性分类头的ER-ACE诊断实验中，当某类同时缺席新到批次和回放批次时，会产生单侧的分类器偏置梯度；更长的缺席时段与更大的负向偏置位移相关，而移除输入损失掩码会削弱这种调度效应。

    arXiv:2609.25735v1 Announce Type: new  Abstract: Class-balanced replay controls class frequency but does not determine the interval between successive replay appearances of a class. We study this interval, the rehearsal gap, separately from the class marginal and class co-occurrence, and introduce randomised-pass replay (RPR), which visits each resident class once per shuffled pass. For a fixed set of C resident classes and replay batch size b less than or equal to C, RPR preserves the balanced time-averaged class marginal and bounds every gap by 2*ceil(C/b)-1; a churn-conditional bound applies while the resident set changes. The scheduler uses no future class information and adds no replay examples or forward passes. In a linear-head ER-ACE diagnostic, joint absence from the incoming and replay batches produces a one-sided classifier-bias gradient. Longer absence episodes are associated with larger negative bias displacement, and removing the incoming-loss mask attenuates the scheduli
    
[^18]: 神经逼近与极小极大回归中网络规模与参数幅值之间的最优权衡

    Optimal Tradeoffs Between Network Size and Parameter Magnitude in Neural Approximation and Minimax Regression

    [https://arxiv.org/abs/2609.25710](https://arxiv.org/abs/2609.25710)

    本文建立了固定深度神经网络中网络宽度与参数幅值之间的最优权衡，证明使用有界1-Lipschitz二分-三角激活函数逼近β-Hölder类的最优误差率为[N²log(eNT)]^(-β/d)，且该速率对所有固定全局Hölder激活函数均匹配下界。

    

    神经网络的统计精度取决于其逼近能力以及从数据中拟合的函数类的复杂度。虽然增加网络规模是改善逼近的自然途径，但参数幅值提供了另一种资源，其在这两方面的作用必须被量化。我们使用一个初等的有界1-Lipschitz二分-三角激活函数，在固定深度下建立了尖锐的宽度-幅值权衡。对于[0,1]^d上的单位β-Hölder球（0<β≤1），当网络宽度满足N≥2d+3且参数幅值以T≥1为界时，最优L^p逼近误差（0<p<∞）的阶为[N²log(eNT)]^(-β/d)。对于每个固定的全局Hölder激活函数，均成立匹配的下界；其Hölder指数只影响常数而不影响速率。在有界设计密度和独立中心化亚高斯噪声条件下，在该函数类上的近似最小二乘……

    arXiv:2609.25710v1 Announce Type: cross  Abstract: The statistical accuracy of neural networks depends on both their approximation power and the complexity of the class fitted from data. While increasing network size is a natural way to improve approximation, parameter magnitude provides another resource whose role must be quantified in both respects. We establish a sharp width--magnitude tradeoff at fixed depth using one elementary bounded $1$-Lipschitz Dyadic--Triangular Activation. For the unit $\beta$-H\"older ball on $[0,1]^d$ with $0<\beta\leq1$, the optimal $L^p$ approximation error for $0<\infty$ is of order $[N^2\log(eNT)]^{-\beta/d}$ when the network width satisfies $N\geq2d+3$ and the parameter magnitudes are bounded by $T\geq1$. Matching lower bounds hold for every fixed globally H\"older activation; its H\"older exponent affects the constants but not the rate. Under bounded design densities and independent centered sub-Gaussian noise, approximate least squares over the ful
    
[^19]: 关于对抗鲁棒联邦回归的梯度异质性动态

    On the Gradient Heterogeneity Dynamics of Adversarially Robust Federated Regression

    [https://arxiv.org/abs/2609.25705](https://arxiv.org/abs/2609.25705)

    该论文从每轮使用新鲜数据的线性和非线性回归统计模型出发推导联邦学习中的梯度异质性，将其分解为诚实客户端真实模型参数差异、有限样本标签噪声和初始化三个来源，并据此为系数为 O(f/n) 的 稳健聚合器建立理论保证。

    

    联邦学习本质上具有异质性：诚实的客户端可能拥有不同的数据生成模型。在此之上，对抗性客户端可以通过共享任意更新使异质性更加显著。现有分析通常通过梯度差异条件来控制统计异质性与对抗行为之间的相互作用。然而，该界是先验强加的，即使对于最小二乘回归也可能产生保守的保证。我们转而从每轮使用新鲜数据样本的线性和非线性回归的统计模型中推导梯度异质性。我们的界将诚实客户端的真实模型参数之间的异质性、有限样本标签噪声以及初始化分离开来。随后我们证明，对于任何系数为 κ = O(f/n) 的 稳健聚合器，其中 f 为对抗性客户端数量，n 为总[客户端数量]……

    arXiv:2609.25705v1 Announce Type: cross  Abstract: Federated learning (FL) is intrinsically heterogeneous: honest clients may have different data-generating models. On top of that, adversarial clients can make heterogeneity even more pronounced by sharing arbitrary updates. Existing analyses typically control the interaction between statistical heterogeneity and adversarial behavior through gradient-dissimilarity conditions. However, the underlying bound is imposed a priori and may yield conservative guarantees even for least-squares regression. We instead derive the gradient heterogeneity from the statistical model of linear and nonlinear regression with fresh data samples at every round. Our bounds separate heterogeneity among the honest clients' ground-truth model parameters, finite-sample label noise, and initialization. We then demonstrate that, for any $(f,\kappa)$-robust aggregator with coefficient $\kappa = O(f/n)$, where $f$ is the number of adversarial clients and $n$ the tot
    
[^20]: 基于贝叶斯多臂老虎机Gittins指数的高效成本感知LLM评估

    Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices

    [https://arxiv.org/abs/2609.25645](https://arxiv.org/abs/2609.25645)

    提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。

    

    在每个基准测试项目上穷尽评估每个候选LLM配置以找出高性能配置的成本十分高昂。我们将配置选择问题形式化为一个成本感知的贝叶斯多臂老虎机问题，并提出GittinsEval，该方法借鉴贝叶斯最优的Gittins策略来决定下一步评估哪个配置以及何时停止。我们通过一个随时推荐规则对该策略进行了扩展，使其能够同时覆盖完全评估和部分评估的配置，并使用LCB风格的分数来考虑后验不确定性。GittinsEval在计算上非常高效，只需在离线预计算之后进行轻量级的在线更新。在GSM8K、PIQA、AlpacaEval和MMLU响应矩阵上，GittinsEval始终具有竞争力，在大样本基准上相比配置级贝叶斯优化取得了尤为显著的提升，在大候选任务上相比不考虑成本的bandit基线也表现更优。至关重要的是，GittinsEval通常能够以（原文在此处截断）……

    arXiv:2609.25645v1 Announce Type: cross  Abstract: Exhaustively evaluating every candidate LLM configuration on every benchmark item to identify a high-performing one is costly. We formulate configuration selection as a cost-aware Bayesian bandit problem and propose GittinsEval, which draws on the Bayesian-optimal Gittins policy to determine which configuration to evaluate next and when to stop. We extend the policy with an anytime recommendation rule over both fully and partially evaluated configurations, using an LCB-style score to account for posterior uncertainty. GittinsEval is computationally efficient, requiring only lightweight online updates after offline precomputation. Across GSM8K, PIQA, AlpacaEval, and MMLU response matrices, GittinsEval is consistently competitive, with particularly strong gains over configuration-level Bayesian optimization on large-example benchmarks and over cost-unaware bandit baselines on large-candidate tasks. Crucially, GittinsEval often attains ne
    
[^21]: 重复测量的广义深度回归

    Generalized Deep Regression for Repeated Measurements

    [https://arxiv.org/abs/2609.25605](https://arxiv.org/abs/2609.25605)

    该论文提出使用ReLU深度神经网络对具有重复测量的数据进行广义边际回归估计，证明了在β-Hölder光滑类上可达到n^{-1}+(nm)^{-2β/(2β+d)}的最优收敛速率，并提供了预言不等式及逐点推断的理论保证。

    

    在本文中，我们研究了使用ReLU深度神经网络从具有重复二值、计数或连续响应的独立单元中估计边际回归函数的问题。在该模型中，我们假设相关性由每个单元内一个未观测的随机均值函数产生。随后，我们使用凸广义回归损失来拟合神经网络。通过将条件测量变异与单元间变异相分离，我们证明了一个预言（oracle）不等式。此外，我们证明了当有n个单元且每个单元有m次测量时，ReLU网络在β-Hölder光滑函数类上可以达到n^{-1}+(nm)^{-2β/(2β+d)}阶的积分均方误差（至多相差对数因子）。我们还推导了针对不等聚类大小的加权预言不等式，以及组合光滑函数的收敛速率。对于逐点集成推断，我们给出了投影中心极限定理，并证明了无穷小刀切法（infinitesimal jackknife）的一致性。

    arXiv:2609.25605v1 Announce Type: cross  Abstract: In this paper, we study the estimation of a marginal regression function from independent units with repeated binary, count, or continuous responses using ReLU deep neural networks. In the model, we assume that the dependence is generated by an unobserved random mean function within each unit. We then fit a neural network with a convex generalized regression loss. We show an oracle inequality by separating conditional measurement variation from between-unit variation. In addition, we prove that with $n$ units and $m$ measurements per unit, ReLU networks can attain an integrated mean squared error of order $n^{-1}+(nm)^{-2\beta/(2\beta+d)}$, up to logarithmic factors, over $\beta$-H\"older classes. We also derive a weighted oracle inequality for unequal cluster sizes and a rate for compositionally smooth functions. For pointwise ensemble inference, we give a projection central limit theorem and prove infinitesimal jackknife consistency 
    
[^22]: 可扩展的最小体积单纯形估计与非渐近分析

    Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis

    [https://arxiv.org/abs/2609.25576](https://arxiv.org/abs/2609.25576)

    提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。

    

    我们研究从 N 个独立同分布、均匀采样自其内部的点中估计一个 K 维单纯形的问题；观测数据是 K+1 个未知原型的凸组合。现有的多项式时间估计器需要每样本立方级的计算量或 O(NK) 的存储空间，在 N 约为 10^6 至 10^8 的规模下不可行。我们提出 DeepMVSA，以神经隐式形式重新表述最小体积原理：一个轻量级坐标网络生成混合权重，一个三角 LU 型参数化表示对偶单纯形矩阵，从而将可训练状态的内存降至与 N 无关的 O(K^2)，并将每次数据遍历的成本降至 O(NK^2)。我们为局部化代理估计器证明了达到多项式时间基准阶数的非渐近样本复杂度界；为神经目标的每个全局最小值点证明了神谕不等式，包含体积膨胀控制和显式收缩偏差；以及一个条件性的端到端误差预算分离……

    arXiv:2609.25576v1 Announce Type: cross  Abstract: We study the estimation of a $K$-dimensional simplex from $N$ i.i.d.\ points sampled uniformly from its interior; the observations are convex combinations of $K+1$ unknown prototypes. Existing polynomial-time estimators need cubic per-sample work or $O(NK)$ storage and are impractical at $N\sim 10^6$--$10^8$. We propose DeepMVSA, which re-expresses the minimum-volume principle in neural implicit form: a lightweight coordinate network generates the mixing weights and a triangular LU-type parameterization the dual simplex matrix, reducing the trainable-state memory to $O(K^2)$, independent of $N$, and the cost per data pass to $O(NK^2)$. We prove a non-asymptotic sample-complexity bound of the polynomial-time benchmark order for a localized surrogate estimator; an oracle inequality for every global minimizer of the neural objective, with volume-inflation control and an explicit shrinkage bias; a conditional end-to-end error budget separa
    
[^23]: 面向自动定理证明搜索的生成器直接优化

    Direct Optimization of Generators for Search in Automated Theorem Proving

    [https://arxiv.org/abs/2609.25575](https://arxiv.org/abs/2609.25575)

    该论文通过对策略引导搜索的抽象，将计算对齐训练（CAT）扩展到树搜索场景，推导出可处理的搜索感知损失，并提出与搜索无关的均匀分配（UA）损失，从而直接优化用于自动定理证明搜索的生成器。

    

    微调后的大型语言模型（LLM）显著推动了自动定理证明（ATP）的发展，但它们通常被部署为树搜索中的引导策略，而非用于单次尝试生成。近期研究表明，交叉熵对于在聚合或过滤等扁平搜索策略中使用的LLM而言并非最优，相关工作已开发出新的损失函数来纠正这种失配。将这种对齐扩展到树搜索更具挑战性：证明的发现依赖于对监督示范无法揭示的非轨迹状态的探索与恢复。我们通过对策略引导搜索进行抽象，将计算对齐训练（CAT）扩展到这一设定，推导出可处理的、基于轨迹支持的损失。除这些搜索感知的损失外，我们还引入了一种与搜索无关的均匀分配（UA）损失，它在不指定具体搜索方式的情况下考虑计算预算。两者均在每条策略的交叉熵上诱导出标量权重。

    arXiv:2609.25575v1 Announce Type: cross  Abstract: Fine-tuned Large Language Models (LLMs) significantly advance Automated Theorem Proving (ATP), but are often deployed as guiding policies within tree search rather than for single-attempt generation. Recent work shows cross entropy is suboptimal for an LLM used in flat search strategies such as aggregation or filtering and that work has developed new loss functions to correct this misalignment. Extending this alignment to tree search is more challenging: proof discovery depends on exploration and recovery through off-trace states that supervised demonstrations do not reveal. We extend Compute-Aligned Training (CAT) to this setting through an abstraction of policy-guided search, deriving tractable, trace-supported losses. Alongside these search-aware losses, we introduce a search-agnostic uniform-allocation (UA) loss that accounts for the budget without specifying the specific search. Both induce scalar weights on per-tactic cross-entro
    
[^24]: p-进模型的连续优化

    Continuous Optimization for p-adic Models

    [https://arxiv.org/abs/2609.25501](https://arxiv.org/abs/2609.25501)

    该论文首次实现了 p-进参数机器学习模型的原生连续梯度下降，其核心创新是通过 Berkovich 仿射线这一路径连通的度量树扩张，使连续优化、反向传播以及动量和 Adam 等优化器在 p-进设定下成为可能。

    

    我们提出了首个针对具有 p-进参数的机器学习模型的原生连续梯度下降方法。现有的原生优化器都是离散的，主要是组合搜索，这是因为 p-进数 $\mathbb{Q}_p$ 是完全不连通的，且标准损失函数在远离极小值处是平坦的。为了实现连续优化，我们提出通过 Berkovich 仿射线来处理 $\mathbb{Q}_p$：这是 $\mathbb{Q}_p$ 的一个规范的、路径连通的扩张，它保持了 $\mathbb{Q}_p$ 的等距性，并唯一地延拓其解析映射。这个闭包是一个度量树，具有可解释的点和局部导数，我们证明了这使得有效的优化器和反向传播成为可能。我们形式化了梯度下降，并证明其近似方法能够高效地学习系数在 $\mathbb{Q}_p$ 中的线性模型来执行模运算——这是一种类 XOR 任务，无法用 $\mathbb{R}$ 上的线性模型表达。我们还演示了动量和 Adam 优化器。

    arXiv:2609.25501v1 Announce Type: new  Abstract: We present the first method for native, continuous gradient descent for machine learning models with $p$-adic parameters. Existing native optimizers are discrete, mostly combinatorial searches, as the $p$-adic numbers $\mathbb{Q}_p$ are totally disconnected, with standard losses that are flat away from their minima. To enable continuous optimization, we propose working with $\mathbb{Q}_p$ via its Berkovich affine line: a canonical, path-connected expansion of $\mathbb{Q}_p$ that preserves its isometries and uniquely extends its analytic maps. This hull is a metric tree with interpretable points and local derivatives, which we show enables effective optimizers and backpropagation. We formulate gradient descent and show that its approximations efficiently learn linear models with coefficients in $\mathbb{Q}_p$ to do modular arithmetic, an XOR-like task not expressible by linear models in $\mathbb{R}$. We also demonstrate momentum and Adam 
    
[^25]: 半监督联邦自动语音识别的实用方案：在线伪标签与服务器更新稳定化

    A Practical Recipe for Semi-Supervised Federated ASR: Online Pseudo-Labels with Server Update Stabilization

    [https://arxiv.org/abs/2609.25471](https://arxiv.org/abs/2609.25471)

    本文提出一种半监督联邦ASR的实用方案，通过服务器端标注数据更新来稳定客户端在线教师模型以生成可靠伪标签，从而显著缩小与完全监督联邦学习之间的性能差距。

    

    半监督联邦学习（SSFL）利用教师模型在客户端的未标注数据上生成伪标签来训练模型，同时服务器端只保留少量带标注的种子数据集。自动语音识别（ASR）在这种场景下尤为脆弱：伪标签错误会在输出序列以及各训练轮次间不断累积，最终导致训练发散，使其与完全监督的联邦学习之间存在较大差距。我们证明，缩小这一差距取决于两个相互耦合的设计维度——教师（由哪个模型生成伪标签）和锚点（服务器端在标注数据上进行的更新，用于稳定训练过程）。在教师维度上，每客户端的在线教师（即客户端自身不断演化的模型）若单独使用会发散，但一旦被稳定化，其表现可以匹敌甚至超越广播式全局教师（即服务器端的一个模型，在每轮训练内保持固定）——在域内数据上优势明显，在域偏移情况下也具有竞争力。随着种子数据集变得更强，在线教师的优势逐渐缩小，此时一个过渡性的…（原文在此截断）

    arXiv:2609.25471v1 Announce Type: new  Abstract: Semi-supervised federated learning (SSFL) trains models on clients' unlabeled data using a teacher to generate pseudo-labels, with a small labeled seed dataset on the server. Automatic Speech Recognition (ASR) is particularly fragile here: pseudo-label errors compound across the output sequence and across training rounds into divergence, leaving a large gap to fully-supervised FL. We show that closing this gap turns on two coupled design axes -- the teacher (which model generates the pseudo-labels) and the anchor (the server-side updates on labeled data that stabilize training). On the teacher axis, a per-client online teacher (each client's own evolving model) diverges on its own, but once stabilized it matches or beats the broadcast global teacher (one server model, fixed within a round) -- decisively in-domain and competitively under domain shift. As the seed grows stronger and the online teacher's advantage narrows, a transitioning t
    
[^26]: PICPIs：预测区间条件化的预测区间

    PICPIs: Prediction-Interval-Conditional Prediction Intervals

    [https://arxiv.org/abs/2609.25388](https://arxiv.org/abs/2609.25388)

    该论文提出了预测区间条件化预测区间这一新框架，通过自洽条件使预测区间同时定义预测值分层并保证该层内的平均结果落在同一区间内，从而在共形预测中填补了边际有效性分辨率不足与完全条件保证不可实现之间的空白。

    

    arXiv:2609.25388v1 公告类型：交叉 摘要：统计学中的一个经典问题是，在对不可观测的目标进行推断时，应当以哪些可观测的量为条件。对于非参数不确定性量化中的共形预测而言，标准的边际有效性在决策所依据的预测值上提供的分辨率有限，而相对于协变量的完全条件保证已被证明是无法实现的。我们通过引入一个基于预测的条件化框架来填补这一空白，我们将其称为预测区间条件化预测区间。形式上，PICPI 是一个满足自洽条件的区间 I：对于预测模型 p、上下文协变量 X 和结果 Y，有 E[Y | p(X) ∈ I] ∈ I。因此，这样一个区间同时定义了一个预测值的分层，并证明该分层中的平均结果落在同一区间内。这一自洽条件产生了数据自适应的……

    arXiv:2609.25388v1 Announce Type: cross  Abstract: A classical question in statistics is which observable quantities to condition on when drawing inferences about unobservable targets. For conformal prediction in nonparametric uncertainty quantification, standard marginal validity offers limited resolution at the prediction values on which decisions are based, and fully conditional guarantees with respect to the covariates are provably unattainable. We address this gap by introducing a prediction-based conditioning framework that we refer to as Prediction-Interval-Conditional Prediction Intervals (PICPIs). Formally, a PICPI is an interval $I$ satisfying a self-consistency condition: $$\mathbb{E} [Y \mid p(X) \in I] \in I,$$ for predictive model $p$, contextual covariate $X$, and outcome $Y$. Thus, an interval simultaneously defines a stratum of prediction values and certifies that the mean outcome in that stratum lies in the same interval. This self-consistency condition yields data-ad
    
[^27]: 用于约束采样的惩罚性非可逆朗之万算法

    Penalized Nonreversible Langevin for Constrained Sampling

    [https://arxiv.org/abs/2609.25381](https://arxiv.org/abs/2609.25381)

    提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。

    

    我们提出了用于从 $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$ 中采样的惩罚性非可逆朗之万算法，其中 $\mathcal C\subset\mathbb R^d$ 是一个紧凸集。这些算法将平方距离惩罚与能够保持惩罚吉布斯分布的常数型或相容的状态依赖斜对称扰动相结合。对于光滑且可能非凸的 $f$，我们在对数索博列夫不等式条件下推导了全梯度算法的非渐近总变差界。当可获得无偏随机梯度时，我们在适应性二次度量下，基于全漂移项的全局收缩性和利普希茨条件建立了 2-Wasserstein 界。对于固定的惩罚参数，相对于惩罚吉布斯分布的误差以指数速度衰减到一个 $\mathcal{O}(\sqrt{\eta})$ 邻域，其中 $\eta$ 为步长。我们还界定了惩罚吉布斯分布与目标分布之间的差异（摘要在此处被截断）。

    arXiv:2609.25381v1 Announce Type: cross  Abstract: We propose penalized nonreversible Langevin algorithms for sampling from $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$, where $\mathcal C\subset\mathbb R^d$ is a compact convex set. The algorithms combine a squared distance penalty with constant or compatible state dependent skew symmetric perturbations that preserve the penalized Gibbs distribution. For smooth, possibly nonconvex $f$, we derive nonasymptotic total variation bounds for the full gradient algorithm under a log Sobolev inequality. When unbiased stochastic gradients are available, we establish $2$-Wasserstein bounds under global contraction and Lipschitz conditions on the full drift in an adapted quadratic metric. For a fixed penalty parameter, the error relative to the penalized Gibbs distribution decays exponentially to an $\mathcal{O}(\sqrt{\eta})$ neighborhood, where $\eta$ is the stepsize. We also bound the discrepancy between the penalized Gibbs distribution and
    
[^28]: 边隐私图生成器的实证审计

    Empirical Auditing of Edge-Private Graph Generators

    [https://arxiv.org/abs/2609.25155](https://arxiv.org/abs/2609.25155)

    该论文提出了一个针对边隐私图生成器的实证审计框架，通过统计有效的隐私损失下界来比较直接边、局部结构和GNN三类攻击，发现隐私泄露程度依赖于生成机制和网络本身，且GNN学习到的表示能揭示传统局部统计无法捕获的隐私信息。

    

    我们通过测试来自边相邻输入的输出是否仍然可区分，来实证审计隐私泄露情况，并使用统计上有效的隐私损失下界来衡量我们攻击所揭示的泄露程度。我们的框架通过围绕目标边的几何结构，比较直接边攻击、局部结构攻击和基于图神经网络（GNN）的攻击。在两个生成器和两个网络上的实验表明，隐私泄露程度既依赖于生成机制也依赖于网络本身，并且学习到的表示能够揭示传统局部统计方法无法捕获的信息。

    arXiv:2609.25155v1 Announce Type: cross  Abstract: We empirically audit privacy leakage by testing whether outputs from edge-neighbouring inputs remain distinguishable, using statistically valid lower bounds on the privacy loss witnessed by our attacks. Our framework compares direct-edge, local-structural, and GNN-based attacks through the geometry surrounding a target edge. Experiments across two generators and two networks show that privacy leakage is both mechanism- and network-dependent, with learned representations revealing information not captured by conventional local statistics.
    
[^29]: 逆问题中摊销贝叶斯推断的变分目标：后验条件化的作用

    Variational objectives for amortized Bayesian inference in inverse problems: The role of posterior conditioning

    [https://arxiv.org/abs/2609.25145](https://arxiv.org/abs/2609.25145)

    该研究比较了逆问题摊销贝叶斯推断中三种VAE变分目标（反向KL、非对称JS和JS-Wasserstein），并通过广义Fisher基下的局部线性–高斯分析，揭示了后验条件化在弱可辨识参数方向上对后验精度和梯度行为的关键作用。

    

    变分自编码器（VAE）为逆问题的摊销贝叶斯推断提供了一种高效方法，但后验精度可能在很大程度上取决于变分正则化的选择，尤其是当逆问题包含弱可辨识的参数方向时。本研究考察了三种目标函数：反向Kullback–Leibler公式（VAE-KL）、非对称Jensen–Shannon公式（VAE-JS），以及Jensen–Shannon–Wasserstein公式（VAE-JSWA），后者在保留前向Kullback–Leibler后验监督的同时，用平方2-Wasserstein距离替代了反向Kullback–Leibler正则项。研究采用全协方差高斯编码器和预训练的基于物理的代理模型进行摊销后验推断，并在广义Fisher基下发展了一种局部线性–高斯分析，以刻画这三种目标函数依赖于方差的梯度特性。

    arXiv:2609.25145v1 Announce Type: cross  Abstract: Variational autoencoders (VAEs) offer an efficient approach to amortized Bayesian inference for inverse problems, but posterior accuracy can depend strongly on the choice of variational regularization, particularly when the inverse problem contains weakly identified parameter directions. This study investigates three objectives: a reverse Kullback--Leibler formulation (VAE-KL), an asymmetric Jensen--Shannon formulation (VAE-JS), and a Jensen--Shannon--Wasserstein formulation (VAE-JSWA), which replaces the reverse Kullback--Leibler regularizer with the squared 2-Wasserstein distance while retaining forward-Kullback--Leibler posterior supervision. A full-covariance Gaussian encoder and a pre-trained physics-based surrogate are used for amortized posterior inference. A local linear--Gaussian analysis in the generalized Fisher basis is developed to characterize the variance-dependent gradients of the three objectives. The formulations are 
    
[^30]: Lepto方差的信息内容及其与高阶矩的关系

    The Informational Content in Lepto-Variance and Its Relation to Higher Moments

    [https://arxiv.org/abs/2609.25144](https://arxiv.org/abs/2609.25144)

    本文通过正态样本模拟研究1比特Lepto方差与样本方差、偏度和超额峰度的关系，发现正态分布的Lepto比率收敛于36.3%，且美国历史股票收益58%的变异性是无法被任何金融因子解释的1比特Lepto方差。

    

    Lepto回归被定义为将目标特征在其自身上构建回归树的机器学习过程。这是一种新颖的无模型方法，有可能揭示重要样本结构特性的信息。但目前尚不清楚Lepto方差的信息内容是什么，以及它如何与样本的其他知名统计量相关联。一项重要发现是，美国历史股票收益变异性的58%是无法被任何金融因子解释的1比特Lepto方差。本文研究的核心问题是利用小样本的正态N(0,1)抽样，探索1比特样本Lepto方差和Lepto比率与样本方差、偏度和超额峰度之间的关系。通过大样本模拟，发现正态分布的Lepto比率收敛于36.3%。对于较小的正态分布模拟N(0,1)样本，虽然Lepto方差本身与……

    arXiv:2609.25144v1 Announce Type: cross  Abstract: Lepto-regression is defined as the machine learning process of constructing a Regression Tree of a target feature on itself. It is a novel, model-free method potentially revealing information on important sample structure properties. But it is yet not clear what the informational content of lepto-variance is and how it is related to other well-known statistics of a sample. One significant finding is that 58% of the historical US stock return variability is 1-bit lepto-variance that can not be explained by any financial factor. The central question investigated in this paper is to use small normal N(0, 1) drawn samples to explore how the 1-bit sample lepto-variance and lepto-ratio relate to sample variance, skewness and excess kurtosis. Using a large sample simulation, the lepto ratio of a normal is found to converge to 36.3%. For smaller normally distributed simulated N(0, 1) samples, while lepto-variance itself is highly correlated to
    
[^31]: 大语言模型的概率结构

    The Probabilistic Structure of Large Language Models

    [https://arxiv.org/abs/2609.25134](https://arxiv.org/abs/2609.25134)

    本文以统一的概率论框架阐述大语言模型——将其视为令牌序列上的概率测度，把训练归结为最大似然估计、把生成归结为随机过程的顺序模拟，并揭示了KL散度的非对称性与幻觉现象及“统计合理性与真实性之别”之间的内在联系。

    

    本文从概率论的视角对大型语言模型（LLMs）进行了阐述，旨在将文献中通常被分开处理的各类工具整合为一篇自成一体的统一论述。LLMs被描述为定义在令牌（token）序列集合上的概率测度，并通过其自回归条件分布来具体指定。训练被表述为一个最大似然估计问题，通过随机梯度方法求解；而文本生成则被视为对该随机过程的顺序模拟。本文还考察了Kullback–Leibler散度的非对称性在文本生成中所起的作用，并将其与幻觉等典型现象，以及“统计上合理”与“真实”之间的区别联系起来。作为同一观点的补充例证，我们还讨论了围绕分数函数（score function）构建的扩散模型，这类模型并非将生成视为……（原文摘要至此截断）

    arXiv:2609.25134v1 Announce Type: new  Abstract: This paper presents a probabilistic perspective on large language models (LLMs), developed with the aim of bringing together, in a single self-contained account, tools that are usually treated separately across the literature. LLMs are described through probability measures on the set of sequences of tokens, specified via their autoregressive conditional distributions. Training is formulated as a maximum-likelihood estimation problem, addressed by stochastic gradient methods, while text generation is viewed as the sequential simulation of the resulting stochastic process. The role of the asymmetry of the Kullback--Leibler divergence in text generation is examined in relation with characteristic phenomena such as hallucination and the distinction between statistical plausibility and truth. As a complementary illustration of the same viewpoint, we also discuss diffusion models, built around the score function, which cast generation not as 
    
[^32]: FREESIA：面向表达力强且可扩展数据同化的协方差感知后验传输方法

    FREESIA: Covariance-Aware Posterior Transport for Expressive and Scalable Data Assimilation

    [https://arxiv.org/abs/2609.25085](https://arxiv.org/abs/2609.25085)

    提出了一种无需训练、渐近精确的协方差感知后验传输方法 FREESIA，通过将预测交叉协方差嵌入基于流的传输，在高维稀疏观测下准确恢复未观测状态并保持非高斯多峰后验结构。

    

    数据同化旨在基于观测数据推断复杂动力系统的状态。然而，在高维和稀疏观测条件下，准确推断由非线性或非单射观测算子所诱导的多峰后验分布仍然是一个关键挑战。集合滤波器可以扩展到高维场景，但受限于严格的分布假设；而无需训练的生成式滤波器（如 EnSF、EnFF）缓解了这一限制，但在稀疏观测下可能引入结构性误差并阻碍信息传播。为解决这些问题，我们提出了一种无需训练、渐近精确的后验传输方法。首先，设计了一种协方差感知的后验传输方案，将预测交叉协方差嵌入到基于流的传输中，在保持非高斯后验结构的同时准确恢复未观测到的状态。此外，该方法结合……（摘要在此处被截断）

    arXiv:2609.25085v1 Announce Type: cross  Abstract: Data assimilation aims to infer the state of complex dynamical systems based on observational data. However, accurate inference of the multimodal posteriors induced by nonlinear or non-injective observation operators remains a key challenge under high-dimensional and sparse observation conditions. Ensemble filters scale to high dimensions but are confined by restrictive distributional assumptions, while training-free generative filters (e.g., EnSF, EnFF) alleviate this limitation but may introduce structural errors and hinder information propagation under sparse observations. To address these issues, we propose a training-free, asymptotically exact posterior transport method. Firstly, a covariance-aware posterior transport scheme is designed, which embeds the forecast cross-covariance into flow-based transport and accurately recovers unobserved states while preserving the non-Gaussian posterior structure. Furthermore, the method combin
    
[^33]: 思维链熵究竟度量了什么？对脚手架、路由与内容的通道审计

    What Does Chain-of-Thought Entropy Measure? A Channel Audit of Scaffolding, Routing, and Content

    [https://arxiv.org/abs/2609.25039](https://arxiv.org/abs/2609.25039)

    该论文提出通过指定脚手架词表子集，将思维链token熵精确分解为脚手架通道与内容通道，证明两种约定在判断最大分叉位置上可能相互矛盾，并发现脚手架最多可占原始高熵token集合的41%。

    

    arXiv:2609.25039v1 公告类型：交叉发布。摘要：思维链token上的熵决定了哪些token接受策略梯度、哪些token被剪枝、以及一次运行是否已经坍缩，然而每一种这样的统计量所读取的下一token分布都混合了三种选择：是否输出连接性脚手架、输出哪个连接词、以及实质性的续写内容应当是什么。通过指定一个脚手架词表子集，可以将这三者精确地分离，该分离对熵、Kullback–Leibler散度以及softmax策略的一阶熵速度均严格成立。我们证明了在显式构造的开区域上，原始约定与内容约定会对“哪个位置是更大的分叉点”产生分歧，并以内容通道加上一个由测量见证所认证的泄漏项来界定答案多样性。在二十三种配置中，脚手架一侧最多占据原始高熵集合的41%；在分词器匹配的阶梯实验中，耦合仅在数学语料库这一步发生变化，而脚手架的熵份额却持续增长……

    arXiv:2609.25039v1 Announce Type: cross  Abstract: Entropy over chain-of-thought tokens decides which tokens receive the policy gradient, which get pruned, and whether a run has collapsed, yet each such statistic reads a next-token distribution mixing three choices: whether to emit connective scaffolding, which connective, and what the substantive continuation should be. Designating a scaffold vocabulary subset separates the three, exactly, for entropy, Kullback--Leibler divergence, and the first-order entropy velocity of a softmax policy. We prove the raw and content conventions disagree about which position is the larger fork on an explicit open region, and bound answer diversity by the content channel plus a leakage term a measured witness certifies. Across twenty-three configurations the scaffold side carries up to 41% of the raw high-entropy set; on a matched-tokenizer ladder, coupling changes only at the math-corpus step while the scaffold's entropy share keeps growing through di
    
[^34]: 面向不平衡分类的密度比重新评分方法

    Density-Ratio Rescoring for Imbalanced Classification

    [https://arxiv.org/abs/2609.23926](https://arxiv.org/abs/2609.23926)

    提出密度比重新评分（DRR）方法，通过调查整权法构造对偶分数并与基础分类器分数以固定权重融合，无需重采样或重新拟合即可在24个不平衡表格数据基准上一致提升平均精度。

    

    密度比重新评分通过一个基于调查整权法的对偶分数来增强在原始类别先验下训练的分类器。整权法对多数类样本进行重新加权，使其在容差范围内匹配少数类的特征矩。DRR对对偶分数和基础分数进行边际标准化，并以固定权重二分之一将二者融合，直接使用拟合的对偶分数进行预测，无需重采样或重新拟合基础分类器。在精确总体匹配且对数线性倾斜模型被正确设定的条件下，对偶分数等于对数密度比加上一个加性常数。类别可分性分析刻画了在共同类内协方差下，融合能够改善可分性所需的信号强度和相关性条件。在24个表格数据基准上，经过30次试验和五种基础学习器的评估，DRR在D=128随机特征设置下于每个数据集上都比标准化基础模型提升了平均精度，平均增益为0.034。

    arXiv:2609.23926v1 Announce Type: cross  Abstract: Density-Ratio Rescoring (DRR) augments a classifier trained at the original class prior with a survey-raking dual score. Raking reweights the majority sample to match minority feature moments within a tolerance. DRR marginally standardizes the dual and base scores and combines them with a fixed weight of one half, using the fitted dual directly for prediction without resampling or refitting the base classifier. Under exact population matching and a correctly specified log-linear tilt model, the dual equals the log density ratio up to an additive constant. A class-separation analysis characterizes the signal strength and correlation conditions under which fusion improves separation under common within-class covariance. On 24 tabular benchmarks, evaluated over 30 trials and five base learners, DRR at the D=128 random-feature setting improves average precision over the standardized base on every dataset, with a mean gain of 0.034. It exce
    
[^35]: 可解释人工智能的局部蒸馏方法

    Interpretable AI with Local Distillation

    [https://arxiv.org/abs/2608.23538](https://arxiv.org/abs/2608.23538)

    本文提出局部蒸馏方法，利用黑盒教师模型在每个查询点指导正则化线性学生模型，通过定义局部性和锚定预测来实现高精度与可解释性的兼顾。

    

    现代AI模型，如表格基础模型和梯度提升集成模型，在预测性能上优于经典方法，但对其预测的推理依据提供甚少。高风险决策要求模型既准确又具备内在可解释性。局部线性建模提供了一条前进之路：平滑回归函数在局部可由线性函数良好近似，使得在每个查询点附近的线性拟合能够在保持透明性的同时实现高精度。挑战在于学习什么是“局部”以及开发用于解释的统计工具。在此，我们提出局部蒸馏方法，其中黑盒“教师”模型在每个查询点指导一个正则化的线性“学生”模型。教师模型通过增加预测结果相似的训练观测的权重来定义局部性，并将其在查询点的预测作为伪观测包含进来以锚定拟合，该伪观测的权重会被估计。

    arXiv:2608.23538v1 Announce Type: cross  Abstract: Modern AI models such as tabular foundation models and gradient-boosted ensembles can outpredict classical methods, but provide little basis for reasoning about their predictions. High-stakes decisions call for models that are both accurate and interpretable as built. Local linear modeling offers a path forward: a smooth regression function is locally well approximated by a linear one, allowing a linear fit near each query point to achieve high accuracy without sacrificing transparency. The challenges lie in learning what is "local" and developing statistical tools for interpretation.   Here, we propose local distillation, in which a black-box "teacher" guides a regularized linear "student" model at each query point. The teacher (1) defines locality by upweighting training observations with similar predicted outcomes, and (2) anchors the fit with its prediction at the query point, included as a pseudo-observation whose weight is estima
    
[^36]: 量子与经典示例的Oracle分离：关于“制造”内容的研究

    A Quantum/Classical Example Oracle Separation for Making Things Up

    [https://arxiv.org/abs/2608.11648](https://arxiv.org/abs/2608.11648)

    本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。

    

    我们研究了在PAC学习框架中，量子示例相对于经典示例的能力。这里，我们考虑两种学习算法，它们都能访问量子计算，但一种获得量子示例，而另一种获得经典示例。此前尚不清楚是否存在学习任务，其中前者能高效完成而后者不能。我们的主要结果是，相对于一个Oracle，存在一些分布，可以由访问量子示例的量子学习者高效生成，但无法由仅访问经典示例的量子学习者生成，这为肯定回答此问题取得了进展。

    arXiv:2608.11648v1 Announce Type: cross  Abstract: We study the power of quantum examples, as compared to classical examples, in the PAC learning framework. Here, we have two learning algorithms, both with access to quantum computation, but one gets quantum examples, whereas the other gets classical examples. It was previously unknown whether there were learning tasks that can be efficiently performed but not by the latter. Our primary result is to show that relative to an oracle, there are distributions that can be efficiently generated by a quantum learner with access to quantum examples, but not by a quantum learner with access to only classical examples, making progress to answering this question in the affirmative.
    
[^37]: 老虎机后推断中偏差的精确刻画

    Sharp Characterization of Bias in Post-Bandit Inference

    [https://arxiv.org/abs/2608.01069](https://arxiv.org/abs/2608.01069)

    本文通过引入“有效探索率”这一关键量，精确刻画了老虎机算法后推断中样本均值偏差的算法来源，发现UCB1下偏差以极慢的1/√(log T)速率衰减，并揭示了探索带来的遗憾-偏差权衡。

    

    老虎机算法为下游推断生成数据，但自适应采样会使老虎机后的样本均值产生偏差。我们针对稳定的指数算法（包括UCB1及其推广形式）分析了这种偏差，并在固定时域T的老虎机实验中，推导出了样本均值偏差和期望Z统计量的精确首阶表达式。我们的刻画通过一个依赖于指数函数的关键量揭示了偏差的算法来源，我们将其称为“有效探索率”。例如，在UCB1算法下，有效探索率的量级为√(log T)，而任何臂（非唯一最优臂）的标准化偏差以极慢的速率1/√(log T)衰减。我们还展示了指数函数的选择如何同时影响遗憾和偏差，这揭示了一种遗憾-偏差权衡：更具探索性的算法会减少偏差但增加遗憾。我们进一步展示了偏差如何最严重地扭曲置信区间……

    arXiv:2608.01069v2 Announce Type: replace  Abstract: Bandit algorithms generate data for downstream inference, but adaptive sampling biases post-bandit sample means. We analyze this bias for stable index algorithms, including UCB1 and its generalizations, and derive sharp leading-order expressions for the sample-mean bias and expected $Z$-statistic, in bandit experiments of fixed horizon $T$. Our characterization reveals the algorithmic origin of bias through a key index-function-dependent quantity, which we term effective exploration rate. For example, under UCB1, the effective exploration rate is of order $\sqrt{\log T}$, and the standardized bias of any arm (that is not uniquely optimal) decays at the extremely slow rate $1/\sqrt{\log T}$. We also show how the choice of the index function affects both regret and bias, which reveals a regret-bias trade-off: more exploratory algorithm reduces bias but increases regret. We further show how bias most severely distorts confidence interva
    
[^38]: 混沌即是阶梯：通过重加权实现超越不变性的域泛化

    Chaos Is a LADDER: Domain Generalization Beyond Invariance via Reweighting

    [https://arxiv.org/abs/2607.26458](https://arxiv.org/abs/2607.26458)

    提出LADDER方法，将多域风格的“混沌”转化为定位未见目标域的阶梯，通过潜在域解耦与环境重加权，超越传统不变性原则实现更灵活的域泛化。

    

    域泛化（DG）旨在从多个源域中学习并泛化到未见的目标域。大多数DG方法追求不变性：它们寻找一种因果表示，使其预测规则在各域之间保持不变。当因果机制稳定时，这一原则行之有效，但当域本身会调节因果内容如何映射到响应时，该原则就变得具有局限性。在这种情况下，直接将域风格输入预测器可能会产生误导性的捷径，因为风格本身并不引起响应。然而，多种风格呈现的表面混沌可以成为阶梯：风格可以在源域中定位未见的目标域，并指导哪些依赖域的预测规则应当被信任。我们提出了潜在自适应域解耦与环境重加权（LADDER），这是一种固定模型的DG流水线，它学习因果/风格表示，冻结编码器，拟合源特定的……

    arXiv:2607.26458v2 Announce Type: replace-cross  Abstract: Domain generalization (DG) aims to learn from multiple source domains and generalize to unseen target domains. Most DG methods pursue invariance: they seek a causal representation whose prediction rule is invariant across domains. This principle is effective when the causal mechanism is stable, but becomes restrictive when the domain itself modulates how causal content maps to the response. In this case, directly feeding domain style into the predictor can create misleading shortcuts, since style does not by itself cause the response. Yet the apparent chaos of multiple styles can become a ladder: style can locate the unseen target domain among source domains and guide which domain-dependent prediction rules should be trusted. We propose \emph{Latent Adaptive Domain Disentanglement and Environment Reweighting} (LADDER), a fixed-model DG pipeline that learns causal/style representations, freezes the encoders, fits source-specific
    
[^39]: 并非所有目标生而平等：面向分层多目标优化的优先级约束下降

    Not All Objectives Are Born Equal: Priority-Constrained Descent for Hierarchical Multi-Objective Optimization

    [https://arxiv.org/abs/2606.29521](https://arxiv.org/abs/2606.29521)

    提出优先级约束下降（PCD）框架，通过单一参数控制最小失真，在保持主要目标下降方向的同时保证次要目标取得进展，且对目标缩放不变并给出两、三目标问题的精确闭式解。

    

    深度学习问题很少涉及重要性相等的目标。一个主要目标定义了任务目标，而次要目标（如稀疏性、压缩或鲁棒性）则对解施加约束。尽管现有的多目标方法在实践中已被证明有效，但它们存在明显的对称性问题，忽视了这些目标空间中固有的目标层级结构。我们提出了优先级约束下降（Priority-Constrained Descent, PCD），这是一种旨在显式利用分层目标结构的基于梯度的优化框架。PCD在保持主要目标下降方向的同时，允许为保证次要目标取得进展所必需的最小失真，该失真强度由单个参数 τ ∈ [0, 1] 控制。所得的公式对目标缩放具有不变性，并且对于两目标和三目标问题可求得精确的闭式解。我们评估……

    arXiv:2606.29521v2 Announce Type: replace  Abstract: Deep learning problems rarely involve objectives that are equal in importance. A primary objective defines the goal, whilst secondary objectives, such as sparsity, compression, or robustness constrain the solution. While existing multi-objective methods have proven effective in practice, they have a clear symmetry problem and neglect the inherent objective hierarchy built into these objective spaces. We introduce Priority-Constrained Descent (PCD), a gradient-based optimization framework designed to explicitly exploit hierarchical objective structures. PCD preserves the direction of primary descent whilst allowing for the minimal distortion necessary to guarantee progress on secondary objectives, controlled by a single $\tau \in [0, 1]$ that dictates the strength of the distortion. The resulting formulation is invariant to objective scaling and admits exact closed-form solutions for problems with two and three objectives. We evaluate
    
[^40]: 无偏梯度与移动的稳定性边界：线性自注意力中的精确小批量几何

    Unbiased Gradients, Moving Stability Boundaries: Exact Mini-Batch Geometry in Linear Self-Attention

    [https://arxiv.org/abs/2605.21292](https://arxiv.org/abs/2605.21292)

    该论文在线性自注意力的上下文回归中证明，无偏小批量随机梯度虽在期望上等价于全梯度，却会因采样曲率与目标相关性导致稳定性边界严格向外漂移，并给出了精确的边界穿越准则及几乎必然逃逸的证明。

    

    无偏随机梯度可以在期望意义上匹配全梯度，同时改变有限步的稳定性。我们在用于上下文线性回归的单层线性自注意力中研究这一效应，其中共享模式的小批量训练可以精确地归结为一个随机双因子映射。全批量映射保持一个椭圆区域，而采样得到的曲率和目标相关性会产生依赖于批量的稳定性边界。它们的联合波动在参数更新层面是居中的，但会使全批量边界坐标产生严格向外的漂移。我们推导了精确的单步穿越准则、自适应集中界和条件漂移恒等式，并在平衡切换模型中证明了几乎必然的逃逸。对于独立的长提示，模式泄漏按提示长度平方根的倒数衰减。对于有限token的softmax注意力，我们推导了一个由有效因子支配的显式边界穿越定律。

    arXiv:2605.21292v2 Announce Type: replace-cross  Abstract: Unbiased stochastic gradients can match the full gradient in expectation while changing finite-step stability. We study this effect in one-layer linear self-attention for in-context linear regression, where shared-mode mini-batch training reduces exactly to a random two-factor map. The full-batch map preserves an elliptic region, whereas sampled curvature and target correlation create batch-dependent stability boundaries. Their combined fluctuation is centered at the parameter-update level but induces a strictly outward drift of the full-batch boundary coordinate. We derive an exact one-step crossing criterion, adaptive concentration bounds, and conditional drift identities, and prove almost-sure escape in a balanced switching model. For independent long prompts, mode leakage decays at the inverse square root of prompt length. For finite-token softmax attention, we derive an explicit boundary-crossing law governed by an effecti
    
[^41]: 熵风险度量下最优策略识别的紧致样本复杂度界

    Tight Sample Complexity Bounds for Entropic Best Policy Identification

    [https://arxiv.org/abs/2605.13717](https://arxiv.org/abs/2605.13717)

    该论文通过基于KL探索奖励的前向模型算法改进了对指数效用的集中性控制，将熵风险敏感强化学习中最优策略识别的样本复杂度上界从 O(e^{2|β|H}) 降至与下界匹配的 O(e^{|β|H})，从而弥合了长期存在的指数级差距。

    

    我们研究在熵风险度量下有限时域风险敏感强化学习中的最优策略识别问题。近期工作表明，识别近似最优策略所需样本数的下界与上界在指数时域依赖性上存在常数倍的差距。具体而言，已知下界的量级为 Ω(e^{|β|H})，其中 H 是 MDP 的时域，而最先进的上界在使用生成模型的情况下最多只能达到 O(e^{2|β|H})（arXiv:2506.00286v2）。我们证明这一额外的指数因子可以追溯到对指数效用过于宽松的集中性控制。为了弥合这一悬而未决的差距，我们通过一种基于前向模型的算法重新审视该问题的分析，该算法建立在基于 KL 散度的探索奖励之上，并将其适配到熵准则。我们所获得的改进得益于两项主要的新颖技术创新。我们利用……（平滑性质）

    arXiv:2605.13717v2 Announce Type: replace  Abstract: We study best-policy identification for finite-horizon risk-sensitive reinforcement learning under the entropic risk measure. Recent work established a constant gap in the exponential horizon dependence between lower and upper bounds on the number of samples required to identify an approximately optimal policy. Precisely, known lower bounds scale in $\Omega(e^{|\beta| H})$ where $H$ is the horizon of the MDP, while the state-of-the-art upper bound achieves at best $O(e^{2|\beta| H})$ (arXiv:2506.00286v2) using a generative model. We show that this extra exponential factor can be traced to overly loose concentration control for exponential utilities. To close this open gap, we revisit the analysis of this problem through a forward-model based algorithm building on KL-based exploration bonuses that we adapt to the entropic criterion. The improvement we get is due to two main novel technical innovations. We leverage the smoothness prope
    
[^42]: 面向计数数据的流匹配

    Flow Matching for Count Data

    [https://arxiv.org/abs/2605.07746](https://arxiv.org/abs/2605.07746)

    提出了count-FM，一种基于连续时间生灭过程的计数数据流匹配框架，通过免模拟训练条件转移速率，在计数空间中高效实现任意计数分布间的分布传输，适用于单细胞RNA测序等高维计数数据场景。

    

    高维计数数据出现在诸如单细胞RNA测序和神经尖峰序列等应用中，其中跨连续批次或时间点的分布间映射构成数据分析的关键组成部分。扩散模型和基于流的深度生成模型在图像、视频和文本领域的近期成功，启发人们将这些思想扩展到计数型数据场景，但许多现有方法要么将每个计数视为离散类别状态，要么将计数转换到连续空间，当计数范围较大时，这两种方式都不自然且效率低下。我们提出了count-FM，一个基于具有局部单位跳变的连续时间生灭过程的计数数据流匹配框架。Count-FM通过对条件转移速率进行免模拟训练，在计数空间中高效地学习边际转移，从而实现任意计数分布的源群体与目标群体之间的传输。在仿真中……

    arXiv:2605.07746v2 Announce Type: replace-cross  Abstract: High-dimensional count data arise in applications such as single-cell RNA sequencing and neural spike trains, where mappings between distributions across successive batches or time points form critical components of data analysis. The recent success of diffusion- and flow-based deep generative models for images, video, and text motivates extending these ideas to count-valued settings, but many existing methods either treat each count as a categorical state or transform counts into a continuous space, neither of which is natural or efficient when the count range is large. We propose count-FM, a flow-matching framework for count data based on a continuous-time birth-death process with local unit jumps. Count-FM learns marginal transitions efficiently in count space through simulation-free training of conditional transition rates, allowing transport between arbitrary count-distributed source and target populations. In simulation, 
    
[^43]: SPLICE：基于JEPA嵌入的潜空间扩散用于保形时间序列修复

    SPLICE: Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting

    [https://arxiv.org/abs/2605.00126](https://arxiv.org/abs/2605.00126)

    SPLICE将JEPA潜空间生成式插补与自适应保形推断相结合，为电力负荷时间序列修复同时提供高质量重建与有限样本覆盖率保证的预测区间，且流匹配变体实现5-10倍加速。

    

    用于时间序列插补的生成模型虽然具有出色的重建精度，但无法提供有限样本下的可靠性保证，这在电力系统中是一个关键局限，因为插补值会直接影响调度与规划决策。我们提出了SPLICE（带保形包络的自监督预测潜空间插补），这是一个将潜空间生成式插补与无分布、在线自适应预测区间相结合的模块化框架。JEPA编码器将每日负荷片段映射到64维潜空间；具备四种采样模式的条件潜空间桥接模块生成候选缺失轨迹；小时条件解码器将其映射回信号空间；自适应保形推断（ACI）为输出包裹具有覆盖率保证的预测带。流匹配变体仅需5-10个ODE步骤即可达到与DDIM相当的质量（5-10倍加速）。在十三个负荷数据集（九个专有数据集、三个UCI电力数据集、ETTh1）上，SPLICE……（摘要在此处截断）

    arXiv:2605.00126v2 Announce Type: replace  Abstract: Generative models for time-series imputation achieve strong reconstruction accuracy, yet provide no finite-sample reliability guarantees, a critical limitation in power systems where imputed values inform dispatch and planning. We introduce SPLICE (Self-supervised Predictive Latent Inpainting with Conformal Envelopes), a modular framework coupling latent generative imputation with distribution-free, online-adaptive prediction intervals. A JEPA encoder maps daily load segments into a 64-dimensional latent space; a conditional latent bridge with four sampling modes generates candidate gap trajectories; an hourly-conditioned decoder maps back to signal space; and Adaptive Conformal Inference (ACI) wraps the output with coverage-guaranteed prediction bands. The flow-matching variant achieves comparable quality to DDIM in 5--10 ODE steps (5-10x speedup). On thirteen load datasets (nine proprietary, three UCI Electricity, ETTh1), SPLICE ac
    
[^44]: 条件分布处理效应：双重稳健估计与检验

    Conditional Distributional Treatment Effects: Doubly Robust Estimation and Testing

    [https://arxiv.org/abs/2603.16829](https://arxiv.org/abs/2603.16829)

    本文提出了捕捉条件分布处理效应的新估计对象及其双重稳健的极小极大最优估计方法，并开发了首个具有有效第一类错误控制和一致性的条件潜在结果分布同质性检验。

    

    超越条件平均处理效应，处理可能以依赖于协变量的方式影响整个结果分布，例如，改变特定子群体的方差或尾部风险。我们提出了一个新的估计对象来捕捉这种条件分布处理效应，并开发了一种在局部渐近意义上具有极小极大最优性的双重稳健估计器。基于此，我们开发了一种针对条件潜在结果分布全局同质性的检验方法，该方法能够容纳超越最大均值差异（MMD）的多种差异度量，具有可证明有效的第一类错误率，且对固定备择假设具有一致性——据我们所知，这是在该设置下首个具有此类保证的检验方法。随后，我们提供了一种在核带宽选择网格上聚合证据的检验方法。此外，我们推导出了两种自然差异度量（包括MMD）的精确闭式表达式，并提供了计算……

    arXiv:2603.16829v2 Announce Type: replace-cross  Abstract: Beyond conditional average treatment effects, treatments may impact the entire outcome distribution in covariate-dependent ways, for example, by altering the variance or tail risks for specific subpopulations. We propose a novel estimand to capture such conditional distributional treatment effects, and develop a doubly robust estimator that is minimax optimal in the local asymptotic sense. Using this, we develop a test for the global homogeneity of conditional potential outcome distributions that accommodates discrepancies beyond the maximum mean discrepancy (MMD), has provably valid type 1 error, and is consistent against fixed alternatives---the first test, to our knowledge, with such guarantees in this setting. We then provide a test that aggregates evidence across a grid of kernel-bandwidth choices. Furthermore, we derive exact closed-form expressions for two natural discrepancies (including the MMD), and provide a computat
    
[^45]: 通过部分共享实现通信高效、拜占庭鲁棒的联邦保形预测

    Communication-Efficient Byzantine-Robust Federated Conformal Prediction via Partial Sharing

    [https://arxiv.org/abs/2602.18396](https://arxiv.org/abs/2602.18396)

    提出PRISM-FCP框架，通过部分模型共享以M/D比例衰减训练阶段拜占庭投毒攻击的扰动能量，并结合基于直方图的过滤抵御对抗性校准提交，实现了兼顾两个阶段安全性与通信效率的联邦保形预测。

    

    我们提出了PRISM-FCP（通过部分共享与统计边际鲁棒校准实现联邦保形预测），这是一种通信高效、拜占庭鲁棒的联邦保形预测框架，它利用部分模型共享来缓解训练期间的随机模型投毒攻击，并利用基于直方图的过滤来抵御对抗性校准提交。现有的鲁棒FCP方法主要针对校准阶段的对抗行为，而将训练阶段的投毒问题留给独立的鲁棒训练机制处理。PRISM-FCP则考虑了这两个阶段之间的耦合。在训练过程中，客户端通过每轮仅传输D个参数中的M个来部分共享更新。在所述的随机攻击模型下，与完全共享相比，这将每个拜占庭客户端对聚合结果的扰动贡献的期望能量衰减为原来的M/D倍。当这种收益超过优化方面的（摘要在此处被截断）

    arXiv:2602.18396v3 Announce Type: replace  Abstract: We propose PRISM-FCP (Partial shaRing and robust calIbration with Statistical Margins for Federated Conformal Prediction), a communication-efficient Byzantine-robust federated conformal prediction framework that uses partial model sharing to mitigate stochastic model-poisoning attacks during training and histogram-based filtering to mitigate adversarial calibration submissions. Existing robust FCP approaches primarily address adversarial behavior during calibration, leaving training-stage poisoning to separate robust-training mechanisms. PRISM-FCP instead considers the coupling between the two stages. During training, clients partially share updates by transmitting only $M$ of $D$ parameters per round. Under the stated stochastic attack model, this attenuates the expected energy of each Byzantine client's perturbation contribution to the aggregate by a factor of $M/D$ relative to full sharing. When this benefit outweighs the optimiza
    
[^46]: 高效且可扩展的生存曲线聚类方法

    Efficient and scalable clustering of survival curves

    [https://arxiv.org/abs/2512.16481](https://arxiv.org/abs/2512.16481)

    该论文提出一种结合k-means聚类与log-rank检验的新方法，无需计算昂贵的自助法重抽样，即可高效且可扩展地对生存曲线进行聚类分析。

    

    生存分析涵盖了分析时间-事件数据的广泛方法，其中一个关键目标是比较不同组别之间的生存曲线。识别生存曲线聚类的传统方法通常依赖于计算密集型的自助法技术来近似原假设分布。这些方法虽然有效，但会带来巨大的计算负担。在这项工作中，我们提出了一种新颖的方法，利用k-means聚类和log-rank检验来高效地识别和聚类生存曲线。我们的方法消除了对计算昂贵的重抽样的需求，在保持统计可靠性的同时显著减少了处理时间。通过系统地评估生存曲线并确定最优聚类，所提出的方法为大规模生存数据分析提供了一种实用且可扩展的替代方案。通过模拟研究，我们证明了……

    arXiv:2512.16481v2 Announce Type: replace-cross  Abstract: Survival analysis encompasses a broad range of methods for analyzing time-to-event data, with one key objective being the comparison of survival curves across groups. Traditional approaches for identifying clusters of survival curves often rely on computationally intensive bootstrap techniques to approximate the null hypothesis distribution. While effective, these methods impose significant computational burdens. In this work, we propose a novel approach that leverages the k-means and log-rank test to efficiently identify and cluster survival curves. Our method eliminates the need for computationally expensive resampling, significantly reducing processing time while maintaining statistical reliability. By systematically evaluating survival curves and determining optimal clusters, the proposed method ensures a practical and scalable alternative for large-scale survival data analysis. Through simulation studies, we demonstrate th
    
[^47]: 非线性上下文老虎机中可证明的任意时刻集成采样算法

    Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits

    [https://arxiv.org/abs/2510.10730](https://arxiv.org/abs/2510.10730)

    该论文提出了一个统一的集成采样算法框架，为广义线性和神经两类非线性上下文老虎机提供了可证明的后悔界，其中GLM-ES达到了与最先进随机探索算法相匹配的性能。

    

    我们为非线性上下文老虎机中的集成采样提供了一个统一的算法框架，并针对两种最常见的非线性上下文老虎机设置开发了相应的后悔界：用于广义线性上下文老虎机的广义线性模型集成采样（GLM-ES），以及用于神经上下文老虎机的神经集成采样。两种方法都通过对随机扰动数据进行最大似然估计来维护奖励模型参数的多个估计器。我们证明了GLM-ES的高概率频率派后悔界为 $\widetilde{\mathcal{O}}(d^{3/2} \sqrt{T} + d^{4})$，Neural-ES的后悔界为 $\widetilde{\mathcal{O}}(\widetilde{d}^{3/2} \sqrt{T})$，其中 $d$ 是特征向量的维度，$\widetilde{d}$ 是神经正切核（NTK）矩阵的有效维度，$T$ 是总轮数。GLM-ES的后悔界与随机探索算法的最先进结果相匹配。

    arXiv:2510.10730v3 Announce Type: replace  Abstract: We provide a unified algorithmic framework for ensemble sampling in nonlinear contextual bandits and develop corresponding regret bounds for two most common nonlinear contextual bandit settings: Generalized Linear Model Ensemble Sampling (GLM-ES) for generalized linear contextual bandits and Neural Ensemble Sampling (Neural-ES) for neural contextual bandits. Both methods maintain multiple estimators for the reward model parameters via maximum likelihood estimation on randomly perturbed data. We prove high-probability frequentist regret bounds of $\widetilde{\mathcal{O}}(d^{3/2} \sqrt{T} + d^{4})$ for GLM-ES and $\widetilde{\mathcal{O}}(\widetilde{d}^{3/2} \sqrt{T})$ for Neural-ES, where $d$ is the dimension of feature vectors, $\widetilde{d}$ is the effective dimension of a neural tangent kernel (NTK) matrix and $T$ is the number of rounds. The regret bound of GLM-ES matches the state-of-the-art result of randomized exploration algor
    
[^48]: 无维数依赖的自归一化集中的变分方法

    A variational approach to dimension-free self-normalized concentration

    [https://arxiv.org/abs/2508.06483](https://arxiv.org/abs/2508.06483)

    本文提出一种变分方法，为向量值随机过程的"sub-ψ"类过程建立无维数依赖的自归一化集中界，推广了经典结果，并首次给出无维数依赖的自归一化经验 Bernstein 不等式。

    

    我们研究向量值随机过程的自归一化集中性。我们聚焦于“sub-ψ”过程的界，这是一类众所周知且相当一般的类别，涵盖了多种著名的尾部条件（包括次指数、次高斯、次伽马、次泊松，以及若干没有矩生成函数的重尾情形，例如对称或二阶、三阶矩有界的情形）。我们的结果在次高斯情形下恢复并推广了 de la Peña 等人 [20] 提出的有影响力的界（该界在 Abbasi-Yadkori 等人 [2] 中被再次证明）。此外，我们填补了文献中基于行列式的界与近期基于条件数的界之间的空白。作为应用，我们证明了满足矩条件（一个比有界性更一般的条件）的随机向量的 Bernstein 不等式，并首次提供了无维数依赖的自归一化经验 Bernstein 不等式。我们的技术……

    arXiv:2508.06483v3 Announce Type: replace-cross  Abstract: We study the self-normalized concentration of vector-valued stochastic processes. We focus on bounds for "sub-$\psi$" processes, a well-known and quite general class that encompasses a wide variety of well-known tail conditions (including sub-exponential, sub-Gaussian, sub-gamma, sub-Poisson, and several heavy-tailed settings without a moment generating function such as symmetric or bounded 2nd or 3rd moments). Our results recover and generalize the influential bound of de la Pe\~na et al. [20] (proved again in Abbasi-Yadkori et al. [2]) in the sub-Gaussian case. Further, we fill a gap in the literature between determinant-based bounds and more recent bounds based on condition numbers. As applications we prove a Bernstein inequality for random vectors satisfying a moment condition (a more general condition than boundedness), and also provide the first dimension-free self-normalized empirical Bernstein inequality. Our techniques
    
[^49]: 具有难解归一化常数的完全观测与部分观测指数族图模型的基于似然的推断

    Likelihood Based Inference in Fully and Partially Observed Exponential Family Graphical Models with Intractable Normalizing Constants

    [https://arxiv.org/abs/2404.17763](https://arxiv.org/abs/2404.17763)

    本文针对具有难解归一化常数的完全观测与部分观测（含潜变量）指数族图模型，提出了基于似然的统计推断方法。

    

    编码潜在马尔可夫随机场的概率图模型是生成建模的基本构建模块，用于在现代具有复杂依赖结构的多变量数据集中学习潜在表示。其中，指数族图模型尤其受欢迎，因为它们具有相当成熟且被充分理解的统计性质，并且基于伪似然方法可在计算上扩展到高维数据。这些模型已成功应用于许多领域，例如统计物理学中的伊辛（Ising）模型和基因组学中的计数图模型。另一类模型允许某些节点为潜变量，从而使可观测节点的边缘分布可以偏离指数族形式，以捕捉更复杂的依赖关系。这些方法构成了人工智能中生成模型的基础，例如玻尔兹曼机及其受限版本。

    arXiv:2404.17763v3 Announce Type: replace-cross  Abstract: Probabilistic graphical models that encode an underlying Markov random field are fundamental building blocks of generative modeling to learn latent representations in modern multivariate data sets with complex dependency structures. Among these, the exponential family graphical models are especially popular, given their fairly well-understood statistical properties and computational scalability to high-dimensional data based on pseudo-likelihood methods. These models have been successfully applied in many fields, such as the Ising model in statistical physics and count graphical models in genomics. Another strand of models allows some nodes to be latent, so as to allow the marginal distribution of the observable nodes to depart from exponential family to capture more complex dependence. These approaches form the basis of generative models in artificial intelligence, such as the Boltzmann machines and their restricted versions. 
    

