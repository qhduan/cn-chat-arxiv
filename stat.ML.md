# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fold Weighting for Cross-Validation Under Continuous Distributional Drift](https://arxiv.org/abs/2611.23184) | 该论文提出在连续分布漂移下对交叉验证的各折进行加权，证明了对权重最小化风险界是一个凸优化问题，且将权重向近期折偏移可以有效降低由漂移导致的部署风险，同时需权衡验证噪声的增加。 |
| [^2] | [Amortizing Tempered Sampling with Corrected Self-Training](https://arxiv.org/abs/2611.23072) | 提出了一种在退火目标序列上进行校正自训练的神经采样方法SSS，通过基于得分的扩散提议与路径空间重要性权重校正，将目标分布的全局结构编码进模型以摊销重复采样成本，在基准测试中取得有竞争力的采样精度且运行时间更短。 |
| [^3] | [Symmetries of Counterfactuals: Identification and Testability of Counterfactual Maps from Covariate Heterogeneity](https://arxiv.org/abs/2611.22948) | 该论文刻画了与观测数据一致的所有反事实映射——它们彼此仅相差一个由协变量条件结果分布决定的对称群（协变量异质性越强，对称群越小），并证明对于三角单调映射和最优传输映射等常见反事实类别，与数据的一致性可归结为一个可检验的条件独立性命题。 |
| [^4] | [Robust Actor-Critic Learning under Transport and Divergence Uncertainty](https://arxiv.org/abs/2611.22759) | 本文提出一种面向连续状态-动作、标称转移未知的折扣鲁棒马尔可夫决策过程的执行器-评论家方法，通过一次性拟合条件扩散模型，并结合共享目标极小极大准则与指数加权得分匹配，处理由Wasserstein距离、全变差及多种散度定义的转移不确定性。 |
| [^5] | [TRAIL-IRL: Crossed Reward-Temperature Mixtures for Unlabeled Trajectory Clustering](https://arxiv.org/abs/2611.22715) | 该论文提出TRAIL-IRL方法，通过将奖励（表示偏好）与共享温度集合（表示动作随机性）交叉配对的混合模型对无标签轨迹进行聚类，建立了该模型的可辨识性条件，并证明传统的共同温度奖励混合模型无法重现这种交叉混合结构。 |
| [^6] | [The Price of Hard Coverage Constraints in Online Conformal Prediction](https://arxiv.org/abs/2611.22646) | 本文刻画了在线共形预测中强制硬覆盖约束（累积误覆盖受预算限制）不可避免的最小预测集膨胀代价，并证明即使在数据独立同分布、分数分布已知的最理想情况下，这一代价依然无法避免。 |
| [^7] | [SOTA or Luck? The Winner's Curse in LLM Leaderboards](https://arxiv.org/abs/2611.22227) | 该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。 |
| [^8] | [Training Neural Networks to be Locally Adaptive](https://arxiv.org/abs/2611.22198) | 本文提出针对每个查询点仅用其k个最近邻训练“局部神经网络”的局部自适应拟合方法，并证明了适当选择k时，估计器在局部α-Hölder函数上可达到 $n^{-2\alpha/(2\alpha+d^{*})}$ 的逐点风险速率。 |
| [^9] | [Centroid Spectral Coupling: Implicit Alignment and Representation Diversity in Self-Supervised Learning](https://arxiv.org/abs/2611.22158) | 该论文提出质心谱耦合（CSC）——一种单项自监督学习目标函数，通过基于质心二阶矩矩阵的谱优化，隐式地同时实现视图对齐、表示多样性和抗坍缩，并给出精确分解与理论保证。 |
| [^10] | [Bandit Online Submodular Minimisation Using Zeroth-Order Oracles](https://arxiv.org/abs/2611.22130) | 本文提出了首个在bandit反馈下可保证跟踪移动最小化元的在线次模最小化零阶方法，其基于Lovász扩展的无偏两点梯度估计并建立了动态遗憾界。 |
| [^11] | [COSI: A Framework for Robustly Valid Inference](https://arxiv.org/abs/2611.22096) | COSI框架利用估计量的尺度不变性，通过分批估计构建置信集，在较弱的假设下即可实现渐近有效推断，且小样本覆盖率更接近名义水平。 |
| [^12] | [Two Counterexamples for e-means](https://arxiv.org/abs/2611.22065) | 本文给出两个反例，证明 e-means 算法在相同初始中心下最终损失可能高于标准 k-means，且其基于质量的聚类复活机制可能破坏全局最优解，导致期望损失之比无界。 |
| [^13] | [Locally Linear Generalized Single Index Regression](https://arxiv.org/abs/2611.22011) | 提出了一种在对协变量分布要求极低的假设下估计广义单指标回归模型的局部加权线性最小二乘方法，能够联合估计梯度向量和条件均值（即使链接函数不连续），并为其建立了中心极限定理。 |
| [^14] | [DEFTS: Data-Efficient, Training-Free Few-Shot Learning for CLIP in the Text Subspace](https://arxiv.org/abs/2611.21993) | DEFTS是一种数据高效、免训练的小样本分类方法，其核心创新在于通过将图像原型投影到文本嵌入所张成的判别性子空间中来缩小图像-文本模态差距，从而提升CLIP的小样本分类性能。 |
| [^15] | [Modeling Document and Section Hierarchies in Embedding Space](https://arxiv.org/abs/2611.21980) | 该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。 |
| [^16] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^17] | [Adaptive Memory for Online Reinforcement Learning under Sparse Model Changes](https://arxiv.org/abs/2611.21879) | 针对部署间稀疏变化的强化学习环境，提出一种自适应记忆算法，为每个状态-动作对动态决定复用多少历史观测，从而在利用过去数据与避免已变化条目带来的估计偏差之间取得平衡。 |
| [^18] | [Bias-Reduced and Parallel Heavy-Tailed Stochastic Convex Optimization in $\ell_p$ Geometries](https://arxiv.org/abs/2611.21844) | 本文在重尾噪声模型下研究 ℓ_p 几何中的并行随机凸优化，提出了一类结合随机去偏差技术与多层蒙特卡罗方法的新颖 ODC_p 算法，可高效估计一致凸目标函数的最优值并给出相应的收敛速率。 |
| [^19] | [Variance-adaptive confidence regions for fixed-benchmark binomial mixtures, with application to LLM evaluation](https://arxiv.org/abs/2611.21266) | 该论文提出方差自适应（VA）置信区域方法，基于固定基准上的二项计数为经验分布线性泛函构造对任意有界泛函同时具有有限样本覆盖保证的置信区间，其可通过线性规划计算并在一阶最坏情形意义下达到最优宽度，可直接应用于大语言模型评估。 |
| [^20] | [Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds](https://arxiv.org/abs/2611.00771) | 本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。 |
| [^21] | [Formalizing the Intrinsic Dimension of Objective Landscapes by a Generalized Notion of Co-dimension](https://arxiv.org/abs/2610.39674) | 本文将目标景观的内在维度形式化为一种依赖于集合全局几何的广义余维数概念，并为浅层和深层ReLU网络学生-教师设置中的最优参数集合推导出内在维度的上下界。 |
| [^22] | [Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences](https://arxiv.org/abs/2610.33262) | 该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。 |
| [^23] | [Spectral Recovery of Point Clouds from Noisy Geometric Graphs](https://arxiv.org/abs/2610.08634) | 该论文证明在点数与维度均趋于无穷的高维信号+噪声图模型下，只要满足谱间隙条件，利用邻接矩阵的顶部特征向量和特征值即可在正交变换意义下近似恢复被高斯噪声扰动的低维点云。 |
| [^24] | [Generative sequence modeling for infinite memory processes via predictive states](https://arxiv.org/abs/2609.38524) | 本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。 |
| [^25] | [Representation Learning with Quantum Signal Processing](https://arxiv.org/abs/2608.28828) | 该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。 |
| [^26] | [FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data](https://arxiv.org/abs/2606.17516) | FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。 |
| [^27] | [Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning](https://arxiv.org/abs/2606.05046) | Graph Cascades提出了一种基于传染扩散的介观图重连策略，在线性时间内将受多跳强化支持的节点对提升为直接邻居，并从理论（标签一致性条件、SBM同配性、有效电阻刻画）和实验上证明其能提升多种GNN与稀疏图Transformer在节点分类任务上的性能。 |
| [^28] | [Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression](https://arxiv.org/abs/2606.00512) | 该论文提出一种简单的两阶段半监督回归估计器——从大量含噪声的预训练代理协变量中学习核特征值特征、并在少量标签上拟合岭回归预测器，从理论上证明当代理扰动可控且无标签数据充足时可恢复快速标签样本率，并将分布回归作为特例纳入统一的泛化保证框架。 |
| [^29] | [Batched Stochastic Linear Bandits with 1-Bit Communication Constraints](https://arxiv.org/abs/2605.30976) | 该论文首次研究了批处理与1比特通信约束相结合的随机线性老虎机问题，并证明了即使在无噪声环境下，1比特通信瓶颈也会导致 $\Omega(B\min\{d,\log|\mathcal{A}|\})$ 的不可避免的遗憾下界。 |
| [^30] | [Kernel-based guarantees for nonlinear parametric models in Bayesian optimization](https://arxiv.org/abs/2605.13160) | 本文提出一个基于核方法的框架，为自适应采集数据上训练的正则化非线性参数模型建立置信界，并据此为近似汤普森采样等贝叶斯优化方法提供收敛保证。 |
| [^31] | [Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks](https://arxiv.org/abs/2605.10395) | 该论文利用留一解耦论证与封闭不动点方程，在高维广延宽度设定下首次给出贝叶斯最优泛化误差的渐近精确刻画，并揭示教师特征随数据量增长依次通过不连续跳变被学习的尖锐相变序列及其背后的神经标度律。 |
| [^32] | [Measuring and Decomposing Mode Separation via the Canonical Diffusion](https://arxiv.org/abs/2605.08777) | 该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。 |
| [^33] | [On Minimal Observation Time for Recovering Latent Hawkes Networks](https://arxiv.org/abs/2605.08400) | 本文针对稀疏弱相互作用的平稳霍克斯过程，证明了精确恢复潜在网络所需的最小观测时间为 $\log d$ 量级（其中 $d$ 为实体数量），该时间界既是充分的也是必要的，并提出了达到该界的两阶段估计方法。 |
| [^34] | [A Cherry-Picking Approach to Large Load Shaping for More Effective Carbon Reduction](https://arxiv.org/abs/2601.17990) | 本文通过构建近8900节点的ERCOT日前市场出清流程校准复现模型进行全年因果调度反事实分析，发现基于节点边际电价的负荷整形在年度电网碳减排中优于其他静态信号，但没有单一信号能在所有运行条件下均表现最佳，因而提出择优选择信号的负荷整形策略以实现更有效的碳减排。 |
| [^35] | [On the use of case estimate and transactional payment data in neural networks for individual loss reserving](https://arxiv.org/abs/2601.05274) | 该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。 |
| [^36] | [Functional Compression as Occam's Razor: Algebraic Complexity from Sensitivity Minimization](https://arxiv.org/abs/2511.23152) | 该论文为代数补全分解模型建立了以参数敏感度下确界度量的普适复杂度下界，证明达到该下界会迫使潜在表示几何对齐且幺正，并使模型通过共享代数组合（功能压缩）而非低秩表示来学习到全局统一的代数结构。 |
| [^37] | [Scalable Learning from Probability Measures with Mean Measure Quantization](https://arxiv.org/abs/2502.04907) | 该论文提出一种均值测度量化方法，将所有输入概率测度近似为共享公共支撑集的K点离散测度，在保证量化测度一致性和下游任务收敛性的同时，大幅降低最优传输计算的时间成本并保持与单独量化相当的性能。 |
| [^38] | [Exact Sequence Interpolation with Transformers](https://arxiv.org/abs/2502.02270) | 该论文通过交替使用前馈层与自注意力层并采用低秩参数矩阵，证明了Transformer能够以与输入序列长度无关的复杂度精确插值有限序列数据集。 |
| [^39] | [Clustering in pure-attention hardmax transformers and its role in sentiment analysis](https://arxiv.org/abs/2407.01602) | 本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。 |
| [^40] | [High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media](https://arxiv.org/abs/2403.01318) | 提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。 |
| [^41] | [Algorithmic Insurance](https://arxiv.org/abs/2106.00839) | 该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。 |
| [^42] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |

# 详细

[^1]: 连续分布漂移下交叉验证的折加权方法

    Fold Weighting for Cross-Validation Under Continuous Distributional Drift

    [https://arxiv.org/abs/2611.23184](https://arxiv.org/abs/2611.23184)

    该论文提出在连续分布漂移下对交叉验证的各折进行加权，证明了对权重最小化风险界是一个凸优化问题，且将权重向近期折偏移可以有效降低由漂移导致的部署风险，同时需权衡验证噪声的增加。

    

    交叉验证对所有折赋予相同权重，因此在分布漂移存在时，最旧的折与最新的折具有同等的重要性。我们通过全变差距离下的赫尔德（Hölder）条件对漂移进行建模，并给出了滑动窗口交叉验证所选配置在部署时的超额风险的有限样本界。每个界都分解为漂移项、估计项和验证噪声项。在使用等权重时，随着数据记录的增长，漂移项会趋于饱和。当折的权重预先固定时，漂移项变为加权平均值，并随着权重向部署时刻移动而缩小。噪声项随着权重集中而增大，并随批量大小的增加而缩小。由于漂移项关于权重是线性的，而噪声项是凸的，因此在该风险界上对权重进行最小化是一个凸优化问题。在美国收入数据和航班延误数据上的应用展示了这一权衡的两个方面。当最优预测器的变化持续存在且每个时期……（摘要原文在此处截断）

    arXiv:2611.23184v1 Announce Type: cross  Abstract: Cross-validation weights all folds equally, so under drift the oldest folds count as much as the newest. We model drift by a H\"older condition in total variation distance and give finite-sample bounds on the excess deployment risk of the configuration that sliding-window cross-validation selects. Each bound splits into drift, estimation, and validation-noise terms. With equal weights the drift terms saturate as the record grows. With fold weights fixed in advance, the drift terms become weighted averages, which shrink as weight moves toward deployment. The noise term grows as the weights concentrate and shrinks with the batch size. The drift terms are linear in the weights and the noise term is convex, so minimizing the bound over the weights is a convex problem. Applications to U.S. earnings and flight delays show both sides of this trade-off. Weight on recent folds pays off when a change in the best predictor persists and each perio
    
[^2]: 通过校正自训练实现摊销式退火采样

    Amortizing Tempered Sampling with Corrected Self-Training

    [https://arxiv.org/abs/2611.23072](https://arxiv.org/abs/2611.23072)

    提出了一种在退火目标序列上进行校正自训练的神经采样方法SSS，通过基于得分的扩散提议与路径空间重要性权重校正，将目标分布的全局结构编码进模型以摊销重复采样成本，在基准测试中取得有竞争力的采样精度且运行时间更短。

    

    学习一个神经采样器可以降低从非归一化分布中重复采样的成本，但获得良好的训练数据仍然困难。我们开发了一种跨越一系列退火（温度调节）目标的校正自训练方案，利用重要性采样和马尔可夫链蒙特卡洛（MCMC）来校正样本，并使用重放缓冲区来存储它们。我们引入了序列得分采样器作为一个具体实例。在SSS中，提议分布是通过在能量梯度上进行回归训练的基于得分的扩散模型；样本则通过路径空间上的重要性权重进行校正。训练完成后，可以直接从模型本身、从其经重要性校正的提议、或从最终温度的缓冲区中收集样本。该方法通过将目标分布的全局结构编码到模型中，从而摊销了重复采样的成本。SSS在各项基准测试中取得了具有竞争力的采样精度，并且在若干目标上以更短的运行时间……

    arXiv:2611.23072v1 Announce Type: cross  Abstract: Learning a neural sampler can reduce the cost of repeated sampling from an un-normalized distribution, but obtaining good training data remains difficult. We develop a corrected self-training scheme across a sequence of tempered targets, making use of importance sampling and Markov chain Monte Carlo to correct the samples and replay buffers to store them. We introduce Sequential Score Sampler (SSS) as a particular case. In SSS, the proposal is a score-based diffusion trained by regression on energy gradients; and the samples are corrected by importance weights on the path space. After training, samples can be collected directly from the model, from its importance-corrected proposals, or from the final-temperature buffer. The method amortizes repeated sampling by encoding the global structure of the target in the model. SSS achieves competitive sampling accuracy across the benchmarks and, on several targets, uses shorter runtimes with f
    
[^3]: 反事实的对称性：从协变量异质性出发研究反事实映射的可识别性与可检验性

    Symmetries of Counterfactuals: Identification and Testability of Counterfactual Maps from Covariate Heterogeneity

    [https://arxiv.org/abs/2611.22948](https://arxiv.org/abs/2611.22948)

    该论文刻画了与观测数据一致的所有反事实映射——它们彼此仅相差一个由协变量条件结果分布决定的对称群（协变量异质性越强，对称群越小），并证明对于三角单调映射和最优传输映射等常见反事实类别，与数据的一致性可归结为一个可检验的条件独立性命题。

    

    一般而言，反事实无法仅凭数据得到识别，因为许多因果模型会产生相同的观测分布，却对应不同的反事实。规避这一问题的常见做法是将机制限制在使其唯一确定的类别中，例如三角单调映射或最优传输映射。然而，所假设的类别是否正确通常并未得到检验。在本工作中，我们研究了当存在仅通过潜在原因影响结果的协变量时，数据能够告诉我们什么。首先，我们刻画了所有与数据一致的反事实映射：它们与真实映射的差异仅在于一组保持“给定协变量条件下结果分布”不变的对称性；这些条件分布在协变量取值之间变化越大，该对称群就越小。其次，我们证明，对于上述类别，与数据的一致性等价于一个单一的条件独立性命题（摘要原文在此处截断）。

    arXiv:2611.22948v1 Announce Type: cross  Abstract: In general, counterfactuals are not identified from data alone, because many causal models produce the same observed distributions but different counterfactuals. A common approach to circumvent this problem is to restrict the mechanism to a class in which it becomes unique, such as triangular monotone maps or optimal transport maps. However, whether the assumed class is correct is usually not examined. In this work, we study what the data can tell us when covariates are available that influence the outcome only through its latent cause. First, we characterise all counterfactual maps that are consistent with the data. They differ from the true map by a group of symmetries that preserve the conditional outcome distributions given the covariates. The more these distributions vary across covariate values, the smaller the group. Second, we show that for the classes above, consistency with the data is equivalent to a single conditional indep
    
[^4]: 运输距离与散度不确定性下的鲁棒执行器-评论家学习

    Robust Actor-Critic Learning under Transport and Divergence Uncertainty

    [https://arxiv.org/abs/2611.22759](https://arxiv.org/abs/2611.22759)

    本文提出一种面向连续状态-动作、标称转移未知的折扣鲁棒马尔可夫决策过程的执行器-评论家方法，通过一次性拟合条件扩散模型，并结合共享目标极小极大准则与指数加权得分匹配，处理由Wasserstein距离、全变差及多种散度定义的转移不确定性。

    

    学习可靠的控制策略需要考虑转移动力学中的不确定性，以及函数估计和动作生成所引入的误差。我们针对具有连续状态与动作且标称转移规律未知的折扣鲁棒马尔可夫决策过程，开发了一种执行器-评论家方法。我们考虑由Wasserstein距离、全变差、Kullback-Leibler散度或Pearson卡方散度所定义的转移不确定性集合。该方法只需对离线转移数据一次性拟合一个条件扩散模型。在随后的每次迭代中，我们使用共享目标的极小极大准则，从当前策略的观测数据中估计鲁棒动作价值函数，然后通过指数加权得分匹配来更新生成式执行器。正则化在保持探索性的同时，为条件分布估计提供了所需的密度正则性。在覆盖性与正则性条件下，我们建立了有限……（摘要原文在此处截断）

    arXiv:2611.22759v1 Announce Type: cross  Abstract: Learning reliable control policies requires accounting for uncertainty in transition dynamics and errors introduced by function estimation and action generation. We develop an actor--critic method for discounted robust Markov decision processes with continuous states and actions and an unknown nominal transition law. We consider transition uncertainty sets defined by Wasserstein distance, total variation, Kullback--Leibler divergence, or Pearson chi-square divergence. A conditional diffusion model is fitted once to offline transitions. At each subsequent iteration, we estimate the robust action-value function from current-policy observations using a shared-target minimax criterion, then update a generative actor through exponentially weighted score matching. Regularization preserves exploration and provides the density regularity needed for conditional distribution estimation. Under coverage and regularity conditions, we establish fini
    
[^5]: TRAIL-IRL：用于无标签轨迹聚类的奖励-温度交叉混合模型

    TRAIL-IRL: Crossed Reward-Temperature Mixtures for Unlabeled Trajectory Clustering

    [https://arxiv.org/abs/2611.22715](https://arxiv.org/abs/2611.22715)

    该论文提出TRAIL-IRL方法，通过将奖励（表示偏好）与共享温度集合（表示动作随机性）交叉配对的混合模型对无标签轨迹进行聚类，建立了该模型的可辨识性条件，并证明传统的共同温度奖励混合模型无法重现这种交叉混合结构。

    

    不同的路径并不一定意味着不同的偏好，轨迹在动作选择的随机性程度上同样存在差异。许多多奖励逆强化学习（IRL）方法假设动作随机性是共同的，这可能会将其与偏好异质性相混淆。我们提出了TRAIL-IRL，该方法使用奖励来刻画偏好，使用温度来刻画动作随机性，从而对无标签轨迹进行聚类。其交叉混合模型将每个奖励与一组共享的温度水平配对，允许动作规律性在偏好组内部发生变化。在已知有限状态环境中使用中心化和归一化的奖励，我们建立了可辨识性条件，在这些条件下，足够长的轨迹分布能够确定奖励、温度、混合权重以及类别数量（在标签重排的意义下）。在相同的条件和奖励约束下，具有相同分量数量的共同温度奖励混合模型无法重现交叉混合模型。

    arXiv:2611.22715v1 Announce Type: cross  Abstract: Different paths need not imply different preferences, trajectories also differ in how randomly actions are chosen. Many multi-reward inverse reinforcement learning (IRL) methods assume common action randomness and can confound it with preference heterogeneity. We introduce TRAIL-IRL, which clusters unlabeled trajectories using rewards for preferences and temperatures for action randomness. Its crossed mixture pairs each reward with a shared set of temperature levels, allowing action regularity to vary within preference groups. With centered and normalized rewards in a known finite-state environment, we establish the identifiability conditions under which the distribution of sufficiently long trajectories determines rewards, temperatures, mixing weights, and class counts up to relabeling. Under the same conditions and reward constraints, common-temperature reward mixtures with the same component count cannot reproduce a crossed mixture 
    
[^6]: 在线共形预测中硬覆盖约束的代价

    The Price of Hard Coverage Constraints in Online Conformal Prediction

    [https://arxiv.org/abs/2611.22646](https://arxiv.org/abs/2611.22646)

    本文刻画了在线共形预测中强制硬覆盖约束（累积误覆盖受预算限制）不可避免的最小预测集膨胀代价，并证明即使在数据独立同分布、分数分布已知的最理想情况下，这一代价依然无法避免。

    

    共形预测提供了无分布假设的不确定性量化方法，但在序列化部署中，标准的覆盖率保证仍可能允许误覆盖次数随时间出现大幅波动。我们研究了一种在线设置：预测集逐轮产生，且累积误覆盖次数必须在整个部署过程中保持在规定的预算之内。这种硬覆盖率保证可以通过扩大预测集来强制实现，但从根本上需要多大的额外预测集规模仍不清楚。我们在 $T$ 轮预测中刻画了这种不可避免的效率代价。值得注意的是，即使在数据独立同分布、分数分布已知、以及对总体最优预测集拥有完全了解的理想情况下，这种代价依然存在。在将覆盖率与预测集规模相关联的适当正则性条件下，我们确定了任何在线方法为实现硬覆盖保证所必须付出的最小累积超额预测集规模。

    arXiv:2611.22646v1 Announce Type: new  Abstract: Conformal prediction provides distribution-free uncertainty quantification, but in sequential deployment, standard coverage guarantees can still allow substantial fluctuations in the number of miscoverages over time. We study an online setting in which prediction sets are produced one round at a time and the cumulative number of miscoverages must stay within a prescribed budget throughout deployment.Such a hard coverage guarantee can be enforced by enlarging prediction sets, but how much additional prediction-set size is fundamentally necessary remains unclear. We characterize this unavoidable efficiency cost over $T$ prediction rounds. Remarkably, the cost persists even with i.i.d.\ data, a known score distribution, and perfect knowledge of the population-optimal prediction set. Under suitable regularity conditions relating coverage and prediction-set size, we determine the minimum cumulative excess prediction-set size required by any o
    
[^7]: SOTA还是运气？LLM排行榜中的赢家诅咒

    SOTA or Luck? The Winner's Curse in LLM Leaderboards

    [https://arxiv.org/abs/2611.22227](https://arxiv.org/abs/2611.22227)

    该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。

    

    公共LLM排行榜指导着模型的评估与采用，然而排名靠前的模型所报告的性能可能被系统性高估。这种“排行榜赢家诅咒”源于对有噪声的基准测试分数进行排名时，会同时奖励采样运气和真实能力。同样的选择效应可能会扭曲整个排行榜上的分数，我们将这一现象称为“排名诱导偏差”。我们开发了一种m-out-of-N自助法（bootstrap）来纠正这种偏差。该方法假设模型池固定、任务独立同分布且有界，同时允许模型间在任务内存在任意相关性。在由正间隔分隔的精确总体层级下，我们证明了该自助法能将Θ(N^{-1/2})的排名偏差降低至期望残差偏差o(N^{-1/2})。在来自5个基准家族的14个基准测试上，排名第一模型的分数调整幅度约为0.09至3.86个百分点。在我们的SWE-bench Verified快照中，并列领先的模型

    arXiv:2611.22227v1 Announce Type: cross  Abstract: Public LLM leaderboards guide model evaluation and adoption, yet the reported performance of top-ranked models can be systematically overstated. This leaderboard winner's curse arises because ranking noisy benchmark scores rewards sampling luck alongside genuine ability. The same selection effect can distort scores throughout the leaderboard, a phenomenon we call the rank-induced bias. We develop an $m$-out-of-$N$ bootstrap to correct for this bias. It assumes a fixed model pool with independent, identically distributed bounded tasks while allowing arbitrary within-task correlation among models. Under exact population tiers separated by positive gaps, we prove that the bootstrap reduces $\Theta(N^{-1/2})$ ranking bias to expected residual bias $o(N^{-1/2})$. Across 14 benchmarks from 5 benchmark families, the rank-1 adjustments range from approximately 0.09 to 3.86 percentage points. In our SWE-bench Verified snapshot, the tied leaders
    
[^8]: 训练神经网络使其具有局部自适应性

    Training Neural Networks to be Locally Adaptive

    [https://arxiv.org/abs/2611.22198](https://arxiv.org/abs/2611.22198)

    本文提出针对每个查询点仅用其k个最近邻训练“局部神经网络”的局部自适应拟合方法，并证明了适当选择k时，估计器在局部α-Hölder函数上可达到 $n^{-2\alpha/(2\alpha+d^{*})}$ 的逐点风险速率。

    

    通过全局经验风险最小化拟合的深度神经网络会在整个协变量空间上采用单一模型。在回归问题中，如果回归函数 $f_0$ 是非齐性的——在某一区域变化剧烈，而在另一区域几乎平坦——那么没有任何单一模型能够处处适用，且拟合效果在函数变化最剧烈的区域会退化。本文研究了一种替代方案：不再对网络进行全局拟合，而是进行局部拟合。对于每个查询点 $x$，仅在其 $k$ 个最近邻样本上训练一个“局部神经网络”，使训练样本和拟合函数均依赖于查询点。我们通过一种局部化方法分析了所得估计器的性能，证明了当 $f_0$ 局部满足 $\alpha$-Hölder（条件时，通过适当选择 $k$，该估计器的逐点风险（在相差对数因子的情况下）可以达到 $n^{-2\alpha/(2\alpha+d^{*})}$ 的速率。

    arXiv:2611.22198v1 Announce Type: cross  Abstract: Deep neural networks fitted by global empirical risk minimization commit to a single model for the entire covariate space. In a regression problem, if the regression function $f_0$ is inhomogeneous --- varying rapidly in one region, while being nearly flat in another --- no single model can be appropriate everywhere, and the fit degrades where the function varies most rapidly. Here we study an alternative in which, instead of fitting the network globally, we fit it locally. For each query point $x$, a \emph{local neural network} is trained on the $k$ nearest-neighbours of $x$ alone, making the training sample and the fitted function query-dependent. We analyze the performance of the resulting estimator through a localization approach, establishing that the pointwise risk of the estimator with an appropriate choice of $k$ attains the rate $n^{-2\alpha/(2\alpha+d^{*})}$, up to logarithmic factors, when $f_0$ is locally $\alpha$-H\"older 
    
[^9]: 质心谱耦合：自监督学习中的隐式对齐与表示多样性

    Centroid Spectral Coupling: Implicit Alignment and Representation Diversity in Self-Supervised Learning

    [https://arxiv.org/abs/2611.22158](https://arxiv.org/abs/2611.22158)

    该论文提出质心谱耦合（CSC）——一种单项自监督学习目标函数，通过基于质心二阶矩矩阵的谱优化，隐式地同时实现视图对齐、表示多样性和抗坍缩，并给出精确分解与理论保证。

    

    自监督学习（SSL）旨在对齐增强视图，同时保持表示多样性并防止坍缩。现有的非对比方法通常使用多个加权损失项来实现这些目标。我们提出了质心谱耦合（CSC），这是一个单项的SSL目标函数，它隐式地将视图对齐、表示多样性和抗坍缩能力耦合在一起。CSC从独立归一化的增强视图对构建质心，并对其未中心化的二阶矩矩阵应用基于岭正则化的谱目标。保持质心的幅值编码了视图之间的一致性，同时促使表示能量在各个谱方向上均衡分布。我们建立了一个精确的分解，将其分解为非负的对齐分量与谱不平衡分量，刻画了全局最优解的几何性质，并推导出对齐与多样性的有限损失保证。我们还证明了完全恒定输出的坍缩是一个……（原文摘要在此处截断）

    arXiv:2611.22158v1 Announce Type: cross  Abstract: Self-Supervised Learning (SSL) aims to align augmented views while preserving representation diversity and preventing collapse. Existing non-contrastive methods often address these goals using multiple weighted loss terms. We introduce Centroid Spectral Coupling (CSC), a single-term SSL objective that implicitly couples view alignment, representation diversity, and collapse resistance. CSC constructs centroids from independently normalized augmented-view pairs and applies a ridge-based spectral objective to their uncentered second-moment matrix. Preserving centroid magnitudes encodes view agreement while encouraging balanced representation energy across spectral directions. We establish an exact decomposition into nonnegative alignment and spectral-imbalance components, characterize the geometry of global optima, and derive finite-loss guarantees for alignment and diversity. We also show that complete constant-output collapse is a stri
    
[^10]: 使用零阶预言机的Bandit在线次模最小化

    Bandit Online Submodular Minimisation Using Zeroth-Order Oracles

    [https://arxiv.org/abs/2611.22130](https://arxiv.org/abs/2611.22130)

    本文提出了首个在bandit反馈下可保证跟踪移动最小化元的在线次模最小化零阶方法，其基于Lovász扩展的无偏两点梯度估计并建立了动态遗憾界。

    

    我们考虑在bandit反馈下的在线次模最小化问题：在每一轮中，决策者选择一个子集，但仅能观测到该子集的代价，而代价函数（因而其最小化元）会随时间变化。现有的bandit方法只能保证静态遗憾，即与单个固定子集进行比较，当最小化元随时间移动时，这是一个较弱的基准。据我们所知，本文提供了在该设置下首个能够保证跟踪移动最小化元的方法。我们提出了一种基于Lovász扩展两点预言机的零阶方法，其两次评估均为可实际执行的子集。扰动点与基准点通过一个共同的随机阈值进行舍入，而从单位球面上抽取平滑方向使得基准评估成为一个零均值控制变量，因此该预言机是平滑梯度的无偏估计。我们建立了数量级为O((1+...)的动态遗憾界。

    arXiv:2611.22130v1 Announce Type: cross  Abstract: We consider online submodular minimisation under bandit feedback, where at each round a decision maker plays a subset and observes only its cost, while the cost functions, and hence their minimiser, change over time. Existing bandit methods only guarantee static regret, which compares against a single fixed subset and is a weak benchmark when the minimiser moves. This paper provides, to the best of our knowledge, the first method with guarantees for tracking a moving minimiser in this setting. We propose a zeroth-order method built on a two-point oracle for the Lov\'asz extension whose two evaluations are both subsets that can be played. The perturbed and base points are rounded by a common random threshold, and drawing the smoothing direction from the unit sphere makes the base evaluation a zero-mean control variate, so the oracle is an unbiased estimator of the smoothed gradient. We establish dynamic regret bounds of order $O\big((1+
    
[^11]: COSI：一个稳健有效推断的框架

    COSI: A Framework for Robustly Valid Inference

    [https://arxiv.org/abs/2611.22096](https://arxiv.org/abs/2611.22096)

    COSI框架利用估计量的尺度不变性，通过分批估计构建置信集，在较弱的假设下即可实现渐近有效推断，且小样本覆盖率更接近名义水平。

    

    我们提出了COSI（基于尺度不变性的置信方法，COnfidence from Scale Invariance），这是一个用于构建统计泛函置信集的框架。COSI建立在HulC（基于包络的置信方法，Hull-based Confidence）之上，后者利用中位数无偏性从分批估计中构建置信集。COSI利用了许多常见估计量所共有的尺度不变性。我们将数据划分为若干批次，在每个批次中计算估计值，并保留那些批次估计通过尺度不变性检验的候选参数值。所得的置信集在比经典Wald方法和bootstrap方法更弱的假设下具有渐近有效性，且在小样本情况下其覆盖率可以更接近名义水平。应用包括对均值的稳健推断、误设线性模型中的回归系数、张量范数以及Fréchet均值的推断。

    arXiv:2611.22096v1 Announce Type: cross  Abstract: We introduce COSI (COnfidence from Scale Invariance), a framework for constructing confidence sets for statistical functionals. COSI builds on the HulC (Hull-based Confidence), which uses median unbiasedness to construct confidence sets from batch estimates. COSI uses scale-invariance properties shared by many common estimators. We divide the data into batches, compute an estimate in each batch, and retain each candidate parameter value for which the batch estimates pass a test of scale invariance. The resulting confidence sets are asymptotically valid under weaker assumptions than classical Wald and bootstrap procedures require, and their coverage can be closer to the nominal level in small samples. Applications include robust inference for the mean, regression coefficients in misspecified linear models, tensor norms, and Fr\'echet means.
    
[^12]: 关于 e-means 的两个反例

    Two Counterexamples for e-means

    [https://arxiv.org/abs/2611.22065](https://arxiv.org/abs/2611.22065)

    本文给出两个反例，证明 e-means 算法在相同初始中心下最终损失可能高于标准 k-means，且其基于质量的聚类复活机制可能破坏全局最优解，导致期望损失之比无界。

    

    e-means（Bessa 等人，2026）将指数平均与聚类复活机制相结合。我们针对其已发表的算法方案给出了两个反例，证明在相同的初始中心条件下，该算法的最终损失高于标准 k-means（Lloyd，1982）。基于质量的复活机制可能会破坏全局最优解，即使使用 k-means++ 进行初始化，期望损失之比仍可能是无界的。

    arXiv:2611.22065v1 Announce Type: new  Abstract: e-means (Bessa et al., 2026) combines exponential averaging and cluster revival. We give two counterexamples to its published recipe with higher final loss than standard k-means (Lloyd, 1982) from identical initial centers. Mass-based revival can destroy a global optimum and make the ratio of expected losses unbounded even with k-means++ seeding.
    
[^13]: 局部线性广义单指标回归

    Locally Linear Generalized Single Index Regression

    [https://arxiv.org/abs/2611.22011](https://arxiv.org/abs/2611.22011)

    提出了一种在对协变量分布要求极低的假设下估计广义单指标回归模型的局部加权线性最小二乘方法，能够联合估计梯度向量和条件均值（即使链接函数不连续），并为其建立了中心极限定理。

    

    在本工作中，我们提出并分析了用于估计广义单指标回归模型的一种方法。该广义单指标回归框架将响应变量 $Y$ 的条件均值建模为协变量 $X$ 的函数，其中映射被假设具有 $f \circ g$ 的形式，其中 $g$ 是 $\beta \geq 2$ 的 $\beta$-Hölder 函数，而 $f$ 可以是一个有界变差的不连续函数。我们提出了一种通用方法，该方法在关于协变量 $X$ 分布的极小假设下运行，只要求有界性、足够的平滑性以及关于勒贝格测度的绝对连续性。我们的方法通过求解局部加权线性最小二乘问题，联合估计梯度向量 $\hat v(x) \approx \nabla g(x)$ 和条件均值 $\mu(x) = f(g(x))$。我们推导了梯度向量估计 $\hat{v}_x$ 和回归估计 $\hat{\mu}(x)$ 的中心极限定理，并进一步推导了 r（摘要内容在此处被截断）

    arXiv:2611.22011v1 Announce Type: cross  Abstract: In this work we propose and analyze a procedure for estimating a generalized single-index regression model. This generalized single-index regression framework models the conditional mean of the responses $Y$ as a function of the covariates $X$ where the mapping is assumed to have the form $f \circ g$, where $g$ is $\beta$-H\"older with $\beta \geq 2$ and $f$ can be a discontinuous function of bounded variation. We propose a general procedure that operates under minimal assumptions about the distribution of covariates $X$, requiring only boundedness, sufficient smoothness and absolute continuity with respect to Lebesgue. Our procedure jointly estimates the gradient vector $\hat v(x) \approx \nabla g(x)$ and the conditional mean $\mu(x) = f(g(x))$ by solving a localized weighted linear least squares problem. We derive central limit theorems for gradient vector estimates $\hat{v}_x$ and regression estimates $\hat{\mu}(x)$, and we derive r
    
[^14]: DEFTS：面向CLIP的文本子空间中数据高效、免训练的小样本学习

    DEFTS: Data-Efficient, Training-Free Few-Shot Learning for CLIP in the Text Subspace

    [https://arxiv.org/abs/2611.21993](https://arxiv.org/abs/2611.21993)

    DEFTS是一种数据高效、免训练的小样本分类方法，其核心创新在于通过将图像原型投影到文本嵌入所张成的判别性子空间中来缩小图像-文本模态差距，从而提升CLIP的小样本分类性能。

    

    对比语言-图像预训练（CLIP）通过将图像嵌入与类别特定的文本嵌入进行比较，实现了令人印象深刻的零样本图像分类性能。尽管近期的小样本方法利用带标签的训练样本来改进这一零样本基线，但它们通常主要依赖图像嵌入，而未能充分利用文本嵌入的语义结构。在本文中，我们提出了DEFTS，一种严格在文本子空间中进行的数据高效、免训练的小样本分类方法。我们首先通过实证表明，虽然原始文本嵌入作为直接分类器可能是次优的，但它们所张成的线性子空间仍然具有高度判别性。因此，DEFTS提出将估计的图像原型直接投影到该子空间中，以缩小图像-文本模态差距。重要的是，通过将这些原型表示为文本嵌入的线性组合，DEFTS降低了高维估计（原文在此处截断）

    arXiv:2611.21993v1 Announce Type: cross  Abstract: Contrastive Language-Image Pre-training (CLIP) achieves impressive zero-shot image classification performance by comparing image embeddings against class-specific text embeddings. While recent few-shot methods leverage labeled training samples to improve upon this zero-shot baseline, they often predominantly rely on image embeddings and underutilize the semantic structure of the text embeddings. In this paper, we introduce DEFTS, a data-efficient, training-free few-shot classification method strictly in the text subspace. We first demonstrate empirically that while raw text embeddings can be suboptimal as direct classifiers, the linear subspace they span remains highly discriminative. DEFTS therefore proposes to reduce the image-text modality gap by projecting estimated image prototypes directly into this subspace. Importantly, by expressing these prototypes as linear combinations of text embeddings, DEFTS reduces a high-dimensional es
    
[^15]: 在嵌入空间中对文档与章节层次结构建模

    Modeling Document and Section Hierarchies in Embedding Space

    [https://arxiv.org/abs/2611.21980](https://arxiv.org/abs/2611.21980)

    该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。

    

    嵌入式主题模型将词嵌入整合到主题生成过程中，能够产生连贯的主题，并在大词表上表现出稳健的性能。然而，每篇文档仅由单个主题分布表示，忽略了其内部结构。许多现实世界的语料库，包括法律文档、公司和政府报告以及在线讨论论坛，都呈现出层次化的组织结构：文档由多个章节组成，而这些章节往往涉及不同的子主题。我们提出了层次化嵌入式主题模型，这是一种结构化主题模型，通过嵌入空间中的层次先验将文档的主题分布与其各章节的主题分布相耦合。我们为 hETM 推导了一种摊销变分推断过程，并在来自上述各领域的结构化语料库上对其进行评估。实验结果表明，hETM 在扁平模型与……之中取得了最低或接近最低的留出集困惑度（摘要原文在此处截断）。

    arXiv:2611.21980v1 Announce Type: cross  Abstract: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and
    
[^16]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^17]: 稀疏模型变化下在线强化学习的自适应记忆

    Adaptive Memory for Online Reinforcement Learning under Sparse Model Changes

    [https://arxiv.org/abs/2611.21879](https://arxiv.org/abs/2611.21879)

    针对部署间稀疏变化的强化学习环境，提出一种自适应记忆算法，为每个状态-动作对动态决定复用多少历史观测，从而在利用过去数据与避免已变化条目带来的估计偏差之间取得平衡。

    

    我们研究底层环境在部署之间发生变化的折扣强化学习（RL）问题。每次部署提供来自一个固定MDP的H个观测，奖励函数和转移核在每次部署内部保持固定；而在不同部署之间，最多有比例为 ε>0 的奖励和转移核条目可能发生变化。在这种设定下，从头学习每个待部署模型可能会丢弃以往部署中有用的观测，而将所有过去的观测汇集在一起则可能使已经发生变化条目的估计产生偏差。为此，我们提出一种算法：针对每个状态-动作对，选择复用多少历史数据来估计奖励和下一状态的概率。我们从当前部署的观测开始，按由近及远的顺序逐个加入更早部署的观测。在每次添加后，我们检验是否单一的……

    arXiv:2611.21879v1 Announce Type: new  Abstract: We study the discounted reinforcement learning (RL) problem in a setting where the underlying environment changes between deployments. Each deployment provides $H$ observations from a fixed MDP. The reward function and transition kernel are fixed within each deployment. Across deployments, at most an $\epsilon >0$ fraction of the reward and transition kernel entries may change. In this setting, learning each model to deploy from scratch may discard useful observations from past deployments, whereas pooling all past observations can bias the estimates of entries that have already changed. We then propose an algorithm that for each state and action pair, selects how much history to reuse to estimate the reward and probabilities of the next state. We start with observations from the current deployment and add earlier deployment observations one at a time, the most recent first. After each addition, we check whether a single average reward a
    
[^18]: ℓ_p 几何下去偏差与并行化的重尾随机凸优化

    Bias-Reduced and Parallel Heavy-Tailed Stochastic Convex Optimization in $\ell_p$ Geometries

    [https://arxiv.org/abs/2611.21844](https://arxiv.org/abs/2611.21844)

    本文在重尾噪声模型下研究 ℓ_p 几何中的并行随机凸优化，提出了一类结合随机去偏差技术与多层蒙特卡罗方法的新颖 ODC_p 算法，可高效估计一致凸目标函数的最优值并给出相应的收敛速率。

    

    我们考虑并行随机凸优化（SCO）这一基本问题，其目标是在每一轮可以多次并行查询矩有界的随机梯度预言机的条件下，近似最小化一个凸目标函数。以往的大多数结果都集中在方差有界的欧几里得（ℓ_2 范数）设定上，即目标函数在 ℝ^d 中 ℓ_2 有界的可行域上关于 ℓ_2 范数是 Lipschitz 的；而本工作则研究重尾噪声模型下 2 ≤ p < ∞ 的 ℓ_p 设定中的并行随机凸优化。我们首先引入了一类新的 ℓ_p 最优距离收敛（ODC_p）算法，借鉴了先前的随机去偏差技术 [ACJ+21]，该算法与多层蒙特卡罗方法相结合，可以高效地估计一致凸目标函数的最优值。作为 ODC_p 算法的结果，我们首先建立了相应速率……（摘要在此处截断）

    arXiv:2611.21844v1 Announce Type: cross  Abstract: We consider the fundamental problem of parallel stochastic convex optimization (SCO), whose aim is to approximately minimize a convex objective when given the ability to query a moment-bounded stochastic gradient oracle multiple times in parallel per round. Whereas most previous results have focused on the variance-bounded Euclidean ($\ell_2$-norm) setting, meaning the objective is Lipschitz with respect to the $\ell_2$-norm over an $\ell_2$-bounded feasible set in $\mathbb{R}^d$, in this work we study parallel SCO in $\ell_p$ settings for $2 \le p < \infty$ under a heavy-tailed noise model. We begin by introducing a new class of $\ell_p$-optimal-distance-convergence (ODC$_p$) algorithms, drawing on prior stochastic bias-reduced techniques [ACJ+21], which combine with multilevel Monte Carlo methods to efficiently estimate the optimum of uniformly convex objectives. As a consequence of our ODC$_p$ algorithms, we first establish rates fo
    
[^19]: 固定基准下二项混合分布的方差自适应置信区域及其在大语言模型评估中的应用

    Variance-adaptive confidence regions for fixed-benchmark binomial mixtures, with application to LLM evaluation

    [https://arxiv.org/abs/2611.21266](https://arxiv.org/abs/2611.21266)

    该论文提出方差自适应（VA）置信区域方法，基于固定基准上的二项计数为经验分布线性泛函构造对任意有界泛函同时具有有限样本覆盖保证的置信区间，其可通过线性规划计算并在一阶最坏情形意义下达到最优宽度，可直接应用于大语言模型评估。

    

    我们为成功概率 $p_1,\dots,p_m$ 的经验分布 $F_m$ 的线性泛函 $\theta(F_m)=\int h\,dF_m$ 构造置信区间，数据来自独立计数 $c_i\sim\mathrm{Bin}(n,p_i)$，例如语言模型在每个问题上运行 $n$ 次的情形。对于任意的 $n$、$m$ 和固定的 $(p_i)$，覆盖率至少为 $1-\alpha$，且对所有有界的 $h$ 同时成立。方差自适应（VA）区域用它所蕴含的尾部计数的条件方差来检验每个候选混合分布。该方差关于分布是线性的，因此该区域可表示为一个线性规划，而胞腔包络使其在网格之外仍然有效。为独立同分布抽样校准的区域针对的是超总体分布，对于 $\theta(F_m)$ 而言是保守的。在利用全部 $n$ 个尾部计数、固定 $n$、连续 $h$ 且网格随 $m\to\infty$ 不断加密的条件下，VA 宽度的识别部分在一阶最坏情形意义上是最优的，与具有相同指标集的任何局部化方法相当。在预先固定的规则下……（原文摘要在此处截断）

    arXiv:2611.21266v1 Announce Type: cross  Abstract: We construct confidence intervals for linear functionals $\theta(F_m)=\int h\,dF_m$ of the empirical law $F_m$ of success probabilities $p_1,\dots,p_m$, from independent counts $c_i\sim\mathrm{Bin}(n,p_i)$, as for a language model run $n$ times per problem. Coverage is at least $1-\alpha$ for every $n$, $m$ and fixed $(p_i)$, simultaneously for all bounded $h$. The variance-adaptive (VA) region tests each candidate mixing law with the conditional variance of the tail counts it implies. This variance is linear in the law, so the region is a linear program, and cell envelopes keep it valid off-grid. Regions calibrated for iid sampling target a super-population law and are conservative for $\theta(F_m)$. With all $n$ tail counts, fixed $n$, continuous $h$ and a grid refined as $m\to\infty$, the identification part of VA's width is first-order worst-case optimal, as for any localization with the same index set. Under a rule fixed before th
    
[^20]: 黎曼流形上积分曲线的抽样设计与置信管

    Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds

    [https://arxiv.org/abs/2611.00771](https://arxiv.org/abs/2611.00771)

    本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。

    

    我们在已知紧黎曼流形上，利用带噪声的切向量场来估计未知向量场的积分曲线。我们考虑两种抽样设置：一种是在已收集的随机位置上的独立观测，另一种是在固定响应预算下，允许利用先导信息来选择后续观测位置。在适当的正则性假设下，被动核估计量的曲线误差以概率达到 $n^{-2/(d+3)}$ 阶，其中 $n$ 为样本量，$d$ 为流形维数。其函数型中心极限定理的极限由一个线性随机微分方程刻画。对于主动抽样，一个独立的先导样本将新的一批测量引导至目标曲线的邻域。在局部化、光滑性和设计条件下，这种集中性使得期望最大曲线误差达到 $N^{-1/2}$ 阶，其中先导样本计入总响应预算 $N$。两种方法均……（摘要原文在此处截断）

    arXiv:2611.00771v2 Announce Type: replace-cross  Abstract: We estimate the integral curve of an unknown vector field from noisy tangent vector field on a known compact Riemannian manifold. We consider two sampling settings: independent observations at random locations that have already been collected, and a fixed response budget that allows subsequent locations to be chosen using pilot information. Under suitable regularity assumptions, the passive kernel estimator has curve error of order $n^{-2/(d+3)}$ in probability, with sample size $n$ and manifold dimension $d$. Its functional central limit theorem has a limit characterized by a linear stochastic differential equation. For active sampling, an independent pilot directs one new batch of measurements to neighborhoods of the target curve. Under localization, smoothness and design conditions, this concentration yields expected maximum curve error of order $N^{-1/2}$, including the pilot in the total response budget $N$. Both methods p
    
[^21]: 通过广义余维数概念形式化目标景观的内在维度

    Formalizing the Intrinsic Dimension of Objective Landscapes by a Generalized Notion of Co-dimension

    [https://arxiv.org/abs/2610.39674](https://arxiv.org/abs/2610.39674)

    本文将目标景观的内在维度形式化为一种依赖于集合全局几何的广义余维数概念，并为浅层和深层ReLU网络学生-教师设置中的最优参数集合推导出内在维度的上下界。

    

    在本工作中，我们形式化了Li等人（2018）提出的目标景观内在维度的概念，该概念通过沿随机生成的子空间进行训练来衡量学习问题的结构复杂度。我们将集合Θ⊂ℝ^p的内在维度定义为最小维度d_int(Θ)，使得ℝ^p中维度为d_int(Θ)的随机平面以概率一与Θ相交。我们证明它是一种依赖于Θ的全局几何的广义余维数概念。我们研究了内在维度的数学性质，并开发了估计学习问题中出现的集合Θ的内在维度的技术。更具体地说，我们推导了当Θ是学生-教师设置中浅层和深层ReLU网络最优参数集合时的上界和/或下界，包括有限和无限……

    arXiv:2610.39674v2 Announce Type: replace-cross  Abstract: In this work, we formalize the concept of intrinsic dimension of objective landscapes introduced in Li et al. (2018), which measures the structural complexity of learning problems by training along randomly generated subspaces. We define the intrinsic dimension of a set $\Theta\subset\mathbb{R}^p$ as the minimal dimension $d_{\mathrm{int}}(\Theta)$ such that a random plane of dimension $d_{\mathrm{int}}(\Theta)$ in $\mathbb{R}^p$ intersects $\Theta$ with probability one. We show that it is a generalized notion of co-dimension that depends on the global geometry of $\Theta$. We study mathematical properties of the intrinsic dimension and develop techniques to estimate the intrinsic dimension of sets $\Theta$ that arise in learning problems. More specifically, we derive upper and/or lower bounds where $\Theta$ is the set of optimal parameters in student-teacher settings with shallow and deep ReLU networks, including finite and in
    
[^22]: 并非所有错误标签都同等重要：基于置信序列的高效任务感知审计

    Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences

    [https://arxiv.org/abs/2610.33262](https://arxiv.org/abs/2610.33262)

    该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。

    

    arXiv:2610.33262v1 公告类型：交叉。摘要：我们研究在固定数据集中对类别标签进行高效、自适应的审计，其动机来自生物多样性信用项目：在这类项目中，专家验证物种记录的成本高昂，且下游评分（如付款金额）对某些标签上错误的敏感度高于其他标签。我们提出的ATLAS框架将任意时刻有效推断与任务感知的自适应审计策略相结合。通过混淆矩阵来表示标签修正，我们根据标签不确定性以及潜在标签修正对下游评分函数的影响，来决定接下来优先检查哪些数据点。对于误标率等线性评分，我们的优先级感知方法在肯尼亚生物多样性数据和广泛使用的CIFAR-10N基准数据集上，分别以比随机抽样少24%和11%的审计量，将整体标签错误率认证到±2个百分点的精度。对于希尔-香农多样性等非线性评分，我们探讨了（原文摘要至此截断）

    arXiv:2610.33262v1 Announce Type: cross  Abstract: We study efficient, adaptive auditing of categorical labels in a fixed dataset, motivated by biodiversity-credit programs in which expert verification of species records is costly and downstream scores (e.g., payment) are more sensitive to errors on certain labels than on others. Our proposed ATLAS framework couples anytime-valid inference with a task-aware adaptive auditing policy. Representing label corrections through a confusion matrix, we prioritize which data points to inspect next based on both label uncertainty and the influence of potential label corrections on the downstream score function. For linear scores such as the mislabel rate, our priority-aware approach certifies the overall label error rate to $\pm 2$ percentage points with 24% and 11% fewer audits than random sampling on biodiversity data from Kenya and a widely used CIFAR-10N benchmark, respectively. For nonlinear scores such as the Hill-Shannon diversity, we expl
    
[^23]: 从带噪声几何图中谱恢复点云

    Spectral Recovery of Point Clouds from Noisy Geometric Graphs

    [https://arxiv.org/abs/2610.08634](https://arxiv.org/abs/2610.08634)

    该论文证明在点数与维度均趋于无穷的高维信号+噪声图模型下，只要满足谱间隙条件，利用邻接矩阵的顶部特征向量和特征值即可在正交变换意义下近似恢复被高斯噪声扰动的低维点云。

    

    我们研究从由带噪声的高维数据生成的随机几何图中恢复低维潜在几何的问题。具体而言，我们分析谱嵌入算法在信号+噪声图模型上的性能，该模型中顶点与被高斯噪声扰动的点相关联，当点对的内积超过指定的对齐阈值时则连接边。在点数 $n$ 和环境维度 $d$ 都趋于无穷的高维情况下，我们证明在谱间隙条件下，图的邻接矩阵的顶部特征向量和特征值可用于在正交变换意义下近似恢复点云。我们在从嵌套球面和高维正弦曲线中采样的点云上展示了我们的结果。

    arXiv:2610.08634v1 Announce Type: cross  Abstract: We study the problem of recovering low-dimensional latent geometry from a random geometric graph generated by noisy, high-dimensional data. Specifically, we analyze the performance of a spectral embedding algorithm on the Signal+Noise Graph Model, in which vertices are associated to points perturbed by Gaussian noise, and edges are included for pairs whose inner product exceeds a specified alignment threshold. In the high-dimensional regime where the number $n$ of points and the ambient dimension $d$ both tend to infinity, we show that under a spectral gap condition, the top eigenvectors and eigenvalues of the graph's adjacency matrix can be used to approximately recover the point cloud up to an orthogonal transformation. We illustrate our results on point clouds sampled from nested spheres and high-dimensional sinusoid curves.
    
[^24]: 基于预测状态的无穷记忆过程生成式序列建模

    Generative sequence modeling for infinite memory processes via predictive states

    [https://arxiv.org/abs/2609.38524](https://arxiv.org/abs/2609.38524)

    本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。

    

    我们考虑估计多元随机过程的一步预测条件分布。许多现有方法依赖于有限范围记忆、稀疏性或可加性等假设，这些假设对于具有长程非线性交互作用的过程可能并不适用。然而，在没有这些结构性假设的情况下，由于维数灾难，非参数估计面临巨大挑战。为了应对这一挑战，我们提出了一种基于过程预测状态的新估计方法，该方法可以处理可能具有无穷范围记忆的过程。我们证明了当过去的历史信息能够被压缩成一个足以预测未来的低维统计量时，我们的估计器能够达到快速收敛速度。具体而言，我们证明了该估计问题的统计复杂度由预测状态空间的内在维数决定。我们为基于该方法的一种具体实现建立了理论保证（摘要在此处被截断）。

    arXiv:2609.38524v1 Announce Type: new  Abstract: We consider estimating the one-step-ahead conditional distribution of a multivariate stochastic process. Many existing approaches rely on assumptions such as finite-range memory, sparsity, or additivity, which can be poorly suited to processes with long-range nonlinear interactions. However, without such structural assumptions, nonparametric estimation is challenging due to the curse of dimensionality. To address this challenge, we introduce a new estimation approach based on the predictive states of a process, possibly with infinite-range memory. We show that our estimator achieves fast convergence rates when the past history can be compressed into a low-dimensional statistic that is sufficient for predicting the future. Specifically, we show that the statistical complexity of the estimation problem is determined by the intrinsic dimension of the predictive state space. We establish guarantees for an instantiation of our method based on
    
[^25]: 基于量子信号处理的表示学习

    Representation Learning with Quantum Signal Processing

    [https://arxiv.org/abs/2608.28828](https://arxiv.org/abs/2608.28828)

    该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。

    

    表示学习始于训练改变了定义数据间相似性的特征之时，而冻结核模型只能对固定几何进行重新加权。我们建立了量子信号处理（QSP）作为表示学习机制的可解量子模型。在任意深度下，我们计算了其量子神经正切核的精确均值与方差，揭示了一种依赖于输入的角几何结构，即使在底层幺正算子趋近Haar随机性时，其对角线仍保持非自平均特性。我们还为完整的非线性梯度流证明了稀疏数据保证，无需冻结或对核进行系综平均：实际动力学收敛于一个具有时间依赖核闭包的可积标量流，并给出显式收敛时间。有限深度速度极限对每个数据集和轨迹均成立。在更高的数据密度下，数值结果表明存在超越标量和冻结核描述的耦合演化。

    arXiv:2608.28828v1 Announce Type: cross  Abstract: Representation learning begins when training changes the features that define similarity between data. A frozen-kernel model only reweights a fixed geometry. We establish quantum signal processing (QSP) as a solvable quantum model of the representation-learning regime. At arbitrary depth, we compute the exact mean and variance of its quantum neural tangent kernel, revealing an input-dependent angular geometry whose diagonal remains non-self-averaging even when the underlying unitary approaches Haar randomness. We also prove a sparse-data guarantee for the full nonlinear gradient flow without freezing or ensemble-averaging the kernel: the realized dynamics converges to an integrable scalar flow with a time-dependent kernel closure and explicit convergence times. A finite-depth speed limit holds for every data set and trajectory. At higher data density, numerical results show coupled evolution beyond both the scalar and frozen-kernel des
    
[^26]: FoundCause：一个从观测数据中发现含潜在混杂因素因果关系的基础模型

    FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data

    [https://arxiv.org/abs/2606.17516](https://arxiv.org/abs/2606.17516)

    FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。

    

    从观测数据中进行因果发现需要在不进行干预的情况下恢复有向关系和潜在混杂因素。我们提出了FoundCause，这是一个摊销式（amortized）因果发现基础模型，仅在合成结构因果模型上训练，能够在单次前向传播中将数据集映射为因果图。一个置换等变的Transformer交替关注样本和变量，以建模变量间的依赖关系和各变量的分布。成对依赖性和不对称性统计信息引导其注意力机制、边解码器和混杂因素模块。解码器将边的存在性与方向分离，而三角细化则融合来自链结构和碰撞结构的证据。可学习的潜在token预测哪些变量对共享隐藏的共同原因，掩码输入表示则用于处理缺失数据。据我们所知，FoundCause是首个针对独立同分布观测数据的摊销式因果发现方法……（摘要原文在此处截断）

    arXiv:2606.17516v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data requires recovering directed relations and latent confounding without interventions. We propose FoundCause, an {\em amortized causal discovery foundation model} trained only on synthetic structural causal models that maps datasets to causal graphs in a single forward pass. A permutation-equivariant transformer alternates attention over samples and variables to model cross-variable dependence and per-variable distributions. Pairwise dependence and asymmetry statistics guide its attention, edge decoder and confounder module. The decoder separates edge existence from direction, while triangular refinement incorporates evidence from chains and colliders. Learnable latent tokens predict which variable pairs share a hidden common cause, and a masked input representation handles missing data. To our knowledge, FoundCause is the first amortized causal discovery approach for i.i.d. observational 
    
[^27]: Graph Cascades：基于传染扩散的介观重连方法，用于结构感知的图机器学习

    Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning

    [https://arxiv.org/abs/2606.05046](https://arxiv.org/abs/2606.05046)

    Graph Cascades提出了一种基于传染扩散的介观图重连策略，在线性时间内将受多跳强化支持的节点对提升为直接邻居，并从理论（标签一致性条件、SBM同配性、有效电阻刻画）和实验上证明其能提升多种GNN与稀疏图Transformer在节点分类任务上的性能。

    

    我们提出了Graph Cascades，这是一种面向图神经网络（GNN）和图Transformer（GT）的介观重连策略，能够捕获超越纯局部边或完全全局注意力的中间尺度图结构。Graph Cascades使用具有固定级联参数的基于传染（contagion）的扩散过程，在O(|V|+|E|)时间内构建一个稀疏辅助图，其中受到重复多跳强化（reinforcement）支持的节点对被提升为直接邻居。我们从理论上刻画了基于强化的重连何时有效：给出了基于强化的边选择比直接邻接更具标签一致性的充分条件；提供了一个随机块模型（SBM）实例，其中两跳强化是完全同配的；并通过图有效电阻对介观连通性进行了形式化。实证结果表明，在多个节点分类基准上，Graph Cascades改进了多种GNN和稀疏图Transformer骨干网络，其中最可靠的……（原文摘要在此处截断）

    arXiv:2606.05046v2 Announce Type: replace  Abstract: We introduce Graph Cascades, a mesoscopic rewiring strategy for Graph Neural Networks (GNNs) and Graph Transformers (GTs) that captures intermediate-scale graph structure beyond purely local edges or fully global attention. Using contagion-based diffusion with fixed cascade parameters, Graph Cascades constructs, in O(|V|+|E|) time, a sparse auxiliary graph where node pairs supported by repeated multi-hop reinforcement are promoted to direct neighbors. We theoretically characterize when reinforcement-based rewiring helps: sufficient conditions under which reinforcement-based edge selection is more label-aligned than direct adjacency, a stochastic block model (SBM) witness in which two-hop reinforcement is perfectly homophilic, and a formalization of mesoscopic connectivity via graph effective resistance. Empirically, across node-classification benchmarks, Graph Cascades improves multiple GNN and sparse-GT backbones, with the most reli
    
[^28]: 基于噪声代理协变量的半监督学习：泛化界与分布回归

    Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression

    [https://arxiv.org/abs/2606.00512](https://arxiv.org/abs/2606.00512)

    该论文提出一种简单的两阶段半监督回归估计器——从大量含噪声的预训练代理协变量中学习核特征值特征、并在少量标签上拟合岭回归预测器，从理论上证明当代理扰动可控且无标签数据充足时可恢复快速标签样本率，并将分布回归作为特例纳入统一的泛化保证框架。

    

    在许多现代机器学习流程中，大量预训练表示被用作含噪声的代理协变量，而特定任务的标签仍然稀缺。我们研究了这一设定下的半监督回归问题，并提出了一种简单的两阶段估计器：从所有代理协变量中学习核特征值特征，然后在有标签数据上拟合岭回归预测器。我们推导了有限样本界，表明当代理扰动得到控制且无标签代理协变量足够充足时，可以恢复快速的有标签样本收敛率。我们还证明分布回归是该框架的直接特例，当有限的袋大小足够大时，同样可以获得类似的保证。实验表明，该方法相比监督和半监督基线取得了一致的性能提升，尤其是在标签稀缺的场景下。

    arXiv:2606.00512v2 Announce Type: replace  Abstract: In many modern machine learning pipelines, abundant pretrained representations serve as noisy proxy covariates, while task-specific labels remain scarce. We study semi-supervised regression in this setting, and propose a simple two stage estimator that learns kernel eigenfeatures from all proxy covariates and fits a ridge predictor on labeled data. We derive finite sample bounds showing that fast labeled sample rates are recovered when proxy perturbation is controlled and unlabeled proxy covariates are sufficiently abundant. We also show that distribution regression is a direct special case, with analogous guarantees when the finite bag size is large enough. Experiments show consistent gains over supervised and semi-supervised baselines, especially in low label regimes.
    
[^29]: 具有1比特通信约束的批式随机线性老虎机

    Batched Stochastic Linear Bandits with 1-Bit Communication Constraints

    [https://arxiv.org/abs/2605.30976](https://arxiv.org/abs/2605.30976)

    该论文首次研究了批处理与1比特通信约束相结合的随机线性老虎机问题，并证明了即使在无噪声环境下，1比特通信瓶颈也会导致 $\Omega(B\min\{d,\log|\mathcal{A}|\})$ 的不可避免的遗憾下界。

    

    我们研究了在批处理与通信约束的一种自然组合下的随机线性老虎机问题：整个时间范围被划分为大小均为 $B$ 的批次，在每个批次中，学习者向代理发送 $B$ 个所请求的动作拉取，代理随后观测相应的 $B$ 个奖励，并向学习者反馈仅一个比特的信息。对于每个批次，学习者指定代理所使用的1比特量化规则，该规则可以依赖于之前接收到的所有比特，但不能直接依赖于任何过去的奖励。这一设置填补了以往仅考虑逐轮量化模型与仅有总比特预算模型之间一个重要但尚未被探索的“中间地带”。我们建立了一个极小极大下界，表明由于1比特通信瓶颈，即使在没有噪声的情况下，$\Omega(B\min\{d,\log\lvert \mathcal{A} \rvert\})$ 的遗憾也是不可避免的。结合标准的统计极限，这得到一个一般的低……

    arXiv:2605.30976v2 Announce Type: replace-cross  Abstract: We study stochastic linear bandits under a natural combination of batching and communication constraints: the time horizon is partitioned into batches of equal size $B$, and during each batch the learner sends $B$ requested arm pulls to an agent, who then observes the corresponding $B$ rewards and responds with a single bit of feedback to the learner. For each batch, the learner specifies the 1-bit quantization rule the agent uses, which may depend on all previously received bits but not on any past rewards directly. This setting addresses a significant yet unexplored ``middle ground'' between previous models having per-round quantization only or total bit budgets only. We establish a minimax lower bound showing that $\Omega(B\min\{d,\log\lvert \mathcal{A} \rvert\})$ regret is unavoidable due to the 1-bit communication bottleneck, even in the absence of noise. Combined with standard statistical limits, this yields a general low
    
[^30]: 贝叶斯优化中非线性参数模型的基于核方法的理论保证

    Kernel-based guarantees for nonlinear parametric models in Bayesian optimization

    [https://arxiv.org/abs/2605.13160](https://arxiv.org/abs/2605.13160)

    本文提出一个基于核方法的框架，为自适应采集数据上训练的正则化非线性参数模型建立置信界，并据此为近似汤普森采样等贝叶斯优化方法提供收敛保证。

    

    现代贝叶斯优化和自适应采样方法越来越依赖于非线性参数模型（例如神经网络），然而此类模型在自适应数据采集下的理论保证仍然有限。现有分析主要集中于高斯过程、核机器、线性模型或线性化的神经近似，导致理论与实践所用的非线性模型之间存在差距。我们开发了一个基于核方法的框架，用于分析在自适应采集数据上训练的正则化参数模型。我们的方法利用参数空间上的核在相应模型类上诱导出再生核希尔伯特空间结构，从而为使用广泛一类正则化损失训练的模型给出置信界。我们展示了这些置信界如何为近似汤普森采样建立收敛保证，并为分析更广泛的无似然贝叶斯优化方法奠定基础。

    arXiv:2605.13160v2 Announce Type: replace-cross  Abstract: Modern Bayesian optimization and adaptive sampling methods increasingly rely on nonlinear parametric models, such as neural networks, yet theoretical guarantees for such models under adaptive data collection remain limited. Existing analyses largely focus on Gaussian processes, kernel machines, linear models, or linearized neural approximations, leaving a gap between theory and the nonlinear models used in practice. We develop a kernel-based framework for analyzing regularized parametric models trained on adaptively collected data. Our approach uses kernels over the parameter space to induce reproducing kernel Hilbert space structures over the corresponding model class, yielding confidence bounds for models trained with a broad class of regularized losses. We show how these bounds can establish convergence guarantees for approximate Thompson sampling and provide a foundation for analyzing broader likelihood-free Bayesian optimi
    
[^31]: 广延宽度网络中的尖锐特征学习转变与贝叶斯最优神经标度律

    Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks

    [https://arxiv.org/abs/2605.10395](https://arxiv.org/abs/2605.10395)

    该论文利用留一解耦论证与封闭不动点方程，在高维广延宽度设定下首次给出贝叶斯最优泛化误差的渐近精确刻画，并揭示教师特征随数据量增长依次通过不连续跳变被学习的尖锐相变序列及其背后的神经标度律。

    

    我们研究了在向更小的学生模型进行知识迁移的背景下，从带噪查询中学习具有层级特征的单隐层教师网络的信息论极限。我们工作在教师宽度 $k$ 与输入维度 $d$ 呈线性标度的高维区间内——这一设定刻画了宽度较大但有限的网络，并且直到最近才在解析上变得可处理。借助一种启发式的留一解耦论证（并在全过程中通过数值验证），我们通过一组封闭的不动点方程，推导出了贝叶斯最优泛化误差和各个特征重叠的渐近精确刻画。这些方程揭示了特征可学习性由一系列尖锐的相变所支配：随着数据量的增长，教师特征被依次恢复，每次恢复都表现为重叠的不连续跳变。这种顺序获取机制构成了精确的……（摘要原文在此处截断）

    arXiv:2605.10395v2 Announce Type: replace-cross  Abstract: We study the information-theoretic limits of learning a one-hidden-layer teacher network with hierarchical features from noisy queries, in the context of knowledge transfer to a smaller student model. We work in the high-dimensional regime where the teacher width $k$ scales linearly with the input dimension $d$ -- a setting that captures large-but-finite-width networks and has only recently become analytically tractable. Using a heuristic leave-one-out decoupling argument, validated numerically throughout, we derive asymptotically sharp characterizations of the Bayes-optimal generalization error and individual feature overlaps via a system of closed fixed-point equations. These equations reveal that feature learnability is governed by a sequence of sharp phase transitions: as data grows, teacher features become recoverable sequentially, each through a discontinuous jump in overlap. This sequential acquisition underlies a precis
    
[^32]: 通过规范扩散测量与分解模态分离

    Measuring and Decomposing Mode Separation via the Canonical Diffusion

    [https://arxiv.org/abs/2605.08777](https://arxiv.org/abs/2605.08777)

    该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。

    

    模态分离，即一个分布被势垒分隔成簇的锐利程度，是密度的基本几何属性，在高维空间中难以量化。模态往往是数据中有意义的选项，例如分子的构象，或生成模型输出中不同的语义类型。模态分离不同于离散度，而后者正是标准无标签工具所捕获的内容：微分熵随散布程度增大而上升，与是否碎裂无关；PCA 按方差对方向排序，与势垒无关。互信息确实能够捕获模态分离，但前提是给定潜在的成分标签，而这通常是不可获得的。我们通过密度所固有的单一随机过程来测量模态分离，即以 $f$ 为平稳分布且扩散系数为常数的唯一可逆扩散过程（规范扩散）。从其自协方差矩阵中，我们提取出两个……

    arXiv:2605.08777v2 Announce Type: replace  Abstract: Mode separation, namely how sharply a distribution fragments into barrier-separated clusters, is a fundamental geometric property of densities, difficult to quantify in high dimensions. Modes are often the meaningful alternatives in the data, such as the conformations of a molecule or the distinct semantic types among a generative model's outputs. Mode separation is distinct from dispersion, which is what standard label-free tools capture: differential entropy rises with spread regardless of fragmentation, and PCA orders directions by variance regardless of barriers. Mutual information does capture it, but only given the latent component labels, which are usually unavailable. We measure mode separation through a single stochastic process intrinsic to the density, namely the unique reversible diffusion with $f$ as its stationary distribution and a constant scalar diffusion coefficient. From its autocovariance matrix we extract two rea
    
[^33]: 关于恢复潜在霍克斯网络的最小观测时间

    On Minimal Observation Time for Recovering Latent Hawkes Networks

    [https://arxiv.org/abs/2605.08400](https://arxiv.org/abs/2605.08400)

    本文针对稀疏弱相互作用的平稳霍克斯过程，证明了精确恢复潜在网络所需的最小观测时间为 $\log d$ 量级（其中 $d$ 为实体数量），该时间界既是充分的也是必要的，并提出了达到该界的两阶段估计方法。

    

    工程、社会和自然中相互作用系统的动力学通常在潜在网络上演化，这些潜在网络决定了哪些实体之间可以发生相互作用。我们研究从基于事件的观测中推断这些网络的问题，这类观测自然地出现在金融、地震学和神经科学等领域。尽管已有大量算法方面的工作来处理这一重要问题，但相关的理论结果仍然稀缺。在本文中，我们提出了以下基本问题：为了精确恢复底层网络，人们需要观察动力学多长时间，即所需时间作为相互作用实体数量 $d$ 的函数是多少？对于一类具有稀疏、弱相互作用的平稳霍克斯过程，我们证明了量级为 $\log d$ 的观测时间既是充分的也是必要的。对于上界部分，我们构造了一个两阶段估计器，该估计器使用裁剪和分箱处理的事件数据进行初步筛选，随后进行最小二乘精化，并应用……

    arXiv:2605.08400v2 Announce Type: replace-cross  Abstract: Dynamics of interacting systems in engineering, society, and nature often evolve over latent networks that govern which entities can interact. We study the problem of inferring these networks from event-based observations, which arise naturally in finance, seismology, and neuroscience. While there is substantial algorithmic work addressing this important problem, theoretical results are scarce. In this paper we ask the following fundamental question: what is the minimum time that one must observe the dynamics in order to exactly recover the underlying network, as a function of the number $d$ of interacting entities? For a class of stationary Hawkes processes with sparse, weak interactions, we prove that an observation time of order $\log d$ is sufficient and necessary. For the upper bound we construct a two-stage estimator that uses clipped and binned event data for screening, followed by a least-squares refinement, and apply c
    
[^34]: 一种“择优挑选”方法用于大规模负荷整形以实现更有效的碳减排

    A Cherry-Picking Approach to Large Load Shaping for More Effective Carbon Reduction

    [https://arxiv.org/abs/2601.17990](https://arxiv.org/abs/2601.17990)

    本文通过构建近8900节点的ERCOT日前市场出清流程校准复现模型进行全年因果调度反事实分析，发现基于节点边际电价的负荷整形在年度电网碳减排中优于其他静态信号，但没有单一信号能在所有运行条件下均表现最佳，因而提出择优选择信号的负荷整形策略以实现更有效的碳减排。

    

    对数据中心等多兆瓦级柔性负荷进行整形会改变电网上的发电机调度，从而影响整个系统的二氧化碳排放和批发电力成本。然而，评估常见的负荷整形信号——例如电网平均碳强度、节点边际电价（LMP）、可再生能源发电量或商业边际排放估算——一直十分困难，因为完整规模的市场出清模型和发电机投标数据在系统运营商之外很少能够获得。本文利用经过校准的、近8900节点规模的ERCOT日前安全约束机组组合（SCUC）与安全约束经济调度（SCED）市场出清流程复现模型，在可再生能源丰富区和化石能源主导区开展了为期一年的因果调度反事实分析。我们发现，虽然基于LMP的负荷整形在年度电网CO2减排方面优于其他静态信号，但没有任何单一信号能在所有运行条件下均表现最佳。

    arXiv:2601.17990v2 Announce Type: replace-cross  Abstract: Shaping multi-megawatt flexible loads such as data centers alters generator dispatch on the electric grid, affecting system-wide CO2 emissions and wholesale electricity costs. However, evaluating prevalent load-shaping signals - such as grid-average carbon intensity, locational marginal price, renewable generation, or commercial marginal emission estimates - has remained difficult because full-scale market-clearing models and generator bid data are rarely accessible outside system operators. Here we conduct year-long causal dispatch counterfactuals using a calibrated, nearly 8,900-bus replica of ERCOT's day-ahead Security Constrained Unit Commitment and Security Constrained Economic Dispatch market-clearing process across renewable-rich and fossil-dominated zones. We find that while LMP-based shaping outperforms other static signals in annual grid CO2 reductions, no single signal performs best across all operating conditions. M
    
[^35]: 论在个体损失准备金评估的神经网络中使用个案估计与交易赔付数据

    On the use of case estimate and transactional payment data in neural networks for individual loss reserving

    [https://arxiv.org/abs/2601.05274](https://arxiv.org/abs/2601.05274)

    该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。

    

    在精算准备金文献中，使用基于个体索赔数据训练的神经网络已日益流行。我们研究了如何在神经网络模型中最优地输入历史赔付数据。此外，个案估计也以时间序列的形式存在，我们进一步扩展了分析以评估其预测能力。在本文中，我们将基于汇总交易数据训练的前馈神经网络与能够分析索赔完整赔付历史和/或个案估计发展历史的循环神经网络进行了比较。我们通过在由SPLICE（Avanzi、Taylor和Wang，2023）模拟出的多个可比较的高度复杂数据集上训练并比较各模型的性能，从而得出结论。在协变量数据有限的环境中，我们发现有证据表明个案估计能够显著改善预测效果，但为神经网络配备记忆机制只会带来……（原文摘要至此处被截断）

    arXiv:2601.05274v2 Announce Type: replace-cross  Abstract: The use of neural networks trained on individual claims data has become increasingly popular in the actuarial reserving literature. We consider how to best input historical payment data in neural network models. Additionally, case estimates are also available in the format of a time series, and we extend our analysis to assessing their predictive power. In this paper, we compare a feed-forward neural network trained on summarised transactions to a recurrent neural network equipped to analyse a claim's entire payment history and/or case estimate development history. We draw conclusions from training and comparing the performance of the models on multiple, comparable highly complex datasets simulated from SPLICE (Avanzi, Taylor and Wang, 2023). In environments where covariate data is limited, we find evidence that case estimates will improve predictions substantially, but that equipping the neural network with memory only leads t
    
[^36]: 功能压缩作为奥卡姆剃刀：由敏感度最小化导出的代数复杂度

    Functional Compression as Occam's Razor: Algebraic Complexity from Sensitivity Minimization

    [https://arxiv.org/abs/2511.23152](https://arxiv.org/abs/2511.23152)

    该论文为代数补全分解模型建立了以参数敏感度下确界度量的普适复杂度下界，证明达到该下界会迫使潜在表示几何对齐且幺正，并使模型通过共享代数组合（功能压缩）而非低秩表示来学习到全局统一的代数结构。

    

    从数据中发现离散代数结构是机器学习中的一个根本性挑战。我们探讨连续的学习目标如何能够偏好功能压缩——即通过共享的代数组合而非低秩表示来表征数据。我们分析了用于代数补全（恢复有限二元运算）的规范分解模型的全局最优解，该模型已被证明对群运算具有偏好。我们以精确重构条件下参数敏感度的下确界来度量代数复杂度。我们建立了一个普适的复杂度下界（地板），达到该下界会迫使潜在表示具有几何对齐性与幺正性。在该下界处，输入表示组合成相应的输出表示；这些表示在整个运算表中的复用将局部恒等式传播开来，从而强制形成全局共享的代数结构。若下界未被达到，则存在一个正的间隙将……（摘要在此处截断）

    arXiv:2511.23152v5 Announce Type: replace  Abstract: Discovering discrete algebraic structures from data is a fundamental challenge in machine learning. We ask how continuous learning objectives can favor functional compression---representing data through shared algebraic composition rather than low-rank representations. We analyze the global optima of a canonical factorization model for algebraic completion (recovering finite binary operations) that demonstrated a preference for group operations. We measure algebraic complexity by the infimum of parameter sensitivity under exact reconstruction. We establish a universal complexity floor, whose attainment forces geometric alignment and unitarity of the latent representations. At the floor, input representations compose into the corresponding output representations; their reuse across the operation table propagates these local identities to enforce a globally shared algebraic structure. If the floor is not attained, a positive gap separa
    
[^37]: 基于均值测度量化的可扩展概率测度学习

    Scalable Learning from Probability Measures with Mean Measure Quantization

    [https://arxiv.org/abs/2502.04907](https://arxiv.org/abs/2502.04907)

    该论文提出一种均值测度量化方法，将所有输入概率测度近似为共享公共支撑集的K点离散测度，在保证量化测度一致性和下游任务收敛性的同时，大幅降低最优传输计算的时间成本并保持与单独量化相当的性能。

    

    我们考虑以一组概率测度的形式观测数据的统计学习问题。最优传输（OT）是比较和操作此类对象的常用工具，但当测度具有较大支撑集时，其计算成本变得难以承受。我们研究了一种基于量化的方法，将所有输入测度近似为共享公共支撑集的K点离散测度。我们证明了所得量化测度的一致性，并进一步为基于这些量化测度计算的多个OT相关下游任务推导出收敛性保证。在合成数据集和真实数据集上的数值实验表明，所提出的方法在大幅减少运行时间的同时，实现了与单独量化相当的性能。

    arXiv:2502.04907v5 Announce Type: replace-cross  Abstract: We consider statistical learning problems in which data are observed as a set of probability measures. Optimal transport (OT) is a popular tool to compare and manipulate such objects, but its computational cost becomes prohibitive when the measures have large support. We study a quantization-based approach in which all input measures are approximated by $K$-point discrete measures sharing a common support. We establish consistency of the resulting quantized measures. We further derive convergence guarantees for several OT-based downstream tasks computed from the quantized measures. Numerical experiments on synthetic and real datasets demonstrate that the proposed approach achieves performance comparable to individual quantization while substantially reducing runtime.
    
[^38]: 基于Transformer的精确序列插值

    Exact Sequence Interpolation with Transformers

    [https://arxiv.org/abs/2502.02270](https://arxiv.org/abs/2502.02270)

    该论文通过交替使用前馈层与自注意力层并采用低秩参数矩阵，证明了Transformer能够以与输入序列长度无关的复杂度精确插值有限序列数据集。

    

    我们证明了Transformer能够精确插值由 $\mathbb{R}^d$（$d\geq 2$）中有限输入序列及其对应的不超过输入长度的输出序列组成的数据集。具体而言，给定 $N$ 个在 $\mathbb{R}^d$ 中长度任意但有限的序列，以及长度为 $m^1, \dots, m^N \in \mathbb{N}$ 的输出序列，我们构造了一个具有 $\mathcal{O}(\sum_{j=1}^N m^j)$ 个块和 $\mathcal{O}(d \sum_{j=1}^N m^j)$ 个参数的Transformer，可以对该数据集实现精确插值。我们的构造通过交替使用前馈层和自注意力层，并利用自注意力层固有的聚类效应，提供了与输入序列长度无关的复杂度估计。我们新颖的构造方法还在自注意力机制中使用了低秩参数矩阵，这也是实际Transformer实现中的一个常见特征。这些结果首先在hardmax自注意力设定下建立。

    arXiv:2502.02270v4 Announce Type: replace  Abstract: We prove that transformers can exactly interpolate datasets of finite input sequences in $\mathbb{R}^d$, $d\geq 2$, with corresponding output sequences of smaller or equal length. Specifically, given $N$ sequences of arbitrary but finite lengths in $\mathbb{R}^d$ and output sequences of lengths $m^1, \dots, m^N \in \mathbb{N}$, we construct a transformer with $\mathcal{O}(\sum_{j=1}^N m^j)$ blocks and $\smash{\mathcal{O}(d \sum_{j=1}^N m^j)}$ parameters that exactly interpolates the dataset. Our construction provides complexity estimates that are independent of the input sequence length, by alternating feed-forward and self-attention layers and by capitalizing on the clustering effect inherent to the latter. Our novel constructive method also uses low-rank parameter matrices in the self-attention mechanism, a common feature of practical transformer implementations. These results are first established in the hardmax self-attention set
    
[^39]: 纯注意力硬最大值Transformer中的聚类及其在情感分析中的作用

    Clustering in pure-attention hardmax transformers and its role in sentiment analysis

    [https://arxiv.org/abs/2407.01602](https://arxiv.org/abs/2407.01602)

    本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。

    

    Transformer是极其成功的机器学习模型，但其数学性质仍未被充分理解。本文严格刻画了具有硬最大值自注意力和归一化子层的Transformer在层数趋于无穷时的行为。通过将此类Transformer视为描述欧几里得空间中点演化的离散时间动力系统，并借助基于超平面分离的自注意力机制的几何解释，我们证明了Transformer的输入渐近收敛到由称为“领导者”的特殊点所决定的聚类平衡态。随后，我们利用这一理论认识，采用完全可解释的Transformer模型来解决语言处理中的情感分析问题，该模型通过将无意义词汇聚类到承载最多语义的领导者词汇周围，有效捕捉了“上下文”。最后，我们概述了仍然存在的（问题）……

    arXiv:2407.01602v3 Announce Type: replace  Abstract: Transformers are extremely successful machine learning models whose mathematical properties remain poorly understood. Here, we rigorously characterize the behavior of transformers with hardmax self-attention and normalization sublayers as the number of layers tends to infinity. By viewing such transformers as discrete-time dynamical systems describing the evolution of points in a Euclidean space, and thanks to a geometric interpretation of the self-attention mechanism based on hyperplane separation, we show that the transformer inputs asymptotically converge to a clustered equilibrium determined by special points called \textit{leaders}. We then leverage this theoretical understanding to solve sentiment analysis problems from language processing using a fully interpretable transformer model, which effectively captures `context' by clustering meaningless words around leader words carrying the most meaning. Finally, we outline remainin
    
[^40]: 高维尾指数回归：以社交媒体病毒帖子文本分析为例

    High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media

    [https://arxiv.org/abs/2403.01318](https://arxiv.org/abs/2403.01318)

    提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。

    

    受社交媒体病毒帖子的点赞分布（如点赞数量）经验性幂律的启发，我们引入了高维尾指数回归及其参数的估计和推断方法。我们提出了一种正则化估计量，证明了它的一致性，并推导了其收敛速度。为了进行推断，我们提出了去偏正则化估计，证明了去偏估计量的渐近正态性。仿真研究支持了我们的理论。这些方法被应用于对涉及 LGBTQ+ 话题的 X（原 Twitter）病毒帖子的文本分析。

    arXiv:2403.01318v1 Announce Type: cross  Abstract: Motivated by the empirical power law of the distributions of credits (e.g., the number of "likes") of viral posts in social media, we introduce the high-dimensional tail index regression and methods of estimation and inference for its parameters. We propose a regularized estimator, establish its consistency, and derive its convergence rate. To conduct inference, we propose to debias the regularized estimate, and establish the asymptotic normality of the debiased estimator. Simulation studies support our theory. These methods are applied to text analyses of viral posts in X (formerly Twitter) concerning LGBTQ+.
    
[^41]: 算法保险

    Algorithmic Insurance

    [https://arxiv.org/abs/2106.00839](https://arxiv.org/abs/2106.00839)

    该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。

    

    当人工智能（AI）系统在医学诊断或自动驾驶等高风险领域出现错误时，一个在不同情境中相同的单一算法缺陷可能产生异质性的损失，这挑战了传统保险的假设。算法保险，即为AI造成的损害提供财务保障的新兴市场正在兴起，然而保险公司在为这类风险定价时面临困难，AI开发者也缺乏将系统设计与责任风险敞口相关联的框架。我们将二元分类中的一个关键操作选择——从业者如何设置决策阈值——与尾部风险敞口联系起来。利用条件风险价值（CVaR）来刻画极端损失，我们证明了准确率最大化等既有方法相对于尾部风险优化会显著增加最坏情况下的损失，且当阈值偏离最优值时，惩罚至少以二次方速度增长。我们提出了一种责任保险合同，要求……（原文摘要在此处截断）

    arXiv:2106.00839v4 Announce Type: replace  Abstract: When artificial intelligence (AI) systems make errors in high-stakes domains such as medical diagnosis or autonomous driving, a single algorithmic flaw across varying contexts can generate heterogeneous losses that challenge traditional insurance assumptions. Algorithmic insurance, financial coverage for AI-induced damages, is an emerging market, yet insurers struggle to price these risks and AI developers lack frameworks linking system design to liability exposure. We connect a key operational choice in binary classification, how practitioners set decision thresholds, to tail risk exposure. Using conditional value-at-risk (CVaR) to capture extreme losses, we prove that established approaches such as accuracy maximization can significantly increase worst-case losses relative to tail risk optimization, with penalties growing at least quadratically as thresholds deviate from optimal. We propose a liability insurance contract that manda
    
[^42]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    

