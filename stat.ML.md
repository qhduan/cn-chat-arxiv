# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal online model aggregation](https://arxiv.org/abs/2403.15527) | 该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。 |
| [^2] | [Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization](https://arxiv.org/abs/2403.04764) | 该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色. |
| [^3] | [Diffusive Gibbs Sampling](https://arxiv.org/abs/2402.03008) | 扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。 |
| [^4] | [Learning to Embed Time Series Patches Independently](https://arxiv.org/abs/2312.16427) | 学习独立嵌入时间序列片段可以产生更好的时间序列表示，通过简单的块重构任务和独立嵌入每个块的MLP模型以及互补对比学习来实现。 |
| [^5] | [Differentially Private Bayesian Tests.](http://arxiv.org/abs/2401.15502) | 本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。 |
| [^6] | [Scalable network reconstruction in subquadratic time.](http://arxiv.org/abs/2401.01404) | 这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。 |
| [^7] | [Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions.](http://arxiv.org/abs/2310.05921) | 符合决策理论是一种框架，可以通过不完美的机器学习预测产生安全的自主决策。该理论的创新之处在于可以在没有对世界模型做出任何假设的情况下提供具有低风险的统计保证的决策。 |
| [^8] | [Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances.](http://arxiv.org/abs/2310.02246) | 本文提出了一种解决一系列线性系统实例中设置求解器参数的方法，通过使用在线学习算法选择参数，可以接近最佳总迭代次数的性能，而无需进行额外的矩阵计算。 |
| [^9] | [Delegating Data Collection in Decentralized Machine Learning.](http://arxiv.org/abs/2309.01837) | 这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。 |
| [^10] | [A Probabilistic Framework for Modular Continual Learning.](http://arxiv.org/abs/2306.06545) | 本文提出了一种名为PICLE的模块化增量学习框架，利用概率模型快速计算每个组合的适应度来加速搜索，是第一个可以实现不同类型的转移的模块化增量学习算法。 |
| [^11] | [Sharp high-probability sample complexities for policy evaluation with linear function approximation.](http://arxiv.org/abs/2305.19001) | 本文研究线性函数逼近下的策略评估问题，提出了两个广泛使用的算法所需的样本复杂度，具有高概率收敛保证且与容差水平的关联性最佳。 |
| [^12] | [Learning linear dynamical systems under convex constraints.](http://arxiv.org/abs/2303.15121) | 本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。 |
| [^13] | [Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure.](http://arxiv.org/abs/2303.11786) | 这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。 |
| [^14] | [Spectral Regularized Kernel Two-Sample Tests.](http://arxiv.org/abs/2212.09201) | 本文研究了基于概率分布的再生核希尔伯特空间嵌入的双样本检验的最优性。我们发现最大均值差异（MMD）检验在分离边界方面并不是最优的，因此我们提出了一种基于谱正则化的修改方法，使得检验具有更小的分离边界。同时，我们还提出了自适应版本的检验，通过数据驱动的策略选择正则化参数，展示了其近乎最优的性能。 |
| [^15] | [Quadratic models for understanding neural network dynamics.](http://arxiv.org/abs/2205.11787) | 神经二次模型可以展示出神经网络在大学习率情况下的“弹弓阶段”，并且在泛化特性上与神经网络有相似之处，是分析神经网络的有效工具。 |

# 详细

[^1]: 依从在线模型聚合

    Conformal online model aggregation

    [https://arxiv.org/abs/2403.15527](https://arxiv.org/abs/2403.15527)

    该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。

    

    依从预测为机器学习模型提供了一种合理的不确定性量化概念，而不需要做出强烈的分布假设。它适用于任何黑盒预测模型，并将点预测转换成具有预定义边际覆盖保证的集预测。然而，依从预测只在事先确定底层机器学习模型的情况下起作用。依从预测中相对较少涉及的问题是模型选择和/或聚合：对于给定的问题，应该如何依从化众多预测方法（随机森林、神经网络、正则化线性模型等）？本文提出了一种新的依从模型聚合方法，用于在线设置，该方法基于将来自多个算法的预测集进行投票，其中根据过去表现调整模型上的权重。

    arXiv:2403.15527v1 Announce Type: cross  Abstract: Conformal prediction equips machine learning models with a reasonable notion of uncertainty quantification without making strong distributional assumptions. It wraps around any black-box prediction model and converts point predictions into set predictions that have a predefined marginal coverage guarantee. However, conformal prediction only works if we fix the underlying machine learning model in advance. A relatively unaddressed issue in conformal prediction is that of model selection and/or aggregation: for a given problem, which of the plethora of prediction methods (random forests, neural nets, regularized linear models, etc.) should we conformalize? This paper proposes a new approach towards conformal model aggregation in online settings that is based on combining the prediction sets from several algorithms by voting, where weights on the models are adapted over time based on past performance.
    
[^2]: 将Thompson抽样遗憾与Sigma比率（TS-RSR）最小化：一种用于批量贝叶斯优化的经过证明的高效算法

    Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization

    [https://arxiv.org/abs/2403.04764](https://arxiv.org/abs/2403.04764)

    该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色.

    

    本文提出了一个新的方法，用于批量贝叶斯优化（BO），其中抽样通过最小化Thompson抽样方法的遗憾与不确定性比率来进行。我们的目标是能够协调每个批次中选择的动作，以最小化点之间的冗余，同时关注具有高预测均值或高不确定性的点。我们对算法的遗憾提供了高概率的理论保证。最后，从数字上看，我们证明了我们的方法在一系列非凸测试函数上达到了最先进的性能，在平均值上比几个竞争对手的基准批量BO算法表现提高了一个数量级。

    arXiv:2403.04764v1 Announce Type: new  Abstract: This paper presents a new approach for batch Bayesian Optimization (BO), where the sampling takes place by minimizing a Thompson Sampling approximation of a regret to uncertainty ratio. Our objective is able to coordinate the actions chosen in each batch in a way that minimizes redundancy between points whilst focusing on points with high predictive means or high uncertainty. We provide high-probability theoretical guarantees on the regret of our algorithm. Finally, numerically, we demonstrate that our method attains state-of-the-art performance on a range of nonconvex test functions, where it outperforms several competitive benchmark batch BO algorithms by an order of magnitude on average.
    
[^3]: 扩散吉布斯采样

    Diffusive Gibbs Sampling

    [https://arxiv.org/abs/2402.03008](https://arxiv.org/abs/2402.03008)

    扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。

    

    传统马尔可夫链蒙特卡洛（MCMC）方法在多模态分布的混合不足方面存在着挑战，特别是在贝叶斯推断和分子动力学等实际应用中。针对这个问题，我们提出了一种创新的采样方法——扩散吉布斯采样（DiGS），用于有效采样具有远程和断开模态特征的分布。DiGS集成了扩散模型的最新发展，利用高斯卷积创建一个辅助噪声分布，以在原始空间中连接孤立的模态，并应用吉布斯采样从两个空间中交替抽取样本。我们的方法在采样多模态分布方面表现出比并行温度法等最先进方法更好的混合性能。我们证明我们的采样器在各种任务中取得了显著改进的结果，包括高斯混合模型、贝叶斯神经网络和分子动力学。

    The inadequate mixing of conventional Markov Chain Monte Carlo (MCMC) methods for multi-modal distributions presents a significant challenge in practical applications such as Bayesian inference and molecular dynamics. Addressing this, we propose Diffusive Gibbs Sampling (DiGS), an innovative family of sampling methods designed for effective sampling from distributions characterized by distant and disconnected modes. DiGS integrates recent developments in diffusion models, leveraging Gaussian convolution to create an auxiliary noisy distribution that bridges isolated modes in the original space and applying Gibbs sampling to alternately draw samples from both spaces. Our approach exhibits a better mixing property for sampling multi-modal distributions than state-of-the-art methods such as parallel tempering. We demonstrate that our sampler attains substantially improved results across various tasks, including mixtures of Gaussians, Bayesian neural networks and molecular dynamics.
    
[^4]: 独立学习将时间序列片段嵌入

    Learning to Embed Time Series Patches Independently

    [https://arxiv.org/abs/2312.16427](https://arxiv.org/abs/2312.16427)

    学习独立嵌入时间序列片段可以产生更好的时间序列表示，通过简单的块重构任务和独立嵌入每个块的MLP模型以及互补对比学习来实现。

    

    最近，掩码时间序列建模作为一种自监督表示学习策略引起了广泛关注。受计算机视觉中的掩码图像建模启发，最近的研究首先将时间序列进行分块处理并部分掩盖，然后训练Transformer模型通过从未掩盖的块预测被掩盖块来捕捉块之间的依赖关系。然而，我们认为捕捉这种块之间的依赖关系可能不是时间序列表示学习的最佳策略；相反，独立学习嵌入片段会产生更好的时间序列表示。具体而言，我们建议使用1）简单的块重构任务，自动将每个块进行编码而不查看其他块，以及2）独自嵌入每个块的简单块式MLP。此外，我们引入互补对比学习来有效地分层捕获相邻时间序列信息。

    arXiv:2312.16427v2 Announce Type: replace-cross  Abstract: Masked time series modeling has recently gained much attention as a self-supervised representation learning strategy for time series. Inspired by masked image modeling in computer vision, recent works first patchify and partially mask out time series, and then train Transformers to capture the dependencies between patches by predicting masked patches from unmasked patches. However, we argue that capturing such patch dependencies might not be an optimal strategy for time series representation learning; rather, learning to embed patches independently results in better time series representations. Specifically, we propose to use 1) the simple patch reconstruction task, which autoencode each patch without looking at other patches, and 2) the simple patch-wise MLP that embeds each patch independently. In addition, we introduce complementary contrastive learning to hierarchically capture adjacent time series information efficiently. 
    
[^5]: 差分隐私贝叶斯检验

    Differentially Private Bayesian Tests. (arXiv:2401.15502v1 [stat.ML])

    [http://arxiv.org/abs/2401.15502](http://arxiv.org/abs/2401.15502)

    本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。

    

    在利用机密数据进行科学假设检验的领域中，差分隐私已经成为一个重要的基石。在报告科学发现时，广泛采用贝叶斯检验，因为它们有效地避免了P值的主要批评，即缺乏可解释性和无法量化对竞争假设的支持证据。我们提出了一个新颖的差分隐私贝叶斯假设检验框架，该框架在基于规范化的数据生成机制基础上自然产生，从而保持了推断结果的可解释性。此外，通过专注于基于广泛使用的检验统计量的差分隐私贝叶斯因子，我们避免了对完整数据生成机制建模的需求，并确保了实质性的计算优势。我们还提供了一组充分条件，以在所提框架下确立贝叶斯因子一致性的结果。

    Differential privacy has emerged as an significant cornerstone in the realm of scientific hypothesis testing utilizing confidential data. In reporting scientific discoveries, Bayesian tests are widely adopted since they effectively circumnavigate the key criticisms of P-values, namely, lack of interpretability and inability to quantify evidence in support of the competing hypotheses. We present a novel differentially private Bayesian hypotheses testing framework that arise naturally under a principled data generative mechanism, inherently maintaining the interpretability of the resulting inferences. Furthermore, by focusing on differentially private Bayes factors based on widely used test statistics, we circumvent the need to model the complete data generative mechanism and ensure substantial computational benefits. We also provide a set of sufficient conditions to establish results on Bayes factor consistency under the proposed framework. The utility of the devised technology is showc
    
[^6]: 可扩展的子二次时间网络重建

    Scalable network reconstruction in subquadratic time. (arXiv:2401.01404v1 [cs.DS])

    [http://arxiv.org/abs/2401.01404](http://arxiv.org/abs/2401.01404)

    这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。

    

    网络重建是指在只有关于条件偶联的观测数据，例如时间序列或图模型的独立样本的情况下，确定N个节点之间未观测到的成对耦合。针对这个问题提出的算法的可扩展性的主要障碍是似乎无法避免的二次复杂度O(N^2)，即要考虑每种可能的成对耦合至少一次，尽管大多数感兴趣的网络都是稀疏的，非零耦合的数量只有O(N)。在这里，我们提出了一个适用于广泛重建问题的通用算法，其在子二次时间内实现结果，其数据相关复杂度宽松上界为O(N^(3/2)logN)，但具有更典型的对数线性复杂度O(Nlog^2 N)。我们的算法依赖于一个随机的二阶邻居搜索，产生了最佳的边候选。

    Network reconstruction consists in determining the unobserved pairwise couplings between $N$ nodes given only observational data on the resulting behavior that is conditioned on those couplings -- typically a time-series or independent samples from a graphical model. A major obstacle to the scalability of algorithms proposed for this problem is a seemingly unavoidable quadratic complexity of $O(N^2)$, corresponding to the requirement of each possible pairwise coupling being contemplated at least once, despite the fact that most networks of interest are sparse, with a number of non-zero couplings that is only $O(N)$. Here we present a general algorithm applicable to a broad range of reconstruction problems that achieves its result in subquadratic time, with a data-dependent complexity loosely upper bounded by $O(N^{3/2}\log N)$, but with a more typical log-linear complexity of $O(N\log^2N)$. Our algorithm relies on a stochastic second neighbor search that produces the best edge candidat
    
[^7]: 符合决策理论: 通过不完美的预测产生安全的自主决策

    Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions. (arXiv:2310.05921v1 [stat.ML])

    [http://arxiv.org/abs/2310.05921](http://arxiv.org/abs/2310.05921)

    符合决策理论是一种框架，可以通过不完美的机器学习预测产生安全的自主决策。该理论的创新之处在于可以在没有对世界模型做出任何假设的情况下提供具有低风险的统计保证的决策。

    

    我们介绍了一种符合决策理论的框架，可以在机器学习预测不完美的情况下产生安全的自主决策。这种决策的例子是普遍存在的，从依赖于行人预测的机器人规划算法，到校准自动化制造以实现高吞吐量和低错误率，再到在运行时选择信任名义策略还是切换到安全备份策略。我们算法产生的决策在统计保证的情况下是安全的，无需对世界模型作出任何假设；观测数据可以不满足独立同分布(I.I.D.)的条件，甚至可能是对抗性的。该理论将符合预测的结果扩展到直接校准决策，而不需要构建预测集合。实验证明了我们方法在围绕人类进行机器人运动规划、自动股票交易和机器人制造方面的实用性。

    We introduce Conformal Decision Theory, a framework for producing safe autonomous decisions despite imperfect machine learning predictions. Examples of such decisions are ubiquitous, from robot planning algorithms that rely on pedestrian predictions, to calibrating autonomous manufacturing to exhibit high throughput and low error, to the choice of trusting a nominal policy versus switching to a safe backup policy at run-time. The decisions produced by our algorithms are safe in the sense that they come with provable statistical guarantees of having low risk without any assumptions on the world model whatsoever; the observations need not be I.I.D. and can even be adversarial. The theory extends results from conformal prediction to calibrate decisions directly, without requiring the construction of prediction sets. Experiments demonstrate the utility of our approach in robot motion planning around humans, automated stock trading, and robot manufacturin
    
[^8]: 学习放松：在一系列线性系统实例中设置求解器参数

    Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances. (arXiv:2310.02246v1 [cs.LG])

    [http://arxiv.org/abs/2310.02246](http://arxiv.org/abs/2310.02246)

    本文提出了一种解决一系列线性系统实例中设置求解器参数的方法，通过使用在线学习算法选择参数，可以接近最佳总迭代次数的性能，而无需进行额外的矩阵计算。

    

    解决线性系统$Ax=b$是一种基本的科学计算原理，已经开发了许多求解器和预处理器。它们带有参数，其最佳值取决于要解决的系统，并且通常无法或成本过高以确定；因此在实践中使用次优启发式。我们考虑在需要解决许多相关线性系统的常见情况下，例如在单个数值模拟期间。在这种情况下，我们是否可以顺序选择参数，以获得接近最佳总迭代次数的性能，而无需进行额外的矩阵计算？对于过度轻松（SOR）这种标准求解器，我们回答肯定的。这种方法能够使用仅迭代次数作为反馈的赌徒在线学习算法，选择序列实例的参数，使得总成本接近最佳固定的ω值。

    Solving a linear system $Ax=b$ is a fundamental scientific computing primitive for which numerous solvers and preconditioners have been developed. These come with parameters whose optimal values depend on the system being solved and are often impossible or too expensive to identify; thus in practice sub-optimal heuristics are used. We consider the common setting in which many related linear systems need to be solved, e.g. during a single numerical simulation. In this scenario, can we sequentially choose parameters that attain a near-optimal overall number of iterations, without extra matrix computations? We answer in the affirmative for Successive Over-Relaxation (SOR), a standard solver whose parameter $\omega$ has a strong impact on its runtime. For this method, we prove that a bandit online learning algorithm -- using only the number of iterations as feedback -- can select parameters for a sequence of instances such that the overall cost approaches that of the best fixed $\omega$ as
    
[^9]: 委托分散机器学习中的数据收集

    Delegating Data Collection in Decentralized Machine Learning. (arXiv:2309.01837v1 [cs.LG])

    [http://arxiv.org/abs/2309.01837](http://arxiv.org/abs/2309.01837)

    这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。

    

    受分散机器学习生态系统的出现的启发，我们研究了数据收集的委托问题。以契约理论为出发点，我们设计了解决两个基本机器学习挑战的最优和近似最优契约：模型质量评估的不确定性和对任何模型最优性能的缺乏知识。我们证明，通过简单的线性契约可以解决不确定性问题，即使委托人只有一个小的测试集，也能实现1-1/e的一等效用水平。此外，我们给出了委托人测试集大小的充分条件，可以达到对最优效用的逼近。为了解决对最优性能缺乏预先知识的问题，我们提出了一个凸问题，可以自适应和高效地计算最优契约。

    Motivated by the emergence of decentralized machine learning ecosystems, we study the delegation of data collection. Taking the field of contract theory as our starting point, we design optimal and near-optimal contracts that deal with two fundamental machine learning challenges: lack of certainty in the assessment of model quality and lack of knowledge regarding the optimal performance of any model. We show that lack of certainty can be dealt with via simple linear contracts that achieve 1-1/e fraction of the first-best utility, even if the principal has a small test set. Furthermore, we give sufficient conditions on the size of the principal's test set that achieves a vanishing additive approximation to the optimal utility. To address the lack of a priori knowledge regarding the optimal performance, we give a convex program that can adaptively and efficiently compute the optimal contract.
    
[^10]: 一种基于概率框架的模块化增量学习方法

    A Probabilistic Framework for Modular Continual Learning. (arXiv:2306.06545v1 [cs.LG])

    [http://arxiv.org/abs/2306.06545](http://arxiv.org/abs/2306.06545)

    本文提出了一种名为PICLE的模块化增量学习框架，利用概率模型快速计算每个组合的适应度来加速搜索，是第一个可以实现不同类型的转移的模块化增量学习算法。

    

    模块化方法是增量学习领域的有前途方向，每个问题使用不同的模块组合且避免遗忘。然而，搜索可能的模块组合是一个挑战，因为评估组合性能需要一轮神经网络训练。为了解决这个问题，我们发展了一种名为PICLE的模块化增量学习框架，通过使用概率模型来快速计算每个组合的适应度来加速搜索。模型结合先前关于良好模块组合的知识与数据集特定信息。它的使用被分为感知和潜在子集等子集的搜索空间加以补充。我们展示了PICLE是第一个可以实现不同类型的转移的模块化增量学习算法，同时还能扩展到大型搜索空间。我们在两个基准套件上对其进行评估，这些套件旨在捕捉增量学习技术的不同要求。

    Modular approaches, which use a different composition of modules for each problem and avoid forgetting by design, have been shown to be a promising direction in continual learning (CL). However, searching through the large, discrete space of possible module compositions is a challenge because evaluating a composition's performance requires a round of neural network training. To address this challenge, we develop a modular CL framework, called PICLE, that accelerates search by using a probabilistic model to cheaply compute the fitness of each composition. The model combines prior knowledge about good module compositions with dataset-specific information. Its use is complemented by splitting up the search space into subsets, such as perceptual and latent subsets. We show that PICLE is the first modular CL algorithm to achieve different types of transfer while scaling to large search spaces. We evaluate it on two benchmark suites designed to capture different desiderata of CL techniques. 
    
[^11]: 线性函数逼近下的策略评估的高概率样本复杂度

    Sharp high-probability sample complexities for policy evaluation with linear function approximation. (arXiv:2305.19001v1 [stat.ML])

    [http://arxiv.org/abs/2305.19001](http://arxiv.org/abs/2305.19001)

    本文研究线性函数逼近下的策略评估问题，提出了两个广泛使用的算法所需的样本复杂度，具有高概率收敛保证且与容差水平的关联性最佳。

    

    本文涉及使用线性函数逼近在无限时间马尔可夫决策过程中进行策略评估的问题。我们研究了两种广泛使用的策略评估算法（时间差分学习算法和带有梯度校正的两个时间尺度线性时间差分算法）所需的样本复杂度，以保证最佳线性系数的预定义估计误差。在策略设置和离线设置中，我们建立了第一个具有高概率收敛保证的样本复杂度界限，达到了与容差水平的最佳关联性。我们还展示了与问题相关量明确的关系，并在策略设置中展示了我们的上限界限与关键问题参数上的极小极大下限界限相匹配。

    This paper is concerned with the problem of policy evaluation with linear function approximation in discounted infinite horizon Markov decision processes. We investigate the sample complexities required to guarantee a predefined estimation error of the best linear coefficients for two widely-used policy evaluation algorithms: the temporal difference (TD) learning algorithm and the two-timescale linear TD with gradient correction (TDC) algorithm. In both the on-policy setting, where observations are generated from the target policy, and the off-policy setting, where samples are drawn from a behavior policy potentially different from the target policy, we establish the first sample complexity bound with high-probability convergence guarantee that attains the optimal dependence on the tolerance level. We also exhihit an explicit dependence on problem-related quantities, and show in the on-policy setting that our upper bound matches the minimax lower bound on crucial problem parameters, in
    
[^12]: 在凸约束下学习线性动态系统

    Learning linear dynamical systems under convex constraints. (arXiv:2303.15121v1 [math.ST])

    [http://arxiv.org/abs/2303.15121](http://arxiv.org/abs/2303.15121)

    本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。

    

    我们考虑从单个轨迹中识别线性动态系统的问题。最近的研究主要关注未对系统矩阵 $A^* \in \mathbb{R}^{n \times n}$ 进行结构假设的情况，并对普通最小二乘 (OLS) 估计器进行了详细分析。我们假设可用先前的 $A^*$ 的结构信息，可以在包含 $A^*$ 的凸集 $\mathcal{K}$ 中捕获。对于随后的受约束最小二乘估计的解，我们推导出 Frobenius 范数下依赖于 $\mathcal{K}$ 在 $A^*$ 处切锥的局部大小的非渐进误差界。为了说明这一结果的有用性，我们将其实例化为以下设置：(i) $\mathcal{K}$ 是 $\mathbb{R}^{n \times n}$ 中的 $d$ 维子空间，或者 (ii) $A^*$ 是 $k$ 稀疏的，$\mathcal{K}$ 是适当缩放的 $\ell_1$ 球。在 $d, k \ll n^2$ 的区域中，我们的误差界对于相同的统计和噪声假设比 OLS 估计器获得了改进。

    We consider the problem of identification of linear dynamical systems from a single trajectory. Recent results have predominantly focused on the setup where no structural assumption is made on the system matrix $A^* \in \mathbb{R}^{n \times n}$, and have consequently analyzed the ordinary least squares (OLS) estimator in detail. We assume prior structural information on $A^*$ is available, which can be captured in the form of a convex set $\mathcal{K}$ containing $A^*$. For the solution of the ensuing constrained least squares estimator, we derive non-asymptotic error bounds in the Frobenius norm which depend on the local size of the tangent cone of $\mathcal{K}$ at $A^*$. To illustrate the usefulness of this result, we instantiate it for the settings where, (i) $\mathcal{K}$ is a $d$ dimensional subspace of $\mathbb{R}^{n \times n}$, or (ii) $A^*$ is $k$-sparse and $\mathcal{K}$ is a suitably scaled $\ell_1$ ball. In the regimes where $d, k \ll n^2$, our bounds improve upon those obta
    
[^13]: Skeleton Regression：一种基于流形结构估计的基于图形的方法。

    Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure. (arXiv:2303.11786v1 [cs.LG])

    [http://arxiv.org/abs/2303.11786](http://arxiv.org/abs/2303.11786)

    这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。

    

    我们引入了一个新的回归框架，旨在处理围绕低维流形的复杂数据。我们的方法首先构建一个图形表示，称为骨架，以捕获潜在的几何结构。然后，我们在骨架图上定义指标，应用非参数回归技术，以及基于图形的特征转换来估计回归函数。除了包括的非参数方法外，我们还讨论了一些非参数回归器在骨架图等一般度量空间方面的限制。所提出的回归框架使我们能够避开维度灾难，具有可以处理多个流形的并集并且鲁棒性能应对加性噪声和嘈杂观察的额外优势。我们为所提出的方法提供了统计保证，并通过模拟和实际数据示例证明了其有效性。

    We introduce a new regression framework designed to deal with large-scale, complex data that lies around a low-dimensional manifold. Our approach first constructs a graph representation, referred to as the skeleton, to capture the underlying geometric structure. We then define metrics on the skeleton graph and apply nonparametric regression techniques, along with feature transformations based on the graph, to estimate the regression function. In addition to the included nonparametric methods, we also discuss the limitations of some nonparametric regressors with respect to the general metric space such as the skeleton graph. The proposed regression framework allows us to bypass the curse of dimensionality and provides additional advantages that it can handle the union of multiple manifolds and is robust to additive noise and noisy observations. We provide statistical guarantees for the proposed method and demonstrate its effectiveness through simulations and real data examples.
    
[^14]: 具有谱正则化的核双样本检验

    Spectral Regularized Kernel Two-Sample Tests. (arXiv:2212.09201v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2212.09201](http://arxiv.org/abs/2212.09201)

    本文研究了基于概率分布的再生核希尔伯特空间嵌入的双样本检验的最优性。我们发现最大均值差异（MMD）检验在分离边界方面并不是最优的，因此我们提出了一种基于谱正则化的修改方法，使得检验具有更小的分离边界。同时，我们还提出了自适应版本的检验，通过数据驱动的策略选择正则化参数，展示了其近乎最优的性能。

    

    在过去的十年中，一种在非参数检验问题中广受欢迎的方法是基于概率分布的再生核希尔伯特空间（RKHS）嵌入的概念来处理一般（即非欧几里得）域上的问题。我们工作的主要目标是理解基于这种方法构建的双样本检验的最优性。首先，我们展示了流行的最大均值差异（MMD）双样本检验在Hellinger距离下的分离边界方面并不是最优的。其次，我们提出了一种基于谱正则化的MMD检验修改方法，通过考虑协方差信息（MMD检验无法捕获），证明了所提出的检验具有比MMD检验更小的分离边界的极小极大最优性。第三，我们提出了上述检验的自适应版本，其中涉及一种数据驱动策略来选择正则化参数，并展示了自适应检验几乎是最优的。

    Over the last decade, an approach that has gained a lot of popularity to tackle non-parametric testing problems on general (i.e., non-Euclidean) domains is based on the notion of reproducing kernel Hilbert space (RKHS) embedding of probability distributions. The main goal of our work is to understand the optimality of two-sample tests constructed based on this approach. First, we show that the popular MMD (maximum mean discrepancy) two-sample test is not optimal in terms of the separation boundary measured in Hellinger distance. Second, we propose a modification to the MMD test based on spectral regularization by taking into account the covariance information (which is not captured by the MMD test) and prove the proposed test to be minimax optimal with a smaller separation boundary than that achieved by the MMD test. Third, we propose an adaptive version of the above test which involves a data-driven strategy to choose the regularization parameter and show the adaptive test to be almos
    
[^15]: 用于理解神经网络动态的二次模型

    Quadratic models for understanding neural network dynamics. (arXiv:2205.11787v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.11787](http://arxiv.org/abs/2205.11787)

    神经二次模型可以展示出神经网络在大学习率情况下的“弹弓阶段”，并且在泛化特性上与神经网络有相似之处，是分析神经网络的有效工具。

    

    当神经网络的宽度增加时，可以用线性模型来逼近神经网络，但宽神经网络的某些特性不能被线性模型捕捉。在这项工作中，我们展示了最近提出的神经二次模型可以展示“弹弓阶段”[Lewkowycz等人，2020]，当使用大学习率训练此类模型时会出现。接着，我们经验证明，神经二次模型的行为与神经网络在泛化特性上有相似之处，尤其是在弹弓阶段范围内。我们的分析进一步表明，二次模型可以成为分析神经网络的有效工具。

    While neural networks can be approximated by linear models as their width increases, certain properties of wide neural networks cannot be captured by linear models. In this work we show that recently proposed Neural Quadratic Models can exhibit the "catapult phase" [Lewkowycz et al. 2020] that arises when training such models with large learning rates. We then empirically show that the behaviour of neural quadratic models parallels that of neural networks in generalization, especially in the catapult phase regime. Our analysis further demonstrates that quadratic models can be an effective tool for analysis of neural networks.
    

