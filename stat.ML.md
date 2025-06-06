# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Discovery from Conditionally Stationary Time Series](https://arxiv.org/abs/2110.06257) | 该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。 |
| [^2] | [High-Dimensional Independence Testing via Maximum and Average Distance Correlations](https://arxiv.org/abs/2001.01095) | 本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。 |
| [^3] | [Entropy-based Training Methods for Scalable Neural Implicit Sampler.](http://arxiv.org/abs/2306.04952) | 本文提出了一种高效且可扩展的神经隐式采样器，并引入了KL训练法和Fisher训练法来训练它，实现了低计算成本下生成大批量样本。 |
| [^4] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: 从有条件平稳时间序列中进行因果发现

    Causal Discovery from Conditionally Stationary Time Series

    [https://arxiv.org/abs/2110.06257](https://arxiv.org/abs/2110.06257)

    该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。

    

    因果发现，即从观测数据推断潜在的因果关系，已被证明对AI系统具有极大挑战。在时间序列建模背景下，传统的因果发现方法主要考虑具有完全观测变量和/或来自平稳时间序列的数据的受限场景。我们开发了一种因果发现方法来处理一类宽泛的非平稳时间序列，即在条件上是平稳的条件平稳时间序列，其中非平稳行为被建模为在一组（可能是隐藏的）状态变量上的平稳性。命名为State-Dependent Causal Inference（SDCI），我们的方法能够可证地回复出潜在的因果依赖关系，证明在完全观察到的状态下，并在存在隐藏状态时经验性地实现。后者通过对合成线性系统和非线性粒子相互作用数据的实验进行验证，SDCI实现了优于基线因果发现方法的性能。

    arXiv:2110.06257v2 Announce Type: replace  Abstract: Causal discovery, i.e., inferring underlying causal relationships from observational data, has been shown to be highly challenging for AI systems. In time series modeling context, traditional causal discovery methods mainly consider constrained scenarios with fully observed variables and/or data from stationary time-series. We develop a causal discovery approach to handle a wide class of non-stationary time-series that are conditionally stationary, where the non-stationary behaviour is modeled as stationarity conditioned on a set of (possibly hidden) state variables. Named State-Dependent Causal Inference (SDCI), our approach is able to recover the underlying causal dependencies, provably with fully-observed states and empirically with hidden states. The latter is confirmed by experiments on synthetic linear system and nonlinear particle interaction data, where SDCI achieves superior performance over baseline causal discovery methods
    
[^2]: 高维度独立性检测: 通过最大和平均距离相关性

    High-Dimensional Independence Testing via Maximum and Average Distance Correlations

    [https://arxiv.org/abs/2001.01095](https://arxiv.org/abs/2001.01095)

    本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。

    

    本文介绍并研究了利用最大和平均距离相关性进行多元独立性检测的方法。我们在高维环境中表征了它们相对于边际相关维度数量的一致性特性，评估了每个检验统计量的优势，检查了它们各自的零分布，并提出了一种基于快速卡方检验的检测程序。得出的检验是非参数的，并适用于欧氏距离和高斯核作为底层度量。为了更好地理解所提出的测试的实际使用情况，我们在各种多元相关场景中评估了最大距离相关性、平均距离相关性和原始距离相关性的实证表现，同时进行了一个真实数据实验，以检测人类血浆中不同癌症类型和肽水平的存在。

    This paper introduces and investigates the utilization of maximum and average distance correlations for multivariate independence testing. We characterize their consistency properties in high-dimensional settings with respect to the number of marginally dependent dimensions, assess the advantages of each test statistic, examine their respective null distributions, and present a fast chi-square-based testing procedure. The resulting tests are non-parametric and applicable to both Euclidean distance and the Gaussian kernel as the underlying metric. To better understand the practical use cases of the proposed tests, we evaluate the empirical performance of the maximum distance correlation, average distance correlation, and the original distance correlation across various multivariate dependence scenarios, as well as conduct a real data experiment to test the presence of various cancer types and peptide levels in human plasma.
    
[^3]: 基于熵的训练方法用于可扩展的神经隐式采样器

    Entropy-based Training Methods for Scalable Neural Implicit Sampler. (arXiv:2306.04952v1 [stat.ML])

    [http://arxiv.org/abs/2306.04952](http://arxiv.org/abs/2306.04952)

    本文提出了一种高效且可扩展的神经隐式采样器，并引入了KL训练法和Fisher训练法来训练它，实现了低计算成本下生成大批量样本。

    

    高效地从非标准目标分布中采样是科学计算和机器学习中的一个基本问题。传统方法如马尔科夫蒙特卡洛（MCMC）可保证从这些分布中渐进无偏采样，但在处理高维目标时计算效率低下，需要多次迭代生成一批样本。本文提出了一种高效且可扩展的神经隐式采样器，通过利用直接将易于采样的潜在向量映射到目标样本的神经变换，可以在低计算成本下生成大批量样本。为了训练神经隐式采样器，我们引入了两种新方法：KL训练法和Fisher训练法。前者最小化Kullback-Leibler散度，而后者则最小化Fisher散度。

    Efficiently sampling from un-normalized target distributions is a fundamental problem in scientific computing and machine learning. Traditional approaches like Markov Chain Monte Carlo (MCMC) guarantee asymptotically unbiased samples from such distributions but suffer from computational inefficiency, particularly when dealing with high-dimensional targets, as they require numerous iterations to generate a batch of samples. In this paper, we propose an efficient and scalable neural implicit sampler that overcomes these limitations. Our sampler can generate large batches of samples with low computational costs by leveraging a neural transformation that directly maps easily sampled latent vectors to target samples without the need for iterative procedures. To train the neural implicit sampler, we introduce two novel methods: the KL training method and the Fisher training method. The former minimizes the Kullback-Leibler divergence, while the latter minimizes the Fisher divergence. By empl
    
[^4]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

