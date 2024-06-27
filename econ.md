# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Individual and Collective Welfare in Risk Sharing with Many States.](http://arxiv.org/abs/2401.07337) | 在多状态的风险分享中，我们定量评估了福利，并证明了随着状态数增加，改善个体福利的空间消失，分配给改进聚合资源的范围也消失，并且在低效分配中，通过扰动聚合资源来增强福利的可能性随状态数的指数增加而变为零，同时，某些先验概率集随着状态空间的大小而缩小。 |
| [^3] | [STEEL: Singularity-aware Reinforcement Learning.](http://arxiv.org/abs/2301.13152) | 这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 多状态风险分享中的个体和集体福利

    Individual and Collective Welfare in Risk Sharing with Many States. (arXiv:2401.07337v1 [econ.TH])

    [http://arxiv.org/abs/2401.07337](http://arxiv.org/abs/2401.07337)

    在多状态的风险分享中，我们定量评估了福利，并证明了随着状态数增加，改善个体福利的空间消失，分配给改进聚合资源的范围也消失，并且在低效分配中，通过扰动聚合资源来增强福利的可能性随状态数的指数增加而变为零，同时，某些先验概率集随着状态空间的大小而缩小。

    

    我们对风险分享和不确定性下的经典模型中的福利进行了定量评估。我们证明了三种结果。首先，在均衡分配中，通过给定的边际提高个体福利（ε-提高）的空间随着状态数的增加而消失。其次，为了提高个体福利，分配给改进聚合资源的范围也消失。等价地说：在低效分配中，对于给定水平的资源次优化（根据资源未充分利用的系数度量），通过扰动聚合资源来增强福利的可能性随状态数的指数增加而变为零。最后，我们考虑了具有多个先验概率的标准不确定性规避中的高效风险分享，并显示在低效分配中，某些先验概率集随着状态空间的大小而缩小。

    We provide a quantitative assessment of welfare in the classical model of risk-sharing and exchange under uncertainty. We prove three kinds of results. First, that in an equilibrium allocation, the scope for improving individual welfare by a given margin (an $\ve$-improvement) vanishes as the number of states increases. Second, that the scope for a change in aggregate resources that may be distributed to enhance individual welfare by a given margin also vanishes. Equivalently: in an inefficient allocation, for a given level of resource sub-optimality (as measured by the coefficient of resource under-utilization), the possibilities for enhancing welfare by perturbing aggregate resources decrease exponentially to zero with the number of states. Finally, we consider efficient risk-sharing in standard models of uncertainty aversion with multiple priors, and show that, in an inefficient allocation, certain sets of priors shrink with the size of the state space.
    
[^3]: STEEL: 奇异性感知的强化学习

    STEEL: Singularity-aware Reinforcement Learning. (arXiv:2301.13152v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.13152](http://arxiv.org/abs/2301.13152)

    这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。

    

    批量强化学习旨在利用预先收集的数据，在动态环境中找到最优策略，以最大化期望总回报。然而，几乎所有现有算法都依赖于目标策略诱导的分布绝对连续假设，以便通过变换测度使用批量数据来校准目标策略。本文提出了一种新的批量强化学习算法，不需要在具有连续状态和行动的无限时马尔可夫决策过程中绝对连续性假设。我们称这个算法为STEEL：SingulariTy-awarE rEinforcement Learning。我们的算法受到关于离线评估的新误差分析的启发，其中我们使用了最大均值偏差，以及带有分布鲁棒优化的策略定向误差评估方法，以确保异常情况下的性能，并提出了一种用于处理奇异情况的定向算法。

    Batch reinforcement learning (RL) aims at leveraging pre-collected data to find an optimal policy that maximizes the expected total rewards in a dynamic environment. Nearly all existing algorithms rely on the absolutely continuous assumption on the distribution induced by target policies with respect to the data distribution, so that the batch data can be used to calibrate target policies via the change of measure. However, the absolute continuity assumption could be violated in practice (e.g., no-overlap support), especially when the state-action space is large or continuous. In this paper, we propose a new batch RL algorithm without requiring absolute continuity in the setting of an infinite-horizon Markov decision process with continuous states and actions. We call our algorithm STEEL: SingulariTy-awarE rEinforcement Learning. Our algorithm is motivated by a new error analysis on off-policy evaluation, where we use maximum mean discrepancy, together with distributionally robust opti
    

