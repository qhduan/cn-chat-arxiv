# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Negatively dependent optimal risk sharing.](http://arxiv.org/abs/2401.03328) | 本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。 |
| [^2] | [Design of Cluster-Randomized Trials with Cross-Cluster Interference.](http://arxiv.org/abs/2310.18836) | 该论文提出了一种新的集群随机试验设计方法，考虑了交叉集群干扰的问题。通过排除可能受到干扰影响的单元，提出了新的估计器，并证明了这种方法可以大大减少偏差。这项研究还提供了优化估计器收敛速率的集群设计方法。 |
| [^3] | [Estimating Nonlinear Network Data Models with Fixed Effects.](http://arxiv.org/abs/2203.15603) | 本文提出一种使用杰克刀程序修正带有个体固定效应的非线性网络数据模型偏差的新方法，可适用于有向和无向网络，对非二元结果变量进行估计，并可用于修正平均效应和反事实结局的估算。 |

# 详细

[^1]: 负相关的最优风险共担问题研究

    Negatively dependent optimal risk sharing. (arXiv:2401.03328v1 [econ.TH])

    [http://arxiv.org/abs/2401.03328](http://arxiv.org/abs/2401.03328)

    本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。

    

    本文分析了使用表现出反单调性的分配方式来优化共享风险的问题。反单调分配的形式有“赢者通吃”或“输者全军覆没”型彩票，我们分别将其归为标准化的“大奖”或“替罪羊”分配。我们的主要定理——反单调改进定理，说明对于一组随机变量，无论它们是全部下界有界还是全部上界有界，总是可以找到一组反单调随机变量，其中每个分量都大于或等于凸序中对应的分量。我们证明了如果帕累托最优分配存在且所有代理都追求风险，那么它们必须是大奖分配。而当所有代理的不连续伯努利效用函数时，我们得到了相反的结论，替罪羊分配使得超过不连续阈值的概率最大化。

    We analyze the problem of optimally sharing risk using allocations that exhibit counter-monotonicity, the most extreme form of negative dependence. Counter-monotonic allocations take the form of either "winner-takes-all" lotteries or "loser-loses-all" lotteries, and we respectively refer to these (normalized) cases as jackpot or scapegoat allocations. Our main theorem, the counter-monotonic improvement theorem, states that for a given set of random variables that are either all bounded from below or all bounded from above, one can always find a set of counter-monotonic random variables such that each component is greater or equal than its counterpart in the convex order. We show that Pareto optimal allocations, if they exist, must be jackpot allocations when all agents are risk seeking. We essentially obtain the opposite when all agents have discontinuous Bernoulli utility functions, as scapegoat allocations maximize the probability of being above the discontinuity threshold. We also c
    
[^2]: 设计具有交叉集群干扰的集群随机试验

    Design of Cluster-Randomized Trials with Cross-Cluster Interference. (arXiv:2310.18836v1 [stat.ME])

    [http://arxiv.org/abs/2310.18836](http://arxiv.org/abs/2310.18836)

    该论文提出了一种新的集群随机试验设计方法，考虑了交叉集群干扰的问题。通过排除可能受到干扰影响的单元，提出了新的估计器，并证明了这种方法可以大大减少偏差。这项研究还提供了优化估计器收敛速率的集群设计方法。

    

    集群随机试验经常涉及空间上分布不规律且没有明显分离社区的单元。在这种情况下，由于潜在的交叉集群干扰，集群构建是设计的一个关键方面。现有的文献依赖于部分干扰模型，该模型将集群视为给定，并假设没有交叉集群干扰。我们通过允许干扰与单元之间的地理距离衰减来放宽这个假设。这导致了一个偏差-方差的权衡：构建较少、较大的集群可以减少干扰引起的偏差，但会增加方差。我们提出了一种新的估计器，排除可能受到交叉集群干扰影响的单元，并显示相对于传统的均值差估计器，这大大降低了渐近偏差。然后，我们研究了优化估计器收敛速率的集群设计。我们提供了一个新的设计的正式证明，该设计选择了集群的数量。

    Cluster-randomized trials often involve units that are irregularly distributed in space without well-separated communities. In these settings, cluster construction is a critical aspect of the design due to the potential for cross-cluster interference. The existing literature relies on partial interference models, which take clusters as given and assume no cross-cluster interference. We relax this assumption by allowing interference to decay with geographic distance between units. This induces a bias-variance trade-off: constructing fewer, larger clusters reduces bias due to interference but increases variance. We propose new estimators that exclude units most potentially impacted by cross-cluster interference and show that this substantially reduces asymptotic bias relative to conventional difference-in-means estimators. We then study the design of clusters to optimize the estimators' rates of convergence. We provide formal justification for a new design that chooses the number of clus
    
[^3]: 修正带有固定效应的非线性网络数据模型的估计方法

    Estimating Nonlinear Network Data Models with Fixed Effects. (arXiv:2203.15603v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.15603](http://arxiv.org/abs/2203.15603)

    本文提出一种使用杰克刀程序修正带有个体固定效应的非线性网络数据模型偏差的新方法，可适用于有向和无向网络，对非二元结果变量进行估计，并可用于修正平均效应和反事实结局的估算。

    

    本论文提出一种新的方法来修正带有特定个体的固定特征的二元模型——其中包括同质性和度异质性的二元链接形成模型的偏差。所提出的方法使用了杰克刀（jackknife）程序来处理关于偶然参数的问题。该方法可应用于有向和无向网络，并且允许使用非二元结果变量，并可用于修正平均效应和反事实结局的估计。作者还展示了如何使用杰克刀来纠正对多个节点的固定效应平均值的偏差，例如三元组或四元组。最后，作者在一个关于跨国进出口关系的引力模型的应用中展示了该估计器的实用性。

    I introduce a new method for bias correction of dyadic models with agent-specific fixed-effects, including the dyadic link formation model with homophily and degree heterogeneity. The proposed approach uses a jackknife procedure to deal with the incidental parameters problem. The method can be applied to both directed and undirected networks, allows for non-binary outcome variables, and can be used to bias correct estimates of average effects and counterfactual outcomes. I also show how the jackknife can be used to bias-correct fixed effect averages over functions that depend on multiple nodes, e.g. triads or tetrads in the network. As an example, I implement specification tests for dependence across dyads, such as reciprocity or transitivity. Finally, I demonstrate the usefulness of the estimator in an application to a gravity model for import/export relationships across countries.
    

