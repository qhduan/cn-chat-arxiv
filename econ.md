# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [The Assignment Game: New Mechanisms for Equitable Core Imputations](https://arxiv.org/abs/2402.11437) | 本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: 《分配博弈：公平核分配的新机制》

    The Assignment Game: New Mechanisms for Equitable Core Imputations

    [https://arxiv.org/abs/2402.11437](https://arxiv.org/abs/2402.11437)

    本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。

    

    《分配博弈》的核分配集形成一个（非有限）分配格。迄今为止，仅已知有效算法用于计算其两个极端分配；但是，其中每一个都最大程度地偏袒一个方，不利于双方的分配，导致盈利不均衡。另一个问题是，由一个玩家组成的子联盟（或者来自分配两侧的一系玩家）可以获得零利润，因此核分配不必给予他们任何利润。因此，核分配在个体代理人层面上不提供任何公平性保证。这引出一个问题，即如何计算更公平的核分配。在本文中，我们提出了一个计算分配博弈的Leximin和Leximax核分配的组合（即，该机制不涉及LP求解器）多项式时间机制。这些分配以不同方式实现了“公平性”：

    arXiv:2402.11437v1 Announce Type: cross  Abstract: The set of core imputations of the assignment game forms a (non-finite) distributive lattice. So far, efficient algorithms were known for computing only its two extreme imputations; however, each of them maximally favors one side and disfavors the other side of the bipartition, leading to inequitable profit sharing. Another issue is that a sub-coalition consisting of one player (or a set of players from the same side of the bipartition) can make zero profit, therefore a core imputation is not obliged to give them any profit. Hence core imputations make no fairness guarantee at the level of individual agents. This raises the question of computing {\em more equitable core imputations}.   In this paper, we give combinatorial (i.e., the mechanism does not invoke an LP-solver) polynomial time mechanisms for computing the leximin and leximax core imputations for the assignment game. These imputations achieve ``fairness'' in different ways: w
    

