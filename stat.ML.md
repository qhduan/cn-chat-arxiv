# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Off-Policy Evaluation and Learning for Large Action Spaces](https://arxiv.org/abs/2402.14664) | 该论文提出了一个统一的贝叶斯框架，通过结构化和信息丰富的先验捕捉动作之间的相关性，提出了一个适用于离策略评估和学习的通用贝叶斯方法sDM，并引入了能评估算法在多问题实例中平均表现的贝叶斯指标，分析了sDM在OPE和OPL中利用动作相关性的优势，并展示了其强大性能 |

# 详细

[^1]: 大动作空间的贝叶斯离策略评估与学习

    Bayesian Off-Policy Evaluation and Learning for Large Action Spaces

    [https://arxiv.org/abs/2402.14664](https://arxiv.org/abs/2402.14664)

    该论文提出了一个统一的贝叶斯框架，通过结构化和信息丰富的先验捕捉动作之间的相关性，提出了一个适用于离策略评估和学习的通用贝叶斯方法sDM，并引入了能评估算法在多问题实例中平均表现的贝叶斯指标，分析了sDM在OPE和OPL中利用动作相关性的优势，并展示了其强大性能

    

    在交互式系统中，动作经常是相关的，这为大动作空间中更有效的离策略评估（OPE）和学习（OPL）提供了机会。我们引入了一个统一的贝叶斯框架，通过结构化和信息丰富的先验来捕捉这些相关性。在该框架中，我们提出了sDM，一个为OPE和OPL设计的通用贝叶斯方法，既有算法基础又有理论基础。值得注意的是，sDM利用动作相关性而不会影响计算效率。此外，受在线贝叶斯赌博机启发，我们引入了评估算法在多个问题实例中平均性能的贝叶斯指标，偏离传统的最坏情况评估。我们分析了sDM在OPE和OPL中的表现，凸显了利用动作相关性的好处。实证证据展示了sDM的强大性能。

    arXiv:2402.14664v1 Announce Type: cross  Abstract: In interactive systems, actions are often correlated, presenting an opportunity for more sample-efficient off-policy evaluation (OPE) and learning (OPL) in large action spaces. We introduce a unified Bayesian framework to capture these correlations through structured and informative priors. In this framework, we propose sDM, a generic Bayesian approach designed for OPE and OPL, grounded in both algorithmic and theoretical foundations. Notably, sDM leverages action correlations without compromising computational efficiency. Moreover, inspired by online Bayesian bandits, we introduce Bayesian metrics that assess the average performance of algorithms across multiple problem instances, deviating from the conventional worst-case assessments. We analyze sDM in OPE and OPL, highlighting the benefits of leveraging action correlations. Empirical evidence showcases the strong performance of sDM.
    

