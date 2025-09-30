# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two-Stage Nuisance Function Estimation for Causal Mediation Analysis](https://arxiv.org/abs/2404.00735) | 通过两阶段估计策略，该研究提出了一种针对因果中介分析中干扰函数的方法，旨在根据其在偏差结构中的作用来估计干扰函数。 |
| [^2] | [Off-Policy Evaluation in Markov Decision Processes under Weak Distributional Overlap](https://arxiv.org/abs/2402.08201) | 本文研究了弱分布重叠下马尔可夫决策过程中的离策略评估问题，并提出了一种截断双重稳健（TDR）估计器，在这种情况下表现良好。 |
| [^3] | [Causal inference for the expected number of recurrent events in the presence of a terminal event.](http://arxiv.org/abs/2306.16571) | 在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。 |

# 详细

[^1]: 因果中介分析的两阶段干扰函数估计

    Two-Stage Nuisance Function Estimation for Causal Mediation Analysis

    [https://arxiv.org/abs/2404.00735](https://arxiv.org/abs/2404.00735)

    通过两阶段估计策略，该研究提出了一种针对因果中介分析中干扰函数的方法，旨在根据其在偏差结构中的作用来估计干扰函数。

    

    在使用基于影响函数的中介功能估计器估计直接和间接因果效应时，了解应该关注治疗、中介和结果的哪些方面是至关重要的。具体而言，将它们视为干扰函数，并试图尽可能准确地拟合这些干扰函数并不一定是最好的方法。在这项工作中，我们提出了一种针对干扰函数的两阶段估计策略，该策略根据干扰函数在影响函数的中介功能估计器的偏差结构中发挥的作用来估计干扰函数。我们对所提出方法进行了稳健性分析，以及参数估计器的一致性和渐近正态性的充分条件。

    arXiv:2404.00735v1 Announce Type: cross  Abstract: When estimating the direct and indirect causal effects using the influence function-based estimator of the mediation functional, it is crucial to understand what aspects of the treatment, the mediator, and the outcome mean mechanisms should be focused on. Specifically, considering them as nuisance functions and attempting to fit these nuisance functions as accurate as possible is not necessarily the best approach to take. In this work, we propose a two-stage estimation strategy for the nuisance functions that estimates the nuisance functions based on the role they play in the structure of the bias of the influence function-based estimator of the mediation functional. We provide robustness analysis of the proposed method, as well as sufficient conditions for consistency and asymptotic normality of the estimator of the parameter of interest.
    
[^2]: 弱分布重叠下马尔可夫决策过程中的离策略评估

    Off-Policy Evaluation in Markov Decision Processes under Weak Distributional Overlap

    [https://arxiv.org/abs/2402.08201](https://arxiv.org/abs/2402.08201)

    本文研究了弱分布重叠下马尔可夫决策过程中的离策略评估问题，并提出了一种截断双重稳健（TDR）估计器，在这种情况下表现良好。

    

    在马尔可夫决策过程（MDP）中，双重稳健方法在序列可忽略性下对离策略评估具有很大的潜力：它们已经证明了随着时长T的收敛速度为$1/\sqrt{T}$，在大样本中具有统计效率，并且可以通过标准强化学习技术执行预估任务，具有模块化实现的能力。然而，现有结果在很大程度上使用了强分布重叠假设，即目标政策和数据收集政策的稳态分布相差在有限因子内，而这个假设通常只在MDP的状态空间有界时才可信。在本文中，我们重新审视了在弱分布重叠概念下的MDP离策略评估任务，并引入了一类截断双重稳健（TDR）估计器，在这种情况下表现良好。当目标和数据收集的分布比率有界时，我们证明了这些估计器的一致性。

    Doubly robust methods hold considerable promise for off-policy evaluation in Markov decision processes (MDPs) under sequential ignorability: They have been shown to converge as $1/\sqrt{T}$ with the horizon $T$, to be statistically efficient in large samples, and to allow for modular implementation where preliminary estimation tasks can be executed using standard reinforcement learning techniques. Existing results, however, make heavy use of a strong distributional overlap assumption whereby the stationary distributions of the target policy and the data-collection policy are within a bounded factor of each other -- and this assumption is typically only credible when the state space of the MDP is bounded. In this paper, we re-visit the task of off-policy evaluation in MDPs under a weaker notion of distributional overlap, and introduce a class of truncated doubly robust (TDR) estimators which we find to perform well in this setting. When the distribution ratio of the target and data-coll
    
[^3]: 在存在终结事件的情况下，关于经常性事件的因果推断

    Causal inference for the expected number of recurrent events in the presence of a terminal event. (arXiv:2306.16571v1 [stat.ME])

    [http://arxiv.org/abs/2306.16571](http://arxiv.org/abs/2306.16571)

    在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。

    

    我们研究了在存在终结事件的情况下，关于经常性事件的因果推断和高效估计。我们将估计目标定义为包括经常性事件的预期数量以及在一系列里程碑时间点处评估的失败生存函数的向量。我们在右截尾和因果选择的情况下确定了估计目标，作为观察数据的功能性，推导了非参数效率界限，并提出了一种多重鲁棒估计器，该估计器达到了界限，并允许非参数估计辅助参数。在整个过程中，我们对失败、截尾或观察数据的概率分布没有做绝对连续性的假设。此外，当分割分布已知时，我们导出了影响函数的类别，并回顾了已发表估计器如何属于该类别。在此过程中，我们强调了因果生命周期中一些有趣的不一致性。

    We study causal inference and efficient estimation for the expected number of recurrent events in the presence of a terminal event. We define our estimand as the vector comprising both the expected number of recurrent events and the failure survival function evaluated along a sequence of landmark times. We identify the estimand in the presence of right-censoring and causal selection as an observed data functional under coarsening at random, derive the nonparametric efficiency bound, and propose a multiply-robust estimator that achieves the bound and permits nonparametric estimation of nuisance parameters. Throughout, no absolute continuity assumption is made on the underlying probability distributions of failure, censoring, or the observed data. Additionally, we derive the class of influence functions when the coarsening distribution is known and review how published estimators may belong to the class. Along the way, we highlight some interesting inconsistencies in the causal lifetime 
    

