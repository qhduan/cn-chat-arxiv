# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Negatively dependent optimal risk sharing.](http://arxiv.org/abs/2401.03328) | 本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。 |
| [^2] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |

# 详细

[^1]: 负相关的最优风险共担问题研究

    Negatively dependent optimal risk sharing. (arXiv:2401.03328v1 [econ.TH])

    [http://arxiv.org/abs/2401.03328](http://arxiv.org/abs/2401.03328)

    本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。

    

    本文分析了使用表现出反单调性的分配方式来优化共享风险的问题。反单调分配的形式有“赢者通吃”或“输者全军覆没”型彩票，我们分别将其归为标准化的“大奖”或“替罪羊”分配。我们的主要定理——反单调改进定理，说明对于一组随机变量，无论它们是全部下界有界还是全部上界有界，总是可以找到一组反单调随机变量，其中每个分量都大于或等于凸序中对应的分量。我们证明了如果帕累托最优分配存在且所有代理都追求风险，那么它们必须是大奖分配。而当所有代理的不连续伯努利效用函数时，我们得到了相反的结论，替罪羊分配使得超过不连续阈值的概率最大化。

    We analyze the problem of optimally sharing risk using allocations that exhibit counter-monotonicity, the most extreme form of negative dependence. Counter-monotonic allocations take the form of either "winner-takes-all" lotteries or "loser-loses-all" lotteries, and we respectively refer to these (normalized) cases as jackpot or scapegoat allocations. Our main theorem, the counter-monotonic improvement theorem, states that for a given set of random variables that are either all bounded from below or all bounded from above, one can always find a set of counter-monotonic random variables such that each component is greater or equal than its counterpart in the convex order. We show that Pareto optimal allocations, if they exist, must be jackpot allocations when all agents are risk seeking. We essentially obtain the opposite when all agents have discontinuous Bernoulli utility functions, as scapegoat allocations maximize the probability of being above the discontinuity threshold. We also c
    
[^2]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    

