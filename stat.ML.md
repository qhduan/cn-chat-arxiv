# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerated First-Order Optimization under Nonlinear Constraints.](http://arxiv.org/abs/2302.00316) | 设计了一种新的加速非线性约束下的一阶优化算法，其优点是避免了在整个可行集上进行优化，而且利用速度来表达约束，使得算法在决策变量数量和约束数量上的复杂度增长适度，适用于机器学习应用。 |
| [^2] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: 加速非线性约束下的一阶优化

    Accelerated First-Order Optimization under Nonlinear Constraints. (arXiv:2302.00316v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.00316](http://arxiv.org/abs/2302.00316)

    设计了一种新的加速非线性约束下的一阶优化算法，其优点是避免了在整个可行集上进行优化，而且利用速度来表达约束，使得算法在决策变量数量和约束数量上的复杂度增长适度，适用于机器学习应用。

    

    我们利用约束优化和非光滑动力系统之间的类比，设计了一种新的加速非线性约束下的一阶优化算法。与Frank-Wolfe或投影梯度不同，这些算法避免了每次迭代在整个可行集上进行优化。我们证明了在非凸设置中的收敛性，并推导了在连续时间和离散时间中的凸设置的加速率。这些算法的一个重要特性是使用速度而不是位置来表达约束，这自然地导致可行集的稀疏、局部和凸近似（即使可行集是非凸的）。因此，复杂度在决策变量数量和约束数量上适度增长，使得该算法适用于机器学习应用。我们将算法应用于压缩感知和稀疏···

    We exploit analogies between first-order algorithms for constrained optimization and non-smooth dynamical systems to design a new class of accelerated first-order algorithms for constrained optimization. Unlike Frank-Wolfe or projected gradients, these algorithms avoid optimization over the entire feasible set at each iteration. We prove convergence to stationary points even in a nonconvex setting and we derive accelerated rates for the convex setting both in continuous time, as well as in discrete time. An important property of these algorithms is that constraints are expressed in terms of velocities instead of positions, which naturally leads to sparse, local and convex approximations of the feasible set (even if the feasible set is nonconvex). Thus, the complexity tends to grow mildly in the number of decision variables and in the number of constraints, which makes the algorithms suitable for machine learning applications. We apply our algorithms to a compressed sensing and a sparse
    
[^2]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

