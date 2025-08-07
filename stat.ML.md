# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Thompson Exploration with Best Challenger Rule in Best Arm Identification.](http://arxiv.org/abs/2310.00539) | 本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。 |
| [^2] | [Symmetry & Critical Points for Symmetric Tensor Decompositions Problems.](http://arxiv.org/abs/2306.07886) | 本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。 |
| [^3] | [Optimal Learning via Moderate Deviations Theory.](http://arxiv.org/abs/2305.14496) | 本文提出了一种能够在广泛模型中进行最优学习的方法，利用中度偏差原理构建高度准确的置信区间，满足指数精度、一致性和最大精度等标准，为该方法提供了理论依据。 |

# 详细

[^1]: 最佳候选规则下的Thompson探索在最佳臂识别中的应用

    Thompson Exploration with Best Challenger Rule in Best Arm Identification. (arXiv:2310.00539v1 [stat.ML])

    [http://arxiv.org/abs/2310.00539](http://arxiv.org/abs/2310.00539)

    本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。

    

    本文研究了在经典单参数指数模型下，固定置信度下的最佳臂识别（BAI）问题。针对这个问题，目前已有很多策略被提出，但大多数需要在每一轮解决一个最优化问题和/或者需要探索一个臂至少一定次数，除非是针对高斯模型的限制。为了解决这些限制，我们提出了一种新的策略，将Thompson采样与一个计算效率高的方法——最佳候选规则相结合。虽然Thompson采样最初被考虑用于最大化累积奖励，但我们证明它也可以自然地用于在BAI中探索臂而不强迫最大化奖励。我们证明了我们的策略在任意两臂赌博机问题上是渐近最优的，并且在一般的$K$臂赌博机问题上（$K\geq 3$）达到接近最优的性能。然而，在数值实验中，我们的策略与现有方法相比表现出了竞争性的性能。

    This paper studies the fixed-confidence best arm identification (BAI) problem in the bandit framework in the canonical single-parameter exponential models. For this problem, many policies have been proposed, but most of them require solving an optimization problem at every round and/or are forced to explore an arm at least a certain number of times except those restricted to the Gaussian model. To address these limitations, we propose a novel policy that combines Thompson sampling with a computationally efficient approach known as the best challenger rule. While Thompson sampling was originally considered for maximizing the cumulative reward, we demonstrate that it can be used to naturally explore arms in BAI without forcing it. We show that our policy is asymptotically optimal for any two-armed bandit problems and achieves near optimality for general $K$-armed bandit problems for $K\geq 3$. Nevertheless, in numerical experiments, our policy shows competitive performance compared to as
    
[^2]: 对称张量分解问题的对称性与临界点

    Symmetry & Critical Points for Symmetric Tensor Decompositions Problems. (arXiv:2306.07886v1 [math.OC])

    [http://arxiv.org/abs/2306.07886](http://arxiv.org/abs/2306.07886)

    本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。

    

    本文考虑了将一个实对称张量分解成秩为1项之和的非凸优化问题。利用其丰富的对称结构，导出Puiseux级数表示的一系列临界点，并获得了关于临界值和Hessian谱的精确分析估计。这些结果揭示了各种几何障碍，阻碍了局部优化方法的使用，最后，利用一个牛顿多面体论证了固定对称性的所有临界点的完全枚举，并证明了与全局最小值的集合相比，由于对称性的存在，临界点的集合可能会显示出组合的丰富性。

    We consider the non-convex optimization problem associated with the decomposition of a real symmetric tensor into a sum of rank one terms. Use is made of the rich symmetry structure to derive Puiseux series representations of families of critical points, and so obtain precise analytic estimates on the critical values and the Hessian spectrum. The sharp results make possible an analytic characterization of various geometric obstructions to local optimization methods, revealing in particular a complex array of saddles and local minima which differ by their symmetry, structure and analytic properties. A desirable phenomenon, occurring for all critical points considered, concerns the index of a point, i.e., the number of negative Hessian eigenvalues, increasing with the value of the objective function. Lastly, a Newton polytope argument is used to give a complete enumeration of all critical points of fixed symmetry, and it is shown that contrarily to the set of global minima which remains 
    
[^3]: 通过中度偏差理论进行最优学习

    Optimal Learning via Moderate Deviations Theory. (arXiv:2305.14496v1 [stat.ML])

    [http://arxiv.org/abs/2305.14496](http://arxiv.org/abs/2305.14496)

    本文提出了一种能够在广泛模型中进行最优学习的方法，利用中度偏差原理构建高度准确的置信区间，满足指数精度、一致性和最大精度等标准，为该方法提供了理论依据。

    

    本文提出了一种在广泛模型中使用置信区间学习函数值的统计最优方法，包括描述为随机规划问题或各种SDE模型的期望损失的一般非参数估计。更准确地说，我们通过采用基于中度偏差原理的方法系统地构建高度准确的置信区间。研究表明，所提出的置信区间在统计意义上是最优的，因为它们满足以指数精度、最小性、一致性、误判概率以及最终的一致最大精度为标准的要求。该方法提出的置信区间是通过强化优化问题的解来表达的，其中不确定性通过数据生成过程引发的中度偏差率函数来表示。我们演示了对于许多模型，这些优化问题具有易于解的结果。

    This paper proposes a statistically optimal approach for learning a function value using a confidence interval in a wide range of models, including general non-parametric estimation of an expected loss described as a stochastic programming problem or various SDE models. More precisely, we develop a systematic construction of highly accurate confidence intervals by using a moderate deviation principle-based approach. It is shown that the proposed confidence intervals are statistically optimal in the sense that they satisfy criteria regarding exponential accuracy, minimality, consistency, mischaracterization probability, and eventual uniformly most accurate (UMA) property. The confidence intervals suggested by this approach are expressed as solutions to robust optimization problems, where the uncertainty is expressed via the underlying moderate deviation rate function induced by the data-generating process. We demonstrate that for many models these optimization problems admit tractable r
    

