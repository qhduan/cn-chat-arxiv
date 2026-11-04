# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Efficient Kernel Methods for Learning Differential Equations and Their Solution Operators: Algorithms and Error Analysis](https://arxiv.org/abs/2503.01036) | 提出了一种数据高效的核方法框架用于学习微分方程及其解算子，结合高斯-牛顿类优化算法并具有严格的最坏情况误差界理论保证，在精度上比现有最先进算法提升一到两个数量级。 |

# 详细

[^1]: 用于学习微分方程及其解算子的数据高效核方法：算法与误差分析

    Data-Efficient Kernel Methods for Learning Differential Equations and Their Solution Operators: Algorithms and Error Analysis

    [https://arxiv.org/abs/2503.01036](https://arxiv.org/abs/2503.01036)

    提出了一种数据高效的核方法框架用于学习微分方程及其解算子，结合高斯-牛顿类优化算法并具有严格的最坏情况误差界理论保证，在精度上比现有最先进算法提升一到两个数量级。

    

    我们提出了一种新颖的基于核方法的框架，用于学习微分方程及其解映射。该框架在数据需求方面非常高效，包括所需观测状态的数量以及每个状态可用的数据量。我们的公式化方法还允许我们在训练过程中采用高效的高斯-牛顿类优化算法，以提高训练过程的精度和速度。我们还为算法提供了严格的理论保证，其形式为学习到的方程的定量最坏情况误差界，并且在光滑性假设下该误差界可以扩展到后续的解算子。数值基准测试表明，与某些最先进的算法相比，该框架在计算复杂度和鲁棒性方面有显著改进，同时在精度方面实现了一到两个数量级的提升。

    arXiv:2503.01036v3 Announce Type: replace  Abstract: We introduce a novel kernel-based framework for learning differential equations and their solution maps. The framework is efficient in its data requirements in terms of the number of observed states and the amount of data available per state. Our formulation also allows us to employ efficient Gauss-Newton-type optimization algorithms in the training procedure to improve accuracy and speed of the training procedure. We also support our algorithm with rigorous theoretical guarantees in the form of quantitative worst-case error bounds for the learned equation that can be extended to the subsequent solution operators under smoothness assumptions. Numerical benchmarks demonstrate significant improvements in computational complexity and robustness while achieving one to two orders of magnitude improvements in terms of accuracy compared to certain state-of-the-art algorithms.
    

