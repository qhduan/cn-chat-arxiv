# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Short Review on Novel Approaches for Maximum Clique Problem: from Classical algorithms to Graph Neural Networks and Quantum algorithms](https://arxiv.org/abs/2403.09742) | 该综述回顾了解决最大团问题的经典算法，同时也涵盖了图神经网络和量子算法的最新进展，并提出了用于测试这些算法的基准。 |
| [^2] | [Bridging the Gap between Newton-Raphson Method and Regularized Policy Iteration.](http://arxiv.org/abs/2310.07211) | 正则化策略迭代和牛顿-拉弗森方法在使用强凸函数对贝尔曼方程平滑化的条件下严格等价，为正则化策略迭代的全局和局部收敛行为提供了统一分析。该算法具有全局线性收敛和局部二次收敛特性。 |

# 详细

[^1]: 最大团问题的新方法简要回顾：从经典算法到图神经网络和量子算法

    A Short Review on Novel Approaches for Maximum Clique Problem: from Classical algorithms to Graph Neural Networks and Quantum algorithms

    [https://arxiv.org/abs/2403.09742](https://arxiv.org/abs/2403.09742)

    该综述回顾了解决最大团问题的经典算法，同时也涵盖了图神经网络和量子算法的最新进展，并提出了用于测试这些算法的基准。

    

    这篇手稿全面回顾了最大团问题，这是一个涉及在图中找到所有两两相邻的顶点子集的计算问题。手稿以简单的方式涵盖了解决该问题的经典算法，并包括了对图神经网络和量子算法最近发展的审查。该综述以基准测试来评估经典以及新的学习和量子算法。

    arXiv:2403.09742v1 Announce Type: new  Abstract: This manuscript provides a comprehensive review of the Maximum Clique Problem, a computational problem that involves finding subsets of vertices in a graph that are all pairwise adjacent to each other. The manuscript covers in a simple way classical algorithms for solving the problem and includes a review of recent developments in graph neural networks and quantum algorithms. The review concludes with benchmarks for testing classical as well as new learning, and quantum algorithms.
    
[^2]: 缩小牛顿-拉弗森方法和正规化策略迭代之间的差距

    Bridging the Gap between Newton-Raphson Method and Regularized Policy Iteration. (arXiv:2310.07211v1 [cs.LG])

    [http://arxiv.org/abs/2310.07211](http://arxiv.org/abs/2310.07211)

    正则化策略迭代和牛顿-拉弗森方法在使用强凸函数对贝尔曼方程平滑化的条件下严格等价，为正则化策略迭代的全局和局部收敛行为提供了统一分析。该算法具有全局线性收敛和局部二次收敛特性。

    

    正则化是强化学习算法中最重要的技术之一。众所周知，软演员-评论家算法是正则化策略迭代的一个特例，其中正则化项选择为Shannon熵。尽管正则化策略迭代在实践中取得了一些成功，但其理论基础仍不清楚。本文证明，在使用强凸函数对贝尔曼方程平滑化的条件下，正则化策略迭代在严格意义上等价于标准的牛顿-拉弗森方法。这种等价性为正则化策略迭代的全局和局部收敛行为奠定了基础。我们证明正则化策略迭代具有全局线性收敛性，收敛速度为$\gamma$（折扣因子）。此外，一旦进入最优值周围的局部区域，该算法将二次收敛。我们还展示了正则化策略迭代的改进版本，即有限的正-----------此处省略部分内容---------------

    Regularization is one of the most important techniques in reinforcement learning algorithms. The well-known soft actor-critic algorithm is a special case of regularized policy iteration where the regularizer is chosen as Shannon entropy. Despite some empirical success of regularized policy iteration, its theoretical underpinnings remain unclear. This paper proves that regularized policy iteration is strictly equivalent to the standard Newton-Raphson method in the condition of smoothing out Bellman equation with strongly convex functions. This equivalence lays the foundation of a unified analysis for both global and local convergence behaviors of regularized policy iteration. We prove that regularized policy iteration has global linear convergence with the rate being $\gamma$ (discount factor). Furthermore, this algorithm converges quadratically once it enters a local region around the optimal value. We also show that a modified version of regularized policy iteration, i.e., with finite
    

