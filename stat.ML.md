# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |
| [^2] | [Distributionally robust risk evaluation with a causality constraint and structural information.](http://arxiv.org/abs/2203.10571) | 本文提出了具有因果约束的分布鲁棒风险评估方法，并用神经网络逼近测试函数。在结构信息有限制时，提供了高效的优化方法。 |

# 详细

[^1]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    
[^2]: 具有因果约束和结构信息的分布鲁棒风险评估

    Distributionally robust risk evaluation with a causality constraint and structural information. (arXiv:2203.10571v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2203.10571](http://arxiv.org/abs/2203.10571)

    本文提出了具有因果约束的分布鲁棒风险评估方法，并用神经网络逼近测试函数。在结构信息有限制时，提供了高效的优化方法。

    

    本文研究了基于时间数据的期望函数值的分布鲁棒评估。一组备选度量通过因果最优传输进行表征。我们证明了强对偶性，并将因果约束重构为无限维测试函数空间的最小化问题。我们通过神经网络逼近测试函数，并用Rademacher复杂度证明了样本复杂度。此外，当结构信息可用于进一步限制模糊集时，我们证明了对偶形式并提供高效的优化方法。对实现波动率和股指的经验分析表明，我们的框架为经典最优传输公式提供了一种有吸引力的替代方案。

    This work studies distributionally robust evaluation of expected function values over temporal data. A set of alternative measures is characterized by the causal optimal transport. We prove the strong duality and recast the causality constraint as minimization over an infinite-dimensional test function space. We approximate test functions by neural networks and prove the sample complexity with Rademacher complexity. Moreover, when structural information is available to further restrict the ambiguity set, we prove the dual formulation and provide efficient optimization methods. Empirical analysis of realized volatility and stock indices demonstrates that our framework offers an attractive alternative to the classic optimal transport formulation.
    

