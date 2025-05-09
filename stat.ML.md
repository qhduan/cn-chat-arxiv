# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Objective Optimization Using the R2 Utility.](http://arxiv.org/abs/2305.11774) | 本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。 |

# 详细

[^1]: 使用R2效用的多目标优化

    Multi-Objective Optimization Using the R2 Utility. (arXiv:2305.11774v1 [math.OC])

    [http://arxiv.org/abs/2305.11774](http://arxiv.org/abs/2305.11774)

    本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。

    

    多目标优化的目标是确定描述多目标之间最佳权衡的点集合。为了解决这个矢量值优化问题，从业者常常使用标量化函数将多目标问题转化为一组单目标问题。这组标量化问题可以使用传统的单目标优化技术来解决。在这项工作中，我们将这个约定形式化为一个通用的数学框架。我们展示了这种策略如何有效地将原始的多目标优化问题重新转化为定义在集合上的单目标优化问题。针对这个新问题的适当类别的目标函数是R2效用函数，它被定义为标量化优化问题的加权积分。我们证明了这个效用函数是单调的和次模的集合函数，可以通过贪心优化算法有效地计算出全局最优解。

    The goal of multi-objective optimization is to identify a collection of points which describe the best possible trade-offs between the multiple objectives. In order to solve this vector-valued optimization problem, practitioners often appeal to the use of scalarization functions in order to transform the multi-objective problem into a collection of single-objective problems. This set of scalarized problems can then be solved using traditional single-objective optimization techniques. In this work, we formalise this convention into a general mathematical framework. We show how this strategy effectively recasts the original multi-objective optimization problem into a single-objective optimization problem defined over sets. An appropriate class of objective functions for this new problem is the R2 utility function, which is defined as a weighted integral over the scalarized optimization problems. We show that this utility function is a monotone and submodular set function, which can be op
    

