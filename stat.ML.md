# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Hypothesis Testing for Information Value (IV).](http://arxiv.org/abs/2309.13183) | 该论文提出了信息价值（IV）的统计假设检验方法，为模型建立前的特征选择提供了理论框架，并通过实验证明了该方法的有效性。 |
| [^2] | [An efficient, provably exact algorithm for the 0-1 loss linear classification problem.](http://arxiv.org/abs/2306.12344) | 该研究详细介绍了一种名为增量单元枚举（ICE）的算法，该算法可以精确解决定维度0-1损失线性分类问题。 |

# 详细

[^1]: 信息价值（IV）的统计假设检验

    Statistical Hypothesis Testing for Information Value (IV). (arXiv:2309.13183v1 [math.ST])

    [http://arxiv.org/abs/2309.13183](http://arxiv.org/abs/2309.13183)

    该论文提出了信息价值（IV）的统计假设检验方法，为模型建立前的特征选择提供了理论框架，并通过实验证明了该方法的有效性。

    

    信息价值（IV）是模型建立前进行特征选择的一种常用技术。目前存在一些实际标准，但基于IV的判断是否一个预测因子具有足够的预测能力的理论依据依然神秘且缺乏。然而，关于该技术的数学发展和统计推断方法在文献中几乎没有提及。在本研究中，我们提出了一个关于IV的理论框架，并提出了一种非参数假设检验方法来测试预测能力。我们展示了如何高效计算检验统计量，并在模拟数据上研究其表现。此外，我们将这一方法应用于银行欺诈数据，并提供了一个实现我们结果的Python库。

    Information value (IV) is a quite popular technique for feature selection prior to the modeling phase. There are practical criteria, but at the same time mysterious and lacking theoretical arguments, based on the IV, to decide if a predictor has sufficient predictive power to be considered in the modeling phase. However, the mathematical development and statistical inference methods for this technique is almost non-existent in the literature. In this work we present a theoretical framework for the IV and propose a non-parametric hypothesis test to test the predictive power. We show how to efficiently calculate the test statistic and study its performance on simulated data. Additionally, we apply our test on bank fraud data and provide a Python library where we implement our results.
    
[^2]: 一种有效且可证明精确的0-1损失线性分类问题算法

    An efficient, provably exact algorithm for the 0-1 loss linear classification problem. (arXiv:2306.12344v1 [cs.LG])

    [http://arxiv.org/abs/2306.12344](http://arxiv.org/abs/2306.12344)

    该研究详细介绍了一种名为增量单元枚举（ICE）的算法，该算法可以精确解决定维度0-1损失线性分类问题。

    

    解决线性分类问题的算法具有悠久的历史，至少可以追溯到1936年的线性判别分析。对于线性可分数据，许多算法可以有效地得到相应的0-1损失分类问题的精确解，但对于非线性可分数据，已经证明这个问题在完全范围内是NP难的。所有替代方法都涉及某种形式的近似，包括使用0-1损失的代理（例如hinge或logistic损失）或近似的组合搜索，这些都不能保证完全解决问题。找到解决定维度0-1损失线性分类问题的全局最优解的有效算法仍然是一个未解决的问题。在本研究中，我们详细介绍了一个新算法的构建过程，增量单元枚举（ICE），它可以精确解决0-1损失分类问题。

    Algorithms for solving the linear classification problem have a long history, dating back at least to 1936 with linear discriminant analysis. For linearly separable data, many algorithms can obtain the exact solution to the corresponding 0-1 loss classification problem efficiently, but for data which is not linearly separable, it has been shown that this problem, in full generality, is NP-hard. Alternative approaches all involve approximations of some kind, including the use of surrogates for the 0-1 loss (for example, the hinge or logistic loss) or approximate combinatorial search, none of which can be guaranteed to solve the problem exactly. Finding efficient algorithms to obtain an exact i.e. globally optimal solution for the 0-1 loss linear classification problem with fixed dimension, remains an open problem. In research we report here, we detail the construction of a new algorithm, incremental cell enumeration (ICE), that can solve the 0-1 loss classification problem exactly in po
    

