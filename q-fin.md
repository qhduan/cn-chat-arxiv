# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fitted Value Iteration Methods for Bicausal Optimal Transport.](http://arxiv.org/abs/2306.12658) | 本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。 |

# 详细

[^1]: 拟合值迭代方法求解适应结构双因果最优传输问题

    Fitted Value Iteration Methods for Bicausal Optimal Transport. (arXiv:2306.12658v1 [stat.ML])

    [http://arxiv.org/abs/2306.12658](http://arxiv.org/abs/2306.12658)

    本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。

    

    本文提出一种拟合值迭代方法(FVI)用于计算具有适应结构的双因果最优传输(OT)。基于动态规划的形式化表述，FVI采用函数类用于近似双因果OT中的值函数。在可集中条件和近似完备性假设下，我们使用（局部）Rademacher复杂度证明了样本复杂度。此外，我们证明了深度多层神经网络具有适当结构，满足样本复杂度证明所需的关键假设条件。数值实验表明，FVI在时间跨度增加时优于线性规划和适应性Sinkhorn方法，在保持可接受精度的同时具有很好的可扩展性。

    We develop a fitted value iteration (FVI) method to compute bicausal optimal transport (OT) where couplings have an adapted structure. Based on the dynamic programming formulation, FVI adopts a function class to approximate the value functions in bicausal OT. Under the concentrability condition and approximate completeness assumption, we prove the sample complexity using (local) Rademacher complexity. Furthermore, we demonstrate that multilayer neural networks with appropriate structures satisfy the crucial assumptions required in sample complexity proofs. Numerical experiments reveal that FVI outperforms linear programming and adapted Sinkhorn methods in scalability as the time horizon increases, while still maintaining acceptable accuracy.
    

