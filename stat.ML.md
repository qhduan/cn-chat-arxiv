# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Precise Error Rates for Computationally Efficient Testing.](http://arxiv.org/abs/2311.00289) | 在高维设置中，我们提出了一种基于线性谱统计的测试方法，该方法在计算上非常高效，并且在所有计算上高效的测试中，实现了类型 I 和类型 II 错误率之间的最佳权衡曲线。 |
| [^2] | [Wasserstein Gradient Flow over Variational Parameter Space for Variational Inference.](http://arxiv.org/abs/2310.16705) | 本文将变分推断重新框架为在变分参数空间上的概率分布优化问题，提出了沃瑟斯坦梯度下降方法来解决优化问题，有效性经过实证实验证实。 |
| [^3] | [Optimal Estimation in Mixed-Membership Stochastic Block Models.](http://arxiv.org/abs/2307.14530) | 本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。 |

# 详细

[^1]: 高效测试的精确错误率

    Precise Error Rates for Computationally Efficient Testing. (arXiv:2311.00289v1 [math.ST])

    [http://arxiv.org/abs/2311.00289](http://arxiv.org/abs/2311.00289)

    在高维设置中，我们提出了一种基于线性谱统计的测试方法，该方法在计算上非常高效，并且在所有计算上高效的测试中，实现了类型 I 和类型 II 错误率之间的最佳权衡曲线。

    

    我们重新审视了简单与简单假设检验的基本问题，特别关注计算复杂度，因为在高维设置中，统计上最优的似然比检验通常是计算上难以处理的。在经典的尖峰维格纳模型（具有一般性 i.i.d. 尖峰先验）中，我们展示了一个基于线性谱统计的现有测试实现了在计算上高效测试之间的最佳权衡曲线，即使存在更好的指数时间测试。这个结果是在一个适当复杂性理论的猜想条件下得到的，即一个自然加强已经建立的低次数猜想。我们的结果表明，谱是计算受限的测试的充分统计量（但不是所有测试的充分统计量）。据我们所知，我们的方法提供了首个用于推理关于有效计算所能实现的精确渐近测试误差的工具。

    We revisit the fundamental question of simple-versus-simple hypothesis testing with an eye towards computational complexity, as the statistically optimal likelihood ratio test is often computationally intractable in high-dimensional settings. In the classical spiked Wigner model (with a general i.i.d. spike prior) we show that an existing test based on linear spectral statistics achieves the best possible tradeoff curve between type I and type II error rates among all computationally efficient tests, even though there are exponential-time tests that do better. This result is conditional on an appropriate complexity-theoretic conjecture, namely a natural strengthening of the well-established low-degree conjecture. Our result shows that the spectrum is a sufficient statistic for computationally bounded tests (but not for all tests).  To our knowledge, our approach gives the first tool for reasoning about the precise asymptotic testing error achievable with efficient computation. The main
    
[^2]: 沃瑟斯坦梯度流在变分推断的变分参数空间上的应用

    Wasserstein Gradient Flow over Variational Parameter Space for Variational Inference. (arXiv:2310.16705v1 [cs.LG])

    [http://arxiv.org/abs/2310.16705](http://arxiv.org/abs/2310.16705)

    本文将变分推断重新框架为在变分参数空间上的概率分布优化问题，提出了沃瑟斯坦梯度下降方法来解决优化问题，有效性经过实证实验证实。

    

    变分推断可以被看作是一个优化问题，其中变分参数被调整以使变分分布与真实后验尽可能接近。可以通过黑箱变分推断中的普通梯度下降或自然梯度变分推断中的自然梯度下降来解决优化任务。在本文中，我们将变分推断重新框架为在一个“变分参数空间”中定义的概率分布的目标优化问题。随后，我们提出了沃瑟斯坦梯度下降方法来解决这个优化问题。值得注意的是，这些优化技术，即黑箱变分推断和自然梯度变分推断，可以重新解释为所提出的沃瑟斯坦梯度下降的特定实例。为了提高优化效率，我们开发了实用的方法来数值求解离散梯度流。通过在一个合成数据集上的实证实验，我们验证了所提出方法的有效性。

    Variational inference (VI) can be cast as an optimization problem in which the variational parameters are tuned to closely align a variational distribution with the true posterior. The optimization task can be approached through vanilla gradient descent in black-box VI or natural-gradient descent in natural-gradient VI. In this work, we reframe VI as the optimization of an objective that concerns probability distributions defined over a \textit{variational parameter space}. Subsequently, we propose Wasserstein gradient descent for tackling this optimization problem. Notably, the optimization techniques, namely black-box VI and natural-gradient VI, can be reinterpreted as specific instances of the proposed Wasserstein gradient descent. To enhance the efficiency of optimization, we develop practical methods for numerically solving the discrete gradient flows. We validate the effectiveness of the proposed methods through empirical experiments on a synthetic dataset, supplemented by theore
    
[^3]: 混合成员随机块模型中的最优估计

    Optimal Estimation in Mixed-Membership Stochastic Block Models. (arXiv:2307.14530v1 [stat.ML])

    [http://arxiv.org/abs/2307.14530](http://arxiv.org/abs/2307.14530)

    本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。

    

    社区检测是现代网络科学中最关键的问题之一。其应用可以在各个领域找到，从蛋白质建模到社交网络分析。最近，出现了许多论文研究重叠社区检测问题，即网络中的每个节点可能属于多个社区。在本文中，我们考虑了由Airoldi等人（2008）首次提出的混合成员随机块模型（MMSB）。MMSB在图中对重叠社区结构提供了相当一般的设置。本文的核心问题是在观察到的网络中重建社区之间的关系。我们比较了不同的方法，并建立了估计误差的极小下界。然后，我们提出了一个与这个下界匹配的新估计器。理论结果在对所考虑的模型的相当普遍条件下得到证明。最后，我们通过一系列实验来说明这个理论。

    Community detection is one of the most critical problems in modern network science. Its applications can be found in various fields, from protein modeling to social network analysis. Recently, many papers appeared studying the problem of overlapping community detection, where each node of a network may belong to several communities. In this work, we consider Mixed-Membership Stochastic Block Model (MMSB) first proposed by Airoldi et al. (2008). MMSB provides quite a general setting for modeling overlapping community structure in graphs. The central question of this paper is to reconstruct relations between communities given an observed network. We compare different approaches and establish the minimax lower bound on the estimation error. Then, we propose a new estimator that matches this lower bound. Theoretical results are proved under fairly general conditions on the considered model. Finally, we illustrate the theory in a series of experiments.
    

