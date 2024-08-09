# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Generalization and Regularization of Nonhomogeneous Temporal Poisson Processes](https://arxiv.org/abs/2402.12808) | 将NHPPs的估计问题转化为学习泛化问题，提出了正则化学习NHPPs的框架与两种新的自适应和数据驱动的分箱方法，有效解决了数据量有限时过拟合的问题。 |
| [^2] | [Graph Matching via convex relaxation to the simplex.](http://arxiv.org/abs/2310.20609) | 本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。 |
| [^3] | [Prediction Error Estimation in Random Forests.](http://arxiv.org/abs/2309.00736) | 本文通过量化评估分类随机森林的误差估计方法，发现随机森林的预测误差估计比平均预测误差更接近真实误差率，并且这一结果适用于不同的误差估计策略。 |
| [^4] | [Distributionally robust risk evaluation with a causality constraint and structural information.](http://arxiv.org/abs/2203.10571) | 本文提出了具有因果约束的分布鲁棒风险评估方法，并用神经网络逼近测试函数。在结构信息有限制时，提供了高效的优化方法。 |

# 详细

[^1]: 学习非齐次时间泊松过程的泛化和正则化

    Learning Generalization and Regularization of Nonhomogeneous Temporal Poisson Processes

    [https://arxiv.org/abs/2402.12808](https://arxiv.org/abs/2402.12808)

    将NHPPs的估计问题转化为学习泛化问题，提出了正则化学习NHPPs的框架与两种新的自适应和数据驱动的分箱方法，有效解决了数据量有限时过拟合的问题。

    

    泊松过程，尤其是非齐次泊松过程(NHPP)，是一种在许多实际应用中非常重要的计数过程。目前，文献中几乎所有的工作都致力于使用非数据驱动的分箱方法对具有无穷数据的NHPP进行估计。本文将有限和有限数据下的NHPP估计问题公式化为一个学习泛化问题。我们在数学上证明，尽管分箱方法对于估计NHPPs很重要，但在数据量有限时会带来过拟合的风险。我们提出了一个正则化学习NHPPs的框架，其中包括两种新的自适应和数据驱动的分箱方法，帮助消除分箱参数的即兴调整。我们在合成和实际数据集上对我们的方法进行了实验证明了其有效性。

    arXiv:2402.12808v1 Announce Type: new  Abstract: The Poisson process, especially the nonhomogeneous Poisson process (NHPP), is an essentially important counting process with numerous real-world applications. Up to date, almost all works in the literature have been on the estimation of NHPPs with infinite data using non-data driven binning methods. In this paper, we formulate the problem of estimation of NHPPs from finite and limited data as a learning generalization problem. We mathematically show that while binning methods are essential for the estimation of NHPPs, they pose a threat of overfitting when the amount of data is limited. We propose a framework for regularized learning of NHPPs with two new adaptive and data-driven binning methods that help to remove the ad-hoc tuning of binning parameters. Our methods are experimentally tested on synthetic and real-world datasets and the results show their effectiveness.
    
[^2]: 通过对单纯形进行凸松弛解决图匹配问题

    Graph Matching via convex relaxation to the simplex. (arXiv:2310.20609v1 [stat.ML])

    [http://arxiv.org/abs/2310.20609](http://arxiv.org/abs/2310.20609)

    本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。

    

    本文针对图匹配问题进行研究，该问题包括在两个输入图之间找到最佳对齐，并在计算机视觉、网络去匿名化和蛋白质对齐等领域有许多应用。解决这个问题的常见方法是通过对NP难问题“二次分配问题”（QAP）进行凸松弛。本文引入了一种新的凸松弛方法，即对单位单纯形进行松弛，并开发了一种具有闭合迭代形式的高效镜像下降方案来解决该问题。在相关高斯Wigner模型下，我们证明了单纯形松弛法在高概率下具有唯一解。在无噪声情况下，这被证明可以精确恢复地面真实排列。此外，我们建立了一种新的输入矩阵假设条件，用于标准贪心取整方法，并且这个条件比常用的“对角线优势”条件更宽松。我们使用这个条件证明了地面真实排列的精确一步恢复。

    This paper addresses the Graph Matching problem, which consists of finding the best possible alignment between two input graphs, and has many applications in computer vision, network deanonymization and protein alignment. A common approach to tackle this problem is through convex relaxations of the NP-hard \emph{Quadratic Assignment Problem} (QAP).  Here, we introduce a new convex relaxation onto the unit simplex and develop an efficient mirror descent scheme with closed-form iterations for solving this problem. Under the correlated Gaussian Wigner model, we show that the simplex relaxation admits a unique solution with high probability. In the noiseless case, this is shown to imply exact recovery of the ground truth permutation. Additionally, we establish a novel sufficiency condition for the input matrix in standard greedy rounding methods, which is less restrictive than the commonly used `diagonal dominance' condition. We use this condition to show exact one-step recovery of the gro
    
[^3]: 随机森林中的预测误差估计

    Prediction Error Estimation in Random Forests. (arXiv:2309.00736v1 [stat.ML])

    [http://arxiv.org/abs/2309.00736](http://arxiv.org/abs/2309.00736)

    本文通过量化评估分类随机森林的误差估计方法，发现随机森林的预测误差估计比平均预测误差更接近真实误差率，并且这一结果适用于不同的误差估计策略。

    

    本文定量评估了分类随机森林的误差估计。在Bates等人（2023年）建立的初步理论框架的基础上，从理论和经验角度探讨了随机森林中常见的各种误差估计方法在真实误差率和期望误差率方面的情况。我们发现，在分类情况下，随机森林的预测误差估计平均更接近真实误差率，而不是平均预测误差。与Bates等人（2023年）对逻辑回归的研究结果相反。我们进一步证明，这个结果适用于交叉验证、自举和数据划分等不同的误差估计策略。

    In this paper, error estimates of classification Random Forests are quantitatively assessed. Based on the initial theoretical framework built by Bates et al. (2023), the true error rate and expected error rate are theoretically and empirically investigated in the context of a variety of error estimation methods common to Random Forests. We show that in the classification case, Random Forests' estimates of prediction error is closer on average to the true error rate instead of the average prediction error. This is opposite the findings of Bates et al. (2023) which were given for logistic regression. We further show that this result holds across different error estimation strategies such as cross-validation, bagging, and data splitting.
    
[^4]: 具有因果约束和结构信息的分布鲁棒风险评估

    Distributionally robust risk evaluation with a causality constraint and structural information. (arXiv:2203.10571v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2203.10571](http://arxiv.org/abs/2203.10571)

    本文提出了具有因果约束的分布鲁棒风险评估方法，并用神经网络逼近测试函数。在结构信息有限制时，提供了高效的优化方法。

    

    本文研究了基于时间数据的期望函数值的分布鲁棒评估。一组备选度量通过因果最优传输进行表征。我们证明了强对偶性，并将因果约束重构为无限维测试函数空间的最小化问题。我们通过神经网络逼近测试函数，并用Rademacher复杂度证明了样本复杂度。此外，当结构信息可用于进一步限制模糊集时，我们证明了对偶形式并提供高效的优化方法。对实现波动率和股指的经验分析表明，我们的框架为经典最优传输公式提供了一种有吸引力的替代方案。

    This work studies distributionally robust evaluation of expected function values over temporal data. A set of alternative measures is characterized by the causal optimal transport. We prove the strong duality and recast the causality constraint as minimization over an infinite-dimensional test function space. We approximate test functions by neural networks and prove the sample complexity with Rademacher complexity. Moreover, when structural information is available to further restrict the ambiguity set, we prove the dual formulation and provide efficient optimization methods. Empirical analysis of realized volatility and stock indices demonstrates that our framework offers an attractive alternative to the classic optimal transport formulation.
    

