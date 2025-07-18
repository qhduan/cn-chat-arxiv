# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrated path stability selection](https://arxiv.org/abs/2403.15877) | 该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。 |
| [^2] | [Boosting for Bounding the Worst-class Error.](http://arxiv.org/abs/2310.14890) | 该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。 |
| [^3] | [Conformal inference for regression on Riemannian Manifolds.](http://arxiv.org/abs/2310.08209) | 本文研究了在黎曼流形上进行回归场景的预测集，并证明了这些区域的经验版本在大样本下的收敛性。 |

# 详细

[^1]: 集成路径稳定选择

    Integrated path stability selection

    [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

    该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。

    

    稳定选择是一种广泛用于改善特征选择算法性能的方法。然而，已发现稳定选择过于保守，导致灵敏度较低。此外，对期望的假阳性数量的理论界限E(FP)相对较松，难以知道实践中会有多少假阳性。在本文中，我们提出一种基于集成稳定路径而非最大化稳定路径的新方法。这产生了对E(FP)更紧密的界限，导致实践中具有更高灵敏度的特征选择标准，并且在与目标E(FP)匹配方面更好地校准。我们提出的方法与原始稳定选择算法需要相同数量的计算，且仅需要用户指定一个输入参数，即E(FP)的目标值。我们提供了性能的理论界限。

    arXiv:2403.15877v1 Announce Type: cross  Abstract: Stability selection is a widely used method for improving the performance of feature selection algorithms. However, stability selection has been found to be highly conservative, resulting in low sensitivity. Further, the theoretical bound on the expected number of false positives, E(FP), is relatively loose, making it difficult to know how many false positives to expect in practice. In this paper, we introduce a novel method for stability selection based on integrating the stability paths rather than maximizing over them. This yields a tighter bound on E(FP), resulting in a feature selection criterion that has higher sensitivity in practice and is better calibrated in terms of matching the target E(FP). Our proposed method requires the same amount of computation as the original stability selection algorithm, and only requires the user to specify one input parameter, a target value for E(FP). We provide theoretical bounds on performance
    
[^2]: Boosting用于界定最差分类误差

    Boosting for Bounding the Worst-class Error. (arXiv:2310.14890v1 [stat.ML])

    [http://arxiv.org/abs/2310.14890](http://arxiv.org/abs/2310.14890)

    该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。

    

    本文解决了最差类别误差率的问题，而不是针对所有类别的标准误差率的平均。例如，一个三类别分类任务，其中各类别的误差率分别为10％，10％和40％，其最差类别误差率为40％，而在类别平衡条件下的平均误差率为20％。最差类别错误在许多应用中很重要。例如，在医学图像分类任务中，对于恶性肿瘤类别具有40％的错误率而良性和健康类别具有10％的错误率是不能被接受的。我们提出了一种保证最差类别训练误差上界的提升算法，并推导出其泛化界。实验结果表明，该算法降低了最差类别的测试误差率，同时避免了对训练集的过拟合。

    This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10\%, 10\%, and 40\% has a worst-class error rate of 40\%, whereas the average is 20\% under the class-balanced condition. The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have 10\% error rates.We propose a boosting algorithm that guarantees an upper bound of the worst-class training error and derive its generalization bound. Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.
    
[^3]: 在黎曼流形上进行回归的一致推断

    Conformal inference for regression on Riemannian Manifolds. (arXiv:2310.08209v1 [stat.ML])

    [http://arxiv.org/abs/2310.08209](http://arxiv.org/abs/2310.08209)

    本文研究了在黎曼流形上进行回归场景的预测集，并证明了这些区域的经验版本在大样本下的收敛性。

    

    在流形上进行回归，以及更广泛地说，对流形上的统计学有了重要的关注，因为这种类型的数据有大量的应用。圆形数据是一个经典示例，但协方差矩阵空间上的数据、主成分分析得到的Grassmann流形上的数据等也是如此。在本文中，我们研究了当响应变量$Y$位于流形上，而协变量$X$位于欧几里德空间时，回归场景的预测集。这扩展了[Lei and Wasserman, 2014]中在这一新领域中概述的概念。与一致推断中的传统原则一致，这些预测集是无分布的，表明对$(X, Y)$的联合分布没有施加特定的假设，而且它们保持非参数性质。我们证明了这些区域的经验版本在几乎必然收敛于无穷大时的收敛性。

    Regression on manifolds, and, more broadly, statistics on manifolds, has garnered significant importance in recent years due to the vast number of applications for this type of data. Circular data is a classic example, but so is data in the space of covariance matrices, data on the Grassmannian manifold obtained as a result of principal component analysis, among many others. In this work we investigate prediction sets for regression scenarios when the response variable, denoted by $Y$, resides in a manifold, and the covariable, denoted by X, lies in Euclidean space. This extends the concepts delineated in [Lei and Wasserman, 2014] to this novel context. Aligning with traditional principles in conformal inference, these prediction sets are distribution-free, indicating that no specific assumptions are imposed on the joint distribution of $(X, Y)$, and they maintain a non-parametric character. We prove the asymptotic almost sure convergence of the empirical version of these regions on th
    

