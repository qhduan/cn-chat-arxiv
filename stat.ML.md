# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variable Selection in Maximum Mean Discrepancy for Interpretable Distribution Comparison.](http://arxiv.org/abs/2311.01537) | 本文研究了数据集比较中的变量选择问题，提出了一种基于最大平均差异的两样本测试方法，通过优化自动相关性检测权重来增强测试的功效，并引入稀疏正则化方法来解决正则化参数选择的问题。 |
| [^2] | [Neural networks learn to magnify areas near decision boundaries.](http://arxiv.org/abs/2301.11375) | 神经网络训练能够放大决策边界附近的局部区域，改善整个系统的泛化能力。 |
| [^3] | [Uniform-in-time propagation of chaos for mean field Langevin dynamics.](http://arxiv.org/abs/2212.03050) | 研究了平均场 Langevin 动力学，证明了边缘分布 $L^p$ 收敛性和混沌现象的均匀时间传播。 |

# 详细

[^1]: 在可解释的分布比较中的最大平均差异中的变量选择

    Variable Selection in Maximum Mean Discrepancy for Interpretable Distribution Comparison. (arXiv:2311.01537v1 [stat.ML])

    [http://arxiv.org/abs/2311.01537](http://arxiv.org/abs/2311.01537)

    本文研究了数据集比较中的变量选择问题，提出了一种基于最大平均差异的两样本测试方法，通过优化自动相关性检测权重来增强测试的功效，并引入稀疏正则化方法来解决正则化参数选择的问题。

    

    两样本测试是为了判断两个数据集是否来自同一分布。本文研究了两样本测试中的变量选择问题，即识别造成两个分布差异的变量（或维度）的任务。这个任务与模式分析和机器学习的许多问题相关，如数据集漂移适应、因果推断和模型验证。我们的方法基于基于最大平均差异（MMD）的两样本检验。我们优化针对各个变量定义的自动相关性检测（ARD）权重，以最大化基于MMD的检验的功率。对于这种优化，我们引入了稀疏正则化，并提出了两种方法来解决选择适当正则化参数的问题。一种方法是以数据驱动的方式确定正则化参数，另一种方法是合并不同正则化参数的结果。我们确认了这个方法的有效性。

    Two-sample testing decides whether two datasets are generated from the same distribution. This paper studies variable selection for two-sample testing, the task being to identify the variables (or dimensions) responsible for the discrepancies between the two distributions. This task is relevant to many problems of pattern analysis and machine learning, such as dataset shift adaptation, causal inference and model validation. Our approach is based on a two-sample test based on the Maximum Mean Discrepancy (MMD). We optimise the Automatic Relevance Detection (ARD) weights defined for individual variables to maximise the power of the MMD-based test. For this optimisation, we introduce sparse regularisation and propose two methods for dealing with the issue of selecting an appropriate regularisation parameter. One method determines the regularisation parameter in a data-driven way, and the other aggregates the results of different regularisation parameters. We confirm the validity of the pr
    
[^2]: 神经网络学习放大决策边界附近的区域

    Neural networks learn to magnify areas near decision boundaries. (arXiv:2301.11375v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11375](http://arxiv.org/abs/2301.11375)

    神经网络训练能够放大决策边界附近的局部区域，改善整个系统的泛化能力。

    

    我们研究了训练如何塑造神经网络特征图诱导的黎曼几何。在宽度为无限的情况下，具有随机参数的神经网络在输入空间上引导高度对称的度量。训练分类任务的网络中的特征学习放大了沿决策边界的局部区域。这些变化与先前提出的用于手动调整核方法以改善泛化的几何方法一致。

    We study how training molds the Riemannian geometry induced by neural network feature maps. At infinite width, neural networks with random parameters induce highly symmetric metrics on input space. Feature learning in networks trained to perform classification tasks magnifies local areas along decision boundaries. These changes are consistent with previously proposed geometric approaches for hand-tuning of kernel methods to improve generalization.
    
[^3]: 均匀时间传播混沌的平均场 Langevin 动力学

    Uniform-in-time propagation of chaos for mean field Langevin dynamics. (arXiv:2212.03050v2 [math.PR] UPDATED)

    [http://arxiv.org/abs/2212.03050](http://arxiv.org/abs/2212.03050)

    研究了平均场 Langevin 动力学，证明了边缘分布 $L^p$ 收敛性和混沌现象的均匀时间传播。

    

    本文研究了平均场 Langevin 动力学及其相关粒子系统。通过假设能量函数的凸性，我们得出了边缘分布收敛到平均场动力学唯一不变测度的 $L^p$ 收敛性。此外，我们证明了在 $L^2$ Wasserstein 距离和相对熵两方面，混沌现象的均匀时间传播。

    We study the mean field Langevin dynamics and the associated particle system. By assuming the functional convexity of the energy, we obtain the $L^p$-convergence of the marginal distributions towards the unique invariant measure for the mean field dynamics. Furthermore, we prove the uniform-in-time propagation of chaos in both the $L^2$-Wasserstein metric and relative entropy.
    

