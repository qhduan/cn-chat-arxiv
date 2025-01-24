# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models.](http://arxiv.org/abs/2310.06715) | 本研究解析了基于深度学习的睡眠阶段分类模型的设计空间，找到了适用于不同输入表示的稳健架构，并在睡眠数据集上实现了显著的性能提升。 |
| [^3] | [Spectral Regularized Kernel Goodness-of-Fit Tests.](http://arxiv.org/abs/2308.04561) | 本文提出了具有谱正则化的核拟合优度检验方法，用于处理非欧几里得数据。相比之前的方法，本方法在选择适当的正则化参数时能达到最小化最大风险。同时，本方法还克服了之前方法对均值元素为零和积分操作符特征函数均匀有界性条件的限制，并且能够计算更多种类的核函数。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: S4Sleep: 解析基于深度学习的睡眠阶段分类模型的设计空间

    S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models. (arXiv:2310.06715v1 [cs.LG])

    [http://arxiv.org/abs/2310.06715](http://arxiv.org/abs/2310.06715)

    本研究解析了基于深度学习的睡眠阶段分类模型的设计空间，找到了适用于不同输入表示的稳健架构，并在睡眠数据集上实现了显著的性能提升。

    

    对于多通道睡眠脑电图记录进行睡眠阶段打分是一项耗时且存在显著的评分人员之间差异的任务。因此，应用机器学习算法可以带来很大的益处。虽然已经为此提出了许多算法，但某些关键的架构决策并未得到系统性的探索。在本研究中，我们详细调查了广泛的编码器-预测器架构范畴内的这些设计选择。我们找到了适用于时间序列和声谱图输入表示的稳健架构。这些架构将结构化状态空间模型作为组成部分，对广泛的SHHS数据集的性能进行了统计显著的提升。这些改进通过统计和系统误差估计进行了评估。我们预计，从本研究中获得的架构洞察不仅对未来的睡眠分期研究有价值，而且对整体睡眠研究都有价值。

    Scoring sleep stages in polysomnography recordings is a time-consuming task plagued by significant inter-rater variability. Therefore, it stands to benefit from the application of machine learning algorithms. While many algorithms have been proposed for this purpose, certain critical architectural decisions have not received systematic exploration. In this study, we meticulously investigate these design choices within the broad category of encoder-predictor architectures. We identify robust architectures applicable to both time series and spectrogram input representations. These architectures incorporate structured state space models as integral components, leading to statistically significant advancements in performance on the extensive SHHS dataset. These improvements are assessed through both statistical and systematic error estimations. We anticipate that the architectural insights gained from this study will not only prove valuable for future research in sleep staging but also hol
    
[^3]: 具有谱正则化的核拟合优度检验

    Spectral Regularized Kernel Goodness-of-Fit Tests. (arXiv:2308.04561v1 [math.ST])

    [http://arxiv.org/abs/2308.04561](http://arxiv.org/abs/2308.04561)

    本文提出了具有谱正则化的核拟合优度检验方法，用于处理非欧几里得数据。相比之前的方法，本方法在选择适当的正则化参数时能达到最小化最大风险。同时，本方法还克服了之前方法对均值元素为零和积分操作符特征函数均匀有界性条件的限制，并且能够计算更多种类的核函数。

    

    在许多机器学习和统计应用中，最大均值差异(MMD)因其处理非欧几里得数据的能力而获得了很多成功，包括非参数假设检验。最近，Balasubramanian等人(2021)通过实验证明，基于MMD的拟合优度检验在适当选择正则化参数时，并不是最小化最大风险，而其Tikhonov正则化版本则是最小化最大风险的。然而，Balasubramanian等人(2021)的结果是在均值元素为零的限制性假设和积分操作符特征函数的均匀有界性条件下获得的。此外，Balasubramanian等人(2021)提出的检验在许多核函数中是不可计算的，因此不实用。本文解决了这些问题，并将结果推广到包括Tikhonov正则化在内的一般谱正则化方法中。

    Maximum mean discrepancy (MMD) has enjoyed a lot of success in many machine learning and statistical applications, including non-parametric hypothesis testing, because of its ability to handle non-Euclidean data. Recently, it has been demonstrated in Balasubramanian et al.(2021) that the goodness-of-fit test based on MMD is not minimax optimal while a Tikhonov regularized version of it is, for an appropriate choice of the regularization parameter. However, the results in Balasubramanian et al. (2021) are obtained under the restrictive assumptions of the mean element being zero, and the uniform boundedness condition on the eigenfunctions of the integral operator. Moreover, the test proposed in Balasubramanian et al. (2021) is not practical as it is not computable for many kernels. In this paper, we address these shortcomings and extend the results to general spectral regularizers that include Tikhonov regularization.
    

