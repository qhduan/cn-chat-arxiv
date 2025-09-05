# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bootstrapping the Cross-Validation Estimate.](http://arxiv.org/abs/2307.00260) | 本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。 |

# 详细

[^1]: 基于自助法的交叉验证估计

    Bootstrapping the Cross-Validation Estimate. (arXiv:2307.00260v1 [stat.ME])

    [http://arxiv.org/abs/2307.00260](http://arxiv.org/abs/2307.00260)

    本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。

    

    交叉验证是一种广泛应用于评估预测模型性能的技术。它可以避免对错误估计中的乐观偏差，尤其对于使用复杂统计学习算法构建的模型。然而，由于交叉验证估计是依赖于观测数据的随机值，因此准确量化估计的不确定性非常重要。特别是当使用交叉验证比较两个模型的性能时，必须确定错误估计的差异是否是由于偶然波动。尽管已经发展了各种方法来对交叉验证估计进行推断，但它们往往有许多限制，如严格的模型假设。本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。

    Cross-validation is a widely used technique for evaluating the performance of prediction models. It helps avoid the optimism bias in error estimates, which can be significant for models built using complex statistical learning algorithms. However, since the cross-validation estimate is a random value dependent on observed data, it is essential to accurately quantify the uncertainty associated with the estimate. This is especially important when comparing the performance of two models using cross-validation, as one must determine whether differences in error estimates are a result of chance fluctuations. Although various methods have been developed for making inferences on cross-validation estimates, they often have many limitations, such as stringent model assumptions This paper proposes a fast bootstrap method that quickly estimates the standard error of the cross-validation estimate and produces valid confidence intervals for a population parameter measuring average model performance
    

