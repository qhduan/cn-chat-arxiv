# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Matching Control Method.](http://arxiv.org/abs/2306.02584) | 本文提出了一种叫做合成匹配控制方法的简单而有效的方法来解决合成控制方法中的选择对齐问题和欠拟合问题，并通过对比实验表明该方法有效地提高了因果效应的估计精度。 |

# 详细

[^1]: 合成匹配控制方法

    Synthetic Matching Control Method. (arXiv:2306.02584v1 [econ.EM])

    [http://arxiv.org/abs/2306.02584](http://arxiv.org/abs/2306.02584)

    本文提出了一种叫做合成匹配控制方法的简单而有效的方法来解决合成控制方法中的选择对齐问题和欠拟合问题，并通过对比实验表明该方法有效地提高了因果效应的估计精度。

    

    在合成控制方法中，估计权重涉及同时选择和对齐对照组以接近匹配处理组的优化过程。然而，这种同时选择和对齐控制组单位的方式可能导致合成控制方法效率的下降。另一个担忧是由于不完美的预处理拟合而导致的欠拟合问题。为了解决这两个问题，本文提出了一种简单有效的方法——合成匹配控制方法。该方法通过执行单变量线性回归来建立控制组的预处理期与处理组之间的适当匹配，并通过合成匹配来获得一个SMC估计量，从而改善了估计的效率和拟合度。

    Estimating weights in the synthetic control method involves an optimization procedure that simultaneously selects and aligns control units in order to closely match the treated unit. However, this simultaneous selection and alignment of control units may lead to a loss of efficiency in the synthetic control method. Another concern arising from the aforementioned procedure is its susceptibility to under-fitting due to imperfect pretreatment fit. It is not uncommon for the linear combination, using nonnegative weights, of pre-treatment period outcomes for the control units to inadequately approximate the pre-treatment outcomes for the treated unit. To address both of these issues, this paper proposes a simple and effective method called Synthetic Matching Control (SMC). The SMC method begins by performing the univariate linear regression to establish a proper match between the pre-treatment periods of the control units and the treated unit. Subsequently, a SMC estimator is obtained by sy
    

