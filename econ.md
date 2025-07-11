# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Rank-Rank Regressions.](http://arxiv.org/abs/2310.15512) | 本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。 |
| [^2] | [Convexity Not Required: Estimation of Smooth Moment Condition Models.](http://arxiv.org/abs/2304.14386) | 本文针对平滑问题，使用全局秩条件下的特定算法证明了非凸结构下的通用矩估计和模拟矩估计是全局收敛的。 |

# 详细

[^1]: 推论用于等级回归

    Inference for Rank-Rank Regressions. (arXiv:2310.15512v1 [econ.EM])

    [http://arxiv.org/abs/2310.15512](http://arxiv.org/abs/2310.15512)

    本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。

    

    在等级回归中，斜率系数是衡量代际流动性的常用指标，例如在子女收入等级与父母收入等级回归中。本文首先指出，常用的方差估计器如同方差估计器或鲁棒方差估计器未能一致估计OLS估计器在等级回归中的渐进方差。我们表明，这些估计器的概率极限可能过大或过小，取决于子女收入和父母收入的联合分布函数的形状。其次，我们导出了等级回归的一般渐进理论，并提供了OLS估计器渐进方差的一致估计器。然后，我们将渐进理论扩展到其他经验工作中涉及等级的回归。最后，我们将新的推论方法应用于三个经验研究。我们发现，基于正确方差的估计器的置信区间有时可能欠精确。

    Slope coefficients in rank-rank regressions are popular measures of intergenerational mobility, for instance in regressions of a child's income rank on their parent's income rank. In this paper, we first point out that commonly used variance estimators such as the homoskedastic or robust variance estimators do not consistently estimate the asymptotic variance of the OLS estimator in a rank-rank regression. We show that the probability limits of these estimators may be too large or too small depending on the shape of the copula of child and parent incomes. Second, we derive a general asymptotic theory for rank-rank regressions and provide a consistent estimator of the OLS estimator's asymptotic variance. We then extend the asymptotic theory to other regressions involving ranks that have been used in empirical work. Finally, we apply our new inference methods to three empirical studies. We find that the confidence intervals based on estimators of the correct variance may sometimes be sub
    
[^2]: 无需凸性：平滑时刻条件模型的估计方法

    Convexity Not Required: Estimation of Smooth Moment Condition Models. (arXiv:2304.14386v1 [econ.EM])

    [http://arxiv.org/abs/2304.14386](http://arxiv.org/abs/2304.14386)

    本文针对平滑问题，使用全局秩条件下的特定算法证明了非凸结构下的通用矩估计和模拟矩估计是全局收敛的。

    

    广义矩及模拟矩法常用于估计结构性经济模型。然而，由于相应的目标函数非凸，优化往往具有挑战性。本文针对平滑问题，证明了不需要凸性：在涉及样本矩阵雅可比的全局秩条件下，某些算法是全局收敛的。这些算法包括梯度下降和高斯牛顿算法及需要适当选择调整参数。该结果具有鲁棒性，对非凸性、单一非线性再参数化以及适度的错误建模具有鲁棒性。相比之下，由于非凸性，牛顿-拉夫森和拟牛顿方法可能无法收敛。简单的例子说明了一个满足前述秩条件的非凸 GMM 估计问题。对随机系数需求估计和脉冲响应匹配的实证应用进一步说明了本文的研究方法。

    Generalized and Simulated Method of Moments are often used to estimate structural Economic models. Yet, it is commonly reported that optimization is challenging because the corresponding objective function is non-convex. For smooth problems, this paper shows that convexity is not required: under a global rank condition involving the Jacobian of the sample moments, certain algorithms are globally convergent. These include a gradient-descent and a Gauss-Newton algorithm with appropriate choice of tuning parameters. The results are robust to 1) non-convexity, 2) one-to-one non-linear reparameterizations, and 3) moderate misspecification. In contrast, Newton-Raphson and quasi-Newton methods can fail to converge for the same estimation because of non-convexity. A simple example illustrates a non-convex GMM estimation problem that satisfies the aforementioned rank condition. Empirical applications to random coefficient demand estimation and impulse response matching further illustrate the re
    

