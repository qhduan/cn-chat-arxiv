# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy Learning with Distributional Welfare.](http://arxiv.org/abs/2311.15878) | 本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。 |
| [^2] | [The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties.](http://arxiv.org/abs/2304.09310) | 本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。 |

# 详细

[^1]: 分配福利的政策学习

    Policy Learning with Distributional Welfare. (arXiv:2311.15878v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.15878](http://arxiv.org/abs/2311.15878)

    本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。

    

    本文探讨了针对分配福利的最优治疗分配策略。大部分关于治疗选择的文献都考虑了基于条件平均治疗效应（ATE）的功利福利。虽然平均福利是直观的，但在个体异质化（例如，存在离群值）情况下可能会产生不理想的分配 - 这正是个性化治疗引入的原因之一。这个观察让我们提出了一种根据个体治疗效应的条件分位数（QoTE）来分配治疗的最优策略。根据分位数概率的选择，这个准则可以适应谨慎或粗心的决策者。确定QoTE的挑战在于其需要对反事实结果的联合分布有所了解，但即使使用实验数据，通常也很难恢复出来。因此，我们介绍了鲁棒的最小最大化策略

    In this paper, we explore optimal treatment allocation policies that target distributional welfare. Most literature on treatment choice has considered utilitarian welfare based on the conditional average treatment effect (ATE). While average welfare is intuitive, it may yield undesirable allocations especially when individuals are heterogeneous (e.g., with outliers) - the very reason individualized treatments were introduced in the first place. This observation motivates us to propose an optimal policy that allocates the treatment based on the conditional quantile of individual treatment effects (QoTE). Depending on the choice of the quantile probability, this criterion can accommodate a policymaker who is either prudent or negligent. The challenge of identifying the QoTE lies in its requirement for knowledge of the joint distribution of the counterfactual outcomes, which is generally hard to recover even with experimental data. Therefore, we introduce minimax policies that are robust 
    
[^2]: 自适应 $\tau$-Lasso：其健壮性和最优性质。

    The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties. (arXiv:2304.09310v1 [stat.ML])

    [http://arxiv.org/abs/2304.09310](http://arxiv.org/abs/2304.09310)

    本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。

    

    本文介绍了一种用于分析高维数据集的新型正则化鲁棒 $\tau$-回归估计器，以应对响应变量和协变量的严重污染。我们称这种估计器为自适应 $\tau$-Lasso，它对异常值和高杠杆点具有鲁棒性，同时采用自适应 $\ell_1$-范数惩罚项来减少真实回归系数的偏差。具体而言，该自适应 $\ell_1$-范数惩罚项为每个回归系数分配一个权重。对于固定数量的预测变量 $p$，我们显示出自适应 $\tau$-Lasso 具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。然后我们通过有限样本断点和影响函数来表征其健壮性。我们进行了广泛的模拟来比较不同的估计器的性能。

    This paper introduces a new regularized version of the robust $\tau$-regression estimator for analyzing high-dimensional data sets subject to gross contamination in the response variables and covariates. We call the resulting estimator adaptive $\tau$-Lasso that is robust to outliers and high-leverage points and simultaneously employs adaptive $\ell_1$-norm penalty term to reduce the bias associated with large true regression coefficients. More specifically, this adaptive $\ell_1$-norm penalty term assigns a weight to each regression coefficient. For a fixed number of predictors $p$, we show that the adaptive $\tau$-Lasso has the oracle property with respect to variable-selection consistency and asymptotic normality for the regression vector corresponding to the true support, assuming knowledge of the true regression vector support. We then characterize its robustness via the finite-sample breakdown point and the influence function. We carry-out extensive simulations to compare the per
    

