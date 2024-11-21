# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A mean-field model of optimal investment](https://arxiv.org/abs/2404.02871) | 该论文建立了一个关于最优投资的随机均场博弈模型，证明了均衡的存在性和唯一性，探讨了有限和无限时间范围的情况，同时研究了确定性对应物。 |
| [^2] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^3] | [Adjustment with Many Regressors Under Covariate-Adaptive Randomizations.](http://arxiv.org/abs/2304.08184) | 本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。 |

# 详细

[^1]: 一个关于最优投资的均场模型

    A mean-field model of optimal investment

    [https://arxiv.org/abs/2404.02871](https://arxiv.org/abs/2404.02871)

    该论文建立了一个关于最优投资的随机均场博弈模型，证明了均衡的存在性和唯一性，探讨了有限和无限时间范围的情况，同时研究了确定性对应物。

    

    我们建立了一个关于最优投资的随机均场博弈的均衡存在性和唯一性。分析涵盖了有限和无限时间范围，代表公司与大量相同和不可区分的公司群体之间的均场相互作用通过生产商品的售价进行建模。在均衡状态下，这个价格以代表公司的预期(最优控制)生产能力的非线性函数来表示。均场均衡存在性和唯一性的证明依赖于先验估计和非线性积分方程的研究，但对于有限和无限时间范围的情况采用了不同的技术。此外，我们还研究了所讨论的均场博弈的确定性对应物。

    arXiv:2404.02871v1 Announce Type: cross  Abstract: We establish the existence and uniqueness of the equilibrium for a stochastic mean-field game of optimal investment. The analysis covers both finite and infinite time horizons, and the mean-field interaction of the representative company with a mass of identical and indistinguishable firms is modeled through the time-dependent price at which the produced good is sold. At equilibrium, this price is given in terms of a nonlinear function of the expected (optimally controlled) production capacity of the representative company at each time. The proof of the existence and uniqueness of the mean-field equilibrium relies on a priori estimates and the study of nonlinear integral equations, but employs different techniques for the finite and infinite horizon cases. Additionally, we investigate the deterministic counterpart of the mean-field game under study.
    
[^2]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^3]: 协变量自适应随机化下的多个回归器的调整

    Adjustment with Many Regressors Under Covariate-Adaptive Randomizations. (arXiv:2304.08184v1 [econ.EM])

    [http://arxiv.org/abs/2304.08184](http://arxiv.org/abs/2304.08184)

    本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。

    

    本文针对协变量自适应随机化（CAR）中的因果推断使用回归调整（RA）时存在的权衡进行了研究。RA可以通过整合未用于随机分配的协变量信息来提高因果估计器的效率。但是，当回归器数量与样本量同阶时，RA的估计误差不能渐近忽略，会降低估计效率。没有考虑RA成本的结果可能导致在零假设下过度拒绝因果推断。为了解决这个问题，我们在CAR下为经过回归调整的平均处理效应（ATE）估计器开发了一种统一的推断理论。我们的理论具有两个关键特征：（1）确保在零假设下的精确渐近大小，无论协变量数量是固定还是最多以样本大小的速度发散，（2）确保在协变量维度方面都比未调整的估计器弱效提高.

    Our paper identifies a trade-off when using regression adjustments (RAs) in causal inference under covariate-adaptive randomizations (CARs). On one hand, RAs can improve the efficiency of causal estimators by incorporating information from covariates that are not used in the randomization. On the other hand, RAs can degrade estimation efficiency due to their estimation errors, which are not asymptotically negligible when the number of regressors is of the same order as the sample size. Failure to account for the cost of RAs can result in over-rejection of causal inference under the null hypothesis. To address this issue, we develop a unified inference theory for the regression-adjusted average treatment effect (ATE) estimator under CARs. Our theory has two key features: (1) it ensures the exact asymptotic size under the null hypothesis, regardless of whether the number of covariates is fixed or diverges at most at the rate of the sample size, and (2) it guarantees weak efficiency impro
    

