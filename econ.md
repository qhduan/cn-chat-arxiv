# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Regression under Cluster Sampling](https://arxiv.org/abs/2403.04766) | 本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论，并提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并验证了集群稳健带宽选择的有效性。 |
| [^2] | [Ordinal Intensity-Efficient Allocations](https://arxiv.org/abs/2011.04306) | 这篇论文研究了一种新的分配问题，考虑了代理的序数强度，提出了基于排名的准则，定义了强度高效的分配。论文初步讨论了这种分配的存在性和非存在性。 |
| [^3] | [The Fragility of Sparsity.](http://arxiv.org/abs/2311.02299) | 稀疏性的线性回归估计在选择回归矩阵和假设检验上存在脆弱性，OLS能够提供更健壮的结果而效率损失较小。 |
| [^4] | [Recursive Preferences, Correlation Aversion, and the Temporal Resolution of Uncertainty.](http://arxiv.org/abs/2304.04599) | 本文探讨了递归效用模型中的相关厌恶行为特征，提出了它们对资产定价、气候政策和最优财政政策等领域的应用意义。 |
| [^5] | [Clustered Covariate Regression.](http://arxiv.org/abs/2302.09255) | 本文提出了一种聚类协变量回归方法，该方法通过使用聚类和紧凑参数支持的自然限制来解决高维度协变量问题。与竞争估计器相比，该方法在偏差减小和尺寸控制方面表现出色，并在估计汽油需求的价格和收入弹性方面具有实用性。 |

# 详细

[^1]: 集群抽样下的非参数回归

    Nonparametric Regression under Cluster Sampling

    [https://arxiv.org/abs/2403.04766](https://arxiv.org/abs/2403.04766)

    本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论，并提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并验证了集群稳健带宽选择的有效性。

    

    本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论。我们研究了非参数密度估计、Nadaraya-Watson核回归和局部线性估计。我们的理论考虑了增长和异质的簇大小。我们推导了渐近条件偏差和方差，确立了一致收敛性，并证明了渐近正态性。我们的发现表明，在异质的簇大小下，渐近方差包括一个反映簇内相关性的新项，当假定簇大小有界时被忽略。我们提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并证明了它们的一致性。在模拟中，我们验证了集群稳健带宽选择的有效性，并展示了推导的集群稳健置信区间提高了覆盖率。

    arXiv:2403.04766v1 Announce Type: new  Abstract: This paper develops a general asymptotic theory for nonparametric kernel regression in the presence of cluster dependence. We examine nonparametric density estimation, Nadaraya-Watson kernel regression, and local linear estimation. Our theory accommodates growing and heterogeneous cluster sizes. We derive asymptotic conditional bias and variance, establish uniform consistency, and prove asymptotic normality. Our findings reveal that under heterogeneous cluster sizes, the asymptotic variance includes a new term reflecting within-cluster dependence, which is overlooked when cluster sizes are presumed to be bounded. We propose valid approaches for bandwidth selection and inference, introduce estimators of the asymptotic variance, and demonstrate their consistency. In simulations, we verify the effectiveness of the cluster-robust bandwidth selection and show that the derived cluster-robust confidence interval improves the coverage ratio. We 
    
[^2]: 序数强度高效分配

    Ordinal Intensity-Efficient Allocations

    [https://arxiv.org/abs/2011.04306](https://arxiv.org/abs/2011.04306)

    这篇论文研究了一种新的分配问题，考虑了代理的序数强度，提出了基于排名的准则，定义了强度高效的分配。论文初步讨论了这种分配的存在性和非存在性。

    

    我们研究了分配问题，其中代理除了具有序数偏好外，还具有"序数强度"：他们可以进行简单和内部一致的比较，例如“我更喜欢$a$而不是$b$，比我更喜欢$c$而不是$d$”，而不一定能够量化它们。在这种新的信息社会选择环境中，我们首先引入了一个基于排名的准则，使得可以进行这些序数强度的跨个人比较。基于这个准则，我们定义了一种分配为“强度高效”，如果它在代理的强度引发的偏好方面是帕累托有效的，并且当另一种分配以相同的方式将相同的物品对分配给相同的代理对时，前一种分配将每对中普遍更受偏好的物品分配给更喜欢它的代理。我们在不对偏好施加限制的情况下提出了关于这种分配的一些初步结果。

    We study the assignment problem in situations where, in addition to having ordinal preferences, agents also have *ordinal intensities*: they can make simple and internally consistent comparisons such as "I prefer $a$ to $b$ more than I prefer $c$ to $d$" without necessarily being able to quantify them. In this new informational social-choice environment we first introduce a rank-based criterion that enables interpersonal comparability of such ordinal intensities. Building on this criterion, we define an allocation to be *"intensity-efficient"* if it is Pareto efficient with respect to the preferences induced by the agents' intensities and also such that, when another allocation assigns the same pairs of items to the same pairs of agents but in a "flipped" way, the former allocation assigns the commonly preferred item in every such pair to the agent who prefers it more. We present some first results on the (non-)existence of such allocations without imposing restrictions on preferences 
    
[^3]: 稀疏性的脆弱性

    The Fragility of Sparsity. (arXiv:2311.02299v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2311.02299](http://arxiv.org/abs/2311.02299)

    稀疏性的线性回归估计在选择回归矩阵和假设检验上存在脆弱性，OLS能够提供更健壮的结果而效率损失较小。

    

    我们使用三个实证应用展示了线性回归估计在依赖稀疏性假设时存在两种脆弱性。首先，我们证明在不影响普通最小二乘(OLS)估计的情况下，如基线类别的选择与分类控制相关，可能会使稀疏性估计值移动超过两个标准误。其次，我们开发了两个基于将稀疏性估计与OLS估计进行比较的稀疏性假设检验。在所有三个应用中，这些检验倾向于拒绝稀疏性假设。除非自变量的数量与样本量相当或超过样本量，否则OLS能够以较小的效率损失产生更健壮的结果。

    We show, using three empirical applications, that linear regression estimates which rely on the assumption of sparsity are fragile in two ways. First, we document that different choices of the regressor matrix that do not impact ordinary least squares (OLS) estimates, such as the choice of baseline category with categorical controls, can move sparsity-based estimates two standard errors or more. Second, we develop two tests of the sparsity assumption based on comparing sparsity-based estimators with OLS. The tests tend to reject the sparsity assumption in all three applications. Unless the number of regressors is comparable to or exceeds the sample size, OLS yields more robust results at little efficiency cost.
    
[^4]: 递归偏好、相关厌恶和时间不确定性的解决方式

    Recursive Preferences, Correlation Aversion, and the Temporal Resolution of Uncertainty. (arXiv:2304.04599v1 [econ.TH])

    [http://arxiv.org/abs/2304.04599](http://arxiv.org/abs/2304.04599)

    本文探讨了递归效用模型中的相关厌恶行为特征，提出了它们对资产定价、气候政策和最优财政政策等领域的应用意义。

    

    递归效用模型在许多经济应用程序中起着重要作用。本文研究了这些模型所表现的一种新的行为特征：对时间上呈现持续性（正自相关）风险的厌恶，称为相关厌恶。我引入了这种属性的形式概念，并提供了一个基于风险态度的特征，同时还表明相关厌恶的偏好具有特定的变分表示。我讨论了这些发现如何说明对相关性的态度是推动递归效用在资产定价、气候政策和最优财政政策等领域应用的关键行为因素。

    Models of recursive utility are of central importance in many economic applications. This paper investigates a new behavioral feature exhibited by these models: aversion to risks that exhibit persistence (positive autocorrelation) through time, referred to as correlation aversion. I introduce a formal notion of such a property and provide a characterization based on risk attitudes, and show that correlation averse preferences admit a specific variational representation. I discuss how these findings imply that attitudes toward correlation are a crucial behavioral aspect driving the applications of recursive utility in fields such as asset pricing, climate policy, and optimal fiscal policy.
    
[^5]: 聚类协变量回归

    Clustered Covariate Regression. (arXiv:2302.09255v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.09255](http://arxiv.org/abs/2302.09255)

    本文提出了一种聚类协变量回归方法，该方法通过使用聚类和紧凑参数支持的自然限制来解决高维度协变量问题。与竞争估计器相比，该方法在偏差减小和尺寸控制方面表现出色，并在估计汽油需求的价格和收入弹性方面具有实用性。

    

    模型估计中协变量维度的高度增加，解决这个问题的现有技术通常需要无序性或不可观测参数向量的离散异质性。然而，在某些经验背景下，经济理论可能不支持任何限制，这可能导致严重的偏差和误导性推断。本文介绍的基于聚类的分组参数估计器（GPE）放弃这两个限制，而选择参数支持是紧凑的自然限制。在标准条件下，GPE具有稳健的大样本性质，并适应了支持可以远离零点的稀疏和非稀疏参数。广泛的蒙特卡洛模拟证明了与竞争估计器相比，GPE在偏差减小和尺寸控制方面的出色性能。对于估计汽油需求的价格和收入弹性的实证应用突显了GPE的实用性。

    High covariate dimensionality is increasingly occurrent in model estimation, and existing techniques to address this issue typically require sparsity or discrete heterogeneity of the unobservable parameter vector. However, neither restriction may be supported by economic theory in some empirical contexts, leading to severe bias and misleading inference. The clustering-based grouped parameter estimator (GPE) introduced in this paper drops both restrictions in favour of the natural one that the parameter support be compact. GPE exhibits robust large sample properties under standard conditions and accommodates both sparse and non-sparse parameters whose support can be bounded away from zero. Extensive Monte Carlo simulations demonstrate the excellent performance of GPE in terms of bias reduction and size control compared to competing estimators. An empirical application of GPE to estimating price and income elasticities of demand for gasoline highlights its practical utility.
    

