# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Machine Learning for Moderation Effects.](http://arxiv.org/abs/2401.08290) | 本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。 |
| [^2] | [Substitutes markets with budget constraints: solving for competitive and optimal prices.](http://arxiv.org/abs/2310.03692) | 本文研究了一个有预算限制的市场，买家具有准线性效用和线性替代估值，卖家需要找到最大化收入或福利的价格和无嫉妒的分配。研究发现竞争均衡价格也是无嫉妒收入最大化的，并提供了对“可行”价格集的新颖描述。 |
| [^3] | [Doubly Robust Uniform Confidence Bands for Group-Time Conditional Average Treatment Effects in Difference-in-Differences.](http://arxiv.org/abs/2305.02185) | 本研究提出了一种双重稳健推断方法，该方法可用于构建在差分中的组时间条件平均处理效应函数的一致置信带。 |

# 详细

[^1]: 因果机器学习用于中介效应。 (arXiv:2401.08290v1 [econ.EM])

    Causal Machine Learning for Moderation Effects. (arXiv:2401.08290v1 [econ.EM])

    [http://arxiv.org/abs/2401.08290](http://arxiv.org/abs/2401.08290)

    本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。

    

    对于任何决策者来说，了解决策（处理）对整体和子群的影响是非常有价值的。因果机器学习最近提供了用于估计群体平均处理效应（GATE）的工具，以更好地理解处理的异质性。本文解决了在考虑其他协变量变化的情况下解释群体间处理效应差异的难题。我们提出了一个新的参数，即平衡群体平均处理效应（BGATE），它衡量了具有特定分布的先验确定协变量的GATE。通过比较两个BGATE的差异，我们可以更有意义地分析异质性，而不仅仅比较两个GATE。这个参数的估计策略是基于无混淆设置中离散处理的双重/去偏机器学习，该估计量在标准条件下表现为$\sqrt{N}$一致性和渐近正态性。添加额外的标识

    It is valuable for any decision maker to know the impact of decisions (treatments) on average and for subgroups. The causal machine learning literature has recently provided tools for estimating group average treatment effects (GATE) to understand treatment heterogeneity better. This paper addresses the challenge of interpreting such differences in treatment effects between groups while accounting for variations in other covariates. We propose a new parameter, the balanced group average treatment effect (BGATE), which measures a GATE with a specific distribution of a priori-determined covariates. By taking the difference of two BGATEs, we can analyse heterogeneity more meaningfully than by comparing two GATEs. The estimation strategy for this parameter is based on double/debiased machine learning for discrete treatments in an unconfoundedness setting, and the estimator is shown to be $\sqrt{N}$-consistent and asymptotically normal under standard conditions. Adding additional identifyin
    
[^2]: 有预算限制的替代市场：解决竞争和最优价格问题

    Substitutes markets with budget constraints: solving for competitive and optimal prices. (arXiv:2310.03692v1 [econ.TH])

    [http://arxiv.org/abs/2310.03692](http://arxiv.org/abs/2310.03692)

    本文研究了一个有预算限制的市场，买家具有准线性效用和线性替代估值，卖家需要找到最大化收入或福利的价格和无嫉妒的分配。研究发现竞争均衡价格也是无嫉妒收入最大化的，并提供了对“可行”价格集的新颖描述。

    

    多个可分割商品的市场从收入和福利的角度得到了广泛的研究。一般来说，众所周知，无嫉妒收入最大化的结果可能会导致比竞争均衡的结果更低的福利。我们研究了一个市场，买家具有准线性效用和线性替代估值，并且有预算限制，卖家必须找到最大化收入或福利的价格和无嫉妒的分配。我们的设置与广告拍卖和金融资产交换拍卖等市场相似。我们证明了唯一的竞争均衡价格也是无嫉妒收入最大化的。这种最大收入和福利的巧合令人惊讶，在买家具有分段线性估值时甚至会失败。我们提供了一个对“可行”价格集的新颖描述，表明这个集合有一个按元素最小的价格向量，证明了这些价格最大化收入和福利。

    Markets with multiple divisible goods have been studied widely from the perspective of revenue and welfare. In general, it is well known that envy-free revenue-maximal outcomes can result in lower welfare than competitive equilibrium outcomes. We study a market in which buyers have quasilinear utilities with linear substitutes valuations and budget constraints, and the seller must find prices and an envy-free allocation that maximise revenue or welfare. Our setup mirrors markets such as ad auctions and auctions for the exchange of financial assets. We prove that the unique competitive equilibrium prices are also envy-free revenue-maximal. This coincidence of maximal revenue and welfare is surprising and breaks down even when buyers have piecewise-linear valuations. We present a novel characterisation of the set of "feasible" prices at which demand does not exceed supply, show that this set has an elementwise minimal price vector, and demonstrate that these prices maximise revenue and w
    
[^3]: 双重稳健一致置信带在差分中的组时间条件平均处理效应中的应用

    Doubly Robust Uniform Confidence Bands for Group-Time Conditional Average Treatment Effects in Difference-in-Differences. (arXiv:2305.02185v1 [econ.EM])

    [http://arxiv.org/abs/2305.02185](http://arxiv.org/abs/2305.02185)

    本研究提出了一种双重稳健推断方法，该方法可用于构建在差分中的组时间条件平均处理效应函数的一致置信带。

    

    本研究考虑了对面板数据进行分析，以研究在Callaway和Sant'Anna（2021）的错位差分设置中，针对感兴趣的预处理协变量的治疗效应异质性。在一组标准识别条件下，一个基于协变量的双重稳健估计值识别了给定协变量的组时间条件平均处理效应。鉴于这个识别结果，我们提出了一个基于非参数局部线性回归和参数估计方法的三步估计程序，并开发了一个双重稳健推断方法来构建组时间条件平均处理效应函数的一致置信带。

    This study considers a panel data analysis to examine the heterogeneity in treatment effects with respect to a pre-treatment covariate of interest in the staggered difference-in-differences setting in Callaway and Sant'Anna (2021). Under a set of standard identification conditions, a doubly robust estimand conditional on the covariate identifies the group-time conditional average treatment effect given the covariate. Given this identification result, we propose a three-step estimation procedure based on nonparametric local linear regressions and parametric estimation methods, and develop a doubly robust inference method to construct a uniform confidence band of the group-time conditional average treatment effect function.
    

