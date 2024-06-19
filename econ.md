# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Speed, Accuracy, and Complexity](https://arxiv.org/abs/2403.11240) | 本文重新审视了使用反应时间来推断问题复杂性的有效性，提出了一个新方法来正确推断问题的复杂性，强调了在简单和复杂问题中决策速度都很快的特点，并探讨了反应时间与能力之间的非单调关系。 |
| [^2] | [A dual approach to nonparametric characterization for random utility models](https://arxiv.org/abs/2403.04328) | 提出了一种新颖的非参数特征方法，构建了一个矩阵来揭示随机效用模型中的偏好关系结构，提供了关于循环选择概率和最大理性选择模式权重的信息。 |
| [^3] | [Uniform Inference for Nonlinear Endogenous Treatment Effects with High-Dimensional Covariates.](http://arxiv.org/abs/2310.08063) | 该论文提出了一种用于高维协变量的非线性内生处理效应的一致推断方法，主要创新是在高维度下对非参数工具变量模型进行双偏差校正，并提供了边际效应函数的均匀置信带。 |
| [^4] | [Rankings-Dependent Preferences: A Real Goods Matching Experiment.](http://arxiv.org/abs/2305.03644) | 研究探讨了代理人通过匹配机制获得对象的偏好是否受到其在排名列表中排名的影响，结果发现更高的排名可以带来更大的效用。研究发现真实偏好交换机制可能不是一种占优策略，且能够提供给更多的代理人报告为更高排名的对象的非占优机制可能会增加市场福利。 |
| [^5] | [Standard Errors for Calibrated Parameters.](http://arxiv.org/abs/2109.08109) | 本文提出了一种针对校准参数的保守标准误差和置信区间的方法，使用仅个别经验矩的方差进行估计，有效应对最坏的相关结构情况。同时提供了超定或参数限制的检验方法，并在实证应用中展示了该方法的有效性。 |

# 详细

[^1]: 速度、准确性和复杂性

    Speed, Accuracy, and Complexity

    [https://arxiv.org/abs/2403.11240](https://arxiv.org/abs/2403.11240)

    本文重新审视了使用反应时间来推断问题复杂性的有效性，提出了一个新方法来正确推断问题的复杂性，强调了在简单和复杂问题中决策速度都很快的特点，并探讨了反应时间与能力之间的非单调关系。

    

    这篇论文重新审视了使用反应时间来推断问题复杂性的有效性。重新审视了一个经典的Wald模型，将信噪比作为问题复杂性的度量。虽然在问题复杂性上，选择质量是单调的，但期望的停止时间是倒U形的。事实上，决策在非常简单和非常复杂的问题中都很快：在简单问题中很快就能理解哪个选择是最好的，而在复杂问题中将会成本过高--这一洞察力也适用于一般的昂贵信息获取模型。这种非单调性也构成了反应时间与能力之间模糊关系的基础，即更高的能力意味着在非常复杂的问题中决策更慢，但在简单问题中决策更快。最后，本文提出了一种新方法，根据选择对激励变化的反应更多来正确推断问题的复杂性。

    arXiv:2403.11240v1 Announce Type: new  Abstract: This paper re-examines the validity of using response time to infer problem complexity. It revisits a canonical Wald model of optimal stopping, taking signal-to-noise ratio as a measure of problem complexity. While choice quality is monotone in problem complexity, expected stopping time is inverse $U$-shaped. Indeed decisions are fast in both very simple and very complex problems: in simple problems it is quick to understand which alternative is best, while in complex problems it would be too costly -- an insight which extends to general costly information acquisition models. This non-monotonicity also underlies an ambiguous relationship between response time and ability, whereby higher ability entails slower decisions in very complex problems, but faster decisions in simple problems. Finally, this paper proposes a new method to correctly infer problem complexity based on the finding that choices react more to changes in incentives in mo
    
[^2]: 随机效用模型的非参数特征的双重方法

    A dual approach to nonparametric characterization for random utility models

    [https://arxiv.org/abs/2403.04328](https://arxiv.org/abs/2403.04328)

    提出了一种新颖的非参数特征方法，构建了一个矩阵来揭示随机效用模型中的偏好关系结构，提供了关于循环选择概率和最大理性选择模式权重的信息。

    

    本文提出了一种新颖的随机效用模型（RUM）特征，这种特征事实上是对Kitamura和Stoye（2018，ECMA）的特征的对偶表示。对于给定的预算家庭及其“补丁”表示，我们构建一个矩阵Ξ，其中每个行向量表明每个预算子家庭中可能的显式偏好关系的结构。然后，证明了在预算线的“补丁”上的随机需求系统，记为π，与RUM一致当且仅当Ξπ≥1。除了提供简洁的封闭形式特征之外，特别是当π与RUM不一致时，向量Ξπ还包含关于（1）必须以正概率发生循环选择的预算子家庭，以及（2）人口中理性选择模式的最大可能权重的信息。

    arXiv:2403.04328v1 Announce Type: new  Abstract: This paper develops a novel characterization for random utility models (RUM), which turns out to be a dual representation of the characterization by Kitamura and Stoye (2018, ECMA). For a given family of budgets and its "patch" representation \'a la Kitamura and Stoye, we construct a matrix $\Xi$ of which each row vector indicates the structure of possible revealed preference relations in each subfamily of budgets. Then, it is shown that a stochastic demand system on the patches of budget lines, say $\pi$, is consistent with a RUM, if and only if $\Xi\pi \geq \mathbb{1}$. In addition to providing a concise closed form characterization, especially when $\pi$ is inconsistent with RUMs, the vector $\Xi\pi$ also contains information concerning (1) sub-families of budgets in which cyclical choices must occur with positive probabilities, and (2) the maximal possible weights on rational choice patterns in a population. The notion of Chv\'atal r
    
[^3]: 带有高维协变量的非线性内生处理效应的一致推断

    Uniform Inference for Nonlinear Endogenous Treatment Effects with High-Dimensional Covariates. (arXiv:2310.08063v1 [econ.EM])

    [http://arxiv.org/abs/2310.08063](http://arxiv.org/abs/2310.08063)

    该论文提出了一种用于高维协变量的非线性内生处理效应的一致推断方法，主要创新是在高维度下对非参数工具变量模型进行双偏差校正，并提供了边际效应函数的均匀置信带。

    

    非线性和内生性在观测数据的因果效应研究中很常见。本文提出了一种新的估计和推断方法，用于具有内生性和潜在高维协变量的非参数处理效应函数。本文的主要创新是在高维度下对非参数工具变量（NPIV）模型进行双偏差校正。我们提供了一个有用的边际效应函数均匀置信带，其定义为非参数处理函数的导数。理论上验证了置信带的渐近诚实性。通过模拟和对空气污染和迁移的实证研究，证明了我们的方法的有效性。

    Nonlinearity and endogeneity are common in causal effect studies with observational data. In this paper, we propose new estimation and inference procedures for nonparametric treatment effect functions with endogeneity and potentially high-dimensional covariates. The main innovation of this paper is the double bias correction procedure for the nonparametric instrumental variable (NPIV) model under high dimensions. We provide a useful uniform confidence band of the marginal effect function, defined as the derivative of the nonparametric treatment function. The asymptotic honesty of the confidence band is verified in theory. Simulations and an empirical study of air pollution and migration demonstrate the validity of our procedures.
    
[^4]: 排序依赖偏好：一项真实货物匹配实验

    Rankings-Dependent Preferences: A Real Goods Matching Experiment. (arXiv:2305.03644v1 [econ.GN])

    [http://arxiv.org/abs/2305.03644](http://arxiv.org/abs/2305.03644)

    研究探讨了代理人通过匹配机制获得对象的偏好是否受到其在排名列表中排名的影响，结果发现更高的排名可以带来更大的效用。研究发现真实偏好交换机制可能不是一种占优策略，且能够提供给更多的代理人报告为更高排名的对象的非占优机制可能会增加市场福利。

    

    本研究探讨通过匹配机制获得对象的偏好是否受到代理人在其报告的排名列表中对其排名的影响。我们假设其他条件相同，代理人对同一对象在其排名更高时会获得更大的效用。排序依赖的效用的引入意味着向真实偏好交换机制提交真实偏好可能不是一种占优策略，并且能够提供给更多的代理人报告为更高排名的对象的非占优机制可能会增加市场福利。我们通过在一个占优策略机制——随机串行独裁——和一个非占优机制——波士顿机制中进行匹配实验来测试这些假设。我们实验设计的一个新特点是在匹配市场分配的对象是真实物品，这使我们能够通过征集内外机制中物品的价值来直接测量排序依赖性。我们的实验结果证实了我们的假设。

    We investigate whether preferences for objects received via a matching mechanism are influenced by how highly agents rank them in their reported rank order list. We hypothesize that all else equal, agents receive greater utility for the same object when they rank it higher. The addition of rankings-dependent utility implies that it may not be a dominant strategy to submit truthful preferences to a strategyproof mechanism, and that non-strategyproof mechanisms that give more agents objects they report as higher ranked may increase market welfare. We test these hypotheses with a matching experiment in a strategyproof mechanism, the random serial dictatorship, and a non-strategyproof mechanism, the Boston mechanism. A novel feature of our experimental design is that the objects allocated in the matching markets are real goods, which allows us to directly measure rankings-dependence by eliciting values for goods both inside and outside of the mechanism. Our experimental results confirm tha
    
[^5]: 校准参数的标准误差

    Standard Errors for Calibrated Parameters. (arXiv:2109.08109v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2109.08109](http://arxiv.org/abs/2109.08109)

    本文提出了一种针对校准参数的保守标准误差和置信区间的方法，使用仅个别经验矩的方差进行估计，有效应对最坏的相关结构情况。同时提供了超定或参数限制的检验方法，并在实证应用中展示了该方法的有效性。

    

    校准是选择结构模型参数以匹配某些经验矩的实践，可以看作是最小距离估计。现有的这类估计量的标准误差公式要求对经验矩的相关结构进行一致估计，但这在实践中通常不可得。相反，个别经验矩的方差通常是可以估计的。我们仅使用这些方差，为结构参数导出保守的标准误差和置信区间，即使在最坏的相关结构下也是有效的。在超定情况下，我们证明使最坏估计方差最小的矩权重方案等同于一个简单解决方案的矩选择问题。最后，我们开发了超定或参数限制的检验方法。我们在一个多产品公司的菜单成本定价模型和一个异质代理新凯恩斯模型上实证应用了我们的方法。

    Calibration, the practice of choosing the parameters of a structural model to match certain empirical moments, can be viewed as minimum distance estimation. Existing standard error formulas for such estimators require a consistent estimate of the correlation structure of the empirical moments, which is often unavailable in practice. Instead, the variances of the individual empirical moments are usually readily estimable. Using only these variances, we derive conservative standard errors and confidence intervals for the structural parameters that are valid even under the worst-case correlation structure. In the over-identified case, we show that the moment weighting scheme that minimizes the worst-case estimator variance amounts to a moment selection problem with a simple solution. Finally, we develop tests of over-identifying or parameter restrictions. We apply our methods empirically to a model of menu cost pricing for multi-product firms and to a heterogeneous agent New Keynesian mod
    

