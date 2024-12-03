# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Identification in Discrete Choice](https://arxiv.org/abs/2403.13773) | 我们研究了随机效用模型中的识别和线性独立性，发现随着备选方案数量的增加，任何线性独立的偏好集都是所有偏好集的微不足道的子集，并引入了一个新的条件来确保线性独立性。 |
| [^2] | [Inference on LATEs with covariates](https://arxiv.org/abs/2402.12607) | 本文提出了一种推断程序，能够在控制虚拟变量和工具交互作用的数量很大的情况下，针对饱和经济数据的特征实现LATEs加权平均的有效测试和置信区间。 |
| [^3] | [The Assignment Game: New Mechanisms for Equitable Core Imputations](https://arxiv.org/abs/2402.11437) | 本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。 |
| [^4] | [What To Do (and Not to Do) with Causal Panel Analysis under Parallel Trends: Lessons from A Large Reanalysis Study.](http://arxiv.org/abs/2309.15983) | 通过对37篇使用观察面板数据的文章进行复制和重新分析，我们发现双向固定效应模型的核心结论在新的异质治疗效应鲁棒估计器下基本保持不变，但平行趋势假设违反和统计力量不足仍然是可信推论的重要障碍。 |
| [^5] | [The Robust F-Statistic as a Test for Weak Instruments.](http://arxiv.org/abs/2309.01637) | 该论文提出了一种鲁棒的F统计量作为检验弱工具的方法，并应用于线性广义矩估计器。这一方法通过计算误差项的偏差，简化了弱工具的临界值计算。 |
| [^6] | [Composite Quantile Factor Models.](http://arxiv.org/abs/2308.02450) | 该论文介绍了一种在高维面板数据中进行因子分析的新方法，即复合分位数因子模型。其创新之处在于在不同分位数上估计因子和因子载荷，提高了估计结果的适应性，并引入了一种一致选择因子数的信息准则。模拟结果和实证研究表明，该方法在非正态分布下具有良好的性质。 |
| [^7] | [Antimonotonicity for Preference Axioms: The Natural Counterpart to Comonotonicity.](http://arxiv.org/abs/2307.08542) | 本论文研究了偏序理论中的反单调性，它是共单调性的自然对应物。通过对一些传统公理进行反单调性限制，我们发现并推广了现有模型的广义化公理化。这些推广显示出经典公理在反单调性情况下的最关键测试点。 |
| [^8] | [New possibilities in identification of binary choice models with fixed effects.](http://arxiv.org/abs/2206.10475) | 本文研究了带有固定效应的二选一模型的识别问题，提出了一种符号饱和条件并证明其足以实现模型的识别。同时，我们还提供了一个测试来检验符号饱和条件，并可使用现有的最大分数估计算法进行实施。 |

# 详细

[^1]: 离散选择中的识别限制

    The Limits of Identification in Discrete Choice

    [https://arxiv.org/abs/2403.13773](https://arxiv.org/abs/2403.13773)

    我们研究了随机效用模型中的识别和线性独立性，发现随着备选方案数量的增加，任何线性独立的偏好集都是所有偏好集的微不足道的子集，并引入了一个新的条件来确保线性独立性。

    

    我们研究了随机效用模型中的识别和线性独立性。我们将随机选择数据的特定图形表示的圈复杂度特征化为随机效用模型的维数。我们表明，随着备选方案数量的增加，任何线性独立的偏好集都是所有偏好集的微不足道的子集。我们引入了对偏好集的新条件，该条件足以确保线性独立性。我们通过示例展示该条件并非必要条件，但严格弱于其他现有的足够条件。

    arXiv:2403.13773v1 Announce Type: new  Abstract: We study identification and linear independence in random utility models. We characterize the dimension of the random utility model as the cyclomatic complexity of a specific graphical representation of stochastic choice data. We show that, as the number of alternatives grows, any linearly independent set of preferences is a vanishingly small subset of the set of all preferences. We introduce a new condition on sets of preferences which is sufficient for linear independence. We demonstrate by example that the condition is not necessary, but is strictly weaker than other existing sufficient conditions.
    
[^2]: 具有协变量的LATEs推断

    Inference on LATEs with covariates

    [https://arxiv.org/abs/2402.12607](https://arxiv.org/abs/2402.12607)

    本文提出了一种推断程序，能够在控制虚拟变量和工具交互作用的数量很大的情况下，针对饱和经济数据的特征实现LATEs加权平均的有效测试和置信区间。

    

    在理论上，两阶段最小二乘法（TSLS）能够从饱和规范中识别协变量特定的局部平均处理效应（LATEs）的加权平均，而不对可用协变量进入模型的参数做出假设。在实践中，当饱和导致控制虚拟变量的数量与样本量的数量级相同时，以及使用许多、可以说是弱的工具时，TSLS严重偏倚。本文针对识别目标为饱和TSLS定位的LATEs加权平均的估值，提出了渐近有效的测试和置信区间，即使控制虚拟变量和工具交互作用的数量很大。所提出的推断程序对于饱和经济数据的四个关键特征具有鲁棒性：处理效应异质性、具有丰富支持的协变量、弱识别强度以及条件异方差性。

    arXiv:2402.12607v1 Announce Type: new  Abstract: In theory, two-stage least squares (TSLS) identifies a weighted average of covariate-specific local average treatment effects (LATEs) from a saturated specification without making parametric assumptions on how available covariates enter the model. In practice, TSLS is severely biased when saturation leads to a number of control dummies that is of the same order of magnitude as the sample size, and the use of many, arguably weak, instruments. This paper derives asymptotically valid tests and confidence intervals for an estimand that identifies the weighted average of LATEs targeted by saturated TSLS, even when the number of control dummies and instrument interactions is large. The proposed inference procedure is robust against four key features of saturated economic data: treatment effect heterogeneity, covariates with rich support, weak identification strength, and conditional heteroskedasticity.
    
[^3]: 《分配博弈：公平核分配的新机制》

    The Assignment Game: New Mechanisms for Equitable Core Imputations

    [https://arxiv.org/abs/2402.11437](https://arxiv.org/abs/2402.11437)

    本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。

    

    《分配博弈》的核分配集形成一个（非有限）分配格。迄今为止，仅已知有效算法用于计算其两个极端分配；但是，其中每一个都最大程度地偏袒一个方，不利于双方的分配，导致盈利不均衡。另一个问题是，由一个玩家组成的子联盟（或者来自分配两侧的一系玩家）可以获得零利润，因此核分配不必给予他们任何利润。因此，核分配在个体代理人层面上不提供任何公平性保证。这引出一个问题，即如何计算更公平的核分配。在本文中，我们提出了一个计算分配博弈的Leximin和Leximax核分配的组合（即，该机制不涉及LP求解器）多项式时间机制。这些分配以不同方式实现了“公平性”：

    arXiv:2402.11437v1 Announce Type: cross  Abstract: The set of core imputations of the assignment game forms a (non-finite) distributive lattice. So far, efficient algorithms were known for computing only its two extreme imputations; however, each of them maximally favors one side and disfavors the other side of the bipartition, leading to inequitable profit sharing. Another issue is that a sub-coalition consisting of one player (or a set of players from the same side of the bipartition) can make zero profit, therefore a core imputation is not obliged to give them any profit. Hence core imputations make no fairness guarantee at the level of individual agents. This raises the question of computing {\em more equitable core imputations}.   In this paper, we give combinatorial (i.e., the mechanism does not invoke an LP-solver) polynomial time mechanisms for computing the leximin and leximax core imputations for the assignment game. These imputations achieve ``fairness'' in different ways: w
    
[^4]: 如何在平行趋势下进行因果面板分析：一项大规模再分析研究的教训

    What To Do (and Not to Do) with Causal Panel Analysis under Parallel Trends: Lessons from A Large Reanalysis Study. (arXiv:2309.15983v1 [stat.ME])

    [http://arxiv.org/abs/2309.15983](http://arxiv.org/abs/2309.15983)

    通过对37篇使用观察面板数据的文章进行复制和重新分析，我们发现双向固定效应模型的核心结论在新的异质治疗效应鲁棒估计器下基本保持不变，但平行趋势假设违反和统计力量不足仍然是可信推论的重要障碍。

    

    双向固定效应模型在政治科学中的因果面板分析中普遍应用。然而，最近的方法论讨论挑战了其在存在异质治疗效应和平行趋势假设违反情况下的有效性。这一新兴的文献引入了多个估计器和诊断方法，导致实证研究人员在两个方面产生了困惑：基于双向固定效应模型的现有结果的可靠性和目前的最佳实践。为了解决这些问题，我们考察、复制和重新分析了三个领先政治科学期刊上共37篇运用观察面板数据和二元治疗的文章。使用六种新引入的异质治疗效应鲁棒估计器，我们发现尽管精确性可能受到影响，但基于双向固定效应估计的核心结论在很大程度上保持不变。然而，平行趋势假设的违反和统计力量不足仍然是可信推论的重要障碍。

    Two-way fixed effects (TWFE) models are ubiquitous in causal panel analysis in political science. However, recent methodological discussions challenge their validity in the presence of heterogeneous treatment effects (HTE) and violations of the parallel trends assumption (PTA). This burgeoning literature has introduced multiple estimators and diagnostics, leading to confusion among empirical researchers on two fronts: the reliability of existing results based on TWFE models and the current best practices. To address these concerns, we examined, replicated, and reanalyzed 37 articles from three leading political science journals that employed observational panel data with binary treatments. Using six newly introduced HTE-robust estimators, we find that although precision may be affected, the core conclusions derived from TWFE estimates largely remain unchanged. PTA violations and insufficient statistical power, however, continue to be significant obstacles to credible inferences. Based 
    
[^5]: 作为弱工具检验的鲁棒F统计量

    The Robust F-Statistic as a Test for Weak Instruments. (arXiv:2309.01637v1 [econ.EM])

    [http://arxiv.org/abs/2309.01637](http://arxiv.org/abs/2309.01637)

    该论文提出了一种鲁棒的F统计量作为检验弱工具的方法，并应用于线性广义矩估计器。这一方法通过计算误差项的偏差，简化了弱工具的临界值计算。

    

    Montiel Olea and Pflueger (2013)提出了一种有效的F统计量作为检验弱工具的方法，该方法基于两阶段最小二乘估计器相对于最坏情况下的基准偏差。我们表明，他们的方法适用于一类线性广义矩估计器（GMM），其中包含了一类广义的有效F统计量。标准的非齐次方差鲁棒F统计量属于这一类别。相关的扩展估计器GMMf，扩展为第一阶段，是一种新颖而不寻常的估计器，因为权重矩阵基于第一阶段的残差。由于鲁棒F统计量也可以用作识别不足的检验，通过计算GMMf估计器相对于基准的偏差，可以简化弱工具的临界值计算，无需使用模拟方法或Patnaik (1949)的分布近似方法。在Andrews (2018)的分组数据IV 设计中，...

    Montiel Olea and Pflueger (2013) proposed the effective F-statistic as a test for weak instruments in terms of the Nagar bias of the two-stage least squares (2SLS) estimator relative to a benchmark worst-case bias. We show that their methodology applies to a class of linear generalized method of moments (GMM) estimators with an associated class of generalized effective F-statistics. The standard nonhomoskedasticity robust F-statistic is a member of this class. The associated GMMf estimator, with the extension f for first-stage, is a novel and unusual estimator as the weight matrix is based on the first-stage residuals. As the robust F-statistic can also be used as a test for underidentification, expressions for the calculation of the weak-instruments critical values in terms of the Nagar bias of the GMMf estimator relative to the benchmark simplify and no simulation methods or Patnaik (1949) distributional approximations are needed. In the grouped-data IV designs of Andrews (2018), whe
    
[^6]: 复合分位数因子模型

    Composite Quantile Factor Models. (arXiv:2308.02450v1 [econ.EM])

    [http://arxiv.org/abs/2308.02450](http://arxiv.org/abs/2308.02450)

    该论文介绍了一种在高维面板数据中进行因子分析的新方法，即复合分位数因子模型。其创新之处在于在不同分位数上估计因子和因子载荷，提高了估计结果的适应性，并引入了一种一致选择因子数的信息准则。模拟结果和实证研究表明，该方法在非正态分布下具有良好的性质。

    

    本文介绍了一种在高维面板数据中进行因子分析的方法，即复合分位数因子模型。我们提出了在不同分位数上估计因子和因子载荷的方法，使得估计结果能够更好地适应不同分位数下数据的特征，并且仍然能够对数据的均值进行建模。我们推导了估计的因子和因子载荷的极限分布，并讨论了一种一致选择因子数的信息准则。模拟结果表明，所提出的估计器和信息准则对于几种非正态分布具有良好的有限样本性质。我们还对246个季度宏观经济变量的因子分析进行了实证研究，并开发了一个名为cqrfactor的伴随R包。

    This paper introduces the method of composite quantile factor model for factor analysis in high-dimensional panel data. We propose to estimate the factors and factor loadings across different quantiles of the data, allowing the estimates to better adapt to features of the data at different quantiles while still modeling the mean of the data. We develop the limiting distribution of the estimated factors and factor loadings, and an information criterion for consistent factor number selection is also discussed. Simulations show that the proposed estimator and the information criterion have good finite sample properties for several non-normal distributions under consideration. We also consider an empirical study on the factor analysis for 246 quarterly macroeconomic variables. A companion R package cqrfactor is developed.
    
[^7]: 偏序理论中的反单调性：共单调性的自然对应物

    Antimonotonicity for Preference Axioms: The Natural Counterpart to Comonotonicity. (arXiv:2307.08542v1 [econ.TH])

    [http://arxiv.org/abs/2307.08542](http://arxiv.org/abs/2307.08542)

    本论文研究了偏序理论中的反单调性，它是共单调性的自然对应物。通过对一些传统公理进行反单调性限制，我们发现并推广了现有模型的广义化公理化。这些推广显示出经典公理在反单调性情况下的最关键测试点。

    

    共单调性（“相同变化”）的随机变量最小化了对冲可能性，在许多领域得到广泛应用。传统公理的共单调限制导致了决策模型中的重大发明，包括Gilboa和Schmeidler的模糊模型。本文研究了反单调性（“相反变化”），它是共单调性的自然对应物，最小化了杠杆作用的可能性。令人惊讶的是，传统公理的反单调限制通常并不产生新的模型，而是给出现有模型的广义化公理。因此，我们推广了：（a）通过柯西方程对线性泛函的经典公理化；（b）通过无套利原则对即时无风险定价；（c）通过簿记来描述主观概率；（d）Anscombe-Aumann的期望效用；（e）在Savage的主观期望效用中的风险规避。在每种情况下，我们的广义化公理化展示了经典公理的最关键测试在反单调性情况下的位置。

    Comonotonicity ("same variation") of random variables minimizes hedging possibilities and has been widely used in many fields. Comonotonic restrictions of traditional axioms have led to impactful inventions in decision models, including Gilboa and Schmeidler's ambiguity models. This paper investigates antimonotonicity ("opposite variation"), the natural counterpart to comonotonicity, minimizing leveraging possibilities. Surprisingly, antimonotonic restrictions of traditional axioms often do not give new models but, instead, give generalized axiomatizations of existing ones. We, thus, generalize: (a) classical axiomatizations of linear functionals through Cauchy's equation; (b) as-if-risk-neutral pricing through no-arbitrage; (c) subjective probabilities through bookmaking; (d) Anscombe-Aumann expected utility; (e) risk aversion in Savage's subjective expected utility. In each case, our generalizations show where the most critical tests of classical axioms lie: in the antimonotonic case
    
[^8]: 二选一模型中固定效应识别的新可能性

    New possibilities in identification of binary choice models with fixed effects. (arXiv:2206.10475v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.10475](http://arxiv.org/abs/2206.10475)

    本文研究了带有固定效应的二选一模型的识别问题，提出了一种符号饱和条件并证明其足以实现模型的识别。同时，我们还提供了一个测试来检验符号饱和条件，并可使用现有的最大分数估计算法进行实施。

    

    本文研究了带有固定效应的二选一模型的识别问题。我们提供了一种称为符号饱和条件的条件，并证明该条件足以对该模型进行识别。特别是，我们可以保证即使在有界的回归变量情况下也能实现识别。我们还证明了在没有符号饱和条件的情况下，除非误差分布属于一个小类别，否则无法对该模型进行识别。相同的符号饱和条件也对于识别治疗效应的符号至关重要。我们提供了一个测试来检查符号饱和条件，并可以使用现有的最大分数估计算法进行实施。

    We study the identification of binary choice models with fixed effects. We provide a condition called sign saturation and show that this condition is sufficient for the identification of the model. In particular, we can guarantee identification even with bounded regressors. We also show that without this condition, the model is not identified unless the error distribution belongs to a small class. The same sign saturation condition is also essential for identifying the sign of treatment effects. A test is provided to check the sign saturation condition and can be implemented using existing algorithms for the maximum score estimator.
    

