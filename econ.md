# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Invalid proxies and volatility changes](https://arxiv.org/abs/2403.08753) | 在代理SVARs中，当外部永久突破导致目标脉冲响应函数随波动率体制变化时，必须明确纳入无条件波动的变化，以点辨识目标结构性冲击并恢复一致性。 |
| [^2] | [On the Value of Information Structures in Stochastic Games.](http://arxiv.org/abs/2308.09211) | 本文研究了不完全公开监测的随机博弈中改进监测如何影响极限均衡收益集。我们通过引入加权混淆信息结构，证明了极限完美公开均衡收益集的单调性。我们还介绍并讨论了另一个较弱的条件，用于扩展极限均衡收益集。 |
| [^3] | [Simple Estimation of Semiparametric Models with Measurement Errors.](http://arxiv.org/abs/2306.14311) | 本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。 |
| [^4] | [Non-Market Allocation Mechanisms: Optimal Design and Investment Incentives.](http://arxiv.org/abs/2303.11805) | 研究了如何对选择机制进行最优设计，考虑代理商的投资激励，确定性的“过/不过”选择规则实际上是最优的。 |
| [^5] | [Identification-robust inference for the LATE with high-dimensional covariates.](http://arxiv.org/abs/2302.09756) | 本文提出了一种适用于高维协变量下的局部平均处理效应的检验统计量，证明其具有统一正确的大小，并通过双重/无偏机器学习方法实现了推断和置信区间计算。模拟结果表明，该检验具有鲁棒性，可以有效处理识别力较弱和高维设置下的数据。应用于实证研究中，该方法在铁路通达对城市人口增长的影响研究中表现出更短的置信区间和更小的点估计。 |

# 详细

[^1]: 无效的代理和波动变化

    Invalid proxies and volatility changes

    [https://arxiv.org/abs/2403.08753](https://arxiv.org/abs/2403.08753)

    在代理SVARs中，当外部永久突破导致目标脉冲响应函数随波动率体制变化时，必须明确纳入无条件波动的变化，以点辨识目标结构性冲击并恢复一致性。

    

    当在代理SVARs中，VAR扰动的协方差矩阵受到外生、永久、非重复性突破的影响，从而产生随着波动率体制变化而变化的目标脉冲响应函数(IRFs)时，即使是强大的外生外部工具也可能导致对所感兴趣的动态因果效应的估计值不一致，如果不适当考虑这些突破。在这种情况下，必须明确地将无条件波动的变化纳入考虑，以便点辨识出目标结构冲击并可能恢复一致性。我们证明，在利用波动性变化所暗示的瞬时时刻的必要充分秩条件下，目标IRFs可以被点辨识并一致估计。重要的是，标准渐近推断在这种情况下仍然是有效的，尽管(I)代理和被工具化的结构性冲击之间的协方差是接近于零，就像Stai中的情况一样。

    arXiv:2403.08753v1 Announce Type: new  Abstract: When in proxy-SVARs the covariance matrix of VAR disturbances is subject to exogenous, permanent, nonrecurring breaks that generate target impulse response functions (IRFs) that change across volatility regimes, even strong, exogenous external instruments can result in inconsistent estimates of the dynamic causal effects of interest if the breaks are not properly accounted for. In such cases, it is essential to explicitly incorporate the shifts in unconditional volatility in order to point-identify the target structural shocks and possibly restore consistency. We demonstrate that, under a necessary and sufficient rank condition that leverages moments implied by changes in volatility, the target IRFs can be point-identified and consistently estimated. Importantly, standard asymptotic inference remains valid in this context despite (i) the covariance between the proxies and the instrumented structural shocks being local-to-zero, as in Stai
    
[^2]: 关于信息结构在随机博弈中的价值研究

    On the Value of Information Structures in Stochastic Games. (arXiv:2308.09211v1 [econ.TH])

    [http://arxiv.org/abs/2308.09211](http://arxiv.org/abs/2308.09211)

    本文研究了不完全公开监测的随机博弈中改进监测如何影响极限均衡收益集。我们通过引入加权混淆信息结构，证明了极限完美公开均衡收益集的单调性。我们还介绍并讨论了另一个较弱的条件，用于扩展极限均衡收益集。

    

    本文研究了改进监测如何影响不完全公开监测的随机博弈的极限均衡收益集。我们引入了一个简单的Blackwell混淆的推广，称为加权混淆，以便比较这类博弈的不同信息结构。我们的主要结果是极限完美公开均衡（PPE）收益集相对于这种信息顺序的单调性：我们证明了如果后一种信息结构是前一种信息结构的加权混淆，则按状态比较后一种信息结构的极限PPE收益集要大于前一种信息结构的极限PPE收益集。我们还证明了这种单调性结果在强对称均衡类中也成立。最后，我们引入并讨论了另一个较弱的足够条件，用于扩展极限PPE收益集。尽管更加复杂和难以验证，但在某些特殊情况下很有用。

    This paper studies how improved monitoring affects the limit equilibrium payoff set for stochastic games with imperfect public monitoring. We introduce a simple generalization of Blackwell garbling called weighted garbling in order to compare different information structures for this class of games. Our main result is the monotonicity of the limit perfect public equilibrium (PPE) payoff set with respect to this information order: we show that the limit PPE payoff sets with one information structure is larger than the limit PPE payoff sets with another information structure state by state if the latter information structure is a weighted garbling of the former. We show that this monotonicity result also holds for the class of strongly symmetric equilibrium. Finally, we introduce and discuss another weaker sufficient condition for the expansion of limit PPE payoff set. It is more complex and difficult to verify, but useful in some special cases.
    
[^3]: 测量误差中半参数模型的简单估计

    Simple Estimation of Semiparametric Models with Measurement Errors. (arXiv:2306.14311v1 [econ.EM])

    [http://arxiv.org/abs/2306.14311](http://arxiv.org/abs/2306.14311)

    本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    

    我们在广义矩量方法（GMM）框架下开发了一种解决变量误差（EIV）问题的实用方法。我们关注的是EIV的可变性是测量误差变量的一小部分的情况，这在实证应用中很常见。对于任何初始矩条件，我们的方法提供了纠正后对EIV具有鲁棒性的矩条件集。我们表明，基于这些矩的GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论。即使EIV很大，朴素估计量（忽略EIV问题）可能严重偏误并且置信区间的覆盖率为0％，我们的方法也能处理。我们的方法不涉及非参数估计，这对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    We develop a practical way of addressing the Errors-In-Variables (EIV) problem in the Generalized Method of Moments (GMM) framework. We focus on the settings in which the variability of the EIV is a fraction of that of the mismeasured variables, which is typical for empirical applications. For any initial set of moment conditions our approach provides a corrected set of moment conditions that are robust to the EIV. We show that the GMM estimator based on these moments is root-n-consistent, with the standard tests and confidence intervals providing valid inference. This is true even when the EIV are so large that naive estimators (that ignore the EIV problem) may be heavily biased with the confidence intervals having 0% coverage. Our approach involves no nonparametric estimation, which is particularly important for applications with multiple covariates, and settings with multivariate, serially correlated, or non-classical EIV.
    
[^4]: 非市场分配机制：最优设计与投资激励研究

    Non-Market Allocation Mechanisms: Optimal Design and Investment Incentives. (arXiv:2303.11805v1 [econ.TH])

    [http://arxiv.org/abs/2303.11805](http://arxiv.org/abs/2303.11805)

    研究了如何对选择机制进行最优设计，考虑代理商的投资激励，确定性的“过/不过”选择规则实际上是最优的。

    

    本文研究了如何对选择机制进行最优设计，并考虑到代理商的投资激励。主体希望将均质资源分配给异质人群的代理商。主体确定了一个可能是随机的选择规则，该规则取决于代理商本质上看重的一维特征。代理商严格偏爱被主体选择，可能会进行昂贵的投资以提高在主体揭示之前的特征。我们表明，即使随机选择规则促进了代理商的投资，特别是在特征分布的顶部，确定性的“过/不过”选择规则实际上是最优的。

    We study how to optimally design selection mechanisms, accounting for agents' investment incentives. A principal wishes to allocate a resource of homogeneous quality to a heterogeneous population of agents. The principal commits to a possibly random selection rule that depends on a one-dimensional characteristic of the agents she intrinsically values. Agents have a strict preference for being selected by the principal and may undertake a costly investment to improve their characteristic before it is revealed to the principal. We show that even if random selection rules foster agents' investments, especially at the top of the characteristic distribution, deterministic "pass-fail" selection rules are in fact optimal.
    
[^5]: 高维协变量下的局部平均处理效应鲁棒性推断

    Identification-robust inference for the LATE with high-dimensional covariates. (arXiv:2302.09756v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.09756](http://arxiv.org/abs/2302.09756)

    本文提出了一种适用于高维协变量下的局部平均处理效应的检验统计量，证明其具有统一正确的大小，并通过双重/无偏机器学习方法实现了推断和置信区间计算。模拟结果表明，该检验具有鲁棒性，可以有效处理识别力较弱和高维设置下的数据。应用于实证研究中，该方法在铁路通达对城市人口增长的影响研究中表现出更短的置信区间和更小的点估计。

    

    本文研究了高维协变量下的局部平均处理效应(LATE)，不论识别力如何。我们提出了一种新的高维LATE检验统计量，并证明了我们的检验在渐进情况下具有统一正确的大小。通过采用双重/无偏机器学习(DML)方法来估计干扰参数，我们开发了简单易实施的算法来推断和计算高维LATE的置信区间。模拟结果表明，我们的检验对于识别力较弱和高维设置下的大小控制和功效表现具有鲁棒性，优于其他传统检验方法。将所提出的检验应用于铁路和人口数据，研究铁路通达对城市人口增长的影响，我们观察到与传统检验相比，铁路通达系数的置信区间长度更短，点估计更小。

    This paper investigates the local average treatment effect (LATE) with high-dimensional covariates, irrespective of the strength of identification. We propose a novel test statistic for the high-dimensional LATE, demonstrating that our test has uniformly correct asymptotic size. By employing the double/debiased machine learning (DML) method to estimate nuisance parameters, we develop easy-to-implement algorithms for inference and confidence interval calculation of the high-dimensional LATE. Simulations indicate that our test is robust against both weak identification and high-dimensional setting concerning size control and power performance, outperforming other conventional tests. Applying the proposed test to railroad and population data to study the effect of railroad access on urban population growth, we observe the shorter length of confidence intervals and smaller point estimates for the railroad access coefficients compared to the conventional tests.
    

