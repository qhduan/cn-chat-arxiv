# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantile Granger Causality in the Presence of Instability](https://arxiv.org/abs/2402.09744) | 我们提出了一种在不稳定环境中评估分位数格兰杰因果关系的新框架，该框架具有一致性、非平凡的功效和某些重要特殊情况下的中心性。蒙特卡洛模拟显示，所提出的检验统计量具有正确的经验大小和较高的功效，即使在没有结构性变化的情况下也是如此。两个实证应用进一步证明了我们方法的适用性。 |
| [^2] | [Flexible Covariate Adjustments in Regression Discontinuity Designs.](http://arxiv.org/abs/2107.07942) | 本研究提出了一种在回归不连续设计中更有效地利用协变量信息的估计器类，可以容纳大量的离散或连续协变量，并经由机器学习、非参数回归或经典参数方法来估计。通过对结果变量适当修改，这种估计器易于实现，可以选择类似于传统RD分析的调整参数。 |
| [^3] | [Filtered and Unfiltered Treatment Effects with Targeting Instruments.](http://arxiv.org/abs/2007.10432) | 本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。 |

# 详细

[^1]: 在不稳定环境中评估分位数格兰杰因果关系的新框架

    Quantile Granger Causality in the Presence of Instability

    [https://arxiv.org/abs/2402.09744](https://arxiv.org/abs/2402.09744)

    我们提出了一种在不稳定环境中评估分位数格兰杰因果关系的新框架，该框架具有一致性、非平凡的功效和某些重要特殊情况下的中心性。蒙特卡洛模拟显示，所提出的检验统计量具有正确的经验大小和较高的功效，即使在没有结构性变化的情况下也是如此。两个实证应用进一步证明了我们方法的适用性。

    

    我们提出了一种在不稳定环境中评估分位数格兰杰因果关系的新框架，该框架可以针对固定分位数或连续分位数水平进行评估。我们提出的检验统计量在固定备择假设下是一致的，对于局部备择假设具有非平凡的功效，并且在某些重要特殊情况下是中心的。此外，我们还展示了当渐近分布依赖于干扰参数时一种自举过程的有效性。蒙特卡洛模拟显示，所提出的检验统计量具有正确的经验大小和较高的功效，即使在没有结构性变化的情况下也是如此。最后，两个能源经济学和宏观经济学的实证应用突出了我们方法的适用性，因为新的检验提供了更强的格兰杰因果关系证据。

    arXiv:2402.09744v1 Announce Type: new  Abstract: We propose a new framework for assessing Granger causality in quantiles in unstable environments, for a fixed quantile or over a continuum of quantile levels. Our proposed test statistics are consistent against fixed alternatives, they have nontrivial power against local alternatives, and they are pivotal in certain important special cases. In addition, we show the validity of a bootstrap procedure when asymptotic distributions depend on nuisance parameters. Monte Carlo simulations reveal that the proposed test statistics have correct empirical size and high power, even in absence of structural breaks. Finally, two empirical applications in energy economics and macroeconomics highlight the applicability of our method as the new tests provide stronger evidence of Granger causality.
    
[^2]: 回归不连续设计中的灵活协变量调整

    Flexible Covariate Adjustments in Regression Discontinuity Designs. (arXiv:2107.07942v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2107.07942](http://arxiv.org/abs/2107.07942)

    本研究提出了一种在回归不连续设计中更有效地利用协变量信息的估计器类，可以容纳大量的离散或连续协变量，并经由机器学习、非参数回归或经典参数方法来估计。通过对结果变量适当修改，这种估计器易于实现，可以选择类似于传统RD分析的调整参数。

    

    实证回归不连续（RD）研究通常使用协变量来增加其估计结果的精度。本文提出了一种新颖的估计器类，比目前实践中广泛使用的线性调整估计器更有效地利用这些协变量信息。我们的方法可以容纳可能大量的离散或连续协变量。它涉及使用适当修改了的结果变量运行标准RD分析，该变量的形式为原始结果与协变量函数的差异。我们表征了导致渐近方差最小的估计器的函数，并展示了如何通过现代机器学习、非参数回归或经典参数方法来估计它。由此产生的估计器易于实现，因为可以选择类似于传统RD分析的调整参数。广泛的模拟研究说明了我们的方法在有限样本中的性能，另外，一个案例研究突出了它的实证相关性。

    Empirical regression discontinuity (RD) studies often use covariates to increase the precision of their estimates. In this paper, we propose a novel class of estimators that use such covariate information more efficiently than the linear adjustment estimators that are currently used widely in practice. Our approach can accommodate a possibly large number of either discrete or continuous covariates. It involves running a standard RD analysis with an appropriately modified outcome variable, which takes the form of the difference between the original outcome and a function of the covariates. We characterize the function that leads to the estimator with the smallest asymptotic variance, and show how it can be estimated via modern machine learning, nonparametric regression, or classical parametric methods. The resulting estimator is easy to implement, as tuning parameters can be chosen as in a conventional RD analysis. An extensive simulation study illustrates the performance of our approac
    
[^3]: 有目标工具的过滤与未过滤处理效果

    Filtered and Unfiltered Treatment Effects with Targeting Instruments. (arXiv:2007.10432v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2007.10432](http://arxiv.org/abs/2007.10432)

    本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。

    

    在应用中，多值处理是很常见的。我们探讨了在这种情况下使用离散工具来控制选择偏差的方法。我们强调了有关定位（工具定位于哪些处理）和过滤（限制分析师对给定观测的处理分配的知识）的假设作用。这允许我们建立条件，使得针对组合编译器群体，可以确定反事实平均值和处理效果。我们通过将其应用于Head Start Impact Study和Student Achievement and Retention Project的数据来说明我们框架的实用性。

    Multivalued treatments are commonplace in applications. We explore the use of discrete-valued instruments to control for selection bias in this setting. Our discussion stresses the role of assumptions on targeting (which instruments target which treatments) and filtering (limits on the analyst's knowledge of the treatment assigned to a given observation). It allows us to establish conditions under which counterfactual averages and treatment effects are identified for composite complier groups. We illustrate the usefulness of our framework by applying it to data from the Head Start Impact Study and the Student Achievement and Retention Project.
    

