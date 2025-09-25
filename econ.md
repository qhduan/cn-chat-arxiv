# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modelling with Discretized Variables](https://arxiv.org/abs/2403.15220) | 该文提出了一种通过离散化方法来实现回归参数点标识的模型，同时保护数据机密性，适用于解决面板数据中多元线性回归参数估计的方法。 |
| [^2] | [Adaptive Neyman Allocation.](http://arxiv.org/abs/2309.08808) | 该论文介绍了一种自适应的Neyman分配方法，该方法能够根据不同阶段的观测来估计标准差，从而指导后续阶段的分组决策。 |

# 详细

[^1]: 使用离散变量建模

    Modelling with Discretized Variables

    [https://arxiv.org/abs/2403.15220](https://arxiv.org/abs/2403.15220)

    该文提出了一种通过离散化方法来实现回归参数点标识的模型，同时保护数据机密性，适用于解决面板数据中多元线性回归参数估计的方法。

    

    本文讨论了在计量经济模型中，因变量、一些解释变量或两者均以被截断的区间数据的形式观测到的情况。这种离散化通常是由于敏感变量如收入的保密性造成的。使用这些变量的模型无法对回归参数进行点识别，因为条件矩未知，这导致文献使用区间估计。在这里，我们提出了一种离散化方法，通过该方法可以点识别回归参数，同时保护数据的机密性。我们展示了OLS估计器在面板数据的多元线性回归中的渐近性质。理论发现得到蒙特卡洛实验的支持，并通过应用到澳大利亚性别工资差距上进行了说明。

    arXiv:2403.15220v1 Announce Type: new  Abstract: This paper deals with econometric models in which the dependent variable, some explanatory variables, or both are observed as censored interval data. This discretization often happens due to confidentiality of sensitive variables like income. Models using these variables cannot point identify regression parameters as the conditional moments are unknown, which led the literature to use interval estimates. Here, we propose a discretization method through which the regression parameters can be point identified while preserving data confidentiality. We demonstrate the asymptotic properties of the OLS estimator for the parameters in multivariate linear regressions for cross-sectional data. The theoretical findings are supported by Monte Carlo experiments and illustrated with an application to the Australian gender wage gap.
    
[^2]: 自适应的Neyman分配

    Adaptive Neyman Allocation. (arXiv:2309.08808v1 [stat.ME])

    [http://arxiv.org/abs/2309.08808](http://arxiv.org/abs/2309.08808)

    该论文介绍了一种自适应的Neyman分配方法，该方法能够根据不同阶段的观测来估计标准差，从而指导后续阶段的分组决策。

    

    在实验设计中，Neyman分配是指将受试者分配到处理组和对照组的做法，可能按照它们各自的标准差成比例地分配，其目标是最小化治疗效应估计器的方差。这种广泛认可的方法在处理组和对照组具有不同标准差的情况下增加了统计效力，这在社会实验、临床试验、市场研究和在线A/B测试中经常发生。然而，除非提前知道标准差，否则无法实施Neyman分配。幸运的是，上述应用的多阶段性质使得可以使用较早阶段的观测来估计标准差，进一步指导后续阶段的分配决策。在本文中，我们引入了一个竞争分析框架来研究这个多阶段实验设计问题。我们提出了一种简单的自适应Neyman分配方法。

    In experimental design, Neyman allocation refers to the practice of allocating subjects into treated and control groups, potentially in unequal numbers proportional to their respective standard deviations, with the objective of minimizing the variance of the treatment effect estimator. This widely recognized approach increases statistical power in scenarios where the treated and control groups have different standard deviations, as is often the case in social experiments, clinical trials, marketing research, and online A/B testing. However, Neyman allocation cannot be implemented unless the standard deviations are known in advance. Fortunately, the multi-stage nature of the aforementioned applications allows the use of earlier stage observations to estimate the standard deviations, which further guide allocation decisions in later stages. In this paper, we introduce a competitive analysis framework to study this multi-stage experimental design problem. We propose a simple adaptive Neym
    

