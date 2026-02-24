# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Inference of Optimal Allocations I: Regularities and their Implications](https://arxiv.org/abs/2403.18248) | 这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。 |
| [^2] | [Stochastic Localization via Iterative Posterior Sampling](https://arxiv.org/abs/2402.10758) | 本论文提出了一种名为SLIPS的方法，通过迭代后验抽样实现随机定位，填补了从非标准化目标密度中抽样的问题的空白。 |
| [^3] | [Dynamic covariate balancing: estimating treatment effects over time with potential local projections.](http://arxiv.org/abs/2103.01280) | 本文提出了一种通过动态协变量平衡方法，基于过去历史上潜在结果期望的局部投影，估计面板数据中动态变化的治疗效果，并考虑结果和时间变化的协变量与治疗轨迹的关系以及治疗效应的异质性。研究结果表明该方法具有良好的渐近性质和数值特性，在实证应用中具有优势。 |

# 详细

[^1]: 统计推断中的最优分配I：规律性及其影响

    Statistical Inference of Optimal Allocations I: Regularities and their Implications

    [https://arxiv.org/abs/2403.18248](https://arxiv.org/abs/2403.18248)

    这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。

    

    在这篇论文中，我们提出了一种用于解决统计最优分配问题的函数可微方法。通过对排序运算符的一般属性进行详细分析，我们首先推导出了值函数的Hadamard可微性。在我们的框架中，Hausdorff测度的概念以及几何测度论中的面积和共面积积分公式是核心。基于我们的Hadamard可微性结果，我们展示了如何利用函数偏微分法直接推导出二元约束最优分配问题的值函数过程以及两步ROC曲线估计量的渐近性质。此外，利用对凸和局部Lipschitz泛函的深刻见解，我们得到了最优分配问题的值函数的额外一般Frechet可微性结果。这些引人入胜的发现激励了我们

    arXiv:2403.18248v1 Announce Type: new  Abstract: In this paper, we develp a functional differentiability approach for solving statistical optimal allocation problems. We first derive Hadamard differentiability of the value function through a detailed analysis of the general properties of the sorting operator. Central to our framework are the concept of Hausdorff measure and the area and coarea integration formulas from geometric measure theory. Building on our Hadamard differentiability results, we demonstrate how the functional delta method can be used to directly derive the asymptotic properties of the value function process for binary constrained optimal allocation problems, as well as the two-step ROC curve estimator. Moreover, leveraging profound insights from geometric functional analysis on convex and local Lipschitz functionals, we obtain additional generic Fr\'echet differentiability results for the value functions of optimal allocation problems. These compelling findings moti
    
[^2]: 通过迭代后验抽样实现随机定位

    Stochastic Localization via Iterative Posterior Sampling

    [https://arxiv.org/abs/2402.10758](https://arxiv.org/abs/2402.10758)

    本论文提出了一种名为SLIPS的方法，通过迭代后验抽样实现随机定位，填补了从非标准化目标密度中抽样的问题的空白。

    

    建立在基于得分学习的基础上，近期对随机定位技术产生了新的兴趣。在这些模型中，人们通过随机过程（称为观测过程）为数据分布中的样本引入噪声，并逐渐学习与该动力学关联的去噪器。除了特定应用之外，对于从非标准化目标密度中抽样的问题，对随机定位的使用尚未得到广泛探讨。本项工作旨在填补这一空白。我们考虑了一个通用的随机定位框架，并引入了一类明确的观测过程，与灵活的去噪时间表相关联。我们提供了一种完整的方法论，即“通过迭代后验抽样实现随机定位”（SLIPS），以获得该动力学的近似样本，并作为副产品，样本来自目标分布。我们的方案基于马尔可夫链蒙特卡洛估计。

    arXiv:2402.10758v1 Announce Type: cross  Abstract: Building upon score-based learning, new interest in stochastic localization techniques has recently emerged. In these models, one seeks to noise a sample from the data distribution through a stochastic process, called observation process, and progressively learns a denoiser associated to this dynamics. Apart from specific applications, the use of stochastic localization for the problem of sampling from an unnormalized target density has not been explored extensively. This work contributes to fill this gap. We consider a general stochastic localization framework and introduce an explicit class of observation processes, associated with flexible denoising schedules. We provide a complete methodology, $\textit{Stochastic Localization via Iterative Posterior Sampling}$ (SLIPS), to obtain approximate samples of this dynamics, and as a by-product, samples from the target distribution. Our scheme is based on a Markov chain Monte Carlo estimati
    
[^3]: 动态协变量平衡：基于潜在局部投影的治疗效果随时间估计

    Dynamic covariate balancing: estimating treatment effects over time with potential local projections. (arXiv:2103.01280v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2103.01280](http://arxiv.org/abs/2103.01280)

    本文提出了一种通过动态协变量平衡方法，基于过去历史上潜在结果期望的局部投影，估计面板数据中动态变化的治疗效果，并考虑结果和时间变化的协变量与治疗轨迹的关系以及治疗效应的异质性。研究结果表明该方法具有良好的渐近性质和数值特性，在实证应用中具有优势。

    

    本文研究了面板数据中治疗历史的估计和推断问题，特别是在治疗在时间上动态变化的情况下。我们提出了一种方法，允许治疗根据高维协变量、过去的结果和治疗动态分配，同时考虑结果和时间变化的协变量与治疗轨迹的关系，以及治疗效应的异质性。我们的方法通过在过去历史上对潜在结果期望进行递归投影，然后通过平衡动态可观测特征来控制偏差。我们研究了估计量的渐近性质和数值特性，并在实证应用中展示了该方法的优势。

    This paper studies the estimation and inference of treatment histories in panel data settings when treatments change dynamically over time.  We propose a method that allows for (i) treatments to be assigned dynamically over time based on high-dimensional covariates, past outcomes and treatments; (ii) outcomes and time-varying covariates to depend on treatment trajectories; (iii) heterogeneity of treatment effects.  Our approach recursively projects potential outcomes' expectations on past histories. It then controls the bias by balancing dynamically observable characteristics. We study the asymptotic and numerical properties of the estimator and illustrate the benefits of the procedure in an empirical application.
    

