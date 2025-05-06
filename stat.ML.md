# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multivariate Gaussian Approximation for Random Forest via Region-based Stabilization](https://arxiv.org/abs/2403.09960) | 该论文通过基于区域稳定性的方法，推导出了随机森林预测的高斯逼近界限，并建立了适用于各种相关统计问题的概率结果。 |
| [^2] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^3] | [Conformal Predictive Programming for Chance Constrained Optimization](https://arxiv.org/abs/2402.07407) | 可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。 |
| [^4] | [Robust Transfer Learning with Unreliable Source Data.](http://arxiv.org/abs/2310.04606) | 本文提出了一个新的鲁棒性迁移学习方法TAB模型，通过衡量目标与源回归函数之间的模糊度水平来改善分类任务，并避免负迁移。通过实验验证，TAB模型在非参数分类和逻辑回归任务上表现出了优越的性能。 |
| [^5] | [Bayesian Analysis for Over-parameterized Linear Model without Sparsity.](http://arxiv.org/abs/2305.15754) | 本文提出了一种基于数据的特征向量的先验方法，用于处理非稀疏超参数线性模型。从导出的后验分布收缩率和开发的截断高斯近似两个方面来证明了该方法的有效性，可以解决之前的先验稀疏性限制。 |
| [^6] | [Microcanonical Langevin Monte Carlo.](http://arxiv.org/abs/2303.18221) | 我们提出的微正则 Langevin Monte Carlo 方法能够高效地采样 $\exp[-S(\x)]$ 分布，同时具有无偏性。 |

# 详细

[^1]: 通过基于区域稳定性的多元高斯逼近改进随机森林

    Multivariate Gaussian Approximation for Random Forest via Region-based Stabilization

    [https://arxiv.org/abs/2403.09960](https://arxiv.org/abs/2403.09960)

    该论文通过基于区域稳定性的方法，推导出了随机森林预测的高斯逼近界限，并建立了适用于各种相关统计问题的概率结果。

    

    我们在给定由泊松过程产生的一组训练点的情况下，推导了随机森林预测的高斯逼近界限，假设数据生成过程存在相当温和的正则性假设。我们的方法基于一个关键观察：随机森林的预测满足一定的称为基于区域稳定性的几何属性。在为随机森林开发结果的过程中，我们还为基于区域稳定的泊松过程的一般泛函建立了一个概率结果，这可能是独立感兴趣的。这一普遍结果利用了Malliavin-Stein方法，并且可能适用于各种相关的统计问题。

    arXiv:2403.09960v1 Announce Type: cross  Abstract: We derive Gaussian approximation bounds for random forest predictions based on a set of training points given by a Poisson process, under fairly mild regularity assumptions on the data generating process. Our approach is based on the key observation that the random forest predictions satisfy a certain geometric property called region-based stabilization. In the process of developing our results for the random forest, we also establish a probabilistic result, which might be of independent interest, on multivariate Gaussian approximation bounds for general functionals of Poisson process that are region-based stabilizing. This general result makes use of the Malliavin-Stein method, and is potentially applicable to various related statistical problems.
    
[^2]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^3]: 可容许预测规划用于机遇受限优化

    Conformal Predictive Programming for Chance Constrained Optimization

    [https://arxiv.org/abs/2402.07407](https://arxiv.org/abs/2402.07407)

    可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。

    

    在对预测规划（CP）的进展的激励下，我们提出了可容许预测规划（CPP），一种解决机遇受限优化（CCO）问题的方法，即受任意随机参数影响的非线性约束函数的优化问题。CPP利用这些随机参数的样本以及量子引理（CP的核心）将CCO问题转化为确定性优化问题。然后，我们通过：（1）将量子表示为线性规划以及其KKT条件（CPP-KKT）；（2）使用混合整数规划（CPP-MIP）来呈现CPP的两种易于处理的改进。CPP具备对CCO问题进行边际概率可行性保证，这与现有方法（例如样本逼近和场景方法）在概念上有所不同。尽管我们探讨了与样本逼近方法的算法相似之处，但我们强调CPP的优势在于易于扩展。

    Motivated by the advances in conformal prediction (CP), we propose conformal predictive programming (CPP), an approach to solve chance constrained optimization (CCO) problems, i.e., optimization problems with nonlinear constraint functions affected by arbitrary random parameters. CPP utilizes samples from these random parameters along with the quantile lemma -- which is central to CP -- to transform the CCO problem into a deterministic optimization problem. We then present two tractable reformulations of CPP by: (1) writing the quantile as a linear program along with its KKT conditions (CPP-KKT), and (2) using mixed integer programming (CPP-MIP). CPP comes with marginal probabilistic feasibility guarantees for the CCO problem that are conceptually different from existing approaches, e.g., the sample approximation and the scenario approach. While we explore algorithmic similarities with the sample approximation approach, we emphasize that the strength of CPP is that it can easily be ext
    
[^4]: 具有不可靠源数据的鲁棒性迁移学习

    Robust Transfer Learning with Unreliable Source Data. (arXiv:2310.04606v1 [stat.ML])

    [http://arxiv.org/abs/2310.04606](http://arxiv.org/abs/2310.04606)

    本文提出了一个新的鲁棒性迁移学习方法TAB模型，通过衡量目标与源回归函数之间的模糊度水平来改善分类任务，并避免负迁移。通过实验验证，TAB模型在非参数分类和逻辑回归任务上表现出了优越的性能。

    

    本文针对鲁棒性迁移学习中的贝叶斯分类器的模糊性和目标与源分布之间的弱可转移信号所带来的挑战进行了研究。我们引入了一种新的量，称为“模糊度水平”，用于衡量目标与源回归函数之间的差异，并提出了一种简单的迁移学习方法，并建立了一个一般定理，说明了这个新量与学习的迁移性在风险改善方面的关系。我们提出的“边界周围转移”(Transfer Around Boundary, TAB)模型通过在目标数据和源数据性能之间进行平衡的阈值，既高效又鲁棒，能够改善分类并避免负迁移。此外，我们还展示了TAB模型在非参数分类和逻辑回归任务上的有效性，达到了最优的上界，只有对数因子的差距。通过仿真研究进一步支持了TAB的有效性。

    This paper addresses challenges in robust transfer learning stemming from ambiguity in Bayes classifiers and weak transferable signals between the target and source distribution. We introduce a novel quantity called the ''ambiguity level'' that measures the discrepancy between the target and source regression functions, propose a simple transfer learning procedure, and establish a general theorem that shows how this new quantity is related to the transferability of learning in terms of risk improvements. Our proposed ''Transfer Around Boundary'' (TAB) model, with a threshold balancing the performance of target and source data, is shown to be both efficient and robust, improving classification while avoiding negative transfer. Moreover, we demonstrate the effectiveness of the TAB model on non-parametric classification and logistic regression tasks, achieving upper bounds which are optimal up to logarithmic factors. Simulation studies lend further support to the effectiveness of TAB. We 
    
[^5]: 非稀疏超参数线性模型的贝叶斯分析

    Bayesian Analysis for Over-parameterized Linear Model without Sparsity. (arXiv:2305.15754v1 [math.ST])

    [http://arxiv.org/abs/2305.15754](http://arxiv.org/abs/2305.15754)

    本文提出了一种基于数据的特征向量的先验方法，用于处理非稀疏超参数线性模型。从导出的后验分布收缩率和开发的截断高斯近似两个方面来证明了该方法的有效性，可以解决之前的先验稀疏性限制。

    

    在高维贝叶斯统计学中，发展了许多方法，包括许多先验分布，它们导致估计参数的稀疏性。然而，这种先验在处理数据的谱特征向量结构方面有局限性，因此不适用于分析最近发展的不假设稀疏性的高维线性模型。本文介绍了一种贝叶斯方法，它使用一个依赖于数据协方差矩阵的特征向量的先验，但不会引起参数的稀疏性。我们还提供了导出的后验分布的收缩率，并开发了后验分布的截断高斯近似。前者证明了后验估计的效率，而后者则使用Bernstein-von Mises类型方法来量化参数不确定性。这些结果表明，任何能够处理谱特征向量的贝叶斯方法，都可以用于非稀疏超参数线性模型分析，从而解决了先前的限制。

    In high-dimensional Bayesian statistics, several methods have been developed, including many prior distributions that lead to the sparsity of estimated parameters. However, such priors have limitations in handling the spectral eigenvector structure of data, and as a result, they are ill-suited for analyzing over-parameterized models (high-dimensional linear models that do not assume sparsity) that have been developed in recent years. This paper introduces a Bayesian approach that uses a prior dependent on the eigenvectors of data covariance matrices, but does not induce the sparsity of parameters. We also provide contraction rates of derived posterior distributions and develop a truncated Gaussian approximation of the posterior distribution. The former demonstrates the efficiency of posterior estimation, while the latter enables quantification of parameter uncertainty using a Bernstein-von Mises-type approach. These results indicate that any Bayesian method that can handle the spectrum
    
[^6]: 微正则 Langevin Monte Carlo

    Microcanonical Langevin Monte Carlo. (arXiv:2303.18221v1 [hep-lat])

    [http://arxiv.org/abs/2303.18221](http://arxiv.org/abs/2303.18221)

    我们提出的微正则 Langevin Monte Carlo 方法能够高效地采样 $\exp[-S(\x)]$ 分布，同时具有无偏性。

    

    我们提出了一种方法，用于以可用渐变 $ \nabla S(\x)$ 的形式采样自一任意分布 $ \exp[-S(\x)]$，该方法被制定为保持能量的随机微分方程（SDE）。我们推导出 Fokker-Planck 方程，并证明确定性漂移和随机扩散分别保持平稳分布。这意味着漂移扩散离散化方案无偏，而标准 Langevin 动力学则不是。我们将该方法应用于 $\phi^4$ 晶格场论，展示了结果与标准采样方法一致，但比当前最先进的采样器效率显著提高。

    We propose a method for sampling from an arbitrary distribution $\exp[-S(\x)]$ with an available gradient $\nabla S(\x)$, formulated as an energy-preserving stochastic differential equation (SDE). We derive the Fokker-Planck equation and show that both the deterministic drift and the stochastic diffusion separately preserve the stationary distribution. This implies that the drift-diffusion discretization schemes are bias-free, in contrast to the standard Langevin dynamics. We apply the method to the $\phi^4$ lattice field theory, showing the results agree with the standard sampling methods but with significantly higher efficiency compared to the current state-of-the-art samplers.
    

