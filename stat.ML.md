# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [All you need is log](https://arxiv.org/abs/2606.27349) | 本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。 |
| [^2] | [Scalable Bayesian inference for the generalized linear mixed model](https://arxiv.org/abs/2403.03007) | 该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。 |
| [^3] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^4] | [Temporal-spatial model via Trend Filtering.](http://arxiv.org/abs/2308.16172) | 本研究通过趋势滤波方法对具有时空依赖性的数据进行了非参数回归函数的估计，研究了该方法在单变量和多变量情况下的应用，并验证了其极小化性。研究发现了以往未曾探索的独特相变现象，并通过仿真和实际数据应用验证了方法的优越性能。 |

# 详细

[^1]: 你只需要对数

    All you need is log

    [https://arxiv.org/abs/2606.27349](https://arxiv.org/abs/2606.27349)

    本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。

    

    arXiv:2606.27349v1 公告类型：交叉 摘要：比较两个概率分布是统计学和机器学习的基本构建块，而正确的族已被充分理解：阶数为α∈[0,∞]的Rényi散度是在数据处理下单调且在独立乘积上可加的唯一族。许多问题却需要同时比较两个以上的分布——多群体公平性、多先验PAC-Bayes界、多假设检验——而Rényi族的多分布泛化正确形式一直是一个开放问题。我们对此进行了刻画。每个在数据处理下单调且在独立乘积上可加的W元分布泛函，都可以表示为多路重合散度C_α(π_1,…,π_W) := -log∫ π_1^{α_1}…π_W^{α_W}（其中∑_k α_k = 1）在具有四个分层参数空间上的正积分：单纯形内部；混合符号指数锥。

    arXiv:2606.27349v1 Announce Type: cross  Abstract: Comparing two probability distributions is a basic building block of statistics and machine learning, and the right family is well understood: the R\'enyi divergences of order $\alpha\in[0,\infty]$ are the unique family monotone under data processing and additive on independent products. Many problems instead compare more than two distributions at once -- multi-population fairness, multi-prior PAC-Bayes bounds, multi-hypothesis testing -- and the right multi-distribution generalization of the R\'enyi family has been an open question.   We characterize it. Every functional of $W$-tuples of distributions that is monotone under data processing and additive on independent products is a positive integral of multi-way coincidence divergences $C_{\alpha}(\pi_1,\dots,\pi_W) := -\log\int \pi_1^{\alpha_1}\cdots\pi_W^{\alpha_W}$ (with $\sum_k \alpha_k = 1$) over a parameter space with four strata: the simplex interior; mixed-sign exponent cones (
    
[^2]: 通用线性混合模型的可扩展贝叶斯推断

    Scalable Bayesian inference for the generalized linear mixed model

    [https://arxiv.org/abs/2403.03007](https://arxiv.org/abs/2403.03007)

    该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。

    

    通用线性混合模型（GLMM）是处理相关数据的一种流行统计方法，在包括生物医学数据等大数据常见的应用领域被广泛使用。本文的重点是针对GLMM的可扩展统计推断，我们将统计推断定义为：（i）对总体参数的估计以及（ii）在存在不确定性的情况下评估科学假设。人工智能（AI）学习算法擅长可扩展的统计估计，但很少包括不确定性量化。相比之下，贝叶斯推断提供完整的统计推断，因为不确定性量化自动来自后验分布。不幸的是，包括马尔可夫链蒙特卡洛（MCMC）在内的贝叶斯推断算法在大数据环境中变得难以计算。在本文中，我们介绍了一个统计推断算法

    arXiv:2403.03007v1 Announce Type: cross  Abstract: The generalized linear mixed model (GLMM) is a popular statistical approach for handling correlated data, and is used extensively in applications areas where big data is common, including biomedical data settings. The focus of this paper is scalable statistical inference for the GLMM, where we define statistical inference as: (i) estimation of population parameters, and (ii) evaluation of scientific hypotheses in the presence of uncertainty. Artificial intelligence (AI) learning algorithms excel at scalable statistical estimation, but rarely include uncertainty quantification. In contrast, Bayesian inference provides full statistical inference, since uncertainty quantification results automatically from the posterior distribution. Unfortunately, Bayesian inference algorithms, including Markov Chain Monte Carlo (MCMC), become computationally intractable in big data settings. In this paper, we introduce a statistical inference algorithm 
    
[^3]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^4]: 通过趋势滤波进行时空模型建模

    Temporal-spatial model via Trend Filtering. (arXiv:2308.16172v1 [stat.ME])

    [http://arxiv.org/abs/2308.16172](http://arxiv.org/abs/2308.16172)

    本研究通过趋势滤波方法对具有时空依赖性的数据进行了非参数回归函数的估计，研究了该方法在单变量和多变量情况下的应用，并验证了其极小化性。研究发现了以往未曾探索的独特相变现象，并通过仿真和实际数据应用验证了方法的优越性能。

    

    本研究侧重于对具有同时时间和空间依赖性的数据进行非参数回归函数的估计。在这种情况下，我们研究了趋势滤波，这是一种非参数估计方法，由Mammen和Rudin提出。在单变量设置中，我们考虑的信号假设具有有界总变异度的k次弱导数，允许一定程度的平滑性。在多变量情况下，我们研究了Padilla等人的K最近邻融合套索估计器，采用适用于具有有界变异度且符合分段利普希茨连续性准则的信号的ADMM算法。通过与下界对齐，我们验证了我们估计器的极小化性。通过分析，我们发现了以往趋势滤波研究中未曾探索过的独特相变现象。仿真研究和实际数据应用都突出了我们方法的出色性能。

    This research focuses on the estimation of a non-parametric regression function designed for data with simultaneous time and space dependencies. In such a context, we study the Trend Filtering, a nonparametric estimator introduced by \cite{mammen1997locally} and \cite{rudin1992nonlinear}. For univariate settings, the signals we consider are assumed to have a kth weak derivative with bounded total variation, allowing for a general degree of smoothness. In the multivariate scenario, we study a $K$-Nearest Neighbor fused lasso estimator as in \cite{padilla2018adaptive}, employing an ADMM algorithm, suitable for signals with bounded variation that adhere to a piecewise Lipschitz continuity criterion. By aligning with lower bounds, the minimax optimality of our estimators is validated. A unique phase transition phenomenon, previously uncharted in Trend Filtering studies, emerges through our analysis. Both Simulation studies and real data applications underscore the superior performance of o
    

