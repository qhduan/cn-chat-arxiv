# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributional Off-policy Evaluation with Bellman Residual Minimization](https://arxiv.org/abs/2402.01900) | 这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。 |
| [^2] | [On the Generalization Properties of Diffusion Models.](http://arxiv.org/abs/2311.01797) | 本文对扩散模型的泛化属性进行了理论研究，建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，并在停止训练时可以避免维度诅咒。进一步将定量分析扩展到了数据依赖的情景。 |
| [^3] | [FaiREE: Fair Classification with Finite-Sample and Distribution-Free Guarantee.](http://arxiv.org/abs/2211.15072) | 本研究提出了FaiREE算法，它是一种可满足群体公平性约束的公平分类算法，并且具有有限样本和无分布理论保证。在实验中表现优异。 |

# 详细

[^1]: 使用Bellman残差最小化的分布式离线策略评估

    Distributional Off-policy Evaluation with Bellman Residual Minimization

    [https://arxiv.org/abs/2402.01900](https://arxiv.org/abs/2402.01900)

    这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。

    

    我们考虑分布式离线策略评估的问题，它是许多分布式强化学习（DRL）算法的基础。与大多数现有的方法（依赖于最大值-扩展的统计距离，如最大值Wasserstein距离）不同，我们研究用于量化分布式Bellman残差的期望-扩展的统计距离，并且证明它可以上界估计返回分布的期望误差。基于这个有吸引力的性质，通过将Bellman残差最小化框架推广到DRL，我们提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。我们在可实现性假设下建立了EBRM估计器的有限样本误差界。此外，我们引入了一种基于多步引导过程的方法的变体，以实现多步扩展。通过选择适当的步长，我们获得了更好的误差界。

    We consider the problem of distributional off-policy evaluation which serves as the foundation of many distributional reinforcement learning (DRL) algorithms. In contrast to most existing works (that rely on supremum-extended statistical distances such as supremum-Wasserstein distance), we study the expectation-extended statistical distance for quantifying the distributional Bellman residuals and show that it can upper bound the expected error of estimating the return distribution. Based on this appealing property, by extending the framework of Bellman residual minimization to DRL, we propose a method called Energy Bellman Residual Minimizer (EBRM) to estimate the return distribution. We establish a finite-sample error bound for the EBRM estimator under the realizability assumption. Furthermore, we introduce a variant of our method based on a multi-step bootstrapping procedure to enable multi-step extension. By selecting an appropriate step level, we obtain a better error bound for thi
    
[^2]: 关于扩散模型的泛化属性

    On the Generalization Properties of Diffusion Models. (arXiv:2311.01797v1 [cs.LG])

    [http://arxiv.org/abs/2311.01797](http://arxiv.org/abs/2311.01797)

    本文对扩散模型的泛化属性进行了理论研究，建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，并在停止训练时可以避免维度诅咒。进一步将定量分析扩展到了数据依赖的情景。

    

    扩散模型是一类生成模型，用于建立一个随机传输映射，将经验观测到的但未知的目标分布与已知的先验分布联系起来。尽管在实际应用中取得了显著的成功，但对其泛化能力的理论理解仍未充分发展。本文对扩散模型的泛化属性进行了全面的理论研究。我们建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，表明在样本大小$n$和模型容量$m$上都存在多项式小的泛化误差($O(n^{-2/5}+m^{-4/5})$)，在停止训练时可以避免维度诅咒（即数据维度不呈指数级增长）。此外，我们将定量分析扩展到了一个数据依赖的情景，其中目标分布被描绘为一系列的概率密度函数。

    Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error ($O(n^{-2/5}+m^{-4/5})$) on both the sample size $n$ and the model capacity $m$, evading the curse of dimensionality (i.e., not exponentially large in the data dimension) when early-stopped. Furthermore, we extend our quantitative analysis to a data-dependent scenario, wherein target distributions are portrayed as a succession of densities with progr
    
[^3]: FaiREE：具有有限样本和无分布保证的公平分类算法

    FaiREE: Fair Classification with Finite-Sample and Distribution-Free Guarantee. (arXiv:2211.15072v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.15072](http://arxiv.org/abs/2211.15072)

    本研究提出了FaiREE算法，它是一种可满足群体公平性约束的公平分类算法，并且具有有限样本和无分布理论保证。在实验中表现优异。

    

    算法公平性在机器学习研究中发挥着越来越重要的作用。已经提出了几种群体公平性概念和算法。然而，现有公平分类方法的公平保证主要依赖于特定的数据分布假设，通常需要大样本量，并且在样本量较小的情况下可能会违反公平性，而这在实践中经常发生。本文提出了FaiREE算法，它是一种公平分类算法，可以在有限样本和无分布理论保证下满足群体公平性约束。FaiREE可以适应各种群体公平性概念（例如，机会平等，平衡几率，人口统计学平衡等）并实现最佳准确性。这些理论保证进一步得到了对合成和实际数据的实验支持。FaiREE表现出比最先进的算法更好的性能。

    Algorithmic fairness plays an increasingly critical role in machine learning research. Several group fairness notions and algorithms have been proposed. However, the fairness guarantee of existing fair classification methods mainly depends on specific data distributional assumptions, often requiring large sample sizes, and fairness could be violated when there is a modest number of samples, which is often the case in practice. In this paper, we propose FaiREE, a fair classification algorithm that can satisfy group fairness constraints with finite-sample and distribution-free theoretical guarantees. FaiREE can be adapted to satisfy various group fairness notions (e.g., Equality of Opportunity, Equalized Odds, Demographic Parity, etc.) and achieve the optimal accuracy. These theoretical guarantees are further supported by experiments on both synthetic and real data. FaiREE is shown to have favorable performance over state-of-the-art algorithms.
    

