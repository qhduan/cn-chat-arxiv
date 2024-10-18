# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Moments of Clarity: Streamlining Latent Spaces in Machine Learning using Moment Pooling](https://arxiv.org/abs/2403.08854) | 提出了一种称为Moment Pooling的新方法，通过将Deep Sets中的求和泛化为任意的多变量矩，显著降低机器学习网络的潜在空间维度，在固定的潜在维度下实现更高的有效潜在维度，从而可以直接可视化和解释内部表示。 |
| [^2] | [Distributional Off-policy Evaluation with Bellman Residual Minimization](https://arxiv.org/abs/2402.01900) | 这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。 |
| [^3] | [Stage-Aware Learning for Dynamic Treatments.](http://arxiv.org/abs/2310.19300) | 本论文提出了一种针对动态治疗的阶段感知学习方法，该方法通过估计DTR并优先考虑治疗轨迹与最佳治疗方案在决策阶段上的一致性，在提高样本效率和稳定性方面取得了重要进展。 |
| [^4] | [Concentration for high-dimensional linear processes with dependent innovations.](http://arxiv.org/abs/2307.12395) | 本论文提出了一种针对高维线性过程的具有相关创新的集中度不等式，并利用该不等式获得了线性过程滞后自协方差矩阵最大分量范数的集中度界限。这些结果在估计高维向量自回归过程、时间序列引导和长期协方差矩阵估计中具有重要应用价值。 |
| [^5] | [Online-to-PAC Conversions: Generalization Bounds via Regret Analysis.](http://arxiv.org/abs/2305.19674) | 本文提出了在线学习游戏“泛化游戏”的框架，将在线学习算法的表现和统计学习算法的泛化界限联系了起来，并得出了一些标准的泛化限制。 |
| [^6] | [Orthogonal polynomial approximation and Extended Dynamic Mode Decomposition in chaos.](http://arxiv.org/abs/2305.08074) | 本文在简单的混沌映射上证明了扩展动态模态分解（EDMD）对于多项式可观测字典有指数效率，从而有效处理了混沌动力学中的正则函数问题，并展示了在这种情况下使用EDMD产生的预测和Koopman谱数据收敛至物理上有意义的极限。 |
| [^7] | [FAStEN: an efficient adaptive method for feature selection and estimation in high-dimensional functional regressions.](http://arxiv.org/abs/2303.14801) | 提出了一种新的自适应方法FAStEN，用于在高维函数回归问题中执行特征选择和参数估计，通过利用函数主成分和对偶增广Lagrangian问题的稀疏性质，具有显著的计算效率和选择准确性。 |

# 详细

[^1]: 清晰瞬间：使用Moment Pooling简化机器学习中的潜在空间

    Moments of Clarity: Streamlining Latent Spaces in Machine Learning using Moment Pooling

    [https://arxiv.org/abs/2403.08854](https://arxiv.org/abs/2403.08854)

    提出了一种称为Moment Pooling的新方法，通过将Deep Sets中的求和泛化为任意的多变量矩，显著降低机器学习网络的潜在空间维度，在固定的潜在维度下实现更高的有效潜在维度，从而可以直接可视化和解释内部表示。

    

    许多机器学习应用涉及学习数据的潜在表示，通常是高维且难以直接解释。在这项工作中，我们提出了“Moment Pooling”，这是Deep Sets网络的一个自然延伸，可大幅减少这些网络的潜在空间维度，同时维持甚至提高性能。Moment Pooling将Deep Sets中的求和泛化为任意的多变量矩，使模型能够在固定的潜在维度下实现更高的有效潜在维度。我们将Moment Pooling应用于夸克/胶子喷注分类的对撞机物理任务，通过将Energy Flow Networks（EFNs）扩展为Moment EFNs。我们发现，具有小至1的潜在维度的Moment EFNs表现与具有较高潜在维度的普通EFNs类似。这种小潜在维度使内部表示可以直接可视化和解释。

    arXiv:2403.08854v1 Announce Type: cross  Abstract: Many machine learning applications involve learning a latent representation of data, which is often high-dimensional and difficult to directly interpret. In this work, we propose "Moment Pooling", a natural extension of Deep Sets networks which drastically decrease latent space dimensionality of these networks while maintaining or even improving performance. Moment Pooling generalizes the summation in Deep Sets to arbitrary multivariate moments, which enables the model to achieve a much higher effective latent dimensionality for a fixed latent dimension. We demonstrate Moment Pooling on the collider physics task of quark/gluon jet classification by extending Energy Flow Networks (EFNs) to Moment EFNs. We find that Moment EFNs with latent dimensions as small as 1 perform similarly to ordinary EFNs with higher latent dimension. This small latent dimension allows for the internal representation to be directly visualized and interpreted, w
    
[^2]: 使用Bellman残差最小化的分布式离线策略评估

    Distributional Off-policy Evaluation with Bellman Residual Minimization

    [https://arxiv.org/abs/2402.01900](https://arxiv.org/abs/2402.01900)

    这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。

    

    我们考虑分布式离线策略评估的问题，它是许多分布式强化学习（DRL）算法的基础。与大多数现有的方法（依赖于最大值-扩展的统计距离，如最大值Wasserstein距离）不同，我们研究用于量化分布式Bellman残差的期望-扩展的统计距离，并且证明它可以上界估计返回分布的期望误差。基于这个有吸引力的性质，通过将Bellman残差最小化框架推广到DRL，我们提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。我们在可实现性假设下建立了EBRM估计器的有限样本误差界。此外，我们引入了一种基于多步引导过程的方法的变体，以实现多步扩展。通过选择适当的步长，我们获得了更好的误差界。

    We consider the problem of distributional off-policy evaluation which serves as the foundation of many distributional reinforcement learning (DRL) algorithms. In contrast to most existing works (that rely on supremum-extended statistical distances such as supremum-Wasserstein distance), we study the expectation-extended statistical distance for quantifying the distributional Bellman residuals and show that it can upper bound the expected error of estimating the return distribution. Based on this appealing property, by extending the framework of Bellman residual minimization to DRL, we propose a method called Energy Bellman Residual Minimizer (EBRM) to estimate the return distribution. We establish a finite-sample error bound for the EBRM estimator under the realizability assumption. Furthermore, we introduce a variant of our method based on a multi-step bootstrapping procedure to enable multi-step extension. By selecting an appropriate step level, we obtain a better error bound for thi
    
[^3]: 针对动态治疗的阶段感知学习

    Stage-Aware Learning for Dynamic Treatments. (arXiv:2310.19300v1 [stat.ML])

    [http://arxiv.org/abs/2310.19300](http://arxiv.org/abs/2310.19300)

    本论文提出了一种针对动态治疗的阶段感知学习方法，该方法通过估计DTR并优先考虑治疗轨迹与最佳治疗方案在决策阶段上的一致性，在提高样本效率和稳定性方面取得了重要进展。

    

    最近对动态治疗方案（DTRs）的研究取得了重要进展，提出了强大的优化治疗搜索算法，根据个体具体需求量身定制，并能最大化其预期的临床效益。然而，现有算法在优化治疗下可能会受到样本量不足的困扰，尤其是在涉及长时间决策阶段的慢性疾病中。为了解决这些挑战，我们提出了一种新颖的个体化学习方法，重点是估计DTR，并优先考虑观察到的治疗轨迹与最佳治疗方案在决策阶段上的一致性。通过放宽观察到的轨迹必须完全与最佳治疗一致的限制，我们的方法大大提高了基于倒数概率加权方法的样本效率和稳定性。具体而言，所提出的学习方案构建了一个更通用的框架，包括了流行的结果加权学习框架。

    Recent advances in dynamic treatment regimes (DTRs) provide powerful optimal treatment searching algorithms, which are tailored to individuals' specific needs and able to maximize their expected clinical benefits. However, existing algorithms could suffer from insufficient sample size under optimal treatments, especially for chronic diseases involving long stages of decision-making. To address these challenges, we propose a novel individualized learning method which estimates the DTR with a focus on prioritizing alignment between the observed treatment trajectory and the one obtained by the optimal regime across decision stages. By relaxing the restriction that the observed trajectory must be fully aligned with the optimal treatments, our approach substantially improves the sample efficiency and stability of inverse probability weighted based methods. In particular, the proposed learning scheme builds a more general framework which includes the popular outcome weighted learning framewo
    
[^4]: 高维线性过程中具有相关创新的集中度

    Concentration for high-dimensional linear processes with dependent innovations. (arXiv:2307.12395v1 [math.ST])

    [http://arxiv.org/abs/2307.12395](http://arxiv.org/abs/2307.12395)

    本论文提出了一种针对高维线性过程的具有相关创新的集中度不等式，并利用该不等式获得了线性过程滞后自协方差矩阵最大分量范数的集中度界限。这些结果在估计高维向量自回归过程、时间序列引导和长期协方差矩阵估计中具有重要应用价值。

    

    我们针对具有子韦布尔尾的混合序列上的线性过程的$l_\infty$范数开发了集中不等式。这些不等式利用了Beveridge-Nelson分解，将问题简化为向量混合序列或其加权和的上确界范数的集中度。这个不等式用于得到线性过程的滞后$h$自协方差矩阵的最大分量范数的集中度界限。这些结果对于使用$l_1$正则化估计的高维向量自回归过程的估计界限、时间序列的高维高斯引导、以及长期协方差矩阵估计非常有用。

    We develop concentration inequalities for the $l_\infty$ norm of a vector linear processes on mixingale sequences with sub-Weibull tails. These inequalities make use of the Beveridge-Nelson decomposition, which reduces the problem to concentration for sup-norm of a vector-mixingale or its weighted sum. This inequality is used to obtain a concentration bound for the maximum entrywise norm of the lag-$h$ autocovariance matrices of linear processes. These results are useful for estimation bounds for high-dimensional vector-autoregressive processes estimated using $l_1$ regularisation, high-dimensional Gaussian bootstrap for time series, and long-run covariance matrix estimation.
    
[^5]: 在线到PAC的转换: 通过遗憾分析得出泛化界限

    Online-to-PAC Conversions: Generalization Bounds via Regret Analysis. (arXiv:2305.19674v1 [stat.ML])

    [http://arxiv.org/abs/2305.19674](http://arxiv.org/abs/2305.19674)

    本文提出了在线学习游戏“泛化游戏”的框架，将在线学习算法的表现和统计学习算法的泛化界限联系了起来，并得出了一些标准的泛化限制。

    

    我们提出了一个新的框架，通过在线学习的视角推导出统计学习算法的泛化界限。具体而言，我们构建了一个在线学习游戏称为“泛化游戏”，其中在线学习器试图与固定的统计学习算法竞争，预测独立同分布数据点训练集上的泛化间隙序列。我们通过展示在这个游戏中存在有界遗憾的在线学习算法与统计学习设置之间的联系来建立这种关联，这意味着统计学习算法的泛化错误存在一个界限，直到与统计学习方法的复杂性无关的鞅浓度项。这种技术允许我们恢复几个标准的泛化限制，包括一系列的PAC-Bayesian保证和信息理论保证，以及它们的推广。

    We present a new framework for deriving bounds on the generalization bound of statistical learning algorithms from the perspective of online learning. Specifically, we construct an online learning game called the "generalization game", where an online learner is trying to compete with a fixed statistical learning algorithm in predicting the sequence of generalization gaps on a training set of i.i.d. data points. We establish a connection between the online and statistical learning setting by showing that the existence of an online learning algorithm with bounded regret in this game implies a bound on the generalization error of the statistical learning algorithm, up to a martingale concentration term that is independent of the complexity of the statistical learning method. This technique allows us to recover several standard generalization bounds including a range of PAC-Bayesian and information-theoretic guarantees, as well as generalizations thereof.
    
[^6]: 正交多项式逼近和扩展动态模态分解在混沌中的应用

    Orthogonal polynomial approximation and Extended Dynamic Mode Decomposition in chaos. (arXiv:2305.08074v1 [math.NA])

    [http://arxiv.org/abs/2305.08074](http://arxiv.org/abs/2305.08074)

    本文在简单的混沌映射上证明了扩展动态模态分解（EDMD）对于多项式可观测字典有指数效率，从而有效处理了混沌动力学中的正则函数问题，并展示了在这种情况下使用EDMD产生的预测和Koopman谱数据收敛至物理上有意义的极限。

    

    扩展动态模态分解（EDMD）是一种数据驱动的工具，用于动态的预测和模型简化，在物理科学领域得到广泛应用。虽然这种方法在概念上很简单，但在确定性混沌中，它的性质或者它的收敛性还不清楚。特别是，EDMD的最小二乘逼近如何处理需要描绘混沌动力学含义的正则函数的类别，这也是不清楚的。本文在分析上简单的一个圆环展开映射的最简单例子上，发展了关于EDMD的一般的、严格的理论。证明了一个新的关于在单位圆上的正交多项式（OPUC）的理论结果，我们证明在无限数据极限时，针对多项式的可观测字典的最小二乘投影具有指数效率。因此，我们展示了在这种情况下使用EDMD产生的预测和Koopman谱数据收敛到物理上有意义的极限的指数速率。

    Extended Dynamic Mode Decomposition (EDMD) is a data-driven tool for forecasting and model reduction of dynamics, which has been extensively taken up in the physical sciences. While the method is conceptually simple, in deterministic chaos it is unclear what its properties are or even what it converges to. In particular, it is not clear how EDMD's least-squares approximation treats the classes of regular functions needed to make sense of chaotic dynamics.  In this paper we develop a general, rigorous theory of EDMD on the simplest examples of chaotic maps: analytic expanding maps of the circle. Proving a new result in the theory of orthogonal polynomials on the unit circle (OPUC), we show that in the infinite-data limit, the least-squares projection is exponentially efficient for polynomial observable dictionaries. As a result, we show that the forecasts and Koopman spectral data produced using EDMD in this setting converge to the physically meaningful limits, at an exponential rate.  
    
[^7]: 高维函数回归中特征选择和估计的一种高效自适应方法--FAStEN

    FAStEN: an efficient adaptive method for feature selection and estimation in high-dimensional functional regressions. (arXiv:2303.14801v1 [stat.ME])

    [http://arxiv.org/abs/2303.14801](http://arxiv.org/abs/2303.14801)

    提出了一种新的自适应方法FAStEN，用于在高维函数回归问题中执行特征选择和参数估计，通过利用函数主成分和对偶增广Lagrangian问题的稀疏性质，具有显著的计算效率和选择准确性。

    

    函数回归分析是许多当代科学应用的已建立工具。涉及大规模和复杂数据集的回归问题是普遍存在的，特征选择对于避免过度拟合和实现准确预测至关重要。我们提出了一种新的、灵活的、超高效的方法，用于在稀疏高维函数回归问题中执行特征选择，并展示了如何将其扩展到标量对函数框架中。我们的方法将函数数据、优化和机器学习技术相结合，以同时执行特征选择和参数估计。我们利用函数主成分的特性以及对偶增广Lagrangian问题的稀疏性质，显著降低了计算成本，并引入了自适应方案来提高选择准确性。通过广泛的模拟研究，我们将我们的方法与最佳现有竞争对手进行了基准测试，并证明了我们的方法的高效性。

    Functional regression analysis is an established tool for many contemporary scientific applications. Regression problems involving large and complex data sets are ubiquitous, and feature selection is crucial for avoiding overfitting and achieving accurate predictions. We propose a new, flexible, and ultra-efficient approach to perform feature selection in a sparse high dimensional function-on-function regression problem, and we show how to extend it to the scalar-on-function framework. Our method combines functional data, optimization, and machine learning techniques to perform feature selection and parameter estimation simultaneously. We exploit the properties of Functional Principal Components, and the sparsity inherent to the Dual Augmented Lagrangian problem to significantly reduce computational cost, and we introduce an adaptive scheme to improve selection accuracy. Through an extensive simulation study, we benchmark our approach to the best existing competitors and demonstrate a 
    

