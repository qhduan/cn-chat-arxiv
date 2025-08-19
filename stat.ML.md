# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficiently matching random inhomogeneous graphs via degree profiles.](http://arxiv.org/abs/2310.10441) | 本文提出了一种通过度特征匹配算法高效匹配随机不均匀图的方法，要求最小平均度和最小相关性达到一定阈值。 |
| [^2] | [Convergence analysis of online algorithms for vector-valued kernel regression.](http://arxiv.org/abs/2309.07779) | 本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。 |
| [^3] | [A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models.](http://arxiv.org/abs/2309.06230) | 该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。 |
| [^4] | [Kernel Ridge Regression Inference.](http://arxiv.org/abs/2302.06578) | 我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。 |

# 详细

[^1]: 通过度特征高效匹配随机不均匀图

    Efficiently matching random inhomogeneous graphs via degree profiles. (arXiv:2310.10441v1 [cs.DS])

    [http://arxiv.org/abs/2310.10441](http://arxiv.org/abs/2310.10441)

    本文提出了一种通过度特征匹配算法高效匹配随机不均匀图的方法，要求最小平均度和最小相关性达到一定阈值。

    

    本文研究了恢复两个相关的随机图之间潜在顶点对应关系的问题，这两个图具有极不均匀且未知的不同顶点对之间的边概率。在Ding、Ma、Wu和Xu(2021)提出的度特征匹配算法的基础上，我们扩展出了一种高效的匹配算法，只要最小平均度至少为$\Omega(\log^{2} n)$，最小相关性至少为$1 - O(\log^{-2} n)$。

    In this paper, we study the problem of recovering the latent vertex correspondence between two correlated random graphs with vastly inhomogeneous and unknown edge probabilities between different pairs of vertices. Inspired by and extending the matching algorithm via degree profiles by Ding, Ma, Wu and Xu (2021), we obtain an efficient matching algorithm as long as the minimal average degree is at least $\Omega(\log^{2} n)$ and the minimal correlation is at least $1 - O(\log^{-2} n)$.
    
[^2]: 在向量值内核回归的在线算法的收敛分析

    Convergence analysis of online algorithms for vector-valued kernel regression. (arXiv:2309.07779v1 [stat.ML])

    [http://arxiv.org/abs/2309.07779](http://arxiv.org/abs/2309.07779)

    本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。

    

    我们考虑使用适当的再生核希尔伯特空间（RKHS）作为先验，通过在线学习算法从噪声向量值数据中逼近回归函数的问题。在在线算法中，独立同分布的样本通过随机过程逐个可用，并依次处理以构建对回归函数的近似。我们关注这种在线逼近算法的渐近性能，并证明了在RKHS范数中的期望平方误差可以被$C^2(m+1)^{-s/(2+s)}$绑定，其中$m$为当下处理的数据数量，参数$0<s\leq 1$表示对回归函数的额外光滑性假设，常数$C$取决于输入噪声的方差、回归函数的光滑性以及算法的其他参数。

    We consider the problem of approximating the regression function from noisy vector-valued data by an online learning algorithm using an appropriate reproducing kernel Hilbert space (RKHS) as prior. In an online algorithm, i.i.d. samples become available one by one by a random process and are successively processed to build approximations to the regression function. We are interested in the asymptotic performance of such online approximation algorithms and show that the expected squared error in the RKHS norm can be bounded by $C^2 (m+1)^{-s/(2+s)}$, where $m$ is the current number of processed data, the parameter $0<s\leq 1$ expresses an additional smoothness assumption on the regression function and the constant $C$ depends on the variance of the input noise, the smoothness of the regression function and further parameters of the algorithm.
    
[^3]: 单指数模型中最佳子集选择的一致性和可扩展算法

    A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models. (arXiv:2309.06230v1 [stat.ML])

    [http://arxiv.org/abs/2309.06230](http://arxiv.org/abs/2309.06230)

    该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。

    

    高维数据的分析引发了对单指数模型（SIMs）和最佳子集选择的增加兴趣。SIMs为高维数据提供了一种可解释和灵活的建模框架，而最佳子集选择旨在从大量的预测因子中找到稀疏模型。然而，在高维模型中的最佳子集选择被认为是计算上难以处理的。现有的方法倾向于放宽选择，但不能得到最佳子集解。在本文中，我们通过提出第一个经过证明的针对高维SIMs中最佳子集选择的可扩展算法，直接解决了计算难题。我们的算法解具有子集选择一致性，并且几乎肯定具有用于参数估计的虚拟属性。该算法包括一个广义信息准则来确定回归系数的支持大小，消除模型选择调整。此外，我们的方法不假设误差分布或特定参数。

    Analysis of high-dimensional data has led to increased interest in both single index models (SIMs) and best subset selection. SIMs provide an interpretable and flexible modeling framework for high-dimensional data, while best subset selection aims to find a sparse model from a large set of predictors. However, best subset selection in high-dimensional models is known to be computationally intractable. Existing methods tend to relax the selection, but do not yield the best subset solution. In this paper, we directly tackle the intractability by proposing the first provably scalable algorithm for best subset selection in high-dimensional SIMs. Our algorithmic solution enjoys the subset selection consistency and has the oracle property with a high probability. The algorithm comprises a generalized information criterion to determine the support size of the regression coefficients, eliminating the model selection tuning. Moreover, our method does not assume an error distribution or a specif
    
[^4]: 核岭回归推断

    Kernel Ridge Regression Inference. (arXiv:2302.06578v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.06578](http://arxiv.org/abs/2302.06578)

    我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。

    

    我们提供了核岭回归(KRR)的一致推断和置信带，这是一种广泛应用于包括排名、图像和图表在内的一般数据类型的非参数回归估计器。尽管这些数据的普遍存在，如学校分配中的排序优先级列表，但KRR的推断理论尚未完全知悉，限制了它在经济学和其他科学领域中的作用。我们构建了针对一般回归器的尖锐、一致的置信区间。为了进行推断，我们开发了一种有效的自举程序，通过对称化来消除偏差并限制计算开销。为了证明该程序，我们推导了再生核希尔伯特空间(RKHS)中部分和的有限样本、均匀高斯和自举耦合。这些推导暗示了基于RKHS单位球的经验过程的强逼近，对覆盖数具有对数依赖关系。模拟验证了置信度。

    We provide uniform inference and confidence bands for kernel ridge regression (KRR), a widely-used non-parametric regression estimator for general data types including rankings, images, and graphs. Despite the prevalence of these data -e.g., ranked preference lists in school assignment -- the inferential theory of KRR is not fully known, limiting its role in economics and other scientific domains. We construct sharp, uniform confidence sets for KRR, which shrink at nearly the minimax rate, for general regressors. To conduct inference, we develop an efficient bootstrap procedure that uses symmetrization to cancel bias and limit computational overhead. To justify the procedure, we derive finite-sample, uniform Gaussian and bootstrap couplings for partial sums in a reproducing kernel Hilbert space (RKHS). These imply strong approximation for empirical processes indexed by the RKHS unit ball with logarithmic dependence on the covering number. Simulations verify coverage. We use our proce
    

