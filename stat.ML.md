# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interaction Screening and Pseudolikelihood Approaches for Tensor Learning in Ising Models.](http://arxiv.org/abs/2310.13232) | 本文研究了在Ising模型中的张量学习中，通过伪似然方法和相互作用筛选方法可以恢复出底层的超网络结构，并且性能比较表明张量恢复速率与最大耦合强度呈指数关系。 |
| [^2] | [Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders.](http://arxiv.org/abs/2305.16189) | 该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。 |
| [^3] | [Adaptive, Rate-Optimal Hypothesis Testing in Nonparametric IV Models.](http://arxiv.org/abs/2006.09587) | 我们提出了一种自适应检验方法，用于处理非参数仪器变量模型中的结构函数的不等式和等式限制。该方法可以适应未知的平滑度和工具强度，并达到了最小值率的自适应最优检验率。 |

# 详细

[^1]: Ising模型中的张量学习的相互作用筛选和伪似然方法

    Interaction Screening and Pseudolikelihood Approaches for Tensor Learning in Ising Models. (arXiv:2310.13232v1 [stat.ME])

    [http://arxiv.org/abs/2310.13232](http://arxiv.org/abs/2310.13232)

    本文研究了在Ising模型中的张量学习中，通过伪似然方法和相互作用筛选方法可以恢复出底层的超网络结构，并且性能比较表明张量恢复速率与最大耦合强度呈指数关系。

    

    本文研究了在$k$-spin Ising模型中的张量恢复中，伪似然方法和相互作用筛选方法两种已知的Ising结构学习方法。我们证明，在适当的正则化下，这两种方法可以使用样本数对数级别大小的样本恢复出底层的超网络结构，且与最大相互作用强度和最大节点度指数级依赖。我们还对这两种方法的张量恢复速率与交互阶数$k$的确切关系进行了跟踪，并允许$k$随样本数和节点数增长。最后，我们通过仿真研究对这两种方法的性能进行了比较讨论，结果也显示了张量恢复速率与最大耦合强度之间的指数依赖关系。

    In this paper, we study two well known methods of Ising structure learning, namely the pseudolikelihood approach and the interaction screening approach, in the context of tensor recovery in $k$-spin Ising models. We show that both these approaches, with proper regularization, retrieve the underlying hypernetwork structure using a sample size logarithmic in the number of network nodes, and exponential in the maximum interaction strength and maximum node-degree. We also track down the exact dependence of the rate of tensor recovery on the interaction order $k$, that is allowed to grow with the number of samples and nodes, for both the approaches. Finally, we provide a comparative discussion of the performance of the two approaches based on simulation studies, which also demonstrate the exponential dependence of the tensor recovery rate on the maximum coupling strength.
    
[^2]: 火星时间序列分解：一种多尺度嵌套方法中的因子变分自编码器

    Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders. (arXiv:2305.16189v1 [cs.LG])

    [http://arxiv.org/abs/2305.16189](http://arxiv.org/abs/2305.16189)

    该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。

    

    无监督的源分离涉及通过混合操作记录的未知源信号的分解，其中对源的先验知识有限，仅可以访问信号混合数据集。这个问题本质上是不适用的，并且进一步受到时间序列数据中源展现出的多种时间尺度的挑战。为了解决这个问题，我们提出了一种无监督的多尺度聚类和源分离框架，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程。在这个表示空间中，我们开发了一个因子高斯混合变分自动编码器，它被训练用于(1)概率地对不同时间尺度上的源进行聚类和逐层非监督源分离，(2)在每个时间尺度上提取低维表示，(3)学习源信号的因子表示，(4)在表示空间中进行采样，以生成未知源信号。我们在MRO上的三个频道的可见数据集上进行了评估，结果表明所提出的方法比目前最先进的技术具有更好的性能。

    Unsupervised source separation involves unraveling an unknown set of source signals recorded through a mixing operator, with limited prior knowledge about the sources, and only access to a dataset of signal mixtures. This problem is inherently ill-posed and is further challenged by the variety of time-scales exhibited by sources in time series data. Existing methods typically rely on a preselected window size that limits their capacity to handle multi-scale sources. To address this issue, instead of operating in the time domain, we propose an unsupervised multi-scale clustering and source separation framework by leveraging wavelet scattering covariances that provide a low-dimensional representation of stochastic processes, capable of distinguishing between different non-Gaussian stochastic processes. Nested within this representation space, we develop a factorial Gaussian-mixture variational autoencoder that is trained to (1) probabilistically cluster sources at different time-scales a
    
[^3]: 非参数IV模型中的自适应高效假设检验

    Adaptive, Rate-Optimal Hypothesis Testing in Nonparametric IV Models. (arXiv:2006.09587v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2006.09587](http://arxiv.org/abs/2006.09587)

    我们提出了一种自适应检验方法，用于处理非参数仪器变量模型中的结构函数的不等式和等式限制。该方法可以适应未知的平滑度和工具强度，并达到了最小值率的自适应最优检验率。

    

    我们提出了一种新的自适应假设检验方法，用于非参数仪器变量（NPIV）模型中结构函数的不等式（如单调性、凸性）和等式（如参数、半参数）限制。我们的检验统计量基于修改版的留一法样本模拟，计算受限和不受限筛子NPIV估计量间的二次距离。我们提供了计算简单、数据驱动的筛子调参和Bonferroni调整卡方临界值的选择。我们的检验适应未知的内生性平滑度和工具强度，达到了$L^2$最小值率的自适应最优检验率。也就是说，在复合零假设下其类型I误差的总体和其类型II误差的总体均不能被任何其他NPIV模型的假设检验所提高。我们还提出了基于数据的置信区间。

    We propose a new adaptive hypothesis test for inequality (e.g., monotonicity, convexity) and equality (e.g., parametric, semiparametric) restrictions on a structural function in a nonparametric instrumental variables (NPIV) model. Our test statistic is based on a modified leave-one-out sample analog of a quadratic distance between the restricted and unrestricted sieve NPIV estimators. We provide computationally simple, data-driven choices of sieve tuning parameters and Bonferroni adjusted chi-squared critical values. Our test adapts to the unknown smoothness of alternative functions in the presence of unknown degree of endogeneity and unknown strength of the instruments. It attains the adaptive minimax rate of testing in $L^2$.  That is, the sum of its type I error uniformly over the composite null and its type II error uniformly over nonparametric alternative models cannot be improved by any other hypothesis test for NPIV models of unknown regularities. Data-driven confidence sets in 
    

