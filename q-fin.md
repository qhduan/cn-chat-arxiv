# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of an Order Book Dependent Hawkes Process for Large Datasets.](http://arxiv.org/abs/2307.09077) | 本研究提出了一种用于高频交易事件到达的点过程，其中强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。算法可以在存在数十亿数据点的情况下进行估计，并证明了其收敛性和一致性。样本外测试结果显示，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。 |
| [^2] | [On the Well-posedness of Hamilton-Jacobi-Bellman Equations of the Equilibrium Type.](http://arxiv.org/abs/2307.01986) | 本文研究了一类非局部抛物型偏微分方程的良定性，这与时不变的随机控制问题中平衡策略和相关价值函数的表征有关。通过连续性方法和线性化方法，证明了非局部非线性情况的局部和全局良定性。 |

# 详细

[^1]: 《大数据集上基于委托簿相关Hawkes过程的估计》

    Estimation of an Order Book Dependent Hawkes Process for Large Datasets. (arXiv:2307.09077v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.09077](http://arxiv.org/abs/2307.09077)

    本研究提出了一种用于高频交易事件到达的点过程，其中强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。算法可以在存在数十亿数据点的情况下进行估计，并证明了其收敛性和一致性。样本外测试结果显示，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。

    

    本研究介绍了一种用于高频交易事件到达的点过程。强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。讨论了该过程稳定性的条件。并提出了一种算法，即使在存在数十亿数据点的情况下，也可以进行模型估计，可能需要将协变量映射到高维空间。大样本量是常见于使用多个流动工具的高频数据应用中的情况。证明了算法的收敛性，建立了在弱条件下的一致性结果，并提出了一种测试统计量来评估不同模型规范的样本外表现。将该方法应用于纽约证券交易所（NYSE）上交易的四只股票的研究中。样本外测试过程表明，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。

    A point process for event arrivals in high frequency trading is presented. The intensity is the product of a Hawkes process and high dimensional functions of covariates derived from the order book. Conditions for stationarity of the process are stated. An algorithm is presented to estimate the model even in the presence of billions of data points, possibly mapping covariates into a high dimensional space. The large sample size can be common for high frequency data applications using multiple liquid instruments. Convergence of the algorithm is shown, consistency results under weak conditions is established, and a test statistic to assess out of sample performance of different model specifications is suggested. The methodology is applied to the study of four stocks that trade on the New York Stock Exchange (NYSE). The out of sample testing procedure suggests that capturing the nonlinearity of the order book information adds value to the self exciting nature of high frequency trading even
    
[^2]: 关于平衡型哈密尔顿-雅可比-贝尔曼方程的良定性

    On the Well-posedness of Hamilton-Jacobi-Bellman Equations of the Equilibrium Type. (arXiv:2307.01986v1 [math.AP])

    [http://arxiv.org/abs/2307.01986](http://arxiv.org/abs/2307.01986)

    本文研究了一类非局部抛物型偏微分方程的良定性，这与时不变的随机控制问题中平衡策略和相关价值函数的表征有关。通过连续性方法和线性化方法，证明了非局部非线性情况的局部和全局良定性。

    

    本文研究了一类非局部抛物型偏微分方程（PDEs）的良定性，或等效地说是平衡型哈密尔顿-雅可比-贝尔曼方程，它与时不变的随机控制问题中平衡策略和相关价值函数的表征有密切关系。具体而言，我们考虑了时间和空间上的非局部性，这允许对具有初始时间和状态依赖目标函数的随机控制问题进行建模。我们利用连续性方法，在我们提出的巴拿赫空间中展示了线性化的非局部PDE的全局良定性，同时建立了舒尔德尔先验估计。然后，我们采用线性化方法和巴拿赫的不动点论证明了非局部完全非线性情况的局部良定性，而全局良定性是可达到的，前提是有一个非常精确的先验估计。

    This paper studies the well-posedness of a class of nonlocal parabolic partial differential equations (PDEs), or equivalently equilibrium Hamilton-Jacobi-Bellman equations, which has a strong tie with the characterization of the equilibrium strategies and the associated value functions for time-inconsistent stochastic control problems. Specifically, we consider nonlocality in both time and space, which allows for modelling of the stochastic control problems with initial-time-and-state dependent objective functionals. We leverage the method of continuity to show the global well-posedness within our proposed Banach space with our established Schauder prior estimate for the linearized nonlocal PDE. Then, we adopt a linearization method and Banach's fixed point arguments to show the local well-posedness of the nonlocal fully nonlinear case, while the global well-posedness is attainable provided that a very sharp a-priori estimate is available. On top of the well-posedness results, we also 
    

