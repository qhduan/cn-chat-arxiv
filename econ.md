# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross-Temporal Forecast Reconciliation at Digital Platforms with Machine Learning](https://arxiv.org/abs/2402.09033) | 本论文介绍了一种使用机器学习方法在数字平台上进行跨时预测协调的非线性分层预测协调方法，该方法能够直接且自动化地生成跨时预测协调的预测，通过对来自按需交付平台的大规模流式数据集进行实证测试。 |
| [^2] | [Nonparametric estimation of conditional densities by generalized random forests.](http://arxiv.org/abs/2309.13251) | 本文提出了一种基于广义随机森林的非参数估计方法，用于估计给定X条件下Y的条件密度。该方法采用了指数级数来表示条件密度，并通过解决一个非线性方程组来得到系数。实验结果表明，该方法是一致的，并且在允许基函数维度无限增长的情况下渐近正态。同时，本文还提供了一个标准误公式来构建置信区间。 |
| [^3] | [Sources of capital growth.](http://arxiv.org/abs/2309.03403) | 资本增长和加速不依赖于净储蓄或消费的限制，对经济教育和公共政策有重要影响。 |

# 详细

[^1]: 使用机器学习在数字平台上进行跨时预测协调

    Cross-Temporal Forecast Reconciliation at Digital Platforms with Machine Learning

    [https://arxiv.org/abs/2402.09033](https://arxiv.org/abs/2402.09033)

    本论文介绍了一种使用机器学习方法在数字平台上进行跨时预测协调的非线性分层预测协调方法，该方法能够直接且自动化地生成跨时预测协调的预测，通过对来自按需交付平台的大规模流式数据集进行实证测试。

    

    平台业务在数字核心上运作，其决策需要不同层次（例如地理区域）和时间聚合（例如分钟到天）的高维准确预测流。为了确保不同规划单元（如定价、产品、控制和战略）之间的决策一致，也需要在层次结构的所有级别上进行协调预测。鉴于平台数据流具有复杂的特征和相互依赖关系，我们引入了一种非线性分层预测协调方法，通过使用流行的机器学习方法，以直接和自动化的方式生成跨时预测协调的预测。该方法足够快，可以满足平台所需的基于预测的高频决策。我们使用来自领先的按需交付平台的独特大规模流式数据集对我们的框架进行了实证测试。

    arXiv:2402.09033v1 Announce Type: new Abstract: Platform businesses operate on a digital core and their decision making requires high-dimensional accurate forecast streams at different levels of cross-sectional (e.g., geographical regions) and temporal aggregation (e.g., minutes to days). It also necessitates coherent forecasts across all levels of the hierarchy to ensure aligned decision making across different planning units such as pricing, product, controlling and strategy. Given that platform data streams feature complex characteristics and interdependencies, we introduce a non-linear hierarchical forecast reconciliation method that produces cross-temporal reconciled forecasts in a direct and automated way through the use of popular machine learning methods. The method is sufficiently fast to allow forecast-based high-frequency decision making that platforms require. We empirically test our framework on a unique, large-scale streaming dataset from a leading on-demand delivery plat
    
[^2]: 概率无限制估计下的条件密度的非参数估计方法：广义随机森林

    Nonparametric estimation of conditional densities by generalized random forests. (arXiv:2309.13251v1 [econ.EM])

    [http://arxiv.org/abs/2309.13251](http://arxiv.org/abs/2309.13251)

    本文提出了一种基于广义随机森林的非参数估计方法，用于估计给定X条件下Y的条件密度。该方法采用了指数级数来表示条件密度，并通过解决一个非线性方程组来得到系数。实验结果表明，该方法是一致的，并且在允许基函数维度无限增长的情况下渐近正态。同时，本文还提供了一个标准误公式来构建置信区间。

    

    在考虑连续随机变量Y和连续随机向量X的情况下，本文提出了一种非参数估计器f^(.|x)，用于给定X=x条件下Y的条件密度。该估计器采用了一个指数级数的形式，其系数T = (T1,...,TJ)是一组依赖于条件期望估计器E[p(Y)|X=x]的非线性方程组的解，其中p(.)是一个J维基函数向量。一个关键特点是E[p(Y)|X=x]通过广义随机森林（Athey, Tibshirani, and Wager, 2019）来进行估计，以针对不同x下T的异质性。我证明了f^(.|x)是一致的，并且在允许J无限增长的情况下渐近正态，并提供了一个标准误公式来构建渐近有效的置信区间。通过Monte Carlo实验和实证分析得到了结果。

    Considering a continuous random variable Y together with a continuous random vector X, I propose a nonparametric estimator f^(.|x) for the conditional density of Y given X=x. This estimator takes the form of an exponential series whose coefficients T = (T1,...,TJ) are the solution of a system of nonlinear equations that depends on an estimator of the conditional expectation E[p(Y)|X=x], where p(.) is a J-dimensional vector of basis functions. A key feature is that E[p(Y)|X=x] is estimated by generalized random forest (Athey, Tibshirani, and Wager, 2019), targeting the heterogeneity of T across x. I show that f^(.|x) is uniformly consistent and asymptotically normal, while allowing J to grow to infinity. I also provide a standard error formula to construct asymptotically valid confidence intervals. Results from Monte Carlo experiments and an empirical illustration are provided.
    
[^3]: 资本增长的来源

    Sources of capital growth. (arXiv:2309.03403v1 [econ.GN])

    [http://arxiv.org/abs/2309.03403](http://arxiv.org/abs/2309.03403)

    资本增长和加速不依赖于净储蓄或消费的限制，对经济教育和公共政策有重要影响。

    

    根据国民账户数据显示，净储蓄或消费的变化与市值资本增长率的变化（资本加速度）之间没有影响。因此，资本增长和加速似乎不依赖于净储蓄或消费的限制。我们探讨了这种可能性，并讨论了对经济教育和公共政策的影响。

    Data from national accounts show no effect of change in net saving or consumption, in ratio to market-value capital, on change in growth rate of market-value capital (capital acceleration). Thus it appears that capital growth and acceleration arrive without help from net saving or consumption restraint. We explore ways in which this is possible, and discuss implications for economic teaching and public policy
    

