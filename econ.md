# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric estimation of conditional densities by generalized random forests.](http://arxiv.org/abs/2309.13251) | 本文提出了一种基于广义随机森林的非参数估计方法，用于估计给定X条件下Y的条件密度。该方法采用了指数级数来表示条件密度，并通过解决一个非线性方程组来得到系数。实验结果表明，该方法是一致的，并且在允许基函数维度无限增长的情况下渐近正态。同时，本文还提供了一个标准误公式来构建置信区间。 |
| [^2] | [Inverse estimation of the transfer velocity of money.](http://arxiv.org/abs/2209.01512) | 本文提出了一种新的方法，利用微观级交易数据逆向估计货币的传输速度。通过测量货币在个人账户上的持有时间，并计算其逆反，我们发现Sarafu在肯尼亚的传输速度比总体上更高，因为并非所有的Sarafu单位都保持在活跃循环中。 |

# 详细

[^1]: 概率无限制估计下的条件密度的非参数估计方法：广义随机森林

    Nonparametric estimation of conditional densities by generalized random forests. (arXiv:2309.13251v1 [econ.EM])

    [http://arxiv.org/abs/2309.13251](http://arxiv.org/abs/2309.13251)

    本文提出了一种基于广义随机森林的非参数估计方法，用于估计给定X条件下Y的条件密度。该方法采用了指数级数来表示条件密度，并通过解决一个非线性方程组来得到系数。实验结果表明，该方法是一致的，并且在允许基函数维度无限增长的情况下渐近正态。同时，本文还提供了一个标准误公式来构建置信区间。

    

    在考虑连续随机变量Y和连续随机向量X的情况下，本文提出了一种非参数估计器f^(.|x)，用于给定X=x条件下Y的条件密度。该估计器采用了一个指数级数的形式，其系数T = (T1,...,TJ)是一组依赖于条件期望估计器E[p(Y)|X=x]的非线性方程组的解，其中p(.)是一个J维基函数向量。一个关键特点是E[p(Y)|X=x]通过广义随机森林（Athey, Tibshirani, and Wager, 2019）来进行估计，以针对不同x下T的异质性。我证明了f^(.|x)是一致的，并且在允许J无限增长的情况下渐近正态，并提供了一个标准误公式来构建渐近有效的置信区间。通过Monte Carlo实验和实证分析得到了结果。

    Considering a continuous random variable Y together with a continuous random vector X, I propose a nonparametric estimator f^(.|x) for the conditional density of Y given X=x. This estimator takes the form of an exponential series whose coefficients T = (T1,...,TJ) are the solution of a system of nonlinear equations that depends on an estimator of the conditional expectation E[p(Y)|X=x], where p(.) is a J-dimensional vector of basis functions. A key feature is that E[p(Y)|X=x] is estimated by generalized random forest (Athey, Tibshirani, and Wager, 2019), targeting the heterogeneity of T across x. I show that f^(.|x) is uniformly consistent and asymptotically normal, while allowing J to grow to infinity. I also provide a standard error formula to construct asymptotically valid confidence intervals. Results from Monte Carlo experiments and an empirical illustration are provided.
    
[^2]: 货币传输速度的逆推估计

    Inverse estimation of the transfer velocity of money. (arXiv:2209.01512v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.01512](http://arxiv.org/abs/2209.01512)

    本文提出了一种新的方法，利用微观级交易数据逆向估计货币的传输速度。通过测量货币在个人账户上的持有时间，并计算其逆反，我们发现Sarafu在肯尼亚的传输速度比总体上更高，因为并非所有的Sarafu单位都保持在活跃循环中。

    

    监测货币供应是进行健全货币政策的重要前提，但传统上货币指标的估计是以总体为单位进行的。本文提出了一种能够利用现实世界支付系统的微观级交易数据的新方法。我们应用一种新颖的计算技术来测量货币在个人账户中持有的持续时间，并通过其逆反计算货币的传输速度。我们的新定义在传统假设下退化为现有定义。然而，在总余额波动和消费模式发生改变的支付系统中，逆估计仍然适用。我们的方法应用于肯尼亚的一个小型数字社区货币Sarafu的研究中，从2020年1月25日到2021年6月15日可以获得交易数据。我们发现，Sarafu的传输速度比看起来的要高，因为并非所有的Sarafu单位都保持在活跃循环中。

    Monitoring the money supply is an important prerequisite for conducting sound monetary policy, yet monetary indicators are conventionally estimated in aggregate. This paper proposes a new methodology that is able to leverage micro-level transaction data from real-world payment systems. We apply a novel computational technique to measure the durations for which money is held in individual accounts, and compute the transfer velocity of money from its inverse. Our new definition reduces to existing definitions under conventional assumptions. However, inverse estimation remains suitable for payment systems where the total balance fluctuates and spending patterns change in time. Our method is applied to study Sarafu, a small digital community currency in Kenya, where transaction data is available from 25 January 2020 to 15 June 2021. We find that the transfer velocity of Sarafu was higher than it would seem, in aggregate, because not all units of Sarafu remained in active circulation. Moreo
    

