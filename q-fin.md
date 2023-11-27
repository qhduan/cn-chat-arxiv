# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Agent-based Modelling of Credit Card Promotions.](http://arxiv.org/abs/2311.01901) | 本研究提出了一种基于Agent的信用卡促销模型，通过校准和验证，可以优化不同市场情景下的信用卡促销策略。 |
| [^2] | [Black-Litterman, Bayesian Shrinkage, and Factor Models in Portfolio Selection: You Can Have It All.](http://arxiv.org/abs/2308.09264) | 该论文提出了一个融合缩小估计、观点纳入和因子模型的贝叶斯蓝图，该蓝图在投资组合选择中应用并优于简单的$1/N$投资组合。 |
| [^3] | [Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo.](http://arxiv.org/abs/2305.09166) | 本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。 |

# 详细

[^1]: 基于Agent的信用卡促销模型

    Agent-based Modelling of Credit Card Promotions. (arXiv:2311.01901v1 [cs.MA])

    [http://arxiv.org/abs/2311.01901](http://arxiv.org/abs/2311.01901)

    本研究提出了一种基于Agent的信用卡促销模型，通过校准和验证，可以优化不同市场情景下的信用卡促销策略。

    

    无息促销是信用卡发行商吸引新客户的常用策略，然而对于它们对消费者和发行商的影响的研究相对较少。选择最优促销策略的过程是复杂的，涉及在竞争机制、市场动态和复杂的消费者行为背景下确定无息期限和促销可用窗口。在本文中，我们介绍了一种基于Agent的模型，可以在不同市场情景下探索各种信用卡促销策略。我们的方法与以往的基于Agent的模型不同，专注于优化促销策略，并使用2019年至2020年英国信用卡市场的基准数据进行校准，代理属性来自大致同一时期的英国人口的历史分布。我们通过结构化事实和时间序列数据对模型进行验证。

    Interest-free promotions are a prevalent strategy employed by credit card lenders to attract new customers, yet the research exploring their effects on both consumers and lenders remains relatively sparse. The process of selecting an optimal promotion strategy is intricate, involving the determination of an interest-free period duration and promotion-availability window, all within the context of competing offers, fluctuating market dynamics, and complex consumer behaviour. In this paper, we introduce an agent-based model that facilitates the exploration of various credit card promotions under diverse market scenarios. Our approach, distinct from previous agent-based models, concentrates on optimising promotion strategies and is calibrated using benchmarks from the UK credit card market from 2019 to 2020, with agent properties derived from historical distributions of the UK population from roughly the same period. We validate our model against stylised facts and time-series data, there
    
[^2]: Black-Litterman、Bayesian Shrinkage和Factor Models在投资组合选择中的应用：拥有全面的选择。

    Black-Litterman, Bayesian Shrinkage, and Factor Models in Portfolio Selection: You Can Have It All. (arXiv:2308.09264v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.09264](http://arxiv.org/abs/2308.09264)

    该论文提出了一个融合缩小估计、观点纳入和因子模型的贝叶斯蓝图，该蓝图在投资组合选择中应用并优于简单的$1/N$投资组合。

    

    均值方差分析被广泛应用于投资组合管理，以在预期收益和波动性之间实现最优的权衡。然而，该方法存在一些限制，尤其是对估计误差的脆弱性和对历史数据的依赖性。虽然缩小估计器和因子模型已被引入以通过偏差-方差权衡来提高估计准确性，而Black-Litterman模型已被开发用于整合投资者观点，但缺乏一个将三种方法结合起来的统一框架。我们的研究首次提出了一个贝叶斯蓝图，将缩小估计与观点纳入融合，将两者都概念化为贝叶斯更新。然后，我们将该模型应用于Fama-French因子模型的上下文中，从而整合了每种方法的优势。最后，通过在跨越十年的美国股票市场进行全面的实证研究，我们证明该模型优于简单的$1/N$投资组合。

    Mean-variance analysis is widely used in portfolio management to identify the best portfolio that makes an optimal trade-off between expected return and volatility. Yet, this method has its limitations, notably its vulnerability to estimation errors and its reliance on historical data. While shrinkage estimators and factor models have been introduced to improve estimation accuracy through bias-variance trade-offs, and the Black-Litterman model has been developed to integrate investor opinions, a unified framework combining three approaches has been lacking. Our study debuts a Bayesian blueprint that fuses shrinkage estimation with view inclusion, conceptualizing both as Bayesian updates. This model is then applied within the context of the Fama-French approach factor models, thereby integrating the advantages of each methodology. Finally, through a comprehensive empirical study in the US equity market spanning a decade, we show that the model outperforms both the simple $1/N$ portfolio
    
[^3]: 最小二乘蒙特卡罗中的有限差分解法

    Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo. (arXiv:2305.09166v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.09166](http://arxiv.org/abs/2305.09166)

    本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。

    

    本文提出了一种简单而有效的方法，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。关键思想是使用低维有限差分法的精确解来构建条件期望继续支付的假设，用于线性回归。该方法在解决向后偏微分方程和蒙特卡罗模拟方面建立了桥梁，旨在实现两者的最佳结合。我们通过实际示例说明该技术，包括百慕大期权和最差发行人可赎回票据。该方法可被视为跨越各种资产类别的通用数值方案，特别是在任意维度下，作为定价美式衍生产品的准确方法。

    This article presents a simple but effective approach to improve the accuracy of Least-Squares Monte Carlo for American-style options. The key idea is to construct the ansatz of conditional expected continuation payoff using the exact solution from low dimensional finite difference methods, to be used in linear regression. This approach builds a bridge between solving backward partial differential equations and a Monte Carlo simulation, aiming at achieving the best of both worlds. We illustrate the technique with realistic examples including Bermuda options and worst of issuer callable notes. The method can be considered as a generic numerical scheme across various asset classes, in particular, as an accurate method for pricing American-style derivatives under arbitrary dimensions.
    

