# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^2] | [Online Ensemble of Models for Optimal Predictive Performance with Applications to Sector Rotation Strategy.](http://arxiv.org/abs/2304.09947) | 通过机器学习模型和资产特定因素在预测行业回报和测量行业特定风险溢价方面获得更大经济收益，开发了一种新型在线集成算法来学习优化预测性能，特别适用于时间序列问题和可能的黑盒模型系统。 |

# 详细

[^1]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^2]: 在线模型集成对最优预测性能的应用和行业轮换策略

    Online Ensemble of Models for Optimal Predictive Performance with Applications to Sector Rotation Strategy. (arXiv:2304.09947v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.09947](http://arxiv.org/abs/2304.09947)

    通过机器学习模型和资产特定因素在预测行业回报和测量行业特定风险溢价方面获得更大经济收益，开发了一种新型在线集成算法来学习优化预测性能，特别适用于时间序列问题和可能的黑盒模型系统。

    

    资产特定因素通常用于预测金融回报并量化资产特定风险溢价。我们使用各种机器学习模型证明，这些因素包含的信息可以在预测行业回报和测量行业特定风险溢价方面带来更大的经济收益。为了利用不同行业表现的单个模型的强预测结果，我们开发了一种新型在线集成算法，该算法学习优化预测性能。该算法随着时间的推移不断适应，通过分析它们最近的预测性能来确定个体模型的最佳组合。这使它特别适用于时间序列问题，滚动窗口回测程序和可能的黑盒模型系统。我们推导出最优增益函数，用样本外R平方度量表达相应的遗憾界，并推导出最优解。

    Asset-specific factors are commonly used to forecast financial returns and quantify asset-specific risk premia. Using various machine learning models, we demonstrate that the information contained in these factors leads to even larger economic gains in terms of forecasts of sector returns and the measurement of sector-specific risk premia. To capitalize on the strong predictive results of individual models for the performance of different sectors, we develop a novel online ensemble algorithm that learns to optimize predictive performance. The algorithm continuously adapts over time to determine the optimal combination of individual models by solely analyzing their most recent prediction performance. This makes it particularly suited for time series problems, rolling window backtesting procedures, and systems of potentially black-box models. We derive the optimal gain function, express the corresponding regret bounds in terms of the out-of-sample R-squared measure, and derive optimal le
    

