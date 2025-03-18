# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Financial sentiment analysis using FinBERT with application in predicting stock movement.](http://arxiv.org/abs/2306.02136) | 本文使用FinBERT进行财经情绪分析，构建了基于LSTM的深度神经网络模型来预测市场运动，发现情绪是预测市场运动的有效因素，并提出了改进模型的方法。 |
| [^2] | [Model-free Analysis of Dynamic Trading Strategies.](http://arxiv.org/abs/2011.02870) | 这篇论文介绍了一种基于交易信号奇点的无模型方法，用于分析各种动态交易策略的风险和收益。它提出了一种数学框架来描述这些策略的风险，并展示了使用奇点分解进行情景分析的简单表达式。 |

# 详细

[^1]: 使用FinBERT进行财经情绪分析及应用于股票预测

    Financial sentiment analysis using FinBERT with application in predicting stock movement. (arXiv:2306.02136v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.02136](http://arxiv.org/abs/2306.02136)

    本文使用FinBERT进行财经情绪分析，构建了基于LSTM的深度神经网络模型来预测市场运动，发现情绪是预测市场运动的有效因素，并提出了改进模型的方法。

    

    我们使用FinBERT进行财经情绪分析，并构建了基于LSTM的深度神经网络模型来预测金融市场的运动。我们将这个模型应用于股票新闻数据集，并将其有效性与BERT，LSTM和经典的ARIMA模型进行比较。我们发现情绪是预测市场运动的有效因素。我们还提出了几种改进模型的方法。

    We apply sentiment analysis in financial context using FinBERT, and build a deep neural network model based on LSTM to predict the movement of financial market movement. We apply this model on stock news dataset, and compare its effectiveness to BERT, LSTM and classical ARIMA model. We find that sentiment is an effective factor in predicting market movement. We also propose several method to improve the model.
    
[^2]: 动态交易策略的无模型分析

    Model-free Analysis of Dynamic Trading Strategies. (arXiv:2011.02870v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2011.02870](http://arxiv.org/abs/2011.02870)

    这篇论文介绍了一种基于交易信号奇点的无模型方法，用于分析各种动态交易策略的风险和收益。它提出了一种数学框架来描述这些策略的风险，并展示了使用奇点分解进行情景分析的简单表达式。

    

    我们引入了一种基于交易信号奇点的无模型方法，用于分析广泛类别的动态交易策略的风险和收益，包括对冲交易和其他统计套利策略。我们提出了一个数学框架，用于基于离开参考水平的价格奇点描述这些策略的风险分析，在没有任何概率假设的路径设置中。我们引入了delta-奇点的概念，定义为一条路径在偏离一个参考水平后返回到该水平的路径。我们展示了每条连续路径都能以delta-奇点的唯一分解形式表示，这对于动态交易策略的情景分析非常有用，可以得到关于交易次数、实现利润、最大亏损和回撤的简单表达式。我们展示了高频极限，对应于delta趋近于零的情况，由(p-th order)的局部时间决定。

    We introduce a model-free approach based on excursions of trading signals for analyzing the risk and return for a broad class of dynamic trading strategies, including pairs trading and other statistical arbitrage strategies. We propose a mathematical framework for the risk analysis of such strategies, based on a description in terms of excursions of prices away from a reference level, in a pathwise setting without any probabilistic assumptions.  We introduce the notion of delta-excursion, defined as a path that deviates by delta from a reference level before returning to this level. We show that every continuous path has a unique decomposition into delta-excursions, which is useful for the scenario analysis of dynamic trading strategies, leading to simple expressions for the number of trades, realized profit, maximum loss, and drawdown. We show that the high-frequency limit, which corresponds to the case where delta decreases to zero, is described by the (p-th order) local time of the 
    

