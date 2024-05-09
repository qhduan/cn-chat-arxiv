# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zonal vs. Nodal Pricing: An Analysis of Different Pricing Rules in the German Day-Ahead Market](https://arxiv.org/abs/2403.09265) | 德国电力市场研究了区域和节点定价模型的比较，发现不同配置下的平均价格差异小，总成本相似。 |
| [^2] | [Dimensionality reduction techniques to support insider trading detection](https://arxiv.org/abs/2403.00707) | 提出了一种无监督机器学习方法，利用主成分分析和自动编码器作为降维技术，用于支持市场监控以识别潜在内幕交易活动。 |
| [^3] | [An Intraday GARCH Model for Discrete Price Changes and Irregularly Spaced Observations.](http://arxiv.org/abs/2211.12376) | 这篇论文提出了一种新的适用于高频价格的日内GARCH模型，该模型考虑了不规则时间间隔观测、同时交易、价格离散性和市场微观结构噪声。通过使用平滑样条捕捉交易持续时间与价格波动性之间的关系，并且使用零膨胀Skellam分布的得分驱动框架，该模型可以很好地拟合数据，并且可以用于测量每日实现波动性。 |

# 详细

[^1]: 区域vs. 节点定价：德国日前市场不同定价规则的分析

    Zonal vs. Nodal Pricing: An Analysis of Different Pricing Rules in the German Day-Ahead Market

    [https://arxiv.org/abs/2403.09265](https://arxiv.org/abs/2403.09265)

    德国电力市场研究了区域和节点定价模型的比较，发现不同配置下的平均价格差异小，总成本相似。

    

    欧洲电力市场基于拥有统一日前价格的大型定价区域。能源转型导致供需变化和再调度成本增加。为了确保市场清算高效和拥塞管理，欧盟委员会委托进行出价区域审查（BZR）以重新评估欧洲出价区域的配置。基于BZR背景下公布的独特数据集，我们比较了德国电力市场的各种定价规则。我们比较了国内、区域和节点模型的市场清算和定价，包括它们的发电成本和相关的再调度成本。此外，我们研究了不同的非统一定价规则及其对德国电力市场的经济影响。我们的结果表明，不同区域的平均价格差异较小。不同配置下的总成本相似，降低了...

    arXiv:2403.09265v1 Announce Type: new  Abstract: The European electricity market is based on large pricing zones with a uniform day-ahead price. The energy transition leads to shifts in supply and demand and increasing redispatch costs. In an attempt to ensure efficient market clearing and congestion management, the EU Commission has mandated the Bidding Zone Review (BZR) to reevaluate the configuration of European bidding zones. Based on a unique data set published in the context of the BZR, we compare various pricing rules for the German power market. We compare market clearing and pricing for national, zonal, and nodal models, including their generation costs and associated redispatch costs. Moreover, we investigate different non-uniform pricing rules and their economic implications for the German electricity market. Our results indicate that the differences in the average prices in different zones are small. The total costs across different configurations are similar and the reduct
    
[^2]: 降维技术用于支持内幕交易检测

    Dimensionality reduction techniques to support insider trading detection

    [https://arxiv.org/abs/2403.00707](https://arxiv.org/abs/2403.00707)

    提出了一种无监督机器学习方法，利用主成分分析和自动编码器作为降维技术，用于支持市场监控以识别潜在内幕交易活动。

    

    识别市场滥用是一项非常复杂的活动，需要分析大量复杂的数据集。我们提出了一种无监督机器学习方法，用于上下文异常检测，可以支持旨在识别潜在内幕交易活动的市场监控。该方法基于重建范式，采用主成分分析和自动编码器作为降维技术。该方法的唯一输入是每位投资者在对我们具有价格敏感事件（PSE）的资产上的交易位置。在确定与交易配置文件相关的重建错误后，我们会施加几个条件，以识别那些行为可疑的投资者，其行为可能涉及与PSE有关的内幕交易。作为案例研究，我们将我们的方法应用于围绕收购要约的意大利股票的投资者解析数据。

    arXiv:2403.00707v1 Announce Type: new  Abstract: Identification of market abuse is an extremely complicated activity that requires the analysis of large and complex datasets. We propose an unsupervised machine learning method for contextual anomaly detection, which allows to support market surveillance aimed at identifying potential insider trading activities. This method lies in the reconstruction-based paradigm and employs principal component analysis and autoencoders as dimensionality reduction techniques. The only input of this method is the trading position of each investor active on the asset for which we have a price sensitive event (PSE). After determining reconstruction errors related to the trading profiles, several conditions are imposed in order to identify investors whose behavior could be suspicious of insider trading related to the PSE. As a case study, we apply our method to investor resolved data of Italian stocks around takeover bids.
    
[^3]: 一种适用于离散价格变动和不规则时间间隔观测的日内GARCH模型

    An Intraday GARCH Model for Discrete Price Changes and Irregularly Spaced Observations. (arXiv:2211.12376v3 [q-fin.ST] UPDATED)

    [http://arxiv.org/abs/2211.12376](http://arxiv.org/abs/2211.12376)

    这篇论文提出了一种新的适用于高频价格的日内GARCH模型，该模型考虑了不规则时间间隔观测、同时交易、价格离散性和市场微观结构噪声。通过使用平滑样条捕捉交易持续时间与价格波动性之间的关系，并且使用零膨胀Skellam分布的得分驱动框架，该模型可以很好地拟合数据，并且可以用于测量每日实现波动性。

    

    我们提出了一种新颖的基于观测的高频价格模型。我们考虑了不规则时间间隔观测、同时交易、价格离散性和市场微观结构噪声。使用平滑样条捕捉交易持续时间与价格波动性之间的关系，以及交易持续时间和价格波动性的日内模式。动态模型基于具有时间变化波动性的零膨胀Skellam分布的得分驱动框架。通过包括移动平均分量来滤除市场微观结构噪声。该模型采用最大似然方法进行估计。在对IBM股票的实证研究中，我们证明了该模型对数据拟合较好。除了对日内波动性建模外，它还可以用于测量每日实现波动性。

    We develop a novel observation-driven model for high-frequency prices. We account for irregularly spaced observations, simultaneous transactions, discreteness of prices, and market microstructure noise. The relation between trade durations and price volatility, as well as intraday patterns of trade durations and price volatility, is captured using smoothing splines. The dynamic model is based on the zero-inflated Skellam distribution with time-varying volatility in a score-driven framework. Market microstructure noise is filtered by including a moving average component. The model is estimated by the maximum likelihood method. In an empirical study of the IBM stock, we demonstrate that the model provides a good fit to the data. Besides modeling intraday volatility, it can also be used to measure daily realized volatility.
    

