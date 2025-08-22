# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-period static hedging of European options.](http://arxiv.org/abs/2310.01104) | 本文研究了基于单因素马尔可夫框架的多期欧式期权的静态对冲，并对相对性能进行了实验比较。 |
| [^2] | [From constant to rough: A survey of continuous volatility modeling.](http://arxiv.org/abs/2309.01033) | 本文综述了连续波动率模型的发展和关键事实，重点介绍了分数和粗糙方法的模型，并讨论了VIX建模问题和SPX-VIX联合校准难题的最新进展。 |
| [^3] | [Modelling Determinants of Cryptocurrency Prices: A Bayesian Network Approach.](http://arxiv.org/abs/2303.16148) | 本文使用贝叶斯网络方法，从因果分析的角度研究了影响替代加密货币价格的因素，包括五种主要替代加密货币、传统金融资产和社交媒体，提供了一种解决加密货币价格预测问题的方法。 |

# 详细

[^1]: 多期欧式期权的静态对冲

    Multi-period static hedging of European options. (arXiv:2310.01104v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.01104](http://arxiv.org/abs/2310.01104)

    本文研究了基于单因素马尔可夫框架的多期欧式期权的静态对冲，并对相对性能进行了实验比较。

    

    本文考虑了在基础资产价格遵循单因素马尔可夫框架的情况下对欧式期权进行对冲。Carr和Wu [1]在这样的设置下，导出了给定期权与在同一资产上写的一系列较短期限期权之间的跨度关系。在本文中，我们将他们的方法扩展到同时包括多个短期到期的期权。然后，我们使用高斯求积方法通过有限的一组短期期权确定对冲误差的实际实现。我们对\textit{Black-Scholes}和\textit{Merton Jump Diffusion}模型进行了广泛的实验，展示了这两种方法的比较性能。

    We consider the hedging of European options when the price of the underlying asset follows a single-factor Markovian framework. By working in such a setting, Carr and Wu \cite{carr2014static} derived a spanning relation between a given option and a continuum of shorter-term options written on the same asset. In this paper, we have extended their approach to simultaneously include options over multiple short maturities. We then show a practical implementation of this with a finite set of shorter-term options to determine the hedging error using a Gaussian Quadrature method. We perform a wide range of experiments for both the \textit{Black-Scholes} and \textit{Merton Jump Diffusion} models, illustrating the comparative performance of the two methods.
    
[^2]: 从恒定到粗糙：连续波动率建模的综述

    From constant to rough: A survey of continuous volatility modeling. (arXiv:2309.01033v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.01033](http://arxiv.org/abs/2309.01033)

    本文综述了连续波动率模型的发展和关键事实，重点介绍了分数和粗糙方法的模型，并讨论了VIX建模问题和SPX-VIX联合校准难题的最新进展。

    

    本文全面调查了连续随机波动率模型，并讨论了它们的历史发展和驱动该领域的关键事实。我们特别关注分数和粗糙方法：我们概述了它们背后的动机，并对一些里程碑式模型进行了表征。此外，我们还简要讨论了VIX建模的问题以及SPX-VIX联合校准难题的最新进展。

    In this paper, we present a comprehensive survey of continuous stochastic volatility models, discussing their historical development and the key stylized facts that have driven the field. Special attention is dedicated to fractional and rough methods: we outline the motivation behind them and characterize some landmark models. In addition, we briefly touch the problem of VIX modeling and recent advances in the SPX-VIX joint calibration puzzle.
    
[^3]: 加密货币价格因素的建模：一种贝叶斯网络方法

    Modelling Determinants of Cryptocurrency Prices: A Bayesian Network Approach. (arXiv:2303.16148v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16148](http://arxiv.org/abs/2303.16148)

    本文使用贝叶斯网络方法，从因果分析的角度研究了影响替代加密货币价格的因素，包括五种主要替代加密货币、传统金融资产和社交媒体，提供了一种解决加密货币价格预测问题的方法。

    

    市场总值和替代比特币的加密货币数量的增长提供了投资机会，同时也增加了预测其价格波动的复杂度。在这个波动性相对较弱的市场中，预测加密货币价格的一个重要挑战是需要确定影响价格的因素。本研究的重点是从因果分析的角度研究影响替代比特币价格的因素，特别地，研究了五个主要的替代加密货币，包括黄金、石油和标准普尔500指数等传统金融资产以及社交媒体之间的相互作用。为了回答这个问题，我们创建了由五个传统金融资产的历史价格数据、社交媒体数据和替代加密货币价格数据构成的因果网络，这些网络用于因果推理和诊断。

    The growth of market capitalisation and the number of altcoins (cryptocurrencies other than Bitcoin) provide investment opportunities and complicate the prediction of their price movements. A significant challenge in this volatile and relatively immature market is the problem of predicting cryptocurrency prices which needs to identify the factors influencing these prices. The focus of this study is to investigate the factors influencing altcoin prices, and these factors have been investigated from a causal analysis perspective using Bayesian networks. In particular, studying the nature of interactions between five leading altcoins, traditional financial assets including gold, oil, and S\&P 500, and social media is the research question. To provide an answer to the question, we create causal networks which are built from the historic price data of five traditional financial assets, social media data, and price data of altcoins. The ensuing networks are used for causal reasoning and diag
    

