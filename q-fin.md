# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nash equilibria for relative investors with (non)linear price impact.](http://arxiv.org/abs/2303.18161) | 本文研究相对投资者之间的战略互动，找到了可以在线性价格影响下存在纳什均衡投资策略的结论，并发现当价格影响超过一个临界参数时，投资者会表现出激进的行为。 |
| [^2] | [Taureau: A Stock Market Movement Inference Framework Based on Twitter Sentiment Analysis.](http://arxiv.org/abs/2303.17667) | Taureau是一个基于Twitter情感分析预测股市走势的框架，利用历史推文进行矢量化和词语嵌入生成，用TextBlob分类推文情感并使用监督机器学习模型预测股市走势，并在实验评估中取得了高达78%的准确性。 |
| [^3] | [Do price trajectory data increase the efficiency of market impact estimation?.](http://arxiv.org/abs/2205.13423) | 本文研究了是否可以利用来自元订单的部分价格轨迹数据提高市场影响估计的效率，结果发现这种方法在流行的市场影响模型中效果更好。 |
| [^4] | [Firm Heterogeneity, Market Power and Macroeconomic Fragility.](http://arxiv.org/abs/2205.03908) | 本文研究了公司异质性和市场垄断如何影响宏观经济脆弱性，尤其是如何引起经济竞争驱动的贫困陷阱。研究模型的校准表明，已知的公司异质性趋势会显著增加缓慢复苏的可能性和持续时间。 |

# 详细

[^1]: 相对投资者的Nash平衡及（非）线性价格影响

    Nash equilibria for relative investors with (non)linear price impact. (arXiv:2303.18161v1 [math.OC])

    [http://arxiv.org/abs/2303.18161](http://arxiv.org/abs/2303.18161)

    本文研究相对投资者之间的战略互动，找到了可以在线性价格影响下存在纳什均衡投资策略的结论，并发现当价格影响超过一个临界参数时，投资者会表现出激进的行为。

    

    本文考虑了$n$个投资者的战略互动，他们能够影响股票价格，同时相对于其他投资者测量他们的效用。我们的主要目的是在由布朗运动驱动的金融市场中找到纳什均衡投资策略，并研究价格影响对均衡的影响。我们考虑了CRRA和CARA效用函数。我们的研究表明，只要价格影响最多是线性的，问题就是良好定义的。此外，数值结果表明，当价格影响超过一个临界参数时，投资者的行为非常激进。

    We consider the strategic interaction of $n$ investors who are able to influence a stock price process and at the same time measure their utilities relative to the other investors. Our main aim is to find Nash equilibrium investment strategies in this setting in a financial market driven by a Brownian motion and investigate the influence the price impact has on the equilibrium. We consider both CRRA and CARA utility functions. Our findings show that the problem is well-posed as long as the price impact is at most linear. Moreover, numerical results reveal that the investors behave very aggressively when the price impact is beyond a critical parameter.
    
[^2]: Taureau: 基于 Twitter 情感分析的股市走势推测框架

    Taureau: A Stock Market Movement Inference Framework Based on Twitter Sentiment Analysis. (arXiv:2303.17667v1 [cs.CY])

    [http://arxiv.org/abs/2303.17667](http://arxiv.org/abs/2303.17667)

    Taureau是一个基于Twitter情感分析预测股市走势的框架，利用历史推文进行矢量化和词语嵌入生成，用TextBlob分类推文情感并使用监督机器学习模型预测股市走势，并在实验评估中取得了高达78%的准确性。

    

    随着信息传播和检索的快速发展，利用自动化手段预测股市价格变得至关重要。本文提出了 Taureau，一个利用 Twitter 情感分析预测股市走势的框架。我们旨在确定 Twitter 是否代表一般大众，能否揭示公众对特定公司的看法并与公司的股价走势存在任何相关性。我们旨在利用这种相关性来预测股价走势。我们首先利用 Tweepy 和 getOldTweets 获取一组顶级公司在重大事件期间代表公众观点的历史推文。我们使用标准程序库对推文进行过滤和标记。然后，我们从获得的推文中进行矢量化和词语嵌入生成。之后，我们利用 TextBlob，一个最先进的情感分析引擎，帮助分类推文的情感为积极、消极或中性情绪。然后我们使用这些情感分数以及其他特征训练监督机器学习模型来预测股市的走势。实验评估表明，我们的框架优于几个现有基线模型，在预测股市走势方面实现高达78%的准确性。

    With the advent of fast-paced information dissemination and retrieval, it has become inherently important to resort to automated means of predicting stock market prices. In this paper, we propose Taureau, a framework that leverages Twitter sentiment analysis for predicting stock market movement. The aim of our research is to determine whether Twitter, which is assumed to be representative of the general public, can give insight into the public perception of a particular company and has any correlation to that company's stock price movement. We intend to utilize this correlation to predict stock price movement. We first utilize Tweepy and getOldTweets to obtain historical tweets indicating public opinions for a set of top companies during periods of major events. We filter and label the tweets using standard programming libraries. We then vectorize and generate word embedding from the obtained tweets. Afterward, we leverage TextBlob, a state-of-the-art sentiment analytics engine, to ass
    
[^3]: 价格轨迹数据是否能提高市场影响估计的效率？

    Do price trajectory data increase the efficiency of market impact estimation?. (arXiv:2205.13423v2 [q-fin.TR] UPDATED)

    [http://arxiv.org/abs/2205.13423](http://arxiv.org/abs/2205.13423)

    本文研究了是否可以利用来自元订单的部分价格轨迹数据提高市场影响估计的效率，结果发现这种方法在流行的市场影响模型中效果更好。

    

    市场影响是大型机构投资者和积极的市场参与者面临的一个重要问题。本文从统计估计的渐近视角，严格探讨了来自元订单的价格轨迹数据是否能提高估计的效率。我们展示了对于流行的市场影响模型，基于部分价格轨迹数据的估计方法，特别是包含早期交易价格的方法，可以在渐近意义下优于已建立的估计方法（例如以VWAP为基础的方法）。我们讨论了这种现象的理论和实证影响，以及如何将其轻松地应用到实践中。

    Market impact is an important problem faced by large institutional investor and active market participant. In this paper, we rigorously investigate whether price trajectory data from the metaorder increases the efficiency of estimation, from an asymptotic view of statistical estimation. We show that, for popular market impact models, estimation methods based on partial price trajectory data, especially those containing early trade prices, can outperform established estimation methods (e.g., VWAP-based) asymptotically. We discuss theoretical and empirical implications of such phenomenon, and how they could be readily incorporated into practice.
    
[^4]: 市场异质性、垄断和宏观经济脆弱性

    Firm Heterogeneity, Market Power and Macroeconomic Fragility. (arXiv:2205.03908v7 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.03908](http://arxiv.org/abs/2205.03908)

    本文研究了公司异质性和市场垄断如何影响宏观经济脆弱性，尤其是如何引起经济竞争驱动的贫困陷阱。研究模型的校准表明，已知的公司异质性趋势会显著增加缓慢复苏的可能性和持续时间。

    

    本文研究了公司异质性和市场垄断如何影响宏观经济的脆弱性，即长期衰退的概率。我们提出了一种理论，其中公司进入、竞争和要素供给之间的积极相互作用可以产生多个稳定状态。我们表明，当公司的异质性很大时，即使是小的暂时性冲击也可以引起公司退出并使经济进入一个竞争驱动的贫困陷阱。我们根据美国经济上不断上升的公司异质性的已知趋势来校准我们的模型，我们表明它们显着增加了经济缓慢复苏的可能性和持续时间。我们使用我们的框架来研究2008-09年的经济衰退，并证明该模型可以解释产出和大多数宏观经济聚合物与趋势的持久偏差，包括净进入、加权平均和劳动份额的行为。后危机经济数据也支持了我们的提出的机制。我们最后指出，公司补贴可以是防止经济陷入低增长均衡的强有力工具。

    We study how firm heterogeneity and market power affect macroeconomic fragility, defined as the probability of long slumps. We propose a theory in which the positive interaction between firm entry, competition and factor supply can give rise to multiple steady-states. We show that when firm heterogeneity is large, even small temporary shocks can trigger firm exit and make the economy spiral in a competition-driven poverty trap. We calibrate our model to incorporate the well-documented trends on rising firm heterogeneity in the US economy, we show that they significantly increase the likelihood and length of slow recoveries. We use our framework to study the 2008-09 recession and show that the model can rationalize the persistent deviation of output and most macroeconomic aggregates from trend, including the behavior of net entry, markups and the labor share. Post-crisis cross-industry data corroborates our proposed mechanism. We conclude by showing that firm subsidies can be powerful i
    

