# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Visibility graph analysis of crude oil futures markets: Insights from the COVID-19 pandemic and Russia-Ukraine conflict.](http://arxiv.org/abs/2310.18903) | 该研究通过可见度图方法对原油期货市场进行了深入分析，发现了VG的幂律衰减和显著聚类特性。研究结果揭示了原油期货市场的小世界属性和复杂同类混合特征。WTI和Brent市场表现一致，而SC市场有所偏离。 |
| [^2] | [Effective a Posteriori Ratemaking with Large Insurance Portfolios via Surrogate Modeling.](http://arxiv.org/abs/2211.06568) | 本文针对保险中高效的后验费率制定问题，提出了使用代理模型方法来获得封闭式表达式计算保费的方案，同时引进了可信度指数作为保费计算的摘要统计信息，避免了因计算代价高昂或计算过程成为“黑盒”而带来的限制。 |
| [^3] | [PreBit -- A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin.](http://arxiv.org/abs/2206.00648) | 本文提出了一种利用多模态模型进行比特币极端价格波动预测的方法，将相关资产、技术指标和Twitter内容作为输入。通过使用预训练的金融词汇表的句级FinBERT嵌入，模型可以有效地捕捉推文中的内容，从而预测比特币的价格波动。 |

# 详细

[^1]: 原油期货市场的可见度图分析：来自COVID-19疫情和俄乌冲突的见解

    Visibility graph analysis of crude oil futures markets: Insights from the COVID-19 pandemic and Russia-Ukraine conflict. (arXiv:2310.18903v1 [q-fin.ST])

    [http://arxiv.org/abs/2310.18903](http://arxiv.org/abs/2310.18903)

    该研究通过可见度图方法对原油期货市场进行了深入分析，发现了VG的幂律衰减和显著聚类特性。研究结果揭示了原油期货市场的小世界属性和复杂同类混合特征。WTI和Brent市场表现一致，而SC市场有所偏离。

    

    在全球金融市场受到俄乌冲突和COVID-19疫情的重大影响的背景下，本研究对三个主要的原油期货市场（WTI、Brent和上海（SC））进行了彻底的分析。采用可见度图（VG）方法，我们使用日常和高频数据来研究静态和动态特性。我们发现大多数VG度分布显示出明显的幂律衰减，并强调了原油期货VG内的显著聚类特性。我们的研究结果还证实了聚类系数与节点度数之间的负相关关系，并进一步揭示了所有VG不仅遵循小世界属性，而且呈现复杂的同类混合。通过VG的时变特性，我们发现WTI和Brent表现出一致的行为，而具有独特交易机制的SC市场则有所偏离。5分钟VG的同类混合系数提供了一种深入分析原油期货市场的方法。

    Drawing inspiration from the significant impact of the ongoing Russia-Ukraine conflict and the recent COVID-19 pandemic on global financial markets, this study conducts a thorough analysis of three key crude oil futures markets: WTI, Brent, and Shanghai (SC). Employing the visibility graph (VG) methodology, we examine both static and dynamic characteristics using daily and high-frequency data. We identified a clear power-law decay in most VG degree distributions and highlighted the pronounced clustering tendencies within crude oil futures VGs. Our results also confirm an inverse correlation between clustering coefficient and node degree and further reveal that all VGs not only adhere to the small-world property but also exhibit intricate assortative mixing. Through the time-varying characteristics of VGs, we found that WTI and Brent demonstrate aligned behavior, while the SC market, with its unique trading mechanics, deviates. The 5-minute VGs' assortativity coefficient provides a deep
    
[^2]: 基于代理建模的大型保险投资组合高效后验费率制定

    Effective a Posteriori Ratemaking with Large Insurance Portfolios via Surrogate Modeling. (arXiv:2211.06568v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2211.06568](http://arxiv.org/abs/2211.06568)

    本文针对保险中高效的后验费率制定问题，提出了使用代理模型方法来获得封闭式表达式计算保费的方案，同时引进了可信度指数作为保费计算的摘要统计信息，避免了因计算代价高昂或计算过程成为“黑盒”而带来的限制。

    

    保险中的后验费率制定利用贝叶斯可信度模型来升级合同的当前保费，考虑到投保人的属性和索赔记录。大多数用于此任务的数据驱动模型在数学上是难以处理的，因此必须通过数值方法如MCMC等模拟方法来获得保费。然而，这些方法在策略保持者层面上进行时，对于大型投资组合而言计算代价昂贵且具有限制性。此外，这些计算变成了“黑盒”处理，因为没有表达式显示如何使用策略保持者的索赔历史来升级其保费。为了解决这些挑战，本文提出采用代理建模方法，以便于低成本地为任何给定模型计算出贝叶斯可信度保费的封闭式表达式。作为方法的一部分，本文介绍了“可信度指数”，它是策略保持者索赔的摘要统计信息。

    A posteriori ratemaking in insurance uses a Bayesian credibility model to upgrade the current premiums of a contract by taking into account policyholders' attributes and their claim history. Most data-driven models used for this task are mathematically intractable, and premiums must be then obtained through numerical methods such as simulation such MCMC. However, these methods can be computationally expensive and prohibitive for large portfolios when applied at the policyholder level. Additionally, these computations become ``black-box" procedures as there is no expression showing how the claim history of policyholders is used to upgrade their premiums. To address these challenges, this paper proposes the use of a surrogate modeling approach to inexpensively derive a closed-form expression for computing the Bayesian credibility premiums for any given model. As a part of the methodology, the paper introduces the ``credibility index", which is a summary statistic of a policyholder's clai
    
[^3]: PreBit -- 一种利用Twitter FinBERT嵌入的多模态模型，用于比特币的极端价格波动预测。

    PreBit -- A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin. (arXiv:2206.00648v2 [q-fin.ST] UPDATED)

    [http://arxiv.org/abs/2206.00648](http://arxiv.org/abs/2206.00648)

    本文提出了一种利用多模态模型进行比特币极端价格波动预测的方法，将相关资产、技术指标和Twitter内容作为输入。通过使用预训练的金融词汇表的句级FinBERT嵌入，模型可以有效地捕捉推文中的内容，从而预测比特币的价格波动。

    

    比特币以其不断增长的受欢迎程度展示了自其诞生以来的极端价格波动性。这种波动性，加上其去中心化的性质，使比特币相对于更传统的资产更容易受到投机交易的影响。本文提出了一种用于预测极端价格波动的多模态模型。该模型将各种相关资产、技术指标以及Twitter内容作为输入。在一项深入研究中，我们探讨了来自大众对比特币的社交媒体讨论是否具有极端价格波动的预测能力。我们收集了从2015年到2021年每天包含关键词“比特币”的5,000条推文的数据集，称为PreBit，并将其在网上提供。在我们的混合模型中，我们使用在金融词汇表上预训练的句级FinBERT嵌入，以便以可理解的方式捕捉推文的全部内容并将其提供给模型。通过将这些嵌入与一种卷积层结合起来，我们可以提取推文的特征信息，并在模型中进行极端价格波动的预测。

    Bitcoin, with its ever-growing popularity, has demonstrated extreme price volatility since its origin. This volatility, together with its decentralised nature, make Bitcoin highly subjective to speculative trading as compared to more traditional assets. In this paper, we propose a multimodal model for predicting extreme price fluctuations. This model takes as input a variety of correlated assets, technical indicators, as well as Twitter content. In an in-depth study, we explore whether social media discussions from the general public on Bitcoin have predictive power for extreme price movements. A dataset of 5,000 tweets per day containing the keyword `Bitcoin' was collected from 2015 to 2021. This dataset, called PreBit, is made available online. In our hybrid model, we use sentence-level FinBERT embeddings, pretrained on financial lexicons, so as to capture the full contents of the tweets and feed it to the model in an understandable way. By combining these embeddings with a Convoluti
    

