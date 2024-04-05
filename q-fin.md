# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BERTopic-Driven Stock Market Predictions: Unraveling Sentiment Insights](https://arxiv.org/abs/2404.02053) | 这项研究利用BERTopic分析股市评论中的情感，整合深度学习模型，显示情感分析显著提升了股市预测性能，揭示了NLP在丰富金融分析方面的潜力。 |
| [^2] | [Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection.](http://arxiv.org/abs/2401.03737) | 本文介绍了MarketSenseAI，一个利用GPT-4进行股票选择的人工智能框架，融合了多种数据源和推理能力，提供具有可行解释的投资信号。 |
| [^3] | [Reconciling Open Interest with Traded Volume in Perpetual Swaps.](http://arxiv.org/abs/2310.14973) | 本研究提出一种方法，通过调和持仓量和交易量来解决永续掉期中的问题。持仓量是衍生品市场中的关键指标，它提供了市场活动、情绪和整体流动性的见解。研究还发现，通过估计担保的下限，可以评估交易所的杠杆水平是否可持续运营。 |
| [^4] | [Approximating the set of Nash equilibria for convex games.](http://arxiv.org/abs/2310.04176) | 本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。 |
| [^5] | [Improved Financial Forecasting via Quantum Machine Learning.](http://arxiv.org/abs/2306.12965) | 本研究利用量子机器学习提升了金融预测的表现，包括使用行列式点过程来增强随机森林模型进行流失预测并设计了量子神经网络架构用于信用风险评估，比传统方法使用更少的参数达到相似的性能。 |
| [^6] | [The use of trade data in the analysis of global phosphate flows.](http://arxiv.org/abs/2305.07362) | 本文提出了一种利用贸易数据追踪磷流动的新方法，可以为环境会计的准确性做出贡献。 |
| [^7] | [Beyond Unbounded Beliefs: How Preferences and Information Interplay in Social Learning.](http://arxiv.org/abs/2103.02754) | 本文研究了在社会学习中，偏好和信息如何相互作用。通过排他性条件，我们发现社会在学习中最重要的是一个单个代理人能够替代任何错误的行动，即使不能采取正确的行动。 |

# 详细

[^1]: BERTopic驱动的股市预测：解析情感洞见

    BERTopic-Driven Stock Market Predictions: Unraveling Sentiment Insights

    [https://arxiv.org/abs/2404.02053](https://arxiv.org/abs/2404.02053)

    这项研究利用BERTopic分析股市评论中的情感，整合深度学习模型，显示情感分析显著提升了股市预测性能，揭示了NLP在丰富金融分析方面的潜力。

    

    这篇论文探讨了自然语言处理（NLP）和金融分析的交叉领域，重点关注情感分析在股价预测中的影响。我们采用了BERTopic，一种先进的NLP技术，分析从股市评论中得出的主题的情感。我们的方法将这种情感分析与各种深度学习模型相整合，这些模型以其在时间序列和股票预测任务中的有效性而闻名。通过全面的实验，我们证明了整合主题情感显著提升了这些模型的性能。结果表明，股市评论中的主题提供了对股市波动和价格趋势的隐含、有价值的洞见。这项研究通过展示NLP在丰富金融分析方面的潜力，为实时情感分析和探索情感和情景相关性打开了研究途径。

    arXiv:2404.02053v1 Announce Type: new  Abstract: This paper explores the intersection of Natural Language Processing (NLP) and financial analysis, focusing on the impact of sentiment analysis in stock price prediction. We employ BERTopic, an advanced NLP technique, to analyze the sentiment of topics derived from stock market comments. Our methodology integrates this sentiment analysis with various deep learning models, renowned for their effectiveness in time series and stock prediction tasks. Through comprehensive experiments, we demonstrate that incorporating topic sentiment notably enhances the performance of these models. The results indicate that topics in stock market comments provide implicit, valuable insights into stock market volatility and price trends. This study contributes to the field by showcasing the potential of NLP in enriching financial analysis and opens up avenues for further research into real-time sentiment analysis and the exploration of emotional and contextua
    
[^2]: 能否打败华尔街？揭示人工智能在股票选择中的潜力

    Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection. (arXiv:2401.03737v1 [q-fin.CP])

    [http://arxiv.org/abs/2401.03737](http://arxiv.org/abs/2401.03737)

    本文介绍了MarketSenseAI，一个利用GPT-4进行股票选择的人工智能框架，融合了多种数据源和推理能力，提供具有可行解释的投资信号。

    

    在金融市场动态和数据驱动的环境中，本文介绍了MarketSenseAI，一个利用GPT-4先进推理能力进行可扩展股票选择的新型人工智能框架。MarketSenseAI整合了“思维链”和“上下文学习”方法，分析包括市场价格动态、财经新闻、公司基本面和宏观经济报告等多种数据源，模仿知名金融投资团队的决策过程。文章详细介绍了MarketSenseAI的开发、实施和实证验证，重点关注其提供具有充分解释支撑的可行投资信号（买入、持有、卖出）的能力。本研究的一个显著特点是使用GPT-4不仅作为预测工具，还作为评估器，揭示了人工智能生成的解释对所建议的投资信号的可靠性和接受度的重要影响。通过广泛的实证评估

    In the dynamic and data-driven landscape of financial markets, this paper introduces MarketSenseAI, a novel AI-driven framework leveraging the advanced reasoning capabilities of GPT-4 for scalable stock selection. MarketSenseAI incorporates Chain of Thought and In-Context Learning methodologies to analyze a wide array of data sources, including market price dynamics, financial news, company fundamentals, and macroeconomic reports emulating the decision making process of prominent financial investment teams. The development, implementation, and empirical validation of MarketSenseAI are detailed, with a focus on its ability to provide actionable investment signals (buy, hold, sell) backed by cogent explanations. A notable aspect of this study is the use of GPT-4 not only as a predictive tool but also as an evaluator, revealing the significant impact of the AI-generated explanations on the reliability and acceptance of the suggested investment signals. In an extensive empirical evaluation
    
[^3]: 在永续掉期中实现持仓量与交易量的调和

    Reconciling Open Interest with Traded Volume in Perpetual Swaps. (arXiv:2310.14973v1 [q-fin.TR])

    [http://arxiv.org/abs/2310.14973](http://arxiv.org/abs/2310.14973)

    本研究提出一种方法，通过调和持仓量和交易量来解决永续掉期中的问题。持仓量是衍生品市场中的关键指标，它提供了市场活动、情绪和整体流动性的见解。研究还发现，通过估计担保的下限，可以评估交易所的杠杆水平是否可持续运营。

    

    永续掉期是一种衍生合约，允许交易者就加密货币的价格变动进行投机或对冲。与传统意义上的期货合约不同，永续掉期没有结算或到期日。资金费率作为一种机制，通过套利者的帮助将永续掉期与其基础资产联系起来。在永续掉期和衍生合约的背景下，持仓量指的是在特定时间点的未平仓合约的总数。作为衍生品市场的关键指标，持仓量可以提供市场活动、情绪和整体流动性的见解。它还提供了一种估计交易所上每个加密货币市场所需担保的下限的方式。通过在交易所上所有市场上累积这个数字，并结合储备证明，可以判断所讨论的交易所是否采用不可持续的杠杆水平运营，这可能会带来偿付能力的影响。

    Perpetual swaps are derivative contracts that allow traders to speculate on, or hedge, the price movements of cryptocurrencies. Unlike futures contracts perpetual swaps have no settlement or expiration, in the traditional sense. The funding rate acts as the mechanism that tethers the perpetual swap to its underlying with the help of arbitrageurs.  Open interest, in the context of perpetual swaps and derivative contracts in general, refers to the total number of outstanding contracts at a given point in time. It is a critical metric in derivatives markets as it can provide insight into market activity, sentiment and overall liquidity. It also provides a way to estimate a lower bound on the collateral required for every cryptocurrency market on an exchange. This number, cumulated across all markets on the exchange in combination with proof of reserves can be used to gauge whether the exchange in question operates with unsustainable levels of leverage; which could have solvency implicatio
    
[^4]: 近似计算凸博弈中纳什均衡解集合

    Approximating the set of Nash equilibria for convex games. (arXiv:2310.04176v1 [math.OC])

    [http://arxiv.org/abs/2310.04176](http://arxiv.org/abs/2310.04176)

    本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。

    

    在Feinstein和Rudloff（2023）中，他们证明了对于任意非合作$N$人博弈，纳什均衡解集合与具有非凸顺序锥的某个向量优化问题的帕累托最优点集合是一致的。为了避免处理非凸顺序锥，我们证明了将纳什均衡解集合等价地表示为$N$个多目标问题（即具有自然顺序锥）的帕累托最优点的交集。目前，计算多目标问题的精确帕累托最优点集合的算法仅适用于线性问题的类别，这将导致这些算法只能用于解线性博弈的真实纳什均衡集合的可能性降低。本文中，我们将考虑更大类别的凸博弈。由于通常只能为凸向量优化问题计算近似解，我们首先展示了类似于上述结果的结果，即$\epsilon$-近似纳什均衡解集合与问题完全相似。

    In Feinstein and Rudloff (2023), it was shown that the set of Nash equilibria for any non-cooperative $N$ player game coincides with the set of Pareto optimal points of a certain vector optimization problem with non-convex ordering cone. To avoid dealing with a non-convex ordering cone, an equivalent characterization of the set of Nash equilibria as the intersection of the Pareto optimal points of $N$ multi-objective problems (i.e.\ with the natural ordering cone) is proven. So far, algorithms to compute the exact set of Pareto optimal points of a multi-objective problem exist only for the class of linear problems, which reduces the possibility of finding the true set of Nash equilibria by those algorithms to linear games only.  In this paper, we will consider the larger class of convex games. As, typically, only approximate solutions can be computed for convex vector optimization problems, we first show, in total analogy to the result above, that the set of $\epsilon$-approximate Nash
    
[^5]: 基于量子机器学习的金融预测的改进

    Improved Financial Forecasting via Quantum Machine Learning. (arXiv:2306.12965v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.12965](http://arxiv.org/abs/2306.12965)

    本研究利用量子机器学习提升了金融预测的表现，包括使用行列式点过程来增强随机森林模型进行流失预测并设计了量子神经网络架构用于信用风险评估，比传统方法使用更少的参数达到相似的性能。

    

    量子算法有潜力提高机器学习在各领域和应用中的表现。在本文中，我们展示了如何使用量子机器学习来改进金融预测。首先，我们使用经典和量子行列式点过程来增强随机森林模型以进行流失预测，提高了近6％的精度。其次，我们设计了具有正交和复合层的量子神经网络架构，用较少的参数达到了与经典性能相当的信用风险评估效果。我们的结果表明，利用量子思想可以有效提升机器学习的表现，无论是现在作为量子启发式的经典ML解决方案，还是在未来更好的量子硬件的到来时。

    Quantum algorithms have the potential to enhance machine learning across a variety of domains and applications. In this work, we show how quantum machine learning can be used to improve financial forecasting. First, we use classical and quantum Determinantal Point Processes to enhance Random Forest models for churn prediction, improving precision by almost 6%. Second, we design quantum neural network architectures with orthogonal and compound layers for credit risk assessment, which match classical performance with significantly fewer parameters. Our results demonstrate that leveraging quantum ideas can effectively enhance the performance of machine learning, both today as quantum-inspired classical ML solutions, and even more in the future, with the advent of better quantum hardware.
    
[^6]: 利用贸易数据分析全球磷流动的研究

    The use of trade data in the analysis of global phosphate flows. (arXiv:2305.07362v1 [econ.GN])

    [http://arxiv.org/abs/2305.07362](http://arxiv.org/abs/2305.07362)

    本文提出了一种利用贸易数据追踪磷流动的新方法，可以为环境会计的准确性做出贡献。

    

    本文介绍了一种跟踪磷从开采国到农业生产国使用的新方法。我们通过将磷岩采矿数据与化肥使用数据和磷相关产品的国际贸易数据相结合来实现目标。我们展示了通过对净出口数据进行某些调整，我们可以在很大程度上推导出国家层面上的磷流矩阵，并因此为物质流分析的准确性做出贡献，这对于改进环境会计不仅对于磷，还适用于许多其他资源至关重要。

    In this paper we present a new method to trace the flows of phosphate from the countries where it is mined to the counties where it is used in agricultural production. We achieve this by combining data on phosphate rock mining with data on fertilizer use and data on international trade of phosphate-related products. We show that by making certain adjustments to data on net exports we can derive the matrix of phosphate flows on the country level to a large degree and thus contribute to the accuracy of material flow analyses, a results that is important for improving environmental accounting, not only for phosphorus but for many other resources.
    
[^7]: 超越无界信念：偏好和信息在社会学习中的相互作用

    Beyond Unbounded Beliefs: How Preferences and Information Interplay in Social Learning. (arXiv:2103.02754v4 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2103.02754](http://arxiv.org/abs/2103.02754)

    本文研究了在社会学习中，偏好和信息如何相互作用。通过排他性条件，我们发现社会在学习中最重要的是一个单个代理人能够替代任何错误的行动，即使不能采取正确的行动。

    

    在社会网络中的连续学习模型中，我们确定了一种称为排他性的学习条件，用于判断社会何时最终学到真理或采取正确的行动。排他性是代理人偏好和信息的共同特征。当需要适用于所有偏好时，它等同于信息具有“无界信念”，这要求任何代理人都可以以小概率单独确定真相。但是对于超过两个状态，无界信念可能是不能持久的：例如，它与单调似然比特性不相容。排他性揭示了对于学习来说至关重要的是，单个代理人必须能够替代任何错误的行动，即使她不能采取正确的行动。我们提出了两类偏好和信息，它们共同满足了排他性条件。

    When does society eventually learn the truth, or take the correct action, via observational learning? In a general model of sequential learning over social networks, we identify a simple condition for learning dubbed excludability. Excludability is a joint property of agents' preferences and their information. When required to hold for all preferences, it is equivalent to information having "unbounded beliefs", which demands that any agent can individually identify the truth, even if only with small probability. But unbounded beliefs may be untenable with more than two states: e.g., it is incompatible with the monotone likelihood ratio property. Excludability reveals that what is crucial for learning, instead, is that a single agent must be able to displace any wrong action, even if she cannot take the correct action. We develop two classes of preferences and information that jointly satisfy excludability: (i) for a one-dimensional state, preferences with single-crossing differences an
    

