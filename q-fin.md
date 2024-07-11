# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interbank network reconstruction enforcing density and reciprocity](https://arxiv.org/abs/2402.11136) | 银行间网络重构研究中，通过约束密度和互惠性，成功解决了有向循环复制问题，揭示了高阶循环统计量受最短循环数量约束的重要性 |
| [^2] | [Regime-Aware Asset Allocation: a Statistical Jump Model Approach](https://arxiv.org/abs/2402.05272) | 本文研究了制度转换对资产配置决策的影响，并比较了不同的制度识别模型。通过使用统计跳跃模型，作者开发了一种决策制度感知的资产配置策略，通过优化跳跃模型的性能指标，并在实证分析中展示了该策略相对于传统方法的表现优势。 |
| [^3] | [Consumption Partial Insurance in the Presence of Tail Income Risk.](http://arxiv.org/abs/2306.13208) | 该研究测量了收入冲击对消费保险的影响程度，证实尾部收入风险对消费具有重要影响，收入负面冲击对消费的传导率大于正面冲击。 |
| [^4] | [Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Iterative Refinement Labeling.](http://arxiv.org/abs/2107.11972) | 本论文提出了一种名为LARA的新型价格变动预测框架，包括两个主要部分：局部感知注意力和迭代细化标注。在真实世界的金融数据集上的实验结果表明，LARA在准确性和盈利能力方面优于现有的最先进方法。 |

# 详细

[^1]: 银行间网络重构: 密度和互惠性约束

    Interbank network reconstruction enforcing density and reciprocity

    [https://arxiv.org/abs/2402.11136](https://arxiv.org/abs/2402.11136)

    银行间网络重构研究中，通过约束密度和互惠性，成功解决了有向循环复制问题，揭示了高阶循环统计量受最短循环数量约束的重要性

    

    金融风险和危机在银行之间传播的网络是关键，但由于机密性，它们的实证结构并不公开。这一限制促使了从部分、总量信息中重构网络的方法的发展。不幸的是，即使是目前最好的方法也无法复制有向循环的数量，而这些循环在决定图谱及网络稳定度和系统性风险程度方面起着至关重要的作用。本文通过利用高阶循环统计量受最短循环数量（即具有互惠链接的双核数量）约束的假设来应对这一挑战。首先，我们在意大利银行e-MID数据集上对互惠链接进行了详细分析，发现互惠链接之间的相关性随着时间分辨率的增加而系统性增加，通常呈现混乱性

    arXiv:2402.11136v1 Announce Type: new  Abstract: Networks of financial exposures are the key propagators of risk and distress among banks, but their empirical structure is not publicly available because of confidentiality. This limitation has triggered the development of methods of network reconstruction from partial, aggregate information. Unfortunately, even the best methods available fail in replicating the number of directed cycles, which on the other hand play a crucial role in determining graph spectra and hence the degree of network stability and systemic risk. Here we address this challenge by exploiting the hypothesis that the statistics of higher-order cycles is strongly constrained by that of the shortest ones, i.e. by the amount of dyads with reciprocated links. First, we provide a detailed analysis of link reciprocity on the e-MID dataset of Italian banks, finding that correlations between reciprocal links systematically increase with the temporal resolution, typically cha
    
[^2]: 决策制度感知资产配置：基于统计跳跃模型方法的研究

    Regime-Aware Asset Allocation: a Statistical Jump Model Approach

    [https://arxiv.org/abs/2402.05272](https://arxiv.org/abs/2402.05272)

    本文研究了制度转换对资产配置决策的影响，并比较了不同的制度识别模型。通过使用统计跳跃模型，作者开发了一种决策制度感知的资产配置策略，通过优化跳跃模型的性能指标，并在实证分析中展示了该策略相对于传统方法的表现优势。

    

    本文研究了制度转换对资产配置决策的影响，重点比较了不同制度识别模型。与传统的马尔可夫转换模型不同，我们采用了统计跳跃模型，这是一种近期被提出的稳健模型，以其能够通过应用显式跳跃惩罚来捕捉持久的市场制度而闻名。我们的跳跃模型的特征集仅包括从价格序列中得出的收益和波动率特征。我们引入了一种在时间序列交叉验证框架内选择跳跃惩罚的数据驱动方法，该方法直接优化决策制度感知资产配置策略的性能指标，这一策略经历了全面的多步骤过程构建。通过使用美国主要股票指数的日回报序列进行实证分析，我们突出了采用跳跃模型相对于买入持有策略和马尔可夫转换资产配置方法的表现优势。

    This article investigates the impact of regime switching on asset allocation decisions, with a primary focus on comparing different regime identification models. In contrast to traditional Markov-switching models, we adopt the statistical jump model, a recently proposed robust model known for its ability to capture persistent market regimes by applying an explicit jump penalty. The feature set of our jump model comprises return and volatility features derived solely from the price series. We introduce a data-driven approach for selecting the jump penalty within a time-series cross-validation framework, which directly optimizes the performance metric of the regime-aware asset allocation strategy constructed following a comprehensive multi-step process. Through empirical analysis using daily return series from major US equity indices, we highlight the outperformance of employing jump models in comparison to both buy-and-hold strategies and Markov-switching asset allocation approaches. Th
    
[^3]: 在尾部收入风险的存在下，消费部分保险

    Consumption Partial Insurance in the Presence of Tail Income Risk. (arXiv:2306.13208v1 [econ.GN])

    [http://arxiv.org/abs/2306.13208](http://arxiv.org/abs/2306.13208)

    该研究测量了收入冲击对消费保险的影响程度，证实尾部收入风险对消费具有重要影响，收入负面冲击对消费的传导率大于正面冲击。

    

    我们通过考虑收入分布高阶矩的影响，衡量了收入冲击对消费保险的影响程度。我们导出了一个非线性消费函数，其中保险程度随着收入冲击的符号和大小而变化。利用PSID数据，我们估计了坏的相对于好的永久性冲击的非对称传导率-- 3 sigma负冲击的的17%传导到消费，而相同大小的正冲击只有9%传导到消费-- 随着冲击恶化，传导率增加。我们的结果与对假想事件消费反应的调查一致，并表明尾部收入风险对消费具有重要影响。

    We measure the extent of consumption insurance to income shocks accounting for high-order moments of the income distribution. We derive a nonlinear consumption function, in which the extent of insurance varies with the sign and magnitude of income shocks. Using PSID data, we estimate an asymmetric pass-through of bad versus good permanent shocks -- 17% of a 3 sigma negative shock transmits to consumption compared to 9% of an equal-sized positive shock -- and the pass-through increases as the shock worsens. Our results are consistent with surveys of consumption responses to hypothetical events and suggest that tail income risk matters substantially for consumption.
    
[^4]: 当机会来临时进行交易：基于注意力机制和迭代细化标注的价格变动预测

    Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Iterative Refinement Labeling. (arXiv:2107.11972v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2107.11972](http://arxiv.org/abs/2107.11972)

    本论文提出了一种名为LARA的新型价格变动预测框架，包括两个主要部分：局部感知注意力和迭代细化标注。在真实世界的金融数据集上的实验结果表明，LARA在准确性和盈利能力方面优于现有的最先进方法。

    

    价格变动预测旨在根据当前市场情况和其他相关信息预测金融资产的未来趋势。最近，机器学习（ML）方法在学术界和工业界中越来越受欢迎，并取得了令人满意的结果。然而，由于金融数据的低信噪比和随机性极强，好的交易机会极为稀少。因此，如果不仔细选择潜在的盈利样本，这些ML方法容易捕捉到噪声而不是真实信号的模式。为解决这个问题，本研究提出了一种名为LARA的新型价格变动预测框架，包括两个主要部分：局部感知注意力（LA-Attention）和迭代细化标注（IRL）。LA-Attention旨在有选择地关注金融数据中最具信息量的局部区域，而IRL则旨在迭代地细化标注过程，过滤掉噪声和无关样本。在真实世界的金融数据集上的实验结果表明，LARA在准确性和盈利能力方面优于现有的最先进方法。

    Price movement forecasting aims at predicting the future trends of financial assets based on the current market conditions and other relevant information. Recently, machine learning (ML) methods have become increasingly popular and achieved promising results for price movement forecasting in both academia and industry. Most existing ML solutions formulate the forecasting problem as a classification (to predict the direction) or a regression (to predict the return) problem over the entire set of training data. However, due to the extremely low signal-to-noise ratio and stochastic nature of financial data, good trading opportunities are extremely scarce. As a result, without careful selection of potentially profitable samples, such ML methods are prone to capture the patterns of noises instead of real signals. To address this issue, we propose a novel price movement forecasting framework named LARA consisting of two main components: Locality-Aware Attention (LA-Attention) and Iterative R
    

