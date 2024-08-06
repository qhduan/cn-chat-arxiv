# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting Elastic String Models of Forward Interest Rates](https://arxiv.org/abs/2403.18126) | 该论文重新审视了正向利率的弹性字符串模型，通过解释市场力量对邻近期限利率的自我参照影响，实现了对整个FRC相关结构的准确再现，仅需两个稳定参数。 |
| [^2] | [A Note on Optimal Liquidation with Linear Price Impact](https://arxiv.org/abs/2402.14100) | 在二次交易成本设定下，提出了一个非常简单的概率解来最大化期望的终端财富，为分数布朗运动下投资者信息流多样化的情况提供了一般结果。 |
| [^3] | [Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending.](http://arxiv.org/abs/2401.16458) | 本文研究了如何利用P2P借贷平台上借款人提供的文本描述来构建风险指标。结果显示，利用大型语言模型生成的风险评分可以明显提高信用风险分类器的性能。 |
| [^4] | [The Inflation Attention Threshold and Inflation Surges.](http://arxiv.org/abs/2308.09480) | 本论文研究了通胀关注度与通胀激增之间的关系，发现在通胀率低稳定时，人们关注度较低，但一旦通胀率超过4%，关注度会明显增加，高关注区域的关注度是低关注区域的两倍。这种关注阈值的存在导致了状态依赖效应，即在宽松货币政策时，成本推动冲击对通胀的影响更大。这些结论有助于理解最近美国的通胀激增现象。 |
| [^5] | [Augmented Dynamic Gordon Growth Model.](http://arxiv.org/abs/2201.06012) | 本文介绍了一种增强的动态戈登增长模型，通过引入时间变化的即期利率和戈登增长模型扩充了模型。利用风险中性估值方法和局部风险最小化策略，获得了红利支付的欧式看涨和看跌期权、Margrabe交换期权以及股权链接的人寿保险产品的定价和对冲公式。同时提供了该模型的最大似然估计。 |

# 详细

[^1]: 重新审视正向利率的弹性字符串模型

    Revisiting Elastic String Models of Forward Interest Rates

    [https://arxiv.org/abs/2403.18126](https://arxiv.org/abs/2403.18126)

    该论文重新审视了正向利率的弹性字符串模型，通过解释市场力量对邻近期限利率的自我参照影响，实现了对整个FRC相关结构的准确再现，仅需两个稳定参数。

    

    25年前，几位作者提出将正向利率曲线（FRC）建模为一根弹性字符串，沿其传播特异冲击，考虑到不同到期日间回报相关性的独特结构。本文重新审视巴奎和布夏尔（2004）的“刚性”弹性字符串场论，使其微观基础更加透明。我们的模型可以解释市场力量对邻近期限利率的自我参照方式的影响。该模型简洁且能准确地在1994-2023年间再现整个FRC的相关结构，误差低于2%。我们仅需两个参数，其值除了2009-2014年量化宽松期间可能有所变化外非常稳定。时间分辨率（也称为Epps效应）对相关性的依赖也被忠实地再现。

    arXiv:2403.18126v1 Announce Type: new  Abstract: Twenty five years ago, several authors proposed to model the forward interest rate curve (FRC) as an elastic string along which idiosyncratic shocks propagate, accounting for the peculiar structure of the return correlation across different maturities. In this paper, we revisit the specific "stiff'' elastic string field theory of Baaquie and Bouchaud (2004) in a way that makes its micro-foundation more transparent. Our model can be interpreted as capturing the effect of market forces that set the rates of nearby tenors in a self-referential fashion. The model is parsimonious and accurately reproduces the whole correlation structure of the FRC over the time period 1994-2023, with an error below 2%. We need only two parameters, the values of which being very stable except perhaps during the Quantitative Easing period 2009-2014. The dependence of correlation on time resolution (also called the Epps effect) is also faithfully reproduced with
    
[^2]: 有关线性价格冲击下的最优清算的注释

    A Note on Optimal Liquidation with Linear Price Impact

    [https://arxiv.org/abs/2402.14100](https://arxiv.org/abs/2402.14100)

    在二次交易成本设定下，提出了一个非常简单的概率解来最大化期望的终端财富，为分数布朗运动下投资者信息流多样化的情况提供了一般结果。

    

    在这篇笔记中，我们考虑了在二次交易成本设定下最大化期望的终端财富。首先，我们为该问题提供了一个非常简单的概率解。尽管这个问题已经得到广泛研究，但据我们所知，这种简单和概率形式的解决方案在文献中尚未出现。接下来，我们应用了这一通用结果来研究风险资产由分数布朗运动给定且投资者的信息流可以进行多样化的情况。

    arXiv:2402.14100v1 Announce Type: new  Abstract: In this note we consider the maximization of the expected terminal wealth for the setup of quadratic transaction costs. First, we provide a very simple probabilistic solution to the problem. Although the problem was largely studied, as far as we know up to date this simple and probabilistic form of the solution has not appeared in the literature. Next, we apply the general result for the study of the case where the risky asset is given by a fractional Brownian Motion and the information flow of the investor can be diversified.
    
[^3]: 信用风险与大型语言模型相结合：从P2P借贷的贷款描述中构建风险指标。

    Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending. (arXiv:2401.16458v1 [q-fin.RM])

    [http://arxiv.org/abs/2401.16458](http://arxiv.org/abs/2401.16458)

    本文研究了如何利用P2P借贷平台上借款人提供的文本描述来构建风险指标。结果显示，利用大型语言模型生成的风险评分可以明显提高信用风险分类器的性能。

    

    P2P借贷作为一种独特的融资机制，通过在线平台将借款人与放款人联系起来。然而，P2P借贷面临信息不对称的挑战，因为放款人往往缺乏足够的数据来评估借款人的信用价值。本文提出了一种新颖的方法来解决这个问题，即利用借款人在贷款申请过程中提供的文本描述。我们的方法涉及使用大型语言模型（LLM）处理这些文本描述，LLM是一种能够识别文本中的模式和语义的强大工具。将迁移学习应用于将LLM适应特定任务。我们从Lending Club数据集的分析结果显示，BERT生成的风险评分显著提高了信用风险分类器的性能。然而，基于LLM的系统固有的不透明性，以及潜在偏差的不确定性，限制了其应用。

    Peer-to-peer (P2P) lending has emerged as a distinctive financing mechanism, linking borrowers with lenders through online platforms. However, P2P lending faces the challenge of information asymmetry, as lenders often lack sufficient data to assess the creditworthiness of borrowers. This paper proposes a novel approach to address this issue by leveraging the textual descriptions provided by borrowers during the loan application process. Our methodology involves processing these textual descriptions using a Large Language Model (LLM), a powerful tool capable of discerning patterns and semantics within the text. Transfer learning is applied to adapt the LLM to the specific task at hand.  Our results derived from the analysis of the Lending Club dataset show that the risk score generated by BERT, a widely used LLM, significantly improves the performance of credit risk classifiers. However, the inherent opacity of LLM-based systems, coupled with uncertainties about potential biases, unders
    
[^4]: 通胀关注阈值与通胀激增

    The Inflation Attention Threshold and Inflation Surges. (arXiv:2308.09480v1 [econ.GN])

    [http://arxiv.org/abs/2308.09480](http://arxiv.org/abs/2308.09480)

    本论文研究了通胀关注度与通胀激增之间的关系，发现在通胀率低稳定时，人们关注度较低，但一旦通胀率超过4%，关注度会明显增加，高关注区域的关注度是低关注区域的两倍。这种关注阈值的存在导致了状态依赖效应，即在宽松货币政策时，成本推动冲击对通胀的影响更大。这些结论有助于理解最近美国的通胀激增现象。

    

    在最近通胀激增爆发时，公众对通胀的关注度很低，但一旦通胀开始上升，关注度迅速增加。本文构建了一个一般均衡货币模型，该模型在通胀低而稳定时，最优化的策略是对通胀关注较少，但一旦通胀超过某个阈值，就会增加关注度。利用调查问卷中的通胀预期，我估计关注阈值在4%通胀率，高关注区域的关注度是低关注区域的两倍。当校准到这些发现时，该模型产生与美国最近通胀激增一致的通胀和通胀预期动态。关注阈值导致状态依赖性：成本推动冲击在货币宽松政策时更加通胀。这些状态依赖性效应在恒定关注或理性预期模型中是不存在的。

    At the outbreak of the recent inflation surge, the public's attention to inflation was low but increased rapidly once inflation started to rise. In this paper, I develop a general equilibrium monetary model where it is optimal for agents to pay little attention to inflation when inflation is low and stable, but in which they increase their attention once inflation exceeds a certain threshold. Using survey inflation expectations, I estimate the attention threshold to be at an inflation rate of 4%, with attention in the high-attention regime being twice as high as in the low-attention regime. When calibrated to match these findings, the model generates inflation and inflation expectation dynamics consistent with the recent inflation surge in the US. The attention threshold induces a state dependency: cost-push shocks become more inflationary in times of loose monetary policy. These state-dependent effects are absent in the model with constant attention or under rational expectations. Fol
    
[^5]: 增强的动态戈登增长模型

    Augmented Dynamic Gordon Growth Model. (arXiv:2201.06012v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2201.06012](http://arxiv.org/abs/2201.06012)

    本文介绍了一种增强的动态戈登增长模型，通过引入时间变化的即期利率和戈登增长模型扩充了模型。利用风险中性估值方法和局部风险最小化策略，获得了红利支付的欧式看涨和看跌期权、Margrabe交换期权以及股权链接的人寿保险产品的定价和对冲公式。同时提供了该模型的最大似然估计。

    

    本文引入了一种动态的戈登增长模型，其通过时间变化的即期利率和戈登增长模型来扩充。利用风险中性估值方法和局部风险最小化策略，我们获得了红利支付的欧式看涨和看跌期权、Margrabe交换期权以及股权链接的人寿保险产品的定价和对冲公式。此外，我们还提供了该模型的最大似然估计。

    In this paper, we introduce a dynamic Gordon growth model, which is augmented by a time-varying spot interest rate and the Gordon growth model for dividends. Using the risk-neutral valuation method and locally risk-minimizing strategy, we obtain pricing and hedging formulas for the dividend--paying European call and put options, Margrabe exchange options, and equity--linked life insurance products. Also, we provide ML estimator of the model.
    

