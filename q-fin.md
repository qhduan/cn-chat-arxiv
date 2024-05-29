# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rationality Report Cards: Assessing the Economic Rationality of Large Language Models](https://arxiv.org/abs/2402.09552) | 本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。 |
| [^2] | [Decomposing Smiles: A Time Change Approach.](http://arxiv.org/abs/2401.03776) | 本论文提出了一种新颖的时间变化方法来研究隐含波动率微笑的形态。该方法适用于常见的半鞘模型，改进了基于矩的公式来近似平价选择权的偏斜和曲率，通过明确计算出这些矩，考虑了几个模型的极限偏斜和曲率，并通过数值方法和实证数据进行了测试，最后应用于校准问题。 |
| [^3] | [Proofs for the New Definitions in Financial Markets.](http://arxiv.org/abs/2309.03003) | 本研究提供了金融市场新定义的证明，通过构造定理来确定新定义中的效用曲线形状。与标准理论不同，新定义中出现了严格凹性、严格凸性或线性的情况。 |
| [^4] | [Empirical Evidence for the New Definitions in Financial Markets.](http://arxiv.org/abs/2305.03468) | 研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。 |
| [^5] | [Automated Market Making and Loss-Versus-Rebalancing.](http://arxiv.org/abs/2208.06046) | 本文研究常量函数市场制造者（CFMM）的市场微观结构，发现CFMM表现不如再平衡策略，提出了损失与再平衡(LVR)的概念，可用于评估投资决策和指导协议设计。 |

# 详细

[^1]: 理性报告卡：评估大型语言模型的经济合理性

    Rationality Report Cards: Assessing the Economic Rationality of Large Language Models

    [https://arxiv.org/abs/2402.09552](https://arxiv.org/abs/2402.09552)

    本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。

    

    越来越多的人对将LLM用作决策"代理人"兴趣日益增加。这包括很多自由度：应该使用哪个模型；如何进行提示；是否要求其进行内省、进行思考链等。解决这些问题（更广泛地说，确定LLM代理人是否足够可靠以便获得信任）需要一种评估这种代理人经济合理性的方法论，在本文中我们提供了一个方法。我们首先对理性决策的经济文献进行了调研、将代理人应该展现的大量细粒度"要素"进行分类，并确定了它们之间的依赖关系。然后，我们提出了一个基准分布，以定量评分LLM在这些要素上的表现，并结合用户提供的评分标准，生成一份"理性报告卡"。最后，我们描述了与14种不同的LLM进行的大规模实证实验的结果。

    arXiv:2402.09552v1 Announce Type: new  Abstract: There is increasing interest in using LLMs as decision-making "agents." Doing so includes many degrees of freedom: which model should be used; how should it be prompted; should it be asked to introspect, conduct chain-of-thought reasoning, etc? Settling these questions -- and more broadly, determining whether an LLM agent is reliable enough to be trusted -- requires a methodology for assessing such an agent's economic rationality. In this paper, we provide one. We begin by surveying the economic literature on rational decision making, taxonomizing a large set of fine-grained "elements" that an agent should exhibit, along with dependencies between them. We then propose a benchmark distribution that quantitatively scores an LLMs performance on these elements and, combined with a user-provided rubric, produces a "rationality report card." Finally, we describe the results of a large-scale empirical experiment with 14 different LLMs, characte
    
[^2]: 分解微笑：一种时间变化方法

    Decomposing Smiles: A Time Change Approach. (arXiv:2401.03776v1 [q-fin.PR])

    [http://arxiv.org/abs/2401.03776](http://arxiv.org/abs/2401.03776)

    本论文提出了一种新颖的时间变化方法来研究隐含波动率微笑的形态。该方法适用于常见的半鞘模型，改进了基于矩的公式来近似平价选择权的偏斜和曲率，通过明确计算出这些矩，考虑了几个模型的极限偏斜和曲率，并通过数值方法和实证数据进行了测试，最后应用于校准问题。

    

    我们采用一种新颖的时间变化方法来研究隐含波动率微笑的形状。该方法适用于常见的半鞘模型，包括跳跃扩散模型、粗糙波动模型和无限活动模型。我们用改进的基于矩的公式近似平价选择权的偏斜和曲率。在时间变化框架下明确计算出这些矩。对于几个模型，我们考虑了极限的偏斜和曲率。我们还通过数值方法和实证数据对短期逼近结果的准确性进行了测试。最后，我们将该方法应用于校准问题。

    We develop a novel time-change approach to study the shape of implied volatility smiles. The method is applicable to common semimartingale models, including jump-diffusion, rough volatility and infinite activity models. We approximate the at-the-money skew and curvature with an improved moment-based formula. The moments are further explicitly computed under a time change framework. The limiting skew and curvature for several models are considered. We also test the accuracy of the short-term approximation results on models via numerical methods and on empirical data. Finally, we apply the method to the calibration problem.
    
[^3]: 金融市场新定义的证明

    Proofs for the New Definitions in Financial Markets. (arXiv:2309.03003v1 [q-fin.GN])

    [http://arxiv.org/abs/2309.03003](http://arxiv.org/abs/2309.03003)

    本研究提供了金融市场新定义的证明，通过构造定理来确定新定义中的效用曲线形状。与标准理论不同，新定义中出现了严格凹性、严格凸性或线性的情况。

    

    构造定理可以帮助确定金融市场新定义中构成的某些效用曲线的形状。本研究旨在为这些定理提供证明。尽管风险厌恶、风险爱好和风险中性等术语在标准理论中分别等同于严格凹性、严格凸性和线性，但某些新定义满足严格凹性或严格凸性，或线性。

    Constructing theorems can help to determine the shape of certain utility curves that make up the new definitions in financial markets. The aim of this study was to present proofs for these theorems. Although the terms of risk-averse, risk-loving, and risk-neutral are equivalent to strict concavity, strict convexity, and linearity, respectively, in standard theory, certain new definitions satisfy strict concavity or strict convexity, or linearity.
    
[^4]: 金融市场新定义的经验证据

    Empirical Evidence for the New Definitions in Financial Markets. (arXiv:2305.03468v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.03468](http://arxiv.org/abs/2305.03468)

    研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。

    

    本研究给出了支持金融市场新定义的经验证据。分析了1889-1978年美国金融市场投资者的风险态度，结果表明，1977年在投资综合S＆P 500指数的股票投资者是风险规避者。相反，投资美国国债的无风险资产投资者则表现出不足的风险偏爱，这可以被认为是一种风险规避行为。这些发现表明，金融市场新定义准确反映了投资者的行为，应考虑在投资策略中。

    This study presents empirical evidence to support the validity of new definitions in financial markets. The risk attitudes of investors in US financial markets from 1889-1978 are analyzed and the results indicate that equity investors who invested in the composite S&P 500 index were risk-averse in 1977. Conversely, risk-free asset investors who invested in US Treasury bills were found to exhibit not enough risk-loving behavior, which can be considered a type of risk-averse behavior. These findings suggest that the new definitions in financial markets accurately reflect the behavior of investors and should be considered in investment strategies.
    
[^5]: 自动化市场做市商及损失与再平衡。

    Automated Market Making and Loss-Versus-Rebalancing. (arXiv:2208.06046v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2208.06046](http://arxiv.org/abs/2208.06046)

    本文研究常量函数市场制造者（CFMM）的市场微观结构，发现CFMM表现不如再平衡策略，提出了损失与再平衡(LVR)的概念，可用于评估投资决策和指导协议设计。

    

    我们从被动流动性提供者的角度考虑了常量函数市场制造者（CFMM）的市场微观结构。在Black-Scholes情境下，我们比较CFMM的表现与再平衡策略的表现，该策略在市场价格上复制CFMM的交易。由于CFMM以比市场价格更差的价格执行所有交易，因此CFMM系统地表现不佳。两种策略之间的绩效差异“损失与再平衡”（LVR，发音为“lever”）取决于基础资产的波动率和CFMM bonding函数的边际流动性。我们模型中的CFMM损失表达式与Uniswap v2 WETH-USDC对的实际损失相匹配。LVR为CFMM流动性提供者的投资决策的前后评估提供了可交易的见解，并且还可以指导CFMM协议的设计。

    We consider the market microstructure of constant function market makers (CFMMs) from the perspective of passive liquidity providers (LPs). In a Black-Scholes setting, we compare the CFMM's performance to that of a rebalancing strategy, which replicates the CFMM's trades at market prices. The CFMM systematically underperforms the rebalancing strategy, because it executes all trades at worse-than-market prices. The performance gap between the two strategies, "loss-versus-rebalancing" (LVR, pronounced "lever"), depends on the volatility of the underlying asset and the marginal liquidity of the CFMM bonding function. Our model's expressions for CFMM losses match actual losses from the Uniswap v2 WETH-USDC pair. LVR provides tradeable insight in both the ex ante and ex post assessment of CFMM LP investment decisions, and can also inform the design of CFMM protocols.
    

