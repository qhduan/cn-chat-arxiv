# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Rebalancing in Dynamic AMMs](https://arxiv.org/abs/2403.18737) | 该论文提出了在动态AMM中实现最优再平衡的方法，通过引入套利机会来调整持有物，采用便宜计算的近似最优方法取得了性能改进。 |
| [^2] | [Limited Attention Allocation in a Stochastic Linear Quadratic System with Multiplicative Noise](https://arxiv.org/abs/2403.18528) | 本研究在具有乘法噪声的随机线性二次系统中解决了有限注意分配问题，通过战略资源分配提高噪声估计并改善控制决策。 |
| [^3] | [Redirecting Flows -- Navigating the Future of the Amazon](https://arxiv.org/abs/2403.18521) | 亚马逊流域及拉丁美洲和加勒比地区正经历环境挑战与变革潜力并存的时刻，本报告旨在揭示各种社会生态问题、技术进步、社区倡议和战略行动，有助于推动区域的生物圈可持续性和韧性。 |
| [^4] | [Growth rate of liquidity provider's wealth in G3Ms](https://arxiv.org/abs/2403.18177) | 该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。 |
| [^5] | [Revisiting Elastic String Models of Forward Interest Rates](https://arxiv.org/abs/2403.18126) | 该论文重新审视了正向利率的弹性字符串模型，通过解释市场力量对邻近期限利率的自我参照影响，实现了对整个FRC相关结构的准确再现，仅需两个稳定参数。 |
| [^6] | [Deep Limit Order Book Forecasting](https://arxiv.org/abs/2403.09267) | 该研究利用深度学习方法预测纳斯达克交易所股票的限价订单簿中间价格变动，提出了一个创新的操作框架来评估预测的实用性。 |
| [^7] | [Closed-form solutions for generic N-token AMM arbitrage](https://arxiv.org/abs/2402.06731) | 本文提出了通用N-token AMM套利的闭合形式解决方案。该解决方案在模拟中表现出比凸优化更好的套利机会，并能更快地利用这些机会。并且，所提出的方法具有并行计算能力，能够在GPU上扩展，为AMM建模提供了一种新方法。此外，该机制的低计算成本还可以实现多资产池的链上套利机器人。 |
| [^8] | [NLP-based detection of systematic anomalies among the narratives of consumer complaints.](http://arxiv.org/abs/2308.11138) | 本文开发了一种基于自然语言处理的方法，用于检测消费者投诉叙述中的系统异常。这种方法可以解决分类算法对于较小且频繁出现的系统异常检测的问题，并将投诉叙述转化为定量数据进行分析。 |
| [^9] | [The multidimensional COS method for option pricing.](http://arxiv.org/abs/2307.12843) | 多维COS方法是一种用于定价依赖于多个标的的金融期权的数值工具。本文证明了该方法的收敛性，并分析了数值不确定度对该方法的影响。 |
| [^10] | [Coevolution of cognition and cooperation in structured populations under reinforcement learning.](http://arxiv.org/abs/2306.11376) | 本研究发现在囚徒困境中，确定的重复交互概率阈值将导致智能体从直觉的背叛者转变为双重过程的合作者；更小的节点度数降低了双重过程合作者的成功率。 |
| [^11] | [Coarse Wage-Setting and Behavioral Firms.](http://arxiv.org/abs/2206.01114) | 本文研究表明公司粗略的工资设定导致工资在整数上的聚集，同时对公司的市场结果产生了负面影响。 |

# 详细

[^1]: 动态AMM中的最优再平衡

    Optimal Rebalancing in Dynamic AMMs

    [https://arxiv.org/abs/2403.18737](https://arxiv.org/abs/2403.18737)

    该论文提出了在动态AMM中实现最优再平衡的方法，通过引入套利机会来调整持有物，采用便宜计算的近似最优方法取得了性能改进。

    

    动态AMM池，如时间函数做市所发现的那样，会将其持有物重新平衡到新的期望比例（例如，从两种资产之间的50-50移动到其中一种占优势的90-10），通过引入一个套利机会来实现，当其持有物与目标一致时，该套利机会会消失。构造这种套利机会可以简化为通过其交易函数向市场公开的投资组合权重序列的选择问题。从起始权重线性插值到结束权重已被用于降低池向套利者支付的成本以重新平衡。在这里，我们在权重变化很小的极限情况下获得了$\textit{最优}$插值（缺点是需要调用一个超越函数），然后获得了一个便宜计算的近似最优方法，几乎实现了相同的性能改进。然后我们展示了这种方法在一系列市场中的应用。

    arXiv:2403.18737v1 Announce Type: new  Abstract: Dynamic AMM pools, as found in Temporal Function Market Making, rebalance their holdings to a new desired ratio (e.g. moving from being 50-50 between two assets to being 90-10 in favour of one of them) by introducing an arbitrage opportunity that disappears when their holdings are in line with their target. Structuring this arbitrage opportunity reduces to the problem of choosing the sequence of portfolio weights the pool exposes to the market via its trading function. Linear interpolation from start weights to end weights has been used to reduce the cost paid by pools to arbitrageurs to rebalance. Here we obtain the $\textit{optimal}$ interpolation in the limit of small weight changes (which has the downside of requiring a call to a transcendental function) and then obtain a cheap-to-compute approximation to that optimal approach that gives almost the same performance improvement. We then demonstrate this method on a range of market bac
    
[^2]: 具有乘法噪声的随机线性二次系统中的有限注意分配

    Limited Attention Allocation in a Stochastic Linear Quadratic System with Multiplicative Noise

    [https://arxiv.org/abs/2403.18528](https://arxiv.org/abs/2403.18528)

    本研究在具有乘法噪声的随机线性二次系统中解决了有限注意分配问题，通过战略资源分配提高噪声估计并改善控制决策。

    

    本研究探讨了具有乘法噪声的随机线性二次系统中的有限注意分配。我们的方法能够实现战略资源分配，以增强噪声估计并改善控制决策。我们提供了解析最优控制，并提出了一种用于最优注意分配的数值方法。此外，我们将我们的研究成果应用于动态均值方差组合选择，展示了跨时间段和因素的有效资源分配，为投资者提供了有价值的见解。

    arXiv:2403.18528v1 Announce Type: cross  Abstract: This study addresses limited attention allocation in a stochastic linear quadratic system with multiplicative noise. Our approach enables strategic resource allocation to enhance noise estimation and improve control decisions. We provide analytical optimal control and propose a numerical method for optimal attention allocation. Additionally, we apply our ffndings to dynamic mean-variance portfolio selection, showing effective resource allocation across time periods and factors, providing valuable insights for investors.
    
[^3]: 重定向流动--探索亚马逊未来

    Redirecting Flows -- Navigating the Future of the Amazon

    [https://arxiv.org/abs/2403.18521](https://arxiv.org/abs/2403.18521)

    亚马逊流域及拉丁美洲和加勒比地区正经历环境挑战与变革潜力并存的时刻，本报告旨在揭示各种社会生态问题、技术进步、社区倡议和战略行动，有助于推动区域的生物圈可持续性和韧性。

    

    亚马逊流域及拉丁美洲和加勒比地区正处于一个关键时刻，正在应对紧迫的环境挑战，同时通过创新解决方案有巨大潜力实现变革。本报告阐明了社会生态问题、技术进步、社区主导的倡议和战略行动等多样化领域，这些可以帮助促进整个地区的生物圈可持续性和韧性。

    arXiv:2403.18521v1 Announce Type: new  Abstract: The Amazon Basin, and the Latin America and Caribbean (LAC) region more broadly, stands at a critical juncture, grappling with pressing environmental challenges while holding immense potential for transformative change through innovative solutions. This report illuminates the diverse landscape of social-ecological issues, technological advancements, community-led initiatives, and strategic actions that could help foster biosphere-based sustainability and resilience across the region.
    
[^4]: G3M中做市商财富增长率

    Growth rate of liquidity provider's wealth in G3Ms

    [https://arxiv.org/abs/2403.18177](https://arxiv.org/abs/2403.18177)

    该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。

    

    几何均值市场做市商（G3M），如Uniswap和Balancer，代表一类广泛使用的自动做市商（AMM）。这些G3M的特点在于：每笔交易前后，AMM的储备必须保持相同（加权）的几何均值。本文研究了交易费对G3M中流动性提供者（LP）盈利能力的影响，以及LP面临的由涉及参考市场的套利活动导致的逆向选择。我们的工作扩展了先前研究中描述的G3M模型，将交易费和连续时间套利整合到分析中。在这个背景下，我们分析了具有随机存储过程特征的G3M动态，并计算了LP财富的增长率。特别地，我们的结果与扩展了关于常数乘积市场做市商的结果相一致，通常称为Uniswap v2。

    arXiv:2403.18177v1 Announce Type: new  Abstract: Geometric mean market makers (G3Ms), such as Uniswap and Balancer, represent a widely used class of automated market makers (AMMs). These G3Ms are characterized by the following rule: the reserves of the AMM must maintain the same (weighted) geometric mean before and after each trade. This paper investigates the effects of trading fees on liquidity providers' (LP) profitability in a G3M, as well as the adverse selection faced by LPs due to arbitrage activities involving a reference market. Our work expands the model described in previous studies for G3Ms, integrating transaction fees and continuous-time arbitrage into the analysis. Within this context, we analyze G3M dynamics, characterized by stochastic storage processes, and calculate the growth rate of LP wealth. In particular, our results align with and extend the results concerning the constant product market maker, commonly referred to as Uniswap v2.
    
[^5]: 重新审视正向利率的弹性字符串模型

    Revisiting Elastic String Models of Forward Interest Rates

    [https://arxiv.org/abs/2403.18126](https://arxiv.org/abs/2403.18126)

    该论文重新审视了正向利率的弹性字符串模型，通过解释市场力量对邻近期限利率的自我参照影响，实现了对整个FRC相关结构的准确再现，仅需两个稳定参数。

    

    25年前，几位作者提出将正向利率曲线（FRC）建模为一根弹性字符串，沿其传播特异冲击，考虑到不同到期日间回报相关性的独特结构。本文重新审视巴奎和布夏尔（2004）的“刚性”弹性字符串场论，使其微观基础更加透明。我们的模型可以解释市场力量对邻近期限利率的自我参照方式的影响。该模型简洁且能准确地在1994-2023年间再现整个FRC的相关结构，误差低于2%。我们仅需两个参数，其值除了2009-2014年量化宽松期间可能有所变化外非常稳定。时间分辨率（也称为Epps效应）对相关性的依赖也被忠实地再现。

    arXiv:2403.18126v1 Announce Type: new  Abstract: Twenty five years ago, several authors proposed to model the forward interest rate curve (FRC) as an elastic string along which idiosyncratic shocks propagate, accounting for the peculiar structure of the return correlation across different maturities. In this paper, we revisit the specific "stiff'' elastic string field theory of Baaquie and Bouchaud (2004) in a way that makes its micro-foundation more transparent. Our model can be interpreted as capturing the effect of market forces that set the rates of nearby tenors in a self-referential fashion. The model is parsimonious and accurately reproduces the whole correlation structure of the FRC over the time period 1994-2023, with an error below 2%. We need only two parameters, the values of which being very stable except perhaps during the Quantitative Easing period 2009-2014. The dependence of correlation on time resolution (also called the Epps effect) is also faithfully reproduced with
    
[^6]: 深度限价订单簿预测

    Deep Limit Order Book Forecasting

    [https://arxiv.org/abs/2403.09267](https://arxiv.org/abs/2403.09267)

    该研究利用深度学习方法预测纳斯达克交易所股票的限价订单簿中间价格变动，提出了一个创新的操作框架来评估预测的实用性。

    

    我们利用尖端的深度学习方法探索了在纳斯达克交易所上交易的一组异质股票的高频限价订单簿中间价格变动的可预测性。在此过程中，我们发布了“LOBFrame”，一个开源代码库，可以高效处理大规模限价订单簿数据，并定量评估最先进的深度学习模型的预测能力。我们的结果是双重的。我们证明股票的微观结构特征影响深度学习方法的有效性，并且它们的高预测能力不一定对应可操作的交易信号。我们认为传统的机器学习指标未能充分评估限价订单簿环境中预测的质量。作为替代，我们提出了一个创新的操作框架，通过专注于准确预测的概率来评估预测的实用性。

    arXiv:2403.09267v1 Announce Type: cross  Abstract: We exploit cutting-edge deep learning methodologies to explore the predictability of high-frequency Limit Order Book mid-price changes for a heterogeneous set of stocks traded on the NASDAQ exchange. In so doing, we release `LOBFrame', an open-source code base, to efficiently process large-scale Limit Order Book data and quantitatively assess state-of-the-art deep learning models' forecasting capabilities. Our results are twofold. We demonstrate that the stocks' microstructural characteristics influence the efficacy of deep learning methods and that their high forecasting power does not necessarily correspond to actionable trading signals. We argue that traditional machine learning metrics fail to adequately assess the quality of forecasts in the Limit Order Book context. As an alternative, we propose an innovative operational framework that assesses predictions' practicality by focusing on the probability of accurately forecasting com
    
[^7]: 通用N-token AMM套利的闭合形式解决方案

    Closed-form solutions for generic N-token AMM arbitrage

    [https://arxiv.org/abs/2402.06731](https://arxiv.org/abs/2402.06731)

    本文提出了通用N-token AMM套利的闭合形式解决方案。该解决方案在模拟中表现出比凸优化更好的套利机会，并能更快地利用这些机会。并且，所提出的方法具有并行计算能力，能够在GPU上扩展，为AMM建模提供了一种新方法。此外，该机制的低计算成本还可以实现多资产池的链上套利机器人。

    

    凸优化自从自动市场制造商（AMM）诞生以来已经提供了一种确定套利交易的机制。在这里，我们概述了通用闭合形式解决方案，用于$N$个代币的几何平均市场制造商池套利。在模拟中（使用合成和历史数据），此解决方案提供了比凸优化更好的套利机会，并且能够更早地利用这些机会。此外，所提出的方法天然地支持并行计算（与凸优化不同），能够在GPU上进行扩展，为AMM建模提供了一种新方法，提供了一种基于替代数值求解方法的选择。这种新机制的低计算成本还可以实现多资产池的链上套利机器人。

    Convex optimisation has provided a mechanism to determine arbitrage trades on automated market markets (AMMs) since almost their inception. Here we outline generic closed-form solutions for $N$-token geometric mean market maker pool arbitrage, that in simulation (with synthetic and historic data) provide better arbitrage opportunities than convex optimisers and is able to capitalise on those opportunities sooner. Furthermore, the intrinsic parallelism of the proposed approach (unlike convex optimisation) offers the ability to scale on GPUs, opening up a new approach to AMM modelling by offering an alternative to numerical-solver-based methods. The lower computational cost of running this new mechanism can also enable on-chain arbitrage bots for multi-asset pools.
    
[^8]: 基于自然语言处理的消费者投诉叙述中系统异常的检测方法

    NLP-based detection of systematic anomalies among the narratives of consumer complaints. (arXiv:2308.11138v1 [stat.ME])

    [http://arxiv.org/abs/2308.11138](http://arxiv.org/abs/2308.11138)

    本文开发了一种基于自然语言处理的方法，用于检测消费者投诉叙述中的系统异常。这种方法可以解决分类算法对于较小且频繁出现的系统异常检测的问题，并将投诉叙述转化为定量数据进行分析。

    

    我们开发了一种基于自然语言处理的方法，用于检测投诉叙述中的系统异常，简称为系统异常。尽管分类算法被用于检测明显的异常，但在较小且频繁出现的系统异常情况下，算法可能会因为各种原因而失效，包括技术原因和人工分析师的自然限制。因此，在分类之后的下一步中，我们将投诉叙述转化为定量数据，然后使用一种算法来检测系统异常。我们使用消费者金融保护局的消费者投诉数据库中的投诉叙述来说明整个过程。

    We develop an NLP-based procedure for detecting systematic nonmeritorious consumer complaints, simply called systematic anomalies, among complaint narratives. While classification algorithms are used to detect pronounced anomalies, in the case of smaller and frequent systematic anomalies, the algorithms may falter due to a variety of reasons, including technical ones as well as natural limitations of human analysts. Therefore, as the next step after classification, we convert the complaint narratives into quantitative data, which are then analyzed using an algorithm for detecting systematic anomalies. We illustrate the entire procedure using complaint narratives from the Consumer Complaint Database of the Consumer Financial Protection Bureau.
    
[^9]: 多维COS方法用于期权定价

    The multidimensional COS method for option pricing. (arXiv:2307.12843v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.12843](http://arxiv.org/abs/2307.12843)

    多维COS方法是一种用于定价依赖于多个标的的金融期权的数值工具。本文证明了该方法的收敛性，并分析了数值不确定度对该方法的影响。

    

    多维COS方法是一种用于定价依赖于多个标的的金融期权的数值工具。该方法利用标的的对数收益率的特征函数φ，并且当支付函数的傅里叶余弦系数v_k以闭式形式给定时，具有优势。然而，在重要情况下，φ和v_k都没有解析地给出，而需要通过数值方法恢复。在本文中，我们证明了多维COS方法的收敛性，包括对φ和v_k的数值不确定度的考虑。我们的分析有助于理解COS方法中对φ和v_k的近似误差如何传播。

    The multidimensional COS method is a numerical tool to price financial options, which depend on several underlyings. The method makes use of the characteristic function $\varphi$ of the logarithmic returns of the underlyings and it is advantageous if the Fourier-cosine coefficients $v_{\boldsymbol{k}}$ of the payoff function are given in closed-form. However, in important cases, neither $\varphi$ nor $v_{\boldsymbol{k}}$ are given analytically but need to be recovered numerically. In this article, we prove the convergence of the multidimensional COS method including numerical uncertainty on $\varphi$ and $v_{\boldsymbol{k}}$. Our analysis helps to understand how the approximation errors on $\varphi$ and $v_{\boldsymbol{k}}$ propagate in the COS method.
    
[^10]: 结构化人群中认知和合作的共同演化在增强学习中的研究

    Coevolution of cognition and cooperation in structured populations under reinforcement learning. (arXiv:2306.11376v1 [physics.soc-ph])

    [http://arxiv.org/abs/2306.11376](http://arxiv.org/abs/2306.11376)

    本研究发现在囚徒困境中，确定的重复交互概率阈值将导致智能体从直觉的背叛者转变为双重过程的合作者；更小的节点度数降低了双重过程合作者的成功率。

    

    本文研究了受强化学习影响下囚徒困境问题中智能体在规则化网络中相互交互，并能了解到它们是否支付考虑交互的费用。相对于文献中使用的其他行为规则，（i）我们确认了重复交互概率的阈值，从直觉的背叛者转变为双重过程的合作者；（ii）我们发现节点度数扮演了不同的角色，更小的度数降低了双重过程合作者的进化成功率；（iii）我们观察到了更高频繁的沉思。

    We study the evolution of behavior under reinforcement learning in a Prisoner's Dilemma where agents interact in a regular network and can learn about whether they play one-shot or repeatedly by incurring a cost of deliberation. With respect to other behavioral rules used in the literature, (i) we confirm the existence of a threshold value of the probability of repeated interaction, switching the emergent behavior from intuitive defector to dual-process cooperator; (ii) we find a different role of the node degree, with smaller degrees reducing the evolutionary success of dual-process cooperators; (iii) we observe a higher frequency of deliberation.
    
[^11]: 粗略工资设定与行为公司

    Coarse Wage-Setting and Behavioral Firms. (arXiv:2206.01114v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2206.01114](http://arxiv.org/abs/2206.01114)

    本文研究表明公司粗略的工资设定导致工资在整数上的聚集，同时对公司的市场结果产生了负面影响。

    

    本文显示出工资在整数上的聚集部分是由公司粗略的工资设定所驱动的。通过对巴西超过2亿新员工的数据进行分析，首先得出了合同工资倾向于集中在整数上的结论。然后，研究表明倾向于以整数工资雇佣工人的公司不够复杂且市场成果较差。接下来，本文提出了一种工资发布模型，在该模型中，由于优化成本的考虑，公司采用了粗略的整数工资，并通过两种研究设计提供了与模型预测相关的证据。最后，本文研究了粗略工资设定对相关经济结果的一些影响。

    This paper shows that the bunching of wages at round numbers is partly driven by firm coarse wage-setting. Using data from over 200 million new hires in Brazil, I first establish that contracted salaries tend to cluster at round numbers. Then, I show that firms that tend to hire workers at round-numbered salaries are less sophisticated and have worse market outcomes. Next, I develop a wage-posting model in which optimization costs lead to the adoption of coarse rounded wages and provide evidence supporting two model predictions using two research designs. Finally, I examine some consequences of coarse wage-setting for relevant economic outcomes.
    

