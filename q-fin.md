# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spanning Multi-Asset Payoffs With ReLUs](https://arxiv.org/abs/2403.14231) | 提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。 |
| [^2] | [DiffSTOCK: Probabilistic relational Stock Market Predictions using Diffusion Models](https://arxiv.org/abs/2403.14063) | 提出了一种使用扩散模型进行概率关系型股市预测的方法，能更好地处理金融数据中的不确定性。 |
| [^3] | [Synthetic Data Applications in Finance](https://arxiv.org/abs/2401.00081) | 合成数据在金融领域的广泛应用，涉及各种不同数据类型，有助于解决隐私、公平性和可解释性相关问题，对其质量和效果进行了评估，并探讨了未来发展方向。 |
| [^4] | [Callable convertible bonds under liquidity constraints and hybrid priorities](https://arxiv.org/abs/2111.02554) | 本文研究了在流动性约束下的可调转换债券问题，提出了一个完整解决方案，并介绍了一个非有序情况下处理的方法。 |
| [^5] | [Designing Digital Voting Systems for Citizens: Achieving Fairness and Legitimacy in Digital Participatory Budgeting.](http://arxiv.org/abs/2310.03501) | 本研究调查了数字参与式预算中投票和聚合方法的权衡，并通过行为实验确定了有利的投票设计组合。研究发现，设计选择对集体决策、市民感知和结果公平性有深远影响，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程提供了可行的见解。 |
| [^6] | [Wildfire Modeling: Designing a Market to Restore Assets.](http://arxiv.org/abs/2205.13773) | 该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。 |

# 详细

[^1]: 用ReLU跨度多资产回报

    Spanning Multi-Asset Payoffs With ReLUs

    [https://arxiv.org/abs/2403.14231](https://arxiv.org/abs/2403.14231)

    提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。

    

    我们提出了利用香草篮子期权的分布式形式来解决多资产回报的跨度问题。我们发现，只有当回报函数为偶函数且绝对齐次函数时，此问题才有唯一解，并且我们建立了一个基于傅立叶的公式来计算解决方案。金融回报通常是分段线性的，导致可能可以明确推导出解决方案，但在数值上可能难以利用。相比于基于单资产香草避险的行业偏爱方法，单隐藏层前馈神经网络为离散跨度提供了一种自然而高效的数值替代方法。我们测试了这种方法用于一些典型回报，并发现与单资产香草避险的产业偏好方法相比，利用香草篮子期权获得了更好的避险结果。

    arXiv:2403.14231v1 Announce Type: new  Abstract: We propose a distributional formulation of the spanning problem of a multi-asset payoff by vanilla basket options. This problem is shown to have a unique solution if and only if the payoff function is even and absolutely homogeneous, and we establish a Fourier-based formula to calculate the solution. Financial payoffs are typically piecewise linear, resulting in a solution that may be derived explicitly, yet may also be hard to numerically exploit. One-hidden-layer feedforward neural networks instead provide a natural and efficient numerical alternative for discrete spanning. We test this approach for a selection of archetypal payoffs and obtain better hedging results with vanilla basket options compared to industry-favored approaches based on single-asset vanilla hedges.
    
[^2]: DiffSTOCK：使用扩散模型进行概率关系型股市预测

    DiffSTOCK: Probabilistic relational Stock Market Predictions using Diffusion Models

    [https://arxiv.org/abs/2403.14063](https://arxiv.org/abs/2403.14063)

    提出了一种使用扩散模型进行概率关系型股市预测的方法，能更好地处理金融数据中的不确定性。

    

    在这项工作中，我们提出了一种推广去噪扩散概率模型用于股市预测和投资组合管理的方法。现有的工作已经展示了建模股票间关系用于市场时间序列预测的功效，并利用基于图的学习模型进行价值预测和投资组合管理。尽管令人信服，这些确定性方法仍然无法处理不确定性，即由于金融数据的信噪比较低，学习有效的确定性模型相当具有挑战性。由于概率方法已被证明能有效模拟时间序列预测的更高不确定性。为此，我们展示了去噪扩散概率模型(DDPM)的有效利用，开发了一个基于历史财务指标和股票间关系提供更好市场预测的架构。

    arXiv:2403.14063v1 Announce Type: new  Abstract: In this work, we propose an approach to generalize denoising diffusion probabilistic models for stock market predictions and portfolio management. Present works have demonstrated the efficacy of modeling interstock relations for market time-series forecasting and utilized Graph-based learning models for value prediction and portfolio management. Though convincing, these deterministic approaches still fall short of handling uncertainties i.e., due to the low signal-to-noise ratio of the financial data, it is quite challenging to learn effective deterministic models. Since the probabilistic methods have shown to effectively emulate higher uncertainties for time-series predictions. To this end, we showcase effective utilisation of Denoising Diffusion Probabilistic Models (DDPM), to develop an architecture for providing better market predictions conditioned on the historical financial indicators and inter-stock relations. Additionally, we al
    
[^3]: 金融领域中的合成数据应用

    Synthetic Data Applications in Finance

    [https://arxiv.org/abs/2401.00081](https://arxiv.org/abs/2401.00081)

    合成数据在金融领域的广泛应用，涉及各种不同数据类型，有助于解决隐私、公平性和可解释性相关问题，对其质量和效果进行了评估，并探讨了未来发展方向。

    

    合成数据在包括金融、医疗保健和虚拟现实在内的各种商业领域取得了巨大进展。本文概述了合成数据在金融领域的原型应用，并为其中的一些特定应用提供了更丰富的细节。这些应用涵盖了来自市场和零售金融应用的表格、时间序列、事件序列和非结构化数据模态的广泛数据类型。由于金融是一个受高度监管的行业，合成数据是处理与隐私、公平性和可解释性相关问题的一种潜在方法。在这些应用中，使用各种指标评估了我们方法的质量和有效性。最后，我们探讨了在金融领域合成数据的未来方向。

    arXiv:2401.00081v2 Announce Type: replace  Abstract: Synthetic data has made tremendous strides in various commercial settings including finance, healthcare, and virtual reality. We present a broad overview of prototypical applications of synthetic data in the financial sector and in particular provide richer details for a few select ones. These cover a wide variety of data modalities including tabular, time-series, event-series, and unstructured arising from both markets and retail financial applications. Since finance is a highly regulated industry, synthetic data is a potential approach for dealing with issues related to privacy, fairness, and explainability. Various metrics are utilized in evaluating the quality and effectiveness of our approaches in these applications. We conclude with open directions in synthetic data in the context of the financial domain.
    
[^4]: 可调转换债券在流动性约束和混合优先级下的研究

    Callable convertible bonds under liquidity constraints and hybrid priorities

    [https://arxiv.org/abs/2111.02554](https://arxiv.org/abs/2111.02554)

    本文研究了在流动性约束下的可调转换债券问题，提出了一个完整解决方案，并介绍了一个非有序情况下处理的方法。

    

    本文研究了在由泊松信号建模的流动性约束下的可调转换债券问题。我们假设当债券持有人和公司同时停止游戏时，他们都没有绝对优先级，而是一部分$m\in[0,1]$的债券转换为公司的股票，其余被公司调用。因此，本文推广了[Liang和Sun，带泊松随机干预时间的Dynkin博弈，SIAM控制和优化杂志，57(2019)，2962-2991]中研究的特殊情况（债券持有人有优先权，$m=1$），并提出了具有流动性约束的可调转换债券问题的完整解决方案。可调转换债券是Dynkin博弈的一个例子，但不属于标准范式，因为收益不以有序方式取决于哪个代理停止游戏。我们展示了如何通过引入...

    arXiv:2111.02554v2 Announce Type: replace  Abstract: This paper investigates the callable convertible bond problem in the presence of a liquidity constraint modelled by Poisson signals. We assume that neither the bondholder nor the firm has absolute priority when they stop the game simultaneously, but instead, a proportion $m\in[0,1]$ of the bond is converted to the firm's stock and the rest is called by the firm. The paper thus generalizes the special case studied in [Liang and Sun, Dynkin games with Poisson random intervention times, SIAM Journal on Control and Optimization, 57 (2019), 2962-2991] where the bondholder has priority ($m=1$), and presents a complete solution to the callable convertible bond problem with liquidity constraint. The callable convertible bond is an example of a Dynkin game, but falls outside the standard paradigm since the payoffs do not depend in an ordered way upon which agent stops the game. We show how to deal with this non-ordered situation by introducin
    
[^5]: 为市民设计数字投票系统：在数字参与式预算中实现公平和合法性

    Designing Digital Voting Systems for Citizens: Achieving Fairness and Legitimacy in Digital Participatory Budgeting. (arXiv:2310.03501v1 [cs.HC])

    [http://arxiv.org/abs/2310.03501](http://arxiv.org/abs/2310.03501)

    本研究调查了数字参与式预算中投票和聚合方法的权衡，并通过行为实验确定了有利的投票设计组合。研究发现，设计选择对集体决策、市民感知和结果公平性有深远影响，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程提供了可行的见解。

    

    数字参与式预算（PB）已成为城市资源分配的重要民主工具。在数字平台的支持下，新的投票输入格式和聚合方法已被利用。然而，实现公平和合法性仍然面临挑战。本研究调查了数字PB中各种投票和聚合方法之间的权衡。通过行为实验，我们确定了在认知负荷、比例和感知合法性方面的有利投票设计组合。研究揭示了设计选择如何深刻影响集体决策、市民感知和结果公平性。我们的发现为人机交互、机制设计和计算社会选择提供了可行的见解，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程做出贡献。

    Digital Participatory Budgeting (PB) has become a key democratic tool for resource allocation in cities. Enabled by digital platforms, new voting input formats and aggregation have been utilised. Yet, challenges in achieving fairness and legitimacy persist. This study investigates the trade-offs in various voting and aggregation methods within digital PB. Through behavioural experiments, we identified favourable voting design combinations in terms of cognitive load, proportionality, and perceived legitimacy. The research reveals how design choices profoundly influence collective decision-making, citizen perceptions, and outcome fairness. Our findings offer actionable insights for human-computer interaction, mechanism design, and computational social choice, contributing to the development of fairer and more transparent digital PB systems and multi-winner collective decision-making process for citizens.
    
[^6]: 森林火灾模型：设计市场以恢复资产

    Wildfire Modeling: Designing a Market to Restore Assets. (arXiv:2205.13773v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.13773](http://arxiv.org/abs/2205.13773)

    该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。

    

    在过去的十年里，夏季森林火灾已经成为加利福尼亚和美国的常态。这些火灾的原因多种多样。州政府会收取森林火灾基金来帮助受灾人员。然而，基金只在特定条件下发放，并且在整个加利福尼亚州均匀收取。因此，该项目的整体思路是寻找关于电力公司如何引发火灾以及如何帮助收取森林火灾基金或者公平收费以最大限度地实现社会影响的数量结果。该研究项目旨在提出与植被、输电线路相关的森林火灾风险，并将其与金钱挂钩。因此，该项目有助于解决与每个地点相关的森林火灾基金收取问题，并结合能源价格根据地点的森林火灾风险向客户收费，以实现社会的总盈余最大化。

    In the past decade, summer wildfires have become the norm in California, and the United States of America. These wildfires are caused due to variety of reasons. The state collects wildfire funds to help the impacted customers. However, the funds are eligible only under certain conditions and are collected uniformly throughout California. Therefore, the overall idea of this project is to look for quantitative results on how electrical corporations cause wildfires and how they can help to collect the wildfire funds or charge fairly to the customers to maximize the social impact. The research project aims to propose the implication of wildfire risk associated with vegetation, and due to power lines and incorporate that in dollars. Therefore, the project helps to solve the problem of collecting wildfire funds associated with each location and incorporate energy prices to charge their customers according to their wildfire risk related to the location to maximize the social surplus for the s
    

