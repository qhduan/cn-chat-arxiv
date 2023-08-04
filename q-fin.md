# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A novel approach for quantum financial simulation and quantum state preparation.](http://arxiv.org/abs/2308.01844) | 本研究引入了一种新的量子模拟算法，即多重分割步骤量子行走算法，它结合了参数化量子电路和经典模拟器上的变分求解器，可用于学习和加载复杂概率分布。该算法通过引入多智能体决策过程，使其适用于金融市场建模，并展示出良好的潜力。 |
| [^2] | [Trading and wealth evolution in the Proof of Stake protocol.](http://arxiv.org/abs/2308.01803) | 本文研究了Proof of Stake（PoS）协议中的交易和财富演化。通过分析 PoS 协议中没有交易的情况下的财富演化、考虑矿工的交易激励和策略，以及研究 PoS 交易环境中矿工的集体行为，总结了该协议的经济特征。 |
| [^3] | [Quantifying Retrospective Human Responsibility in Intelligent Systems.](http://arxiv.org/abs/2308.01752) | 本论文提出了三种衡量使用智能系统时追溯人类责任的方法，包括对人类与系统重复交互的量化，对决策信息形成中人类独特贡献的量化，以及对人类行动合理性的量化。 |
| [^4] | [Path Shadowing Monte-Carlo.](http://arxiv.org/abs/2308.01486) | 该论文提出了一种路径阴影蒙特卡洛方法，可以根据任何生成模型对未来路径进行预测，并通过对生成的价格路径进行平均来实现。该方法可以产生最先进的未来实现波动率预测，并且在确定S\&P500的条件期权笑曲线方面优于当前版本的路径依赖波动率模型和期权市场本身。 |
| [^5] | [A new probabilistic analysis of the yard-sale model.](http://arxiv.org/abs/2308.01485) | 这个论文通过新的概率分析证明了Chakraborti的庙会模型中几乎肯定会出现财富凝聚的现象，并且这一结果还扩展到了具有财富获取优势的修改后模型中。 |
| [^6] | [Graph Neural Networks for Forecasting Multivariate Realized Volatility with Spillover Effects.](http://arxiv.org/abs/2308.01419) | 本研究提出了一种使用图神经网络建模和预测多变量实现波动的新方法，能够有效地捕捉非线性关系，并利用多跳邻居的溢出效应提高实现波动的预测准确性。此外，使用拟似似然损失进行训练可以显著提升模型性能。 |
| [^7] | [A lower bound for the volatility swap in the lognormal SABR model.](http://arxiv.org/abs/2306.14602) | 本研究对于条件性对数正态的SABR模型，证明了零散度隐含波动率是波动率互换行权价的下界，该结论适用于所有相关参数，并且在相关性小于或等于零时比平值隐含波动率更为精确。 |
| [^8] | [Abnormal Trading Detection in the NFT Market.](http://arxiv.org/abs/2306.04643) | 本文提出了一种通过聚类算法检测非同质化代币（NFT）交易市场中的异常行为的方法，并探讨了监管对减少欺诈行为的影响。 |
| [^9] | [Price Discovery for Derivatives.](http://arxiv.org/abs/2302.13426) | 本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。 |
| [^10] | [Can we infer microscopic financial information from the long memory in market-order flow?: a quantitative test of the Lillo-Mike-Farmer model.](http://arxiv.org/abs/2301.13505) | 该研究通过对市场订单流的长时记忆进行定量测试，验证了Lillo-Mike-Farmer模型中的微观假设关于订单拆分行为的预测与宏观订单方向相关性的定量关系。 |
| [^11] | [Generative CVaR Portfolio Optimization with Attention-Powered Dynamic Factor Learning.](http://arxiv.org/abs/2301.07318) | 本论文提出了一种使用注意力驱动的动态因子学习的生成CVaR组合优化算法，通过随机变量转换作为分布建模的隐式方式，以及使用注意力-GRU网络进行动态学习和预测，捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。通过在每个投资日期从学习到的生成模型中模拟新样本，并进一步应用CVaR组合优化策略，实现了更明智的投资决策。 |
| [^12] | [Robust utility maximization with nonlinear continuous semimartingales.](http://arxiv.org/abs/2206.14015) | 本文研究模型不确定性下的连续时间健壮效用最大化问题，通过对偶性证明了对数、指数和幂效用的最优组合的存在性。 |
| [^13] | [The Log Private Company Valuation Model.](http://arxiv.org/abs/2206.09666) | 本文提出了一种基于动态戈登增长模型的对数私人公司估值模型，其中包含闭合式期权定价公式、对冲公式以及权益连接寿险产品的净保费公式。该模型可用于私人公司和公共公司。 |

# 详细

[^1]: 量子金融模拟和量子态制备的一种新方法

    A novel approach for quantum financial simulation and quantum state preparation. (arXiv:2308.01844v1 [quant-ph])

    [http://arxiv.org/abs/2308.01844](http://arxiv.org/abs/2308.01844)

    本研究引入了一种新的量子模拟算法，即多重分割步骤量子行走算法，它结合了参数化量子电路和经典模拟器上的变分求解器，可用于学习和加载复杂概率分布。该算法通过引入多智能体决策过程，使其适用于金融市场建模，并展示出良好的潜力。

    

    量子态制备在量子计算和信息处理中至关重要。准确可靠地制备特定的量子态对于各种应用至关重要。量子计算的一个有希望的应用是量子模拟。这需要制备表示我们要模拟系统的量子态。本研究引入了一种新的模拟算法，即多重分割步骤量子行走（multi-SSQW），采用借助参数化量子电路（PQC）和经典模拟器上的变分求解器来学习和加载复杂概率分布。多重分割步骤量子行走算法是分割步骤量子行走的改进版本，增加了多智能体决策过程，使其适用于金融市场建模。本研究提供了多重分割步骤量子行走算法的理论描述和实证调查，以证明其在概率分布模拟中的潜力。

    Quantum state preparation is vital in quantum computing and information processing. The ability to accurately and reliably prepare specific quantum states is essential for various applications. One of the promising applications of quantum computers is quantum simulation. This requires preparing a quantum state representing the system we are trying to simulate. This research introduces a novel simulation algorithm, the multi-Split-Steps Quantum Walk (multi-SSQW), designed to learn and load complicated probability distributions using parameterized quantum circuits (PQC) with a variational solver on classical simulators. The multi-SSQW algorithm is a modified version of the split-steps quantum walk, enhanced to incorporate a multi-agent decision-making process, rendering it suitable for modeling financial markets. The study provides theoretical descriptions and empirical investigations of the multi-SSQW algorithm to demonstrate its promising capabilities in probability distribution simula
    
[^2]: Proof of Stake协议中的交易和财富演化

    Trading and wealth evolution in the Proof of Stake protocol. (arXiv:2308.01803v1 [econ.GN])

    [http://arxiv.org/abs/2308.01803](http://arxiv.org/abs/2308.01803)

    本文研究了Proof of Stake（PoS）协议中的交易和财富演化。通过分析 PoS 协议中没有交易的情况下的财富演化、考虑矿工的交易激励和策略，以及研究 PoS 交易环境中矿工的集体行为，总结了该协议的经济特征。

    

    随着Proof of Stake（PoS）区块链的越来越广泛采用，研究这种区块链创造的经济是及时的。在本章中，我们将调查依据PoS协议发行新币的加密货币中的交易和财富演化的最新进展。我们首先考虑在没有交易的情况下PoS协议的财富演化，并关注去中心化的问题。接下来，我们通过最优控制的视角考虑每个矿工的交易激励和策略，矿工需要权衡PoS挖矿和交易。最后，我们通过均场模型研究在PoS交易环境中矿工的集体行为。在我们的研究中使用了随机和分析工具。同时还提出了一系列未解决的问题。

    With the increasing adoption of the Proof of Stake (PoS) blockchain, it is timely to study the economy created by such blockchain. In this chapter, we will survey recent progress on the trading and wealth evolution in a cryptocurrency where the new coins are issued according to the PoS protocol. We first consider the wealth evolution in the PoS protocol assuming no trading, and focus on the problem of decentralisation. Next we consider each miner's trading incentive and strategy through the lens of optimal control, where the miner needs to trade off PoS mining and trading. Finally, we study the collective behavior of the miners in a PoS trading environment by a mean field model. We use both stochastic and analytic tools in our study. A list of open problems are also presented.
    
[^3]: 对智能系统中的人类追溯责任进行量化

    Quantifying Retrospective Human Responsibility in Intelligent Systems. (arXiv:2308.01752v1 [cs.HC])

    [http://arxiv.org/abs/2308.01752](http://arxiv.org/abs/2308.01752)

    本论文提出了三种衡量使用智能系统时追溯人类责任的方法，包括对人类与系统重复交互的量化，对决策信息形成中人类独特贡献的量化，以及对人类行动合理性的量化。

    

    智能系统已成为我们生活中的重要组成部分。在与这些系统的互动中，人类对结果的责任变得模糊，因为信息获取、决策和行动实施的部分可能由人类和系统共同完成。在导致不利结果的事件中，确定人类因果责任尤为重要。我们提出了三种衡量使用智能系统时的追溯人类因果责任的方法。 第一种方法涉及人类与系统的重复交互。利用信息论，它量化了人类对过去事件结果的平均独特贡献。第二和第三种方法分别涉及人类在与智能系统的单次过去交互中的因果责任。它们分别量化了人类在形成决策所使用的信息方面的独特贡献以及人类实施行动的合理性。

    Intelligent systems have become a major part of our lives. Human responsibility for outcomes becomes unclear in the interaction with these systems, as parts of information acquisition, decision-making, and action implementation may be carried out jointly by humans and systems. Determining human causal responsibility with intelligent systems is particularly important in events that end with adverse outcomes. We developed three measures of retrospective human causal responsibility when using intelligent systems. The first measure concerns repetitive human interactions with a system. Using information theory, it quantifies the average human's unique contribution to the outcomes of past events. The second and third measures concern human causal responsibility in a single past interaction with an intelligent system. They quantify, respectively, the unique human contribution in forming the information used for decision-making and the reasonability of the actions that the human carried out. T
    
[^4]: 路径阴影蒙特卡洛方法

    Path Shadowing Monte-Carlo. (arXiv:2308.01486v1 [q-fin.MF])

    [http://arxiv.org/abs/2308.01486](http://arxiv.org/abs/2308.01486)

    该论文提出了一种路径阴影蒙特卡洛方法，可以根据任何生成模型对未来路径进行预测，并通过对生成的价格路径进行平均来实现。该方法可以产生最先进的未来实现波动率预测，并且在确定S\&P500的条件期权笑曲线方面优于当前版本的路径依赖波动率模型和期权市场本身。

    

    我们介绍了一种路径阴影蒙特卡洛方法，可以根据任何生成模型对未来路径进行预测。在任何给定日期，该方法通过对生成的价格路径进行平均，其中过去的历史与实际观察到的历史相匹配或“阴影”。我们使用从金融价格的最大熵模型生成的路径进行测试，该模型基于最近提出的标准偏斜度和峰度的多尺度模拟称为“散射谱”。该模型促进了生成路径的多样性，同时重现了金融价格的主要统计特性，包括波动率粗糙度的特征事实。我们的方法为未来实现波动率提供了最先进的预测，并且可以确定超越当前版本的路径依赖波动率模型和期权市场本身的S\&P500的条件期权笑曲线。

    We introduce a Path Shadowing Monte-Carlo method, which provides prediction of future paths, given any generative model. At any given date, it averages future quantities over generated price paths whose past history matches, or `shadows', the actual (observed) history. We test our approach using paths generated from a maximum entropy model of financial prices, based on a recently proposed multi-scale analogue of the standard skewness and kurtosis called `Scattering Spectra'. This model promotes diversity of generated paths while reproducing the main statistical properties of financial prices, including stylized facts on volatility roughness. Our method yields state-of-the-art predictions for future realized volatility and allows one to determine conditional option smiles for the S\&P500 that outperform both the current version of the Path-Dependent Volatility model and the option market itself.
    
[^5]: 庙会模型的新概率分析

    A new probabilistic analysis of the yard-sale model. (arXiv:2308.01485v1 [q-fin.GN])

    [http://arxiv.org/abs/2308.01485](http://arxiv.org/abs/2308.01485)

    这个论文通过新的概率分析证明了Chakraborti的庙会模型中几乎肯定会出现财富凝聚的现象，并且这一结果还扩展到了具有财富获取优势的修改后模型中。

    

    在 Chakraborti 的庙会模型中，相同的代理人进行交易，导致财富的交换，但保持代理人的总财富和每个代理人的预期财富不变。在这个模型中，几乎可以确定地出现财富的凝聚，即收敛到一个代理人拥有一切而其他代理人一无所有的状态。我们给出了一个比现有证明更短且适用于修改后模型的证明，其中存在财富增益优势，即财富较多的两个交易伙伴更有可能从交易中受益。

    In Chakraborti's yard-sale model of an economy, identical agents engage in trades that result in wealth exchanges, but conserve the combined wealth of all agents and each agent's expected wealth. In this model, wealth condensation, that is, convergence to a state in which one agent owns everything and the others own nothing, occurs almost surely. We give a proof of this fact that is much shorter than existing ones and extends to a modified model in which there is a wealth-acquired advantage, i.e., the wealthier of two trading partners is more likely to benefit from the trade.
    
[^6]: 使用图神经网络预测具有溢出效应的多变量实现波动

    Graph Neural Networks for Forecasting Multivariate Realized Volatility with Spillover Effects. (arXiv:2308.01419v1 [q-fin.ST])

    [http://arxiv.org/abs/2308.01419](http://arxiv.org/abs/2308.01419)

    本研究提出了一种使用图神经网络建模和预测多变量实现波动的新方法，能够有效地捕捉非线性关系，并利用多跳邻居的溢出效应提高实现波动的预测准确性。此外，使用拟似似然损失进行训练可以显著提升模型性能。

    

    我们提出了一种新的方法，使用定制的图神经网络来建模和预测多变量的实际波动，以整合股票间的溢出效应。所提出的模型具有将多跳邻居的溢出效应纳入考虑、捕捉非线性关系和使用不同损失函数进行灵活训练的优势。我们的实证结果提供了有力的证据，表明仅考虑多跳邻居的溢出效应并不能在预测准确性方面明显优势。然而，建模非线性溢出效应可以提高实现波动的预测准确性，尤其是对于最长一周的短期预测。此外，我们的结果不断表明，使用拟似似然损失进行训练比常用的均方误差具有明显的模型性能改善。在其他设置中进行的全面一系列实证评估证实了该模型的稳健性。

    We present a novel methodology for modeling and forecasting multivariate realized volatilities using customized graph neural networks to incorporate spillover effects across stocks. The proposed model offers the benefits of incorporating spillover effects from multi-hop neighbors, capturing nonlinear relationships, and flexible training with different loss functions. Our empirical findings provide compelling evidence that incorporating spillover effects from multi-hop neighbors alone does not yield a clear advantage in terms of predictive accuracy. However, modeling nonlinear spillover effects enhances the forecasting accuracy of realized volatilities, particularly for short-term horizons of up to one week. Moreover, our results consistently indicate that training with the Quasi-likelihood loss leads to substantial improvements in model performance compared to the commonly-used mean squared error. A comprehensive series of empirical evaluations in alternative settings confirm the robus
    
[^7]: 一种对数正态SABR模型波动率互换的下界

    A lower bound for the volatility swap in the lognormal SABR model. (arXiv:2306.14602v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.14602](http://arxiv.org/abs/2306.14602)

    本研究对于条件性对数正态的SABR模型，证明了零散度隐含波动率是波动率互换行权价的下界，该结论适用于所有相关参数，并且在相关性小于或等于零时比平值隐含波动率更为精确。

    

    本研究证明，在短期到期的限制条件下，对于条件性对数正态的SABR模型，零散度隐含波动率是波动率互换行权价的下界。对于所有相关参数，该结果都是有效的，并且在相关性小于或等于零时比平值隐含波动率更为精确。

    In the short time to maturity limit it is proved that for the conditionally lognormal SABR model the zero vanna implied volatility is a lower bound for the volatility swap strike. The result is valid for all values of the correlation parameter and is a sharper lower bound than the at-the-money implied volatility for correlation less than or equal to zero.
    
[^8]: NFT市场中的异常交易检测

    Abnormal Trading Detection in the NFT Market. (arXiv:2306.04643v1 [q-fin.TR])

    [http://arxiv.org/abs/2306.04643](http://arxiv.org/abs/2306.04643)

    本文提出了一种通过聚类算法检测非同质化代币（NFT）交易市场中的异常行为的方法，并探讨了监管对减少欺诈行为的影响。

    

    非同质化代币（NFT）市场近年来呈爆炸性增长。据DappRadar统计，最大的NFT市场OpenSea的总交易额在2023年2月达到了347亿美元。然而，NFT市场大部分是未受监管的，存在着重大的洗钱、欺诈和虚假交易等问题。在本文中，我们试图揭示常见的欺诈行为，如虚拟交易，这可能会误导其他交易者。我们使用市场数据从网络、货币和时间的角度设计量化特征，并将其输入到基于K均值聚类的无监督学习算法中，以对交易者进行分类。最后，我们讨论了聚类结果的重要性以及如何通过监管来减少不良行为。我们的工作可能有助于重新建立交易者的信任。

    The Non-Fungible-Token (NFT) market has experienced explosive growth in recent years. According to DappRadar, the total transaction volume on OpenSea, the largest NFT marketplace, reached 34.7 billion dollars in February 2023. However, the NFT market is mostly unregulated and there are significant concerns about money laundering, fraud and wash trading. Amateur traders and retail investors comprise a significant fraction of the NFT market. Hence it is important that researchers highlight the relevant risks involved in NFT trading. In this paper, we attempt to uncover common fraudulent behaviors such as wash trading that could mislead other traders. Using market data, we design quantitative features from the network, monetary, and temporal perspectives that are fed into K-means clustering unsupervised learning algorithm to sort traders into groups. Lastly, we discuss the clustering results' significance and how regulations can reduce undesired behaviors. Our work can potentially help re
    
[^9]: 期权的价格发现

    Price Discovery for Derivatives. (arXiv:2302.13426v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.13426](http://arxiv.org/abs/2302.13426)

    本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。

    

    本文通过一个模型，考虑了私有信息和高阶信息对期权市场价格的影响。模型允许有私有信息的交易者在状态-索赔集市场上交易。等价的期权形式下，我们考虑了拥有关于基础资产收益的分布的私有信息，并允许交易任意期权组合的操纵者。我们得出了通货膨胀需求、价格冲击和信息效率的闭式解，这些解提供了关于内幕交易的高阶信息，如任何给定的时刻交易期权策略，并将这些策略泛化到了波动率交易等实践领域。

    We obtain a basic theory of price discovery across derivative markets with respect to higher-order information, using a model where an agent with general private information regarding state probabilities is allowed to trade arbitrary portfolios of state-contingent claims. In an equivalent options formulation, the informed agent has private information regarding arbitrary aspects of the payoff distribution of an underlying asset and is allowed to trade arbitrary option portfolios. We characterize, in closed form, the informed demand, price impact, and information efficiency of prices. Our results offer a theory of insider trading on higher moments of the underlying payoff as a special case. The informed demand formula prescribes option strategies for trading on any given moment and extends those used in practice for, e.g. volatility trading.
    
[^10]: 我们能否从市场订单流的长时记忆中推断微观金融信息？：对Lillo-Mike-Farmer模型的定量检验。

    Can we infer microscopic financial information from the long memory in market-order flow?: a quantitative test of the Lillo-Mike-Farmer model. (arXiv:2301.13505v2 [q-fin.TR] UPDATED)

    [http://arxiv.org/abs/2301.13505](http://arxiv.org/abs/2301.13505)

    该研究通过对市场订单流的长时记忆进行定量测试，验证了Lillo-Mike-Farmer模型中的微观假设关于订单拆分行为的预测与宏观订单方向相关性的定量关系。

    

    在金融市场中，市场订单的方向表现出强烈的持续性，被广泛称为订单流的长时关联（LRC）; 具体而言，订单方向相关函数显示出具有幂律指数$\gamma$的长时记忆，使得对于较大的时间延迟$\tau$，$C(\tau) \propto \tau^{-\gamma}$。其中最有希望的微观假设之一是个体交易者水平上的订单拆分行为。事实上，Lillo、Mike和Farmer（LMF）在2005年提出了一个简单的微观订单拆分行为模型，该模型预测宏观订单方向相关与微观元订单分布定量相关。尽管这个假设一直是计量经济物理学中的一个核心争议问题，但直接定量验证一直缺乏，因为它需要具有高分辨率的大量微观数据集来观察所有个体交易者的订单拆分行为。在这里，我们首次对这种LMF模型的预测进行了定量验证。

    In financial markets, the market order sign exhibits strong persistence, widely known as the long-range correlation (LRC) of order flow; specifically, the sign correlation function displays long memory with power-law exponent $\gamma$, such that $C(\tau) \propto \tau^{-\gamma}$ for large time-lag $\tau$. One of the most promising microscopic hypotheses is the order-splitting behaviour at the level of individual traders. Indeed, Lillo, Mike, and Farmer (LMF) introduced in 2005 a simple microscopic model of order-splitting behaviour, which predicts that the macroscopic sign correlation is quantitatively associated with the microscopic distribution of metaorders. While this hypothesis has been a central issue of debate in econophysics, its direct quantitative validation has been missing because it requires large microscopic datasets with high resolution to observe the order-splitting behaviour of all individual traders. Here we present the first quantitative validation of this LFM predict
    
[^11]: 使用注意力驱动的动态因子学习的生成CVaR组合优化算法

    Generative CVaR Portfolio Optimization with Attention-Powered Dynamic Factor Learning. (arXiv:2301.07318v3 [q-fin.PM] UPDATED)

    [http://arxiv.org/abs/2301.07318](http://arxiv.org/abs/2301.07318)

    本论文提出了一种使用注意力驱动的动态因子学习的生成CVaR组合优化算法，通过随机变量转换作为分布建模的隐式方式，以及使用注意力-GRU网络进行动态学习和预测，捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。通过在每个投资日期从学习到的生成模型中模拟新样本，并进一步应用CVaR组合优化策略，实现了更明智的投资决策。

    

    动态组合构建问题需要动态建模多元股票收益的联合分布。为了实现这一目标，我们提出了一种动态生成因子模型，它使用随机变量转换作为分布建模的隐式方式，并依赖于使用注意力-GRU网络进行动态学习和预测。所提出的模型捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。我们还提出了一个两步迭代算法来训练模型，然后预测时变的模型参数，包括时不变的尾部参数。在每个投资日期，我们可以轻松地从学习到的生成模型中模拟新样本，并进一步使用模拟样本来进行CVaR组合优化，形成动态组合策略。对股票数据的数值实验表明，我们的模型可以带来更明智的投资，承诺更高的回报风险比，并呈现出...

    The dynamic portfolio construction problem requires dynamic modeling of the joint distribution of multivariate stock returns. To achieve this, we propose a dynamic generative factor model which uses random variable transformation as an implicit way of distribution modeling and relies on the Attention-GRU network for dynamic learning and forecasting. The proposed model captures the dynamic dependence among multivariate stock returns, especially focusing on the tail-side properties. We also propose a two-step iterative algorithm to train the model and then predict the time-varying model parameters, including the time-invariant tail parameters. At each investment date, we can easily simulate new samples from the learned generative model, and we further perform CVaR portfolio optimization with the simulated samples to form a dynamic portfolio strategy. The numerical experiment on stock data shows that our model leads to wiser investments that promise higher reward-risk ratios and present l
    
[^12]: 非线性连续半鞅下的健壮效用最大化问题

    Robust utility maximization with nonlinear continuous semimartingales. (arXiv:2206.14015v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2206.14015](http://arxiv.org/abs/2206.14015)

    本文研究模型不确定性下的连续时间健壮效用最大化问题，通过对偶性证明了对数、指数和幂效用的最优组合的存在性。

    

    本文研究在模型不确定性下的连续时间健壮效用最大化问题。模型不确定性由具有不确定局部特征的连续半鞅所控制。在此处，微分特征由一个取决于时间和路径的集合值函数所规定。我们展示了健壮效用最大化问题与共轭问题的对偶性，并研究了对数、指数和幂效用的最优组合的存在性。

    In this paper we study a robust utility maximization problem in continuous time under model uncertainty. The model uncertainty is governed by a continuous semimartingale with uncertain local characteristics. Here, the differential characteristics are prescribed by a set-valued function that depends on time and path. We show that the robust utility maximization problem is in duality with a conjugate problem, and study the existence of optimal portfolios for logarithmic, exponential and power utilities.
    
[^13]: 对数私人公司估值模型

    The Log Private Company Valuation Model. (arXiv:2206.09666v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2206.09666](http://arxiv.org/abs/2206.09666)

    本文提出了一种基于动态戈登增长模型的对数私人公司估值模型，其中包含闭合式期权定价公式、对冲公式以及权益连接寿险产品的净保费公式。该模型可用于私人公司和公共公司。

    

    针对私人公司由于价格不可观测而导致期权、寿险定价以及对冲等任务具有挑战性，本文介绍了一种基于动态戈登增长模型的对数私人公司估值模型。在本文中，我们得到了针对私人公司的闭合式期权定价公式、对冲公式以及权益连接寿险产品的净保费公式。此外，本文提供了我们模型的极大似然估计器、期望极大化算法和私人公司估值公式。该模型既可用于私人公司，也可用于公共公司。

    For a public company, the option pricing models, hedging models, and pricing and hedging models of equity--linked life insurance products have been developed. However, for a private company, because of unobserved prices, the option and the life insurance pricing, and the hedging are challenging tasks. For this reason, this paper introduces a log private company valuation model, which is based on the dynamic Gordon growth model. In this paper, we obtain closed-form option pricing formulas, hedging formulas, and net premium formulas of equity-linked life insurance products for a private company. Also, the paper provides ML estimators of our model, EM algorithm, and valuation formula for the private company. The suggested model can be used not only by private companies but also by public companies
    

