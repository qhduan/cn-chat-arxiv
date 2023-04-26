# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The cross-sectional stock return predictions via quantum neural network and tensor network.](http://arxiv.org/abs/2304.12501) | 本文研究将量子神经网络和张量网络应用于股票收益预测，在日本股市中张量网络模型表现优于传统模型，并在最新市场环境下呈现出卓越表现。 |
| [^2] | [Asymptotic Expansions for High-Frequency Option Data.](http://arxiv.org/abs/2304.12450) | 本文推导出了适用于一般Ito半鞅类的高阶渐近展开式，用于分析小时间变化下的条件特征函数，该结论可直接应用于高频期权数据分析中。 |
| [^3] | [Long memory, fractional integration and cointegration analysis of real convergence in Spain.](http://arxiv.org/abs/2304.12433) | 本文研究了西班牙自治区实际收入方面的趋同，使用分数阶协整检验发现没有协整的证据，从而排除了所有或一些西班牙地区之间趋同的可能性。 |
| [^4] | [Assessing the difference between integrated quantiles and integrated cumulative distribution functions.](http://arxiv.org/abs/2210.16880) | 本文提供一种数学发明，将经常在风险度量中出现的综合分位数转化为更易处理的综合累积分布函数，避免了存在概率密度函数的要求，可评估模型不确定性和错误设置，以及推导统计推断结果。 |
| [^5] | [Universal approximation of credit portfolio losses using Restricted Boltzmann Machines.](http://arxiv.org/abs/2202.11060) | 本论文提出了一种基于受限玻尔兹曼机的信用组合风险模型，能够更好地拟合实际损失分布并提供更准确的风险度量估计，同时引入的重要性采样算法使得估计高置信度水平下的风险度量变得更加高效。 |
| [^6] | [Empirical Analysis of EIP-1559: Transaction Fees, Waiting Time, and Consensus Security.](http://arxiv.org/abs/2201.05574) | 本文利用以太坊数据，研究了早期的TFM- EIP-1559，它可以改善用户体验，缓解燃料价格的内部区块差异并缩短用户等待时间。然而，它对燃气费用水平和共识安全性的影响仅很小。 |
| [^7] | [Parameterised-Response Zero-Intelligence Traders.](http://arxiv.org/abs/2103.11341) | PRZI是一种新型的零智能交易者，其概率质量函数由一个实值控制变量s确定，可以使其报价价格更加紧急或偏离交易者的限价，用于模拟连续双向拍卖市场动态的研究中。 |
| [^8] | [A New Stock Market Valuation Measure with Applications to Equity-Linked Annuities.](http://arxiv.org/abs/1905.04603) | 本文提出了一种新的股票市场估值方法，采用对数财富与对数收益之间的一阶自回归，可以用于预测股票市场的未来总回报，结果否定了有效市场假说。 |

# 详细

[^1]: 量子神经网络和张量网络在截面股票收益预测中的应用

    The cross-sectional stock return predictions via quantum neural network and tensor network. (arXiv:2304.12501v1 [cs.LG])

    [http://arxiv.org/abs/2304.12501](http://arxiv.org/abs/2304.12501)

    本文研究将量子神经网络和张量网络应用于股票收益预测，在日本股市中张量网络模型表现优于传统模型，并在最新市场环境下呈现出卓越表现。

    

    本文研究了利用量子和量子启发式的机器学习算法进行股票收益预测的应用。其中，我们将量子神经网络（一种适用于噪声中等规模量子计算机的算法）和张量网络（一种受量子启发的机器学习算法）的性能与传统模型如线性回归和神经网络进行比较。通过构建基于模型预测的投资组合并测量投资绩效，我们发现在日本股市中，张量网络模型表现优于传统基准模型（包括线性和神经网络模型）。虽然量子神经网络模型在整个周期内具有降低风险调整超额收益的能力，但最新的市场环境下，量子神经网络和张量网络模型均表现出卓越的性能。

    In this paper we investigate the application of quantum and quantum-inspired machine learning algorithms to stock return predictions. Specifically, we evaluate performance of quantum neural network, an algorithm suited for noisy intermediate-scale quantum computers, and tensor network, a quantum-inspired machine learning algorithm, against classical models such as linear regression and neural networks. To evaluate their abilities, we construct portfolios based on their predictions and measure investment performances. The empirical study on the Japanese stock market shows the tensor network model achieves superior performance compared to classical benchmark models, including linear and neural network models. Though the quantum neural network model attains the lowered risk-adjusted excess return than the classical neural network models over the whole period, both the quantum neural network and tensor network models have superior performances in the latest market environment, which sugges
    
[^2]: 高频期权数据的渐近展开

    Asymptotic Expansions for High-Frequency Option Data. (arXiv:2304.12450v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.12450](http://arxiv.org/abs/2304.12450)

    本文推导出了适用于一般Ito半鞅类的高阶渐近展开式，用于分析小时间变化下的条件特征函数，该结论可直接应用于高频期权数据分析中。

    

    我们推导出了一个非参数的高阶渐近展开式，用于分析Ito半鞅增量的条件特征函数小时间变化。该渐近分析涵盖了时间区间长度和条件特征函数评估时间间隔同时缩小的情形，其中基础过程的现货半鞅特征以及它们的现货半鞅特征作为渐近展开的主项。分析适用于一般It\^o半鞅类，包括L\'evy驱动的SDEs和时间变化的L\'evy过程。这些渐近展开结果可直接用于从标的资产的高频数据中构建关于其随机波动率动态的非参数估计。

    We derive a nonparametric higher-order asymptotic expansion for small-time changes of conditional characteristic functions of It\^o semimartingale increments. The asymptotics setup is of joint type: both the length of the time interval of the increment of the underlying process and the time gap between evaluating the conditional characteristic function are shrinking. The spot semimartingale characteristics of the underlying process as well as their spot semimartingale characteristics appear as leading terms in the derived asymptotic expansions. The analysis applies to a general class of It\^o semimartingales that includes in particular L\'evy-driven SDEs and time-changed L\'evy processes. The asymptotic expansion results are of direct use for constructing nonparametric estimates pertaining to the stochastic volatility dynamics of an asset from high-frequency data of options written on the underlying asset.
    
[^3]: 西班牙实际收入趋同的长记忆、分数阶积分和协整分析

    Long memory, fractional integration and cointegration analysis of real convergence in Spain. (arXiv:2304.12433v1 [econ.GN])

    [http://arxiv.org/abs/2304.12433](http://arxiv.org/abs/2304.12433)

    本文研究了西班牙自治区实际收入方面的趋同，使用分数阶协整检验发现没有协整的证据，从而排除了所有或一些西班牙地区之间趋同的可能性。

    

    本文研究了西班牙自治区人均实际收入方面的经济趋同。为了实现趋同，这些系列应该是协整的。本文使用最近提出的两种分数阶协整检验策略来检查这个必要条件，发现没有协整的证据，这排除了所有或一些西班牙地区之间趋同的可能性。作为额外的贡献，对于一个不同的变量数量和样本大小，本文提供了一个分数阶协整检验的一个测试的临界值扩展，这个测试最初的值由作者提供，适合本文考虑的值。

    This paper investigates economic convergence in terms of real income per capita among the autonomous regions of Spain. In order to converge, the series should cointegrate. This necessary condition is checked using two testing strategies recently proposed for fractional cointegration, finding no evidence of cointegration, which rules out the possibility of convergence between all or some of the Spanish regions. As an additional contribution, an extension of the critical values of one of the tests of fractional cointegration is provided for a different number of variables and sample sizes from those originally provided by the author, fitting those considered in this paper.
    
[^4]: 综合分位数与综合累积分布函数之间的差异评估

    Assessing the difference between integrated quantiles and integrated cumulative distribution functions. (arXiv:2210.16880v3 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2210.16880](http://arxiv.org/abs/2210.16880)

    本文提供一种数学发明，将经常在风险度量中出现的综合分位数转化为更易处理的综合累积分布函数，避免了存在概率密度函数的要求，可评估模型不确定性和错误设置，以及推导统计推断结果。

    

    本文提供了一种数学发明，展示了如何将常常在风险度量中出现的综合分位数转化为技术上更易处理的综合累积分布函数。该发明有助于避免在处理包含分位数的量时通常需要强加的一些技术假设。特别地，它可以完全避免要求存在概率密度函数的要求。本文阐释和说明了这一发明，其副产品包括模型不确定性和错误设置的评估，以及统计推断结果的推导。

    This paper offers a mathematical invention that shows how to convert integrated quantiles, which often appear in risk measures, into integrated cumulative distribution functions, which are technically more tractable from various perspectives. The invention helps to avoid a number of technical assumptions that have been traditionally imposed when working with quantities containing quantiles. In particular it helps to completely avoid the requirement of the existence of a probability density function. The developed results explain and illustrate the invention, whose byproducts include the assessment of model uncertainty and misspecification, and the derivation of statistical inference results.
    
[^5]: 利用受限玻尔兹曼机实现信用组合损失的通用逼近

    Universal approximation of credit portfolio losses using Restricted Boltzmann Machines. (arXiv:2202.11060v3 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2202.11060](http://arxiv.org/abs/2202.11060)

    本论文提出了一种基于受限玻尔兹曼机的信用组合风险模型，能够更好地拟合实际损失分布并提供更准确的风险度量估计，同时引入的重要性采样算法使得估计高置信度水平下的风险度量变得更加高效。

    

    我们介绍了一种基于受限玻尔兹曼机（RBMs）的新型组合信用风险模型，这是一种具有通用逼近损失分布能力的随机神经网络。我们在一个包含1,012家美国公司违约概率的实证数据集上测试了该模型，并且证明它在多个信用风险管理任务中优于常用的参数因子Copula模型，如高斯或t因子Copula模型。特别地，该模型可以更好地拟合实际损失分布，并提供更准确的风险度量估计。我们引入了一种重要性采样过程，可以以计算高效的方式估计置信度水平较高的风险度量，并且这是目前对于Copula模型可用的蒙特卡罗技术的重大改进。此外，该模型提取的统计因素在组合行业结构方面具有解释性，并为实践提供了参考。

    We introduce a new portfolio credit risk model based on Restricted Boltzmann Machines (RBMs), which are stochastic neural networks capable of universal approximation of loss distributions. We test the model on an empirical dataset of default probabilities of 1'012 US companies and we show that it outperforms commonly used parametric factor copula models -- such as the Gaussian or the t factor copula models -- across several credit risk management tasks. In particular, the model leads to better fits for the empirical loss distribution and more accurate risk measure estimations. We introduce an importance sampling procedure which allows risk measures to be estimated at high confidence levels in a computationally efficient way and which is a substantial improvement over the Monte Carlo techniques currently available for copula models. Furthermore, the statistical factors extracted by the model admit an interpretation in terms of the underlying portfolio sector structure and provide practi
    
[^6]: EIP-1559的实证分析：交易费用、等待时间和共识安全性

    Empirical Analysis of EIP-1559: Transaction Fees, Waiting Time, and Consensus Security. (arXiv:2201.05574v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2201.05574](http://arxiv.org/abs/2201.05574)

    本文利用以太坊数据，研究了早期的TFM- EIP-1559，它可以改善用户体验，缓解燃料价格的内部区块差异并缩短用户等待时间。然而，它对燃气费用水平和共识安全性的影响仅很小。

    

    交易费机制（TFM）是区块链协议的重要组成部分。然而，TFMs实际影响的系统评估仍然缺乏。本文利用以太坊区块链、mempool和交易所的丰富数据，研究了EIP-1559的影响，它是最早部署的TFMs之一，其不同于传统的一级价格拍卖范例。我们进行了严谨和全面的实证研究，以检查其对区块链交易费用动态、交易等待时间和共识安全性的因果影响。我们的结果表明，EIP-1559通过减缓燃料价格的内部区块差异和缩短用户等待时间来改善用户体验。但是，EIP-1559对燃气费用水平和共识安全性的影响仅很小。此外，我们发现当Ether的价格更加波动时，等待时间显著增加。我们还证实较大的区块大小增加了姐妹块的存在。这些发现提示了新的研究方向。

    A transaction fee mechanism (TFM) is an essential component of a blockchain protocol. However, a systematic evaluation of the real-world impact of TFMs is still absent. Using rich data from the Ethereum blockchain, the mempool, and exchanges, we study the effect of EIP-1559, one of the earliest-deployed TFMs that depart from the traditional first-price auction paradigm. We conduct a rigorous and comprehensive empirical study to examine its causal effect on blockchain transaction fee dynamics, transaction waiting times, and consensus security. Our results show that EIP-1559 improves the user experience by mitigating intrablock differences in the gas price paid and reducing users' waiting times. However, EIP-1559 has only a small effect on gas fee levels and consensus security. In addition, we find that when Ether's price is more volatile, the waiting time is significantly higher. We also verify that a larger block size increases the presence of siblings. These findings suggest new direc
    
[^7]: 参数化响应零智能交易者

    Parameterised-Response Zero-Intelligence Traders. (arXiv:2103.11341v7 [q-fin.TR] UPDATED)

    [http://arxiv.org/abs/2103.11341](http://arxiv.org/abs/2103.11341)

    PRZI是一种新型的零智能交易者，其概率质量函数由一个实值控制变量s确定，可以使其报价价格更加紧急或偏离交易者的限价，用于模拟连续双向拍卖市场动态的研究中。

    

    我介绍了PRZI（参数化响应零智能）交易者，这是一种新型的零智能交易者，旨在用于模拟连续双向拍卖市场动态的研究中。与Gode＆Sunder的经典ZIC交易者类似，PRZI从允许的询价价格的随机分布中生成报价价格。与使用均匀分布生成价格的ZIC不同，PRZI交易者的概率分布被参数化，其概率质量函数（PMF）由在[-1.0，+1.0]范围内确定该交易者策略的实值控制变量s确定。当s = 0时，PRZI交易者与均匀PMF的ZIC相同；但当| s | =~1时，PRZI交易者的PMF变得极度偏斜，向价格范围的某一端倾斜，从而使其报价价格更加紧急，将报价价格分布偏向或远离交易者的限价。为了探索共同演化的动态，我用PRZI代替了ZIC来构建一个人工市场，并发现，随着交易者策略通过基于收益的选择进行演化，市场会出现平衡状态。

    I introduce PRZI (Parameterised-Response Zero Intelligence), a new form of zero-intelligence trader intended for use in simulation studies of the dynamics of continuous double auction markets. Like Gode & Sunder's classic ZIC trader, PRZI generates quote-prices from a random distribution over some specified domain of allowable quote-prices. Unlike ZIC, which uses a uniform distribution to generate prices, the probability distribution in a PRZI trader is parameterised in such a way that its probability mass function (PMF) is determined by a real-valued control variable s in the range [-1.0, +1.0] that determines the _strategy_ for that trader. When s=0, a PRZI trader is identical to ZIC, with a uniform PMF; but when |s|=~1 the PRZI trader's PMF becomes maximally skewed to one extreme or the other of the price-range, thereby making its quote-prices more or less urgent, biasing the quote-price distribution toward or away from the trader's limit-price. To explore the co-evolutionary dynami
    
[^8]: 一种新的股票市场估值方法及其在股票相关养老金中的应用

    A New Stock Market Valuation Measure with Applications to Equity-Linked Annuities. (arXiv:1905.04603v13 [q-fin.ST] UPDATED)

    [http://arxiv.org/abs/1905.04603](http://arxiv.org/abs/1905.04603)

    本文提出了一种新的股票市场估值方法，采用对数财富与对数收益之间的一阶自回归，可以用于预测股票市场的未来总回报，结果否定了有效市场假说。

    

    本文对用于预测股票市场未来总回报的经典席勒循环调整市盈率（CAPE）进行了概括和推广。在本文中，收益增长被视为外生变量。通过将对数财富与对数收益之间的差建模为具有线性趋势4.5％和高斯创新的一阶自回归，我们得到了一种新的估值方法。这个自回归与随机漫步有很大的区别。因此，我们的结果否定了有效市场假说。长期总回报等于长期收益增长加上4.5％。我们将这些结果应用于退休计划中。

    We generalize the classic Shiller cyclically adjusted price-earnings ratio (CAPE) used for prediction of future total returns of the stock market. We treat earnings growth as exogenous. The difference between log wealth and log earnings is modeled as an autoregression of order 1 with linear trend 4.5% and Gaussian innovations. Detrending gives us a new valuation measure. This autoregression is significantly different from the random walk. Therefore, our results disprove the Efficient Market Hypothesis. Therefore, long-run total returns equal long-run earnings growth plus 4.5%. We apply results to retirement planning.
    

