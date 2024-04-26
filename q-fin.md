# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Probabilistic Forecasting with Applications in Market Operations](https://arxiv.org/abs/2403.05743) | 提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。 |
| [^2] | [Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior](https://arxiv.org/abs/2402.03953) | 本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。 |
| [^3] | [Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision.](http://arxiv.org/abs/2309.08431) | 本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。 |
| [^4] | [A stochastic control problem arising from relaxed wealth tracking with a monotone benchmark process.](http://arxiv.org/abs/2302.08302) | 本文研究了一种非标准的随机控制问题，旨在最大化消费效用，同时满足底部约束和基准过程。通过引入两个带反射的辅助状态过程，建立了等效的辅助控制问题，得到了关键的对偶价值函数性质，并推导出了一些有趣的经济影响。 |
| [^5] | [Convolution Bounds on Quantile Aggregation.](http://arxiv.org/abs/2007.09320) | 本文建立了一种新的分位数聚合分析界限——卷积界限，其是目前一般情况下最好的界限。卷积界限易于计算，在许多相关情况下尖锐，并且允许最极端的依赖结构的可解释性。此外，本文还讨论了相关应用问题。 |

# 详细

[^1]: 具有市场运营应用的生成式概率预测

    Generative Probabilistic Forecasting with Applications in Market Operations

    [https://arxiv.org/abs/2403.05743](https://arxiv.org/abs/2403.05743)

    提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。

    

    本文提出了一种新颖的生成式概率预测方法，该方法源自于非参数时间序列的Wiener-Kallianpur创新表示。在生成人工智能的范式下，所提出的预测架构包括一个自编码器，将非参数多变量随机过程转化为规范的创新序列，从中根据过去样本生成未来时间序列样本，条件是它们的概率分布取决于过去样本。提出了一种新的深度学习算法，将潜在过程限制为具有匹配自编码器输入-输出条件概率分布的独立同分布序列。建立了所提出的生成式预测方法的渐近最优性和结构收敛性质。该方法在实时市场运营中涉及高度动态和波动时间序列的三个应用方面。

    arXiv:2403.05743v1 Announce Type: cross  Abstract: This paper presents a novel generative probabilistic forecasting approach derived from the Wiener-Kallianpur innovation representation of nonparametric time series. Under the paradigm of generative artificial intelligence, the proposed forecasting architecture includes an autoencoder that transforms nonparametric multivariate random processes into canonical innovation sequences, from which future time series samples are generated according to their probability distributions conditioned on past samples. A novel deep-learning algorithm is proposed that constrains the latent process to be an independent and identically distributed sequence with matching autoencoder input-output conditional probability distributions. Asymptotic optimality and structural convergence properties of the proposed generative forecasting approach are established. Three applications involving highly dynamic and volatile time series in real-time market operations a
    
[^2]: 中心化和去中心化交易所中的永续期货合约: 机制和交易者行为

    Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior

    [https://arxiv.org/abs/2402.03953](https://arxiv.org/abs/2402.03953)

    本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。

    

    本研究提出了一个具有开创性的知识系统化(SoK)计划，重点深入探索交易者在中心化交易所(CEXs)和去中心化交易所(DEXs)中关于永续期货合约的动态和行为。我们改进了现有模型，以研究交易者对价格波动的反应，创建了一个针对这些合约平台的新的分析框架，同时突出了区块链技术在其应用中的作用。我们的研究包括对CEXs的历史数据的比较分析，以及对DEXs上的完整交易数据的更详尽的研究。在虚拟自动化市场做市商(VAMM)模型的DEX上，多头和空头持仓量对价格波动产生相反的影响，这归因于VAMM的价格形成机制。在采用预言机定价模型的DEX中，我们观察到买家和卖家之间交易者行为上存在明显的不对称性。

    This study presents a groundbreaking Systematization of Knowledge (SoK) initiative, focusing on an in-depth exploration of the dynamics and behavior of traders on perpetual future contracts across both centralized exchanges (CEXs), and decentralized exchanges (DEXs). We have refined the existing model for investigating traders' behavior in reaction to price volatility to create a new analytical framework specifically for these contract platforms, while also highlighting the role of blockchain technology in their application. Our research includes a comparative analysis of historical data from CEXs and a more extensive examination of complete transactional data on DEXs. On DEX of Virtual Automated Market Making (VAMM) Model, open interest on short and long positions exert effect on price volatility in opposite direction, attributable to VAMM's price formation mechanism. In the DEXs with Oracle Pricing Model, we observed a distinct asymmetry in trader behavior between buyers and sellers.
    
[^3]: 去中心化金融与自动化市场做市：可预测的损失和最优流动性提供

    Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision. (arXiv:2309.08431v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.08431](http://arxiv.org/abs/2309.08431)

    本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    

    在这篇论文中，我们对动态调整其在集中流动性池中提供流动性范围的战略性流动性提供者的连续时间财富动态进行了表征。他们的财富来自手续费收入和他们在池中持有的资产的价值。接下来，我们推导出了一种自融资和封闭形式的最优流动性提供策略，其中流动性提供者的范围宽度由池的盈利能力（提供费用减去燃气费用）、可预测损失（持仓的损失）和集中风险决定。集中风险是指如果池中的边际兑换率（类似于限价订单簿中的中间价）超出流动性提供者的范围，费用收入会下降。当边际兑换率由随机漂移驱动时，我们展示了如何通过最优调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    Constant product markets with concentrated liquidity (CL) are the most popular type of automated market makers. In this paper, we characterise the continuous-time wealth dynamics of strategic LPs who dynamically adjust their range of liquidity provision in CL pools. Their wealth results from fee income and the value of their holdings in the pool. Next, we derive a self-financing and closed-form optimal liquidity provision strategy where the width of the LP's liquidity range is determined by the profitability of the pool (provision fees minus gas fees), the predictable losses (PL) of the LP's position, and concentration risk. Concentration risk refers to the decrease in fee revenue if the marginal exchange rate (akin to the midprice in a limit order book) in the pool exits the LP's range of liquidity. When the marginal rate is driven by a stochastic drift, we show how to optimally skew the range of liquidity to increase fee revenue and profit from the expected changes in the marginal ra
    
[^4]: 起源于带有单调基准过程的松弛财富跟踪的随机控制问题

    A stochastic control problem arising from relaxed wealth tracking with a monotone benchmark process. (arXiv:2302.08302v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.08302](http://arxiv.org/abs/2302.08302)

    本文研究了一种非标准的随机控制问题，旨在最大化消费效用，同时满足底部约束和基准过程。通过引入两个带反射的辅助状态过程，建立了等效的辅助控制问题，得到了关键的对偶价值函数性质，并推导出了一些有趣的经济影响。

    

    本文研究了一种非标准的随机控制问题，其灵感来源于带有非递减基准过程的最优消费与财富跟踪。具体而言，单调基准是通过漂移布朗运动的运行最大值来建模的。我们考虑使用资本注入的松弛跟踪公式，使得注入的资本所补偿的财富在任何时候都优于基准过程。随机控制问题是在动态底部约束下，最大化削减资本注入成本后的消费的期望效用。通过引入两个带反射的辅助状态过程，我们制定了一个等效的辅助控制问题，同时研究了杠杆资本注入和底部约束控制，使其隐匿起来。为了解决带有两个Neumann边界条件的HJB方程，我们使用一些新颖的概率技巧建立了对偶PDE的唯一经典解分离形式的存在性。我们的主要贡献是确定了对偶价值函数的一些关键性质，从而使我们能够建立松弛财富跟踪问题的最优解的存在性和唯一性，并得出了一些有趣的经济影响。

    This paper studies a nonstandard stochastic control problem motivated by the optimal consumption with wealth tracking of a non-decreasing benchmark process. In particular, the monotone benchmark is modelled by the running maximum of a drifted Brownian motion. We consider a relaxed tracking formulation using capital injection such that the wealth compensated by the injected capital dominates the benchmark process at all times. The stochastic control problem is to maximize the expected utility on consumption deducted by the cost of the capital injection under the dynamic floor constraint. By introducing two auxiliary state processes with reflections, an equivalent auxiliary control problem is formulated and studied such that the singular control of capital injection and the floor constraint can be hidden. To tackle the HJB equation with two Neumann boundary conditions, we establish the existence of a unique classical solution to the dual PDE in a separation form using some novel probabil
    
[^5]: 推断分位数聚合的卷积界限

    Convolution Bounds on Quantile Aggregation. (arXiv:2007.09320v3 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2007.09320](http://arxiv.org/abs/2007.09320)

    本文建立了一种新的分位数聚合分析界限——卷积界限，其是目前一般情况下最好的界限。卷积界限易于计算，在许多相关情况下尖锐，并且允许最极端的依赖结构的可解释性。此外，本文还讨论了相关应用问题。

    

    带有依赖不确定性的分位数聚合在概率论中有着悠久的历史，广泛应用于金融、风险管理、统计学和运筹学等领域。本文利用关于基于分位数风险度量的 inf-卷积最近的结果，建立了我们称之为卷积界限的新的分位数聚合分析界限。卷积界限统一了分位数聚合中的每一个可用分析结果，并启发我们对这些方法的理解。这些界限是一般情况下最好的。此外，卷积界限易于计算，并且我们展示它们在许多相关情况下是尖锐的。它们还允许最极端的依赖结构的可解释性。结果直接导致随机变量和的分布界限与任意相关性的随机变量。我们讨论了风险管理和经济学中的相关应用问题。

    Quantile aggregation with dependence uncertainty has a long history in probability theory with wide applications in finance, risk management, statistics, and operations research. Using a recent result on inf-convolution of quantile-based risk measures, we establish new analytical bounds for quantile aggregation which we call convolution bounds. Convolution bounds both unify every analytical result available in quantile aggregation and enlighten our understanding of these methods. These bounds are the best available in general. Moreover, convolution bounds are easy to compute, and we show that they are sharp in many relevant cases. They also allow for interpretability on the extremal dependence structure. The results directly lead to bounds on the distribution of the sum of random variables with arbitrary dependence. We discuss relevant applications in risk management and economics.
    

