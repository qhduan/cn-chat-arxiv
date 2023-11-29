# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating Stable Fixed Points and Langevin Potentials for Financial Dynamics.](http://arxiv.org/abs/2309.12082) | 本文将几何布朗运动模型推广为具有多项式漂移的随机微分方程，并通过模型选择确定最优模型为阶数为2的模型。势函数集合表明存在明显的势能井，表明稳定价格的存在。 |
| [^2] | [AI Regulation in the European Union: Examining Non-State Actor Preferences.](http://arxiv.org/abs/2305.11523) | 本篇文章研究了欧洲联盟AI法案，对非国家行为者的规制偏好进行了系统分析。所有类型的非国家行为者都支持对AI进行某种形式的规制，但是在规制范围和严格性上存在显着差异，这可以解释为行业水平的竞争和法规可能带来的分配结果差异。 |
| [^3] | [The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations.](http://arxiv.org/abs/2210.15914) | 这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。 |
| [^4] | [Automated Market Making and Loss-Versus-Rebalancing.](http://arxiv.org/abs/2208.06046) | 本文研究常量函数市场制造者（CFMM）的市场微观结构，发现CFMM表现不如再平衡策略，提出了损失与再平衡(LVR)的概念，可用于评估投资决策和指导协议设计。 |
| [^5] | [Missing Values and the Dimensionality of Expected Returns.](http://arxiv.org/abs/2207.13071) | 该研究比较了多种插值方法，发现预测变量通常独立，因此即使采用特定值估计，最终结果与最大似然估计相似。此外，50个主成分可捕捉等权重的期望回报，而使用神经网络投资组合也有类似的结果。 |

# 详细

[^1]: 估计金融动力学的稳定不动点和朗之万势能

    Estimating Stable Fixed Points and Langevin Potentials for Financial Dynamics. (arXiv:2309.12082v1 [q-fin.ST])

    [http://arxiv.org/abs/2309.12082](http://arxiv.org/abs/2309.12082)

    本文将几何布朗运动模型推广为具有多项式漂移的随机微分方程，并通过模型选择确定最优模型为阶数为2的模型。势函数集合表明存在明显的势能井，表明稳定价格的存在。

    

    几何布朗运动是量化金融中的标准模型，但其随机微分方程的势函数不能包含稳定的非零价格。本文将几何布朗运动推广为阶数为q的多项式漂移的随机微分方程，并通过模型选择表明q=2最常被认为是描述数据的最优模型。此外，通过马尔科夫链蒙特卡洛的势函数集合表明存在明显的势能井，表明稳定价格的存在。

    The Geometric Brownian Motion (GBM) is a standard model in quantitative finance, but the potential function of its stochastic differential equation (SDE) cannot include stable nonzero prices. This article generalises the GBM to an SDE with polynomial drift of order q and shows via model selection that q=2 is most frequently the optimal model to describe the data. Moreover, Markov chain Monte Carlo ensembles of the accompanying potential functions show a clear and pronounced potential well, indicating the existence of a stable price.
    
[^2]: 欧洲联盟的人工智能规制: 探究非国家行为者的偏好

    AI Regulation in the European Union: Examining Non-State Actor Preferences. (arXiv:2305.11523v1 [econ.GN])

    [http://arxiv.org/abs/2305.11523](http://arxiv.org/abs/2305.11523)

    本篇文章研究了欧洲联盟AI法案，对非国家行为者的规制偏好进行了系统分析。所有类型的非国家行为者都支持对AI进行某种形式的规制，但是在规制范围和严格性上存在显着差异，这可以解释为行业水平的竞争和法规可能带来的分配结果差异。

    

    随着人工智能（AI）的发展和应用不断增长，政策制定者越来越在努力解决如何规制该技术的问题。最具影响力的国际倡议是欧洲联盟（EU）的AI法案，旨在建立第一个全面的AI规制框架。本文首次系统分析了非国家行为者对人工智能国际规制的偏好，重点研究了EU AI法案的情况。在理论上，我们阐述了商业行为者和其他非国家行为者在不同的AI行业竞争条件下的规制偏好的论点。在经验上，我们使用关于欧洲AI规制的公共咨询中非国家行为者偏好的数据来测试这些期望。我们的研究结果有三个方面。首先，所有类型的非国家行为者都表达了对AI的担忧，并支持以某种形式规制AI。其次，尽管如此，不同类型的非国家行为者在AI的规制严格性和范围方面存在显着差异。第三，这些差异部分可以通过规制可能带来的行业水平竞争和分配结果的差异来解释。总体而言，我们的分析揭示了AI规制的复杂和有争议的政治，不仅在欧盟内部，而且超越欧盟的背景下也是如此。

    As the development and use of artificial intelligence (AI) continues to grow, policymakers are increasingly grappling with the question of how to regulate this technology. The most far-reaching international initiative is the European Union (EU) AI Act, which aims to establish the first comprehensive framework for regulating AI. In this article, we offer the first systematic analysis of non-state actor preferences toward international regulation of AI, focusing on the case of the EU AI Act. Theoretically, we develop an argument about the regulatory preferences of business actors and other non-state actors under varying conditions of AI sector competitiveness. Empirically, we test these expectations using data on non-state actor preferences from public consultations on European AI regulation. Our findings are threefold. First, all types of non-state actors express concerns about AI and support regulation in some form. Second, there are nonetheless significant differences across actor ty
    
[^3]: 移民、移民者和当地人在欧洲知识聚集形成中的历史角色

    The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations. (arXiv:2210.15914v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2210.15914](http://arxiv.org/abs/2210.15914)

    这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。

    

    移民是不是让巴黎成为了艺术圣地，维也纳成为了古典音乐的灯塔？还是他们的崛起纯粹是当地人的结果？在这里，我们使用了关于22000多名生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。我们发现，一个地区在某种活动（基于著名物理学家、画家等的出生）发展或保持专业化的概率随着对该活动具有知识的移民和对相关活动具有知识的移民的存在而增加。相比之下，我们并没有找到有力的证据表明当地人具有相关知识的存在解释了进入和/或退出。我们通过考虑任何特定地点-时期-活动因素（例如吸引科学家的新大学的存在）的固定效应模型来解决一些内生性问题。

    Did migrants make Paris a Mecca for the arts and Vienna a beacon of classical music? Or was their rise a pure consequence of local actors? Here, we use data on more than 22,000 historical individuals born between the years 1000 and 2000 to estimate the contribution of famous immigrants, emigrants, and locals to the knowledge specializations of European regions. We find that the probability that a region develops or keeps specialization in an activity (based on the birth of famous physicists, painters, etc.) grows with both, the presence of immigrants with knowledge on that activity and immigrants with knowledge in related activities. In contrast, we do not find robust evidence that the presence of locals with related knowledge explains entries and/or exits. We address some endogeneity concerns using fixed-effects models considering any location-period-activity specific factors (e.g. the presence of a new university attracting scientists).
    
[^4]: 自动化市场做市商及损失与再平衡。

    Automated Market Making and Loss-Versus-Rebalancing. (arXiv:2208.06046v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2208.06046](http://arxiv.org/abs/2208.06046)

    本文研究常量函数市场制造者（CFMM）的市场微观结构，发现CFMM表现不如再平衡策略，提出了损失与再平衡(LVR)的概念，可用于评估投资决策和指导协议设计。

    

    我们从被动流动性提供者的角度考虑了常量函数市场制造者（CFMM）的市场微观结构。在Black-Scholes情境下，我们比较CFMM的表现与再平衡策略的表现，该策略在市场价格上复制CFMM的交易。由于CFMM以比市场价格更差的价格执行所有交易，因此CFMM系统地表现不佳。两种策略之间的绩效差异“损失与再平衡”（LVR，发音为“lever”）取决于基础资产的波动率和CFMM bonding函数的边际流动性。我们模型中的CFMM损失表达式与Uniswap v2 WETH-USDC对的实际损失相匹配。LVR为CFMM流动性提供者的投资决策的前后评估提供了可交易的见解，并且还可以指导CFMM协议的设计。

    We consider the market microstructure of constant function market makers (CFMMs) from the perspective of passive liquidity providers (LPs). In a Black-Scholes setting, we compare the CFMM's performance to that of a rebalancing strategy, which replicates the CFMM's trades at market prices. The CFMM systematically underperforms the rebalancing strategy, because it executes all trades at worse-than-market prices. The performance gap between the two strategies, "loss-versus-rebalancing" (LVR, pronounced "lever"), depends on the volatility of the underlying asset and the marginal liquidity of the CFMM bonding function. Our model's expressions for CFMM losses match actual losses from the Uniswap v2 WETH-USDC pair. LVR provides tradeable insight in both the ex ante and ex post assessment of CFMM LP investment decisions, and can also inform the design of CFMM protocols.
    
[^5]: 缺失值和期望回报维度的研究

    Missing Values and the Dimensionality of Expected Returns. (arXiv:2207.13071v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2207.13071](http://arxiv.org/abs/2207.13071)

    该研究比较了多种插值方法，发现预测变量通常独立，因此即使采用特定值估计，最终结果与最大似然估计相似。此外，50个主成分可捕捉等权重的期望回报，而使用神经网络投资组合也有类似的结果。

    

    通过将许多横截面回报预测变量组合在一起（例如，在机器学习中），通常需要插值缺失值。我们比较了几种方法，包括最大似然估计和特定值估计。令人惊讶的是，最大似然估计和特定值估计产生了类似的结果。这是因为预测变量在很大程度上是独立的：相关性集群接近于零，10个主成分(PC)跨度少于总方差的50％。独立性意味着观察到的变量对缺失变量没有信息贡献，使特定方法有效。 在PC回归测试中，需要50个PC才能捕捉等权重期望回报（30个PC权值加权），不管imputation是什么。 我们发现神经网络投资组合中也有类似的不变性。

    Combining many cross-sectional return predictors (for example, in machine learning) often requires imputing missing values. We compare ad-hoc mean imputation with several methods including maximum likelihood. Surprisingly, maximum likelihood and ad-hoc methods lead to similar results. This is because predictors are largely independent: Correlations cluster near zero and 10 principal components (PCs) span less than 50% of total variance. Independence implies observed predictors are uninformative about missing predictors, making ad-hoc methods valid. In PC regression tests, 50 PCs are required to capture equal-weighted expected returns (30 PCs value-weighted), regardless of the imputation. We find similar invariance in neural network portfolios.
    

