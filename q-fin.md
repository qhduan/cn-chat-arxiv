# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic expansion for the pricing of Asian options](https://arxiv.org/abs/2402.17684) | 该论文提出了在Black-Scholes模型下利用随机泰勒展开得到的近似解析公式，用于离散平均的亚洲期权定价，实践中表现高度准确。 |
| [^2] | [The critical disordered pinning measure](https://arxiv.org/abs/2402.17642) | 论文研究了一个由随机游走引起的无序钉扎模型，发现在临界区间内点对点的分区函数收敛到一种唯一的极限随机测度，并命名为临界无序钉扎测度。 |
| [^3] | [Political Pandering and Bureaucratic Influence](https://arxiv.org/abs/2402.17526) | 研究发现在政策实施中，政治家和官僚之间的互动在选举激励下会产生讨好行为，不同类型的官僚对此有不同态度，分析显示官僚主义的影响程度能够最大化选民福利。 |
| [^4] | [Navigating Complexity: Constrained Portfolio Analysis in High Dimensions with Tracking Error and Weight Constraints](https://arxiv.org/abs/2402.17523) | 本文分析了高维度投资组合中受限条件下的统计特性，展示了在大维度下如何一致估计这些投资组合，同时提供了受限组合权重、风险和夏普比率的收敛速度结果。 |
| [^5] | [Limit Order Book Simulations: A Review](https://arxiv.org/abs/2402.17359) | 本综述研究了当前先进的各种限价订单簿（LOB）模拟模型，在方法学分类的基础上提供了流行风格事实的整体视图，重点研究了模型中的价格冲击现象。 |
| [^6] | [The Random Forest Model for Analyzing and Forecasting the US Stock Market in the Context of Smart Finance](https://arxiv.org/abs/2402.17194) | 本研究评估了结合人工智能的随机森林模型在预测美国股市走势上的预测性能，为投资者提供了决策参考。 |
| [^7] | [Withdrawal Success Optimization in a Pooled Annuity Fund](https://arxiv.org/abs/2402.17164) | 在合并年金基金中，通过优化投资组合权重函数，最大化特定年金购买者完成规定提取直至死亡的概率，并且得到明显增加的最大概率。 |
| [^8] | [Time series generation for option pricing on quantum computers using tensor network](https://arxiv.org/abs/2402.17148) | 提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。 |
| [^9] | [A monotone piecewise constant control integration approach for the two-factor uncertain volatility model](https://arxiv.org/abs/2402.06840) | 这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。 |
| [^10] | [Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools](https://arxiv.org/abs/2311.10801) | 使用EarnMore方法，我们提出了一种新的RL方法，可以允许RL代理与可定制股票池（CSPs）交互，而不需要重新训练。 |
| [^11] | [Forecasting Volatility with Machine Learning and Rough Volatility: Example from the Crypto-Winter](https://arxiv.org/abs/2311.04727) | 通过结合LSTM和粗糙波动率，我们发现波动率预测模型在加密货币领域的表现优于传统模型，同时从简约的参数模型中得到类似预测表现，进一步证明了波动率形成过程机制的普遍性。 |
| [^12] | [Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks.](http://arxiv.org/abs/2401.08610) | 这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。 |
| [^13] | [Discrete time optimal investment under model uncertainty.](http://arxiv.org/abs/2307.11919) | 本文研究了在模型不确定性下的离散时间最优投资问题，利用原始方法证明了存在一个最优投资策略，并引入了类型(A)的效用函数。 |
| [^14] | [The cross-sectional stock return predictions via quantum neural network and tensor network.](http://arxiv.org/abs/2304.12501) | 本文研究将量子神经网络和张量网络应用于股票收益预测，在日本股市中张量网络模型表现优于传统模型，并在最新市场环境下呈现出卓越表现。 |
| [^15] | [Uncertainty over Uncertainty in Environmental Policy Adoption: Bayesian Learning of Unpredictable Socioeconomic Costs.](http://arxiv.org/abs/2304.10344) | 本文开发了一个类似于实物期权的模型，以捕捉环境政策采用中的两层不确定性。决策者能够通过跟踪成本的实际演变来学习未知的漂移，并形成后验动态信念其真正价值。 |

# 详细

[^1]: 亚洲期权定价的随机扩展

    Stochastic expansion for the pricing of Asian options

    [https://arxiv.org/abs/2402.17684](https://arxiv.org/abs/2402.17684)

    该论文提出了在Black-Scholes模型下利用随机泰勒展开得到的近似解析公式，用于离散平均的亚洲期权定价，实践中表现高度准确。

    

    我们在具有时间相关参数的Black-Scholes模型下，针对离散平均的亚洲期权定价提出了封闭的解析近似。这些公式是通过在一个对数正态代理模型周围进行随机泰勒展开得到的，并在实践中被发现非常准确。

    arXiv:2402.17684v1 Announce Type: new  Abstract: We present closed analytical approximations for the pricing of Asian options with discrete averaging under the Black-Scholes model with time-dependent parameters. The formulae are obtained by using a stochastic Taylor expansion around a log-normal proxy model and are found to be highly accurate in practice.
    
[^2]: 临界无序钉扎测度

    The critical disordered pinning measure

    [https://arxiv.org/abs/2402.17642](https://arxiv.org/abs/2402.17642)

    论文研究了一个由随机游走引起的无序钉扎模型，发现在临界区间内点对点的分区函数收敛到一种唯一的极限随机测度，并命名为临界无序钉扎测度。

    

    在这篇论文中，我们研究了一个由随机游走引起的无序钉扎模型，其增量具有有限的四阶矩和消失的一阶和三阶矩。已知该模型是临界相关的，并且在中等随机性区域中会发生相变。我们展示，在临界区间内，点对点的分区函数收敛到一个唯一的极限随机测度，我们称之为临界无序钉扎测度。我们还为钉扎模型的连续对应物获得了类似的结果，该连续对应物与另外两个模型密切相关：一个是导致粗糙波动率模型的临界随机Volterra方程，另一个是具有乘性噪声的临界随机热方程，其在时间上是白色的，在空间上是δ函数。

    arXiv:2402.17642v1 Announce Type: cross  Abstract: In this paper, we study a disordered pinning model induced by a random walk whose increments have a finite fourth moment and vanishing first and third moments. It is known that this model is marginally relevant, and moreover, it undergoes a phase transition in an intermediate disorder regime. We show that, in the critical window, the point-to-point partition functions converge to a unique limiting random measure, which we call the critical disordered pinning measure. We also obtain an analogous result for a continuous counterpart to the pinning model, which is closely related to two other models: one is a critical stochastic Volterra equation that gives rise to a rough volatility model, and the other is a critical stochastic heat equation with multiplicative noise that is white in time and delta in space.
    
[^3]: 政治讨好和官僚主义影响

    Political Pandering and Bureaucratic Influence

    [https://arxiv.org/abs/2402.17526](https://arxiv.org/abs/2402.17526)

    研究发现在政策实施中，政治家和官僚之间的互动在选举激励下会产生讨好行为，不同类型的官僚对此有不同态度，分析显示官僚主义的影响程度能够最大化选民福利。

    

    本文研究了在选举激励产生讨好行为的环境中，官僚主义对政策实施的影响。通过发展一个两期模型来分析政治家和官僚之间的互动，他们被分类为与选民偏好政策相一致的对齐型或者意图制定有利于精英集团的政策的官僚。研究结果揭示了在对齐的政治家诉诸于讨好的均衡状态，而对齐的官僚则支持或反对这种行为。分析进一步表明，根据参数的不同，官僚主义的任何程度都可能使选民的福利最大化，范围从全能的到无力的官僚体系情景都有。

    arXiv:2402.17526v1 Announce Type: new  Abstract: This paper examines the impact of bureaucracy on policy implementation in environments where electoral incentives generate pandering. A two-period model is developed to analyze the interactions between politicians and bureaucrats, who are categorized as either aligned -- sharing the voters' preferences over policies -- or intent on enacting policies that favor elite groups. The findings reveal equilibria in which aligned politicians resort to pandering, whereas aligned bureaucrats either support or oppose such behavior. The analysis further indicates that, depending on parameters, any level of bureaucratic influence can maximize the voters' welfare, ranging from scenarios with an all-powerful to a toothless bureaucracy.
    
[^4]: 在高维度中带有跟踪误差和权重约束的受限组合分析中导航复杂性

    Navigating Complexity: Constrained Portfolio Analysis in High Dimensions with Tracking Error and Weight Constraints

    [https://arxiv.org/abs/2402.17523](https://arxiv.org/abs/2402.17523)

    本文分析了高维度投资组合中受限条件下的统计特性，展示了在大维度下如何一致估计这些投资组合，同时提供了受限组合权重、风险和夏普比率的收敛速度结果。

    

    本文分析了高维度投资组合中受限制条件下投资组合形成的统计特性。具体来说，我们考虑了具有跟踪误差约束的投资组合、同时带有权重（等式或不等式）限制的投资组合，以及仅带有权重限制的投资组合。跟踪误差是指投资组合表现相对于基准（通常是指数）的度量，而权重约束是指投资组合内资产的具体分配，通常以监管要求或基金章程形式出现。我们展示了即使资产数量多于投资组合的时间跨度，这些投资组合如何可以在大维度下被一致估计。我们还提供了受限组合权重、受限组合风险和受限组合夏普比率的收敛速度结果。

    arXiv:2402.17523v1 Announce Type: new  Abstract: This paper analyzes the statistical properties of constrained portfolio formation in a high dimensional portfolio with a large number of assets. Namely, we consider portfolios with tracking error constraints, portfolios with tracking error jointly with weight (equality or inequality) restrictions, and portfolios with only weight restrictions. Tracking error is the portfolio's performance measured against a benchmark (an index usually), {\color{black}{and weight constraints refers to specific allocation of assets within the portfolio, which often come in the form of regulatory requirement or fund prospectus.}} We show how these portfolios can be estimated consistently in large dimensions, even when the number of assets is larger than the time span of the portfolio. We also provide rate of convergence results for weights of the constrained portfolio, risk of the constrained portfolio and the Sharpe Ratio of the constrained portfolio. To ac
    
[^5]: 限价订单簿模拟：一项综述

    Limit Order Book Simulations: A Review

    [https://arxiv.org/abs/2402.17359](https://arxiv.org/abs/2402.17359)

    本综述研究了当前先进的各种限价订单簿（LOB）模拟模型，在方法学分类的基础上提供了流行风格事实的整体视图，重点研究了模型中的价格冲击现象。

    

    限价订单簿（LOBs）作为买家和卖家在金融市场中相互交互的机制。对LOB进行建模和模拟通常是校准和微调算法交易研究中开发的自动交易策略时的必要步骤。近年来，人工智能革命和更快、更便宜的计算能力的可用性使得建模和模拟变得更加丰富，甚至使用现代人工智能技术。在这项综述中，我们考察了当前最先进的各种LOB模拟模型。我们在方法论基础上对这些模型进行分类，并提供了文献中用于测试模型的流行风格事实的整体视图。此外，我们重点研究模型中价格冲击的存在，因为这是算法交易中一个更为关键的现象之一。最后，我们进行了一项比较研究。

    arXiv:2402.17359v1 Announce Type: new  Abstract: Limit Order Books (LOBs) serve as a mechanism for buyers and sellers to interact with each other in the financial markets. Modelling and simulating LOBs is quite often necessary} for calibrating and fine-tuning the automated trading strategies developed in algorithmic trading research. The recent AI revolution and availability of faster and cheaper compute power has enabled the modelling and simulations to grow richer and even use modern AI techniques. In this review we \highlight{examine} the various kinds of LOB simulation models present in the current state of the art. We provide a classification of the models on the basis of their methodology and provide an aggregate view of the popular stylized facts used in the literature to test the models. We additionally provide a focused study of price impact's presence in the models since it is one of the more crucial phenomena to model in algorithmic trading. Finally, we conduct a comparative
    
[^6]: 在智能金融背景下，用于分析和预测美国股市的随机森林模型

    The Random Forest Model for Analyzing and Forecasting the US Stock Market in the Context of Smart Finance

    [https://arxiv.org/abs/2402.17194](https://arxiv.org/abs/2402.17194)

    本研究评估了结合人工智能的随机森林模型在预测美国股市走势上的预测性能，为投资者提供了决策参考。

    

    股市是金融市场的重要组成部分，在投资者财富积累、上市公司融资成本和国民宏观经济稳定发展方面发挥着关键作用。股市的显著波动可能损害股票投资者的利益，并导致产业结构失衡，从而干扰国民经济的宏观发展。预测股价走势是学术界的热门研究课题。预测股价上涨、横盘和下跌这三种趋势可以帮助投资者做出买入、持有或卖出股票的明智决策。建立有效的预测模型以预测这些趋势具有极其重要的实际意义。本文评估了结合人工智能的随机森林模型在四只股票测试集上的预测性能。

    arXiv:2402.17194v1 Announce Type: new  Abstract: The stock market is a crucial component of the financial market, playing a vital role in wealth accumulation for investors, financing costs for listed companies, and the stable development of the national macroeconomy. Significant fluctuations in the stock market can damage the interests of stock investors and cause an imbalance in the industrial structure, which can interfere with the macro level development of the national economy. The prediction of stock price trends is a popular research topic in academia. Predicting the three trends of stock pricesrising, sideways, and falling can assist investors in making informed decisions about buying, holding, or selling stocks. Establishing an effective forecasting model for predicting these trends is of substantial practical importance. This paper evaluates the predictive performance of random forest models combined with artificial intelligence on a test set of four stocks using optimal param
    
[^7]: 在合并年金基金中优化提取成功

    Withdrawal Success Optimization in a Pooled Annuity Fund

    [https://arxiv.org/abs/2402.17164](https://arxiv.org/abs/2402.17164)

    在合并年金基金中，通过优化投资组合权重函数，最大化特定年金购买者完成规定提取直至死亡的概率，并且得到明显增加的最大概率。

    

    考虑一个投资于n种资产且具有离散时间再平衡的封闭型合并年金基金。在时间0，每位年金购买者向基金做出初始投资，并承诺按照预定的提取时间表提取。要求年金购买者在初始投资和预定提取时间表上是同质的，他们的死亡分布是相同且独立的。在上述设置下，最大化特定年金购买者直至死亡完成规定提取的概率，逐步可度量的组合权重函数。具有两种资产混合的基金组合的应用考虑了标准普尔综合指数和一种通胀保护债券。为初始投资和随后年均提取直至死亡的调整时间表计算了最大概率。通过调整组合权重函数进一步提高了最大概率。

    arXiv:2402.17164v1 Announce Type: new  Abstract: Consider a closed pooled annuity fund investing in n assets with discrete-time rebalancing. At time 0, each annuitant makes an initial contribution to the fund, committing to a predetermined schedule of withdrawals. Require annuitants to be homogeneous in the sense that their initial contributions and predetermined withdrawal schedules are identical, and their mortality distributions are identical and independent. Under the forementioned setup, the probability for a particular annuitant to complete the prescribed withdrawals until death is maximized over progressively measurable portfolio weight functions. Applications consider fund portfolios that mix two assets: the S&P Composite Index and an inflation-protected bond. The maximum probability is computed for annually rebalanced schedules consisting of an initial investment and then equal annual withdrawals until death. A considerable increase in the maximum probability is achieved by in
    
[^8]: 使用张量网络在量子计算机上生成期权定价的时间序列

    Time series generation for option pricing on quantum computers using tensor network

    [https://arxiv.org/abs/2402.17148](https://arxiv.org/abs/2402.17148)

    提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。

    

    金融，特别是期权定价，是一个有望从量子计算中受益的行业。尽管已经提出了用于期权定价的量子算法，但人们希望在算法中设计出更高效的实现方式，其中之一是准备编码基础资产价格概率分布的量子态。特别是在定价依赖路径的期权时，我们需要生成一个编码多个时间点处基础资产价格的联合分布的态，这更具挑战性。为解决这些问题，我们提出了一种使用矩阵乘积态（MPS）作为时间序列生成的生成模型的新方法。为了验证我们的方法，以Heston模型为目标，我们进行数值实验以在模型中生成时间序列。我们的研究结果表明MPS模型能够生成Heston模型中的路径，突显了...

    arXiv:2402.17148v1 Announce Type: cross  Abstract: Finance, especially option pricing, is a promising industrial field that might benefit from quantum computing. While quantum algorithms for option pricing have been proposed, it is desired to devise more efficient implementations of costly operations in the algorithms, one of which is preparing a quantum state that encodes a probability distribution of the underlying asset price. In particular, in pricing a path-dependent option, we need to generate a state encoding a joint distribution of the underlying asset price at multiple time points, which is more demanding. To address these issues, we propose a novel approach using Matrix Product State (MPS) as a generative model for time series generation. To validate our approach, taking the Heston model as a target, we conduct numerical experiments to generate time series in the model. Our findings demonstrate the capability of the MPS model to generate paths in the Heston model, highlightin
    
[^9]: 两因素不确定波动率模型的单调分段常数控制积分方法

    A monotone piecewise constant control integration approach for the two-factor uncertain volatility model

    [https://arxiv.org/abs/2402.06840](https://arxiv.org/abs/2402.06840)

    这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。

    

    在不确定波动率模型中，两种资产期权合约的价格满足具有交叉导数项的二维Hamilton-Jacobi-Bellman（HJB）偏微分方程（PDE）。传统方法主要涉及有限差分和策略迭代。本文提出了一种新颖且更简化的“分解和积分，然后优化”的方法来解决上述的HJB PDE。在每个时间步内，我们的策略采用分段常数控制，将HJB PDE分解为独立的线性二维PDE。利用已知的与这些PDE相关的Green函数的Fourier变换的闭式表达式，我们确定了这些函数的显式公式。由于Green函数是非负的，将PDE转化为二维卷积积分的解可以b

    Prices of option contracts on two assets within uncertain volatility models for worst and best-case scenarios satisfy a two-dimensional Hamilton-Jacobi-Bellman (HJB) partial differential equation (PDE) with cross derivatives terms. Traditional methods mainly involve finite differences and policy iteration. This "discretize, then optimize" paradigm requires complex rotations of computational stencils for monotonicity.   This paper presents a novel and more streamlined "decompose and integrate, then optimize" approach to tackle the aforementioned HJB PDE. Within each timestep, our strategy employs a piecewise constant control, breaking down the HJB PDE into independent linear two-dimensional PDEs. Using known closed-form expressions for the Fourier transforms of the Green's functions associated with these PDEs, we determine an explicit formula for these functions. Since the Green's functions are non-negative, the solutions to the PDEs, cast as two-dimensional convolution integrals, can b
    
[^10]: 使用可屏蔽股票表示的强化学习在可定制股票池中进行投资组合管理

    Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools

    [https://arxiv.org/abs/2311.10801](https://arxiv.org/abs/2311.10801)

    使用EarnMore方法，我们提出了一种新的RL方法，可以允许RL代理与可定制股票池（CSPs）交互，而不需要重新训练。

    

    投资组合管理（PM）是一项基本的金融交易任务，探索定期将资金重新配置到不同股票中以追求长期利润。最近，强化学习（RL）显示出其潜力，通过与金融市场互动来训练具有盈利能力的PM代理。但是，现有工作主要集中在固定股票池上，这与投资者的实际需求不一致。为应对这一挑战，我们提出EarnMore，一种新的RL方法，可以允许RL代理与可定制股票池（CSPs）交互，而不需要重新训练。

    arXiv:2311.10801v3 Announce Type: replace-cross  Abstract: Portfolio management (PM) is a fundamental financial trading task, which explores the optimal periodical reallocation of capitals into different stocks to pursue long-term profits. Reinforcement learning (RL) has recently shown its potential to train profitable agents for PM through interacting with financial markets. However, existing work mostly focuses on fixed stock pools, which is inconsistent with investors' practical demand. Specifically, the target stock pool of different investors varies dramatically due to their discrepancy on market states and individual investors may temporally adjust stocks they desire to trade (e.g., adding one popular stocks), which lead to customizable stock pools (CSPs). Existing RL methods require to retrain RL agents even with a tiny change of the stock pool, which leads to high computational cost and unstable performance. To tackle this challenge, we propose EarnMore, a rEinforcement leARNin
    
[^11]: 使用机器学习和粗糙波动率预测波动：来自加密寒冬的案例

    Forecasting Volatility with Machine Learning and Rough Volatility: Example from the Crypto-Winter

    [https://arxiv.org/abs/2311.04727](https://arxiv.org/abs/2311.04727)

    通过结合LSTM和粗糙波动率，我们发现波动率预测模型在加密货币领域的表现优于传统模型，同时从简约的参数模型中得到类似预测表现，进一步证明了波动率形成过程机制的普遍性。

    

    我们扩展了一种最近引入的波动率预测框架的应用并测试其性能，该框架包括LSTM和粗糙波动率。我们感兴趣的资产类别是加密货币，在2022年的“加密寒冬”初期。我们首先展示了，为了预测波动率，一个基于资产池训练的通用LSTM方法优于传统模型。然后，我们考虑基于粗糙波动率和Zumbach效应的简约参数模型。我们得到类似的预测表现，仅使用五个参数，其值不依赖于资产。我们的发现进一步证实了波动率形成过程背后机制的普遍性。

    arXiv:2311.04727v2 Announce Type: replace  Abstract: We extend the application and test the performance of a recently introduced volatility prediction framework encompassing LSTM and rough volatility. Our asset class of interest is cryptocurrencies, at the beginning of the "crypto-winter" in 2022. We first show that to forecast volatility, a universal LSTM approach trained on a pool of assets outperforms traditional models. We then consider a parsimonious parametric model based on rough volatility and Zumbach effect. We obtain similar prediction performances with only five parameters whose values are non-asset-dependent. Our findings provide further evidence on the universality of the mechanisms underlying the volatility formation process.
    
[^12]: 使用Liquid Staking Derivatives (LSDs)进行杠杆质押: 机会与风险

    Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks. (arXiv:2401.08610v1 [q-fin.GN])

    [http://arxiv.org/abs/2401.08610](http://arxiv.org/abs/2401.08610)

    这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。

    

    Lido是以太坊上最主要的Liquid Staking Derivative (LSD)提供商，允许用户抵押任意数量的ETH来获得stETH，这可以与DeFi协议如Aave进行整合。Lido与Aave之间的互通性使得一种新型策略“杠杆质押”得以实现，用户在Lido上质押ETH获取stETH，将stETH作为Aave上的抵押品借入ETH，然后将借入的ETH重新投入Lido。用户可以迭代执行此过程，根据自己的风险偏好来优化潜在回报。本文系统地研究了杠杆质押所涉及的机会和风险。我们是第一个在Lido-Aave生态系统中对杠杆质押策略进行形式化的研究。我们的经验研究发现，在以太坊上有262个杠杆质押头寸，总质押金额为295,243 ETH（482M USD）。我们发现，90.13%的杠杆质押头寸实现了比传统质押更高的回报。

    Lido, the leading Liquid Staking Derivative (LSD) provider on Ethereum, allows users to stake an arbitrary amount of ETH to receive stETH, which can be integrated with Decentralized Finance (DeFi) protocols such as Aave. The composability between Lido and Aave enables a novel strategy called "leverage staking", where users stake ETH on Lido to acquire stETH, utilize stETH as collateral on Aave to borrow ETH, and then restake the borrowed ETH on Lido. Users can iteratively execute this process to optimize potential returns based on their risk profile.  This paper systematically studies the opportunities and risks associated with leverage staking. We are the first to formalize the leverage staking strategy within the Lido-Aave ecosystem. Our empirical study identifies 262 leverage staking positions on Ethereum, with an aggregated staking amount of 295,243 ETH (482M USD). We discover that 90.13% of leverage staking positions have achieved higher returns than conventional staking. Furtherm
    
[^13]: 在模型不确定性下的离散时间最优投资

    Discrete time optimal investment under model uncertainty. (arXiv:2307.11919v1 [q-fin.MF])

    [http://arxiv.org/abs/2307.11919](http://arxiv.org/abs/2307.11919)

    本文研究了在模型不确定性下的离散时间最优投资问题，利用原始方法证明了存在一个最优投资策略，并引入了类型(A)的效用函数。

    

    我们研究了一个在一般离散时间无摩擦市场中的鲁棒效用最大化问题，其中假设投资者对整个实数线上的随机凹效用函数进行定义。她还面临对市场的信念的模型不确定性，该不确定性通过一组先验模型进行建模。我们证明了使用原始方法仅需要存在一个最优投资策略。为此，我们假设市场和随机效用函数具有经典假设，具有渐进弹性约束。我们的其他大多数假设是基于逐前假设的，并对市场无不确定性的文献中的普遍接受假设相对应。我们还引入了类型(A)的效用函数，其包括具有基准的效用函数，我们的假设很容易验证。

    We study a robust utility maximization problem in a general discrete-time frictionless market under quasi-sure no-arbitrage. The investor is assumed to have a random and concave utility function defined on the whole real-line. She also faces model ambiguity on her beliefs about the market, which is modeled through a set of priors. We prove the existence of an optimal investment strategy using only primal methods. For that we assume classical assumptions on the market and on the random utility function as asymptotic elasticity constraints. Most of our other assumptions are stated on a prior-by-prior basis and correspond to generally accepted assumptions in the literature on markets without ambiguity. We also introduce utility functions of type (A), which include utility functions with benchmark and for which our assumptions are easily checked.
    
[^14]: 量子神经网络和张量网络在截面股票收益预测中的应用

    The cross-sectional stock return predictions via quantum neural network and tensor network. (arXiv:2304.12501v1 [cs.LG])

    [http://arxiv.org/abs/2304.12501](http://arxiv.org/abs/2304.12501)

    本文研究将量子神经网络和张量网络应用于股票收益预测，在日本股市中张量网络模型表现优于传统模型，并在最新市场环境下呈现出卓越表现。

    

    本文研究了利用量子和量子启发式的机器学习算法进行股票收益预测的应用。其中，我们将量子神经网络（一种适用于噪声中等规模量子计算机的算法）和张量网络（一种受量子启发的机器学习算法）的性能与传统模型如线性回归和神经网络进行比较。通过构建基于模型预测的投资组合并测量投资绩效，我们发现在日本股市中，张量网络模型表现优于传统基准模型（包括线性和神经网络模型）。虽然量子神经网络模型在整个周期内具有降低风险调整超额收益的能力，但最新的市场环境下，量子神经网络和张量网络模型均表现出卓越的性能。

    In this paper we investigate the application of quantum and quantum-inspired machine learning algorithms to stock return predictions. Specifically, we evaluate performance of quantum neural network, an algorithm suited for noisy intermediate-scale quantum computers, and tensor network, a quantum-inspired machine learning algorithm, against classical models such as linear regression and neural networks. To evaluate their abilities, we construct portfolios based on their predictions and measure investment performances. The empirical study on the Japanese stock market shows the tensor network model achieves superior performance compared to classical benchmark models, including linear and neural network models. Though the quantum neural network model attains the lowered risk-adjusted excess return than the classical neural network models over the whole period, both the quantum neural network and tensor network models have superior performances in the latest market environment, which sugges
    
[^15]: 环境政策采用中的不确定性：对不可预测的社会经济成本的贝叶斯学习(arXiv:2304.10344v1 [math.OC])

    Uncertainty over Uncertainty in Environmental Policy Adoption: Bayesian Learning of Unpredictable Socioeconomic Costs. (arXiv:2304.10344v1 [math.OC])

    [http://arxiv.org/abs/2304.10344](http://arxiv.org/abs/2304.10344)

    本文开发了一个类似于实物期权的模型，以捕捉环境政策采用中的两层不确定性。决策者能够通过跟踪成本的实际演变来学习未知的漂移，并形成后验动态信念其真正价值。

    

    污染的社会经济影响自然而然地伴随着不确定性，例如，排放减少的新技术发展或人口变化。此外，环境破坏未来成本的趋势是未知的：全球变暖是否占主导地位，还是技术进步会占据主导地位？事实上，我们不知道哪种情况会实现，科学界的辩论仍然存在。本文通过开发一个类似于实物期权的模型，捕捉这两层不确定性。在该模型中，决策者的目标是在当前的排放率中采取一次性昂贵的减少，当污染的社会经济成本的随机动态受到布朗运动的冲击时，漂移是不可观察的随机变量。通过跟踪成本的实际演变，决策者能够学习未知的漂移，并形成后验动态信念其真正价值。由此产生的决策时机问题公司

    The socioeconomic impact of pollution naturally comes with uncertainty due to, e.g., current new technological developments in emissions' abatement or demographic changes. On top of that, the trend of the future costs of the environmental damage is unknown: Will global warming dominate or technological advancements prevail? The truth is that we do not know which scenario will be realised and the scientific debate is still open. This paper captures those two layers of uncertainty by developing a real-options-like model in which a decision maker aims at adopting a once-and-for-all costly reduction in the current emissions rate, when the stochastic dynamics of the socioeconomic costs of pollution are subject to Brownian shocks and the drift is an unobservable random variable. By keeping track of the actual evolution of the costs, the decision maker is able to learn the unknown drift and to form a posterior dynamic belief of its true value. The resulting decision maker's timing problem boi
    

