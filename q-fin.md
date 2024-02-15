# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extended mean-field games with multi-dimensional singular controls and non-linear jump impact](https://arxiv.org/abs/2402.09317) | 这篇论文提出了一个概率框架，用于分析具有多维奇异控制和状态相关跳跃动力学和成本的扩展均场博弈。通过限制可接受的奇异控制的集合为可由连续控制逼近的控制，解决了状态动力学不连续和奖励函数不下半连续的问题。同时引入了一类参数化的均场博弈来处理连续性缺失问题。 |
| [^2] | [Exact simulation scheme for the Ornstein-Uhlenbeck driven stochastic volatility model with the Karhunen-Lo\`eve expansions](https://arxiv.org/abs/2402.09243) | 本研究提出了一种新的精确模拟方案，可以更快地模拟Ornstein-Uhlenbeck驱动的随机波动率模型。使用Karhunen-Loève展开来表示波动率路径，并通过解析推导的方式获得了波动率和方差的时间积分。通过采用条件蒙特卡洛方法和保持鞅性的控制变量来进一步改进模拟算法。这一方法比现有方法更快速且更高效。 |
| [^3] | [The Boosted Difference of Convex Functions Algorithm for Value-at-Risk Constrained Portfolio Optimization](https://arxiv.org/abs/2402.09194) | 本文提出了一种基于凸函数差分和线搜索的算法（BDCA），用于解决线性优化问题，并且证明了其收敛性和其他特性。 |
| [^4] | [Optimal Automated Market Makers: Differentiable Economics and Strong Duality](https://arxiv.org/abs/2402.09129) | 本文研究了在多商品市场中，包括复杂捆绑行为的情况下，找到最优的自动做市商的问题，发现该问题对偶于一个最优运输问题，并且具有特定的几何约束条件。 |
| [^5] | [Database for the meta-analysis of the social cost of carbon (v2024.0)](https://arxiv.org/abs/2402.09125) | 该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。 |
| [^6] | [Time preference, wealth and utility inequality: A microeconomic interaction and dynamic macroeconomic model connection approach](https://arxiv.org/abs/2402.08905) | 本研究通过将个体之间的互动与社会规范参考相连接，揭示了时间偏好对财富分配和不平等的影响。通过模型计算和实证分析，结果显示，个体之间的消费比较对财富不平等和效用产生显著影响。社会规范在一定程度上可以减缓这种影响。这个方法提供了一个新的框架，用于理解时间偏好在经济不平等中的作用。 |
| [^7] | [LLM-driven Imitation of Subrational Behavior : Illusion or Reality?](https://arxiv.org/abs/2402.08755) | 本文探讨了使用LLMs生成合成人类演示的方法，然后通过模仿学习来学习亚理性代理策略，从而模拟亚理性行为，并为我们理解人类行为提供了改进的可能性。 |
| [^8] | [Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior](https://arxiv.org/abs/2402.03953) | 本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。 |
| [^9] | [Well Posedness of Utility Maximization Problems Under Partial Information in a Market with Gaussian Drift](https://arxiv.org/abs/2205.08614) | 本文研究了金融市场中的效用最大化问题的良定义性，为具有高斯漂移的部分信息市场提供了模型参数的充分条件。 |
| [^10] | [A generalization of the rational rough Heston approximation.](http://arxiv.org/abs/2310.09181) | 这个论文将理性逼近从粗Heston分数ODE扩展到Mittag-Leffler核情况，并提供了数值证据表明解的收敛性。 |
| [^11] | [Commitment and the Dynamics of Household Labor Supply.](http://arxiv.org/abs/2307.10983) | 本文发展了一个家庭生命周期集体模型，研究了三种承诺类型的家庭行为，并通过测试方法确定了家庭承诺程度的异质性。使用最新数据，研究发现有限承诺是最普遍的类型。 |
| [^12] | [Reinforcement learning for optimization of energy trading strategy.](http://arxiv.org/abs/2303.16266) | 本文使用强化学习算法优化了一种黑盒交易策略，该策略通过在马尔可夫决策过程中使用真实数据进行优化，在 DA 能源市场上由中型生产者自动进行交易。 |

# 详细

[^1]: 具有多维奇异控制和非线性跳跃影响的扩展均场博弈

    Extended mean-field games with multi-dimensional singular controls and non-linear jump impact

    [https://arxiv.org/abs/2402.09317](https://arxiv.org/abs/2402.09317)

    这篇论文提出了一个概率框架，用于分析具有多维奇异控制和状态相关跳跃动力学和成本的扩展均场博弈。通过限制可接受的奇异控制的集合为可由连续控制逼近的控制，解决了状态动力学不连续和奖励函数不下半连续的问题。同时引入了一类参数化的均场博弈来处理连续性缺失问题。

    

    我们建立了一个概率框架，用于分析具有多维奇异控制和状态相关跳跃动力学和成本的扩展均场博弈。在分析这样的博弈时，出现了两个关键挑战：状态动力学可能不连续地依赖于控制，奖励函数可能不是下半连续的。这些问题可以通过限制可接受的奇异控制的集合为可由连续控制逼近的控制来克服。我们证明了相应的可接受弱控制的集合由Marcus型随机微分方程的弱解给出，并给出了奖励函数的明确表征。奖励函数通常只是下半连续的。为了解决连续性缺失问题，我们引入了一类新颖的具有更广泛的可接受控制集合的均场博弈，称为参数化的均场博弈。参数化是状态/控制过程的连续插值跳跃的法则。我们证明了用参数化表示的均场博弈具有所需的性质。

    arXiv:2402.09317v1 Announce Type: cross Abstract: We establish a probabilistic framework for analysing extended mean-field games with multi-dimensional singular controls and state-dependent jump dynamics and costs. Two key challenges arise when analysing such games: the state dynamics may not depend continuously on the control and the reward function may not be u.s.c.~Both problems can be overcome by restricting the set of admissible singular controls to controls that can be approximated by continuous ones. We prove that the corresponding set of admissible weak controls is given by the weak solutions to a Marcus-type SDE and provide an explicit characterisation of the reward function. The reward function will in general only be u.s.c.~To address the lack of continuity we introduce a novel class of MFGs with a broader set of admissible controls, called MFGs of parametrisations. Parametrisations are laws of state/control processes that continuously interpolate jumps. We prove that the re
    
[^2]: Ornstein-Uhlenbeck驱动的随机波动率模型的精确模拟方案与Karhunen-Loève展开

    Exact simulation scheme for the Ornstein-Uhlenbeck driven stochastic volatility model with the Karhunen-Lo\`eve expansions

    [https://arxiv.org/abs/2402.09243](https://arxiv.org/abs/2402.09243)

    本研究提出了一种新的精确模拟方案，可以更快地模拟Ornstein-Uhlenbeck驱动的随机波动率模型。使用Karhunen-Loève展开来表示波动率路径，并通过解析推导的方式获得了波动率和方差的时间积分。通过采用条件蒙特卡洛方法和保持鞅性的控制变量来进一步改进模拟算法。这一方法比现有方法更快速且更高效。

    

    本研究提出了一种新的Ornstein-Uhlenbeck驱动的随机波动率模型的精确模拟方案。利用Karhunen-Loève展开，将遵循Ornstein-Uhlenbeck过程的随机波动率路径表示为正弦级数，并将波动率和方差的时间积分解析地推导为独立正态随机变量的和。这种新方法比依赖于计算昂贵的数值变换反演的Li和Wu [Eur. J. Oper. Res., 2019, 275(2), 768-779] 方法快几百倍。进一步采用了条件蒙特卡洛方法和保持鞅性的控制变量对实时价格进行模拟算法改进。

    arXiv:2402.09243v1 Announce Type: new Abstract: This study proposes a new exact simulation scheme of the Ornstein-Uhlenbeck driven stochastic volatility model. With the Karhunen-Lo\`eve expansions, the stochastic volatility path following the Ornstein-Uhlenbeck process is expressed as a sine series, and the time integrals of volatility and variance are analytically derived as the sums of independent normal random variates. The new method is several hundred times faster than Li and Wu [Eur. J. Oper. Res., 2019, 275(2), 768-779] that relies on computationally expensive numerical transform inversion. The simulation algorithm is further improved with the conditional Monte-Carlo method and the martingale-preserving control variate on the spot price.
    
[^3]: 基于凸函数差分算法的价值风险约束组合优化

    The Boosted Difference of Convex Functions Algorithm for Value-at-Risk Constrained Portfolio Optimization

    [https://arxiv.org/abs/2402.09194](https://arxiv.org/abs/2402.09194)

    本文提出了一种基于凸函数差分和线搜索的算法（BDCA），用于解决线性优化问题，并且证明了其收敛性和其他特性。

    

    现代金融中一个非常相关的问题是设计价值风险（VaR）最优组合。由于当代金融监管的要求，银行和其他金融机构需要使用风险度量来控制他们的信用风险、市场风险和运营风险。对于具有离散收益分布和有限个情景的组合，可以推导出凸函数差分（DC）函数表示的VaR。Wozabal（2012）证明了这可以通过使用凸函数差分算法（DCA）来解决VaR约束的Markowitz风格组合选择问题。最近的算法扩展是Boosted Difference of Convex Functions Algorithm（BDCA），它通过额外的线搜索步骤加速收敛。已经证明BDCA对于解决具有线性不等式约束的非光滑二次问题具有线性收敛性。本文证明了该算法在线性优化问题中的收敛性以及其它一些性质。

    arXiv:2402.09194v1 Announce Type: cross Abstract: A highly relevant problem of modern finance is the design of Value-at-Risk (VaR) optimal portfolios. Due to contemporary financial regulations, banks and other financial institutions are tied to use the risk measure to control their credit, market and operational risks. For a portfolio with a discrete return distribution and finitely many scenarios, a Difference of Convex (DC) functions representation of the VaR can be derived. Wozabal (2012) showed that this yields a solution to a VaR constrained Markowitz style portfolio selection problem using the Difference of Convex Functions Algorithm (DCA). A recent algorithmic extension is the so-called Boosted Difference of Convex Functions Algorithm (BDCA) which accelerates the convergence due to an additional line search step. It has been shown that the BDCA converges linearly for solving non-smooth quadratic problems with linear inequality constraints. In this paper, we prove that the linear
    
[^4]: 最优自动做市商: 可微经济学和强对偶性

    Optimal Automated Market Makers: Differentiable Economics and Strong Duality

    [https://arxiv.org/abs/2402.09129](https://arxiv.org/abs/2402.09129)

    本文研究了在多商品市场中，包括复杂捆绑行为的情况下，找到最优的自动做市商的问题，发现该问题对偶于一个最优运输问题，并且具有特定的几何约束条件。

    

    做市商的作用是同时以指定价格购买和出售商品数量，通常是金融资产如股票。自动做市商（AMM）是一种根据预定的时间表提供交易的机制；选择最佳的时间表取决于做市商的目标。现有研究主要集中在预测市场上，目的是信息收集。近期的工作则主要关注利润最大化的目标，但仅仅考虑了一种类型的商品（用衡量货币进行交易），包括逆向选择的情况。关于存在多种商品以及可能出现复杂捆绑行为的最优做市问题尚不清楚。本文表明，在多个商品存在且可能出现复杂捆绑行为的情况下，找到一个最优的做市商是一个对偶于最优运输问题的问题，并且具有特定的几何约束条件。

    arXiv:2402.09129v1 Announce Type: cross Abstract: The role of a market maker is to simultaneously offer to buy and sell quantities of goods, often a financial asset such as a share, at specified prices. An automated market maker (AMM) is a mechanism that offers to trade according to some predetermined schedule; the best choice of this schedule depends on the market maker's goals. The literature on the design of AMMs has mainly focused on prediction markets with the goal of information elicitation. More recent work motivated by DeFi has focused instead on the goal of profit maximization, but considering only a single type of good (traded with a numeraire), including under adverse selection (Milionis et al. 2022). Optimal market making in the presence of multiple goods, including the possibility of complex bundling behavior, is not well understood. In this paper, we show that finding an optimal market maker is dual to an optimal transport problem, with specific geometric constraints on t
    
[^5]: 社会碳成本的元分析数据库 (v2024.0)

    Database for the meta-analysis of the social cost of carbon (v2024.0)

    [https://arxiv.org/abs/2402.09125](https://arxiv.org/abs/2402.09125)

    该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。

    

    本文介绍了社会碳成本估计元分析数据库的新版本。新增了记录，并添加了关于气候变化影响和福利函数形状的新字段。该数据库还扩展了合作者和引用网络。

    arXiv:2402.09125v1 Announce Type: new Abstract: A new version of the database for the meta-analysis of estimates of the social cost of carbon is presented. New records were added, and new fields on the impact of climate change and the shape of the welfare function. The database was extended to co-author and citation networks.
    
[^6]: 时间偏好、财富和效用不平等：一种微观经济互动和动态宏观经济模型连接方法

    Time preference, wealth and utility inequality: A microeconomic interaction and dynamic macroeconomic model connection approach

    [https://arxiv.org/abs/2402.08905](https://arxiv.org/abs/2402.08905)

    本研究通过将个体之间的互动与社会规范参考相连接，揭示了时间偏好对财富分配和不平等的影响。通过模型计算和实证分析，结果显示，个体之间的消费比较对财富不平等和效用产生显著影响。社会规范在一定程度上可以减缓这种影响。这个方法提供了一个新的框架，用于理解时间偏好在经济不平等中的作用。

    

    基于个体与他人的互动与对社会规范的参考，本研究揭示了时间偏好的异质性对财富分配和不平等的影响。我们提出了一种新颖的方法，将产生异质性的微观经济主体之间的互动与宏观经济模型中的资本和消费动态方程相连接。利用这种方法，我们估计了微观经济互动引起的折现率变化对资本、消费和效用以及不平等程度的影响。结果显示，与他人的消费比较显著影响资本，即财富不平等。此外，对效用的影响从未微小，并且社会规范可以减少这种影响。我们的支持证据显示，不平等计算的定量结果与队列和跨文化研究的调查数据相符。本研究的微观-宏观连接方法为理解时间偏好对经济不平等的影响提供了新的框架。

    arXiv:2402.08905v1 Announce Type: new Abstract: Based on interactions between individuals and others and references to social norms, this study reveals the impact of heterogeneity in time preference on wealth distribution and inequality. We present a novel approach that connects the interactions between microeconomic agents that generate heterogeneity to the dynamic equations for capital and consumption in macroeconomic models. Using this approach, we estimate the impact of changes in the discount rate due to microeconomic interactions on capital, consumption and utility and the degree of inequality. The results show that intercomparisons with others regarding consumption significantly affect capital, i.e. wealth inequality. Furthermore, the impact on utility is never small and social norms can reduce this impact. Our supporting evidence shows that the quantitative results of inequality calculations correspond to survey data from cohort and cross-cultural studies. This study's micro-ma
    
[^7]: LLM驱动的模拟亚理性行为：幻觉还是现实？

    LLM-driven Imitation of Subrational Behavior : Illusion or Reality?

    [https://arxiv.org/abs/2402.08755](https://arxiv.org/abs/2402.08755)

    本文探讨了使用LLMs生成合成人类演示的方法，然后通过模仿学习来学习亚理性代理策略，从而模拟亚理性行为，并为我们理解人类行为提供了改进的可能性。

    

    建模亚理性代理，如人类或经济家庭，由于校准强化学习模型的困难或收集涉及人类主体的数据的难度而具有挑战性。现有研究强调了大型语言模型（LLMs）解决复杂推理任务和模仿人类交流的能力，而使用LLMs作为代理进行模拟显示出出现的社交行为，可能提高我们对人类行为的理解。在本文中，我们提议研究使用LLMs生成合成的人类演示，然后通过模仿学习来学习亚理性代理策略。我们假设LLMs可以用作人类的隐式计算模型，并提出了一个框架，使用从LLMs派生的合成演示来建模人类特有的亚理性行为（例如，目光短浅的行为或对风险规避的偏好）。我们进行了实验证明:

    arXiv:2402.08755v1 Announce Type: new Abstract: Modeling subrational agents, such as humans or economic households, is inherently challenging due to the difficulty in calibrating reinforcement learning models or collecting data that involves human subjects. Existing work highlights the ability of Large Language Models (LLMs) to address complex reasoning tasks and mimic human communication, while simulation using LLMs as agents shows emergent social behaviors, potentially improving our comprehension of human conduct. In this paper, we propose to investigate the use of LLMs to generate synthetic human demonstrations, which are then used to learn subrational agent policies though Imitation Learning. We make an assumption that LLMs can be used as implicit computational models of humans, and propose a framework to use synthetic demonstrations derived from LLMs to model subrational behaviors that are characteristic of humans (e.g., myopic behavior or preference for risk aversion). We experim
    
[^8]: 中心化和去中心化交易所中的永续期货合约: 机制和交易者行为

    Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior

    [https://arxiv.org/abs/2402.03953](https://arxiv.org/abs/2402.03953)

    本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。

    

    本研究提出了一个具有开创性的知识系统化(SoK)计划，重点深入探索交易者在中心化交易所(CEXs)和去中心化交易所(DEXs)中关于永续期货合约的动态和行为。我们改进了现有模型，以研究交易者对价格波动的反应，创建了一个针对这些合约平台的新的分析框架，同时突出了区块链技术在其应用中的作用。我们的研究包括对CEXs的历史数据的比较分析，以及对DEXs上的完整交易数据的更详尽的研究。在虚拟自动化市场做市商(VAMM)模型的DEX上，多头和空头持仓量对价格波动产生相反的影响，这归因于VAMM的价格形成机制。在采用预言机定价模型的DEX中，我们观察到买家和卖家之间交易者行为上存在明显的不对称性。

    This study presents a groundbreaking Systematization of Knowledge (SoK) initiative, focusing on an in-depth exploration of the dynamics and behavior of traders on perpetual future contracts across both centralized exchanges (CEXs), and decentralized exchanges (DEXs). We have refined the existing model for investigating traders' behavior in reaction to price volatility to create a new analytical framework specifically for these contract platforms, while also highlighting the role of blockchain technology in their application. Our research includes a comparative analysis of historical data from CEXs and a more extensive examination of complete transactional data on DEXs. On DEX of Virtual Automated Market Making (VAMM) Model, open interest on short and long positions exert effect on price volatility in opposite direction, attributable to VAMM's price formation mechanism. In the DEXs with Oracle Pricing Model, we observed a distinct asymmetry in trader behavior between buyers and sellers.
    
[^9]: 在具有高斯漂移的部分信息市场中的效用最大化问题的良定义性研究

    Well Posedness of Utility Maximization Problems Under Partial Information in a Market with Gaussian Drift

    [https://arxiv.org/abs/2205.08614](https://arxiv.org/abs/2205.08614)

    本文研究了金融市场中的效用最大化问题的良定义性，为具有高斯漂移的部分信息市场提供了模型参数的充分条件。

    

    本文研究了金融市场中的效用最大化问题的良定义性，其中股票回报依赖于隐藏的高斯均值回归漂移过程。由于该过程可能是无界的，对于不受上界限制的效用函数，无法保证其良定义性。对于相对风险厌恶小于对数效用的功率效用函数，这导致了对模型参数的选择限制，例如投资周期和控制资产价格和漂移过程方差的参数。我们推导出了模型参数的充分条件，以实现具有完全和部分信息的模型的终端财富的有界最大预期效用。

    arXiv:2205.08614v2 Announce Type: replace Abstract: This paper investigates well posedness of utility maximization problems for financial markets where stock returns depend on a hidden Gaussian mean reverting drift process. Since that process is potentially unbounded, well posedness cannot be guaranteed for utility functions which are not bounded from above. For power utility with relative risk aversion smaller than those of log-utility this leads to restrictions on the choice of model parameters such as the investment horizon and parameters controlling the variance of the asset price and drift processes. We derive sufficient conditions to the model parameters leading to bounded maximum expected utility of terminal wealth for models with full and partial information.
    
[^10]: 理性粗Heston逼近的一般化

    A generalization of the rational rough Heston approximation. (arXiv:2310.09181v1 [q-fin.CP])

    [http://arxiv.org/abs/2310.09181](http://arxiv.org/abs/2310.09181)

    这个论文将理性逼近从粗Heston分数ODE扩展到Mittag-Leffler核情况，并提供了数值证据表明解的收敛性。

    

    我们将理性逼近从[GR19]中的粗Heston分数ODE扩展到Mittag-Leffler核情况。我们提供了数值证据表明解的收敛性。

    We extend the rational approximation of the solution of the rough Heston fractional ODE in [GR19] to the case of the Mittag-Leffler kernel. We provide numerical evidence of the convergence of the solution.
    
[^11]: 承诺与家庭劳动力供给的动态。 (arXiv:2307.10983v1 [econ.GN])

    Commitment and the Dynamics of Household Labor Supply. (arXiv:2307.10983v1 [econ.GN])

    [http://arxiv.org/abs/2307.10983](http://arxiv.org/abs/2307.10983)

    本文发展了一个家庭生命周期集体模型，研究了三种承诺类型的家庭行为，并通过测试方法确定了家庭承诺程度的异质性。使用最新数据，研究发现有限承诺是最普遍的类型。

    

    个体对伴侣的承诺程度对社会有重要影响。本文通过发展一个家庭的生命周期集体模型，描述了三种不同类型的承诺：全面承诺、有限承诺和无承诺。我们提出了一种测试方法，通过观察当下和历史消息对家庭行为的影响来区分这三种类型。我们的测试允许家庭之间承诺程度的异质性。使用最近的“收入动态面板研究”数据，我们拒绝了全面承诺和无承诺，同时找到了有限承诺的强有力证据。

    The extent to which individuals commit to their partner for life has important implications. This paper develops a lifecycle collective model of the household, through which it characterizes behavior in three prominent alternative types of commitment: full, limited, and no commitment. We propose a test that distinguishes between all three types based on how contemporaneous and historical news affect household behavior. Our test permits heterogeneity in the degree of commitment across households. Using recent data from the Panel Study of Income Dynamics, we reject full and no commitment, while we find strong evidence for limited commitment.
    
[^12]: 强化学习用于能源交易策略的优化

    Reinforcement learning for optimization of energy trading strategy. (arXiv:2303.16266v1 [cs.LG])

    [http://arxiv.org/abs/2303.16266](http://arxiv.org/abs/2303.16266)

    本文使用强化学习算法优化了一种黑盒交易策略，该策略通过在马尔可夫决策过程中使用真实数据进行优化，在 DA 能源市场上由中型生产者自动进行交易。

    

    越来越多的能源来自大量小型生产者的可再生能源，这些来源的效率是不稳定的，在某种程度上也是随机的，加剧了能源市场平衡问题。在许多国家，这种平衡是在预测日（DA）能源市场上完成的。本文考虑由中型生产者在DA能源市场上的自动化交易。我们将此活动建模为马尔可夫决策过程，并规范了一个框架，其中可以使用现实数据优化即用策略。我们合成参数化交易策略，并使用进化算法优化它们。我们还使用最先进的强化学习算法优化一个黑盒交易策略，该策略利用来自环境的可用信息来影响未来价格。

    An increasing part of energy is produced from renewable sources by a large number of small producers. The efficiency of these sources is volatile and, to some extent, random, exacerbating the energy market balance problem. In many countries, that balancing is performed on day-ahead (DA) energy markets. In this paper, we consider automated trading on a DA energy market by a medium size prosumer. We model this activity as a Markov Decision Process and formalize a framework in which a ready-to-use strategy can be optimized with real-life data. We synthesize parametric trading strategies and optimize them with an evolutionary algorithm. We also use state-of-the-art reinforcement learning algorithms to optimize a black-box trading strategy fed with available information from the environment that can impact future prices.
    

