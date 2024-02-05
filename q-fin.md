# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning the Market: Sentiment-Based Ensemble Trading Agents](https://rss.arxiv.org/abs/2402.01441) | 该论文提出了将情感分析和深度强化学习集成算法应用于股票交易的方法，设计了可以根据市场情绪动态调整的交易策略。实验结果表明，这种方法比传统的集成策略、单一智能体算法和市场指标更具盈利性、稳健性和风险最小化。相关研究还发现，传统的固定更换集成智能体的做法并不是最优的，而基于情感的动态框架可以显著提高交易智能体的性能。 |
| [^2] | [Forecasting Volatility of Oil-based Commodities: The Model of Dynamic Persistence](https://rss.arxiv.org/abs/2402.01354) | 该论文提出了一种新的方法来预测基于石油的商品的波动性，通过将异质持续性的冲击在时间上平滑变化，这种模型在改进波动性预测方面表现出色，并且特别适用于较长时间段的预测。 |
| [^3] | [Convergence rates for Backward SDEs driven by L\'evy processes](https://rss.arxiv.org/abs/2402.01337) | 这篇论文研究了由复合Poisson过程逼近的Lévy过程驱动的BSDEs的收敛速度，并推导出了在L^2范数和Wasserstein距离下的最优收敛速度。 |
| [^4] | [A simple method for joint evaluation of skill in directional forecasts of multiple variables](https://rss.arxiv.org/abs/2402.01142) | 本文介绍了一种用于联合评估多变量方向预测技能的简单方法，并利用泰国三个组织在2001年至2021年间的GDP增长和通胀预测数据，展示了该方法的应用。 |
| [^5] | [Income distribution in Thailand is scale-invariant](https://rss.arxiv.org/abs/2402.01141) | 本研究发现泰国的收入分配具有尺度不变性，在近三十年来保持不变，意味着改变收入分配需要进行类似物理学中物质相变的过程。 |
| [^6] | [Municipal cyber risk modeling using cryptographic computing to inform cyber policymaking](https://rss.arxiv.org/abs/2402.01007) | 通过利用密码计算平台收集到的市政机构数据，我们建立了基于数据驱动的市政网络风险模型和网络安全基准，可以帮助指导网络政策制定，量化网络风险，识别差距，优先考虑政策干预，以及跟踪干预措施的进展。 |
| [^7] | [Measures of Resilience to Cyber Contagion -- An Axiomatic Approach for Complex Systems](https://rss.arxiv.org/abs/2312.13884) | 通过基于公理的方法，我们引入了一种新颖的风险度量方法，旨在管理复杂系统中的系统性风险，特别是网络中的系统性网络风险。这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。 |
| [^8] | [Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data.](http://arxiv.org/abs/2310.06000) | 该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。 |
| [^9] | [Uncertainty Propagation and Dynamic Robust Risk Measures.](http://arxiv.org/abs/2308.12856) | 这篇论文提出了一种用于量化动态环境中不确定性传播的框架。它定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合，以捕捉围绕随机过程和模型的不确定性。论文还探讨了导致动态稳健风险度量的一些已知性质的不确定性集合的条件，并发现由$f$-divergences引起的不确定性集合导致较强的时间一致性。 |
| [^10] | [An Empirical Assessment of Characteristics and Optimal Portfolios.](http://arxiv.org/abs/2104.12975) | 通过离样本外效用函数的视角，本文分析特征的联合预测信息。在动量、大小和残差波动率的条件下，形成的投资组合对所有投资者来说都具有显著更高的确定等价值。特征的互补作用带来了这些好处，例如动量减轻了其他特征固有的过度拟合。最优组合的回报超出了传统因子的范围。 |

# 详细

[^1]: 学习市场：基于情感的集成交易智能体

    Learning the Market: Sentiment-Based Ensemble Trading Agents

    [https://rss.arxiv.org/abs/2402.01441](https://rss.arxiv.org/abs/2402.01441)

    该论文提出了将情感分析和深度强化学习集成算法应用于股票交易的方法，设计了可以根据市场情绪动态调整的交易策略。实验结果表明，这种方法比传统的集成策略、单一智能体算法和市场指标更具盈利性、稳健性和风险最小化。相关研究还发现，传统的固定更换集成智能体的做法并不是最优的，而基于情感的动态框架可以显著提高交易智能体的性能。

    

    我们提出了将情感分析和深度强化学习集成算法应用于股票交易，并设计了一种能够根据当前市场情绪动态调整所使用的智能体的策略。我们创建了一个简单但有效的方法来提取新闻情感，并将其与对现有作品的一般改进结合起来，从而得到有效考虑定性市场因素和定量股票数据的自动交易智能体。我们证明了我们的方法导致了一种盈利、稳健且风险最小的策略，优于传统的集成策略以及单一智能体算法和市场指标。我们的发现表明，传统的每隔固定月份更换集成智能体的做法并不是最优的，基于情感的动态框架极大地提升了这些智能体的性能。此外，由于我们的算法设计简单且...

    We propose the integration of sentiment analysis and deep-reinforcement learning ensemble algorithms for stock trading, and design a strategy capable of dynamically altering its employed agent given concurrent market sentiment. In particular, we create a simple-yet-effective method for extracting news sentiment and combine this with general improvements upon existing works, resulting in automated trading agents that effectively consider both qualitative market factors and quantitative stock data. We show that our approach results in a strategy that is profitable, robust, and risk-minimal -- outperforming the traditional ensemble strategy as well as single agent algorithms and market metrics. Our findings determine that the conventional practice of switching ensemble agents every fixed-number of months is sub-optimal, and that a dynamic sentiment-based framework greatly unlocks additional performance within these agents. Furthermore, as we have designed our algorithm with simplicity and
    
[^2]: 预测基于石油的商品波动性: 动态持续性模型

    Forecasting Volatility of Oil-based Commodities: The Model of Dynamic Persistence

    [https://rss.arxiv.org/abs/2402.01354](https://rss.arxiv.org/abs/2402.01354)

    该论文提出了一种新的方法来预测基于石油的商品的波动性，通过将异质持续性的冲击在时间上平滑变化，这种模型在改进波动性预测方面表现出色，并且特别适用于较长时间段的预测。

    

    时间变动和持续性是波动性的关键属性，通常在基于石油的波动性预测模型中分别研究。在这里，我们提出了一种新的方法，允许具有异质持续性的冲击在时间上平滑变化，并将两者结合在一起建模。我们认为这很重要，因为这种动态是由于基于石油商品的冲击的动态性质自然产生的。我们通过局部回归从数据中识别出这种动态，并建立了一个显著改进波动性预测的模型。这种基于在时间上平滑变化的丰富持续性结构的预测模型，超越了最先进的基准模型，并在较长的预测时间段内特别有用。

    Time variation and persistence are crucial properties of volatility that are often studied separately in oil-based volatility forecasting models. Here, we propose a novel approach that allows shocks with heterogeneous persistence to vary smoothly over time, and thus model the two together. We argue that this is important because such dynamics arise naturally from the dynamic nature of shocks in oil-based commodities. We identify such dynamics from the data using localised regressions and build a model that significantly improves volatility forecasts. Such forecasting models, based on a rich persistence structure that varies smoothly over time, outperform state-of-the-art benchmark models and are particularly useful for forecasting over longer horizons.
    
[^3]: 这篇论文的翻译题目是《由Lévy过程驱动的反向SDE的收敛速度》

    Convergence rates for Backward SDEs driven by L\'evy processes

    [https://rss.arxiv.org/abs/2402.01337](https://rss.arxiv.org/abs/2402.01337)

    这篇论文研究了由复合Poisson过程逼近的Lévy过程驱动的BSDEs的收敛速度，并推导出了在L^2范数和Wasserstein距离下的最优收敛速度。

    

    本文考虑了由复合Poisson过程逼近的Lévy过程以及相应的由复合Poisson逼近的Lévy过程驱动的BSDEs。我们对近似BSDEs收敛到Lévy过程驱动的BSDEs的速度感兴趣。Lévy过程的收敛速度取决于过程的Blumenthal--Getoor指数。我们推导了BSDEs在L^2范数和Wasserstein距离下的收敛速度，并表明，在这两种情况下，收敛速度与相应Lévy过程的收敛速度相等，因此是最优的。

    We consider L\'evy processes that are approximated by compound Poisson processes and, correspondingly, BSDEs driven by L\'evy processes that are approximated by BSDEs driven by their compound Poisson approximations. We are interested in the rate of convergence of the approximate BSDEs to the ones driven by the L\'evy processes. The rate of convergence of the L\'evy processes depends on the Blumenthal--Getoor index of the process. We derive the rate of convergence for the BSDEs in the $\mathbb L^2$-norm and in the Wasserstein distance, and show that, in both cases, this equals the rate of convergence of the corresponding L\'evy process, and thus is optimal.
    
[^4]: 一种用于多变量方向预测技能联合评估的简单方法

    A simple method for joint evaluation of skill in directional forecasts of multiple variables

    [https://rss.arxiv.org/abs/2402.01142](https://rss.arxiv.org/abs/2402.01142)

    本文介绍了一种用于联合评估多变量方向预测技能的简单方法，并利用泰国三个组织在2001年至2021年间的GDP增长和通胀预测数据，展示了该方法的应用。

    

    主要宏观经济变量的预测通常由同一组织同时进行，并且在政策分析和决策中一起呈现和使用。因此，了解预测者是否具备足够的技能来预测这些变量的未来值非常重要。本文介绍了一种用于联合评估多变量方向预测技能的简单方法。该方法易于使用，不依赖于传统统计方法中衡量方向预测准确性所需的复杂假设。采用泰国三个组织（泰国银行、财政政策办公室和国家经济和社会发展委员会办公室）关于2001年至2021年间泰国GDP增长和通胀预测以及实际数据，以演示该方法如何用于评估预测者的技能。

    Forecasts for key macroeconomic variables are almost always made simultaneously by the same organizations, presented together, and used together in policy analyses and decision-makings. It is therefore important to know whether the forecasters are skillful enough to forecast the future values of those variables. Here a method for joint evaluation of skill in directional forecasts of multiple variables is introduced. The method is simple to use and does not rely on complicated assumptions required by the conventional statistical methods for measuring accuracy of directional forecast. The data on GDP growth and inflation forecasts of three organizations from Thailand, namely, the Bank of Thailand, the Fiscal Policy Office, and the Office of the National Economic and Social Development Council as well as the actual data on GDP growth and inflation of Thailand between 2001 and 2021 are employed in order to demonstrate how the method could be used to evaluate the skills of forecasters in pr
    
[^5]: 泰国的收入分配具有尺度不变性

    Income distribution in Thailand is scale-invariant

    [https://rss.arxiv.org/abs/2402.01141](https://rss.arxiv.org/abs/2402.01141)

    本研究发现泰国的收入分配具有尺度不变性，在近三十年来保持不变，意味着改变收入分配需要进行类似物理学中物质相变的过程。

    

    本研究考察了泰国的收入分配是否具有尺度不变性或年度自相似性。通过使用1988年至2021年泰国五分位和十分位的收入份额数据，306个配对的Kolmogorov-Smirnov检验结果表明，泰国的收入分配在统计上具有尺度不变性或年度自相似性，p值范围在0.988至1.000之间。根据这些实证结果，本研究认为，为了改变持续了三十多年的泰国收入分配模式，变革本身不能是渐进的，而应该像物理学中的相变过程一样。

    This study examines whether income distribution in Thailand has a property of scale invariance or self-similarity across years. By using the data on income shares by quintile and by decile of Thailand from 1988 to 2021, the results from 306-pairwise Kolmogorov-Smirnov tests indicate that income distribution in Thailand is statistically scale-invariant or self-similar across years with p-values ranging between 0.988 and 1.000. Based on these empirical findings, this study would like to propose that, in order to change income distribution in Thailand whose pattern had persisted for over three decades, the change itself cannot be gradual but has to be like a phase transition of substance in physics.
    
[^6]: 利用密码计算进行市政网络风险建模以为网络政策制定提供信息

    Municipal cyber risk modeling using cryptographic computing to inform cyber policymaking

    [https://rss.arxiv.org/abs/2402.01007](https://rss.arxiv.org/abs/2402.01007)

    通过利用密码计算平台收集到的市政机构数据，我们建立了基于数据驱动的市政网络风险模型和网络安全基准，可以帮助指导网络政策制定，量化网络风险，识别差距，优先考虑政策干预，以及跟踪干预措施的进展。

    

    市政机构容易受到网络攻击并造成灾难性后果，但缺乏评估自身风险并与同行机构的安全状况进行比较的关键信息。我们利用通过密码计算平台收集的83个市政机构的数据，包括安全状况、安全事件、安全控制失效以及损失，建立基于数据驱动的市政网络风险模型和网络安全基准。我们生成了一个行业中安全状况的基准、网络安全事件的频率、组织根据其防御状况预测的年度损失，以及根据各个控制措施的失效率和相关损失加权。综合这四项指标可以通过量化行业中的网络风险、识别需要解决的差距、优先考虑政策干预，并跟踪这些干预措施的进展来指导网络政策制定。对于市政机构来说，这些新的指标可以帮助他们评估自身网络风险并制定相应的改进措施。

    Municipalities are vulnerable to cyberattacks with devastating consequences, but they lack key information to evaluate their own risk and compare their security posture to peers. Using data from 83 municipalities collected via a cryptographically secure computation platform about their security posture, incidents, security control failures, and losses, we build data-driven cyber risk models and cyber security benchmarks for municipalities. We produce benchmarks of the security posture in a sector, the frequency of cyber incidents, forecasted annual losses for organizations based on their defensive posture, and a weighting of cyber controls based on their individual failure rates and associated losses. Combined, these four items can help guide cyber policymaking by quantifying the cyber risk in a sector, identifying gaps that need to be addressed, prioritizing policy interventions, and tracking progress of those interventions over time. In the case of the municipalities, these newly der
    
[^7]: 对于复杂系统的网络的弹性度量 -- 基于公理的方法

    Measures of Resilience to Cyber Contagion -- An Axiomatic Approach for Complex Systems

    [https://rss.arxiv.org/abs/2312.13884](https://rss.arxiv.org/abs/2312.13884)

    通过基于公理的方法，我们引入了一种新颖的风险度量方法，旨在管理复杂系统中的系统性风险，特别是网络中的系统性网络风险。这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。

    

    我们引入了一种新颖的风险度量方法，旨在管理网络中的系统性风险。与现有方法相比，这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。虽然我们的讨论主要围绕数字网络中系统性网络风险的管理，但我们同时将类比方法应用于其他复杂系统的风险管理，以确定是否适当。

    We introduce a novel class of risk measures designed for the management of systemic risk in networks. In contrast to prevailing approaches, these risk measures target the topological configuration of the network in order to mitigate the propagation risk of contagious threats. While our discussion primarily revolves around the management of systemic cyber risks in digital networks, we concurrently draw parallels to risk management of other complex systems where analogous approaches may be adequate.
    
[^8]: 鼓励数据共享以进行能源预测：具有相关数据的分析市场

    Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data. (arXiv:2310.06000v1 [econ.GN])

    [http://arxiv.org/abs/2310.06000](http://arxiv.org/abs/2310.06000)

    该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。

    

    准确地预测不确定的电力产量对于电力市场的社会福利具有益处，可以减少平衡资源的需求。将这种预测描述为一项分析任务，当前文献提出了以分析市场作为激励手段来改善精度的数据共享方法，例如利用时空相关性。挑战在于，当相关数据用作预测的输入特征时，重叠信息的价值在于收入分配方面使市场设计复杂化，因为这种价值在本质上是组合的。我们为风力预测应用开发了一个考虑相关性的分析市场。为了分配收入，我们采用了基于Shapley值的归因策略，将代理人的特征视为玩家，将他们的相互作用视为一个特征函数博弈。我们说明了描述这种博弈的多种选项，每个选项都有因果细微差别，影响着特征相关时的市场行为。

    Reliably forecasting uncertain power production is beneficial for the social welfare of electricity markets by reducing the need for balancing resources. Describing such forecasting as an analytics task, the current literature proposes analytics markets as an incentive for data sharing to improve accuracy, for instance by leveraging spatio-temporal correlations. The challenge is that, when used as input features for forecasting, correlated data complicates the market design with respect to the revenue allocation, as the value of overlapping information is inherently combinatorial. We develop a correlation-aware analytics market for a wind power forecasting application. To allocate revenue, we adopt a Shapley value-based attribution policy, framing the features of agents as players and their interactions as a characteristic function game. We illustrate that there are multiple options to describe such a game, each having causal nuances that influence market behavior when features are cor
    
[^9]: 不确定性传播和动态稳健风险度量

    Uncertainty Propagation and Dynamic Robust Risk Measures. (arXiv:2308.12856v1 [q-fin.RM])

    [http://arxiv.org/abs/2308.12856](http://arxiv.org/abs/2308.12856)

    这篇论文提出了一种用于量化动态环境中不确定性传播的框架。它定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合，以捕捉围绕随机过程和模型的不确定性。论文还探讨了导致动态稳健风险度量的一些已知性质的不确定性集合的条件，并发现由$f$-divergences引起的不确定性集合导致较强的时间一致性。

    

    我们引入了一个用于量化动态环境中不确定性传播的框架。具体地，我们定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合。这些动态不确定性集合捕捉了围绕随机过程和模型的不确定性，包括分布模糊等因素。不确定性集合的例子包括由Wasserstein距离和$f$-divergences引起的不确定性集合。我们进一步将动态稳健风险度量定义为不确定性集合内所有候选风险的上确界。我们以公理化的方式讨论了导致动态稳健风险度量的一些已知性质（如凸性和一致性）的不确定性集合的条件。此外，我们还讨论了导致稳健动态风险度量时间一致性的动态不确定性集合的必要和充分特性。我们发现，由$f$-divergences引起的不确定性集合导致较强的时间一致性。

    We introduce a framework for quantifying propagation of uncertainty arising in a dynamic setting. Specifically, we define dynamic uncertainty sets designed explicitly for discrete stochastic processes over a finite time horizon. These dynamic uncertainty sets capture the uncertainty surrounding stochastic processes and models, accounting for factors such as distributional ambiguity. Examples of uncertainty sets include those induced by the Wasserstein distance and $f$-divergences.  We further define dynamic robust risk measures as the supremum of all candidates' risks within the uncertainty set. In an axiomatic way, we discuss conditions on the uncertainty sets that lead to well-known properties of dynamic robust risk measures, such as convexity and coherence. Furthermore, we discuss the necessary and sufficient properties of dynamic uncertainty sets that lead to time-consistencies of robust dynamic risk measures. We find that uncertainty sets stemming from $f$-divergences lead to stro
    
[^10]: 一个对特征和最优组合的实证评估

    An Empirical Assessment of Characteristics and Optimal Portfolios. (arXiv:2104.12975v3 [q-fin.GN] UPDATED)

    [http://arxiv.org/abs/2104.12975](http://arxiv.org/abs/2104.12975)

    通过离样本外效用函数的视角，本文分析特征的联合预测信息。在动量、大小和残差波动率的条件下，形成的投资组合对所有投资者来说都具有显著更高的确定等价值。特征的互补作用带来了这些好处，例如动量减轻了其他特征固有的过度拟合。最优组合的回报超出了传统因子的范围。

    

    本文通过离样本外效用函数的视角分析特征的联合预测信息。我们通过最大化样本内损失函数来减轻特征权重估计误差，该损失函数比效用函数更凹。尽管没有单一特征可以提高所有投资者的效用，但在动力、大小和残差波动率的条件下，形成的投资组合的确定等价值显著高于基准值。特征的互补作用带来了这些好处，例如动力减轻了其他特征固有的过度拟合。最优组合的回报在很大程度上超出了传统因子的范围。

    We analyze characteristics' joint predictive information through the lens of out-of-sample power utility functions. Linking weights to characteristics to form optimal portfolios suffers from estimation error which we mitigate by maximizing an in-sample loss function that is more concave than the utility function. While no single characteristic can be used to enhance utility by all investors, conditioning on momentum, size, and residual volatility produces portfolios with significantly higher certainty equivalents than benchmarks for all investors. Characteristic complementarities produce the benefits, for example momentum mitigates overfitting inherent in other characteristics. Optimal portfolios' returns lie largely outside the span of traditional factors.
    

