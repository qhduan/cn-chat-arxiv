# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semi-parametric financial risk forecasting incorporating multiple realized measures](https://arxiv.org/abs/2402.09985) | 本研究提出了一个半参数金融风险预测框架，它利用多个实现指标建模联合VaR和ES，并采用非对称拉普拉斯分布进行贝叶斯推断。实证结果表明，该框架在涵盖COVID-19期间的多个股市上表现良好。 |
| [^2] | [Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment](https://arxiv.org/abs/2402.09746) | Alpha-GPT 2.0是一个量化投资框架，采用人机协同的方法，将人类研究者的见解融入到alpha研究中，提高了量化投资研究的效率和准确性。 |
| [^3] | [ABIDES-Economist: Agent-Based Simulation of Economic Systems with Learning Agents](https://arxiv.org/abs/2402.09563) | ABIDES-Economist是一个多智能体模拟器，用于经济系统，具有学习代理、规则性策略和基于现实数据的设计。它提供了一种使用强化学习策略的模拟环境，并可以模拟和分析各种经济情景。 |
| [^4] | [A game theoretic approach to lowering incentives to violate speed limits in Finland](https://arxiv.org/abs/2402.09556) | 本论文通过使用无限重复博弈模型，并采用子博弈完美均衡策略配置作为解决概念，提出了一种胡萝卜和棍子的方法来降低违反芬兰限速规定的激励，并在短期重复博弈中构建了一个纳什均衡策略配置。 |
| [^5] | [Rationality Report Cards: Assessing the Economic Rationality of Large Language Models](https://arxiv.org/abs/2402.09552) | 本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。 |
| [^6] | [On the Potential of Network-Based Features for Fraud Detection](https://arxiv.org/abs/2402.09495) | 本文研究了基于网络特征在欺诈检测中的潜力，通过使用个性化的PageRank算法来捕捉欺诈的社会动态。实验结果表明，集成PPR可以提高模型的预测能力并提供独特有价值的信息。 |
| [^7] | [Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior](https://arxiv.org/abs/2402.03953) | 本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。 |
| [^8] | [Statistical inference for rough volatility: Minimax Theory](https://arxiv.org/abs/2210.01214) | 本文提供了对粗糙波动模型进行严格统计分析的研究，通过建立极小化下界和使用小波方法，成功推断了粗糙波动模型的参数，并扩展了现有的结果。 |
| [^9] | [Thiele's PIDE for unit-linked policies in the Heston-Hawkes stochastic volatility model.](http://arxiv.org/abs/2309.03541) | 本文研究了在Heston-Hawkes随机波动率模型中单位连结政策的Thiele's PIDE(偏微分方程)，推导了Thiele's微分方程以计算保险公司的储备。 |
| [^10] | [Equilibrium in Functional Stochastic Games with Mean-Field Interaction.](http://arxiv.org/abs/2306.05433) | 该论文提出了一种新的方法来明确推导出带有平均场相互作用的功能随机博弈的纳什均衡，同时证明了均衡的收敛性和存在的条件比有限玩家博弈的条件更少。 |
| [^11] | [Implementing a Hierarchical Deep Learning Approach for Simulating Multi-Level Auction Data.](http://arxiv.org/abs/2207.12255) | 我们提出了一种基于深度学习的方法，能够模拟现实中复杂的多层次拍卖数据，并将其应用于代理学习和建模应用，为模拟性研究的进展做出了贡献。 |

# 详细

[^1]: 半参数金融风险预测模型中引入多个实现指标

    Semi-parametric financial risk forecasting incorporating multiple realized measures

    [https://arxiv.org/abs/2402.09985](https://arxiv.org/abs/2402.09985)

    本研究提出了一个半参数金融风险预测框架，它利用多个实现指标建模联合VaR和ES，并采用非对称拉普拉斯分布进行贝叶斯推断。实证结果表明，该框架在涵盖COVID-19期间的多个股市上表现良好。

    

    本文提出了一种半参数联合VaR和ES预测框架，其中引入了多个实现指标。该框架通过使用多个实现指标作为外生变量来扩展分位数回归模型VaR。然后，利用实现指标的信息来建模VaR和ES之间的时变关系。最后，使用模拟量和实现指标之间的同期相关性建立的度量方程来完成模型。基于非对称拉普拉斯分布的准似然方法实现了对该模型的贝叶斯推断。使用自适应马尔可夫链蒙特卡洛方法对模型进行估计。经验证实，该框架在涵盖COVID-19期间的2000年1月至2022年6月间的六个股市上具有较好的预测性能。

    arXiv:2402.09985v1 Announce Type: new  Abstract: A semi-parametric joint Value-at-Risk (VaR) and Expected Shortfall (ES) forecasting framework employing multiple realized measures is developed. The proposed framework extends the quantile regression using multiple realized measures as exogenous variables to model the VaR. Then, the information from realized measures is used to model the time-varying relationship between VaR and ES. Finally, a measurement equation that models the contemporaneous dependence between the quantile and realized measures is used to complete the model. A quasi-likelihood, built on the asymmetric Laplace distribution, enables the Bayesian inference for the proposed model. An adaptive Markov Chain Monte Carlo method is used for the model estimation. The empirical section evaluates the performance of the proposed framework with six stock markets from January 2000 to June 2022, covering the period of COVID-19. Three realized measures, including 5-minute realized va
    
[^2]: Alpha-GPT 2.0：量化投资中的人机协同AI

    Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment

    [https://arxiv.org/abs/2402.09746](https://arxiv.org/abs/2402.09746)

    Alpha-GPT 2.0是一个量化投资框架，采用人机协同的方法，将人类研究者的见解融入到alpha研究中，提高了量化投资研究的效率和准确性。

    

    最近，我们在量化投资领域引入了一种新的alpha挖掘范 Paradigm，开发了一个新的交互式的alpha挖掘系统框架，即Alpha-GPT。该系统围绕基于大型语言模型的迭代式人机交互，引入了一种人机协同的alpha发现方法。在本文中，我们介绍了下一代Alpha-GPT 2.0，这是一个量化投资框架，进一步涵盖了量化投资中的关键建模和分析阶段。该框架强调人机之间的迭代互动研究，并贯穿整个量化投资流程，体现了一种人机协同的策略。通过将人类研究人员的见解融入到系统性的alpha研究过程中，我们有效地利用了人机协同的方法，提高了量化投资研究的效率和准确性。

    arXiv:2402.09746v1 Announce Type: new  Abstract: Recently, we introduced a new paradigm for alpha mining in the realm of quantitative investment, developing a new interactive alpha mining system framework, Alpha-GPT. This system is centered on iterative Human-AI interaction based on large language models, introducing a Human-in-the-Loop approach to alpha discovery. In this paper, we present the next-generation Alpha-GPT 2.0 \footnote{Draft. Work in progress}, a quantitative investment framework that further encompasses crucial modeling and analysis phases in quantitative investment. This framework emphasizes the iterative, interactive research between humans and AI, embodying a Human-in-the-Loop strategy throughout the entire quantitative investment pipeline. By assimilating the insights of human researchers into the systematic alpha research process, we effectively leverage the Human-in-the-Loop approach, enhancing the efficiency and precision of quantitative investment research.
    
[^3]: ABIDES-Economist: 具有学习代理的经济系统的基于代理的模拟

    ABIDES-Economist: Agent-Based Simulation of Economic Systems with Learning Agents

    [https://arxiv.org/abs/2402.09563](https://arxiv.org/abs/2402.09563)

    ABIDES-Economist是一个多智能体模拟器，用于经济系统，具有学习代理、规则性策略和基于现实数据的设计。它提供了一种使用强化学习策略的模拟环境，并可以模拟和分析各种经济情景。

    

    我们介绍了一个多智能体模拟器，用于由异质家庭、异质公司、中央银行和政府代理组成的经济系统，该系统可以受到外生的随机冲击。代理之间的互动定义了经济中商品的生产和消费以及资金的流动。每个代理可以根据固定的、规则性的策略行动，也可以通过与模拟器中其他代理的互动来学习自己的策略。我们通过选择基于经济文献的代理异质性参数，并将其行动空间设计与美国的实际数据相一致，来使我们的模拟器具备现实基础。我们的模拟器通过为经济系统定义 OpenAI Gym 风格的环境，促进了代理使用强化学习策略的能力。通过模拟和分析两种假设的（但有趣的）经济情景，我们展示了我们模拟器的实用性。

    arXiv:2402.09563v1 Announce Type: cross  Abstract: We introduce a multi-agent simulator for economic systems comprised of heterogeneous Households, heterogeneous Firms, Central Bank and Government agents, that could be subjected to exogenous, stochastic shocks. The interaction between agents defines the production and consumption of goods in the economy alongside the flow of money. Each agent can be designed to act according to fixed, rule-based strategies or learn their strategies using interactions with others in the simulator. We ground our simulator by choosing agent heterogeneity parameters based on economic literature, while designing their action spaces in accordance with real data in the United States. Our simulator facilitates the use of reinforcement learning strategies for the agents via an OpenAI Gym style environment definition for the economic system. We demonstrate the utility of our simulator by simulating and analyzing two hypothetical (yet interesting) economic scenar
    
[^4]: 降低芬兰违反限速的激励的博弈论方法

    A game theoretic approach to lowering incentives to violate speed limits in Finland

    [https://arxiv.org/abs/2402.09556](https://arxiv.org/abs/2402.09556)

    本论文通过使用无限重复博弈模型，并采用子博弈完美均衡策略配置作为解决概念，提出了一种胡萝卜和棍子的方法来降低违反芬兰限速规定的激励，并在短期重复博弈中构建了一个纳什均衡策略配置。

    

    我们通过讨论一个无限重复博弈模型来扩展之前的研究，该模型以子博弈完美均衡策略配置（SPE）作为解决概念，以胡萝卜和棍子的方式降低违反限速的激励。在构建SPE策略配置时，选择初始状态使得驾驶员采取混合策略，而警察不以确定性执行。我们还假设了一个短期版本的重复博弈，该博弈具有广义阶段博弈的回报。对于这个博弈，我们构建了一个多阶段策略配置，它是一个纳什均衡但不是SPE。通过展示某些解的候选者不满足一个必要条件，即在完全信息的重复博弈中满足一次性偏离属性，排除了一些解。

    arXiv:2402.09556v1 Announce Type: new  Abstract: We expand on earlier research on the topic by discussing an infinitely repeated game model with a subgame perfect equilibrium strategy profile (SPE) as a solution concept that diminishes incentives to violate speed limits in a carrot and stick fashion. In attempts to construct an SPE strategy profile, the initial state is chosen such that the drivers are playing a mixed strategy whereas the police is not enforcing with certainty. We also postulate a short period version of the repeated game with generalized stage game payoffs. For this game, we construct a multistage strategy profile that is a Nash equilibrium but not an SPE. Some solution candidates are excluded by showing that they do not satisfy a one shot deviation property that is a necessary condition for an SPE profile in a repeated game of perfect information.
    
[^5]: 理性报告卡：评估大型语言模型的经济合理性

    Rationality Report Cards: Assessing the Economic Rationality of Large Language Models

    [https://arxiv.org/abs/2402.09552](https://arxiv.org/abs/2402.09552)

    本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。

    

    越来越多的人对将LLM用作决策"代理人"兴趣日益增加。这包括很多自由度：应该使用哪个模型；如何进行提示；是否要求其进行内省、进行思考链等。解决这些问题（更广泛地说，确定LLM代理人是否足够可靠以便获得信任）需要一种评估这种代理人经济合理性的方法论，在本文中我们提供了一个方法。我们首先对理性决策的经济文献进行了调研、将代理人应该展现的大量细粒度"要素"进行分类，并确定了它们之间的依赖关系。然后，我们提出了一个基准分布，以定量评分LLM在这些要素上的表现，并结合用户提供的评分标准，生成一份"理性报告卡"。最后，我们描述了与14种不同的LLM进行的大规模实证实验的结果。

    arXiv:2402.09552v1 Announce Type: new  Abstract: There is increasing interest in using LLMs as decision-making "agents." Doing so includes many degrees of freedom: which model should be used; how should it be prompted; should it be asked to introspect, conduct chain-of-thought reasoning, etc? Settling these questions -- and more broadly, determining whether an LLM agent is reliable enough to be trusted -- requires a methodology for assessing such an agent's economic rationality. In this paper, we provide one. We begin by surveying the economic literature on rational decision making, taxonomizing a large set of fine-grained "elements" that an agent should exhibit, along with dependencies between them. We then propose a benchmark distribution that quantitatively scores an LLMs performance on these elements and, combined with a user-provided rubric, produces a "rationality report card." Finally, we describe the results of a large-scale empirical experiment with 14 different LLMs, characte
    
[^6]: 关于基于网络特征在欺诈检测中潜力的研究

    On the Potential of Network-Based Features for Fraud Detection

    [https://arxiv.org/abs/2402.09495](https://arxiv.org/abs/2402.09495)

    本文研究了基于网络特征在欺诈检测中的潜力，通过使用个性化的PageRank算法来捕捉欺诈的社会动态。实验结果表明，集成PPR可以提高模型的预测能力并提供独特有价值的信息。

    

    在线交易欺诈给企业和消费者带来了重大挑战，面临着重大的经济损失。传统的基于规则的系统难以跟上欺诈战术的演变，导致高误报率和漏报率。机器学习技术通过利用历史数据识别欺诈模式提供了一个有希望的解决方案。本文探讨使用个性化的PageRank（PPR）算法通过分析金融账户之间的关系来捕捉欺诈的社会动态。主要目标是比较传统特征与添加PPR在欺诈检测模型中的性能。结果表明，集成PPR可以提高模型的预测能力，超过基准模型。此外，PPR特征提供了独特而有价值的信息，通过其高特征重要性得分得以证明。特征稳定性分析证实了一致的结果。

    arXiv:2402.09495v1 Announce Type: cross  Abstract: Online transaction fraud presents substantial challenges to businesses and consumers, risking significant financial losses. Conventional rule-based systems struggle to keep pace with evolving fraud tactics, leading to high false positive rates and missed detections. Machine learning techniques offer a promising solution by leveraging historical data to identify fraudulent patterns. This article explores using the personalised PageRank (PPR) algorithm to capture the social dynamics of fraud by analysing relationships between financial accounts. The primary objective is to compare the performance of traditional features with the addition of PPR in fraud detection models. Results indicate that integrating PPR enhances the model's predictive power, surpassing the baseline model. Additionally, the PPR feature provides unique and valuable information, evidenced by its high feature importance score. Feature stability analysis confirms consist
    
[^7]: 中心化和去中心化交易所中的永续期货合约: 机制和交易者行为

    Perpetual Future Contracts in Centralized and Decentralized Exchanges: Mechanism and Traders' Behavior

    [https://arxiv.org/abs/2402.03953](https://arxiv.org/abs/2402.03953)

    本研究系统化地研究了中心化和去中心化交易所中永续期货合约的交易者行为，并提出了新的分析框架。研究发现，在VAMM模型的DEX中，多头和空头持仓量对价格波动产生相反的影响，而在采用预言机定价模型的DEX中，买家和卖家之间的交易者行为存在明显的不对称性。

    

    本研究提出了一个具有开创性的知识系统化(SoK)计划，重点深入探索交易者在中心化交易所(CEXs)和去中心化交易所(DEXs)中关于永续期货合约的动态和行为。我们改进了现有模型，以研究交易者对价格波动的反应，创建了一个针对这些合约平台的新的分析框架，同时突出了区块链技术在其应用中的作用。我们的研究包括对CEXs的历史数据的比较分析，以及对DEXs上的完整交易数据的更详尽的研究。在虚拟自动化市场做市商(VAMM)模型的DEX上，多头和空头持仓量对价格波动产生相反的影响，这归因于VAMM的价格形成机制。在采用预言机定价模型的DEX中，我们观察到买家和卖家之间交易者行为上存在明显的不对称性。

    This study presents a groundbreaking Systematization of Knowledge (SoK) initiative, focusing on an in-depth exploration of the dynamics and behavior of traders on perpetual future contracts across both centralized exchanges (CEXs), and decentralized exchanges (DEXs). We have refined the existing model for investigating traders' behavior in reaction to price volatility to create a new analytical framework specifically for these contract platforms, while also highlighting the role of blockchain technology in their application. Our research includes a comparative analysis of historical data from CEXs and a more extensive examination of complete transactional data on DEXs. On DEX of Virtual Automated Market Making (VAMM) Model, open interest on short and long positions exert effect on price volatility in opposite direction, attributable to VAMM's price formation mechanism. In the DEXs with Oracle Pricing Model, we observed a distinct asymmetry in trader behavior between buyers and sellers.
    
[^8]: 粗糙波动的统计推断：极小化理论

    Statistical inference for rough volatility: Minimax Theory

    [https://arxiv.org/abs/2210.01214](https://arxiv.org/abs/2210.01214)

    本文提供了对粗糙波动模型进行严格统计分析的研究，通过建立极小化下界和使用小波方法，成功推断了粗糙波动模型的参数，并扩展了现有的结果。

    

    近年来，粗糙波动模型在量化金融界引起了相当大的关注。在这个模型中，资产价格的波动性由一个具有小的赫斯特参数$H$值的分数布朗运动所驱动。本文对这些模型进行了严格的统计分析。为此，我们建立了参数估计的极小化下界，并设计了基于小波的程序来达到这些下界的估计。我们尤其得到了基于n个采样数据对H进行估计的收敛速度的最优结果为$n^{-1/(4H+2)}$，这扩展了目前仅对于较容易的情况$H>1/2$已知的结果。因此，我们得出结论，粗糙波动模型的参数可以在所有情况下以最优准确性进行推断。

    arXiv:2210.01214v2 Announce Type: replace-cross  Abstract: Rough volatility models have gained considerable interest in the quantitative finance community in recent years. In this paradigm, the volatility of the asset price is driven by a fractional Brownian motion with a small value for the Hurst parameter $H$. In this work, we provide a rigorous statistical analysis of these models. To do so, we establish minimax lower bounds for parameter estimation and design procedures based on wavelets attaining them. We notably obtain an optimal speed of convergence of $n^{-1/(4H+2)}$ for estimating $H$ based on n sampled data, extending results known only for the easier case $H>1/2$ so far. We therefore establish that the parameters of rough volatility models can be inferred with optimal accuracy in all regimes.
    
[^9]: Thiele在Heston-Hawkes随机波动率模型中的单位连结政策的PIDE(偏微分方程)

    Thiele's PIDE for unit-linked policies in the Heston-Hawkes stochastic volatility model. (arXiv:2309.03541v1 [q-fin.PR])

    [http://arxiv.org/abs/2309.03541](http://arxiv.org/abs/2309.03541)

    本文研究了在Heston-Hawkes随机波动率模型中单位连结政策的Thiele's PIDE(偏微分方程)，推导了Thiele's微分方程以计算保险公司的储备。

    

    本文的主要目的是推导出在arXiv:2210.15343中介绍的Heston-Hawkes随机波动率模型中单位连结政策的Thiele的微分方程。该模型是著名的Heston模型的扩展，通过在波动率中添加一个复合Hawkes过程，引入了波动率聚类特征。由于该模型是无套利的，因此可以通过在$\mathbb{Q}$下的等价原则下定价单位连结政策。我们检验了一些可积条件，并找到了合适的风险中性概率测度家族以得到Thiele的微分方程。在人寿保险公司中，计算储备的建立和实践方法是通过解决Thiele的方程，这对于保证保险公司的偿付能力是至关重要的。

    The main purpose of the paper is to derive Thiele's differential equation for unit-linked policies in the Heston-Hawkes stochastic volatility model presented in arXiv:2210.15343. This model is an extension of the well-known Heston model that incorporates the volatility clustering feature by adding a compound Hawkes process in the volatility. Since the model is arbitrage-free, pricing unit-linked policies via the equivalence principle under $\mathbb{Q}$ is possible. Some integrability conditions are checked and a suitable family of risk neutral probability measures is found to obtain Thiele's differential equation. The established and practical method to compute reserves in life insurance is by solving Thiele's equation, which is crucial to guarantee the solvency of the insurance company.
    
[^10]: 带有平均场相互作用的功能随机博弈的均衡

    Equilibrium in Functional Stochastic Games with Mean-Field Interaction. (arXiv:2306.05433v1 [math.OC])

    [http://arxiv.org/abs/2306.05433](http://arxiv.org/abs/2306.05433)

    该论文提出了一种新的方法来明确推导出带有平均场相互作用的功能随机博弈的纳什均衡，同时证明了均衡的收敛性和存在的条件比有限玩家博弈的条件更少。

    

    我们考虑了一个一般的有限玩家带有平均场相互作用的随机博弈，其中线性二次成本函数包括作用于$L^2$中的控制的线性算子。我们提出了一种新的方法，通过将相关的一阶条件减少到第二类随机Fredholm方程组的系统，并推导出它们的闭形式解来明确推导出了博弈的纳什均衡。此外，通过证明随机Fredholm方程组的稳定性结果，我们推导出了$N$人博弈的均衡收敛到相应的平均场均衡。作为一个副产品，我们还推导出了平均场博弈的$\varepsilon$-纳什均衡，在这种情况下，它具有价值，因为我们表明，在平均场极限存在均衡的条件比有限玩家博弈的条件要少。最后，我们将我们的一般框架应用于解决各种例子。

    We consider a general class of finite-player stochastic games with mean-field interaction, in which the linear-quadratic cost functional includes linear operators acting on controls in $L^2$. We propose a novel approach for deriving the Nash equilibrium of the game explicitly in terms of operator resolvents, by reducing the associated first order conditions to a system of stochastic Fredholm equations of the second kind and deriving their closed form solution. Furthermore, by proving stability results for the system of stochastic Fredholm equations we derive the convergence of the equilibrium of the $N$-player game to the corresponding mean-field equilibrium. As a by-product we also derive an $\varepsilon$-Nash equilibrium for the mean-field game, which is valuable in this setting as we show that the conditions for existence of an equilibrium in the mean-field limit are less restrictive than in the finite-player game. Finally we apply our general framework to solve various examples, su
    
[^11]: 实现基于分层深度学习的多层次拍卖数据模拟方法

    Implementing a Hierarchical Deep Learning Approach for Simulating Multi-Level Auction Data. (arXiv:2207.12255v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2207.12255](http://arxiv.org/abs/2207.12255)

    我们提出了一种基于深度学习的方法，能够模拟现实中复杂的多层次拍卖数据，并将其应用于代理学习和建模应用，为模拟性研究的进展做出了贡献。

    

    我们提出了一种深度学习解决方案，以应对模拟现实合理的合同拍卖数据所遇到的挑战。这种类型的拍卖数据所遇到的复杂性包括高基数离散特征空间和由与单个拍卖实例相关联的多个出价引起的多层级结构。我们的方法将深度生成建模（DGM）与预测基于拍卖特征的条件出价分布的人工学习器相结合，为模拟性研究的进展做出贡献。这种方法为创建适用于代理学习和建模应用的真实拍卖环境奠定了基础。我们的贡献有两个方面：我们引入了一种综合的方法来模拟多层次离散拍卖数据，我们强调了DGM作为优化模拟技术和促进基于生成型人工智能的经济模型发展的有力工具的潜力。

    We present a deep learning solution to address the challenges of simulating realistic synthetic first-price sealed-bid auction data. The complexities encountered in this type of auction data include high-cardinality discrete feature spaces and a multilevel structure arising from multiple bids associated with a single auction instance. Our methodology combines deep generative modeling (DGM) with an artificial learner that predicts the conditional bid distribution based on auction characteristics, contributing to advancements in simulation-based research. This approach lays the groundwork for creating realistic auction environments suitable for agent-based learning and modeling applications. Our contribution is twofold: we introduce a comprehensive methodology for simulating multilevel discrete auction data, and we underscore the potential of DGM as a powerful instrument for refining simulation techniques and fostering the development of economic models grounded in generative AI.
    

