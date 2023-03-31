# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BloombergGPT: A Large Language Model for Finance.](http://arxiv.org/abs/2303.17564) | 本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。 |
| [^2] | [Coskewness under dependence uncertainty.](http://arxiv.org/abs/2303.17266) | 本文研究在依赖不确定的情况下，解决序列随机变量乘积的期望问题，并介绍了一种新的“标准排名共偏度”的概念。 |
| [^3] | [Capitalising the Network Externalities of New Land Supply in the Metaverse.](http://arxiv.org/abs/2303.17180) | 本文基于区块链的虚拟世界The Sandbox的实验数据，研究了首次供应在元宇宙中的网络外部性的资本化，结果发现新土地的供应将会与现有土地形成竞争，但是新增供应也能提升现有土地的网络价值。 |
| [^4] | [Entrepreneurial Capability And Engagement Of Persons With Disabilities Toward A Framework For Inclusive Entrepreneurship.](http://arxiv.org/abs/2303.17130) | 该研究旨在建立包容性创业框架，调查残疾人的创业能力和参与度。调查结果发现，残疾人在创业方面的感知能力水平较为合适，大多数参与水平回应时投入较多，受访者认为缺乏获得信贷和资金的能力和困难是创业过程中的主要挑战。 |
| [^5] | [Liquidity Constraints, Cash Windfalls, and Entrepreneurship: Evidence from Administrative Data on Lottery Winners.](http://arxiv.org/abs/2303.17029) | 本文利用台湾彩票获奖者的行政数据，发现丰厚的现金盈利增加了流动性受限家庭创业的可能性，创业财富弹性为0.25至0.36。连续创业家更有可能开始一家新业务，但不会改变他们继续经营当前业务的决策。 |
| [^6] | [Option pricing using a skew random walk pricing tree.](http://arxiv.org/abs/2303.17014) | 本论文提出了基于二叉树模型和It\^o-Mckean偏斜布朗运动的期权定价模型，继承自然界的偏斜性质，并在风险中性世界中得以保留，具备完备性质。应用于估值的数字应用，可计算交易所的欧式看涨期权和看跌期权的价格。 |
| [^7] | [How to handle the COS method for option pricing.](http://arxiv.org/abs/2303.16012) | 介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。 |
| [^8] | [The State of Food Systems Worldwide: Counting Down to 2030.](http://arxiv.org/abs/2303.13669) | 本文介绍了食物系统到2030倒计时倡议提出的五个主题和指标体系结构，并应用最新可用数据构建了第一个全球食物系统基线以跟踪转型。 |
| [^9] | [Quantum algorithm for stochastic optimal stopping problems with applications in finance.](http://arxiv.org/abs/2111.15332) | 本文提出用于金融领域随机最优停止问题的量子算法可用于美式期权定价，相对于传统算法具有近二次方速度提升。 |
| [^10] | [Learning in a Small/Big World.](http://arxiv.org/abs/2009.11917) | 这篇论文研究了在小/大世界中最优学习行为的特征，发现随着环境变得更复杂和决策者的认知能力变弱，最优行为逐渐不同。在大世界中，最优学习行为可能表现出多种非贝叶斯学习行为。 |

# 详细

[^1]: BloombergGPT：金融领域的大型语言模型

    BloombergGPT: A Large Language Model for Finance. (arXiv:2303.17564v1 [cs.LG])

    [http://arxiv.org/abs/2303.17564](http://arxiv.org/abs/2303.17564)

    本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。

    

    自然语言处理在金融技术领域有着广泛而复杂的应用，从情感分析和命名实体识别到问答。大型语言模型（LLM）已被证明在各种任务上非常有效；然而，专为金融领域设计的LLM尚未在文献中报告。在本文中，我们提出了BloombergGPT，一个拥有500亿个参数的语言模型，它是基于广泛的金融数据进行训练的。我们构建了一种3630亿个标记的数据集，该数据集基于彭博社的广泛数据来源，可能是迄今最大的领域特定数据集，同时又增加了来自通用数据集的3450亿个标记。我们在标准LLM基准、开放式金融基准和一套最能准确反映我们预期用途的内部基准上验证了BloombergGPT。我们的混合数据集训练产生了一个在金融任务上明显优于现有模型的模型，同时不会牺牲普通任务的性能。

    The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no LLM specialized for the financial domain has been reported in literature. In this work, we present BloombergGPT, a 50 billion parameter language model that is trained on a wide range of financial data. We construct a 363 billion token dataset based on Bloomberg's extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345 billion tokens from general purpose datasets. We validate BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most accurately reflect our intended usage. Our mixed dataset training leads to a model that outperforms existing models on financial tasks by significant margins without sacrificing performance on general 
    
[^2]: 处理依赖不确定性下的共偏度问题

    Coskewness under dependence uncertainty. (arXiv:2303.17266v1 [math.ST])

    [http://arxiv.org/abs/2303.17266](http://arxiv.org/abs/2303.17266)

    本文研究在依赖不确定的情况下，解决序列随机变量乘积的期望问题，并介绍了一种新的“标准排名共偏度”的概念。

    

    本论文研究了$X_i\sim F_i$的$d$个随机变量乘积$\mathbb{E}(X_1X_2\cdots X_d)$在依赖关系不确定时的期望影响。在对$F_i$的一些限制条件下，得到了明确的尖锐限制，并提供了数值方法，以逼近任意选择的$F_i$的结果。结果应用于评估依赖不确定性对共偏度的影响。在这方面，我们介绍了一种新的“标准排名共偏度”的概念，其在严格递增变换下不变，并取值为[-1,1]。

    We study the impact of dependence uncertainty on the expectation of the product of $d$ random variables, $\mathbb{E}(X_1X_2\cdots X_d)$ when $X_i \sim F_i$ for all~$i$. Under some conditions on the $F_i$, explicit sharp bounds are obtained and a numerical method is provided to approximate them for arbitrary choices of the $F_i$. The results are applied to assess the impact of dependence uncertainty on coskewness. In this regard, we introduce a novel notion of "standardized rank coskewness," which is invariant under strictly increasing transformations and takes values in $[-1,\ 1]$.
    
[^3]: 首次供应在元宇宙中的网络外部性的资本化

    Capitalising the Network Externalities of New Land Supply in the Metaverse. (arXiv:2303.17180v1 [econ.GN])

    [http://arxiv.org/abs/2303.17180](http://arxiv.org/abs/2303.17180)

    本文基于区块链的虚拟世界The Sandbox的实验数据，研究了首次供应在元宇宙中的网络外部性的资本化，结果发现新土地的供应将会与现有土地形成竞争，但是新增供应也能提升现有土地的网络价值。

    

    当土地变得更加连接时，由于网络外部性，其价值可能会发生变化。这种想法对于开发人员和政策制定者来说直观而有吸引力，但是在实践中难以分离土地价值的决定因素，这是数据挖掘的一个难点。我们使用基于区块链的虚拟经济——The Sandbox中的房地产，应用一系列自然实验来估计基于网络外部性的土地的因果影响，从而解决了这个挑战。我们的研究结果表明，当新土地可用时，现有土地的网络价值会增加，但是要权衡新土地与现有供应的竞争关系。我们的工作展示了使用虚拟世界进行政策实验的好处。

    When land becomes more connected, its value can change because of network externalities. This idea is intuitive and appealing to developers and policymakers, but documenting their importance is empirically challenging because it is difficult to isolate the determinants of land value in practice. We address this challenge with real estate in The Sandbox, a virtual economy built on blockchain, which provides a series of natural experiments that can be used to estimate the causal impact of land-based of network externalities. Our results show that when new land becomes available, the network value of existing land increases, but there is a trade-off as new land also competes with existing supply. Our work illustrates the benefits of using virtual worlds to conduct policy experiments.
    
[^4]: 残疾人的创业能力和参与度：建立包容性创业框架

    Entrepreneurial Capability And Engagement Of Persons With Disabilities Toward A Framework For Inclusive Entrepreneurship. (arXiv:2303.17130v1 [econ.GN])

    [http://arxiv.org/abs/2303.17130](http://arxiv.org/abs/2303.17130)

    该研究旨在建立包容性创业框架，调查残疾人的创业能力和参与度。调查结果发现，残疾人在创业方面的感知能力水平较为合适，大多数参与水平回应时投入较多，受访者认为缺乏获得信贷和资金的能力和困难是创业过程中的主要挑战。

    

    该研究旨在确定残疾人的创业能力和参与度，以建立包容性创业框架。研究者通过有目的的随机抽样，采用描述性和相关方法进行调查研究。样本来自通用三角市和Rosario市的残疾人事务办公室（PDAO）注册的人群。调查结果表明，受访者大多来自工薪阶层，女性居多，大多数是单身，拥有大学学位，住在中等大小的住房中，收入仅够维生。此外，残疾人在创业方面的感知能力水平较为合适，大多数参与水平回应时会有所投入。值得一提的是，年龄和婚姻状况与研究对象的大部分变量存在显著关系。最后，残疾人受访者所认为的挑战包括以下几点：缺乏财务能力、获得信贷和资金的难度，身体状况等。

    The study was designed to determine the entrepreneurial capability and engagement of persons with disabilities toward a framework for inclusive entrepreneurship. The researcher used descriptive and correlational approaches through purposive random sampling. The sample came from the City of General Trias and the Municipality of Rosario, registered under their respective Persons with Disabilities Affairs Offices (PDAO). The findings indicated that the respondents are from the working class, are primarily female, are mostly single, have college degrees, live in a medium-sized home, and earn the bare minimum. Furthermore, PWDs' perceived capability level in entrepreneurship was somehow capable, and the majority of engagement level responses were somehow engaged. Considerably, age and civil status have significant relationships with most of the variables under study. Finally, the perceived challenges of PWDs' respondents noted the following: lack of financial capacity, access to credit and 
    
[^5]: 流动性约束、现金盈利和创业：台湾彩票获奖者的实证研究

    Liquidity Constraints, Cash Windfalls, and Entrepreneurship: Evidence from Administrative Data on Lottery Winners. (arXiv:2303.17029v1 [econ.GN])

    [http://arxiv.org/abs/2303.17029](http://arxiv.org/abs/2303.17029)

    本文利用台湾彩票获奖者的行政数据，发现丰厚的现金盈利增加了流动性受限家庭创业的可能性，创业财富弹性为0.25至0.36。连续创业家更有可能开始一家新业务，但不会改变他们继续经营当前业务的决策。

    

    本文利用台湾彩票获奖者的行政数据，考察现金盈利对创业的影响。我们将在特定年份中赢得超过150万新台币（50,000美元）彩票的家庭的创业决策与赢得不到15,000新台币（500美元）的家庭相比较。结果表明，丰厚的盈利增加了1.5个百分点（从基线平均值增加了125%）的创业可能性。创业财富弹性为0.25至0.36。此外，流动性受限的家庭驱动了盈利引发的创业反应。最后，我们研究了拥有企业的家庭如何应对现金盈利，并发现连续创业家更有可能开始一家新业务，但不会改变他们继续经营当前业务的决策。

    Using administrative data on Taiwanese lottery winners, this paper examines the effects of cash windfalls on entrepreneurship. We compare the start-up decisions of households winning more than 1.5 million NTD (50,000 USD) in the lottery in a particular year with those of households winning less than 15,000 NTD (500 USD). Our results suggest that a substantial windfall increases the likelihood of starting a business by 1.5 percentage points (125% from the baseline mean). Startup wealth elasticity is 0.25 to 0.36. Moreover, households who tend to be liquidity-constrained drive the windfall-induced entrepreneurial response. Finally, we examine how households with a business react to a cash windfall and find that serial entrepreneurs are more likely to start a new business but do not change their decision to continue the current business.
    
[^6]: 使用偏斜随机漫步定价树的期权定价

    Option pricing using a skew random walk pricing tree. (arXiv:2303.17014v1 [q-fin.MF])

    [http://arxiv.org/abs/2303.17014](http://arxiv.org/abs/2303.17014)

    本论文提出了基于二叉树模型和It\^o-Mckean偏斜布朗运动的期权定价模型，继承自然界的偏斜性质，并在风险中性世界中得以保留，具备完备性质。应用于估值的数字应用，可计算交易所的欧式看涨期权和看跌期权的价格。

    

    受Corns-Satchell 连续时间期权定价模型的启发，我们开发了一个二叉树定价模型，其中基础资产价格动态遵循It\^o-Mckean偏斜布朗运动。尽管Corns-Satchell市场模型不完备，但我们的离散时间市场模型定义于自然世界中，在无套利条件下在风险中性世界下进行扩展，其中衍生品是以唯一确定的风险中性概率进行定价的。并且是完备的。自然界中引入的偏斜在风险中性世界中被保留。此外，我们证明了该模型在连续时间极限下保持偏斜。我们向交易所交易的跟踪S＆P全球1200指数的ETF上的欧式看涨期权和看跌期权的估值提供了数字应用。

    Motivated by the Corns-Satchell, continuous time, option pricing model, we develop a binary tree pricing model with underlying asset price dynamics following It\^o-Mckean skew Brownian motion. While the Corns-Satchell market model is incomplete, our discrete time market model is defined in the natural world; extended to the risk neutral world under the no-arbitrage condition where derivatives are priced under uniquely determined risk-neutral probabilities; and is complete. The skewness introduced in the natural world is preserved in the risk neutral world. Furthermore, we show that the model preserves skewness under the continuous-time limit. We provide numerical applications of our model to the valuation of European put and call options on exchange-traded funds tracking the S&P Global 1200 index.
    
[^7]: 如何处理用于期权定价的 COS 方法

    How to handle the COS method for option pricing. (arXiv:2303.16012v1 [q-fin.CP])

    [http://arxiv.org/abs/2303.16012](http://arxiv.org/abs/2303.16012)

    介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。

    

    Fourier余弦展开（COS）方法用于高效地计算欧式期权价格。要应用COS方法，必须指定两个参数：对数收益率密度的截断范围和用余弦级数逼近截断密度的项数N。如何选择截断范围已经为人所知。在这里，我们还能找到一个明确的并且有用的项数N的界限。我们还进一步表明，如果密度是平滑的并且呈指数衰减，则COS方法至少具有指数收敛阶数。但是，如果密度平滑但有重尾巴，就像在有限矩阵log稳定模型中一样，则COS方法没有指数收敛阶数。数值实验确认了理论发现。

    The Fourier cosine expansion (COS) method is used for pricing European options numerically very efficiently. To apply the COS method, one has to specify two parameters: a truncation range for the density of the log-returns and a number of terms N to approximate the truncated density by a cosine series. How to choose the truncation range is already known. Here, we are able to find an explicit and useful bound for N as well. We further show that the COS method has at least an exponential order of convergence if the density is smooth and decays exponentially. But, if the density is smooth and has heavy tails like in the Finite Moment Log Stable model, the COS method has not an exponential order of convergence. Numerical experiments confirm the theoretical findings.
    
[^8]: 全球食物系统的现状：倒计时至2030年

    The State of Food Systems Worldwide: Counting Down to 2030. (arXiv:2303.13669v1 [econ.GN])

    [http://arxiv.org/abs/2303.13669](http://arxiv.org/abs/2303.13669)

    本文介绍了食物系统到2030倒计时倡议提出的五个主题和指标体系结构，并应用最新可用数据构建了第一个全球食物系统基线以跟踪转型。

    

    转变食物系统是实现全球发展和可持续目标、带来更健康、公平、可持续和有弹性的未来所必需的。迄今为止，没有综合性框架来跟踪食物系统的转型及其对全球目标的贡献。在2021年，食物系统到2030倒计时倡议（FSCI）阐明了一种体系结构来监测食物系统，分为五个主题：1 饮食、营养和健康；2 环境、自然资源和生产；3 生计、贫困和公平；4 治理；和5弹性和可持续性。每个主题包括三至五个指标领域。本文基于该体系结构，介绍了包容性、咨询性的过程用于选择指标，并应用最新可用的数据构建了第一个全球食物系统基线以跟踪转型。虽然涵盖了大多数主题和领域的数据，但关键指标仍需进一步开发。

    Transforming food systems is essential to bring about a healthier, equitable, sustainable, and resilient future, including achieving global development and sustainability goals. To date, no comprehensive framework exists to track food systems transformation and their contributions to global goals. In 2021, the Food Systems Countdown to 2030 Initiative (FSCI) articulated an architecture to monitor food systems across five themes: 1 diets, nutrition, and health; 2 environment, natural resources, and production; 3 livelihoods, poverty, and equity; 4 governance; and 5 resilience and sustainability. Each theme comprises three-to-five indicator domains. This paper builds on that architecture, presenting the inclusive, consultative process used to select indicators and an application of the indicator framework using the latest available data, constructing the first global food systems baseline to track transformation. While data are available to cover most themes and domains, critical indicat
    
[^9]: 用于金融领域随机最优停止问题的量子算法

    Quantum algorithm for stochastic optimal stopping problems with applications in finance. (arXiv:2111.15332v3 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2111.15332](http://arxiv.org/abs/2111.15332)

    本文提出用于金融领域随机最优停止问题的量子算法可用于美式期权定价，相对于传统算法具有近二次方速度提升。

    

    著名的最小二乘蒙特卡罗（LSM）算法将线性最小二乘回归与蒙特卡罗模拟相结合，近似解决随机最优停止理论问题。本文提出了基于量子访问随机过程、计算最优停止时间的量子电路以及蒙特卡罗的量子LSM算法。对于该算法，我们阐明了函数逼近与量子蒙特卡罗算法的复杂相互作用。在一些温和的假设下，我们的算法在运行时间上实现了近二次方速度提升，相对于LSM算法而言。具体来说，我们的量子算法可应用于美式期权定价，我们对布朗运动和几何布朗运动过程的共同情况进行了案例研究。

    The famous least squares Monte Carlo (LSM) algorithm combines linear least square regression with Monte Carlo simulation to approximately solve problems in stochastic optimal stopping theory. In this work, we propose a quantum LSM based on quantum access to a stochastic process, on quantum circuits for computing the optimal stopping times, and on quantum techniques for Monte Carlo. For this algorithm, we elucidate the intricate interplay of function approximation and quantum algorithms for Monte Carlo. Our algorithm achieves a nearly quadratic speedup in the runtime compared to the LSM algorithm under some mild assumptions. Specifically, our quantum algorithm can be applied to American option pricing and we analyze a case study for the common situation of Brownian motion and geometric Brownian motion processes.
    
[^10]: 小/大世界中的学习

    Learning in a Small/Big World. (arXiv:2009.11917v8 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2009.11917](http://arxiv.org/abs/2009.11917)

    这篇论文研究了在小/大世界中最优学习行为的特征，发现随着环境变得更复杂和决策者的认知能力变弱，最优行为逐渐不同。在大世界中，最优学习行为可能表现出多种非贝叶斯学习行为。

    

    复杂性和有限能力对我们在不确定性下的学习和决策有深刻的影响。本文使用有限自动机理论来模拟信念形成过程，研究了在小世界和大世界中，即环境复杂度相对于决策者的认知能力较低或较高的情况下，最优学习行为的特征。在非常小的世界中，最优行为非常接近贝叶斯基准，但随着世界的变大，最优行为则越来越不同。此外，在大世界中，最优学习行为可能表现出多种已有文献报道过的非贝叶斯学习行为，包括启发式的使用、相关忽视、持续的过度自信、不注意学习以及模型简化或误设等行为。这些结果建立了非贝叶斯学习行为、复杂度和认知能力之间明确可验证的关系。

    Complexity and limited ability have profound effect on how we learn and make decisions under uncertainty. Using the theory of finite automaton to model belief formation, this paper studies the characteristics of optimal learning behavior in small and big worlds, where the complexity of the environment is low and high, respectively, relative to the cognitive ability of the decision maker. Optimal behavior is well approximated by the Bayesian benchmark in very small world but is more different as the world gets bigger. In addition, in big worlds, the optimal learning behavior could exhibit a wide range of well-documented non-Bayesian learning behavior, including the use of heuristics, correlation neglect, persistent over-confidence, inattentive learning, and other behaviors of model simplification or misspecification. These results establish a clear and testable relationship among the prominence of non-Bayesian learning behavior, complexity, and cognitive ability.
    

