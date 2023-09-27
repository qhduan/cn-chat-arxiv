# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The ATM implied skew in the ADO-Heston model.](http://arxiv.org/abs/2309.15044) | 本文构建了ADO-Heston模型，通过合适选择的市场风险价格和闭式表达式，能够近似再现香草期权的已知行为。 |
| [^2] | [A dynamic systems approach to harness the potential of social tipping.](http://arxiv.org/abs/2309.14964) | 这篇论文提出了一个动态系统方法来利用社会转折的潜力，以实现净零温室气体排放目标。该方法包括系统观、定向数据收集和全球综合模型等多层面考虑。 |
| [^3] | [Nuclear Energy Acceptance in Poland: From Societal Attitudes to Effective Policy Strategies -- Network Modeling Approach.](http://arxiv.org/abs/2309.14869) | 本研究通过网络建模方法调查了波兰社会对能源的态度，发现政治意识形态、环境态度、风险感知、安全顾虑和经济变量等因素对能源接受度起重要作用。研究结果为改善波兰能源政策提供了基础，强调了与波兰人口的多样化价值观、信仰和偏好相 resonating 的政策的重要性。 |
| [^4] | [Approximation Rates for Deep Calibration of (Rough) Stochastic Volatility Models.](http://arxiv.org/abs/2309.14784) | 本论文提出了利用深度神经网络逼近随机波动性模型的定量误差界限，证明了在适当假设下，可以通过较小的网络规模学习得到期权价格，而不会受到维度灾难的影响。 |
| [^5] | [Gray-box Adversarial Attack of Deep Reinforcement Learning-based Trading Agents.](http://arxiv.org/abs/2309.14615) | 本研究展示了一种通过在同一股票市场进行交易的方式，利用灰盒方法对基于深度强化学习的交易代理进行攻击的可能性。这种方法可以应对交易代理受到对手操纵的问题。 |
| [^6] | [Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems.](http://arxiv.org/abs/2309.14548) | 这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。 |
| [^7] | [Designing Effective Music Excerpts.](http://arxiv.org/abs/2309.14475) | 本研究通过分析iTunes音乐商店中歌曲摘录的准实验变化，发现延长摘录时间可以显著增加歌曲的独立每月听众，特别是对于陌生歌曲和陌生艺术家的效果更为显著。此外，摘录的重复性和可预测性对需求的增强效果有一定的压制作用。因此，该研究支持平台采用更长的摘录时间来改善内容发现。 |
| [^8] | [Common Subcontracting and Airline Prices.](http://arxiv.org/abs/2301.05999) | 地区航空公司的共同子承包一方面会导致更低的价格，另一方面会导致更高的价格，这表明地区航空公司的增长可能对航空业产生反竞争影响。 |
| [^9] | [What is the Price of a Skill? The Value of Complementarity.](http://arxiv.org/abs/2210.01535) | 本研究表明，技能的价值很大程度上由互补性决定，大多数技能在与不同类型的技能组合使用时价值最高。人工智能技能由于其强大的互补性和近年来不断增长的需求，其价值尤为突出，平均提高工人工资21％。 |
| [^10] | [A Mean-Field Control Problem of Optimal Portfolio Liquidation with Semimartingale Strategies.](http://arxiv.org/abs/2207.00446) | 本文研究了具有半鞅策略的最优组合清算问题，证明了价值函数采用线性-二次形式，且最优策略仅在交易期的开始和结束时进行跳跃。 |
| [^11] | [The Shared Cost of Pursuing Shareholder Value.](http://arxiv.org/abs/2103.12138) | 文章使用股东大会时间差异的方法研究了股东偏好和对公司利他决策的影响，发现追求（某些）股东的价值具有分配成本，但大股东的监控可以避免这种由偏好异质性驱动的成本。 |

# 详细

[^1]: ADO-Heston模型中的ATM隐含波动率倾斜

    The ATM implied skew in the ADO-Heston model. (arXiv:2309.15044v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.15044](http://arxiv.org/abs/2309.15044)

    本文构建了ADO-Heston模型，通过合适选择的市场风险价格和闭式表达式，能够近似再现香草期权的已知行为。

    

    本文构建了ADO-Heston模型，该模型是粗糙Heston-like波动率模型的一个马尔可夫近似。通过风险中性和实际度量推导了该模型的特征函数（CF），这是一个不稳定的三维偏微分方程，其系数是时间$t$和Hurst指数$H$的函数。通过合理选择市场风险价格并找到对数价格和ATM隐含波动率的闭式表达式，我们能够复制市场隐含波动率倾斜的已知行为。通过提供的示例，我们声称ADO-Heston模型（纯扩散模型，但具有方差过程的随机均值回归速度，或者是粗糙Heston模型的马尔可夫近似）能够（近似地）再现小$T$时期香草期权的已知行为。我们得出结论认为我们的隐含波动率倾斜曲线${\cal S}(T) \prop$

    In this paper similar to [P. Carr, A. Itkin, 2019] we construct another Markovian approximation of the rough Heston-like volatility model - the ADO-Heston model. The characteristic function (CF) of the model is derived under both risk-neutral and real measures which is an unsteady three-dimensional PDE with some coefficients being functions of the time $t$ and the Hurst exponent $H$. To replicate known behavior of the market implied skew we proceed with a wise choice of the market price of risk, and then find a closed form expression for the CF of the log-price and the ATM implied skew. Based on the provided example, we claim that the ADO-Heston model (which is a pure diffusion model but with a stochastic mean-reversion speed of the variance process, or a Markovian approximation of the rough Heston model) is able (approximately) to reproduce the known behavior of the vanilla implied skew at small $T$. We conclude that the behavior of our implied volatility skew curve ${\cal S}(T) \prop
    
[^2]: 动态系统方法来利用社会转折的潜力

    A dynamic systems approach to harness the potential of social tipping. (arXiv:2309.14964v1 [econ.GN])

    [http://arxiv.org/abs/2309.14964](http://arxiv.org/abs/2309.14964)

    这篇论文提出了一个动态系统方法来利用社会转折的潜力，以实现净零温室气体排放目标。该方法包括系统观、定向数据收集和全球综合模型等多层面考虑。

    

    社会转折点是实现净零温室气体排放目标的有希望的手段。如果触发级联的正反馈机制，它们描述了社会、政治、经济或技术系统如何迅速转入新状态。分析社会转折对快速脱碳的潜力需要考虑社会系统固有的复杂性。在这里，我们指出现有的科学文献倾向于基于叙事的社会转折记述，缺乏广泛的实证框架和多系统视角。随后，我们概述了一个动态系统方法，包括（i）涉及相互关联的反馈机制的系统观；

    Social tipping points are promising levers to achieve net-zero greenhouse gas emission targets. They describe how social, political, economic or technological systems can move rapidly into a new state if cascading positive feedback mechanisms are triggered. Analysing the potential of social tipping for rapid decarbonization requires considering the inherent complexity of social systems. Here, we identify that existing scientific literature is inclined to a narrative-based account of social tipping, lacks a broad empirical framework and a multi-systems view. We subsequently outline a dynamic systems approach that entails (i) a systems outlook involving interconnected feedback mechanisms alongside cross-system and cross-scale interactions, and including a socioeconomic and environmental injustice perspective (ii) directed data collection efforts to provide empirical evidence for and monitor social tipping dynamics, (iii) global, integrated, descriptive modelling to project future dynamic
    
[^3]: 波兰的核能接受度：从社会态度到有效政策策略--网络建模方法

    Nuclear Energy Acceptance in Poland: From Societal Attitudes to Effective Policy Strategies -- Network Modeling Approach. (arXiv:2309.14869v1 [econ.GN])

    [http://arxiv.org/abs/2309.14869](http://arxiv.org/abs/2309.14869)

    本研究通过网络建模方法调查了波兰社会对能源的态度，发现政治意识形态、环境态度、风险感知、安全顾虑和经济变量等因素对能源接受度起重要作用。研究结果为改善波兰能源政策提供了基础，强调了与波兰人口的多样化价值观、信仰和偏好相 resonating 的政策的重要性。

    

    波兰能源部门目前正经历重大转型，赢得公众支持对其能源政策的成功至关重要。我们通过与338名波兰参与者进行研究，调查了社会对各种能源来源的态度，包括核能和可再生能源。应用了一种新颖的网络方法，我们确定了影响能源接受度的多种因素。政治意识形态是塑造公众接受度的核心因素，然而我们还发现环境态度、风险感知、安全顾虑和经济变量也起着重要作用。鉴于核能的长期承诺和其在波兰能源转型中的角色，我们的研究为改善波兰的能源政策提供了基础。我们的研究强调了与人口的多样化价值观、信仰和偏好相 resonating 的政策的重要性。

    Poland is currently undergoing substantial transformation in its energy sector, and gaining public support is pivotal for the success of its energy policies. We conducted a study with 338 Polish participants to investigate societal attitudes towards various energy sources, including nuclear energy and renewables. Applying a novel network approach, we identified a multitude of factors influencing energy acceptance. Political ideology is the central factor in shaping public acceptance, however we also found that environmental attitudes, risk perception, safety concerns, and economic variables play substantial roles. Considering the long-term commitment associated with nuclear energy and its role in Poland's energy transformation, our findings provide a foundation for improving energy policy in Poland. Our research underscores the importance of policies that resonate with the diverse values, beliefs, and preferences of the population. While the risk-risk trade-off and technology-focused s
    
[^4]: 对（粗糙）随机波动性模型进行深度校准的逼近速率

    Approximation Rates for Deep Calibration of (Rough) Stochastic Volatility Models. (arXiv:2309.14784v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.14784](http://arxiv.org/abs/2309.14784)

    本论文提出了利用深度神经网络逼近随机波动性模型的定量误差界限，证明了在适当假设下，可以通过较小的网络规模学习得到期权价格，而不会受到维度灾难的影响。

    

    我们推导了深度神经网络(DNN)逼近$d$维风险资产期权价格的定量误差界限，这些界限与基础模型参数、支付参数和初始条件相关。我们涵盖了马尔可夫性质的广义随机波动性模型以及粗糙Bergomi模型。特别地，在适当假设下，我们证明了在DNN的网络规模仅次于资产向量维度$d$和精度的倒数$\varepsilon^{-1}$时，期权价格可以被学习得到的误差可以任意小于一个小于1/2的误差$\varepsilon$。因此，这种逼近不会受到维度灾难的影响。由于在我们的设置中，适用于DNN的定量逼近结果是基于紧致定义域上的函数，因此我们首先考虑了资产价格限制在紧致集合上的情况，然后再通过对期权价格的收敛性论证将这些结果推广到一般情况下。

    We derive quantitative error bounds for deep neural networks (DNNs) approximating option prices on a $d$-dimensional risky asset as functions of the underlying model parameters, payoff parameters and initial conditions. We cover a general class of stochastic volatility models of Markovian nature as well as the rough Bergomi model. In particular, under suitable assumptions we show that option prices can be learned by DNNs up to an arbitrary small error $\varepsilon \in (0,1/2)$ while the network size grows only sub-polynomially in the asset vector dimension $d$ and the reciprocal $\varepsilon^{-1}$ of the accuracy. Hence, the approximation does not suffer from the curse of dimensionality. As quantitative approximation results for DNNs applicable in our setting are formulated for functions on compact domains, we first consider the case of the asset price restricted to a compact set, then we extend these results to the general case by using convergence arguments for the option prices.
    
[^5]: 深度强化学习交易代理的灰盒对抗攻击

    Gray-box Adversarial Attack of Deep Reinforcement Learning-based Trading Agents. (arXiv:2309.14615v1 [cs.LG])

    [http://arxiv.org/abs/2309.14615](http://arxiv.org/abs/2309.14615)

    本研究展示了一种通过在同一股票市场进行交易的方式，利用灰盒方法对基于深度强化学习的交易代理进行攻击的可能性。这种方法可以应对交易代理受到对手操纵的问题。

    

    近年来，深度强化学习（Deep RL）已成功应用于诸如复杂游戏、自动驾驶汽车和聊天机器人等许多系统中，其中一个有趣的应用案例是将其作为自动化股票交易代理。一般来说，任何自动化交易代理都容易受到交易环境中的对手的操纵，因此研究其鲁棒性对于其实践成功至关重要。然而，用于研究RL鲁棒性的典型机制，即基于白盒梯度基础的对抗样本生成技术（如FGSM），对于这种用例来说已经过时，因为模型受到安全的国际交易所API的保护，如纳斯达克。在这项研究中，我们证明了一种“灰盒”方法可以攻击基于Deep RL的交易代理，仅通过在同一股票市场进行交易，而无需额外接触交易代理。在我们提出的方法中，对手代理使用了一个混合的深度神经网络

    In recent years, deep reinforcement learning (Deep RL) has been successfully implemented as a smart agent in many systems such as complex games, self-driving cars, and chat-bots. One of the interesting use cases of Deep RL is its application as an automated stock trading agent. In general, any automated trading agent is prone to manipulations by adversaries in the trading environment. Thus studying their robustness is vital for their success in practice. However, typical mechanism to study RL robustness, which is based on white-box gradient-based adversarial sample generation techniques (like FGSM), is obsolete for this use case, since the models are protected behind secure international exchange APIs, such as NASDAQ. In this research, we demonstrate that a "gray-box" approach for attacking a Deep RL-based trading agent is possible by trading in the same stock market, with no extra access to the trading agent. In our proposed approach, an adversary agent uses a hybrid Deep Neural Netwo
    
[^6]: 算法勾结还是竞争：平台推荐系统的角色

    Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems. (arXiv:2309.14548v1 [cs.AI])

    [http://arxiv.org/abs/2309.14548](http://arxiv.org/abs/2309.14548)

    这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。

    

    最近的学术研究广泛探讨了基于人工智能(AI)的动态定价算法导致的算法勾结。然而，电子商务平台使用推荐算法来分配不同产品的曝光，而这一重要方面在先前的算法勾结研究中被大部分忽视。我们的研究填补了文献中这一重要的空白，并检验了推荐算法如何决定基于AI的定价算法的竞争或勾结动态。具体而言，我们研究了两种常用的推荐算法：(i)以最大化卖家总利润为目标的推荐系统和(ii)以最大化平台上产品需求为目标的推荐系统。我们构建了一个重复博弈框架，将卖家的定价算法和平台的推荐算法进行了整合。

    Recent academic research has extensively examined algorithmic collusion resulting from the utilization of artificial intelligence (AI)-based dynamic pricing algorithms. Nevertheless, e-commerce platforms employ recommendation algorithms to allocate exposure to various products, and this important aspect has been largely overlooked in previous studies on algorithmic collusion. Our study bridges this important gap in the literature and examines how recommendation algorithms can determine the competitive or collusive dynamics of AI-based pricing algorithms. Specifically, two commonly deployed recommendation algorithms are examined: (i) a recommender system that aims to maximize the sellers' total profit (profit-based recommender system) and (ii) a recommender system that aims to maximize the demand for products sold on the platform (demand-based recommender system). We construct a repeated game framework that incorporates both pricing algorithms adopted by sellers and the platform's recom
    
[^7]: 设计有效的音乐摘录

    Designing Effective Music Excerpts. (arXiv:2309.14475v1 [econ.GN])

    [http://arxiv.org/abs/2309.14475](http://arxiv.org/abs/2309.14475)

    本研究通过分析iTunes音乐商店中歌曲摘录的准实验变化，发现延长摘录时间可以显著增加歌曲的独立每月听众，特别是对于陌生歌曲和陌生艺术家的效果更为显著。此外，摘录的重复性和可预测性对需求的增强效果有一定的压制作用。因此，该研究支持平台采用更长的摘录时间来改善内容发现。

    

    音乐摘录被广泛用于预览和推广音乐作品。有效的摘录可以促使源音乐作品的消费，从而产生收入。然而，什么使得摘录有效仍然未被探索。我们利用苹果的一项政策变化，通过iTunes音乐商店中歌曲摘录的准实验变化，估计摘录时间延长60秒可以平均增加歌曲的独立每月听众5.4％，陌生歌曲增加9.7％，陌生艺术家增加11.1％。这相当于被收录在Spotify全球Top 50播放列表中的影响。我们开发了音乐重复性和不可预测性的衡量指标，以考察信息提供作为一种机制，并发现当摘录过于重复、过于可预测或过于不可预测时，延长摘录时间对需求的增强效果会被压制。我们的发现支持平台采用更长的摘录来改善内容发现，而我们的衡量指标可以帮助提供摘录选择的信息。

    Excerpts are widely used to preview and promote musical works. Effective excerpts induce consumption of the source musical work and thus generate revenue. Yet, what makes an excerpt effective remains unexplored. We leverage a policy change by Apple that generates quasi-exogenous variation in the excerpts of songs in the iTunes Music Store to estimate that having a 60 second longer excerpt increases songs' unique monthly listeners by 5.4% on average, by 9.7% for lesser known songs, and by 11.1% for lesser known artists. This is comparable to the impact of being featured on the Spotify Global Top 50 playlist. We develop measures of musical repetition and unpredictability to examine information provision as a mechanism, and find that the demand-enhancing effect of longer excerpts is suppressed when they are repetitive, too predictable, or too unpredictable. Our findings support platforms' adoption of longer excerpts to improve content discovery and our measures can help inform excerpt sel
    
[^8]: 共同子承包和航空价格

    Common Subcontracting and Airline Prices. (arXiv:2301.05999v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.05999](http://arxiv.org/abs/2301.05999)

    地区航空公司的共同子承包一方面会导致更低的价格，另一方面会导致更高的价格，这表明地区航空公司的增长可能对航空业产生反竞争影响。

    

    在美国航空业中，独立的地区航空公司代表几家全国航空公司在不同的市场上为乘客飞行，这产生了“共同子承包”的情况。一方面，我们发现子承包与较低的价格有关，这符合地区航空公司比主要航空公司更低成本运输乘客的想法。另一方面，我们发现“共同”子承包与更高的价格有关。这两种相互冲突的效应表明，地区航空公司的增长可能对该行业产生反竞争影响。

    In the US airline industry, independent regional airlines fly passengers on behalf of several national airlines across different markets, giving rise to $\textit{common subcontracting}$. On the one hand, we find that subcontracting is associated with lower prices, consistent with the notion that regional airlines tend to fly passengers at lower costs than major airlines. On the other hand, we find that $\textit{common}$ subcontracting is associated with higher prices. These two countervailing effects suggest that the growth of regional airlines can have anticompetitive implications for the industry.
    
[^9]: 技能的价格是多少？互补性的价值

    What is the Price of a Skill? The Value of Complementarity. (arXiv:2210.01535v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2210.01535](http://arxiv.org/abs/2210.01535)

    本研究表明，技能的价值很大程度上由互补性决定，大多数技能在与不同类型的技能组合使用时价值最高。人工智能技能由于其强大的互补性和近年来不断增长的需求，其价值尤为突出，平均提高工人工资21％。

    

    全球劳动力被敦促不断提升自己的技能，因为技术变革偏好某些新技能，同时使其他技能变得多余。但是对于工人和公司来说，哪些技能是一个好的投资呢？由于技能很少是孤立应用的，我们提出互补性强烈决定了技能的经济价值。我们对962种技能进行了研究，证明了它们的价值在很大程度上由互补性决定-即一个能力可以与多少不同的技能理想情况下结合，这些技能具有很高的价值。我们显示，技能的价值是相对的，因为它取决于工人的技能背景。对于大多数技能来说，当与不同类型的技能组合使用时，它们的价值最高。我们用与人工智能（AI）相关的技能集对我们的模型进行了测试。我们发现，人工智能技能尤其有价值-平均提高工人工资21％-这是因为它们的互补性强大，并且近年来需求不断增加。模型和指标

    The global workforce is urged to constantly reskill, as technological change favours particular new skills while making others redundant. But which skills are a good investment for workers and firms? As skills are seldomly applied in isolation, we propose that complementarity strongly determines a skill's economic value. For 962 skills, we demonstrate that their value is strongly determined by complementarity - that is, how many different skills, ideally of high value, a competency can be combined with. We show that the value of a skill is relative, as it depends on the skill background of the worker. For most skills, their value is highest when used in combination with skills of a different type. We put our model to the test with a set of skills related to Artificial Intelligence (AI). We find that AI skills are particularly valuable - increasing worker wages by 21% on average - because of their strong complementarities and their rising demand in recent years. The model and metrics of
    
[^10]: 具有半鞅策略的最优组合清算均值场控制问题

    A Mean-Field Control Problem of Optimal Portfolio Liquidation with Semimartingale Strategies. (arXiv:2207.00446v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2207.00446](http://arxiv.org/abs/2207.00446)

    本文研究了具有半鞅策略的最优组合清算问题，证明了价值函数采用线性-二次形式，且最优策略仅在交易期的开始和结束时进行跳跃。

    

    我们考虑了一个具有瞬态市场影响和自激励订单流的组合清算模型，其中涉及到 c\`adl\`ag 半鞅策略的均值场控制问题。我们证明了价值函数仅通过其分布依赖于状态过程，并且它采用线性-二次形式，其系数满足非标准 Riccati 类型方程的耦合系统。通过由离散时间模型转换到连续时间极限来启发式地获得 Riccati 方程。通过复杂的变换，我们可以将系统带入标准 Riccati 形式，从而推断出全局解的存在性。我们的分析表明，最优策略仅在交易期的开始和结束时进行跳跃。

    We consider a mean-field control problem with c\`adl\`ag semimartingale strategies arising in portfolio liquidation models with transient market impact and self-exciting order flow. We show that the value function depends on the state process only through its law, and that it is of linear-quadratic form and that its coefficients satisfy a coupled system of non-standard Riccati-type equations. The Riccati equations are obtained heuristically by passing to the continuous-time limit from a sequence of discrete-time models. A sophisticated transformation shows that the system can be brought into standard Riccati form from which we deduce the existence of a global solution. Our analysis shows that the optimal strategy jumps only at the beginning and the end of the trading period.
    
[^11]: 追求股东价值的共同成本

    The Shared Cost of Pursuing Shareholder Value. (arXiv:2103.12138v9 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2103.12138](http://arxiv.org/abs/2103.12138)

    文章使用股东大会时间差异的方法研究了股东偏好和对公司利他决策的影响，发现追求（某些）股东的价值具有分配成本，但大股东的监控可以避免这种由偏好异质性驱动的成本。

    

    本文采用准实验性的方法，根据公司股东大会（AGMs）的时间差异，提出了一个可移植的框架，推断股东的偏好和对公司利他决策的影响，并将其应用于covid相关捐赠、最近针对俄罗斯的私人制裁以及公司2012-19年的利他立场。AGMs的媒体曝光带来的形象收益，使得与公司同义的股东（如密切相关的个人）支持昂贵的利他变革，而其他股东（如金融公司）反对这些变革。支持这些变革的影响使收益下降了30％：追求（某些）股东的价值具有分配成本，大股东的监控可以避免由偏好异质性驱动的成本。

    Using quasi-experimental variations from the timing of firms' Annual General Meetings (AGMs), we propose a portable framework to infer shareholders' preferences and influences on firms' prosocial decisions and apply it to covid-related donations, recent private sanctions on Russia, and firms' prosocial stances over 2012-19. Image gains from AGMs' media exposure drive shareholders synonymous with a firm, like closely-connected individuals, to support costly prosocial changes, while others, like financial corporations, oppose them. Influence supporting these changes lowers earnings by 30\%: pursuing the values of (some) shareholders has distributional costs, which the monitoring of large shareholders motivated by heterogeneous preferences could prevent.
    

