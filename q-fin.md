# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Advancing Ad Auction Realism: Practical Insights & Modeling Implications.](http://arxiv.org/abs/2307.11732) | 本文提出了一个学习模型来模拟现实的在线广告拍卖环境，并发现在这样的环境中，使用"软底价"可以提高关键绩效指标，即使投标者来自相同的人群。 |
| [^2] | [Towards Generalizable Reinforcement Learning for Trade Execution.](http://arxiv.org/abs/2307.11685) | 本论文提出了一种面向通用化的交易执行的强化学习方法。研究表明，现有的强化学习方法存在过拟合问题，阻碍了实际应用。作者通过使用离线强化学习和动态上下文建模来解决过拟合问题，并提出了学习上下文的紧凑表示方法。 |
| [^3] | [Assessing the role of small farmers and households in agriculture and the rural economy and measures to support their sustainable development.](http://arxiv.org/abs/2307.11683) | 该论文旨在评估小农户和家庭在农业和农村经济中的作用，以及提出支持其可持续发展的措施。同时，它还研究了如何减少乌克兰影子农业市场的规模和采取何种措施。 |
| [^4] | [ESG Reputation Risk Matters: An Event Study Based on Social Media Data.](http://arxiv.org/abs/2307.11571) | 本研究探讨了股东对环境、社会和治理相关的声誉风险的反应，并通过社交媒体数据观察到ESG风险事件对相关资产回报的显著负面影响。 |
| [^5] | [A Robust Site Selection Model under uncertainty for Special Hospital Wards in Hong Kong.](http://arxiv.org/abs/2307.11508) | 本研究提出了针对香港特殊医院病房的选址问题的鲁棒模型，考虑了不确定性水平、不可行容忍度和可靠性水平，并采用了一种适用于不确定性的优化协议。 |
| [^6] | [Optimal Bubble Riding with Price-dependent Entry: a Mean Field Game of Controls with Common Noise.](http://arxiv.org/abs/2307.11340) | 本文扩展了唐派和王提出的最佳泡沫乘车模型，允许价格相关的进入时间。我们证明了带有共同噪音和随机进入时间的控制均值场博弈的存在性。共同噪音来自资产价格和外生泡沫破裂时间。 |
| [^7] | [Of Models and Tin Men -- a behavioural economics study of principal-agent problems in AI alignment using large-language models.](http://arxiv.org/abs/2307.11137) | 本研究基于行为经济学角度，对使用大语言模型进行AI对齐中的委托-代理问题进行研究，发现现实世界中的AI安全问题不仅涉及设计者与代理之间的冲突，还涉及到多个代理之间的信息不对称与效用函数之间的错位。 |
| [^8] | [Optimal execution and speculation with trade signals.](http://arxiv.org/abs/2306.00621) | 本文提出了一个价格冲击模型，基于市场的订单流动来推导市场的随机价格变化，提出了一个短期信号过程来帮助交易员了解订单流的变化，为最优执行问题提供了解决方案。 |
| [^9] | [On random number generators and practical market efficiency.](http://arxiv.org/abs/2305.17419) | 本研究利用随机数生成器来检测资本市场中信息的有效性，发现市场效率因年份和事件影响而变化，并与公司规模和投资者群体有关。 |
| [^10] | [The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations.](http://arxiv.org/abs/2210.15914) | 这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。 |
| [^11] | [Forecasting consumer confidence through semantic network analysis of online news.](http://arxiv.org/abs/2105.04900) | 本研究使用语义网络分析在线新闻对消费者信心的影响，结果表明该方法能够预测消费者对经济形势的判断，提供了一种补充方法来估计消费者信心。 |

# 详细

[^1]: 推进广告拍卖的现实性：实际见解与建模影响

    Advancing Ad Auction Realism: Practical Insights & Modeling Implications. (arXiv:2307.11732v1 [cs.LG])

    [http://arxiv.org/abs/2307.11732](http://arxiv.org/abs/2307.11732)

    本文提出了一个学习模型来模拟现实的在线广告拍卖环境，并发现在这样的环境中，使用"软底价"可以提高关键绩效指标，即使投标者来自相同的人群。

    

    本文提出了一个在线广告拍卖学习模型，允许考虑当代在线拍卖的四个关键现实特征：（1）广告槽可以根据用户的搜索查询具有不同的价值和点击率，（2）竞争广告商的数量和身份是不可观察的，并且在每次竞拍中会发生更改，（3）广告商仅接收到部分的汇总反馈，（4）付款规则只部分确定。我们将广告商建模为受对抗性赌博算法驱动的代理，独立于拍卖机制的复杂性。我们的目标是为了模拟广告商的行为，进行反事实分析、预测和推理。我们的研究结果表明，在这种更复杂的环境中，即使投标者来自相同的人群，"软底价"也可以提高关键绩效指标。我们进一步展示了如何从观察到的竞标中推断广告商价值分布，从而证实了该方法的实际功效。

    This paper proposes a learning model of online ad auctions that allows for the following four key realistic characteristics of contemporary online auctions: (1) ad slots can have different values and click-through rates depending on users' search queries, (2) the number and identity of competing advertisers are unobserved and change with each auction, (3) advertisers only receive partial, aggregated feedback, and (4) payment rules are only partially specified. We model advertisers as agents governed by an adversarial bandit algorithm, independent of auction mechanism intricacies. Our objective is to simulate the behavior of advertisers for counterfactual analysis, prediction, and inference purposes. Our findings reveal that, in such richer environments, "soft floors" can enhance key performance metrics even when bidders are drawn from the same population. We further demonstrate how to infer advertiser value distributions from observed bids, thereby affirming the practical efficacy of o
    
[^2]: 面向通用化的交易执行的强化学习方法

    Towards Generalizable Reinforcement Learning for Trade Execution. (arXiv:2307.11685v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.11685](http://arxiv.org/abs/2307.11685)

    本论文提出了一种面向通用化的交易执行的强化学习方法。研究表明，现有的强化学习方法存在过拟合问题，阻碍了实际应用。作者通过使用离线强化学习和动态上下文建模来解决过拟合问题，并提出了学习上下文的紧凑表示方法。

    

    优化的交易执行是在给定时间内以最低的交易成本卖出（或买入）给定资产的过程。最近，强化学习方法被应用于优化的交易执行，以从市场数据中学习更智能的策略。然而，我们发现许多现有的强化学习方法存在显著的过拟合问题，从而阻碍了它们的实际应用。在本文中，我们对优化的交易执行中的过拟合问题进行了广泛研究。首先，我们将优化的交易执行建模为带有动态上下文（ORDC）的离线强化学习问题，其中上下文表示不能受到交易策略影响并以离线方式收集的市场变量。在这个框架下，我们推导了泛化界限，并发现过拟合问题是由于离线环境中上下文空间巨大且上下文样本有限所导致的。因此，我们提出了学习上下文的紧凑表示来解决过拟合问题，可以通过...

    Optimized trade execution is to sell (or buy) a given amount of assets in a given time with the lowest possible trading cost. Recently, reinforcement learning (RL) has been applied to optimized trade execution to learn smarter policies from market data. However, we find that many existing RL methods exhibit considerable overfitting which prevents them from real deployment. In this paper, we provide an extensive study on the overfitting problem in optimized trade execution. First, we model the optimized trade execution as offline RL with dynamic context (ORDC), where the context represents market variables that cannot be influenced by the trading policy and are collected in an offline manner. Under this framework, we derive the generalization bound and find that the overfitting issue is caused by large context space and limited context samples in the offline setting. Accordingly, we propose to learn compact representations for context to address the overfitting problem, either by levera
    
[^3]: 评估小农户和家庭在农业和农村经济中的角色以及支持其可持续发展的措施

    Assessing the role of small farmers and households in agriculture and the rural economy and measures to support their sustainable development. (arXiv:2307.11683v1 [econ.GN])

    [http://arxiv.org/abs/2307.11683](http://arxiv.org/abs/2307.11683)

    该论文旨在评估小农户和家庭在农业和农村经济中的作用，以及提出支持其可持续发展的措施。同时，它还研究了如何减少乌克兰影子农业市场的规模和采取何种措施。

    

    经济部对于如何增加乌克兰合法登记的小型家庭农民的数量以及研究能够减少乌克兰影子农业市场规模的措施有兴趣和需求。在以上政治经济背景和需求的基础上，我们将分别进行可持续小规模（家庭）农业发展和探索减少乌克兰影子农业市场规模和措施的分析。

    The Ministry of Economy has an interest and demand in exploring how to increase the set of [legally registered] small family farmers in Ukraine and to examine more in details measures that could reduce the scale of the shadow agricultural market in Ukraine. Building upon the above political economy background and demand, we will be undertaking the analysis along the two separate but not totally independents streams of analysis, i.e. sustainable small scale (family) farming development and exploring the scale and measures for reducing the shadow agricultural market in Ukraine
    
[^4]: ESG声誉风险的重要性：基于社交媒体数据的事件研究

    ESG Reputation Risk Matters: An Event Study Based on Social Media Data. (arXiv:2307.11571v1 [econ.GN])

    [http://arxiv.org/abs/2307.11571](http://arxiv.org/abs/2307.11571)

    本研究探讨了股东对环境、社会和治理相关的声誉风险的反应，并通过社交媒体数据观察到ESG风险事件对相关资产回报的显著负面影响。

    

    本文研究了股东对环境、社会和治理相关的声誉风险（ESG风险）的反应，重点关注社交媒体的影响。我们使用了2016年至2022年间114百万条关于S&P100指数上市公司的推文数据集，提取了讨论ESG事项的对话。通过事件研究设计，我们将异常的推文活动高峰定义为与ESG风险有关的事件，并检查相关资产回报的相应变化。通过关注社交媒体，我们可以了解公众舆论和投资者情绪，而这些方面在单纯关注ESG争议新闻中并未捕捉到。据我们所知，我们的方法是首次将社交媒体的声誉影响与负面ESG争议新闻的实际成本明确分离开来。我们的结果表明，ESG风险事件的发生导致平均异常回报下降0.29%，在统计上是显著的。

    We investigate the response of shareholders to Environmental, Social, and Governance-related reputational risk (ESG-risk), focusing exclusively on the impact of social media. Using a dataset of 114 million tweets about firms listed on the S&P100 index between 2016 and 2022, we extract conversations discussing ESG matters. In an event study design, we define events as unusual spikes in message posting activity linked to ESG-risk, and we then examine the corresponding changes in the returns of related assets. By focusing on social media, we gain insight into public opinion and investor sentiment, an aspect not captured through ESG controversies news alone. To the best of our knowledge, our approach is the first to distinctly separate the reputational impact on social media from the physical costs associated with negative ESG controversy news. Our results show that the occurrence of an ESG-risk event leads to a statistically significant average reduction of 0.29% in abnormal returns. Furt
    
[^5]: 香港特殊医院病房在不确定性下的鲁棒性选址模型

    A Robust Site Selection Model under uncertainty for Special Hospital Wards in Hong Kong. (arXiv:2307.11508v1 [econ.GN])

    [http://arxiv.org/abs/2307.11508](http://arxiv.org/abs/2307.11508)

    本研究提出了针对香港特殊医院病房的选址问题的鲁棒模型，考虑了不确定性水平、不可行容忍度和可靠性水平，并采用了一种适用于不确定性的优化协议。

    

    本文提出了两个鲁棒性选址问题的模型，针对香港一家主要医院的病房。考虑了三个参数：不确定性水平、不可行容忍度和可靠性水平。然后，研究了两种不确定性，即对称和有界不确定性。因此，考虑了在不确定性下的调度问题，可以通过给定的概率分布函数来表示未知的问题因素。在这方面，Lin, Janak和Floudas（2004）引入了一种新开发的强优化协议。因此，计算机和化学工程领域已经发展了一种考虑通过给定概率分布表示的不确定性的方法。最后，我们的准确优化协议基于一个min-max框架，并且在应用于（MILP）问题时产生了一个具有对不确定数据免疫性的精确解决方案。

    This paper process two robust models for site selection problems for one of the major Hospitals in Hong Kong. Three parameters, namely, level of uncertainty, infeasibility tolerance as well as the level of reliability, are incorporated. Then, 2 kinds of uncertainty; that is, the symmetric and bounded uncertainties have been investigated. Therefore, the issue of scheduling under uncertainty has been considered wherein unknown problem factors could be illustrated via a given probability distribution function. In this regard, Lin, Janak, and Floudas (2004) introduced one of the newly developed strong optimisation protocols. Hence, computers as well as the chemical engineering [1069-1085] has been developed for considering uncertainty illustrated through a given probability distribution. Finally, our accurate optimisation protocol has been on the basis of a min-max framework and in a case of application to the (MILP) problems it produced a precise solution that has immunity to uncertain da
    
[^6]: 使用与价格相关的进入时间的最佳泡沫乘车：带有共同噪音的控制均值场博弈

    Optimal Bubble Riding with Price-dependent Entry: a Mean Field Game of Controls with Common Noise. (arXiv:2307.11340v1 [q-fin.MF])

    [http://arxiv.org/abs/2307.11340](http://arxiv.org/abs/2307.11340)

    本文扩展了唐派和王提出的最佳泡沫乘车模型，允许价格相关的进入时间。我们证明了带有共同噪音和随机进入时间的控制均值场博弈的存在性。共同噪音来自资产价格和外生泡沫破裂时间。

    

    本文通过允许价格相关的进入时间进一步扩展了唐派和王提出的最佳泡沫乘车模型。代理商通过个体进入阈值来表示他们对泡沫强度的信念。而泡沫的增长动力则来源于玩家的涌入。价格相关的进入自然地引导了一个带有共同噪音和随机进入时间的控制均值场博弈，我们提供了一个存在性结果。平衡通过首先解决弱形式下的离散化游戏，然后在极限中检验可测性性质来获得。本文中，共同噪音来自两个来源：所有代理商交易的资产价格，以及外生泡沫破裂时间，我们还通过渐进增大滤波来离散化并将其纳入模型。

    In this paper we further extend the optimal bubble riding model proposed by Tangpi and Wang by allowing for price-dependent entry times. Agents are characterized by their individual entry threshold that represents their belief in the strength of the bubble. Conversely, the growth dynamics of the bubble is fueled by the influx of players. Price-dependent entry naturally leads to a mean field game of controls with common noise and random entry time, for which we provide an existence result. The equilibrium is obtained by first solving discretized versions of the game in the weak formulation and then examining the measurability property in the limit. In this paper, the common noise comes from two sources: the price of the asset which all agents trade, and also the exogenous bubble burst time, which we also discretize and incorporate into the model via progressive enlargement of filtration.
    
[^7]: 模型与锡人之间——使用大语言模型研究AI对齐中的委托-代理问题的行为经济学研究

    Of Models and Tin Men -- a behavioural economics study of principal-agent problems in AI alignment using large-language models. (arXiv:2307.11137v1 [cs.AI])

    [http://arxiv.org/abs/2307.11137](http://arxiv.org/abs/2307.11137)

    本研究基于行为经济学角度，对使用大语言模型进行AI对齐中的委托-代理问题进行研究，发现现实世界中的AI安全问题不仅涉及设计者与代理之间的冲突，还涉及到多个代理之间的信息不对称与效用函数之间的错位。

    

    AI对齐通常被描述为一个设计者与人工智能代理之间的相互作用，设计者试图确保代理的行为与其目的一致，并且风险仅仅是由于设计者意图中的效用函数与代理的内部效用函数之间的意外错位而导致的冲突。然而，随着使用大语言模型（LLM）实例化的代理的出现，这种描述不能捕捉到AI安全的核心方面，因为现实世界中设计者与代理之间并没有一对一的对应关系，而且许多代理，无论是人工智能还是人类，都具有多样的价值观。因此，AI安全具有经济方面的问题，委托-代理问题可能会出现。

    AI Alignment is often presented as an interaction between a single designer and an artificial agent in which the designer attempts to ensure the agent's behavior is consistent with its purpose, and risks arise solely because of conflicts caused by inadvertent misalignment between the utility function intended by the designer and the resulting internal utility function of the agent. With the advent of agents instantiated with large-language models (LLMs), which are typically pre-trained, we argue this does not capture the essential aspects of AI safety because in the real world there is not a one-to-one correspondence between designer and agent, and the many agents, both artificial and human, have heterogeneous values. Therefore, there is an economic aspect to AI safety and the principal-agent problem is likely to arise. In a principal-agent problem conflict arises because of information asymmetry together with inherent misalignment between the utility of the agent and its principal, an
    
[^8]: 带有交易信号的最优执行和投机

    Optimal execution and speculation with trade signals. (arXiv:2306.00621v1 [q-fin.TR])

    [http://arxiv.org/abs/2306.00621](http://arxiv.org/abs/2306.00621)

    本文提出了一个价格冲击模型，基于市场的订单流动来推导市场的随机价格变化，提出了一个短期信号过程来帮助交易员了解订单流的变化，为最优执行问题提供了解决方案。

    

    我们提出了一个价格冲击模型，在这个模型中，价格的变化纯粹由市场的订单流动驱动。 市场订单的随机价格冲击和限价订单和市场订单的到达率是市场流动性过程的函数，该流动性过程反映了市场流动性的供需平衡。 限价订单和市场订单相互激发，使得流动性具有均值回归性质。我们使用 Meyers-$\sigma$-场的理论引入一个短期信号过程，从中交易员可以了解订单流的即将发生的变化。在这种情况下，我们研究了最优执行问题，并推导了其价值函数的Hamilton-Jacobi-Bellman（HJB）方程。 HJB方程经过数值求解后，我们演示了交易员如何使用信号来增强执行问题的性能并执行投机策略。

    We propose a price impact model where changes in prices are purely driven by the order flow in the market. The stochastic price impact of market orders and the arrival rates of limit and market orders are functions of the market liquidity process which reflects the balance of the demand and supply of liquidity. Limit and market orders mutually excite each other so that liquidity is mean reverting. We use the theory of Meyer-$\sigma$-fields to introduce a short-term signal process from which a trader learns about imminent changes in order flow. In this setting, we examine an optimal execution problem and derive the Hamilton--Jacobi--Bellman (HJB) equation for the value function. The HJB equation is solved numerically and we illustrate how the trader uses the signal to enhance the performance of execution problems and to execute speculative strategies.
    
[^9]: 随机数生成器与实际市场效率研究

    On random number generators and practical market efficiency. (arXiv:2305.17419v1 [q-fin.ST])

    [http://arxiv.org/abs/2305.17419](http://arxiv.org/abs/2305.17419)

    本研究利用随机数生成器来检测资本市场中信息的有效性，发现市场效率因年份和事件影响而变化，并与公司规模和投资者群体有关。

    

    现代主流金融理论基于有效市场假说，认为相关信息迅速纳入资产定价。先前的运筹研究文献中仅有少量研究利用为随机数生成器设计的检测方法来检查这些信息效率。将二元日收益视为硬件随机数发生器类比，重叠置换检验表明，这些时间序列具有特异性的循环模式。与以往的研究相反，我们将分析分为年度和公司层面两部分，并对纳斯达克上市公司进行长期效率研究，缓解交易噪声的影响，使市场能够真实地消化新信息。结果表明，信息效率在不同年份中有所不同，并反映了金融危机等大规模市场影响。我们还展示了信息效率与公司规模和投资者群体有关。

    Modern mainstream financial theory is underpinned by the efficient market hypothesis, which posits the rapid incorporation of relevant information into asset pricing. Limited prior studies in the operational research literature have investigated the use of tests designed for random number generators to check for these informational efficiencies. Treating binary daily returns as a hardware random number generator analogue, tests of overlapping permutations have indicated that these time series feature idiosyncratic recurrent patterns. Contrary to prior studies, we split our analysis into two streams at the annual and company level, and investigate longer-term efficiency over a larger time frame for Nasdaq-listed public companies to diminish the effects of trading noise and allow the market to realistically digest new information. Our results demonstrate that information efficiency varies across different years and reflects large-scale market impacts such as financial crises. We also sho
    
[^10]: 移民、移民者和当地人在欧洲知识聚集形成中的历史角色

    The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations. (arXiv:2210.15914v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2210.15914](http://arxiv.org/abs/2210.15914)

    这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。

    

    移民是不是让巴黎成为了艺术圣地，维也纳成为了古典音乐的灯塔？还是他们的崛起纯粹是当地人的结果？在这里，我们使用了关于22000多名生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。我们发现，一个地区在某种活动（基于著名物理学家、画家等的出生）发展或保持专业化的概率随着对该活动具有知识的移民和对相关活动具有知识的移民的存在而增加。相比之下，我们并没有找到有力的证据表明当地人具有相关知识的存在解释了进入和/或退出。我们通过考虑任何特定地点-时期-活动因素（例如吸引科学家的新大学的存在）的固定效应模型来解决一些内生性问题。

    Did migrants make Paris a Mecca for the arts and Vienna a beacon of classical music? Or was their rise a pure consequence of local actors? Here, we use data on more than 22,000 historical individuals born between the years 1000 and 2000 to estimate the contribution of famous immigrants, emigrants, and locals to the knowledge specializations of European regions. We find that the probability that a region develops or keeps specialization in an activity (based on the birth of famous physicists, painters, etc.) grows with both, the presence of immigrants with knowledge on that activity and immigrants with knowledge in related activities. In contrast, we do not find robust evidence that the presence of locals with related knowledge explains entries and/or exits. We address some endogeneity concerns using fixed-effects models considering any location-period-activity specific factors (e.g. the presence of a new university attracting scientists).
    
[^11]: 通过在线新闻的语义网络分析预测消费者信心

    Forecasting consumer confidence through semantic network analysis of online news. (arXiv:2105.04900v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2105.04900](http://arxiv.org/abs/2105.04900)

    本研究使用语义网络分析在线新闻对消费者信心的影响，结果表明该方法能够预测消费者对经济形势的判断，提供了一种补充方法来估计消费者信心。

    

    本研究通过语义网络分析研究在线新闻对社会经济消费者态度的影响。使用覆盖四年的意大利媒体上的超过180万篇在线文章，我们计算特定经济相关关键词的语义重要性，以确定文章中出现的词语是否能够预测消费者对经济形势和消费者信心指数的判断。我们运用创新方法分析大规模文本数据，结合了文本挖掘和社会网络分析的方法和工具。结果显示，该指标对于判断当前家庭和国家情况具有较强的预测能力。我们的指标为消费者信心的估计提供了一种补充方法，减轻了传统基于调查的方法的局限性。

    This research studies the impact of online news on social and economic consumer perceptions through semantic network analysis. Using over 1.8 million online articles on Italian media covering four years, we calculate the semantic importance of specific economic-related keywords to see if words appearing in the articles could anticipate consumers' judgments about the economic situation and the Consumer Confidence Index. We use an innovative approach to analyze big textual data, combining methods and tools of text mining and social network analysis. Results show a strong predictive power for the judgments about the current households and national situation. Our indicator offers a complementary approach to estimating consumer confidence, lessening the limitations of traditional survey-based methods.
    

