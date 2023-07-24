# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Advancing Ad Auction Realism: Practical Insights & Modeling Implications.](http://arxiv.org/abs/2307.11732) | 本文提出了一个学习模型来模拟现实的在线广告拍卖环境，并发现在这样的环境中，使用"软底价"可以提高关键绩效指标，即使投标者来自相同的人群。 |
| [^2] | [Assessing the role of small farmers and households in agriculture and the rural economy and measures to support their sustainable development.](http://arxiv.org/abs/2307.11683) | 该论文旨在评估小农户和家庭在农业和农村经济中的作用，以及提出支持其可持续发展的措施。同时，它还研究了如何减少乌克兰影子农业市场的规模和采取何种措施。 |
| [^3] | [ESG Reputation Risk Matters: An Event Study Based on Social Media Data.](http://arxiv.org/abs/2307.11571) | 本研究探讨了股东对环境、社会和治理相关的声誉风险的反应，并通过社交媒体数据观察到ESG风险事件对相关资产回报的显著负面影响。 |
| [^4] | [A Robust Site Selection Model under uncertainty for Special Hospital Wards in Hong Kong.](http://arxiv.org/abs/2307.11508) | 本研究提出了针对香港特殊医院病房的选址问题的鲁棒模型，考虑了不确定性水平、不可行容忍度和可靠性水平，并采用了一种适用于不确定性的优化协议。 |
| [^5] | [Functional Differencing in Networks.](http://arxiv.org/abs/2307.11484) | 本文提出了一种在网络中应用功能差异方法的估计方法，该方法不受网络稠密程度和异质性形式约束，适用于网络经济实证分析。 |
| [^6] | [Of Models and Tin Men -- a behavioural economics study of principal-agent problems in AI alignment using large-language models.](http://arxiv.org/abs/2307.11137) | 本研究基于行为经济学角度，对使用大语言模型进行AI对齐中的委托-代理问题进行研究，发现现实世界中的AI安全问题不仅涉及设计者与代理之间的冲突，还涉及到多个代理之间的信息不对称与效用函数之间的错位。 |
| [^7] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |
| [^8] | [The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations.](http://arxiv.org/abs/2210.15914) | 这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。 |
| [^9] | [A Comparison of Group Criticality Notions for Simple Games.](http://arxiv.org/abs/2207.03565) | 本文比较了简单游戏中基于合作和分裂的关键程度概念和差分关键性概念，提出了满足扩展强单调性、不分配权力给哑元玩家和自由骑手的新概念，并介绍了一个群体机会测试来统一两种观点。 |
| [^10] | [Agreement and Statistical Efficiency in Bayesian Perception Models.](http://arxiv.org/abs/2205.11561) | 本文研究了贝叶斯感知模型在连通网络上的重复沟通中的渐近行为，发现尽管个体代理人不是经典意义上的效用最大化者，最终他们会达成一致，并且极限后验分布是贝叶斯最优的。 |
| [^11] | [Forecasting consumer confidence through semantic network analysis of online news.](http://arxiv.org/abs/2105.04900) | 本研究使用语义网络分析在线新闻对消费者信心的影响，结果表明该方法能够预测消费者对经济形势的判断，提供了一种补充方法来估计消费者信心。 |

# 详细

[^1]: 推进广告拍卖的现实性：实际见解与建模影响

    Advancing Ad Auction Realism: Practical Insights & Modeling Implications. (arXiv:2307.11732v1 [cs.LG])

    [http://arxiv.org/abs/2307.11732](http://arxiv.org/abs/2307.11732)

    本文提出了一个学习模型来模拟现实的在线广告拍卖环境，并发现在这样的环境中，使用"软底价"可以提高关键绩效指标，即使投标者来自相同的人群。

    

    本文提出了一个在线广告拍卖学习模型，允许考虑当代在线拍卖的四个关键现实特征：（1）广告槽可以根据用户的搜索查询具有不同的价值和点击率，（2）竞争广告商的数量和身份是不可观察的，并且在每次竞拍中会发生更改，（3）广告商仅接收到部分的汇总反馈，（4）付款规则只部分确定。我们将广告商建模为受对抗性赌博算法驱动的代理，独立于拍卖机制的复杂性。我们的目标是为了模拟广告商的行为，进行反事实分析、预测和推理。我们的研究结果表明，在这种更复杂的环境中，即使投标者来自相同的人群，"软底价"也可以提高关键绩效指标。我们进一步展示了如何从观察到的竞标中推断广告商价值分布，从而证实了该方法的实际功效。

    This paper proposes a learning model of online ad auctions that allows for the following four key realistic characteristics of contemporary online auctions: (1) ad slots can have different values and click-through rates depending on users' search queries, (2) the number and identity of competing advertisers are unobserved and change with each auction, (3) advertisers only receive partial, aggregated feedback, and (4) payment rules are only partially specified. We model advertisers as agents governed by an adversarial bandit algorithm, independent of auction mechanism intricacies. Our objective is to simulate the behavior of advertisers for counterfactual analysis, prediction, and inference purposes. Our findings reveal that, in such richer environments, "soft floors" can enhance key performance metrics even when bidders are drawn from the same population. We further demonstrate how to infer advertiser value distributions from observed bids, thereby affirming the practical efficacy of o
    
[^2]: 评估小农户和家庭在农业和农村经济中的角色以及支持其可持续发展的措施

    Assessing the role of small farmers and households in agriculture and the rural economy and measures to support their sustainable development. (arXiv:2307.11683v1 [econ.GN])

    [http://arxiv.org/abs/2307.11683](http://arxiv.org/abs/2307.11683)

    该论文旨在评估小农户和家庭在农业和农村经济中的作用，以及提出支持其可持续发展的措施。同时，它还研究了如何减少乌克兰影子农业市场的规模和采取何种措施。

    

    经济部对于如何增加乌克兰合法登记的小型家庭农民的数量以及研究能够减少乌克兰影子农业市场规模的措施有兴趣和需求。在以上政治经济背景和需求的基础上，我们将分别进行可持续小规模（家庭）农业发展和探索减少乌克兰影子农业市场规模和措施的分析。

    The Ministry of Economy has an interest and demand in exploring how to increase the set of [legally registered] small family farmers in Ukraine and to examine more in details measures that could reduce the scale of the shadow agricultural market in Ukraine. Building upon the above political economy background and demand, we will be undertaking the analysis along the two separate but not totally independents streams of analysis, i.e. sustainable small scale (family) farming development and exploring the scale and measures for reducing the shadow agricultural market in Ukraine
    
[^3]: ESG声誉风险的重要性：基于社交媒体数据的事件研究

    ESG Reputation Risk Matters: An Event Study Based on Social Media Data. (arXiv:2307.11571v1 [econ.GN])

    [http://arxiv.org/abs/2307.11571](http://arxiv.org/abs/2307.11571)

    本研究探讨了股东对环境、社会和治理相关的声誉风险的反应，并通过社交媒体数据观察到ESG风险事件对相关资产回报的显著负面影响。

    

    本文研究了股东对环境、社会和治理相关的声誉风险（ESG风险）的反应，重点关注社交媒体的影响。我们使用了2016年至2022年间114百万条关于S&P100指数上市公司的推文数据集，提取了讨论ESG事项的对话。通过事件研究设计，我们将异常的推文活动高峰定义为与ESG风险有关的事件，并检查相关资产回报的相应变化。通过关注社交媒体，我们可以了解公众舆论和投资者情绪，而这些方面在单纯关注ESG争议新闻中并未捕捉到。据我们所知，我们的方法是首次将社交媒体的声誉影响与负面ESG争议新闻的实际成本明确分离开来。我们的结果表明，ESG风险事件的发生导致平均异常回报下降0.29%，在统计上是显著的。

    We investigate the response of shareholders to Environmental, Social, and Governance-related reputational risk (ESG-risk), focusing exclusively on the impact of social media. Using a dataset of 114 million tweets about firms listed on the S&P100 index between 2016 and 2022, we extract conversations discussing ESG matters. In an event study design, we define events as unusual spikes in message posting activity linked to ESG-risk, and we then examine the corresponding changes in the returns of related assets. By focusing on social media, we gain insight into public opinion and investor sentiment, an aspect not captured through ESG controversies news alone. To the best of our knowledge, our approach is the first to distinctly separate the reputational impact on social media from the physical costs associated with negative ESG controversy news. Our results show that the occurrence of an ESG-risk event leads to a statistically significant average reduction of 0.29% in abnormal returns. Furt
    
[^4]: 香港特殊医院病房在不确定性下的鲁棒性选址模型

    A Robust Site Selection Model under uncertainty for Special Hospital Wards in Hong Kong. (arXiv:2307.11508v1 [econ.GN])

    [http://arxiv.org/abs/2307.11508](http://arxiv.org/abs/2307.11508)

    本研究提出了针对香港特殊医院病房的选址问题的鲁棒模型，考虑了不确定性水平、不可行容忍度和可靠性水平，并采用了一种适用于不确定性的优化协议。

    

    本文提出了两个鲁棒性选址问题的模型，针对香港一家主要医院的病房。考虑了三个参数：不确定性水平、不可行容忍度和可靠性水平。然后，研究了两种不确定性，即对称和有界不确定性。因此，考虑了在不确定性下的调度问题，可以通过给定的概率分布函数来表示未知的问题因素。在这方面，Lin, Janak和Floudas（2004）引入了一种新开发的强优化协议。因此，计算机和化学工程领域已经发展了一种考虑通过给定概率分布表示的不确定性的方法。最后，我们的准确优化协议基于一个min-max框架，并且在应用于（MILP）问题时产生了一个具有对不确定数据免疫性的精确解决方案。

    This paper process two robust models for site selection problems for one of the major Hospitals in Hong Kong. Three parameters, namely, level of uncertainty, infeasibility tolerance as well as the level of reliability, are incorporated. Then, 2 kinds of uncertainty; that is, the symmetric and bounded uncertainties have been investigated. Therefore, the issue of scheduling under uncertainty has been considered wherein unknown problem factors could be illustrated via a given probability distribution function. In this regard, Lin, Janak, and Floudas (2004) introduced one of the newly developed strong optimisation protocols. Hence, computers as well as the chemical engineering [1069-1085] has been developed for considering uncertainty illustrated through a given probability distribution. Finally, our accurate optimisation protocol has been on the basis of a min-max framework and in a case of application to the (MILP) problems it produced a precise solution that has immunity to uncertain da
    
[^5]: 网络中的功能差异

    Functional Differencing in Networks. (arXiv:2307.11484v1 [econ.EM])

    [http://arxiv.org/abs/2307.11484](http://arxiv.org/abs/2307.11484)

    本文提出了一种在网络中应用功能差异方法的估计方法，该方法不受网络稠密程度和异质性形式约束，适用于网络经济实证分析。

    

    经济交往通常发生在网络中，其中具有异质性的代理人（如工人或企业）进行排序和生产。然而，大多数现有的估计方法要么要求网络密集，与许多实证网络相矛盾，要么要求限制异质性的形式和网络形成过程。我们展示了如何将Bonhomme（2012）在面板数据背景下引入的功能差异方法应用于网络设置中，从而导出模型参数和平均效应上的矩约束。这些约束与异质性的形式无关，适用于密集和稀疏网络。我们借助线性和非线性的雇主-员工数据模型来说明分析方法，这与Abowd，Kramarz和Margolis（1999）提出的模型相似。

    Economic interactions often occur in networks where heterogeneous agents (such as workers or firms) sort and produce. However, most existing estimation approaches either require the network to be dense, which is at odds with many empirical networks, or they require restricting the form of heterogeneity and the network formation process. We show how the functional differencing approach introduced by Bonhomme (2012) in the context of panel data, can be applied in network settings to derive moment restrictions on model parameters and average effects. Those restrictions are valid irrespective of the form of heterogeneity, and they hold in both dense and sparse networks. We illustrate the analysis with linear and nonlinear models of matched employer-employee data, in the spirit of the model introduced by Abowd, Kramarz, and Margolis (1999).
    
[^6]: 模型与锡人之间——使用大语言模型研究AI对齐中的委托-代理问题的行为经济学研究

    Of Models and Tin Men -- a behavioural economics study of principal-agent problems in AI alignment using large-language models. (arXiv:2307.11137v1 [cs.AI])

    [http://arxiv.org/abs/2307.11137](http://arxiv.org/abs/2307.11137)

    本研究基于行为经济学角度，对使用大语言模型进行AI对齐中的委托-代理问题进行研究，发现现实世界中的AI安全问题不仅涉及设计者与代理之间的冲突，还涉及到多个代理之间的信息不对称与效用函数之间的错位。

    

    AI对齐通常被描述为一个设计者与人工智能代理之间的相互作用，设计者试图确保代理的行为与其目的一致，并且风险仅仅是由于设计者意图中的效用函数与代理的内部效用函数之间的意外错位而导致的冲突。然而，随着使用大语言模型（LLM）实例化的代理的出现，这种描述不能捕捉到AI安全的核心方面，因为现实世界中设计者与代理之间并没有一对一的对应关系，而且许多代理，无论是人工智能还是人类，都具有多样的价值观。因此，AI安全具有经济方面的问题，委托-代理问题可能会出现。

    AI Alignment is often presented as an interaction between a single designer and an artificial agent in which the designer attempts to ensure the agent's behavior is consistent with its purpose, and risks arise solely because of conflicts caused by inadvertent misalignment between the utility function intended by the designer and the resulting internal utility function of the agent. With the advent of agents instantiated with large-language models (LLMs), which are typically pre-trained, we argue this does not capture the essential aspects of AI safety because in the real world there is not a one-to-one correspondence between designer and agent, and the many agents, both artificial and human, have heterogeneous values. Therefore, there is an economic aspect to AI safety and the principal-agent problem is likely to arise. In a principal-agent problem conflict arises because of information asymmetry together with inherent misalignment between the utility of the agent and its principal, an
    
[^7]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    
[^8]: 移民、移民者和当地人在欧洲知识聚集形成中的历史角色

    The Role of Immigrants, Emigrants, and Locals in the Historical Formation of European Knowledge Agglomerations. (arXiv:2210.15914v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2210.15914](http://arxiv.org/abs/2210.15914)

    这项研究通过使用超过22000位生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。研究发现，对某种活动具有知识的移民和对相关活动具有知识的移民的存在可以增加一个地区发展或保持专业化的概率，而当地人的相关知识则不能解释进入和/或退出。

    

    移民是不是让巴黎成为了艺术圣地，维也纳成为了古典音乐的灯塔？还是他们的崛起纯粹是当地人的结果？在这里，我们使用了关于22000多名生于1000年至2000年之间的历史人物的数据，估计了著名移民、移民者和当地人对欧洲地区知识专业化的贡献。我们发现，一个地区在某种活动（基于著名物理学家、画家等的出生）发展或保持专业化的概率随着对该活动具有知识的移民和对相关活动具有知识的移民的存在而增加。相比之下，我们并没有找到有力的证据表明当地人具有相关知识的存在解释了进入和/或退出。我们通过考虑任何特定地点-时期-活动因素（例如吸引科学家的新大学的存在）的固定效应模型来解决一些内生性问题。

    Did migrants make Paris a Mecca for the arts and Vienna a beacon of classical music? Or was their rise a pure consequence of local actors? Here, we use data on more than 22,000 historical individuals born between the years 1000 and 2000 to estimate the contribution of famous immigrants, emigrants, and locals to the knowledge specializations of European regions. We find that the probability that a region develops or keeps specialization in an activity (based on the birth of famous physicists, painters, etc.) grows with both, the presence of immigrants with knowledge on that activity and immigrants with knowledge in related activities. In contrast, we do not find robust evidence that the presence of locals with related knowledge explains entries and/or exits. We address some endogeneity concerns using fixed-effects models considering any location-period-activity specific factors (e.g. the presence of a new university attracting scientists).
    
[^9]: 简单游戏中群体重要性概念的比较

    A Comparison of Group Criticality Notions for Simple Games. (arXiv:2207.03565v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2207.03565](http://arxiv.org/abs/2207.03565)

    本文比较了简单游戏中基于合作和分裂的关键程度概念和差分关键性概念，提出了满足扩展强单调性、不分配权力给哑元玩家和自由骑手的新概念，并介绍了一个群体机会测试来统一两种观点。

    

    我们基于玩家之间的合作（或分裂）来定义简单单调博弈中玩家的关键程度概念，要求所有参与者都具有重要角色。我们将其与Beisbart提出的差分关键性概念进行比较，后者将权力定义为其他玩家留下的机会。我们证明了我们的提议满足Young引入的强单调性的扩展，不会为哑元玩家和自由骑手分配任何权力，并且可以从最小的获胜和阻塞联盟中轻松计算。我们的分析表明，迄今为止定义的群体关键性度量无法衡量重要的玩家，同时仅保持为机会度量。我们提出了一个群体机会测试来协调两种观点。

    We define a notion of the criticality of a player for simple monotone games based on cooperation with other players, either to form a winning coalition or to break a winning one, with an essential role for all the players involved. We compare it with the notion of differential criticality given by Beisbart that measures power as the opportunity left by other players.  We prove that our proposal satisfies an extension of the strong monotonicity introduced by Young, it assigns no power to dummy players and free riders, and it can easily be computed from the minimal winning and blocking coalitions. Our analysis shows that the measures of group criticality defined so far cannot weigh essential players while only remaining an opportunity measure. We propose a group opportunity test to reconcile the two views.
    
[^10]: Bayesian感知模型中的一致性和统计效率

    Agreement and Statistical Efficiency in Bayesian Perception Models. (arXiv:2205.11561v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2205.11561](http://arxiv.org/abs/2205.11561)

    本文研究了贝叶斯感知模型在连通网络上的重复沟通中的渐近行为，发现尽管个体代理人不是经典意义上的效用最大化者，最终他们会达成一致，并且极限后验分布是贝叶斯最优的。

    

    自1970年代以来，经济学领域一直研究贝叶斯群体学习模型，最近在计算语言学方面也进行了研究。经济学模型假设代理人在沟通和行动中最大化效用。然而，这些经济学模型无法解释在许多实验研究中观察到的 "概率匹配" 现象。为了解决这些观察结果，引入了一些不完全符合经济学效用最大化框架的贝叶斯模型。在这些模型中，个体在沟通中从其后验分布中进行抽样。本研究探讨了这种模型在连通网络上进行重复沟通时的渐近行为。令人惊讶的是，尽管个体代理人在经典意义上并不是效用最大化者，但我们证明他们最终会达成一致，并且进一步证明极限后验分布是贝叶斯最优的。

    Bayesian models of group learning are studied in Economics since the 1970s and more recently in computational linguistics. The models from Economics postulate that agents maximize utility in their communication and actions. The Economics models do not explain the ``probability matching" phenomena that are observed in many experimental studies. To address these observations, Bayesian models that do not formally fit into the economic utility maximization framework were introduced. In these models individuals sample from their posteriors in communication. In this work we study the asymptotic behavior of such models on connected networks with repeated communication. Perhaps surprisingly, despite the fact that individual agents are not utility maximizers in the classical sense, we establish that the individuals ultimately agree and furthermore show that the limiting posterior is Bayes optimal.
    
[^11]: 通过在线新闻的语义网络分析预测消费者信心

    Forecasting consumer confidence through semantic network analysis of online news. (arXiv:2105.04900v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2105.04900](http://arxiv.org/abs/2105.04900)

    本研究使用语义网络分析在线新闻对消费者信心的影响，结果表明该方法能够预测消费者对经济形势的判断，提供了一种补充方法来估计消费者信心。

    

    本研究通过语义网络分析研究在线新闻对社会经济消费者态度的影响。使用覆盖四年的意大利媒体上的超过180万篇在线文章，我们计算特定经济相关关键词的语义重要性，以确定文章中出现的词语是否能够预测消费者对经济形势和消费者信心指数的判断。我们运用创新方法分析大规模文本数据，结合了文本挖掘和社会网络分析的方法和工具。结果显示，该指标对于判断当前家庭和国家情况具有较强的预测能力。我们的指标为消费者信心的估计提供了一种补充方法，减轻了传统基于调查的方法的局限性。

    This research studies the impact of online news on social and economic consumer perceptions through semantic network analysis. Using over 1.8 million online articles on Italian media covering four years, we calculate the semantic importance of specific economic-related keywords to see if words appearing in the articles could anticipate consumers' judgments about the economic situation and the Consumer Confidence Index. We use an innovative approach to analyze big textual data, combining methods and tools of text mining and social network analysis. Results show a strong predictive power for the judgments about the current households and national situation. Our indicator offers a complementary approach to estimating consumer confidence, lessening the limitations of traditional survey-based methods.
    

