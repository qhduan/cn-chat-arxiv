# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Composite Quantile Factor Models.](http://arxiv.org/abs/2308.02450) | 该论文介绍了一种在高维面板数据中进行因子分析的新方法，即复合分位数因子模型。其创新之处在于在不同分位数上估计因子和因子载荷，提高了估计结果的适应性，并引入了一种一致选择因子数的信息准则。模拟结果和实证研究表明，该方法在非正态分布下具有良好的性质。 |
| [^2] | [Matrix Completion When Missing Is Not at Random and Its Applications in Causal Panel Data Models.](http://arxiv.org/abs/2308.02364) | 本文提出了一个矩阵补全的推断框架，可以处理非随机缺失且无需强信号，方法包括将缺失条目分组并应用核范数正则化进行估计。研究受到Tick Size Pilot Program的启发，该实验评估了扩大Tick Size对股票市场品质的影响。 |
| [^3] | [Game theoretic foundations of the Gately power measure for directed networks.](http://arxiv.org/abs/2308.02274) | 该论文基于合作博弈中的Gately值，引入了一种新的网络中心性度量方法，用于衡量有向网络的权力分配并探讨了其特性和与其他衡量方法的关系。 |
| [^4] | [Should we trust web-scraped data?.](http://arxiv.org/abs/2308.02231) | 本论文指出天真的网络抓取程序可能导致收集数据中的抽样偏差，并描述了来源于网络内容易变性、个性化和未索引的抽样偏差。通过例子说明了抽样偏差的普遍性和程度，并提供了克服抽样偏差的建议。 |
| [^5] | [A Non-Parametric Test of Risk Aversion.](http://arxiv.org/abs/2308.02083) | 本研究提出了一种检验期望效用和凹凸性的简单方法，结果发现几乎没有支持这两个模型。此外，研究还证明使用流行的多价格清单方法测量风险规避是不合适的，由于参数错误导致了风险规避高普遍性的现象。 |
| [^6] | [Human and Machine Intelligence in n-Person Games with Partial Knowledge.](http://arxiv.org/abs/2302.13937) | 提出了有限知识下n人博弈的框架，引入了“游戏智能”机制和“防作弊性”概念，GI机制可以实际评估玩家的智能，应用广泛。 |
| [^7] | [When do Default Nudges Work?.](http://arxiv.org/abs/2301.08797) | 通过在瑞典Covid-19疫苗推出中的区域变化，研究了对激励有差异的群体的提示效果，结果显示提示对于个人无意义的选择更有效。 |
| [^8] | [Learning from Viral Content.](http://arxiv.org/abs/2210.01267) | 本文研究了社交媒体上的学习，发现向用户展示病毒性故事可以增加信息汇集，但也可能导致大多数分享的故事是错误的稳定状态。这些误导性的稳定状态会自我维持，对平台设计和鲁棒性产生多种后果。 |
| [^9] | [SoK: Blockchain Decentralization.](http://arxiv.org/abs/2205.04256) | 该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。 |
| [^10] | [Pigeonhole Design: Balancing Sequential Experiments from an Online Matching Perspective.](http://arxiv.org/abs/2201.12936) | 本研究研究了一种新颖的在线实验设计问题，称为“在线阻塞问题”，旨在平衡顺序到达的具有异质协变信息的实验对象，以最小化总差异。作者提出了一种名为“鸽巢设计”的方法来解决该问题。 |
| [^11] | [A Computational Approach to Identification of Treatment Effects for Policy Evaluation.](http://arxiv.org/abs/2009.13861) | 本文研究了一种计算方法，能够在存在未观察到的异质性和仅有二值工具变量的情况下，推广局部治疗效应到不同的反事实情境，并且提出了一个新的框架，可以系统地计算各种与政策相关的治疗参数的尖锐非参数界限。 |

# 详细

[^1]: 复合分位数因子模型

    Composite Quantile Factor Models. (arXiv:2308.02450v1 [econ.EM])

    [http://arxiv.org/abs/2308.02450](http://arxiv.org/abs/2308.02450)

    该论文介绍了一种在高维面板数据中进行因子分析的新方法，即复合分位数因子模型。其创新之处在于在不同分位数上估计因子和因子载荷，提高了估计结果的适应性，并引入了一种一致选择因子数的信息准则。模拟结果和实证研究表明，该方法在非正态分布下具有良好的性质。

    

    本文介绍了一种在高维面板数据中进行因子分析的方法，即复合分位数因子模型。我们提出了在不同分位数上估计因子和因子载荷的方法，使得估计结果能够更好地适应不同分位数下数据的特征，并且仍然能够对数据的均值进行建模。我们推导了估计的因子和因子载荷的极限分布，并讨论了一种一致选择因子数的信息准则。模拟结果表明，所提出的估计器和信息准则对于几种非正态分布具有良好的有限样本性质。我们还对246个季度宏观经济变量的因子分析进行了实证研究，并开发了一个名为cqrfactor的伴随R包。

    This paper introduces the method of composite quantile factor model for factor analysis in high-dimensional panel data. We propose to estimate the factors and factor loadings across different quantiles of the data, allowing the estimates to better adapt to features of the data at different quantiles while still modeling the mean of the data. We develop the limiting distribution of the estimated factors and factor loadings, and an information criterion for consistent factor number selection is also discussed. Simulations show that the proposed estimator and the information criterion have good finite sample properties for several non-normal distributions under consideration. We also consider an empirical study on the factor analysis for 246 quarterly macroeconomic variables. A companion R package cqrfactor is developed.
    
[^2]: 矩阵补全: 非随机缺失及其在因果面板数据模型中的应用

    Matrix Completion When Missing Is Not at Random and Its Applications in Causal Panel Data Models. (arXiv:2308.02364v1 [stat.ME])

    [http://arxiv.org/abs/2308.02364](http://arxiv.org/abs/2308.02364)

    本文提出了一个矩阵补全的推断框架，可以处理非随机缺失且无需强信号，方法包括将缺失条目分组并应用核范数正则化进行估计。研究受到Tick Size Pilot Program的启发，该实验评估了扩大Tick Size对股票市场品质的影响。

    

    本文在缺失数据非随机的情况下，开发了一个推断框架来进行矩阵补全，并且无需强信号的要求。我们的方法基于观察到，如果缺失条目的数量相对于面板的大小足够小，即使缺失是非随机的，也可以很好地估计它们。利用这个事实，我们将缺失的条目分成较小的组，并通过核范数正则化来估计每组。此外，我们还证明了，在适当的去偏估计下，我们提出的估计量即使对于相当弱的信号也是渐近正态的。我们的研究受到了最近关于Tick Size Pilot Program的研究的启发，这是一项由美国证券交易委员会（SEC）从2016年到2018年进行的实验，旨在评估扩大股票最小变动价位（Tick Size）对市场品质的影响。而以往的研究都是基于传统的回归或差分法，假设处理效应在不变的情况下进行的。

    This paper develops an inferential framework for matrix completion when missing is not at random and without the requirement of strong signals. Our development is based on the observation that if the number of missing entries is small enough compared to the panel size, then they can be estimated well even when missing is not at random. Taking advantage of this fact, we divide the missing entries into smaller groups and estimate each group via nuclear norm regularization. In addition, we show that with appropriate debiasing, our proposed estimate is asymptotically normal even for fairly weak signals. Our work is motivated by recent research on the Tick Size Pilot Program, an experiment conducted by the Security and Exchange Commission (SEC) to evaluate the impact of widening the tick size on the market quality of stocks from 2016 to 2018. While previous studies were based on traditional regression or difference-in-difference methods by assuming that the treatment effect is invariant wit
    
[^3]: 基于Gately博弈论的有向网络Gately能量衡量的基础

    Game theoretic foundations of the Gately power measure for directed networks. (arXiv:2308.02274v1 [cs.GT])

    [http://arxiv.org/abs/2308.02274](http://arxiv.org/abs/2308.02274)

    该论文基于合作博弈中的Gately值，引入了一种新的网络中心性度量方法，用于衡量有向网络的权力分配并探讨了其特性和与其他衡量方法的关系。

    

    我们引入了一种新的网络中心性度量，基于可转移效用的合作博弈中的Gately值。有向网络被解释为代表玩家之间的控制或权威关系 - 构成一个等级网络。层级网络的权力分配可以通过TU博弈来表示。我们研究了这个TU表示的特性，并研究了由此产生的Gately能量衡量中的Gately值。我们确定了何时Gately衡量为核心能量衡量，研究了Gately与β-衡量的关系，并构建了Gately衡量的公理化。

    We introduce a new network centrality measure founded on the Gately value for cooperative games with transferable utilities. A directed network is interpreted as representing control or authority relations between players--constituting a hierarchical network. The power distribution of a hierarchical network can be represented through a TU-game. We investigate the properties of this TU-representation and investigate the Gately value of the TU-representation resulting in the Gately power measure. We establish when the Gately measure is a Core power gauge, investigate the relationship of the Gately with the $\beta$-measure, and construct an axiomatisation of the Gately measure.
    
[^4]: 我们应该相信网络抓取的数据吗？

    Should we trust web-scraped data?. (arXiv:2308.02231v1 [econ.GN])

    [http://arxiv.org/abs/2308.02231](http://arxiv.org/abs/2308.02231)

    本论文指出天真的网络抓取程序可能导致收集数据中的抽样偏差，并描述了来源于网络内容易变性、个性化和未索引的抽样偏差。通过例子说明了抽样偏差的普遍性和程度，并提供了克服抽样偏差的建议。

    

    实证研究人员越来越多地采用计量经济学和机器学习方法，导致了对一种数据收集方法的广泛使用：网络抓取。网络抓取指的是使用自动化计算机程序访问网站并下载其内容。本文的主要论点是，天真的网络抓取程序可能会导致收集数据中的抽样偏差。本文描述了网络抓取数据中的三种抽样偏差来源。更具体地说，抽样偏差源于网络内容的易变性（即可能发生变化）、个性化（即根据请求特征呈现）和未索引（即人口登记簿的丰富性）。通过一系列例子，我说明了抽样偏差的普遍性和程度。为了支持研究人员和审稿人，本文提供了关于对网络抓取数据的抽样偏差进行预期、检测和克服的建议。

    The increasing adoption of econometric and machine-learning approaches by empirical researchers has led to a widespread use of one data collection method: web scraping. Web scraping refers to the use of automated computer programs to access websites and download their content. The key argument of this paper is that na\"ive web scraping procedures can lead to sampling bias in the collected data. This article describes three sources of sampling bias in web-scraped data. More specifically, sampling bias emerges from web content being volatile (i.e., being subject to change), personalized (i.e., presented in response to request characteristics), and unindexed (i.e., abundance of a population register). In a series of examples, I illustrate the prevalence and magnitude of sampling bias. To support researchers and reviewers, this paper provides recommendations on anticipating, detecting, and overcoming sampling bias in web-scraped data.
    
[^5]: 风险规避的非参数检验

    A Non-Parametric Test of Risk Aversion. (arXiv:2308.02083v1 [econ.GN])

    [http://arxiv.org/abs/2308.02083](http://arxiv.org/abs/2308.02083)

    本研究提出了一种检验期望效用和凹凸性的简单方法，结果发现几乎没有支持这两个模型。此外，研究还证明使用流行的多价格清单方法测量风险规避是不合适的，由于参数错误导致了风险规避高普遍性的现象。

    

    在经济学中，通过期望效用范式内的凹凸伯努利效用来建模风险规避。我们提出了一种简单的期望效用和凹凸性的测试方法。我们发现很少支持任何一种模型：只有30%的选择符合凹凸效用，只有72名受试者中的两人符合期望效用，而其中只有一人符合经济学中的风险规避模型。我们的结果与使用流行的多价格清单方法获取的看似“风险规避”的选择的普遍现象形成对比，这一结果我们在本文中重复了。我们证明了这种方法不适用于衡量风险规避，并且它产生风险规避高普遍性的原因是参数错误。

    In economics, risk aversion is modeled via a concave Bernoulli utility within the expected-utility paradigm. We propose a simple test of expected utility and concavity. We find little support for either: only 30 percent of the choices are consistent with a concave utility, only two out of 72 subjects are consistent with expected utility, and only one of them fits the economic model of risk aversion. Our findings contrast with the preponderance of seemingly "risk-averse" choices that have been elicited using the popular multiple-price list methodology, a result we replicate in this paper. We demonstrate that this methodology is unfit to measure risk aversion, and that the high prevalence of risk aversion it produces is due to parametric misspecification.
    
[^6]: 有限知识下n人博弈的人类与机器智能

    Human and Machine Intelligence in n-Person Games with Partial Knowledge. (arXiv:2302.13937v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2302.13937](http://arxiv.org/abs/2302.13937)

    提出了有限知识下n人博弈的框架，引入了“游戏智能”机制和“防作弊性”概念，GI机制可以实际评估玩家的智能，应用广泛。

    

    本文提出了一个新的框架——有限知识下的n人博弈，其中玩家只对游戏的某些方面（包括行动、结果和其他玩家）有限的了解。为了分析这些游戏，我介绍了一组新的概念和机制，重点关注人类和机器决策之间的相互作用。具体而言，我引入了两个主要概念：第一个是“游戏智能”（GI）机制，它通过考虑参考机器智能下的“错误”，不仅仅是游戏的结果，量化了玩家在游戏中展示出的智能。第二个是“防作弊性”，这是一种实用的、可计算的策略无关性的概念。GI机制提供了一种实用的方法来评估玩家，可以潜在地应用于从在线游戏到现实生活决策的各种游戏。

    In this note, I introduce a new framework called n-person games with partial knowledge, in which players have only limited knowledge about the aspects of the game -- including actions, outcomes, and other players. For example, playing an actual game of chess is a game of partial knowledge. To analyze these games, I introduce a set of new concepts and mechanisms for measuring the intelligence of players, with a focus on the interplay between human- and machine-based decision-making. Specifically, I introduce two main concepts: firstly, the Game Intelligence (GI) mechanism, which quantifies a player's demonstrated intelligence in a game by considering not only the game's outcome but also the "mistakes" made during the game according to the reference machine's intelligence. Secondly, I define gaming-proofness, a practical and computational concept of strategy-proofness. The GI mechanism provides a practicable way to assess players and can potentially be applied to a wide range of games, f
    
[^7]: 默认提示什么时候有效?

    When do Default Nudges Work?. (arXiv:2301.08797v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.08797](http://arxiv.org/abs/2301.08797)

    通过在瑞典Covid-19疫苗推出中的区域变化，研究了对激励有差异的群体的提示效果，结果显示提示对于个人无意义的选择更有效。

    

    提示是科学和政策上一个新兴的话题，但不同激励群体中的提示效果的证据还不足。本文利用瑞典Covid-19疫苗推出过程中的区域变化，研究了对激励有差异的群体的提示效果：对于16-17岁的孩子，Covid-19不危险，而50-59岁的人则面临着严重疾病或死亡的巨大风险。我们发现，年轻人的反应显著强烈，这与提示对于个人无意义的选择更有效的理论相一致。

    Nudging is a burgeoning topic in science and in policy, but evidence on the effectiveness of nudges among differentially-incentivized groups is lacking. This paper exploits regional variations in the roll-out of the Covid-19 vaccine in Sweden to examine the effect of a nudge on groups whose intrinsic incentives are different: 16-17-year-olds, for whom Covid-19 is not dangerous, and 50-59-year-olds, who face a substantial risk of death or severe dis-ease. We find a significantly stronger response in the younger group, consistent with the theory that nudges are more effective for choices that are not meaningful to the individual.
    
[^8]: 从病毒性内容中学习

    Learning from Viral Content. (arXiv:2210.01267v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2210.01267](http://arxiv.org/abs/2210.01267)

    本文研究了社交媒体上的学习，发现向用户展示病毒性故事可以增加信息汇集，但也可能导致大多数分享的故事是错误的稳定状态。这些误导性的稳定状态会自我维持，对平台设计和鲁棒性产生多种后果。

    

    我们研究了社交媒体上的学习，采用了一个均衡模型来描述用户与共享新闻故事进行交互。理性用户按顺序到达，观察到原始故事（即私有信号）和新闻推送中前辈故事的样本，然后决定分享哪些故事。观察到的故事样本取决于前辈分享的内容以及生成新闻推送的抽样算法。我们重点研究了这个算法如何选择更具病毒性（即被广泛分享）的故事的频率。向用户展示病毒性故事可以增加信息汇集，但也可能产生大多数分享故事错误的稳定状态。这些误导性的稳定状态自我持续，因为观察到错误故事的用户会形成错误的信念，从而理性地继续分享它们。最后，我们描述了平台设计和鲁棒性方面的若干后果。

    We study learning on social media with an equilibrium model of users interacting with shared news stories. Rational users arrive sequentially, observe an original story (i.e., a private signal) and a sample of predecessors' stories in a news feed, and then decide which stories to share. The observed sample of stories depends on what predecessors share as well as the sampling algorithm generating news feeds. We focus on how often this algorithm selects more viral (i.e., widely shared) stories. Showing users viral stories can increase information aggregation, but it can also generate steady states where most shared stories are wrong. These misleading steady states self-perpetuate, as users who observe wrong stories develop wrong beliefs, and thus rationally continue to share them. Finally, we describe several consequences for platform design and robustness.
    
[^9]: SoK：区块链去中心化

    SoK: Blockchain Decentralization. (arXiv:2205.04256v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.04256](http://arxiv.org/abs/2205.04256)

    该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。

    

    区块链通过在点对点网络中实现分布式信任，为去中心化经济提供了支持。然而，令人惊讶的是，目前还缺乏广泛接受的去中心化定义或度量标准。我们通过全面分析现有研究，探索了区块链去中心化的知识系统化（SoK）。首先，我们通过对现有研究的定性分析，在共识、网络、治理、财富和交易等五个方面建立了用于分析区块链去中心化的分类法。我们发现，除了共识去中心化以外，其他方面的研究相对较少。其次，我们提出了一种指数，通过转换香农熵来衡量和量化区块链在不同方面的去中心化水平。我们通过比较静态模拟验证了该指数的可解释性。我们还提供了其他指数的定义和讨论，包括基尼系数、中本聪系数和赫尔曼-赫尔东指数等。我们的工作概述了当前区块链去中心化的景象，并提出了一个量化的度量标准，为未来的研究提供基准。

    Blockchain empowers a decentralized economy by enabling distributed trust in a peer-to-peer network. However, surprisingly, a widely accepted definition or measurement of decentralization is still lacking. We explore a systematization of knowledge (SoK) on blockchain decentralization by comprehensively analyzing existing studies in various aspects. First, we establish a taxonomy for analyzing blockchain decentralization in the five facets of consensus, network, governance, wealth, and transaction bu qualitative analysis of existing research. We find relatively little research on aspects other than consensus decentralization. Second, we propose an index that measures and quantifies the decentralization level of blockchain across different facets by transforming Shannon entropy. We verify the explainability of the index via comparative static simulations. We also provide the definition and discussion of alternative indices including the Gini Coefficient, Nakamoto Coefficient, and Herfind
    
[^10]: 鸽巢设计：从在线匹配的视角平衡顺序实验

    Pigeonhole Design: Balancing Sequential Experiments from an Online Matching Perspective. (arXiv:2201.12936v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2201.12936](http://arxiv.org/abs/2201.12936)

    本研究研究了一种新颖的在线实验设计问题，称为“在线阻塞问题”，旨在平衡顺序到达的具有异质协变信息的实验对象，以最小化总差异。作者提出了一种名为“鸽巢设计”的方法来解决该问题。

    

    实践者和学者长期以来都认识到实验对公司带来的好处。然而，对于运行在线A/B测试的面向Web的公司来说，在顺序到达实验对象时平衡协变信息仍然具有挑战性。在本文中，我们研究了一种新颖的在线实验设计问题，我们称之为“在线阻塞问题”。在这个问题中，具有异质协变信息的实验对象按顺序到达，必须立即分配到控制组或处理组中，目标是最小化总差异，即两组之间的最小权重完美匹配。为了解决这个问题，我们提出了一种新颖的实验设计方法，称为“鸽巢设计”。鸽巢设计首先将协变空间划分为较小的空间，我们称之为鸽巢，然后在每个鸽巢中到达实验对象时。

    Practitioners and academics have long appreciated the benefits that experimentation brings to firms. For web-facing firms running online A/B tests, however, it still remains challenging in balancing covariate information when experimental subjects arrive sequentially. In this paper, we study a novel online experimental design problem, which we refer to as the "Online Blocking Problem." In this problem, experimental subjects with heterogeneous covariate information arrive sequentially and must be immediately assigned into either the control or the treatment group, with an objective of minimizing the total discrepancy, which is defined as the minimum weight perfect matching between the two groups. To solve this problem, we propose a novel experimental design approach, which we refer to as the "Pigeonhole Design." The pigeonhole design first partitions the covariate space into smaller spaces, which we refer to as pigeonholes, and then, when the experimental subjects arrive at each pigeonh
    
[^11]: 一种用于政策评估的治疗效果识别的计算方法

    A Computational Approach to Identification of Treatment Effects for Policy Evaluation. (arXiv:2009.13861v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.13861](http://arxiv.org/abs/2009.13861)

    本文研究了一种计算方法，能够在存在未观察到的异质性和仅有二值工具变量的情况下，推广局部治疗效应到不同的反事实情境，并且提出了一个新的框架，可以系统地计算各种与政策相关的治疗参数的尖锐非参数界限。

    

    对于反事实政策评估，确保治疗参数与所讨论的政策相关非常重要。在存在未观察到的异质性的情况下，这一点尤为具有挑战性，而这在局部平均治疗效应（LATE）的定义中得到了很好的体现。作为内在的局部效应，LATE已知在反事实环境下缺乏外部效度。本文研究了当工具变量仅为二值时，将局部治疗效应推广到不同反事实情境的可能性。我们提出了一个新的框架，系统地计算各种与政策相关的治疗参数的尖锐非参数界限，这些参数被定义为边际治疗效应（MTE）的加权平均。我们的框架足够灵活，能够充分融入工具变量的统计独立性（而不仅仅是均值独立性），并且能够考虑多种形状约束之外的鉴定假设，这些形状约束在之前的研究中已经被考虑过了。

    For counterfactual policy evaluation, it is important to ensure that treatment parameters are relevant to policies in question. This is especially challenging under unobserved heterogeneity, as is well featured in the definition of the local average treatment effect (LATE). Being intrinsically local, the LATE is known to lack external validity in counterfactual environments. This paper investigates the possibility of extrapolating local treatment effects to different counterfactual settings when instrumental variables are only binary. We propose a novel framework to systematically calculate sharp nonparametric bounds on various policy-relevant treatment parameters that are defined as weighted averages of the marginal treatment effect (MTE). Our framework is flexible enough to fully incorporate statistical independence (rather than mean independence) of instruments and a large menu of identifying assumptions beyond the shape restrictions on the MTE that have been considered in prior stu
    

