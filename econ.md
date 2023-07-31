# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings.](http://arxiv.org/abs/2307.15702) | 强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。 |
| [^2] | [Global air quality inequality over 2000-2020.](http://arxiv.org/abs/2307.15669) | 2000-2020年间全球空气质量不平等水平持续上升，主要由国家间差异引起。研究结果指出，仅关注国内差异的努力忽视了一个重要问题。 |
| [^3] | [Fast but multi-partisan: Bursts of communication increase opinion diversity in the temporal Deffuant model.](http://arxiv.org/abs/2307.15614) | 本文研究了沟通突发如何影响观点多样性，发现突发性可以通过促进本地意见的加强来阻止达成共识和极化，特别是在个体容忍度较低且更喜欢调整给相似的同伴时。 |
| [^4] | [Quantifying the Influence of Climate on Human Mind and Culture: Evidence from Visual Art.](http://arxiv.org/abs/2307.15540) | 本研究通过分析大量绘画作品和艺术家的生物数据，发现气候变化对人类思维和文化产生了显著和持久的影响，而且这种影响在更依赖艺术家想象力的艺术流派中更加明显。 |
| [^5] | [Only-child matching penalty in the marriage market.](http://arxiv.org/abs/2307.15336) | 本研究探讨了独生子女在婚姻市场中的匹配情况，并发现独生子女在配对时受到了惩罚，尤其是在伴侣的社会经济地位方面。 |
| [^6] | [Group-Heterogeneous Changes-in-Changes and Distributional Synthetic Controls.](http://arxiv.org/abs/2307.15313) | 本文针对改变中的改变和分布式合成对照的问题，提出了解决团体异质性的新方法，包括在控制组中找出与处理组具有相似群体特征的适当子群，以及使用在不同时间段内具有相似团体异质性的单位构建合成对照。 |
| [^7] | [On the mathematics of the circular flow of economic activity with applications to the topic of caring for the vulnerable during pandemics.](http://arxiv.org/abs/2307.15197) | 该论文研究了在缺乏政府支持的情况下如何帮助贫穷和易受困者的问题。通过线性代数和图论方法，提出了一个新的模型来解释收入循环动力学，并将社会划分为分散的和凝聚的。在分散社会中，说服富裕代理人支持贫穷和易受困者将非常困难，而在凝聚社会中，富裕代理人有动机和能力支持贫穷和易受困者。 |
| [^8] | [On the Efficiency of Finely Stratified Experiments.](http://arxiv.org/abs/2307.15181) | 本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。 |
| [^9] | [Predictability Tests Robust against Parameter Instability.](http://arxiv.org/abs/2307.15151) | 该论文提出了一种具有鲁棒性的可预测性检验方法，能在考虑参数不稳定性的情况下，通过仪器变量方法检验联合预测和结构性突变。该方法通过分析可追踪的渐近理论，对非平稳回归器下的统计结果进行了推导。实验证明该方法具有较好的实际应用价值。 |
| [^10] | [Managed Campaigns and Data-Augmented Auctions for Digital Advertising.](http://arxiv.org/abs/2304.08432) | 拍卖模型为数字广告提供了数据增强的竞标机制，同时托管活动机制有助于优化广告配置。平台上的社会有效配置会影响广告主在平台外的产品价格，导致平台外的分配效率低下。 |
| [^11] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |
| [^12] | [Buying Opinions.](http://arxiv.org/abs/2202.05249) | 本文研究了当一个委托人雇佣一个代理人来获取信息时，如何保证代理人可以被诱导获得并诚实报告信息，以实现最优结果，但代理人的风险厌恶可能导致最优结果无法实现。 |

# 详细

[^1]: 强大的最大环算法：一种集成偏好排序的新方法

    The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings. (arXiv:2307.15702v1 [cs.SI])

    [http://arxiv.org/abs/2307.15702](http://arxiv.org/abs/2307.15702)

    强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。

    

    我们提出了一种基于优化的方法，用于在每个决策者或选民对一对选择进行偏好表达的情况下集成偏好。挑战在于在一些冲突的投票情况下，尽可能与投票结果一致地得出一个排序。只有不包含环路的投票集合才是非冲突的，并且可以在选择之间引发一个部分顺序。我们的方法是基于这样一个观察：构成一个环路的投票集合可以被视为平局。然后，方法是从投票图中删除环路的并集，并根据剩余部分确定集成偏好。我们引入了强大的最大环路，它由一组环路的并集形成，删除它可以保证在引发的部分顺序中获得唯一结果。此外，它还包含在消除任何最大环路后剩下的所有集成偏好。与之相反的是，wel

    We present a new optimization-based method for aggregating preferences in settings where each decision maker, or voter, expresses preferences over pairs of alternatives. The challenge is to come up with a ranking that agrees as much as possible with the votes cast in cases when some of the votes conflict. Only a collection of votes that contains no cycles is non-conflicting and can induce a partial order over alternatives. Our approach is motivated by the observation that a collection of votes that form a cycle can be treated as ties. The method is then to remove unions of cycles of votes, or circulations, from the vote graph and determine aggregate preferences from the remainder.  We introduce the strong maximum circulation which is formed by a union of cycles, the removal of which guarantees a unique outcome in terms of the induced partial order. Furthermore, it contains all the aggregate preferences remaining following the elimination of any maximum circulation. In contrast, the wel
    
[^2]: 2000-2020年全球空气质量不平等研究

    Global air quality inequality over 2000-2020. (arXiv:2307.15669v1 [econ.GN])

    [http://arxiv.org/abs/2307.15669](http://arxiv.org/abs/2307.15669)

    2000-2020年间全球空气质量不平等水平持续上升，主要由国家间差异引起。研究结果指出，仅关注国内差异的努力忽视了一个重要问题。

    

    空气污染在全世界范围内产生了巨大的健康损害和经济成本。污染暴露的程度在国家间和国内差异很大。然而，全球范围内空气质量不平等及其随时间变化的程度尚未得到量化。本研究使用经济不平等指数来衡量各国暴露于直径为2.5微米或以下的环境细颗粒物（PM2.5）的不平等情况。研究发现全球空气质量不平等水平高且持续上升。全球PM2.5基尼系数从2000年的0.32上升到2020年的0.36，超过了许多国家的收入不平等水平。空气质量不平等主要是由国家间差异引起的，而国内差异的影响较小，分解分析结果显示。面对最高水平PM2.5暴露的人口大部分集中在只有少数几个国家。研究结果表明，仅关注国内差异的研究和政策努力忽视了一个重要问题。

    Air pollution generates substantial health damages and economic costs worldwide. Pollution exposure varies greatly, both between countries and within them. However, the degree of air quality inequality and its' trajectory over time have not been quantified at a global level. Here I use economic inequality indices to measure global inequality in exposure to ambient fine particles with 2.5 microns or less in diameter (PM2.5). I find high and rising levels of global air quality inequality. The global PM2.5 Gini Index increased from 0.32 in 2000 to 0.36 in 2020, exceeding levels of income inequality in many countries. Air quality inequality is mostly driven by differences between countries and less so by variation within them, as decomposition analysis shows. A large share of people facing the highest levels of PM2.5 exposure are concentrated in only a few countries. The findings suggest that research and policy efforts that focus only on differences within countries are overlooking an imp
    
[^3]: 快速而多派系的：沟通突发增加时间Deffuant模型中的观点多样性

    Fast but multi-partisan: Bursts of communication increase opinion diversity in the temporal Deffuant model. (arXiv:2307.15614v1 [physics.soc-ph])

    [http://arxiv.org/abs/2307.15614](http://arxiv.org/abs/2307.15614)

    本文研究了沟通突发如何影响观点多样性，发现突发性可以通过促进本地意见的加强来阻止达成共识和极化，特别是在个体容忍度较低且更喜欢调整给相似的同伴时。

    

    人类互动创建了社会网络，形成社会的骨干。个体通过社交互动交换信息来调整观点。两个经常出现的问题是社会结构是否促进了观点极化或共识以及是否可以避免极化，尤其是在社交媒体上。在本文中，我们假设社交互动的时间结构不仅调节观点集群的形成，而且定时性本身足以阻止达成共识和极化，通过促进本地观点的强化。我们设计了一种Deffuant观点模型的时间版本，其中双方交互遵循时间模式，并表明仅仅靠突发性就足以通过促进本地意见的加强来避免共识和极化。由于网络聚类，个体自组织成为一个多派系的社会，但在突发性的情况下，意见集群的多样性进一步增加，特别是当个体的容忍度较低且更喜欢调整给相似的同伴时。

    Human interactions create social networks forming the backbone of societies. Individuals adjust their opinions by exchanging information through social interactions. Two recurrent questions are whether social structures promote opinion polarisation or consensus in societies and whether polarisation can be avoided, particularly on social media. In this paper, we hypothesise that not only network structure but also the timings of social interactions regulate the emergence of opinion clusters. We devise a temporal version of the Deffuant opinion model where pairwise interactions follow temporal patterns and show that burstiness alone is sufficient to refrain from consensus and polarisation by promoting the reinforcement of local opinions. Individuals self-organise into a multi-partisan society due to network clustering, but the diversity of opinion clusters further increases with burstiness, particularly when individuals have low tolerance and prefer to adjust to similar peers. The emerge
    
[^4]: 量化气候对人类思维和文化的影响: 来自视觉艺术的证据

    Quantifying the Influence of Climate on Human Mind and Culture: Evidence from Visual Art. (arXiv:2307.15540v1 [q-bio.PE])

    [http://arxiv.org/abs/2307.15540](http://arxiv.org/abs/2307.15540)

    本研究通过分析大量绘画作品和艺术家的生物数据，发现气候变化对人类思维和文化产生了显著和持久的影响，而且这种影响在更依赖艺术家想象力的艺术流派中更加明显。

    

    本文研究了气候变化对人类思维和文化的影响，时间跨度从13世纪到21世纪。通过对10万幅绘画作品和2000多位艺术家的生物数据进行定量分析，发现了绘画明亮度的一种有趣的U型模式，与全球温度趋势相关。事件研究分析发现，当艺术家遭受高温冲击时，他们的作品在后期变得更加明亮。此外，这种影响在更多依赖艺术家想象力而非现实事物的艺术流派中更加明显，表明了艺术家思维的影响。总体而言，本研究证明了气候对人类思维和文化的显著和持久影响。

    This paper examines the influence of climate change on the human mind and culture from the 13th century to the 21st century. By quantitatively analyzing 100,000 paintings and the biological data of over 2,000 artists, an interesting U-shaped pattern in the lightness of paintings was found, which correlated with trends in global temperature. Event study analysis revealed that when an artist is subjected to a high-temperature shock, their paintings become brighter in later periods. Moreover, the effects are more pronounced in art genres that rely less on real things and more on the artist's imagination, indicating the influence of artists' minds. Overall, this study demonstrates the significant and enduring influence of climate on the human mind and culture over centuries.
    
[^5]: 独生子女在婚姻市场中的配对惩罚

    Only-child matching penalty in the marriage market. (arXiv:2307.15336v1 [econ.GN])

    [http://arxiv.org/abs/2307.15336](http://arxiv.org/abs/2307.15336)

    本研究探讨了独生子女在婚姻市场中的匹配情况，并发现独生子女在配对时受到了惩罚，尤其是在伴侣的社会经济地位方面。

    

    本研究探讨了独生子女的婚姻配对及其结果。具体而言，我们分析了两个方面。首先，我们调查了独生子女和非独生子女之间的婚姻状况（即与独生子女、非独生子女结婚或保持单身）。这个分析让我们了解人们在选择配偶时是否根据独生子女身份采用正相关配对方式，以及独生子女是否从婚姻配对中获益或受到伴侣吸引力的惩罚。其次，我们通过独生子女和非独生子女之间伴侣社会经济地位（在这里，指受教育年限）差距的大小来衡量配对溢价/惩罚。传统的经济理论和对独生子女身份的正相关配对的观察到的婚姻模式预测，在婚姻市场上，独生子女会受到配对惩罚，尤其是当

    This study explores the marriage matching of only-child individuals and its outcome. Specifically, we analyze two aspects. First, we investigate how marital status (i.e., marriage with an only child, that with a non-only child and remaining single) differs between only children and non-only children. This analysis allows us to know whether people choose mates in a positive or a negative assortative manner regarding only-child status, and to predict whether only-child individuals benefit from marriage matching premiums or are subject to penalties regarding partner attractiveness. Second, we measure the premium/penalty by the size of the gap in partner's socio economic status (SES, here, years of schooling) between only-child and non--only-child individuals. The conventional economic theory and the observed marriage patterns of positive assortative mating on only-child status predict that only-child individuals are subject to a matching penalty in the marriage market, especially when the
    
[^6]: 团体异质改变与分布式合成对照的改进及其分布异质性控制

    Group-Heterogeneous Changes-in-Changes and Distributional Synthetic Controls. (arXiv:2307.15313v1 [econ.EM])

    [http://arxiv.org/abs/2307.15313](http://arxiv.org/abs/2307.15313)

    本文针对改变中的改变和分布式合成对照的问题，提出了解决团体异质性的新方法，包括在控制组中找出与处理组具有相似群体特征的适当子群，以及使用在不同时间段内具有相似团体异质性的单位构建合成对照。

    

    我们在存在团体层面异质性的情况下，开发了新的方法来处理改变中的改变和分布式合成对照。对于改变中的改变，我们允许个体属于大量异质群体。新方法扩展了Athey和Imbens（2006）的改变中的改变方法，通过找到控制组内与处理组具有相似团体层面未观测特征的适当子群来进行改进。对于分布式合成对照，我们表明适当的合成对照需要使用在不同时间段内具有与处理组相当的团体层面异质性的单位进行构建，而不仅仅是像Gunsilius（2023）中的相同时间段的单位。我们简要讨论了这些新方法的实施和数据要求。

    We develop new methods for changes-in-changes and distributional synthetic controls when there exists group level heterogeneity. For changes-in-changes, we allow individuals to belong to a large number of heterogeneous groups. The new method extends the changes-in-changes method in Athey and Imbens (2006) by finding appropriate subgroups within the control groups which share similar group level unobserved characteristics to the treatment groups. For distributional synthetic control, we show that the appropriate synthetic control needs to be constructed using units in potentially different time periods in which they have comparable group level heterogeneity to the treatment group, instead of units that are only in the same time period as in Gunsilius (2023). Implementation and data requirements for these new methods are briefly discussed.
    
[^7]: 关于经济活动循环流动数学的研究及其在关怀易受困者的应用

    On the mathematics of the circular flow of economic activity with applications to the topic of caring for the vulnerable during pandemics. (arXiv:2307.15197v1 [econ.GN])

    [http://arxiv.org/abs/2307.15197](http://arxiv.org/abs/2307.15197)

    该论文研究了在缺乏政府支持的情况下如何帮助贫穷和易受困者的问题。通过线性代数和图论方法，提出了一个新的模型来解释收入循环动力学，并将社会划分为分散的和凝聚的。在分散社会中，说服富裕代理人支持贫穷和易受困者将非常困难，而在凝聚社会中，富裕代理人有动机和能力支持贫穷和易受困者。

    

    我们从根本层面探讨了在缺乏政府机构支持的情况下，为什么、何时和如何去帮助贫穷和易受困者的问题。我们提出了一个基于线性代数和基础图论的简单而新颖的方法来捕捉经济主体之间的收入循环动力学。引入了一个针对收入循环的新的线性代数模型，根据该模型我们可以将社会划分为分散的或凝聚的。我们表明，在分散的社会中，说服社会层级顶部的富裕代理人支持贫穷和易受困者将极为困难。我们还强调了线性代数和简单图论方法如何从基本观点解释分散社会中阶级斗争的一些机制。然后，我们直观地说明并从数学上证明了为什么在凝聚社会中，社会层级顶部的富裕代理人有动机和能力支持贫穷和易受困者。

    We investigate, at the fundamental level, the questions of `why', `when' and `how' one could or should reach out to poor and vulnerable people to support them in the absence of governmental institutions. We provide a simple and new approach that is rooted in linear algebra and basic graph theory to capture the dynamics of income circulation among economic agents. A new linear algebraic model for income circulation is introduced, based on which we are able to categorize societies as fragmented or cohesive. We show that, in the case of fragmented societies, convincing wealthy agents at the top of the social hierarchy to support the poor and vulnerable will be very difficult. We also highlight how linear-algebraic and simple graph-theoretic methods help explain, from a fundamental point of view, some of the mechanics of class struggle in fragmented societies. Then, we explain intuitively and prove mathematically why, in cohesive societies, wealthy agents at the top of the social hierarchy
    
[^8]: 关于细分实验效率的研究

    On the Efficiency of Finely Stratified Experiments. (arXiv:2307.15181v1 [econ.EM])

    [http://arxiv.org/abs/2307.15181](http://arxiv.org/abs/2307.15181)

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。

    

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计。在这里，效率是指对于一类广泛的处理分配方案而言的，其中任何单位被分配到处理的边际概率等于预先指定的值，例如一半。重要的是，我们不要求处理状态是以i.i.d.的方式分配的，因此可以适应实践中使用的复杂处理分配方案，如分层随机化和匹配对。所考虑的参数类别是可以表示为已知观测数据的一个已知函数的期望的约束的解的那些参数，其中可能包括处理分配边际概率的预先指定值。我们证明了这类参数包括平均处理效应、分位数处理效应、局部平均处理效应等。

    This paper studies the efficient estimation of a large class of treatment effect parameters that arise in the analysis of experiments. Here, efficiency is understood to be with respect to a broad class of treatment assignment schemes for which the marginal probability that any unit is assigned to treatment equals a pre-specified value, e.g., one half. Importantly, we do not require that treatment status is assigned in an i.i.d. fashion, thereby accommodating complicated treatment assignment schemes that are used in practice, such as stratified block randomization and matched pairs. The class of parameters considered are those that can be expressed as the solution to a restriction on the expectation of a known function of the observed data, including possibly the pre-specified value for the marginal probability of treatment assignment. We show that this class of parameters includes, among other things, average treatment effects, quantile treatment effects, local average treatment effect
    
[^9]: 对参数不稳定性有鲁棒性的可预测性检验

    Predictability Tests Robust against Parameter Instability. (arXiv:2307.15151v1 [econ.EM])

    [http://arxiv.org/abs/2307.15151](http://arxiv.org/abs/2307.15151)

    该论文提出了一种具有鲁棒性的可预测性检验方法，能在考虑参数不稳定性的情况下，通过仪器变量方法检验联合预测和结构性突变。该方法通过分析可追踪的渐近理论，对非平稳回归器下的统计结果进行了推导。实验证明该方法具有较好的实际应用价值。

    

    我们考虑基于Phillips和Magdalinos（2009年）的仪器变量方法设计的联合预测和结构性突变检验的Wald类型统计量。我们表明，在非平稳预测变量的假设下：（i）基于OLS估计量的检验收敛于依赖于扰动系数的非标准极限分布；（ii）基于IVX估计量的检验可以在某些参数限制下滤除持续性，这是由于最高函数的原因。这些结果通过在考虑非平稳回归器时提供了解析可追踪的渐近理论，为联合预测和参数不稳定性测试的文献做出了贡献。我们通过广泛的蒙特卡罗实验比较了两种估计量下Wald检验的有限样本大小和功率性能。临界值是使用标准的自助法推断方法计算的。我们展示了所提出的框架的实用性。

    We consider Wald type statistics designed for joint predictability and structural break testing based on the instrumentation method of Phillips and Magdalinos (2009). We show that under the assumption of nonstationary predictors: (i) the tests based on the OLS estimators converge to a nonstandard limiting distribution which depends on the nuisance coefficient of persistence; and (ii) the tests based on the IVX estimators can filter out the persistence under certain parameter restrictions due to the supremum functional. These results contribute to the literature of joint predictability and parameter instability testing by providing analytical tractable asymptotic theory when taking into account nonstationary regressors. We compare the finite-sample size and power performance of the Wald tests under both estimators via extensive Monte Carlo experiments. Critical values are computed using standard bootstrap inference methodologies. We illustrate the usefulness of the proposed framework to
    
[^10]: 数字广告的托管活动和数据增强拍卖模型

    Managed Campaigns and Data-Augmented Auctions for Digital Advertising. (arXiv:2304.08432v1 [econ.TH])

    [http://arxiv.org/abs/2304.08432](http://arxiv.org/abs/2304.08432)

    拍卖模型为数字广告提供了数据增强的竞标机制，同时托管活动机制有助于优化广告配置。平台上的社会有效配置会影响广告主在平台外的产品价格，导致平台外的分配效率低下。

    

    本研究针对数字广告开发了拍卖模型。垄断平台具有广告主和消费者之间匹配价值的数据。平台支持利用额外信息进行竞标，从而增加平台内匹配的可行剩余。广告主共同决定其在平台上和平台外的定价策略，以及他们在平台上进行数字广告的竞标。我们比较了数据增强的二价拍卖和托管活动机制。在数据增强的拍卖中，广告主的竞标受到平台匹配价值数据的影响。这导致了平台上的社会有效配置，但广告主会提高其在平台外的产品价格，以在平台上更具竞争力。因此，由于产品价格过高，平台外的分配效率不高。托管活动机制允许广告主提交预算，然后转移

    We develop an auction model for digital advertising. A monopoly platform has access to data on the value of the match between advertisers and consumers. The platform support bidding with additional information and increase the feasible surplus for on-platform matches. Advertisers jointly determine their pricing strategy both on and off the platform, as well as their bidding for digital advertising on the platform. We compare a data-augmented second-price auction and a managed campaign mechanism. In the data-augmented auction, the bids by the advertisers are informed by the data of the platform regarding the value of the match. This results in a socially efficient allocation on the platform, but the advertisers increase their product prices off the platform to be more competitive on the platform. In consequence, the allocation off the platform is inefficient due to excessively high product prices. The managed campaign mechanism allows advertisers to submit budgets that are then transfor
    
[^11]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    
[^12]: 意见购买

    Buying Opinions. (arXiv:2202.05249v4 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2202.05249](http://arxiv.org/abs/2202.05249)

    本文研究了当一个委托人雇佣一个代理人来获取信息时，如何保证代理人可以被诱导获得并诚实报告信息，以实现最优结果，但代理人的风险厌恶可能导致最优结果无法实现。

    

    一个委托人雇佣一个代理人来获取关于未知状态的软信息。虽然代理人学习的方式和发现的信息都无法在合同中规定，但我们证明了在代理人是风险中性的情况下，只要代理人被诱导获得并诚实报告信息的成本不太高，或者面临足够大的惩罚，那么委托人可以胜任任何信息获取的任务。然而代理人的风险厌恶确保了最佳结果的无法实现。

    A principal hires an agent to acquire soft information about an unknown state. Even though neither how the agent learns nor what the agent discovers are contractible, we show the principal is unconstrained as to what information the agent can be induced to acquire and report honestly. When the agent is risk neutral, and a) is not asked to learn too much, b) can acquire information sufficiently cheaply, or c) can face sufficiently large penalties, the principal can attain the first-best outcome. Risk aversion (on the part of the agent) ensures the first-best is unattainable.
    

