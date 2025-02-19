# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |
| [^2] | [Recurring Auctions with Costly Entry: Theory and Evidence.](http://arxiv.org/abs/2306.17355) | 重复拍卖在买方面临高成本参与时，在效率和收入方面优于单轮拍卖。原因是重复拍卖允许不同价值的买家在不同时间进入，从而节省了进入成本并增加了整体售卖概率。 |
| [^3] | [Self-Resolving Prediction Markets for Unverifiable Outcomes.](http://arxiv.org/abs/2306.04305) | 该论文提出了一种新颖的预测市场机制，允许在无法验证结果的情况下从一组代理中收集信息和汇总预测，使用机器学习模型，实现自治以及避免了需进一步的验证或干预。 |
| [^4] | [Adjustment with Many Regressors Under Covariate-Adaptive Randomizations.](http://arxiv.org/abs/2304.08184) | 本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。 |
| [^5] | [Correlation between upstreamness and downstreamness in random global value chains.](http://arxiv.org/abs/2303.06603) | 本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。 |
| [^6] | [Media Slant is Contagious.](http://arxiv.org/abs/2202.07269) | 本文研究了国家有线电视新闻对美国本土报纸的影响，发现当地报纸的内容会因为当地 FNC 观众数量的增加而趋向于 FNC 的倾向，并且有线电视倾向会极化地方新闻内容。 |

# 详细

[^1]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    
[^2]: 高成本参与的重复拍卖：理论与证据

    Recurring Auctions with Costly Entry: Theory and Evidence. (arXiv:2306.17355v1 [econ.TH])

    [http://arxiv.org/abs/2306.17355](http://arxiv.org/abs/2306.17355)

    重复拍卖在买方面临高成本参与时，在效率和收入方面优于单轮拍卖。原因是重复拍卖允许不同价值的买家在不同时间进入，从而节省了进入成本并增加了整体售卖概率。

    

    重复拍卖在出售耐用资产，如土地、房屋或艺术品方面普遍存在：当卖方在初始拍卖中无法出售物品时，她通常会在不久的将来举行下一个拍卖。本文在理论和实证两方面对重复拍卖的设计进行了表征。在理论方面，我们表明当潜在买家面临高成本参与时，重复拍卖在效率和收入方面优于单轮拍卖。这是因为重复拍卖允许具有不同价值的潜在买家在不同时间进入，从而降低了进入成本并增加了整体售卖概率。我们进一步推导了在重复拍卖中保留价格的最佳序列，这取决于卖方的目标是最大化效率还是收入。在实证方面，我们将理论应用于中国的房屋被收购拍卖中，一套被收购的房屋最多连续拍卖三次。在估计结构参数之后

    Recurring auctions are ubiquitous for selling durable assets, such as land, home, or artwork: When the seller cannot sell the item in the initial auction, she often holds a subsequent auction in the near future. This paper characterizes the design of recurring auctions, both theoretically and empirically. On the theoretical side, we show that recurring auctions outperform single-round auctions in efficiency and revenue when potential buyers face costly entry. This occurs because recurring auctions allow potential buyers with different values to enter at different times, which generates savings in entry costs and increases the overall probability of sale. We further derive the optimal sequence of reserve prices in recurring auctions, depending on whether the seller aims to maximize efficiency or revenue. On the empirical side, we apply the theory to home foreclosure auctions in China, where a foreclosed home is auctioned up to three times in a row. After estimating structural parameters
    
[^3]: 无法验证结果的自动解决预测市场

    Self-Resolving Prediction Markets for Unverifiable Outcomes. (arXiv:2306.04305v1 [cs.GT])

    [http://arxiv.org/abs/2306.04305](http://arxiv.org/abs/2306.04305)

    该论文提出了一种新颖的预测市场机制，允许在无法验证结果的情况下从一组代理中收集信息和汇总预测，使用机器学习模型，实现自治以及避免了需进一步的验证或干预。

    

    预测市场通过根据预测是否接近可验证的未来结果向代理支付费用来激励和汇总信念。然而，许多重要问题的结果难以验证或不可验证，因为可能很难或不可能获取实际情况。我们提出了一种新颖而又不直观的结果，表明可以通过向代理支付其预测与精心选择的参考代理的预测之间的负交叉熵来运行一个ε-激励兼容的预测市场，从代理池中获取信息并进行有效的汇总，而不需要观察结果。我们的机制利用一个离线的机器学习模型，该模型根据市场设计者已知的一组特征来预测结果，从而使市场能够在观察到结果后自行解决并立即向代理支付报酬，而不需要进一步的验证或干预。我们对我们的机制的效率、激励兼容性和收敛性提供了理论保证，同时在几个真实世界的数据集上进行了验证。

    Prediction markets elicit and aggregate beliefs by paying agents based on how close their predictions are to a verifiable future outcome. However, outcomes of many important questions are difficult to verify or unverifiable, in that the ground truth may be hard or impossible to access. Examples include questions about causal effects where it is infeasible or unethical to run randomized trials; crowdsourcing and content moderation tasks where it is prohibitively expensive to verify ground truth; and questions asked over long time horizons, where the delay until the realization of the outcome skews agents' incentives to report their true beliefs. We present a novel and unintuitive result showing that it is possible to run an $\varepsilon-$incentive compatible prediction market to elicit and efficiently aggregate information from a pool of agents without observing the outcome by paying agents the negative cross-entropy between their prediction and that of a carefully chosen reference agen
    
[^4]: 协变量自适应随机化下的多个回归器的调整

    Adjustment with Many Regressors Under Covariate-Adaptive Randomizations. (arXiv:2304.08184v1 [econ.EM])

    [http://arxiv.org/abs/2304.08184](http://arxiv.org/abs/2304.08184)

    本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。

    

    本文针对协变量自适应随机化（CAR）中的因果推断使用回归调整（RA）时存在的权衡进行了研究。RA可以通过整合未用于随机分配的协变量信息来提高因果估计器的效率。但是，当回归器数量与样本量同阶时，RA的估计误差不能渐近忽略，会降低估计效率。没有考虑RA成本的结果可能导致在零假设下过度拒绝因果推断。为了解决这个问题，我们在CAR下为经过回归调整的平均处理效应（ATE）估计器开发了一种统一的推断理论。我们的理论具有两个关键特征：（1）确保在零假设下的精确渐近大小，无论协变量数量是固定还是最多以样本大小的速度发散，（2）确保在协变量维度方面都比未调整的估计器弱效提高.

    Our paper identifies a trade-off when using regression adjustments (RAs) in causal inference under covariate-adaptive randomizations (CARs). On one hand, RAs can improve the efficiency of causal estimators by incorporating information from covariates that are not used in the randomization. On the other hand, RAs can degrade estimation efficiency due to their estimation errors, which are not asymptotically negligible when the number of regressors is of the same order as the sample size. Failure to account for the cost of RAs can result in over-rejection of causal inference under the null hypothesis. To address this issue, we develop a unified inference theory for the regression-adjusted average treatment effect (ATE) estimator under CARs. Our theory has two key features: (1) it ensures the exact asymptotic size under the null hypothesis, regardless of whether the number of covariates is fixed or diverges at most at the rate of the sample size, and (2) it guarantees weak efficiency impro
    
[^5]: 随机全球价值链中上游和下游之间的相关性

    Correlation between upstreamness and downstreamness in random global value chains. (arXiv:2303.06603v1 [stat.AP])

    [http://arxiv.org/abs/2303.06603](http://arxiv.org/abs/2303.06603)

    本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。

    This paper studies the upstreamness and downstreamness of industries and countries in global value chains, and finds a positive correlation between upstreamness and downstreamness of the same industrial sector.

    本文关注全球价值链中产业和国家的上游和下游。上游和下游分别衡量产业部门与最终消费和初级输入之间的平均距离，并基于最常用的全球投入产出表数据库（例如世界投入产出数据库（WIOD））进行计算。最近，Antr\`as和Chor在1995-2011年的数据中报告了一个令人困惑和反直觉的发现，即（在国家层面上）上游似乎与下游呈正相关，相关斜率接近+1。这种效应随时间和跨国家稳定存在，并已得到后续分析的确认和验证。我们分析了一个简单的随机投入产出表模型，并展示了在最小和现实的结构假设下，同一产业部门的上游和下游之间存在正相关性，具有相关性。

    This paper is concerned with upstreamness and downstreamness of industries and countries in global value chains. Upstreamness and downstreamness measure respectively the average distance of an industrial sector from final consumption and from primary inputs, and they are computed from based on the most used global Input-Output tables databases, e.g., the World Input-Output Database (WIOD). Recently, Antr\`as and Chor reported a puzzling and counter-intuitive finding in data from the period 1995-2011, namely that (at country level) upstreamness appears to be positively correlated with downstreamness, with a correlation slope close to $+1$. This effect is stable over time and across countries, and it has been confirmed and validated by later analyses. We analyze a simple model of random Input/Output tables, and we show that, under minimal and realistic structural assumptions, there is a positive correlation between upstreamness and downstreamness of the same industrial sector, with corre
    
[^6]: 媒体倾向是具有传染性的。

    Media Slant is Contagious. (arXiv:2202.07269v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2202.07269](http://arxiv.org/abs/2202.07269)

    本文研究了国家有线电视新闻对美国本土报纸的影响，发现当地报纸的内容会因为当地 FNC 观众数量的增加而趋向于 FNC 的倾向，并且有线电视倾向会极化地方新闻内容。

    

    本研究考察了媒体倾向的传播，具体来说是国家有线电视新闻对美国本土报纸（2005-2008）的影响。我们使用一种基于 Fox News Channel（FNC）、CNN 和 MSNBC 内容的有线电视倾向文本度量方法，分析地方报纸如何采用 FNC 的倾向而不是 CNN/MSNBC 的倾向。研究结果显示，地方新闻随着当地 FNC 观众人数的外部增长而变得更加类似于 FNC 的内容。这种转变不仅限于从有线电视借鉴，而是地方报纸自身内容的改变。此外，有线电视倾向极化了地方新闻内容。

    We examine the diffusion of media slant, specifically how partisan content from national cable news affects local newspapers in the U.S., 2005-2008. We use a text-based measure of cable news slant trained on content from Fox News Channel (FNC), CNN, and MSNBC to analyze how local newspapers adopt FNC's slant over CNN/MSNBC's. Our findings show that local news becomes more similar to FNC content in response to an exogenous increase in local FNC viewership. This shift is not limited to borrowing from cable news, but rather, local newspapers' own content changes. Further, cable TV slant polarizes local news content.
    

