# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Massive Scale Semantic Similarity Dataset of Historical English.](http://arxiv.org/abs/2306.17810) | 本研究利用重新数字化的无版权美国本地报纸文章，构建了一个大规模的跨越了70年的语义相似性数据集，并包含近4亿个正向语义相似性对。 |
| [^2] | [Obvious Manipulations in Matching with and without Contracts.](http://arxiv.org/abs/2306.17773) | 在医生多对一匹配模型中，任何稳定的匹配规则都可以被医生操纵。但在有合同的模型中，医生最优匹配规则不容易被明显操纵，而医院最优匹配规则则容易被明显操纵。对于没有合同的多对一模型，医院最优匹配规则不容易被明显操纵。 |
| [^3] | [Two characterizations of the dense rank.](http://arxiv.org/abs/2306.17546) | 本文研究了在弱序中为替代方案分配位置的密集排名，并提出了两个特征化方法来确定密集排名。 |
| [^4] | [On the Behavior of the Payoff Amounts in Simple Interest Loans in Arbitrage-Free Markets.](http://arxiv.org/abs/2306.17467) | 本文研究了在无套利市场中的简单利息贷款的回报金额行为，并提出了适用于这种贷款的新的回报金额公式。 |
| [^5] | [Recurring Auctions with Costly Entry: Theory and Evidence.](http://arxiv.org/abs/2306.17355) | 重复拍卖在买方面临高成本参与时，在效率和收入方面优于单轮拍卖。原因是重复拍卖允许不同价值的买家在不同时间进入，从而节省了进入成本并增加了整体售卖概率。 |
| [^6] | [A Comparison of Sequential Ranked-Choice Voting and Single Transferable Vote.](http://arxiv.org/abs/2306.17341) | 顺序排名选择投票与单转移票数是两种用于多赢家选举的不同方法。通过研究发现，顺序排名选择投票常常产生与单转移票数不同的获胜者集合，且更关注卓越性而非实现比例性，可能会牺牲少数群体的利益。 |
| [^7] | [Retail Pricing Format and Rigidity of Regular Prices.](http://arxiv.org/abs/2306.17309) | 本研究通过对三家位于加拿大的不同定价格式商店的数据进行研究，发现不同的定价格式会影响正常价格的刚性，也会导致对价格下调和特价价格的定义有所不同。 |
| [^8] | [Bounded (O(1)) Regret Recommendation Learning via Synthetic Controls Oracle.](http://arxiv.org/abs/2301.12571) | 通过合成控制理论，本论文提出了一种实现有界遗憾的推荐学习方法，并解决了线性模型的精确知识、潜在协变量的存在、不均匀的用户到达速率和用户选择退出私人数据跟踪等实践中的问题。 |
| [^9] | [Optimal Scoring Rules for Multi-dimensional Effort.](http://arxiv.org/abs/2211.03302) | 本研究提出了用于优化激励多维努力的评分规则设计框架，并确定了两个简单的评分规则族，可以近似最优地激励代理人。最佳规则的近似最优性是鲁棒的。 |
| [^10] | [Variational inference for large Bayesian vector autoregressions.](http://arxiv.org/abs/2202.12644) | 我们提出了一种新颖的变分贝叶斯方法，用于估计高维向量自回归模型。该方法不依赖于传统的结构VAR表示，而是直接对回归系数矩阵进行层次收缩先验，并通过后验推理得到简化形式转换矩阵，使得估计结果更为稳健。我们的方法在模拟研究和实证分析中展现出良好的性能。 |
| [^11] | [Heckman-Selection or Two-Part models for alcohol studies? Depends.](http://arxiv.org/abs/2112.10542) | 本研究重新介绍了Heckman模型作为酒精研究中的有效经验技术，并通过使用Heckman模型和Two-Part估计模型对问题饮酒的决定因素进行估计。研究结果表明，在应用研究中，Heckman模型相较于Two-Part模型更具可行性和普适性，对选择偏差进行了纠正，并揭示了问题饮酒的影响因素。 |
| [^12] | [Dynamic Ordered Panel Logit Models.](http://arxiv.org/abs/2107.03253) | 本文研究了固定效应面板数据的动态有序Logit模型，构建了一组不受固定效应约束的有效矩条件，并给出了矩条件识别模型公共参数的充分条件。通过广义矩法估计公共参数，并用蒙特卡罗模拟和实证研究验证了估计器的性能。 |

# 详细

[^1]: 一个历史英语的大规模语义相似性数据集

    A Massive Scale Semantic Similarity Dataset of Historical English. (arXiv:2306.17810v1 [cs.CL])

    [http://arxiv.org/abs/2306.17810](http://arxiv.org/abs/2306.17810)

    本研究利用重新数字化的无版权美国本地报纸文章，构建了一个大规模的跨越了70年的语义相似性数据集，并包含近4亿个正向语义相似性对。

    

    各种任务使用在语义相似性数据上训练的语言模型。虽然有多种数据集可捕捉语义相似性，但它们要么是从现代网络数据构建的，要么是由人工标注员在过去十年中创建的相对较小的数据集。本研究利用一种新颖的来源，即重新数字化的无版权美国本地报纸文章，构建了一个大规模的语义相似性数据集，跨越了1920年到1989年的70年，并包含近4亿个正向语义相似性对。在美国本地报纸中，大约一半的文章来自新闻机构的新闻稿，而本地报纸复制了新闻稿的文章，并撰写了自己的标题，这些标题形成了与文章相关的提取性摘要。我们通过利用文档布局和语言理解将文章和标题关联起来。然后，我们使用深度神经方法来检测哪些文章来自相同的基础来源。

    A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in th
    
[^2]: 匹配中的明显操纵问题：有合同和无合同的情况。

    Obvious Manipulations in Matching with and without Contracts. (arXiv:2306.17773v1 [econ.TH])

    [http://arxiv.org/abs/2306.17773](http://arxiv.org/abs/2306.17773)

    在医生多对一匹配模型中，任何稳定的匹配规则都可以被医生操纵。但在有合同的模型中，医生最优匹配规则不容易被明显操纵，而医院最优匹配规则则容易被明显操纵。对于没有合同的多对一模型，医院最优匹配规则不容易被明显操纵。

    

    在一个医生多对一匹配模型中，无论是否有合同，医生的偏好是私有信息，医院的偏好是可替代的和公开的信息，任何稳定的匹配规则都可以被医生操纵。由于操纵无法完全避免，我们考虑了“明显操纵”的概念，并寻找至少可以防止这些操纵的稳定匹配规则（对医生而言）。对于有合同的模型，我们证明了：（i）医生最优匹配规则不容易被明显操纵；（ii）医院最优匹配规则即使在一对一模型中也容易被明显操纵。与（ii）相反，对于没有合同的多对一模型，我们证明了医院最优匹配规则不容易被明显操纵（当医院的偏好是可替代的）。此外，如果我们关注分位数稳定规则，则证明了医生最优匹配规则是唯一的不容易被明显操纵的规则。

    In a many-to-one matching model, with or without contracts, where doctors' preferences are private information and hospitals' preferences are substitutable and public information, any stable matching rule could be manipulated for doctors. Since manipulations can not be completely avoided, we consider the concept of \textit{obvious manipulations} and look for stable matching rules that prevent at least such manipulations (for doctors). For the model with contracts, we prove that: \textit{(i)} the doctor-optimal matching rule is non-obviously manipulable and \textit{(ii)} the hospital-optimal matching rule is obviously manipulable, even in the one-to-one model. In contrast to \textit{(ii)}, for a many-to-one model without contracts, we prove that the hospital-optimal matching rule is not obviously manipulable.% when hospitals' preferences are substitutable. Furthermore, if we focus on quantile stable rules, then we prove that the doctor-optimal matching rule is the only non-obviously man
    
[^3]: 密集排名的两个特征化方法

    Two characterizations of the dense rank. (arXiv:2306.17546v1 [econ.TH])

    [http://arxiv.org/abs/2306.17546](http://arxiv.org/abs/2306.17546)

    本文研究了在弱序中为替代方案分配位置的密集排名，并提出了两个特征化方法来确定密集排名。

    

    本文考虑了在弱序中为替代方案分配位置的密集排名。如果我们将替代方案按层次（即不确定类）排列，密集排名将将位置1分配给顶层所有替代方案，将位置2分配给第二层所有替代方案，依此类推。我们提出了一个形式化框架来分析密集排名与其他知名位置算子（如标准排名、修改排名和分数排名）的比较。作为主要结果，我们提供了两个不同的公理特征化方法，通过考虑水平扩展（复制）、垂直减少和移动（截断以及上下独立性）来确定密集排名。

    In this paper, we have considered the dense rank for assigning positions to alternatives in weak orders. If we arrange the alternatives in tiers (i.e., indifference classes), the dense rank assigns position 1 to all the alternatives in the top tier, 2 to all the alternatives in the second tier, and so on. We have proposed a formal framework to analyze the dense rank when compared to other well-known position operators such as the standard, modified and fractional ranks. As the main results, we have provided two different axiomatic characterizations which determine the dense rank by considering position invariance conditions along horizontal extensions (duplication), as well as through vertical reductions and movements (truncation, and upwards or downwards independency).
    
[^4]: 在无套利市场中的简单利息贷款的回报金额行为研究

    On the Behavior of the Payoff Amounts in Simple Interest Loans in Arbitrage-Free Markets. (arXiv:2306.17467v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.17467](http://arxiv.org/abs/2306.17467)

    本文研究了在无套利市场中的简单利息贷款的回报金额行为，并提出了适用于这种贷款的新的回报金额公式。

    

    消费者金融保护局定义了回报金额的概念，即在贷款提前偿还时，需要在特定时间支付的完全偿还债务的金额。在复利贷款中，这个金额被广泛理解，但在使用简单利息时不太清楚。最近，Aretusi和Mari（2018）提出了一种适用于简单利息贷款的回报金额公式。本文的第一个目标是研究这个新公式，并在一个贷款市场模型中推导出它，其中贷款以简单利息购买和出售，利率随时间变化，且不存在套利机会。第二个目标是展示这个公式表现出与复利贷款不同的行为。

    The Consumer Financial Protection Bureau defines the notion of payoff amount as the amount that has to be payed at a particular time in order to completely pay off the debt, in case the lender intends to pay off the loan early, way before the last installment is due (CFPB 2020). This amount is well-understood for loans at compound interest, but much less so when simple interest is used.  Recently, Aretusi and Mari (2018) have proposed a formula for the payoff amount for loans at simple interest. We assume that the payoff amounts are established contractually at time zero, whence the requirement that no arbitrage may arise this way  The first goal of this paper is to study this new formula and derive it within a model of a loan market in which loans are bought and sold at simple interest, interest rates change over time, and no arbitrage opportunities exist.  The second goal is to show that this formula exhibits a behaviour rather different from the one which occurs when compound intere
    
[^5]: 高成本参与的重复拍卖：理论与证据

    Recurring Auctions with Costly Entry: Theory and Evidence. (arXiv:2306.17355v1 [econ.TH])

    [http://arxiv.org/abs/2306.17355](http://arxiv.org/abs/2306.17355)

    重复拍卖在买方面临高成本参与时，在效率和收入方面优于单轮拍卖。原因是重复拍卖允许不同价值的买家在不同时间进入，从而节省了进入成本并增加了整体售卖概率。

    

    重复拍卖在出售耐用资产，如土地、房屋或艺术品方面普遍存在：当卖方在初始拍卖中无法出售物品时，她通常会在不久的将来举行下一个拍卖。本文在理论和实证两方面对重复拍卖的设计进行了表征。在理论方面，我们表明当潜在买家面临高成本参与时，重复拍卖在效率和收入方面优于单轮拍卖。这是因为重复拍卖允许具有不同价值的潜在买家在不同时间进入，从而降低了进入成本并增加了整体售卖概率。我们进一步推导了在重复拍卖中保留价格的最佳序列，这取决于卖方的目标是最大化效率还是收入。在实证方面，我们将理论应用于中国的房屋被收购拍卖中，一套被收购的房屋最多连续拍卖三次。在估计结构参数之后

    Recurring auctions are ubiquitous for selling durable assets, such as land, home, or artwork: When the seller cannot sell the item in the initial auction, she often holds a subsequent auction in the near future. This paper characterizes the design of recurring auctions, both theoretically and empirically. On the theoretical side, we show that recurring auctions outperform single-round auctions in efficiency and revenue when potential buyers face costly entry. This occurs because recurring auctions allow potential buyers with different values to enter at different times, which generates savings in entry costs and increases the overall probability of sale. We further derive the optimal sequence of reserve prices in recurring auctions, depending on whether the seller aims to maximize efficiency or revenue. On the empirical side, we apply the theory to home foreclosure auctions in China, where a foreclosed home is auctioned up to three times in a row. After estimating structural parameters
    
[^6]: 顺序排名选择投票和单转移票数的比较

    A Comparison of Sequential Ranked-Choice Voting and Single Transferable Vote. (arXiv:2306.17341v1 [econ.GN])

    [http://arxiv.org/abs/2306.17341](http://arxiv.org/abs/2306.17341)

    顺序排名选择投票与单转移票数是两种用于多赢家选举的不同方法。通过研究发现，顺序排名选择投票常常产生与单转移票数不同的获胜者集合，且更关注卓越性而非实现比例性，可能会牺牲少数群体的利益。

    

    单转移票数（STV）和顺序排名选择投票（RCV）是用于在多赢家选举中选出一组获胜者的不同方法。STV是一种经典的投票方法，在国际上已被广泛使用多年。相比之下，顺序RCV很少被使用，并且只有最近几年，随着犹他州几个城市采用这种方法选举市议会成员，使用率有所增加。通过蒙特卡洛模拟和大量真实选举数据，我们比较了顺序RCV和STV的行为。我们的总体发现是，顺序RCV通常产生不同的获胜者集合。此外，顺序RCV最好理解为一种基于卓越性的方法，不会产生比例结果，这往往以少数利益为代价。

    The methods of single transferable vote (STV) and sequential ranked-choice voting (RCV) are different methods for electing a set of winners in multiwinner elections. STV is a classical voting method that has been widely used internationally for many years. By contrast, sequential RCV has rarely been used, and only recently has seen an increase in usage as several cities in Utah have adopted the method to elect city council members. We use Monte Carlo simulations and a large database of real-world ranked-choice elections to investigate the behavior of sequential RCV by comparing it to STV. Our general finding is that sequential RCV often produces different winner sets than STV. Furthermore, sequential RCV is best understood as an excellence-based method which will not produce proportional results, often at the expense of minority interests.
    
[^7]: 零售定价格式与正常价格的刚性研究

    Retail Pricing Format and Rigidity of Regular Prices. (arXiv:2306.17309v1 [econ.GN])

    [http://arxiv.org/abs/2306.17309](http://arxiv.org/abs/2306.17309)

    本研究通过对三家位于加拿大的不同定价格式商店的数据进行研究，发现不同的定价格式会影响正常价格的刚性，也会导致对价格下调和特价价格的定义有所不同。

    

    本文研究了正常价格和特价价格的价格刚性，以及它们如何受到定价格式（定价策略）的影响。我们使用了三个位于彼此1公里半径内的加拿大大型商店的数据，这些商店具有不同的定价格式（每日低价，高低价和混合）。我们的数据包含实际交易价格和在商店货架上显示的实际正常价格。我们将这些数据与两个生成的正常价格序列（过滤价格和参考价格）结合起来进行研究其刚性。由于不同的商店格式对待特价价格的方式不同，因此正常价格的刚性因商店格式而异，从而导致对正常价格的定义不同。相应地，价格下调和特价价格在不同的商店格式下的含义也不同。为了解释研究结果，我们考虑了美国各种商店定价格式的分布情况。

    We study the price rigidity of regular and sale prices, and how it is affected by pricing formats (pricing strategies). We use data from three large Canadian stores with different pricing formats (Every-Day-Low-Price, Hi-Lo, and Hybrid) that are located within a 1 km radius of each other. Our data contains both the actual transaction prices and actual regular prices as displayed on the store shelves. We combine these data with two generated regular price series (filtered prices and reference prices) and study their rigidity. Regular price rigidity varies with store formats because different format stores treat sale prices differently, and consequently define regular prices differently. Correspondingly, the meanings of price cuts and sale prices vary across store formats. To interpret the findings, we consider the store pricing format distribution across the US.
    
[^8]: 通过合成控制理论实现有界（O(1)）遗憾的推荐学习

    Bounded (O(1)) Regret Recommendation Learning via Synthetic Controls Oracle. (arXiv:2301.12571v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12571](http://arxiv.org/abs/2301.12571)

    通过合成控制理论，本论文提出了一种实现有界遗憾的推荐学习方法，并解决了线性模型的精确知识、潜在协变量的存在、不均匀的用户到达速率和用户选择退出私人数据跟踪等实践中的问题。

    

    在在线探索系统中，当具有固定偏好的用户重复到达时，最近已经证明可以将系统建模为线性情境广告带来O(1)的有界遗憾。这个结果可能对推荐系统具有兴趣，因为它们的物品的流行度通常是短暂的，即探索本身可能在潜在的长期非稳态开始之前很快完成。然而，在实践中，线性模型的精确知识往往难以证明。此外，潜在协变量的存在，不均匀的用户到达速率，对必要等级条件的解释以及用户选择退出私人数据跟踪等都需要在实际的推荐系统应用中解决。在这项工作中，我们进行了理论研究，以解决所有这些问题，同时仍然实现了有界遗憾。除了证明技术，我们在这里所做的关键区别性假设是有效合成控制理论的存在。

    In online exploration systems where users with fixed preferences repeatedly arrive, it has recently been shown that O(1), i.e., bounded regret, can be achieved when the system is modeled as a linear contextual bandit. This result may be of interest for recommender systems, where the popularity of their items is often short-lived, as the exploration itself may be completed quickly before potential long-run non-stationarities come into play. However, in practice, exact knowledge of the linear model is difficult to justify. Furthermore, potential existence of unobservable covariates, uneven user arrival rates, interpretation of the necessary rank condition, and users opting out of private data tracking all need to be addressed for practical recommender system applications. In this work, we conduct a theoretical study to address all these issues while still achieving bounded regret. Aside from proof techniques, the key differentiating assumption we make here is the presence of effective Sy
    
[^9]: 多维努力的优化评分规则

    Optimal Scoring Rules for Multi-dimensional Effort. (arXiv:2211.03302v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2211.03302](http://arxiv.org/abs/2211.03302)

    本研究提出了用于优化激励多维努力的评分规则设计框架，并确定了两个简单的评分规则族，可以近似最优地激励代理人。最佳规则的近似最优性是鲁棒的。

    

    本文提出了一种用于设计评分规则的框架，以最佳方式激励代理人进行多维努力。这个框架是对经典背包问题（cf. Briest, Krysta, and V\"ocking, 2005, Singer, 2010）的战略代理人的推广，并且是将算法机制设计应用于教室的基础。本文确定了两个简单的评分规则族，可以保证对最优评分规则进行常数逼近。截断分离评分规则是一组单维评分规则的总和，被截断在可行得分的有界范围内。阈值评分规则在报告超过阈值时给出最高得分，否则为零。这两种简单评分规则的近似最优性类似于Babaioff, Immorlica, Lucier, and Weinberg（2014）的捆绑或单独销售结果。最后，我们展示了这两种简单评分规则中最佳规则的近似最优性是鲁棒的。

    This paper develops a framework for the design of scoring rules to optimally incentivize an agent to exert a multi-dimensional effort. This framework is a generalization to strategic agents of the classical knapsack problem (cf. Briest, Krysta, and V\"ocking, 2005, Singer, 2010) and it is foundational to applying algorithmic mechanism design to the classroom. The paper identifies two simple families of scoring rules that guarantee constant approximations to the optimal scoring rule. The truncated separate scoring rule is the sum of single dimensional scoring rules that is truncated to the bounded range of feasible scores. The threshold scoring rule gives the maximum score if reports exceed a threshold and zero otherwise. Approximate optimality of one or the other of these rules is similar to the bundling or selling separately result of Babaioff, Immorlica, Lucier, and Weinberg (2014). Finally, we show that the approximate optimality of the best of those two simple scoring rules is robu
    
[^10]: 大规模贝叶斯向量自回归的变分推理

    Variational inference for large Bayesian vector autoregressions. (arXiv:2202.12644v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2202.12644](http://arxiv.org/abs/2202.12644)

    我们提出了一种新颖的变分贝叶斯方法，用于估计高维向量自回归模型。该方法不依赖于传统的结构VAR表示，而是直接对回归系数矩阵进行层次收缩先验，并通过后验推理得到简化形式转换矩阵，使得估计结果更为稳健。我们的方法在模拟研究和实证分析中展现出良好的性能。

    

    我们提出了一种新颖的变分贝叶斯方法，用于估计具有层次收缩先验的高维向量自回归（VAR）模型。我们的方法不依赖于传统的结构VAR参数空间的后验推理。相反，我们直接对回归系数矩阵进行层次收缩先验，以便（1）先验结构直接映射到对简化形式转换矩阵的后验推理，（2）后验估计对变量置换更稳健。广泛的模拟研究证明，我们的方法与现有的线性和非线性马尔科夫链蒙特卡罗和变分贝叶斯方法相比具有优势。在一大组不同行业投资组合的均值-方差投资者的背景下，我们研究了我们变分推理方法的预测的统计和经济价值。结果表明，我们的方法可以得到更准确的估计。

    We propose a novel variational Bayes approach to estimate high-dimensional vector autoregression (VAR) models with hierarchical shrinkage priors. Our approach does not rely on a conventional structural VAR representation of the parameter space for posterior inference. Instead, we elicit hierarchical shrinkage priors directly on the matrix of regression coefficients so that (1) the prior structure directly maps into posterior inference on the reduced-form transition matrix, and (2) posterior estimates are more robust to variables permutation. An extensive simulation study provides evidence that our approach compares favourably against existing linear and non-linear Markov Chain Monte Carlo and variational Bayes methods. We investigate both the statistical and economic value of the forecasts from our variational inference approach within the context of a mean-variance investor allocating her wealth in a large set of different industry portfolios. The results show that more accurate estim
    
[^11]: Heckman-Selection或者Two-Part模型在酒精研究中的应用？取决于情况。

    Heckman-Selection or Two-Part models for alcohol studies? Depends. (arXiv:2112.10542v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.10542](http://arxiv.org/abs/2112.10542)

    本研究重新介绍了Heckman模型作为酒精研究中的有效经验技术，并通过使用Heckman模型和Two-Part估计模型对问题饮酒的决定因素进行估计。研究结果表明，在应用研究中，Heckman模型相较于Two-Part模型更具可行性和普适性，对选择偏差进行了纠正，并揭示了问题饮酒的影响因素。

    

    目的：重新介绍Heckman模型作为酒精研究中有效的经验技术。设计：使用Heckman模型和Two-Part估计模型估计问题饮酒的决定因素。心理学和神经科学研究证明了我基本的估计假设和协变量排除限制的合理性。高阶检验验证了在应用研究中使用Heckman模型而不是Two-Part估计模型的可行性。我讨论了这两个模型在应用研究中的普适性。研究场景和参与者：使用2016年和2017年的两个全国人口调查数据集：行为风险因素监测调查（BRFS）和国家药物使用和健康调查（NSDUH）。测量：问题饮酒的参与和满足问题饮酒的标准。发现：美国的两项全国调查都在Heckman模型下表现良好，并通过了所有的高阶检验。Heckman模型纠正了选择偏差，揭示了生物社会学因素和经济因素对问题饮酒的影响方向。

    Aims: To re-introduce the Heckman model as a valid empirical technique in alcohol studies. Design: To estimate the determinants of problem drinking using a Heckman and a two-part estimation model. Psychological and neuro-scientific studies justify my underlying estimation assumptions and covariate exclusion restrictions. Higher order tests checking for multicollinearity validate the use of Heckman over the use of two-part estimation models. I discuss the generalizability of the two models in applied research. Settings and Participants: Two pooled national population surveys from 2016 and 2017 were used: the Behavioral Risk Factor Surveillance Survey (BRFS), and the National Survey of Drug Use and Health (NSDUH). Measurements: Participation in problem drinking and meeting the criteria for problem drinking. Findings: Both U.S. national surveys perform well with the Heckman model and pass all higher order tests. The Heckman model corrects for selection bias and reveals the direction of bi
    
[^12]: 动态有序面板Logit模型

    Dynamic Ordered Panel Logit Models. (arXiv:2107.03253v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2107.03253](http://arxiv.org/abs/2107.03253)

    本文研究了固定效应面板数据的动态有序Logit模型，构建了一组不受固定效应约束的有效矩条件，并给出了矩条件识别模型公共参数的充分条件。通过广义矩法估计公共参数，并用蒙特卡罗模拟和实证研究验证了估计器的性能。

    

    本文研究了一个用于固定效应面板数据的动态有序Logit模型。本文的主要贡献是构建了一组不受固定效应约束的有效矩条件。这些矩函数可以利用四个或更多期的数据计算，并且本文给出了矩条件能够识别模型的公共参数（回归系数、自回归参数和阈值参数）的充分条件。矩条件的可利用性表明可以利用广义矩法估计这些公共参数，本文通过蒙特卡罗模拟和利用英国家庭面板调查的自报健康状况进行经验说明，评估了该估计器的性能。

    This paper studies a dynamic ordered logit model for panel data with fixed effects. The main contribution of the paper is to construct a set of valid moment conditions that are free of the fixed effects. The moment functions can be computed using four or more periods of data, and the paper presents sufficient conditions for the moment conditions to identify the common parameters of the model, namely the regression coefficients, the autoregressive parameters, and the threshold parameters. The availability of moment conditions suggests that these common parameters can be estimated using the generalized method of moments, and the paper documents the performance of this estimator using Monte Carlo simulations and an empirical illustration to self-reported health status using the British Household Panel Survey.
    

