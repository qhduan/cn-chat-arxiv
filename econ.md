# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating Causal Effects with Double Machine Learning -- A Method Evaluation](https://arxiv.org/abs/2403.14385) | 双重/无偏机器学习（DML）方法改进了因果效应估计中对非线性混淆关系的调整，摆脱传统函数形式假设，但仍然依赖于标准因果假设。 |
| [^2] | [A Gaussian smooth transition vector autoregressive model: An application to the macroeconomic effects of severe weather shocks](https://arxiv.org/abs/2403.14216) | 提出了一种新的高斯平滑过渡向量自回归模型，能够更好地捕捉复杂的转换动态，并在实证应用中发现美国经济逐渐适应了严重天气带来的影响。 |
| [^3] | [Fused LASSO as Non-Crossing Quantile Regression](https://arxiv.org/abs/2403.14036) | 通过扩展非交叉约束，通过变化单一超参数（$\alpha$），可以获得常用的分位数估计量，显示了非交叉约束只是融合收缩的一个特殊类型。 |
| [^4] | [Robust Communication Between Parties with Nearly Independent Preferences](https://arxiv.org/abs/2403.13983) | 通信博弈中当发送方偏好近乎独立时，鲁棒说服可能通过逼近透明偏好博弈的‘无环’均衡实现，前提是发送方的偏好满足一种单调性条件。 |
| [^5] | [Ex-Ante Design of Persuasion Games](https://arxiv.org/abs/2312.02465) | 研究了先见之明机制设计设置中的信息披露激励，提出了一种新的原则，揭示了全局激励约束受到有限后验信念下的“最坏情况”惩罚的影响，同时发现接收方在解决最优分配问题时将从对确定性结果进行随机化中受益。 |
| [^6] | [Identification and Statistical Decision Theory](https://arxiv.org/abs/2204.11318) | 此论文探讨了辨识分析对统计决策理论的重要性，发现辨识分析可以为有限样本性能提供上限，对于点辨识参数来说比较简单，而在部分辨识且存在模糊情况下做出决策时则更为复杂。 |
| [^7] | [Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments](https://arxiv.org/abs/1905.07812) | 提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。 |
| [^8] | [Designing Digital Voting Systems for Citizens: Achieving Fairness and Legitimacy in Digital Participatory Budgeting.](http://arxiv.org/abs/2310.03501) | 本研究调查了数字参与式预算中投票和聚合方法的权衡，并通过行为实验确定了有利的投票设计组合。研究发现，设计选择对集体决策、市民感知和结果公平性有深远影响，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程提供了可行的见解。 |
| [^9] | [The Dictator Equation: The Distortion of Information Flow in Autocratic Regimes and Its Consequences.](http://arxiv.org/abs/2310.01666) | 专制政权中的信息流扭曲及其后果的模型表明，在短期可以带来改善，但随后会导致国家逐渐恶化。顾问的欺骗程度与困难程度成正比。 |
| [^10] | [A Majority Rule Philosophy for Instant Runoff Voting.](http://arxiv.org/abs/2308.08430) | 在即时投票中，引入了有序多数规则的概念，它建立了一种社会顺序，确保了从多数党或联盟中选出候选人，并防止对立的反对党或联盟对选举结果的影响。有序多数规则与康多塞合规性、无关因素的独立性和单调性等特性不兼容，并且主张有序多数规则可能优于成对多数规则。 |
| [^11] | [Wildfire Modeling: Designing a Market to Restore Assets.](http://arxiv.org/abs/2205.13773) | 该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。 |
| [^12] | [Homophily in preferences or meetings? Identifying and estimating an iterative network formation model.](http://arxiv.org/abs/2201.06694) | 本文讨论同质性在社交和经济网络中产生的机制，提出了一种区分偏好和机会的迭代网络游戏的方法，并且进行了应用研究，发现偏好的同质性比机会的同质性更强，追踪学生可能会提高福利，但是效益随着时间推移而减弱。 |

# 详细

[^1]: 用双机器学习估计因果效应--一种方法评估

    Estimating Causal Effects with Double Machine Learning -- A Method Evaluation

    [https://arxiv.org/abs/2403.14385](https://arxiv.org/abs/2403.14385)

    双重/无偏机器学习（DML）方法改进了因果效应估计中对非线性混淆关系的调整，摆脱传统函数形式假设，但仍然依赖于标准因果假设。

    

    使用观测数据估计因果效应仍然是一个非常活跃的研究领域。近年来，研究人员开发了利用机器学习放宽传统假设以估计因果效应的新框架。在本文中，我们回顾了其中一个最重要的方法-"双/无偏机器学习"（DML），并通过比较它在模拟数据上相对于更传统的统计方法的表现，然后将其应用于真实世界数据进行了实证评估。我们的研究发现表明，在DML中应用一个适当灵活的机器学习算法可以改进对各种非线性混淆关系的调整。这种优势使得可以摆脱通常在因果效应估计中必需的传统函数形式假设。然而，我们表明该方法在关于因果关系的标准假设方面仍然至关重要。

    arXiv:2403.14385v1 Announce Type: cross  Abstract: The estimation of causal effects with observational data continues to be a very active research area. In recent years, researchers have developed new frameworks which use machine learning to relax classical assumptions necessary for the estimation of causal effects. In this paper, we review one of the most prominent methods - "double/debiased machine learning" (DML) - and empirically evaluate it by comparing its performance on simulated data relative to more traditional statistical methods, before applying it to real-world data. Our findings indicate that the application of a suitably flexible machine learning algorithm within DML improves the adjustment for various nonlinear confounding relationships. This advantage enables a departure from traditional functional form assumptions typically necessary in causal effect estimation. However, we demonstrate that the method continues to critically depend on standard assumptions about causal 
    
[^2]: 一种高斯平滑过渡向量自回归模型：对严重天气冲击的宏观经济影响的应用

    A Gaussian smooth transition vector autoregressive model: An application to the macroeconomic effects of severe weather shocks

    [https://arxiv.org/abs/2403.14216](https://arxiv.org/abs/2403.14216)

    提出了一种新的高斯平滑过渡向量自回归模型，能够更好地捕捉复杂的转换动态，并在实证应用中发现美国经济逐渐适应了严重天气带来的影响。

    

    我们介绍了一种新的平滑过渡向量自回归模型，其具有高斯条件分布和转换权重，对于第$p$阶模型，这些权重取决于前$p$个观测值的完整分布。具体而言，每个状态的转换权重随其相对加权似然性而增加。这种数据驱动方法有助于捕捉复杂的转换动态，增强逐渐状态转变的识别。在一个关于严重天气冲击对宏观经济影响的实证应用中，我们发现在1961年第1季度至2022年第3季度的美国月度数据中，冲击的影响在样本早期的状态和某些危机时期比在样本后期主导的状态中更为显著。这表明美国经济总体上逐渐适应了随时间增加的严重天气。

    arXiv:2403.14216v1 Announce Type: new  Abstract: We introduce a new smooth transition vector autoregressive model with a Gaussian conditional distribution and transition weights that, for a $p$th order model, depend on the full distribution of the preceding $p$ observations. Specifically, the transition weight of each regime increases in its relative weighted likelihood. This data-driven approach facilitates capturing complex switching dynamics, enhancing the identification of gradual regime shifts. In an empirical application to the macroeconomic effects of a severe weather shock, we find that in monthly U.S. data from 1961:1 to 2022:3, the impacts of the shock are stronger in the regime prevailing in the early part of the sample and in certain crisis periods than in the regime dominating the latter part of the sample. This suggests overall adaptation of the U.S. economy to increased severe weather over time.
    
[^3]: Fused LASSO作为非交叉分位数回归

    Fused LASSO as Non-Crossing Quantile Regression

    [https://arxiv.org/abs/2403.14036](https://arxiv.org/abs/2403.14036)

    通过扩展非交叉约束，通过变化单一超参数（$\alpha$），可以获得常用的分位数估计量，显示了非交叉约束只是融合收缩的一个特殊类型。

    

    分位数交叉一直是分位数回归中一个长久存在的问题，推动了对获得遵守分位数单调性性质的密度和系数的研究。本文扩展了非交叉约束，展示通过变化单一超参数（$\alpha$）可以获得常用的分位数估计量。具体来说，当 $\alpha=0$ 时，我们获得了Koenker和Bassett（1978）的分位数回归估计量，当 $\alpha=1$ 时，获得了Bondell等人（2010）的非交叉分位数回归估计量，当 $\alpha\rightarrow\infty$ 时，获得了Koenker（1984）和Zou以及Yuan（2008）的复合分位数回归估计量。因此，我们展示了非交叉约束只是融合收缩的一个特殊类型。

    arXiv:2403.14036v1 Announce Type: new  Abstract: Quantile crossing has been an ever-present thorn in the side of quantile regression. This has spurred research into obtaining densities and coefficients that obey the quantile monotonicity property. While important contributions, these papers do not provide insight into how exactly these constraints influence the estimated coefficients. This paper extends non-crossing constraints and shows that by varying a single hyperparameter ($\alpha$) one can obtain commonly used quantile estimators. Namely, we obtain the quantile regression estimator of Koenker and Bassett (1978) when $\alpha=0$, the non crossing quantile regression estimator of Bondell et al. (2010) when $\alpha=1$, and the composite quantile regression estimator of Koenker (1984) and Zou and Yuan (2008) when $\alpha\rightarrow\infty$. As such, we show that non-crossing constraints are simply a special type of fused-shrinkage.
    
[^4]: 具有近乎独立偏好的各方之间的鲁棒通信

    Robust Communication Between Parties with Nearly Independent Preferences

    [https://arxiv.org/abs/2403.13983](https://arxiv.org/abs/2403.13983)

    通信博弈中当发送方偏好近乎独立时，鲁棒说服可能通过逼近透明偏好博弈的‘无环’均衡实现，前提是发送方的偏好满足一种单调性条件。

    

    我们研究了有限状态通信博弈，在该博弈中，发送方的偏好受到随机私人特异性的扰动。在统计独立发送方/接收方偏好类别中，一般无法进行说服，这与先前的研究相反，先前的研究建立了当发送方偏好完全透明时的说服均衡。然而，当发送方的偏好仅略微依赖于状态/特异时，仍可能发生鲁棒说服。这需要逼近透明偏好博弈的‘无环’均衡，通常意味着这种均衡也是‘连接的’—这是对部分混合均衡的推广。然后，发送方的偏好相对于逼近均衡满足一种单调性条件是必要且充分的。 如果发送方的偏好进一步满足‘半局部’递增差异的版本，则这种分析将延伸到发送方

    arXiv:2403.13983v1 Announce Type: new  Abstract: We study finite-state communication games in which the sender's preference is perturbed by random private idiosyncrasies. Persuasion is generically impossible within the class of statistically independent sender/receiver preferences -- contrary to prior research establishing persuasive equilibria when the sender's preference is precisely transparent.   Nevertheless, robust persuasion may occur when the sender's preference is only slightly state-dependent/idiosyncratic. This requires approximating an `acyclic' equilibrium of the transparent preference game, generically implying that this equilibrium is also `connected' -- a generalization of partial-pooling equilibria. It is then necessary and sufficient that the sender's preference satisfy a monotonicity condition relative to the approximated equilibrium.   If the sender's preference further satisfies a `semi-local' version of increasing differences, then this analysis extends to sender 
    
[^5]: 先见之明的说服博弈设计

    Ex-Ante Design of Persuasion Games

    [https://arxiv.org/abs/2312.02465](https://arxiv.org/abs/2312.02465)

    研究了先见之明机制设计设置中的信息披露激励，提出了一种新的原则，揭示了全局激励约束受到有限后验信念下的“最坏情况”惩罚的影响，同时发现接收方在解决最优分配问题时将从对确定性结果进行随机化中受益。

    

    接受方的承诺如何影响贝叶斯说服中信息披露的激励？我们研究了许多发送方的说服博弈，其中单个接收方在发送方设计信息环境之前承诺一个后验依赖的行动配置或分配。我们在先见之明机制设计设置中提出了一种新的类似披露的原则，其中发送方报告是布莱克韦尔实验，并用它来表征我们模型中可实施的分配集合。我们展示了全局激励约束受到“最坏情况”惩罚的固定，该固定在有限后验信念中独立于分配的值。此外，当解决受约束的最优分配时，与标准机制设计模型形成对比，接收方将从能够对确定性结果进行随机化中获益。最后，我们将结果应用于分析多商品分配问题中的效率。

    arXiv:2312.02465v3 Announce Type: replace  Abstract: How does receiver commitment affect incentives for information revelation in Bayesian persuasion? We study many-sender persuasion games where a single receiver commits to a posterior-dependent action profile, or allocation, before senders design the informational environment. We develop a novel revelation-like principle for ex-ante mechanism design settings where sender reports are Blackwell experiments and use it to characterize the set of implementable allocations in our model. We show global incentive constraints are pinned down by "worst-case" punishments at finitely many posterior beliefs, whose values are independent of the allocation. Moreover, the receiver will generically benefit from the ability to randomize over deterministic outcomes when solving for the constrained optimal allocation, in contrast to standard mechanism design models. Finally, we apply our results to analyze efficiency in multi-good allocation problems, fu
    
[^6]: 辨识和统计决策理论

    Identification and Statistical Decision Theory

    [https://arxiv.org/abs/2204.11318](https://arxiv.org/abs/2204.11318)

    此论文探讨了辨识分析对统计决策理论的重要性，发现辨识分析可以为有限样本性能提供上限，对于点辨识参数来说比较简单，而在部分辨识且存在模糊情况下做出决策时则更为复杂。

    

    经济计量学家将估计研究有用地分为辨识和统计两部分。辨识分析假设已知生成观测数据的概率分布，为有限样本数据可学到的人口参数设定了上限。然而Wald的统计决策理论研究了只用样本数据做决策，没有提及辨识，事实上也没有提及估计。本文探讨了辨识分析对统计决策理论是否有用。答案是肯定的，因为它可以为决策标准的有限样本性能提供信息丰富且易于处理的上限。当决策相关参数为点估计时，推理很简单。当真实状态部分辨识且在模糊下必须做出决策时，推理就变得更加微妙。然后，某些准则的性能，s

    arXiv:2204.11318v2 Announce Type: replace  Abstract: Econometricians have usefully separated study of estimation into identification and statistical components. Identification analysis, which assumes knowledge of the probability distribution generating observable data, places an upper bound on what may be learned about population parameters of interest with finite sample data. Yet Wald's statistical decision theory studies decision making with sample data without reference to identification, indeed without reference to estimation. This paper asks if identification analysis is useful to statistical decision theory. The answer is positive, as it can yield an informative and tractable upper bound on the achievable finite sample performance of decision criteria. The reasoning is simple when the decision relevant parameter is point identified. It is more delicate when the true state is partially identified and a decision must be made under ambiguity. Then the performance of some criteria, s
    
[^7]: 连续内生变量和离散工具的非参数回归的迭代估计

    Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments

    [https://arxiv.org/abs/1905.07812](https://arxiv.org/abs/1905.07812)

    提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。

    

    我们考虑了一个具有连续内生独立变量的非参数回归模型，当只有与误差项独立的离散工具可用时。虽然这个框架在应用研究中非常相关，但其实现很麻烦，因为回归函数成为了非线性积分方程的解。我们提出了一个简单的迭代过程来估计这样的模型，并展示了一些其渐近性质。在一个模拟实验中，我们讨论了在工具变量为二进制时其实现细节。我们总结了一个实证应用，其中我们研究了美国几个县的房价对污染的影响。

    arXiv:1905.07812v2 Announce Type: replace  Abstract: We consider a nonparametric regression model with continuous endogenous independent variables when only discrete instruments are available that are independent of the error term. While this framework is very relevant for applied research, its implementation is cumbersome, as the regression function becomes the solution to a nonlinear integral equation. We propose a simple iterative procedure to estimate such models and showcase some of its asymptotic properties. In a simulation experiment, we discuss the details of its implementation in the case when the instrumental variable is binary. We conclude with an empirical application in which we examine the effect of pollution on house prices in a short panel of U.S. counties.
    
[^8]: 为市民设计数字投票系统：在数字参与式预算中实现公平和合法性

    Designing Digital Voting Systems for Citizens: Achieving Fairness and Legitimacy in Digital Participatory Budgeting. (arXiv:2310.03501v1 [cs.HC])

    [http://arxiv.org/abs/2310.03501](http://arxiv.org/abs/2310.03501)

    本研究调查了数字参与式预算中投票和聚合方法的权衡，并通过行为实验确定了有利的投票设计组合。研究发现，设计选择对集体决策、市民感知和结果公平性有深远影响，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程提供了可行的见解。

    

    数字参与式预算（PB）已成为城市资源分配的重要民主工具。在数字平台的支持下，新的投票输入格式和聚合方法已被利用。然而，实现公平和合法性仍然面临挑战。本研究调查了数字PB中各种投票和聚合方法之间的权衡。通过行为实验，我们确定了在认知负荷、比例和感知合法性方面的有利投票设计组合。研究揭示了设计选择如何深刻影响集体决策、市民感知和结果公平性。我们的发现为人机交互、机制设计和计算社会选择提供了可行的见解，为开发更公平和更透明的数字PB系统和市民的多胜者集体决策过程做出贡献。

    Digital Participatory Budgeting (PB) has become a key democratic tool for resource allocation in cities. Enabled by digital platforms, new voting input formats and aggregation have been utilised. Yet, challenges in achieving fairness and legitimacy persist. This study investigates the trade-offs in various voting and aggregation methods within digital PB. Through behavioural experiments, we identified favourable voting design combinations in terms of cognitive load, proportionality, and perceived legitimacy. The research reveals how design choices profoundly influence collective decision-making, citizen perceptions, and outcome fairness. Our findings offer actionable insights for human-computer interaction, mechanism design, and computational social choice, contributing to the development of fairer and more transparent digital PB systems and multi-winner collective decision-making process for citizens.
    
[^9]: 独裁者方程：专制政权中信息流的扭曲及其后果

    The Dictator Equation: The Distortion of Information Flow in Autocratic Regimes and Its Consequences. (arXiv:2310.01666v1 [nlin.AO])

    [http://arxiv.org/abs/2310.01666](http://arxiv.org/abs/2310.01666)

    专制政权中的信息流扭曲及其后果的模型表明，在短期可以带来改善，但随后会导致国家逐渐恶化。顾问的欺骗程度与困难程度成正比。

    

    人们对专制与民主政权的利弊争论已有数千年之久。例如，《理想国》中的柏拉图更青睐于精英专制政权，认为这是启蒙的政权形式，而非民主体制。现代独裁者通常在上台时承诺快速解决社会问题和长期稳定。我提出了一个以国家最佳利益为出发点的独裁模型。该模型基于以下前提：a) 独裁者只依赖顾问的信息来决定国家发展的方向；b) 顾问的欺骗不会随时间减少；c) 顾问的欺骗程度会随国家所遇到的困难而增加。该模型展示了短期改善（几个月到一年），随后出现不稳定情况，导致国家逐渐恶化多年。我推导出了一些适用于所有独裁者的普遍参数，并证明了顾问的欺骗程度与困难程度成正比。

    Humans have been arguing about the benefits of dictatorial versus democratic regimes for millennia. For example, Plato, in The Republic, favored Aristocracy,, the enlightened autocratic regime, to democracy. Modern dictators typically come to power promising quick solutions to societal problems and long-term stability}. I present a model of a dictatorship with the country's best interests in mind. The model is based on the following premises: a) the dictator forces the country to follow the desired trajectory of development only from the information from the advisors; b) the deception from the advisors cannot decrease in time; and c) the deception increases based on the difficulties the country encounters. The model shows an improvement in the short term (a few months to a year), followed by instability leading to the country's gradual deterioration over many years. I derive some universal parameters applicable to all dictators and show that advisors' deception increases in parallel wi
    
[^10]: 关于即时投票中多数规则理念的研究

    A Majority Rule Philosophy for Instant Runoff Voting. (arXiv:2308.08430v1 [econ.TH])

    [http://arxiv.org/abs/2308.08430](http://arxiv.org/abs/2308.08430)

    在即时投票中，引入了有序多数规则的概念，它建立了一种社会顺序，确保了从多数党或联盟中选出候选人，并防止对立的反对党或联盟对选举结果的影响。有序多数规则与康多塞合规性、无关因素的独立性和单调性等特性不兼容，并且主张有序多数规则可能优于成对多数规则。

    

    我们介绍了有序多数规则的概念，它是即时投票的一个特性，并将其与康多塞方法的成对多数规则进行了比较。有序多数规则在候选人中建立一种社会顺序，使得相对顺序由不倾向于其他主要候选人的选民决定。它确保了从多数党或联盟中选出一位候选人，并防止对立的反对党或联盟对可能成为候选人的选择产生影响。我们展示了即时投票是唯一满足有序多数规则的投票方法，对主要候选人和次要候选人进行了自洽确定，并且有序多数规则与康多塞合规性、无关因素的独立性和单调性等特性不兼容。最后，我们提出了一些关于为什么有序多数规则可能优于成对多数规则的论点，利用到2022年阿拉斯加的案例。

    We present the concept of ordered majority rule, a property of Instant Runoff Voting, and compare it to the familiar concept of pairwise majority rule of Condorcet methods. Ordered majority rule establishes a social order among the candidates such that that relative order between any two candidates is determined by voters who do not prefer another major candidate. It ensures the election of a candidate from the majority party or coalition while preventing an antagonistic opposition party or coalition from influencing which candidate that may be. We show how IRV is the only voting method to satisfy ordered majority rule, for a self-consistently determined distinction between major and minor candidates, and that ordered majority rule is incompatible with the properties of Condorcet compliance, independence of irrelevant alternatives, and monotonicity. Finally, we present some arguments as to why ordered majority rule may be preferable to pairwise majority rule, using the 2022 Alaska spec
    
[^11]: 森林火灾模型：设计市场以恢复资产

    Wildfire Modeling: Designing a Market to Restore Assets. (arXiv:2205.13773v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.13773](http://arxiv.org/abs/2205.13773)

    该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。

    

    在过去的十年里，夏季森林火灾已经成为加利福尼亚和美国的常态。这些火灾的原因多种多样。州政府会收取森林火灾基金来帮助受灾人员。然而，基金只在特定条件下发放，并且在整个加利福尼亚州均匀收取。因此，该项目的整体思路是寻找关于电力公司如何引发火灾以及如何帮助收取森林火灾基金或者公平收费以最大限度地实现社会影响的数量结果。该研究项目旨在提出与植被、输电线路相关的森林火灾风险，并将其与金钱挂钩。因此，该项目有助于解决与每个地点相关的森林火灾基金收取问题，并结合能源价格根据地点的森林火灾风险向客户收费，以实现社会的总盈余最大化。

    In the past decade, summer wildfires have become the norm in California, and the United States of America. These wildfires are caused due to variety of reasons. The state collects wildfire funds to help the impacted customers. However, the funds are eligible only under certain conditions and are collected uniformly throughout California. Therefore, the overall idea of this project is to look for quantitative results on how electrical corporations cause wildfires and how they can help to collect the wildfire funds or charge fairly to the customers to maximize the social impact. The research project aims to propose the implication of wildfire risk associated with vegetation, and due to power lines and incorporate that in dollars. Therefore, the project helps to solve the problem of collecting wildfire funds associated with each location and incorporate energy prices to charge their customers according to their wildfire risk related to the location to maximize the social surplus for the s
    
[^12]: 偏好或相遇中的同质性？识别并估计一种迭代网络形成模型。

    Homophily in preferences or meetings? Identifying and estimating an iterative network formation model. (arXiv:2201.06694v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.06694](http://arxiv.org/abs/2201.06694)

    本文讨论同质性在社交和经济网络中产生的机制，提出了一种区分偏好和机会的迭代网络游戏的方法，并且进行了应用研究，发现偏好的同质性比机会的同质性更强，追踪学生可能会提高福利，但是效益随着时间推移而减弱。

    

    社交和经济网络中的同质性是由于对同质性（偏好）的偏好还是更可能遇到具有类似属性的个体（机会）驱动的？本文研究了一种迭代网络游戏的识别和估计方法，可区分这两种机制。我们的方法使我们能够评估改变代理人之间会议协议的反事实效应。作为一种应用，我们研究偏好和会议在塑造巴西的课堂友谊网络中的作用。在一种网络结构中，由于偏好而产生的同质性比由于会议机会而产生的同质性更强，追踪学生可能会提高福利。然而，这项政策的相对效益随着学年的推移而减弱。

    Is homophily in social and economic networks driven by a taste for homogeneity (preferences) or by a higher probability of meeting individuals with similar attributes (opportunity)? This paper studies identification and estimation of an iterative network game that distinguishes between these two mechanisms. Our approach enables us to assess the counterfactual effects of changing the meeting protocol between agents. As an application, we study the role of preferences and meetings in shaping classroom friendship networks in Brazil. In a network structure in which homophily due to preferences is stronger than homophily due to meeting opportunities, tracking students may improve welfare. Still, the relative benefit of this policy diminishes over the school year.
    

