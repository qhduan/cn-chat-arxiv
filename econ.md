# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Machine Learning Catch Economic Recessions Using Economic and Market Sentiments?.](http://arxiv.org/abs/2308.16200) | 本研究通过使用市场情绪和经济指标的机器学习技术，以月度为频率，预测美国的经济衰退，解决了缺失数据点的问题，并利用降维、相关分析和解决多重共线性问题进行模型分析。 |
| [^2] | [High Dimensional Time Series Regression Models: Applications to Statistical Learning Methods.](http://arxiv.org/abs/2308.16192) | 这篇论文概述了高维时间序列回归模型的方法和最新发展，包括相关数据的极限理论结果、与许多协变量的时间序列回归模型相关的渐近理论和统计学习方法的多种应用。 |
| [^3] | [Cognitive Aging and Labor Share.](http://arxiv.org/abs/2308.14982) | 该研究将劳动份额的下降与认知衰老联系起来，提出了一个新颖的宏观经济模型。模型表明，工业化导致人口老龄化，老龄消费者认知能力的下降减少了对新产出变体的需求，从而导致劳动份额的下降。 |
| [^4] | [Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings.](http://arxiv.org/abs/2308.10131) | 该研究使用自我关注模块的深度学习模型，根据FOMC会议的异议记录和会议记录，测量了每位成员在每个会议中的异议程度。研究发现，尽管异议很少见，成员们经常对政策决策持保留意见。异议程度主要受到当前或预测的宏观经济数据的影响，而成员的个人特征几乎不起作用。此外，研究还发现了会议之间成员的演讲与随后会议的异议程度之间存在弱相关性。最后，研究发现，每当货币政策行动更加激进时，异议程度会增加。 |
| [^5] | [Measuring Value Added in Gross Trade: Endogenous Approach of Vertical Differentiation.](http://arxiv.org/abs/2307.10660) | 本论文讨论了一种用于区分和衡量垂直差异产业内贸易的替代方法。 |
| [^6] | [Unraveling Coordination Problems.](http://arxiv.org/abs/2307.08557) | 本文通过解析协调问题，建立了一种特殊的补贴设计模型，首次发现了“解谜效应”，通过对补贴的特性进行表征，找到实现给定博弈结果的唯一均衡。该模型具有对称性，全局连续性，随机机会成本的增加和负溢出效应的减少。应用包括联合投资问题、参与决策和委托代理合同。 |
| [^7] | [Difference-in-Differences with Interference: A Finite Population Perspective.](http://arxiv.org/abs/2306.12003) | 本文提出了一种新的双重稳健估计器，以应对邻居干扰对于“区别于差分”类型的估计器效果的影响。在本文提出的有限人口视角下，可以处理直接平均处理效应以及平均溢出效应，具有一定的研究价值。 |
| [^8] | [Dynamic Transportation of Economic Agents.](http://arxiv.org/abs/2303.12567) | 本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。 |
| [^9] | [Quality Selection in Two-Sided Markets: A Constrained Price Discrimination Approach.](http://arxiv.org/abs/1912.02251) | 本论文研究了两边市场中的信息披露问题，探讨了平台应该允许哪些卖家参与以及与买家分享多少关于卖家质量的信息。研究结果表明，在特定条件下，简单的信息结构可以最大化平台的收入。 |
| [^10] | [Identifying the Effects of a Program Offer with an Application to Head Start.](http://arxiv.org/abs/1711.02048) | 该论文提出了一种治疗选择模型，通过分析Head Start学前计划的数据，发现方案提供对于参与者有积极效果，特别是那些没有其他学前教育选择的孩子。通过成本效益分析，发现与考试成绩提高相关的收入效益超过了方案提供的净成本。 |

# 详细

[^1]: 机器学习能否利用经济和市场情绪预测经济衰退？

    Can Machine Learning Catch Economic Recessions Using Economic and Market Sentiments?. (arXiv:2308.16200v1 [econ.EM])

    [http://arxiv.org/abs/2308.16200](http://arxiv.org/abs/2308.16200)

    本研究通过使用市场情绪和经济指标的机器学习技术，以月度为频率，预测美国的经济衰退，解决了缺失数据点的问题，并利用降维、相关分析和解决多重共线性问题进行模型分析。

    

    定量模型对政策制定者和投资者来说是重要的决策因素。准确可靠地预测经济衰退将对社会非常有益。本文评估了使用市场情绪和经济指标（75个解释变量）从1986年1月至2022年6月的美国经济衰退预测的机器学习技术，以月度为频率。为了解决缺失的时间序列数据点的问题，使用自回归积分滑动平均（ARIMA）方法对解释变量进行反向预测。分析从使用Boruta算法、相关矩阵和解决多重共线性问题对高维数据集进行降维开始。然后，建立了各种交叉验证模型，包括概率回归方法和机器学习技术，预测衰退二进制结果。考虑的方法包括Probit、Logit、弹性网络、随机森林、渐变。

    Quantitative models are an important decision-making factor for policy makers and investors. Predicting an economic recession with high accuracy and reliability would be very beneficial for the society. This paper assesses machine learning technics to predict economic recessions in United States using market sentiment and economic indicators (seventy-five explanatory variables) from Jan 1986 - June 2022 on a monthly basis frequency. In order to solve the issue of missing time-series data points, Autoregressive Integrated Moving Average (ARIMA) method used to backcast explanatory variables. Analysis started with reduction in high dimensional dataset to only most important characters using Boruta algorithm, correlation matrix and solving multicollinearity issue. Afterwards, built various cross-validated models, both probability regression methods and machine learning technics, to predict recession binary outcome. The methods considered are Probit, Logit, Elastic Net, Random Forest, Gradi
    
[^2]: 高维时间序列回归模型：应用于统计学习方法的研究

    High Dimensional Time Series Regression Models: Applications to Statistical Learning Methods. (arXiv:2308.16192v1 [econ.EM])

    [http://arxiv.org/abs/2308.16192](http://arxiv.org/abs/2308.16192)

    这篇论文概述了高维时间序列回归模型的方法和最新发展，包括相关数据的极限理论结果、与许多协变量的时间序列回归模型相关的渐近理论和统计学习方法的多种应用。

    

    这些讲稿概述了高维时间序列回归模型的现有方法和最新发展，包括高维相关数据的主要极限理论结果、与许多协变量的时间序列回归模型相关的渐近理论主要方面以及统计学习方法在时间序列分析中的各种应用。

    These lecture notes provide an overview of existing methodologies and recent developments for estimation and inference with high dimensional time series regression models. First, we present main limit theory results for high dimensional dependent data which is relevant to covariance matrix structures as well as to dependent time series sequences. Second, we present main aspects of the asymptotic theory related to time series regression models with many covariates. Third, we discuss various applications of statistical learning methodologies for time series analysis purposes.
    
[^3]: 认知衰老与劳动份额

    Cognitive Aging and Labor Share. (arXiv:2308.14982v1 [econ.GN])

    [http://arxiv.org/abs/2308.14982](http://arxiv.org/abs/2308.14982)

    该研究将劳动份额的下降与认知衰老联系起来，提出了一个新颖的宏观经济模型。模型表明，工业化导致人口老龄化，老龄消费者认知能力的下降减少了对新产出变体的需求，从而导致劳动份额的下降。

    

    劳动份额，即经济产出的工资比例，在工业化国家中不可理解地在下降。虽然许多之前的研究试图通过经济因素来解释这种下降，但我们的新颖方法将这种下降与生物因素联系起来。具体而言，我们提出了一个理论宏观经济模型，劳动份额反映了劳动力自动化现有产出和消费者需求新的依赖人力劳动的产出变体之间的动态平衡。工业化导致人口老龄化，虽然在工作年限内认知表现稳定，但之后急剧下降。因此，老龄消费者认知能力的下降减少了对新的产出变体的需求，导致劳动份额下降。我们的模型将劳动份额表达为中位数年龄的代数函数，并通过非线性随机回归在工业化经济体的历史数据上以惊人的准确性进行了验证。

    Labor share, the fraction of economic output accrued as wages, is inexplicably declining in industrialized countries. Whilst numerous prior works attempt to explain the decline via economic factors, our novel approach links the decline to biological factors. Specifically, we propose a theoretical macroeconomic model where labor share reflects a dynamic equilibrium between the workforce automating existing outputs, and consumers demanding new output variants that require human labor. Industrialization leads to an aging population, and while cognitive performance is stable in the working years it drops sharply thereafter. Consequently, the declining cognitive performance of aging consumers reduces the demand for new output variants, leading to a decline in labor share. Our model expresses labor share as an algebraic function of median age, and is validated with surprising accuracy on historical data across industrialized economies via non-linear stochastic regression.
    
[^4]: 持不同意见：测量FOMC会议中的隐藏异议

    Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings. (arXiv:2308.10131v1 [econ.GN])

    [http://arxiv.org/abs/2308.10131](http://arxiv.org/abs/2308.10131)

    该研究使用自我关注模块的深度学习模型，根据FOMC会议的异议记录和会议记录，测量了每位成员在每个会议中的异议程度。研究发现，尽管异议很少见，成员们经常对政策决策持保留意见。异议程度主要受到当前或预测的宏观经济数据的影响，而成员的个人特征几乎不起作用。此外，研究还发现了会议之间成员的演讲与随后会议的异议程度之间存在弱相关性。最后，研究发现，每当货币政策行动更加激进时，异议程度会增加。

    

    基于1976年至2017年的FOMC投票异议记录和会议记录，我们开发了一个基于自我关注模块的深度学习模型，用于确定每个成员在每个会议中的异议程度。虽然异议很少见，但我们发现成员们经常对政策决策持保留意见。异议程度主要由当前或预测的宏观经济数据驱动，成员的个人特征几乎不起作用。我们还利用模型评估会议之间成员的演讲，并发现它们所揭示的异议程度与随后的会议异议程度之间存在弱相关性。最后，我们发现每当货币政策行动更加激进时，异议程度会增加。

    Based on a record of dissents on FOMC votes and transcripts of the meetings from 1976 to 2017, we develop a deep learning model based on self-attention modules to create a measure of the level of disagreement for each member in each meeting. While dissents are rare, we find that members often have reservations with the policy decision. The level of disagreement is mostly driven by current or predicted macroeconomic data, and personal characteristics of the members play almost no role. We also use our model to evaluate speeches made by members between meetings, and we find a weak correlation between the level of disagreement revealed in them and that of the following meeting. Finally, we find that the level of disagreement increases whenever monetary policy action is more aggressive.
    
[^5]: 测量毛贸易中的增加值: 垂直差异的内生方法

    Measuring Value Added in Gross Trade: Endogenous Approach of Vertical Differentiation. (arXiv:2307.10660v1 [econ.TH])

    [http://arxiv.org/abs/2307.10660](http://arxiv.org/abs/2307.10660)

    本论文讨论了一种用于区分和衡量垂直差异产业内贸易的替代方法。

    

    从1980年代开始，对于产业内贸易的第一次理论分析表明，这种贸易的决定因素和后果取决于交易产品是否在质量上存在差异。当交易产品在两个具有不同质量的国家之间进行产业内贸易时，这种贸易被称为垂直差异。否则，被称为水平差异。有一种方法可以区分垂直差异中的产业内贸易和水平差异中的产业内贸易。该方法比较了每个产业的产业内贸易中出口的单价与进口的单价。当出口的单价与进口的单价存在显著差异时，认为该产业内的产业内贸易是垂直差异。但是这种方法存在局限性。下文将引导我们思考一种替代方法，用于区分和衡量产业内贸易。

    From the beginning of the 1980s, the first theoretical analysis of intra-industry trade showed that the determinants and consequences of this type of trade are different, depending on whether the traded products differ in quality. When the products are subject to intra-industry trade between two countries with distinct qualities, this trade is vertically differentiated. Otherwise, it is called horizontal differentiation. There is a method for distinguishing intra-industry trade between two countries in vertical differentiation from those in horizontal differentiation. This method compares exports' unit value to imports for each industry's intra-industry trade. It considers the intra-industry trading carried out in this industry as vertical differentiation when the unit value of exports differs significantly from that of imports. This approach has limitations. The discussion below will lead us to think about an alternative method for separating and measuring intra-industry trade into ho
    
[^6]: 解析协调问题

    Unraveling Coordination Problems. (arXiv:2307.08557v1 [econ.TH])

    [http://arxiv.org/abs/2307.08557](http://arxiv.org/abs/2307.08557)

    本文通过解析协调问题，建立了一种特殊的补贴设计模型，首次发现了“解谜效应”，通过对补贴的特性进行表征，找到实现给定博弈结果的唯一均衡。该模型具有对称性，全局连续性，随机机会成本的增加和负溢出效应的减少。应用包括联合投资问题、参与决策和委托代理合同。

    

    本文研究协调问题中的政策设计。在协调博弈中，补贴提高了玩家i选择补贴行动的动机。这进一步提高了j选择相同行动的动机，进而激励i，依此类推。基于这种“解谜效应”，我们对实施给定博弈结果的补贴进行了表征，将其唯一均衡。在其他属性中，我们确定了以下特点：（i）对于相同的参与者是对称的；（ii）在模型参数上全局连续；（iii）与机会成本正相关；（iv）与溢出效应负相关。该模型的应用包括联合投资问题、参与决策和委托代理合同。

    This paper studies policy design in coordination problems. In coordination games, a subsidy raises player $i$'s incentive to play the subsidized action. This raises $j$'s incentive to play the same action, which further incentivizes $i$, and so on. Building upon this ``unraveling effect'', we characterize the subsidies that implement a given outcome of the game as its unique equilibrium. Among other properties, we establish that subsidies are: (i) symmetric for identical players; (ii) globally continuous in model parameters; (iii) increasing in opportunity costs; and (iv) decreasing in spillovers. Applications of the model include joint investment problems, participation decisions, and principal-agent contracting.
    
[^7]: 区别于差分的干扰效应：有限人口视角下的研究

    Difference-in-Differences with Interference: A Finite Population Perspective. (arXiv:2306.12003v1 [econ.EM])

    [http://arxiv.org/abs/2306.12003](http://arxiv.org/abs/2306.12003)

    本文提出了一种新的双重稳健估计器，以应对邻居干扰对于“区别于差分”类型的估计器效果的影响。在本文提出的有限人口视角下，可以处理直接平均处理效应以及平均溢出效应，具有一定的研究价值。

    

    在许多情况下，如对基于地点的政策进行评估时，潜在结果不仅取决于单位本身的处理，还取决于其邻居的处理。尽管如此，“区别于差分”（DID）类型的估计器通常忽略邻居之间的干扰。本文指出，在邻居干扰存在的情况下，经典的DID估计器通常无法识别有趣的因果效应。为了将干扰结构引入DID估计中，笔者提出了双重稳健估计器，用于处理直接平均处理效应以及改进后的平行趋势假设下的平均溢出效应。当需要考虑溢出效应时，通常会对整个人口进行抽样。因此，本文采用了有限人口视角，即将估计量定义为人口平均值，并且推断是有条件的，基于所有人口单位的属性。本文提出的一般和统一的方法将干扰效应纳入到DID估计中，具有一定的研究价值。

    In many scenarios, such as the evaluation of place-based policies, potential outcomes are not only dependent upon the unit's own treatment but also its neighbors' treatment. In spite of this, the "difference-in-differences" (DID) type estimators typically ignore such interference among neighbors. I show in this paper that the canonical DID estimators generally do not identify interesting causal effects in the presence of neighborhood interference. To incorporate interference structure into DID estimation, I propose doubly robust estimators for the direct average treatment effect on the treated as well as the average spillover effects under a modified parallel trends assumption. When spillover effects are of interest, we often sample the entire population. Thus, I adopt a finite population perspective in the sense that the estimands are defined as population averages and inference is conditional on the attributes of all population units. The general and unified approach in this paper re
    
[^8]: 经济主体的动态运输

    Dynamic Transportation of Economic Agents. (arXiv:2303.12567v1 [econ.GN])

    [http://arxiv.org/abs/2303.12567](http://arxiv.org/abs/2303.12567)

    本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。

    

    本文是在发现了一个共同的策略未能将某些异质性代理人不完全市场模型的宏观经济均衡定位到广泛引用的基准研究中而引发的。通过模仿Dumas和Lyasoff（2012）提出的方法，本文提供了一个新的描述，在面对不可保险的总体和个体风险的大量互动经济体代表的私人状态分布的运动定律。提出了一种新的算法，用于确定回报、最优私人配置和平衡状态下的人口运输，并在两个众所周知的基准研究中进行了测试。

    The paper was prompted by the surprising discovery that the common strategy, adopted in a large body of research, for producing macroeconomic equilibrium in certain heterogeneous-agent incomplete-market models fails to locate the equilibrium in a widely cited benchmark study. By mimicking the approach proposed by Dumas and Lyasoff (2012), the paper provides a novel description of the law of motion of the distribution over the range of private states of a large population of interacting economic agents faced with uninsurable aggregate and idiosyncratic risk. A new algorithm for identifying the returns, the optimal private allocations, and the population transport in the state of equilibrium is developed and is tested in two well known benchmark studies.
    
[^9]: 两边市场中的质量选择问题: 以受限的价格歧视方法为例

    Quality Selection in Two-Sided Markets: A Constrained Price Discrimination Approach. (arXiv:1912.02251v7 [econ.TH] UPDATED)

    [http://arxiv.org/abs/1912.02251](http://arxiv.org/abs/1912.02251)

    本论文研究了两边市场中的信息披露问题，探讨了平台应该允许哪些卖家参与以及与买家分享多少关于卖家质量的信息。研究结果表明，在特定条件下，简单的信息结构可以最大化平台的收入。

    

    在线平台收集有关参与者的丰富信息，然后将其中一些信息与参与者共享，以改善市场结果。在本文中，我们研究了两边市场中的信息披露问题：如果一个平台想最大化收入，应该允许哪些卖家参与平台，并且平台应该与买家分享多少关于参与卖家质量的可用信息？我们将这个信息披露问题研究应用到两种不同的两边市场模型中：一个模型中平台选择价格，卖家选择数量（类似于网约车），另一个模型中卖家选择价格（类似于电商）。我们的主要结果提供了一些条件，当普遍观察到简单的信息结构，比如禁止某些卖家参与平台但不区分参与卖家时，平台的收入最大化。平台的信息披露问题是一个关键且具有实际意义的问题。

    Online platforms collect rich information about participants and then share some of this information back with them to improve market outcomes. In this paper we study the following information disclosure problem in two-sided markets: If a platform wants to maximize revenue, which sellers should the platform allow to participate, and how much of its available information about participating sellers' quality should the platform share with buyers? We study this information disclosure problem in the context of two distinct two-sided market models: one in which the platform chooses prices and the sellers choose quantities (similar to ride-sharing), and one in which the sellers choose prices (similar to e-commerce). Our main results provide conditions under which simple information structures commonly observed in practice, such as banning certain sellers from the platform while not distinguishing between participating sellers, maximize the platform's revenue. The platform's information discl
    
[^10]: 识别一个方案提供的效果：以Head Start为例

    Identifying the Effects of a Program Offer with an Application to Head Start. (arXiv:1711.02048v6 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1711.02048](http://arxiv.org/abs/1711.02048)

    该论文提出了一种治疗选择模型，通过分析Head Start学前计划的数据，发现方案提供对于参与者有积极效果，特别是那些没有其他学前教育选择的孩子。通过成本效益分析，发现与考试成绩提高相关的收入效益超过了方案提供的净成本。

    

    我提出了一个治疗选择模型，引入了选择集和偏好中的未观测异质性，以评估方案提供的平均效果。我展示了如何利用模型结构来定义捕捉这些效果的参数，然后在选择集的工具变异下计算其识别集。我通过使用Head Start Impact Study的数据分析提供Head Start学前计划的效果来说明这些工具。我发现这样的政策影响了许多接受方案的孩子，并且随后对考试成绩产生了积极的影响。这些效果来自那些没有学前教育作为选择的选项的孩子。成本效益分析显示，与考试成绩增长相关的收入效益可以很大，超过了接受方案所带来的净成本。

    I propose a treatment selection model that introduces unobserved heterogeneity in both choice sets and preferences to evaluate the average effects of a program offer. I show how to exploit the model structure to define parameters capturing these effects and then computationally characterize their identified sets under instrumental variable variation in choice sets. I illustrate these tools by analyzing the effects of providing an offer to the Head Start preschool program using data from the Head Start Impact Study. I find that such a policy affects a large number of children who take up the offer, and that they subsequently have positive effects on test scores. These effects arise from children who do not have any preschool as an outside option. A cost-benefit analysis reveals that the earning benefits associated with the test score gains can be large and outweigh the net costs associated with offer take up.
    

