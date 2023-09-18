# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision.](http://arxiv.org/abs/2309.08431) | 本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。 |
| [^2] | [On Sparse Grid Interpolation for American Option Pricing with Multiple Underlying Assets.](http://arxiv.org/abs/2309.08287) | 本文提出了一种基于稀疏网格插值的方法，用于定价包含多种标的资产的美式期权。通过动态规划和静态稀疏网格插值技术，我们能够高效地计算美式期权的继续价值函数，并通过减少插值点的数量实现计算效率的提高。数值实验结果表明该方法在定价美式算术和几何篮子看跌期权方面表现出色。 |
| [^3] | [A Markovian empirical model for the VIX index and the pricing of the corresponding derivatives.](http://arxiv.org/abs/2309.08175) | 本文提出了一种VIX指数的经验模型，发现VIX具有长期经验分布，利用马尔可夫过程和适当函数$h$进行动态建模，并使用分离变量法解决了VIX期货和认购期权的定价问题。 |
| [^4] | [Sources of capital growth.](http://arxiv.org/abs/2309.03403) | 资本增长和加速不依赖于净储蓄或消费的限制，对经济教育和公共政策有重要影响。 |
| [^5] | [Sustainability assessment of Low Earth Orbit (LEO) satellite broadband mega-constellations.](http://arxiv.org/abs/2309.02338) | 本研究对低地球轨道（LEO）卫星宽带星座进行了可持续性评估，发现卫星发射增加带来了环境排放和碳足迹的问题，需要对其进行有效管理。 |
| [^6] | [Large-Scale Education Reform in General Equilibrium: Regression Discontinuity Evidence from India: Comment.](http://arxiv.org/abs/2303.11956) | 本文重新分析了Khanna (2023)的研究，指出缺失数据及绘图软件等因素对结果造成了干扰，表明他们之前的结论不能被有效支持。 |
| [^7] | [The Impact of the #MeToo Movement on Language at Court -- A text-based causal inference approach.](http://arxiv.org/abs/2209.00409) | 本研究通过一种基于文本的因果推断方法，评估了#MeToo运动对美国法庭语言的影响，结果发现该运动对司法意见中关于性暴力案件的语言产生了显著影响。 |
| [^8] | [On the skew and curvature of implied and local volatilities.](http://arxiv.org/abs/2205.11185) | 本文研究了本地波动率和隐含波动率的短期关系，并提出了关于斜率和曲率之间的规则。 |

# 详细

[^1]: 去中心化金融与自动化市场做市：可预测的损失和最优流动性提供

    Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision. (arXiv:2309.08431v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.08431](http://arxiv.org/abs/2309.08431)

    本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    

    在这篇论文中，我们对动态调整其在集中流动性池中提供流动性范围的战略性流动性提供者的连续时间财富动态进行了表征。他们的财富来自手续费收入和他们在池中持有的资产的价值。接下来，我们推导出了一种自融资和封闭形式的最优流动性提供策略，其中流动性提供者的范围宽度由池的盈利能力（提供费用减去燃气费用）、可预测损失（持仓的损失）和集中风险决定。集中风险是指如果池中的边际兑换率（类似于限价订单簿中的中间价）超出流动性提供者的范围，费用收入会下降。当边际兑换率由随机漂移驱动时，我们展示了如何通过最优调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    Constant product markets with concentrated liquidity (CL) are the most popular type of automated market makers. In this paper, we characterise the continuous-time wealth dynamics of strategic LPs who dynamically adjust their range of liquidity provision in CL pools. Their wealth results from fee income and the value of their holdings in the pool. Next, we derive a self-financing and closed-form optimal liquidity provision strategy where the width of the LP's liquidity range is determined by the profitability of the pool (provision fees minus gas fees), the predictable losses (PL) of the LP's position, and concentration risk. Concentration risk refers to the decrease in fee revenue if the marginal exchange rate (akin to the midprice in a limit order book) in the pool exits the LP's range of liquidity. When the marginal rate is driven by a stochastic drift, we show how to optimally skew the range of liquidity to increase fee revenue and profit from the expected changes in the marginal ra
    
[^2]: 关于稀疏网格插值在多标的资产美式期权定价中的应用

    On Sparse Grid Interpolation for American Option Pricing with Multiple Underlying Assets. (arXiv:2309.08287v1 [math.NA])

    [http://arxiv.org/abs/2309.08287](http://arxiv.org/abs/2309.08287)

    本文提出了一种基于稀疏网格插值的方法，用于定价包含多种标的资产的美式期权。通过动态规划和静态稀疏网格插值技术，我们能够高效地计算美式期权的继续价值函数，并通过减少插值点的数量实现计算效率的提高。数值实验结果表明该方法在定价美式算术和几何篮子看跌期权方面表现出色。

    

    本文提出了一种基于高效积分和稀疏网格的多项式插值方法，用于定价包含多种标的资产的美式期权。该方法首先利用动态规划的思想对美式期权进行定价，然后使用静态稀疏网格对每个时间步长的继续价值函数进行插值。为了提高效率，我们首先通过缩放tanh映射将定义域从$\mathbb{R}^d$转换到$(-1,1)^d$，然后通过一个气泡函数消除在$(-1,1)^d$上的边界奇异性，并同时显著减少插值点的数量。我们严格证明了通过适当选择气泡函数，所得到的函数在一定阶数的混合导数上具有有界性，从而为使用稀疏网格提供了理论基础。数值实验结果表明，该方法在美式算术和几何篮子看跌期权定价中效果显著。

    In this work, we develop a novel efficient quadrature and sparse grid based polynomial interpolation method to price American options with multiple underlying assets. The approach is based on first formulating the pricing of American options using dynamic programming, and then employing static sparse grids to interpolate the continuation value function at each time step. To achieve high efficiency, we first transform the domain from $\mathbb{R}^d$ to $(-1,1)^d$ via a scaled tanh map, and then remove the boundary singularity of the resulting multivariate function over $(-1,1)^d$ by a bubble function and simultaneously, to significantly reduce the number of interpolation points. We rigorously establish that with a proper choice of the bubble function, the resulting function has bounded mixed derivatives up to a certain order, which provides theoretical underpinnings for the use of sparse grids. Numerical experiments for American arithmetic and geometric basket put options with the number
    
[^3]: 《VIX指数的马尔可夫经验模型及相应衍生品的定价》

    A Markovian empirical model for the VIX index and the pricing of the corresponding derivatives. (arXiv:2309.08175v1 [q-fin.PR])

    [http://arxiv.org/abs/2309.08175](http://arxiv.org/abs/2309.08175)

    本文提出了一种VIX指数的经验模型，发现VIX具有长期经验分布，利用马尔可夫过程和适当函数$h$进行动态建模，并使用分离变量法解决了VIX期货和认购期权的定价问题。

    

    本文提出了一种VIX指数的经验模型。我们的研究结果表明VIX指数具有长期的经验分布。为了描述其动态变化，我们采用了一个具有均匀分布作为不变分布且包含适当函数$h$的连续时间马尔可夫过程。我们发现$h$是VIX数据经验分布的反函数。此外，我们利用分离变量法得到了对VIX期货和认购期权定价问题的精确解。

    In this paper, we propose an empirical model for the VIX index. Our findings indicate that the VIX has a long-term empirical distribution. To model its dynamics, we utilize a continuous-time Markov process with a uniform distribution as its invariant distribution and a suitable function $h$. We determined that $h$ is the inverse function of the VIX data's empirical distribution. Additionally, we use the method of variables of separation to get the exact solution to the pricing problem for VIX futures and call options.
    
[^4]: 资本增长的来源

    Sources of capital growth. (arXiv:2309.03403v1 [econ.GN])

    [http://arxiv.org/abs/2309.03403](http://arxiv.org/abs/2309.03403)

    资本增长和加速不依赖于净储蓄或消费的限制，对经济教育和公共政策有重要影响。

    

    根据国民账户数据显示，净储蓄或消费的变化与市值资本增长率的变化（资本加速度）之间没有影响。因此，资本增长和加速似乎不依赖于净储蓄或消费的限制。我们探讨了这种可能性，并讨论了对经济教育和公共政策的影响。

    Data from national accounts show no effect of change in net saving or consumption, in ratio to market-value capital, on change in growth rate of market-value capital (capital acceleration). Thus it appears that capital growth and acceleration arrive without help from net saving or consumption restraint. We explore ways in which this is possible, and discuss implications for economic teaching and public policy
    
[^5]: 低地球轨道（LEO）卫星宽带星座的可持续性评估

    Sustainability assessment of Low Earth Orbit (LEO) satellite broadband mega-constellations. (arXiv:2309.02338v1 [astro-ph.EP])

    [http://arxiv.org/abs/2309.02338](http://arxiv.org/abs/2309.02338)

    本研究对低地球轨道（LEO）卫星宽带星座进行了可持续性评估，发现卫星发射增加带来了环境排放和碳足迹的问题，需要对其进行有效管理。

    

    超大型星座的增长迅速增加了将新卫星送入空间所需的火箭发射次数。虽然低地球轨道（LEO）宽带卫星有助于连接未连通的社区并实现可持续发展目标，但也存在一系列负面环境外部性，包括火箭燃料燃烧和由此产生的环境排放。我们对三个主要LEO星座的第一阶段进行可持续性分析，包括Amazon Kuiper（3,236颗卫星）、OneWeb（648颗卫星）和SpaceX Starlink（4,425颗卫星）。在基准方案下，经过五年，我们发现Kuiper的每位用户二氧化碳当量（CO$_2$eq）为0.70±0.34吨，OneWeb为1.41±0.71吨，Starlink为0.47±0.15吨CO$_2$eq/用户。然而，在最坏情况下的排放情景中，这些值增加到Kuiper的3.02±1.48吨，OneWeb的1.7±0.71吨和Starlink的1.04±0.33吨CO$_2$eq/用户。

    The growth of mega-constellations is rapidly increasing the number of rocket launches required to place new satellites in space. While Low Earth Orbit (LEO) broadband satellites help to connect unconnected communities and achieve the Sustainable Development Goals, there are also a range of negative environmental externalities, from the burning of rocket fuels and resulting environmental emissions. We present sustainability analytics for phase 1 of the three main LEO constellations including Amazon Kuiper (3,236 satellites), OneWeb (648 satellites), and SpaceX Starlink (4,425 satellites). In baseline scenarios over five years, we find a per subscriber carbon dioxide equivalent (CO$_2$eq) of 0.70$\pm$0.34 tonnes for Kuiper, 1.41$\pm$0.71 tonnes for OneWeb and 0.47$\pm$0.15 tonnes CO$_2$eq/subscriber for Starlink. However, in the worst-case emissions scenario these values increase to 3.02$\pm$1.48 tonnes for Kuiper, 1.7$\pm$0.71 tonnes for OneWeb and 1.04$\pm$0.33 tonnes CO$_2$eq/subscrib
    
[^6]: 基于一般均衡的大规模教育改革：印度回归不连续证据的评论

    Large-Scale Education Reform in General Equilibrium: Regression Discontinuity Evidence from India: Comment. (arXiv:2303.11956v1 [econ.GN])

    [http://arxiv.org/abs/2303.11956](http://arxiv.org/abs/2303.11956)

    本文重新分析了Khanna (2023)的研究，指出缺失数据及绘图软件等因素对结果造成了干扰，表明他们之前的结论不能被有效支持。

    

    本文重新分析了 Khanna (2023) 中通过回归不连续设计研究印度教育对劳动力市场的影响的内容。在图形初步分析中，反转绘图软件默认值的覆盖极大地减少了不连续性的出现。在数据中缺少离不连续点四个街区；修复后削减了对学校和对数工资的简化形式影响分别为62％和75％。使用一致的方差估计器，并将其聚类处理到地理治疗单元，进一步削弱了积极影响的推断。一般均衡效应和替代弹性的估计不是无偏的，且有效方差为无限大。

    This paper reanalyzes Khanna (2023), which studies labor market effects of schooling in India through a regression discontinuity design. In graphical preliminaries, reversing overrides of the plotting software's defaults greatly reduces the appearance of discontinuities. Absent from the data are four districts close to the discontinuity; restoring them cuts the reduced-form impacts on schooling and log wages by 62% and 75%. Using a consistent variance estimator, and clustering it at the geographic unit of treatment, further weakens the inference of positive impact. The estimates of general equilibrium effects and elasticities of substitution are not unbiased and have effectively infinite variance.
    
[^7]: #MeToo运动对法庭语言的影响--一种基于文本因果推断方法的研究

    The Impact of the #MeToo Movement on Language at Court -- A text-based causal inference approach. (arXiv:2209.00409v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.00409](http://arxiv.org/abs/2209.00409)

    本研究通过一种基于文本的因果推断方法，评估了#MeToo运动对美国法庭语言的影响，结果发现该运动对司法意见中关于性暴力案件的语言产生了显著影响。

    

    本研究评估了#MeToo运动对51个美国州和联邦上诉法院关于性暴力相关案件的司法意见中使用的语言的影响。该研究引入了各种指标来量化法庭中的参与者使用的语言，这种语言暗示性地将责任转嫁给受害者。其中一个指标衡量了作为语法主语提及受害者的频率，因为心理学领域的研究表明，受害者被作为语法主语提及的次数越多，受到的责怪就越多。另外两个衡量受害者指责程度的指数捕捉了句子中涉及受害者和/或施害者的情感和上下文。此外，司法意见被转化为词袋和tf-idf向量，以便研究语言随时间的演变。通过D因果效应估算了#MeToo运动的因果影响。

    This study assesses the effect of the #MeToo movement on the language used in judicial opinions on sexual violence related cases from 51 U.S. state and federal appellate courts. The study introduces various indicators to quantify the extent to which actors in courtrooms employ language that implicitly shifts responsibility away from the perpetrator and onto the victim. One indicator measures how frequently the victim is mentioned as the grammatical subject, as research in the field of psychology suggests that victims are assigned more blame the more often they are referred to as the grammatical subject. The other two indices designed to gauge the level of victim-blaming capture the sentiment of and the context in sentences referencing the victim and/or perpetrator. Additionally, judicial opinions are transformed into bag-of-words and tf-idf vectors to facilitate the examination of the evolution of language over time. The causal effect of the #MeToo movement is estimated by means of a D
    
[^8]: 关于暗含波动率和本地波动率的偏斜和曲率

    On the skew and curvature of implied and local volatilities. (arXiv:2205.11185v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2205.11185](http://arxiv.org/abs/2205.11185)

    本文研究了本地波动率和隐含波动率的短期关系，并提出了关于斜率和曲率之间的规则。

    

    本文研究了本地波动率曲面和隐含波动率曲面的短期关系。基于Malliavin微积分技术，我们的结果得到了最近关于粗糙波动率的$\frac{1}{H+3/2}$规则（其中$H$表示波动率过程的Hurst参数），该规则表明无风险价格的隐含波动率的短期斜率是相应本地波动率斜率的$\frac{1}{H+3/2}$。此外，我们发现隐含波动率的无风险价格的短端曲率可以用本地波动率的短端偏斜和曲率表示，反之亦然，并且这种关系取决于$H$。

    In this paper, we study the relationship between the short-end of the local and the implied volatility surfaces. Our results, based on Malliavin calculus techniques, recover the recent $\frac{1}{H+3/2}$ rule (where $H$ denotes the Hurst parameter of the volatility process) for rough volatilitites (see Bourgey, De Marco, Friz, and Pigato (2022)), that states that the short-time skew slope of the at-the-money implied volatility is $\frac{1}{H+3/2}$ the corresponding slope for local volatilities. Moreover, we see that the at-the-money short-end curvature of the implied volatility can be written in terms of the short-end skew and curvature of the local volatility and viceversa, and that this relationship depends on $H$.
    

