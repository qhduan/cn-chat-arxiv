# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision.](http://arxiv.org/abs/2309.08431) | 本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。 |
| [^2] | [Artificial Intelligence and Dual Contract.](http://arxiv.org/abs/2303.12350) | 本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。 |

# 详细

[^1]: 去中心化金融与自动化市场做市：可预测的损失和最优流动性提供

    Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision. (arXiv:2309.08431v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.08431](http://arxiv.org/abs/2309.08431)

    本文研究了集中流动性的常量产品市场，对动态调整流动性的战略性流动性提供者的财富动态进行了描述。通过推导出自融资和封闭形式的最优流动性提供策略，结合盈利能力、预测损失和集中风险，可以通过调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    

    在这篇论文中，我们对动态调整其在集中流动性池中提供流动性范围的战略性流动性提供者的连续时间财富动态进行了表征。他们的财富来自手续费收入和他们在池中持有的资产的价值。接下来，我们推导出了一种自融资和封闭形式的最优流动性提供策略，其中流动性提供者的范围宽度由池的盈利能力（提供费用减去燃气费用）、可预测损失（持仓的损失）和集中风险决定。集中风险是指如果池中的边际兑换率（类似于限价订单簿中的中间价）超出流动性提供者的范围，费用收入会下降。当边际兑换率由随机漂移驱动时，我们展示了如何通过最优调整流动性范围来增加费用收入并从边际率的预期变化中获利。

    Constant product markets with concentrated liquidity (CL) are the most popular type of automated market makers. In this paper, we characterise the continuous-time wealth dynamics of strategic LPs who dynamically adjust their range of liquidity provision in CL pools. Their wealth results from fee income and the value of their holdings in the pool. Next, we derive a self-financing and closed-form optimal liquidity provision strategy where the width of the LP's liquidity range is determined by the profitability of the pool (provision fees minus gas fees), the predictable losses (PL) of the LP's position, and concentration risk. Concentration risk refers to the decrease in fee revenue if the marginal exchange rate (akin to the midprice in a limit order book) in the pool exits the LP's range of liquidity. When the marginal rate is driven by a stochastic drift, we show how to optimally skew the range of liquidity to increase fee revenue and profit from the expected changes in the marginal ra
    
[^2]: 人工智能与双重合同

    Artificial Intelligence and Dual Contract. (arXiv:2303.12350v1 [cs.AI])

    [http://arxiv.org/abs/2303.12350](http://arxiv.org/abs/2303.12350)

    本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。

    

    随着人工智能算法的快速进步，人们希望算法很快就能在各个领域取代人类决策者，例如合同设计。我们通过实验研究了由人工智能（多智能体Q学习）驱动的算法在双重委托-代理问题的经典“双重合同”模型中的行为。我们发现，这些AI算法可以自主学习设计合适的激励相容合同，而无需外部引导或者它们之间的通信。我们强调，由不同AI算法支持的委托人可以采用混合和零和博弈行为。我们还发现，更具智能的委托人往往会变得合作，而智能较低的委托人则会出现内生性近视并倾向于竞争。在最优合同下，代理的较低合同激励由委托人之间的勾结策略维持。

    With the dramatic progress of artificial intelligence algorithms in recent times, it is hoped that algorithms will soon supplant human decision-makers in various fields, such as contract design. We analyze the possible consequences by experimentally studying the behavior of algorithms powered by Artificial Intelligence (Multi-agent Q-learning) in a workhorse \emph{dual contract} model for dual-principal-agent problems. We find that the AI algorithms autonomously learn to design incentive-compatible contracts without external guidance or communication among themselves. We emphasize that the principal, powered by distinct AI algorithms, can play mixed-sum behavior such as collusion and competition. We find that the more intelligent principals tend to become cooperative, and the less intelligent principals are endogenizing myopia and tend to become competitive. Under the optimal contract, the lower contract incentive to the agent is sustained by collusive strategies between the principals
    

