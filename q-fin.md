# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Appointment Scheduling with Waiting Time Guarantees](https://arxiv.org/abs/2402.12561) | 该研究提出了具有等待时间保证的稳健预约调度问题，旨在同时最小化总成本并提供所有客户等待时间保证，通过引入混合整数线性规划解决了这一NP困难问题，并证明了特殊情况下采用的两种调度规则的最优性。 |
| [^2] | [Hedging Valuation Adjustment for Callable Claims.](http://arxiv.org/abs/2304.02479) | 本文扩展了对冲估价调整的概念到可赎回资产上，用于量化和处理达尔文模型风险。具体地，通过在好模型中模拟坏模型的对冲行为，揭示了可赎回资产的模型风险数学特性，可赎回债券的对冲估价调整可能比所得收益几倍更大。 |

# 详细

[^1]: 具有等待时间保证的稳健预约调度

    Robust Appointment Scheduling with Waiting Time Guarantees

    [https://arxiv.org/abs/2402.12561](https://arxiv.org/abs/2402.12561)

    该研究提出了具有等待时间保证的稳健预约调度问题，旨在同时最小化总成本并提供所有客户等待时间保证，通过引入混合整数线性规划解决了这一NP困难问题，并证明了特殊情况下采用的两种调度规则的最优性。

    

    在不确定性条件下进行的预约调度问题面临着成本最小化和客户等待时间之间的基本权衡。大多数现有研究采用加权和方法来解决这一权衡，这种方法很少强调个体等待时间，因此对客户满意度关注较少。相反，我们研究了如何在为所有客户提供等待时间保证的同时最小化总成本。鉴于服务时间和客户未到的盒状不确定性集，我们引入了具有等待时间保证的稳健预约调度问题。我们证明了该问题在一般情况下是NP困难的，并引入了一个可以在合理计算时间内解决的混合整数线性规划。对于特殊情况，我们证明了众所周知的最小方差优先排序规则和贝利-韦尔奇调度规则的多项式时间变体是最优的。此外，我们进行了一个案例研究，使用了一个大型大学放射学部门的数据

    arXiv:2402.12561v1 Announce Type: new  Abstract: Appointment scheduling problems under uncertainty encounter a fundamental trade-off between cost minimization and customer waiting times. Most existing studies address this trade-off using a weighted sum approach, which puts little emphasis on individual waiting times and, thus, customer satisfaction. In contrast, we study how to minimize total cost while providing waiting time guarantees to all customers. Given box uncertainty sets for service times and no-shows, we introduce the Robust Appointment Scheduling Problem with Waiting Time Guarantees. We show that the problem is NP-hard in general and introduce a mixed-integer linear program that can be solved in reasonable computation time. For special cases, we prove that polynomial-time variants of the well-known Smallest-Variance-First sequencing rule and the Bailey-Welch scheduling rule are optimal. Furthermore, a case study with data from the radiology department of a large university 
    
[^2]: 可赎回债券的对冲估价调整

    Hedging Valuation Adjustment for Callable Claims. (arXiv:2304.02479v1 [q-fin.RM])

    [http://arxiv.org/abs/2304.02479](http://arxiv.org/abs/2304.02479)

    本文扩展了对冲估价调整的概念到可赎回资产上，用于量化和处理达尔文模型风险。具体地，通过在好模型中模拟坏模型的对冲行为，揭示了可赎回资产的模型风险数学特性，可赎回债券的对冲估价调整可能比所得收益几倍更大。

    

    达尔文模型风险是交易者寄生于中期系统利润的错价和对冲风险，它们仅仅是长期亏损的补偿因素，因为在极端情况下，交易者的错误模型不再与市场相吻合。这种达尔文模型风险的Alpha泄漏无法被市场风险工具（如风险价值、预期缺口或强调风险价值）所检测。达尔文模型风险只能通过模拟坏模型在好模型中的对冲行为来观察。本文将对冲估价调整的概念扩展到可赎回资产上，用于量化和处理这种风险。以一个具有样式的可赎回范围计价为例，说明了可赎回资产的达尔文模型风险数学性质。在考虑错误的对冲和行权决策时，对冲估价调整的大小可能比所得收益几倍更大。

    Darwinian model risk is the risk of mis-price-and-hedge biased toward short-to-medium systematic profits of a trader, which are only the compensator of long term losses becoming apparent under extreme scenarios where the bad model of the trader no longer calibrates to the market. The alpha leakages that characterize Darwinian model risk are undetectable by the usual market risk tools such as value-at-risk, expected shortfall, or stressed value-at-risk.Darwinian model risk can only be seen by simulating the hedging behavior of a bad model within a good model. In this paper we extend to callable assets the notion of hedging valuation adjustment introduced in previous work for quantifying and handling such risk. The mathematics of Darwinian model risk for callable assets are illustrated by exact numerics on a stylized callable range accrual example. Accounting for the wrong hedges and exercise decisions, the magnitude of the hedging valuation adjustment can be several times larger than th
    

