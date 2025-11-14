# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unwinding Stochastic Order Flow: When to Warehouse Trades.](http://arxiv.org/abs/2310.14144) | 本论文研究了如何以最小的交易成本解构随机订单流程，模拟并解决了这个问题，并详细研究了订单流入特征对最优策略和核心交易指标的影响。 |
| [^2] | [Unbiased estimators for the Heston model with stochastic interest rates.](http://arxiv.org/abs/2301.12072) | 本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。 |

# 详细

[^1]: 解构随机订单流程：何时存储交易

    Unwinding Stochastic Order Flow: When to Warehouse Trades. (arXiv:2310.14144v1 [q-fin.TR])

    [http://arxiv.org/abs/2310.14144](http://arxiv.org/abs/2310.14144)

    本论文研究了如何以最小的交易成本解构随机订单流程，模拟并解决了这个问题，并详细研究了订单流入特征对最优策略和核心交易指标的影响。

    

    我们研究如何以最小的交易成本解构随机订单流程。随机订单流程在金融机构的中央风险簿（CRB）中出现，CRB是一个集中交易台，用于汇总金融机构内的订单流程。该交易台可以存储流入订单，理想情况下将其与随后的相反订单进行净化（内部化），或将其路由到市场（外部化）并承担与价格影响和买卖价差相关的成本。我们对一般类别的流入过程建模并解决了这个问题，使我们能够详细研究流入特征如何影响最优策略和核心交易指标。我们的模型允许半闭合形式的分析解，并且可以进行数值实现。与已知订单大小的标准执行问题相比，解构策略对预期未来流入有一个附加调整。其符号取决于订单的自相关性；只有真实流（鞅）才会被解构。

    We study how to unwind stochastic order flow with minimal transaction costs. Stochastic order flow arises, e.g., in the central risk book (CRB), a centralized trading desk that aggregates order flows within a financial institution. The desk can warehouse in-flow orders, ideally netting them against subsequent opposite orders (internalization), or route them to the market (externalization) and incur costs related to price impact and bid-ask spread. We model and solve this problem for a general class of in-flow processes, enabling us to study in detail how in-flow characteristics affect optimal strategy and core trading metrics. Our model allows for an analytic solution in semi-closed form and is readily implementable numerically. Compared with a standard execution problem where the order size is known upfront, the unwind strategy exhibits an additive adjustment for projected future in-flows. Its sign depends on the autocorrelation of orders; only truth-telling (martingale) flow is unwou
    
[^2]: 具有随机利率的Heston模型的无偏估计器

    Unbiased estimators for the Heston model with stochastic interest rates. (arXiv:2301.12072v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2301.12072](http://arxiv.org/abs/2301.12072)

    本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。

    

    我们结合了Rhee和Glynn（Operations Research: 63(5), 1026-1043，2015）中的无偏估计器和具有随机利率的Heston模型。具体地，我们首先为具有随机利率的Heston模型开发了一个半精确的对数欧拉方案。然后，在一些温和的假设下，我们证明收敛率在L^2范数中是O(h)，其中h是步长。该结果适用于许多模型，如Heston-Hull-While模型，Heston-CIR模型和Heston-Black-Karasinski模型。数值实验支持我们的理论收敛率。

    We combine the unbiased estimators in Rhee and Glynn (Operations Research: 63(5), 1026-1043, 2015) and the Heston model with stochastic interest rates. Specifically, we first develop a semi-exact log-Euler scheme for the Heston model with stochastic interest rates. Then, under mild assumptions, we show that the convergence rate in the $L^2$ norm is $O(h)$, where $h$ is the step size. The result applies to a large class of models, such as the Heston-Hull-While model, the Heston-CIR model and the Heston-Black-Karasinski model. Numerical experiments support our theoretical convergence rate.
    

