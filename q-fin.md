# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Growth rate of liquidity provider's wealth in G3Ms](https://arxiv.org/abs/2403.18177) | 该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。 |
| [^2] | [A Multilevel Stochastic Approximation Algorithm for Value-at-Risk and Expected Shortfall Estimation.](http://arxiv.org/abs/2304.01207) | 提出了一种多层次随机逼近算法用于计算金融损失的价值风险和预期损失的估计。该方法的性能优于传统的蒙特卡罗方法和其他标准的基于优化的方法。 |

# 详细

[^1]: G3M中做市商财富增长率

    Growth rate of liquidity provider's wealth in G3Ms

    [https://arxiv.org/abs/2403.18177](https://arxiv.org/abs/2403.18177)

    该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。

    

    几何均值市场做市商（G3M），如Uniswap和Balancer，代表一类广泛使用的自动做市商（AMM）。这些G3M的特点在于：每笔交易前后，AMM的储备必须保持相同（加权）的几何均值。本文研究了交易费对G3M中流动性提供者（LP）盈利能力的影响，以及LP面临的由涉及参考市场的套利活动导致的逆向选择。我们的工作扩展了先前研究中描述的G3M模型，将交易费和连续时间套利整合到分析中。在这个背景下，我们分析了具有随机存储过程特征的G3M动态，并计算了LP财富的增长率。特别地，我们的结果与扩展了关于常数乘积市场做市商的结果相一致，通常称为Uniswap v2。

    arXiv:2403.18177v1 Announce Type: new  Abstract: Geometric mean market makers (G3Ms), such as Uniswap and Balancer, represent a widely used class of automated market makers (AMMs). These G3Ms are characterized by the following rule: the reserves of the AMM must maintain the same (weighted) geometric mean before and after each trade. This paper investigates the effects of trading fees on liquidity providers' (LP) profitability in a G3M, as well as the adverse selection faced by LPs due to arbitrage activities involving a reference market. Our work expands the model described in previous studies for G3Ms, integrating transaction fees and continuous-time arbitrage into the analysis. Within this context, we analyze G3M dynamics, characterized by stochastic storage processes, and calculate the growth rate of LP wealth. In particular, our results align with and extend the results concerning the constant product market maker, commonly referred to as Uniswap v2.
    
[^2]: 一种多层次随机逼近算法用于价值风险和期望损失估计

    A Multilevel Stochastic Approximation Algorithm for Value-at-Risk and Expected Shortfall Estimation. (arXiv:2304.01207v1 [q-fin.CP])

    [http://arxiv.org/abs/2304.01207](http://arxiv.org/abs/2304.01207)

    提出了一种多层次随机逼近算法用于计算金融损失的价值风险和预期损失的估计。该方法的性能优于传统的蒙特卡罗方法和其他标准的基于优化的方法。

    

    我们提出了一种多层次随机逼近(MLSA)方案用于计算金融损失的价值风险(VaR)和预期损失(ES)，这只能在未来风险因素的实现条件下通过模拟计算。因此，估计其VaR和ES的问题是嵌套的，可以视为具有偏移创新的随机逼近问题的实例。在这个框架中，对于预定的精度$\epsilon$，标准随机逼近算法的最优复杂度被证明是$\epsilon$ - 3的顺序。为了估计VaR，我们的MLSA算法实现了$\epsilon$ -2- $\delta$的顺序的最优复杂度，其中$\delta$ <1是某个参数，取决于损失的可积度，而为了估计ES，它实现了$\epsilon$ -2 | ln $\epsilon$ | 2的顺序的最优复杂度。误差率和执行时间的联合演变的数值研究表明，我们的MLSA算法优于传统的蒙特卡罗方法和其他标准的基于优化的方法。

    We propose a multilevel stochastic approximation (MLSA) scheme for the computation of the Value-at-Risk (VaR) and the Expected Shortfall (ES) of a financial loss, which can only be computed via simulations conditional on the realization of future risk factors. Thus, the problem of estimating its VaR and ES is nested in nature and can be viewed as an instance of a stochastic approximation problem with biased innovation. In this framework, for a prescribed accuracy $\epsilon$, the optimal complexity of a standard stochastic approximation algorithm is shown to be of order $\epsilon$ --3. To estimate the VaR, our MLSA algorithm attains an optimal complexity of order $\epsilon$ --2--$\delta$ , where $\delta$ < 1 is some parameter depending on the integrability degree of the loss, while to estimate the ES, it achieves an optimal complexity of order $\epsilon$ --2 |ln $\epsilon$| 2. Numerical studies of the joint evolution of the error rate and the execution time demonstrate how our MLSA algo
    

