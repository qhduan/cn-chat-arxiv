# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximating the set of Nash equilibria for convex games.](http://arxiv.org/abs/2310.04176) | 本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。 |
| [^2] | [Modeling Large Spot Price Deviations in Electricity Markets.](http://arxiv.org/abs/2306.07731) | 本文研究了电力市场中的电价波动模型，并发现在非危机时期四因子模型比三因子模型更为适用。 |
| [^3] | [Distributional dynamic risk measures in Markov decision processes.](http://arxiv.org/abs/2203.09612) | 本文提出了分布动态风险测度的概念，并将其用于研究马尔可夫决策过程。这些动态风险测度允许风险规避在过程中动态变化，我们还推导出了动态规划原理和最优策略的存在性。另外，我们还提供了确定性动作最优性的充分条件。示例研究包括限价定单簿和自动驾驶。 |

# 详细

[^1]: 近似计算凸博弈中纳什均衡解集合

    Approximating the set of Nash equilibria for convex games. (arXiv:2310.04176v1 [math.OC])

    [http://arxiv.org/abs/2310.04176](http://arxiv.org/abs/2310.04176)

    本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。

    

    在Feinstein和Rudloff（2023）中，他们证明了对于任意非合作$N$人博弈，纳什均衡解集合与具有非凸顺序锥的某个向量优化问题的帕累托最优点集合是一致的。为了避免处理非凸顺序锥，我们证明了将纳什均衡解集合等价地表示为$N$个多目标问题（即具有自然顺序锥）的帕累托最优点的交集。目前，计算多目标问题的精确帕累托最优点集合的算法仅适用于线性问题的类别，这将导致这些算法只能用于解线性博弈的真实纳什均衡集合的可能性降低。本文中，我们将考虑更大类别的凸博弈。由于通常只能为凸向量优化问题计算近似解，我们首先展示了类似于上述结果的结果，即$\epsilon$-近似纳什均衡解集合与问题完全相似。

    In Feinstein and Rudloff (2023), it was shown that the set of Nash equilibria for any non-cooperative $N$ player game coincides with the set of Pareto optimal points of a certain vector optimization problem with non-convex ordering cone. To avoid dealing with a non-convex ordering cone, an equivalent characterization of the set of Nash equilibria as the intersection of the Pareto optimal points of $N$ multi-objective problems (i.e.\ with the natural ordering cone) is proven. So far, algorithms to compute the exact set of Pareto optimal points of a multi-objective problem exist only for the class of linear problems, which reduces the possibility of finding the true set of Nash equilibria by those algorithms to linear games only.  In this paper, we will consider the larger class of convex games. As, typically, only approximate solutions can be computed for convex vector optimization problems, we first show, in total analogy to the result above, that the set of $\epsilon$-approximate Nash
    
[^2]: 电力市场中的大型电价波动建模

    Modeling Large Spot Price Deviations in Electricity Markets. (arXiv:2306.07731v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.07731](http://arxiv.org/abs/2306.07731)

    本文研究了电力市场中的电价波动模型，并发现在非危机时期四因子模型比三因子模型更为适用。

    

    过去两年里，能源市场的不确定性增加，导致电力即期价格出现了大幅波动。本文研究了经典三因子模型与高斯基信号、一个正跳跃信号和一个负跳跃信号的拟合，以及在这个新的市场环境下添加第二个高斯基信号的影响。我们使用基于所谓的Gibbs采样的马尔可夫链蒙特卡罗算法来校准模型。将得到的四因子模型与特定时期的三因子模型进行比较，并使用后验预测检验进行评估。此外，我们还推导出了基于四因子即期价格模型的期货合约价格的闭式解。我们发现，四因子模型在非危机时期的表现优于三因子模型。在危机时期，第二个高斯基信号并没有导致更好的拟合效果。

    Increased insecurities on the energy markets have caused massive fluctuations of the electricity spot price within the past two years. In this work, we investigate the fit of a classical 3-factor model with a Gaussian base signal as well as one positive and one negative jump signal in this new market environment. We also study the influence of adding a second Gaussian base signal to the model. For the calibration of our model we use a Markov Chain Monte Carlo algorithm based on the so-called Gibbs sampling. The resulting 4-factor model is than compared to the 3-factor model in different time periods of particular interest and evaluated using posterior predictive checking. Additionally, we derive closed-form solutions for the price of futures contracts in our 4-factor spot price model. We find that the 4-factor model outperforms the 3-factor model in times of non-crises. In times of crises, the second Gaussian base signal does not lead to a better the fit of the model. To the best of ou
    
[^3]: 马尔可夫决策过程中的分布动态风险测度

    Distributional dynamic risk measures in Markov decision processes. (arXiv:2203.09612v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2203.09612](http://arxiv.org/abs/2203.09612)

    本文提出了分布动态风险测度的概念，并将其用于研究马尔可夫决策过程。这些动态风险测度允许风险规避在过程中动态变化，我们还推导出了动态规划原理和最优策略的存在性。另外，我们还提供了确定性动作最优性的充分条件。示例研究包括限价定单簿和自动驾驶。

    

    基于不变凸风险测度的概念，我们引入了分布凸风险测度的概念，并将其用于定义分布动态风险测度。然后，我们将这些动态风险测度应用于研究马尔可夫决策过程，包括潜在成本，随机动作和弱连续转移核。此外，所提出的动态风险测度允许风险规避动态变化。在温和的假设下，我们推导出动态规划原理，并证明了有限和无限时间范围内最优策略的存在。此外，我们提供了确定性动作最优性的充分条件。为了说明，我们以限价定单簿和自动驾驶为例总结本文。

    Based on the concept of law-invariant convex risk measures, we introduce the notion of distributional convex risk measures and employ them to define distributional dynamic risk measures. We then apply these dynamic risk measures to investigate Markov decision processes, incorporating latent costs, random actions, and weakly continuous transition kernels. Furthermore, the proposed dynamic risk measures allows risk aversion to change dynamically. Under mild assumptions, we derive a dynamic programming principle and show the existence of an optimal policy in both finite and infinite time horizons. Moreover, we provide a sufficient condition for the optimality of deterministic actions. For illustration, we conclude the paper with examples from optimal liquidation with limit order books and autonomous driving.
    

