# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exact simulation scheme for the Ornstein-Uhlenbeck driven stochastic volatility model with the Karhunen-Lo\`eve expansions](https://arxiv.org/abs/2402.09243) | 本研究提出了一种新的精确模拟方案，可以更快地模拟Ornstein-Uhlenbeck驱动的随机波动率模型。使用Karhunen-Loève展开来表示波动率路径，并通过解析推导的方式获得了波动率和方差的时间积分。通过采用条件蒙特卡洛方法和保持鞅性的控制变量来进一步改进模拟算法。这一方法比现有方法更快速且更高效。 |
| [^2] | [Optimal Trading in Automatic Market Makers with Deep Learning.](http://arxiv.org/abs/2304.02180) | 本文利用深度学习方法，建立了一个考虑到自动市场制造商和中心化交易所互动的模型，实现了在不使用市场动态近似的前提下，通过优化隐藏订单的交易速率，优化交易策略，该策略可减少价格滑动，跑赢无脑策略。 |

# 详细

[^1]: Ornstein-Uhlenbeck驱动的随机波动率模型的精确模拟方案与Karhunen-Loève展开

    Exact simulation scheme for the Ornstein-Uhlenbeck driven stochastic volatility model with the Karhunen-Lo\`eve expansions

    [https://arxiv.org/abs/2402.09243](https://arxiv.org/abs/2402.09243)

    本研究提出了一种新的精确模拟方案，可以更快地模拟Ornstein-Uhlenbeck驱动的随机波动率模型。使用Karhunen-Loève展开来表示波动率路径，并通过解析推导的方式获得了波动率和方差的时间积分。通过采用条件蒙特卡洛方法和保持鞅性的控制变量来进一步改进模拟算法。这一方法比现有方法更快速且更高效。

    

    本研究提出了一种新的Ornstein-Uhlenbeck驱动的随机波动率模型的精确模拟方案。利用Karhunen-Loève展开，将遵循Ornstein-Uhlenbeck过程的随机波动率路径表示为正弦级数，并将波动率和方差的时间积分解析地推导为独立正态随机变量的和。这种新方法比依赖于计算昂贵的数值变换反演的Li和Wu [Eur. J. Oper. Res., 2019, 275(2), 768-779] 方法快几百倍。进一步采用了条件蒙特卡洛方法和保持鞅性的控制变量对实时价格进行模拟算法改进。

    arXiv:2402.09243v1 Announce Type: new Abstract: This study proposes a new exact simulation scheme of the Ornstein-Uhlenbeck driven stochastic volatility model. With the Karhunen-Lo\`eve expansions, the stochastic volatility path following the Ornstein-Uhlenbeck process is expressed as a sine series, and the time integrals of volatility and variance are analytically derived as the sums of independent normal random variates. The new method is several hundred times faster than Li and Wu [Eur. J. Oper. Res., 2019, 275(2), 768-779] that relies on computationally expensive numerical transform inversion. The simulation algorithm is further improved with the conditional Monte-Carlo method and the martingale-preserving control variate on the spot price.
    
[^2]: 深度学习在自动做市商中的最优交易优化

    Optimal Trading in Automatic Market Makers with Deep Learning. (arXiv:2304.02180v1 [q-fin.TR])

    [http://arxiv.org/abs/2304.02180](http://arxiv.org/abs/2304.02180)

    本文利用深度学习方法，建立了一个考虑到自动市场制造商和中心化交易所互动的模型，实现了在不使用市场动态近似的前提下，通过优化隐藏订单的交易速率，优化交易策略，该策略可减少价格滑动，跑赢无脑策略。

    

    本文探讨了在常数函数市场制造商(CFMMs)和集中式交易所中优化交易策略的方法。我们开发了一个模型，考虑了这两个市场之间的互动，使用条件可激发性的概念估计变量之间的条件依赖关系。此外，我们提出了一个最优执行问题，其中代理通过控制交易速率来隐藏订单。我们这样做时没有近似市场动态。由此产生的动态规划方程不是解析可追踪的，因此我们采用深度Galerkin方法来解决它。最后，我们进行了数字实验，并说明最优策略不会容易受到价格滑移影响，且优于幼稚的策略。

    This article explores the optimisation of trading strategies in Constant Function Market Makers (CFMMs) and centralised exchanges. We develop a model that accounts for the interaction between these two markets, estimating the conditional dependence between variables using the concept of conditional elicitability. Furthermore, we pose an optimal execution problem where the agent hides their orders by controlling the rate at which they trade. We do so without approximating the market dynamics. The resulting dynamic programming equation is not analytically tractable, therefore, we employ the deep Galerkin method to solve it. Finally, we conduct numerical experiments and illustrate that the optimal strategy is not prone to price slippage and outperforms na\"ive strategies.
    

