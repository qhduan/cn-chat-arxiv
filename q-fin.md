# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Market Making of Options via Reinforcement Learning.](http://arxiv.org/abs/2307.01814) | 本研究提出了一种通过结合随机策略和强化学习技术的方法来进行期权市场做市。当市场订单的到达与价差呈线性反比时，最优策略为正态分布。 |
| [^2] | [Ledoit-Wolf linear shrinkage with unknown mean.](http://arxiv.org/abs/2304.07045) | 本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。 |

# 详细

[^1]: 通过强化学习进行期权市场做市

    Market Making of Options via Reinforcement Learning. (arXiv:2307.01814v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.01814](http://arxiv.org/abs/2307.01814)

    本研究提出了一种通过结合随机策略和强化学习技术的方法来进行期权市场做市。当市场订单的到达与价差呈线性反比时，最优策略为正态分布。

    

    由于其高维度特性，期权市场做市对于不同到期日和行权价格的期权来说是一个具有挑战性的问题。在本文中，我们提出了一种新颖的方法，将随机策略和受强化学习启发的技术相结合，确定期权市场做市商为交易不同到期日和行权价格的期权时所发布的买卖价差的最优策略。当市场订单的到达与价差呈线性反比时，最优策略为正态分布。

    Market making of options with different maturities and strikes is a challenging problem due to its high dimensional nature. In this paper, we propose a novel approach that combines a stochastic policy and reinforcement learning-inspired techniques to determine the optimal policy for posting bid-ask spreads for an options market maker who trades options with different maturities and strikes. When the arrival of market orders is linearly inverse to the spreads, the optimal policy is normally distributed.
    
[^2]: Ledoit-Wolf线性收缩方法在未知均值的情况下的应用(arXiv:2304.07045v1 [math.ST])

    Ledoit-Wolf linear shrinkage with unknown mean. (arXiv:2304.07045v1 [math.ST])

    [http://arxiv.org/abs/2304.07045](http://arxiv.org/abs/2304.07045)

    本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。

    

    本研究探讨了在未知均值下的大维协方差矩阵估计问题。当维数和样本数成比例并趋向于无穷大时，经验协方差估计器失效，此时称为Kolmogorov渐进性。当均值已知时，Ledoit和Wolf（2004）提出了一个线性收缩估计器，并证明了在这些演进下的收敛性。据我们所知，当均值未知时，尚未提出正式证明。为了解决这个问题，我们提出了一个新的估计器，并在Ledoit和Wolf的假设下证明了它的二次收敛性。最后，我们通过实验证明它胜过了其他标准估计器。

    This work addresses large dimensional covariance matrix estimation with unknown mean. The empirical covariance estimator fails when dimension and number of samples are proportional and tend to infinity, settings known as Kolmogorov asymptotics. When the mean is known, Ledoit and Wolf (2004) proposed a linear shrinkage estimator and proved its convergence under those asymptotics. To the best of our knowledge, no formal proof has been proposed when the mean is unknown. To address this issue, we propose a new estimator and prove its quadratic convergence under the Ledoit and Wolf assumptions. Finally, we show empirically that it outperforms other standard estimators.
    

