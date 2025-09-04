# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?](https://arxiv.org/abs/2212.14511) | 该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。 |
| [^2] | [The Curse of Memory in Stochastic Approximation: Extended Version.](http://arxiv.org/abs/2309.02944) | 本文研究了随机逼近中的记忆诅咒问题，并探讨了在不同情况下的结果。在具有几何遍历马尔可夫扰动的情况下，目标偏差一般非零。此外，当参数估计使用平均法时，估计值收敛到渐近无偏，且具有近似最优的渐近协方差。 |

# 详细

[^1]: 直接潜在模型学习能够解决线性二次高斯控制问题吗？

    Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?

    [https://arxiv.org/abs/2212.14511](https://arxiv.org/abs/2212.14511)

    该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。

    

    我们研究了从潜在高维观测中学习状态表示的任务，目标是控制未知的部分可观察系统。我们采用直接潜在模型学习方法，通过预测与规划直接相关的数量（例如成本）来学习潜在状态空间中的动态模型，而无需重建观测。具体来说，我们专注于一种直观的基于成本驱动的状态表示学习方法，用于解决线性二次高斯（LQG）控制问题，这是最基本的部分可观察控制问题之一。作为我们的主要结果，我们建立了在有限样本下找到近似最优状态表示函数和使用直接学习的潜在模型找到近似最优控制器的保证。据我们所知，尽管以前的相关工作取得了各种经验成功，但在这项工作之前，尚不清楚这种基于成本驱动的潜在模型学习方法是否具有有限样本保证。

    arXiv:2212.14511v2 Announce Type: replace  Abstract: We study the task of learning state representations from potentially high-dimensional observations, with the goal of controlling an unknown partially observable system. We pursue a direct latent model learning approach, where a dynamic model in some latent state space is learned by predicting quantities directly related to planning (e.g., costs) without reconstructing the observations. In particular, we focus on an intuitive cost-driven state representation learning method for solving Linear Quadratic Gaussian (LQG) control, one of the most fundamental partially observable control problems. As our main results, we establish finite-sample guarantees of finding a near-optimal state representation function and a near-optimal controller using the directly learned latent model. To the best of our knowledge, despite various empirical successes, prior to this work it was unclear if such a cost-driven latent model learner enjoys finite-sampl
    
[^2]: 随机逼近中的记忆诅咒：扩展版本

    The Curse of Memory in Stochastic Approximation: Extended Version. (arXiv:2309.02944v1 [math.ST])

    [http://arxiv.org/abs/2309.02944](http://arxiv.org/abs/2309.02944)

    本文研究了随机逼近中的记忆诅咒问题，并探讨了在不同情况下的结果。在具有几何遍历马尔可夫扰动的情况下，目标偏差一般非零。此外，当参数估计使用平均法时，估计值收敛到渐近无偏，且具有近似最优的渐近协方差。

    

    自适应控制的最早的日子以来，随机逼近（SA）的理论和应用在控制系统的社区中得到了快速发展。本文以新的视角重新审视了这个主题，受到最近的结果的启发，该结果证明使用（足够小的）恒定步长α>0的SA具有非凡的性能。如果采用平均法获取最终的参数估计，则估计值在渐近无偏和近似最优的渐近协方差下收敛。这些结果是针对具有独立同分布系数的随机线性SA递归获得的。本文在更常见的几何遍历马尔可夫扰动的情况下获得了非常不同的结论：（i）在非线性SA的情况下，识别出了“目标偏差”，并且一般上不为零。其余的结果是针对线性SA递归建立的：（ii）双变量参数扰动过程在拓扑意义上具有几何遍历性；（iii）偏差的表示具有简单的性质。

    Theory and application of stochastic approximation (SA) has grown within the control systems community since the earliest days of adaptive control. This paper takes a new look at the topic, motivated by recent results establishing remarkable performance of SA with (sufficiently small) constant step-size $\alpha>0$. If averaging is implemented to obtain the final parameter estimate, then the estimates are asymptotically unbiased with nearly optimal asymptotic covariance. These results have been obtained for random linear SA recursions with i.i.d.\ coefficients. This paper obtains very different conclusions in the more common case of geometrically ergodic Markovian disturbance: (i) The \textit{target bias} is identified, even in the case of non-linear SA, and is in general non-zero. The remaining results are established for linear SA recursions: (ii) the bivariate parameter-disturbance process is geometrically ergodic in a topological sense; (iii) the representation for bias has a simple
    

