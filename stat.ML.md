# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits](https://arxiv.org/abs/2402.15171) | 提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。 |
| [^2] | [Convergence of Dirichlet Forms for MCMC Optimal Scaling with Dependent Target Distributions on Large Graphs.](http://arxiv.org/abs/2210.17042) | 本文利用Dirichlet形式的Mosco收敛性分析了在大图上的随机游走Metropolis（RWM）算法，证明了RWM算法的最优比例缩放具有收敛性，将已知的几个结果推广到了大图上的依赖目标分布的情况，并为大图上的MCMC算法开辟了许多新的可能性。 |
| [^3] | [Time-Varying Parameters as Ridge Regressions.](http://arxiv.org/abs/2009.00401) | 该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。 |

# 详细

[^1]: 用于随机组合半臂老虎机的协方差自适应最小二乘算法

    Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2402.15171](https://arxiv.org/abs/2402.15171)

    提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。

    

    我们解决了随机组合半臂老虎机问题，其中玩家可以从包含d个基本项的P个子集中进行选择。大多数现有算法（如CUCB、ESCB、OLS-UCB）需要对奖励分布有先验知识，比如子高斯代理-方差的上界，这很难准确估计。在这项工作中，我们设计了OLS-UCB的方差自适应版本，依赖于协方差结构的在线估计。在实际设置中，估计协方差矩阵的系数要容易得多，并且相对于基于代理方差的算法，导致改进的遗憾上界。当协方差系数全为非负时，我们展示了我们的方法有效地利用了半臂反馈，并且可以明显优于老虎机反馈方法，在指数级别P≫d以及P≤d的情况下，这一点并不来自大多数现有分析。

    arXiv:2402.15171v1 Announce Type: new  Abstract: We address the problem of stochastic combinatorial semi-bandits, where a player can select from P subsets of a set containing d base items. Most existing algorithms (e.g. CUCB, ESCB, OLS-UCB) require prior knowledge on the reward distribution, like an upper bound on a sub-Gaussian proxy-variance, which is hard to estimate tightly. In this work, we design a variance-adaptive version of OLS-UCB, relying on an online estimation of the covariance structure. Estimating the coefficients of a covariance matrix is much more manageable in practical settings and results in improved regret upper bounds compared to proxy variance-based algorithms. When covariance coefficients are all non-negative, we show that our approach efficiently leverages the semi-bandit feedback and provably outperforms bandit feedback approaches, not only in exponential regimes where P $\gg$ d but also when P $\le$ d, which is not straightforward from most existing analyses.
    
[^2]: 依赖于大图的MCMC最优比例缩放的Dirichlet形式的收敛性

    Convergence of Dirichlet Forms for MCMC Optimal Scaling with Dependent Target Distributions on Large Graphs. (arXiv:2210.17042v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2210.17042](http://arxiv.org/abs/2210.17042)

    本文利用Dirichlet形式的Mosco收敛性分析了在大图上的随机游走Metropolis（RWM）算法，证明了RWM算法的最优比例缩放具有收敛性，将已知的几个结果推广到了大图上的依赖目标分布的情况，并为大图上的MCMC算法开辟了许多新的可能性。

    

    Markov Chain Monte Carlo (MCMC)算法在统计学、物理学、机器学习等方面发挥了重要作用，并且对于一些高维问题，它们是唯一已知的通用和有效的方法。本文利用Dirichlet形式的Mosco收敛性分析了在大图上的随机游走Metropolis（RWM）算法，其目标分布是包括任何满足Markov性质的概率测度的Gibbs测度。Dirichlet形式的抽象且强大的理论使我们能够直接和自然地在无限维空间上工作，我们的Mosco收敛性概念允许与RWM链相关联的Dirichlet形式位于变化的图序列上，其中图的大小可以是无界的，图可以是相关的。我们证明了在强空间依赖性存在的情况下，RWM算法的最优比例缩放具有收敛性。我们的结果将已知的几个结果推广到了大图上的依赖目标分布的情况，并为大图上的MCMC算法开辟了许多新的可能性。

    Markov chain Monte Carlo (MCMC) algorithms have played a significant role in statistics, physics, machine learning and others, and they are the only known general and efficient approach for some high-dimensional problems. The random walk Metropolis (RWM) algorithm as the most classical MCMC algorithm, has had a great influence on the development and practice of science and engineering. The behavior of the RWM algorithm in high-dimensional problems is typically investigated through a weak convergence result of diffusion processes. In this paper, we utilize the Mosco convergence of Dirichlet forms in analyzing the RWM algorithm on large graphs, whose target distribution is the Gibbs measure that includes any probability measure satisfying a Markov property. The abstract and powerful theory of Dirichlet forms allows us to work directly and naturally on the infinite-dimensional space, and our notion of Mosco convergence allows Dirichlet forms associated with the RWM chains to lie on changi
    
[^3]: 使用岭回归法的时变参数模型

    Time-Varying Parameters as Ridge Regressions. (arXiv:2009.00401v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.00401](http://arxiv.org/abs/2009.00401)

    该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。

    

    时变参数模型(TVPs)经常被用于经济学中来捕捉结构性变化。我强调了一个被忽视的事实——这些实际上是岭回归。这使得计算、调整和实现比状态空间范式更容易。在高维情况下，解决等价的双重岭问题的计算非常快,关键的“时间变化量”通常是由交叉验证来调整的。使用两步回归岭回归来处理不断变化的波动性。我考虑了基于稀疏性(算法选择哪些参数变化, 哪些不变)和降低秩约束的扩展(变化与因子模型相关联)。为了展示这种方法的有用性, 我使用它来研究加拿大货币政策的演变, 并使用大规模时变局部投影估计约4600个TVPs, 这一任务完全可以利用这种新方法完成。

    Time-varying parameters (TVPs) models are frequently used in economics to capture structural change. I highlight a rather underutilized fact -- that these are actually ridge regressions. Instantly, this makes computations, tuning, and implementation much easier than in the state-space paradigm. Among other things, solving the equivalent dual ridge problem is computationally very fast even in high dimensions, and the crucial "amount of time variation" is tuned by cross-validation. Evolving volatility is dealt with using a two-step ridge regression. I consider extensions that incorporate sparsity (the algorithm selects which parameters vary and which do not) and reduced-rank restrictions (variation is tied to a factor model). To demonstrate the usefulness of the approach, I use it to study the evolution of monetary policy in Canada using large time-varying local projections. The application requires the estimation of about 4600 TVPs, a task well within the reach of the new method.
    

