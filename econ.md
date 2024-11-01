# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasi-randomization tests for network interference](https://arxiv.org/abs/2403.16673) | 构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。 |
| [^2] | [Continuous-Time Best-Response and Related Dynamics in Tullock Contests with Convex Costs](https://arxiv.org/abs/2402.08541) | 本研究证明了在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点，并提供了计算近似均衡的算法。同时，我们还证明了相关离散时间动态的收敛性，这表明均衡是这些游戏中代理人行为的可靠预测器。 |
| [^3] | [Game Connectivity and Adaptive Dynamics.](http://arxiv.org/abs/2309.10609) | 通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。 |

# 详细

[^1]: 网络干扰的准随机化检验

    Quasi-randomization tests for network interference

    [https://arxiv.org/abs/2403.16673](https://arxiv.org/abs/2403.16673)

    构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。

    

    许多经典的推理方法在人口单位之间存在干扰时失效。这意味着一个单位的处理状态会影响人口中其他单位的潜在结果。在这种情况下测试这种影响的零假设会使零假设非尖锐。解决这种设置中零假设非尖锐性的一个有趣方法是构建条件随机化检验，使得零假设在受限人口上是尖锐的。在随机实验中，条件随机化检验具有有限样本有效性。这种方法可能会带来计算挑战，因为根据实验设计找到这些适当的子人口可能涉及解决一个NP难的问题。在这篇论文中，我们将人口之间的网络视为一个随机变量而不是固定的。我们提出了一种建立条件准随机化检验的新方法。我们的主要思想是

    arXiv:2403.16673v1 Announce Type: cross  Abstract: Many classical inferential approaches fail to hold when interference exists among the population units. This amounts to the treatment status of one unit affecting the potential outcome of other units in the population. Testing for such spillover effects in this setting makes the null hypothesis non-sharp. An interesting approach to tackling the non-sharp nature of the null hypothesis in this setup is constructing conditional randomization tests such that the null is sharp on the restricted population. In randomized experiments, conditional randomized tests hold finite sample validity. Such approaches can pose computational challenges as finding these appropriate sub-populations based on experimental design can involve solving an NP-hard problem. In this paper, we view the network amongst the population as a random variable instead of being fixed. We propose a new approach that builds a conditional quasi-randomization test. Our main ide
    
[^2]: Tullock竞赛中基于连续时间最优响应和相关动态的凸成本模型

    Continuous-Time Best-Response and Related Dynamics in Tullock Contests with Convex Costs

    [https://arxiv.org/abs/2402.08541](https://arxiv.org/abs/2402.08541)

    本研究证明了在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点，并提供了计算近似均衡的算法。同时，我们还证明了相关离散时间动态的收敛性，这表明均衡是这些游戏中代理人行为的可靠预测器。

    

    Tullock竞赛模型适用于各种现实情景，包括PoW区块链矿工之间的竞争、寻租和游说活动。我们利用李雅普诺夫方式的论证结果表明，在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点。然后，我们利用这一结果提供了一种计算近似均衡的算法。我们还证明了相关离散时间动态的收敛性，例如，当代理人对其他代理人的经验平均行动做出最优响应时。这些结果表明均衡是这些游戏中代理人行为的可靠预测器。

    Tullock contests model real-life scenarios that range from competition among proof-of-work blockchain miners to rent-seeking and lobbying activities. We show that continuous-time best-response dynamics in Tullock contests with convex costs converges to the unique equilibrium using Lyapunov-style arguments. We then use this result to provide an algorithm for computing an approximate equilibrium. We also establish convergence of related discrete-time dynamics, e.g., when the agents best-respond to the empirical average action of other agents. These results indicate that the equilibrium is a reliable predictor of the agents' behavior in these games.
    
[^3]: 游戏连通性与自适应动态

    Game Connectivity and Adaptive Dynamics. (arXiv:2309.10609v1 [econ.TH])

    [http://arxiv.org/abs/2309.10609](http://arxiv.org/abs/2309.10609)

    通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。

    

    我们通过分析最佳响应图的连通特性，分析了游戏的典型结构。特别是，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的，这意味着每个非均衡的行动配置都可以通过最佳响应路径到达每个纯纳什均衡。这对于游戏中的动态过程有着重要意义：许多自适应动态，例如带有惯性的最佳响应动态，在连通的游戏中会导致均衡。因此，存在简单的、不耦合的自适应动态，按周期游戏将几乎确定地收敛到具有纯纳什均衡的“大型”通用游戏的情况下。

    We analyse the typical structure of games in terms of the connectivity properties of their best-response graphs. In particular, we show that almost every 'large' generic game that has a pure Nash equilibrium is connected, meaning that every non-equilibrium action profile can reach every pure Nash equilibrium via best-response paths. This has implications for dynamics in games: many adaptive dynamics, such as the best-response dynamic with inertia, lead to equilibrium in connected games. It follows that there are simple, uncoupled, adaptive dynamics for which period-by-period play converges almost surely to a pure Nash equilibrium in almost every 'large' generic game that has one. We build on recent results in probabilistic combinatorics for our characterisation of game connectivity.
    

