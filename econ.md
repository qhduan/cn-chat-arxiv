# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Symmetric mechanisms for two-sided matching problems](https://arxiv.org/abs/2404.01404) | 该研究关注对称双边匹配问题，介绍了稳定和坚决机制在匹配过程中的重要性。 |
| [^2] | [Data-driven Policy Learning for a Continuous Treatment](https://arxiv.org/abs/2402.02535) | 本论文研究了在连续治疗条件下的政策学习，通过使用核方法估计政策福利，并引入一种半数据驱动的策略来平衡福利损失的组成部分。 |
| [^3] | [Market Design for Dynamic Pricing and Pooling in Capacitated Networks.](http://arxiv.org/abs/2307.03994) | 本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。 |

# 详细

[^1]: 对称机制用于双边匹配问题

    Symmetric mechanisms for two-sided matching problems

    [https://arxiv.org/abs/2404.01404](https://arxiv.org/abs/2404.01404)

    该研究关注对称双边匹配问题，介绍了稳定和坚决机制在匹配过程中的重要性。

    

    我们专注于基本的一对一双边匹配模型，在这个模型中有两个相等规模的不相交代理人集合，每个代理人在一个集合中对另一个集合中的代理人有偏好，这些偏好通过线性排序进行建模。目标是基于代理人的偏好找到将一个集合中的每个代理人与另一个集合中的一个且仅一个代理人关联的匹配。机制是将一组匹配与每个偏好配置关联的规则。稳定性指的是机制应满足的只选择稳定匹配的能力，而另一个特别适用于应用的关键特性是坚决性，这要求机制总是选择唯一的匹配。延迟接受算法的两个版本是稳定和坚决的机制的示例。然而，这些机制非常不公平，因为它们严重偏向于市场的两侧。在这篇论文中，我们...

    arXiv:2404.01404v1 Announce Type: new  Abstract: We focus on the basic one-to-one two-sided matching model, where there are two disjoint sets of agents of equal size, and each agent in a set has preferences on the agents in the other set, modelled by linear orders. The goal is to find a matching that associates each agent in one set with one and only one agent in the other set based on the agents' preferences. A mechanism is a rule that associates a set of matchings to each preference profile. Stability, which refers to the capability to select only stable matchings, is an important property a mechanism should fulfill. Another crucial property, especially useful for applications, is resoluteness, which requires that the mechanism always selects a unique matching. The two versions of the deferred acceptance algorithm are examples of stable and resolute mechanisms. However, these mechanisms are severely unfair since they strongly favor one of the two sides of the market. In this paper, w
    
[^2]: 基于数据驱动的连续治疗政策学习

    Data-driven Policy Learning for a Continuous Treatment

    [https://arxiv.org/abs/2402.02535](https://arxiv.org/abs/2402.02535)

    本论文研究了在连续治疗条件下的政策学习，通过使用核方法估计政策福利，并引入一种半数据驱动的策略来平衡福利损失的组成部分。

    

    本研究针对连续治疗变量条件下的政策学习进行了研究。我们采用基于核方法的倒数估计权重(IPW)方法来估计政策福利。我们的目标是在由无穷的Vapnik-Chervonenkis(VC)维度特征的全局政策类中近似最优政策。通过使用一系列具有有限VC维度的筛选政策类，实现了这一目标。初步分析表明，福利损失包括三个组成部分：全局福利不足、方差和偏差。这导致同时选择估计的最优带宽和福利近似的最优政策类成为必要。为了应对这一挑战，我们引入了一种半数据驱动的策略，采用了惩罚技术。这种方法产生了奥拉克不等式，能够在不事先了解福利不足的情况下，灵活平衡福利损失的三个组成部分。

    This paper studies policy learning under the condition of unconfoundedness with a continuous treatment variable. Our research begins by employing kernel-based inverse propensity-weighted (IPW) methods to estimate policy welfare. We aim to approximate the optimal policy within a global policy class characterized by infinite Vapnik-Chervonenkis (VC) dimension. This is achieved through the utilization of a sequence of sieve policy classes, each with finite VC dimension. Preliminary analysis reveals that welfare regret comprises of three components: global welfare deficiency, variance, and bias. This leads to the necessity of simultaneously selecting the optimal bandwidth for estimation and the optimal policy class for welfare approximation. To tackle this challenge, we introduce a semi-data-driven strategy that employs penalization techniques. This approach yields oracle inequalities that adeptly balance the three components of welfare regret without prior knowledge of the welfare deficie
    
[^3]: 动态定价和汇聚网络的市场设计

    Market Design for Dynamic Pricing and Pooling in Capacitated Networks. (arXiv:2307.03994v1 [cs.GT])

    [http://arxiv.org/abs/2307.03994](http://arxiv.org/abs/2307.03994)

    本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。

    

    我们研究了一种市场机制，通过设置边缘价格来激励战略性代理商组织旅行，以有效共享有限的网络容量。该市场允许代理商组成团队共享旅行，做出出发时间和路线选择的决策，并支付边缘价格和其他成本。我们发展了一种新的方法来分析市场均衡的存在和计算，建立在组合拍卖和动态网络流理论的基础上。我们的方法解决了市场均衡特征化中的挑战，包括：（a）共享有限边缘容量中旅行的动态流量所引发的整数和网络约束；（b）战略性代理商的异质和私人偏好。我们提供了关于网络拓扑和代理商偏好的充分条件，以确保市场均衡的存在和多项式时间计算。我们确定了一个特定的市场均衡，实现了所有代理商的最大效用，并且与经典的最大流最小割问题等价。

    We study a market mechanism that sets edge prices to incentivize strategic agents to organize trips that efficiently share limited network capacity. This market allows agents to form groups to share trips, make decisions on departure times and route choices, and make payments to cover edge prices and other costs. We develop a new approach to analyze the existence and computation of market equilibrium, building on theories of combinatorial auctions and dynamic network flows. Our approach tackles the challenges in market equilibrium characterization arising from: (a) integer and network constraints on the dynamic flow of trips in sharing limited edge capacity; (b) heterogeneous and private preferences of strategic agents. We provide sufficient conditions on the network topology and agents' preferences that ensure the existence and polynomial-time computation of market equilibrium. We identify a particular market equilibrium that achieves maximum utilities for all agents, and is equivalen
    

