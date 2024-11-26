# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Assignment Game: New Mechanisms for Equitable Core Imputations](https://arxiv.org/abs/2402.11437) | 本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。 |
| [^2] | [Persuading a Learning Agent](https://arxiv.org/abs/2402.09721) | 在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。 |

# 详细

[^1]: 《分配博弈：公平核分配的新机制》

    The Assignment Game: New Mechanisms for Equitable Core Imputations

    [https://arxiv.org/abs/2402.11437](https://arxiv.org/abs/2402.11437)

    本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。

    

    《分配博弈》的核分配集形成一个（非有限）分配格。迄今为止，仅已知有效算法用于计算其两个极端分配；但是，其中每一个都最大程度地偏袒一个方，不利于双方的分配，导致盈利不均衡。另一个问题是，由一个玩家组成的子联盟（或者来自分配两侧的一系玩家）可以获得零利润，因此核分配不必给予他们任何利润。因此，核分配在个体代理人层面上不提供任何公平性保证。这引出一个问题，即如何计算更公平的核分配。在本文中，我们提出了一个计算分配博弈的Leximin和Leximax核分配的组合（即，该机制不涉及LP求解器）多项式时间机制。这些分配以不同方式实现了“公平性”：

    arXiv:2402.11437v1 Announce Type: cross  Abstract: The set of core imputations of the assignment game forms a (non-finite) distributive lattice. So far, efficient algorithms were known for computing only its two extreme imputations; however, each of them maximally favors one side and disfavors the other side of the bipartition, leading to inequitable profit sharing. Another issue is that a sub-coalition consisting of one player (or a set of players from the same side of the bipartition) can make zero profit, therefore a core imputation is not obliged to give them any profit. Hence core imputations make no fairness guarantee at the level of individual agents. This raises the question of computing {\em more equitable core imputations}.   In this paper, we give combinatorial (i.e., the mechanism does not invoke an LP-solver) polynomial time mechanisms for computing the leximin and leximax core imputations for the assignment game. These imputations achieve ``fairness'' in different ways: w
    
[^2]: 说服一位学习代理

    Persuading a Learning Agent

    [https://arxiv.org/abs/2402.09721](https://arxiv.org/abs/2402.09721)

    在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。

    

    我们研究了一个重复的贝叶斯说服问题（更一般地，任何具有完全信息的广义委托-代理问题），其中委托人没有承诺能力，代理人使用算法来学习如何对委托人的信号做出响应。我们将这个问题简化为一个一次性的广义委托-代理问题，代理人近似地最佳响应。通过这个简化，我们可以证明：如果代理人使用上下文无遗憾学习算法，则委托人可以保证其效用与经典无学习模型中具有承诺的委托人的最优效用之间可以无限接近；如果代理人使用上下文无交换遗憾学习算法，则委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。委托人在学习模型与非学习模型中可以获得的效用之间的差距是有界的。

    arXiv:2402.09721v1 Announce Type: cross  Abstract: We study a repeated Bayesian persuasion problem (and more generally, any generalized principal-agent problem with complete information) where the principal does not have commitment power and the agent uses algorithms to learn to respond to the principal's signals. We reduce this problem to a one-shot generalized principal-agent problem with an approximately-best-responding agent. This reduction allows us to show that: if the agent uses contextual no-regret learning algorithms, then the principal can guarantee a utility that is arbitrarily close to the principal's optimal utility in the classic non-learning model with commitment; if the agent uses contextual no-swap-regret learning algorithms, then the principal cannot obtain any utility significantly more than the optimal utility in the non-learning model with commitment. The difference between the principal's obtainable utility in the learning model and the non-learning model is bound
    

