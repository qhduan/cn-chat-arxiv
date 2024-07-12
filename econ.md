# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Characterizing Random Serial Dictatorship.](http://arxiv.org/abs/2303.11976) | 本文通过使用组合最佳优先算法和二次规划，扩展了随机串行独裁的特征到$n\le5$的情况，接近解决该特征是否适用于任意$n$的问题。 |
| [^2] | [Individualized Treatment Allocation in Sequential Network Games.](http://arxiv.org/abs/2302.05747) | 本文针对顺序决策博弈中的互动主体，提出了一种个体化治疗分配方法，通过评估结果的固定分布并采用变分近似和贪婪优化算法，最大化了社会福利准则。 |

# 详细

[^1]: 描述随机串行独裁的特征

    Characterizing Random Serial Dictatorship. (arXiv:2303.11976v1 [econ.TH])

    [http://arxiv.org/abs/2303.11976](http://arxiv.org/abs/2303.11976)

    本文通过使用组合最佳优先算法和二次规划，扩展了随机串行独裁的特征到$n\le5$的情况，接近解决该特征是否适用于任意$n$的问题。

    

    随机串行独裁是一种随机分配规则，它给定了一组对$n$个房屋的选择具有严格偏好的$n$个代理人，满足平等对待、后验效率和策略无关性。对于$n \le 3$，Bogomolnaia和Moulin（2001）已经表明随机串行独裁以这三个公理为特征。本文利用组合最佳优先搜索和二次规划将这个特征扩展到$n \le 5$，更接近回答长期存在的开放性问题，即这个特征是否适用于任意$n$。在此过程中，我们描述了后验效率和策略无关性的削弱形式，这些形式足以满足我们的特征，并确定在针对更大$n$时做出陈述时存在的问题。

    Random serial dictatorship (RSD) is a randomized assignment rule that - given a set of $n$ agents with strict preferences over $n$ houses - satisfies equal treatment of equals, ex post efficiency, and strategyproofness. For $n \le 3$, Bogomolnaia and Moulin (2001) have shown that RSD is characterized by these three axioms. Using best first search in combination with quadratic programming, we extend this characterization to $n \le 5$, getting closer to answering the long-standing open question whether the characterization holds for arbitrary $n$. On the way, we describe weakenings of ex post efficiency and strategyproofness that are sufficient for our characterization and identify problems when making statements for larger $n$.
    
[^2]: 顺序网络博弈中的个体化治疗分配

    Individualized Treatment Allocation in Sequential Network Games. (arXiv:2302.05747v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.05747](http://arxiv.org/abs/2302.05747)

    本文针对顺序决策博弈中的互动主体，提出了一种个体化治疗分配方法，通过评估结果的固定分布并采用变分近似和贪婪优化算法，最大化了社会福利准则。

    

    设计个体化的治疗分配，以最大化互动主体的均衡福利，在政策相关的应用中有很大的意义。本文针对互动主体的顺序决策博弈，开发了一种方法来获得最优的治疗分配规则，通过评估结果的固定分布来最大化社会福利准则。在顺序决策博弈中，固定分布由Gibbs分布给出，由于解析和计算复杂性，很难对治疗分配进行优化。我们采用变分近似来优化固定分布，并使用贪婪优化算法来优化近似平衡福利的治疗分配。我们通过福利遗憾界限推导了变分近似的性能，对贪婪优化算法的性能进行了表征。我们在模拟实验中实现了我们提出的方法。

    Designing individualized allocation of treatments so as to maximize the equilibrium welfare of interacting agents has many policy-relevant applications. Focusing on sequential decision games of interacting agents, this paper develops a method to obtain optimal treatment assignment rules that maximize a social welfare criterion by evaluating stationary distributions of outcomes. Stationary distributions in sequential decision games are given by Gibbs distributions, which are difficult to optimize with respect to a treatment allocation due to analytical and computational complexity. We apply a variational approximation to the stationary distribution and optimize the approximated equilibrium welfare with respect to treatment allocation using a greedy optimization algorithm. We characterize the performance of the variational approximation, deriving a performance guarantee for the greedy optimization algorithm via a welfare regret bound. We implement our proposed method in simulation exerci
    

