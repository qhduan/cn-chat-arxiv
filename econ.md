# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequential unanimity voting rules for binary social choice](https://arxiv.org/abs/2402.13009) | 提出了一种新颖的中立和策略证明规则家族，称为顺序一致性规则，通过算法将M-获胜联盟规则转换为等效的顺序一致性规则，支持选民在完整偏好领域中可能的中立立场。 |
| [^2] | [Approximating the set of Nash equilibria for convex games.](http://arxiv.org/abs/2310.04176) | 本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。 |
| [^3] | [The Power of Tests for Detecting $p$-Hacking.](http://arxiv.org/abs/2205.07950) | 本文从理论和模拟的层面，研究了$p$-hacking的可能形式对$p-value$分布和现有检测方法的影响，证明了测试揭示“p-hacking”的能力可能非常低，并且关键取决于具体的$p$-hacking策略和研究测试的实际效果的分布。 |

# 详细

[^1]: 二元社会选择的顺序一致性投票规则

    Sequential unanimity voting rules for binary social choice

    [https://arxiv.org/abs/2402.13009](https://arxiv.org/abs/2402.13009)

    提出了一种新颖的中立和策略证明规则家族，称为顺序一致性规则，通过算法将M-获胜联盟规则转换为等效的顺序一致性规则，支持选民在完整偏好领域中可能的中立立场。

    

    我们考虑一个需要在两个候选人之间做决定的选民群体。我们提出了一种新颖的中立和策略证明规则家族，我们称之为顺序一致性规则。通过展示它们与Moulin（1983）的M-获胜联盟规则的形式等价，我们表明顺序一致性规则的特征是中立性和策略证明。我们通过开发算法将给定的M-获胜联盟规则转换为等效的顺序一致性规则以及反之来建立我们的结果。这种分析可扩展以适应选民可能在候选人之间持中立立场的完整偏好领域。

    arXiv:2402.13009v1 Announce Type: new  Abstract: We consider a group of voters that needs to decide between two candidates. We propose a novel family of neutral and strategy-proof rules, which we call sequential unanimity rules. By demonstrating their formal equivalence to the M-winning coalition rules of Moulin (1983), we show that sequential unanimity rules are characterized by neutrality and strategy-proofness. We establish our results by developing algorithms that transform a given M-winning coalition rule into an equivalent sequential unanimity rule and vice versa. The analysis can be extended to accommodate the full preference domain in which voters may be indifferent between candidates.
    
[^2]: 近似计算凸博弈中纳什均衡解集合

    Approximating the set of Nash equilibria for convex games. (arXiv:2310.04176v1 [math.OC])

    [http://arxiv.org/abs/2310.04176](http://arxiv.org/abs/2310.04176)

    本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。

    

    在Feinstein和Rudloff（2023）中，他们证明了对于任意非合作$N$人博弈，纳什均衡解集合与具有非凸顺序锥的某个向量优化问题的帕累托最优点集合是一致的。为了避免处理非凸顺序锥，我们证明了将纳什均衡解集合等价地表示为$N$个多目标问题（即具有自然顺序锥）的帕累托最优点的交集。目前，计算多目标问题的精确帕累托最优点集合的算法仅适用于线性问题的类别，这将导致这些算法只能用于解线性博弈的真实纳什均衡集合的可能性降低。本文中，我们将考虑更大类别的凸博弈。由于通常只能为凸向量优化问题计算近似解，我们首先展示了类似于上述结果的结果，即$\epsilon$-近似纳什均衡解集合与问题完全相似。

    In Feinstein and Rudloff (2023), it was shown that the set of Nash equilibria for any non-cooperative $N$ player game coincides with the set of Pareto optimal points of a certain vector optimization problem with non-convex ordering cone. To avoid dealing with a non-convex ordering cone, an equivalent characterization of the set of Nash equilibria as the intersection of the Pareto optimal points of $N$ multi-objective problems (i.e.\ with the natural ordering cone) is proven. So far, algorithms to compute the exact set of Pareto optimal points of a multi-objective problem exist only for the class of linear problems, which reduces the possibility of finding the true set of Nash equilibria by those algorithms to linear games only.  In this paper, we will consider the larger class of convex games. As, typically, only approximate solutions can be computed for convex vector optimization problems, we first show, in total analogy to the result above, that the set of $\epsilon$-approximate Nash
    
[^3]: 测试揭示“p-hacking”的能力

    The Power of Tests for Detecting $p$-Hacking. (arXiv:2205.07950v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.07950](http://arxiv.org/abs/2205.07950)

    本文从理论和模拟的层面，研究了$p$-hacking的可能形式对$p-value$分布和现有检测方法的影响，证明了测试揭示“p-hacking”的能力可能非常低，并且关键取决于具体的$p$-hacking策略和研究测试的实际效果的分布。

    

    $p$-Hacking可能会削弱经验研究的有效性。一篇繁荣的经验文献调查了基于报告的$p$-value在研究中的分布的$p$-hacking的普遍存在。解释这个文献中的结果需要仔细理解用于检测不同类型$p$-hacking的方法的能力。我们从理论上研究了$p$-hacking的可能形式对报告的$p$-value分布和现有检测方法的能力的影响。能力可能非常低，关键取决于特定的$p$-hacking策略和研究测试的实际效果的分布。出版偏差可以增强测试无$p$-hacking和无出版偏差的联合零假设的能力。我们将测试的能力与$p$-hacking的成本相关联，并显示当$p$-hacking非常昂贵时，能力倾向于更大。蒙特卡罗模拟支持我们的理论结果。

    $p$-Hacking can undermine the validity of empirical studies. A flourishing empirical literature investigates the prevalence of $p$-hacking based on the empirical distribution of reported $p$-values across studies. Interpreting results in this literature requires a careful understanding of the power of methods used to detect different types of $p$-hacking. We theoretically study the implications of likely forms of $p$-hacking on the distribution of reported $p$-values and the power of existing methods for detecting it. Power can be quite low, depending crucially on the particular $p$-hacking strategy and the distribution of actual effects tested by the studies. Publication bias can enhance the power for testing the joint null hypothesis of no $p$-hacking and no publication bias. We relate the power of the tests to the costs of $p$-hacking and show that power tends to be larger when $p$-hacking is very costly. Monte Carlo simulations support our theoretical results.
    

