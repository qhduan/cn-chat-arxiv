# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^2] | [Stable Menus of Public Goods: A Matching Problem](https://arxiv.org/abs/2402.11370) | 研究匹配问题中的稳定菜单问题，提出了保证存在稳定解决方案的条件，对无策略性稳定匹配给出了积极和消极结果。 |
| [^3] | [Using Forests in Multivariate Regression Discontinuity Designs.](http://arxiv.org/abs/2303.11721) | 本文提出了一种基于森林的估计器，可以灵活地建模多元得分中的回归不连续设计，相比于局部线性回归在高维空间运算时具有优势。 |
| [^4] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |

# 详细

[^1]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^2]: 公共物品的稳定菜单: 一个匹配问题

    Stable Menus of Public Goods: A Matching Problem

    [https://arxiv.org/abs/2402.11370](https://arxiv.org/abs/2402.11370)

    研究匹配问题中的稳定菜单问题，提出了保证存在稳定解决方案的条件，对无策略性稳定匹配给出了积极和消极结果。

    

    我们研究了在没有货币转移的情境下，代理者和公共物品之间的匹配问题。由于物品是公共的，它们没有容量限制。没有外生定义的提供物品的预算。相反，每个提供的物品必须证明其成本，导致在物品的“偏好”中存在很强的互补性。此外，鉴于已提供的其他物品，那些需求量高的物品也必须得到提供。存在一个稳定解决方案（提供的公共物品的菜单）的问题展示了丰富的组合结构。我们揭示了保证存在稳定解决方案的充分条件和必要条件，并为无策略性稳定匹配得出了积极和消极的结果。

    arXiv:2402.11370v1 Announce Type: cross  Abstract: We study a matching problem between agents and public goods, in settings without monetary transfers. Since goods are public, they have no capacity constraints. There is no exogenously defined budget of goods to be provided. Rather, each provided good must justify its cost, leading to strong complementarities in the "preferences" of goods. Furthermore, goods that are in high demand given other already-provided goods must also be provided. The question of the existence of a stable solution (a menu of public goods to be provided) exhibits a rich combinatorial structure. We uncover sufficient conditions and necessary conditions for guaranteeing the existence of a stable solution, and derive both positive and negative results for strategyproof stable matching.
    
[^3]: 多元回归不连续设计中的森林应用

    Using Forests in Multivariate Regression Discontinuity Designs. (arXiv:2303.11721v1 [econ.EM])

    [http://arxiv.org/abs/2303.11721](http://arxiv.org/abs/2303.11721)

    本文提出了一种基于森林的估计器，可以灵活地建模多元得分中的回归不连续设计，相比于局部线性回归在高维空间运算时具有优势。

    

    本文讨论在具有多个得分的回归不连续设计中估计条件治疗效应。虽然当治疗状态完全由一个连续变量描述时，局部线性回归已经成为常见方法，但在包含多个治疗分配规则的实际应用中，这些方法不易推广。我们提出一种基于森林的估计器，可以灵活地建模多元得分，通过在治疗边界两侧建立两个基于Wager和Athey（2018）诚实森林。该估计器是渐近正常的，并避免了在高维空间中运行局部线性回归的缺陷。在模拟中，我们发现我们提出的估计器在多元设计中优于局部线性回归，并且与Imbens和Wager（2019）的最小极小估计器竞争。

    We discuss estimating conditional treatment effects in regression discontinuity designs with multiple scores. While local linear regressions have been popular in settings where the treatment status is completely described by one running variable, they do not easily generalize to empirical applications involving multiple treatment assignment rules. In practice, the multivariate problem is usually reduced to a univariate one where using local linear regressions is suitable. Instead, we propose a forest-based estimator that can flexibly model multivariate scores, where we build two honest forests in the sense of Wager and Athey (2018) on both sides of the treatment boundary. This estimator is asymptotically normal and sidesteps the pitfalls of running local linear regressions in higher dimensions. In simulations, we find our proposed estimator outperforms local linear regressions in multivariate designs and is competitive against the minimax-optimal estimator of Imbens and Wager (2019). T
    
[^4]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    

