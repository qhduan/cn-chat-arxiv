# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Irrational Random Utility Models](https://arxiv.org/abs/2403.10208) | 理性决策者群体的选择可以被足够不相关的偏好的非理性决策者群体所代表，即非理性RUM。几乎所有的RUMs可以通过一部分决策者是非理性的群体来表示，并且在特定条件下他们的非理性行为是不受限制的。 |
| [^2] | [Regularizing Discrimination in Optimal Policy Learning with Distributional Targets](https://arxiv.org/abs/2401.17909) | 为了解决优化策略学习中的歧视问题，研究者提出了一个框架，允许决策者通过惩罚来防止在特定人群中的不公平结果分布，该框架对目标函数和歧视度量具有很大的灵活性，通过数据驱动的参数调整，可以在实践中具备遗憾和一致性保证。 |
| [^3] | [Endogenous Barriers to Learning.](http://arxiv.org/abs/2306.16904) | 本文通过建模代理的行为，研究了学习的内在障碍。实验发现，在某些游戏中，增加准确性会导致不稳定的最佳反应动态。通过将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平，提出了一个limitQR均衡。同时，讨论了策略限制在减少或增加学习障碍方面的作用。 |
| [^4] | [Q-based Equilibria.](http://arxiv.org/abs/2304.12647) | 该论文研究了基于Q的策略规则族中的均衡偏差（或 Qb-equilibria），即Q值在不同监测技术下的效果。 |

# 详细

[^1]: 非理性随机效用模型

    Irrational Random Utility Models

    [https://arxiv.org/abs/2403.10208](https://arxiv.org/abs/2403.10208)

    理性决策者群体的选择可以被足够不相关的偏好的非理性决策者群体所代表，即非理性RUM。几乎所有的RUMs可以通过一部分决策者是非理性的群体来表示，并且在特定条件下他们的非理性行为是不受限制的。

    

    我们展示了一个理性决策者群体的集合选择 - 随机效用模型（RUMs） - 只有当他们的偏好足够不相关时，可以通过一个非理性决策者群体来表示。我们称这种表示为：非理性RUM。然后我们展示几乎所有的RUMs都可以通过至少一部分决策者是非理性决策者的群体来表示，并且在特定条件下他们的非理性行为是不受限制的。

    arXiv:2403.10208v1 Announce Type: new  Abstract: We show that the set of aggregate choices of a population of rational decision-makers - random utility models (RUMs) - can be represented by a population of irrational ones if, and only if, their preferences are sufficiently uncorrelated. We call this representation: Irrational RUM. We then show that almost all RUMs can be represented by a population in which at least some decision-makers are irrational and that under specific conditions their irrational behavior is unconstrained.
    
[^2]: 优化策略学习中正则化歧视问题的研究

    Regularizing Discrimination in Optimal Policy Learning with Distributional Targets

    [https://arxiv.org/abs/2401.17909](https://arxiv.org/abs/2401.17909)

    为了解决优化策略学习中的歧视问题，研究者提出了一个框架，允许决策者通过惩罚来防止在特定人群中的不公平结果分布，该框架对目标函数和歧视度量具有很大的灵活性，通过数据驱动的参数调整，可以在实践中具备遗憾和一致性保证。

    

    决策者通常通过训练数据学习治疗的相对效果，并选择一个实施机制，该机制根据某个目标函数预测了“最优”结果分布。然而，一个意识到歧视问题的决策者可能不满意以严重歧视人群子组的代价来实现该优化，即在子组中的结果分布明显偏离整体最优结果分布。我们研究了一个框架，允许决策者惩罚这种偏差，并可以使用各种目标函数和歧视度量。我们对具有数据驱动调参的经验成功策略建立了遗憾和一致性保证，并提供了数值结果。此外，我们还对两个实证场景进行了简要说明。

    A decision maker typically (i) incorporates training data to learn about the relative effectiveness of the treatments, and (ii) chooses an implementation mechanism that implies an "optimal" predicted outcome distribution according to some target functional. Nevertheless, a discrimination-aware decision maker may not be satisfied achieving said optimality at the cost of heavily discriminating against subgroups of the population, in the sense that the outcome distribution in a subgroup deviates strongly from the overall optimal outcome distribution. We study a framework that allows the decision maker to penalize for such deviations, while allowing for a wide range of target functionals and discrimination measures to be employed. We establish regret and consistency guarantees for empirical success policies with data-driven tuning parameters, and provide numerical results. Furthermore, we briefly illustrate the methods in two empirical settings.
    
[^3]: 学习的内在障碍

    Endogenous Barriers to Learning. (arXiv:2306.16904v1 [econ.GN])

    [http://arxiv.org/abs/2306.16904](http://arxiv.org/abs/2306.16904)

    本文通过建模代理的行为，研究了学习的内在障碍。实验发现，在某些游戏中，增加准确性会导致不稳定的最佳反应动态。通过将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平，提出了一个limitQR均衡。同时，讨论了策略限制在减少或增加学习障碍方面的作用。

    

    动机在于缺乏经验会导致错误，而经验可以减少这些错误，我们使用随机选择模型来建模代理的行为，将其选择的准确性设定为内生变量。在某些游戏中，增加准确性有可能导致不稳定的最佳反应动态。我们将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平。使用逻辑量化响应，这定义了一个limitQR均衡。我们将该概念应用于蜈蚣、旅行者困境和11-20钱求游戏，以及一价和全付拍卖，并讨论策略限制在减少或增加学习障碍方面的作用。

    Motivated by the idea that lack of experience is a source of errors but that experience should reduce them, we model agents' behavior using a stochastic choice model, leaving endogenous the accuracy of their choice. In some games, increased accuracy is conducive to unstable best-response dynamics. We define the barrier to learning as the minimum level of noise which keeps the best-response dynamic stable. Using logit Quantal Response, this defines a limitQR Equilibrium. We apply the concept to centipede, travelers' dilemma, and 11-20 money-request games and to first-price and all-pay auctions, and discuss the role of strategy restrictions in reducing or amplifying barriers to learning.
    
[^4]: 基于Q的均衡

    Q-based Equilibria. (arXiv:2304.12647v1 [econ.TH])

    [http://arxiv.org/abs/2304.12647](http://arxiv.org/abs/2304.12647)

    该论文研究了基于Q的策略规则族中的均衡偏差（或 Qb-equilibria），即Q值在不同监测技术下的效果。

    

    在动态环境中，Q学习是一种自适应规则，其为每个替代方案提供估计值(即Q值)，该值与之前的决策相关。一个朴素的策略是始终选择具有最高Q值的替代方案。我们考虑一族基于Q的策略规则，这些规则可能系统地支持某些替代方案而不是其他替代方案，例如包含有利合作的宽容偏差的规则。在 Compte 和 Postlewaite [2018] 的精神下，我们在这个 Q-based 规则族中寻找均衡偏差（或 Qb-equilibria）。我们研究了不同监测技术下的经典博弈。

    In dynamic environments, Q-learning is an adaptative rule that provides an estimate (a Q-value) of the continuation value associated with each alternative. A naive policy consists in always choosing the alternative with highest Q-value. We consider a family of Q-based policy rules that may systematically favor some alternatives over others, for example rules that incorporate a leniency bias that favors cooperation. In the spirit of Compte and Postlewaite [2018], we look for equilibrium biases (or Qb-equilibria) within this family of Q-based rules. We examine classic games under various monitoring technologies.
    

