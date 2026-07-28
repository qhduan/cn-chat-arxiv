# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalizing Better Response Paths and Weakly Acyclic Games](https://arxiv.org/abs/2403.18086) | 本文提出了弱环游戏的一个泛化概念，在多智能体学习中具有重要作用，尤其在智能体无法做出最佳响应时。这种泛化概念与最近的满足性路径理论密切相关。 |
| [^2] | [Causal Interpretation of Estimands Defined by Exposure Mappings](https://arxiv.org/abs/2403.08183) | 该论文研究了在对相互干扰施加弱限制条件时因果解释估计量的问题，并提出了用于因果可解释性的符号保留标准。 |
| [^3] | [Managing Persuasion Robustly: The Optimality of Quota Rules.](http://arxiv.org/abs/2310.10024) | 接收者在承诺决策规则时，最优策略是采用配额规则，以完全保持发送者的激励一致。 |
| [^4] | [Equivalence of inequality indices: Three dimensions of impact revisited.](http://arxiv.org/abs/2304.07479) | 本文探讨了不平等是自然演化的结果和通过数学模型和迭代过程来量化不平等的方法。其中，冲击分布可以通过时间依赖性代理模型来准确建模并结合富者越富和纯粹偶然成分。 |

# 详细

[^1]: 泛化更好的响应路径和弱环游戏

    Generalizing Better Response Paths and Weakly Acyclic Games

    [https://arxiv.org/abs/2403.18086](https://arxiv.org/abs/2403.18086)

    本文提出了弱环游戏的一个泛化概念，在多智能体学习中具有重要作用，尤其在智能体无法做出最佳响应时。这种泛化概念与最近的满足性路径理论密切相关。

    

    弱环游戏泛化潜在游戏，并且对于研究博弈论控制是基础性的。本文提出了弱环游戏的一个泛化，并观察到在多智能体学习中的重要性，当智能体在无法做出最佳响应时采用实验策略更新。虽然弱环性是用游戏的更好响应图的路径连接特性来定义的，我们的泛化是用泛化更好响应图来定义的。我们给出了这种泛化弱环性在两人游戏和$n$人游戏中的充分条件。为了证明我们的泛化并非微不足道，我们提供了存在纯纳什均衡的博弈的例子，这些博弈不具有泛化弱环性。本文提出的泛化与最近的满足性路径理论密切相关，这里提供的反例构成了

    arXiv:2403.18086v1 Announce Type: cross  Abstract: Weakly acyclic games generalize potential games and are fundamental to the study of game theoretic control. In this paper, we present a generalization of weakly acyclic games, and we observe its importance in multi-agent learning when agents employ experimental strategy updates in periods where they fail to best respond. While weak acyclicity is defined in terms of path connectivity properties of a game's better response graph, our generalization is defined using a generalized better response graph. We provide sufficient conditions for this notion of generalized weak acyclicity in both two-player games and $n$-player games. To demonstrate that our generalization is not trivial, we provide examples of games admitting a pure Nash equilibrium that are not generalized weakly acyclic. The generalization presented in this work is closely related to the recent theory of satisficing paths, and the counterexamples presented here constitute the 
    
[^2]: 由暴露映射定义的估计量的因果解释

    Causal Interpretation of Estimands Defined by Exposure Mappings

    [https://arxiv.org/abs/2403.08183](https://arxiv.org/abs/2403.08183)

    该论文研究了在对相互干扰施加弱限制条件时因果解释估计量的问题，并提出了用于因果可解释性的符号保留标准。

    

    在存在相互干扰的情况下，通常利用由暴露映射定义的估计量来总结与个体相邻的治疗分配变化的影响。本文研究了在对相互干扰施加弱限制条件时它们的因果解释。我们证明在传统的识别条件下，这些估计量可能出现不可取的符号反转。这促使提出用于因果可解释性的符号保留标准。为满足首选标准，有必要对相互干扰施加约束，无论是在潜在结果还是在治疗选择中。我们提供了充分条件，并展示它们由一个允许在结果和选择阶段中存在复杂干扰形式的非参数模型满足。

    arXiv:2403.08183v1 Announce Type: new  Abstract: In settings with interference, it is common to utilize estimands defined by exposure mappings to summarize the impact of variation in treatment assignments local to the ego. This paper studies their causal interpretation under weak restrictions on interference. We demonstrate that the estimands can exhibit unpalatable sign reversals under conventional identification conditions. This motivates the formulation of sign preservation criteria for causal interpretability. To satisfy preferred criteria, it is necessary to impose restrictions on interference, either in potential outcomes or selection into treatment. We provide sufficient conditions and show that they are satisfied by a nonparametric model allowing for a complex form of interference in both the outcome and selection stages.
    
[^3]: 高效管理说服力：定额规则的最优性

    Managing Persuasion Robustly: The Optimality of Quota Rules. (arXiv:2310.10024v1 [econ.TH])

    [http://arxiv.org/abs/2310.10024](http://arxiv.org/abs/2310.10024)

    接收者在承诺决策规则时，最优策略是采用配额规则，以完全保持发送者的激励一致。

    

    我们研究了一个发送者-接收者模型，其中接收者可以在发送者确定信息策略之前承诺一个决策规则。决策规则可以依赖于信号结构和发送者采用的信号实现。这个框架涵盖了从一个利益相关方（发送者）那里征求意见的决策者（接收者）面临的不确定性。在这些应用中，接收者面临着对发送者偏好和可行信号结构集合的不确定性。因此，我们采用了一个统一的鲁棒分析框架，将最大最小效用、最小最大遗憾和最小最大近似比纳入了特殊情况。我们表明，为了完全保持发送者的激励一致，接收者在实现后期最优性的同时，牺牲了一致性定额规则下行动的边际分布。最优决策规则是一个配额规则，即决策规则在保证约束条件下，最大化接收者的期望收益。

    We study a sender-receiver model where the receiver can commit to a decision rule before the sender determines the information policy. The decision rule can depend on the signal structure and the signal realization that the sender adopts. This framework captures applications where a decision-maker (the receiver) solicit advice from an interested party (sender). In these applications, the receiver faces uncertainty regarding the sender's preferences and the set of feasible signal structures. Consequently, we adopt a unified robust analysis framework that includes max-min utility, min-max regret, and min-max approximation ratio as special cases. We show that it is optimal for the receiver to sacrifice ex-post optimality to perfectly align the sender's incentive. The optimal decision rule is a quota rule, i.e., the decision rule maximizes the receiver's ex-ante payoff subject to the constraint that the marginal distribution over actions adheres to a consistent quota, regardless of the sen
    
[^4]: 不等式指数的等效性：重审三维影响

    Equivalence of inequality indices: Three dimensions of impact revisited. (arXiv:2304.07479v1 [physics.soc-ph])

    [http://arxiv.org/abs/2304.07479](http://arxiv.org/abs/2304.07479)

    本文探讨了不平等是自然演化的结果和通过数学模型和迭代过程来量化不平等的方法。其中，冲击分布可以通过时间依赖性代理模型来准确建模并结合富者越富和纯粹偶然成分。

    

    不平等是我们生活中固有的一部分：我们可以在收入、才能、资源和引用等方面看到它。不平等的强度因不同的环境而异：从相对均匀分布的环境，到少数利益相关者控制大部分可用资源的环境。我们想了解为什么不平等会自然地成为任何系统发展的结果。研究受直觉假设支配的简单数学模型可以为解决这个问题带来很多见解。特别是，我们最近观察到（Siudem et al.，PNAS 117:13896-13900, 2020），冲击分布可以通过涉及富者越富和纯粹偶然成分的时间依赖性代理模型来准确建模。在这里，我们指出这种模型与生成任意长度和Gini指数预定义的不平等水平的等级分布的迭代过程之间的关系。许多指数 quantifying

    Inequality is an inherent part of our lives: we see it in the distribution of incomes, talents, resources, and citations, amongst many others. Its intensity varies across different environments: from relatively evenly distributed ones, to where a small group of stakeholders controls the majority of the available resources. We would like to understand why inequality naturally arises as a consequence of the natural evolution of any system. Studying simple mathematical models governed by intuitive assumptions can bring many insights into this problem. In particular, we recently observed (Siudem et al., PNAS 117:13896-13900, 2020) that impact distribution might be modelled accurately by a time-dependent agent-based model involving a mixture of the rich-get-richer and sheer chance components. Here we point out its relationship to an iterative process that generates rank distributions of any length and a predefined level of inequality, as measured by the Gini index.  Many indices quantifying
    

