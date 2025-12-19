# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Managing Persuasion Robustly: The Optimality of Quota Rules.](http://arxiv.org/abs/2310.10024) | 接收者在承诺决策规则时，最优策略是采用配额规则，以完全保持发送者的激励一致。 |
| [^2] | [Optimal Decision Rules when Payoffs are Partially Identified.](http://arxiv.org/abs/2204.11748) | 该论文导出了处理部分识别的离散选择问题的最优统计决策规则，通过优化最大风险或遗憾来实现最优决策，并通过bootstrap和贝叶斯方法在参数和半参数模型中实现，可用于解决最优治疗选择和最优定价等问题。 |

# 详细

[^1]: 高效管理说服力：定额规则的最优性

    Managing Persuasion Robustly: The Optimality of Quota Rules. (arXiv:2310.10024v1 [econ.TH])

    [http://arxiv.org/abs/2310.10024](http://arxiv.org/abs/2310.10024)

    接收者在承诺决策规则时，最优策略是采用配额规则，以完全保持发送者的激励一致。

    

    我们研究了一个发送者-接收者模型，其中接收者可以在发送者确定信息策略之前承诺一个决策规则。决策规则可以依赖于信号结构和发送者采用的信号实现。这个框架涵盖了从一个利益相关方（发送者）那里征求意见的决策者（接收者）面临的不确定性。在这些应用中，接收者面临着对发送者偏好和可行信号结构集合的不确定性。因此，我们采用了一个统一的鲁棒分析框架，将最大最小效用、最小最大遗憾和最小最大近似比纳入了特殊情况。我们表明，为了完全保持发送者的激励一致，接收者在实现后期最优性的同时，牺牲了一致性定额规则下行动的边际分布。最优决策规则是一个配额规则，即决策规则在保证约束条件下，最大化接收者的期望收益。

    We study a sender-receiver model where the receiver can commit to a decision rule before the sender determines the information policy. The decision rule can depend on the signal structure and the signal realization that the sender adopts. This framework captures applications where a decision-maker (the receiver) solicit advice from an interested party (sender). In these applications, the receiver faces uncertainty regarding the sender's preferences and the set of feasible signal structures. Consequently, we adopt a unified robust analysis framework that includes max-min utility, min-max regret, and min-max approximation ratio as special cases. We show that it is optimal for the receiver to sacrifice ex-post optimality to perfectly align the sender's incentive. The optimal decision rule is a quota rule, i.e., the decision rule maximizes the receiver's ex-ante payoff subject to the constraint that the marginal distribution over actions adheres to a consistent quota, regardless of the sen
    
[^2]: 支付部分确认时的最优决策规则

    Optimal Decision Rules when Payoffs are Partially Identified. (arXiv:2204.11748v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2204.11748](http://arxiv.org/abs/2204.11748)

    该论文导出了处理部分识别的离散选择问题的最优统计决策规则，通过优化最大风险或遗憾来实现最优决策，并通过bootstrap和贝叶斯方法在参数和半参数模型中实现，可用于解决最优治疗选择和最优定价等问题。

    

    当支付依赖于部分确认的参数$\theta$且决策者可以使用点确认的参数$P$来推断对$\theta$的限制时，我们导出了离散选择问题的最优统计决策规则，其中先进的例子包括部分识别下的最优治疗选择和具有丰富未观察到的异质性的最优定价。我们的最优决策规则最小化在$P$条件下支付确认集合上的最大风险或遗憾，并有效地利用数据来学习$P$。我们讨论了最优决策规则的通过bootstrap和贝叶斯方法在参数和半参数模型中的实现。我们提供了关于治疗选择和最优定价的详细应用。使用实验极限框架，我们展示了我们的最优决策规则可以超越看似自然的替代方案。我们的渐近方法非常适合现实经验设置，其中有限样本最优的推导是困难的。

    We derive optimal statistical decision rules for discrete choice problems when payoffs depend on a partially-identified parameter $\theta$ and the decision maker can use a point-identified parameter $P$ to deduce restrictions on $\theta$. Leading examples include optimal treatment choice under partial identification and optimal pricing with rich unobserved heterogeneity. Our optimal decision rules minimize the maximum risk or regret over the identified set of payoffs conditional on $P$ and use the data efficiently to learn about $P$. We discuss implementation of optimal decision rules via the bootstrap and Bayesian methods, in both parametric and semiparametric models. We provide detailed applications to treatment choice and optimal pricing. Using a limits of experiments framework, we show that our optimal decision rules can dominate seemingly natural alternatives. Our asymptotic approach is well suited for realistic empirical settings in which the derivation of finite-sample optimal r
    

