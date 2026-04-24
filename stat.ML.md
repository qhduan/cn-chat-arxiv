# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal and Fair Encouragement Policy Evaluation and Learning.](http://arxiv.org/abs/2309.07176) | 本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。 |
| [^2] | [Convergence Rates for Non-Log-Concave Sampling and Log-Partition Estimation.](http://arxiv.org/abs/2303.03237) | 非对数凹势V的高维采样速率可以在一些条件下实现与凸函数相同的收敛速率。 |

# 详细

[^1]: 最优和公平的鼓励政策评估与学习

    Optimal and Fair Encouragement Policy Evaluation and Learning. (arXiv:2309.07176v1 [cs.LG])

    [http://arxiv.org/abs/2309.07176](http://arxiv.org/abs/2309.07176)

    本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。

    

    在关键领域中，强制个体接受治疗通常是不可能的，因此在人类不遵循治疗建议的情况下，最优策略规则只是建议。在这些领域中，接受治疗的个体可能存在异质性，治疗效果也可能存在异质性。虽然最优治疗规则可以最大化整个人群的因果结果，但在鼓励的情况下，对于访问平等限制或其他公平考虑因素可能是相关的。例如，在社会服务领域，一个持久的难题是那些最有可能从中受益的人中那些获益服务的使用差距。当决策者对访问和平均结果都有分配偏好时，最优决策规则会发生变化。我们研究了因果识别、统计方差减少估计和稳健估计的最优治疗规则，包括在违反阳性条件的情况下。

    In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. While optimal treatment rules can maximize causal outcomes across the population, access parity constraints or other fairness considerations can be relevant in the case of encouragement. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study causal identification, statistical variance-reduced estimation, and robust estimation of optimal treatment rules, including under potential violations of positivity. We c
    
[^2]: 非对数凹采样和对数分区估计的收敛速率

    Convergence Rates for Non-Log-Concave Sampling and Log-Partition Estimation. (arXiv:2303.03237v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.03237](http://arxiv.org/abs/2303.03237)

    非对数凹势V的高维采样速率可以在一些条件下实现与凸函数相同的收敛速率。

    

    从吉布斯分布$p(x)\propto\exp(-V(x)/\epsilon)$中采样并计算其对数分区函数是统计学、机器学习和统计物理中的基本任务。然而，虽然有效的算法已知于凸势函数$V$，但非凸情况下的情况要困难得多，算法必然在最坏情况下受到维度灾难的困扰。最近，已经证明在适当的条件下，高维采样非对数凹势V的速率也可以达到同样快的速度。本文对这些结果进行了回顾，并强调了领域中的一些开放问题。

    Sampling from Gibbs distributions $p(x) \propto \exp(-V(x)/\varepsilon)$ and computing their log-partition function are fundamental tasks in statistics, machine learning, and statistical physics. However, while efficient algorithms are known for convex potentials $V$, the situation is much more difficult in the non-convex case, where algorithms necessarily suffer from the curse of dimensionality in the worst case. For optimization, which can be seen as a low-temperature limit of sampling, it is known that smooth functions $V$ allow faster convergence rates. Specifically, for $m$-times differentiable functions in $d$ dimensions, the optimal rate for algorithms with $n$ function evaluations is known to be $O(n^{-m/d})$, where the constant can potentially depend on $m, d$ and the function to be optimized. Hence, the curse of dimensionality can be alleviated for smooth functions at least in terms of the convergence rate. Recently, it has been shown that similarly fast rates can also be ach
    

