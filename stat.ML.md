# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation.](http://arxiv.org/abs/2401.03893) | 本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。 |
| [^2] | [Incentivizing Honesty among Competitors in Collaborative Learning and Optimization.](http://arxiv.org/abs/2305.16272) | 这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。 |

# 详细

[^1]: 非线性双时间尺度随机逼近中的有限时间解耦收敛

    Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation. (arXiv:2401.03893v1 [math.OC])

    [http://arxiv.org/abs/2401.03893](http://arxiv.org/abs/2401.03893)

    本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。

    

    在双时间尺度随机逼近中，使用不同的步长以不同的速度更新两个迭代，每次更新都会影响另一个。先前的线性双时间尺度随机逼近研究发现，这些更新的均方误差的收敛速度仅仅取决于它们各自的步长，导致了所谓的解耦收敛。然而，在非线性随机逼近中实现这种解耦收敛的可能性仍不明确。我们的研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力。我们发现，在较弱的Lipschitz条件下，传统分析无法实现解耦收敛。这一发现在数值上得到了进一步的支持。但是通过引入一个嵌套局部线性条件，我们证明了在适当选择与平滑度相关的步长的情况下，解耦收敛仍然是可行的。

    In two-time-scale stochastic approximation (SA), two iterates are updated at varying speeds using different step sizes, with each update influencing the other. Previous studies in linear two-time-scale SA have found that the convergence rates of the mean-square errors for these updates are dependent solely on their respective step sizes, leading to what is referred to as decoupled convergence. However, the possibility of achieving this decoupled convergence in nonlinear SA remains less understood. Our research explores the potential for finite-time decoupled convergence in nonlinear two-time-scale SA. We find that under a weaker Lipschitz condition, traditional analyses are insufficient for achieving decoupled convergence. This finding is further numerically supported by a counterexample. But by introducing an additional condition of nested local linearity, we show that decoupled convergence is still feasible, contingent on the appropriate choice of step sizes associated with smoothnes
    
[^2]: 在协同学习和优化中激励竞争对手诚实行为的研究

    Incentivizing Honesty among Competitors in Collaborative Learning and Optimization. (arXiv:2305.16272v1 [cs.LG])

    [http://arxiv.org/abs/2305.16272](http://arxiv.org/abs/2305.16272)

    这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。

    

    协同学习技术能够让机器学习模型的训练比仅利用单一数据源的模型效果更好。然而，在许多情况下，潜在的参与者是下游任务中的竞争对手，如每个都希望通过提供最佳推荐来吸引客户的公司。这可能会激励不诚实的更新，损害其他参与者的模型，从而可能破坏协作的好处。在这项工作中，我们制定了一个模型来描述这种交互，并在该框架内研究了两个学习任务：单轮均值估计和强凸目标的多轮 SGD。对于一类自然的参与者行为，我们发现理性的客户会被激励强烈地操纵他们的更新，从而防止学习。然后，我们提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。最后，我们通过实验证明了这一点。

    Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entity's data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the
    

