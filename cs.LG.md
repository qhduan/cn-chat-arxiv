# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Value function interference and greedy action selection in value-based multi-objective reinforcement learning](https://arxiv.org/abs/2402.06266) | 基于价值的多目标强化学习中，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，会导致价值函数干扰并收敛到次优策略。 |

# 详细

[^1]: 基于价值的多目标强化学习中的价值函数干扰和贪婪动作选择

    Value function interference and greedy action selection in value-based multi-objective reinforcement learning

    [https://arxiv.org/abs/2402.06266](https://arxiv.org/abs/2402.06266)

    基于价值的多目标强化学习中，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，会导致价值函数干扰并收敛到次优策略。

    

    多目标强化学习（MORL）算法将传统的强化学习（RL）扩展到具有多个相互冲突目标的更一般情况下，这些目标由向量值奖励表示。广泛使用的标量RL方法（如Q学习）可以通过（1）学习向量值的价值函数和（2）使用反映用户对不同目标的效用的标量化或排序算子来处理多个目标。然而，正如我们在这里所示，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，这可能会导致代理学习的价值函数干扰，从而收敛到次优策略。这在优化预期标量化回报准则时在随机环境中最为普遍，但我们提供了一个简单的例子证明干扰也可能在确定性环境中出现。

    Multi-objective reinforcement learning (MORL) algorithms extend conventional reinforcement learning (RL) to the more general case of problems with multiple, conflicting objectives, represented by vector-valued rewards. Widely-used scalar RL methods such as Q-learning can be modified to handle multiple objectives by (1) learning vector-valued value functions, and (2) performing action selection using a scalarisation or ordering operator which reflects the user's utility with respect to the different objectives. However, as we demonstrate here, if the user's utility function maps widely varying vector-values to similar levels of utility, this can lead to interference in the value-function learned by the agent, leading to convergence to sub-optimal policies. This will be most prevalent in stochastic environments when optimising for the Expected Scalarised Return criterion, but we present a simple example showing that interference can also arise in deterministic environments. We demonstrat
    

