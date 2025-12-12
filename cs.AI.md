# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2306.01270) | 提出了一种多机器人路径规划方法MAPPOHR，该方法结合了启发式搜索、经验规则和多智能体强化学习。实验结果表明，该方法在规划效率和避碰能力方面优于现有方法。 |

# 详细

[^1]: 组合启发式和多智能体强化学习的多机器人路径规划

    Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning. (arXiv:2306.01270v1 [cs.AI])

    [http://arxiv.org/abs/2306.01270](http://arxiv.org/abs/2306.01270)

    提出了一种多机器人路径规划方法MAPPOHR，该方法结合了启发式搜索、经验规则和多智能体强化学习。实验结果表明，该方法在规划效率和避碰能力方面优于现有方法。

    

    动态环境下的多机器人路径规划是一个极具挑战性的经典问题。在移动过程中，机器人需要避免与其他移动机器人发生碰撞，同时最小化它们的行驶距离。以往的方法要么使用启发式搜索方法不断重新规划路径以避免冲突，要么基于学习方法选择适当的避碰策略。前者可能由于频繁的重新规划导致行驶距离较长，而后者可能由于低样本探索和利用率而导致学习效率低，从而使模型的训练成本较高。为解决这些问题，我们提出了一种路径规划方法MAPPOHR，该方法结合了启发式搜索、经验规则和多智能体强化学习。该方法包含两个层次：基于多智能体强化学习算法MAPPO的实时规划器，其将经验规则嵌入到动作输出层和奖励函数中；以及一个启发式规划器，它生成初始路径并向MAPPO规划器添加约束。我们的实验结果表明，所提出的方法在规划效率和避碰能力方面优于现有方法。

    Multi-robot path finding in dynamic environments is a highly challenging classic problem. In the movement process, robots need to avoid collisions with other moving robots while minimizing their travel distance. Previous methods for this problem either continuously replan paths using heuristic search methods to avoid conflicts or choose appropriate collision avoidance strategies based on learning approaches. The former may result in long travel distances due to frequent replanning, while the latter may have low learning efficiency due to low sample exploration and utilization, and causing high training costs for the model. To address these issues, we propose a path planning method, MAPPOHR, which combines heuristic search, empirical rules, and multi-agent reinforcement learning. The method consists of two layers: a real-time planner based on the multi-agent reinforcement learning algorithm, MAPPO, which embeds empirical rules in the action output layer and reward functions, and a heuri
    

