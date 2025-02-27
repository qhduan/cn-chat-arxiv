# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement.](http://arxiv.org/abs/2310.03984) | AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。 |

# 详细

[^1]: AdaRec：用于增强用户长期参与度的自适应顺序推荐算法

    AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement. (arXiv:2310.03984v1 [cs.IR])

    [http://arxiv.org/abs/2310.03984](http://arxiv.org/abs/2310.03984)

    AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。

    

    在顺序推荐任务中，人们越来越关注使用强化学习算法来优化用户的长期参与度。大规模在线推荐系统面临的一个挑战是用户行为模式（如互动频率和保留倾向）的不断复杂变化。当将问题建模为马尔科夫决策过程时，推荐系统的动态和奖励函数会不断受到这些变化的影响。现有的推荐系统强化学习算法会受到分布偏移问题的困扰，并难以适应这种马尔科夫决策过程。本文介绍了一种新的范式，称为自适应顺序推荐（AdaRec），来解决这个问题。AdaRec提出了一种基于距离的表示损失，从用户的互动轨迹中提取潜在信息。这些信息反映了强化学习策略与当前用户行为模式的匹配程度，并帮助策略识别推荐系统中的细微变化。

    Growing attention has been paid to Reinforcement Learning (RL) algorithms when optimizing long-term user engagement in sequential recommendation tasks. One challenge in large-scale online recommendation systems is the constant and complicated changes in users' behavior patterns, such as interaction rates and retention tendencies. When formulated as a Markov Decision Process (MDP), the dynamics and reward functions of the recommendation system are continuously affected by these changes. Existing RL algorithms for recommendation systems will suffer from distribution shift and struggle to adapt in such an MDP. In this paper, we introduce a novel paradigm called Adaptive Sequential Recommendation (AdaRec) to address this issue. AdaRec proposes a new distance-based representation loss to extract latent information from users' interaction trajectories. Such information reflects how RL policy fits to current user behavior patterns, and helps the policy to identify subtle changes in the recomm
    

