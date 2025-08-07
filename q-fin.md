# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluation of Deep Reinforcement Learning Algorithms for Portfolio Optimisation.](http://arxiv.org/abs/2307.07694) | 该论文评估了投资组合优化任务中的深度强化学习算法，并发现在包括市场冲击和参数变化的情况下，基于策略的算法PPO和A2C在处理噪声方面表现良好，而离策略算法DDPG、TD3和SAC则效果较差。 |

# 详细

[^1]: 评估深度强化学习算法在投资组合优化中的应用

    Evaluation of Deep Reinforcement Learning Algorithms for Portfolio Optimisation. (arXiv:2307.07694v1 [cs.CE])

    [http://arxiv.org/abs/2307.07694](http://arxiv.org/abs/2307.07694)

    该论文评估了投资组合优化任务中的深度强化学习算法，并发现在包括市场冲击和参数变化的情况下，基于策略的算法PPO和A2C在处理噪声方面表现良好，而离策略算法DDPG、TD3和SAC则效果较差。

    

    我们对投资组合优化任务中的基准深度强化学习（DRL）算法进行了评估，并使用模拟器作为评估依据。该模拟器基于相关几何布朗运动（GBM）与Bertsimas-Lo（BL）市场冲击模型。使用凯利准则（对数效用）作为目标，我们可以在没有市场冲击的情况下通过分析推导出最优策略，并将其用作包括市场冲击时性能的上限。我们发现，离策略算法DDPG、TD3和SAC由于噪声奖励的存在无法学习到正确的Q函数，因此表现不佳。而基于策略的算法PPO和A2C，在广义优势估计（GAE）的使用下能够应对噪声并得出接近最优策略。PPO的剪切变体在防止策略在收敛后偏离最优值方面发挥了重要作用。在GBM参数发生制度性变化的更具挑战性的环境中，我们发现PPO、TD3和SAC算法仍能保持较好的性能。

    We evaluate benchmark deep reinforcement learning (DRL) algorithms on the task of portfolio optimisation under a simulator. The simulator is based on correlated geometric Brownian motion (GBM) with the Bertsimas-Lo (BL) market impact model. Using the Kelly criterion (log utility) as the objective, we can analytically derive the optimal policy without market impact and use it as an upper bound to measure performance when including market impact. We found that the off-policy algorithms DDPG, TD3 and SAC were unable to learn the right Q function due to the noisy rewards and therefore perform poorly. The on-policy algorithms PPO and A2C, with the use of generalised advantage estimation (GAE), were able to deal with the noise and derive a close to optimal policy. The clipping variant of PPO was found to be important in preventing the policy from deviating from the optimal once converged. In a more challenging environment where we have regime changes in the GBM parameters, we found that PPO,
    

