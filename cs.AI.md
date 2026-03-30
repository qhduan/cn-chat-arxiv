# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Error Estimation for Physics-informed Neural Networks Approximating Semilinear Wave Equations](https://arxiv.org/abs/2402.07153) | 本文提供了物理信息神经网络逼近半线性波动方程的严格误差界限，包括泛化误差和训练误差的界限，并在数值实验中展示了理论界限的有效性。 |
| [^2] | [Scale-Adaptive Balancing of Exploration and Exploitation in Classical Planning.](http://arxiv.org/abs/2305.09840) | 本文提出了一种MCTS/THTS算法GreedyUCT-Normal，该算法能够通过采用奖励变化的尺度处理不同尺度的分布，以在经典计划中平衡探索和开发。 |

# 详细

[^1]: 物理信息神经网络逼近半线性波动方程的误差估计

    Error Estimation for Physics-informed Neural Networks Approximating Semilinear Wave Equations

    [https://arxiv.org/abs/2402.07153](https://arxiv.org/abs/2402.07153)

    本文提供了物理信息神经网络逼近半线性波动方程的严格误差界限，包括泛化误差和训练误差的界限，并在数值实验中展示了理论界限的有效性。

    

    本文对物理信息神经网络逼近半线性波动方程提供了严格的误差界限。我们针对具有两个隐藏层的tanh神经网络，基于网络层宽度和训练点数量，提供了对泛化误差和训练误差的界限。我们的主要结果是在一些假设下，将总误差以$H^1([0,T];L^2(\Omega))$-范数的形式表示，并能够随着训练点数量的增加而任意减小。我们通过数值实验验证了我们的理论界限。

    This paper provides rigorous error bounds for physics-informed neural networks approximating the semilinear wave equation. We provide bounds for the generalization and training error in terms of the width of the network's layers and the number of training points for a tanh neural network with two hidden layers. Our main result is a bound of the total error in the $H^1([0,T];L^2(\Omega))$-norm in terms of the training error and the number of training points, which can be made arbitrarily small under some assumptions. We illustrate our theoretical bounds with numerical experiments.
    
[^2]: 经典规划中探索和开发的自适应平衡

    Scale-Adaptive Balancing of Exploration and Exploitation in Classical Planning. (arXiv:2305.09840v1 [cs.AI])

    [http://arxiv.org/abs/2305.09840](http://arxiv.org/abs/2305.09840)

    本文提出了一种MCTS/THTS算法GreedyUCT-Normal，该算法能够通过采用奖励变化的尺度处理不同尺度的分布，以在经典计划中平衡探索和开发。

    

    在游戏树搜索和自动化规划中，平衡探索和开发一直是一个重要的问题。然而，虽然这个问题在多臂赌博机（MAB）文献中已经被广泛分析，但规划社区在试图应用这些结果时取得的成功有限。我们展示了MAB文献更详细的理论理解有助于改进基于蒙特卡罗树搜索（MCTS）/基于试验的启发式树搜索（THTS）的现有规划算法。具体而言，THTS在一种临时方法中使用UCB1 MAB算法，因为在启发式搜索中UCB1理论上需要有界支持奖励分布的要求在经典规划中不被满足。核心问题在于UCB1缺乏对不同奖励尺度的自适应。我们提出了GreedyUCT-Normal，这是一种具有UCB1-Normal赌博机的MCTS/THTS算法，用于敏捷经典计划，它通过采用奖励变化的尺度处理不同尺度的分布。

    Balancing exploration and exploitation has been an important problem in both game tree search and automated planning. However, while the problem has been extensively analyzed within the Multi-Armed Bandit (MAB) literature, the planning community has had limited success when attempting to apply those results. We show that a more detailed theoretical understanding of MAB literature helps improve existing planning algorithms that are based on Monte Carlo Tree Search (MCTS) / Trial Based Heuristic Tree Search (THTS). In particular, THTS uses UCB1 MAB algorithms in an ad hoc manner, as UCB1's theoretical requirement of fixed bounded support reward distributions is not satisfied within heuristic search for classical planning. The core issue lies in UCB1's lack of adaptations to the different scales of the rewards. We propose GreedyUCT-Normal, a MCTS/THTS algorithm with UCB1-Normal bandit for agile classical planning, which handles distributions with different scales by taking the reward vari
    

