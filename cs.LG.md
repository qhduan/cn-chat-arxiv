# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite-Time Error Analysis of Online Model-Based Q-Learning with a Relaxed Sampling Model](https://arxiv.org/abs/2402.11877) | 本文通过有限时间分析以及实证评估，探讨了集成模型方法的Q学习在样本复杂度方面的优势。 |
| [^2] | [Unichain and Aperiodicity are Sufficient for Asymptotic Optimality of Average-Reward Restless Bandits](https://arxiv.org/abs/2402.05689) | 该论文提出了一种新的策略类别，用于解决无限期平均回报好转胆冒险问题。研究表明，在单臂松弛问题是Unichain和非周期性的情况下，该策略类别具有渐进最优性。 |
| [^3] | [Deep Neural Networks: A Formulation Via Non-Archimedean Analysis](https://arxiv.org/abs/2402.00094) | 该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。 |
| [^4] | [Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks.](http://arxiv.org/abs/2307.07753) | 本文提出了一种用于神经网络的先验学习方法，通过利用可扩展和结构化的神经网络后验作为推广的信息先验，提高了神经网络的推广和不确定性估计能力。我们的方法在大规模上提供了表达性的概率表示，并产生了非空推广界限。我们的技术贡献是推导出可处理的目标函数，并提出了改进的推广界限计算方法。在经验上，我们证明了该方法在不确定性估计和推广方面的有效性。 |
| [^5] | [Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning.](http://arxiv.org/abs/2306.05494) | 本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。 |
| [^6] | [Convergence of the Inexact Langevin Algorithm and Score-based Generative Models in KL Divergence.](http://arxiv.org/abs/2211.01512) | 本文研究了不精确 Langevin 算法与基于得分的生成模型在 KL 散度中的收敛性，提出了建立稳定偏差收敛保证的两个关键假设：目标分布满足对数 Sobolev 不等式和分数估计器展示出有界的矩阵生成函数误差。作者探讨了如何获得可靠的分数估计器，并证明了基于核密度估计的简单估计器满足假设。 |

# 详细

[^1]: 在具有放松采样模型的在线模型的有限时间误差分析下的Q学习

    Finite-Time Error Analysis of Online Model-Based Q-Learning with a Relaxed Sampling Model

    [https://arxiv.org/abs/2402.11877](https://arxiv.org/abs/2402.11877)

    本文通过有限时间分析以及实证评估，探讨了集成模型方法的Q学习在样本复杂度方面的优势。

    

    强化学习在模型为基础的方法的出现下取得了显著进展。在这些方法中，Q学习在无模型设置中被证明是一种强大的算法。然而，将Q学习扩展到基于模型的框架仍然相对未被探索。在本文中，我们深入研究了Q学习与基于模型方法相结合时的样本复杂度。通过理论分析和实证评估，我们试图阐明在哪些条件下，基于模型的Q学习在样本效率方面优于其无模型对应物。

    arXiv:2402.11877v1 Announce Type: cross  Abstract: Reinforcement learning has witnessed significant advancements, particularly with the emergence of model-based approaches. Among these, $Q$-learning has proven to be a powerful algorithm in model-free settings. However, the extension of $Q$-learning to a model-based framework remains relatively unexplored. In this paper, we delve into the sample complexity of $Q$-learning when integrated with a model-based approach. Through theoretical analyses and empirical evaluations, we seek to elucidate the conditions under which model-based $Q$-learning excels in terms of sample efficiency compared to its model-free counterpart.
    
[^2]: Unichain和非周期性足以保证平均回报好转胆冒险目标的渐进最优性

    Unichain and Aperiodicity are Sufficient for Asymptotic Optimality of Average-Reward Restless Bandits

    [https://arxiv.org/abs/2402.05689](https://arxiv.org/abs/2402.05689)

    该论文提出了一种新的策略类别，用于解决无限期平均回报好转胆冒险问题。研究表明，在单臂松弛问题是Unichain和非周期性的情况下，该策略类别具有渐进最优性。

    

    我们考虑了离散时间下的无限期平均回报的好转胆冒险问题。我们提出了一种新的策略类别，旨在将逐渐扩大的臂子集向最佳分布方向推进。我们证明了我们的策略在N臂问题中是渐进最优的，如果单臂松弛问题是Unichain和非周期性的，那么就会有一个$O(1/\sqrt{N})$的最优间隙。我们的方法不同于大多数现有的研究，这些研究侧重于指数或优先级策略，这些策略依赖于统一全局吸引子属性（UGAP）来保证收敛到最优，或者依赖于最近开发的基于模拟的策略，该策略要求遵循同步假设（SA）。

    We consider the infinite-horizon, average-reward restless bandit problem in discrete time. We propose a new class of policies that are designed to drive a progressively larger subset of arms toward the optimal distribution. We show that our policies are asymptotically optimal with an $O(1/\sqrt{N})$ optimality gap for an $N$-armed problem, provided that the single-armed relaxed problem is unichain and aperiodic. Our approach departs from most existing work that focuses on index or priority policies, which rely on the Uniform Global Attractor Property (UGAP) to guarantee convergence to the optimum, or a recently developed simulation-based policy, which requires a Synchronization Assumption (SA).
    
[^3]: 深度神经网络: 非阿基米德分析的表述方式

    Deep Neural Networks: A Formulation Via Non-Archimedean Analysis

    [https://arxiv.org/abs/2402.00094](https://arxiv.org/abs/2402.00094)

    该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。

    

    我们引入了一种新的深度神经网络（DNNs），采用多层树状结构的架构。这些架构使用非阿基米德局部域的整数环中的数字进行编码。这些环具有自然的层次结构，类似无限根树。这些环上的自然态射使我们能够构建有限的多层架构。新的DNNs是对在所提到的环上定义的实值函数的稳健的普遍逼近器。我们还证明了DNNs也是对在单位区间上定义的实值平方可积函数的稳健的普遍逼近器。

    We introduce a new class of deep neural networks (DNNs) with multilayered tree-like architectures. The architectures are codified using numbers from the ring of integers of non-Archimdean local fields. These rings have a natural hierarchical organization as infinite rooted trees. Natural morphisms on these rings allow us to construct finite multilayered architectures. The new DNNs are robust universal approximators of real-valued functions defined on the mentioned rings. We also show that the DNNs are robust universal approximators of real-valued square-integrable functions defined in the unit interval.
    
[^4]: 学习神经网络中的表达性先验，提高推广和不确定性估计

    Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks. (arXiv:2307.07753v1 [cs.LG])

    [http://arxiv.org/abs/2307.07753](http://arxiv.org/abs/2307.07753)

    本文提出了一种用于神经网络的先验学习方法，通过利用可扩展和结构化的神经网络后验作为推广的信息先验，提高了神经网络的推广和不确定性估计能力。我们的方法在大规模上提供了表达性的概率表示，并产生了非空推广界限。我们的技术贡献是推导出可处理的目标函数，并提出了改进的推广界限计算方法。在经验上，我们证明了该方法在不确定性估计和推广方面的有效性。

    

    在这项工作中，我们提出了一种新的先验学习方法，用于提高深度神经网络中的推广和不确定性估计。关键思想是利用可扩展和结构化的神经网络后验作为具有推广保证的信息先验。我们学习到的先验在大规模上提供了表达性的概率表示，类似于在ImageNet上预训练模型的贝叶斯对应物，并进一步产生了非空推广界限。我们还将这个想法扩展到连续学习框架中，我们的先验的有利特性是可取的。主要的推动因素是我们的技术贡献：(1) Kronecker积求和的计算，(2) 推导和优化可处理的目标函数，从而导致改进的推广界限。在经验上，我们详尽地展示了该方法在不确定性估计和推广方面的有效性。

    In this work, we propose a novel prior learning method for advancing generalization and uncertainty estimation in deep neural networks. The key idea is to exploit scalable and structured posteriors of neural networks as informative priors with generalization guarantees. Our learned priors provide expressive probabilistic representations at large scale, like Bayesian counterparts of pre-trained models on ImageNet, and further produce non-vacuous generalization bounds. We also extend this idea to a continual learning framework, where the favorable properties of our priors are desirable. Major enablers are our technical contributions: (1) the sums-of-Kronecker-product computations, and (2) the derivations and optimizations of tractable objectives that lead to improved generalization bounds. Empirically, we exhaustively show the effectiveness of this method for uncertainty estimation and generalization.
    
[^5]: 神经网络中对抗性漏洞攻击的实用性测试：动态学习的影响

    Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning. (arXiv:2306.05494v1 [cs.CR])

    [http://arxiv.org/abs/2306.05494](http://arxiv.org/abs/2306.05494)

    本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。

    

    机器学习被广泛应用于网络入侵检测系统(NIDS)中，由于其自动化的特性和在处理和分类大量数据上的高精度。但机器学习存在缺陷，其中最大的问题之一是对抗性攻击，其目的是使机器学习模型产生错误的预测。本文提出了两个独特的贡献：对抗性攻击对基于机器学习的NIDS实用性问题的分类和对持续训练对NIDS对抗性攻击的影响进行了研究。我们的实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。虽然对抗性攻击可能会危及基于机器学习的NIDS，但持续再训练可带来一定的缓解效果。

    Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy in processing and classifying large volumes of data. However, ML has been found to have several flaws, on top of them are adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the practicality of such attacks against ML-based network security entities, especially NIDS.  This paper presents two distinct contributions: a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS and an investigation of the impact of continuous training on adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effect of adversarial attacks. While adversarial attacks can harm ML-based NIDSs, ou
    
[^6]: 不精确 Langevin 算法与基于得分的生成模型在 KL 散度中的收敛性

    Convergence of the Inexact Langevin Algorithm and Score-based Generative Models in KL Divergence. (arXiv:2211.01512v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01512](http://arxiv.org/abs/2211.01512)

    本文研究了不精确 Langevin 算法与基于得分的生成模型在 KL 散度中的收敛性，提出了建立稳定偏差收敛保证的两个关键假设：目标分布满足对数 Sobolev 不等式和分数估计器展示出有界的矩阵生成函数误差。作者探讨了如何获得可靠的分数估计器，并证明了基于核密度估计的简单估计器满足假设。

    

    本文研究了不精确 Langevin 动力学（ILD）、不精确 Langevin 算法（ILA）和基于得分的生成建模（SGM）在利用估计得分函数进行采样时的情况。我们的重点在于建立关于 Kullback-Leibler（KL）散度的稳定偏差收敛保证。为了实现这些保证，我们采用了两个关键假设：1）目标分布满足对数 Sobolev 不等式（LSI），2）分数估计器展示出一个有界的矩阵生成函数（MGF）误差。值得注意的是，我们采用的 MGF 误差假设相比现有文献中使用的 $L^\infty$ 误差假设更为宽松。然而，它比最近的作品中使用的 $L^2$ 误差假设更强，后者常常导致不稳定的边界。我们探讨了如何获得满足 MGF 误差假设的可靠分数估计器的问题。具体来说，我们证明了一种基于核密度估计的简单估计器满足 MGF 误差假设。

    We study the Inexact Langevin Dynamics (ILD), Inexact Langevin Algorithm (ILA), and Score-based Generative Modeling (SGM) when utilizing estimated score functions for sampling. Our focus lies in establishing stable biased convergence guarantees in terms of the Kullback-Leibler (KL) divergence. To achieve these guarantees, we impose two key assumptions: 1) the target distribution satisfies the log-Sobolev inequality (LSI), and 2) the score estimator exhibits a bounded Moment Generating Function (MGF) error. Notably, the MGF error assumption we adopt is more lenient compared to the $L^\infty$ error assumption used in existing literature. However, it is stronger than the $L^2$ error assumption utilized in recent works, which often leads to unstable bounds. We explore the question of how to obtain a provably accurate score estimator that satisfies the MGF error assumption. Specifically, we demonstrate that a simple estimator based on kernel density estimation fulfills the MGF error assumpt
    

