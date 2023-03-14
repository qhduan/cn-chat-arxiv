# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedLP: Layer-wise Pruning Mechanism for Communication-Computation Efficient Federated Learning.](http://arxiv.org/abs/2303.06360) | 本文提出了一种显式的FL剪枝框架FedLP，采用局部训练和联邦更新中的层次剪枝，对不同类型的深度学习模型具有普适性，可以缓解通信和计算的系统瓶颈，并且性能下降较小。 |
| [^2] | [Learning Sparse Graphon Mean Field Games.](http://arxiv.org/abs/2209.03880) | 本文提出了一种新型的GMFG公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题，特别是幂律网络。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。 |
| [^3] | [Bayesian Optimization-based Combinatorial Assignment.](http://arxiv.org/abs/2208.14698) | 本文提出了一种基于贝叶斯优化的组合分配（BOCA）机制，通过将捕获模型不确定性的方法集成到迭代组合拍卖机制中，解决了组合分配领域中先前工作的主要缺点，能够更好地引导代理提供信息。 |
| [^4] | [DM$^2$: Decentralized Multi-Agent Reinforcement Learning for Distribution Matching.](http://arxiv.org/abs/2206.00233) | 本文提出了一种基于分布匹配的去中心化多智能体强化学习方法，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配，可以实现收敛到生成目标分布的联合策略。 |

# 详细

[^1]: FedLP: 一种用于通信计算高效的联邦学习的层次剪枝机制

    FedLP: Layer-wise Pruning Mechanism for Communication-Computation Efficient Federated Learning. (arXiv:2303.06360v1 [cs.LG])

    [http://arxiv.org/abs/2303.06360](http://arxiv.org/abs/2303.06360)

    本文提出了一种显式的FL剪枝框架FedLP，采用局部训练和联邦更新中的层次剪枝，对不同类型的深度学习模型具有普适性，可以缓解通信和计算的系统瓶颈，并且性能下降较小。

    This paper proposes an explicit FL pruning framework, FedLP, which adopts layer-wise pruning in local training and federated updating, and is model-agnostic and universal for different types of deep learning models. FedLP can relieve the system bottlenecks of communication and computation with marginal performance decay.

    联邦学习（FL）已经成为一种高效且隐私保护的分布式学习方案。本文主要关注FL中计算和通信的优化，采用局部训练和联邦更新中的层次剪枝，提出了一个显式的FL剪枝框架FedLP（Federated Layer-wise Pruning），该框架对不同类型的深度学习模型具有普适性。为具有同质本地模型和异质本地模型的场景设计了两种特定的FedLP方案。通过理论和实验评估，证明了FedLP可以缓解通信和计算的系统瓶颈，并且性能下降较小。据我们所知，FedLP是第一个正式将层次剪枝引入FL的框架。在联邦学习范围内，可以基于FedLP进一步设计更多的变体和组合。

    Federated learning (FL) has prevailed as an efficient and privacy-preserved scheme for distributed learning. In this work, we mainly focus on the optimization of computation and communication in FL from a view of pruning. By adopting layer-wise pruning in local training and federated updating, we formulate an explicit FL pruning framework, FedLP (Federated Layer-wise Pruning), which is model-agnostic and universal for different types of deep learning models. Two specific schemes of FedLP are designed for scenarios with homogeneous local models and heterogeneous ones. Both theoretical and experimental evaluations are developed to verify that FedLP relieves the system bottlenecks of communication and computation with marginal performance decay. To the best of our knowledge, FedLP is the first framework that formally introduces the layer-wise pruning into FL. Within the scope of federated learning, more variants and combinations can be further designed based on FedLP.
    
[^2]: 学习稀疏图形均场博弈

    Learning Sparse Graphon Mean Field Games. (arXiv:2209.03880v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2209.03880](http://arxiv.org/abs/2209.03880)

    本文提出了一种新型的GMFG公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题，特别是幂律网络。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。

    This paper proposes a novel formulation of GMFGs, called LPGMFG, which leverages the graph theoretical concept of $L^p$ graphons and provides a machine learning tool to efficiently and accurately approximate solutions for sparse network problems, especially power law networks. The paper derives theoretical existence and convergence guarantees and gives empirical examples that demonstrate the accuracy of the learning method.

    尽管多智能体强化学习（MARL）领域在过去几年中取得了相当大的进展，但解决具有大量代理的系统仍然是一个难题。图形均场博弈（GMFG）使得可以对否则难以处理的MARL问题进行可扩展的分析。由于图形的数学结构，这种方法仅限于描述许多现实世界网络（如幂律图）的稠密图形，这是不足的。我们的论文介绍了GMFG的新型公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，并提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题。这尤其包括在各种应用领域中经验观察到的幂律网络，这些网络无法被标准图形所捕捉。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。

    Although the field of multi-agent reinforcement learning (MARL) has made considerable progress in the last years, solving systems with a large number of agents remains a hard challenge. Graphon mean field games (GMFGs) enable the scalable analysis of MARL problems that are otherwise intractable. By the mathematical structure of graphons, this approach is limited to dense graphs which are insufficient to describe many real-world networks such as power law graphs. Our paper introduces a novel formulation of GMFGs, called LPGMFGs, which leverages the graph theoretical concept of $L^p$ graphons and provides a machine learning tool to efficiently and accurately approximate solutions for sparse network problems. This especially includes power law networks which are empirically observed in various application areas and cannot be captured by standard graphons. We derive theoretical existence and convergence guarantees and give empirical examples that demonstrate the accuracy of our learning ap
    
[^3]: 基于贝叶斯优化的组合分配

    Bayesian Optimization-based Combinatorial Assignment. (arXiv:2208.14698v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.14698](http://arxiv.org/abs/2208.14698)

    本文提出了一种基于贝叶斯优化的组合分配（BOCA）机制，通过将捕获模型不确定性的方法集成到迭代组合拍卖机制中，解决了组合分配领域中先前工作的主要缺点，能够更好地引导代理提供信息。

    This paper proposes a Bayesian optimization-based combinatorial assignment (BOCA) mechanism, which addresses the main shortcoming of prior work in the combinatorial assignment domain by integrating a method for capturing model uncertainty into an iterative combinatorial auction mechanism, and can better elicit information from agents.

    本文研究组合分配领域，包括组合拍卖和课程分配。该领域的主要挑战是随着物品数量的增加，捆绑空间呈指数增长。为了解决这个问题，最近有几篇论文提出了基于机器学习的偏好引导算法，旨在从代理中仅引导出最重要的信息。然而，这些先前工作的主要缺点是它们没有对尚未引导出的捆绑值的机制不确定性进行建模。本文通过提出一种基于贝叶斯优化的组合分配（BOCA）机制来解决这个缺点。我们的关键技术贡献是将捕获模型不确定性的方法集成到迭代组合拍卖机制中。具体而言，我们设计了一种新的方法来估计可用于定义获取函数以确定下一个查询的上限不确定性界限。这使得机制能够

    We study the combinatorial assignment domain, which includes combinatorial auctions and course allocation. The main challenge in this domain is that the bundle space grows exponentially in the number of items. To address this, several papers have recently proposed machine learning-based preference elicitation algorithms that aim to elicit only the most important information from agents. However, the main shortcoming of this prior work is that it does not model a mechanism's uncertainty over values for not yet elicited bundles. In this paper, we address this shortcoming by presenting a Bayesian optimization-based combinatorial assignment (BOCA) mechanism. Our key technical contribution is to integrate a method for capturing model uncertainty into an iterative combinatorial auction mechanism. Concretely, we design a new method for estimating an upper uncertainty bound that can be used to define an acquisition function to determine the next query to the agents. This enables the mechanism 
    
[^4]: DM$^2$: 基于分布匹配的去中心化多智能体强化学习

    DM$^2$: Decentralized Multi-Agent Reinforcement Learning for Distribution Matching. (arXiv:2206.00233v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2206.00233](http://arxiv.org/abs/2206.00233)

    本文提出了一种基于分布匹配的去中心化多智能体强化学习方法，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配，可以实现收敛到生成目标分布的联合策略。

    This paper proposes a decentralized multi-agent reinforcement learning method based on distribution matching, where each agent independently minimizes the distribution mismatch to the corresponding component of a target visitation distribution, achieving convergence to the joint policy that generated the target distribution.

    当前的多智能体协作方法往往依赖于集中式机制或显式通信协议以确保收敛。本文研究了分布匹配在不依赖于集中式组件或显式通信的分布式多智能体学习中的应用。在所提出的方案中，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配。理论分析表明，在某些条件下，每个智能体最小化其个体分布不匹配可以实现收敛到生成目标分布的联合策略。此外，如果目标分布来自优化合作任务的联合策略，则该任务奖励和分布匹配奖励的组合的最优策略是相同的联合策略。这一见解被用来制定一个实用的算法。

    Current approaches to multi-agent cooperation rely heavily on centralized mechanisms or explicit communication protocols to ensure convergence. This paper studies the problem of distributed multi-agent learning without resorting to centralized components or explicit communication. It examines the use of distribution matching to facilitate the coordination of independent agents. In the proposed scheme, each agent independently minimizes the distribution mismatch to the corresponding component of a target visitation distribution. The theoretical analysis shows that under certain conditions, each agent minimizing its individual distribution mismatch allows the convergence to the joint policy that generated the target distribution. Further, if the target distribution is from a joint policy that optimizes a cooperative task, the optimal policy for a combination of this task reward and the distribution matching reward is the same joint policy. This insight is used to formulate a practical al
    

