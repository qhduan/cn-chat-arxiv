# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond](https://arxiv.org/abs/2403.14151) | 本文综述了深度学习在轨迹数据管理与挖掘中的发展和最新进展，探讨了其在预处理、存储、分析、预测、推荐、分类、估计和检测等方面的应用。 |
| [^2] | [Factorized Tensor Networks for Multi-Task and Multi-Domain Learning.](http://arxiv.org/abs/2310.06124) | 本文提出了一种分解张量网络（FTN），它可以克服多任务多领域学习中的共享信息利用挑战，并在准确性、存储成本、计算量和样本复杂度等方面实现高效率。实验结果表明，FTN相对于现有方法需要更少的任务特定参数，并且可以适应大量的目标领域和任务。 |

# 详细

[^1]: 轨迹数据管理与挖掘的深度学习：调查与展望

    Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond

    [https://arxiv.org/abs/2403.14151](https://arxiv.org/abs/2403.14151)

    本文综述了深度学习在轨迹数据管理与挖掘中的发展和最新进展，探讨了其在预处理、存储、分析、预测、推荐、分类、估计和检测等方面的应用。

    

    arXiv:2403.14151v1 公告类型：跨越 抽象：轨迹计算是一个重要的领域，涵盖轨迹数据管理和挖掘，因其在诸如位置服务、城市交通和公共安全等各种实际应用中的关键作用而受到广泛关注。传统方法侧重于简单的时空特征，面临复杂计算、有限的可扩展性和不足以适应现实复杂性的挑战。在本文中，我们对轨迹计算中深度学习的发展和最新进展进行了全面的回顾（DL4Traj）。我们首先定义轨迹数据，并简要介绍了广泛使用的深度学习模型。系统地探讨了深度学习在轨迹管理（预处理、存储、分析和可视化）和挖掘（与轨迹相关的预测、轨迹相关的推荐、轨迹分类、旅行时间估计、异常检测）

    arXiv:2403.14151v1 Announce Type: cross  Abstract: Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety. Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in deep learning for trajectory computing (DL4Traj). We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (trajectory-related forecasting, trajectory-related recommendation, trajectory classification, travel time estimation, anomaly detecti
    
[^2]: 分解张量网络用于多任务和多领域学习

    Factorized Tensor Networks for Multi-Task and Multi-Domain Learning. (arXiv:2310.06124v1 [cs.LG])

    [http://arxiv.org/abs/2310.06124](http://arxiv.org/abs/2310.06124)

    本文提出了一种分解张量网络（FTN），它可以克服多任务多领域学习中的共享信息利用挑战，并在准确性、存储成本、计算量和样本复杂度等方面实现高效率。实验结果表明，FTN相对于现有方法需要更少的任务特定参数，并且可以适应大量的目标领域和任务。

    

    多任务和多领域学习方法旨在使用单个统一的网络共同学习多个任务/领域，或者先后学习它们。关键挑战和机会是利用任务和领域之间的共享信息，提高统一网络的效率，包括准确性、存储成本、计算量或样本复杂度。本文提出了一种分解张量网络（FTN），可以通过增加少量附加参数实现与独立单任务/领域网络相当的准确性。FTN使用源模型的冻结主干网络，并逐步添加任务/领域特定的低秩张量因子到共享的冻结网络中。这种方法可以适应大量目标领域和任务，而不会出现灾难性遗忘。此外，与现有方法相比，FTN需要较少的任务特定参数。我们在广泛使用的多领域和多任务数据集上进行了实验。

    Multi-task and multi-domain learning methods seek to learn multiple tasks/domains, jointly or one after another, using a single unified network. The key challenge and opportunity is to exploit shared information across tasks and domains to improve the efficiency of the unified network. The efficiency can be in terms of accuracy, storage cost, computation, or sample complexity. In this paper, we propose a factorized tensor network (FTN) that can achieve accuracy comparable to independent single-task/domain networks with a small number of additional parameters. FTN uses a frozen backbone network from a source model and incrementally adds task/domain-specific low-rank tensor factors to the shared frozen network. This approach can adapt to a large number of target domains and tasks without catastrophic forgetting. Furthermore, FTN requires a significantly smaller number of task-specific parameters compared to existing methods. We performed experiments on widely used multi-domain and multi-
    

