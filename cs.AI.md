# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recursive Joint Simulation in Games](https://arxiv.org/abs/2402.08128) | 本文研究了游戏中AI代理之间的递归协同模拟的互动方式，并证明了这种方式与原始游戏的无限重复版本在战略上是等价的。 |
| [^2] | [A DeepLearning Framework for Dynamic Estimation of Origin-Destination Sequence.](http://arxiv.org/abs/2307.05623) | 本文提出了一个综合方法，使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化，解决了交通领域中静态和动态OD矩阵估计中的欠定和滞后挑战。 |
| [^3] | [Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization.](http://arxiv.org/abs/2303.18031) | 该论文评估了基于领域泛化的方法在开放领域泛化中的表现，证明了CORAL和MMD等简单DG方法在某些情况下的竞争力，提出了这些方法的简单扩展。 |

# 详细

[^1]: 递归协同模拟在游戏中的应用

    Recursive Joint Simulation in Games

    [https://arxiv.org/abs/2402.08128](https://arxiv.org/abs/2402.08128)

    本文研究了游戏中AI代理之间的递归协同模拟的互动方式，并证明了这种方式与原始游戏的无限重复版本在战略上是等价的。

    

    人工智能(AI)代理之间的博弈动态与传统的人-人互动可能存在各种不同之处。其中一个区别是可能能够准确地模拟AI代理，例如因为其源代码是已知的。我们的目标是探索利用这种可能性在战略设置中实现更合作的结果的方法。在本文中，我们研究了AI代理之间运行递归协同模拟的互动。即，代理首先共同观察他们所面对情境的模拟。这种模拟反过来递归地包括了额外的模拟（为了避免无限递归，具有小概率的失败），并且在选择行动之前观察所有这些嵌套模拟的结果。我们证明，由此产生的互动在战略上等价于原始游戏的无限重复版本，从而可以直接转移诸如各种民间定理等现有结果。

    Game-theoretic dynamics between AI agents could differ from traditional human-human interactions in various ways. One such difference is that it may be possible to accurately simulate an AI agent, for example because its source code is known. Our aim is to explore ways of leveraging this possibility to achieve more cooperative outcomes in strategic settings. In this paper, we study an interaction between AI agents where the agents run a recursive joint simulation. That is, the agents first jointly observe a simulation of the situation they face. This simulation in turn recursively includes additional simulations (with a small chance of failure, to avoid infinite recursion), and the results of all these nested simulations are observed before an action is chosen. We show that the resulting interaction is strategically equivalent to an infinitely repeated version of the original game, allowing a direct transfer of existing results such as the various folk theorems.
    
[^2]: 用于动态估计出发地-目的地序列的深度学习框架

    A DeepLearning Framework for Dynamic Estimation of Origin-Destination Sequence. (arXiv:2307.05623v1 [cs.LG])

    [http://arxiv.org/abs/2307.05623](http://arxiv.org/abs/2307.05623)

    本文提出了一个综合方法，使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化，解决了交通领域中静态和动态OD矩阵估计中的欠定和滞后挑战。

    

    OD矩阵估计是交通领域的一个关键问题。主要方法使用交通传感器测量信息（如交通流量）来估计由OD矩阵表示的交通需求。该问题分为静态OD矩阵估计和动态OD矩阵序列（简称OD序列）估计两类。上述两种方法面临由于大量估计参数和不足的约束信息造成的欠定问题。此外，OD序列估计还面临滞后挑战：由于拥堵等不同交通条件，相同的车辆在同一观测时段内会出现在不同的路段上，导致相同的OD需求对应不同的行程。为此，本文提出了一种综合方法，它使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化。我们的实验显示...

    OD matrix estimation is a critical problem in the transportation domain. The principle method uses the traffic sensor measured information such as traffic counts to estimate the traffic demand represented by the OD matrix. The problem is divided into two categories: static OD matrix estimation and dynamic OD matrices sequence(OD sequence for short) estimation. The above two face the underdetermination problem caused by abundant estimated parameters and insufficient constraint information. In addition, OD sequence estimation also faces the lag challenge: due to different traffic conditions such as congestion, identical vehicle will appear on different road sections during the same observation period, resulting in identical OD demands correspond to different trips. To this end, this paper proposes an integrated method, which uses deep learning methods to infer the structure of OD sequence and uses structural constraints to guide traditional numerical optimization. Our experiments show th
    
[^3]: 简单的领域泛化方法是开放领域泛化的强大基准方法

    Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization. (arXiv:2303.18031v1 [cs.CV])

    [http://arxiv.org/abs/2303.18031](http://arxiv.org/abs/2303.18031)

    该论文评估了基于领域泛化的方法在开放领域泛化中的表现，证明了CORAL和MMD等简单DG方法在某些情况下的竞争力，提出了这些方法的简单扩展。

    

    在现实世界的应用中，机器学习模型需要处理开放集识别（OSR），即在推理过程中出现未知类别，以及领域漂移（domain shift），即训练和推理阶段之间数据分布不同的情况。领域泛化（DG）旨在处理推理阶段的目标领域在模型训练期间不可访问的情况下的领域漂移情况。开放领域泛化（ODG）同时考虑了DG和OSR。领域增强元学习（DAML）是一个面向ODG的方法，但其学习过程较为复杂。另一方面，尽管提出了各种DG方法，但它们尚未在ODG情况下进行评估。本文全面评估现有的DG方法在ODG中的表现，并展示了两种简单的DG方法，即CORrelation ALignment（CORAL）和Maximum Mean Discrepancy（MMD）在若干情况下与DAML具有竞争力。此外，我们通过引入一个小调整，提出了CORAL和MMD的简单扩展。

    In real-world applications, a machine learning model is required to handle an open-set recognition (OSR), where unknown classes appear during the inference, in addition to a domain shift, where the distribution of data differs between the training and inference phases. Domain generalization (DG) aims to handle the domain shift situation where the target domain of the inference phase is inaccessible during model training. Open domain generalization (ODG) takes into account both DG and OSR. Domain-Augmented Meta-Learning (DAML) is a method targeting ODG but has a complicated learning process. On the other hand, although various DG methods have been proposed, they have not been evaluated in ODG situations. This work comprehensively evaluates existing DG methods in ODG and shows that two simple DG methods, CORrelation ALignment (CORAL) and Maximum Mean Discrepancy (MMD), are competitive with DAML in several cases. In addition, we propose simple extensions of CORAL and MMD by introducing th
    

