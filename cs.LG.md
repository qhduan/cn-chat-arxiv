# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Filter for Real-world Graph Clustering](https://arxiv.org/abs/2403.03666) | 提出了一种可证实的方案，针对实际世界图聚类问题，在处理同源和异源图时表现出色，并构建了低通和高通滤波器来捕捉全面信息。 |
| [^2] | [A Survey on Decentralized Federated Learning.](http://arxiv.org/abs/2308.04604) | 最近几年，联邦学习成为训练分布式、大规模、保护隐私的机器学习系统的流行范式。然而，其中一个关键挑战是克服集中式编排的单点故障问题。 |

# 详细

[^1]: 可证滤波器用于现实世界图聚类

    Provable Filter for Real-world Graph Clustering

    [https://arxiv.org/abs/2403.03666](https://arxiv.org/abs/2403.03666)

    提出了一种可证实的方案，针对实际世界图聚类问题，在处理同源和异源图时表现出色，并构建了低通和高通滤波器来捕捉全面信息。

    

    图聚类是一个重要的无监督问题，已经被证明对图神经网络（GNNs）的进展更具抵抗力。此外，几乎所有的聚类方法都专注于同源图，忽略异源性。这严重限制了它们在实践中的适用性，因为现实世界图展现出结构不一致，不能简单地被归类为同源性和异源性。因此，迫切需要一种处理实际图的原则性方法。为了填补这一空白，我们提供了一个具有理论支持的新颖解决方案。有趣的是，我们发现大多数同源和异源边可以基于邻居信息被正确识别。受到这一发现的启发，我们构建了两个分别高度同源和异源的图。它们用于构建低通和高通滤波器以捕捉整体信息。重要的特征进一步由挤压-激励块增强。

    arXiv:2403.03666v1 Announce Type: new  Abstract: Graph clustering, an important unsupervised problem, has been shown to be more resistant to advances in Graph Neural Networks (GNNs). In addition, almost all clustering methods focus on homophilic graphs and ignore heterophily. This significantly limits their applicability in practice, since real-world graphs exhibit a structural disparity and cannot simply be classified as homophily and heterophily. Thus, a principled way to handle practical graphs is urgently needed. To fill this gap, we provide a novel solution with theoretical support. Interestingly, we find that most homophilic and heterophilic edges can be correctly identified on the basis of neighbor information. Motivated by this finding, we construct two graphs that are highly homophilic and heterophilic, respectively. They are used to build low-pass and high-pass filters to capture holistic information. Important features are further enhanced by the squeeze-and-excitation block
    
[^2]: 分散式联邦学习综述

    A Survey on Decentralized Federated Learning. (arXiv:2308.04604v1 [cs.LG])

    [http://arxiv.org/abs/2308.04604](http://arxiv.org/abs/2308.04604)

    最近几年，联邦学习成为训练分布式、大规模、保护隐私的机器学习系统的流行范式。然而，其中一个关键挑战是克服集中式编排的单点故障问题。

    

    最近几年，联邦学习（FL）已经成为训练分布式、大规模、保护隐私的机器学习（ML）系统的流行范式。与标准ML不同，需要将数据收集在训练执行的确切位置，FL利用数百万边缘设备的计算能力来协同训练共享的全局模型，同时不会披露其本地私有数据。在典型的FL系统中，中央服务器只充当协调器的角色；它迭代地收集和汇总每个客户端在自己的私有数据上训练的本地模型，直到收敛。尽管FL在设计上具有许多优点（例如通过设计保护私有数据所有权），但也存在一些弱点。其中最关键的挑战之一是克服经典FL客户端-服务器架构的集中式编排，这被认为是易受单点故障攻击的。

    In recent years, federated learning (FL) has become a very popular paradigm for training distributed, large-scale, and privacy-preserving machine learning (ML) systems. In contrast to standard ML, where data must be collected at the exact location where training is performed, FL takes advantage of the computational capabilities of millions of edge devices to collaboratively train a shared, global model without disclosing their local private data. Specifically, in a typical FL system, the central server acts only as an orchestrator; it iteratively gathers and aggregates all the local models trained by each client on its private data until convergence. Although FL undoubtedly has several benefits over traditional ML (e.g., it protects private data ownership by design), it suffers from several weaknesses. One of the most critical challenges is to overcome the centralized orchestration of the classical FL client-server architecture, which is known to be vulnerable to single-point-of-failur
    

