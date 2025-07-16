# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FairTargetSim: An Interactive Simulator for Understanding and Explaining the Fairness Effects of Target Variable Definition](https://arxiv.org/abs/2403.06031) | FairTargetSim提供了一个交互式模拟器，展示了目标变量定义对公平性的影响，适用于算法开发者、研究人员和非技术利益相关者。 |
| [^2] | [Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption.](http://arxiv.org/abs/2401.13444) | 该论文介绍了一种以低计算资源消耗为中心的高效知识库问答框架，通过引入线索引导路径探索的方式，将知识库与大型语言模型高效地融合，从而降低了对模型能力的要求，并在实验证明了其优越性能。 |
| [^3] | [VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models.](http://arxiv.org/abs/2310.13367) | VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。 |
| [^4] | [GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks.](http://arxiv.org/abs/2310.03399) | GRAPES是一种自适应图采样方法，通过学习识别在训练图神经网络分类器时具有影响力的节点集合，解决了可扩展图神经网络中的内存问题，并且在准确性和可扩展性方面表现出色。 |

# 详细

[^1]: FairTargetSim：用于理解和解释目标变量定义公平性影响的交互式模拟器

    FairTargetSim: An Interactive Simulator for Understanding and Explaining the Fairness Effects of Target Variable Definition

    [https://arxiv.org/abs/2403.06031](https://arxiv.org/abs/2403.06031)

    FairTargetSim提供了一个交互式模拟器，展示了目标变量定义对公平性的影响，适用于算法开发者、研究人员和非技术利益相关者。

    

    机器学习需要为预测或决策定义目标变量，这个过程可能对公平性产生深远影响：偏见通常已经被编码在目标变量定义本身中，而不是在任何数据收集或训练之前。我们提出了一个交互式模拟器，FairTargetSim (FTS)，展示了目标变量定义如何影响公平性。FTS是一个有价值的工具，适用于算法开发者、研究人员和非技术利益相关者。FTS使用了算法招聘的案例研究，使用真实世界数据和用户定义的目标变量。FTS是开源的，可在以下网址找到：http://tinyurl.com/ftsinterface。本文附带的视频网址为：http://tinyurl.com/ijcaifts。

    arXiv:2403.06031v1 Announce Type: cross  Abstract: Machine learning requires defining one's target variable for predictions or decisions, a process that can have profound implications on fairness: biases are often encoded in target variable definition itself, before any data collection or training. We present an interactive simulator, FairTargetSim (FTS), that illustrates how target variable definition impacts fairness. FTS is a valuable tool for algorithm developers, researchers, and non-technical stakeholders. FTS uses a case study of algorithmic hiring, using real-world data and user-defined target variables. FTS is open-source and available at: http://tinyurl.com/ftsinterface. The video accompanying this paper is here: http://tinyurl.com/ijcaifts.
    
[^2]: 以低计算资源消耗为中心的高效知识库问答框架：基于线索引导路径探索

    Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption. (arXiv:2401.13444v1 [cs.CL])

    [http://arxiv.org/abs/2401.13444](http://arxiv.org/abs/2401.13444)

    该论文介绍了一种以低计算资源消耗为中心的高效知识库问答框架，通过引入线索引导路径探索的方式，将知识库与大型语言模型高效地融合，从而降低了对模型能力的要求，并在实验证明了其优越性能。

    

    在最近的研究中，大型语言模型（LLMs）展示了出色的能力。然而，更新它们的知识面会带来挑战，当面对不熟悉的查询时可能导致不准确性。虽然已经研究了将知识图谱与LLMs集成的方法，但现有方法将LLMs视为主要的决策者，对其能力提出了较高的要求。对于计算成本较低且性能相对较差的LLMs来说，这是不太合适的。本文介绍了一种以线索引导路径探索为核心的知识库问答框架（CGPE），它将知识库与LLMs高效地融合，对模型的能力要求较低。受人类手动检索知识的方法启发，CGPE利用问题中的信息作为线索，系统地探索知识库中所需的知识路径。开源数据集上的实验证明，CGPE优于先前的方法，并且非常适用于计算成本较低且性能较差的LLMs。

    In recent times, large language models (LLMs) have showcased remarkable capabilities. However, updating their knowledge poses challenges, potentially leading to inaccuracies when confronted with unfamiliar queries. While integrating knowledge graphs with LLMs has been explored, existing approaches treat LLMs as primary decision-makers, imposing high demands on their capabilities. This is particularly unsuitable for LLMs with lower computational costs and relatively poorer performance. In this paper, we introduce a Clue-Guided Path Exploration framework (CGPE) that efficiently merges a knowledge base with an LLM, placing less stringent requirements on the model's capabilities. Inspired by the method humans use to manually retrieve knowledge, CGPE employs information from the question as clues to systematically explore the required knowledge path within the knowledge base. Experiments on open-source datasets reveal that CGPE outperforms previous methods and is highly applicable to LLMs w
    
[^3]: VFedMH: 垂直联合学习用于训练多参与方异构模型

    VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models. (arXiv:2310.13367v1 [cs.LG])

    [http://arxiv.org/abs/2310.13367](http://arxiv.org/abs/2310.13367)

    VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。

    

    垂直联合学习（VFL）作为一种集成样本对齐和特征合并的新型训练范式，已经引起了越来越多的关注。然而，现有的VFL方法在处理参与者之间存在异构本地模型时面临挑战，这影响了优化收敛性和泛化能力。为了解决这个问题，本文提出了一种名为VFedMH的新方法，用于训练多方异构模型。VFedMH的重点是在前向传播期间聚合每个参与者知识的嵌入，而不是中间结果。主动方，拥有样本的标签和特征，在VFedMH中安全地聚合本地嵌入以获得全局知识嵌入，并将其发送给被动方。被动方仅拥有样本的特征，然后利用全局嵌入在其本地异构网络上进行前向传播。然而，被动方不拥有标签。

    Vertical Federated Learning (VFL) has gained increasing attention as a novel training paradigm that integrates sample alignment and feature union. However, existing VFL methods face challenges when dealing with heterogeneous local models among participants, which affects optimization convergence and generalization. To address this issue, this paper proposes a novel approach called Vertical Federated learning for training Multi-parties Heterogeneous models (VFedMH). VFedMH focuses on aggregating the embeddings of each participant's knowledge instead of intermediate results during forward propagation. The active party, who possesses labels and features of the sample, in VFedMH securely aggregates local embeddings to obtain global knowledge embeddings, and sends them to passive parties. The passive parties, who own only features of the sample, then utilize the global embeddings to propagate forward on their local heterogeneous networks. However, the passive party does not own the labels, 
    
[^4]: GRAPES: 学习用于可扩展图神经网络的图采样

    GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks. (arXiv:2310.03399v1 [cs.LG])

    [http://arxiv.org/abs/2310.03399](http://arxiv.org/abs/2310.03399)

    GRAPES是一种自适应图采样方法，通过学习识别在训练图神经网络分类器时具有影响力的节点集合，解决了可扩展图神经网络中的内存问题，并且在准确性和可扩展性方面表现出色。

    

    图神经网络（GNNs）通过以不同方式聚合周围信息来学习图中节点的表示。随着这些网络的加深，由于邻域尺寸的增加，它们的感受野呈指数增长，导致高内存消耗。图采样通过对图中节点进行抽样来解决GNNs中的内存问题。通过这种方式，GNNs可以扩展到更大的图。大多数采样方法专注于固定的采样启发式算法，这可能无法推广到不同的结构或任务。我们引入了GRAPES，一种自适应的图采样方法，该方法学习识别用于训练GNN分类器的一组具有影响力的节点。GRAPES使用GFlowNet来学习给定分类目标的节点采样概率。我们在几个小规模和大规模图基准上评估了GRAPES，并展示了其在准确性和可扩展性方面的有效性。与现有的采样方法相比，GRAPES即使在采样比例较低的情况下仍保持高准确性。

    Graph neural networks (GNNs) learn the representation of nodes in a graph by aggregating the neighborhood information in various ways. As these networks grow in depth, their receptive field grows exponentially due to the increase in neighborhood sizes, resulting in high memory costs. Graph sampling solves memory issues in GNNs by sampling a small ratio of the nodes in the graph. This way, GNNs can scale to much larger graphs. Most sampling methods focus on fixed sampling heuristics, which may not generalize to different structures or tasks. We introduce GRAPES, an adaptive graph sampling method that learns to identify sets of influential nodes for training a GNN classifier. GRAPES uses a GFlowNet to learn node sampling probabilities given the classification objectives. We evaluate GRAPES across several small- and large-scale graph benchmarks and demonstrate its effectiveness in accuracy and scalability. In contrast to existing sampling methods, GRAPES maintains high accuracy even with 
    

