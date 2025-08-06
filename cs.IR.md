# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterophily-Aware Fair Recommendation using Graph Convolutional Networks](https://arxiv.org/abs/2402.03365) | 本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。 |

# 详细

[^1]: 利用图卷积网络的異质友善推荐方法

    Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

    [https://arxiv.org/abs/2402.03365](https://arxiv.org/abs/2402.03365)

    本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。

    

    近年来，图神经网络（GNNs）已成为提高推荐系统准确性和性能的流行工具。现代推荐系统不仅设计为为最终用户服务，还要让其他参与者（如项目和项目供应商）从中受益。这些参与者可能具有不同或冲突的目标和利益，这引发了对公平性和流行度偏差考虑的需求。基于GNN的推荐方法也面临不公平性和流行度偏差的挑战，其归一化和聚合过程受到这些挑战的影响。在本文中，我们提出了一种公平的基于GNN的推荐系统，称为HetroFair，旨在提高项目侧的公平性。HetroFair使用两个独立的组件生成具有公平性意识的嵌入：i）公平注意力，它在GNN的归一化过程中结合了点积，以减少节点度数的影响；ii）异质性特征加权，为不同的特征分配不同的权重。

    In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items' side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes' degrees, and ii) heterophily feature weighting to assign distinct weights to 
    

