# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterophily-Aware Fair Recommendation using Graph Convolutional Networks](https://arxiv.org/abs/2402.03365) | 本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。 |
| [^2] | [Backdoor to the Hidden Ground State: Planted Vertex Cover Example.](http://arxiv.org/abs/2305.06610) | 本论文发现了在正则随机图中存在一种新类型的自由能重构，称为eureka点，通过eureka点可以轻易访问具有消失自由能屏障的隐藏基态。 |

# 详细

[^1]: 利用图卷积网络的異质友善推荐方法

    Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

    [https://arxiv.org/abs/2402.03365](https://arxiv.org/abs/2402.03365)

    本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。

    

    近年来，图神经网络（GNNs）已成为提高推荐系统准确性和性能的流行工具。现代推荐系统不仅设计为为最终用户服务，还要让其他参与者（如项目和项目供应商）从中受益。这些参与者可能具有不同或冲突的目标和利益，这引发了对公平性和流行度偏差考虑的需求。基于GNN的推荐方法也面临不公平性和流行度偏差的挑战，其归一化和聚合过程受到这些挑战的影响。在本文中，我们提出了一种公平的基于GNN的推荐系统，称为HetroFair，旨在提高项目侧的公平性。HetroFair使用两个独立的组件生成具有公平性意识的嵌入：i）公平注意力，它在GNN的归一化过程中结合了点积，以减少节点度数的影响；ii）异质性特征加权，为不同的特征分配不同的权重。

    In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items' side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes' degrees, and ii) heterophily feature weighting to assign distinct weights to 
    
[^2]: 针对正则随机图的种植顶点覆盖问题及其发现的自由能重构

    Backdoor to the Hidden Ground State: Planted Vertex Cover Example. (arXiv:2305.06610v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2305.06610](http://arxiv.org/abs/2305.06610)

    本论文发现了在正则随机图中存在一种新类型的自由能重构，称为eureka点，通过eureka点可以轻易访问具有消失自由能屏障的隐藏基态。

    

    我们引入了一个针对正则随机图的种植顶点覆盖问题，并通过空穴方法对其进行研究。此二元自旋交互作用系统的平衡序相变具有不连续的性质，不同于常规的类似伊辛模型的连续相变，并且在广泛的自由能屏障的动态阻塞下。我们发现，该系统的无序对称相在除了唯一的eureka点$\beta_b$之外的所有逆温度下都可以在有序相的情况下局部稳定。 eureka点$\beta_b$为访问具有消失自由能屏障的隐藏基态提供了一个便道。它存在于无限系列的种植随机图集合中，并且我们通过分析确定了它们的结构参数。揭示出的新类型的自由能景观也可能存在于统计物理学和统计学界面的其他种植随机图优化问题中。

    We introduce a planted vertex cover problem on regular random graphs and study it by the cavity method. The equilibrium ordering phase transition of this binary-spin two-body interaction system is discontinuous in nature distinct from the continuous one of conventional Ising-like models, and it is dynamically blocked by an extensive free energy barrier. We discover that the disordered symmetric phase of this system may be locally stable with respect to the ordered phase at all inverse temperatures except for a unique eureka point $\beta_b$ at which it is only marginally stable. The eureka point $\beta_b$ serves as a backdoor to access the hidden ground state with vanishing free energy barrier. It exists in an infinite series of planted random graph ensembles and we determine their structural parameters analytically. The revealed new type of free energy landscape may also exist in other planted random-graph optimization problems at the interface of statistical physics and statistical in
    

