# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Evaluation of Generative Models in Distributed Learning Tasks.](http://arxiv.org/abs/2310.11714) | 本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。通过研究Fr\'echet inception距离（FID），并考虑不同聚合分数，发现FID-all和FID-avg分数的模型排名可能不一致。 |
| [^2] | [NervePool: A Simplicial Pooling Layer.](http://arxiv.org/abs/2305.06315) | 单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。 |

# 详细

[^1]: 在分布式学习任务中评估生成模型

    On the Evaluation of Generative Models in Distributed Learning Tasks. (arXiv:2310.11714v1 [cs.LG])

    [http://arxiv.org/abs/2310.11714](http://arxiv.org/abs/2310.11714)

    本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。通过研究Fr\'echet inception距离（FID），并考虑不同聚合分数，发现FID-all和FID-avg分数的模型排名可能不一致。

    

    在文献中已经广泛研究了对包括生成对抗网络（GAN）和扩散模型在内的深度生成模型的评估。然而，现有的评估方法主要针对单个客户端存储的训练数据的集中式学习问题，而生成模型的许多应用涉及到分布式学习环境，例如联邦学习场景，其中训练数据由多个客户端收集并分发。本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。首先，我们关注Fr\'echet inception距离（FID），并考虑以下基于FID的聚合分数：1）FID-avg作为客户端个体FID分数的平均值，2）FID-all作为训练模型与包含所有客户端数据的集体数据集之间的FID距离。我们证明了根据FID-all和FID-avg分数的模型排名可能不一致。

    The evaluation of deep generative models including generative adversarial networks (GANs) and diffusion models has been extensively studied in the literature. While the existing evaluation methods mainly target a centralized learning problem with training data stored by a single client, many applications of generative models concern distributed learning settings, e.g. the federated learning scenario, where training data are collected by and distributed among several clients. In this paper, we study the evaluation of generative models in distributed learning tasks with heterogeneous data distributions. First, we focus on the Fr\'echet inception distance (FID) and consider the following FID-based aggregate scores over the clients: 1) FID-avg as the mean of clients' individual FID scores, 2) FID-all as the FID distance of the trained model to the collective dataset containing all clients' data. We prove that the model rankings according to the FID-all and FID-avg scores could be inconsist
    
[^2]: NervePool: 一个单纯复形池化层

    NervePool: A Simplicial Pooling Layer. (arXiv:2305.06315v1 [cs.CG])

    [http://arxiv.org/abs/2305.06315](http://arxiv.org/abs/2305.06315)

    单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。

    

    对于图结构数据的深度学习问题，池化层对于降采样、减少计算成本和减少过拟合都很重要。我们提出了一个池化层，NervePool，适用于单纯复形结构的数据，这种结构是图的推广，包括比顶点和边更高维度的单纯形；这种结构可以更灵活地建模更高阶的关系。所提出的单纯复合缩小方案基于顶点的分区构建，这使得我们可以生成单纯复形的分层表示，以一种学习的方式折叠信息。NervePool建立在学习的顶点群集分配的基础上，并以一种确定性的方式扩展到高维单纯形的缩小。虽然在实践中，池化操作是通过一系列矩阵运算来计算的，但是其拓扑动机是一个基于单纯形星星的并集和神经复合体的集合构造。

    For deep learning problems on graph-structured data, pooling layers are important for down sampling, reducing computational cost, and to minimize overfitting. We define a pooling layer, NervePool, for data structured as simplicial complexes, which are generalizations of graphs that include higher-dimensional simplices beyond vertices and edges; this structure allows for greater flexibility in modeling higher-order relationships. The proposed simplicial coarsening scheme is built upon partitions of vertices, which allow us to generate hierarchical representations of simplicial complexes, collapsing information in a learned fashion. NervePool builds on the learned vertex cluster assignments and extends to coarsening of higher dimensional simplices in a deterministic fashion. While in practice, the pooling operations are computed via a series of matrix operations, the topological motivation is a set-theoretic construction based on unions of stars of simplices and the nerve complex
    

