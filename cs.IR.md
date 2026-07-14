# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Research Team Identification Based on Representation Learning of Academic Heterogeneous Information Network.](http://arxiv.org/abs/2311.00922) | 本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法，通过利用节点级和元路径级的注意机制学习低维稠密实值向量表示，以有效识别和发现学术网络中的科研团队。 |
| [^2] | [Federated Topic Model and Model Pruning Based on Variational Autoencoder.](http://arxiv.org/abs/2311.00314) | 本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。 |

# 详细

[^1]: 基于学术异构信息网络表示学习的研究团队识别

    Research Team Identification Based on Representation Learning of Academic Heterogeneous Information Network. (arXiv:2311.00922v1 [cs.IR])

    [http://arxiv.org/abs/2311.00922](http://arxiv.org/abs/2311.00922)

    本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法，通过利用节点级和元路径级的注意机制学习低维稠密实值向量表示，以有效识别和发现学术网络中的科研团队。

    

    现实世界中的学术网络通常可以由由多类型节点和关系组成的异构信息网络来描述。现有关于同构信息网络的表示学习方法缺乏对异构信息网络的探索能力，无法应用于异构信息网络。针对从由庞大复杂的科技大数据组成的学术异构信息网络中有效识别和发现科研团队的实际需求，本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法。该方法利用节点级和元路径级的注意机制，在保留网络中节点的丰富拓扑信息和语义信息的基础上，学习低维稠密实值向量表示。

    Academic networks in the real world can usually be described by heterogeneous information networks composed of multi-type nodes and relationships. Some existing research on representation learning for homogeneous information networks lacks the ability to explore heterogeneous information networks in heterogeneous information networks. It cannot be applied to heterogeneous information networks. Aiming at the practical needs of effectively identifying and discovering scientific research teams from the academic heterogeneous information network composed of massive and complex scientific and technological big data, this paper proposes a scientific research team identification method based on representation learning of academic heterogeneous information networks. The attention mechanism at node level and meta-path level learns low-dimensional, dense and real-valued vector representations on the basis of retaining the rich topological information of nodes in the network and the semantic info
    
[^2]: 基于变分自编码器的联邦主题模型和模型剪枝

    Federated Topic Model and Model Pruning Based on Variational Autoencoder. (arXiv:2311.00314v1 [cs.LG])

    [http://arxiv.org/abs/2311.00314](http://arxiv.org/abs/2311.00314)

    本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。

    

    主题建模已经成为在大规模文档集合中发现模式和主题的有价值工具。然而，当跨多个方参与交叉分析时，数据隐私成为一个关键问题。联邦主题建模已经被开发出来解决这个问题，允许多个参与方在保护隐私的同时共同训练模型。然而，在联邦场景中存在通信和性能挑战。为了解决上述问题，本文提出了一种建立联邦主题模型并确保每个节点隐私的方法，并使用神经网络模型剪枝加速模型，其中客户端定期将模型神经元累积梯度和模型权重发送给服务器，服务器对模型进行剪枝。为了满足不同的要求，提出了两种确定模型剪枝率的不同方法。

    Topic modeling has emerged as a valuable tool for discovering patterns and topics within large collections of documents. However, when cross-analysis involves multiple parties, data privacy becomes a critical concern. Federated topic modeling has been developed to address this issue, allowing multiple parties to jointly train models while protecting pri-vacy. However, there are communication and performance challenges in the federated sce-nario. In order to solve the above problems, this paper proposes a method to establish a federated topic model while ensuring the privacy of each node, and use neural network model pruning to accelerate the model, where the client periodically sends the model neu-ron cumulative gradients and model weights to the server, and the server prunes the model. To address different requirements, two different methods are proposed to determine the model pruning rate. The first method involves slow pruning throughout the entire model training process, which has 
    

