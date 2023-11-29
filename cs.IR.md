# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks for Recommendation: Reproducibility, Graph Topology, and Node Representation.](http://arxiv.org/abs/2310.11270) | 本教程针对推荐中的图神经网络的三个关键方面进行了探讨：最先进方法的可复现性，图拓扑特征对模型性能的影响，以及学习节点表示的策略。 |
| [^2] | [Towards Populating Generalizable Engineering Design Knowledge.](http://arxiv.org/abs/2307.06985) | 这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。 |

# 详细

[^1]: 推荐系统中的图神经网络: 可复现性、图拓扑和节点表示

    Graph Neural Networks for Recommendation: Reproducibility, Graph Topology, and Node Representation. (arXiv:2310.11270v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.11270](http://arxiv.org/abs/2310.11270)

    本教程针对推荐中的图神经网络的三个关键方面进行了探讨：最先进方法的可复现性，图拓扑特征对模型性能的影响，以及学习节点表示的策略。

    

    最近几年，图神经网络（GNNs）在推荐系统中变得越来越重要。通过将用户-物品矩阵表示为一个二部图和无向图，GNN能够捕捉用户-物品之间的近距离和远距离交互，从而比传统推荐方法学习到更准确的偏好模式。与之前的同类教程不同，本教程旨在展示和探讨推荐中GNNs的三个关键方面：（i）最先进方法的可复现性，（ii）图拓扑特征对模型性能的潜在影响，以及（iii）在从零开始训练特征或利用预训练嵌入作为额外物品信息（例如多模态特征）时，学习节点表示的策略。我们的目标是为该领域提供三个新颖的理论和实践视角，目前在图学习中存在争议。

    Graph neural networks (GNNs) have gained prominence in recommendation systems in recent years. By representing the user-item matrix as a bipartite and undirected graph, GNNs have demonstrated their potential to capture short- and long-distance user-item interactions, thereby learning more accurate preference patterns than traditional recommendation approaches. In contrast to previous tutorials on the same topic, this tutorial aims to present and examine three key aspects that characterize GNNs for recommendation: (i) the reproducibility of state-of-the-art approaches, (ii) the potential impact of graph topological characteristics on the performance of these models, and (iii) strategies for learning node representations when training features from scratch or utilizing pre-trained embeddings as additional item information (e.g., multimodal features). The goal is to provide three novel theoretical and practical perspectives on the field, currently subject to debate in graph learning but l
    
[^2]: 迈向填充通用工程设计知识的方法

    Towards Populating Generalizable Engineering Design Knowledge. (arXiv:2307.06985v1 [cs.CL])

    [http://arxiv.org/abs/2307.06985](http://arxiv.org/abs/2307.06985)

    这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。

    

    为了填充通用工程设计知识，我们提出了一种从专利文件中提取head entity :: relationship :: tail entity形式事实的方法。这些事实可以在专利文件内部和跨文件之间组合形成知识图，用作表示和存储设计知识的方案。现有的工程设计文献中的方法通常利用一组预定义的关系来填充统计近似而非事实的三元组。在我们的方法中，我们训练一个标记器来识别句子中的实体和关系。在确定了一对实体后，我们训练另一个标记器来识别特定表示这对实体之间关系的关系标记。为了训练这些标记器，我们手动构建了一个包含44,227个句子和相应事实的数据集。我们还将该方法的性能与通常推荐的方法进行了比较，其中我们预.

    Aiming to populate generalizable engineering design knowledge, we propose a method to extract facts of the form head entity :: relationship :: tail entity from sentences found in patent documents. These facts could be combined within and across patent documents to form knowledge graphs that serve as schemes for representing as well as storing design knowledge. Existing methods in engineering design literature often utilise a set of predefined relationships to populate triples that are statistical approximations rather than facts. In our method, we train a tagger to identify both entities and relationships from a sentence. Given a pair of entities thus identified, we train another tagger to identify the relationship tokens that specifically denote the relationship between the pair. For training these taggers, we manually construct a dataset of 44,227 sentences and corresponding facts. We also compare the performance of the method against typically recommended approaches, wherein, we pre
    

