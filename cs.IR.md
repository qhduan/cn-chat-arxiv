# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Location Sensitive Embedding for Knowledge Graph Embedding.](http://arxiv.org/abs/2401.10893) | 这篇论文介绍了一种新颖的位置敏感嵌入（LSE）方法，该方法通过关系特定的映射来修改头实体，将关系概念化为线性变换。LSE在知识图谱嵌入领域具有理论基础，同时提出了更高效的变体LSEd。实验证明LSEd在链接预测任务上具有竞争力。 |

# 详细

[^1]: 知识图谱嵌入的位置敏感嵌入

    Location Sensitive Embedding for Knowledge Graph Embedding. (arXiv:2401.10893v1 [cs.IR])

    [http://arxiv.org/abs/2401.10893](http://arxiv.org/abs/2401.10893)

    这篇论文介绍了一种新颖的位置敏感嵌入（LSE）方法，该方法通过关系特定的映射来修改头实体，将关系概念化为线性变换。LSE在知识图谱嵌入领域具有理论基础，同时提出了更高效的变体LSEd。实验证明LSEd在链接预测任务上具有竞争力。

    

    知识图谱嵌入将知识图谱转化为连续的、低维度的空间，有助于推理和补全任务。该领域主要分为传统的距离模型和语义匹配模型。传统的距离模型面临的关键挑战是无法有效区分图谱中的“头实体”和“尾实体”。为了解决这个问题，提出了新颖的位置敏感嵌入（LSE）方法。LSE通过关系特定的映射修改头实体，将关系概念化为线性变换而不仅仅是平移。LSE的理论基础，包括其表示能力和与现有模型的联系，都进行了详细研究。一种更简化的变体LSEd利用对角矩阵进行变换以提高实用性能。在对四个大规模数据集进行链接预测的测试中，LSEd要么表现更好，要么具有竞争力。

    Knowledge graph embedding transforms knowledge graphs into a continuous, low-dimensional space, facilitating inference and completion tasks. This field is mainly divided into translational distance models and semantic matching models. A key challenge in translational distance models is their inability to effectively differentiate between 'head' and 'tail' entities in graphs. To address this, the novel location-sensitive embedding (LSE) method has been developed. LSE innovatively modifies the head entity using relation-specific mappings, conceptualizing relations as linear transformations rather than mere translations. The theoretical foundations of LSE, including its representational capabilities and its connections to existing models, have been thoroughly examined. A more streamlined variant, LSEd, employs a diagonal matrix for transformations to enhance practical efficiency. In tests conducted on four large-scale datasets for link prediction, LSEd either outperforms or is competitive
    

