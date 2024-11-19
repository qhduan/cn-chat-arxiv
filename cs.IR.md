# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unlocking the `Why' of Buying: Introducing a New Dataset and Benchmark for Purchase Reason and Post-Purchase Experience](https://arxiv.org/abs/2402.13417) | 引入了一个新的数据集和基准，旨在揭示用户购买决策背后的原因，提出了一个有效的基于LLM的方法来生成高质量、个性化的购买原因解释。 |
| [^2] | [Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT](https://arxiv.org/abs/2402.07440) | 该论文介绍了LoCoV1，一个用于评估长上下文检索性能的新型基准测试，并提出了M2-BERT检索编码器，用于处理长上下文检索，解决了如何评估性能、预训练语言模型以及如何进行微调的挑战。 |
| [^3] | [SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.](http://arxiv.org/abs/2401.15299) | SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。 |

# 详细

[^1]: 解锁购买的“为何”：引入一个新的数据集和购买原因与后购买体验的基准

    Unlocking the `Why' of Buying: Introducing a New Dataset and Benchmark for Purchase Reason and Post-Purchase Experience

    [https://arxiv.org/abs/2402.13417](https://arxiv.org/abs/2402.13417)

    引入了一个新的数据集和基准，旨在揭示用户购买决策背后的原因，提出了一个有效的基于LLM的方法来生成高质量、个性化的购买原因解释。

    

    解释对于提高现代推荐系统中用户信任和理解至关重要。为了构建真正可解释的系统，我们需要能阐明用户为何做出选择的高质量数据集。我们提出了一个新颖的购买原因解释任务。为此，我们引入了一种基于LLM的方法来生成一个由真实用户解释为何做出某些购买决策的文本解释的数据集。我们诱导LLM明确区分用户评论中购买产品背后的原因和购买后的体验。自动化的LLM驱动评估以及小规模人工评估证实了我们方法获取高质量、个性化解释的有效性。我们在两个个性化数据集上对该数据集进行基准测试。

    arXiv:2402.13417v1 Announce Type: new  Abstract: Explanations are crucial for enhancing user trust and understanding within modern recommendation systems. To build truly explainable systems, we need high-quality datasets that elucidate why users make choices. While previous efforts have focused on extracting users' post-purchase sentiment in reviews, they ignore the reasons behind the decision to buy.   In our work, we propose a novel purchase reason explanation task. To this end, we introduce an LLM-based approach to generate a dataset that consists of textual explanations of why real users make certain purchase decisions. We induce LLMs to explicitly distinguish between the reasons behind purchasing a product and the experience after the purchase in a user review. An automated, LLM-driven evaluation, as well as a small scale human evaluation, confirms the effectiveness of our approach to obtaining high-quality, personalized explanations. We benchmark this dataset on two personalized 
    
[^2]: 使用LoCo和M2-BERT进行基准测试和构建长上下文检索模型

    Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT

    [https://arxiv.org/abs/2402.07440](https://arxiv.org/abs/2402.07440)

    该论文介绍了LoCoV1，一个用于评估长上下文检索性能的新型基准测试，并提出了M2-BERT检索编码器，用于处理长上下文检索，解决了如何评估性能、预训练语言模型以及如何进行微调的挑战。

    

    检索管道是许多机器学习系统中的重要组成部分，在文档很长（例如10K个标记或更多）且需要在整个文本中合成信息来确定相关文档的领域中表现不佳。开发适用于这些领域的长上下文检索编码器面临三个挑战：（1）如何评估长上下文检索性能，（2）如何预训练基本语言模型以表示短上下文（对应查询）和长上下文（对应文档），以及（3）如何根据GPU内存限制下的批量大小限制对该模型进行微调。为了解决这些挑战，我们首先介绍了LoCoV1，这是一个新颖的12个任务基准测试，用于测量在不可分块或不有效的情况下的长上下文检索。接下来，我们提出了M2-BERT检索编码器，这是一个80M参数状态空间编码器模型，采用Monarch Mixer架构构建，能够进行可扩展的检索。

    Retrieval pipelines-an integral component of many machine learning systems-perform poorly in domains where documents are long (e.g., 10K tokens or more) and where identifying the relevant document requires synthesizing information across the entire text. Developing long-context retrieval encoders suitable for these domains raises three challenges: (1) how to evaluate long-context retrieval performance, (2) how to pretrain a base language model to represent both short contexts (corresponding to queries) and long contexts (corresponding to documents), and (3) how to fine-tune this model for retrieval under the batch size limitations imposed by GPU memory constraints. To address these challenges, we first introduce LoCoV1, a novel 12 task benchmark constructed to measure long-context retrieval where chunking is not possible or not effective. We next present the M2-BERT retrieval encoder, an 80M parameter state-space encoder model built from the Monarch Mixer architecture, capable of scali
    
[^3]: SupplyGraph: 使用图神经网络进行供应链规划的基准数据集

    SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks. (arXiv:2401.15299v1 [cs.LG])

    [http://arxiv.org/abs/2401.15299](http://arxiv.org/abs/2401.15299)

    SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。

    

    图神经网络（GNNs）在不同领域如运输、生物信息学、语言处理和计算机视觉中取得了重要进展。然而，在将GNNs应用于供应链网络方面，目前尚缺乏研究。供应链网络在结构上类似于图形，使其成为应用GNN方法的理想选择。这为优化、预测和解决供应链问题开辟了无限可能。然而，此方法的一个主要障碍在于缺乏真实世界的基准数据集以促进使用GNN来研究和解决供应链问题。为了解决这个问题，我们提供了一个来自孟加拉国一家领先的快速消费品公司的实际基准数据集，该数据集侧重于用于生产目的的供应链规划的时间任务。该数据集包括时间数据作为节点特征，以实现销售预测、生产计划和故障识别。

    Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of fa
    

