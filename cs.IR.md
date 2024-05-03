# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Disaggregated Multi-Tower: Topology-aware Modeling Technique for Efficient Large-Scale Recommendation](https://arxiv.org/abs/2403.00877) | Disaggregated Multi-Tower提出了一种面向拓扑感知的建模技术，通过SPTT、TM和TP三个组件实现了高效的大规模推荐，加速性能提升了1.9倍。 |
| [^2] | [Language Models As Semantic Indexers.](http://arxiv.org/abs/2310.07815) | 本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。 |
| [^3] | [Large language models can accurately predict searcher preferences.](http://arxiv.org/abs/2309.10621) | 大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。 |

# 详细

[^1]: Disaggregated Multi-Tower: 面向拓扑感知的高效大规模推荐建模技术

    Disaggregated Multi-Tower: Topology-aware Modeling Technique for Efficient Large-Scale Recommendation

    [https://arxiv.org/abs/2403.00877](https://arxiv.org/abs/2403.00877)

    Disaggregated Multi-Tower提出了一种面向拓扑感知的建模技术，通过SPTT、TM和TP三个组件实现了高效的大规模推荐，加速性能提升了1.9倍。

    

    我们研究了深度学习推荐模型的扁平架构、常见的分布式训练模式和分层数据中心拓扑之间的不匹配。为了解决相关的低效性，我们提出了Disaggregated Multi-Tower（DMT），这是一种建模技术，包括（1）语义保留的Tower Transform（SPTT），一个将单片全局嵌入查找过程分解为不相交塔以利用数据中心位置关系的新型训练模式；（2）Tower Module（TM），一个附加到每个塔的协同稠密组件，通过分层特征交互降低模型复杂性和通信量；和（3）Tower Partitioner（TP），一个特征分区器，系统地创建具有有意义特征交互和负载平衡分配的塔，通过学习的嵌入来保持模型质量和训练吞吐量。我们展示了DMT相比于最新的方法可以实现高达1.9倍的加速。

    arXiv:2403.00877v1 Announce Type: new  Abstract: We study a mismatch between the deep learning recommendation models' flat architecture, common distributed training paradigm and hierarchical data center topology. To address the associated inefficiencies, we propose Disaggregated Multi-Tower (DMT), a modeling technique that consists of (1) Semantic-preserving Tower Transform (SPTT), a novel training paradigm that decomposes the monolithic global embedding lookup process into disjoint towers to exploit data center locality; (2) Tower Module (TM), a synergistic dense component attached to each tower to reduce model complexity and communication volume through hierarchical feature interaction; and (3) Tower Partitioner (TP), a feature partitioner to systematically create towers with meaningful feature interactions and load balanced assignments to preserve model quality and training throughput via learned embeddings. We show that DMT can achieve up to 1.9x speedup compared to the state-of-th
    
[^2]: 语言模型作为语义索引器

    Language Models As Semantic Indexers. (arXiv:2310.07815v1 [cs.IR])

    [http://arxiv.org/abs/2310.07815](http://arxiv.org/abs/2310.07815)

    本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。

    

    语义标识符（ID）是信息检索中的一个重要概念，旨在保留对象（如文档和项）内部的语义。先前的研究通常采用两阶段流程来学习语义ID，首先使用现成的文本编码器获取嵌入，并根据嵌入来推导ID。然而，每个步骤都会引入潜在的信息损失，并且文本编码器生成的潜在空间内的嵌入分布通常与语义索引所需的预期分布存在固有的不匹配。然而，设计一个既能学习文档的语义表示又能同时学习其分层结构的方法并不容易，因为语义ID是离散和顺序结构的，并且语义监督是不充分的。在本文中，我们引入了LMINDEXER，它是一个自监督框架，用于使用生成性语言模型学习语义ID。

    Semantic identifier (ID) is an important concept in information retrieval that aims to preserve the semantics of objects such as documents and items inside their IDs. Previous studies typically adopt a two-stage pipeline to learn semantic IDs by first procuring embeddings using off-the-shelf text encoders and then deriving IDs based on the embeddings. However, each step introduces potential information loss and there is usually an inherent mismatch between the distribution of embeddings within the latent space produced by text encoders and the anticipated distribution required for semantic indexing. Nevertheless, it is non-trivial to design a method that can learn the document's semantic representations and its hierarchical structure simultaneously, given that semantic IDs are discrete and sequentially structured, and the semantic supervision is deficient. In this paper, we introduce LMINDEXER, a self-supervised framework to learn semantic IDs with a generative language model. We tackl
    
[^3]: 大型语言模型能够准确预测搜索者的偏好

    Large language models can accurately predict searcher preferences. (arXiv:2309.10621v1 [cs.IR])

    [http://arxiv.org/abs/2309.10621](http://arxiv.org/abs/2309.10621)

    大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。

    

    相关性标签是评估和优化搜索系统的关键。获取大量相关性标签通常需要第三方标注人员，但存在低质量数据的风险。本论文介绍了一种改进标签质量的替代方法，通过从真实用户那里获得仔细反馈来获取高质量的第一方数据。

    Relevance labels, which indicate whether a search result is valuable to a searcher, are key to evaluating and optimising search systems. The best way to capture the true preferences of users is to ask them for their careful feedback on which results would be useful, but this approach does not scale to produce a large number of labels. Getting relevance labels at scale is usually done with third-party labellers, who judge on behalf of the user, but there is a risk of low-quality data if the labeller doesn't understand user needs. To improve quality, one standard approach is to study real users through interviews, user studies and direct feedback, find areas where labels are systematically disagreeing with users, then educate labellers about user needs through judging guidelines, training and monitoring. This paper introduces an alternate approach for improving label quality. It takes careful feedback from real users, which by definition is the highest-quality first-party gold data that 
    

