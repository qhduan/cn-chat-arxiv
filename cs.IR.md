# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoMLP: Automated MLP for Sequential Recommendations.](http://arxiv.org/abs/2303.06337) | AutoMLP是一种新颖的序列推荐系统，通过自动化和自适应搜索算法，更好地模拟用户的长期/短期兴趣，实现更好的推荐效果。 |
| [^2] | [NASTyLinker: NIL-Aware Scalable Transformer-based Entity Linker.](http://arxiv.org/abs/2303.04426) | NASTyLinker是一种NIL感知的实体链接器，它通过生成提及簇来表示NIL实体，并在保持已知实体高链接性能的同时解决冲突。 |
| [^3] | [A Dataset for Learning Graph Representations to Predict Customer Returns in Fashion Retail.](http://arxiv.org/abs/2302.14096) | 该论文介绍了一个由ASOS收集的新型数据集，用于解决时尚零售生态系统中预测客户退货的挑战。研究者使用图表示学习方法，提高了退货预测分类任务的F1分数至0.792，这比其他模型有所改进。 |
| [^4] | [A Survey on Event-based News Narrative Extraction.](http://arxiv.org/abs/2302.08351) | 本文综述了事件驱动新闻叙事提取的研究，通过筛选超过900篇文章，得到了54篇相关文章，这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。 |
| [^5] | [COMET: Convolutional Dimension Interaction for Collaborative Filtering.](http://arxiv.org/abs/2007.14129) | COMET是一种新的基于表示学习的模型，它可以同时模拟历史交互和嵌入维度之间的高阶交互模式。 |

# 详细

[^1]: AutoMLP: 自动化MLP用于序列推荐

    AutoMLP: Automated MLP for Sequential Recommendations. (arXiv:2303.06337v1 [cs.IR])

    [http://arxiv.org/abs/2303.06337](http://arxiv.org/abs/2303.06337)

    AutoMLP是一种新颖的序列推荐系统，通过自动化和自适应搜索算法，更好地模拟用户的长期/短期兴趣，实现更好的推荐效果。

    AutoMLP is a novel sequential recommender system that models users' long/short-term interests through an automated and adaptive search algorithm, achieving better recommendation performance.

    序列推荐系统旨在根据用户的历史交互来预测他们下一个感兴趣的项目。然而，长期存在的问题是如何区分用户的长期/短期兴趣，这可能是异质的并对下一个推荐产生不同的贡献。现有方法通常通过穷举搜索或经验经验设置预定义的短期兴趣长度，这既高度低效又产生次优结果。最近的先进基于变压器的模型可以实现最先进的性能，尽管存在上述问题，但它们对输入序列的长度具有二次计算复杂度。为此，本文提出了一种新颖的序列推荐系统AutoMLP，旨在更好地模拟用户的长期/短期兴趣。此外，我们设计了一种自动化和自适应搜索算法，以通过端到端优化获得更好的短期兴趣长度。通过实验，我们证明了AutoMLP的有效性和效率。

    Sequential recommender systems aim to predict users' next interested item given their historical interactions. However, a long-standing issue is how to distinguish between users' long/short-term interests, which may be heterogeneous and contribute differently to the next recommendation. Existing approaches usually set pre-defined short-term interest length by exhaustive search or empirical experience, which is either highly inefficient or yields subpar results. The recent advanced transformer-based models can achieve state-of-the-art performances despite the aforementioned issue, but they have a quadratic computational complexity to the length of the input sequence. To this end, this paper proposes a novel sequential recommender system, AutoMLP, aiming for better modeling users' long/short-term interests from their historical interactions. In addition, we design an automated and adaptive search algorithm for preferable short-term interest length via end-to-end optimization. Through ext
    
[^2]: NASTyLinker：NIL感知可扩展基于Transformer的实体链接器

    NASTyLinker: NIL-Aware Scalable Transformer-based Entity Linker. (arXiv:2303.04426v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.04426](http://arxiv.org/abs/2303.04426)

    NASTyLinker是一种NIL感知的实体链接器，它通过生成提及簇来表示NIL实体，并在保持已知实体高链接性能的同时解决冲突。

    NASTyLinker is a NIL-aware entity linker that represents NIL entities by producing mention clusters and resolves conflicts while maintaining high linking performance for known entities.

    实体链接（EL）是检测文本中实体提及并将其消歧为参考知识库的任务。大多数流行的EL方法假定参考知识库是完整的。然而，在实践中，需要处理链接到不包含在知识库中的实体（NIL实体）的情况。最近的研究表明，考虑提及之间的亲和力可以用于表示NIL实体，方法是通过生成提及簇。同时，提及之间的亲和力可以帮助显著提高已知实体的链接性能。通过NASTyLinker，我们介绍了一种EL方法，它知道NIL实体并产生相应的提及簇，同时保持已知实体的高链接性能。该方法基于Transformer的密集表示对提及和实体进行聚类，并解决冲突（如果一个实体有多个提及）。

    Entity Linking (EL) is the task of detecting mentions of entities in text and disambiguating them to a reference knowledge base. Most prevalent EL approaches assume that the reference knowledge base is complete. In practice, however, it is necessary to deal with the case of linking to an entity that is not contained in the knowledge base (NIL entity). Recent works have shown that, instead of focusing only on affinities between mentions and entities, considering inter-mention affinities can be used to represent NIL entities by producing clusters of mentions. At the same time, inter-mention affinities can help to substantially improve linking performance for known entities. With NASTyLinker, we introduce an EL approach that is aware of NIL entities and produces corresponding mention clusters while maintaining high linking performance for known entities. The approach clusters mentions and entities based on dense representations from Transformers and resolves conflicts (if more than one en
    
[^3]: 一份用于学习图表示以预测时尚零售客户退货的数据集

    A Dataset for Learning Graph Representations to Predict Customer Returns in Fashion Retail. (arXiv:2302.14096v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.14096](http://arxiv.org/abs/2302.14096)

    该论文介绍了一个由ASOS收集的新型数据集，用于解决时尚零售生态系统中预测客户退货的挑战。研究者使用图表示学习方法，提高了退货预测分类任务的F1分数至0.792，这比其他模型有所改进。

    This paper introduces a novel dataset collected by ASOS for predicting customer returns in a fashion retail ecosystem. The researchers use Graph Representation Learning to improve the F1-score of the return prediction classification task to 0.792, outperforming other models.

    我们提出了一个由ASOS（一家主要的在线时尚零售商）收集的新型数据集，以解决在时尚零售生态系统中预测客户退货的挑战。通过发布这个庞大的数据集，我们希望激发研究社区和时尚行业之间的进一步合作。我们首先探讨了这个数据集的结构，重点关注图表示学习的应用，以利用自然数据结构并提供对数据中特定特征的统计洞察。除此之外，我们展示了一个退货预测分类任务的示例，其中包括一些基线模型（即没有中间表示学习步骤）和基于图表示的模型。我们展示了在下游退货预测分类任务中，使用图神经网络（GNN）可以找到F1分数为0.792，这比本文讨论的其他模型有所改进。除了这个增加的F1分数，我们还提出了一个l

    We present a novel dataset collected by ASOS (a major online fashion retailer) to address the challenge of predicting customer returns in a fashion retail ecosystem. With the release of this substantial dataset we hope to motivate further collaboration between research communities and the fashion industry. We first explore the structure of this dataset with a focus on the application of Graph Representation Learning in order to exploit the natural data structure and provide statistical insights into particular features within the data. In addition to this, we show examples of a return prediction classification task with a selection of baseline models (i.e. with no intermediate representation learning step) and a graph representation based model. We show that in a downstream return prediction classification task, an F1-score of 0.792 can be found using a Graph Neural Network (GNN), improving upon other models discussed in this work. Alongside this increased F1-score, we also present a l
    
[^4]: 事件驱动新闻叙事提取综述

    A Survey on Event-based News Narrative Extraction. (arXiv:2302.08351v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08351](http://arxiv.org/abs/2302.08351)

    本文综述了事件驱动新闻叙事提取的研究，通过筛选超过900篇文章，得到了54篇相关文章，这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。

    This survey presents an extensive study of research in the area of event-based news narrative extraction, screening over 900 articles and synthesizing 54 relevant articles organized by representation model, extraction criteria, and evaluation application.

    叙事是我们理解世界的基础，为我们提供了一个自然的时间知识表示结构。计算机叙事提取是人工智能的一个子领域，它大量使用信息检索和自然语言处理技术。尽管计算机叙事提取的重要性，但在综合以前的研究和策划未来的研究方面，学术研究相对较少。特别是，本文侧重于从事件中心的角度提取新闻叙事。从新闻数据中提取叙事在理解不断变化的信息景观方面具有多种应用。本综述对事件驱动新闻叙事提取领域的研究进行了广泛的研究。特别是，我们筛选了超过900篇文章，得到了54篇相关文章。这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。

    Narratives are fundamental to our understanding of the world, providing us with a natural structure for knowledge representation over time. Computational narrative extraction is a subfield of artificial intelligence that makes heavy use of information retrieval and natural language processing techniques. Despite the importance of computational narrative extraction, relatively little scholarly work exists on synthesizing previous research and strategizing future research in the area. In particular, this article focuses on extracting news narratives from an event-centric perspective. Extracting narratives from news data has multiple applications in understanding the evolving information landscape. This survey presents an extensive study of research in the area of event-based news narrative extraction. In particular, we screened over 900 articles that yielded 54 relevant articles. These articles are synthesized and organized by representation model, extraction criteria, and evaluation app
    
[^5]: COMET: 卷积维度交互用于协同过滤

    COMET: Convolutional Dimension Interaction for Collaborative Filtering. (arXiv:2007.14129v6 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2007.14129](http://arxiv.org/abs/2007.14129)

    COMET是一种新的基于表示学习的模型，它可以同时模拟历史交互和嵌入维度之间的高阶交互模式。

    COMET is a novel representation learning-based model that can simultaneously model the high-order interaction patterns among historical interactions and embedding dimensions.

    基于表示学习的推荐模型在推荐技术中扮演着主导角色。然而，大多数现有方法假设历史交互和嵌入维度相互独立，因此遗憾地忽略了历史交互和嵌入维度之间的高阶交互信息。在本文中，我们提出了一种新的基于表示学习的模型，称为COMET（COnvolutional diMEnsion inTeraction），它同时模拟历史交互和嵌入维度之间的高阶交互模式。具体而言，COMET首先将历史交互的嵌入水平堆叠，从而产生两个“嵌入映射”。通过卷积神经网络（CNN）同时使用不同大小的内部交互和维度交互内核，可以利用内部交互和维度交互。然后应用全连接的多层感知器（MLP）来获得两个交互向量。

    Representation learning-based recommendation models play a dominant role among recommendation techniques. However, most of the existing methods assume both historical interactions and embedding dimensions are independent of each other, and thus regrettably ignore the high-order interaction information among historical interactions and embedding dimensions. In this paper, we propose a novel representation learning-based model called COMET (COnvolutional diMEnsion inTeraction), which simultaneously models the high-order interaction patterns among historical interactions and embedding dimensions. To be specific, COMET stacks the embeddings of historical interactions horizontally at first, which results in two "embedding maps". In this way, internal interactions and dimensional interactions can be exploited by convolutional neural networks (CNN) with kernels of different sizes simultaneously. A fully-connected multi-layer perceptron (MLP) is then applied to obtain two interaction vectors. 
    

