# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CaseLink: Inductive Graph Learning for Legal Case Retrieval](https://arxiv.org/abs/2403.17780) | 该论文提出了一种基于归纳图学习的方法，通过充分利用案例间的连接关系，提高了法律案例检索性能。 |
| [^2] | [A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer.](http://arxiv.org/abs/2401.01497) | 本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。 |
| [^3] | [Integrating the Wikidata Taxonomy into YAGO.](http://arxiv.org/abs/2308.11884) | 本文介绍了将整个Wikidata分类体系尽可能地合并到YAGO知识库中的工作，为YAGO添加了丰富的信息类别，并保持了知识库的逻辑一致性。 |
| [^4] | [Leveraging Recommender Systems to Reduce Content Gaps on Peer Production Platforms.](http://arxiv.org/abs/2307.08669) | 该研究通过在SuggestBot上进行离线分析和为期三个月的对照实验，发现推荐被低估主题的文章可以增加在这些文章上的编辑工作量，同时不会明显降低对推荐内容的接受程度。这一发现对解决对等生产平台上的内容缺口问题具有重要意义。 |
| [^5] | [Ripple Knowledge Graph Convolutional Networks For Recommendation Systems.](http://arxiv.org/abs/2305.01147) | 本文介绍了一种基于知识图谱的深度学习模型RKGCN，它能够动态分析用户的偏好并推荐出合适的物品。该模型在包括电影、书籍和音乐在内的三个真实世界的数据集上比5个基准模型表现更好。 |

# 详细

[^1]: CaseLink:法律案例检索的归纳图学习

    CaseLink: Inductive Graph Learning for Legal Case Retrieval

    [https://arxiv.org/abs/2403.17780](https://arxiv.org/abs/2403.17780)

    该论文提出了一种基于归纳图学习的方法，通过充分利用案例间的连接关系，提高了法律案例检索性能。

    

    在案例法中，先例是用来支持法官做出决定以及律师对特定案例的观点的相关案例。为了从大量案例池中高效地找到相关案例，法律从业者广泛使用检索工具。现有的法律案例检索模型主要通过比较单个案例的文本表示来工作。尽管它们获得了不错的检索准确性，但案例之间的固有连接关系未被充分利用于案例编码，从而限制了进一步提高检索性能。在案例池中，有三种案例连接关系：案例引用关系、案例语义关系和案例法律指控关系。由于法律案例检索任务的归纳方式的特点，使用案例引用作为输入

    arXiv:2403.17780v1 Announce Type: new  Abstract: In case law, the precedents are the relevant cases that are used to support the decisions made by the judges and the opinions of lawyers towards a given case. This relevance is referred to as the case-to-case reference relation. To efficiently find relevant cases from a large case pool, retrieval tools are widely used by legal practitioners. Existing legal case retrieval models mainly work by comparing the text representations of individual cases. Although they obtain a decent retrieval accuracy, the intrinsic case connectivity relationships among cases have not been well exploited for case encoding, therefore limiting the further improvement of retrieval performance. In a case pool, there are three types of case connectivity relationships: the case reference relationship, the case semantic relationship, and the case legal charge relationship. Due to the inductive manner in the task of legal case retrieval, using case reference as input 
    
[^2]: 一个预训练的顺序推荐框架：基于流行度动态的零-shot迁移

    A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer. (arXiv:2401.01497v1 [cs.IR])

    [http://arxiv.org/abs/2401.01497](http://arxiv.org/abs/2401.01497)

    本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。

    

    顺序推荐对于在线应用如电子商务、视频流媒体和社交媒体的成功至关重要。尽管模型架构不断改进，但对于每个新的应用领域，我们仍然需要从头训练一个新模型以获得高质量的推荐。另一方面，预训练的语言和视觉模型已经在零-shot或少-shot适应新应用领域方面取得了巨大成功。受到同行AI领域预训练模型成功的启发，我们提出了一种新颖的预训练顺序推荐框架：PrepRec。我们通过建模物品流行度动态来学习通用物品表示。通过在五个真实世界数据集上的大量实验证明，PrepRec在没有任何辅助信息的情况下不仅能够零-shot迁移到新领域，并且与同类最先进的顺序推荐模型相比，模型大小仅相当一小部分的情况下，可以实现竞争性的性能。

    Sequential recommenders are crucial to the success of online applications, \eg e-commerce, video streaming, and social media. While model architectures continue to improve, for every new application domain, we still have to train a new model from scratch for high quality recommendations. On the other hand, pre-trained language and vision models have shown great success in zero-shot or few-shot adaptation to new application domains. Inspired by the success of pre-trained models in peer AI fields, we propose a novel pre-trained sequential recommendation framework: PrepRec. We learn universal item representations by modeling item popularity dynamics. Through extensive experiments on five real-world datasets, we show that PrepRec, without any auxiliary information, can not only zero-shot transfer to a new domain, but achieve competitive performance compared to state-of-the-art sequential recommender models with only a fraction of the model size. In addition, with a simple post-hoc interpol
    
[^3]: 将Wikidata分类体系集成到YAGO中

    Integrating the Wikidata Taxonomy into YAGO. (arXiv:2308.11884v1 [cs.AI])

    [http://arxiv.org/abs/2308.11884](http://arxiv.org/abs/2308.11884)

    本文介绍了将整个Wikidata分类体系尽可能地合并到YAGO知识库中的工作，为YAGO添加了丰富的信息类别，并保持了知识库的逻辑一致性。

    

    Wikidata是最大的公共通用知识库之一。然而，由于它的合作性质，其模式和分类体系变得复杂。在YAGO 4知识库中，我们将Wikidata与Schema.org的本体论结合起来，减少和清理分类体系和约束条件，并使其能够在数据上运行自动推理器。然而，这也舍弃了大部分的Wikidata分类体系。在本文中，我们展示了将整个Wikidata分类体系尽可能地合并到YAGO知识库中的工作。我们特别关注逻辑约束和类与实例的细致区分。我们的工作创建了YAGO 4.5，为YAGO添加了丰富的信息类别，同时保持了知识库的逻辑一致性。

    Wikidata is one of the largest public general-purpose Knowledge Bases (KBs). Yet, due to its collaborative nature, its schema and taxonomy have become convoluted. For the YAGO 4 KB, we combined Wikidata with the ontology from Schema.org, which reduced and cleaned up the taxonomy and constraints and made it possible to run automated reasoners on the data. However, it also cut away large parts of the Wikidata taxonomy. In this paper, we present our effort to merge the entire Wikidata taxonomy into the YAGO KB as much as possible. We pay particular attention to logical constraints and a careful distinction of classes and instances. Our work creates YAGO 4.5, which adds a rich layer of informative classes to YAGO, while at the same time keeping the KB logically consistent.
    
[^4]: 利用推荐系统缩小对等生产平台上的内容差距

    Leveraging Recommender Systems to Reduce Content Gaps on Peer Production Platforms. (arXiv:2307.08669v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2307.08669](http://arxiv.org/abs/2307.08669)

    该研究通过在SuggestBot上进行离线分析和为期三个月的对照实验，发现推荐被低估主题的文章可以增加在这些文章上的编辑工作量，同时不会明显降低对推荐内容的接受程度。这一发现对解决对等生产平台上的内容缺口问题具有重要意义。

    

    维基百科等对等生产平台常常存在内容缺口。先前研究表明，推荐系统可以帮助解决这个问题，通过引导编辑人员关注被低估的主题。然而，目前尚不清楚这种方法是否会导致不太相关的推荐，从而降低对推荐内容的整体参与度。为了回答这个问题，我们首先在SuggestBot上进行了离线分析（研究1），然后进行了为期三个月的对照实验（研究2）。我们的结果显示，向用户展示来自被低估主题的文章可以增加在这些文章上的工作量，而不会明显降低对推荐内容的接受程度。我们讨论了结果的意义，包括如何忽视文章发现过程可能会人为地限制推荐。我们以"过滤气泡"的常见问题来展示这一现象，并对任何平台都存在的类似问题进行了类比。

    Peer production platforms like Wikipedia commonly suffer from content gaps. Prior research suggests recommender systems can help solve this problem, by guiding editors towards underrepresented topics. However, it remains unclear whether this approach would result in less relevant recommendations, leading to reduced overall engagement with recommended items. To answer this question, we first conducted offline analyses (Study 1) on SuggestBot, a task-routing recommender system for Wikipedia, then did a three-month controlled experiment (Study 2). Our results show that presenting users with articles from underrepresented topics increased the proportion of work done on those articles without significantly reducing overall recommendation uptake. We discuss the implications of our results, including how ignoring the article discovery process can artificially narrow recommendations. We draw parallels between this phenomenon and the common issue of "filter bubbles" to show how any platform tha
    
[^5]: 基于知识图谱的卷积神经网络在推荐系统中的应用

    Ripple Knowledge Graph Convolutional Networks For Recommendation Systems. (arXiv:2305.01147v1 [cs.IR])

    [http://arxiv.org/abs/2305.01147](http://arxiv.org/abs/2305.01147)

    本文介绍了一种基于知识图谱的深度学习模型RKGCN，它能够动态分析用户的偏好并推荐出合适的物品。该模型在包括电影、书籍和音乐在内的三个真实世界的数据集上比5个基准模型表现更好。

    

    最近已经证明，使用知识图谱来辅助深度学习模型进行推荐决策能有效提高模型的可解释性和准确性。本文介绍了一种端到端的深度学习模型，命名为RKGCN，它动态分析每个用户的偏好，并推荐出合适的物品。它在物品和用户双方面利用知识图谱来丰富它们的表示，最大化知识图谱中丰富的信息的利用。 RKGCN能够在三种不同的场景下提供更个性化和相关的推荐。实验结果表明，在包括电影、书籍和音乐在内的三个真实世界的数据集上，我们的模型比5个基准模型更有效。

    Using knowledge graphs to assist deep learning models in making recommendation decisions has recently been proven to effectively improve the model's interpretability and accuracy. This paper introduces an end-to-end deep learning model, named RKGCN, which dynamically analyses each user's preferences and makes a recommendation of suitable items. It combines knowledge graphs on both the item side and user side to enrich their representations to maximize the utilization of the abundant information in knowledge graphs. RKGCN is able to offer more personalized and relevant recommendations in three different scenarios. The experimental results show the superior effectiveness of our model over 5 baseline models on three real-world datasets including movies, books, and music.
    

