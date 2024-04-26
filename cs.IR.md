# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Antitrust, Amazon, and Algorithmic Auditing](https://arxiv.org/abs/2403.18623) | 本研究调查了亚马逊是否存在自我偏好的做法，讨论了如何利用计算机科学工具进行基于算法审计的监管，以规范数字市场。 |
| [^2] | [Retrieving Texts based on Abstract Descriptions.](http://arxiv.org/abs/2305.12517) | 本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。 |

# 详细

[^1]: 反垄断、亚马逊和算法审计

    Antitrust, Amazon, and Algorithmic Auditing

    [https://arxiv.org/abs/2403.18623](https://arxiv.org/abs/2403.18623)

    本研究调查了亚马逊是否存在自我偏好的做法，讨论了如何利用计算机科学工具进行基于算法审计的监管，以规范数字市场。

    

    在数字市场中，反垄断法和特殊法规旨在确保市场保持竞争，尽管数字平台在每个人生活中扮演着主导角色。与传统市场不同，这些市场中的市场参与者行为很容易被观察到。我们展示了一系列实证调查，探讨亚马逊在多大程度上参与了通常被描述为自我偏好的做法。我们讨论了本文中使用的计算机科学工具如何在基于算法审计的监管环境中使用，并要求在规模上监管数字市场。

    arXiv:2403.18623v1 Announce Type: cross  Abstract: In digital markets, antitrust law and special regulations aim to ensure that markets remain competitive despite the dominating role that digital platforms play today in everyone's life. Unlike traditional markets, market participant behavior is easily observable in these markets. We present a series of empirical investigations into the extent to which Amazon engages in practices that are typically described as self-preferencing. We discuss how the computer science tools used in this paper can be used in a regulatory environment that is based on algorithmic auditing and requires regulating digital markets at scale.
    
[^2]: 基于摘要描述的文本检索

    Retrieving Texts based on Abstract Descriptions. (arXiv:2305.12517v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12517](http://arxiv.org/abs/2305.12517)

    本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。

    

    虽然针对文本的信息提取，指令优化的大型语言模型表现优异，但对于在大规模文档集合中定位符合给定描述的文本（语义检索）并不适用。基于嵌入向量的相似度搜索可以通过查询执行检索，但嵌入中的相似度定义不明确且不一致，并且对于许多用例来说都是次优的。那么，什么是有效检索的好的查询表示？我们确定了根据内容的摘要描述检索句子的明确定义且一致的任务。我们展示了当前文本嵌入的不足，并提出了一种替代模型，在标准最近邻搜索中的表现显著提升。该模型使用通过提示LLM获得的正负样本对进行训练。虽然很容易从LLM中获得训练材料，但LLM无法直接执行检索任务。

    While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?  We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This de
    

