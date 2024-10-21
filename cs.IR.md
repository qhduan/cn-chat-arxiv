# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-shot Query Reformulation for Conversational Search.](http://arxiv.org/abs/2307.09384) | 提出了一种零样本查询重构（ZeQR）框架，通过利用机器阅读理解任务的语言模型来解决对话搜索中的数据稀疏性、解释性不足和歧义的问题。 |
| [^2] | [An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval.](http://arxiv.org/abs/2307.08803) | 本文研究了对话式段落检索中混合主动查询重构的探索，并提出了一个混合主动查询重构模块，该模块能够基于用户与系统之间的混合主动交互对原始查询进行重构，以提高检索效果。 |

# 详细

[^1]: 零样本对话搜索中的查询重构

    Zero-shot Query Reformulation for Conversational Search. (arXiv:2307.09384v1 [cs.IR])

    [http://arxiv.org/abs/2307.09384](http://arxiv.org/abs/2307.09384)

    提出了一种零样本查询重构（ZeQR）框架，通过利用机器阅读理解任务的语言模型来解决对话搜索中的数据稀疏性、解释性不足和歧义的问题。

    

    随着语音助手的普及，对话搜索在信息检索领域引起了更多的关注。然而，对话搜索中的数据稀疏性问题严重阻碍了监督式对话搜索方法的进展。因此，研究人员更加关注零样本对话搜索方法。然而，现有的零样本方法存在三个主要限制：它们不适用于所有的检索器，它们的有效性缺乏足够的解释性，并且他们无法解决因省略而导致的常见对话歧义。为了解决这些限制，我们引入了一种新颖的零样本查询重构（ZeQR）框架，该框架根据先前的对话上下文重构查询，而无需对话搜索数据的监督。具体来说，我们的框架利用了设计用于机器阅读理解任务的语言模型来明确解决两个常见的歧义：协调和省略。

    As the popularity of voice assistants continues to surge, conversational search has gained increased attention in Information Retrieval. However, data sparsity issues in conversational search significantly hinder the progress of supervised conversational search methods. Consequently, researchers are focusing more on zero-shot conversational search approaches. Nevertheless, existing zero-shot methods face three primary limitations: they are not universally applicable to all retrievers, their effectiveness lacks sufficient explainability, and they struggle to resolve common conversational ambiguities caused by omission. To address these limitations, we introduce a novel Zero-shot Query Reformulation (ZeQR) framework that reformulates queries based on previous dialogue contexts without requiring supervision from conversational search data. Specifically, our framework utilizes language models designed for machine reading comprehension tasks to explicitly resolve two common ambiguities: cor
    
[^2]: 《混合主动查询重构在对话式段落检索中的探索研究》的研究报告

    An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval. (arXiv:2307.08803v1 [cs.IR])

    [http://arxiv.org/abs/2307.08803](http://arxiv.org/abs/2307.08803)

    本文研究了对话式段落检索中混合主动查询重构的探索，并提出了一个混合主动查询重构模块，该模块能够基于用户与系统之间的混合主动交互对原始查询进行重构，以提高检索效果。

    

    在本文中，我们报告了我们在TREC Conversational Assistance Track (CAsT) 2022中的方法和实验。我们的目标是复现多阶段的检索管线，并探索在对话式段落检索场景中涉及混合主动交互的潜在好处之一：对原始查询进行重构。在多阶段检索管线的第一个排名阶段之前，我们提出了一个混合主动查询重构模块，它通过用户与系统之间的混合主动交互实现查询重构，作为神经重构方法的替代品。具体而言，我们设计了一个算法来生成与原始查询中的歧义相关的适当问题，以及另一个算法来解析用户的反馈并将其融入到原始查询中以进行查询重构。对于我们的多阶段管线的第一个排名阶段，我们采用了一个稀疏排名函数：BM25和一个密集检索方法：TCT-ColBERT。

    In this paper, we report our methods and experiments for the TREC Conversational Assistance Track (CAsT) 2022. In this work, we aim to reproduce multi-stage retrieval pipelines and explore one of the potential benefits of involving mixed-initiative interaction in conversational passage retrieval scenarios: reformulating raw queries. Before the first ranking stage of a multi-stage retrieval pipeline, we propose a mixed-initiative query reformulation module, which achieves query reformulation based on the mixed-initiative interaction between the users and the system, as the replacement for the neural reformulation method. Specifically, we design an algorithm to generate appropriate questions related to the ambiguities in raw queries, and another algorithm to reformulate raw queries by parsing users' feedback and incorporating it into the raw query. For the first ranking stage of our multi-stage pipelines, we adopt a sparse ranking function: BM25, and a dense retrieval method: TCT-ColBERT
    

