# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers](https://arxiv.org/abs/2403.18276) | Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。 |
| [^2] | [Retrieve Anything To Augment Large Language Models.](http://arxiv.org/abs/2310.07554) | 这项工作提出了一种新的方法，即LLM-Embedder，通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。 |

# 详细

[^1]: RankMamba，在Transformer时代对Mamba文档排名性能的基准测试

    RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers

    [https://arxiv.org/abs/2403.18276](https://arxiv.org/abs/2403.18276)

    Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。

    

    Transformer结构在自然语言处理（NLP）、计算机视觉（CV）和信息检索(IR)等多个应用的机器学习领域取得了巨大成功。Transformer架构的核心机制--注意力，在训练中需要$O(n^2)$的时间复杂度，在推断中需要$O(n)$的时间复杂度。许多工作已经提出改进注意力机制的可扩展性，比如Flash Attention和Multi-query Attention。另一方面的工作旨在设计新的机制来取代注意力。最近，基于状态空间模型的一个显著模型结构--Mamba，在多个序列建模任务中取得了与Transformer相当的性能。

    arXiv:2403.18276v1 Announce Type: cross  Abstract: Transformer structure has achieved great success in multiple applied machine learning communities, such as natural language processing (NLP), computer vision (CV) and information retrieval (IR). Transformer architecture's core mechanism -- attention requires $O(n^2)$ time complexity in training and $O(n)$ time complexity in inference. Many works have been proposed to improve the attention mechanism's scalability, such as Flash Attention and Multi-query Attention. A different line of work aims to design new mechanisms to replace attention. Recently, a notable model structure -- Mamba, which is based on state space models, has achieved transformer-equivalent performance in multiple sequence modeling tasks.   In this work, we examine \mamba's efficacy through the lens of a classical IR task -- document ranking. A reranker model takes a query and a document as input, and predicts a scalar relevance score. This task demands the language mod
    
[^2]: 检索任何内容来增强大型语言模型

    Retrieve Anything To Augment Large Language Models. (arXiv:2310.07554v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.07554](http://arxiv.org/abs/2310.07554)

    这项工作提出了一种新的方法，即LLM-Embedder，通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。

    

    大型语言模型(LLMs)面临着由于其在知识、记忆、对齐和行动方面的固有限制而产生的重要挑战。这些挑战不能单靠LLMs自行解决，而应依赖于来自外部世界（如知识库、记忆存储、演示示例和工具）的辅助。检索增强作为LLMs与外部辅助之间的重要机制。然而，传统方法遇到两个紧迫问题。一方面，通用检索器未能适当优化LLMs的检索增强。另一方面，任务特定的检索器缺乏所需的多样性，阻碍其在各种检索增强场景中的性能表现。在这项工作中，我们提出了一种新的方法，即LLM-Embedder，它通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。训练这样的统一模型并不容易，由于不同检索增强场景的多样性。

    Large language models (LLMs) face significant challenges stemming from their inherent limitations in knowledge, memory, alignment, and action. These challenges cannot be addressed by LLMs alone, but should rely on assistance from the external world, such as knowledge base, memory store, demonstration examples, and tools. Retrieval augmentation stands as a vital mechanism for bridging the gap between LLMs and the external assistance. However, conventional methods encounter two pressing issues. On the one hand, the general-purpose retrievers are not properly optimized for the retrieval augmentation of LLMs. On the other hand, the task-specific retrievers lack the required versatility, hindering their performance across the diverse retrieval augmentation scenarios.  In this work, we present a novel approach, the LLM-Embedder, which comprehensively supports the diverse retrieval augmentation needs of LLMs with one unified embedding model. Training such a unified model is non-trivial, as va
    

