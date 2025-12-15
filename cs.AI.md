# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216) | BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。 |

# 详细

[^1]: BGE M3-嵌入：通过自知识蒸馏实现多语言、多功能和多粒度的文本嵌入

    BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

    [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)

    BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。

    

    在本文中，我们提出了一种新的嵌入模型，称为M3-嵌入，以其在多语言、多功能和多粒度方面的多样性而著称。它可以支持超过100种工作语言，在多语言和跨语言检索任务上取得了新的最先进性能。它可以同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索，为现实世界的IR应用提供了统一的模型基础。它能够处理不同粒度的输入，从短句到长达8192个标记的文档。M3-嵌入的有效训练包括以下技术贡献。我们提出了一种新颖的自知识蒸馏方法，可以将来自不同检索功能的相关性分数整合为教师信号，以提高训练质量。我们还优化了批处理策略。

    In this paper, we present a new embedding model, called M3-Embedding, which is distinguished for its versatility in Multi-Linguality, Multi-Functionality, and Multi-Granularity. It can support more than 100 working languages, leading to new state-of-the-art performances on multi-lingual and cross-lingual retrieval tasks. It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval, which provides a unified model foundation for real-world IR applications. It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens. The effective training of M3-Embedding involves the following technical contributions. We propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strat
    

