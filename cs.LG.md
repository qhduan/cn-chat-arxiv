# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216) | BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。 |
| [^2] | [Data as voters: instance selection using approval-based multi-winner voting.](http://arxiv.org/abs/2304.09995) | 该论文提出了一种基于赞成票的多获胜者投票的实例选择方法，通过代表性投票规则选择获胜者，并将其作为减少训练集的数据实例。 |
| [^3] | [GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism.](http://arxiv.org/abs/2303.13775) | 本文提出了一种新的并行小批量训练方法，即分裂并行，应用在图神经网络训练上，能有效缓解数据并行方法的性能瓶颈，同时在大规模图上的性能表现优越。 |

# 详细

[^1]: BGE M3-嵌入：通过自知识蒸馏实现多语言、多功能和多粒度的文本嵌入

    BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

    [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)

    BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。

    

    在本文中，我们提出了一种新的嵌入模型，称为M3-嵌入，以其在多语言、多功能和多粒度方面的多样性而著称。它可以支持超过100种工作语言，在多语言和跨语言检索任务上取得了新的最先进性能。它可以同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索，为现实世界的IR应用提供了统一的模型基础。它能够处理不同粒度的输入，从短句到长达8192个标记的文档。M3-嵌入的有效训练包括以下技术贡献。我们提出了一种新颖的自知识蒸馏方法，可以将来自不同检索功能的相关性分数整合为教师信号，以提高训练质量。我们还优化了批处理策略。

    In this paper, we present a new embedding model, called M3-Embedding, which is distinguished for its versatility in Multi-Linguality, Multi-Functionality, and Multi-Granularity. It can support more than 100 working languages, leading to new state-of-the-art performances on multi-lingual and cross-lingual retrieval tasks. It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval, which provides a unified model foundation for real-world IR applications. It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens. The effective training of M3-Embedding involves the following technical contributions. We propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strat
    
[^2]: 基于投票的实例筛选方法：使用基于赞成票的多获胜者投票

    Data as voters: instance selection using approval-based multi-winner voting. (arXiv:2304.09995v1 [cs.LG])

    [http://arxiv.org/abs/2304.09995](http://arxiv.org/abs/2304.09995)

    该论文提出了一种基于赞成票的多获胜者投票的实例选择方法，通过代表性投票规则选择获胜者，并将其作为减少训练集的数据实例。

    

    我们提出了一种新的机器学习（或数据挖掘）中的实例筛选方法。我们的方法基于最近关于基于赞成票的多获胜者选举中代表性表征的结果。在我们的模型中，实例扮演选民和候选人的双重角色。每个训练集中的实例（作为选民）赞成其本地集合中的实例（扮演候选人的角色）（除自身以外的实例），这个概念在文献中已经存在。然后，我们使用代表性投票规则选择选举获胜者，并作为减少训练集中的数据实例保留。

    We present a novel approach to the instance selection problem in machine learning (or data mining). Our approach is based on recent results on (proportional) representation in approval-based multi-winner elections. In our model, instances play a double role as voters and candidates. Each instance in the training set (acting as a voter) approves of the instances (playing the role of candidates) belonging to its local set (except itself), a concept already existing in the literature. We then select the election winners using a representative voting rule, and such winners are the data instances kept in the reduced training set.
    
[^3]: GSplit: 通过分裂并行实现大规模图神经网络训练的扩展

    GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism. (arXiv:2303.13775v1 [cs.DC])

    [http://arxiv.org/abs/2303.13775](http://arxiv.org/abs/2303.13775)

    本文提出了一种新的并行小批量训练方法，即分裂并行，应用在图神经网络训练上，能有效缓解数据并行方法的性能瓶颈，同时在大规模图上的性能表现优越。

    

    在许多行业、科学和工程领域（如推荐系统、社交图分析、知识库、材料科学和生物学）中，拥有数十亿个边的大规模图形是普遍存在的。图神经网络（GNN）作为一种新兴的机器学习模型，由于在各种图分析任务中具有卓越的性能，因此越来越多地被采用来学习这些图形。在大型图形上训练通常采用小批量训练，并且数据并行是将小批量训练扩展到多个 GPU 的标准方法。本文认为，GNN 训练系统的几个基本性能瓶颈与数据并行方法的固有限制有关。我们提出了一种新的并行小批量训练范式- 分裂并行，并将其实现在一个名为gsplit的新系统中。实验表明，gsplit 的性能优于DGL、Quiver和PaGraph等现有的系统。

    Large-scale graphs with billions of edges are ubiquitous in many industries, science, and engineering fields such as recommendation systems, social graph analysis, knowledge base, material science, and biology. Graph neural networks (GNN), an emerging class of machine learning models, are increasingly adopted to learn on these graphs due to their superior performance in various graph analytics tasks. Mini-batch training is commonly adopted to train on large graphs, and data parallelism is the standard approach to scale mini-batch training to multiple GPUs. In this paper, we argue that several fundamental performance bottlenecks of GNN training systems have to do with inherent limitations of the data parallel approach. We then propose split parallelism, a novel parallel mini-batch training paradigm. We implement split parallelism in a novel system called gsplit and show that it outperforms state-of-the-art systems such as DGL, Quiver, and PaGraph.
    

