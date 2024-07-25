# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models Enhanced Collaborative Filtering](https://arxiv.org/abs/2403.17688) | 提出了大型语言模型增强协同过滤（LLM-CF）框架，通过LLMs的世界知识和推理能力来提供更好的协同过滤信息 |
| [^2] | [Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models](https://arxiv.org/abs/2402.07179) | 本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。 |
| [^3] | [Heterophily-Aware Fair Recommendation using Graph Convolutional Networks](https://arxiv.org/abs/2402.03365) | 本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。 |
| [^4] | [Retrieving Texts based on Abstract Descriptions.](http://arxiv.org/abs/2305.12517) | 本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。 |

# 详细

[^1]: 大型语言模型增强协同过滤

    Large Language Models Enhanced Collaborative Filtering

    [https://arxiv.org/abs/2403.17688](https://arxiv.org/abs/2403.17688)

    提出了大型语言模型增强协同过滤（LLM-CF）框架，通过LLMs的世界知识和推理能力来提供更好的协同过滤信息

    

    最近，大型语言模型（LLM）的最新进展吸引了研究人员的广泛兴趣，他们利用这些模型来增强推荐系统（RS）。现有工作主要利用LLM生成知识丰富的文本，或者利用LLM衍生的嵌入作为特征来改进RS。虽然LLM中包含的广泛世界知识通常有利于RS，但该应用只能接受有限数量的用户和项目作为输入，没有充分利用协同过滤信息。考虑到协同过滤在RS中的关键作用，利用LLM增强RS的一个关键挑战在于通过LLM提供更好的协同过滤信息。在本文中，我们从LLM中的上下文学习和思维链推理中汲取灵感，提出了大型语言模型增强协同过滤（LLM-CF）框架，该框架提炼了LLMs的世界知识和推理能力

    arXiv:2403.17688v1 Announce Type: new  Abstract: Recent advancements in Large Language Models (LLMs) have attracted considerable interest among researchers to leverage these models to enhance Recommender Systems (RSs). Existing work predominantly utilizes LLMs to generate knowledge-rich texts or utilizes LLM-derived embeddings as features to improve RSs. Al- though the extensive world knowledge embedded in LLMs generally benefits RSs, the application can only take limited number of users and items as inputs, without adequately exploiting collaborative filtering information. Considering its crucial role in RSs, one key challenge in enhancing RSs with LLMs lies in providing better collaborative filtering information through LLMs. In this paper, drawing inspiration from the in-context learning and chain of thought reasoning in LLMs, we propose the Large Language Models enhanced Collaborative Filtering (LLM-CF) framework, which distils the world knowledge and reasoning capabilities of LLMs
    
[^2]: 在基于检索增强生成的大型语言模型中进行提示扰动

    Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models

    [https://arxiv.org/abs/2402.07179](https://arxiv.org/abs/2402.07179)

    本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。

    

    大型语言模型（LLM）的鲁棒性在其在各个领域的使用迅速增长中变得越来越重要。检索增强生成（RAG）被视为提高从LLM生成文本的可信度的方法。然而，目前对RAG-based LLMs的输出如何受到稍有不同的输入影响的研究还不够充分。在本文中，我们发现即使在提示中插入一个很短的前缀也会导致生成的输出与事实正确答案相去甚远。我们系统地评估了这类前缀对RAG的影响，并引入了一种称为Gradient Guided Prompt Perturbation（GGPP）的新型优化技术。GGPP在将RAG-based LLMs的输出引导到特定错误答案方面取得了很高的成功率。它还可以应对提示中请求忽略无关上下文的指令。我们还利用LLMs在带有和不带有GGPP扰动的提示之间的神经元激活差异来提供一种改进方法。

    The robustness of large language models (LLMs) becomes increasingly important as their use rapidly grows in a wide range of domains. Retrieval-Augmented Generation (RAG) is considered as a means to improve the trustworthiness of text generation from LLMs. However, how the outputs from RAG-based LLMs are affected by slightly different inputs is not well studied. In this work, we find that the insertion of even a short prefix to the prompt leads to the generation of outputs far away from factually correct answers. We systematically evaluate the effect of such prefixes on RAG by introducing a novel optimization technique called Gradient Guided Prompt Perturbation (GGPP). GGPP achieves a high success rate in steering outputs of RAG-based LLMs to targeted wrong answers. It can also cope with instructions in the prompts requesting to ignore irrelevant context. We also exploit LLMs' neuron activation difference between prompts with and without GGPP perturbations to give a method that improves
    
[^3]: 利用图卷积网络的異质友善推荐方法

    Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

    [https://arxiv.org/abs/2402.03365](https://arxiv.org/abs/2402.03365)

    本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。

    

    近年来，图神经网络（GNNs）已成为提高推荐系统准确性和性能的流行工具。现代推荐系统不仅设计为为最终用户服务，还要让其他参与者（如项目和项目供应商）从中受益。这些参与者可能具有不同或冲突的目标和利益，这引发了对公平性和流行度偏差考虑的需求。基于GNN的推荐方法也面临不公平性和流行度偏差的挑战，其归一化和聚合过程受到这些挑战的影响。在本文中，我们提出了一种公平的基于GNN的推荐系统，称为HetroFair，旨在提高项目侧的公平性。HetroFair使用两个独立的组件生成具有公平性意识的嵌入：i）公平注意力，它在GNN的归一化过程中结合了点积，以减少节点度数的影响；ii）异质性特征加权，为不同的特征分配不同的权重。

    In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items' side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes' degrees, and ii) heterophily feature weighting to assign distinct weights to 
    
[^4]: 基于摘要描述的文本检索

    Retrieving Texts based on Abstract Descriptions. (arXiv:2305.12517v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12517](http://arxiv.org/abs/2305.12517)

    本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。

    

    虽然针对文本的信息提取，指令优化的大型语言模型表现优异，但对于在大规模文档集合中定位符合给定描述的文本（语义检索）并不适用。基于嵌入向量的相似度搜索可以通过查询执行检索，但嵌入中的相似度定义不明确且不一致，并且对于许多用例来说都是次优的。那么，什么是有效检索的好的查询表示？我们确定了根据内容的摘要描述检索句子的明确定义且一致的任务。我们展示了当前文本嵌入的不足，并提出了一种替代模型，在标准最近邻搜索中的表现显著提升。该模型使用通过提示LLM获得的正负样本对进行训练。虽然很容易从LLM中获得训练材料，但LLM无法直接执行检索任务。

    While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?  We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This de
    

