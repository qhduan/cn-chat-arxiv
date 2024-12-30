# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Model Interaction Simulator for Cold-Start Item Recommendation](https://arxiv.org/abs/2402.09176) | 我们提出了一个大型语言模型交互模拟器 (LLM-InS)，用于解决冷启动物品推荐的问题。该模拟器能够模拟出逼真的交互，并将冷启动物品转化为热门物品，从而提高推荐性能。 |
| [^2] | [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | 本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。 |

# 详细

[^1]: 大型语言模型交互模拟器用于冷启动物品推荐

    Large Language Model Interaction Simulator for Cold-Start Item Recommendation

    [https://arxiv.org/abs/2402.09176](https://arxiv.org/abs/2402.09176)

    我们提出了一个大型语言模型交互模拟器 (LLM-InS)，用于解决冷启动物品推荐的问题。该模拟器能够模拟出逼真的交互，并将冷启动物品转化为热门物品，从而提高推荐性能。

    

    推荐冷启动物品对协同过滤模型来说是个长期的挑战，因为这些物品缺乏历史用户交互以建模他们的协同特性。冷启动物品的内容与行为模式之间的差距使得很难为其生成准确的行为嵌入。现有的冷启动模型使用映射函数基于冷启动物品的内容特征生成虚假的行为嵌入。然而，这些生成的嵌入与真实的行为嵌入存在显著的差异，对冷启动推荐性能产生负面影响。为了解决这个挑战，我们提出了一个基于内容方面来模拟用户行为模式的LLM交互模拟器 (LLM-InS)。该模拟器允许推荐系统为每个冷启动物品模拟生动的交互，并将其直接从冷启动物品转化为热门物品。

    arXiv:2402.09176v1 Announce Type: new Abstract: Recommending cold items is a long-standing challenge for collaborative filtering models because these cold items lack historical user interactions to model their collaborative features. The gap between the content of cold items and their behavior patterns makes it difficult to generate accurate behavioral embeddings for cold items. Existing cold-start models use mapping functions to generate fake behavioral embeddings based on the content feature of cold items. However, these generated embeddings have significant differences from the real behavioral embeddings, leading to a negative impact on cold recommendation performance. To address this challenge, we propose an LLM Interaction Simulator (LLM-InS) to model users' behavior patterns based on the content aspect. This simulator allows recommender systems to simulate vivid interactions for each cold item and transform them from cold to warm items directly. Specifically, we outline the desig
    
[^2]: LLMs用于知识图谱构建和推理：最新功能与未来机遇

    LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

    [https://arxiv.org/abs/2305.13168](https://arxiv.org/abs/2305.13168)

    本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。

    

    本文对大规模语言模型（LLMs）在知识图谱（KG）构建和推理中的数量化和质化评估进行了详尽的研究。我们在八个不同的数据集上进行了实验，重点关注涵盖实体和关系提取、事件提取、链接预测和问答四个典型任务，从而全面探索了LLMs在构建和推理领域的表现。经验性研究发现，以GPT-4为代表的LLMs更适合作为推理助手，而不是少样本信息提取器。具体而言，虽然GPT-4在与KG构建相关的任务中表现出色，但在推理任务中表现更出色，在某些情况下超越了精调模型。此外，我们的调查还扩展到LLMs在信息提取方面的潜在泛化能力，提出了虚拟知识提取的构想。

    arXiv:2305.13168v2 Announce Type: replace-cross  Abstract: This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extr
    

