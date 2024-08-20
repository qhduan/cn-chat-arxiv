# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pattern-wise Transparent Sequential Recommendation](https://arxiv.org/abs/2402.11480) | 提出了一种模式透明的顺序推荐框架，通过将项目序列分解为多级模式并在概率空间中量化每个模式对结果的贡献，实现了透明的决策过程。 |
| [^2] | [Towards A Unified View of Answer Calibration for Multi-Step Reasoning](https://arxiv.org/abs/2311.09101) | 本文总结了最近答案校准技术的分类法，从统一视角对步级和路径级答案校准进行了彻底评估，结果显示整合两种策略的优势倾向于产生最佳结果。 |
| [^3] | [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | 本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。 |
| [^4] | [Linguistic and Structural Basis of Engineering Design Knowledge.](http://arxiv.org/abs/2312.06355) | 本文通过分析33881份专利文件的样本，将工程设计知识阐释为知识图谱，从而揭示工程设计知识的语言和结构基础。 |
| [^5] | [IncDSI: Incrementally Updatable Document Retrieval.](http://arxiv.org/abs/2307.10323) | IncDSI是一种递增可更新的文档检索方法，它通过最小改变网络参数的约束优化问题，实现实时添加文档而无需重新训练整个模型，具有与重新训练模型相竞争的速度，能够实时更新的文档检索系统的开发。 |
| [^6] | [Prompt Tuning on Graph-augmented Low-resource Text Classification.](http://arxiv.org/abs/2307.10230) | 本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。 |

# 详细

[^1]: 模式透明的顺序推荐

    Pattern-wise Transparent Sequential Recommendation

    [https://arxiv.org/abs/2402.11480](https://arxiv.org/abs/2402.11480)

    提出了一种模式透明的顺序推荐框架，通过将项目序列分解为多级模式并在概率空间中量化每个模式对结果的贡献，实现了透明的决策过程。

    

    透明的决策过程对于开发可靠和值得信赖的推荐系统至关重要。对于顺序推荐来说，意味着模型能够识别关键项目作为其推荐结果的理由。然而，同时实现模型透明度和推荐性能是具有挑战性的，特别是对于将整个项目序列作为输入而不加筛选的模型而言。在本文中，我们提出了一种名为PTSR的可解释框架，它实现了一种模式透明的决策过程。它将项目序列分解为多级模式，这些模式作为整个推荐过程的原子单元。每个模式对结果的贡献在概率空间中得到量化。通过精心设计的模式加权校正，即使在没有真实关键模式的情况下，也能学习模式的贡献。最终推荐

    arXiv:2402.11480v1 Announce Type: new  Abstract: A transparent decision-making process is essential for developing reliable and trustworthy recommender systems. For sequential recommendation, it means that the model can identify critical items asthe justifications for its recommendation results. However, achieving both model transparency and recommendation performance simultaneously is challenging, especially for models that take the entire sequence of items as input without screening. In this paper,we propose an interpretable framework (named PTSR) that enables a pattern-wise transparent decision-making process. It breaks the sequence of items into multi-level patterns that serve as atomic units for the entire recommendation process. The contribution of each pattern to the outcome is quantified in the probability space. With a carefully designed pattern weighting correction, the pattern contribution can be learned in the absence of ground-truth critical patterns. The final recommended
    
[^2]: 朝向多步推理的答案校准统一视图

    Towards A Unified View of Answer Calibration for Multi-Step Reasoning

    [https://arxiv.org/abs/2311.09101](https://arxiv.org/abs/2311.09101)

    本文总结了最近答案校准技术的分类法，从统一视角对步级和路径级答案校准进行了彻底评估，结果显示整合两种策略的优势倾向于产生最佳结果。

    

    大型语言模型（LLMs）使用“思维链”提示扩展了改进多步推理能力的范围。我们通常将多步推理分为两个阶段：路径生成以生成推理路径；和答案校准后处理推理路径以获得最终答案。然而，现有文献缺乏对不同答案校准方法的系统分析。本文总结了最近答案校准技术的分类法，并将其分解为步级和路径级策略。然后，我们从统一视角对这些策略进行了彻底评估，系统地审查了多路径上的步级和路径级答案校准。实验结果表明，整合两种策略的优势倾向于产生最佳结果。我们的研究有可能启示优化多步推理系统的关键见解。

    arXiv:2311.09101v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) employing Chain-of-Thought (CoT) prompting have broadened the scope for improving multi-step reasoning capabilities. We generally divide multi-step reasoning into two phases: path generation to generate the reasoning path(s); and answer calibration post-processing the reasoning path(s) to obtain a final answer. However, the existing literature lacks systematic analysis on different answer calibration approaches. In this paper, we summarize the taxonomy of recent answer calibration techniques and break them down into step-level and path-level strategies. We then conduct a thorough evaluation on these strategies from a unified view, systematically scrutinizing step-level and path-level answer calibration across multiple paths. Experimental results reveal that integrating the dominance of both strategies tends to derive optimal outcomes. Our study holds the potential to illuminate key insights for opti
    
[^3]: LLMs用于知识图谱构建和推理：最新功能与未来机遇

    LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

    [https://arxiv.org/abs/2305.13168](https://arxiv.org/abs/2305.13168)

    本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。

    

    本文对大规模语言模型（LLMs）在知识图谱（KG）构建和推理中的数量化和质化评估进行了详尽的研究。我们在八个不同的数据集上进行了实验，重点关注涵盖实体和关系提取、事件提取、链接预测和问答四个典型任务，从而全面探索了LLMs在构建和推理领域的表现。经验性研究发现，以GPT-4为代表的LLMs更适合作为推理助手，而不是少样本信息提取器。具体而言，虽然GPT-4在与KG构建相关的任务中表现出色，但在推理任务中表现更出色，在某些情况下超越了精调模型。此外，我们的调查还扩展到LLMs在信息提取方面的潜在泛化能力，提出了虚拟知识提取的构想。

    arXiv:2305.13168v2 Announce Type: replace-cross  Abstract: This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extr
    
[^4]: 工程设计知识的语言和结构基础

    Linguistic and Structural Basis of Engineering Design Knowledge. (arXiv:2312.06355v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.06355](http://arxiv.org/abs/2312.06355)

    本文通过分析33881份专利文件的样本，将工程设计知识阐释为知识图谱，从而揭示工程设计知识的语言和结构基础。

    

    物品描述是工程设计知识的主要载体，既是设计过程的产物，也是驱动设计过程的因素。尽管物品可以以不同的内涵进行描述，但设计过程需要一种描述来体现工程设计知识，这通过实体和关系的复杂安排在文本中表现出来。虽然大型语言模型可以从各种文本中学习，但它们尚未生成体现明确的工程设计事实的文本。现有的本体论设计理论很少能指导目前仅限于构思和学习目的的大型语言模型的应用。本文从33881份专利文件的大样本中将工程设计知识阐释为知识图谱。我们研究这些知识图谱的组成部分，以理解工程设计知识的语言和结构基础。

    Artefact descriptions are the primary carriers of engineering design knowledge that is both an outcome and a driver of the design process. While an artefact could be described in different connotations, the design process requires a description to embody engineering design knowledge, which is expressed in the text through intricate placement of entities and relationships. As large-language models learn from all kinds of text merely as a sequence of characters/tokens, these are yet to generate text that embodies explicit engineering design facts. Existing ontological design theories are less likely to guide the large-language models whose applications are currently limited to ideation and learning purposes. In this article, we explicate engineering design knowledge as knowledge graphs from a large sample of 33,881 patent documents. We examine the constituents of these knowledge graphs to understand the linguistic and structural basis of engineering design knowledge. In terms of linguist
    
[^5]: IncDSI：递增可更新的文档检索

    IncDSI: Incrementally Updatable Document Retrieval. (arXiv:2307.10323v1 [cs.IR])

    [http://arxiv.org/abs/2307.10323](http://arxiv.org/abs/2307.10323)

    IncDSI是一种递增可更新的文档检索方法，它通过最小改变网络参数的约束优化问题，实现实时添加文档而无需重新训练整个模型，具有与重新训练模型相竞争的速度，能够实时更新的文档检索系统的开发。

    

    不同iable搜索索引是最近提出的一种文档检索范例，它将文档语料库的信息编码在神经网络的参数中，并直接将查询映射到相应的文档。这些模型在许多基准测试中取得了最先进的性能。这些模型具有一个重要限制：在训练模型之后添加新文档并不容易。我们提出了IncDSI，一种实时添加文档的方法（每个文档约20-50毫秒），而无需对整个数据集（甚至部分数据集）重新训练模型。相反，我们将添加文档的过程形式化为一个在网络参数上进行最小改变的约束优化问题。虽然速度更快几个数量级，但我们的方法与在整个数据集上重新训练模型相竞争，并且可以实时更新的文档检索系统的开发。我们的IncDSI代码

    Differentiable Search Index is a recently proposed paradigm for document retrieval, that encodes information about a corpus of documents within the parameters of a neural network and directly maps queries to corresponding documents. These models have achieved state-of-the-art performances for document retrieval across many benchmarks. These kinds of models have a significant limitation: it is not easy to add new documents after a model is trained. We propose IncDSI, a method to add documents in real time (about 20-50ms per document), without retraining the model on the entire dataset (or even parts thereof). Instead we formulate the addition of documents as a constrained optimization problem that makes minimal changes to the network parameters. Although orders of magnitude faster, our approach is competitive with re-training the model on the whole dataset and enables the development of document retrieval systems that can be updated with new information in real-time. Our code for IncDSI
    
[^6]: 基于图增强的低资源文本分类的Prompt调优

    Prompt Tuning on Graph-augmented Low-resource Text Classification. (arXiv:2307.10230v1 [cs.IR])

    [http://arxiv.org/abs/2307.10230](http://arxiv.org/abs/2307.10230)

    本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。

    

    文本分类是信息检索中的一个基础问题，有许多实际应用，例如预测在线文章的主题和电子商务产品描述的类别。然而，低资源文本分类，即没有或只有很少标注样本的情况，对监督学习构成了严重问题。与此同时，许多文本数据本质上都建立在网络结构上，例如在线文章的超链接/引用网络和电子商务产品的用户-物品购买网络。这些图结构捕捉了丰富的语义关系，有助于增强低资源文本分类。在本文中，我们提出了一种名为Graph-Grounded Pre-training and Prompting (G2P2)的新模型，以两方面方法解决低资源文本分类问题。在预训练阶段，我们提出了三种基于图交互的对比策略，共同预训练图文模型；在下游分类阶段，我们探索了手工设计的提示信息对模型的影响。

    Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with no or few labeled samples, presents a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore handcrafted 
    

