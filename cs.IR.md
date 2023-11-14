# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph.](http://arxiv.org/abs/2310.16452) | 本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。 |
| [^2] | [AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models.](http://arxiv.org/abs/2307.11772) | AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。 |
| [^3] | [Personalized Elastic Embedding Learning for On-Device Recommendation.](http://arxiv.org/abs/2306.10532) | 本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架考虑了设备和用户的异质性与动态资源约束，并在一次性生成个性化嵌入的基础上进行推荐。 |
| [^4] | [Visualization for Recommendation Explainability: A Survey and New Perspectives.](http://arxiv.org/abs/2305.11755) | 本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。 |
| [^5] | [SciRepEval: A Multi-Format Benchmark for Scientific Document Representations.](http://arxiv.org/abs/2211.13308) | SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。 |

# 详细

[^1]: 可解释的基于路径的知识图推荐中的忠实路径语言建模

    Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph. (arXiv:2310.16452v1 [cs.IR])

    [http://arxiv.org/abs/2310.16452](http://arxiv.org/abs/2310.16452)

    本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。

    

    针对知识图谱中的路径推理方法在提高推荐系统透明度方面的潜力，本文提出了一种名为PEARLM的新方法，该方法通过语言建模有效捕获用户行为和产品端知识。我们的方法通过语言模型直接从知识图谱上的路径中学习知识图谱嵌入，并将实体和关系统一在同一优化空间中。序列解码的约束保证了路径对知识图谱的忠实性。在两个数据集上的实验证明了我们方法与现有最先进方法的有效性。

    Path reasoning methods over knowledge graphs have gained popularity for their potential to improve transparency in recommender systems. However, the resulting models still rely on pre-trained knowledge graph embeddings, fail to fully exploit the interdependence between entities and relations in the KG for recommendation, and may generate inaccurate explanations. In this paper, we introduce PEARLM, a novel approach that efficiently captures user behaviour and product-side knowledge through language modelling. With our approach, knowledge graph embeddings are directly learned from paths over the KG by the language model, which also unifies entities and relations in the same optimisation space. Constraints on the sequence decoding additionally guarantee path faithfulness with respect to the KG. Experiments on two datasets show the effectiveness of our approach compared to state-of-the-art baselines. Source code and datasets: AVAILABLE AFTER GETTING ACCEPTED.
    
[^2]: AutoAlign：基于大型语言模型的全自动有效知识图谱对齐方法

    AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models. (arXiv:2307.11772v1 [cs.IR])

    [http://arxiv.org/abs/2307.11772](http://arxiv.org/abs/2307.11772)

    AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。

    

    知识图谱间的实体对齐任务旨在识别出两个不同知识图谱中表示相同实体的每对实体。许多基于机器学习的方法已被提出用于这个任务。然而，据我们所知，现有的方法都需要手工制作的种子对齐，这是非常昂贵的。在本文中，我们提出了第一个名为AutoAlign的完全自动对齐方法，它不需要任何手工制作的种子对齐。具体而言，对于谓词嵌入，AutoAlign使用大型语言模型构建谓词近邻图，自动捕捉两个知识图谱中谓词的相似性。对于实体嵌入，AutoAlign首先使用TransE独立计算每个知识图谱的实体嵌入，然后通过计算基于实体属性的实体相似性，将两个知识图谱的实体嵌入移动到相同的向量空间中。因此，AutoAlign实现了谓词对齐和实体对齐。

    The task of entity alignment between knowledge graphs (KGs) aims to identify every pair of entities from two different KGs that represent the same entity. Many machine learning-based methods have been proposed for this task. However, to our best knowledge, existing methods all require manually crafted seed alignments, which are expensive to obtain. In this paper, we propose the first fully automatic alignment method named AutoAlign, which does not require any manually crafted seed alignments. Specifically, for predicate embeddings, AutoAlign constructs a predicate-proximity-graph with the help of large language models to automatically capture the similarity between predicates across two KGs. For entity embeddings, AutoAlign first computes the entity embeddings of each KG independently using TransE, and then shifts the two KGs' entity embeddings into the same vector space by computing the similarity between entities based on their attributes. Thus, both predicate alignment and entity al
    
[^3]: 个性化弹性嵌入学习用于设备上的推荐

    Personalized Elastic Embedding Learning for On-Device Recommendation. (arXiv:2306.10532v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.10532](http://arxiv.org/abs/2306.10532)

    本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架考虑了设备和用户的异质性与动态资源约束，并在一次性生成个性化嵌入的基础上进行推荐。

    

    为了解决隐私问题并减少网络延迟，近年来一直有将在云端训练的臃肿的推荐模型压缩并在资源受限的设备上部署紧凑的推荐器模型以进行实时推荐的趋势。现有的解决方案通常忽视了设备异质性和用户异质性。它们要么要求所有设备共享相同的压缩模型，要么要求具有相同资源预算的设备共享相同模型。然而，即使是具有相同设备的用户可能也具有不同的偏好。此外，它们假设设备上的推荐器可用资源（如内存）是恒定的，这与现实情况不符。鉴于设备和用户的异质性以及动态资源约束，本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架以一次性方式为具有不同内存预算的设备生成个性化的嵌入。

    To address privacy concerns and reduce network latency, there has been a recent trend of compressing cumbersome recommendation models trained on the cloud and deploying compact recommender models to resource-limited devices for real-time recommendation. Existing solutions generally overlook device heterogeneity and user heterogeneity. They either require all devices to share the same compressed model or the devices with the same resource budget to share the same model. However, even users with the same devices may have different preferences. In addition, they assume the available resources (e.g., memory) for the recommender on a device are constant, which is not reflective of reality. In light of device and user heterogeneities as well as dynamic resource constraints, this paper proposes a Personalized Elastic Embedding Learning framework (PEEL) for on-device recommendation, which generates personalized embeddings for devices with various memory budgets in once-for-all manner, efficien
    
[^4]: 推荐系统解释的可视化：综述和新视角

    Visualization for Recommendation Explainability: A Survey and New Perspectives. (arXiv:2305.11755v1 [cs.IR])

    [http://arxiv.org/abs/2305.11755](http://arxiv.org/abs/2305.11755)

    本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。

    

    为推荐提供系统生成的解释是实现透明且值得信赖的推荐系统的重要步骤。可解释的推荐系统为输出提供了人类可理解的基础。在过去的20年中，可解释的推荐引起了推荐系统研究社区的广泛关注。本文旨在全面回顾推荐系统中有关可视化解释的研究工作。更具体地，我们根据解释目标、解释范围、解释样式和解释格式这四个维度系统地审查推荐系统中有关解释的文献。认识到可视化的重要性，我们从解释性视觉方式的角度途径推荐系统文献，即使用可视化作为解释的显示样式。因此，我们得出了一组可能有益于设计解释性可视化推荐系统的指南。

    Providing system-generated explanations for recommendations represents an important step towards transparent and trustworthy recommender systems. Explainable recommender systems provide a human-understandable rationale for their outputs. Over the last two decades, explainable recommendation has attracted much attention in the recommender systems research community. This paper aims to provide a comprehensive review of research efforts on visual explanation in recommender systems. More concretely, we systematically review the literature on explanations in recommender systems based on four dimensions, namely explanation goal, explanation scope, explanation style, and explanation format. Recognizing the importance of visualization, we approach the recommender system literature from the angle of explanatory visualizations, that is using visualizations as a display style of explanation. As a result, we derive a set of guidelines that might be constructive for designing explanatory visualizat
    
[^5]: SciRepEval：一个用于科学文献表示的多格式基准

    SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. (arXiv:2211.13308v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.13308](http://arxiv.org/abs/2211.13308)

    SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。

    

    学习的科学文献表示可以作为下游任务的有价值输入特征，无需进一步微调。然而，用于评估这些表示的现有基准未能捕捉到相关任务的多样性。为此，我们介绍了 SciRepEval，第一个用于训练和评估科学文献表示的全面基准。它包括四种格式的 25 个具有挑战性和现实性的任务，其中 11 个是新任务：分类、回归、排名和搜索。我们使用该基准来研究和改进科学文档表示模型的泛化能力。我们展示了最先进的模型如何在任务格式方面缺乏泛化性能，简单的多任务训练也不能改进它们。然而，一种新的方法，学习每个文档的多个嵌入，每个嵌入专门针对不同的格式，可以提高性能。我们尝试使用任务格式特定的控制代码和适配器。

    Learned representations of scientific documents can serve as valuable input features for downstream tasks, without the need for further fine-tuning. However, existing benchmarks for evaluating these representations fail to capture the diversity of relevant tasks. In response, we introduce SciRepEval, the first comprehensive benchmark for training and evaluating scientific document representations. It includes 25 challenging and realistic tasks, 11 of which are new, across four formats: classification, regression, ranking and search. We then use the benchmark to study and improve the generalization ability of scientific document representation models. We show how state-of-the-art models struggle to generalize across task formats, and that simple multi-task training fails to improve them. However, a new approach that learns multiple embeddings per document, each tailored to a different format, can improve performance. We experiment with task-format-specific control codes and adapters in 
    

