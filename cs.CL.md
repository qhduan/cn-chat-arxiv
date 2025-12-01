# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey](https://arxiv.org/abs/2403.01528) | 生物分子与自然语言相结合的多模态学习为全面表示和分析生物分子开辟了新途径。 |
| [^2] | [Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization](https://arxiv.org/abs/2402.11414) | 提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。 |
| [^3] | [Extensible Multi-Granularity Fusion Network for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2402.07787) | 这篇论文提出了一种可扩展的多粒度融合网络（EMGF）用于基于方面的情感分析，通过整合不同的语言和结构特征，包括句法依赖、组成、注意力语义和外部知识图谱等，来提高情感分析的性能和准确性。 |
| [^4] | [Enhancing Continual Learning with Global Prototypes: Counteracting Negative Representation Drift.](http://arxiv.org/abs/2205.12186) | 该论文提出了一种基于全局原型的持续学习方法，在自监督信息的正则化下学习数据表示，以缓解负面表示漂移问题，并减少持续学习中的灾难性遗忘。 |
| [^5] | [A Trio Neural Model for Dynamic Entity Relatedness Ranking.](http://arxiv.org/abs/1808.08316) | 这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。 |

# 详细

[^1]: 利用生物分子和自然语言的多模态学习：一项综述

    Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey

    [https://arxiv.org/abs/2403.01528](https://arxiv.org/abs/2403.01528)

    生物分子与自然语言相结合的多模态学习为全面表示和分析生物分子开辟了新途径。

    

    集成生物分子建模与自然语言（BL）已经成为人工智能、化学和生物学交叉领域中的一个具有前景的跨学科领域。这种方法利用文本数据源中包含的生物分子的丰富多面描述，增强我们对基本理解，并实现生物分子性质预测等计算任务。通过将自然语言中表达的微妙叙述与通过各种分子建模技术描述的生物分子的结构和功能细节融合，打开了全面表征和分析生物分子的新途径。通过将围绕生物分子的上下文语言数据纳入建模中，BL旨在捕捉包含语言传达的符号特性以及数量化结构特征的整体视图。

    arXiv:2403.01528v1 Announce Type: cross  Abstract: The integration of biomolecular modeling with natural language (BL) has emerged as a promising interdisciplinary area at the intersection of artificial intelligence, chemistry and biology. This approach leverages the rich, multifaceted descriptions of biomolecules contained within textual data sources to enhance our fundamental understanding and enable downstream computational tasks such as biomolecule property prediction. The fusion of the nuanced narratives expressed through natural language with the structural and functional specifics of biomolecules described via various molecular modeling techniques opens new avenues for comprehensively representing and analyzing biomolecules. By incorporating the contextual language data that surrounds biomolecules into their modeling, BL aims to capture a holistic view encompassing both the symbolic qualities conveyed through language as well as quantitative structural characteristics. In this r
    
[^2]: 用于多模态摘要的细粒度可解释事实评估

    Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization

    [https://arxiv.org/abs/2402.11414](https://arxiv.org/abs/2402.11414)

    提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。

    

    多模态摘要旨在生成基于输入文本和图像的简洁摘要。然而，现有方法可能存在事实性输出的问题。为了评估多模态摘要模型的事实性，我们提出了两种细粒度和可解释的评估框架（FALLACIOUS）用于不同的应用场景，即基于参考的事实性评估框架和无参考的事实性评估框架。值得注意的是，无参考事实性评估框架不需要基准真值，因此具有更广泛的应用场景。为了评估所提出框架的有效性，我们计算了我们的框架与其他指标之间的相关性。实验结果显示了我们提出方法的有效性。我们将通过GitHub发布我们的代码和数据集。

    arXiv:2402.11414v1 Announce Type: new  Abstract: Multimodal summarization aims to generate a concise summary based on the input text and image. However, the existing methods potentially suffer from unfactual output. To evaluate the factuality of multimodal summarization models, we propose two fine-grained and explainable evaluation frameworks (FALLACIOUS) for different application scenarios, i.e. reference-based factuality evaluation framework and reference-free factuality evaluation framework. Notably, the reference-free factuality evaluation framework doesn't need ground truth and hence it has a wider application scenario. To evaluate the effectiveness of the proposed frameworks, we compute the correlation between our frameworks and the other metrics. The experimental results show the effectiveness of our proposed method. We will release our code and dataset via github.
    
[^3]: 可扩展的多粒度融合网络用于基于方面的情感分析

    Extensible Multi-Granularity Fusion Network for Aspect-based Sentiment Analysis

    [https://arxiv.org/abs/2402.07787](https://arxiv.org/abs/2402.07787)

    这篇论文提出了一种可扩展的多粒度融合网络（EMGF）用于基于方面的情感分析，通过整合不同的语言和结构特征，包括句法依赖、组成、注意力语义和外部知识图谱等，来提高情感分析的性能和准确性。

    

    基于方面的情感分析（ABSA）评估文本中的情感表达以理解情感信息。先前的研究整合了外部知识，如知识图谱，以加强ABSA模型中的语义特征。最近的研究探讨了在依赖和组成树上使用图神经网络（GNN）进行句法分析。随着ABSA的不断发展，越来越多的创新的语言和结构特征被融入其中（例如潜在图），但这也引入了复杂性和混淆。目前，尚不存在一个可扩展的框架，可以将多样性的语言和结构特征集成到ABSA中。本文介绍了可扩展的多粒度融合（EMGF）网络，它整合了来自句法依赖和组成、注意力语义和外部知识图谱的信息。EMGF配备了多锚点三元学习和正交投影，高效地利用了这些特征的综合潜力。

    Aspect-based Sentiment Analysis (ABSA) evaluates sentiment expressions within a text to comprehend sentiment information. Previous studies integrated external knowledge, such as knowledge graphs, to enhance the semantic features in ABSA models. Recent research has examined the use of Graph Neural Networks (GNNs) on dependency and constituent trees for syntactic analysis. With the ongoing development of ABSA, more innovative linguistic and structural features are being incorporated (e.g. latent graph), but this also introduces complexity and confusion. As of now, a scalable framework for integrating diverse linguistic and structural features into ABSA does not exist. This paper presents the Extensible Multi-Granularity Fusion (EMGF) network, which integrates information from dependency and constituent syntactic, attention semantic , and external knowledge graphs. EMGF, equipped with multi-anchor triplet learning and orthogonal projection, efficiently harnesses the combined potential of 
    
[^4]: 基于全局原型的增强持续学习: 对抗负表示漂移

    Enhancing Continual Learning with Global Prototypes: Counteracting Negative Representation Drift. (arXiv:2205.12186v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12186](http://arxiv.org/abs/2205.12186)

    该论文提出了一种基于全局原型的持续学习方法，在自监督信息的正则化下学习数据表示，以缓解负面表示漂移问题，并减少持续学习中的灾难性遗忘。

    

    持续学习旨在学习一系列任务，其中数据分布从一个任务转移到另一个任务。在训练新任务数据时，旧任务的数据表示可能会漂移。一些负面的表示漂移可能会导致灾难性遗忘，因为会导致从本地学习的类别原型和数据表示在任务之间的相关性较差。为了缓解这种表示漂移，我们提出一种方法，通过全局原型指导学习，用自监督信息的正则化来学习数据表示。具体来说，对于NLP任务，我们将每个任务以屏蔽语言建模的方式进行公式化，并通过预训练的语言模型进行相邻注意机制学习任务。实验结果表明，我们提出的方法可以学习出具有较少表示漂移的相当一致的表示，并在不重新采样过去任务的数据的情况下显著减少持续学习中的灾难性遗忘。

    Continual learning (CL) aims to learn a sequence of tasks over time, with data distributions shifting from one task to another. When training on new task data, data representations from old tasks may drift. Some negative representation drift can result in catastrophic forgetting, by causing the locally learned class prototypes and data representations to correlate poorly across tasks. To mitigate such representation drift, we propose a method that finds global prototypes to guide the learning, and learns data representations with the regularization of the self-supervised information. Specifically, for NLP tasks, we formulate each task in a masked language modeling style, and learn the task via a neighbor attention mechanism over a pre-trained language model. Experimental results show that our proposed method can learn fairly consistent representations with less representation drift, and significantly reduce catastrophic forgetting in CL without resampling data from past tasks.
    
[^5]: 一种三元神经模型用于动态实体相关性排名

    A Trio Neural Model for Dynamic Entity Relatedness Ranking. (arXiv:1808.08316v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/1808.08316](http://arxiv.org/abs/1808.08316)

    这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。

    

    测量实体相关性是许多自然语言处理和信息检索应用的基本任务。之前的研究通常在静态设置和非监督方式下研究实体相关性。然而，现实世界中的实体往往涉及许多不同的关系，因此实体关系随时间变得非常动态。在这项工作中，我们提出了一种基于神经网络的方法来动态评估实体相关性，利用集体注意力作为监督。我们的模型能够在联合框架中学习丰富而不同的实体表示。通过对大规模数据集的广泛实验，我们证明了我们的方法比竞争基线获得了更好的结果。

    Measuring entity relatedness is a fundamental task for many natural language processing and information retrieval applications. Prior work often studies entity relatedness in static settings and an unsupervised manner. However, entities in real-world are often involved in many different relationships, consequently entity-relations are very dynamic over time. In this work, we propose a neural networkbased approach for dynamic entity relatedness, leveraging the collective attention as supervision. Our model is capable of learning rich and different entity representations in a joint framework. Through extensive experiments on large-scale datasets, we demonstrate that our method achieves better results than competitive baselines.
    

