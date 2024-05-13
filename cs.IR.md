# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation.](http://arxiv.org/abs/2401.14678) | 这项研究提出了一种提升Prompt的联邦内容表示学习方法，用于解决跨领域推荐中的隐私泄露和知识转移挑战。 |
| [^2] | [REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction.](http://arxiv.org/abs/2309.14891) | REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。 |
| [^3] | [Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies.](http://arxiv.org/abs/2309.13063) | 通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。 |
| [^4] | [A Unified Review of Deep Learning for Automated Medical Coding.](http://arxiv.org/abs/2201.02797) | 本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。 |

# 详细

[^1]: 提升Prompt的联邦内容表示学习用于跨领域推荐

    Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation. (arXiv:2401.14678v1 [cs.IR])

    [http://arxiv.org/abs/2401.14678](http://arxiv.org/abs/2401.14678)

    这项研究提出了一种提升Prompt的联邦内容表示学习方法，用于解决跨领域推荐中的隐私泄露和知识转移挑战。

    

    随着数据稀疏问题的缓解，跨领域推荐作为有效的技术已经在近年来得到广泛研究。然而，之前的研究工作可能会导致领域隐私泄露，因为它们在训练过程中需要将各个领域的数据聚合到一个中央服务器上。虽然一些研究通过联邦学习对隐私进行保护的跨领域推荐进行了研究，但仍存在以下限制：1）它们需要将用户的个人信息上传到中央服务器，存在用户隐私泄露的风险。2）现有的联邦方法主要依赖于原子项目ID来表示项目，这使它们无法在统一的特征空间中对项目进行建模，增加了领域之间的知识转移的挑战。3）它们都基于知道领域之间重叠用户的前提，这在实际应用中是不可行的。为了解决上述限制，我们着眼于隐私保护跨领域推荐。

    Cross-domain Recommendation (CDR) as one of the effective techniques in alleviating the data sparsity issues has been widely studied in recent years. However, previous works may cause domain privacy leakage since they necessitate the aggregation of diverse domain data into a centralized server during the training process. Though several studies have conducted privacy preserving CDR via Federated Learning (FL), they still have the following limitations: 1) They need to upload users' personal information to the central server, posing the risk of leaking user privacy. 2) Existing federated methods mainly rely on atomic item IDs to represent items, which prevents them from modeling items in a unified feature space, increasing the challenge of knowledge transfer among domains. 3) They are all based on the premise of knowing overlapped users between domains, which proves impractical in real-world applications. To address the above limitations, we focus on Privacy-preserving Cross-domain Reco
    
[^2]: REFORM: 移除CTR预测中的误关联的多级交互

    REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction. (arXiv:2309.14891v1 [cs.IR])

    [http://arxiv.org/abs/2309.14891](http://arxiv.org/abs/2309.14891)

    REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。

    

    点击率（CTR）预测是在线广告和推荐系统中的关键任务，准确的预测对于用户定位和个性化推荐至关重要。最近的一些前沿方法主要关注复杂的隐式和显式特征交互。然而，这些方法忽视了由混淆因子或选择偏差引起的误关联问题。这个问题在这些交互的复杂性和冗余性下变得更加严重。我们提出了一种CTR预测框架，称为REFORM，在多级特征交互中移除了误关联。所提出的REFORM框架通过两个流式叠加的循环结构利用了大量的多级高阶特征表示，并消除了误关联。该框架有两个关键组成部分：I. 多级叠加循环（MSR）结构使模型能够高效地捕捉到来自特征空间的多样非线性交互。

    Click-through rate (CTR) prediction is a critical task in online advertising and recommendation systems, as accurate predictions are essential for user targeting and personalized recommendations. Most recent cutting-edge methods primarily focus on investigating complex implicit and explicit feature interactions. However, these methods neglect the issue of false correlations caused by confounding factors or selection bias. This problem is further magnified by the complexity and redundancy of these interactions. We propose a CTR prediction framework that removes false correlation in multi-level feature interaction, termed REFORM. The proposed REFORM framework exploits a wide range of multi-level high-order feature representations via a two-stream stacked recurrent structure while eliminating false correlations. The framework has two key components: I. The multi-level stacked recurrent (MSR) structure enables the model to efficiently capture diverse nonlinear interactions from feature spa
    
[^3]: 使用大型语言模型生成、验证和应用用户意图分类方法

    Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies. (arXiv:2309.13063v1 [cs.IR])

    [http://arxiv.org/abs/2309.13063](http://arxiv.org/abs/2309.13063)

    通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。

    

    日志数据可以揭示用户与网络搜索服务的交互方式、用户的需求以及满意程度等宝贵信息。然而，分析日志数据中的用户意图并不容易，尤其是对于新的网络搜索形式，如人工智能驱动的聊天。为了理解日志数据中的用户意图，我们需要一种能够用有意义的分类方式标记它们的方法，以捕捉其多样性和动态性。现有的方法依赖于手动或基于机器学习的标注，这些方法对于大型且不断变化的数据集而言，要么代价高昂要么不够灵活。我们提出了一种使用大型语言模型(LLM)的新方法，这种模型能够生成丰富且相关的概念、描述和示例来表示用户意图。然而，使用LLM生成用户意图分类并将其应用于日志分析可能存在两个主要问题：这样的分类得不到外部验证，并且可能存在不良的反馈回路。为了克服这些问题，我们提出了一种新的方法，通过人工专家和评估者来验证。

    Log data can reveal valuable information about how users interact with web search services, what they want, and how satisfied they are. However, analyzing user intents in log data is not easy, especially for new forms of web search such as AI-driven chat. To understand user intents from log data, we need a way to label them with meaningful categories that capture their diversity and dynamics. Existing methods rely on manual or ML-based labeling, which are either expensive or inflexible for large and changing datasets. We propose a novel solution using large language models (LLMs), which can generate rich and relevant concepts, descriptions, and examples for user intents. However, using LLMs to generate a user intent taxonomy and apply it to do log analysis can be problematic for two main reasons: such a taxonomy is not externally validated, and there may be an undesirable feedback loop. To overcome these issues, we propose a new methodology with human experts and assessors to verify th
    
[^4]: 深度学习在自动医疗编码中的应用综述

    A Unified Review of Deep Learning for Automated Medical Coding. (arXiv:2201.02797v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2201.02797](http://arxiv.org/abs/2201.02797)

    本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。

    

    自动医疗编码是医疗运营和服务的基本任务，通过从临床文档中预测医疗编码来管理非结构化数据。近年来，深度学习和自然语言处理的进步已广泛应用于该任务。但基于深度学习的自动医疗编码缺乏对神经网络架构设计的统一视图。本综述提出了一个统一框架，以提供对医疗编码模型组件的一般理解，并总结了在此框架下最近的高级模型。我们的统一框架将医疗编码分解为四个主要组件，即用于文本特征提取的编码器模块、构建深度编码器架构的机制、用于将隐藏表示转换成医疗代码的解码器模块以及辅助信息的使用。最后，我们介绍了基准和真实世界中的使用情况，讨论了关键的研究挑战和未来方向。

    Automated medical coding, an essential task for healthcare operation and delivery, makes unstructured data manageable by predicting medical codes from clinical documents. Recent advances in deep learning and natural language processing have been widely applied to this task. However, deep learning-based medical coding lacks a unified view of the design of neural network architectures. This review proposes a unified framework to provide a general understanding of the building blocks of medical coding models and summarizes recent advanced models under the proposed framework. Our unified framework decomposes medical coding into four main components, i.e., encoder modules for text feature extraction, mechanisms for building deep encoder architectures, decoder modules for transforming hidden representations into medical codes, and the usage of auxiliary information. Finally, we introduce the benchmarks and real-world usage and discuss key research challenges and future directions.
    

