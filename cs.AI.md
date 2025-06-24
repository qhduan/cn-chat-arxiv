# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [No more optimization rules: LLM-enabled policy-based multi-modal query optimizer (version 1)](https://arxiv.org/abs/2403.13597) | LLM启用了基于策略的多模查询优化器，摆脱了传统的基于规则的优化方法，为查询优化带来全新的可能性。 |
| [^2] | [$L^*LM$: Learning Automata from Examples using Natural Language Oracles](https://arxiv.org/abs/2402.07051) | 该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。 |
| [^3] | [When Large Language Models Meet Vector Databases: A Survey](https://arxiv.org/abs/2402.01763) | 本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。 |
| [^4] | [Unveiling Molecular Moieties through Hierarchical Graph Explainability](https://arxiv.org/abs/2402.01744) | 本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。 |
| [^5] | [Do Concept Bottleneck Models Obey Locality?.](http://arxiv.org/abs/2401.01259) | 本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。 |
| [^6] | [Advective Diffusion Transformers for Topological Generalization in Graph Learning.](http://arxiv.org/abs/2310.06417) | 本研究探索了在不同的图拓扑存在下，图扩散方程如何对GNN进行外推和概括，揭示了基于局部扩散的现有模型在概括能力上的不足，并提出了非局部扩散的潜力。 |
| [^7] | [Human-AI Interactions and Societal Pitfalls.](http://arxiv.org/abs/2309.10448) | 本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。 |
| [^8] | [Latest Trends in Artificial Intelligence Technology: A Scoping Review.](http://arxiv.org/abs/2305.04532) | 本文对当前最先进的人工智能技术进行了范围评估，并要求对技术解决方案进行测试、使用公认数据集以及确保结果可复制。 |
| [^9] | [RPLKG: Robust Prompt Learning with Knowledge Graph.](http://arxiv.org/abs/2304.10805) | 本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。 |
| [^10] | [Indeterminate Probability Neural Network.](http://arxiv.org/abs/2303.11536) | 本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。 |

# 详细

[^1]: 不再有优化规则: 基于LLM的基于策略的多模查询优化器（版本1）

    No more optimization rules: LLM-enabled policy-based multi-modal query optimizer (version 1)

    [https://arxiv.org/abs/2403.13597](https://arxiv.org/abs/2403.13597)

    LLM启用了基于策略的多模查询优化器，摆脱了传统的基于规则的优化方法，为查询优化带来全新的可能性。

    

    大语言模型(LLM)在机器学习和深度学习领域标志着一个重要时刻。最近，人们研究了LLM在查询规划中的能力，包括单模和多模查询。然而，对于LLM的查询优化能力还没有相关研究。作为显著影响查询计划执行性能的关键步骤，不应错过这种分析和尝试。另一方面，现有的查询优化器通常是基于规则或基于规则+基于成本的，即它们依赖于人工创建的规则来完成查询计划重写/转换。鉴于现代优化器包括数百至数千条规则，按照类似方式设计一个多模查询优化器将耗费大量时间，因为我们将不得不列举尽可能多的多模优化规则，而这并没有。

    arXiv:2403.13597v1 Announce Type: cross  Abstract: Large language model (LLM) has marked a pivotal moment in the field of machine learning and deep learning. Recently its capability for query planning has been investigated, including both single-modal and multi-modal queries. However, there is no work on the query optimization capability of LLM. As a critical (or could even be the most important) step that significantly impacts the execution performance of the query plan, such analysis and attempts should not be missed. From another aspect, existing query optimizers are usually rule-based or rule-based + cost-based, i.e., they are dependent on manually created rules to complete the query plan rewrite/transformation. Given the fact that modern optimizers include hundreds to thousands of rules, designing a multi-modal query optimizer following a similar way is significantly time-consuming since we will have to enumerate as many multi-modal optimization rules as possible, which has not be
    
[^2]: $L^*LM$: 通过自然语言定义示例学习自动机

    $L^*LM$: Learning Automata from Examples using Natural Language Oracles

    [https://arxiv.org/abs/2402.07051](https://arxiv.org/abs/2402.07051)

    该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。

    

    专家演示已被证明是简化间接指定复杂任务的一种方法。最近的算法甚至支持从演示中提取明确的形式规范，如确定性有限自动机（DFA）。不幸的是，这些技术通常不具备高样本效率。在本文中，我们介绍了一种名为 $L^*LM$ 的算法，用于从演示和自然语言中学习 DFA。由于自然语言的表达能力，我们观察到从专家演示中学习 DFA 的数据效率显著提高。从技术上讲，$L^*LM$ 利用大型语言模型来回答关于底层任务的成员查询。然后将其与最近的演示学习技术相结合，将学习转化为一系列带标签示例学习问题。在我们的实验中，我们观察到这两种模态相互补充，从而产生了一个强大的少样本学习器。

    Expert demonstrations have proven an easy way to indirectly specify complex tasks. Recent algorithms even support extracting unambiguous formal specifications, e.g. deterministic finite automata (DFA), from demonstrations. Unfortunately, these techniques are generally not sample efficient. In this work, we introduce $L^*LM$, an algorithm for learning DFAs from both demonstrations and natural language. Due to the expressivity of natural language, we observe a significant improvement in the data efficiency of learning DFAs from expert demonstrations. Technically, $L^*LM$ leverages large language models to answer membership queries about the underlying task. This is then combined with recent techniques for transforming learning from demonstrations into a sequence of labeled example learning problems. In our experiments, we observe the two modalities complement each other, yielding a powerful few-shot learner.
    
[^3]: 当大型语言模型遇上向量数据库：一项综述

    When Large Language Models Meet Vector Databases: A Survey

    [https://arxiv.org/abs/2402.01763](https://arxiv.org/abs/2402.01763)

    本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。

    

    最近大型语言模型的突破在人类文字处理和生成方面开启了新的领域。然而，随着它们的显著增长，大型语言模型面临着包括幻觉、偏见、实时知识更新以及在商业环境中实施和维护的高成本等重要挑战。而另一种日益流行的工具，向量数据库则为这些挑战提供了潜在的解决方案。这些数据库擅长处理高维数据，并且对于高效的信息检索和语义搜索等任务至关重要。通过与大型语言模型的整合，它们显著增强了人工智能系统管理和更有效地利用多样数据的能力。本综述论文对大型语言模型和向量数据库之间的交叉点进行了深入而独特的分析。

    The recent burst in Large Language Models has opened new frontiers in human-like text processing and generation. However, alongside their remarkable growth, Large Language Models have encountered critical challenges including issues of hallucination, bias, real-time knowledge updates, and the high costs of implementation and maintenance in commercial settings. Vector Databases, another increasingly popular tool, offer potential solutions to these challenges. These databases are adept at handling high-dimensional data and are crucial for tasks such as efficient information retrieval and semantic search. By integrating with Large Language Models, they significantly enhance AI systems' ability to manage and utilize diverse data more effectively. This survey paper provides an in-depth and unique analysis of the intersection between Large Language Models and Vector Databases.
    
[^4]: 通过分层图解释揭示分子成分

    Unveiling Molecular Moieties through Hierarchical Graph Explainability

    [https://arxiv.org/abs/2402.01744](https://arxiv.org/abs/2402.01744)

    本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。

    

    背景：图神经网络（GNN）作为一种强大的工具，在支持体外虚拟筛选方面已经出现多年。在这项工作中，我们提出了一种使用图卷积架构实现高精度多靶标筛选的GNN。我们还设计了一种分层可解释人工智能（XAI）技术，通过利用信息传递机制，在原子、环和整个分子层面上直接捕获信息，从而找到与生物活性预测相关的最重要的成分。结果：我们在支持虚拟筛选方面的二十个细胞周期依赖性激酶靶标上报道了一种最先进的GNN分类器。我们的分类器超越了作者提出的先前最先进方法。此外，我们还设计了一个仅针对CDK1的高灵敏度版本的GNN，以使用我们的解释器来避免多类别模型固有的偏差。分层解释器已经由一位专家化学家在19个CDK1批准药物上进行了验证。

    Background: Graph Neural Networks (GNN) have emerged in very recent years as a powerful tool for supporting in silico Virtual Screening. In this work we present a GNN which uses Graph Convolutional architectures to achieve very accurate multi-target screening. We also devised a hierarchical Explainable Artificial Intelligence (XAI) technique to catch information directly at atom, ring, and whole molecule level by leveraging the message passing mechanism. In this way, we find the most relevant moieties involved in bioactivity prediction. Results: We report a state-of-the-art GNN classifier on twenty Cyclin-dependent Kinase targets in support of VS. Our classifier outperforms previous SOTA approaches proposed by the authors. Moreover, a CDK1-only high-sensitivity version of the GNN has been designed to use our explainer in order to avoid the inherent bias of multi-class models. The hierarchical explainer has been validated by an expert chemist on 19 approved drugs on CDK1. Our explainer 
    
[^5]: 概念瓶颈模型是否遵循局部性？

    Do Concept Bottleneck Models Obey Locality?. (arXiv:2401.01259v1 [cs.LG])

    [http://arxiv.org/abs/2401.01259](http://arxiv.org/abs/2401.01259)

    本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。

    

    概念基础学习通过解释其预测结果使用人可理解的概念，改善了深度学习模型的可解释性。在这种范式下训练的深度学习模型严重依赖于神经网络能够学习独立于其他概念的给定概念的存在或不存在。然而，最近的研究强烈暗示这种假设可能在概念瓶颈模型（CBMs）这一典型的基于概念的可解释架构中不能成立。本文中，我们研究了当这些概念既在空间上（通过它们的值完全由固定子集的特征定义）又在语义上（通过它们的值仅与预定义的固定子集的概念相关联）定位时，CBMs是否正确捕捉到概念之间的条件独立程度。为了理解局部性，我们分析了概念之外的特征变化对概念预测的影响。

    Concept-based learning improves a deep learning model's interpretability by explaining its predictions via human-understandable concepts. Deep learning models trained under this paradigm heavily rely on the assumption that neural networks can learn to predict the presence or absence of a given concept independently of other concepts. Recent work, however, strongly suggests that this assumption may fail to hold in Concept Bottleneck Models (CBMs), a quintessential family of concept-based interpretable architectures. In this paper, we investigate whether CBMs correctly capture the degree of conditional independence across concepts when such concepts are localised both spatially, by having their values entirely defined by a fixed subset of features, and semantically, by having their values correlated with only a fixed subset of predefined concepts. To understand locality, we analyse how changes to features outside of a concept's spatial or semantic locality impact concept predictions. Our
    
[^6]: 用于图学习中的拓扑概括的流动扩散变压器

    Advective Diffusion Transformers for Topological Generalization in Graph Learning. (arXiv:2310.06417v1 [cs.LG])

    [http://arxiv.org/abs/2310.06417](http://arxiv.org/abs/2310.06417)

    本研究探索了在不同的图拓扑存在下，图扩散方程如何对GNN进行外推和概括，揭示了基于局部扩散的现有模型在概括能力上的不足，并提出了非局部扩散的潜力。

    

    图扩散方程与图神经网络（GNN）密切相关，并且最近引起了人们的关注，作为分析GNN动力学、形式化其表达能力和证明架构选择的有原则的框架。图学习中的一个关键问题是GNN的概括能力。当前方法的一个主要限制在于假设训练集和测试集中的图拓扑来自相同的分布。本文通过探索图扩散方程在不同图拓扑存在下的外推和概括能力，迈出了解析GNN概括性的一步。我们首先展示了基于图上局部扩散的现有模型在概括能力上的不足，这是由于对拓扑变化的指数敏感性引起的。随后的分析揭示了非局部扩散的潜力，它倡导对完全连接的潜在图进行特征传播。

    Graph diffusion equations are intimately related to graph neural networks (GNNs) and have recently attracted attention as a principled framework for analyzing GNN dynamics, formalizing their expressive power, and justifying architectural choices. One key open questions in graph learning is the generalization capabilities of GNNs. A major limitation of current approaches hinges on the assumption that the graph topologies in the training and test sets come from the same distribution. In this paper, we make steps towards understanding the generalization of GNNs by exploring how graph diffusion equations extrapolate and generalize in the presence of varying graph topologies. We first show deficiencies in the generalization capability of existing models built upon local diffusion on graphs, stemming from the exponential sensitivity to topology variation. Our subsequent analysis reveals the promise of non-local diffusion, which advocates for feature propagation over fully-connected latent gr
    
[^7]: 人工智能与人类互动以及社会陷阱

    Human-AI Interactions and Societal Pitfalls. (arXiv:2309.10448v1 [cs.AI])

    [http://arxiv.org/abs/2309.10448](http://arxiv.org/abs/2309.10448)

    本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。

    

    当与生成式人工智能（AI）合作时，用户可能会看到生产力的提升，但AI生成的内容可能不完全符合他们的偏好。为了研究这种影响，我们引入了一个贝叶斯框架，其中异质用户选择与AI共享多少信息，面临输出保真度和通信成本之间的权衡。我们展示了这些个体决策与AI训练之间的相互作用可能导致社会挑战。输出可能变得更加同质化，特别是当AI在AI生成的内容上进行训练时。而任何AI的偏见可能成为社会偏见。解决同质化和偏见问题的办法是改进人工智能与人类的互动，实现个性化输出而不牺牲生产力。

    When working with generative artificial intelligence (AI), users may see productivity gains, but the AI-generated content may not match their preferences exactly. To study this effect, we introduce a Bayesian framework in which heterogeneous users choose how much information to share with the AI, facing a trade-off between output fidelity and communication cost. We show that the interplay between these individual-level decisions and AI training may lead to societal challenges. Outputs may become more homogenized, especially when the AI is trained on AI-generated content. And any AI bias may become societal bias. A solution to the homogenization and bias issues is to improve human-AI interactions, enabling personalized outputs without sacrificing productivity.
    
[^8]: 人工智能技术的最新趋势：一个范围评估研究

    Latest Trends in Artificial Intelligence Technology: A Scoping Review. (arXiv:2305.04532v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.04532](http://arxiv.org/abs/2305.04532)

    本文对当前最先进的人工智能技术进行了范围评估，并要求对技术解决方案进行测试、使用公认数据集以及确保结果可复制。

    

    人工智能技术已经广泛应用于多个领域。智能手机、社交媒体平台、搜索引擎和自主驾驶车辆等应用程序都利用了人工智能技术以提高其性能。本研究按照 PRISMA 框架对当前最先进的人工智能技术进行了范围评估。目标是寻找应用于不同领域人工智能技术研究的最先进技术。从人工智能和机器学习领域选取了三个知名期刊：《人工智能研究杂志》、《机器学习研究杂志》和《机器学习》，并观察了2022年发表的文章。对技术解决方案进行了一定的资格要求：技术必须针对可比较的解决方案进行测试，必须使用公认或其他充分证明的数据集进行应用，并确保结果可复制。

    Artificial intelligence is more ubiquitous in multiple domains. Smartphones, social media platforms, search engines, and autonomous vehicles are just a few examples of applications that utilize artificial intelligence technologies to enhance their performance. This study carries out a scoping review of the current state-of-the-art artificial intelligence technologies following the PRISMA framework. The goal was to find the most advanced technologies used in different domains of artificial intelligence technology research. Three recognized journals were used from artificial intelligence and machine learning domain: Journal of Artificial Intelligence Research, Journal of Machine Learning Research, and Machine Learning, and articles published in 2022 were observed. Certain qualifications were laid for the technological solutions: the technology must be tested against comparable solutions, commonly approved or otherwise well justified datasets must be used while applying, and results must 
    
[^9]: RPLKG: 基于知识图谱的鲁棒提示学习

    RPLKG: Robust Prompt Learning with Knowledge Graph. (arXiv:2304.10805v1 [cs.AI])

    [http://arxiv.org/abs/2304.10805](http://arxiv.org/abs/2304.10805)

    本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。

    

    大规模预训练模型已经被证明是可迁移的，并且对未知数据集具有很好的泛化性能。最近，诸如CLIP之类的多模态预训练模型在各种实验中表现出显着的性能提升。然而，当标记数据集有限时，新数据集或领域的泛化仍然具有挑战性。为了提高小样本学习的泛化性能，已经进行了各种努力，如提示学习和适配器。然而，当前的少样本自适应方法不具备可解释性，并且需要高计算成本来进行自适应。在本研究中，我们提出了一种新的方法，即基于知识图谱的鲁棒提示学习（RPLKG）。基于知识图谱，我们自动设计出各种可解释和有意义的提示集。我们的模型在大型预训练模型的一次正向传递后获得提示集的缓存嵌入。之后，模型使用GumbelSoftmax优化提示选择过程。

    Large-scale pre-trained models have been known that they are transferable, and they generalize well on the unseen dataset. Recently, multimodal pre-trained models such as CLIP show significant performance improvement in diverse experiments. However, when the labeled dataset is limited, the generalization of a new dataset or domain is still challenging. To improve the generalization performance on few-shot learning, there have been diverse efforts, such as prompt learning and adapter. However, the current few-shot adaptation methods are not interpretable, and they require a high computation cost for adaptation. In this study, we propose a new method, robust prompt learning with knowledge graph (RPLKG). Based on the knowledge graph, we automatically design diverse interpretable and meaningful prompt sets. Our model obtains cached embeddings of prompt sets after one forwarding from a large pre-trained model. After that, model optimizes the prompt selection processes with GumbelSoftmax. In
    
[^10]: 不定概率神经网络

    Indeterminate Probability Neural Network. (arXiv:2303.11536v1 [cs.LG])

    [http://arxiv.org/abs/2303.11536](http://arxiv.org/abs/2303.11536)

    本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。

    

    本文提出了一个称为IPNN的新型通用模型，它将神经网络和概率论结合在一起。在传统概率论中，概率的计算是基于事件的发生，而这在当前的神经网络中几乎不使用。因此，我们提出了一种新的概率理论，它是经典概率论的扩展，并使经典概率论成为我们理论的一种特殊情况。此外，对于我们提出的神经网络框架，神经网络的输出被定义为概率事件，并基于这些事件的统计分析，推导出分类任务的推理模型。IPNN展现了新的特性：它在进行分类的同时可以执行无监督聚类。此外，IPNN能够使用非常小的神经网络进行非常大的分类，例如100个输出节点的模型可以分类10亿类别。理论优势体现在新的概率理论和神经网络框架中，并且实验结果展示了IPNN在各种应用中的潜力。

    We propose a new general model called IPNN - Indeterminate Probability Neural Network, which combines neural network and probability theory together. In the classical probability theory, the calculation of probability is based on the occurrence of events, which is hardly used in current neural networks. In this paper, we propose a new general probability theory, which is an extension of classical probability theory, and makes classical probability theory a special case to our theory. Besides, for our proposed neural network framework, the output of neural network is defined as probability events, and based on the statistical analysis of these events, the inference model for classification task is deduced. IPNN shows new property: It can perform unsupervised clustering while doing classification. Besides, IPNN is capable of making very large classification with very small neural network, e.g. model with 100 output nodes can classify 10 billion categories. Theoretical advantages are refl
    

