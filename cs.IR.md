# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Co-evolving Vector Quantization for ID-based Recommendation.](http://arxiv.org/abs/2308.16761) | 这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。 |
| [^2] | [Towards Long-Tailed Recognition for Graph Classification via Collaborative Experts.](http://arxiv.org/abs/2308.16609) | 本文提出了一种新颖的方法，通过合作专家实现了长尾图分类，解决了现有方法在处理图数据上的不足。 |
| [^3] | [How Expressive are Graph Neural Networks in Recommendation?.](http://arxiv.org/abs/2308.11127) | 本文对图神经网络在推荐中的表达能力进行了理论分析，发现现有的表达能力度量标准可能无法有效评估模型在推荐中的能力，提出了一个全面的理论分析方法。 |
| [^4] | [AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models.](http://arxiv.org/abs/2307.11772) | AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。 |
| [^5] | [Retrieval-augmented GPT-3.5-based Text-to-SQL Framework with Sample-aware Prompting and Dynamic Revision Chain.](http://arxiv.org/abs/2307.05074) | 本文提出了一种基于检索增强的GPT-3.5文本到SQL框架，采用了样本感知引导和动态修订链的方法，以应对现有方法在处理语义差距较大的检索示例时面临的挑战。 |
| [^6] | [Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom.](http://arxiv.org/abs/2306.11963) | 本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。 |
| [^7] | [STUDY: Socially Aware Temporally Casual Decoder Recommender Systems.](http://arxiv.org/abs/2306.07946) | 该论文提出了一种基于社交感知和时间因素的解码器推荐系统(STUDY)，使用transformer解码器网络实现对社交网络图中相邻的用户组的联合推断。该方法在教育内容领域中经过测试，能够取得优于社交和顺序方法的结果。 |
| [^8] | [CTRL: Connect Tabular and Language Model for CTR Prediction.](http://arxiv.org/abs/2306.02841) | 提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。 |
| [^9] | [AMR4NLI: Interpretable and robust NLI measures from semantic graphs.](http://arxiv.org/abs/2306.00936) | 该论文提出了一种从语义图中获取可解释和鲁棒的NLI度量方法，与使用上下文嵌入的方法相比具有补充性，可以在混合模型中结合使用。 |
| [^10] | [Iteratively Learning Representations for Unseen Entities with Inter-Rule Correlations.](http://arxiv.org/abs/2305.10531) | 本文提出了一种虚拟邻居网络(VNC)，用于解决知识图谱完成中未知实体表示的问题。该方法通过规则挖掘、规则推理和嵌入三个阶段，实现对规则间相关性进行建模。 |
| [^11] | [DELTA: Dynamic Embedding Learning with Truncated Conscious Attention for CTR Prediction.](http://arxiv.org/abs/2305.04891) | 该论文提出了一种名为DELTA的CTR模型，使用截断意识注意力进行动态嵌入学习，有效地解决了上下文中无效和冗余特征的问题。 |
| [^12] | [MemoNet: Memorizing All Cross Features' Representations Efficiently via Multi-Hash Codebook Network for CTR Prediction.](http://arxiv.org/abs/2211.01334) | 本文提出了一种名为MemoNet的CTR模型，通过引入多哈希码本网络（HCNet）作为记忆机制，高效地学习和记忆交叉特征的表示。实验结果表明MemoNet在性能上优于最先进的方法，并且展现出NLP中的大型语言模型的扩展规律。 |
| [^13] | [Empowering Long-tail Item Recommendation through Cross Decoupling Network (CDN).](http://arxiv.org/abs/2210.14309) | 本论文研究了长尾商品推荐中的偏差问题，并提出了一种交叉解耦网络（CDN）方法，能够在保持整体性能和减少训练和服务成本的条件下提高尾部商品的推荐效果。 |
| [^14] | [Dual Correction Strategy for Ranking Distillation in Top-N Recommender System.](http://arxiv.org/abs/2109.03459) | 本文提出了一种双重修正策略（DCD），用于在推荐系统中更有效地将教师模型的排名信息转移到学生模型。这种方法不仅充分利用了学生模型的预测误差，还提供了更全面的视角，解决了松弛排名蒸馏方法的限制。 |

# 详细

[^1]: 基于ID的推荐的共同演化向量量化

    Co-evolving Vector Quantization for ID-based Recommendation. (arXiv:2308.16761v1 [cs.IR])

    [http://arxiv.org/abs/2308.16761](http://arxiv.org/abs/2308.16761)

    这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。

    

    类别信息对于提高推荐的质量和个性化起着至关重要的作用。然而，在基于ID的推荐中，项目类别信息的可用性并不一致。在这项工作中，我们提出了一种替代方法，以自动学习和生成实体（即用户和项目）在不同粒度级别上的分类信息，特别适用于基于ID的推荐。具体而言，我们设计了一个共同演化向量量化框架，即COVE，它能够同时学习和改进代码表示和实体嵌入，并以从随机初始化状态开始的端到端方式进行。通过其高度适应性，COVE可以轻松集成到现有的推荐模型中。我们验证了COVE在各种推荐任务中的有效性，包括列表完成、协同过滤和点击率预测，涵盖不同的推荐场景。

    Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommend
    
[^2]: 通过合作专家实现长尾图分类的研究

    Towards Long-Tailed Recognition for Graph Classification via Collaborative Experts. (arXiv:2308.16609v1 [cs.LG])

    [http://arxiv.org/abs/2308.16609](http://arxiv.org/abs/2308.16609)

    本文提出了一种新颖的方法，通过合作专家实现了长尾图分类，解决了现有方法在处理图数据上的不足。

    

    图分类旨在学习用于有效类别分配的图级表示，在平衡的类别分布的高质量数据集的支持下取得了杰出成果。事实上，大多数现实世界的图数据自然呈现长尾形式，其中头部类别的样本数量远超过尾部类别，因此在长尾数据上研究图级分类是至关重要的，但仍然较少探索。然而，现有的视觉中的长尾学习方法大多无法同时优化表示学习和分类器训练，并且忽略了难以分类的类别的挖掘。直接将现有方法应用于图可能导致次优性能，因为在图上训练的模型由于复杂的拓扑特征会更加敏感于长尾分布。因此，在本文中，我们提出了一种新颖的对长尾图级分类的方法

    Graph classification, aiming at learning the graph-level representations for effective class assignments, has received outstanding achievements, which heavily relies on high-quality datasets that have balanced class distribution. In fact, most real-world graph data naturally presents a long-tailed form, where the head classes occupy much more samples than the tail classes, it thus is essential to study the graph-level classification over long-tailed data while still remaining largely unexplored. However, most existing long-tailed learning methods in visions fail to jointly optimize the representation learning and classifier training, as well as neglect the mining of the hard-to-classify classes. Directly applying existing methods to graphs may lead to sub-optimal performance, since the model trained on graphs would be more sensitive to the long-tailed distribution due to the complex topological characteristics. Hence, in this paper, we propose a novel long-tailed graph-level classifica
    
[^3]: 图神经网络在推荐中的表达能力有多强？

    How Expressive are Graph Neural Networks in Recommendation?. (arXiv:2308.11127v1 [cs.IR])

    [http://arxiv.org/abs/2308.11127](http://arxiv.org/abs/2308.11127)

    本文对图神经网络在推荐中的表达能力进行了理论分析，发现现有的表达能力度量标准可能无法有效评估模型在推荐中的能力，提出了一个全面的理论分析方法。

    

    图神经网络（GNNs）在各种图学习任务中展示了优越的性能，包括利用图中的用户-物品协作过滤信号进行推荐。然而，尽管它们在最先进的推荐模型中的经验有效性，但对于它们的能力的理论表述非常稀少。最近的研究探讨了GNNs的一般表达能力，证明了消息传递GNNs至多与Weisfeiler-Lehman测试一样强大，并且与随机节点初始化相结合的GNNs是通用的。然而，GNNs的“表达能力”概念仍然定义模糊。大多数现有的工作采用图同构测试作为表达能力的度量标准，但这种图级任务可能不能有效评估模型在推荐中区分不同接近程度节点的能力。在本文中，我们对GNNs在推荐中的表达能力进行了全面的理论分析。

    Graph Neural Networks (GNNs) have demonstrated superior performance on various graph learning tasks, including recommendation, where they leverage user-item collaborative filtering signals in graphs. However, theoretical formulations of their capability are scarce, despite their empirical effectiveness in state-of-the-art recommender models. Recently, research has explored the expressiveness of GNNs in general, demonstrating that message passing GNNs are at most as powerful as the Weisfeiler-Lehman test, and that GNNs combined with random node initialization are universal. Nevertheless, the concept of "expressiveness" for GNNs remains vaguely defined. Most existing works adopt the graph isomorphism test as the metric of expressiveness, but this graph-level task may not effectively assess a model's ability in recommendation, where the objective is to distinguish nodes of different closeness. In this paper, we provide a comprehensive theoretical analysis of the expressiveness of GNNs in 
    
[^4]: AutoAlign：基于大型语言模型的全自动有效知识图谱对齐方法

    AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models. (arXiv:2307.11772v1 [cs.IR])

    [http://arxiv.org/abs/2307.11772](http://arxiv.org/abs/2307.11772)

    AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。

    

    知识图谱间的实体对齐任务旨在识别出两个不同知识图谱中表示相同实体的每对实体。许多基于机器学习的方法已被提出用于这个任务。然而，据我们所知，现有的方法都需要手工制作的种子对齐，这是非常昂贵的。在本文中，我们提出了第一个名为AutoAlign的完全自动对齐方法，它不需要任何手工制作的种子对齐。具体而言，对于谓词嵌入，AutoAlign使用大型语言模型构建谓词近邻图，自动捕捉两个知识图谱中谓词的相似性。对于实体嵌入，AutoAlign首先使用TransE独立计算每个知识图谱的实体嵌入，然后通过计算基于实体属性的实体相似性，将两个知识图谱的实体嵌入移动到相同的向量空间中。因此，AutoAlign实现了谓词对齐和实体对齐。

    The task of entity alignment between knowledge graphs (KGs) aims to identify every pair of entities from two different KGs that represent the same entity. Many machine learning-based methods have been proposed for this task. However, to our best knowledge, existing methods all require manually crafted seed alignments, which are expensive to obtain. In this paper, we propose the first fully automatic alignment method named AutoAlign, which does not require any manually crafted seed alignments. Specifically, for predicate embeddings, AutoAlign constructs a predicate-proximity-graph with the help of large language models to automatically capture the similarity between predicates across two KGs. For entity embeddings, AutoAlign first computes the entity embeddings of each KG independently using TransE, and then shifts the two KGs' entity embeddings into the same vector space by computing the similarity between entities based on their attributes. Thus, both predicate alignment and entity al
    
[^5]: 采用样本感知引导和动态修订链的基于检索增强的GPT-3.5文本到SQL框架

    Retrieval-augmented GPT-3.5-based Text-to-SQL Framework with Sample-aware Prompting and Dynamic Revision Chain. (arXiv:2307.05074v1 [cs.IR])

    [http://arxiv.org/abs/2307.05074](http://arxiv.org/abs/2307.05074)

    本文提出了一种基于检索增强的GPT-3.5文本到SQL框架，采用了样本感知引导和动态修订链的方法，以应对现有方法在处理语义差距较大的检索示例时面临的挑战。

    

    文本到SQL旨在为给定的自然语言问题生成SQL查询，从而帮助用户查询数据库。最近出现了一种基于大型语言模型（LLMs）的提示学习方法，该方法设计提示以引导LLMs理解输入问题并生成相应的SQL。然而，它面临着严格的SQL语法要求的挑战。现有工作使用一系列示例（即问题-SQL对）来提示LLMs生成SQL，但固定的提示几乎无法处理检索出的示例与输入问题之间的语义差距较大的情况。在本文中，我们提出了一种基于检索增强的提示方法，用于基于LLM的文本到SQL框架，包括样本感知提示和动态修订链。我们的方法包括样本感知示例，其中包括SQL运算符的组合和与给定问题相关的细粒度信息。

    Text-to-SQL aims at generating SQL queries for the given natural language questions and thus helping users to query databases. Prompt learning with large language models (LLMs) has emerged as a recent approach, which designs prompts to lead LLMs to understand the input question and generate the corresponding SQL. However, it faces challenges with strict SQL syntax requirements. Existing work prompts the LLMs with a list of demonstration examples (i.e. question-SQL pairs) to generate SQL, but the fixed prompts can hardly handle the scenario where the semantic gap between the retrieved demonstration and the input question is large. In this paper, we propose a retrieval-augmented prompting method for a LLM-based Text-to-SQL framework, involving sample-aware prompting and a dynamic revision chain. Our approach incorporates sample-aware demonstrations, which include the composition of SQL operators and fine-grained information related to the given question. To retrieve questions sharing sim
    
[^6]: 智慧医疗中的多模态融合:从数据、信息、知识到智慧之旅

    Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom. (arXiv:2306.11963v1 [cs.IR])

    [http://arxiv.org/abs/2306.11963](http://arxiv.org/abs/2306.11963)

    本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。

    

    多模态医学数据融合已成为智慧医疗中的一种革新性方法，能够全面了解患者健康状况和个性化治疗方案。本文探讨了多模态融合为智慧医疗带来的从数据、信息和知识到智慧（DIKW）之旅。全面回顾了多模态医学数据融合的研究现状，重点关注了不同数据模态的集成方式。文章探讨了特征选择、基于规则的系统、机器学习、深度学习和自然语言处理等不同方法，用于多模态数据的融合和分析。同时，文章也着重讨论了多模态融合在医疗保健中面临的挑战。通过综合评述的框架和见解，提出了一个符合DIKW机制的通用多模态医疗数据融合框架。此外，文章还探讨了未来与预测、预防、个性化和治疗有关的医疗方向。

    Multimodal medical data fusion has emerged as a transformative approach in smart healthcare, enabling a comprehensive understanding of patient health and personalized treatment plans. In this paper, a journey from data, information, and knowledge to wisdom (DIKW) is explored through multimodal fusion for smart healthcare. A comprehensive review of multimodal medical data fusion focuses on the integration of various data modalities are presented. It explores different approaches such as Feature selection, Rule-based systems, Machine learning, Deep learning, and Natural Language Processing for fusing and analyzing multimodal data. The paper also highlights the challenges associated with multimodal fusion in healthcare. By synthesizing the reviewed frameworks and insights, a generic framework for multimodal medical data fusion is proposed while aligning with the DIKW mechanism. Moreover, it discusses future directions aligned with the four pillars of healthcare: Predictive, Preventive, Pe
    
[^7]: 研究：社交感知时间松散解码器推荐系统

    STUDY: Socially Aware Temporally Casual Decoder Recommender Systems. (arXiv:2306.07946v1 [cs.SI])

    [http://arxiv.org/abs/2306.07946](http://arxiv.org/abs/2306.07946)

    该论文提出了一种基于社交感知和时间因素的解码器推荐系统(STUDY)，使用transformer解码器网络实现对社交网络图中相邻的用户组的联合推断。该方法在教育内容领域中经过测试，能够取得优于社交和顺序方法的结果。

    

    随着现在在线和离线可获取的数据数量过于庞大，推荐系统变得越来越必要，以帮助用户找到符合他们兴趣的物品。当社交网络信息存在时，有一些方法利用这些信息来做出更好的推荐，但这些方法通常有复杂的结构和训练过程。此外，许多现有的方法使用图神经网络，而这些网络训练起来非常困难。为了解决这个问题，我们提出了基于社交感知和时间因素的解码器推荐系统(STUDY)。STUDY采用一个经过修改的transformer解码器网络的单向前传，对社交网络图中相邻的用户组进行联合推断。我们在基于学校课堂结构定义社交网络的教育内容领域测试了我们的方法。我们的方法在保持单一均匀网络设计简单性的同时，优于社交和顺序方法。

    With the overwhelming amount of data available both on and offline today, recommender systems have become much needed to help users find items tailored to their interests. When social network information exists there are methods that utilize this information to make better recommendations, however the methods are often clunky with complex architectures and training procedures. Furthermore many of the existing methods utilize graph neural networks which are notoriously difficult to train. To address this, we propose Socially-aware Temporally caUsal Decoder recommender sYstems (STUDY). STUDY does joint inference over groups of users who are adjacent in the social network graph using a single forward pass of a modified transformer decoder network. We test our method in a school-based educational content setting, using classroom structure to define social networks. Our method outperforms both social and sequential methods while maintaining the design simplicity of a single homogeneous netw
    
[^8]: CTRL: 连接表格和语言模型进行CTR预测

    CTRL: Connect Tabular and Language Model for CTR Prediction. (arXiv:2306.02841v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02841](http://arxiv.org/abs/2306.02841)

    提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。

    

    传统的CTR预测模型将表格数据转换为one-hot向量，并利用特征之间的协作关系来推断用户对项目的偏好。这种建模范式抛弃了基本的语义信息。尽管一些最近的工作（如P5和M6-Rec）已经探索了使用预训练语言模型（PLMs）提取CTR预测的语义信号的潜力，但它们计算成本高，效率低。此外，尚未考虑到有益的协作关系，从而阻碍了推荐的性能。为了解决这些问题，我们提出了一个新的框架CTRL，它是工业友好的和模型不可知的，具有高训练和推理效率。具体而言，原始的表格数据首先被转换为文本数据。两种不同的模态被分别视为两个模态，并分别输入协作CTR模型中以建模它们的交互作用。我们还提出了信息蒸馏机制，从PLMs中提取关于CTR预测的语义信息，进一步提高了模型的性能。在三个真实的工业数据集上，我们的模型在比较其他现有的模型时均达到了最新的SOTA性能水平。

    Traditional click-through rate (CTR) prediction models convert the tabular data into one-hot vectors and leverage the collaborative relations among features for inferring user's preference over items. This modeling paradigm discards the essential semantic information. Though some recent works like P5 and M6-Rec have explored the potential of using Pre-trained Language Models (PLMs) to extract semantic signals for CTR prediction, they are computationally expensive and suffer from low efficiency. Besides, the beneficial collaborative relations are not considered, hindering the recommendation performance. To solve these problems, in this paper, we propose a novel framework \textbf{CTRL}, which is industrial friendly and model-agnostic with high training and inference efficiency. Specifically, the original tabular data is first converted into textual data. Both tabular data and converted textual data are regarded as two different modalities and are separately fed into the collaborative CTR
    
[^9]: AMR4NLI: 从语义图中获得可解释和鲁棒的NLI度量

    AMR4NLI: Interpretable and robust NLI measures from semantic graphs. (arXiv:2306.00936v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.00936](http://arxiv.org/abs/2306.00936)

    该论文提出了一种从语义图中获取可解释和鲁棒的NLI度量方法，与使用上下文嵌入的方法相比具有补充性，可以在混合模型中结合使用。

    

    自然语言推理（NLI）任务要求判断给定的前提（用自然语言表达）是否蕴含给定的假设。NLI基准包含了蕴含性的人工评分，但是驱动这些评分的意义关系并未形式化。是否可以以一种可解释且鲁棒的方式更明确地表示句子对之间的关系？我们比较了表示前提和假设的语义结构，包括一组上下文化嵌入和语义图（抽象意义表示），并使用可解释的度量方法来衡量假设是否是前提的语义子结构。在三个英语基准测试中的评估发现，上下文化嵌入和语义图都有其价值；而且它们提供了互补的信号，并可以在混合模型中一起利用。

    The task of natural language inference (NLI) asks whether a given premise (expressed in NL) entails a given NL hypothesis. NLI benchmarks contain human ratings of entailment, but the meaning relationships driving these ratings are not formalized. Can the underlying sentence pair relationships be made more explicit in an interpretable yet robust fashion? We compare semantic structures to represent premise and hypothesis, including sets of contextualized embeddings and semantic graphs (Abstract Meaning Representations), and measure whether the hypothesis is a semantic substructure of the premise, utilizing interpretable metrics. Our evaluation on three English benchmarks finds value in both contextualized embeddings and semantic graphs; moreover, they provide complementary signals, and can be leveraged together in a hybrid model.
    
[^10]: 迭代学习具有规则间相关性的未知实体表示

    Iteratively Learning Representations for Unseen Entities with Inter-Rule Correlations. (arXiv:2305.10531v1 [cs.IR])

    [http://arxiv.org/abs/2305.10531](http://arxiv.org/abs/2305.10531)

    本文提出了一种虚拟邻居网络(VNC)，用于解决知识图谱完成中未知实体表示的问题。该方法通过规则挖掘、规则推理和嵌入三个阶段，实现对规则间相关性进行建模。

    

    知识图谱完成(KGC)的最新研究侧重于学习知识图谱中实体和关系的嵌入。这些嵌入方法要求所有测试实体在训练时被观察到，导致对超出知识图谱（OOKG）实体的耗时重新训练过程。为解决此问题，当前归纳知识嵌入方法采用图神经网络(GNN)通过聚合已知邻居的信息来表示未知实体。他们面临三个重要挑战:i)数据稀疏性，ii)知识图谱中存在复杂模式(如规则间相关性)，iii)规则挖掘、规则推理和嵌入之间存在交互。在本文中，我们提出了一个包含三个阶段的具有规则间相关性的虚拟邻居网络(VNC):i)规则挖掘，ii)规则推理，和iii)嵌入。

    Recent work on knowledge graph completion (KGC) focused on learning embeddings of entities and relations in knowledge graphs. These embedding methods require that all test entities are observed at training time, resulting in a time-consuming retraining process for out-of-knowledge-graph (OOKG) entities. To address this issue, current inductive knowledge embedding methods employ graph neural networks (GNNs) to represent unseen entities by aggregating information of known neighbors. They face three important challenges: (i) data sparsity, (ii) the presence of complex patterns in knowledge graphs (e.g., inter-rule correlations), and (iii) the presence of interactions among rule mining, rule inference, and embedding. In this paper, we propose a virtual neighbor network with inter-rule correlations (VNC) that consists of three stages: (i) rule mining, (ii) rule inference, and (iii) embedding. In the rule mining process, to identify complex patterns in knowledge graphs, both logic rules and 
    
[^11]: 带有截断意识注意力的动态嵌入学习模型用于CTR预测

    DELTA: Dynamic Embedding Learning with Truncated Conscious Attention for CTR Prediction. (arXiv:2305.04891v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.04891](http://arxiv.org/abs/2305.04891)

    该论文提出了一种名为DELTA的CTR模型，使用截断意识注意力进行动态嵌入学习，有效地解决了上下文中无效和冗余特征的问题。

    

    点击率（CTR）预测是产品和内容推荐中关键的任务，学习有效的特征嵌入具有重要意义。传统方法通常学习固定的特征表示，而缺乏根据上下文信息动态调整特征表示的机制，导致性能不佳。一些近期的方法尝试通过学习位权重或增强嵌入来解决这个问题，但是受到上下文中无信息或冗余特征的影响。为了解决这个问题，我们借鉴了意识加工中全局工作区理论，该理论认为只有特定的产品特征与点击行为相关，其余特征可能会噪音干扰，甚至有害，因此提出了一种带有截断意识注意力的动态嵌入学习模型DELTA进行CTR预测。

    Click-Through Rate (CTR) prediction is a pivotal task in product and content recommendation, where learning effective feature embeddings is of great significance. However, traditional methods typically learn fixed feature representations without dynamically refining feature representations according to the context information, leading to suboptimal performance. Some recent approaches attempt to address this issue by learning bit-wise weights or augmented embeddings for feature representations, but suffer from uninformative or redundant features in the context. To tackle this problem, inspired by the Global Workspace Theory in conscious processing, which posits that only a specific subset of the product features are pertinent while the rest can be noisy and even detrimental to human-click behaviors, we propose a CTR model that enables Dynamic Embedding Learning with Truncated Conscious Attention for CTR prediction, termed DELTA. DELTA contains two key components: (I) conscious truncatio
    
[^12]: MemoNet: 通过多哈希码本网络高效地记忆所有交叉特征表示以实现CTR预测

    MemoNet: Memorizing All Cross Features' Representations Efficiently via Multi-Hash Codebook Network for CTR Prediction. (arXiv:2211.01334v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.01334](http://arxiv.org/abs/2211.01334)

    本文提出了一种名为MemoNet的CTR模型，通过引入多哈希码本网络（HCNet）作为记忆机制，高效地学习和记忆交叉特征的表示。实验结果表明MemoNet在性能上优于最先进的方法，并且展现出NLP中的大型语言模型的扩展规律。

    

    自然语言处理（NLP）中的新发现表明，强大的记忆能力对大型语言模型（LLM）的成功起到了很大作用。这启发我们将独立的记忆机制引入CTR排名模型，以学习和记忆交叉特征的表示。本文提出了多哈希码本网络（HCNet）作为CTR任务中高效学习和记忆交叉特征表示的记忆机制。HCNet使用多哈希码本作为主要的记忆位置，并由多哈希寻址、记忆恢复和特征缩减三个阶段组成。我们还提出了一种名为MemoNet的新型CTR模型，将HCNet与DNN骨干网络相结合。广泛的实验结果在三个公共数据集和在线测试中表明，MemoNet在性能上优于最先进的方法。此外，MemoNet展现出NLP中的大型语言模型的扩展规律，这意味着我们可以扩大模型规模来提高性能。

    New findings in natural language processing (NLP) demonstrate that the strong memorization capability contributes a lot to the success of Large Language Models (LLM). This inspires us to explicitly bring an independent memory mechanism into CTR ranking model to learn and memorize cross features' representations. In this paper, we propose multi-Hash Codebook NETwork (HCNet) as the memory mechanism for efficiently learning and memorizing representations of cross features in CTR tasks. HCNet uses a multi-hash codebook as the main memory place and the whole memory procedure consists of three phases: multi-hash addressing, memory restoring, and feature shrinking. We also propose a new CTR model named MemoNet which combines HCNet with a DNN backbone. Extensive experimental results on three public datasets and online test show that MemoNet reaches superior performance over state-of-the-art approaches. Besides, MemoNet shows scaling law of large language model in NLP, which means we can enlarg
    
[^13]: 通过交叉解耦网络（CDN）增强长尾商品推荐

    Empowering Long-tail Item Recommendation through Cross Decoupling Network (CDN). (arXiv:2210.14309v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.14309](http://arxiv.org/abs/2210.14309)

    本论文研究了长尾商品推荐中的偏差问题，并提出了一种交叉解耦网络（CDN）方法，能够在保持整体性能和减少训练和服务成本的条件下提高尾部商品的推荐效果。

    

    工业推荐系统通常遭受高度倾斜的长尾商品分布，其中少数商品获得大部分用户反馈。这种偏差对没有太多用户反馈的商品影响推荐质量。虽然学术界已经取得了许多研究进展，但在生产中部署这些方法非常困难，工业领域中改进的方法很少。本文旨在提高尾部商品的推荐效果，同时减少培训和服务成本。我们首先发现长尾分布下用户偏好的预测具有偏差。该偏见来自于两个方面在训练和服务数据之间的差异：1）物品分布以及2）给定物品的用户偏好。大多数现有方法主要尝试减轻偏差，但鲜有同时保持整体性能和降低成本的工作。

    Industry recommender systems usually suffer from highly-skewed long-tail item distributions where a small fraction of the items receives most of the user feedback. This skew hurts recommender quality especially for the item slices without much user feedback. While there have been many research advances made in academia, deploying these methods in production is very difficult and very few improvements have been made in industry. One challenge is that these methods often hurt overall performance; additionally, they could be complex and expensive to train and serve. In this work, we aim to improve tail item recommendations while maintaining the overall performance with less training and serving cost. We first find that the predictions of user preferences are biased under long-tail distributions. The bias comes from the differences between training and serving data in two perspectives: 1) the item distributions, and 2) user's preference given an item. Most existing methods mainly attempt t
    
[^14]: 推荐系统中用于排名蒸馏的双重修正策略

    Dual Correction Strategy for Ranking Distillation in Top-N Recommender System. (arXiv:2109.03459v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2109.03459](http://arxiv.org/abs/2109.03459)

    本文提出了一种双重修正策略（DCD），用于在推荐系统中更有效地将教师模型的排名信息转移到学生模型。这种方法不仅充分利用了学生模型的预测误差，还提供了更全面的视角，解决了松弛排名蒸馏方法的限制。

    

    知识蒸馏是将训练充分的大模型（教师）的知识转移到小模型（学生）的重要研究领域，对于推荐系统的实际部署而言，它已成为一个重要的研究方向。最近，松弛排名蒸馏（RRD）表明，在推荐列表中蒸馏排名信息能够显著提高性能。然而，该方法仍然存在以下限制：1）它未充分利用学生模型的预测误差，使得训练效率不高；2）它只蒸馏用户侧的排名信息，在稀疏的隐式反馈下提供的视角不足。本文提出了一种更高效的蒸馏方法，即双重修正策略（DCD），通过教师模型和学生模型预测之间的差异来决定要蒸馏的知识。

    Knowledge Distillation (KD), which transfers the knowledge of a well-trained large model (teacher) to a small model (student), has become an important area of research for practical deployment of recommender systems. Recently, Relaxed Ranking Distillation (RRD) has shown that distilling the ranking information in the recommendation list significantly improves the performance. However, the method still has limitations in that 1) it does not fully utilize the prediction errors of the student model, which makes the training not fully efficient, and 2) it only distills the user-side ranking information, which provides an insufficient view under the sparse implicit feedback. This paper presents Dual Correction strategy for Distillation (DCD), which transfers the ranking information from the teacher model to the student model in a more efficient manner. Most importantly, DCD uses the discrepancy between the teacher model and the student model predictions to decide which knowledge to be disti
    

