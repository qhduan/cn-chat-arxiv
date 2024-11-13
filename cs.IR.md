# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhanced Generative Recommendation via Content and Collaboration Integration](https://arxiv.org/abs/2403.18480) | 本文引入了一种基于内容的协作生成式推荐系统ColaRec，旨在解决生成式推荐中的协作信号集成和信息对齐的挑战。 |
| [^2] | [Explainable Identification of Hate Speech towards Islam using Graph Neural Networks](https://arxiv.org/abs/2311.04916) | 使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。 |
| [^3] | [Explicit and Implicit Semantic Ranking Framework.](http://arxiv.org/abs/2304.04918) | 本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。 |

# 详细

[^1]: 通过内容和协作集成增强生成式推荐

    Enhanced Generative Recommendation via Content and Collaboration Integration

    [https://arxiv.org/abs/2403.18480](https://arxiv.org/abs/2403.18480)

    本文引入了一种基于内容的协作生成式推荐系统ColaRec，旨在解决生成式推荐中的协作信号集成和信息对齐的挑战。

    

    生成式推荐已经出现作为一种有前途的范式，旨在通过生成式人工智能的最新进展来增强推荐系统。本任务被制定为一个序列到序列的生成过程，其中输入序列包含与用户先前交互的项目相关的数据，输出序列表示建议项目的生成标识符。然而，现有的生成式推荐方法仍然面临着以下挑战：有效地在统一生成框架内集成用户-项目协作信号和项目内容信息，以及在内容信息和协作信号之间执行高效的对齐。

    arXiv:2403.18480v1 Announce Type: new  Abstract: Generative recommendation has emerged as a promising paradigm aimed at augmenting recommender systems with recent advancements in generative artificial intelligence. This task has been formulated as a sequence-to-sequence generation process, wherein the input sequence encompasses data pertaining to the user's previously interacted items, and the output sequence denotes the generative identifier for the suggested item. However, existing generative recommendation approaches still encounter challenges in (i) effectively integrating user-item collaborative signals and item content information within a unified generative framework, and (ii) executing an efficient alignment between content information and collaborative signals.   In this paper, we introduce content-based collaborative generation for recommender systems, denoted as ColaRec. To capture collaborative signals, the generative item identifiers are derived from a pretrained collabora
    
[^2]: 使用图神经网络解释伊斯兰教仇恨言论的研究

    Explainable Identification of Hate Speech towards Islam using Graph Neural Networks

    [https://arxiv.org/abs/2311.04916](https://arxiv.org/abs/2311.04916)

    使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。

    

    伊斯兰教仇恨言论在在线社交互动平台上是一个普遍存在的挑战。识别和消除这种仇恨是迈向和谐与和平未来的关键一步。本研究提出了一种新的范例，利用图神经网络来识别和解释针对伊斯兰教的仇恨言论。利用图神经网络发现、提取并利用不同数据点之间的关系的内在能力，我们的模型始终能够在保持出色性能的同时提供对潜在相关性和因果关系的解释。

    arXiv:2311.04916v2 Announce Type: cross  Abstract: Islamophobic language is a prevalent challenge on online social interaction platforms. Identifying and eliminating such hatred is a crucial step towards a future of harmony and peace. This study presents a novel paradigm for identifying and explaining hate speech towards Islam using graph neural networks. Utilizing the intrinsic ability of graph neural networks to find, extract, and use relationships across disparate data points, our model consistently achieves outstanding performance while offering explanations for the underlying correlations and causation.
    
[^3]: 显式和隐式语义排序框架

    Explicit and Implicit Semantic Ranking Framework. (arXiv:2304.04918v1 [cs.IR])

    [http://arxiv.org/abs/2304.04918](http://arxiv.org/abs/2304.04918)

    本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。

    

    在许多实际应用中，核心难题是将一个查询与一个可变且有限的文档集中的最佳文档进行匹配。现有的工业解决方案，特别是延迟受限的服务，通常依赖于相似性算法，这些算法为了速度而牺牲了质量。本文介绍了一个通用的语义学习排名框架，自我训练语义交叉关注排名（sRank）。这个基于transformer的框架使用线性成对损失，具有可变的训练批量大小、实现质量提升和高效率，并已成功应用于微软公司的两个工业任务：智能回复（SR）和环境临床智能（ACI）的真实大规模数据集上。在智能回复中，$sRank$通过基于消费者和支持代理信息的预定义解决方案选择最佳答案，帮助用户实时获得技术支持。在SR任务上，$sRank$实现了11.7%的离线top-one准确度提升，比之前的系统更加优秀。

    The core challenge in numerous real-world applications is to match an inquiry to the best document from a mutable and finite set of candidates. Existing industry solutions, especially latency-constrained services, often rely on similarity algorithms that sacrifice quality for speed. In this paper we introduce a generic semantic learning-to-rank framework, Self-training Semantic Cross-attention Ranking (sRank). This transformer-based framework uses linear pairwise loss with mutable training batch sizes and achieves quality gains and high efficiency, and has been applied effectively to show gains on two industry tasks at Microsoft over real-world large-scale data sets: Smart Reply (SR) and Ambient Clinical Intelligence (ACI). In Smart Reply, $sRank$ assists live customers with technical support by selecting the best reply from predefined solutions based on consumer and support agent messages. It achieves 11.7% gain in offline top-one accuracy on the SR task over the previous system, and 
    

