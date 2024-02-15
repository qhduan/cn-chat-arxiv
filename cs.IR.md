# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Training of Temporal GNNs using Nearest Neighbours based Hard Negatives](https://arxiv.org/abs/2402.09239) | 本研究提出了使用基于重要性的负样本抽样训练TGNNS的方法，并通过实证评估证明了其优越性能。 |
| [^2] | [Large Language Model Interaction Simulator for Cold-Start Item Recommendation](https://arxiv.org/abs/2402.09176) | 我们提出了一个大型语言模型交互模拟器 (LLM-InS)，用于解决冷启动物品推荐的问题。该模拟器能够模拟出逼真的交互，并将冷启动物品转化为热门物品，从而提高推荐性能。 |
| [^3] | [Recommendation Algorithm Based on Recommendation Sessions](https://arxiv.org/abs/2402.09130) | 本研究提出了一种基于推荐会话的推荐算法，该算法使用了来自不同信息集的数据，包括静态信息类别、对象特征和动态用户行为。 |
| [^4] | [Confidence-aware Fine-tuning of Sequential Recommendation Systems via Conformal Prediction](https://arxiv.org/abs/2402.08976) | 本研究提出了CPFT框架，通过在顺序推荐系统中精细调整过程中结合交叉熵损失函数和基于符合性预测的损失函数，增强了推荐系统的置信度。CPFT动态生成潜在真实值的项目集合，提升了训练过程中的性能，并提高了推荐的准确性和可信度。 |
| [^5] | [Enhancing ID and Text Fusion via Alternative Training in Session-based Recommendation](https://arxiv.org/abs/2402.08921) | 本论文研究在基于会话的推荐中增强ID和文本的融合效果。研究发现，通过简单融合的方法并不能始终超越最佳的单模态方法。进一步的调查表明，这可能是由于融合方法中存在的ID和文本模态不平衡问题。 |
| [^6] | [eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data](https://arxiv.org/abs/2402.08831) | 本文利用开源的大规模高质量指导数据集ECInstruct，通过指导调优通用语言模型，开发了一系列电子商务LLMs（eCeLLM），在电子商务中表现出了显著的优势。 |
| [^7] | [Multi-Label Zero-Shot Product Attribute-Value Extraction](https://arxiv.org/abs/2402.08802) | 本文提出了一种用于在没有标注数据的情况下高效提取新产品中未见过的属性值的方法，通过构建异构超图并利用归纳推理来捕捉复杂的高阶关系，以更好地学习。 |
| [^8] | [Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT](https://arxiv.org/abs/2402.07440) | 该论文介绍了LoCoV1，一个用于评估长上下文检索性能的新型基准测试，并提出了M2-BERT检索编码器，用于处理长上下文检索，解决了如何评估性能、预训练语言模型以及如何进行微调的挑战。 |
| [^9] | [In-context Learning with Retrieved Demonstrations for Language Models: A Survey.](http://arxiv.org/abs/2401.11624) | 本综述调查了一种名为检索示范的方法，它通过使用特定于输入查询的示范来提高语言模型的少量样本情境学习（ICL）能力。这种方法不仅提高了学习效率和可扩展性，还减少了手动示例选择中的偏见。 |
| [^10] | [Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model.](http://arxiv.org/abs/2310.09400) | 本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。 |

# 详细

[^1]: 使用基于最近邻的硬负样本的鲁棒性训练时间GNN

    Robust Training of Temporal GNNs using Nearest Neighbours based Hard Negatives

    [https://arxiv.org/abs/2402.09239](https://arxiv.org/abs/2402.09239)

    本研究提出了使用基于重要性的负样本抽样训练TGNNS的方法，并通过实证评估证明了其优越性能。

    

    时间图神经网络(TGNNS)在未来链接预测任务中表现出最先进的性能。这些TGNNS的训练是通过均匀随机抽样的无监督损失进行列举的。在训练过程中，对于正例情况，损失是在无信息的负样本上计算的，这引入了冗余和次优的性能。在本文中，我们提出了改进的TGNNS无监督学习，通过使用基于重要性的负样本抽样来替换均匀负样本抽样。我们从理论上对负例采样的动态计算分布进行了理论验证和定义。最后，通过对三个真实世界数据集进行实证评估，我们展示了使用基于提出的负样本抽样的损失训练的TGNNS提供了一致的优越性能。

    arXiv:2402.09239v1 Announce Type: new Abstract: Temporal graph neural networks Tgnn have exhibited state-of-art performance in future-link prediction tasks. Training of these TGNNs is enumerated by uniform random sampling based unsupervised loss. During training, in the context of a positive example, the loss is computed over uninformative negatives, which introduces redundancy and sub-optimal performance. In this paper, we propose modified unsupervised learning of Tgnn, by replacing the uniform negative sampling with importance-based negative sampling. We theoretically motivate and define the dynamically computed distribution for a sampling of negative examples. Finally, using empirical evaluations over three real-world datasets, we show that Tgnn trained using loss based on proposed negative sampling provides consistent superior performance.
    
[^2]: 大型语言模型交互模拟器用于冷启动物品推荐

    Large Language Model Interaction Simulator for Cold-Start Item Recommendation

    [https://arxiv.org/abs/2402.09176](https://arxiv.org/abs/2402.09176)

    我们提出了一个大型语言模型交互模拟器 (LLM-InS)，用于解决冷启动物品推荐的问题。该模拟器能够模拟出逼真的交互，并将冷启动物品转化为热门物品，从而提高推荐性能。

    

    推荐冷启动物品对协同过滤模型来说是个长期的挑战，因为这些物品缺乏历史用户交互以建模他们的协同特性。冷启动物品的内容与行为模式之间的差距使得很难为其生成准确的行为嵌入。现有的冷启动模型使用映射函数基于冷启动物品的内容特征生成虚假的行为嵌入。然而，这些生成的嵌入与真实的行为嵌入存在显著的差异，对冷启动推荐性能产生负面影响。为了解决这个挑战，我们提出了一个基于内容方面来模拟用户行为模式的LLM交互模拟器 (LLM-InS)。该模拟器允许推荐系统为每个冷启动物品模拟生动的交互，并将其直接从冷启动物品转化为热门物品。

    arXiv:2402.09176v1 Announce Type: new Abstract: Recommending cold items is a long-standing challenge for collaborative filtering models because these cold items lack historical user interactions to model their collaborative features. The gap between the content of cold items and their behavior patterns makes it difficult to generate accurate behavioral embeddings for cold items. Existing cold-start models use mapping functions to generate fake behavioral embeddings based on the content feature of cold items. However, these generated embeddings have significant differences from the real behavioral embeddings, leading to a negative impact on cold recommendation performance. To address this challenge, we propose an LLM Interaction Simulator (LLM-InS) to model users' behavior patterns based on the content aspect. This simulator allows recommender systems to simulate vivid interactions for each cold item and transform them from cold to warm items directly. Specifically, we outline the desig
    
[^3]: 基于推荐会话的推荐算法

    Recommendation Algorithm Based on Recommendation Sessions

    [https://arxiv.org/abs/2402.09130](https://arxiv.org/abs/2402.09130)

    本研究提出了一种基于推荐会话的推荐算法，该算法使用了来自不同信息集的数据，包括静态信息类别、对象特征和动态用户行为。

    

    互联网的巨大发展不仅在地理范围上，也在日常生活中利用其可能性的领域上，决定了巨量数据的创建和收集。由于规模的问题，传统方法不能分析这些数据，因此必须使用现代方法和技术。这种方法主要由推荐领域提供。本研究旨在提出一种基于不同信息集（静态信息类别、对象特征和动态用户行为）的推荐系统中的新算法。

    arXiv:2402.09130v1 Announce Type: new Abstract: The enormous development of the Internet, both in the geographical scale and in the area of using its possibilities in everyday life, determines the creation and collection of huge amounts of data. Due to the scale, it is not possible to analyse them using traditional methods, therefore it makes a necessary to use modern methods and techniques. Such methods are provided, among others, by the area of recommendations. The aim of this study is to present a new algorithm in the area of recommendation systems, the algorithm based on data from various sets of information, both static (categories of objects, features of objects) and dynamic (user behaviour).
    
[^4]: 通过符合性预测实现置信度感知的顺序推荐系统的精细调整

    Confidence-aware Fine-tuning of Sequential Recommendation Systems via Conformal Prediction

    [https://arxiv.org/abs/2402.08976](https://arxiv.org/abs/2402.08976)

    本研究提出了CPFT框架，通过在顺序推荐系统中精细调整过程中结合交叉熵损失函数和基于符合性预测的损失函数，增强了推荐系统的置信度。CPFT动态生成潜在真实值的项目集合，提升了训练过程中的性能，并提高了推荐的准确性和可信度。

    

    在顺序推荐系统中，通常使用交叉熵损失函数，但在训练过程中未能利用项目置信度分数。为了认识到置信度在将训练目标与评估指标对齐中的关键作用，我们提出了CPFT，这是一个多功能的框架，通过在精细调整过程中将基于符合性预测的损失函数与交叉熵损失函数相结合，增强了推荐系统的置信度。CPFT动态生成一组具有高概率包含真实值的项目，通过将验证数据纳入训练过程而不损害其在模型选择中的作用，丰富了训练过程。这种创新的方法与基于符合性预测的损失函数相结合，更专注于改善推荐集合，从而提高潜在项目预测的置信度。通过通过基于符合性预测的损失函数对项目置信度进行精细调整，CPFT显著提高了模型性能，提供了更精确和可信的推荐。

    arXiv:2402.08976v1 Announce Type: new Abstract: In Sequential Recommendation Systems, Cross-Entropy (CE) loss is commonly used but fails to harness item confidence scores during training. Recognizing the critical role of confidence in aligning training objectives with evaluation metrics, we propose CPFT, a versatile framework that enhances recommendation confidence by integrating Conformal Prediction (CP)-based losses with CE loss during fine-tuning. CPFT dynamically generates a set of items with a high probability of containing the ground truth, enriching the training process by incorporating validation data without compromising its role in model selection. This innovative approach, coupled with CP-based losses, sharpens the focus on refining recommendation sets, thereby elevating the confidence in potential item predictions. By fine-tuning item confidence through CP-based losses, CPFT significantly enhances model performance, leading to more precise and trustworthy recommendations th
    
[^5]: 通过替代训练增强ID和文本融合在基于会话的推荐中

    Enhancing ID and Text Fusion via Alternative Training in Session-based Recommendation

    [https://arxiv.org/abs/2402.08921](https://arxiv.org/abs/2402.08921)

    本论文研究在基于会话的推荐中增强ID和文本的融合效果。研究发现，通过简单融合的方法并不能始终超越最佳的单模态方法。进一步的调查表明，这可能是由于融合方法中存在的ID和文本模态不平衡问题。

    

    近年来，基于会话的推荐引起了越来越多的关注，其目标是根据用户在会话中的历史行为提供定制的建议。为了推进这个领域，已经开发了许多方法，其中基于ID的方法通常表现出有希望的性能。然而，这些方法在长尾项目方面经常面临挑战，并且忽视了其他丰富的信息形式，特别是有价值的文本语义信息。为了整合文本信息，引入了各种方法，主要是遵循一个简单的融合框架。令人惊讶的是，我们观察到融合这两种模态并不始终优于遵循简单融合框架的最佳单模态。进一步的调查揭示了简单融合中潜在的不平衡问题，其中ID占主导地位，而文本模态则未充分训练。这表明意外观察可能源于简单融合的潜在问题。

    arXiv:2402.08921v1 Announce Type: cross Abstract: Session-based recommendation has gained increasing attention in recent years, with its aim to offer tailored suggestions based on users' historical behaviors within sessions.   To advance this field, a variety of methods have been developed, with ID-based approaches typically demonstrating promising performance. However, these methods often face challenges with long-tail items and overlook other rich forms of information, notably valuable textual semantic information. To integrate text information, various methods have been introduced, mostly following a naive fusion framework. Surprisingly, we observe that fusing these two modalities does not consistently outperform the best single modality by following the naive fusion framework. Further investigation reveals an potential imbalance issue in naive fusion, where the ID dominates and text modality is undertrained. This suggests that the unexpected observation may stem from naive fusion's
    
[^6]: eCeLLM：从大规模高质量指导数据中将大型语言模型推广到电子商务中

    eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data

    [https://arxiv.org/abs/2402.08831](https://arxiv.org/abs/2402.08831)

    本文利用开源的大规模高质量指导数据集ECInstruct，通过指导调优通用语言模型，开发了一系列电子商务LLMs（eCeLLM），在电子商务中表现出了显著的优势。

    

    通过在开发有效的电子商务模型方面做出巨大努力，传统的电子商务模型在通用电子商务建模上取得了有限的成功，并且在新用户和新产品上的表现不佳——这是一个典型的领域外泛化挑战。与此同时，大型语言模型(LLMs)在许多领域展示出了出色的通用建模和领域外泛化能力。为了充分发挥它们在电子商务中的作用，本文构建了ECInstruct，这是第一个面向电子商务的开源、大规模和高质量的指导数据集。利用ECInstruct，我们通过指导调优通用语言模型开发了一系列电子商务LLMs，称为eCeLLM。我们的综合实验和评估表明，eCeLLM模型在内部环境中明显优于基准模型，包括最先进的GPT-4和最先进的特定任务模型。

    arXiv:2402.08831v1 Announce Type: cross Abstract: With tremendous efforts on developing effective e-commerce models, conventional e-commerce models show limited success in generalist e-commerce modeling, and suffer from unsatisfactory performance on new users and new products - a typical out-of-domain generalization challenge. Meanwhile, large language models (LLMs) demonstrate outstanding performance in generalist modeling and out-of-domain generalizability in many fields. Toward fully unleashing their power for e-commerce, in this paper, we construct ECInstruct, the first open-sourced, large-scale, and high-quality benchmark instruction dataset for e-commerce. Leveraging ECInstruct, we develop eCeLLM, a series of e-commerce LLMs, by instruction-tuning general-purpose LLMs. Our comprehensive experiments and evaluation demonstrate that eCeLLM models substantially outperform baseline models, including the most advanced GPT-4, and the state-of-the-art task-specific models in in-domain ev
    
[^7]: 多标签零样本产品属性值提取

    Multi-Label Zero-Shot Product Attribute-Value Extraction

    [https://arxiv.org/abs/2402.08802](https://arxiv.org/abs/2402.08802)

    本文提出了一种用于在没有标注数据的情况下高效提取新产品中未见过的属性值的方法，通过构建异构超图并利用归纳推理来捕捉复杂的高阶关系，以更好地学习。

    

    电子商务平台应提供详细的产品描述（属性值）以实现有效的产品搜索和推荐。然而，对于新产品，通常无法获得属性值信息。为了预测未见过的属性值，需要大量带标签的训练数据来训练传统的监督学习模型。通常，手动标记大量新产品的配置文件是困难、耗时且昂贵的。在本文中，我们提出了一种新颖的方法，以在没有标注数据的情况下（零样本设置）高效地从新产品中提取未见过的属性值。我们提出了 HyperPAVE，一种利用异构超图进行归纳推理的多标签零样本属性值提取模型。特别是，我们提出的技术构建了异构超图，以捕捉复杂的高阶关系（即用户行为信息），以更好地学习。

    arXiv:2402.08802v1 Announce Type: new Abstract: E-commerce platforms should provide detailed product descriptions (attribute values) for effective product search and recommendation. However, attribute value information is typically not available for new products. To predict unseen attribute values, large quantities of labeled training data are needed to train a traditional supervised learning model. Typically, it is difficult, time-consuming, and costly to manually label large quantities of new product profiles. In this paper, we propose a novel method to efficiently and effectively extract unseen attribute values from new products in the absence of labeled data (zero-shot setting). We propose HyperPAVE, a multi-label zero-shot attribute value extraction model that leverages inductive inference in heterogeneous hypergraphs. In particular, our proposed technique constructs heterogeneous hypergraphs to capture complex higher-order relations (i.e. user behavior information) to learn more 
    
[^8]: 使用LoCo和M2-BERT进行基准测试和构建长上下文检索模型

    Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT

    [https://arxiv.org/abs/2402.07440](https://arxiv.org/abs/2402.07440)

    该论文介绍了LoCoV1，一个用于评估长上下文检索性能的新型基准测试，并提出了M2-BERT检索编码器，用于处理长上下文检索，解决了如何评估性能、预训练语言模型以及如何进行微调的挑战。

    

    检索管道是许多机器学习系统中的重要组成部分，在文档很长（例如10K个标记或更多）且需要在整个文本中合成信息来确定相关文档的领域中表现不佳。开发适用于这些领域的长上下文检索编码器面临三个挑战：（1）如何评估长上下文检索性能，（2）如何预训练基本语言模型以表示短上下文（对应查询）和长上下文（对应文档），以及（3）如何根据GPU内存限制下的批量大小限制对该模型进行微调。为了解决这些挑战，我们首先介绍了LoCoV1，这是一个新颖的12个任务基准测试，用于测量在不可分块或不有效的情况下的长上下文检索。接下来，我们提出了M2-BERT检索编码器，这是一个80M参数状态空间编码器模型，采用Monarch Mixer架构构建，能够进行可扩展的检索。

    Retrieval pipelines-an integral component of many machine learning systems-perform poorly in domains where documents are long (e.g., 10K tokens or more) and where identifying the relevant document requires synthesizing information across the entire text. Developing long-context retrieval encoders suitable for these domains raises three challenges: (1) how to evaluate long-context retrieval performance, (2) how to pretrain a base language model to represent both short contexts (corresponding to queries) and long contexts (corresponding to documents), and (3) how to fine-tune this model for retrieval under the batch size limitations imposed by GPU memory constraints. To address these challenges, we first introduce LoCoV1, a novel 12 task benchmark constructed to measure long-context retrieval where chunking is not possible or not effective. We next present the M2-BERT retrieval encoder, an 80M parameter state-space encoder model built from the Monarch Mixer architecture, capable of scali
    
[^9]: 通过检索示范进行上下文学习的语言模型：一项综述

    In-context Learning with Retrieved Demonstrations for Language Models: A Survey. (arXiv:2401.11624v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.11624](http://arxiv.org/abs/2401.11624)

    本综述调查了一种名为检索示范的方法，它通过使用特定于输入查询的示范来提高语言模型的少量样本情境学习（ICL）能力。这种方法不仅提高了学习效率和可扩展性，还减少了手动示例选择中的偏见。

    

    语言模型，特别是预训练的大型语言模型，已展示出卓越的能力，可以在输入上下文中进行少量样本的情境学习（ICL），并在新任务上具有适应能力。然而，模型的ICL能力对于少样本示范的选择是敏感的。最近的一项研究进展是检索针对每个输入查询定制的示范。示范检索的实现相对简单，利用现有的数据库和检索系统。这不仅提高了学习过程的效率和可扩展性，而且已经证明可以减少手动示例选择中的偏见。鉴于令人鼓舞的结果和在检索示范的ICL方面不断增长的研究，我们进行了广泛的研究综述。在这项综述中，我们讨论和比较了检索模型的不同设计选择，检索训练

    Language models, especially pre-trained large language models, have showcased remarkable abilities as few-shot in-context learners (ICL), adept at adapting to new tasks with just a few demonstrations in the input context. However, the model's ability to perform ICL is sensitive to the choice of the few-shot demonstrations. Instead of using a fixed set of demonstrations, one recent development is to retrieve demonstrations tailored to each input query. The implementation of demonstration retrieval is relatively straightforward, leveraging existing databases and retrieval systems. This not only improves the efficiency and scalability of the learning process but also has been shown to reduce biases inherent in manual example selection. In light of the encouraging results and growing research in ICL with retrieved demonstrations, we conduct an extensive review of studies in this area. In this survey, we discuss and compare different design choices for retrieval models, retrieval training p
    
[^10]: 协作情境化：填补协同过滤和预训练语言模型之间的差距

    Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model. (arXiv:2310.09400v1 [cs.IR])

    [http://arxiv.org/abs/2310.09400](http://arxiv.org/abs/2310.09400)

    本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。

    

    传统的推荐系统在建模用户和物品时 heavily relied on identity representations (IDs)，而预训练语言模型 (PLM) 的兴起丰富了对情境化物品描述的建模。然而，尽管 PLM 在解决 few-shot、zero-shot 或统一建模场景方面非常有效，但常常忽视了关键的协同过滤信号。这种忽视带来了两个紧迫的挑战：(1) 协作情境化，即协同信号与情境化表示的无缝集成。(2) 在保留它们的情境语义的同时，弥合基于ID的表示和情境化表示之间的表示差距的必要性。在本文中，我们提出了CollabContext，一种新颖的模型，能够巧妙地将协同过滤信号与情境化表示结合起来，并将这些表示对齐在情境空间内，保留了重要的情境语义。实验结果表明...

    Traditional recommender systems have heavily relied on identity representations (IDs) to model users and items, while the ascendancy of pre-trained language model (PLM) encoders has enriched the modeling of contextual item descriptions. However, PLMs, although effective in addressing few-shot, zero-shot, or unified modeling scenarios, often neglect the crucial collaborative filtering signal. This neglect gives rise to two pressing challenges: (1) Collaborative Contextualization, the seamless integration of collaborative signals with contextual representations. (2) the imperative to bridge the representation gap between ID-based representations and contextual representations while preserving their contextual semantics. In this paper, we propose CollabContext, a novel model that adeptly combines collaborative filtering signals with contextual representations and aligns these representations within the contextual space, preserving essential contextual semantics. Experimental results acros
    

