# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CoProver: A Recommender System for Proof Construction.](http://arxiv.org/abs/2304.10486) | CoProver是一种证明构造推荐系统，能够从证明构造过程中的过去操作中学习，并通过探索存储在ITP中的关于以前证明的知识来提供有用的建议。 |
| [^2] | [SARF: Aliasing Relation Assisted Self-Supervised Learning for Few-shot Relation Reasoning.](http://arxiv.org/abs/2304.10297) | 本文提出了一个新的自监督学习模型SARF，通过利用同义关系辅助提高了少样本关系推理的泛化性能和准确性，经过实验验证超越了目前最先进的方法。 |
| [^3] | [ZEBRA: Z-order Curve-based Event Retrieval Approach to Efficiently Explore Automotive Data.](http://arxiv.org/abs/2304.10232) | 本论文提出了基于Z形曲线的事件检索方法，可用于高效探索汽车数据。该方法可用于处理大型车队的非注释数据集，使自动驾驶软件更好地掌握现实场景下的驾驶情况。 |
| [^4] | [CoT-MoTE: Exploring ConTextual Masked Auto-Encoder Pre-training with Mixture-of-Textual-Experts for Passage Retrieval.](http://arxiv.org/abs/2304.10195) | 本文提出了一种在段落检索中采用基于文本专家混合的上下文掩码自编码器预训练的新方法，可以有效改进嵌入空间的判别效果。 |
| [^5] | [Is ChatGPT a Good Recommender? A Preliminary Study.](http://arxiv.org/abs/2304.10149) | 本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。 |
| [^6] | [Visualising Personal Data Flows: Insights from a Case Study of Booking.com.](http://arxiv.org/abs/2304.09603) | 本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。 |

# 详细

[^1]: CoProver: 一种证明构造推荐系统

    CoProver: A Recommender System for Proof Construction. (arXiv:2304.10486v1 [cs.LO])

    [http://arxiv.org/abs/2304.10486](http://arxiv.org/abs/2304.10486)

    CoProver是一种证明构造推荐系统，能够从证明构造过程中的过去操作中学习，并通过探索存储在ITP中的关于以前证明的知识来提供有用的建议。

    

    交互式定理证明工具(ITPs)是形式化方法专家工具库中不可或缺的一部分, 用于构建和(形式)验证证明. 证明的复杂性和通常需要的专业水平往往会阻碍ITPs的采用. 最近的一系列工作研究了将基于ITP用户活动跟踪的机器学习模型作为实现完全自动化的可行途径的方法. 虽然这是一条有价值的研究线, 但仍然有许多问题需要人类的监督才能完全完成，因此将学习方法应用于协助用户提供有用建议可能更为有益。跟随用户协助的思路，我们介绍了CoProver，这是一个基于transformers的证明推荐系统，能够从证明构造过程中的过去操作中学习，同时探索存储在ITP中的关于以前证明的知识。

    Interactive Theorem Provers (ITPs) are an indispensable tool in the arsenal of formal method experts as a platform for construction and (formal) verification of proofs. The complexity of the proofs in conjunction with the level of expertise typically required for the process to succeed can often hinder the adoption of ITPs. A recent strain of work has investigated methods to incorporate machine learning models trained on ITP user activity traces as a viable path towards full automation. While a valuable line of investigation, many problems still require human supervision to be completed fully, thus applying learning methods to assist the user with useful recommendations can prove more fruitful.  Following the vein of user assistance, we introduce CoProver, a proof recommender system based on transformers, capable of learning from past actions during proof construction, all while exploring knowledge stored in the ITP concerning previous proofs. CoProver employs a neurally learnt sequenc
    
[^2]: SARF: 利用同义关系辅助的自监督学习以进行少样本关系推理

    SARF: Aliasing Relation Assisted Self-Supervised Learning for Few-shot Relation Reasoning. (arXiv:2304.10297v1 [cs.LG])

    [http://arxiv.org/abs/2304.10297](http://arxiv.org/abs/2304.10297)

    本文提出了一个新的自监督学习模型SARF，通过利用同义关系辅助提高了少样本关系推理的泛化性能和准确性，经过实验验证超越了目前最先进的方法。

    

    知识图谱中针对少量数据、长尾的关系推理（FS-KGR）近年来因其实用性而备受关注。之前的方法需要手动构建元关系集来预训练，导致了大量的劳动成本。自监督学习（SSL）被视为解决这个问题的方案，但在FS-KGR任务中仍处于早期阶段。此外，大多数现有方法忽略了利用与目标数据稀少关系具有类似上下文语义的同义关系（AR）的有益信息。因此，我们提出了一种使用同义关系辅助的自监督学习模型，命名为SARF。具体地，我们的模型设计了四个主要组成部分，即SSL推理模块、AR辅助机制、融合模块和评分函数。我们首先以生成式的方式生成共现模式的表示，同时设计AR辅助机制来捕捉数据稀少关系的相似上下文语义。然后，这两个表示被融合起来生成综合特征表示。最后，我们采用评分函数来预测基于此特征表示的目标关系。两个广泛使用的基准测试上的实验结果表明SARF优于最先进的方法。

    Few-shot relation reasoning on knowledge graphs (FS-KGR) aims to infer long-tail data-poor relations, which has drawn increasing attention these years due to its practicalities. The pre-training of previous methods needs to manually construct the meta-relation set, leading to numerous labor costs. Self-supervised learning (SSL) is treated as a solution to tackle the issue, but still at an early stage for FS-KGR task. Moreover, most of the existing methods ignore leveraging the beneficial information from aliasing relations (AR), i.e., data-rich relations with similar contextual semantics to the target data-poor relation. Therefore, we proposed a novel Self-Supervised Learning model by leveraging Aliasing Relations to assist FS-KGR, termed SARF. Concretely, four main components are designed in our model, i.e., SSL reasoning module, AR-assisted mechanism, fusion module, and scoring function. We first generate the representation of the co-occurrence patterns in a generative manner. Meanwh
    
[^3]: 基于Z形曲线的事件检索方法在高效探索汽车数据中的应用

    ZEBRA: Z-order Curve-based Event Retrieval Approach to Efficiently Explore Automotive Data. (arXiv:2304.10232v1 [cs.IR])

    [http://arxiv.org/abs/2304.10232](http://arxiv.org/abs/2304.10232)

    本论文提出了基于Z形曲线的事件检索方法，可用于高效探索汽车数据。该方法可用于处理大型车队的非注释数据集，使自动驾驶软件更好地掌握现实场景下的驾驶情况。

    

    评估自动驾驶软件的性能主要由现实世界收集的数据驱动。虽然专业测试驾驶员得到技术支持来半自动地注释驾驶操作，更好地识别事件，但是大型车队中的简单数据记录器通常缺乏自动和详细的事件分类，因此需要额外的后处理工作。然而，专业测试驾驶员的数据质量显然比缺少标签的大型车队更高，但大型车队的非注释数据集更具有代表性，可用于自动驾驶车辆处理的典型、现实的驾驶情景。但是，在后处理期间添加有价值的注释越来越昂贵，虽然扩大大型车队的数据相对较简单。在本文中，我们利用Z形空间填充曲线系统地减少数据维数，

    Evaluating the performance of software for automated vehicles is predominantly driven by data collected from the real world. While professional test drivers are supported with technical means to semi-automatically annotate driving maneuvers to allow better event identification, simple data loggers in large vehicle fleets typically lack automatic and detailed event classification and hence, extra effort is needed when post-processing such data. Yet, the data quality from professional test drivers is apparently higher than the one from large fleets where labels are missing, but the non-annotated data set from large vehicle fleets is much more representative for typical, realistic driving scenarios to be handled by automated vehicles. However, while growing the data from large fleets is relatively simple, adding valuable annotations during post-processing has become increasingly expensive. In this paper, we leverage Z-order space-filling curves to systematically reduce data dimensionality
    
[^4]: CoT-MoTE：探索基于文本专家混合的上下文掩码自编码器预训练在段落检索中的应用

    CoT-MoTE: Exploring ConTextual Masked Auto-Encoder Pre-training with Mixture-of-Textual-Experts for Passage Retrieval. (arXiv:2304.10195v1 [cs.CL])

    [http://arxiv.org/abs/2304.10195](http://arxiv.org/abs/2304.10195)

    本文提出了一种在段落检索中采用基于文本专家混合的上下文掩码自编码器预训练的新方法，可以有效改进嵌入空间的判别效果。

    

    段落检索旨在从大规模开放式语料库中检索相关段落。上下文掩码自编码器在单体双编码器的表示瓶颈预训练中证明有效，并常常被采用为基本的检索架构，在预训练和微调阶段中将查询和段落编码为它们的潜在嵌入空间。然而，简单地共享或分离双编码器的参数会导致嵌入空间的不平衡判别。本文中，我们提出了一种预先训练具有文本专家混合的上下文掩码自编码器（CoT-MoTE）。具体来说，我们为查询和段落的不同属性分别编码文本特定的专家。同时，仍保留一个共享的自我注意层，用于统一的注意建模。对大规模段落检索基准测试的结果显示稳定的改进。

    Passage retrieval aims to retrieve relevant passages from large collections of the open-domain corpus. Contextual Masked Auto-Encoding has been proven effective in representation bottleneck pre-training of a monolithic dual-encoder for passage retrieval. Siamese or fully separated dual-encoders are often adopted as basic retrieval architecture in the pre-training and fine-tuning stages for encoding queries and passages into their latent embedding spaces. However, simply sharing or separating the parameters of the dual-encoder results in an imbalanced discrimination of the embedding spaces. In this work, we propose to pre-train Contextual Masked Auto-Encoder with Mixture-of-Textual-Experts (CoT-MoTE). Specifically, we incorporate textual-specific experts for individually encoding the distinct properties of queries and passages. Meanwhile, a shared self-attention layer is still kept for unified attention modeling. Results on large-scale passage retrieval benchmarks show steady improvemen
    
[^5]: ChatGPT是一个好的推荐算法吗？初步研究

    Is ChatGPT a Good Recommender? A Preliminary Study. (arXiv:2304.10149v1 [cs.IR])

    [http://arxiv.org/abs/2304.10149](http://arxiv.org/abs/2304.10149)

    本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。

    

    推荐系统在过去几十年中取得了显著进展并得到广泛应用。然而，大多数传统推荐方法都是特定任务的，因此缺乏有效的泛化能力。最近，ChatGPT的出现通过增强对话模型的能力，显著推进了NLP任务。尽管如此，ChatGPT在推荐领域的应用还没有得到充分的研究。在本文中，我们采用ChatGPT作为通用推荐模型，探讨它将从大规模语料库中获得的广泛语言和世界知识转移到推荐场景中的潜力。具体而言，我们设计了一组提示，并评估ChatGPT在五个推荐场景中的表现。与传统的推荐方法不同的是，在整个评估过程中我们不微调ChatGPT，仅依靠提示自身将推荐任务转化为自然语言。

    Recommendation systems have witnessed significant advancements and have been widely used over the past decades. However, most traditional recommendation methods are task-specific and therefore lack efficient generalization ability. Recently, the emergence of ChatGPT has significantly advanced NLP tasks by enhancing the capabilities of conversational models. Nonetheless, the application of ChatGPT in the recommendation domain has not been thoroughly investigated. In this paper, we employ ChatGPT as a general-purpose recommendation model to explore its potential for transferring extensive linguistic and world knowledge acquired from large-scale corpora to recommendation scenarios. Specifically, we design a set of prompts and evaluate ChatGPT's performance on five recommendation scenarios. Unlike traditional recommendation methods, we do not fine-tune ChatGPT during the entire evaluation process, relying only on the prompts themselves to convert recommendation tasks into natural language 
    
[^6]: 可视化个人数据流：以Booking.com为例的案例研究

    Visualising Personal Data Flows: Insights from a Case Study of Booking.com. (arXiv:2304.09603v1 [cs.CR])

    [http://arxiv.org/abs/2304.09603](http://arxiv.org/abs/2304.09603)

    本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。

    

    商业机构持有和处理的个人数据量越来越多。政策和法律不断变化，要求这些公司在收集、存储、处理和共享这些数据方面更加透明。本文报告了我们以Booking.com为案例研究，从他们的隐私政策中提取个人数据流的可视化工作。通过展示该公司如何分享其消费者的个人数据，我们提出了问题，并扩展了有关使用隐私政策告知客户个人数据流范围的挑战和限制的讨论。更重要的是，本案例研究可以为未来更以数据流为导向的隐私政策分析和在复杂商业生态系统中建立更全面的个人数据流本体论的研究提供参考。

    Commercial organisations are holding and processing an ever-increasing amount of personal data. Policies and laws are continually changing to require these companies to be more transparent regarding collection, storage, processing and sharing of this data. This paper reports our work of taking Booking.com as a case study to visualise personal data flows extracted from their privacy policy. By showcasing how the company shares its consumers' personal data, we raise questions and extend discussions on the challenges and limitations of using privacy policy to inform customers the true scale and landscape of personal data flows. More importantly, this case study can inform us about future research on more data flow-oriented privacy policy analysis and on the construction of a more comprehensive ontology on personal data flows in complicated business ecosystems.
    

