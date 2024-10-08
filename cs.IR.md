# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Empirical Study of Training ID-Agnostic Multi-modal Sequential Recommenders](https://arxiv.org/abs/2403.17372) | 通过研究现有的多模态相关的顺序推荐方法，提炼出视觉编码器、文本编码器、多模态融合模块和顺序架构这四个核心组件。 |
| [^2] | [Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/abs/2403.06259) | 该论文首次研究了为大型语言模型编辑概念知识，通过构建基准数据集和建立新评估指标，发现现有方法虽然能一定程度上修改概念定义，但也可能造成LLMs中相关实例知识的扭曲，导致性能下降。 |
| [^3] | [A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer.](http://arxiv.org/abs/2401.01497) | 本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。 |
| [^4] | [Continually Updating Generative Retrieval on Dynamic Corpora.](http://arxiv.org/abs/2305.18952) | 本文研究了动态语料库上的生成检索。实验结果表明，在静态设置下，生成检索效果优于双编码器，但在动态设置下情况相反。通过使用参数高效的预训练方法，我们的模型DynamicGR在新的语料库上展现出了意外的性能。 |

# 详细

[^1]: 训练独立于ID的多模态顺序推荐器的实证研究

    An Empirical Study of Training ID-Agnostic Multi-modal Sequential Recommenders

    [https://arxiv.org/abs/2403.17372](https://arxiv.org/abs/2403.17372)

    通过研究现有的多模态相关的顺序推荐方法，提炼出视觉编码器、文本编码器、多模态融合模块和顺序架构这四个核心组件。

    

    顺序推荐旨在基于历史交互来预测未来用户-物品交互。许多顺序推荐方法集中在用户ID和物品ID上，人类通过多模态信号（如文本和图像）感知世界的方式启发了研究人员探索如何构建不使用ID的多模态信息的顺序推荐。然而，多模态学习的复杂性体现在不同的特征提取器、融合方法和预训练模型中。因此，设计一个简单且通用的多模态顺序推荐（MMSR）框架仍然是一个巨大挑战。我们系统总结了现有的多模态相关的顺序推荐方法，并将精华提炼成四个核心组件：视觉编码器、文本编码器、多模态融合模块和顺序架构。沿着这些维度，我们剖析了模型设计，并回答了以下问题

    arXiv:2403.17372v1 Announce Type: new  Abstract: Sequential Recommendation (SR) aims to predict future user-item interactions based on historical interactions. While many SR approaches concentrate on user IDs and item IDs, the human perception of the world through multi-modal signals, like text and images, has inspired researchers to delve into constructing SR from multi-modal information without using IDs. However, the complexity of multi-modal learning manifests in diverse feature extractors, fusion methods, and pre-trained models. Consequently, designing a simple and universal \textbf{M}ulti-\textbf{M}odal \textbf{S}equential \textbf{R}ecommendation (\textbf{MMSR}) framework remains a formidable challenge. We systematically summarize the existing multi-modal related SR methods and distill the essence into four core components: visual encoder, text encoder, multimodal fusion module, and sequential architecture. Along these dimensions, we dissect the model designs, and answer the foll
    
[^2]: 大型语言模型的概念知识编辑

    Editing Conceptual Knowledge for Large Language Models

    [https://arxiv.org/abs/2403.06259](https://arxiv.org/abs/2403.06259)

    该论文首次研究了为大型语言模型编辑概念知识，通过构建基准数据集和建立新评估指标，发现现有方法虽然能一定程度上修改概念定义，但也可能造成LLMs中相关实例知识的扭曲，导致性能下降。

    

    最近，对于大型语言模型（LLMs）的知识编辑引起了越来越多的关注。当前的方法和评估仅探讨了实例级别的编辑，然而LLMs是否具有修改概念的能力仍不清楚。本文首次研究了为LLMs编辑概念知识，通过构建一个新颖的基准数据集ConceptEdit并建立了一套新的评估指标。实验结果表明，尽管现有的编辑方法可以有效地在一定程度上修改概念级别的定义，但它们也有潜力扭曲LLMs中相关的实例知识，导致性能不佳。我们期望这可以激发对更好理解LLMs的进一步进展。我们的项目主页位于https://zjunlp.github.io/project/ConceptEdit。

    arXiv:2403.06259v1 Announce Type: cross  Abstract: Recently, there has been a growing interest in knowledge editing for Large Language Models (LLMs). Current approaches and evaluations merely explore the instance-level editing, while whether LLMs possess the capability to modify concepts remains unclear. This paper pioneers the investigation of editing conceptual knowledge for LLMs, by constructing a novel benchmark dataset ConceptEdit and establishing a suite of new metrics for evaluation. The experimental results reveal that, although existing editing methods can efficiently modify concept-level definition to some extent, they also have the potential to distort the related instantial knowledge in LLMs, leading to poor performance. We anticipate this can inspire further progress in better understanding LLMs. Our project homepage is available at https://zjunlp.github.io/project/ConceptEdit.
    
[^3]: 一个预训练的顺序推荐框架：基于流行度动态的零-shot迁移

    A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer. (arXiv:2401.01497v1 [cs.IR])

    [http://arxiv.org/abs/2401.01497](http://arxiv.org/abs/2401.01497)

    本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。

    

    顺序推荐对于在线应用如电子商务、视频流媒体和社交媒体的成功至关重要。尽管模型架构不断改进，但对于每个新的应用领域，我们仍然需要从头训练一个新模型以获得高质量的推荐。另一方面，预训练的语言和视觉模型已经在零-shot或少-shot适应新应用领域方面取得了巨大成功。受到同行AI领域预训练模型成功的启发，我们提出了一种新颖的预训练顺序推荐框架：PrepRec。我们通过建模物品流行度动态来学习通用物品表示。通过在五个真实世界数据集上的大量实验证明，PrepRec在没有任何辅助信息的情况下不仅能够零-shot迁移到新领域，并且与同类最先进的顺序推荐模型相比，模型大小仅相当一小部分的情况下，可以实现竞争性的性能。

    Sequential recommenders are crucial to the success of online applications, \eg e-commerce, video streaming, and social media. While model architectures continue to improve, for every new application domain, we still have to train a new model from scratch for high quality recommendations. On the other hand, pre-trained language and vision models have shown great success in zero-shot or few-shot adaptation to new application domains. Inspired by the success of pre-trained models in peer AI fields, we propose a novel pre-trained sequential recommendation framework: PrepRec. We learn universal item representations by modeling item popularity dynamics. Through extensive experiments on five real-world datasets, we show that PrepRec, without any auxiliary information, can not only zero-shot transfer to a new domain, but achieve competitive performance compared to state-of-the-art sequential recommender models with only a fraction of the model size. In addition, with a simple post-hoc interpol
    
[^4]: 在动态语料库上持续更新生成检索

    Continually Updating Generative Retrieval on Dynamic Corpora. (arXiv:2305.18952v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.18952](http://arxiv.org/abs/2305.18952)

    本文研究了动态语料库上的生成检索。实验结果表明，在静态设置下，生成检索效果优于双编码器，但在动态设置下情况相反。通过使用参数高效的预训练方法，我们的模型DynamicGR在新的语料库上展现出了意外的性能。

    

    先前关于信息检索(IR)的大多数研究假设语料库是静态的，而实际世界中的文档是不断更新的。本文将知识的动态性引入检索系统中，将检索视为动态的知识库，更符合真实环境。我们对双编码器和生成检索进行全面评估，利用StreamingQA基准测试用于时态知识更新。我们的初步结果显示，在静态设置下，生成检索优于双编码器，但在动态设置下情况相反。然而，令人惊讶的是，当我们利用参数高效的预训练方法增强生成检索对新语料库的适应性时，我们的模型Dynamic Generative Retrieval (DynamicGR)展现出意外的发现。它能够在其内部索引中高效压缩新的知识，

    The majority of prior work on information retrieval (IR) assumes that the corpus is static, whereas in the real world, the documents are continually updated. In this paper, we incorporate often overlooked dynamic nature of knowledge into the retrieval systems. Our work treats retrieval not as static archives but as dynamic knowledge bases better aligned with real-world environments. We conduct a comprehensive evaluation of dual encoders and generative retrieval, utilizing the StreamingQA benchmark designed for the temporal knowledge updates. Our initial results show that while generative retrieval outperforms dual encoders in static settings, the opposite is true in dynamic settings. Surprisingly, however, when we utilize a parameter-efficient pre-training method to enhance adaptability of generative retrieval to new corpora, our resulting model, Dynamic Generative Retrieval (DynamicGR), exhibits unexpected findings. It (1) efficiently compresses new knowledge in their internal index, 
    

