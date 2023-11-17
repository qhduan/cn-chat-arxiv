# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Personalized Elastic Embedding Learning for On-Device Recommendation.](http://arxiv.org/abs/2306.10532) | 本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架考虑了设备和用户的异质性与动态资源约束，并在一次性生成个性化嵌入的基础上进行推荐。 |
| [^2] | [Continually Updating Generative Retrieval on Dynamic Corpora.](http://arxiv.org/abs/2305.18952) | 本文研究了动态语料库上的生成检索。实验结果表明，在静态设置下，生成检索效果优于双编码器，但在动态设置下情况相反。通过使用参数高效的预训练方法，我们的模型DynamicGR在新的语料库上展现出了意外的性能。 |

# 详细

[^1]: 个性化弹性嵌入学习用于设备上的推荐

    Personalized Elastic Embedding Learning for On-Device Recommendation. (arXiv:2306.10532v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.10532](http://arxiv.org/abs/2306.10532)

    本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架考虑了设备和用户的异质性与动态资源约束，并在一次性生成个性化嵌入的基础上进行推荐。

    

    为了解决隐私问题并减少网络延迟，近年来一直有将在云端训练的臃肿的推荐模型压缩并在资源受限的设备上部署紧凑的推荐器模型以进行实时推荐的趋势。现有的解决方案通常忽视了设备异质性和用户异质性。它们要么要求所有设备共享相同的压缩模型，要么要求具有相同资源预算的设备共享相同模型。然而，即使是具有相同设备的用户可能也具有不同的偏好。此外，它们假设设备上的推荐器可用资源（如内存）是恒定的，这与现实情况不符。鉴于设备和用户的异质性以及动态资源约束，本文提出了一种用于设备上的个性化弹性嵌入学习框架（PEEL），该框架以一次性方式为具有不同内存预算的设备生成个性化的嵌入。

    To address privacy concerns and reduce network latency, there has been a recent trend of compressing cumbersome recommendation models trained on the cloud and deploying compact recommender models to resource-limited devices for real-time recommendation. Existing solutions generally overlook device heterogeneity and user heterogeneity. They either require all devices to share the same compressed model or the devices with the same resource budget to share the same model. However, even users with the same devices may have different preferences. In addition, they assume the available resources (e.g., memory) for the recommender on a device are constant, which is not reflective of reality. In light of device and user heterogeneities as well as dynamic resource constraints, this paper proposes a Personalized Elastic Embedding Learning framework (PEEL) for on-device recommendation, which generates personalized embeddings for devices with various memory budgets in once-for-all manner, efficien
    
[^2]: 在动态语料库上持续更新生成检索

    Continually Updating Generative Retrieval on Dynamic Corpora. (arXiv:2305.18952v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.18952](http://arxiv.org/abs/2305.18952)

    本文研究了动态语料库上的生成检索。实验结果表明，在静态设置下，生成检索效果优于双编码器，但在动态设置下情况相反。通过使用参数高效的预训练方法，我们的模型DynamicGR在新的语料库上展现出了意外的性能。

    

    先前关于信息检索(IR)的大多数研究假设语料库是静态的，而实际世界中的文档是不断更新的。本文将知识的动态性引入检索系统中，将检索视为动态的知识库，更符合真实环境。我们对双编码器和生成检索进行全面评估，利用StreamingQA基准测试用于时态知识更新。我们的初步结果显示，在静态设置下，生成检索优于双编码器，但在动态设置下情况相反。然而，令人惊讶的是，当我们利用参数高效的预训练方法增强生成检索对新语料库的适应性时，我们的模型Dynamic Generative Retrieval (DynamicGR)展现出意外的发现。它能够在其内部索引中高效压缩新的知识，

    The majority of prior work on information retrieval (IR) assumes that the corpus is static, whereas in the real world, the documents are continually updated. In this paper, we incorporate often overlooked dynamic nature of knowledge into the retrieval systems. Our work treats retrieval not as static archives but as dynamic knowledge bases better aligned with real-world environments. We conduct a comprehensive evaluation of dual encoders and generative retrieval, utilizing the StreamingQA benchmark designed for the temporal knowledge updates. Our initial results show that while generative retrieval outperforms dual encoders in static settings, the opposite is true in dynamic settings. Surprisingly, however, when we utilize a parameter-efficient pre-training method to enhance adaptability of generative retrieval to new corpora, our resulting model, Dynamic Generative Retrieval (DynamicGR), exhibits unexpected findings. It (1) efficiently compresses new knowledge in their internal index, 
    

