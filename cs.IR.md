# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer.](http://arxiv.org/abs/2401.01497) | 本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。 |

# 详细

[^1]: 一个预训练的顺序推荐框架：基于流行度动态的零-shot迁移

    A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer. (arXiv:2401.01497v1 [cs.IR])

    [http://arxiv.org/abs/2401.01497](http://arxiv.org/abs/2401.01497)

    本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。

    

    顺序推荐对于在线应用如电子商务、视频流媒体和社交媒体的成功至关重要。尽管模型架构不断改进，但对于每个新的应用领域，我们仍然需要从头训练一个新模型以获得高质量的推荐。另一方面，预训练的语言和视觉模型已经在零-shot或少-shot适应新应用领域方面取得了巨大成功。受到同行AI领域预训练模型成功的启发，我们提出了一种新颖的预训练顺序推荐框架：PrepRec。我们通过建模物品流行度动态来学习通用物品表示。通过在五个真实世界数据集上的大量实验证明，PrepRec在没有任何辅助信息的情况下不仅能够零-shot迁移到新领域，并且与同类最先进的顺序推荐模型相比，模型大小仅相当一小部分的情况下，可以实现竞争性的性能。

    Sequential recommenders are crucial to the success of online applications, \eg e-commerce, video streaming, and social media. While model architectures continue to improve, for every new application domain, we still have to train a new model from scratch for high quality recommendations. On the other hand, pre-trained language and vision models have shown great success in zero-shot or few-shot adaptation to new application domains. Inspired by the success of pre-trained models in peer AI fields, we propose a novel pre-trained sequential recommendation framework: PrepRec. We learn universal item representations by modeling item popularity dynamics. Through extensive experiments on five real-world datasets, we show that PrepRec, without any auxiliary information, can not only zero-shot transfer to a new domain, but achieve competitive performance compared to state-of-the-art sequential recommender models with only a fraction of the model size. In addition, with a simple post-hoc interpol
    

