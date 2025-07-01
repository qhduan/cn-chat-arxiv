# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Autonomous Large Language Model Agent for Chemical Literature Data Mining](https://arxiv.org/abs/2402.12993) | 介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。 |
| [^2] | [A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer.](http://arxiv.org/abs/2401.01497) | 本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。 |

# 详细

[^1]: 用于化学文献数据挖掘的自主大型语言模型代理

    An Autonomous Large Language Model Agent for Chemical Literature Data Mining

    [https://arxiv.org/abs/2402.12993](https://arxiv.org/abs/2402.12993)

    介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。

    

    化学合成对于推动材料合成和药物发现至关重要，影响着包括环境科学和医疗保健在内的各个领域。化学领域的技术上升使得产生了大量的化学数据，挑战研究人员去识别模式并细化合成过程。人工智能通过分析数据来优化合成并提高产量。然而，人工智能在处理文献数据方面面临着挑战，因为化学文献的结构不规整，写作风格多样。为了克服这些困难，我们引入了一个端到端的人工智能代理框架，能够从广泛的化学文献中高保真地提取信息。这个人工智能代理采用大型语言模型（LLMs）进行快速生成和迭代优化。它充当化学助手的角色，自动化数据收集和分析，从而节省人力并提高性能。

    arXiv:2402.12993v1 Announce Type: cross  Abstract: Chemical synthesis, which is crucial for advancing material synthesis and drug discovery, impacts various sectors including environmental science and healthcare. The rise of technology in chemistry has generated extensive chemical data, challenging researchers to discern patterns and refine synthesis processes. Artificial intelligence (AI) helps by analyzing data to optimize synthesis and increase yields. However, AI faces challenges in processing literature data due to the unstructured format and diverse writing style of chemical literature. To overcome these difficulties, we introduce an end-to-end AI agent framework capable of high-fidelity extraction from extensive chemical literature. This AI agent employs large language models (LLMs) for prompt generation and iterative optimization. It functions as a chemistry assistant, automating data collection and analysis, thereby saving manpower and enhancing performance. Our framework's ef
    
[^2]: 一个预训练的顺序推荐框架：基于流行度动态的零-shot迁移

    A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer. (arXiv:2401.01497v1 [cs.IR])

    [http://arxiv.org/abs/2401.01497](http://arxiv.org/abs/2401.01497)

    本文提出了一个预训练的顺序推荐框架PrepRec，通过建模物品流行度动态学习通用物品表示。在大量实验证明，PrepRec可以零-shot迁移到新领域，并且在模型大小上只有很小一部分，并且实现了竞争性的性能。

    

    顺序推荐对于在线应用如电子商务、视频流媒体和社交媒体的成功至关重要。尽管模型架构不断改进，但对于每个新的应用领域，我们仍然需要从头训练一个新模型以获得高质量的推荐。另一方面，预训练的语言和视觉模型已经在零-shot或少-shot适应新应用领域方面取得了巨大成功。受到同行AI领域预训练模型成功的启发，我们提出了一种新颖的预训练顺序推荐框架：PrepRec。我们通过建模物品流行度动态来学习通用物品表示。通过在五个真实世界数据集上的大量实验证明，PrepRec在没有任何辅助信息的情况下不仅能够零-shot迁移到新领域，并且与同类最先进的顺序推荐模型相比，模型大小仅相当一小部分的情况下，可以实现竞争性的性能。

    Sequential recommenders are crucial to the success of online applications, \eg e-commerce, video streaming, and social media. While model architectures continue to improve, for every new application domain, we still have to train a new model from scratch for high quality recommendations. On the other hand, pre-trained language and vision models have shown great success in zero-shot or few-shot adaptation to new application domains. Inspired by the success of pre-trained models in peer AI fields, we propose a novel pre-trained sequential recommendation framework: PrepRec. We learn universal item representations by modeling item popularity dynamics. Through extensive experiments on five real-world datasets, we show that PrepRec, without any auxiliary information, can not only zero-shot transfer to a new domain, but achieve competitive performance compared to state-of-the-art sequential recommender models with only a fraction of the model size. In addition, with a simple post-hoc interpol
    

