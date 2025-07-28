# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spike No More: Stabilizing the Pre-training of Large Language Models](https://rss.arxiv.org/abs/2312.16903) | 本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。 |
| [^2] | [How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?](https://arxiv.org/abs/2402.13470) | 研究探讨了在生物医学关系提取任务中领域特异性对于语言模型和指导微调的重要性，对比了在生物医学领域与通用领域训练的模型效果，并探讨了在生物医学数据集上指导微调的模型在性能上的优势。 |
| [^3] | [Comparison of pipeline, sequence-to-sequence, and GPT models for end-to-end relation extraction: experiments with the rare disease use-case](https://arxiv.org/abs/2311.13729) | 本文比较了用于端到端关系抽取的管道、序列到序列和GPT模型，发现管道模型仍然是最佳选择，而序列到序列模型紧随其后；参数量增加八倍的GPT模型甚至比序列到序列模型更差，且比管道模型低10个F1点以上。 |
| [^4] | [Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition.](http://arxiv.org/abs/2401.10337) | 该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。 |

# 详细

[^1]: 别再出现尖峰了：稳定大型语言模型的预训练

    Spike No More: Stabilizing the Pre-training of Large Language Models

    [https://rss.arxiv.org/abs/2312.16903](https://rss.arxiv.org/abs/2312.16903)

    本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。

    

    大型语言模型的预训练经常出现损失尖峰。这些尖峰会降低大型语言模型的性能，有时会破坏预训练。由于预训练需要大量的计算资源，我们应该避免这种尖峰的出现。为了研究损失尖峰的原因，我们关注内部层的梯度。通过理论分析，我们揭示了梯度爆炸的两个原因，并提供了预防梯度爆炸的要求。此外，我们提出了一种通过组合初始化方法和对嵌入进行简单修改来满足要求的方法。我们进行了各种实验证明我们的理论分析的有效性。实验结果表明，在预训练过程中，这种组合方法能够有效地防止尖峰的出现。

    Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a vast computational budget, we should avoid such spikes. To investigate the cause of loss spikes, we focus on gradients of internal layers. Through theoretical analyses, we reveal two causes of the exploding gradients, and provide requirements to prevent the explosion. In addition, we propose a method to satisfy the requirements by combining the initialization method and a simple modification to embeddings. We conduct various experiments to verify our theoretical analyses empirically. Experimental results indicate that the combination is effective in preventing spikes during pre-training.
    
[^2]: 语言模型和生物医学关系提取中的领域特异性有多重要？

    How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?

    [https://arxiv.org/abs/2402.13470](https://arxiv.org/abs/2402.13470)

    研究探讨了在生物医学关系提取任务中领域特异性对于语言模型和指导微调的重要性，对比了在生物医学领域与通用领域训练的模型效果，并探讨了在生物医学数据集上指导微调的模型在性能上的优势。

    

    高价值、数据丰富的生物医学领域常常会使用最前沿的通用自然语言处理技术。过去几年来，生成式语言模型、指导微调和少样本学习成为自然语言处理研究的焦点。因此，预训练于生物医学语料库的生成式语言模型不断涌现，同时也尝试对生物医学指导微调，希望领域特异性可以改善下游任务的性能。鉴于训练这些模型所需的非平凡努力，我们研究它们在关系提取这一关键生物医学自然语言处理任务中是否存在任何益处。具体来说，我们探讨了两个问题：(1) 在生物医学语料库上训练的语言模型是否优于在通用领域语料库上训练的模型？(2) 在生物医学数据集上进行指导微调的模型是否优于在各种数据集上进行微调或者仅仅预训练的模型？我们解决这些问题。

    arXiv:2402.13470v1 Announce Type: new  Abstract: Cutting edge techniques developed in the general NLP domain are often subsequently applied to the high-value, data-rich biomedical domain. The past few years have seen generative language models (LMs), instruction finetuning, and few-shot learning become foci of NLP research. As such, generative LMs pretrained on biomedical corpora have proliferated and biomedical instruction finetuning has been attempted as well, all with the hope that domain specificity improves performance on downstream tasks. Given the nontrivial effort in training such models, we investigate what, if any, benefits they have in the key biomedical NLP task of relation extraction. Specifically, we address two questions: (1) Do LMs trained on biomedical corpora outperform those trained on general domain corpora? (2) Do models instruction finetuned on biomedical datasets outperform those finetuned on assorted datasets or those simply pretrained? We tackle these questions
    
[^3]: 比较用于端到端关系抽取的管道、序列到序列和GPT模型：以罕见疾病用例为实验

    Comparison of pipeline, sequence-to-sequence, and GPT models for end-to-end relation extraction: experiments with the rare disease use-case

    [https://arxiv.org/abs/2311.13729](https://arxiv.org/abs/2311.13729)

    本文比较了用于端到端关系抽取的管道、序列到序列和GPT模型，发现管道模型仍然是最佳选择，而序列到序列模型紧随其后；参数量增加八倍的GPT模型甚至比序列到序列模型更差，且比管道模型低10个F1点以上。

    

    端到端关系抽取（E2ERE）是自然语言处理（NLP）在生物医学中的一个重要而现实的应用。本文旨在使用一个关注罕见疾病、涉及不连续和嵌套实体的复杂数据集，比较E2ERE的三种流行范式。我们使用RareDis信息提取数据集评估了三种竞争方法（用于E2ERE）：实体识别（NER）→关系抽取（RE）管道、联合序列到序列模型和生成式预训练变压器（GPT）模型。我们针对每种方法使用可比的最先进模型和最佳实践，并进行错误分析以评估它们的失败模式。我们的发现显示，管道模型仍然是最佳选择，而序列到序列模型紧随其后；参数量增加八倍的GPT模型甚至比序列到序列模型更差，且比管道模型低10个F1点以上。

    arXiv:2311.13729v2 Announce Type: replace  Abstract: End-to-end relation extraction (E2ERE) is an important and realistic application of natural language processing (NLP) in biomedicine. In this paper, we aim to compare three prevailing paradigms for E2ERE using a complex dataset focused on rare diseases involving discontinuous and nested entities. We use the RareDis information extraction dataset to evaluate three competing approaches (for E2ERE): NER $\rightarrow$ RE pipelines, joint sequence to sequence models, and generative pre-trained transformer (GPT) models. We use comparable state-of-the-art models and best practices for each of these approaches and conduct error analyses to assess their failure modes. Our findings reveal that pipeline models are still the best, while sequence-to-sequence models are not far behind; GPT models with eight times as many parameters are worse than even sequence-to-sequence models and lose to pipeline models by over 10 F1 points. Partial matches and
    
[^4]: 基于噪声对比估计的低资源安全攻击模式识别匹配框架

    Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition. (arXiv:2401.10337v1 [cs.LG])

    [http://arxiv.org/abs/2401.10337](http://arxiv.org/abs/2401.10337)

    该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。

    

    战术、技术和程序（TTPs）是网络安全领域中复杂的攻击模式，在文本知识库中有详细的描述。在网络安全写作中识别TTPs，通常称为TTP映射，是一个重要而具有挑战性的任务。传统的学习方法通常以经典的多类或多标签分类设置为目标。由于存在大量的类别（即TTPs），标签分布的不均衡和标签空间的复杂层次结构，这种设置限制了模型的学习能力。我们采用了一种不同的学习范式来解决这个问题，其中将文本与TTP标签之间的直接语义相似度决定为文本分配给TTP标签，从而减少了仅仅在大型标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，促进学习过程。

    Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning pr
    

