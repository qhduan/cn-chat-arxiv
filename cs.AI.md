# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spike No More: Stabilizing the Pre-training of Large Language Models](https://rss.arxiv.org/abs/2312.16903) | 本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。 |
| [^2] | [Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510) | 本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。 |
| [^3] | [Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition.](http://arxiv.org/abs/2401.10337) | 该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。 |

# 详细

[^1]: 别再出现尖峰了：稳定大型语言模型的预训练

    Spike No More: Stabilizing the Pre-training of Large Language Models

    [https://rss.arxiv.org/abs/2312.16903](https://rss.arxiv.org/abs/2312.16903)

    本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。

    

    大型语言模型的预训练经常出现损失尖峰。这些尖峰会降低大型语言模型的性能，有时会破坏预训练。由于预训练需要大量的计算资源，我们应该避免这种尖峰的出现。为了研究损失尖峰的原因，我们关注内部层的梯度。通过理论分析，我们揭示了梯度爆炸的两个原因，并提供了预防梯度爆炸的要求。此外，我们提出了一种通过组合初始化方法和对嵌入进行简单修改来满足要求的方法。我们进行了各种实验证明我们的理论分析的有效性。实验结果表明，在预训练过程中，这种组合方法能够有效地防止尖峰的出现。

    Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a vast computational budget, we should avoid such spikes. To investigate the cause of loss spikes, we focus on gradients of internal layers. Through theoretical analyses, we reveal two causes of the exploding gradients, and provide requirements to prevent the explosion. In addition, we propose a method to satisfy the requirements by combining the initialization method and a simple modification to embeddings. We conduct various experiments to verify our theoretical analyses empirically. Experimental results indicate that the combination is effective in preventing spikes during pre-training.
    
[^2]: 生成式AI代理之间的秘密勾结

    Secret Collusion Among Generative AI Agents

    [https://arxiv.org/abs/2402.07510](https://arxiv.org/abs/2402.07510)

    本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。

    

    最近大型语言模型在能力上的增强为通信的生成式AI代理团队解决联合任务的应用打开了可能性。这引发了关于未经授权分享信息或其他不必要的代理协调形式的隐私和安全挑战。现代隐写术技术可能使这种动态难以检测。本文通过汲取人工智能和安全领域相关概念，全面系统地形式化了生成式AI代理系统中的秘密勾结问题。我们研究了使用隐写术的动机，并提出了各种缓解措施。我们的研究结果是一个模型评估框架，系统地测试了各种形式的秘密勾结所需的能力。我们在各种当代大型语言模型上提供了广泛的实证结果。虽然当前模型的隐写能力仍然有限，但 GPT-4 显示出能力的飞跃，这表明有必要进行进一步的研究。

    Recent capability increases in large language models (LLMs) open up applications in which teams of communicating generative AI agents solve joint tasks. This poses privacy and security challenges concerning the unauthorised sharing of information, or other unwanted forms of agent coordination. Modern steganographic techniques could render such dynamics hard to detect. In this paper, we comprehensively formalise the problem of secret collusion in systems of generative AI agents by drawing on relevant concepts from both the AI and security literature. We study incentives for the use of steganography, and propose a variety of mitigation measures. Our investigations result in a model evaluation framework that systematically tests capabilities required for various forms of secret collusion. We provide extensive empirical results across a range of contemporary LLMs. While the steganographic capabilities of current models remain limited, GPT-4 displays a capability jump suggesting the need fo
    
[^3]: 基于噪声对比估计的低资源安全攻击模式识别匹配框架

    Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition. (arXiv:2401.10337v1 [cs.LG])

    [http://arxiv.org/abs/2401.10337](http://arxiv.org/abs/2401.10337)

    该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。

    

    战术、技术和程序（TTPs）是网络安全领域中复杂的攻击模式，在文本知识库中有详细的描述。在网络安全写作中识别TTPs，通常称为TTP映射，是一个重要而具有挑战性的任务。传统的学习方法通常以经典的多类或多标签分类设置为目标。由于存在大量的类别（即TTPs），标签分布的不均衡和标签空间的复杂层次结构，这种设置限制了模型的学习能力。我们采用了一种不同的学习范式来解决这个问题，其中将文本与TTP标签之间的直接语义相似度决定为文本分配给TTP标签，从而减少了仅仅在大型标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，促进学习过程。

    Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning pr
    

