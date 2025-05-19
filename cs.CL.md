# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear Attention Sequence Parallelism](https://arxiv.org/abs/2404.02882) | 提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。 |
| [^2] | [Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning](https://arxiv.org/abs/2403.11083) | 该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。 |
| [^3] | [COBIAS: Contextual Reliability in Bias Assessment](https://arxiv.org/abs/2402.14889) | 我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。 |
| [^4] | [Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?](https://arxiv.org/abs/2311.07564) | 本文研究了作者归属模型在演讲文本中区分发言人的能力，并提出了以会话演讲文本为重点的发言人归属基准。 |

# 详细

[^1]: 线性注意力序列并行化

    Linear Attention Sequence Parallelism

    [https://arxiv.org/abs/2404.02882](https://arxiv.org/abs/2404.02882)

    提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。

    

    序列并行（SP）作为一种处理超出单个GPU内存限制的长序列的流行策略。然而，现有的SP方法并未利用线性注意力特性，导致在基于线性注意力的语言模型中并行效率和可用性不佳。在本文中，我们介绍了线性注意力序列并行（LASP），这是一种专为基于线性注意力的语言模型量身定制的高效SP方法。具体来说，我们设计了一种高效的点对点通信机制，以利用线性注意力的右乘内核技巧，从而显着降低SP的通信开销。我们还通过执行内核融合和中间状态缓存来增强LASP的实际效率，使LASP在GPU集群上的硬件友好性得到提升。此外，我们还精心确保序列级LASP与所有类型的批级数据兼容。

    arXiv:2404.02882v1 Announce Type: cross  Abstract: Sequence Parallel (SP) serves as a prevalent strategy to handle long sequences that exceed the memory limit of a single GPU. However, existing SP methods do not take advantage of linear attention features, resulting in sub-optimal parallelism efficiency and usability for linear attention-based language models. In this paper, we introduce Linear Attention Sequence Parallel (LASP), an efficient SP method tailored to linear attention-based language models. Specifically, we design an efficient point-to-point communication mechanism to leverage the right-product kernel trick of linear attention, which sharply decreases the communication overhead of SP. We also enhance the practical efficiency of LASP by performing kernel fusion and intermediate state caching, making the implementation of LASP hardware-friendly on GPU clusters. Furthermore, we meticulously ensure the compatibility of sequence-level LASP with all types of batch-level data par
    
[^2]: 为多模态异常检测和推理定制视觉-语言基础模型

    Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning

    [https://arxiv.org/abs/2403.11083](https://arxiv.org/abs/2403.11083)

    该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。

    

    异常检测在各种工业场景中十分重要，包括生产线上异常模式的识别和用于质量控制的制造缺陷检测。本研究旨在开发一种适用于多种场景的通用异常检测模型。为实现这一目标，我们将拥有广泛知识和强大推理能力的通用视觉-语言基础模型定制为异常检测器和推理器。具体来说，我们引入了一种多模态提示策略，将领域专家的领域知识作为条件引导模型。我们的方法考虑多模态提示类型，包括任务描述、类别上下文、正常规则和参考图像。另外，我们将多模态输入表示统一为2D图像格式，使其能够

    arXiv:2403.11083v1 Announce Type: cross  Abstract: Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, we aim to develop a generic anomaly detection model applicable across multiple scenarios. To achieve this, we customize generic visual-language foundation models that possess extensive knowledge and robust reasoning abilities into anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers multi-modal prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling m
    
[^3]: COBIAS：偏见评估中的情境可靠性

    COBIAS: Contextual Reliability in Bias Assessment

    [https://arxiv.org/abs/2402.14889](https://arxiv.org/abs/2402.14889)

    我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。

    

    大型语言模型（LLMs）是基于固有偏见数据训练的。以往的去偏见模型研究依赖基准数据集来衡量模型性能。然而，这些数据集由于对偏见的极其主观理解而存在多个缺陷，凸显出对情境探索的迫切需求。我们提出考虑输入用户内容的情境，考虑到输入语句可能存在的多种情况。这种方法将允许培养偏见意识的框架，而不是伤害用户参与的防护设施。我们的贡献有两个方面：(i) 我们创建了一个包含2287个陈词滥调语句以及添加情境要点的数据集；(ii) 我们开发了面向情境的偏见指标和评估分数（COBIAS）来评估语句在衡量偏见方面的情境可靠性。我们的度量是衡量偏见基准数据集情境可靠性的重要预测因子。

    arXiv:2402.14889v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are trained on inherently biased data. Previous works on debiasing models rely on benchmark datasets to measure model performance. However, these datasets suffer from several pitfalls due to the extremely subjective understanding of bias, highlighting a critical need for contextual exploration. We propose understanding the context of user inputs with consideration of the diverse situations in which input statements are possible. This approach would allow for frameworks that foster bias awareness rather than guardrails that hurt user engagement. Our contribution is twofold: (i) we create a dataset of 2287 stereotyped statements augmented with points for adding context; (ii) we develop the Context-Oriented Bias Indicator and Assessment Score (COBIAS) to assess statements' contextual reliability in measuring bias. Our metric is a significant predictor of the contextual reliability of bias-benchmark datasets ($
    
[^4]: 作者归属模型能否区分演讲文本中的发言人？

    Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?

    [https://arxiv.org/abs/2311.07564](https://arxiv.org/abs/2311.07564)

    本文研究了作者归属模型在演讲文本中区分发言人的能力，并提出了以会话演讲文本为重点的发言人归属基准。

    

    作者归属验证是确定两个不同书面样本是否同属一作者的任务，通常涉及对书面文本的归因。本文探讨了转录演讲的归属问题，这带来了新的挑战。其中一个主要挑战是，许多文体特征，如标点和大写，在这种情境下并不具备信息量。另一方面，转录的演讲呈现其他模式，如填充词和回应性声音（例如“嗯”，“嗯，嗯”），这些可能是不同发言人的特征性表现。我们提出了一个新的以会话演讲文本为重点的发言人归属基准。为了限制发言人与话题之间的虚假关联，我们使用会话提示和参与同一对话的发言人构建不同难度的验证试验。通过比较一系列方法，在这一新基准上建立了最新技术水平。

    arXiv:2311.07564v2 Announce Type: replace  Abstract: Authorship verification is the task of determining if two distinct writing samples share the same author and is typically concerned with the attribution of written text. In this paper, we explore the attribution of transcribed speech, which poses novel challenges. The main challenge is that many stylistic features, such as punctuation and capitalization, are not informative in this setting. On the other hand, transcribed speech exhibits other patterns, such as filler words and backchannels (e.g., 'um', 'uh-huh'), which may be characteristic of different speakers. We propose a new benchmark for speaker attribution focused on conversational speech transcripts. To limit spurious associations of speakers with topic, we employ both conversation prompts and speakers participating in the same conversation to construct verification trials of varying difficulties. We establish the state of the art on this new benchmark by comparing a suite of
    

