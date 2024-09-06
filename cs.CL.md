# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Speaker Assignment in Speaker-Attributed ASR for Real Meeting Applications](https://arxiv.org/abs/2403.06570) | 该研究提出了一种优化实际会议应用中Speaker-Attributed ASR系统的方法，通过使用语音活动检测输出来微调模型以减少说话者错误率，并探讨了增强说话者嵌入模板提取的策略。 |
| [^2] | [Aligning Large Language Models to a Domain-specific Graph Database](https://arxiv.org/abs/2402.16567) | 该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。 |
| [^3] | [Cost-Efficient Subjective Task Annotation and Modeling through Few-Shot Annotator Adaptation](https://arxiv.org/abs/2402.14101) | 引入了一个新颖的框架，通过少样本标注者调适来实现在主观任务中进行高效标注与建模，最大程度减少标注预算，同时最大化每个标注者的预测性能。 |
| [^4] | [Exploring Group and Symmetry Principles in Large Language Models](https://arxiv.org/abs/2402.06120) | 本文提出了一个基于群组和对称性原理的框架，以评估大型语言模型的推理能力。通过研究四个群组属性，发现这些模型在保持群组属性方面表现不佳。 |
| [^5] | [Language-Guided World Models: A Model-Based Approach to AI Control](https://arxiv.org/abs/2402.01695) | 语言引导的世界模型（LWMs）是一种基于模型的人工智能控制方法，它通过阅读语言描述来捕捉环境动态，提高了代理的沟通效率，并允许人类通过简洁的语言反馈同时改变他们在多个任务上的行为。 |
| [^6] | [Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation.](http://arxiv.org/abs/2401.06477) | Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。 |
| [^7] | [Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review.](http://arxiv.org/abs/2310.14735) | 这篇论文解释了提示工程在释放大型语言模型能力方面的关键作用，探讨了不同的提示方法以及外部插件如何协助减少机器幻想，并指出了未来研究方向的重要性。 |
| [^8] | [PESTS: Persian_English Cross Lingual Corpus for Semantic Textual Similarity.](http://arxiv.org/abs/2305.07893) | 本研究提出了跨语言的语义相似性模型PESTS，并通过波斯语-英语的跨语言语料库来验证模型的准确性。 |

# 详细

[^1]: 改进实际会议应用中说话者归因ASR中的说话者分配

    Improving Speaker Assignment in Speaker-Attributed ASR for Real Meeting Applications

    [https://arxiv.org/abs/2403.06570](https://arxiv.org/abs/2403.06570)

    该研究提出了一种优化实际会议应用中Speaker-Attributed ASR系统的方法，通过使用语音活动检测输出来微调模型以减少说话者错误率，并探讨了增强说话者嵌入模板提取的策略。

    

    过去关于端到端会议转录的研究主要集中在模型架构上，并且大多数是在模拟会议数据上进行评估的。我们提出了一项旨在优化Speaker-Attributed ASR (SA-ASR)系统在现实场景中（如AMI会议语料库）使用的研究，以改进语音段的说话者分配。首先，我们提出了一个针对涉及语音活动检测（VAD）、说话者辨识（SD）和SA-ASR的现实生活应用的流程。其次，我们提倡使用VAD输出段来微调SA-ASR模型，考虑到在测试期间它也被应用于VAD段，并展示了这导致说话者错误率（SER）相对减少高达28％。最后，我们探讨增强从SD输出中提取说话者嵌入模板的策略，这些模板作为SA-ASR系统的输入。我们证明，将它们从SD输出中提取而不是从注释的说话者段中提取，会导致一种...

    arXiv:2403.06570v1 Announce Type: new  Abstract: Past studies on end-to-end meeting transcription have focused on model architecture and have mostly been evaluated on simulated meeting data. We present a novel study aiming to optimize the use of a Speaker-Attributed ASR (SA-ASR) system in real-life scenarios, such as the AMI meeting corpus, for improved speaker assignment of speech segments. First, we propose a pipeline tailored to real-life applications involving Voice Activity Detection (VAD), Speaker Diarization (SD), and SA-ASR. Second, we advocate using VAD output segments to fine-tune the SA-ASR model, considering that it is also applied to VAD segments during test, and show that this results in a relative reduction of Speaker Error Rate (SER) up to 28%. Finally, we explore strategies to enhance the extraction of the speaker embedding templates used as inputs by the SA-ASR system. We show that extracting them from SD output rather than annotated speaker segments results in a rela
    
[^2]: 将大型语言模型对齐到特定领域的图数据库

    Aligning Large Language Models to a Domain-specific Graph Database

    [https://arxiv.org/abs/2402.16567](https://arxiv.org/abs/2402.16567)

    该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。

    

    图数据库（Graph DB）被广泛应用于金融、社交网络和医药等各个领域。然而，将自然语言（NL）转换为图查询语言（GQL），通常称为NL2GQL，由于其固有复杂性和专业化特性而变得具有挑战性。一些方法试图利用大型语言模型（LLMs）来解决类似的任务，如文本转SQL。然而，在特定领域的NL2GQL任务中，缺乏特定领域的NL-GQL数据对使得难以建立LLMs和图数据库之间的对齐。为了解决这一挑战，我们提出了一个明确定义的流水线。具体地，我们利用ChatGPT基于给定的图数据库自我生成NL-GQL数据对。然后，我们使用创建的数据来对LLMs进行微调，从而实现LLMs与图数据库之间的对齐。此外，在推断过程中，我们提出了一种提取相关信息的方法。

    arXiv:2402.16567v1 Announce Type: new  Abstract: Graph Databases (Graph DB) are widely applied in various fields, including finance, social networks, and medicine. However, translating Natural Language (NL) into the Graph Query Language (GQL), commonly known as NL2GQL, proves to be challenging due to its inherent complexity and specialized nature. Some approaches have sought to utilize Large Language Models (LLMs) to address analogous tasks like text2SQL. Nevertheless, when it comes to NL2GQL taskson a particular domain, the absence of domain-specific NL-GQL data pairs makes it difficult to establish alignment between LLMs and the graph DB. To address this challenge, we propose a well-defined pipeline. Specifically, we utilize ChatGPT to create NL-GQL data pairs based on the given graph DB with self-instruct. Then, we use the created data to fine-tune LLMs, thereby achieving alignment between LLMs and the graph DB. Additionally, during inference, we propose a method that extracts relev
    
[^3]: 通过少样本标注者调适实现高效主观任务标注与建模

    Cost-Efficient Subjective Task Annotation and Modeling through Few-Shot Annotator Adaptation

    [https://arxiv.org/abs/2402.14101](https://arxiv.org/abs/2402.14101)

    引入了一个新颖的框架，通过少样本标注者调适来实现在主观任务中进行高效标注与建模，最大程度减少标注预算，同时最大化每个标注者的预测性能。

    

    在主观性自然语言处理任务中，由于不存在单一的标准答案，包括不同标注者变得至关重要，因为他们独特的视角显著影响标注。在现实场景中，标注预算通常成为决定数据中包含的视角数量（即标注者）及后续建模的主要因素。我们引入了一个新颖的框架，用于在主观任务中进行标注收集和建模，旨在最大程度地减少标注预算，同时最大化每个标注者的预测性能。我们的框架采用两阶段设计：首先，我们依赖一小组标注者来构建多任务模型，然后，我们通过为每个标注者策略性地标注少量样本，来为新视角增强模型。为了在规模上测试我们的框架，我们介绍并发布了一个独特数据集《道德基础主观语料库》，包含2000个Reddit帖子，由24名标注者进行标注。

    arXiv:2402.14101v1 Announce Type: new  Abstract: In subjective NLP tasks, where a single ground truth does not exist, the inclusion of diverse annotators becomes crucial as their unique perspectives significantly influence the annotations. In realistic scenarios, the annotation budget often becomes the main determinant of the number of perspectives (i.e., annotators) included in the data and subsequent modeling. We introduce a novel framework for annotation collection and modeling in subjective tasks that aims to minimize the annotation budget while maximizing the predictive performance for each annotator. Our framework has a two-stage design: first, we rely on a small set of annotators to build a multitask model, and second, we augment the model for a new perspective by strategically annotating a few samples per annotator. To test our framework at scale, we introduce and release a unique dataset, Moral Foundations Subjective Corpus, of 2000 Reddit posts annotated by 24 annotators for 
    
[^4]: 探索大型语言模型中的群组和对称性原理

    Exploring Group and Symmetry Principles in Large Language Models

    [https://arxiv.org/abs/2402.06120](https://arxiv.org/abs/2402.06120)

    本文提出了一个基于群组和对称性原理的框架，以评估大型语言模型的推理能力。通过研究四个群组属性，发现这些模型在保持群组属性方面表现不佳。

    

    大型语言模型（LLM）在广泛的应用中展示了令人瞩目的性能，然而评估它们的推理能力仍然是一个重大挑战。在本文中，我们引入了一个以群组和对称性原理为基础的框架，这些原理在物理学和数学等领域发挥着关键作用，并提供了另一种评估它们能力的方式。虽然提出的框架是通用的，为了展示使用这些属性的好处，我们关注算术推理，并研究这些模型在四个群组属性（封闭性、恒等性、逆性和结合性）上的性能。我们的发现表明，在本研究中研究的LLM在不同的测试方案中难以保持群组属性。在封闭性测试中，我们观察到对特定输出的偏见，并在特定的序列长度后从100％的性能迅速下降到0％。它们在恒等性测试中表现不佳，代表了相加得到原数的属性。

    Large Language Models (LLMs) have demonstrated impressive performance across a wide range of applications; however, assessing their reasoning capabilities remains a significant challenge. In this paper, we introduce a framework grounded in group and symmetry principles, which have played a crucial role in fields such as physics and mathematics, and offer another way to evaluate their capabilities. While the proposed framework is general, to showcase the benefits of employing these properties, we focus on arithmetic reasoning and investigate the performance of these models on four group properties: closure, identity, inverse, and associativity. Our findings reveal that LLMs studied in this work struggle to preserve group properties across different test regimes. In the closure test, we observe biases towards specific outputs and an abrupt degradation in their performance from 100% to 0% after a specific sequence length. They also perform poorly in the identity test, which represents add
    
[^5]: 语言引导的世界模型：一种基于模型的人工智能控制方法

    Language-Guided World Models: A Model-Based Approach to AI Control

    [https://arxiv.org/abs/2402.01695](https://arxiv.org/abs/2402.01695)

    语言引导的世界模型（LWMs）是一种基于模型的人工智能控制方法，它通过阅读语言描述来捕捉环境动态，提高了代理的沟通效率，并允许人类通过简洁的语言反馈同时改变他们在多个任务上的行为。

    

    将概率世界模型安装到人工智能代理中，为人类与这些代理沟通和控制打开了一个高效的渠道。除了更新代理策略，人类还可以修改他们的内部世界模型，以影响代理的决策。然而，当前现有的世界模型难以适应人类，因为它们缺乏自然的通信界面。为了解决这个问题，我们开发了语言引导的世界模型（LWMs），它们可以通过阅读语言描述来捕捉环境动态。这些模型提高了代理的沟通效率，使人类能够通过简洁的语言反馈同时改变他们在多个任务上的行为。它们还使代理能够从最初用于指导人类的文本中进行自我学习。为了促进LWMs的发展，我们设计了一个基于MESSENGER游戏（Hanjie等人，2021）的挑战基准，需要对新场景进行组合泛化。

    Installing probabilistic world models into artificial agents opens an efficient channel for humans to communicate with and control these agents. In addition to updating agent policies, humans can modify their internal world models in order to influence their decisions. The challenge, however, is that currently existing world models are difficult for humans to adapt because they lack a natural communication interface. Aimed at addressing this shortcoming, we develop Language-Guided World Models (LWMs), which can capture environment dynamics by reading language descriptions. These models enhance agent communication efficiency, allowing humans to simultaneously alter their behavior on multiple tasks with concise language feedback. They also enable agents to self-learn from texts originally written to instruct humans. To facilitate the development of LWMs, we design a challenging benchmark based on the game of MESSENGER (Hanjie et al., 2021), requiring compositional generalization to new l
    
[^6]: Kun: 使用指令反向翻译的中国自对齐问题的答案优化方法

    Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation. (arXiv:2401.06477v1 [cs.CL])

    [http://arxiv.org/abs/2401.06477](http://arxiv.org/abs/2401.06477)

    Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。

    

    在本文中，我们介绍了一种名为Kun的新方法，用于在不依赖手动注释的情况下为大型语言模型（LLMs）创建高质量的指导调整数据集。Kun利用来自吾道、完卷和SkyPile等多个来源的未标记数据，采用基于指令反向翻译和答案优化的自我训练算法，生成了一个超过一百万个中文指导数据点的大规模数据集。该方法通过使用自我筛选过程来完善和选择最有效的指令-输出对，显著偏离传统方法。我们在多个基准测试上对6B参数的Yi模型进行了实验，结果表明Kun具有鲁棒性和可扩展性。我们方法的核心贡献在于算法的改进，增强了数据的保留和清晰度，并且创新的数据生成方法极大地减少了对昂贵和耗时的手动注释的依赖。这种方法ological方法提出了一种解决中文自对齐问题的方法，并提高了数据的准确性和质量。

    In this paper, we introduce Kun, a novel approach for creating high-quality instruction-tuning datasets for large language models (LLMs) without relying on manual annotations. Adapting a self-training algorithm based on instruction back-translation and answer polishment, Kun leverages unlabelled data from diverse sources such as Wudao, Wanjuan, and SkyPile to generate a substantial dataset of over a million Chinese instructional data points. This approach significantly deviates from traditional methods by using a self-curation process to refine and select the most effective instruction-output pairs. Our experiments with the 6B-parameter Yi model across various benchmarks demonstrate Kun's robustness and scalability. Our method's core contributions lie in its algorithmic advancement, which enhances data retention and clarity, and its innovative data generation approach that substantially reduces the reliance on costly and time-consuming manual annotations. This methodology presents a sc
    
[^7]: 激发大型语言模型中的提示工程潜力：一项综述

    Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review. (arXiv:2310.14735v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14735](http://arxiv.org/abs/2310.14735)

    这篇论文解释了提示工程在释放大型语言模型能力方面的关键作用，探讨了不同的提示方法以及外部插件如何协助减少机器幻想，并指出了未来研究方向的重要性。

    

    本文深入探讨了提示工程在释放大型语言模型（LLM）能力方面的关键作用。提示工程是为LLM构建输入文本的过程，是优化LLM有效性的重要技术。本综述阐明了提示工程的基本原理，如角色提示、一次性提示和少量提示，以及更高级的方法，如思维链和思维树提示。本文还阐述了外部插件如何协助此任务，并通过检索外部知识来减少机器幻想。随后，我们勾勒了提示工程研究的前景方向，强调了对结构和代理在人工智能生成内容（AIGC）工具中的作用的深入理解的必要性。我们讨论了如何从不同角度和使用不同的方法评估提示方法的有效性。最后，我们提出了展望未来的研究方向。

    This paper delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). Prompt engineering is the process of structuring input text for LLMs and is a technique integral to optimizing the efficacy of LLMs. This survey elucidates foundational principles of prompt engineering, such as role-prompting, one-shot, and few-shot prompting, as well as more advanced methodologies such as the chain-of-thought and tree-of-thoughts prompting. The paper sheds light on how external assistance in the form of plugins can assist in this task, and reduce machine hallucination by retrieving external knowledge. We subsequently delineate prospective directions in prompt engineering research, emphasizing the need for a deeper understanding of structures and the role of agents in Artificial Intelligence-Generated Content (AIGC) tools. We discuss how to assess the efficacy of prompt methods from different perspectives and using different methods. Finally, we
    
[^8]: PESTS: 波斯语-英语跨语言语料库用于语义文本相似度

    PESTS: Persian_English Cross Lingual Corpus for Semantic Textual Similarity. (arXiv:2305.07893v1 [cs.CL])

    [http://arxiv.org/abs/2305.07893](http://arxiv.org/abs/2305.07893)

    本研究提出了跨语言的语义相似性模型PESTS，并通过波斯语-英语的跨语言语料库来验证模型的准确性。

    

    近来，语义文本相似度成为自然语言处理中备受关注的组件。在计算语言学和自然语言处理中，评估单词、短语、段落和文本之间的语义相似性很重要。同时，语义相似性度量要求在源和目标语言中提供具有一定语义相似性的句子对。许多跨语言的语义相似度模型使用机器翻译来弥补跨语言语料库不可用的不足，但机器翻译的误差会降低模型的准确性。然而，在使用语义相似度特征实现机器翻译时，用相同的机器翻译模型可以提高结果的准确性。

    One of the components of natural language processing that has received a lot of investigation recently is semantic textual similarity. In computational linguistics and natural language processing, assessing the semantic similarity of words, phrases, paragraphs, and texts is crucial. Calculating the degree of semantic resemblance between two textual pieces, paragraphs, or phrases provided in both monolingual and cross-lingual versions is known as semantic similarity. Cross lingual semantic similarity requires corpora in which there are sentence pairs in both the source and target languages with a degree of semantic similarity between them. Many existing cross lingual semantic similarity models use a machine translation due to the unavailability of cross lingual semantic similarity dataset, which the propagation of the machine translation error reduces the accuracy of the model. On the other hand, when we want to use semantic similarity features for machine translation the same machine t
    

