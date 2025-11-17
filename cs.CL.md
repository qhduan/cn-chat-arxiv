# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Vanilla Transformers are Transfer Capability Teachers](https://arxiv.org/abs/2403.01994) | 混合专家（MoE）变压器在模型预训练性能和传输能力方面表现不如香草变压器，为此提出了迁移能力蒸馏的概念，指出香草模型是迁移能力的有效教师，指导MoE模型实现预训练性能和传输能力的结合。 |
| [^3] | [Survey in Characterization of Semantic Change](https://arxiv.org/abs/2402.19088) | 语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。 |
| [^4] | [Metric-Learning Encoding Models Identify Processing Profiles of Linguistic Features in BERT's Representations](https://arxiv.org/abs/2402.11608) | 应用指标学习编码模型（MLEMs）于BERT表示，发现语言特征在不同层中有序分离，神经表示层级组织，中间层解耦，优于其他解码方法。 |
| [^5] | [Identifying and Analyzing Task-Encoding Tokens in Large Language Models](https://arxiv.org/abs/2401.11323) | 本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 香草变压器是迁移能力教师

    Vanilla Transformers are Transfer Capability Teachers

    [https://arxiv.org/abs/2403.01994](https://arxiv.org/abs/2403.01994)

    混合专家（MoE）变压器在模型预训练性能和传输能力方面表现不如香草变压器，为此提出了迁移能力蒸馏的概念，指出香草模型是迁移能力的有效教师，指导MoE模型实现预训练性能和传输能力的结合。

    

    最近，由于在模型容量和计算效率方面的优势，混合专家（MoE）变压器引起了越来越多的关注。然而，研究表明，在许多下游任务中，MoE变压器的表现不及香草变压器，这显著降低了MoE模型的实用价值。为了解释这个问题，我们提出模型的预训练性能和迁移能力是影响其下游任务性能的联合决定因素。与香草模型相比，MoE模型的迁移能力较差，导致它们在下游任务中表现不佳。为了解决这个问题，我们引入了迁移能力蒸馏的概念，认为虽然香草模型性能较弱，但它们是迁移能力的有效教师。由香草模型指导的MoE模型可以实现强大的预训练性能和迁移能力，最终

    arXiv:2403.01994v1 Announce Type: new  Abstract: Recently, Mixture of Experts (MoE) Transformers have garnered increasing attention due to their advantages in model capacity and computational efficiency. However, studies have indicated that MoE Transformers underperform vanilla Transformers in many downstream tasks, significantly diminishing the practical value of MoE models. To explain this issue, we propose that the pre-training performance and transfer capability of a model are joint determinants of its downstream task performance. MoE models, in comparison to vanilla models, have poorer transfer capability, leading to their subpar performance in downstream tasks. To address this issue, we introduce the concept of transfer capability distillation, positing that although vanilla models have weaker performance, they are effective teachers of transfer capability. The MoE models guided by vanilla models can achieve both strong pre-training performance and transfer capability, ultimately
    
[^3]: 对语义变化特征的调查

    Survey in Characterization of Semantic Change

    [https://arxiv.org/abs/2402.19088](https://arxiv.org/abs/2402.19088)

    语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。

    

    活语言不断发展，以吸纳人类社会的文化变化。这种演变通过新词语（新单词）或单词的语义变化（赋予已有单词新的含义）来体现。理解单词的含义对解释来自不同文化（地方用语或俚语）、领域（例如技术术语）或时代的文本至关重要。在计算机科学中，这些单词与计算语言学算法相关，例如翻译、信息检索、问答等。语义变化可能会影响这些算法的结果质量。因此，了解和形式化表征这些变化是很重要的。研究这种影响是计算语言学界近期引起关注的问题。几种方法提出了检测语义变化的方法，具有较高的精度，但需要更多努力来对其进行表征。

    arXiv:2402.19088v1 Announce Type: cross  Abstract: Live languages continuously evolve to integrate the cultural change of human societies. This evolution manifests through neologisms (new words) or \textbf{semantic changes} of words (new meaning to existing words). Understanding the meaning of words is vital for interpreting texts coming from different cultures (regionalism or slang), domains (e.g., technical terms), or periods. In computer science, these words are relevant to computational linguistics algorithms such as translation, information retrieval, question answering, etc. Semantic changes can potentially impact the quality of the outcomes of these algorithms. Therefore, it is important to understand and characterize these changes formally. The study of this impact is a recent problem that has attracted the attention of the computational linguistics community. Several approaches propose methods to detect semantic changes with good precision, but more effort is needed to charact
    
[^4]: 指标学习编码模型识别BERT表示中的语言特征处理特征

    Metric-Learning Encoding Models Identify Processing Profiles of Linguistic Features in BERT's Representations

    [https://arxiv.org/abs/2402.11608](https://arxiv.org/abs/2402.11608)

    应用指标学习编码模型（MLEMs）于BERT表示，发现语言特征在不同层中有序分离，神经表示层级组织，中间层解耦，优于其他解码方法。

    

    我们介绍了指标学习编码模型（MLEMs）作为一种理解神经系统如何表示其处理对象的理论特征的新方法。作为概念验证，我们将MLEMs应用于从BERT中提取的神经表示，并跟踪各种语言特征（例如时态、主语人称、从句类型、从句嵌套等）。我们发现：（1）语言特征是有序的：它们在不同层中以不同程度将句子的表示分开；（2）神经表示是分层组织的：在某些层中，我们发现表示的群集嵌套在更大的群集内部，遵循逐渐重要的语言特征；（3）语言特征在中间层中是解耦的：不同的、选择性单位由不同的语言特征激活。在方法论上，MLEMs（4）优于多变量解码方法，更具抗类型-I错误的鲁棒性，（5）优于单变量

    arXiv:2402.11608v1 Announce Type: new  Abstract: We introduce Metric-Learning Encoding Models (MLEMs) as a new approach to understand how neural systems represent the theoretical features of the objects they process. As a proof-of-concept, we apply MLEMs to neural representations extracted from BERT, and track a wide variety of linguistic features (e.g., tense, subject person, clause type, clause embedding). We find that: (1) linguistic features are ordered: they separate representations of sentences to different degrees in different layers; (2) neural representations are organized hierarchically: in some layers, we find clusters of representations nested within larger clusters, following successively important linguistic features; (3) linguistic features are disentangled in middle layers: distinct, selective units are activated by distinct linguistic features. Methodologically, MLEMs are superior (4) to multivariate decoding methods, being more robust to type-I errors, and (5) to univ
    
[^5]: 辨识并分析大型语言模型中的任务编码标记

    Identifying and Analyzing Task-Encoding Tokens in Large Language Models

    [https://arxiv.org/abs/2401.11323](https://arxiv.org/abs/2401.11323)

    本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。

    

    在上下文学习（ICL）已成为自然语言处理中少样本学习的有效解决方案。然而，我们对ICL的工作机制的理解有限，特别是模型如何从ICL演示中学习执行任务。本文通过识别和分析任务编码标记，调查了这个问题。我们发现，模板标记和停用词标记最容易成为任务编码标记。此外，我们实验证明，词汇意思、重复和文本格式是这些标记的主要区别特征。我们的工作揭示了大型语言模型（LLMs）学习的方式。

    arXiv:2401.11323v2 Announce Type: replace  Abstract: In-context learning (ICL) has become an effective solution for few-shot learning in natural language processing. However, our understanding of ICL's working mechanisms is limited, specifically regarding how models learn to perform tasks from ICL demonstrations. For example, unexpectedly large changes in performance can arise from small changes in the prompt, leaving prompt design a largely empirical endeavour. In this paper, we investigate this problem by identifying and analyzing task-encoding tokens on whose representations the task performance depends. Using experiments that ablate the representations of different token types, we find that template and stopword tokens are the most prone to be task-encoding. In addition, we demonstrate experimentally that lexical meaning, repetition, and text formatting are the main distinguishing characteristics of these tokens. Our work sheds light on how large language models (LLMs) learn to per
    

