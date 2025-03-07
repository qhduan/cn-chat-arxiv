# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data](https://arxiv.org/abs/2402.14973) | 提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。 |
| [^2] | [DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392) | DoraemonGPT是一个由LLMs驱动的系统，旨在处理动态视频任务，通过将视频转换为符号记忆来进行空间-时间查询和推理，并取得简洁的中间结果。 |
| [^3] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |

# 详细

[^1]: GenCeption：使用未标记的单模态数据评估多模态LLM

    GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data

    [https://arxiv.org/abs/2402.14973](https://arxiv.org/abs/2402.14973)

    提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。

    

    多模态大型语言模型（MLLMs）通常使用昂贵的带标注的多模态基准进行评估。然而，这些基准通常难以跟上MLLM评估的快速发展要求。我们提出了GenCeption，这是一个新颖的无需注释的MLLM评估框架，仅需要单模态数据来评估跨模态语义一致性，并反映出模型产生幻觉的倾向。类似于流行的DrawCeption游戏，GenCeption从一个非文本样本开始，并经历一系列迭代的描述和生成步骤。迭代之间的语义漂移使用GC@T指标进行量化。我们的实证发现验证了GenCeption的有效性，并显示出与流行的MLLM基准结果的强相关性。GenCeption可以通过利用普遍存在且以前未见的单模态数据来扩展，以减轻训练数据的污染。

    arXiv:2402.14973v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) are commonly evaluated using costly annotated multimodal benchmarks. However, these benchmarks often struggle to keep pace with the rapidly advancing requirements of MLLM evaluation. We propose GenCeption, a novel and annotation-free MLLM evaluation framework that merely requires unimodal data to assess inter-modality semantic coherence and inversely reflects the models' inclination to hallucinate. Analogous to the popular DrawCeption game, GenCeption initiates with a non-textual sample and undergoes a series of iterative description and generation steps. Semantic drift across iterations is quantified using the GC@T metric. Our empirical findings validate GenCeption's efficacy, showing strong correlations with popular MLLM benchmarking results. GenCeption may be extended to mitigate training data contamination by utilizing ubiquitous, previously unseen unimodal data.
    
[^2]: DoraemonGPT：朝向理解具有大语言模型的动态场景迈进

    DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models

    [https://arxiv.org/abs/2401.08392](https://arxiv.org/abs/2401.08392)

    DoraemonGPT是一个由LLMs驱动的系统，旨在处理动态视频任务，通过将视频转换为符号记忆来进行空间-时间查询和推理，并取得简洁的中间结果。

    

    最近由LLM驱动的视觉代理主要集中于解决基于图像的任务，这限制了它们理解动态场景的能力，使其远离像引导学生进行实验室实验和识别错误这样的真实应用。考虑到视频模态更好地反映了真实世界场景的不断变化性质，我们设计了DoraemonGPT，这是一个由LLM驱动的综合概念简洁系统，用于处理动态视频任务。给定一个带有问题/任务的视频，DoraemonGPT首先将输入视频转换为存储与任务相关属性的符号存储器。这种结构化表示允许通过精心设计的子任务工具进行空间-时间查询和推理，从而产生简洁的中间结果。鉴于LLM在涉及专业领域（例如分析实验中潜在的科学原理）时具有有限的内部知识，我们引入了

    arXiv:2401.08392v2 Announce Type: replace-cross  Abstract: Recent LLM-driven visual agents mainly focus on solving image-based tasks, which limits their ability to understand dynamic scenes, making it far from real-life applications like guiding students in laboratory experiments and identifying their mistakes. Considering the video modality better reflects the ever-changing nature of real-world scenarios, we devise DoraemonGPT, a comprehensive and conceptually elegant system driven by LLMs to handle dynamic video tasks. Given a video with a question/task, DoraemonGPT begins by converting the input video into a symbolic memory that stores task-related attributes. This structured representation allows for spatial-temporal querying and reasoning by well-designed sub-task tools, resulting in concise intermediate results. Recognizing that LLMs have limited internal knowledge when it comes to specialized domains (e.g., analyzing the scientific principles underlying experiments), we incorpor
    
[^3]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    

