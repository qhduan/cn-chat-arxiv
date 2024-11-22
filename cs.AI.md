# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AI-generated faces free from racial and gender stereotypes](https://rss.arxiv.org/abs/2402.01002) | 这项研究发现并解决了AI生成的面孔中存在的种族和性别刻板印象问题，提出了分类器用于预测面部属性的方法，并提出了有效的去偏见解决方案。 |
| [^2] | [PromptCodec: High-Fidelity Neural Speech Codec using Disentangled Representation Learning based Adaptive Feature-aware Prompt Encoders](https://arxiv.org/abs/2404.02702) | 本文提出了PromptCodec，一种使用离散表示学习的特征感知提示编码器的高保真神经语音编解码器，通过引入额外特征表示、自适应特征加权融合和效率优化来解决高压缩率下的高保真音频重建问题。 |
| [^3] | [Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines](https://arxiv.org/abs/2403.11585) | Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。 |
| [^4] | [Graph Neural Networks and Arithmetic Circuits](https://arxiv.org/abs/2402.17805) | 研究者在本文中建立了图神经网络与算术电路之间的表达能力对应关系，结果表明不同激活函数的GNN在表达能力上等价于实数上的算术电路。 |
| [^5] | [Probing Multimodal Large Language Models for Global and Local Semantic Representation](https://arxiv.org/abs/2402.17304) | 通过研究发现，多模态大型语言模型的中间层能够更好地编码全局语义信息，在视觉-语言任务中表现出更好的性能。顶层可能过多关注局部信息，导致理解全局信息的能力下降。 |
| [^6] | [Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction](https://arxiv.org/abs/2402.15368) | 本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。 |
| [^7] | [PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition](https://arxiv.org/abs/2402.04838) | 本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。 |
| [^8] | [HEAM : Hashed Embedding Acceleration using Processing-In-Memory](https://arxiv.org/abs/2402.04032) | HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。 |
| [^9] | [Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis.](http://arxiv.org/abs/2311.16141) | 本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。 |
| [^10] | [Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems.](http://arxiv.org/abs/2311.00207) | 本文提出了Magmaw，这是一种针对基于机器学习的无线通信系统进行模态不可知对抗攻击的黑盒攻击方法。它能够生成通用的对抗扰动，并引入了新的攻击目标。实验证实了其对现有防御方法的韧性。使用实时无线攻击平台进行了概念验证。 |
| [^11] | [Robust Decision-Focused Learning for Reward Transfer.](http://arxiv.org/abs/2304.03365) | 本文介绍了一种稳健决策重点（RDF）算法，利用非识别性的DF解，学习同时最大化期望回报和抵御奖励函数变化的模型，可以显著提高DF对奖励函数变化的稳健性，而不会降低智能体的总回报。 |
| [^12] | [Risk-Sensitive Reinforcement Learning with Exponential Criteria.](http://arxiv.org/abs/2212.09010) | 本文介绍了一种风险敏感的强化学习算法，使用指数判据来提高其系统抗干扰性和实用性。作者进行了在模拟和实际机器人上的实验验证，表明该算法能够有效地提高样本效率和执行效果。 |

# 详细

[^1]: AI生成的面孔摆脱了种族和性别刻板印象

    AI-generated faces free from racial and gender stereotypes

    [https://rss.arxiv.org/abs/2402.01002](https://rss.arxiv.org/abs/2402.01002)

    这项研究发现并解决了AI生成的面孔中存在的种族和性别刻板印象问题，提出了分类器用于预测面部属性的方法，并提出了有效的去偏见解决方案。

    

    诸如Stable Diffusion之类的文本到图像生成AI模型每天都被全球数百万人使用。然而，许多人对这些模型如何放大种族和性别刻板印象提出了关切。为了研究这一现象，我们开发了一个分类器来预测任意给定面部图像的种族、性别和年龄组，并展示其达到了最先进的性能。利用这个分类器，我们对Stable Diffusion在六种种族、两种性别、五个年龄组、32个职业和八个属性上的偏见进行了量化。然后，我们提出了超越最先进替代方案的新型去偏见解决方案。此外，我们还检查了Stable Diffusion在描绘同一种族的个体时相似程度。分析结果显示出高度的刻板印象，例如，将大多数中东男性描绘为皮肤黝黑、留着胡子、戴着传统头饰。我们提出了另一种增加面部多样性的新型解决方案来解决这些限制。

    Text-to-image generative AI models such as Stable Diffusion are used daily by millions worldwide. However, many have raised concerns regarding how these models amplify racial and gender stereotypes. To study this phenomenon, we develop a classifier to predict the race, gender, and age group of any given face image, and show that it achieves state-of-the-art performance. Using this classifier, we quantify biases in Stable Diffusion across six races, two genders, five age groups, 32 professions, and eight attributes. We then propose novel debiasing solutions that outperform state-of-the-art alternatives. Additionally, we examine the degree to which Stable Diffusion depicts individuals of the same race as being similar to one another. This analysis reveals a high degree of stereotyping, e.g., depicting most middle eastern males as being dark-skinned, bearded, and wearing a traditional headdress. We address these limitations by proposing yet another novel solution that increases facial div
    
[^2]: PromptCodec: 使用基于自适应特征感知的离散表示学习的高保真神经语音编解码器

    PromptCodec: High-Fidelity Neural Speech Codec using Disentangled Representation Learning based Adaptive Feature-aware Prompt Encoders

    [https://arxiv.org/abs/2404.02702](https://arxiv.org/abs/2404.02702)

    本文提出了PromptCodec，一种使用离散表示学习的特征感知提示编码器的高保真神经语音编解码器，通过引入额外特征表示、自适应特征加权融合和效率优化来解决高压缩率下的高保真音频重建问题。

    

    神经语音编解码器近来在生成语音建模领域引起广泛关注，例如语音转换、文本转语音合成等。然而，在高压缩率下确保语音编解码器的高保真音频重建仍然是一个未解决且具有挑战性的问题。本文提出了PromptCodec，一种使用基于离散表示学习的特征感知提示编码器的新型端到端神经语音编解码器模型。通过引入来自提示编码器的额外特征表示，PromptCodec可以分配需要处理的语音信息并增强其能力。此外，引入了一种简单而有效的自适应特征加权融合方法，以整合不同编码器的特征。同时，我们提出了一种基于余弦距离的新颖离散表示学习策略，以优化PromptCodec的编码器以确保其效率，进一步改进。

    arXiv:2404.02702v1 Announce Type: cross  Abstract: Neural speech codec has recently gained widespread attention in generative speech modeling domains, like voice conversion, text-to-speech synthesis, etc. However, ensuring high-fidelity audio reconstruction of speech codecs under high compression rates remains an open and challenging issue. In this paper, we propose PromptCodec, a novel end-to-end neural speech codec model using disentangled representation learning based feature-aware prompt encoders. By incorporating additional feature representations from prompt encoders, PromptCodec can distribute the speech information requiring processing and enhance its capabilities. Moreover, a simple yet effective adaptive feature weighted fusion approach is introduced to integrate features of different encoders. Meanwhile, we propose a novel disentangled representation learning strategy based on cosine distance to optimize PromptCodec's encoders to ensure their efficiency, thereby further impr
    
[^3]: Linguacodus：一种在机器学习流水线中进行变革性代码生成的协同框架

    Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines

    [https://arxiv.org/abs/2403.11585](https://arxiv.org/abs/2403.11585)

    Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。

    

    在不断发展的机器学习领域中，将自然语言描述无缝转化为可执行代码仍然是一个巨大的挑战。本文介绍了Linguacodus，这是一个创新性框架，旨在通过部署一个动态流水线，通过高级数据塑形指令，将自然语言任务描述迭代地转换为代码来应对这一挑战。Linguacodus的核心是一个经过精细调整的大型语言模型（LLM），能够评估各种问题的多样解决方案，并为特定任务选择最合适的解决方案。本文详细介绍了精细调整过程，并阐明了如何将自然语言描述转化为功能性代码。Linguacodus代表了自动化代码生成的重大飞跃，有效地弥合了任务描述和可执行代码之间的差距。它对推进跨不同领域的机器学习应用具有巨大潜力。

    arXiv:2403.11585v1 Announce Type: cross  Abstract: In the ever-evolving landscape of machine learning, seamless translation of natural language descriptions into executable code remains a formidable challenge. This paper introduces Linguacodus, an innovative framework designed to tackle this challenge by deploying a dynamic pipeline that iteratively transforms natural language task descriptions into code through high-level data-shaping instructions. The core of Linguacodus is a fine-tuned large language model (LLM), empowered to evaluate diverse solutions for various problems and select the most fitting one for a given task. This paper details the fine-tuning process, and sheds light on how natural language descriptions can be translated into functional code. Linguacodus represents a substantial leap towards automated code generation, effectively bridging the gap between task descriptions and executable code. It holds great promise for advancing machine learning applications across div
    
[^4]: 图神经网络与算术电路

    Graph Neural Networks and Arithmetic Circuits

    [https://arxiv.org/abs/2402.17805](https://arxiv.org/abs/2402.17805)

    研究者在本文中建立了图神经网络与算术电路之间的表达能力对应关系，结果表明不同激活函数的GNN在表达能力上等价于实数上的算术电路。

    

    我们表征了遵循图神经网络（GNN）架构的神经网络的计算能力，不限于聚合-组合GNN或其他特定类型。我们建立了使用不同激活函数的GNN的表达能力与实数上的算术电路之间的准确对应关系。在我们的结果中，网络的激活函数成为电路中的门类型。我们的结果对于常数深度电路和网络家族均成立，无论是在一致还是非一致的情况下，对于所有常见激活函数。

    arXiv:2402.17805v1 Announce Type: cross  Abstract: We characterize the computational power of neural networks that follow the graph neural network (GNN) architecture, not restricted to aggregate-combine GNNs or other particular types. We establish an exact correspondence between the expressivity of GNNs using diverse activation functions and arithmetic circuits over real numbers. In our results the activation function of the network becomes a gate type in the circuit. Our result holds for families of constant depth circuits and networks, both uniformly and non-uniformly, for all common activation functions.
    
[^5]: 探究多模态大型语言模型对全局和局部语义表示的影响

    Probing Multimodal Large Language Models for Global and Local Semantic Representation

    [https://arxiv.org/abs/2402.17304](https://arxiv.org/abs/2402.17304)

    通过研究发现，多模态大型语言模型的中间层能够更好地编码全局语义信息，在视觉-语言任务中表现出更好的性能。顶层可能过多关注局部信息，导致理解全局信息的能力下降。

    

    大型语言模型的成功启发了研究人员将其优秀的表示能力转移到其他模态。最近的一些研究利用图像描述对齐数据集训练多模态大型语言模型（MLLMs），在图像到文本任务中取得了最新的性能表现。然而，很少有研究探讨MLLMs是否真正理解完整的图像信息，即全局信息，或者它们只能捕捉一些局部对象信息。本研究发现模型的中间层可以编码更多全局语义信息，其表示向量在视觉-语言蕴涵任务上表现更好，而不是顶层。我们通过目标检测任务进一步探究模型的局部语义表示。我们得出的结论是顶层可能过多专注于局部信息，导致减弱了对全局信息的理解能力。

    arXiv:2402.17304v1 Announce Type: cross  Abstract: The success of large language models has inspired researchers to transfer their exceptional representing ability to other modalities. Several recent works leverage image-caption alignment datasets to train multimodal large language models (MLLMs), which achieve state-of-the-art performance on image-to-text tasks. However, there are very few studies exploring whether MLLMs truly understand the complete image information, i.e., global information, or if they can only capture some local object information. In this study, we find that the intermediate layers of models can encode more global semantic information, whose representation vectors perform better on visual-language entailment tasks, rather than the topmost layers. We further probe models for local semantic representation through object detection tasks. And we draw a conclusion that the topmost layers may excessively focus on local information, leading to a diminished ability to en
    
[^6]: 使用符合预测的技术实现语言指导多机器人系统的安全任务规划

    Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction

    [https://arxiv.org/abs/2402.15368](https://arxiv.org/abs/2402.15368)

    本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。

    

    本文解决了语言指导机器人团队的任务规划问题。任务用自然语言（NL）表示，要求机器人在各种位置和语义对象上应用它们的能力（例如移动、操作和感知）。最近几篇论文通过利用预训练的大型语言模型（LLMs）设计有效的多机器人计划来解决类似的规划问题。然而，这些方法缺乏任务性能和安全性保证。为了解决这一挑战，我们引入了一种新的基于分布式LLM的规划器，能够实现高任务成功率。这是通过利用符合预测（CP）来实现的，CP是一种基于分布的不确定性量化工具，可以在黑盒模型中对其固有不确定性进行推理。CP允许所提出的多机器人规划器以分布方式推理其固有不确定性，使得机器人在充分信任时能够做出个别决策。

    arXiv:2402.15368v1 Announce Type: cross  Abstract: This paper addresses task planning problems for language-instructed robot teams. Tasks are expressed in natural language (NL), requiring the robots to apply their capabilities (e.g., mobility, manipulation, and sensing) at various locations and semantic objects. Several recent works have addressed similar planning problems by leveraging pre-trained Large Language Models (LLMs) to design effective multi-robot plans. However, these approaches lack mission performance and safety guarantees. To address this challenge, we introduce a new decentralized LLM-based planner that is capable of achieving high mission success rates. This is accomplished by leveraging conformal prediction (CP), a distribution-free uncertainty quantification tool in black-box models. CP allows the proposed multi-robot planner to reason about its inherent uncertainty in a decentralized fashion, enabling robots to make individual decisions when they are sufficiently ce
    
[^7]: PaDeLLM-NER：大型语言模型中的并行解码用于命名实体识别

    PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition

    [https://arxiv.org/abs/2402.04838](https://arxiv.org/abs/2402.04838)

    本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。

    

    本研究旨在使用大型语言模型（LLMs）减少命名实体识别（NER）的生成延迟。LLMs的高延迟的主要原因是顺序解码过程，该过程自回归地生成NER的所有标签和提及，显著增加了序列长度。为此，我们引入了PaDeLLM-NER（Parallel Decoding in LLM for NE），这是一种无需额外模块或架构修改即可无缝集成到现有生成模型框架中的方法。PaDeLLM-NER允许同时解码所有提及，从而减少生成延迟。实验结果显示，PaDeLLM-NER的推理速度显著提高，对英语和中文来说比自回归方法快1.76到10.22倍。与各种数据集上的最先进性能相媲美，同时维持了预测质量。

    In this study, we aim to reduce generation latency for Named Entity Recognition (NER) with Large Language Models (LLMs). The main cause of high latency in LLMs is the sequential decoding process, which autoregressively generates all labels and mentions for NER, significantly increase the sequence length. To this end, we introduce Parallel Decoding in LLM for NE} (PaDeLLM-NER), a approach that integrates seamlessly into existing generative model frameworks without necessitating additional modules or architectural modifications. PaDeLLM-NER allows for the simultaneous decoding of all mentions, thereby reducing generation latency. Experiments reveal that PaDeLLM-NER significantly increases inference speed that is 1.76 to 10.22 times faster than the autoregressive approach for both English and Chinese. Simultaneously it maintains the quality of predictions as evidenced by the performance that is on par with the state-of-the-art across various datasets.
    
[^8]: HEAM: 使用处理-内存进行散列嵌入加速的方法

    HEAM : Hashed Embedding Acceleration using Processing-In-Memory

    [https://arxiv.org/abs/2402.04032](https://arxiv.org/abs/2402.04032)

    HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。

    

    在当今的数据中心中，个性化推荐系统面临着诸多挑战，特别是在执行嵌入操作时需要大容量的内存和高带宽。之前的方法依赖于DIMM-based近内存处理技术或引入3D堆叠DRAM来解决内存限制和扩展内存带宽的问题。然而，这些解决方案在处理日益扩大的个性化推荐系统大小时存在不足之处。推荐模型已经增长到超过数十TB的大小，导致在传统单节点推断服务器上高效运行变得困难。尽管已经提出了各种算法方法来减小嵌入表容量，但通常会导致内存访问增加或内存资源利用低效的问题。本文引入了HEAM，一种异构内存架构，将3D堆叠DRAM与DIMM集成在一起，以加速组合嵌入的推荐系统。

    In today's data centers, personalized recommendation systems face challenges such as the need for large memory capacity and high bandwidth, especially when performing embedding operations. Previous approaches have relied on DIMM-based near-memory processing techniques or introduced 3D-stacked DRAM to address memory-bound issues and expand memory bandwidth. However, these solutions fall short when dealing with the expanding size of personalized recommendation systems. Recommendation models have grown to sizes exceeding tens of terabytes, making them challenging to run efficiently on traditional single-node inference servers. Although various algorithmic methods have been proposed to reduce embedding table capacity, they often result in increased memory access or inefficient utilization of memory resources. This paper introduces HEAM, a heterogeneous memory architecture that integrates 3D-stacked DRAM with DIMM to accelerate recommendation systems in which compositional embedding is util
    
[^9]: SNNs中基于关键性的高效修剪方法，受到关键性大脑假设的启发

    Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis. (arXiv:2311.16141v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2311.16141](http://arxiv.org/abs/2311.16141)

    本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。

    

    由于其节能和无乘法特性，SNNs已经引起了相当大的关注。深度SNNs规模的不断增长给模型部署带来了挑战。网络修剪通过压缩网络规模来减少模型部署的硬件资源需求。然而，现有的SNN修剪方法由于修剪迭代增加了SNNs的训练难度，导致修剪成本高昂且性能损失严重。本文受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的用于SNN修剪的再生机制，以增强特征提取并加速修剪过程。首先，我们提出了一种SNN中用于关键性的低成本度量方式。然后，在修剪后对所修剪结构进行重新排序，并再生那些具有较高关键性的结构，以获取关键网络。我们的方法表现优于当前的最先进方法。

    Spiking Neural Networks (SNNs) have gained considerable attention due to the energy-efficient and multiplication-free characteristics. The continuous growth in scale of deep SNNs poses challenges for model deployment. Network pruning reduces hardware resource requirements of model deployment by compressing the network scale. However, existing SNN pruning methods cause high pruning costs and performance loss because the pruning iterations amplify the training difficulty of SNNs. In this paper, inspired by the critical brain hypothesis in neuroscience, we propose a regeneration mechanism based on the neuron criticality for SNN pruning to enhance feature extraction and accelerate the pruning process. Firstly, we propose a low-cost metric for the criticality in SNNs. Then, we re-rank the pruned structures after pruning and regenerate those with higher criticality to obtain the critical network. Our method achieves higher performance than the current state-of-the-art (SOTA) method with up t
    
[^10]: Magmaw: 对基于机器学习的无线通信系统的模态不可知对抗攻击

    Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems. (arXiv:2311.00207v1 [cs.CR])

    [http://arxiv.org/abs/2311.00207](http://arxiv.org/abs/2311.00207)

    本文提出了Magmaw，这是一种针对基于机器学习的无线通信系统进行模态不可知对抗攻击的黑盒攻击方法。它能够生成通用的对抗扰动，并引入了新的攻击目标。实验证实了其对现有防御方法的韧性。使用实时无线攻击平台进行了概念验证。

    

    机器学习在合并端到端无线通信系统的所有物理层模块以实现联合收发器优化方面发挥了重要作用。尽管已经有许多针对基于机器学习的无线系统的对抗攻击方法，但现有方法并未提供包括源数据的多模态、共同的物理层组件和无线领域约束在内的全面视角。本文提出了Magmaw，这是一种能够针对通过无线信道传输的任何多模态信号生成通用对抗扰动的黑盒攻击方法。我们进一步对基于机器学习的下游应用的对抗攻击引入了新的目标。实验证实了该攻击对现有广泛使用的对抗训练和扰动信号减法防御方法的韧性。为了概念证明，我们使用软件定义无线电系统构建了一个实时无线攻击平台。

    Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experi
    
[^11]: 奖励转移的稳健决策重点学习

    Robust Decision-Focused Learning for Reward Transfer. (arXiv:2304.03365v1 [cs.LG])

    [http://arxiv.org/abs/2304.03365](http://arxiv.org/abs/2304.03365)

    本文介绍了一种稳健决策重点（RDF）算法，利用非识别性的DF解，学习同时最大化期望回报和抵御奖励函数变化的模型，可以显著提高DF对奖励函数变化的稳健性，而不会降低智能体的总回报。

    

    最近，决策重点（Decision-focused，DF）的基于模型的强化学习被介绍为一种强有力的算法，它可以专注于学习最有利于获得高报酬的MDP动态。虽然这种方法通过专注于直接优化报酬来提高智能体的性能，但从MLE的角度来看，它学习的动力学不够准确，因此可能对奖励函数的变化很脆弱。在这项工作中，我们开发了稳健决策重点（RDF）算法，它利用DF解的非识别性，学习同时最大化期望回报和抵御奖励函数变化的模型。我们在各种玩具示例和医疗模拟器上展示了RDF显着增加了DF对奖励函数变化的稳健性，而不会降低智能体的总回报。

    Decision-focused (DF) model-based reinforcement learning has recently been introduced as a powerful algorithm which can focus on learning the MDP dynamics which are most relevant for obtaining high rewards. While this approach increases the performance of agents by focusing the learning towards optimizing for the reward directly, it does so by learning less accurate dynamics (from a MLE standpoint), and may thus be brittle to changes in the reward function. In this work, we develop the robust decision-focused (RDF) algorithm which leverages the non-identifiability of DF solutions to learn models which maximize expected returns while simultaneously learning models which are robust to changes in the reward function. We demonstrate on a variety of toy example and healthcare simulators that RDF significantly increases the robustness of DF to changes in the reward function, without decreasing the overall return the agent obtains.
    
[^12]: 风险敏感的强化学习算法：指数标准的应用

    Risk-Sensitive Reinforcement Learning with Exponential Criteria. (arXiv:2212.09010v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.09010](http://arxiv.org/abs/2212.09010)

    本文介绍了一种风险敏感的强化学习算法，使用指数判据来提高其系统抗干扰性和实用性。作者进行了在模拟和实际机器人上的实验验证，表明该算法能够有效地提高样本效率和执行效果。

    

    尽管风险中性的强化学习已经在很多应用中得到了实验成功，但是这种方法容易受到噪声和系统参数扰动的影响而不够稳健。因此,对风险敏感的强化学习算法进行了研究，以提高其系统抗干扰性，样本效率和实用性。本文介绍了一种新型的无模型风险敏感学习算法，将广泛使用的策略梯度算法进行变体，其实现过程类似。具体来说，本文研究了指数标准对强化学习代理的策略风险敏感性的影响，并开发了蒙特卡罗策略梯度算法和在线(时间差分)演员-评论家算法的变体。分析结果表明，指数标准的使用能够推广常用的特定正则化方法。作者在摆动杆和摆摆杆任务上进行了测试，验证了所提出的算法的实现性能和稳健性。

    While risk-neutral reinforcement learning has shown experimental success in a number of applications, it is well-known to be non-robust with respect to noise and perturbations in the parameters of the system. For this reason, risk-sensitive reinforcement learning algorithms have been studied to introduce robustness and sample efficiency, and lead to better real-life performance. In this work, we introduce new model-free risk-sensitive reinforcement learning algorithms as variations of widely-used Policy Gradient algorithms with similar implementation properties. In particular, we study the effect of exponential criteria on the risk-sensitivity of the policy of a reinforcement learning agent, and develop variants of the Monte Carlo Policy Gradient algorithm and the online (temporal-difference) Actor-Critic algorithm. Analytical results showcase that the use of exponential criteria generalize commonly used ad-hoc regularization approaches. The implementation, performance, and robustness 
    

