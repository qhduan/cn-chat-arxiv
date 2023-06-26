# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bring Your Own Data! Self-Supervised Evaluation for Large Language Models.](http://arxiv.org/abs/2306.13651) | 本研究提出了一种自我监督评估框架，通过分析输入文本上的变换对LLMs的灵敏度或不变性，直接监控LLMs在实际数据上的行为。 |
| [^2] | [GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models.](http://arxiv.org/abs/2306.13649) | 本文提出了广义知识蒸馏（GKD），通过从学生中采样输出序列来缓解分布不匹配，并在优化替代KL等离散度方面处理模型欠规范，达到了在摘要任务上最先进的性能。 |
| [^3] | [Margin Maximization in Attention Mechanism.](http://arxiv.org/abs/2306.13596) | 这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。 |
| [^4] | [System-Level Natural Language Feedback.](http://arxiv.org/abs/2306.13588) | 本文提出了一个通用框架，用于解锁系统级别使用自然语言反馈的方法。我们展示了通过任务度量设计和语言模型提示设计，如何使用反馈在人工交互流程中形式化系统级别的设计决策，以便产生更好的模型，并展示了使用系统级别反馈和实例级别反馈的有效性。 |
| [^5] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |
| [^6] | [Knowledge-Infused Self Attention Transformers.](http://arxiv.org/abs/2306.13501) | 本文介绍了一种将知识注入到变压器模型不同组件的系统方法，可以更彻底地分析知识注入对每个组件和组件之间相互作用的影响，并使变压器模型在各种自然语言处理任务中性能得到了显著提高。 |
| [^7] | [Incorporating Graph Information in Transformer-based AMR Parsing.](http://arxiv.org/abs/2306.13467) | 本论文介绍了一种新的模型和方法LeakDistill，它使用结构适配器将图形信息明确并入到学习的表示中，从而提高了AMR解析性能。实验表明，我们可以通过在训练时使用单词到节点对齐将图形结构信息嵌入编码器中，即使不使用其他数据，也可以通过自我知识蒸馏获得最先进的AMR解析性能。 |
| [^8] | [Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation.](http://arxiv.org/abs/2306.13460) | 本文通过半透过最大似然估计方法，鼓励模型生成更详细的长字幕。 |
| [^9] | [Long-range Language Modeling with Self-retrieval.](http://arxiv.org/abs/2306.13421) | 本论文提出了一种名为Retrieval-Pretrained Transformer的模型，可以从头开始联合训练语言模型和检索器来模拟长文本。模型可以计算文本块的查询表示，并将其用于检索前面的块，从而融合信息以预测下一个目标块。检索器使用一个语义目标进行训练，目标是检索那些增加下一个块概率的块。 |
| [^10] | [Stress Testing BERT Anaphora Resolution Models for Reaction Extraction in Chemical Patents.](http://arxiv.org/abs/2306.13379) | 该论文通过在化学专利中进行压力测试研究了指代消解模型在无噪声和有噪声环境中的性能差异，旨在提高其对噪声的鲁棒性。 |
| [^11] | [Abstractive Text Summarization for Resumes With Cutting Edge NLP Transformers and LSTM.](http://arxiv.org/abs/2306.13315) | 本研究评估了多种技术（包括LSTM、T5、Pegasus、BART和BART-Large模型）在不同数据集上对简历文本进行分类任务的表现，结果显示微调后的BART-Large模型效果最佳。 |
| [^12] | [Mutually Guided Few-shot Learning for Relational Triple Extraction.](http://arxiv.org/abs/2306.13310) | 提出了相互指导的少样本学习框架，以进行关系三元组提取，并引入了一个新的跨域少样本三元组提取任务，实现了在少样本情况下的有竞争力结果。 |
| [^13] | [Towards Effective and Compact Contextual Representation for Conformer Transducer Speech Recognition Systems.](http://arxiv.org/abs/2306.13307) | 本研究旨在为Conformer Transducer语音识别系统建立高效紧凑的上下文表示，通过特定的注意力汇聚层实现跨话语的信息集成，取得了显著的性能提升。 |
| [^14] | [DiversiGATE: A Comprehensive Framework for Reliable Large Language Models.](http://arxiv.org/abs/2306.13230) | DiversiGATE是一个统一框架，汇集了多种LLM验证方法，其中包括自一致性、数学提示和WebGPT，同时提出了一个符合该框架的新模型“SelfLearner”，该模型可以从自己的输出中学习并优化性能，在实验中表现良好，GSM8K基准测试上提高了7%的性能。 |
| [^15] | [Visual Adversarial Examples Jailbreak Large Language Models.](http://arxiv.org/abs/2306.13213) | 本文对将图像引入大型语言模型的安全隐患进行了分析，指出视觉输入空间的连续性和高维性是对抗攻击的丰富领域，同时也为视觉攻击者提供了更广泛的实现对抗目标的可能性。 |
| [^16] | [Prompt to GPT-3: Step-by-Step Thinking Instructions for Humor Generation.](http://arxiv.org/abs/2306.13195) | 本文探讨了如何在GPT-3上实现幽默生成，使用了逐步思维指导的方法，同时探讨了创造幽默的认知距离的作用。 |
| [^17] | [A Reference-less Quality Metric for Automatic Speech Recognition via Contrastive-Learning of a Multi-Language Model with Self-Supervision.](http://arxiv.org/abs/2306.13114) | 本文提出了一种基于反差学习和自监督的多语言模型的无参考质量评价指标，可以在没有真实转录的情况下比较不同ASR模型在语音数据集上的性能，并可将WER降低超过7％。 |
| [^18] | [GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning.](http://arxiv.org/abs/2306.13089) | 本研究提出了一种名为GIMLET的统一图文模型，用于在零样本设置下使用自然语言指令完成分子相关任务。我们解决了现有模型的指令处理不足和图形容量有限的问题，并证明了使用GIMLET能够增强图形特征的泛化能力。 |
| [^19] | [Human-in-the-Loop through Chain-of-Thought.](http://arxiv.org/abs/2306.07932) | 通过人在循环链中的方式，手动校正系统可以通过探究理性中子逻辑的手动校正来提高LLM的推理性能，并且基于经济理论的CAMLOP可以平衡效用和成本。 |
| [^20] | [Visually-Grounded Descriptions Improve Zero-Shot Image Classification.](http://arxiv.org/abs/2306.06077) | 本文提出了一种称为V-GLOSS的新方法，它利用现代语言模型和语义知识库生成具有视觉基础的类别描述，提高了零样本图像分类的准确性，并引入了一个带有类别描述的银标准数据集。 |
| [^21] | [LEACE: Perfect linear concept erasure in closed form.](http://arxiv.org/abs/2306.03819) | 本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。 |
| [^22] | [Revisiting Automated Prompting: Are We Actually Doing Better?.](http://arxiv.org/abs/2304.03609) | 本文重审自动提示技术在六个不同的任务和更广泛范围的K-shot学习设置上的表现，发现自动提示并不能始终优于手动提示，因此手动提示应该作为自动提示的一个基准线。 |
| [^23] | [Extending the Pre-Training of BLOOM for Improved Support of Traditional Chinese: Models, Methods and Results.](http://arxiv.org/abs/2303.04715) | 本文介绍了一种名为BLOOM-zh的多语言语言模型，它扩展了BLOOM的预训练，并具有改进的繁体中文支持。BLOOM-zh在繁体中文基准测试中表现优于其前身。 |
| [^24] | [MarioGPT: Open-Ended Text2Level Generation through Large Language Models.](http://arxiv.org/abs/2302.05981) | MarioGPT是第一个文本到超级马里奥兄弟游戏关卡的生成模型，通过大型语言模型实现开放式的、可控制的关卡生成。 |
| [^25] | [Summarize the Past to Predict the Future: Natural Language Descriptions of Context Boost Multimodal Object Interaction.](http://arxiv.org/abs/2301.09209) | 本文提出了一种TransFusion架构，利用先前训练的图像字幕和视觉语言模型总结动作上下文，实现对多模态对象交互的预测，有效性得到验证。 |
| [^26] | [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.](http://arxiv.org/abs/2212.10509) | 提出了一种名为 IRCoT 的方法，该方法将检索与思路链条步骤交替进行，以引导检索并使用检索结果改进思路链条，从而有效解决了多步问答中先前检索信息不足的问题。 |
| [^27] | [A Natural Bias for Language Generation Models.](http://arxiv.org/abs/2212.09686) | 通过初始化偏差项为log-unigram分布，可为神经语言生成模型赋予单元频率统计的先验知识，从而提高学习效率和翻译质量，准确翻译不经常出现的单词和短语。 |
| [^28] | [INSCIT: Information-Seeking Conversations with Mixed-Initiative Interactions.](http://arxiv.org/abs/2207.00746) | 本文提出了 InSCIt 数据集，用于混合主动交互的信息查寻对话，其中代理人在维基百科上搜索，直接回答、要求澄清或提供相关信息以回应用户的查询。该数据集包含两个子任务（证据段落识别和回答生成）以及人工评估协议，可用于评估模型性能。 |
| [^29] | [Improving Gender Fairness of Pre-Trained Language Models without Catastrophic Forgetting.](http://arxiv.org/abs/2110.05367) | 该论文提出了一种新方法GEEP，用于提高预训练语言模型的性别公平性，同时没有灾难性遗忘问题。透过性别中性数据学习性别相关的提示，GEEP实现了SOTA表现并在GLUE性能上取得了显著提高。 |

# 详细

[^1]: 自带数据！大型语言模型的自我监督评估

    Bring Your Own Data! Self-Supervised Evaluation for Large Language Models. (arXiv:2306.13651v1 [cs.CL])

    [http://arxiv.org/abs/2306.13651](http://arxiv.org/abs/2306.13651)

    本研究提出了一种自我监督评估框架，通过分析输入文本上的变换对LLMs的灵敏度或不变性，直接监控LLMs在实际数据上的行为。

    

    随着大型语言模型（LLMs）的兴起以及它们在各种领域的普及，衡量语言模型在实际数据上的行为变得不可或缺。为了解决这个问题，本研究提出了一种自我监督评估框架，通过分析输入文本上的变换对LLMs的灵敏度或不变性，直接监控LLM在野外收集的数据集或在模型部署期间进行的流数据的行为，实现了评估LLMs的有效和可扩展的解决方案。

    With the rise of Large Language Models (LLMs) and their ubiquitous deployment in diverse domains, measuring language model behavior on realistic data is imperative. For example, a company deploying a client-facing chatbot must ensure that the model will not respond to client requests with profanity. Current evaluations approach this problem using small, domain-specific datasets with human-curated labels. These evaluation sets are often sampled from a narrow and simplified distribution, and data sources can unknowingly be leaked into the training set which can lead to misleading evaluations. To bypass these drawbacks, we propose a framework for self-supervised evaluation of LLMs by analyzing their sensitivity or invariance to transformations on the input text. Self-supervised evaluation can directly monitor LLM behavior on datasets collected in the wild or streamed during live model deployment. We demonstrate self-supervised evaluation strategies for measuring closed-book knowledge, tox
    
[^2]: GKD：自回归序列模型的广义知识蒸馏

    GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models. (arXiv:2306.13649v1 [cs.LG])

    [http://arxiv.org/abs/2306.13649](http://arxiv.org/abs/2306.13649)

    本文提出了广义知识蒸馏（GKD），通过从学生中采样输出序列来缓解分布不匹配，并在优化替代KL等离散度方面处理模型欠规范，达到了在摘要任务上最先进的性能。

    

    知识蒸馏通常用于压缩神经网络，以减少推理成本和内存占用。然而，当前针对自回归模型（如生成语言模型）的蒸馏方法存在两个关键问题：（1）训练期间输出序列和部署时由学生模型生成的序列之间分布不匹配，（2）模型欠规范，学生模型可能不够表达老师分布。为了解决这些问题，我们提出了广义知识蒸馏（GKD）。GKD通过在训练期间从学生中采样输出序列来缓解分布不匹配。此外，GKD通过优化替代KL等离散度来处理模型欠规范，这些离散度集中于生成可能符合老师分布的学生样本。我们证明，在摘要任务上，GKD优于常用的LLM蒸馏方法，在几个基准数据集上实现了最先进的性能。

    Knowledge distillation is commonly used for compressing neural networks to reduce their inference cost and memory footprint. However, current distillation methods for auto-regressive models, such as generative language models (LMs), suffer from two key issues: (1) distribution mismatch between output sequences during training and the sequences generated by the student during its deployment, and (2) model under-specification, where the student model may not be expressive enough to fit the teacher's distribution. To address these issues, we propose Generalized Knowledge Distillation (GKD). GKD mitigates distribution mismatch by sampling output sequences from the student during training. Furthermore, GKD handles model under-specification by optimizing alternative divergences, such as reverse KL, that focus on generating samples from the student that are likely under the teacher's distribution. We demonstrate that GKD outperforms commonly-used approaches for distilling LLMs on summarizatio
    
[^3]: 注意力机制中的边缘最大化

    Margin Maximization in Attention Mechanism. (arXiv:2306.13596v1 [cs.LG])

    [http://arxiv.org/abs/2306.13596](http://arxiv.org/abs/2306.13596)

    这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。

    

    注意力机制是Transformer架构的核心组件，也是大型语言模型取得惊人成功的原因之一。然而，注意力机制背后的理论原则尚不清楚，特别是它的非凸优化动力学。本文探讨了开创性的softmax-attention模型$f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$，其中$\boldsymbol{X}$是标记序列，$(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$是可调参数。我们证明了在$\boldsymbol{p}$或等价的$\boldsymbol{W}$上运行梯度下降会沿着方向收敛到分隔“局部最优”标记和“非最优”标记的最大边缘解。这明确地形式化了注意力作为一种标记分离机制。值得注意的是，我们的结果适用于一般数据，并使用嵌入$\boldsymbol{Xv}$和$\texttt{softmax}(\boldsymbol{XWp})$精细地表征标记的“最优性”。

    Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$, where, $\boldsymbol{X}$ is the token sequence and $(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$ are tunable parameters. We prove that running gradient descent on $\boldsymbol{p}$, or equivalently $\boldsymbol{W}$, converges in direction to a max-margin solution that separates $\textit{locally-optimal}$ tokens from non-optimal ones. This clearly formalizes attention as a token separation mechanism. Remarkably, our results are applicable to general data and precisely characterize $\textit{optimality}$ of tokens in terms of the value embeddings $\boldsymbol{Xv}$ and
    
[^4]: 系统级自然语言反馈

    System-Level Natural Language Feedback. (arXiv:2306.13588v1 [cs.CL])

    [http://arxiv.org/abs/2306.13588](http://arxiv.org/abs/2306.13588)

    本文提出了一个通用框架，用于解锁系统级别使用自然语言反馈的方法。我们展示了通过任务度量设计和语言模型提示设计，如何使用反馈在人工交互流程中形式化系统级别的设计决策，以便产生更好的模型，并展示了使用系统级别反馈和实例级别反馈的有效性。

    

    自然语言反馈包含了丰富的用户体验信息。现有研究聚焦于实例级别的方法，即将反馈用于细化特定例子，而忽略了其系统范围的应用。本文提出了一个通用框架，用于解锁系统级别使用自然语言反馈的方法。我们展示了如何使用反馈在人工交互流程中形式化系统级别的设计决策，以便产生更好的模型。具体而言，这是通过以下两方面实现的：(i) 任务度量设计; (ii) 用于改进模型响应的语言模型提示设计。我们进行了两项案例研究，来改进搜索查询生成和对话响应生成，展示了使用系统级别反馈的有效性。我们表明系统级别反馈和实例级别反馈的组合带来了进一步的收益，并且由人类撰写的实例级别反馈导致比GPT-3.5撰写的反馈更加扎实。

    Natural language (NL) feedback contains rich information about the user experience. Existing studies focus on an instance-level approach, where feedback is used to refine specific examples, disregarding its system-wide application. This paper proposes a general framework for unlocking the system-level use of NL feedback. We show how to use feedback to formalize system-level design decisions in a human-in-the-loop-process -- in order to produce better models. In particular this is done through: (i) metric design for tasks; and (ii) language model prompt design for refining model responses. We conduct two case studies of this approach for improving search query generation and dialog response generation, demonstrating the effectiveness of the use of system-level feedback. We show the combination of system-level feedback and instance-level feedback brings further gains, and that human written instance-level feedback results in more grounded refinements than GPT-3.5 written ones, underlying
    
[^5]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    
[^6]: 知识注入自注意力变压器

    Knowledge-Infused Self Attention Transformers. (arXiv:2306.13501v1 [cs.CL])

    [http://arxiv.org/abs/2306.13501](http://arxiv.org/abs/2306.13501)

    本文介绍了一种将知识注入到变压器模型不同组件的系统方法，可以更彻底地分析知识注入对每个组件和组件之间相互作用的影响，并使变压器模型在各种自然语言处理任务中性能得到了显著提高。

    

    基于变换器的语言模型以其使用自我关注机制捕捉复杂依赖关系和上下文信息的能力，在各种自然语言处理任务中取得了令人瞩目的成功。然而，它们并非没有局限性。这些限制包括幻觉，即它们产生高置信度的错误输出，以及对人类用户生成无用和不安全输出的对齐问题。这些限制源于数据中隐含和缺失上下文的缺乏。为了解决这个问题，研究人员探索了利用来自知识图谱的外部知识来提供必要的附加上下文来增强这些模型的方法。然而，现有方法的临时性质使得在分析知识注入对变压器的许多部分或组件产生的影响以及组件之间的相互影响方面困难重重。本文介绍了一种将知识注入到变压器模型的不同组件中的系统方法。该提出的方法允许更彻底地分析知识注入对每个组件以及组件之间相互作用的影响。实验结果表明，所提出的方法显着提高了变压器模型在各种自然语言处理任务中的性能。

    Transformer-based language models have achieved impressive success in various natural language processing tasks due to their ability to capture complex dependencies and contextual information using self-attention mechanisms. However, they are not without limitations. These limitations include hallucinations, where they produce incorrect outputs with high confidence, and alignment issues, where they generate unhelpful and unsafe outputs for human users. These limitations stem from the absence of implicit and missing context in the data alone. To address this, researchers have explored augmenting these models with external knowledge from knowledge graphs to provide the necessary additional context. However, the ad-hoc nature of existing methods makes it difficult to properly analyze the effects of knowledge infusion on the many moving parts or components of a transformer. This paper introduces a systematic method for infusing knowledge into different components of a transformer-based mod
    
[^7]: 将图表信息融入基于Transformer的AMR分析

    Incorporating Graph Information in Transformer-based AMR Parsing. (arXiv:2306.13467v1 [cs.CL])

    [http://arxiv.org/abs/2306.13467](http://arxiv.org/abs/2306.13467)

    本论文介绍了一种新的模型和方法LeakDistill，它使用结构适配器将图形信息明确并入到学习的表示中，从而提高了AMR解析性能。实验表明，我们可以通过在训练时使用单词到节点对齐将图形结构信息嵌入编码器中，即使不使用其他数据，也可以通过自我知识蒸馏获得最先进的AMR解析性能。

    

    抽象意义表达（AMR）是一种语义解析形式主义，旨在提供表示给定文本的语义图表抽象。当前的方法基于自回归语言模型，例如BART或T5，通过Teacher Forcing进行微调，以从句子得到线性化版本的AMR图表。在本文中，我们提出了LeakDistill，一种探索转换器架构修改的模型和方法，使用结构适配器明确将图形信息并入到学习的表示中，以提高AMR解析性能。我们的实验表明，通过在训练时使用单词到节点对齐将图形结构信息嵌入编码器中，即使不使用其他数据，也可以通过自我知识蒸馏获得最先进的AMR解析性能。我们在\url{this http URL}上发布了代码。

    Abstract Meaning Representation (AMR) is a Semantic Parsing formalism that aims at providing a semantic graph abstraction representing a given text. Current approaches are based on autoregressive language models such as BART or T5, fine-tuned through Teacher Forcing to obtain a linearized version of the AMR graph from a sentence. In this paper, we present LeakDistill, a model and method that explores a modification to the Transformer architecture, using structural adapters to explicitly incorporate graph information into the learned representations and improve AMR parsing performance. Our experiments show how, by employing word-to-node alignment to embed graph structural information into the encoder at training time, we can obtain state-of-the-art AMR parsing through self-knowledge distillation, even without the use of additional data. We release the code at \url{this http URL}.
    
[^8]: 通过半透过最大似然估计学习描述性图像字幕

    Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation. (arXiv:2306.13460v1 [cs.CL])

    [http://arxiv.org/abs/2306.13460](http://arxiv.org/abs/2306.13460)

    本文通过半透过最大似然估计方法，鼓励模型生成更详细的长字幕。

    

    图像字幕旨在用自然语言描述视觉内容。然而，由于最大似然估计是训练目标，字幕模型在预测与标签不匹配时会受到惩罚。本文提出了半透过最大似然估计（SMILE）方法，允许丰富性优化同时阻止简洁性优化，从而鼓励模型生成更详细的长字幕。

    Image captioning aims to describe visual content in natural language. As 'a picture is worth a thousand words', there could be various correct descriptions for an image. However, with maximum likelihood estimation as the training objective, the captioning model is penalized whenever its prediction mismatches with the label. For instance, when the model predicts a word expressing richer semantics than the label, it will be penalized and optimized to prefer more concise expressions, referred to as conciseness optimization. In contrast, predictions that are more concise than labels lead to richness optimization. Such conflicting optimization directions could eventually result in the model generating general descriptions. In this work, we introduce Semipermeable MaxImum Likelihood Estimation (SMILE), which allows richness optimization while blocking conciseness optimization, thus encouraging the model to generate longer captions with more details. Extensive experiments on two mainstream im
    
[^9]: 自检索的长文本语言模型

    Long-range Language Modeling with Self-retrieval. (arXiv:2306.13421v1 [cs.CL])

    [http://arxiv.org/abs/2306.13421](http://arxiv.org/abs/2306.13421)

    本论文提出了一种名为Retrieval-Pretrained Transformer的模型，可以从头开始联合训练语言模型和检索器来模拟长文本。模型可以计算文本块的查询表示，并将其用于检索前面的块，从而融合信息以预测下一个目标块。检索器使用一个语义目标进行训练，目标是检索那些增加下一个块概率的块。

    

    近期，基于检索辅助的语言模型受到了广泛关注。但是，通常检索器并不是作为语言模型的本地组件进行联合训练的，而是被添加到已经预训练好的语言模型中，这限制了语言模型和检索器相互适应的能力。在本文中，我们提出了Retrieval-Pretrained Transformer (RPT)，一种从头开始训练检索辅助的语言模型的架构和训练方法，用于模拟长文本。给定一个最近在长文档中生成的文本块，语言模型计算查询表示，然后用它来检索文档中更早的块，这些块可能跨越数万个标记。检索到的块中的信息被融合到语言模型表示中，以预测下一个目标块。我们用一个语义目标来训练检索器组件，该目标的目的是检索增加下一个块概率的块，根据参考语言模型。我们评估了...

    Retrieval-augmented language models (LMs) have received much attention recently. However, typically the retriever is not trained jointly as a native component of the LM, but added to an already-pretrained LM, which limits the ability of the LM and the retriever to adapt to one another. In this work, we propose the Retrieval-Pretrained Transformer (RPT), an architecture and training procedure for jointly training a retrieval-augmented LM from scratch for the task of modeling long texts. Given a recently generated text chunk in a long document, the LM computes query representations, which are then used to retrieve earlier chunks in the document, located potentially tens of thousands of tokens before. Information from retrieved chunks is fused into the LM representations to predict the next target chunk. We train the retriever component with a semantic objective, where the goal is to retrieve chunks that increase the probability of the next chunk, according to a reference LM. We evaluate 
    
[^10]: 压力测试BERT在化学专利中的回应提取中的指代消解模型

    Stress Testing BERT Anaphora Resolution Models for Reaction Extraction in Chemical Patents. (arXiv:2306.13379v1 [cs.CL])

    [http://arxiv.org/abs/2306.13379](http://arxiv.org/abs/2306.13379)

    该论文通过在化学专利中进行压力测试研究了指代消解模型在无噪声和有噪声环境中的性能差异，旨在提高其对噪声的鲁棒性。

    

    大量的化学专利出版物和及时获取其信息的重要性促使自动化从化学专利中提取信息。 指代消解是综合信息提取的重要组成部分，对于提取反应至关重要。 在化学专利中，存在五种感兴趣的指代关系：共指，转换，反应相关，处理和包含。 我们的目标是研究指代消解模型在无噪声和有噪声环境中用于化学专利中反应文本的性能差异，并在多大程度上可以提高模型对噪声的鲁棒性。

    The high volume of published chemical patents and the importance of a timely acquisition of their information gives rise to automating information extraction from chemical patents. Anaphora resolution is an important component of comprehensive information extraction, and is critical for extracting reactions. In chemical patents, there are five anaphoric relations of interest: co-reference, transformed, reaction associated, work up, and contained. Our goal is to investigate how the performance of anaphora resolution models for reaction texts in chemical patents differs in a noise-free and noisy environment and to what extent we can improve the robustness against noise of the model.
    
[^11]: 利用先进的NLP变形器和LSTM进行简历的提取性文本摘要

    Abstractive Text Summarization for Resumes With Cutting Edge NLP Transformers and LSTM. (arXiv:2306.13315v1 [cs.CL])

    [http://arxiv.org/abs/2306.13315](http://arxiv.org/abs/2306.13315)

    本研究评估了多种技术（包括LSTM、T5、Pegasus、BART和BART-Large模型）在不同数据集上对简历文本进行分类任务的表现，结果显示微调后的BART-Large模型效果最佳。

    

    文本摘要是自然语言处理中的一项基本任务，旨在将大量的文本信息压缩成简洁连贯的摘要。随着内容的指数增长和高效提取关键信息的需求，文本摘要在近年来受到了极大的关注。在本研究中，评估了LSTM和预训练的T5、Pegasus、BART和BART-Large模型在开源数据集（Xsum、CNN/Daily Mail、亚马逊精美食品评论和新闻摘要）和准备的简历数据集上的表现。该简历数据集包括许多信息，如语言、教育、经验、个人信息、技能等，数据集中包括了75份简历。本研究的主要目标是对简历文本进行分类。使用简历数据集评估了各种技术，包括LSTM、预训练模型和微调模型。使用简历数据集微调的BART-Large模型表现最佳。

    Text summarization is a fundamental task in natural language processing that aims to condense large amounts of textual information into concise and coherent summaries. With the exponential growth of content and the need to extract key information efficiently, text summarization has gained significant attention in recent years. In this study, LSTM and pre-trained T5, Pegasus, BART and BART-Large model performances were evaluated on the open source dataset (Xsum, CNN/Daily Mail, Amazon Fine Food Review and News Summary) and the prepared resume dataset. This resume dataset consists of many information such as language, education, experience, personal information, skills, and this data includes 75 resumes. The primary objective of this research was to classify resume text. Various techniques such as LSTM, pre-trained models, and fine-tuned models were assessed using a dataset of resumes. The BART-Large model fine-tuned with the resume dataset gave the best performance.
    
[^12]: 相互指导的少样本学习在关系三元组提取中的应用

    Mutually Guided Few-shot Learning for Relational Triple Extraction. (arXiv:2306.13310v1 [cs.CL])

    [http://arxiv.org/abs/2306.13310](http://arxiv.org/abs/2306.13310)

    提出了相互指导的少样本学习框架，以进行关系三元组提取，并引入了一个新的跨域少样本三元组提取任务，实现了在少样本情况下的有竞争力结果。

    

    知识图谱（KGs）包含许多实体-关系-实体三元组，为下游应用提供了丰富的信息。尽管从非结构化文本中提取三元组已经广泛探索，但大部分方法需要大量标注实例。当只有少量标记数据可用时，性能将急剧下降。为了解决这个问题，我们提出了相互指导的少样本学习框架，以进行关系三元组提取（MG-FTE）。具体而言，我们的方法包含一个以实体为导向的关系原型解码器，首先对关系进行分类，以及一个以关系为导向的实体原型解码器，根据分类的关系提取实体。为了连结实体与关系，我们设计了原型层融合模块，以提高实体提取和关系分类的性能。此外，我们还引入了一个新的跨域少样本三元组提取任务。广泛的实验表明，我们的方法在少样本三元组提取任务中优于许多最先进的方法，即使只有少量标记数据可用时，也能取得有竞争力的结果。

    Knowledge graphs (KGs), containing many entity-relation-entity triples, provide rich information for downstream applications. Although extracting triples from unstructured texts has been widely explored, most of them require a large number of labeled instances. The performance will drop dramatically when only few labeled data are available. To tackle this problem, we propose the Mutually Guided Few-shot learning framework for Relational Triple Extraction (MG-FTE). Specifically, our method consists of an entity-guided relation proto-decoder to classify the relations firstly and a relation-guided entity proto-decoder to extract entities based on the classified relations. To draw the connection between entity and relation, we design a proto-level fusion module to boost the performance of both entity extraction and relation classification. Moreover, a new cross-domain few-shot triple extraction task is introduced. Extensive experiments show that our method outperforms many state-of-the-art
    
[^13]: 为Conformer Transducer语音识别系统建立高效紧凑的上下文表示

    Towards Effective and Compact Contextual Representation for Conformer Transducer Speech Recognition Systems. (arXiv:2306.13307v1 [eess.AS])

    [http://arxiv.org/abs/2306.13307](http://arxiv.org/abs/2306.13307)

    本研究旨在为Conformer Transducer语音识别系统建立高效紧凑的上下文表示，通过特定的注意力汇聚层实现跨话语的信息集成，取得了显著的性能提升。

    

    当前的自动语音识别系统主要在话语级别进行训练和评估。可以纳入长范围的跨话语上下文信息。本文提出，在Conformer-Transducer编码器中使用特殊设计的注意力汇聚层，通过高效缓存的前面话语历史向量，学习了紧凑的低维跨话语上下文特征。在1000小时的Gigaspeech语料库上的实验表明，所提出的上下文化流Conformer-Transducer在开发和测试数据上都比仅使用话语内部上下文的基线模型具有显著的字错率降低（0.7％到0.5％绝对，4.3％到3.1％相对）。

    Current ASR systems are mainly trained and evaluated at the utterance level. Long range cross utterance context can be incorporated. A key task is to derive a suitable compact representation of the most relevant history contexts. In contrast to previous researches based on either LSTM-RNN encoded histories that attenuate the information from longer range contexts, or frame level concatenation of transformer context embeddings, in this paper compact low-dimensional cross utterance contextual features are learned in the Conformer-Transducer Encoder using specially designed attention pooling layers that are applied over efficiently cached preceding utterances history vectors. Experiments on the 1000-hr Gigaspeech corpus demonstrate that the proposed contextualized streaming Conformer-Transducers outperform the baseline using utterance internal context only with statistically significant WER reductions of 0.7% to 0.5% absolute (4.3% to 3.1% relative) on the dev and test data.
    
[^14]: DiversiGATE: 一个可靠的大规模语言模型全面框架

    DiversiGATE: A Comprehensive Framework for Reliable Large Language Models. (arXiv:2306.13230v1 [cs.CL])

    [http://arxiv.org/abs/2306.13230](http://arxiv.org/abs/2306.13230)

    DiversiGATE是一个统一框架，汇集了多种LLM验证方法，其中包括自一致性、数学提示和WebGPT，同时提出了一个符合该框架的新模型“SelfLearner”，该模型可以从自己的输出中学习并优化性能，在实验中表现良好，GSM8K基准测试上提高了7%的性能。

    

    本文提出了DiversiGATE，一个统一的框架，汇集LLM验证的多种方法。该框架包括两个主要组成部分：多样化和聚合，在现有的验证方法上提供了全面的视角，例如自一致性、数学提示和WebGPT。此外，本文提出了一个新颖的“SelfLearner”模型，符合DiversiGATE框架，可以从自己的输出中学习并随着时间的推移不断完善其性能，从而提高准确性。为了评估SelfLearner的有效性，我们进行了一系列严格的实验，包括对合成数据和广泛使用的算术推理基准测试GSM8K的测试。我们的结果表明，我们的方法优于传统的LLMs，在GSM8K基准测试中实现了可观的54.8%->61.8%的提高。

    In this paper, we introduce DiversiGATE, a unified framework that consolidates diverse methodologies for LLM verification. The proposed framework comprises two main components: Diversification and Aggregation which provide a holistic perspective on existing verification approaches, such as Self-Consistency, Math Prompter and WebGPT. Furthermore, we propose a novel `SelfLearner' model that conforms to the DiversiGATE framework which can learn from its own outputs and refine its performance over time, leading to improved accuracy. To evaluate the effectiveness of SelfLearner, we conducted a rigorous series of experiments, including tests on synthetic data as well as on popular arithmetic reasoning benchmarks such as GSM8K. Our results demonstrate that our approach outperforms traditional LLMs, achieving a considerable 54.8% -> 61.8% improvement on the GSM8K benchmark.
    
[^15]: 视觉对抗样本越狱大语言模型的安全隐患分析

    Visual Adversarial Examples Jailbreak Large Language Models. (arXiv:2306.13213v1 [cs.CR])

    [http://arxiv.org/abs/2306.13213](http://arxiv.org/abs/2306.13213)

    本文对将图像引入大型语言模型的安全隐患进行了分析，指出视觉输入空间的连续性和高维性是对抗攻击的丰富领域，同时也为视觉攻击者提供了更广泛的实现对抗目标的可能性。

    

    最近，将图像引入大型语言模型（LLMs）已经引起了人们的高度关注。大型视觉语言模型（VLMs）的普及，例如Flamingo、BLIP-2和GPT-4，标志着视觉和语言基础模型的先进发展相互融合的重要进展。然而，这种综合方法涉及的风险仍未得到详细研究。本文揭示了这一趋势的安全隐患。我们首先指出，视觉输入空间的连续性和高维性在本质上使其成为对抗攻击的丰富领域，这不可避免地扩大了LLMs的攻击面。其次，我们强调，LLMs的广泛功能也为视觉攻击者提供了更广泛的实现对抗目标的可能性，将安全失败的影响扩展到了简单的错误分类之外。为了阐明这些风险，我们研究了VLM视觉输入空间中的对抗性样例。

    Recently, there has been a surge of interest in introducing vision into Large Language Models (LLMs). The proliferation of large Visual Language Models (VLMs), such as Flamingo, BLIP-2, and GPT-4, signifies an exciting convergence of advancements in both visual and language foundation models. Yet, the risks associated with this integrative approach are largely unexamined. In this paper, we shed light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the additional visual input space intrinsically makes it a fertile ground for adversarial attacks. This unavoidably expands the attack surfaces of LLMs. Second, we highlight that the broad functionality of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. To elucidate these risks, we study adversarial examples in the visual input space of a VLM.
    
[^16]: GPT-3中幽默生成的逐步思维指导

    Prompt to GPT-3: Step-by-Step Thinking Instructions for Humor Generation. (arXiv:2306.13195v1 [cs.CL])

    [http://arxiv.org/abs/2306.13195](http://arxiv.org/abs/2306.13195)

    本文探讨了如何在GPT-3上实现幽默生成，使用了逐步思维指导的方法，同时探讨了创造幽默的认知距离的作用。

    

    人工智能在自然语言处理方面取得了重大进展，像GPT-3这样的模型展示了令人印象深刻的能力。但是，当涉及到需要理解用户的复杂任务时，这些模型仍然存在局限性，比如掌握人类喜剧写作策略。本文通过建模人类喜剧写作理论和利用逐步思维指导来探讨使用GPT-3进行幽默生成。此外，我们还探讨了创造幽默的认知距离的作用。

    Artificial intelligence has made significant progress in natural language processing, with models like GPT-3 demonstrating impressive capabilities. However, these models still have limitations when it comes to complex tasks that require an understanding of the user, such as mastering human comedy writing strategies. This paper explores humor generation using GPT-3 by modeling human comedy writing theory and leveraging step-by-step thinking instructions. In addition, we explore the role of cognitive distance in creating humor.
    
[^17]: 一种基于反差学习和自监督的多语言模型的无参考自动语音识别质量评价指标

    A Reference-less Quality Metric for Automatic Speech Recognition via Contrastive-Learning of a Multi-Language Model with Self-Supervision. (arXiv:2306.13114v1 [cs.CL])

    [http://arxiv.org/abs/2306.13114](http://arxiv.org/abs/2306.13114)

    本文提出了一种基于反差学习和自监督的多语言模型的无参考质量评价指标，可以在没有真实转录的情况下比较不同ASR模型在语音数据集上的性能，并可将WER降低超过7％。

    

    自动语音识别（ASR）系统的质量评价通常采用基于参考的指标，如使用耗时且昂贵的手工真实转录来计算的词错误率（WER）。本文提出了一种多语言无参考质量评价指标，可在没有真实转录的情况下比较不同ASR模型在语音数据集上的性能。为了估计ASR假设的质量，使用预训练的语言模型（LM）以自监督学习方式进行反差学习的微调。在对多个未见过的测试数据集进行的实验中，所提出的无参考指标在所有实验中都比最先进的多语言LM的困惑度指标获得了更高的与WER得分及其排名的相关性，并且在用于合并假设时可将WER降低超过7％。

    The common standard for quality evaluation of automatic speech recognition (ASR) systems is reference-based metrics such as the Word Error Rate (WER), computed using manual ground-truth transcriptions that are time-consuming and expensive to obtain. This work proposes a multi-language referenceless quality metric, which allows comparing the performance of different ASR models on a speech dataset without ground truth transcriptions. To estimate the quality of ASR hypotheses, a pre-trained language model (LM) is fine-tuned with contrastive learning in a self-supervised learning manner. In experiments conducted on several unseen test datasets consisting of outputs from top commercial ASR engines in various languages, the proposed referenceless metric obtains a much higher correlation with WER scores and their ranks than the perplexity metric from the state-of-art multi-lingual LM in all experiments, and also reduces WER by more than $7\%$ when used for ensembling hypotheses. The fine-tune
    
[^18]: GIMLET：一种用于基于指令分子零样本学习的统一图文模型

    GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning. (arXiv:2306.13089v1 [cs.LG])

    [http://arxiv.org/abs/2306.13089](http://arxiv.org/abs/2306.13089)

    本研究提出了一种名为GIMLET的统一图文模型，用于在零样本设置下使用自然语言指令完成分子相关任务。我们解决了现有模型的指令处理不足和图形容量有限的问题，并证明了使用GIMLET能够增强图形特征的泛化能力。

    

    分子属性预测近年来受到了广泛关注，但由于昂贵的实验造成的标签不足问题将是其主要瓶颈。为了缓解这个问题并更好地利用文本知识进行任务，本研究探讨了在零样本设置下使用自然语言指令完成分子相关任务的可行性。我们发现现有的分子-文本模型在这种情况下表现不佳，原因是处理指令不足以及图形容量有限。为了克服这些问题，我们提出了GIMLET，它统一了图形和文本数据的语言模型。通过采用广义位置嵌入，我们的模型被扩展以编码图形结构和指令文本，而无需额外的图形编码模块。GIMLET还在注意机制中解耦了图形的编码和任务指令，增强了跨新任务的图形特征的泛化能力。我们构建了一个数据集...

    Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs. To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset 
    
[^19]: 人在循环链中。

    Human-in-the-Loop through Chain-of-Thought. (arXiv:2306.07932v1 [cs.CL])

    [http://arxiv.org/abs/2306.07932](http://arxiv.org/abs/2306.07932)

    通过人在循环链中的方式，手动校正系统可以通过探究理性中子逻辑的手动校正来提高LLM的推理性能，并且基于经济理论的CAMLOP可以平衡效用和成本。

    

    尽管强大的语言模型和思维链提示的出现使自动化变得越来越无处不在，但有时在长期或多步逻辑推理方面显示出其弱点。例如，用户在没有人类参与的情况下不总能得到复杂数学问题的理想答案。在这个背景下，我们提出了手动校正系统（MCS）——一个通过思维链提示增强的人工参与系统，探究了理性中子逻辑的手动校正如何提高LLM的推理性能。更进一步考虑到有人参与的系统不仅要提高性能，还要控制成本。因此，我们提出了基于古典经济理论的人在循环链中成本效用分析模型（CAMLOP）来分析、量化和平衡效用和相应的成本。我们使用12个数据集对MCS和CAMLOP进行了实验。

    While the emergence of powerful language models along with Chain-of-thought prompting has made automation more and more omnipresent, it sometimes demonstrates its weakness in long-term or multi-step logical reasoning. For example, users don't always get desirable answers for complex mathematical problems without human involvement. Against this background, we present the Manual Correction System (MCS) -- a human-in-the-loop system enhanced by Chain-of-Thought prompting, which explores how manual correction of sub-logics in rationales can improve LLM's reasoning performance. Moving one step forward, considering a system with human-in-the-loop involves more than having humans improve performance but also controlling the cost. Therefore, we post a Cost-utility Analysis Model for Human-in-the-Loop systems (CAMLOP) based on classical economics theory to analyze, quantify and balance the utility and the corresponding cost. We conduct experiments of MCS and CAMLOP with twelve datasets. A signi
    
[^20]: 视觉词汇描述提升零样本图像分类

    Visually-Grounded Descriptions Improve Zero-Shot Image Classification. (arXiv:2306.06077v1 [cs.CV])

    [http://arxiv.org/abs/2306.06077](http://arxiv.org/abs/2306.06077)

    本文提出了一种称为V-GLOSS的新方法，它利用现代语言模型和语义知识库生成具有视觉基础的类别描述，提高了零样本图像分类的准确性，并引入了一个带有类别描述的银标准数据集。

    

    语言视觉模型如CLIP在零样本视觉任务（例如零样本图像分类ZSIC）方面取得了显著进展。然而，生成具体和富有表现力的类别描述仍然是一个主要挑战。现有方法存在粒度和标签歧义等问题。为了解决这些挑战，我们提出了一种新方法V-GLOSS：Visual Glosses，它利用现代语言模型和语义知识库来生成具有视觉基础的类别描述。我们通过在基准ZSIC数据集（包括ImageNet和STL-10）上实现最先进的结果来展示V-GLOSS的有效性。此外，我们引入了一个由V-GLOSS生成的带有类别描述的银标准数据集，并展示其用于视觉任务的有用性。我们提供了源代码和数据集。

    Language-vision models like CLIP have made significant progress in zero-shot vision tasks, such as zero-shot image classification (ZSIC). However, generating specific and expressive class descriptions remains a major challenge. Existing approaches suffer from granularity and label ambiguity issues. To tackle these challenges, we propose V-GLOSS: Visual Glosses, a novel method leveraging modern language models and semantic knowledge bases to produce visually-grounded class descriptions. We demonstrate V-GLOSS's effectiveness by achieving state-of-the-art results on benchmark ZSIC datasets including ImageNet and STL-10. In addition, we introduce a silver dataset with class descriptions generated by V-GLOSS, and show its usefulness for vision tasks. We make available our code and dataset.
    
[^21]: LEACE：闭合形式中的完美线性概念擦除

    LEACE: Perfect linear concept erasure in closed form. (arXiv:2306.03819v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.03819](http://arxiv.org/abs/2306.03819)

    本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。

    

    概念擦除旨在从表征中删除指定的特征。它可以提高公平性（例如，防止分类器使用性别或种族）和可解释性（例如，删除概念以观察模型行为的变化）。我们引入了LEAst-squares概念擦除（LEACE），这是一种闭合形式的方法，可证明防止所有线性分类器检测到概念，同时尽可能地改变表示，如广泛类别的范数所测量的那样。我们使用名为“概念擦除”的新方法将LEACE应用于大型语言模型，擦除每个层中的目标概念信息。我们在两个任务上展示了我们的方法：测量语言模型对词性信息的依赖性，以及减少BERT嵌入中的性别偏差。代码可在https://github.com/EleutherAI/concept-erasure上找到。

    Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called "concept scrubbing," which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Code is available at https://github.com/EleutherAI/concept-erasure.
    
[^22]: 重新审视自动提示：我们真的做得更好吗？

    Revisiting Automated Prompting: Are We Actually Doing Better?. (arXiv:2304.03609v1 [cs.CL])

    [http://arxiv.org/abs/2304.03609](http://arxiv.org/abs/2304.03609)

    本文重审自动提示技术在六个不同的任务和更广泛范围的K-shot学习设置上的表现，发现自动提示并不能始终优于手动提示，因此手动提示应该作为自动提示的一个基准线。

    

    当前的文献表明，大型语言模型(LLM)是出色的几乎不用学习的学习者，在几乎不用学习的情况下，提示显着提高了它们在多个下游任务中的表现。随后进行了试图自动化人类提示的尝试，并取得了一定进展。特别是，随后的工作表明，在某些K-shot学习场景中，自动化可以优于微调。在本文中，我们重新审视了自动提示在六个不同的下游任务和更大范围的K-shot学习设置上的技术。我们发现，自动提示不能始终优于简单的手动提示。我们的工作表明，在这一研究领域中，除了微调之外，手动提示应作为基线使用。

    Current literature demonstrates that Large Language Models (LLMs) are great few-shot learners, and prompting significantly increases their performance on a range of downstream tasks in a few-shot learning setting. An attempt to automate human-led prompting followed, with some progress achieved. In particular, subsequent work demonstrates automation can outperform fine-tuning in certain K-shot learning scenarios.  In this paper, we revisit techniques for automated prompting on six different downstream tasks and a larger range of K-shot learning settings. We find that automated prompting does not consistently outperform simple manual prompts. Our work suggests that, in addition to fine-tuning, manual prompts should be used as a baseline in this line of research.
    
[^23]: BLOOM的预训练扩展以改善对繁体中文的支持：模型、方法和结果

    Extending the Pre-Training of BLOOM for Improved Support of Traditional Chinese: Models, Methods and Results. (arXiv:2303.04715v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.04715](http://arxiv.org/abs/2303.04715)

    本文介绍了一种名为BLOOM-zh的多语言语言模型，它扩展了BLOOM的预训练，并具有改进的繁体中文支持。BLOOM-zh在繁体中文基准测试中表现优于其前身。

    

    本文介绍了一种名为BLOOM-zh的多语言语言模型，它具有改进的繁体中文支持。BLOOM-zh起源于由BigScience于2022年推出的开源BLOOM模型。我们在已发布的模型基础上，使用74亿个额外的繁体中文和英文标记进行了扩展，覆盖了各种领域，如新闻文章、书籍、百科全书、教育材料以及口语语言。为了展示BLOOM-zh的性质，我们使用现有的和新创建的基准场景来评估其性能。在大多数的繁体中文基准测试中，BLOOM-zh的性能优于其前身，同时保持了其英文能力。我们将所有模型发布给研究社区。

    In this paper we present the multilingual language model BLOOM-zh that features enhanced support for Traditional Chinese. BLOOM-zh has its origins in the open-source BLOOM models presented by BigScience in 2022. Starting from released models, we extended the pre-training of BLOOM by additional 7.4 billion tokens in Traditional Chinese and English covering a variety of domains such as news articles, books, encyclopedias, educational materials as well as spoken language. In order to show the properties of BLOOM-zh, both existing and newly created benchmark scenarios are used for evaluating the performance. BLOOM-zh outperforms its predecessor on most Traditional Chinese benchmarks while maintaining its English capability. We release all our models to the research community.
    
[^24]: MarioGPT: 通过大语言模型进行开放式文本关卡生成

    MarioGPT: Open-Ended Text2Level Generation through Large Language Models. (arXiv:2302.05981v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.05981](http://arxiv.org/abs/2302.05981)

    MarioGPT是第一个文本到超级马里奥兄弟游戏关卡的生成模型，通过大型语言模型实现开放式的、可控制的关卡生成。

    

    流程内容生成算法可以自动生成复杂数一致的环境。然而，使用流程内容生成方法生成反映特定意图和限制的有意义内容仍然具有挑战性。此外，许多流程内容生成算法缺乏以开放式方式生成内容的能力。最近，大型语言模型在许多不同领域都表现出了非常高的效率。这些训练有素的大型语言模型可以进行微调，重复使用信息并加速新任务的培训。在这项工作中，我们介绍了MarioGPT，这是一个经过优化的GPT2模型，用于生成基于瓷砖的游戏关卡，我们以超级马里奥兄弟的关卡为例。我们展示了MarioGPT不仅可以生成不同的游戏关卡，而且可以通过文本提示控制关卡生成，解决了当前PCG技术的主要挑战之一。据我们所知，MarioGPT是第一个文本到关卡模型。

    Procedural Content Generation (PCG) algorithms provide a technique to generate complex and diverse environments in an automated way. However, while generating content with PCG methods is often straightforward, generating meaningful content that reflects specific intentions and constraints remains challenging. Furthermore, many PCG algorithms lack the ability to generate content in an open-ended manner. Recently, Large Language Models (LLMs) have shown to be incredibly effective in many diverse domains. These trained LLMs can be fine-tuned, re-using information and accelerating training for new tasks. In this work, we introduce MarioGPT, a fine-tuned GPT2 model trained to generate tile-based game levels, in our case Super Mario Bros levels. We show that MarioGPT can not only generate diverse levels, but can be text-prompted for controllable level generation, addressing one of the key challenges of current PCG techniques. As far as we know, MarioGPT is the first text-to-level model. We a
    
[^25]: 总结过去以预测未来：自然语言对场景的描述促进多模态对象交互

    Summarize the Past to Predict the Future: Natural Language Descriptions of Context Boost Multimodal Object Interaction. (arXiv:2301.09209v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.09209](http://arxiv.org/abs/2301.09209)

    本文提出了一种TransFusion架构，利用先前训练的图像字幕和视觉语言模型总结动作上下文，实现对多模态对象交互的预测，有效性得到验证。

    

    本论文针对自我中心视频中的对象交互预测进行了研究。该任务需要理解先前对对象执行的动作所形成的时空上下文，称为动作上下文。我们提出了一种基于多模态transformer的架构TransFusion。它利用语言的表达能力，对动作上下文进行总结。TransFusion利用预先训练的图像字幕和视觉语言模型从过去的视频帧中提取动作上下文。将这个动作上下文与下一个视频帧一起经过多模态融合模块进行处理，从而预测下一个对象交互。我们的模型实现了更高效的端到端学习，大型预训练语言模型则增加了通用性和泛化能力。在Ego4D和EPIC-KITCHENS-100上的实验证实了我们的多模态融合模型的有效性。同时，也凸显了在一个视觉似乎足够的任务中使用基于语言的上下文摘要的好处。我们的方法胜过了现有的方法。

    We study object interaction anticipation in egocentric videos. This task requires an understanding of the spatiotemporal context formed by past actions on objects, coined action context. We propose TransFusion, a multimodal transformer-based architecture. It exploits the representational power of language by summarising the action context. TransFusion leverages pre-trained image captioning and vision-language models to extract the action context from past video frames. This action context together with the next video frame is processed by the multimodal fusion module to forecast the next object interaction. Our model enables more efficient end-to-end learning. The large pre-trained language models add common sense and a generalisation capability. Experiments on Ego4D and EPIC-KITCHENS-100 show the effectiveness of our multimodal fusion model. They also highlight the benefits of using language-based context summaries in a task where vision seems to suffice. Our method outperforms state-
    
[^26]: 利用思路链条推理交错式检索解决知识密集型多步问题

    Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. (arXiv:2212.10509v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10509](http://arxiv.org/abs/2212.10509)

    提出了一种名为 IRCoT 的方法，该方法将检索与思路链条步骤交替进行，以引导检索并使用检索结果改进思路链条，从而有效解决了多步问答中先前检索信息不足的问题。

    

    基于提示的大型语言模型在多步问答中生成自然语言推理步骤或思路链条（CoT）时具有出色的强大性能。然而，当所需知识不可用或不在模型参数中更新时，它们可能出错。虽然使用问题从外部知识源检索相关文本有助于大型语言模型，但我们观察到这种一步检索和阅读方法对于多步问题回答不足够。对于多步问题，需要根据先前得出的内容选择检索内容，而这可能依赖于之前检索过的内容。为了解决这个问题，我们提出了 IRCoT，一种新的多步问答方法，它将检索与思路链条中的步骤进行交错，以思路链条引导检索，并使用检索结果来改进思路链条。在四个数据集上使用 IRCoT 与 GPT3 可以大大提高检索（高达 21 点）和下游问答（高达 15 点）的性能。

    Prompting-based large language models (LLMs) are surprisingly powerful at generating natural language reasoning steps or Chains-of-Thoughts (CoT) for multi-step question answering (QA). They struggle, however, when the necessary knowledge is either unavailable to the LLM or not up-to-date within its parameters. While using the question to retrieve relevant text from an external knowledge source helps LLMs, we observe that this one-step retrieve-and-read approach is insufficient for multi-step QA. Here, \textit{what to retrieve} depends on \textit{what has already been derived}, which in turn may depend on \textit{what was previously retrieved}. To address this, we propose IRCoT, a new approach for multi-step QA that interleaves retrieval with steps (sentences) in a CoT, guiding the retrieval with CoT and in turn using retrieved results to improve CoT. Using IRCoT with GPT3 substantially improves retrieval (up to 21 points) as well as downstream QA (up to 15 points) on four datasets: Ho
    
[^27]: 语言生成模型的自然倾向

    A Natural Bias for Language Generation Models. (arXiv:2212.09686v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09686](http://arxiv.org/abs/2212.09686)

    通过初始化偏差项为log-unigram分布，可为神经语言生成模型赋予单元频率统计的先验知识，从而提高学习效率和翻译质量，准确翻译不经常出现的单词和短语。

    

    经过几百个训练循环之后，一个标准的语言生成概率模型可能还没有学会自然语言的许多语义或句法规则，这使得难以估计下一个令牌的概率分布。但在这一点左右，这些模型已经确定了一种简单的最小化损失的行为：输出目标训练语料库的单元分布。使用这种启发式方法引出了一个问题：我们可以初始化我们的模型，使用这种行为并节省宝贵的计算资源和模型容量吗？在这里，我们展示了一个方法，可以有效地为标准神经语言生成模型赋予反映单元频率统计作为先验知识的单独模块，只需通过将模型的最终线性层的偏差项初始化为log-unigram分布来实现。我们以神经机器翻译为测试基础，观察到：（i）提高了学习效率；（ii）实现了更好的整体翻译质量；（iii）能够更准确地翻译不经常出现的单词和短语。

    After just a few hundred training updates, a standard probabilistic model for language generation has likely not yet learnt many semantic or syntactic rules of natural language, making it difficult to estimate the probability distribution over next tokens. Yet around this point, these models have identified a simple, loss-minimising behaviour: to output the unigram distribution of the target training corpus. The use of such a heuristic raises the question: Can we initialise our models with this behaviour and save precious compute resources and model capacity? Here we show that we can effectively endow standard neural language generation models with a separate module that reflects unigram frequency statistics as prior knowledge, simply by initialising the bias term in a model's final linear layer with the log-unigram distribution. We use neural machine translation as a test bed for this simple technique and observe that it: (i) improves learning efficiency; (ii) achieves better overall 
    
[^28]: INSCIT: 混合主动交互的信息查寻对话

    INSCIT: Information-Seeking Conversations with Mixed-Initiative Interactions. (arXiv:2207.00746v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.00746](http://arxiv.org/abs/2207.00746)

    本文提出了 InSCIt 数据集，用于混合主动交互的信息查寻对话，其中代理人在维基百科上搜索，直接回答、要求澄清或提供相关信息以回应用户的查询。该数据集包含两个子任务（证据段落识别和回答生成）以及人工评估协议，可用于评估模型性能。

    

    在信息查寻对话中，用户可能会提出模糊或无法回答的问题。理想的代理人应根据可用的知识源启动不同的响应类型进行交互。然而，大多数当前的研究要么未能够，要么人为地纳入这种代理人端的主动性。本文提出了 InSCIt 数据集，用于混合主动交互的信息查寻对话。它包含805个人-人对话中4.7K个用户-代理人交互，其中代理人在维基百科上搜索，直接回答、要求澄清或提供相关信息以回应用户的查询。该数据支持两个子任务，证据段落识别和回答生成，以及一个人工评估协议，以评估模型性能。我们报告了两个基于最先进的对话知识识别和开放域问答模型的系统的结果。这两个系统都明显表现不及人类。

    In an information-seeking conversation, a user may ask questions that are under-specified or unanswerable. An ideal agent would interact by initiating different response types according to the available knowledge sources. However, most current studies either fail to or artificially incorporate such agent-side initiative. This work presents InSCIt, a dataset for Information-Seeking Conversations with mixed-initiative Interactions. It contains 4.7K user-agent turns from 805 human-human conversations where the agent searches over Wikipedia and either directly answers, asks for clarification, or provides relevant information to address user queries. The data supports two subtasks, evidence passage identification and response generation, as well as a human evaluation protocol to assess model performance. We report results of two systems based on state-of-the-art models of conversational knowledge identification and open-domain question answering. Both systems significantly underperform huma
    
[^29]: 在不产生灾难性遗忘的情况下提高预训练语言模型的性别公平性。

    Improving Gender Fairness of Pre-Trained Language Models without Catastrophic Forgetting. (arXiv:2110.05367v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2110.05367](http://arxiv.org/abs/2110.05367)

    该论文提出了一种新方法GEEP，用于提高预训练语言模型的性别公平性，同时没有灾难性遗忘问题。透过性别中性数据学习性别相关的提示，GEEP实现了SOTA表现并在GLUE性能上取得了显著提高。

    

    现有的解决预训练语言模型性别偏见的研究通常建立一个小型的性别中性数据集，然后在该数据集上对模型进行第二阶段的预训练。然而，鉴于性别中性数据集的规模有限且集中关注，第二阶段预训练会出现灾难性遗忘。忘记原始训练数据中的信息可能会严重损害模型在下游任务中的性能。在这项工作中，我们通过在GLUE中进行评估，实证地表明这种方法中会发生灾难性遗忘。然后，我们提出了一种新方法，GEnder Equality Prompt (GEEP)，以改善预训练模型的性别公平性，且遗忘较少。 GEEP会冻结预训练模型，并使用性别中性数据学习与性别相关的提示。实证结果显示，GEEP不仅在性别公平任务上实现了SOTA表现，而且在GLUE上遗忘较少，并取得了明显的性能提高。

    Existing studies addressing gender bias of pre-trained language models, usually build a small gender-neutral data set and conduct a second phase pre-training on the model with such data. However, given the limited size and concentrated focus of the gender-neutral data, catastrophic forgetting would occur during second-phase pre-training. Forgetting information in the original training data may damage the model's downstream performance by a large margin. In this work, we empirically show that catastrophic forgetting occurs in such methods by evaluating them with general NLP tasks in GLUE. Then, we propose a new method, GEnder Equality Prompt (GEEP), to improve gender fairness of pre-trained models with less forgetting. GEEP freezes the pre-trained model and learns gender-related prompts with gender-neutral data. Empirical results show that GEEP not only achieves SOTA performances on gender fairness tasks, but also forgets less and performs better on GLUE by a large margin.
    

