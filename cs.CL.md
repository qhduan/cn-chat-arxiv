# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Calibrating the Confidence of Large Language Models by Eliciting Fidelity](https://arxiv.org/abs/2404.02655) | 本文通过将语言模型的置信度分解为问题的不确定性和对答案的忠实性，提出了一种估计语言模型置信度的即插即用方法，经实验证明具有良好的校准性能。 |
| [^2] | [UniMEEC: Towards Unified Multimodal Emotion Recognition and Emotion Cause](https://arxiv.org/abs/2404.00403) | UniMEEC提出了一个统一的多模情绪识别和情绪-原因分析框架，将MERC和MECPE重新定义为两个掩码预测问题，以增强情绪和原因之间的交互作用。 |
| [^3] | [Emergent communication and learning pressures in language models: a language evolution perspective](https://arxiv.org/abs/2403.14427) | 从语言进化的角度研究了新兴沟通文献，发现其在设计和调整模型以恢复自然语言中初始缺失的语言现象方面表现优秀，揭示了关键压力促使恢复最初不显现的人类模式。 |
| [^4] | [A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236) | 这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。 |
| [^5] | [Less is More: Data Value Estimation for Visual Instruction Tuning](https://arxiv.org/abs/2403.09559) | 视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。 |
| [^6] | [Large Language Models are Parallel Multilingual Learners](https://arxiv.org/abs/2403.09073) | 通过将输入翻译为多种语言，为大型语言模型提供多语言平行输入，显著增强了它们的理解能力，实验证明多语言输入可以超越传统学习方法，并发现了神经元激活的反直觉现象 |
| [^7] | [Rebuilding ROME : Resolving Model Collapse during Sequential Model Editing](https://arxiv.org/abs/2403.07175) | 本文重建了ROME，提供了更稳定的r-ROME实现，解决了顺序模型编辑过程中的模型崩溃问题。 |
| [^8] | [Evidence-Focused Fact Summarization for Knowledge-Augmented Zero-Shot Question Answering](https://arxiv.org/abs/2403.02966) | 提出了一种面向证据的事实摘要化框架EFSum，用于增强LLMs的零-shot QA性能，并确保摘要的有益性和忠实性。 |
| [^9] | [Few shot clinical entity recognition in three languages: Masked language models outperform LLM prompting](https://arxiv.org/abs/2402.12801) | 掩盖语言模型在三种语言中的少样本临床实体识别中表现优异，胜过LLM提示方法 |
| [^10] | [FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning](https://arxiv.org/abs/2402.12692) | FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。 |
| [^11] | [Automatic Evaluation for Mental Health Counseling using LLMs](https://arxiv.org/abs/2402.11958) | 使用LLMs自动评估心理咨询对话中的工作联盟，结果显示与人工评估高度一致，并提供宝贵见解。 |
| [^12] | [Can Separators Improve Chain-of-Thought Prompting?](https://arxiv.org/abs/2402.10645) | 分隔符的引入在思维链提示中显著提高了大型语言模型（LLMs）在复杂推理任务上的表现。 |
| [^13] | [Tree-Based Hard Attention with Self-Motivation for Large Language Models](https://arxiv.org/abs/2402.08874) | 提出了一种名为TEAROOM的框架，该框架采用基于树状硬注意力和自我激励的机制，用于处理大型语言模型中的分层文本输入，并通过提示机制使模型能够选择性地关注与特定任务相关的叶子节点。 |
| [^14] | [A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925) | 在LLMs的背景下，本文综合研究了各种解码方法的性能、鲁棒性和解码速度，并发现解码方法的性能与任务相关，受到对齐、模型大小和量化等因素影响；某些方法可以通过大量超参数调整达到更好的性能，但需要权衡取舍。 |
| [^15] | [Limits of Transformer Language Models on Algorithmic Learning](https://arxiv.org/abs/2402.05785) | Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。 |
| [^16] | [Partially Recentralization Softmax Loss for Vision-Language Models Robustness](https://arxiv.org/abs/2402.03627) | 本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。 |
| [^17] | [SWAG: Storytelling With Action Guidance](https://arxiv.org/abs/2402.03483) | SWAG是一种新的故事讲述方法，通过将故事写作简化为搜索问题，使用两个模型的反馈循环来指导故事的发展方向。在GPT-4和人工评估中，SWAG表现出显著的优势，并且使用仅开源模型的SWAG流程超过了GPT-3.5-Turbo。 |
| [^18] | [Preference Poisoning Attacks on Reward Model Learning](https://arxiv.org/abs/2402.01920) | 对于从偏好比较中学习奖励模型的方法存在偏好污染攻击的漏洞，攻击者可以通过翻转少量偏好比较来对目标结果进行操纵。我们提出了两类算法方法，并证明了这些攻击在实施恶意行为方面的有效性。 |
| [^19] | [Recent Advances in Hate Speech Moderation: Multimodality and the Role of Large Models.](http://arxiv.org/abs/2401.16727) | 这项综合调查总结了最近在仇恨言论审核方面的进展，重点介绍了大型语言模型和大型多模态模型的作用。研究发现了文本、视觉和听觉元素在传播仇恨言论中的微妙相互作用，并强调了大型模型对审核能力的重新定义。同时，研究还指出了在少数语言和文化背景下的研究差距和处理低资源环境的需求。 |
| [^20] | [Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models.](http://arxiv.org/abs/2401.10440) | 本论文提出了一种称为X-ELM的跨语言专家语言模型，通过独立训练语言模型的子集来减轻多语言竞争，为多语言处理带来提升。实验表明，X-ELM在各种语言上优于联合训练的多语言模型，并且可以适应新语言的迭代添加。 |
| [^21] | [Axis Tour: Word Tour Determines the Order of Axes in ICA-transformed Embeddings.](http://arxiv.org/abs/2401.06112) | 本研究提出了一种新的方法，Axis Tour，用于确定ICA转换嵌入中轴的顺序，并通过最大化语义连续性来提高词嵌入空间的清晰度。实验证明，Axis Tour构建的低维嵌入比PCA和ICA更好。 |
| [^22] | [Conversational Financial Information Retrieval Model (ConFIRM).](http://arxiv.org/abs/2310.13001) | ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。 |
| [^23] | [Revisiting Supertagging for HPSG.](http://arxiv.org/abs/2309.07590) | 重新审视基于HPSG的Supertagging，在高质量注释的树库和多样化的测试数据集上，通过使用SVM和神经网络方法，取得了较高准确率。相关数据集已整理为标记分类形式，可为现代HPSG解析器提供帮助。 |
| [^24] | [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs.](http://arxiv.org/abs/2309.05516) | 本文提出一种名为SignRound的优化权重舍入的方法，通过使用有符号梯度进行轻量级分块调整，解决了大型语言模型(LLMs)的量化挑战。 |
| [^25] | [HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus.](http://arxiv.org/abs/2309.02731) | 本文介绍了HC3 Plus，一个语义不变的人类ChatGPT对比语料库。与以往的工作相比，该语料库考虑了更多类型的任务，包括语义不变任务。研究发现，在语义不变任务中检测模型生成的文本更加困难。通过大量任务指令微调和Tk-instruct，建立了一个更强大的模型。 |
| [^26] | [Speech Separation based on Contrastive Learning and Deep Modularization.](http://arxiv.org/abs/2305.10652) | 本文提出了一种基于对比学习和深度模块化的完全无监督语音分离方法，解决了有监督学习中存在的排列问题、说话人数量不匹配的问题和高质量标记数据的依赖问题。 |
| [^27] | [Geolocation Predicting of Tweets Using BERT-Based Models.](http://arxiv.org/abs/2303.07865) | 该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。 |

# 详细

[^1]: 通过诱导忠实性校准大型语言模型的置信度

    Calibrating the Confidence of Large Language Models by Eliciting Fidelity

    [https://arxiv.org/abs/2404.02655](https://arxiv.org/abs/2404.02655)

    本文通过将语言模型的置信度分解为问题的不确定性和对答案的忠实性，提出了一种估计语言模型置信度的即插即用方法，经实验证明具有良好的校准性能。

    

    使用RLHF等技术优化的大型语言模型已经取得了良好的对齐，既有帮助性又无害。然而，在对齐之后，这些语言模型经常表现出过度自信，表达的置信度并不准确地与其正确率校准。在本文中，我们将语言模型的置信度分解为关于问题的\textit{不确定性}和对语言模型生成的答案的\textit{忠实性}。然后，我们提出了一种即插即用的方法来估计语言模型的置信度。通过在四个MCQA数据集上对6个RLHF-LMs进行实验，我们的方法表现出很好的校准性能。此外，我们提出了两个新颖的度量标准，IPR和CE，来评估模型的校准性，并对\textit{真正校准的置信度}进行了详细讨论。我们的方法可以作为一个强有力的基线，希望这项工作能提供一些见解。

    arXiv:2404.02655v1 Announce Type: new  Abstract: Large language models optimized with techniques like RLHF have achieved good alignment in being helpful and harmless. However, post-alignment, these language models often exhibit overconfidence, where the expressed confidence does not accurately calibrate with their correctness rate. In this paper, we decompose the language model confidence into the \textit{Uncertainty} about the question and the \textit{Fidelity} to the answer generated by language models. Then, we propose a plug-and-play method to estimate the confidence of language models. Our method has shown good calibration performance by conducting experiments with 6 RLHF-LMs on four MCQA datasets. Moreover, we propose two novel metrics, IPR and CE, to evaluate the calibration of the model, and we have conducted a detailed discussion on \textit{Truly Well-Calibrated Confidence}. Our method could serve as a strong baseline, and we hope that this work will provide some insights into
    
[^2]: UniMEEC:走向统一的多模情绪识别与情绪因果

    UniMEEC: Towards Unified Multimodal Emotion Recognition and Emotion Cause

    [https://arxiv.org/abs/2404.00403](https://arxiv.org/abs/2404.00403)

    UniMEEC提出了一个统一的多模情绪识别和情绪-原因分析框架，将MERC和MECPE重新定义为两个掩码预测问题，以增强情绪和原因之间的交互作用。

    

    最近，对话中的多模情绪识别（MERC）和多模情绪-原因对提取（MECPE）引起了广泛关注。情绪是情感或感受的表达；对特定事件、想法或情况的响应被称为情绪原因。它们如同一枚硬币的两面，共同描述了人类行为和意图。然而，大多数现有作品将MERC和MECPE视为独立任务，这可能导致在整合情绪和原因到现实应用中存在潜在挑战。在本文中，我们提出了一个统一的多模情绪识别和情绪-原因分析框架（UniMEEC），以探索情绪和情绪原因之间的因果关系和互补性。具体来说，UniMEEC将MERC和MECPE任务重新定义为两个掩码预测问题，增强了情绪和原因之间的交互作用。与此同时，UniMEEC在各模态之间共享迅速学习以促进

    arXiv:2404.00403v1 Announce Type: new  Abstract: Multimodal emotion recognition in conversation (MERC) and multimodal emotion-cause pair extraction (MECPE) has recently garnered significant attention. Emotions are the expression of affect or feelings; responses to specific events, thoughts, or situations are known as emotion causes. Both are like two sides of a coin, collectively describing human behaviors and intents. However, most existing works treat MERC and MECPE as separate tasks, which may result in potential challenges in integrating emotion and cause in real-world applications. In this paper, we propose a Unified Multimodal Emotion recognition and Emotion-Cause analysis framework (UniMEEC) to explore the causality and complementarity between emotion and emotion cause. Concretely, UniMEEC reformulates the MERC and MECPE tasks as two mask prediction problems, enhancing the interaction between emotion and cause. Meanwhile, UniMEEC shares the prompt learning among modalities for p
    
[^3]: 语言模型中的紧急沟通和学习压力：语言进化视角

    Emergent communication and learning pressures in language models: a language evolution perspective

    [https://arxiv.org/abs/2403.14427](https://arxiv.org/abs/2403.14427)

    从语言进化的角度研究了新兴沟通文献，发现其在设计和调整模型以恢复自然语言中初始缺失的语言现象方面表现优秀，揭示了关键压力促使恢复最初不显现的人类模式。

    

    语言模型和人类是两种学习系统。发现或促进二者之间的共同点可能会在我们理解语言的习得和演化方面取得重大突破。许多语言进化理论在很大程度上依赖于学习偏好和学习压力。然而，由于学习压力存在着重大差异，对于人类和机器之间的相似性是否足以启发洞见并值得与人类参与者一起进行测试是值得怀疑的。本文从语言进化的角度审视了新兴沟通文献，这是多智能体强化学习的一个子领域。我们发现，新兴沟通文献在设计和调整模型以恢复自然语言的最初不显现的语言现象方面有杰出表现。根据对文献的简要回顾，我们确定了一些在新兴沟通中恢复最初不显现的人类模式的关键压力。

    arXiv:2403.14427v1 Announce Type: new  Abstract: Language models and humans are two types of learning systems. Finding or facilitating commonalities could enable major breakthroughs in our understanding of the acquisition and evolution of language. Many theories of language evolution rely heavily on learning biases and learning pressures. Yet due to substantial differences in learning pressures, it is questionable whether the similarity between humans and machines is sufficient for insights to carry over and to be worth testing with human participants. Here, we review the emergent communication literature, a subfield of multi-agent reinforcement learning, from a language evolution perspective. We find that the emergent communication literature excels at designing and adapting models to recover initially absent linguistic phenomena of natural languages. Based on a short literature review, we identify key pressures that have recovered initially absent human patterns in emergent communica
    
[^4]: 一个统一的模型编辑框架

    A Unified Framework for Model Editing

    [https://arxiv.org/abs/2403.14236](https://arxiv.org/abs/2403.14236)

    这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。

    

    模型编辑是一个不断发展的领域，专注于更新模型中嵌入的知识。在各种方法中，ROME和MEMIT作为主要的“定位和编辑”模型编辑技术脱颖而出。而MEMIT可以批量编辑记忆，ROME则一次只能改变一个事实。本文引入了一个统一的框架，将ROME和MEMIT纳入一个单一的概念框架，优化同一目标，我们称之为“保存-记忆”目标。该目标旨在在记忆新事实信息的同时保留某些选定向量的表示。具体来说，ROME使用等式约束优化此目标，而MEMIT采用更灵活的最小二乘约束。除了批量编辑外，MEMIT还可以在多个层面编辑模型。我们将编辑的分布从多个层面分开，区别于优化目标。

    arXiv:2403.14236v1 Announce Type: cross  Abstract: Model editing is a growing area focused on updating the knowledge embedded within models. Among the various methodologies, ROME and MEMIT stand out as leading "locate-and-edit" model editing techniques. While MEMIT enables batched editing of memories, ROME is limited to changing one fact at a time. This paper introduces a unifying framework that brings ROME and MEMIT under a single conceptual umbrella, optimizing for the same goal, which we call the "preservation-memorization" objective. This objective aims to preserve the representations of certain selected vectors while memorizing the representations of new factual information. Specifically, ROME optimizes this objective using an equality constraint, whereas MEMIT employs a more flexible least-square constraint. In addition to making batched edits, MEMIT also edits the model at multiple layers. We disentangle the distribution of edits to multiple layers from the optimization objectiv
    
[^5]: 数据价值评估对视觉指导调整的影响

    Less is More: Data Value Estimation for Visual Instruction Tuning

    [https://arxiv.org/abs/2403.09559](https://arxiv.org/abs/2403.09559)

    视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。

    

    视觉指导调整是构建多模式大语言模型（MLLMs）的关键，大大提高了大语言模型（LLMs）在视觉场景中的推理能力。然而，现有的MLLMs主要依赖于多个高度多样化的视觉指导数据集的混合训练（甚至超过一百万条指导），这可能引入数据冗余。为了调查这个问题，我们进行了一系列实证研究，揭示了视觉指导数据集内存在显著冗余，并显示大大减少几个指导数据集的数量甚至不会影响性能。根据研究结果，我们提出了一种新的数据选择方法TIVE，以消除视觉指导数据中的冗余。TIVE首先根据计算的梯度估计视觉指导的任务级和实例级价值。然后，根据估计的价值，TIVE确定了任务级和实例级指导选择策略。

    arXiv:2403.09559v1 Announce Type: new  Abstract: Visual instruction tuning is the key to building multimodal large language models (MLLMs), which greatly improves the reasoning capabilities of large language models (LLMs) in vision scenario. However, existing MLLMs mostly rely on a mixture of multiple highly diverse visual instruction datasets for training (even more than a million instructions), which may introduce data redundancy. To investigate this issue, we conduct a series of empirical studies, which reveal a significant redundancy within the visual instruction datasets, and show that greatly reducing the amount of several instruction dataset even do not affect the performance. Based on the findings, we propose a new data selection approach TIVE, to eliminate redundancy within visual instruction data. TIVE first estimates the task-level and instance-level value of the visual instructions based on computed gradients. Then, according to the estimated values, TIVE determines the tas
    
[^6]: 大型语言模型是并行多语言学习者

    Large Language Models are Parallel Multilingual Learners

    [https://arxiv.org/abs/2403.09073](https://arxiv.org/abs/2403.09073)

    通过将输入翻译为多种语言，为大型语言模型提供多语言平行输入，显著增强了它们的理解能力，实验证明多语言输入可以超越传统学习方法，并发现了神经元激活的反直觉现象

    

    在这项研究中，我们揭示了多语言大型语言模型（LLMs）的上下文学习（ICL）能力：通过将输入翻译成多种语言，我们为LLMs提供了多语言平行输入（PiM），显著增强了它们的理解能力。为测试这种能力，我们设计了包括8个典型数据集、7种语言和8种最先进的多语言LLMs在内的大量实验证明结果显示，（1）整合更多语言可以帮助PiM进一步超越传统的ICL；（2）即使与基准性能低劣的翻译结合也是有帮助的。此外，通过检查LLMs中激活的神经元，我们发现了一个令人意外但有趣的现象。与常见观点相反，PiM并不会激活比单语输入更多的神经元来利用从多种语言学习到的知识，而实际上是抑制神经元并促进更精确的神经。

    arXiv:2403.09073v1 Announce Type: new  Abstract: In this study, we reveal an in-context learning (ICL) capability of multilingual large language models (LLMs): by translating the input to several languages, we provide Parallel Input in Multiple Languages (PiM) to LLMs, which significantly enhances their comprehension abilities. To test this capability, we design extensive experiments encompassing 8 typical datasets, 7 languages and 8 state-of-the-art multilingual LLMs. Experimental results show that (1) incorporating more languages help PiM surpass the conventional ICL further; (2) even combining with the translations that are inferior to baseline performance can also help. Moreover, by examining the activated neurons in LLMs, we discover a counterintuitive but interesting phenomenon. Contrary to the common thought that PiM would activate more neurons than monolingual input to leverage knowledge learned from diverse languages, PiM actually inhibits neurons and promotes more precise neu
    
[^7]: 重建ROME: 解决顺序模型编辑过程中的模型崩溃问题

    Rebuilding ROME : Resolving Model Collapse during Sequential Model Editing

    [https://arxiv.org/abs/2403.07175](https://arxiv.org/abs/2403.07175)

    本文重建了ROME，提供了更稳定的r-ROME实现，解决了顺序模型编辑过程中的模型崩溃问题。

    

    最近关于使用Rank-One Model Editing (ROME)进行模型编辑的研究表明，有一些事实表明该算法无法进行编辑而不破坏模型。这些编辑以前被称为禁用编辑。这些禁用编辑会导致立即模型崩溃，并限制了ROME用于顺序编辑的使用。在本文中，我们做出了两个主要贡献。首先，我们展示了在使用CounterFact数据集进行编辑时，ROME仅在此时发生模型崩溃，并在使用zsRE数据集时不会发生。其次，我们发现禁用编辑是ROME原始实现的产物。通过本文，我们提供了一个更稳定的实现ROME，我们将其称为r-ROME，并展示我们在使用ROME进行大规模顺序编辑时不再观察到模型崩溃。

    arXiv:2403.07175v1 Announce Type: cross  Abstract: Recent work on model editing using Rank-One Model Editing (ROME), a popular model editing method, has shown that there are certain facts that the algorithm is unable to edit without breaking the model. Such edits have previously been called disabling edits. These disabling edits cause immediate model collapse and limits the use of ROME for sequential editing. In this paper, we make two main contributions. Firstly, we show that model collapse with ROME only happens when making edits using the CounterFact dataset and does not happen when using the zsRE dataset. Secondly, we find that disabling edits are an artifact of the original implementation of ROME. With this paper, we provide a more stable implementation ROME, which we call r-ROME and show that we no longer observe model collapse when making large scale sequential edits with ROME.
    
[^8]: 面向证据的事实摘要化用于知识增强的零-shot问答

    Evidence-Focused Fact Summarization for Knowledge-Augmented Zero-Shot Question Answering

    [https://arxiv.org/abs/2403.02966](https://arxiv.org/abs/2403.02966)

    提出了一种面向证据的事实摘要化框架EFSum，用于增强LLMs的零-shot QA性能，并确保摘要的有益性和忠实性。

    

    最近的研究探讨了利用知识图谱（KGs）来增强大语言模型（LLMs）的问答（QA）性能，然而结构化的KG形式化仍然具有挑战性。现有方法，如三元组形式或三元组事实的自由文本转换，遇到了一些问题。这些问题包括由于重复实体或关系而导致的证据密度降低，以及由于无法强调关键证据而导致的证据清晰度降低。为解决这些问题，我们提出了EFSum，一个面向证据的事实摘要化框架，用于通过知识增强的LLMs增强QA。我们通过蒸馏和偏好对齐来优化一个开源的LLM作为事实摘要器。我们的广泛实验证明，EFSum提高了LLM的零-shot QA性能，并且可以确保摘要的同时有益和忠实。

    arXiv:2403.02966v1 Announce Type: cross  Abstract: Recent studies have investigated utilizing Knowledge Graphs (KGs) to enhance Quesetion Answering (QA) performance of Large Language Models (LLMs), yet structured KG verbalization remains challengin. Existing methods, such as triple-form or free-form textual conversion of triple-form facts, encounter several issues. These include reduced evidence density due to duplicated entities or relationships, and reduced evidence clarity due to an inability to emphasize crucial evidence. To address these issues, we propose EFSum, an Evidence-focused Fact Summarization framework for enhanced QA with knowledge-augmented LLMs. We optimize an open-source LLM as a fact summarizer through distillation and preference alignment. Our extensive experiments show that EFSum improves LLM's zero-shot QA performance, and it is possible to ensure both the helpfulness and faithfulness of the summary.
    
[^9]: 三种语言中的少样本临床实体识别：掩盖语言模型胜过LLM提示

    Few shot clinical entity recognition in three languages: Masked language models outperform LLM prompting

    [https://arxiv.org/abs/2402.12801](https://arxiv.org/abs/2402.12801)

    掩盖语言模型在三种语言中的少样本临床实体识别中表现优异，胜过LLM提示方法

    

    大型语言模型正成为许多自然语言处理任务的首选解决方案，包括在专业领域中，人们期望它们的少样本能力能在资源匮乏的情况下获得高性能。本文旨在评估大型语言模型在多种语言中进行少样本临床实体识别的性能。我们使用8个领域内（临床）和6个领域外的黄金标准语料库，评估英语、法语和西班牙语中的命名实体识别。我们评估了10个自回归语言模型的性能，这些模型使用提示，并使用16个用于文本编码的掩盖语言模型作为BiLSTM-CRF监督标注器。我们通过限制可用的带标注数据量为100个句子来创建一个少样本设置。我们的实验表明，尽管更大的基于提示的模型往往在临床领域之外的命名实体识别中实现了有竞争力的F-measure，但这种性能水平并未。。。

    arXiv:2402.12801v1 Announce Type: new  Abstract: Large Language Models are becoming the go-to solution for many natural language processing tasks, including in specialized domains where their few-shot capacities are expected to yield high performance in low-resource settings. Herein, we aim to assess the performance of Large Language Models for few shot clinical entity recognition in multiple languages. We evaluate named entity recognition in English, French and Spanish using 8 in-domain (clinical) and 6 out-domain gold standard corpora. We assess the performance of 10 auto-regressive language models using prompting and 16 masked language models used for text encoding in a biLSTM-CRF supervised tagger. We create a few-shot set-up by limiting the amount of annotated data available to 100 sentences. Our experiments show that although larger prompt-based models tend to achieve competitive F-measure for named entity recognition outside the clinical domain, this level of performance does no
    
[^10]: FormulaQA：一个基于公式的数值推理问题问答数据集

    FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning

    [https://arxiv.org/abs/2402.12692](https://arxiv.org/abs/2402.12692)

    FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。

    

    应用公式是人类在解决数值推理问题时的基本能力。然而，现有的数值推理数据集很少明确指出推理步骤中使用的公式。为了弥补这一差距，我们提出了一个基于初中物理考试的公式驱动数值推理问题问答数据集FormulaQA。我们还使用大小从7B到超过100B参数的LLMs进行了零样本和少样本思维链方法的评估，并探索了在提供外部公式数据库时使用检索增强型LLMs的方法。我们还对大小不超过2B的较小模型进行了微调。我们的实证研究强调了当应用于我们复杂、基于公式的FormulaQA时，现有模型在改进方面具有显著潜力。

    arXiv:2402.12692v1 Announce Type: new  Abstract: The application of formulas is a fundamental ability of humans when addressing numerical reasoning problems. However, existing numerical reasoning datasets seldom explicitly indicate the formulas employed during the reasoning steps. To bridge this gap, we propose a question answering dataset for formula-based numerical reasoning called FormulaQA, from junior high school physics examinations. We further conduct evaluations on LLMs with size ranging from 7B to over 100B parameters utilizing zero-shot and few-shot chain-of-thoughts methods and we explored the approach of using retrieval-augmented LLMs when providing an external formula database. We also fine-tune on smaller models with size not exceeding 2B. Our empirical findings underscore the significant potential for improvement in existing models when applied to our complex, formula-driven FormulaQA.
    
[^11]: 使用LLMs自动评估心理健康咨询

    Automatic Evaluation for Mental Health Counseling using LLMs

    [https://arxiv.org/abs/2402.11958](https://arxiv.org/abs/2402.11958)

    使用LLMs自动评估心理咨询对话中的工作联盟，结果显示与人工评估高度一致，并提供宝贵见解。

    

    高质量的心理咨询对全球心理健康至关重要，及时评估对确保其有效性至关重要。然而，为每个咨询会话获取专业评估既昂贵又具挑战性。依赖自我或第三方手动报告来评估咨询质量的现有方法存在主观偏见和耗时的局限性。为了解决上述挑战，本文提出了一种创新高效的自动评估方法，利用大型语言模型(LLMs)来评估咨询对话中的工作联盟。我们收集了一个全面的咨询数据集，并基于治疗关系理论进行了多方评估。我们基于LLMs的评估结合我们的指南，与人工评估高度一致，并为咨询脚本提供了宝贵的见解。这突显了LLMs作为监督的潜力。

    arXiv:2402.11958v1 Announce Type: new  Abstract: High-quality psychological counseling is crucial for mental health worldwide, and timely evaluation is vital for ensuring its effectiveness. However, obtaining professional evaluation for each counseling session is expensive and challenging. Existing methods that rely on self or third-party manual reports to assess the quality of counseling suffer from subjective biases and limitations of time-consuming.   To address above challenges, this paper proposes an innovative and efficient automatic approach using large language models (LLMs) to evaluate the working alliance in counseling conversations. We collected a comprehensive counseling dataset and conducted multiple third-party evaluations based on therapeutic relationship theory. Our LLM-based evaluation, combined with our guidelines, shows high agreement with human evaluations and provides valuable insights into counseling scripts. This highlights the potential of LLMs as supervisory to
    
[^12]: 分隔符是否可以提高思维链提示的效果？

    Can Separators Improve Chain-of-Thought Prompting?

    [https://arxiv.org/abs/2402.10645](https://arxiv.org/abs/2402.10645)

    分隔符的引入在思维链提示中显著提高了大型语言模型（LLMs）在复杂推理任务上的表现。

    

    Chain-of-thought (CoT) prompting是一种简单有效的方法，用于提高大型语言模型（LLMs）的推理能力。CoT的基本理念是通过将示例放在输入提示中，让LLMs逐步拆解他们的思维过程。然而，CoT提示的密集结构可能导致LLMs的认知负荷过重。受人类认知启发，我们引入了CoT-Sep，一种新颖的方法，在CoT提示中每个示例的末尾策略性地应用分隔符。这些分隔符旨在帮助LLMs在推理过程中更好地理解他们的思维过程。结果表明，与不使用分隔符的普通CoT相比，CoT-Sep显著提高了LLMs在复杂推理任务（如GSM-8K、AQuA、CSQA）上的表现。我们还研究了不同类型和位置的分隔符对多个LLMs（包括GPT-3.5-Turbo、GPT-4和LLaMA-27）的影响。

    arXiv:2402.10645v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) prompting is a simple and effective method for improving the reasoning capabilities of Large language models (LLMs). The basic idea of CoT is to let LLMs break down their thought processes step-by-step by putting exemplars in the input prompt. However, the densely structured prompt exemplars of CoT may cause the cognitive overload of LLMs. Inspired by human cognition, we introduce CoT-Sep, a novel method that strategically employs separators at the end of each exemplar in CoT prompting. These separators are designed to help the LLMs understand their thought processes better while reasoning. It turns out that CoT-Sep significantly improves the LLMs' performances on complex reasoning tasks (e.g., GSM-8K, AQuA, CSQA), compared with the vanilla CoT, which does not use separators. We also study the effects of the type and the location of separators tested on multiple LLMs, including GPT-3.5-Turbo, GPT-4, and LLaMA-2 7
    
[^13]: 基于树状硬注意力和自我激励的大型语言模型

    Tree-Based Hard Attention with Self-Motivation for Large Language Models

    [https://arxiv.org/abs/2402.08874](https://arxiv.org/abs/2402.08874)

    提出了一种名为TEAROOM的框架，该框架采用基于树状硬注意力和自我激励的机制，用于处理大型语言模型中的分层文本输入，并通过提示机制使模型能够选择性地关注与特定任务相关的叶子节点。

    

    虽然大型语言模型在理解和生成纯文本方面表现出色，但它们并没有专门设计来处理分层文本结构。从它们的自然语言回复中提取任务所需的属性通常需要额外的处理步骤。事实上，选择性地理解大规模文本的层次结构对于理解其实质至关重要。通过提示将LLM与特定任务的分类或回归值更紧密地对齐也仍然具有挑战性。为此，我们提出了一种新颖的框架，称为Tree-Based Hard Attention with Self-Motivation for Large Language Models（TEAROOM）。TEAROOM将树状硬注意力机制纳入LLM中，以处理分层结构的文本输入。通过利用提示机制，它使冻结的LLM能够选择性地关注与根节点相关的叶子节点，生成一个定制的符号表示。

    arXiv:2402.08874v1 Announce Type: new Abstract: While large language models (LLMs) excel at understanding and generating plain text, they are not specifically tailored to handle hierarchical text structures. Extracting the task-desired property from their natural language responses typically necessitates additional processing steps. In fact, selectively comprehending the hierarchical structure of large-scale text is pivotal to understanding its substance. Aligning LLMs more closely with the classification or regression values of specific task through prompting also remains challenging. To this end, we propose a novel framework called Tree-Based Hard Attention with Self-Motivation for Large Language Models (TEAROOM). TEAROOM incorporates a tree-based hard attention mechanism for LLMs to process hierarchically structured text inputs. By leveraging prompting, it enables a frozen LLM to selectively focus on relevant leaves in relation to the root, generating a tailored symbolic representat
    
[^14]: LLM时代解码方法的综合研究

    A Thorough Examination of Decoding Methods in the Era of LLMs

    [https://arxiv.org/abs/2402.06925](https://arxiv.org/abs/2402.06925)

    在LLMs的背景下，本文综合研究了各种解码方法的性能、鲁棒性和解码速度，并发现解码方法的性能与任务相关，受到对齐、模型大小和量化等因素影响；某些方法可以通过大量超参数调整达到更好的性能，但需要权衡取舍。

    

    解码方法在将语言模型从下一个标记预测器转换为实际任务解决器中起着不可或缺的作用。以往关于解码方法的研究主要集中在任务特定模型上，可能不适用于当前通用型大型语言模型(LLMs)的时代。此外，最近解码策略的涌入进一步复杂了这个领域。本文在LLMs的背景下，对各种解码方法进行了全面而多方位的分析，评估了它们在各种任务、模型和部署环境中的性能、对超参数变化的鲁棒性以及解码速度。我们的研究结果表明，解码方法的性能明显与任务相关，并受到对齐、模型大小和量化等因素的影响。有趣的是，敏感性分析揭示了某些方法在需要进行大量超参数调整的前提下能够实现更好的性能，突出了在达到最佳性能之间的权衡关系。

    Decoding methods play an indispensable role in converting language models from next-token predictors into practical task solvers. Prior research on decoding methods, primarily focusing on task-specific models, may not extend to the current era of general-purpose large language models (LLMs). Moreover, the recent influx of decoding strategies has further complicated this landscape. This paper provides a comprehensive and multifaceted analysis of various decoding methods within the context of LLMs, evaluating their performance, robustness to hyperparameter changes, and decoding speeds across a wide range of tasks, models, and deployment environments. Our findings reveal that decoding method performance is notably task-dependent and influenced by factors such as alignment, model size, and quantization. Intriguingly, sensitivity analysis exposes that certain methods achieve superior performance at the cost of extensive hyperparameter tuning, highlighting the trade-off between attaining opt
    
[^15]: Transformer语言模型在算法学习上的限制

    Limits of Transformer Language Models on Algorithmic Learning

    [https://arxiv.org/abs/2402.05785](https://arxiv.org/abs/2402.05785)

    Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。

    

    我们分析了Transformer语言模型在学习离散算法方面的能力。为此，我们引入了两个要求组合多个离散子任务的新任务。我们通过从头开始训练LLaMA模型和在GPT-4和Gemini上提示来衡量学习学习原语的组合。我们观察到，目前最先进的Transformer语言模型的组合能力非常有限，并且在样本规模方面比为新的算法组合重新学习所有子任务效果更差。我们还提出了一个复杂性理论的定理，证明了记忆前馈模型上的梯度下降可以指数级地浪费数据。

    We analyze the capabilities of Transformer language models on learning discrete algorithms. To this end, we introduce two new tasks demanding the composition of several discrete sub-tasks. On both training LLaMA models from scratch and prompting on GPT-4 and Gemini we measure learning compositions of learned primitives. We observe that the compositional capabilities of state-of-the-art Transformer language models are very limited and sample-wise scale worse than relearning all sub-tasks for a new algorithmic composition. We also present a theorem in complexity theory, showing that gradient descent on memorizing feedforward models can be exponentially data inefficient.
    
[^16]: 近似的中心化softmax损失用于视觉-语言模型的鲁棒性

    Partially Recentralization Softmax Loss for Vision-Language Models Robustness

    [https://arxiv.org/abs/2402.03627](https://arxiv.org/abs/2402.03627)

    本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。

    

    随着大型语言模型在自然语言处理任务中的突破，多模态技术变得非常流行。然而，已经证明多模态自然语言处理模型容易受到对抗攻击，即模型的输出可以通过对输入进行微小扰动而发生巨大变化。虽然计算机视觉和自然语言处理模型中已经提出了几种防御技术，但对多模态模型的鲁棒性还没有进行充分探索。在本文中，我们研究了通过修改预训练多模态模型的损失函数，通过限制前K个softmax输出来提供的对抗鲁棒性。基于评估和评分，我们的实验结果显示，在经过微调后，预训练模型的对抗鲁棒性可以显着提高，对抗常见的攻击有效。进一步的研究应该探索这类损失函数的输出多样性、泛化能力以及鲁棒性和性能之间的平衡。我们的代码将在之后提供。

    As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after th
    
[^17]: SWAG: 带有行动指导的故事讲述

    SWAG: Storytelling With Action Guidance

    [https://arxiv.org/abs/2402.03483](https://arxiv.org/abs/2402.03483)

    SWAG是一种新的故事讲述方法，通过将故事写作简化为搜索问题，使用两个模型的反馈循环来指导故事的发展方向。在GPT-4和人工评估中，SWAG表现出显著的优势，并且使用仅开源模型的SWAG流程超过了GPT-3.5-Turbo。

    

    自动长篇故事生成通常使用长上下文大语言模型（LLMs）进行一次性创建，它可以产生连贯但不一定引人入胜的内容。我们引入了带有行动指导的故事讲述（SWAG）的新方法。我们的方法通过两个模型的反馈循环将故事写作简化为一个搜索问题：一个LLM生成故事内容，另一个辅助LLM用于选择下一个最佳的“行动”，以引导故事的未来发展方向。我们的结果表明，当使用GPT-4和人工评估进行评估时，SWAG能够显著优于以往的端到端故事生成技术，并且我们只使用开源模型的SWAG流程超越了GPT-3.5-Turbo。

    Automated long-form story generation typically employs long-context large language models (LLMs) for one-shot creation, which can produce cohesive but not necessarily engaging content. We introduce Storytelling With Action Guidance (SWAG), a novel approach to storytelling with LLMs. Our approach reduces story writing to a search problem through a two-model feedback loop: one LLM generates story content, and another auxiliary LLM is used to choose the next best "action" to steer the story's future direction. Our results show that SWAG can substantially outperform previous end-to-end story generation techniques when evaluated by GPT-4 and through human evaluation, and our SWAG pipeline using only open-source models surpasses GPT-3.5-Turbo.
    
[^18]: 对奖励模型学习的偏好污染攻击

    Preference Poisoning Attacks on Reward Model Learning

    [https://arxiv.org/abs/2402.01920](https://arxiv.org/abs/2402.01920)

    对于从偏好比较中学习奖励模型的方法存在偏好污染攻击的漏洞，攻击者可以通过翻转少量偏好比较来对目标结果进行操纵。我们提出了两类算法方法，并证明了这些攻击在实施恶意行为方面的有效性。

    

    从两两比较中学习效用或奖励模型是许多应用领域的基础组成部分。这些方法从本质上需要从人们那里收集偏好信息，而反馈通常是匿名提供的。由于偏好是主观的，没有可以比较的黄金标准；然而，对偏好学习的高影响系统的依赖性为恶意行为者倾向于扭曲以达到其目的而采集的数据创造了强烈的动机。我们通过考虑一种威胁模型系统地调查了这种漏洞的性质和程度，其中攻击者可以翻转少量偏好比较，以促进或贬低目标结果。首先，我们提出了两类用于这些攻击的算法方法：基于原则的梯度框架和几种变种的按距离排名的方法。接下来，我们展示了这两类最佳攻击在成功实施恶意行为方面的效果。

    Learning utility, or reward, models from pairwise comparisons is a fundamental component in a number of application domains. These approaches inherently entail collecting preference information from people, with feedback often provided anonymously. Since preferences are subjective, there is no gold standard to compare against; yet, reliance of high-impact systems on preference learning creates a strong motivation for malicious actors to skew data collected in this fashion to their ends. We investigate the nature and extent of this vulnerability systematically by considering a threat model in which an attacker can flip a small subset of preference comparisons with the goal of either promoting or demoting a target outcome. First, we propose two classes of algorithmic approaches for these attacks: a principled gradient-based framework, and several variants of rank-by-distance methods. Next, we demonstrate the efficacy of best attacks in both these classes in successfully achieving malicio
    
[^19]: 最近在仇恨言论审核方面的进展：多模态和大型模型的作用

    Recent Advances in Hate Speech Moderation: Multimodality and the Role of Large Models. (arXiv:2401.16727v1 [cs.CL])

    [http://arxiv.org/abs/2401.16727](http://arxiv.org/abs/2401.16727)

    这项综合调查总结了最近在仇恨言论审核方面的进展，重点介绍了大型语言模型和大型多模态模型的作用。研究发现了文本、视觉和听觉元素在传播仇恨言论中的微妙相互作用，并强调了大型模型对审核能力的重新定义。同时，研究还指出了在少数语言和文化背景下的研究差距和处理低资源环境的需求。

    

    在网络交流的不断发展中，审核仇恨言论（HS）面临着复杂的挑战，这是由数字内容的多模态特性所带来的。这项综合调查深入研究了HS审核的最新进展，着重介绍了大型语言模型（LLMs）和大型多模态模型（LMMs）的崛起角色。我们的研究从对当前文献的全面分析开始，揭示了文本、视觉和听觉元素在传播HS中的微妙相互作用。我们发现了一个明显的趋势，即将这些模态整合在一起，主要是因为HS的传播具有复杂性和微妙性。对于由LLMs和LMMs带来的进展，我们特别强调了其对检测和审核能力边界的重新定义。我们确定了研究中存在的现有差距，特别是在少数语言和文化的背景下，以及在处理低资源环境中需要解决方案的需求。

    In the evolving landscape of online communication, moderating hate speech (HS) presents an intricate challenge, compounded by the multimodal nature of digital content. This comprehensive survey delves into the recent strides in HS moderation, spotlighting the burgeoning role of large language models (LLMs) and large multimodal models (LMMs). Our exploration begins with a thorough analysis of current literature, revealing the nuanced interplay between textual, visual, and auditory elements in propagating HS. We uncover a notable trend towards integrating these modalities, primarily due to the complexity and subtlety with which HS is disseminated. A significant emphasis is placed on the advances facilitated by LLMs and LMMs, which have begun to redefine the boundaries of detection and moderation capabilities. We identify existing gaps in research, particularly in the context of underrepresented languages and cultures, and the need for solutions to handle low-resource settings. The survey
    
[^20]: 用跨语言专家语言模型突破多语言环境的难题

    Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models. (arXiv:2401.10440v1 [cs.CL])

    [http://arxiv.org/abs/2401.10440](http://arxiv.org/abs/2401.10440)

    本论文提出了一种称为X-ELM的跨语言专家语言模型，通过独立训练语言模型的子集来减轻多语言竞争，为多语言处理带来提升。实验表明，X-ELM在各种语言上优于联合训练的多语言模型，并且可以适应新语言的迭代添加。

    

    尽管多语言语言模型在非英语自然语言处理（NLP）中很受欢迎，但由于模型参数之间的跨语言竞争，它们往往表现不及单语语言模型。我们提出了跨语言专家语言模型（X-ELM），通过对多语言语料库的子集进行独立训练，来减轻这种竞争。这个过程使X-ELM针对不同语言进行专门训练，同时作为一个多语言集合保持有效。我们的实验表明，在给定相同计算预算的情况下，X-ELM在所有考虑的语言上优于联合训练的多语言模型，并且这些收益可以转移到下游任务中。X-ELM在性能改进方面提供了额外的好处：可以迭代地添加新的专家，适应新语言而不会产生灾难性的遗忘。此外，训练是异步进行的，减少了多语言训练的硬件要求，实现多语言建模的民主化。

    Despite their popularity in non-English NLP, multilingual language models often underperform monolingual ones due to inter-language competition for model parameters. We propose Cross-lingual Expert Language Models (X-ELM), which mitigate this competition by independently training language models on subsets of the multilingual corpus. This process specializes X-ELMs to different languages while remaining effective as a multilingual ensemble. Our experiments show that when given the same compute budget, X-ELM outperforms jointly trained multilingual models across all considered languages and that these gains transfer to downstream tasks. X-ELM provides additional benefits over performance improvements: new experts can be iteratively added, adapting X-ELM to new languages without catastrophic forgetting. Furthermore, training is asynchronous, reducing the hardware requirements for multilingual training and democratizing multilingual modeling.
    
[^21]: Axis Tour: Word Tour 确定ICA转换嵌入中轴的顺序

    Axis Tour: Word Tour Determines the Order of Axes in ICA-transformed Embeddings. (arXiv:2401.06112v1 [cs.CL])

    [http://arxiv.org/abs/2401.06112](http://arxiv.org/abs/2401.06112)

    本研究提出了一种新的方法，Axis Tour，用于确定ICA转换嵌入中轴的顺序，并通过最大化语义连续性来提高词嵌入空间的清晰度。实验证明，Axis Tour构建的低维嵌入比PCA和ICA更好。

    

    词嵌入是自然语言处理中最重要的组成部分之一，但解释高维嵌入仍然是一个具有挑战性的问题。为了解决这个问题，独立成分分析（ICA）被确定为有效的解决方案。ICA转换的词嵌入揭示了可解释的语义轴，但这些轴的顺序是任意的。在这项研究中，我们着重关注这个特性，并提出了一种新的方法，Axis Tour，它优化了轴的顺序。受到一维词嵌入方法Word Tour的启发，我们旨在通过最大化轴的语义连续性来提高词嵌入空间的清晰度。此外，我们通过在下游任务上的实验证明，与PCA和ICA相比，Axis Tour构建了更好的低维嵌入。

    Word embedding is one of the most important components in natural language processing, but interpreting high-dimensional embeddings remains a challenging problem. To address this problem, Independent Component Analysis (ICA) is identified as an effective solution. ICA-transformed word embeddings reveal interpretable semantic axes; however, the order of these axes are arbitrary. In this study, we focus on this property and propose a novel method, Axis Tour, which optimizes the order of the axes. Inspired by Word Tour, a one-dimensional word embedding method, we aim to improve the clarity of the word embedding space by maximizing the semantic continuity of the axes. Furthermore, we show through experiments on downstream tasks that Axis Tour constructs better low-dimensional embeddings compared to both PCA and ICA.
    
[^22]: 会话式金融信息检索模型（ConFIRM）

    Conversational Financial Information Retrieval Model (ConFIRM). (arXiv:2310.13001v1 [cs.IR])

    [http://arxiv.org/abs/2310.13001](http://arxiv.org/abs/2310.13001)

    ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。

    

    随着大型语言模型（LLM）的指数级增长，利用它们在金融等专门领域的新兴特性具有探索的价值。然而，金融等受监管领域具有独特的约束条件，需要具备针对该领域的优化框架。我们提出了ConFIRM，一种基于LLM的会话式金融信息检索模型，用于查询意图分类和知识库标记。ConFIRM包括两个模块：1）一种合成金融领域特定问答对的方法，以及2）评估参数高效的微调方法来进行查询分类任务。我们生成了一个包含4000多个样本的数据集，并在单独的测试集上评估了准确性。ConFIRM实现了超过90%的准确性，这对于符合监管要求至关重要。ConFIRM提供了一种数据高效的解决方案，用于提取金融对话系统的精确查询意图。

    With the exponential growth in large language models (LLMs), leveraging their emergent properties for specialized domains like finance merits exploration. However, regulated fields such as finance pose unique constraints, requiring domain-optimized frameworks. We present ConFIRM, an LLM-based conversational financial information retrieval model tailored for query intent classification and knowledge base labeling.  ConFIRM comprises two modules:  1) a method to synthesize finance domain-specific question-answer pairs, and  2) evaluation of parameter efficient fine-tuning approaches for the query classification task. We generate a dataset of over 4000 samples, assessing accuracy on a separate test set.  ConFIRM achieved over 90% accuracy, essential for regulatory compliance. ConFIRM provides a data-efficient solution to extract precise query intent for financial dialog systems.
    
[^23]: 重新审视基于HPSG的Supertagging

    Revisiting Supertagging for HPSG. (arXiv:2309.07590v1 [cs.CL])

    [http://arxiv.org/abs/2309.07590](http://arxiv.org/abs/2309.07590)

    重新审视基于HPSG的Supertagging，在高质量注释的树库和多样化的测试数据集上，通过使用SVM和神经网络方法，取得了较高准确率。相关数据集已整理为标记分类形式，可为现代HPSG解析器提供帮助。

    

    我们提出了基于HPSG树库训练的新型supertagger。这些树库基于一个成熟的语言学理论，具有高质量的注释，并且包含了丰富多样和具有挑战性的测试数据集，超出了通常的WSJ第23节和维基百科数据。之前的HPSG supertagging主要依赖于基于MaxEnt的模型。我们使用SVM和基于神经CRF和BERT的方法，并展示出SVM和神经supertagger相对于基准模型取得了显著更高的准确率。我们微调的BERT-based tagger在来自WSJ23的1000个句子上达到了97.26%的准确率，并在完全不同领域的"The Cathedral and the Bazaar"上达到了93.88%的准确率。因此，我们得出结论，将这些新的supertagger集成到现代HPSG解析器中是有意义的，并且我们也希望我们在这里使用的多样且难的数据集在该领域中获得更多的关注。我们贡献了重新格式化为标记分类的完整数据集。

    We present new supertaggers trained on HPSG-based treebanks. These treebanks feature high-quality annotation based on a well-developed linguistic theory and include diverse and challenging test datasets, beyond the usual WSJ section 23 and Wikipedia data. HPSG supertagging has previously relied on MaxEnt-based models. We use SVM and neural CRF- and BERT-based methods and show that both SVM and neural supertaggers achieve considerably higher accuracy compared to the baseline. Our fine-tuned BERT-based tagger achieves 97.26% accuracy on 1000 sentences from WSJ23 and 93.88% on the completely out-of-domain The Cathedral and the Bazaar (cb)). We conclude that it therefore makes sense to integrate these new supertaggers into modern HPSG parsers, and we also hope that the diverse and difficult datasets we used here will gain more popularity in the field. We contribute the complete dataset reformatted for token classification.
    
[^24]: 通过有符号梯度下降优化LLMs量化中的权重舍入

    Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs. (arXiv:2309.05516v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.05516](http://arxiv.org/abs/2309.05516)

    本文提出一种名为SignRound的优化权重舍入的方法，通过使用有符号梯度进行轻量级分块调整，解决了大型语言模型(LLMs)的量化挑战。

    

    大型语言模型(LLMs)在执行语言相关任务方面表现出了非凡的能力。然而，由于其巨大的内存和存储需求，它们的部署面临着重大挑战。为了解决这个问题，仅针对权重的量化，特别是3位和4位仅针对权重的量化，已经成为最可行的解决方案之一。随着位数的减少，量化网格变得更加宽泛，从而强调了上下舍入的重要性。尽管先前的研究表明，在某些情况下，通过添加扰动细调上下舍入可以提高准确性，但我们的研究受制于这些扰动的精确且有限的边界，只有改变舍入值的阈值才具有重要性。因此，我们提出了一种简洁高效的优化权重舍入任务的方法。我们的方法名为SignRound，它涉及使用有符号梯度的轻量级分块调整。

    Large Language Models (LLMs) have proven their exceptional capabilities in performing language-related tasks. However, their deployment poses significant challenges due to their considerable memory and storage requirements. In response to this issue, weight-only quantization, particularly 3 and 4-bit weight-only quantization, has emerged as one of the most viable solutions. As the number of bits decreases, the quantization grid broadens, thus emphasizing the importance of up and down rounding. While previous studies have demonstrated that fine-tuning up and down rounding with the addition of perturbations can enhance accuracy in some scenarios, our study is driven by the precise and limited boundary of these perturbations, where only the threshold for altering the rounding value is of significance. Consequently, we propose a concise and highly effective approach for optimizing the weight rounding task. Our method, named SignRound, involves lightweight block-wise tuning using signed gra
    
[^25]: HC3 Plus：一个语义不变的人类ChatGPT对比语料库

    HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus. (arXiv:2309.02731v1 [cs.CL])

    [http://arxiv.org/abs/2309.02731](http://arxiv.org/abs/2309.02731)

    本文介绍了HC3 Plus，一个语义不变的人类ChatGPT对比语料库。与以往的工作相比，该语料库考虑了更多类型的任务，包括语义不变任务。研究发现，在语义不变任务中检测模型生成的文本更加困难。通过大量任务指令微调和Tk-instruct，建立了一个更强大的模型。

    

    ChatGPT因其出色的性能而引起了人们的广泛关注，但人们对其潜在风险，尤其是对AI生成内容（AIGC）的检测越来越关注，这对未经训练的人类来说往往很难识别。目前用于检测ChatGPT生成文本的数据集主要集中在问答方面，但往往忽视了具有语义不变性的任务，如摘要、翻译和改写。我们的研究表明，在语义不变任务上检测模型生成的文本更加困难。为了填补这一空白，我们引入了一个更广泛、更全面的数据集，考虑了比以前的工作更多类型的任务，包括语义不变任务。此外，经过大量任务指令微调的模型表现出很强的性能。基于以前的成功，我们进一步指导微调了Tk-instruct，并构建了一个更强大的模型。

    ChatGPT has gained significant interest due to its impressive performance, but people are increasingly concerned about its potential risks, particularly around the detection of AI-generated content (AIGC), which is often difficult for untrained humans to identify. Current datasets utilized for detecting ChatGPT-generated text primarily center around question-answering, yet they tend to disregard tasks that possess semantic-invariant properties, such as summarization, translation, and paraphrasing. Our primary studies demonstrate that detecting model-generated text on semantic-invariant tasks is more difficult. To fill this gap, we introduce a more extensive and comprehensive dataset that considers more types of tasks than previous work, including semantic-invariant tasks. In addition, the model after a large number of task instruction fine-tuning shows a strong powerful performance. Owing to its previous success, we further instruct fine-tuning Tk-instruct and built a more powerful det
    
[^26]: 基于对比学习和深度模块化的语音分离

    Speech Separation based on Contrastive Learning and Deep Modularization. (arXiv:2305.10652v1 [cs.SD])

    [http://arxiv.org/abs/2305.10652](http://arxiv.org/abs/2305.10652)

    本文提出了一种基于对比学习和深度模块化的完全无监督语音分离方法，解决了有监督学习中存在的排列问题、说话人数量不匹配的问题和高质量标记数据的依赖问题。

    

    目前，语音分离的最先进工具依赖于有监督学习。这意味着它们必须处理排列问题，它们受到训练和推断中使用的说话者数量不匹配的影响。此外，它们的性能严重依赖于高质量标记数据的存在。这些问题可以通过采用完全无监督的语音分离技术有效地解决。在本文中，我们使用对比学习建立帧的表示，然后在下游的深度模块化任务中使用学习到的表示。具体而言，在语音分离中，说话人的不同帧可以被看作是给定那个说话人的隐含标准帧的增强版。说话人的帧包含足够的韵律信息重叠，这是语音分离的关键。基于此，我们实现了自监督学习，学习缩小帧之间的距离。

    The current monaural state of the art tools for speech separation relies on supervised learning. This means that they must deal with permutation problem, they are impacted by the mismatch on the number of speakers used in training and inference. Moreover, their performance heavily relies on the presence of high-quality labelled data. These problems can be effectively addressed by employing a fully unsupervised technique for speech separation. In this paper, we use contrastive learning to establish the representations of frames then use the learned representations in the downstream deep modularization task. Concretely, we demonstrate experimentally that in speech separation, different frames of a speaker can be viewed as augmentations of a given hidden standard frame of that speaker. The frames of a speaker contain enough prosodic information overlap which is key in speech separation. Based on this, we implement a self-supervised learning to learn to minimize the distance between frames
    
[^27]: 基于BERT模型的推文地理位置预测

    Geolocation Predicting of Tweets Using BERT-Based Models. (arXiv:2303.07865v1 [cs.CL])

    [http://arxiv.org/abs/2303.07865](http://arxiv.org/abs/2303.07865)

    该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。

    

    该研究旨在解决推文/用户地理位置预测任务，并提供了处理文本大数据地理标记的灵活方法。该方法采用基于神经网络的自然语言处理来估计坐标对（经度，纬度）和二维高斯混合模型（GMM）。提出的模型的范围已经在Twitter数据集上使用预训练的BERT模型进行调整。性能指标表明，对于在推文内容和元数据上训练和评估的模型，全球范围内的中位误差小于30公里，美国范围内的中位误差小于15公里。

    This research is aimed to solve the tweet/user geolocation prediction task and provide a flexible methodology for the geotagging of textual big data. The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as coordinate pairs (longitude, latitude) and two-dimensional Gaussian Mixture Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as base models. Performance metrics show a median error of fewer than 30 km on a worldwide-level, and fewer than 15 km on the US-level datasets for the models trained and evaluated on text features of tweets' content and metadata context.
    

