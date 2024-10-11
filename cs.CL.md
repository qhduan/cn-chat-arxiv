# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spike No More: Stabilizing the Pre-training of Large Language Models](https://rss.arxiv.org/abs/2312.16903) | 本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。 |
| [^2] | [Instruction Multi-Constraint Molecular Generation Using a Teacher-Student Large Language Model](https://arxiv.org/abs/2403.13244) | 介绍了一个多约束分子生成大型语言模型TSMMG，通过整合多个小模型和工具来帮助生成符合描述的新分子，在各种约束任务中表现优秀。 |
| [^3] | [Reference-based Metrics Disprove Themselves in Question Generation](https://arxiv.org/abs/2403.12242) | 基于参考文献的指标在问句生成中被推翻，作者提出了一个无需参考文献的多维标准评估方法。 |
| [^4] | [Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/abs/2403.09919) | 本文介绍了一种适用于大型语言模型的循环草稿机制，结合了经典双模型和最新单模型方法，通过运用循环依赖设计，实现了高效的推测解码。 |
| [^5] | [Less is More: Data Value Estimation for Visual Instruction Tuning](https://arxiv.org/abs/2403.09559) | 视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。 |
| [^6] | [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/abs/2403.00953) | AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。 |
| [^7] | [DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2403.00896) | DiaHalu是第一个对话级幻觉评估基准，针对大型语言模型在对话级别上的幻觉问题进行研究。 |
| [^8] | [TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning](https://arxiv.org/abs/2402.19467) | TV-TREES是第一个多模态蕴涵树生成器，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树，实现了可解释联合模态推理，并在挑战性的TVQA数据集上展示了最先进的零-shot性能。 |
| [^9] | [Updating Language Models with Unstructured Facts: Towards Practical Knowledge Editing](https://arxiv.org/abs/2402.18909) | 本文提出了一个新的基准，非结构化知识编辑（UKE），旨在使用非结构化文本作为知识更新，避免了繁琐的结构化事实构建，具有更高效和响应性的知识编辑能力。 |
| [^10] | [LLMs with Chain-of-Thought Are Non-Causal Reasoners](https://arxiv.org/abs/2402.16048) | 本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。 |
| [^11] | [Unlocking the Power of Large Language Models for Entity Alignment](https://arxiv.org/abs/2402.15048) | ChatEA是一个创新性框架，利用大型语言模型提高实体对齐准确性，通过引入KG-code翻译模块和两阶段EA策略来克服传统方法的局限性。 |
| [^12] | [Probabilistically-sound beam search with masked language models](https://arxiv.org/abs/2402.15020) | 提出了在掩码语言模型上进行束搜索的概率健壮方法，表明其在多个领域中优于传统方法。 |
| [^13] | [Investigating Multilingual Instruction-Tuning: Do Polyglot Models Demand for Multilingual Instructions?](https://arxiv.org/abs/2402.13703) | 本研究是第一个对多语模型在不同印欧语言上的性能进行了广泛研究，发现在并行教学调整数据集上进行教学调整可以显著提升跨语言遵循能力，同时提出了对表面对齐假设的质疑 |
| [^14] | [PANDA (Pedantic ANswer-correctness Determination and Adjudication):Improving Automatic Evaluation for Question Answering and Text Generation](https://arxiv.org/abs/2402.11161) | 提出了PANDA方法，引入了更精确的答案正确性评测方式，解决了当前自动评估问答和文本生成过程中的挑战。 |
| [^15] | [Paramanu: A Family of Novel Efficient Indic Generative Foundation Language Models](https://arxiv.org/abs/2401.18034) | Paramanu是一种高效的印度生成式基础语言模型系列，包含多种印度语言模型，并且在单个GPU上进行了从头预训练。它还包括一个先进的印度分词器以及避免多语言诅咒的预训练方法。这些模型在人工评估中展现出良好的语法、连贯性、创造性和事实准确性。 |
| [^16] | [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/abs/2401.17244) | LLaMP是一个多模态的检索增强生成框架，能够在不进行微调的情况下，理解和集成各种材料科学概念的能力，检索相关数据，处理高阶数据以及总结固态合成过程。同时，LLaMP有效纠正了GPT-3.5内部知识的错误。 |
| [^17] | [TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning](https://arxiv.org/abs/2312.09039) | TAP4LLM提出了一个用于生成表格提示的多功能预处理工具箱，通过采样、增补和打包半结构化数据，解决了在大型语言模型推理中处理复杂问题和大型表格的挑战。 |
| [^18] | [Activation Addition: Steering Language Models Without Optimization.](http://arxiv.org/abs/2308.10248) | 这项研究探讨了一种在推理时通过改变激活来预测性地改变语言模型行为的方法，并且相比于传统方法具有更低的计算和实施成本，并且能够保持模型性能。 |
| [^19] | [On the Computational Power of Decoder-Only Transformer Language Models.](http://arxiv.org/abs/2305.17026) | 本篇论文研究了解码器Transformer语言模型的计算普适性，表明即使只有单层和单注意力头，仍然具有图灵完备性，其中单词嵌入的稀疏性/可压缩性是必要条件。 |
| [^20] | [What Makes a Language Easy to Deep-Learn?.](http://arxiv.org/abs/2302.12239) | 本研究通过测试神经网络和人类在学习和推广不同结构程度的语言方面的能力，发现神经网络在系统化概括方面存在困难，这对于模拟人类语言学习和进化构成了一个问题。 |

# 详细

[^1]: 别再出现尖峰了：稳定大型语言模型的预训练

    Spike No More: Stabilizing the Pre-training of Large Language Models

    [https://rss.arxiv.org/abs/2312.16903](https://rss.arxiv.org/abs/2312.16903)

    本论文研究了大型语言模型预训练中的损失尖峰问题，并通过理论分析找出了梯度爆炸的原因，并提出了满足要求的方法。通过实验证明，该方法能够有效地防止尖峰的发生。

    

    大型语言模型的预训练经常出现损失尖峰。这些尖峰会降低大型语言模型的性能，有时会破坏预训练。由于预训练需要大量的计算资源，我们应该避免这种尖峰的出现。为了研究损失尖峰的原因，我们关注内部层的梯度。通过理论分析，我们揭示了梯度爆炸的两个原因，并提供了预防梯度爆炸的要求。此外，我们提出了一种通过组合初始化方法和对嵌入进行简单修改来满足要求的方法。我们进行了各种实验证明我们的理论分析的有效性。实验结果表明，在预训练过程中，这种组合方法能够有效地防止尖峰的出现。

    Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a vast computational budget, we should avoid such spikes. To investigate the cause of loss spikes, we focus on gradients of internal layers. Through theoretical analyses, we reveal two causes of the exploding gradients, and provide requirements to prevent the explosion. In addition, we propose a method to satisfy the requirements by combining the initialization method and a simple modification to embeddings. We conduct various experiments to verify our theoretical analyses empirically. Experimental results indicate that the combination is effective in preventing spikes during pre-training.
    
[^2]: 使用师生大型语言模型进行多约束分子生成

    Instruction Multi-Constraint Molecular Generation Using a Teacher-Student Large Language Model

    [https://arxiv.org/abs/2403.13244](https://arxiv.org/abs/2403.13244)

    介绍了一个多约束分子生成大型语言模型TSMMG，通过整合多个小模型和工具来帮助生成符合描述的新分子，在各种约束任务中表现优秀。

    

    尽管已经提出了各种模型和计算工具用于分子的结构和性质分析，但生成符合所有期望结构和性质的分子仍然是一个挑战。在这里，我们介绍了一个多约束分子生成大型语言模型TSMMG，类似于学生，该模型整合了来自各种小模型和工具（即“老师”）的知识。为了训练TSMMG，我们通过从这些‘老师’中提取的分子知识构建了大量文本-分子对，使其能够通过各种文本提示生成符合描述的新分子。我们通过实验证明，TSMMG在生成符合复杂、自然语言描述的两、三和四约束任务的分子方面表现出色，平均分子有效性超过99％，成功率分别为88.08％、65.27％和61.44％。该模型还ex

    arXiv:2403.13244v1 Announce Type: new  Abstract: While various models and computational tools have been proposed for structure and property analysis of molecules, generating molecules that conform to all desired structures and properties remains a challenge. Here, we introduce a multi-constraint molecular generation large language model, TSMMG, which, akin to a student, incorporates knowledge from various small models and tools, namely, the 'teachers'. To train TSMMG, we construct a large set of text-molecule pairs by extracting molecular knowledge from these 'teachers', enabling it to generate novel molecules that conform to the descriptions through various text prompts. We experimentally show that TSMMG remarkably performs in generating molecules meeting complex, natural language-described property requirements across two-, three-, and four-constraint tasks, with an average molecular validity of over 99% and success ratio of 88.08%, 65.27%, and 61.44%, respectively. The model also ex
    
[^3]: 基于参考文献的指标在问句生成中被推翻

    Reference-based Metrics Disprove Themselves in Question Generation

    [https://arxiv.org/abs/2403.12242](https://arxiv.org/abs/2403.12242)

    基于参考文献的指标在问句生成中被推翻，作者提出了一个无需参考文献的多维标准评估方法。

    

    BLEU和BERTScore等基于参考文献的指标被广泛用于评估问句生成(QG)。本研究在SQuAD和HotpotQA等QG基准数据集上发现，使用人工编写的参考文献并不能保证基于参考文献的指标的有效性。大多数QG基准数据集只有一个参考文献；我们复制了注释过程并收集了另一个参考文献。预期好的指标应该对人工验证的问题的评分不会低于生成的问题。然而，在我们新收集的参考文献上，基于参考文献的指标的结果却证明了这些指标本身是错误的。我们提出了一个无需参考文献的指标，由多维标准组成，如自然性、可回答性和复杂性，利用大型语言模型。这些标准不受限于单个参考问题的句法或语义，该指标也不需要多样化的参考文献。实验证明我们的方法

    arXiv:2403.12242v1 Announce Type: cross  Abstract: Reference-based metrics such as BLEU and BERTScore are widely used to evaluate question generation (QG). In this study, on QG benchmarks such as SQuAD and HotpotQA, we find that using human-written references cannot guarantee the effectiveness of the reference-based metrics. Most QG benchmarks have only one reference; we replicated the annotation process and collect another reference. A good metric was expected to grade a human-validated question no worse than generated questions. However, the results of reference-based metrics on our newly collected reference disproved the metrics themselves. We propose a reference-free metric consisted of multi-dimensional criteria such as naturalness, answerability, and complexity, utilizing large language models. These criteria are not constrained to the syntactic or semantic of a single reference question, and the metric does not require a diverse set of references. Experiments reveal that our met
    
[^4]: 大型语言模型中用于快速推测解码的循环草稿机制

    Recurrent Drafter for Fast Speculative Decoding in Large Language Models

    [https://arxiv.org/abs/2403.09919](https://arxiv.org/abs/2403.09919)

    本文介绍了一种适用于大型语言模型的循环草稿机制，结合了经典双模型和最新单模型方法，通过运用循环依赖设计，实现了高效的推测解码。

    

    在本文中，我们介绍一种改进的推测解码方法，旨在提高大型语言模型的效率。我们的方法利用了两种成熟技术的优势：经典的双模型推测解码方法和较新的单模型方法Medusa。从Medusa得到灵感，我们的方法采用了单模型策略进行推测解码。然而，我们的方法通过使用具有循环依赖设计的单个轻量级草稿头来区分自己，本质上类似于经典推测解码中使用的小型草稿模型，但避免了完整transformer架构的复杂性。由于循环依赖，我们可以使用波束搜索快速过滤出草稿头中不需要的候选项。其结果是一种结合了单模型设计简易性并避免了创建数据相关树依赖的方法。

    arXiv:2403.09919v1 Announce Type: new  Abstract: In this paper, we introduce an improved approach of speculative decoding aimed at enhancing the efficiency of serving large language models. Our method capitalizes on the strengths of two established techniques: the classic two-model speculative decoding approach, and the more recent single-model approach, Medusa. Drawing inspiration from Medusa, our approach adopts a single-model strategy for speculative decoding. However, our method distinguishes itself by employing a single, lightweight draft head with a recurrent dependency design, akin in essence to the small, draft model uses in classic speculative decoding, but without the complexities of the full transformer architecture. And because of the recurrent dependency, we can use beam search to swiftly filter out undesired candidates with the draft head. The outcome is a method that combines the simplicity of single-model design and avoids the need to create a data-dependent tree attent
    
[^5]: 数据价值评估对视觉指导调整的影响

    Less is More: Data Value Estimation for Visual Instruction Tuning

    [https://arxiv.org/abs/2403.09559](https://arxiv.org/abs/2403.09559)

    视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。

    

    视觉指导调整是构建多模式大语言模型（MLLMs）的关键，大大提高了大语言模型（LLMs）在视觉场景中的推理能力。然而，现有的MLLMs主要依赖于多个高度多样化的视觉指导数据集的混合训练（甚至超过一百万条指导），这可能引入数据冗余。为了调查这个问题，我们进行了一系列实证研究，揭示了视觉指导数据集内存在显著冗余，并显示大大减少几个指导数据集的数量甚至不会影响性能。根据研究结果，我们提出了一种新的数据选择方法TIVE，以消除视觉指导数据中的冗余。TIVE首先根据计算的梯度估计视觉指导的任务级和实例级价值。然后，根据估计的价值，TIVE确定了任务级和实例级指导选择策略。

    arXiv:2403.09559v1 Announce Type: new  Abstract: Visual instruction tuning is the key to building multimodal large language models (MLLMs), which greatly improves the reasoning capabilities of large language models (LLMs) in vision scenario. However, existing MLLMs mostly rely on a mixture of multiple highly diverse visual instruction datasets for training (even more than a million instructions), which may introduce data redundancy. To investigate this issue, we conduct a series of empirical studies, which reveal a significant redundancy within the visual instruction datasets, and show that greatly reducing the amount of several instruction dataset even do not affect the performance. Based on the findings, we propose a new data selection approach TIVE, to eliminate redundancy within visual instruction data. TIVE first estimates the task-level and instance-level value of the visual instructions based on computed gradients. Then, according to the estimated values, TIVE determines the tas
    
[^6]: AutoRD：一种基于本体增强的大型语言模型的罕见疾病知识图构建的自动化端到端系统

    AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models

    [https://arxiv.org/abs/2403.00953](https://arxiv.org/abs/2403.00953)

    AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。

    

    目标：我们的目标是创建一个名为AutoRD的端到端系统，该系统自动从临床文本中提取有关罕见疾病的信息。我们进行了各种测试来评估AutoRD的性能，并在本文中强调了其优势和局限性。方法：我们的系统AutoRD是一个软件流水线，涉及数据预处理、实体提取、关系提取、实体校准和知识图构建。我们使用大型语言模型和由开源医学本体发展而来的医学知识图来实现这一目标。我们通过实体提取、关系提取以及知识图构建性能对系统进行定量评估。结果：AutoRD取得了47.3%的整体F1分数，较基础LLM提高了14.4%。具体来说，AutoRD实现了56.1%的整体实体提取F1分数（罕见疾病：83.5%，疾病：35.8%，s

    arXiv:2403.00953v1 Announce Type: cross  Abstract: Objectives: Our objective is to create an end-to-end system called AutoRD, which automates extracting information from clinical text about rare diseases. We have conducted various tests to evaluate the performance of AutoRD and highlighted its strengths and limitations in this paper.   Materials and Methods: Our system, AutoRD, is a software pipeline involving data preprocessing, entity extraction, relation extraction, entity calibration, and knowledge graph construction. We implement this using large language models and medical knowledge graphs developed from open-source medical ontologies. We quantitatively evaluate our system on entity extraction, relation extraction, and the performance of knowledge graph construction.   Results: AutoRD achieves an overall F1 score of 47.3%, a 14.4% improvement compared to the base LLM. In detail, AutoRD achieves an overall entity extraction F1 score of 56.1% (rare_disease: 83.5%, disease: 35.8%, s
    
[^7]: DiaHalu：大型语言模型的对话级幻觉评估基准

    DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models

    [https://arxiv.org/abs/2403.00896](https://arxiv.org/abs/2403.00896)

    DiaHalu是第一个对话级幻觉评估基准，针对大型语言模型在对话级别上的幻觉问题进行研究。

    

    自从最近几年大型语言模型（LLMs）取得了显著成功，幻觉问题仍然是一个挑战，有许多基准被提出来检测这种幻觉。然而，其中一些基准不是由LLMs自然生成的，而是有意引发的。此外，许多基准仅关注事实上的幻觉，而忽视了忠实度的幻觉。此外，尽管在LLMs时代对话模式被广泛应用，但目前的基准仅集中在句子级和段落级的幻觉上。在这项研究中，我们提出 DiaHalu，这是我们所知的第一个对话级幻觉评估基准。首先，我们将收集的主题集成到系统提示中，促进两个ChatGPT3.5之间的对话。随后，我们手动修改不符合人类语言约定的内容，然后让LLMs重新生成，模拟真实的人类-

    arXiv:2403.00896v1 Announce Type: cross  Abstract: Since large language models (LLMs) achieve significant success in recent years, the hallucination issue remains a challenge, numerous benchmarks are proposed to detect the hallucination. Nevertheless, some of these benchmarks are not naturally generated by LLMs but are intentionally induced. Also, many merely focus on the factuality hallucination while ignoring the faithfulness hallucination. Additionally, although dialogue pattern is more widely utilized in the era of LLMs, current benchmarks only concentrate on sentence-level and passage-level hallucination. In this study, we propose DiaHalu, the first dialogue-level hallucination evaluation benchmark to our knowledge. Initially, we integrate the collected topics into system prompts and facilitate a dialogue between two ChatGPT3.5. Subsequently, we manually modify the contents that do not adhere to human language conventions and then have LLMs re-generate, simulating authentic human-
    
[^8]: TV-TREES：用于神经符号视频推理的多模态蕴涵树

    TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning

    [https://arxiv.org/abs/2402.19467](https://arxiv.org/abs/2402.19467)

    TV-TREES是第一个多模态蕴涵树生成器，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树，实现了可解释联合模态推理，并在挑战性的TVQA数据集上展示了最先进的零-shot性能。

    

    在处理电视剪辑等复杂的多模态内容进行问答是一项具有挑战性的任务。这部分是因为当前的视频-语言模型依赖于单模态推理，在处理长输入时性能下降，并且缺乏可解释性。我们提出了TV-TREES，这是第一个多模态蕴涵树生成器。TV-TREES作为一种促进可解释联合模态推理的视频理解方法，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树。随后，我们引入了多模态蕴涵树生成任务来评估此类方法的推理质量。我们的方法在具有挑战性的TVQA数据集上的实验结果展示了可解释的、具有最先进零-shot性能的完整视频剪辑，展示了与黑盒方法相比的最佳实践。

    arXiv:2402.19467v1 Announce Type: cross  Abstract: It is challenging to perform question-answering over complex, multimodal content such as television clips. This is in part because current video-language models rely on single-modality reasoning, have lowered performance on long inputs, and lack interpetability. We propose TV-TREES, the first multimodal entailment tree generator. TV-TREES serves as an approach to video understanding that promotes interpretable joint-modality reasoning by producing trees of entailment relationships between simple premises directly entailed by the videos and higher-level conclusions. We then introduce the task of multimodal entailment tree generation to evaluate the reasoning quality of such methods. Our method's experimental results on the challenging TVQA dataset demonstrate intepretable, state-of-the-art zero-shot performance on full video clips, illustrating a best of both worlds contrast to black-box methods.
    
[^9]: 使用非结构化事实更新语言模型：迈向实用知识编辑

    Updating Language Models with Unstructured Facts: Towards Practical Knowledge Editing

    [https://arxiv.org/abs/2402.18909](https://arxiv.org/abs/2402.18909)

    本文提出了一个新的基准，非结构化知识编辑（UKE），旨在使用非结构化文本作为知识更新，避免了繁琐的结构化事实构建，具有更高效和响应性的知识编辑能力。

    

    知识编辑旨在将知识更新注入语言模型中，使其保持正确性和最新性。然而，当前的评估策略明显不切实际：它们仅使用精心策划的结构化事实（主题、关系和对象的三元组）进行更新，而现实世界的知识更新通常出现在新闻文章等非结构化文本中。本文提出了一个新的基准，非结构化知识编辑（UKE）。它使用非结构化文本直接评估编辑性能，称为非结构化事实。因此，UKE避免了繁琐的结构化事实构建，实现了高效和响应迅速的知识编辑，成为一个更实用的基准。我们在新构建的数据集上进行了大量实验，并展示了UKE对最先进的知识编辑方法构成了重大挑战，导致它们的关键性能下降。

    arXiv:2402.18909v1 Announce Type: cross  Abstract: Knowledge editing aims to inject knowledge updates into language models to keep them correct and up-to-date. However, its current evaluation strategies are notably impractical: they solely update with well-curated structured facts (triplets with subjects, relations, and objects), whereas real-world knowledge updates commonly emerge in unstructured texts like news articles. In this paper, we propose a new benchmark, Unstructured Knowledge Editing (UKE). It evaluates editing performance directly using unstructured texts as knowledge updates, termed unstructured facts. Hence UKE avoids the laborious construction of structured facts and enables efficient and responsive knowledge editing, becoming a more practical benchmark. We conduct extensive experiments on newly built datasets and demonstrate that UKE poses a significant challenge to state-of-the-art knowledge editing methods, resulting in their critical performance declines. We further
    
[^10]: LLMs带有思维链条是非因果推理者

    LLMs with Chain-of-Thought Are Non-Causal Reasoners

    [https://arxiv.org/abs/2402.16048](https://arxiv.org/abs/2402.16048)

    本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。

    

    本文探讨了大型语言模型（LLMs）推理中思维链条（CoT）的作用。尽管它有改善任务性能的潜力，但我们的分析揭示了在LLMs中正确答案跟随不正确CoTs的频率及反之。我们采用因果分析来评估CoTs/指令与LLMs答案之间的因果关系，揭示LLMs近似的结构因果模型（SCM）。通过比较暗示SCM与人类推理的SCM，我们突显了LLM和人类推理过程之间的差异。我们进一步研究了影响暗示SCM因果结构的因素，揭示了语境学习、有监督微调以及对人类反馈的强化学习显著影响因果关系。我们在https://github.com/StevenZHB/CoT_Causal_Analysis发布了代码和结果。

    arXiv:2402.16048v1 Announce Type: cross  Abstract: This paper explores the role of the Chain of Thought (CoT) in Large Language Models (LLMs) reasoning. Despite its potential to improve task performance, our analysis reveals a surprising frequency of correct answers following incorrect CoTs and vice versa. We employ causal analysis to assess the cause-effect relationship between CoTs/instructions and answers in LLMs, uncovering the Structural Causal Model (SCM) that LLMs approximate. By comparing the implied SCM with that of human reasoning, we highlight discrepancies between LLM and human reasoning processes. We further examine the factors influencing the causal structure of the implied SCM, revealing that in-context learning, supervised fine-tuning, and reinforcement learning on human feedback significantly impact the causal relations. We release the code and results at https://github.com/StevenZHB/CoT_Causal_Analysis.
    
[^11]: 发挥大型语言模型在实体对齐中的力量

    Unlocking the Power of Large Language Models for Entity Alignment

    [https://arxiv.org/abs/2402.15048](https://arxiv.org/abs/2402.15048)

    ChatEA是一个创新性框架，利用大型语言模型提高实体对齐准确性，通过引入KG-code翻译模块和两阶段EA策略来克服传统方法的局限性。

    

    实体对齐（EA）对于整合不同知识图（KG）数据至关重要，在数据驱动的人工智能应用中发挥着关键作用。传统的EA方法主要依赖于比较实体嵌入，但受限于有限的输入KG数据和表示学习技术的能力，它们的有效性受到约束。在这一背景下，我们介绍了ChatEA，这是一个创新性框架，它将大型语言模型（LLMs）融入以改善EA。为了解决有限的输入KG数据的限制，ChatEA引入了一个KG-code翻译模块，将KG结构翻译成LLMs可理解的格式，从而使LLMs能够利用其广泛的背景知识提高EA的准确性。为了克服对实体嵌入比较的过度依赖，ChatEA实现了一个两阶段EA策略，利用LLMs在对话格式中的多步推理能力，从而提高准确性。

    arXiv:2402.15048v1 Announce Type: cross  Abstract: Entity Alignment (EA) is vital for integrating diverse knowledge graph (KG) data, playing a crucial role in data-driven AI applications. Traditional EA methods primarily rely on comparing entity embeddings, but their effectiveness is constrained by the limited input KG data and the capabilities of the representation learning techniques. Against this backdrop, we introduce ChatEA, an innovative framework that incorporates large language models (LLMs) to improve EA. To address the constraints of limited input KG data, ChatEA introduces a KG-code translation module that translates KG structures into a format understandable by LLMs, thereby allowing LLMs to utilize their extensive background knowledge to improve EA accuracy. To overcome the over-reliance on entity embedding comparisons, ChatEA implements a two-stage EA strategy that capitalizes on LLMs' capability for multi-step reasoning in a dialogue format, thereby enhancing accuracy wh
    
[^12]: 具有掩码语言模型的概率健壮束搜索

    Probabilistically-sound beam search with masked language models

    [https://arxiv.org/abs/2402.15020](https://arxiv.org/abs/2402.15020)

    提出了在掩码语言模型上进行束搜索的概率健壮方法，表明其在多个领域中优于传统方法。

    

    具有掩码语言模型（MLMs）的束搜索存在挑战，部分原因是由于序列的联合概率分布不像自回归模型那样readily available。然而，估算这样的分布在许多领域中具有应用，包括蛋白工程和古代文本恢复。我们提出了一种具有概率健壮性的使用MLMs进行束搜索的方法。首先，我们阐明了在哪些条件下使用标准束搜索对MLMs执行文本填充在理论上是可靠的。当这些条件失败时，我们提供了一种具有概率健壮性的修改，而且无需额外的计算复杂性，并且证明在预期条件下它优于前述的束搜索。然后，我们提出了比较多个领域中几种使用MLMs进行填充的方法的经验结果。

    arXiv:2402.15020v1 Announce Type: cross  Abstract: Beam search with masked language models (MLMs) is challenging in part because joint probability distributions over sequences are not readily available, unlike for autoregressive models. Nevertheless, estimating such distributions has applications in many domains, including protein engineering and ancient text restoration. We present probabilistically-sound methods for beam search with MLMs. First, we clarify the conditions under which it is theoretically sound to perform text infilling with MLMs using standard beam search. When these conditions fail, we provide a probabilistically-sound modification with no additional computational complexity and demonstrate that it is superior to the aforementioned beam search in the expected conditions. We then present empirical results comparing several infilling approaches with MLMs across several domains.
    
[^13]: 调查多语言教学调整：多语模型是否需要多语教学？

    Investigating Multilingual Instruction-Tuning: Do Polyglot Models Demand for Multilingual Instructions?

    [https://arxiv.org/abs/2402.13703](https://arxiv.org/abs/2402.13703)

    本研究是第一个对多语模型在不同印欧语言上的性能进行了广泛研究，发现在并行教学调整数据集上进行教学调整可以显著提升跨语言遵循能力，同时提出了对表面对齐假设的质疑

    

    arXiv:2402.13703v1 公告类型：新摘要：将多语言预训练大型语言模型（LLMs）转化为雄辩而有用的助手对促进它们在不同语言地区的使用至关重要。基于这一精神，我们是第一个对跨多种印欧语言进行大规模研究的研究者，旨在研究多语模型在选择的最常用的印欧语言上的并行、多轮教学调整基准测试的性能。我们系统地研究了语言和教学数据集大小对中型多语言LLM的影响，通过在并行教学调整数据集上进行教学调整。我们的结果表明，在并行教学调整而不是单语语料库上进行教学调整可以使跨语言遵循能力提高多达4.6%。此外，我们表明，表面对齐假设通常不成立，因为所调查的多语7B参数模型是一个反例，需要大规模的教学调整。

    arXiv:2402.13703v1 Announce Type: new  Abstract: The adaption of multilingual pre-trained Large Language Models (LLMs) into eloquent and helpful assistants is essential to facilitate their use across different language regions. In that spirit, we are the first to conduct an extensive study of the performance of multilingual models on parallel, multi-turn instruction-tuning benchmarks across a selection of the most-spoken Indo-European languages. We systematically examine the effects of language and instruction dataset size on a mid-sized, multilingual LLM by instruction-tuning it on parallel instruction-tuning datasets. Our results demonstrate that instruction-tuning on parallel instead of monolingual corpora benefits cross-lingual instruction following capabilities by up to 4.6%. Furthermore, we show that the Superficial Alignment Hypothesis does not hold in general, as the investigated multilingual 7B parameter model presents a counter-example requiring large-scale instruction-tuning
    
[^14]: PANDA（Pedantic ANswer-correctness Determination and Adjudication）：改进问答和文本生成的自动评估

    PANDA (Pedantic ANswer-correctness Determination and Adjudication):Improving Automatic Evaluation for Question Answering and Text Generation

    [https://arxiv.org/abs/2402.11161](https://arxiv.org/abs/2402.11161)

    提出了PANDA方法，引入了更精确的答案正确性评测方式，解决了当前自动评估问答和文本生成过程中的挑战。

    

    问答（QA）只有在我们知道答案是否正确时才能取得进展，但对于许多最具挑战性和有趣的QA示例，当前的答案正确性（AC）指标与人类判断不一致，特别是来自大型语言模型（LLM）的冗长、自由格式答案。我们提出了两个挑战：缺乏数据和模型过大。基于LLM的评分器与人类更好地相关，但这项昂贵的任务仅在有限的QA数据集上进行了测试。我们通过提供清晰的指南来评估从人类QA比赛中采纳的机器QA，解决了这些问题。我们还引入了精确的答案正确性确定和裁决（Precise ANswer correctness Determination and Adjudication，PANDA），这是一个小巧、高效、确定性的AC分类器（812 KB），更准确地评估答案的正确性。

    arXiv:2402.11161v1 Announce Type: cross  Abstract: Question answering (QA) can only make progress if we know if an answer is correct, but for many of the most challenging and interesting QA examples, current answer correctness (AC) metrics do not align with human judgments, particularly verbose, free form answers from large language models (LLM). There are two challenges: a lack of data and that models are too big. LLM based scorers correlate better with humans, but this expensive task has only been tested on limited QA datasets. We rectify these issues by providing clear guidelines for evaluating machine QA adopted from human QA contests. We also introduce Precise ANswer correctness Determination and Adjudication (PANDA), a small, efficient, deterministic AC classifier (812 KB) that more accurately evaluates answer correctness.
    
[^15]: Paramanu: 一种高效的印度生成式基础语言模型系列

    Paramanu: A Family of Novel Efficient Indic Generative Foundation Language Models

    [https://arxiv.org/abs/2401.18034](https://arxiv.org/abs/2401.18034)

    Paramanu是一种高效的印度生成式基础语言模型系列，包含多种印度语言模型，并且在单个GPU上进行了从头预训练。它还包括一个先进的印度分词器以及避免多语言诅咒的预训练方法。这些模型在人工评估中展现出良好的语法、连贯性、创造性和事实准确性。

    

    我们介绍了Gyan AI Paramanu（“原子”），一种适用于印度语言的新型语言模型系列。它是一个在单个GPU上从头开始预训练的包含单语、双语和多语印度语言模型的集合，涵盖了10种印度语言（阿萨姆语、孟加拉语、印地语、康坎尼语、迈蒂利语、马拉地语、奥迪亚语、梵语、泰米尔语和泰卢固语）以及5种不同大小的字母表（孟加拉语、天城体、奥迪亚语、泰米尔语和泰卢固语）。这些模型以1024的上下文大小在单个GPU上预训练，非常高效、小巧、快速且强大。我们还开发了一种高效的先进的印度语分词器，甚至可以标记未知语言。为了避免我们的多语言mParamanu模型中的“多语言诅咒”，我们使用相同的字母表按语言类型进行了可比较语料库的预训练。我们对我们预训练模型进行了人工评估，评估指标包括语法、连贯性、创造性和事实准确性。

    We present Gyan AI Paramanu ("atom"), a family of novel language models for Indian languages. It is a collection of auto-regressive monolingual, bilingual, and multilingual Indic language models pretrained from scratch on a single GPU for 10 Indian languages (Assamese, Bangla, Hindi, Konkani, Maithili, Marathi, Odia, Sanskrit, Tamil, Telugu) across 5 scripts (Bangla, Devanagari, Odia, Tamil, Telugu) of varying sizes ranging from 13.29M to 367.5M.The models are pretrained with a context size of 1024 on a single GPU. The models are very efficient, small, fast, and powerful. We have also developed an efficient most advanced Indic tokenizer that can even tokenize unseen languages. In order to avoid the "curse of multi-linguality" in our multilingual mParamanu model, we pretrained on comparable corpora by typological grouping using the same script. We performed human evaluation of our pretrained models for open end text generation on grammar, coherence, creativity, and factuality metrics fo
    
[^16]: LLaMP: 大型语言模型在高保真材料知识检索和提炼中的强大应用

    LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation

    [https://arxiv.org/abs/2401.17244](https://arxiv.org/abs/2401.17244)

    LLaMP是一个多模态的检索增强生成框架，能够在不进行微调的情况下，理解和集成各种材料科学概念的能力，检索相关数据，处理高阶数据以及总结固态合成过程。同时，LLaMP有效纠正了GPT-3.5内部知识的错误。

    

    减少大型语言模型（LLM）的错误信息对于科学中的可重复性至关重要。然而，LLM天生缺乏长期记忆，因此在特定领域的文献和数据上对其进行微调是一个非常困难、临时的和不可避免具有偏见的任务。在这里，我们介绍了LLaMP，这是一个多模态的检索增强生成（RAG）框架，由多个数据感知的推理与行动（ReAct）智能体动态与Materials Project (MP)上的计算和实验数据进行交互。在无需微调的情况下，LLaMP展示了理解和集成各种方式的材料科学概念的能力，能够即时获取相关数据存储，处理高阶数据（如晶体结构和弹性张量），并总结固态合成的多步骤过程。我们证明LLaMP有效纠正了GPT-3.5内部知识的错误，将频繁记录的能带间隙MAPE降低了5.21%，将显著的错误降低了1103.54%

    Reducing hallucination of Large Language Models (LLMs) is imperative for use in the sciences where reproducibility is crucial. However, LLMs inherently lack long-term memory, making it a nontrivial, ad hoc, and inevitably biased task to fine-tune them on domain-specific literature and data. Here we introduce LLaMP, a multimodal retrieval-augmented generation (RAG) framework of multiple data-aware reasoning-and-acting (ReAct) agents that dynamically interact with computational and experimental data on Materials Project (MP). Without fine-tuning, LLaMP demonstrates an ability to comprehend and integrate various modalities of materials science concepts, fetch relevant data stores on the fly, process higher-order data (such as crystal structures and elastic tensors), and summarize multi-step procedures for solid-state synthesis. We show that LLaMP effectively corrects errors in GPT-3.5's intrinsic knowledge, reducing a 5.21% MAPE on frequently-documented bandgaps and a significant 1103.54%
    
[^17]: TAP4LLM：用于大型语言模型推理的表格提供者在对半结构化数据进行采样、增补和打包

    TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning

    [https://arxiv.org/abs/2312.09039](https://arxiv.org/abs/2312.09039)

    TAP4LLM提出了一个用于生成表格提示的多功能预处理工具箱，通过采样、增补和打包半结构化数据，解决了在大型语言模型推理中处理复杂问题和大型表格的挑战。

    

    基于表格的推理在结合深度模型和离散推理方面取得了显著进展，这需要对自由形式的自然语言（NL）问题和半结构化表格数据进行推理。然而，先前的表格推理解决方案只考虑小型表格，并且在处理更大表格时存在局限性。此外，大多数现有方法难以推理复杂问题，因为它们缺乏基本信息或分散在不同位置。为了解决这些挑战，我们提出了TAP4LLM作为一个多功能的预处理工具箱，通过平衡标记分配权衡来生成表格提示，实现(1) 表格采样，(2) 表格增补和(3) 表格打包。在每个模块中，我们收集和设计了几种在不同情况下使用的常见方法（例如，速度与准确性的平衡）。我们还对T内部每个组件的性能进行了全面评估。

    arXiv:2312.09039v2 Announce Type: replace-cross  Abstract: Table-based reasoning has shown remarkable progress in combining deep models with discrete reasoning, which requires reasoning over both free-form natural language (NL) questions and semi-structured tabular data. However, previous table reasoning solutions only consider small-sized tables and exhibit limitations in handling larger tables. In addition, most existing methods struggle to reason over complex questions since they lack essential information or they are scattered in different places. To alleviate these challenges, we propose TAP4LLM as a versatile pre-processing toolbox to generate table prompts through (1) table sampling, (2) table augmentation, and (3) table packing while balancing the token allocation trade-off. In each module, we collect and design several common methods for usage in various scenarios (e.g., speed over accuracy). We also provide a comprehensive evaluation on performance of each components inside T
    
[^18]: 激活添加: 无需优化即可操纵语言模型

    Activation Addition: Steering Language Models Without Optimization. (arXiv:2308.10248v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.10248](http://arxiv.org/abs/2308.10248)

    这项研究探讨了一种在推理时通过改变激活来预测性地改变语言模型行为的方法，并且相比于传统方法具有更低的计算和实施成本，并且能够保持模型性能。

    

    可靠地控制大型语言模型的行为是一个紧迫的开放性问题。现有的方法包括有监督微调、根据人类反馈进行强化学习、提示工程和引导解码。我们相反，研究了激活工程：在推理时修改激活以可预测地改变模型行为。特别地，我们通过自然语言隐式指定了一个添加的“导向向量”来偏置前向传播。与以前学习这些导向向量的工作不同，我们的激活添加（ActAdd）方法通过计算来自提示对的激活差异来计算它们。我们在OpenWebText和ConceptNet上展示了ActAdd在GPT-2上的应用。我们的推理时方法控制了输出的高级属性并保持了非目标模型的性能。它所需的计算和实施工作比微调要少得多，允许用户提供自然语言的规范，并且其开销与模型规模自然地扩展。

    Reliably controlling the behavior of large language models is a pressing open problem. Existing methods include supervised finetuning, reinforcement learning from human feedback, prompt engineering, and guided decoding. We instead investigate activation engineering: modifying activations at inference time to predictably alter model behavior. In particular, we bias the forward pass with an added 'steering vector' implicitly specified through natural language.  Unlike past work which learned these steering vectors, our Activation Addition (ActAdd) method computes them by taking the activation differences that result from pairs of prompts. We demonstrate ActAdd on GPT-2 on OpenWebText and ConceptNet. Our inference-time approach yields control over high-level properties of output and preserves off-target model performance. It involves far less compute and implementation effort than finetuning, allows users to provide natural language specifications, and its overhead scales naturally with m
    
[^19]: 论解码器Transformer语言模型的计算能力

    On the Computational Power of Decoder-Only Transformer Language Models. (arXiv:2305.17026v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17026](http://arxiv.org/abs/2305.17026)

    本篇论文研究了解码器Transformer语言模型的计算普适性，表明即使只有单层和单注意力头，仍然具有图灵完备性，其中单词嵌入的稀疏性/可压缩性是必要条件。

    

    本文章对解码器Transformer模型的计算普适性进行了理论评估。我们扩展了Transformer模型的理论文献，并表明仅使用单层和单注意力头的解码器Transformer结构，在合理假设下具备图灵完备性。从理论分析中，我们证明了单词嵌入的稀疏性/可压缩性是图灵完备性成立的必要条件。

    This article presents a theoretical evaluation of the computational universality of decoder-only transformer models. We extend the theoretical literature on transformer models and show that decoder-only transformer architectures (even with only a single layer and single attention head) are Turing complete under reasonable assumptions. From the theoretical analysis, we show sparsity/compressibility of the word embedding to be a necessary condition for Turing completeness to hold.
    
[^20]: 编程什么使一种语言易于深度学习？

    What Makes a Language Easy to Deep-Learn?. (arXiv:2302.12239v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.12239](http://arxiv.org/abs/2302.12239)

    本研究通过测试神经网络和人类在学习和推广不同结构程度的语言方面的能力，发现神经网络在系统化概括方面存在困难，这对于模拟人类语言学习和进化构成了一个问题。

    

    神经网络推动了自然语言处理的成功。语言的一个基本属性是其组成结构，使人类能够系统地产生新的意义形式。然而，与人类不同，神经网络在系统化概括方面一直存在困难，并且在新兴通信模拟中不一定受益于组成结构。这对于使用神经网络模拟人类语言学习和进化构成了一个问题，并且暗示了不同学习系统的偏见的关键差异。在这里，我们直接测试神经网络在学习和概括不同输入语言的能力，这些语言在其结构程度上有所不同。我们评估了一个预训练的语言模型GPT-3.5（类似于成年第二语言学习者）和从头开始训练的递归神经网络（类似于儿童第一语言学习者）的记忆和概括能力。我们的结果显示了令人震惊的

    Neural networks drive the success of natural language processing. A fundamental property of language is its compositional structure, allowing humans to produce forms for new meanings systematically. However, unlike humans, neural networks notoriously struggle with systematic generalization, and do not necessarily benefit from compositional structure in emergent communication simulations. This poses a problem for using neural networks to simulate human language learning and evolution, and suggests crucial differences in the biases of the different learning systems. Here, we directly test how neural networks compare to humans in learning and generalizing different input languages that vary in their degree of structure. We evaluate the memorization and generalization capabilities of a pre-trained language model GPT-3.5 (analagous to an adult second language learner) and recurrent neural networks trained from scratch (analaogous to a child first language learner). Our results show striking
    

