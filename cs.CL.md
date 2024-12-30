# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order](https://arxiv.org/abs/2404.00399) | Aurora-M 是第一个根据美国行政命令进行红队测试的开源多语言模型，通过在英语、芬兰语、印地语、日语、越南语和代码上训练，不断预训练，包括了人工审核的安全说明，总训练 token 数超过 2 万亿个 |
| [^2] | [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919) | 逐层重要性采样的新方法LISA在微调任务中表现出色，记忆成本低且优于传统方法。 |
| [^3] | [Rules still work for Open Information Extraction](https://arxiv.org/abs/2403.10758) | 该论文提出了一种针对中文文本的创新型OIE模型APRCOIE，通过自动生成抽取模式、定义新的模式形式以及设计张量计算过滤器等方法，实现了对多样复杂中文语法现象的处理，并在评估中表现优越，拓展了OIE性能边界。 |
| [^4] | [CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences](https://arxiv.org/abs/2403.09032) | 介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。 |
| [^5] | [OmniPred: Language Models as Universal Regressors](https://arxiv.org/abs/2402.14547) | 本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。 |
| [^6] | [Probabilistic Reasoning in Generative Large Language Models](https://arxiv.org/abs/2402.09614) | 本文针对大型语言模型在概率推理任务中的限制，引入了贝叶斯语言推理数据集（BLInD），并提出了几种解决策略，包括Python代码和概率推理算法。 |
| [^7] | [MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences](https://arxiv.org/abs/2402.08925) | 这项工作提出了一种公平对齐大型语言模型与多样的人类偏好的方法，通过学习混合偏好分布并使用MaxMin对齐目标来更好地表示人类偏好。 |
| [^8] | [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | 本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。 |
| [^9] | [Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models.](http://arxiv.org/abs/2308.10462) | 本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。 |
| [^10] | [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training.](http://arxiv.org/abs/2306.00107) | 提出了一个带有大规模自监督训练的音乐理解模型MERT，利用了教师模型并采用了一种优于传统的语音和音频方法的组合方式。 |

# 详细

[^1]: Aurora-M: 根据美国行政命令，第一个开源的多语言语言模型进行了红队测试

    Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order

    [https://arxiv.org/abs/2404.00399](https://arxiv.org/abs/2404.00399)

    Aurora-M 是第一个根据美国行政命令进行红队测试的开源多语言模型，通过在英语、芬兰语、印地语、日语、越南语和代码上训练，不断预训练，包括了人工审核的安全说明，总训练 token 数超过 2 万亿个

    

    预训练语言模型支持多种人工智能应用，但是它们在训练时高昂的计算成本限制了可访问性。BLOOM 和 StarCoder 等倡议旨在使预训练模型对于协作社区开发更具民主性。然而，目前存在的模型面临一些挑战：多语言能力有限，持续的预训练会导致灾难性遗忘，而从头开始预训练又具有高昂的计算成本，并且需要遵守人工智能安全和发展法律。本文介绍了 Aurora-M，一个包含 15B 参数的多语言开源模型，训练语言包括英语、芬兰语、印地语、日语、越南语和代码。Aurora-M 不断从 StarCoderPlus 上预训练，额外训练了 4350 亿个 token，总训练 token 数超过了 2 万亿个。它是第一个在人工审核的安全说明上进行微调的开源多语言模型，使其开发与传统

    arXiv:2404.00399v1 Announce Type: cross  Abstract: Pretrained language models underpin several AI applications, but their high computational cost for training limits accessibility. Initiatives such as BLOOM and StarCoder aim to democratize access to pretrained models for collaborative community development. However, such existing models face challenges: limited multilingual capabilities, continual pretraining causing catastrophic forgetting, whereas pretraining from scratch is computationally expensive, and compliance with AI safety and development laws. This paper presents Aurora-M, a 15B parameter multilingual open-source model trained on English, Finnish, Hindi, Japanese, Vietnamese, and code. Continually pretrained from StarCoderPlus on 435 billion additional tokens, Aurora-M surpasses 2 trillion tokens in total training token count. It is the first open-source multilingual model fine-tuned on human-reviewed safety instructions, thus aligning its development not only with conventio
    
[^2]: LISA：用于高效内存大型语言模型微调的逐层重要性采样

    LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning

    [https://arxiv.org/abs/2403.17919](https://arxiv.org/abs/2403.17919)

    逐层重要性采样的新方法LISA在微调任务中表现出色，记忆成本低且优于传统方法。

    

    机器学习领域自大型语言模型（LLMs）首次出现以来取得了令人瞩目的进展，然而它们巨大的内存消耗已成为大规模训练的主要障碍。虽然已经提出了诸如低秩调整（LoRA）之类的参数高效微调技术来缓解这一问题，但在大多数大规模微调设置中，它们的性能仍无法与完整参数训练相匹配。为弥补这一不足，我们研究了LoRA在微调任务中的逐层特性，并观察到不同层之间权重范数的异常偏斜。利用这一关键观察，我们发现了一个令人惊讶简单的训练策略，在记忆成本低于LoRA的情况下，在广泛的设置中优于LoRA和完整参数训练。我们将其命名为Layerwise Importance Sampled AdamW（LISA），这是LoRA的一个有希望的替代方案，应用了

    arXiv:2403.17919v1 Announce Type: cross  Abstract: The machine learning community has witnessed impressive advancements since the first appearance of large language models (LLMs), yet their huge memory consumption has become a major roadblock to large-scale training. Parameter Efficient Fine-Tuning techniques such as Low-Rank Adaptation (LoRA) have been proposed to alleviate this problem, but their performance still fails to match full parameter training in most large-scale fine-tuning settings. Attempting to complement this deficiency, we investigate layerwise properties of LoRA on fine-tuning tasks and observe an uncommon skewness of weight norms across different layers. Utilizing this key observation, a surprisingly simple training strategy is discovered, which outperforms both LoRA and full parameter training in a wide range of settings with memory costs as low as LoRA. We name it Layerwise Importance Sampled AdamW (LISA), a promising alternative for LoRA, which applies the idea of
    
[^3]: 规则仍然适用于开放信息抽取

    Rules still work for Open Information Extraction

    [https://arxiv.org/abs/2403.10758](https://arxiv.org/abs/2403.10758)

    该论文提出了一种针对中文文本的创新型OIE模型APRCOIE，通过自动生成抽取模式、定义新的模式形式以及设计张量计算过滤器等方法，实现了对多样复杂中文语法现象的处理，并在评估中表现优越，拓展了OIE性能边界。

    

    开放信息抽取（OIE）旨在从自然语言文本中提取表面关系及其对应的参数，而不考虑领域。本文提出了一种针对中文文本的创新型OIE模型APRCOIE。与先前的模型不同，我们的模型可以自主生成抽取模式。该模型为中文OIE定义了一种新的模式形式，并提出了自动化的模式生成方法。通过这种方式，模型可以处理多样复杂的中文语法现象。我们基于张量计算设计了一个初步过滤器，以便高效地进行抽取过程。为了训练模型，我们手动标注了一个大规模的中文OIE数据集。在比较评估中，我们展示了APRCOIE优于最先进的中文OIE模型，并显著拓展了可实现的OIE性能边界。APRCOIE的代码和标注数据集已发布。

    arXiv:2403.10758v1 Announce Type: new  Abstract: Open information extraction (OIE) aims to extract surface relations and their corresponding arguments from natural language text, irrespective of domain. This paper presents an innovative OIE model, APRCOIE, tailored for Chinese text. Diverging from previous models, our model generates extraction patterns autonomously. The model defines a new pattern form for Chinese OIE and proposes an automated pattern generation methodology. In that way, the model can handle a wide array of complex and diverse Chinese grammatical phenomena. We design a preliminary filter based on tensor computing to conduct the extraction procedure efficiently. To train the model, we manually annotated a large-scale Chinese OIE dataset. In the comparative evaluation, we demonstrate that APRCOIE outperforms state-of-the-art Chinese OIE models and significantly expands the boundaries of achievable OIE performance. The code of APRCOIE and the annotated dataset are releas
    
[^4]: CodeUltraFeedback：一种用于将大型语言模型与编程偏好对齐的LLM作为法官数据集

    CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences

    [https://arxiv.org/abs/2403.09032](https://arxiv.org/abs/2403.09032)

    介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。

    

    评估大型语言模型（LLMs）与用户定义的编程偏好的对齐性是一项具有挑战性的工作，需要评估复杂文本LLMs的输出。现有基准仰赖自动化指标和静态分析工具，未能评估用户指令和LLM输出中的微妙之处，突显了对LLM偏好对齐的大规模数据集和基准的需求。在本文中，我们介绍了CodeUltraFeedback，一个包含10,000个复杂指令的偏好数据集，通过AI反馈来调整和对齐LLMs与编程偏好。我们使用14种不同的LLMs对这些指令生成响应，然后根据它们与五种编程偏好的对齐情况进行注释，使用GPT-3.5的LLM作为法官方法产生数字和文本反馈。我们还提出了CODAL-Bench，一个用于评估LLM与这些编程偏好对齐的基准。我们的结果显示C

    arXiv:2403.09032v1 Announce Type: cross  Abstract: Evaluating the alignment of large language models (LLMs) with user-defined coding preferences is a challenging endeavour that requires assessing intricate textual LLMs' outputs. By relying on automated metrics and static analysis tools, existing benchmarks fail to assess nuances in user instructions and LLM outputs, highlighting the need for large-scale datasets and benchmarks for LLM preference alignment. In this paper, we introduce CodeUltraFeedback, a preference dataset of 10,000 complex instructions to tune and align LLMs to coding preferences through AI feedback. We generate responses to the instructions using a pool of 14 diverse LLMs, which we then annotate according to their alignment with five coding preferences using the LLM-as-a-Judge approach with GPT-3.5, producing both numerical and textual feedback. We also present CODAL-Bench, a benchmark for assessing LLM alignment with these coding preferences. Our results show that C
    
[^5]: OmniPred：语言模型作为通用回归器

    OmniPred: Language Models as Universal Regressors

    [https://arxiv.org/abs/2402.14547](https://arxiv.org/abs/2402.14547)

    本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。

    

    在实验设计的广阔领域中，回归一直是一个强大的工具，可以准确预测系统或模型在给定一组参数的情况下的结果指标，但传统上只限于适用于特定任务的方法。在本文中，我们提出了OmniPred，这是一个用于训练语言模型作为通用端到端回归器的框架，使用来自多样真实世界实验的$(x,y)$评估数据。通过使用源自Google Vizier的数据，这是世界上最大的黑盒优化数据库之一，我们的大量实验表明，仅通过数学参数和值的文本表示，语言模型能够进行非常精确的数值回归，如果有机会训练多个任务，则可以显著优于传统的回归模型。

    arXiv:2402.14547v1 Announce Type: cross  Abstract: Over the broad landscape of experimental design, regression has been a powerful tool to accurately predict the outcome metrics of a system or model given a set of parameters, but has been traditionally restricted to methods which are only applicable to a specific task. In this paper, we propose OmniPred, a framework for training language models as universal end-to-end regressors over $(x,y)$ evaluation data from diverse real world experiments. Using data sourced from Google Vizier, one of the largest blackbox optimization databases in the world, our extensive experiments demonstrate that through only textual representations of mathematical parameters and values, language models are capable of very precise numerical regression, and if given the opportunity to train over multiple tasks, can significantly outperform traditional regression models.
    
[^6]: 生成式大型语言模型中的概率推理

    Probabilistic Reasoning in Generative Large Language Models

    [https://arxiv.org/abs/2402.09614](https://arxiv.org/abs/2402.09614)

    本文针对大型语言模型在概率推理任务中的限制，引入了贝叶斯语言推理数据集（BLInD），并提出了几种解决策略，包括Python代码和概率推理算法。

    

    本文探讨了大型语言模型（LLMs）在处理涉及概率值明确量化的文本推理问题时面临的挑战。这种概率推理对于从日常对话到医学决策等各种情境都很重要。尽管LLMs在数学推理能力方面有所改进，但在概率推理方面仍然存在显著困难。为了解决这个问题，我们首先引入了贝叶斯语言推理数据集（BLInD），这是一个专门设计用于测试LLMs概率推理能力的新数据集。然后，我们利用这个新数据集来详细说明LLMs在涉及概率推理的任务中的特定限制，并提出了几种将问题映射到不同形式表示的策略，包括Python代码和概率推理算法。

    arXiv:2402.09614v1 Announce Type: cross  Abstract: This paper considers the challenges that Large Language Models (LLMs) face when reasoning over text that includes information involving uncertainty explicitly quantified via probability values. This type of reasoning is relevant to a variety of contexts ranging from everyday conversations to medical decision-making. Despite improvements in the mathematical reasoning capabilities of LLMs, they still exhibit significant difficulties when it comes to probabilistic reasoning. To deal with this problem, we first introduce the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs. We then leverage this new dataset to thoroughly illustrate the specific limitations of LLMs for tasks involving probabilistic reasoning and present several strategies that map the problem to different formal representations, including Python code, probabilistic inference algorithm
    
[^7]: MaxMin-RLHF:面向具有多样的人类偏好的大型语言模型的公平对齐

    MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences

    [https://arxiv.org/abs/2402.08925](https://arxiv.org/abs/2402.08925)

    这项工作提出了一种公平对齐大型语言模型与多样的人类偏好的方法，通过学习混合偏好分布并使用MaxMin对齐目标来更好地表示人类偏好。

    

    强化学习从人类反馈中学习(RLHF)通过使用从偏好数据中派生的单一奖励模型来对齐语言模型与人类偏好一致。然而，这种方法忽视了从多个用户收集的数据中固有的人类偏好的丰富多样性。在这项工作中，我们首先推导出了使用单一奖励RLHF进行对齐的不可能性结果，从而凸显了其无法表示多样的人类偏好。为了提供一个公平的解决方案，我们通过期望最大化算法学习了一种混合偏好分布，并提出了一种受社会选择理论中的平等原则启发的MaxMin对齐目标来更好地表示多样的人类偏好。我们阐明了我们提出的方法与分布稳健优化和通用效用RL的联系，从而突显了我们提出的方法的普适性和鲁棒性。

    arXiv:2402.08925v1 Announce Type: cross Abstract: Reinforcement Learning from Human Feedback (RLHF) aligns language models to human preferences by employing a singular reward model derived from preference data. However, such an approach overlooks the rich diversity of human preferences inherent in data collected from multiple users. In this work, we first derive an impossibility result of alignment with single reward RLHF, thereby highlighting its insufficiency in representing diverse human preferences. To provide an equitable solution to the problem, we learn a mixture of preference distributions via an expectation-maximization algorithm and propose a MaxMin alignment objective for policy learning inspired by the Egalitarian principle in social choice theory to better represent diverse human preferences. We elucidate the connection of our proposed approach to distributionally robust optimization and general utility RL, thereby highlighting the generality and robustness of our proposed
    
[^8]: LLMs用于知识图谱构建和推理：最新功能与未来机遇

    LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

    [https://arxiv.org/abs/2305.13168](https://arxiv.org/abs/2305.13168)

    本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。

    

    本文对大规模语言模型（LLMs）在知识图谱（KG）构建和推理中的数量化和质化评估进行了详尽的研究。我们在八个不同的数据集上进行了实验，重点关注涵盖实体和关系提取、事件提取、链接预测和问答四个典型任务，从而全面探索了LLMs在构建和推理领域的表现。经验性研究发现，以GPT-4为代表的LLMs更适合作为推理助手，而不是少样本信息提取器。具体而言，虽然GPT-4在与KG构建相关的任务中表现出色，但在推理任务中表现更出色，在某些情况下超越了精调模型。此外，我们的调查还扩展到LLMs在信息提取方面的潜在泛化能力，提出了虚拟知识提取的构想。

    arXiv:2305.13168v2 Announce Type: replace-cross  Abstract: This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extr
    
[^9]: 探索大语言模型用于代码生成的参数高效微调技术

    Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models. (arXiv:2308.10462v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2308.10462](http://arxiv.org/abs/2308.10462)

    本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。

    

    大型语言模型（LLM）展示了在没有特定微调的情况下，即可根据自然语言意图生成准确的代码片段的印象能力。尽管先前的研究已经突出了微调LLMs的优势，但这个过程代价高，对于拥有数十亿个参数的模型来说，在资源稀缺的环境下是不切实际的。为了解决这些挑战，以前的研究探索了在上下文学习（ICL）作为一种策略，用任务特定的提示示例指导LLM生成过程。然而，ICL引入了一些不便之处，比如需要设计上下文相关的提示和没有学习任务特定的参数，从而限制了下游任务的性能。在这种情况下，我们预见参数高效微调（PEFT）技术作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将LLM专门用于任务特定的数据。

    Large Language Models (LLMs) demonstrate impressive capabilities to generate accurate code snippets given natural language intents in zero-shot, i.e., without the need for specific fine-tuning. While prior studies have highlighted the advantages of fine-tuning LLMs, this process incurs high computational costs, making it impractical in resource-scarce environments, particularly for models with billions of parameters. To address these challenges, previous research explored In-Context Learning (ICL) as a strategy to guide the LLM generative process with task-specific prompt examples. However, ICL introduces inconveniences, such as the need for designing contextually relevant prompts and the absence of learning task-specific parameters, thereby limiting downstream task performance. In this context, we foresee Parameter-Efficient Fine-Tuning (PEFT) techniques as a promising approach to efficiently specialize LLMs to task-specific data while maintaining reasonable resource consumption. In t
    
[^10]: MERT:带有大规模自监督训练的声学音乐理解模型

    MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. (arXiv:2306.00107v1 [cs.SD])

    [http://arxiv.org/abs/2306.00107](http://arxiv.org/abs/2306.00107)

    提出了一个带有大规模自监督训练的音乐理解模型MERT，利用了教师模型并采用了一种优于传统的语音和音频方法的组合方式。

    

    自监督学习（SSL）最近在视觉、文本和语音领域中已被证明是训练通用模型的一种很有前景的范例，对于跨越音乐领域的应用，尤其是对于调性和音高这样的特殊音乐知识的建模颇具挑战性。为了解决这一问题，我们提出了一个基于大规模自监督训练的声学音乐理解模型，即MERT。在我们的探索中，我们确定了更优秀的教师模型组合，这种组合方法在性能方面优于传统的语音和音频方法。

    Self-supervised learning (SSL) has recently emerged as a promising paradigm for training generalisable models on large-scale data in the fields of vision, text, and speech. Although SSL has been proven effective in speech and audio, its application to music audio has yet to be thoroughly explored. This is primarily due to the distinctive challenges associated with modelling musical knowledge, particularly its tonal and pitched characteristics of music. To address this research gap, we propose an acoustic Music undERstanding model with large-scale self-supervised Training (MERT), which incorporates teacher models to provide pseudo labels in the masked language modelling (MLM) style acoustic pre-training. In our exploration, we identified a superior combination of teacher models, which outperforms conventional speech and audio approaches in terms of performance. This combination includes an acoustic teacher based on Residual Vector Quantization - Variational AutoEncoder (RVQ-VAE) and a m
    

