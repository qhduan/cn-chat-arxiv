# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distractor Generation for Multiple-Choice Questions: A Survey of Methods, Datasets, and Evaluation](https://rss.arxiv.org/abs/2402.01512) | 本文综述了多项选择题中干扰项生成的方法、数据集和评估。调查结果显示，现有数据集主要来自特定领域教育资源中，以文本为主，缺乏开放领域和多模态的数据集。 |
| [^2] | [Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon](https://rss.arxiv.org/abs/2401.03462) | 本论文提出了一种称为激活标志的新方法，它通过压缩LLM的激活状态，使其能够以有限的上下文窗口感知更长的上下文，同时保留了LLM在短上下文中的原始能力。这种方法具有竞争力的内存和时间效率，并通过多样化训练有效地支持不同上下文长度。 |
| [^3] | [Consecutive Model Editing with Batch alongside HooK Layers](https://arxiv.org/abs/2403.05330) | 提出了一种内存友好的连续模型编辑与批量支持的方法COMEBA-HK，在实验中表现出优越性。 |
| [^4] | [Few shot chain-of-thought driven reasoning to prompt LLMs for open ended medical question answering](https://arxiv.org/abs/2403.04890) | 本文提出了基于少样本推动推理的链式思维驱动LLMs用于开放式医学问题回答，通过修改MedQA-USMLE数据集并采用奖励训练机制，实现了在医疗场景中正确响应临床问题的有效方法。 |
| [^5] | [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) | 大语言模型中的层级存在较高相似性，有些层对网络功能几乎无影响。研究提出一种称为区块影响的度量，并通过层删除方法显著优于以往的模型修剪方法。 |
| [^6] | [Right for Right Reasons: Large Language Models for Verifiable Commonsense Knowledge Graph Question Answering](https://arxiv.org/abs/2403.01390) | LLM-based KGQA methods struggle with hallucination on commonsense reasoning questions, hindering their applicability in real-world applications. |
| [^7] | [Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment](https://arxiv.org/abs/2402.19085) | 引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。 |
| [^8] | [Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding](https://arxiv.org/abs/2402.14279) | 通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。 |
| [^9] | [I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments](https://arxiv.org/abs/2402.11192) | 将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。 |
| [^10] | [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) | 大型语言模型的性能与代理数量成比例，通过简单的采样和投票方法可以进一步增强性能，这种方法与现有的复杂方法正交。 |
| [^11] | [DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models](https://arxiv.org/abs/2402.02392) | DeLLMa是一个旨在提高不确定环境下决策精度的框架，通过多步骤的脚手架程序，借鉴决策理论和效用理论的原则，可以显著提高大型语言模型的决策性能。 |
| [^12] | [Adapting Large Language Models for Document-Level Machine Translation.](http://arxiv.org/abs/2401.06468) | 本文研究了适应大型语言模型进行文档级机器翻译的过程。实验结果显示，这些专门的模型在某些情况下超过了GPT-4的翻译性能，但在其他情况下仍然存在离标翻译问题，需要进一步改进和探索。 |
| [^13] | [Exploring the Landscape of Large Language Models In Medical Question Answering: Observations and Open Questions.](http://arxiv.org/abs/2310.07225) | 通过评估多种流行的大型语言模型在医学问题方面的知识，本研究提供了对这些模型作为一个群体的初步观察，并提出了进一步研究的开放问题。 |
| [^14] | [Detoxify Language Model Step-by-Step.](http://arxiv.org/abs/2308.08295) | 这项研究提出了一种分步解毒语言模型的方法，通过在输入阶段进行解毒处理，并使用无毒提示进行连续生成来保持生成质量。同时，通过设计Detox-Chain来校准LLMs的推理能力，实现了更安全和可靠的生成。 |
| [^15] | [NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic.](http://arxiv.org/abs/2307.02849) | NatLogAttack是一个用自然逻辑对自然语言推理模型进行系统性攻击的框架，它可以进行保持标签和翻转标签的攻击，并相比现有攻击模型产生了更好的对抗性攻击。 |

# 详细

[^1]: 多项选择题中的干扰项生成：方法、数据集和评估综述

    Distractor Generation for Multiple-Choice Questions: A Survey of Methods, Datasets, and Evaluation

    [https://rss.arxiv.org/abs/2402.01512](https://rss.arxiv.org/abs/2402.01512)

    本文综述了多项选择题中干扰项生成的方法、数据集和评估。调查结果显示，现有数据集主要来自特定领域教育资源中，以文本为主，缺乏开放领域和多模态的数据集。

    

    干扰项在学习评估中具有重要作用。本文调查了针对英语多项选择题的干扰项生成任务，并使用了文本和多模态语境的数据集。具体而言，本文对干扰项生成任务的最新研究进行了全面的文献综述，讨论了多项选择题的组成部分及其特点，分析了相关数据集，并总结了干扰项生成的评估指标。我们的调查结果表明，超过一半的数据集来自特定领域（如科学和英语）中的教育来源，并且主要是以文本为主，缺乏开放领域和多模态的数据集。

    Distractors are important in learning evaluation. This paper surveys distractor generation tasks using English multiple-choice question datasets for textual and multimodal contexts. In particular, this paper presents a thorough literature review of the recent studies on distractor generation tasks, discusses multiple choice components and their characteristics, analyzes the related datasets, and summarizes the evaluation metrics of distractor generation. Our investigation reveals that more than half of datasets are human-generated from educational sources in specific domains such as Science and English, which are largely text-based, with a lack of open domain and multimodal datasets.
    
[^2]: 从4K到400K的飞跃：利用激活标志扩展LLM的上下文

    Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon

    [https://rss.arxiv.org/abs/2401.03462](https://rss.arxiv.org/abs/2401.03462)

    本论文提出了一种称为激活标志的新方法，它通过压缩LLM的激活状态，使其能够以有限的上下文窗口感知更长的上下文，同时保留了LLM在短上下文中的原始能力。这种方法具有竞争力的内存和时间效率，并通过多样化训练有效地支持不同上下文长度。

    

    长上下文的利用对于LLM来说是一个巨大的挑战，因为它们有限的上下文窗口大小。尽管通过微调可以扩展上下文窗口，但这会导致训练和推理时间的显著成本，并对LLM的原始能力产生不利影响。在这项工作中，我们提出了一种名为激活标志的新方法，它将LLM的原始激活压缩成紧凑的形式，使LLM能够以有限的上下文窗口感知更长的上下文。激活标志被引入为插件模块，完全保留了LLM在短上下文中的原始能力。它与滑动窗口一起实时处理长的上下文，从而在训练和推理中实现了竞争力的内存和时间效率。激活标志是通过多样化压缩比的短序列数据进行训练的。得益于这种处理，它可以有效地学习支持不同上下文长度，实现小规模的训练。

    The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considerable cost at both training and inference time, and exert an unfavorable impact to the LLM's original capabilities. In this work, we propose a new method called Activation Beacon, which condenses LLM's raw activations into compact forms such that the LLM can perceive a longer context with a limited context window. Activation Beacon is introduced as a plug-in module, which fully preserves the LLM's original capability in short contexts. It works with the sliding window to streamingly process the long context, which leads to a competitive memory and time efficiency in both training and inference. Activation Beacon is trained with short-sequence data of diversified condensing ratios. Thanks to such a treatment, it can be effectively learned to support different context lengths with a small trai
    
[^3]: 连续模型编辑与批量支持的HooK层

    Consecutive Model Editing with Batch alongside HooK Layers

    [https://arxiv.org/abs/2403.05330](https://arxiv.org/abs/2403.05330)

    提出了一种内存友好的连续模型编辑与批量支持的方法COMEBA-HK，在实验中表现出优越性。

    

    由于典型的重新训练范式耗时且消耗资源，研究人员正在转向模型编辑，以寻找一种有效的、连续的、并支持批量方式直接编辑模型行为的方法。然而，尽管存在所有这些实用期望，现有的模型编辑方法却未能实现所有这些目标。此外，对于这种支持连续性模型编辑方法的内存需求往往是禁止性的，经常需要随着时间的增长逐步增加外部内存。为了应对这些挑战，我们提出了一种名为COMEBA-HK的模型编辑方法，该方法既是连续的又支持批量。COMEBA-HK对于存储几个具有更新权重的hook层仅需少量内存，是内存友好的。实验结果表明，我们的方法在单轮和连续批量编辑场景下优于其他支持批量模型编辑方法。

    arXiv:2403.05330v1 Announce Type: new  Abstract: As the typical retraining paradigm is unacceptably time- and resource-consuming, researchers are turning to model editing in order to seek an effective, consecutive, and batch-supportive way to edit the model behavior directly. Despite all these practical expectations, existing model editing methods fail to realize all of them. Furthermore, the memory demands for such succession-supportive model editing approaches tend to be prohibitive, frequently necessitating an external memory that grows incrementally over time. To cope with these challenges, we propose COMEBA-HK, a model editing method that is both consecutive and batch-supportive. COMEBA-HK is memory-friendly as it only needs a small amount of it to store several hook layers with updated weights. Experimental results demonstrate the superiority of our method over other batch-supportive model editing methods under both single-round and consecutive batch editing scenarios. Extensive 
    
[^4]: 基于少样本推动推理的链式思维驱动LLMs用于开放式医学问题回答

    Few shot chain-of-thought driven reasoning to prompt LLMs for open ended medical question answering

    [https://arxiv.org/abs/2403.04890](https://arxiv.org/abs/2403.04890)

    本文提出了基于少样本推动推理的链式思维驱动LLMs用于开放式医学问题回答，通过修改MedQA-USMLE数据集并采用奖励训练机制，实现了在医疗场景中正确响应临床问题的有效方法。

    

    大型语言模型（LLMs）已经展示了在转变医疗保健方面的巨大潜力，通过自动化诸如临床文档、信息检索和决策支持等任务。在这方面，精心设计的提示已经成为在医疗场景中使用LLMs的强大工具，例如患者临床场景。在本文中，我们提出了MedQA-USMLE数据集的修改版本，目的是模拟真实临床场景。我们探讨了基于主观响应生成的Chain of Thought（CoT）推理，用于修改后的MedQA-USMLE数据集，通过适当的LM驱动前向推理来获得正确的医学问题答案。考虑到在医疗环境中响应验证的重要性，我们利用奖励训练机制，其中语言模型还为特定的临床问题回应提供了适当的验证响应。

    arXiv:2403.04890v1 Announce Type: new  Abstract: Large Language models (LLMs) have demonstrated significant potential in transforming healthcare by automating tasks such as clinical documentation, information retrieval, and decision support. In this aspect, carefully engineered prompts have emerged as a powerful tool for using LLMs for medical scenarios, e.g., patient clinical scenarios. In this paper, we propose a modified version of the MedQA-USMLE dataset, which is subjective, to mimic real-life clinical scenarios. We explore the Chain of Thought (CoT) reasoning based on subjective response generation for the modified MedQA-USMLE dataset with appropriate LM-driven forward reasoning for correct responses to the medical questions. Keeping in mind the importance of response verification in the medical setting, we utilize a reward training mechanism whereby the language model also provides an appropriate verified response for a particular response to a clinical question. In this regard,
    
[^5]: ShortGPT: 大语言模型中的层级比您想象的更冗余

    ShortGPT: Layers in Large Language Models are More Redundant Than You Expect

    [https://arxiv.org/abs/2403.03853](https://arxiv.org/abs/2403.03853)

    大语言模型中的层级存在较高相似性，有些层对网络功能几乎无影响。研究提出一种称为区块影响的度量，并通过层删除方法显著优于以往的模型修剪方法。

    

    随着大语言模型（LLMs）在性能上不断取得进展，其规模显著增加，当前的LLMs包含数十亿甚至数万亿个参数。然而，在这项研究中，我们发现许多LLMs的层之间存在高度相似性，并且一些层在网络功能中起到了可忽略的作用。基于这一观察，我们定义了一种称为区块影响（BI）的度量衡量LLMs中每个层的重要性。然后，我们提出了一种简单的修剪方法：层删除，即根据它们的BI得分直接删除LLMs中的冗余层。实验证明，我们的方法ShortGPT在模型修剪方面明显优于以往的最先进方法。此外，ShortGPT与量化等方法正交，可以进一步减少参数和计算。通过简单的层删除即可获得更好的结果的能力，与传统的精确修剪方法截然不同。

    arXiv:2403.03853v1 Announce Type: new  Abstract: As Large Language Models (LLMs) continue to advance in performance, their size has escalated significantly, with current LLMs containing billions or even trillions of parameters. However, in this study, we discovered that many layers of LLMs exhibit high similarity, and some layers play a negligible role in network functionality. Based on this observation, we define a metric called Block Influence (BI) to gauge the significance of each layer in LLMs. We then propose a straightforward pruning approach: layer removal, in which we directly delete the redundant layers in LLMs based on their BI scores. Experiments demonstrate that our method, which we call ShortGPT, significantly outperforms previous state-of-the-art (SOTA) methods in model pruning. Moreover, ShortGPT is orthogonal to quantization-like methods, enabling further reduction in parameters and computation. The ability to achieve better results through simple layer removal, as oppo
    
[^6]: 正当且充分：可验证的常识知识图问题回答中的大型语言模型

    Right for Right Reasons: Large Language Models for Verifiable Commonsense Knowledge Graph Question Answering

    [https://arxiv.org/abs/2403.01390](https://arxiv.org/abs/2403.01390)

    LLM-based KGQA methods struggle with hallucination on commonsense reasoning questions, hindering their applicability in real-world applications.

    

    知识图问题回答（KGQA）方法旨在利用知识图中存储的关系信息来回答自然语言问题。随着大型语言模型（LLMs）的最新进展及其出色的推理能力，利用它们进行KGQA的趋势日益增长。然而，现有方法仅专注于回答事实性问题，例如“Silvio Berlusconi的第一任妻子出生在哪座城市？”，而忽略了涉及常识推理的问题，这是现实世界用户可能更经常提出的，例如“我需要单独的签证才能看到威伦多夫的维纳斯并参加今年夏天的奥运会吗？”。在这项工作中，我们首先观察到，现有基于LLM的KGQA方法在处理这类问题时难以产生真实的答案，尤其是对针对长尾实体的查询（例如非主流和最近的实体），从而阻碍了它们在现实世界应用中的可应用性。

    arXiv:2403.01390v1 Announce Type: new  Abstract: Knowledge Graph Question Answering (KGQA) methods seek to answer Natural Language questions using the relational information stored in Knowledge Graphs (KGs). With the recent advancements of Large Language Models (LLMs) and their remarkable reasoning abilities, there is a growing trend to leverage them for KGQA. However, existing methodologies have only focused on answering factual questions, e.g., "In which city was Silvio Berlusconi's first wife born?", leaving questions involving commonsense reasoning that real-world users may pose more often, e.g., "Do I need separate visas to see the Venus of Willendorf and attend the Olympics this summer?" unaddressed. In this work, we first observe that existing LLM-based methods for KGQA struggle with hallucination on such questions, especially on queries targeting long-tail entities (e.g., non-mainstream and recent entities), thus hindering their applicability in real-world applications especial
    
[^7]: 可控偏好优化：朝着可控多目标对齐方向发展

    Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment

    [https://arxiv.org/abs/2402.19085](https://arxiv.org/abs/2402.19085)

    引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。

    

    人工智能中的对齐工作旨在追求模型响应与人类偏好和价值的一致性。本文引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。实验分析表明，经过对齐的模型可以提供符合各种偏好的响应。

    arXiv:2402.19085v1 Announce Type: new  Abstract: Alignment in artificial intelligence pursues the consistency between model responses and human preferences as well as values. In practice, the multifaceted nature of human preferences inadvertently introduces what is known as the "alignment tax" -a compromise where enhancements in alignment within one objective (e.g.,harmlessness) can diminish performance in others (e.g.,helpfulness). However, existing alignment techniques are mostly unidirectional, leading to suboptimal trade-offs and poor flexibility over various objectives. To navigate this challenge, we argue the prominence of grounding LLMs with evident preferences. We introduce controllable preference optimization (CPO), which explicitly specifies preference scores for different objectives, thereby guiding the model to generate responses that meet the requirements. Our experimental analysis reveals that the aligned models can provide responses that match various preferences among t
    
[^8]: 使用音素表示减缓语言差异，实现稳健的多语言理解

    Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding

    [https://arxiv.org/abs/2402.14279](https://arxiv.org/abs/2402.14279)

    通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。

    

    为了改善多语言理解，通常需要在训练阶段使用多种语言，依赖复杂的训练技术，并且在高资源语言和低资源语言之间存在显著的性能差距。我们假设语言之间的性能差距受到这些语言之间的语言差异的影响，并通过使用音素表示（具体来说，将音素作为输入标记输入到语言模型中，而不是子词）提供了一种新颖的解决方案，以实现稳健的多语言建模。我们通过三个跨语言任务的定量证据展示了音素表示的有效性，这进一步得到了对跨语言性能差距的理论分析的证明。

    arXiv:2402.14279v1 Announce Type: cross  Abstract: Approaches to improving multilingual language understanding often require multiple languages during the training phase, rely on complicated training techniques, and -- importantly -- struggle with significant performance gaps between high-resource and low-resource languages. We hypothesize that the performance gaps between languages are affected by linguistic gaps between those languages and provide a novel solution for robust multilingual language modeling by employing phonemic representations (specifically, using phonemes as input tokens to LMs rather than subwords). We present quantitative evidence from three cross-lingual tasks that demonstrate the effectiveness of phonemic representation, which is further justified by a theoretical analysis of the cross-lingual performance gap.
    
[^9]: 如果你讲我的语言，我会更好地学习：使用风格对齐响应调整增强大型语言模型微调

    I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments

    [https://arxiv.org/abs/2402.11192](https://arxiv.org/abs/2402.11192)

    将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。

    

    使用小数据集为特定任务微调大型语言模型(LLMs)是一个普遍遇到的但复杂的挑战。在有限的示例上过多拟合可能会对模型的泛化能力和保留原始技能产生负面影响。我们的研究探讨了在微调过程中地实际响应风格的影响。我们发现将地实际响应风格与LLM固有风格匹配会产生更好的学习结果。基于这一观点，我们开发了一种方法，最小程度地修改LLM的现有响应以更正错误，使用这些调整后的响应作为训练目标。这种技术能够实现与模型固有响应风格一致的精确更正，维护模型的核心能力，从而避免过多拟合。我们的研究结果表明，这种方法不仅提高了LLM的特定任务准确性，而且关键地

    arXiv:2402.11192v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) with a small data set for particular tasks is a widely encountered yet complex challenge. The potential for overfitting on a limited number of examples can negatively impact the model's ability to generalize and retain its original skills. Our research explores the impact of the style of ground-truth responses during the fine-tuning process. We found that matching the ground-truth response style with the LLM's inherent style results in better learning outcomes. Building on this insight, we developed a method that minimally alters the LLM's pre-existing responses to correct errors, using these adjusted responses as training targets. This technique enables precise corrections in line with the model's native response style, safeguarding the model's core capabilities and thus avoid overfitting. Our findings show that this approach not only improves the LLM's task-specific accuracy but also crucially
    
[^10]: 更多的代理就是你所需要的

    More Agents Is All You Need

    [https://arxiv.org/abs/2402.05120](https://arxiv.org/abs/2402.05120)

    大型语言模型的性能与代理数量成比例，通过简单的采样和投票方法可以进一步增强性能，这种方法与现有的复杂方法正交。

    

    我们发现，仅通过一种采样和投票的方法，大型语言模型(Large Language Models, LLMs)的性能与实例化的代理数量成比例。此外，这种方法对已有的复杂方法进一步增强LLMs是正交的，而增强的程度与任务的困难程度相关。我们进行了广泛的实验，验证了我们的发现，并研究了能够促进其发生的属性。我们的代码公开在以下网址: \url{https://anonymous.4open.science/r/more_agent_is_all_you_need}

    We find that, simply via a sampling-and-voting method, the performance of large language models (LLMs) scales with the number of agents instantiated. Also, this method is orthogonal to existing complicated methods to further enhance LLMs, while the degree of enhancement is correlated to the task difficulty. We conduct comprehensive experiments on a wide range of LLM benchmarks to verify the presence of our finding, and to study the properties that can facilitate its occurrence. Our code is publicly available at: \url{https://anonymous.4open.science/r/more_agent_is_all_you_need}.
    
[^11]: DeLLMa:一个用于大型语言模型下决策的框架

    DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models

    [https://arxiv.org/abs/2402.02392](https://arxiv.org/abs/2402.02392)

    DeLLMa是一个旨在提高不确定环境下决策精度的框架，通过多步骤的脚手架程序，借鉴决策理论和效用理论的原则，可以显著提高大型语言模型的决策性能。

    

    大型语言模型（LLMs）在商业、工程和医学等领域被广泛应用，这些领域往往面临决策不确定性的问题，这是一个关键但具有挑战性的任务。本文表明，在决策问题上直接使用LLMs往往效果较差，尤其是在问题复杂性增加时。为了克服这个限制，我们提出了DeLLMa（Decision-making Large Language Model assistant）框架，旨在提高不确定环境下的决策精度。DeLLMa包括一个多步骤的脚手架程序，借鉴了决策理论和效用理论的原则，提供了一个最优的、可审计的决策过程。我们在涉及真实农业和金融数据的决策环境中验证了我们的框架。结果表明，DeLLMa可以显著提高LLMs的决策性能，准确性可提高高达40%以上。

    Large language models (LLMs) are increasingly used across society, including in domains like business, engineering, and medicine. These fields often grapple with decision-making under uncertainty, a critical yet challenging task. In this paper, we show that directly prompting LLMs on these types of decision-making problems yields poor results, especially as the problem complexity increases. To overcome this limitation, we propose DeLLMa (Decision-making Large Language Model assistant), a framework designed to enhance decision-making accuracy in uncertain environments. DeLLMa involves a multi-step scaffolding procedure, drawing upon principles from decision theory and utility theory, to provide an optimal and human-auditable decision-making process. We validate our framework on decision-making environments involving real agriculture and finance data. Our results show that DeLLMa can significantly improve LLM decision-making performance, achieving up to a 40% increase in accuracy over co
    
[^12]: 适应大型语言模型进行文档级机器翻译的研究

    Adapting Large Language Models for Document-Level Machine Translation. (arXiv:2401.06468v1 [cs.CL])

    [http://arxiv.org/abs/2401.06468](http://arxiv.org/abs/2401.06468)

    本文研究了适应大型语言模型进行文档级机器翻译的过程。实验结果显示，这些专门的模型在某些情况下超过了GPT-4的翻译性能，但在其他情况下仍然存在离标翻译问题，需要进一步改进和探索。

    

    大型语言模型（LLMs）在各种自然语言处理（NLP）任务中取得了重要进展。最近的研究表明，在任务特定的微调之后，中等规模的LLMs往往胜过其更大的对应模型。在这项工作中，我们深入研究了将LLMs调整为特定语言对的文档级机器翻译（DocMT）的过程。首先，我们探讨了提示策略对下游翻译性能的影响。然后，我们进行了大量实验，使用了两种微调方法、三种LLM主干和18个涉及九种语言对的翻译任务。我们的研究结果表明，在某些情况下，这些专门的模型甚至在翻译性能上超过了GPT-4，而在其他情况下，即使它们专门在双语平行文档上进行了微调，仍然明显存在离标翻译问题。此外，我们对这些针对DocMT量身定制的LLMs进行了深入分析，探讨了如翻译准确度改善、多源信息整合等各个方面。

    Large language models (LLMs) have made significant strides in various natural language processing (NLP) tasks. Recent research shows that the moderately-sized LLMs often outperform their larger counterparts after task-specific fine-tuning. In this work, we delve into the process of adapting LLMs to specialize in document-level machine translation (DocMT) for a specific language pair. Firstly, we explore how prompt strategies affect downstream translation performance. Then, we conduct extensive experiments with two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Our findings indicate that in some cases, these specialized models even surpass GPT-4 in translation performance, while they still significantly suffer from the off-target translation issue in others, even if they are exclusively fine-tuned on bilingual parallel documents. Furthermore, we provide an in-depth analysis of these LLMs tailored for DocMT, exploring aspects such as transl
    
[^13]: 在医学问题回答中探索大型语言模型的领域: 观察和开放问题

    Exploring the Landscape of Large Language Models In Medical Question Answering: Observations and Open Questions. (arXiv:2310.07225v1 [cs.CL])

    [http://arxiv.org/abs/2310.07225](http://arxiv.org/abs/2310.07225)

    通过评估多种流行的大型语言模型在医学问题方面的知识，本研究提供了对这些模型作为一个群体的初步观察，并提出了进一步研究的开放问题。

    

    大型语言模型(LLMs)在医学问题回答领域显示出潜力，通过在标准化考试中取得及格分数，并被认为是支持医疗保健工作者的工具。将LLMs部署到如此高风险的环境中需要对这些模型的限制有清晰的理解。随着新的LLMs的快速发展和发布，识别跨模型存在的模式，并因此可能出现在新版本中，特别有价值。在本文中，我们评估了多种流行LLM在医学问题方面的知识，以更好地了解它们作为一个群体的特性。通过这个比较，我们提供了初步的观察，并提出了进一步研究的开放问题。

    Large Language Models (LLMs) have shown promise in medical question answering by achieving passing scores in standardised exams and have been suggested as tools for supporting healthcare workers. Deploying LLMs into such a high-risk context requires a clear understanding of the limitations of these models. With the rapid development and release of new LLMs, it is especially valuable to identify patterns which exist across models and may, therefore, continue to appear in newer versions. In this paper, we evaluate a wide range of popular LLMs on their knowledge of medical questions in order to better understand their properties as a group. From this comparison, we provide preliminary observations and raise open questions for further research.
    
[^14]: 分步解毒语言模型

    Detoxify Language Model Step-by-Step. (arXiv:2308.08295v1 [cs.CL])

    [http://arxiv.org/abs/2308.08295](http://arxiv.org/abs/2308.08295)

    这项研究提出了一种分步解毒语言模型的方法，通过在输入阶段进行解毒处理，并使用无毒提示进行连续生成来保持生成质量。同时，通过设计Detox-Chain来校准LLMs的推理能力，实现了更安全和可靠的生成。

    

    解毒语言模型具有挑战性，因为它要求模型在保持生成能力的同时避免生成有害内容。为了确保生成的安全性，先前的解毒方法通过改变数据分布或在单步骤中从不同方面约束生成来解毒模型。然而，由于语言模型倾向于沿着有毒提示生成，解毒方法的工作方向与之相反，这些方法将大大影响LLM的生成质量，如话语连贯性和语义一致性。为了处理这种冲突，我们将解毒过程分解为不同的子步骤，其中解毒集中在输入阶段，随后的连续生成基于无毒提示。此外，我们还通过设计一个Detox-Chain来校准LLMs的强大推理能力，以有序的方式连接上述子步骤，这使得LLMs可以进行连续的解毒生成。

    Detoxification for LLMs is challenging since it requires models to avoid generating harmful content while maintaining the generation capability. To ensure the safety of generations, previous detoxification methods detoxify the models by changing the data distributions or constraining the generations from different aspects in a single-step manner. However, these approaches will dramatically affect the generation quality of LLMs, e.g., discourse coherence and semantic consistency, since language models tend to generate along the toxic prompt while detoxification methods work in the opposite direction. To handle such a conflict, we decompose the detoxification process into different sub-steps, where the detoxification is concentrated in the input stage and the subsequent continual generation is based on the non-toxic prompt. Besides, we also calibrate the strong reasoning ability of LLMs by designing a Detox-Chain to connect the above sub-steps in an orderly manner, which allows LLMs to d
    
[^15]: NatLogAttack: 一个用自然逻辑对自然语言推理模型进行攻击的框架

    NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic. (arXiv:2307.02849v1 [cs.CL])

    [http://arxiv.org/abs/2307.02849](http://arxiv.org/abs/2307.02849)

    NatLogAttack是一个用自然逻辑对自然语言推理模型进行系统性攻击的框架，它可以进行保持标签和翻转标签的攻击，并相比现有攻击模型产生了更好的对抗性攻击。

    

    推理自从人工智能的开始就是一个中心话题。近年来在分布式表示和神经网络上取得的进展持续改进了自然语言推理模型的最新性能。然而，这些模型是否通过真正的推理来得出结论，还是依赖于虚假的相关性，这仍然是一个未解决的问题。对抗性攻击已经证明是评估受害模型的致命弱点的重要工具。在这项研究中，我们探讨了基于逻辑形式主义开发攻击模型的基本问题。我们提出了NatLogAttack来执行围绕自然逻辑的系统性攻击，这是一个可追溯到亚里士多德三段论并且与自然语言推理密切相关的经典逻辑形式。该提议的框架可以进行保持标签和翻转标签的攻击。我们展示了与现有攻击模型相比，NatLogAttack产生了更好的对抗性攻击。

    Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial 
    

