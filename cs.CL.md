# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auditing Large Language Models for Enhanced Text-Based Stereotype Detection and Probing-Based Bias Evaluation](https://arxiv.org/abs/2404.01768) | 该研究引入了Multi-Grain Stereotype（MGS）数据集，探索了不同的机器学习方法用于建立陈规检测的基线，并提出了一系列基于MGS数据训练的英文文本的陈规分类器模型。 |
| [^2] | [Encode Once and Decode in Parallel: Efficient Transformer Decoding](https://arxiv.org/abs/2403.13112) | 提出了一种新的编码器-解码器模型配置，称为prompt-in-decoder（PiD），可以一次编码输入并并行解码输出，在结构化输出和问答任务中取得高效率，避免了重复输入编码，大幅减少了解码器的内存占用。 |
| [^3] | [Data-oriented Dynamic Fine-tuning Parameter Selection Strategy for FISH Mask based Efficient Fine-tuning](https://arxiv.org/abs/2403.08484) | 提出了一种数据驱动的动态微调参数选择策略，针对FISH Mask提出了IRD算法，用于在不稳定的数据分布下动态选择最佳参数设置。 |
| [^4] | [Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation](https://arxiv.org/abs/2402.18191) | 本文提出了一种聚类与排序方法（CaR），通过与专家偏好相一致的评分模型排名指令对，保留了数据集的多样性。 |
| [^5] | [How (un)ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries](https://arxiv.org/abs/2402.15302) | 本研究探讨了大型语言模型（LLMs）对指令中心响应的容忍度，并提出了一个包含复杂查询的数据集，旨在揭示触发不道德响应的方法。 |
| [^6] | [Word-Sequence Entropy: Towards Uncertainty Estimation in Free-Form Medical Question Answering Applications and Beyond](https://arxiv.org/abs/2402.14259) | 本论文提出了一种新方法单词序列熵（WSE），用于在自由形式医学问答任务中量化答案的不确定性，相比其他基线方法表现更优秀。 |
| [^7] | [MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages](https://arxiv.org/abs/2402.10691) | MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。 |
| [^8] | [Improving Contextual Congruence Across Modalities for Effective Multimodal Marketing using Knowledge-infused Learning](https://arxiv.org/abs/2402.03607) | 本研究提出了一种将常识知识图谱与大型视觉语言模型相结合的框架，用于改进预测多模态营销活动效果的性能。该方法能够提供早期检测可能具有说服力的多模态活动并评估和增强营销理论的能力。 |
| [^9] | [Large Language Models are Null-Shot Learners](https://arxiv.org/abs/2401.08273) | 本文提出了零射击提示方法，通过利用大规模语言模型中的错误信息来指导模型进行任务，以提高任务表现。实验结果表明，在不同数据集上，包括阅读理解、算术推理和闭卷问答，模型性能有所提升。这些结果也显示出不同模型之间存在不同程度的错误信息。 |
| [^10] | [SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning](https://arxiv.org/abs/2401.07950) | SciGLM引入了自我反思指导注释框架，用于弥补大型语言模型在理解复杂科学概念、推导符号方程式和解决高级数值计算方面的不足，以训练能够进行大学水平科学推理的科学语言模型。 |
| [^11] | [A Comprehensive Study of Knowledge Editing for Large Language Models.](http://arxiv.org/abs/2401.01286) | 本研究全面研究了大型语言模型的知识编辑，旨在有效修改模型的行为，同时保持整体性能。 |
| [^12] | [Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models.](http://arxiv.org/abs/2310.17086) | Transformers学会了高阶优化方法，用于上下文学习，通过实现类似于迭代牛顿法的算法，而不是梯度下降。 |
| [^13] | [Matching Patients to Clinical Trials with Large Language Models.](http://arxiv.org/abs/2307.15051) | 本研究调查了使用大型语言模型（LLMs）来帮助患者和转诊医生识别合适的临床试验的潜力，并引入了TrialGPT架构，该架构能够准确预测合格性并提供解释，实验证明其有效性。 |
| [^14] | [Towards Explainable Evaluation Metrics for Machine Translation.](http://arxiv.org/abs/2306.13041) | 本研究探索机器翻译可解释性评估指标，提供综合综述和最新方法，并贡献下一代方法的愿景。 |

# 详细

[^1]: 用于增强基于文本的陈规检测和基于探测的偏见评估的大规模语言模型审计

    Auditing Large Language Models for Enhanced Text-Based Stereotype Detection and Probing-Based Bias Evaluation

    [https://arxiv.org/abs/2404.01768](https://arxiv.org/abs/2404.01768)

    该研究引入了Multi-Grain Stereotype（MGS）数据集，探索了不同的机器学习方法用于建立陈规检测的基线，并提出了一系列基于MGS数据训练的英文文本的陈规分类器模型。

    

    大型语言模型（LLMs）的最新进展显著提高了它们在面向人类的人工智能（AI）应用中的影响力。然而，LLMs可能会复制甚至加剧自训练数据中的陈规输出。本研究介绍了Multi-Grain Stereotype（MGS）数据集，包括51,867个实例，涵盖性别、种族、职业、宗教和陈规文本，通过融合多个先前公开的陈规检测数据集收集而来。我们探索了旨在为陈规检测建立基线的不同机器学习方法，并微调了多种架构和模型大小的几个语言模型，本文展示了一系列基于MGS训练的英文文本的陈规分类器模型。为了了解我们的陈规检测器是否捕捉到与人类常识一致的相关特征，我们利用了各种可解释的AI工具，

    arXiv:2404.01768v1 Announce Type: cross  Abstract: Recent advancements in Large Language Models (LLMs) have significantly increased their presence in human-facing Artificial Intelligence (AI) applications. However, LLMs could reproduce and even exacerbate stereotypical outputs from training data. This work introduces the Multi-Grain Stereotype (MGS) dataset, encompassing 51,867 instances across gender, race, profession, religion, and stereotypical text, collected by fusing multiple previously publicly available stereotype detection datasets. We explore different machine learning approaches aimed at establishing baselines for stereotype detection, and fine-tune several language models of various architectures and model sizes, presenting in this work a series of stereotypes classifier models for English text trained on MGS. To understand whether our stereotype detectors capture relevant features (aligning with human common sense) we utilise a variety of explanainable AI tools, including 
    
[^2]: 一次编码，多次并行解码：高效Transformer解码

    Encode Once and Decode in Parallel: Efficient Transformer Decoding

    [https://arxiv.org/abs/2403.13112](https://arxiv.org/abs/2403.13112)

    提出了一种新的编码器-解码器模型配置，称为prompt-in-decoder（PiD），可以一次编码输入并并行解码输出，在结构化输出和问答任务中取得高效率，避免了重复输入编码，大幅减少了解码器的内存占用。

    

    基于Transformer的自然语言处理模型功能强大，但计算成本高，限制了部署场景。在专业领域中，微调的编码器-解码器模型备受青睐，可以胜过更大更通用的仅解码器模型，例如GPT-4。我们介绍了一种新的编码器-解码器模型配置，可以提高在结构化输出和问答任务中的效率，在这些任务中，需要从单个输入中产生多个输出。我们的方法，prompt-in-decoder（PiD），只对输入进行一次编码，并且并行解码输出，通过避免重复输入编码，从而减少解码器的内存占用，提升了训练和推断效率。我们实现了计算减少，大致随子任务数量增加而扩展，相比最先进模型，在对话状态追踪、摘要和问答任务中获得高达4.6倍的速度提升，并且性能相当或更好。我们发布了我们的训练/推断代码。

    arXiv:2403.13112v1 Announce Type: new  Abstract: Transformer-based NLP models are powerful but have high computational costs that limit deployment scenarios. Finetuned encoder-decoder models are popular in specialized domains and can outperform larger more generalized decoder-only models, such as GPT-4. We introduce a new configuration for encoder-decoder models that improves efficiency on structured output and question-answering tasks where multiple outputs are required of a single input. Our method, prompt-in-decoder (PiD), encodes the input once and decodes output in parallel, boosting both training and inference efficiency by avoiding duplicate input encoding, thereby reducing the decoder's memory footprint. We achieve computation reduction that roughly scales with the number of subtasks, gaining up to 4.6x speed-up over state-of-the-art models for dialogue state tracking, summarization, and question-answering tasks with comparable or better performance. We release our training/inf
    
[^3]: 数据驱动的动态微调参数选择策略，用于基于FISH Mask的高效微调

    Data-oriented Dynamic Fine-tuning Parameter Selection Strategy for FISH Mask based Efficient Fine-tuning

    [https://arxiv.org/abs/2403.08484](https://arxiv.org/abs/2403.08484)

    提出了一种数据驱动的动态微调参数选择策略，针对FISH Mask提出了IRD算法，用于在不稳定的数据分布下动态选择最佳参数设置。

    

    鉴于大型语言模型(LLMs)的参数数量巨大，调整所有参数成本很高，因此更明智的做法是对特定参数进行微调。大多数参数高效微调(PEFT)集中在参数选择策略上，例如加法方法、选择性方法和基于重新参数化的方法。然而，很少有方法考虑数据样本对参数选择的影响，例如基于Fish Mask的方法。Fish Mask随机选择部分数据样本，并在参数选择过程中对它们进行同等处理，这无法为不稳定的数据分布动态选择最佳参数。在这项工作中，我们采用了数据驱动的视角，提出了一个IRD(迭代样本参数范围减小)算法，以搜索FISH Mask的最佳样本参数对设置。

    arXiv:2403.08484v1 Announce Type: new  Abstract: In view of the huge number of parameters of Large language models (LLMs) , tuning all parameters is very costly, and accordingly fine-tuning specific parameters is more sensible. Most of parameter efficient fine-tuning (PEFT) concentrate on parameter selection strategies, such as additive method, selective method and reparametrization-based method. However, there are few methods that consider the impact of data samples on parameter selecting, such as Fish Mask based method. Fish Mask randomly choose a part of data samples and treat them equally during parameter selection, which is unable to dynamically select optimal parameters for inconstant data distributions. In this work, we adopt a data-oriented perspective, then proposing an IRD ($\mathrm{\underline I}$terative sample-parameter $\mathrm{\underline R}$ange $\mathrm{\underline D}$ecreasing) algorithm to search the best setting of sample-parameter pair for FISH Mask. In each iteration
    
[^4]: 聚类与排序：通过专家定位质量估计实现保留多样性的指令选择

    Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation

    [https://arxiv.org/abs/2402.18191](https://arxiv.org/abs/2402.18191)

    本文提出了一种聚类与排序方法（CaR），通过与专家偏好相一致的评分模型排名指令对，保留了数据集的多样性。

    

    随着开源社区的贡献，涌现了大量指令调优（IT）数据。鉴于训练和评估模型需要大量资源分配，因此有必要采用高效的方法选择高质量的IT数据。然而，现有的指令数据选择方法存在一些限制，比如依赖脆弱的外部API、受GPT模型偏见影响，或减少所选指令数据集的多样性。在本文中，我们提出了一种面向工业的、与专家定位相吻合并保留多样性的指令数据选择方法：聚类与排序（CaR）。CaR分为两个步骤。第一步涉及使用与专家偏好很好对齐的评分模型对指令对进行排名（准确率达到84.25%）。第二步通过聚类过程保留数据集多样性。在我们的实验中，CaR选择了一个子集

    arXiv:2402.18191v1 Announce Type: new  Abstract: With contributions from the open-source community, a vast amount of instruction tuning (IT) data has emerged. Given the significant resource allocation required by training and evaluating models, it is advantageous to have an efficient method for selecting high-quality IT data. However, existing methods for instruction data selection have limitations such as relying on fragile external APIs, being affected by biases in GPT models, or reducing the diversity of the selected instruction dataset. In this paper, we propose an industrial-friendly, expert-aligned and diversity-preserved instruction data selection method: Clustering and Ranking (CaR). CaR consists of two steps. The first step involves ranking instruction pairs using a scoring model that is well aligned with expert preferences (achieving an accuracy of 84.25%). The second step involves preserving dataset diversity through a clustering process.In our experiment, CaR selected a sub
    
[^5]: 有关LLMs指令中心响应的（不道德）程度有多高？揭示安全防护栏对有害查询的漏洞

    How (un)ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries

    [https://arxiv.org/abs/2402.15302](https://arxiv.org/abs/2402.15302)

    本研究探讨了大型语言模型（LLMs）对指令中心响应的容忍度，并提出了一个包含复杂查询的数据集，旨在揭示触发不道德响应的方法。

    

    在这项研究中，我们解决了一个围绕大型语言模型（LLMs）安全和道德使用日益关注的问题。尽管这些模型具有潜力，但它们可能会被各种复杂的方法欺骗，产生有害或不道德内容，包括“越狱”技术和有针对性的操纵。我们的工作集中在一个特定问题上：LLMs在要求它们生成以伪代码、程序或软件片段为中心的响应时，有多大程度上可能会被误导，而不是生成普通文本。为了调查这个问题，我们引入了TechHazardQA，一个数据集，其中包含应以文本和以指令为中心格式（例如伪代码）回答的复杂查询，旨在识别不道德响应的触发器。我们查询了一系列LLMs-- Llama-2-13b，Llama-2-7b，Mistral-V2和Mistral 8X7B--并要求它们生成文本和指令为中心的响应。为了评估我们的方法，

    arXiv:2402.15302v1 Announce Type: new  Abstract: In this study, we tackle a growing concern around the safety and ethical use of large language models (LLMs). Despite their potential, these models can be tricked into producing harmful or unethical content through various sophisticated methods, including 'jailbreaking' techniques and targeted manipulation. Our work zeroes in on a specific issue: to what extent LLMs can be led astray by asking them to generate responses that are instruction-centric such as a pseudocode, a program or a software snippet as opposed to vanilla text. To investigate this question, we introduce TechHazardQA, a dataset containing complex queries which should be answered in both text and instruction-centric formats (e.g., pseudocodes), aimed at identifying triggers for unethical responses. We query a series of LLMs -- Llama-2-13b, Llama-2-7b, Mistral-V2 and Mistral 8X7B -- and ask them to generate both text and instruction-centric responses. For evaluation we rep
    
[^6]: 单词序列熵：走向自由形式医学问答应用及其不确定性估计

    Word-Sequence Entropy: Towards Uncertainty Estimation in Free-Form Medical Question Answering Applications and Beyond

    [https://arxiv.org/abs/2402.14259](https://arxiv.org/abs/2402.14259)

    本论文提出了一种新方法单词序列熵（WSE），用于在自由形式医学问答任务中量化答案的不确定性，相比其他基线方法表现更优秀。

    

    不确定性估计在确保安全关键的人工智能系统与人类互动的可靠性中发挥关键作用，尤其在医疗领域尤为重要。然而，在自由形式的医学问答任务中，尚未建立一种通用方法来量化答案的不确定性，其中无关的词汇和语序含有有限的语义信息可能是不确定性的主要来源，这是由于生成不平等的存在。本文提出了单词序列熵（WSE），该方法根据语义相关性在单词和序列级别上校准不确定性比例，在不确定性量化时更加强调关键词和更相关的序列。我们在5个自由形式医学问答数据集上，利用7种“现成的”大语言模型（LLMs）将WSE与6种基线方法进行比较，并展示了WSE在性能上的优越性。

    arXiv:2402.14259v1 Announce Type: cross  Abstract: Uncertainty estimation plays a pivotal role in ensuring the reliability of safety-critical human-AI interaction systems, particularly in the medical domain. However, a general method for quantifying the uncertainty of free-form answers has yet to be established in open-ended medical question-answering (QA) tasks, where irrelevant words and sequences with limited semantic information can be the primary source of uncertainty due to the presence of generative inequality. In this paper, we propose the Word-Sequence Entropy (WSE), which calibrates the uncertainty proportion at both the word and sequence levels according to the semantic relevance, with greater emphasis placed on keywords and more relevant sequences when performing uncertainty quantification. We compare WSE with 6 baseline methods on 5 free-form medical QA datasets, utilizing 7 "off-the-shelf" large language models (LLMs), and show that WSE exhibits superior performance on ac
    
[^7]: MultiPoT: 多语言思维程序利用多种编程语言

    MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages

    [https://arxiv.org/abs/2402.10691](https://arxiv.org/abs/2402.10691)

    MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。

    

    arXiv:2402.10691v1 公告类型：新的 摘要：思维程序（PoT）是一种以其可执行中间步骤为特征的方法，其确保推理过程中数值计算的准确性。目前，PoT主要使用Python。然而，仅依赖单一语言可能导致次优解决方案，忽视其他编程语言的潜在优势。在本文中，我们对PoT中使用的编程语言进行了全面实验，发现没有一种单一语言在所有任务和模型上始终提供最佳性能。每种语言的有效性取决于具体情景。受此启发，我们提出了一种称为MultiPoT的任务和模型无关方法，该方法从各种语言中获取强大和多样性。实验结果显示，MultiPoT 在很大程度上优于Python 自一致性。此外，与最佳模型相比，它实现了可比或更优异的性能。

    arXiv:2402.10691v1 Announce Type: new  Abstract: Program of Thoughts (PoT) is an approach characterized by its executable intermediate steps, which ensure the accuracy of the numerical calculations in the reasoning process. Currently, PoT primarily uses Python. However, relying solely on a single language may result in suboptimal solutions and overlook the potential benefits of other programming languages. In this paper, we conduct comprehensive experiments on the programming languages used in PoT and find that no single language consistently delivers optimal performance across all tasks and models. The effectiveness of each language varies depending on the specific scenarios. Inspired by this, we propose a task and model agnostic approach called MultiPoT, which harnesses strength and diversity from various languages. Experimental results reveal that it significantly outperforms Python Self-Consistency. Furthermore, it achieves comparable or superior performance compared to the best mo
    
[^8]: 提高多模态营销的上下文一致性：知识基础学习的有效性

    Improving Contextual Congruence Across Modalities for Effective Multimodal Marketing using Knowledge-infused Learning

    [https://arxiv.org/abs/2402.03607](https://arxiv.org/abs/2402.03607)

    本研究提出了一种将常识知识图谱与大型视觉语言模型相结合的框架，用于改进预测多模态营销活动效果的性能。该方法能够提供早期检测可能具有说服力的多模态活动并评估和增强营销理论的能力。

    

    智能设备的普及使用户能够在线体验多模态信息。然而，大型语言模型（LLM）和视觉模型（LVM）仍然受到捕捉跨模态语义关系的整体意义的限制。缺乏明确的常识知识（例如，作为一个知识图谱），视觉语言模型（VLM）仅通过捕捉庞大的语料库中的高级模式来学习隐式表示，从而忽略了重要的上下文跨模态线索。在这项工作中，我们设计了一个框架，将显式的常识知识以知识图谱的形式与大型的VLM相结合，以提高下游任务的性能，即预测多模态营销活动的有效性。虽然营销应用提供了一个有说服力的指标来评估我们的方法，但我们的方法使得早期发现可能具有说服力的多模态活动成为可能，并评估和增强营销理论。

    The prevalence of smart devices with the ability to capture moments in multiple modalities has enabled users to experience multimodal information online. However, large Language (LLMs) and Vision models (LVMs) are still limited in capturing holistic meaning with cross-modal semantic relationships. Without explicit, common sense knowledge (e.g., as a knowledge graph), Visual Language Models (VLMs) only learn implicit representations by capturing high-level patterns in vast corpora, missing essential contextual cross-modal cues. In this work, we design a framework to couple explicit commonsense knowledge in the form of knowledge graphs with large VLMs to improve the performance of a downstream task, predicting the effectiveness of multi-modal marketing campaigns. While the marketing application provides a compelling metric for assessing our methods, our approach enables the early detection of likely persuasive multi-modal campaigns and the assessment and augmentation of marketing theory.
    
[^9]: 大规模语言模型是零射击学习器

    Large Language Models are Null-Shot Learners

    [https://arxiv.org/abs/2401.08273](https://arxiv.org/abs/2401.08273)

    本文提出了零射击提示方法，通过利用大规模语言模型中的错误信息来指导模型进行任务，以提高任务表现。实验结果表明，在不同数据集上，包括阅读理解、算术推理和闭卷问答，模型性能有所提升。这些结果也显示出不同模型之间存在不同程度的错误信息。

    

    本文提出了零射击提示方法。零射击提示利用大规模语言模型（LLMs）中的错误信息，通过指示LLMs利用从“示例”部分中获取的信息（该信息在所提供的上下文中不存在）来完成任务。虽然减少错误信息对于LLMs的日常和重要用途至关重要，但我们提出在目前的环境中，这些LLMs仍然具有错误信息，实际上可以利用错误信息来提高与标准零射击提示相比的任务表现。对八个LLMs进行实验，结果显示在大多数八个数据集（包括阅读理解、算术推理和闭卷问答）中，性能有所提升。观察到的不一致性增加相对性能在LLMs之间的差异，也可能表示每个模型中存在不同程度的错误信息。

    arXiv:2401.08273v2 Announce Type: replace-cross Abstract: This paper presents null-shot prompting. Null-shot prompting exploits hallucination in large language models (LLMs) by instructing LLMs to utilize information from the "Examples" section that never exists within the provided context to perform a task. While reducing hallucination is crucial and non-negligible for daily and critical uses of LLMs, we propose that in the current landscape in which these LLMs still hallucinate, it is possible, in fact, to exploit hallucination to increase performance in performing tasks compared to standard zero-shot prompting. Experiments with eight LLMs show improvements in performance across the majority of eight datasets, including reading comprehension, arithmetic reasoning, and closed-book question answering. The observed inconsistency in increased relative performance across the LLMs also potentially indicates a different degree of inherent hallucination in each model. These differences show 
    
[^10]: SciGLM: 用自我反思指导注释和调整训练科学语言模型

    SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning

    [https://arxiv.org/abs/2401.07950](https://arxiv.org/abs/2401.07950)

    SciGLM引入了自我反思指导注释框架，用于弥补大型语言模型在理解复杂科学概念、推导符号方程式和解决高级数值计算方面的不足，以训练能够进行大学水平科学推理的科学语言模型。

    

    大型语言模型(LLMs)已显示出在协助科学发现方面的潜力。然而，目前LLMs在理解复杂科学概念、推导符号方程式和解决高级数值计算方面存在局限。为了弥补这些差距，我们引入了SciGLM，一套能够进行大学水平科学推理的科学语言模型。我们方法的核心是一种新颖的自我反思指导注释框架，以解决科学领域中数据稀缺挑战。该框架利用现有LLMs为未标记的科学问题生成逐步推理，随后经过自我反思的批评和修改过程。应用这一框架，我们整理了SciInstruct，这是一个涵盖物理、化学、数学和形式证明的多样化、高质量的数据集。我们利用SciInstruct对ChatGLM系列语言模型进行了微调，增强了

    arXiv:2401.07950v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs' deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciGLM, a suite of scientific language models able to conduct college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated SciInstruct, a diverse and high-quality dataset encompassing physics, chemistry, math, and formal proofs. We fine-tuned the ChatGLM family of language models with SciInstruct, enhancing
    
[^11]: 大型语言模型的知识编辑全面研究

    A Comprehensive Study of Knowledge Editing for Large Language Models. (arXiv:2401.01286v1 [cs.CL])

    [http://arxiv.org/abs/2401.01286](http://arxiv.org/abs/2401.01286)

    本研究全面研究了大型语言模型的知识编辑，旨在有效修改模型的行为，同时保持整体性能。

    

    大型语言模型(LLM)在理解和生成与人类交流紧密相似的文本方面展现出了非凡的能力。然而，其主要限制在于训练过程中的显著计算需求，这是由于其广泛的参数化造成的。这一挑战在于世界的动态性，需要频繁更新LLM以修正过时的信息或集成新知识，从而确保其持续的相关性。许多应用需要在训练后进行持续的模型调整，以解决缺陷或不良行为。近年来，对于LLM的知识编辑技术的兴趣越来越高，在特定领域内有效地修改LLM的行为，同时保持整体性能在各种输入中的表现。本文首先定义了知识编辑的目标和挑战，然后综述了现有的知识编辑方法和技术，并讨论了其应用和未来发展的方向。

    Large Language Models (LLMs) have shown extraordinary capabilities in understanding and generating text that closely mirrors human communication. However, a primary limitation lies in the significant computational demands during training, arising from their extensive parameterization. This challenge is further intensified by the dynamic nature of the world, necessitating frequent updates to LLMs to correct outdated information or integrate new knowledge, thereby ensuring their continued relevance. Note that many applications demand continual model adjustments post-training to address deficiencies or undesirable behaviors. There is an increasing interest in efficient, lightweight methods for on-the-fly model modifications. To this end, recent years have seen a burgeoning in the techniques of knowledge editing for LLMs, which aim to efficiently modify LLMs' behaviors within specific domains while preserving overall performance across various inputs. In this paper, we first define the kno
    
[^12]: Transformers学会了高阶优化方法用于上下文学习：一项与线性模型的研究

    Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models. (arXiv:2310.17086v1 [cs.LG])

    [http://arxiv.org/abs/2310.17086](http://arxiv.org/abs/2310.17086)

    Transformers学会了高阶优化方法，用于上下文学习，通过实现类似于迭代牛顿法的算法，而不是梯度下降。

    

    Transformers在上下文学习中表现出色，但是它们是如何进行上下文学习仍然是一个谜。最近的研究表明，Transformers可能通过内部运行梯度下降，即一阶优化方法，来进行上下文学习。本文中，我们展示了Transformers学会了实现高阶优化方法来进行上下文学习。我们以上下文线性回归为重点，展示了Transformers学会了实现一个非常类似于迭代牛顿法的算法，而不是梯度下降。从实证上来看，我们展示了连续的Transformer层的预测与牛顿法的不同迭代非常接近，每个中间层大致计算了3次迭代。相比之下，需要指数级的梯度下降步骤才能匹配额外的Transformer层；这表明Transformers具有相当的收敛速率。

    Transformers are remarkably good at in-context learning (ICL) -- learning from demonstrations without parameter updates -- but how they perform ICL remains a mystery. Recent work suggests that Transformers may learn in-context by internally running Gradient Descent, a first-order optimization method. In this paper, we instead demonstrate that Transformers learn to implement higher-order optimization methods to perform ICL. Focusing on in-context linear regression, we show that Transformers learn to implement an algorithm very similar to Iterative Newton's Method, a higher-order optimization method, rather than Gradient Descent. Empirically, we show that predictions from successive Transformer layers closely match different iterations of Newton's Method linearly, with each middle layer roughly computing 3 iterations. In contrast, exponentially more Gradient Descent steps are needed to match an additional Transformers layer; this suggests that Transformers have an comparable rate of conv
    
[^13]: 使用大型语言模型将患者与临床试验匹配

    Matching Patients to Clinical Trials with Large Language Models. (arXiv:2307.15051v1 [cs.CL])

    [http://arxiv.org/abs/2307.15051](http://arxiv.org/abs/2307.15051)

    本研究调查了使用大型语言模型（LLMs）来帮助患者和转诊医生识别合适的临床试验的潜力，并引入了TrialGPT架构，该架构能够准确预测合格性并提供解释，实验证明其有效性。

    

    临床试验在推动药物研发和基于证据的医学方面非常重要，但患者招募常常受到限制。在这项工作中，我们调查了使用大型语言模型（LLMs）来帮助患者和转诊医生识别合适的临床试验的潜力。具体而言，我们引入了一种新颖的架构TrialGPT，采用LLMs预测基于标准的合格性，并提供详细的解释，并根据患者病历中的自由文本来对候选临床试验进行排名和排除。我们在三个公开可用的184名患者和18,238个注释的临床试验的队列上评估了TrialGPT。实验结果表明几个关键发现：第一，TrialGPT在标准级别的预测准确性上表现出很高的准确率，并提供准确的解释。第二，TrialGPT的综合试验级别评分与专家标注的合格性高度相关。第三，这些评分

    Clinical trials are vital in advancing drug development and evidence-based medicine, but their success is often hindered by challenges in patient recruitment. In this work, we investigate the potential of large language models (LLMs) to assist individual patients and referral physicians in identifying suitable clinical trials from an extensive selection. Specifically, we introduce TrialGPT, a novel architecture employing LLMs to predict criterion-level eligibility with detailed explanations, which are then aggregated for ranking and excluding candidate clinical trials based on free-text patient notes. We evaluate TrialGPT on three publicly available cohorts of 184 patients and 18,238 annotated clinical trials. The experimental results demonstrate several key findings: First, TrialGPT achieves high criterion-level prediction accuracy with faithful explanations. Second, the aggregated trial-level TrialGPT scores are highly correlated with expert eligibility annotations. Third, these scor
    
[^14]: 机器翻译可解释性评估指标的探索

    Towards Explainable Evaluation Metrics for Machine Translation. (arXiv:2306.13041v1 [cs.CL])

    [http://arxiv.org/abs/2306.13041](http://arxiv.org/abs/2306.13041)

    本研究探索机器翻译可解释性评估指标，提供综合综述和最新方法，并贡献下一代方法的愿景。

    

    与传统的词汇重叠度量（如BLEU）不同，大多数当前用于机器翻译评估的指标（例如COMET或BERTScore）基于黑盒子的大型语言模型。它们通常与人类判断具有强相关性，但是最近的研究表明，较低质量的传统指标仍然占主导地位，其中一个潜在原因是它们的决策过程更透明。因此，为了促进新的高质量指标的更广泛接受，解释性变得至关重要。在这篇概念论文中，我们确定了可解释机器翻译指标的关键属性和目标，并提供了最近技术的综合综述，将它们与我们确立的目标和属性联系起来。在这个背景下，我们还讨论基于生成模型（如ChatGPT和GPT4）的可解释指标的最新先进方法。最后，我们贡献了下一代方法的愿景，包括自然语言e。

    Unlike classical lexical overlap metrics such as BLEU, most current evaluation metrics for machine translation (for example, COMET or BERTScore) are based on black-box large language models. They often achieve strong correlations with human judgments, but recent research indicates that the lower-quality classical metrics remain dominant, one of the potential reasons being that their decision processes are more transparent. To foster more widespread acceptance of novel high-quality metrics, explainability thus becomes crucial. In this concept paper, we identify key properties as well as key goals of explainable machine translation metrics and provide a comprehensive synthesis of recent techniques, relating them to our established goals and properties. In this context, we also discuss the latest state-of-the-art approaches to explainable metrics based on generative models such as ChatGPT and GPT4. Finally, we contribute a vision of next-generation approaches, including natural language e
    

