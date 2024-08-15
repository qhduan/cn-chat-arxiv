# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WavLLM: Towards Robust and Adaptive Speech Large Language Model](https://arxiv.org/abs/2404.00656) | WavLLM是一个稳健和自适应语音大语言模型，引入了双编码器和Prompt-aware LoRA权重适配器，通过两阶段课程学习方法优化，解耦不同类型的语音信息，为处理语义内容和说话者身份的独特特征提供了新思路 |
| [^2] | [Retrieval-Enhanced Knowledge Editing for Multi-Hop Question Answering in Language Models](https://arxiv.org/abs/2403.19631) | 提出了用于多跳问题回答的检索增强模型编辑（RAE）框架，利用互信息最大化的检索方法和修剪策略，实现了对语言模型的有效优化。 |
| [^3] | [Lost in Overlap: Exploring Watermark Collision in LLMs](https://arxiv.org/abs/2403.10020) | 本研究探讨了在大型语言模型中关于水印冲突的问题，发现双水印冲突存在时会对水印算法的检测性能造成威胁。 |
| [^4] | [$\texttt{COSMIC}$: Mutual Information for Task-Agnostic Summarization Evaluation](https://arxiv.org/abs/2402.19457) | $\texttt{COSMIC}$是一种以相互信息为基础的新的摘要评估方法，有效预测下游任务表现，并与人类判断相关性强。竞争性能优于$\texttt{BERTScore}$和$\texttt{ROUGE}$。 |
| [^5] | [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) | 大型语言模型中出现了大量激活现象，它们具有非常大的值并且在模型中起到重要作用。 |
| [^6] | [FIPO: Free-form Instruction-oriented Prompt Optimization with Preference Dataset and Modular Fine-tuning Schema](https://arxiv.org/abs/2402.11811) | FIPO提出了基于自由形式指导的提示优化方法，结合偏好数据集和模块化微调模式，重新构思了优化过程并实现了灵活的任务提示生成。 |
| [^7] | [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) | V-STaR利用正确和不正确的解决方案训练验证器，用于选择模型生成的解决方案，实现了自我改进和验证方法在常见代码生成和数学推理任务中达到4%到17%的测试准确率提升。 |
| [^8] | [LLM Voting: Human Choices and AI Collective Decision Making](https://arxiv.org/abs/2402.01766) | 本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。 |
| [^9] | [Agent Instructs Large Language Models to be General Zero-Shot Reasoners.](http://arxiv.org/abs/2310.03710) | 该论文提出了一种方法，通过代理指导的方式，大大提高了大型语言模型在零-shot推理任务上的能力，并在多个数据集上实现了最先进的性能。 |
| [^10] | [Survey on Sociodemographic Bias in Natural Language Processing.](http://arxiv.org/abs/2306.08158) | 本文调查了209篇关于NLP模型偏见的论文，其中大部分涉及社会人口统计偏见。研究者提出了社会人口统计偏见的定义，并确定了NLP偏见研究的三个主要类别。当前去偏见技术只是隐藏了偏见而不是真正去除它，需要进一步改进。 |

# 详细

[^1]: WavLLM：面向稳健和自适应语音大语言模型

    WavLLM: Towards Robust and Adaptive Speech Large Language Model

    [https://arxiv.org/abs/2404.00656](https://arxiv.org/abs/2404.00656)

    WavLLM是一个稳健和自适应语音大语言模型，引入了双编码器和Prompt-aware LoRA权重适配器，通过两阶段课程学习方法优化，解耦不同类型的语音信息，为处理语义内容和说话者身份的独特特征提供了新思路

    

    近年来，大型语言模型(LLMs)的最新进展彻底改变了自然语言处理领域，逐渐拓宽了它们的范围到多模态感知和生成。然而，有效地将听觉能力整合到LLMs中会带来显著挑战，特别是在泛化跨不同语境和执行复杂听觉任务方面。在这项工作中，我们引入了WavLLM，一个具有双编码器和Prompt-aware LoRA权重适配器的稳健和自适应语音大语言模型，通过两阶段课程学习方法进行优化。利用双编码器，我们解耦不同类型的语音信息，利用Whisper编码器处理语音的语义内容，利用WavLM编码器捕捉说话者身份的独特特征。在课程学习框架内，WavLLM首先通过混合要素进行优化来建立其基础能力

    arXiv:2404.00656v1 Announce Type: cross  Abstract: The recent advancements in large language models (LLMs) have revolutionized the field of natural language processing, progressively broadening their scope to multimodal perception and generation. However, effectively integrating listening capabilities into LLMs poses significant challenges, particularly with respect to generalizing across varied contexts and executing complex auditory tasks. In this work, we introduce WavLLM, a robust and adaptive speech large language model with dual encoders, and a prompt-aware LoRA weight adapter, optimized by a two-stage curriculum learning approach. Leveraging dual encoders, we decouple different types of speech information, utilizing a Whisper encoder to process the semantic content of speech, and a WavLM encoder to capture the unique characteristics of the speaker's identity. Within the curriculum learning framework, WavLLM first builds its foundational capabilities by optimizing on mixed elemen
    
[^2]: 多跳问题回答中的检索增强知识编辑在语言模型中的应用

    Retrieval-Enhanced Knowledge Editing for Multi-Hop Question Answering in Language Models

    [https://arxiv.org/abs/2403.19631](https://arxiv.org/abs/2403.19631)

    提出了用于多跳问题回答的检索增强模型编辑（RAE）框架，利用互信息最大化的检索方法和修剪策略，实现了对语言模型的有效优化。

    

    大型语言模型（LLMs）在问题回答任务中显示出高效能，但往往难以整合实时知识更新，导致可能过时或不准确的响应。当处理多跳问题时，这个问题变得更具挑战性，因为它们要求LLMs更新和整合与问题相关的多个知识片段。为了解决这个问题，我们提出了针对多跳问题回答定制的检索增强模型编辑（RAE）框架。RAE首先检索编辑后的事实，然后通过上下文学习来完善语言模型。具体而言，我们的检索方法基于互信息最大化，利用LLMs的推理能力来识别链式事实，而天真的基于相似性的搜索可能会忽略这些事实。此外，我们的框架还采用了修剪策略，从检索到的事实中消除冗余信息，这增强了编辑

    arXiv:2403.19631v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown proficiency in question-answering tasks but often struggle to integrate real-time knowledge updates, leading to potentially outdated or inaccurate responses. This problem becomes even more challenging when dealing with multi-hop questions since they require LLMs to update and integrate multiple knowledge pieces relevant to the questions. To tackle the problem, we propose the Retrieval-Augmented model Editing (RAE) framework tailored for multi-hop question answering. RAE first retrieves edited facts and then refines the language model through in-context learning. Specifically, our retrieval approach, based on mutual information maximization, leverages the reasoning abilities of LLMs to identify chain facts that na\"ive similarity-based searches might miss. Additionally, our framework incorporates a pruning strategy to eliminate redundant information from the retrieved facts, which enhances the edi
    
[^3]: 在重叠中迷失：探索LLMs中的水印冲突

    Lost in Overlap: Exploring Watermark Collision in LLMs

    [https://arxiv.org/abs/2403.10020](https://arxiv.org/abs/2403.10020)

    本研究探讨了在大型语言模型中关于水印冲突的问题，发现双水印冲突存在时会对水印算法的检测性能造成威胁。

    

    由于大型语言模型（LLMs）在生成内容方面的普及，引发了关于文本版权的担忧。水印方法，特别是基于logit的方法，将不可察觉的标识嵌入文本中，以解决这些挑战。然而，水印方法在不同LLMs上的广泛应用导致了一种不可避免的问题，即在常见任务（如问答和改写）中发生的水印冲突。本研究关注双水印冲突，即同一文本中同时存在两个水印的情况。研究表明，水印冲突对上游和下游水印算法的检测器的检测性能构成威胁。

    arXiv:2403.10020v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread use of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks like question answering and paraphrasing. This study focuses on dual watermark collisions, where two watermarks are present simultaneously in the same text. The research demonstrates that watermark collision poses a threat to detection performance for detectors of both upstream and downstream watermark algorithms.
    
[^4]: $\texttt{COSMIC}$: 相互信息用于任务无关摘要评估

    $\texttt{COSMIC}$: Mutual Information for Task-Agnostic Summarization Evaluation

    [https://arxiv.org/abs/2402.19457](https://arxiv.org/abs/2402.19457)

    $\texttt{COSMIC}$是一种以相互信息为基础的新的摘要评估方法，有效预测下游任务表现，并与人类判断相关性强。竞争性能优于$\texttt{BERTScore}$和$\texttt{ROUGE}$。

    

    评估总结质量存在显著挑战。为此，我们提出了一种新颖的面向任务的评估方法，根据总结器生成对下游任务有用且保留任务结果的摘要能力。我们在理论上建立了这些任务的结果错误概率与源文本和生成摘要之间的相互信息之间的直接关系。我们引入了$\texttt{COSMIC}$作为这一度量的实际实现，展示了它与基于人类判断的度量之间的强相关性，以及它在预测下游任务性能方面的有效性。对已建立的度量如$\texttt{BERTScore}$和$\texttt{ROUGE}$的比较分析凸显了$\texttt{COSMIC}$的竞争性能。

    arXiv:2402.19457v1 Announce Type: cross  Abstract: Assessing the quality of summarizers poses significant challenges. In response, we propose a novel task-oriented evaluation approach that assesses summarizers based on their capacity to produce summaries that are useful for downstream tasks, while preserving task outcomes. We theoretically establish a direct relationship between the resulting error probability of these tasks and the mutual information between source texts and generated summaries. We introduce $\texttt{COSMIC}$ as a practical implementation of this metric, demonstrating its strong correlation with human judgment-based metrics and its effectiveness in predicting downstream task performance. Comparative analyses against established metrics like $\texttt{BERTScore}$ and $\texttt{ROUGE}$ highlight the competitive performance of $\texttt{COSMIC}$.
    
[^5]: 大型语言模型中的大量激活

    Massive Activations in Large Language Models

    [https://arxiv.org/abs/2402.17762](https://arxiv.org/abs/2402.17762)

    大型语言模型中出现了大量激活现象，它们具有非常大的值并且在模型中起到重要作用。

    

    我们观察到大型语言模型（LLMs）中的一个经验现象——很少的激活展现出比其他激活明显更大的值（例如，大出 100,000 倍）。我们称之为大量激活。首先，我们展示了大量激活在各种LLMs中的普遍存在，并对其位置进行了表征。其次，我们发现它们的值基本上不受输入影响，并且在LLMs中起到不可或缺的偏置项作用。第三，这些大量激活导致关注概率集中于其对应的标记，并进一步成为自注意输出中的隐式偏置项。最后，我们还研究了视觉Transformer中的大量激活。

    arXiv:2402.17762v1 Announce Type: new  Abstract: We observe an empirical phenomenon in Large Language Models (LLMs) -- very few activations exhibit significantly larger values than others (e.g., 100,000 times larger). We call them massive activations. First, we demonstrate the widespread existence of massive activations across various LLMs and characterize their locations. Second, we find their values largely stay constant regardless of the input, and they function as indispensable bias terms in LLMs. Third, these massive activations lead to the concentration of attention probabilities to their corresponding tokens, and further, implicit bias terms in the self-attention output. Last, we also study massive activations in Vision Transformers.
    
[^6]: FIPO：基于自由形式指导的提示优化与偏好数据集和模块化微调模式

    FIPO: Free-form Instruction-oriented Prompt Optimization with Preference Dataset and Modular Fine-tuning Schema

    [https://arxiv.org/abs/2402.11811](https://arxiv.org/abs/2402.11811)

    FIPO提出了基于自由形式指导的提示优化方法，结合偏好数据集和模块化微调模式，重新构思了优化过程并实现了灵活的任务提示生成。

    

    在促进大语言模型在最终用户-机器人交互中的深度智能方面，提示创作的艺术被视为普通用户的一项关键但复杂的任务。与之前基于模型而不考虑指导的自动提示优化方法形成对比，这些方法为预定义目标模型产生了光滑的结果，但在使用开箱即用模型时容易快速退化，我们提出了基于自由形式指导的提示优化（FIPO）。这种方法得到我们的大规模提示偏好数据集的支持，并采用模块化微调模式。FIPO模式重新构思了优化过程，将其分解为可管理的模块，以动态调整内容的元提示为锚点。这允许灵活整合原始任务指导、可选指导响应和可选真实值，以生成经过精心优化的任务提示。

    arXiv:2402.11811v1 Announce Type: new  Abstract: In the quest to facilitate the deep intelligence of Large Language Models (LLMs) accessible in final-end user-bot interactions, the art of prompt crafting emerges as a critical yet complex task for the average user. Contrast to previous model-oriented yet instruction-agnostic Automatic Prompt Optimization methodologies, yielding polished results for predefined target models while suffering rapid degradation with out-of-box models, we present Free-form Instruction-oriented Prompt Optimization (FIPO). This approach is supported by our large-scale prompt preference dataset and employs a modular fine-tuning schema. The FIPO schema reimagines the optimization process into manageable modules, anchored by a meta prompt that dynamically adapts content. This allows for the flexible integration of the raw task instruction, the optional instruction response, and the optional ground truth to produce finely optimized task prompts. The FIPO preference
    
[^7]: V-STaR: 自学推理器的训练方法

    V-STaR: Training Verifiers for Self-Taught Reasoners

    [https://arxiv.org/abs/2402.06457](https://arxiv.org/abs/2402.06457)

    V-STaR利用正确和不正确的解决方案训练验证器，用于选择模型生成的解决方案，实现了自我改进和验证方法在常见代码生成和数学推理任务中达到4%到17%的测试准确率提升。

    

    大型语言模型（LLM）的常见自我改进方法，例如STaR（Zelikman等人，2022），通过自动生成的解决方案迭代微调LLM以提高其问题解决能力。然而，这些方法在此过程中丢弃了大量的不正确的解决方案，可能忽略了这些解决方案中的宝贵信息。为了解决这个缺点，我们提出了V-STaR，它利用自我改进过程中生成的正确和不正确的解决方案来使用DPO训练一个判断模型生成解决方案的正确性的验证器。在推理时，这个验证器用来在众多候选解决方案中选择一个解决方案。多次运行V-STaR会逐步产生更好的推理器和验证器，在常见代码生成和数学推理基准测试中，使用LLaMA2模型可以取得4%到17%的测试准确率提升。

    Common self-improvement approaches for large language models (LLMs), such as STaR (Zelikman et al., 2022), iteratively fine-tune LLMs on self-generated solutions to improve their problem-solving ability. However, these approaches discard the large amounts of incorrect solutions generated during this process, potentially neglecting valuable information in such solutions. To address this shortcoming, we propose V-STaR that utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using DPO that judges correctness of model-generated solutions. This verifier is used at inference time to select one solution among many candidate solutions. Running V-STaR for multiple iterations results in progressively better reasoners and verifiers, delivering a 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on common code generation and math reasoning benchmarks with LLaMA2 models.
    
[^8]: LLM投票：人类选择和AI集体决策

    LLM Voting: Human Choices and AI Collective Decision Making

    [https://arxiv.org/abs/2402.01766](https://arxiv.org/abs/2402.01766)

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并与人类投票模式进行了对比。我们的方法包括进行人类投票实验以建立人类偏好的基准，并与LLM代理进行平行实验。研究聚焦于集体结果和个体偏好，揭示了人类和LLMs之间在决策和固有偏见方面的差异。我们观察到LLMs在偏好多样性和一致性之间存在权衡，相比人类选民的多样偏好，LLMs有更趋向于一致选择的倾向。这一发现表明，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    This paper investigates the voting behaviors of Large Language Models (LLMs), particularly OpenAI's GPT4 and LLaMA2, and their alignment with human voting patterns. Our approach included a human voting experiment to establish a baseline for human preferences and a parallel experiment with LLM agents. The study focused on both collective outcomes and individual preferences, revealing differences in decision-making and inherent biases between humans and LLMs. We observed a trade-off between preference diversity and alignment in LLMs, with a tendency towards more uniform choices as compared to the diverse preferences of human voters. This finding indicates that LLMs could lead to more homogenized collective outcomes when used in voting assistance, underscoring the need for cautious integration of LLMs into democratic processes.
    
[^9]: 代理指导大型语言模型成为通用的零-shot推理器

    Agent Instructs Large Language Models to be General Zero-Shot Reasoners. (arXiv:2310.03710v1 [cs.CL])

    [http://arxiv.org/abs/2310.03710](http://arxiv.org/abs/2310.03710)

    该论文提出了一种方法，通过代理指导的方式，大大提高了大型语言模型在零-shot推理任务上的能力，并在多个数据集上实现了最先进的性能。

    

    我们引入了一种方法，以提高大型语言模型在一般语言理解任务上的零-shot推理能力。具体而言，我们构建了一个自主代理，来指导大型语言模型的推理过程。我们展示了这种方法进一步释放了大型语言模型的零-shot推理能力，适用于更多的任务。我们在涵盖生成、分类和推理的广泛数据集上研究了我们方法的性能。我们展示了我们的方法适用于大多数任务，并在我们评估的29个数据集中，在20个数据集上获得了最先进的零-shot性能。例如，我们的方法显著提升了最先进的大型语言模型的性能，包括Vicuna-13b（13.3%），Llama-2-70b-chat（23.2%）和GPT-3.5 Turbo（17.0%）。与零-shot思维链相比，我们对推理的改进很明显，平均提高了10.5%。通过我们的方法，Llama-2-70b-chat的性能超过零-shot GPT-3.5 Turbo 10.2%。

    We introduce a method to improve the zero-shot reasoning abilities of large language models on general language understanding tasks. Specifically, we build an autonomous agent to instruct the reasoning process of large language models. We show this approach further unleashes the zero-shot reasoning abilities of large language models to more tasks. We study the performance of our method on a wide set of datasets spanning generation, classification, and reasoning. We show that our method generalizes to most tasks and obtains state-of-the-art zero-shot performance on 20 of the 29 datasets that we evaluate. For instance, our method boosts the performance of state-of-the-art large language models by a large margin, including Vicuna-13b (13.3%), Llama-2-70b-chat (23.2%), and GPT-3.5 Turbo (17.0%). Compared to zero-shot chain of thought, our improvement in reasoning is striking, with an average increase of 10.5%. With our method, Llama-2-70b-chat outperforms zero-shot GPT-3.5 Turbo by 10.2%.
    
[^10]: 自然语言处理中社会人口统计偏见的调查

    Survey on Sociodemographic Bias in Natural Language Processing. (arXiv:2306.08158v1 [cs.CL])

    [http://arxiv.org/abs/2306.08158](http://arxiv.org/abs/2306.08158)

    本文调查了209篇关于NLP模型偏见的论文，其中大部分涉及社会人口统计偏见。研究者提出了社会人口统计偏见的定义，并确定了NLP偏见研究的三个主要类别。当前去偏见技术只是隐藏了偏见而不是真正去除它，需要进一步改进。

    

    深度神经网络在训练过程中往往会学习到非预期的偏见，这在实际应用中可能会产生有害的影响。本文对209篇关于NLP模型中偏见的论文进行了调查，其中大部分论文涉及社会人口统计偏见。为了更好地理解偏见与真实世界的危害之间的区别，我们借鉴心理学和行为经济学的思想，提出了社会人口统计偏见的定义。我们确定了NLP偏见研究的三个主要类别：偏见类型、量化偏见和去偏见。我们认为当前对于量化偏见的方法存在可靠性问题，许多偏见度量并不涉及真实世界中的偏见，当前的去偏见技术是表面的，只是隐藏了偏见，而不是真正去除它。最后，我们提供了未来工作的建议。

    Deep neural networks often learn unintended biases during training, which might have harmful effects when deployed in real-world settings. This paper surveys 209 papers on bias in NLP models, most of which address sociodemographic bias. To better understand the distinction between bias and real-world harm, we turn to ideas from psychology and behavioral economics to propose a definition for sociodemographic bias. We identify three main categories of NLP bias research: types of bias, quantifying bias, and debiasing. We conclude that current approaches on quantifying bias face reliability issues, that many of the bias metrics do not relate to real-world biases, and that current debiasing techniques are superficial and hide bias rather than removing it. Finally, we provide recommendations for future work.
    

