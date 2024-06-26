# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions](https://arxiv.org/abs/2403.19651) | 本研究提出了MagicLens，一系列支持开放式指令的自监督图像检索模型，核心创新在于利用文本指令使得图像检索可以检索到比视觉相似性更丰富关系的图像。 |
| [^2] | [LLMs Are Few-Shot In-Context Low-Resource Language Learners](https://arxiv.org/abs/2403.16512) | 该研究对25种低资源语言和7种相对较高资源语言上的情境学习（ICL）及其跨语言变体进行了研究，发现了在低资源语言中使用LLMs进行ICL的有效性，提出了替代方法查询对齐，并为低资源语言的ICL提供了宝贵见解。 |
| [^3] | [Telecom Language Models: Must They Be Large?](https://arxiv.org/abs/2403.04666) | 小型语言模型Phi-2在电信领域展示出与大型对应模型相媲美的性能，通过检索增强生成方法提升了其能力。 |
| [^4] | [Prompting Explicit and Implicit Knowledge for Multi-hop Question Answering Based on Human Reading Process](https://arxiv.org/abs/2402.19350) | 该研究引入了一个促进显式和隐式知识的框架，用于多跳问题回答，从人类阅读过程的角度连接输入文段和预训练知识。 |
| [^5] | [A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models](https://arxiv.org/abs/2402.15422) | 本研究探讨了使用大型语言模型基于医生笔记生成患者总结的潜力，通过严格的标记协议和医学专家标记实验发现，在无幻觉数据上进行微调能有效减少幻觉的生成，并保留相关信息。 |
| [^6] | [MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues](https://arxiv.org/abs/2402.14762) | 提出了MT-Bench-101用于评估大型语言模型在多轮对话中的细粒度能力，构建了包含4208轮对话数据的三级分层能力分类，并评估了21种流行的语言模型，发现它们在不同对话轮次中表现出不同的趋势。 |
| [^7] | [Harnessing Large Language Models as Post-hoc Correctors](https://arxiv.org/abs/2402.13414) | 通过提出的无需训练的框架 LlmCorr，本文展示了一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。 |
| [^8] | [Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models](https://arxiv.org/abs/2402.11532) | 提出了一种名为指令链（CoI）的新概念，通过逐步解决每个子任务来处理由多个子任务组成的指令，进而提高了大型语言模型（LLMs）的泛化能力和多语言摘要性能 |
| [^9] | [Aligning Large Language Models by On-Policy Self-Judgment](https://arxiv.org/abs/2402.11253) | 本文提出了一个新颖的对齐框架SELF-JUDGE，通过增加式监督微调（JSFT）训练一个同时充当策略和评判器的单一模型，实现了参数高效的基于政策学习，无需额外的奖励模型。 |
| [^10] | [GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements](https://arxiv.org/abs/2402.10963) | 提出了Stepwise ORMs (SORMs)，它们在合成数据上训练，以近似预测最优策略的未来预期奖励 |
| [^11] | [Universal Prompt Optimizer for Safe Text-to-Image Generation](https://arxiv.org/abs/2402.10882) | 提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。 |
| [^12] | [DE-COP: Detecting Copyrighted Content in Language Models Training Data](https://arxiv.org/abs/2402.09910) | DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。 |
| [^13] | [MiMiC: Minimally Modified Counterfactuals in the Representation Space](https://arxiv.org/abs/2402.09631) | 提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。 |
| [^14] | [Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction](https://arxiv.org/abs/2402.02416) | Aligner是一种通过学习校正残差来实现高效对齐的方法，相比于传统的强化学习方法，Aligner具有参数高效、弱到强泛化以及即插即用的优势。 |
| [^15] | [Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge](https://arxiv.org/abs/2402.01677) | 本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。 |
| [^16] | [An Embedded Diachronic Sense Change Model with a Case Study from Ancient Greek.](http://arxiv.org/abs/2311.00541) | 本论文介绍了一个嵌入式历时语义变化模型（EDiSC），结合了词嵌入和DiSC模型，通过无监督学习分析古希腊文本中目标词汇的意义变化。实验证明EDiSC具有优越的性能。 |
| [^17] | [NExT-GPT: Any-to-Any Multimodal LLM.](http://arxiv.org/abs/2309.05519) | NExT-GPT是一个任何到任何的多模态语言模型系统，通过连接多模态适配器和不同扩散解码器，能够接受和生成任意组合的文本、图像、视频和音频内容。 |
| [^18] | [S$^3$HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering.](http://arxiv.org/abs/2305.11725) | 本文提出了一种三阶段的文本表格问答框架S3HQA，该框架采用训练精细的检索器来解决标签噪声问题，采用混合选择器来选择最相关的实际知识并采用基于生成的推理器来获取答案。在WikiTableQuestions和ComplexWebQuestions数据集上实现了最先进的性能，并在很大程度上优于以前的方法。 |

# 详细

[^1]: MagicLens：自监督图像检索与开放式指令

    MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions

    [https://arxiv.org/abs/2403.19651](https://arxiv.org/abs/2403.19651)

    本研究提出了MagicLens，一系列支持开放式指令的自监督图像检索模型，核心创新在于利用文本指令使得图像检索可以检索到比视觉相似性更丰富关系的图像。

    

    图像检索，即根据参考图像查找所需图像，固有地包含难以仅使用基于图像的度量捕捉到的丰富、多方面的搜索意图。最近的工作利用文本指令允许用户更自由地表达他们的搜索意图。然而，现有工作主要集中在那些视觉上相似和/或可以用一小组预定义关系来表征的图像对上。本文的核心论点是文本指令可以使图像检索能够检索到比视觉相似性更丰富关系的图像。为了证明这一点，我们引入了MagicLens，一系列支持开放式指令的自监督图像检索模型。MagicLens建立在一个重要的新颖见解上：自然发生在同一网页上的图像对包含着大量隐式关系（例如，内部视图），我们可以通过综合指令将这些隐式关系变为显式。

    arXiv:2403.19651v1 Announce Type: cross  Abstract: Image retrieval, i.e., finding desired images given a reference image, inherently encompasses rich, multi-faceted search intents that are difficult to capture solely using image-based measures. Recent work leverages text instructions to allow users to more freely express their search intents. However, existing work primarily focuses on image pairs that are visually similar and/or can be characterized by a small set of pre-defined relations. The core thesis of this paper is that text instructions can enable retrieving images with richer relations beyond visual similarity. To show this, we introduce MagicLens, a series of self-supervised image retrieval models that support open-ended instructions. MagicLens is built on a key novel insight: image pairs that naturally occur on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we can bring those implicit relations explicit by synthesizing instructions
    
[^2]: LLMs是少样本情境低资源语言学习器

    LLMs Are Few-Shot In-Context Low-Resource Language Learners

    [https://arxiv.org/abs/2403.16512](https://arxiv.org/abs/2403.16512)

    该研究对25种低资源语言和7种相对较高资源语言上的情境学习（ICL）及其跨语言变体进行了研究，发现了在低资源语言中使用LLMs进行ICL的有效性，提出了替代方法查询对齐，并为低资源语言的ICL提供了宝贵见解。

    

    在情境学习（ICL）的支持下，大型语言模型（LLMs）可以利用短时的情境信息执行各种任务，这为缩小高资源语言和低资源语言之间的差距提供了重要途径。然而，目前只有少数研究探讨了针对低资源语言的ICL，其中大部分集中在相对高资源的语言，比如法语和西班牙语。在这项工作中，我们对25种低资源语言和7种相对较高资源语言上的ICL及其跨语言变体（X-ICL）进行了广泛研究。我们的研究不仅评估了LLMs在低资源语言中使用ICL的有效性，还发现了情境标签对齐的缺陷，并引入了更有效的替代方法：查询对齐。此外，我们为低资源语言的ICL的各个方面提供了宝贵的见解。我们的研究总结了少样本情境学习的重要性。

    arXiv:2403.16512v1 Announce Type: cross  Abstract: In-context learning (ICL) empowers large language models (LLMs) to perform diverse tasks in underrepresented languages using only short in-context information, offering a crucial avenue for narrowing the gap between high-resource and low-resource languages. Nonetheless, there is only a handful of works explored ICL for low-resource languages with most of them focusing on relatively high-resource languages, such as French and Spanish. In this work, we extensively study ICL and its cross-lingual variation (X-ICL) on 25 low-resource and 7 relatively higher-resource languages. Our study not only assesses the effectiveness of ICL with LLMs in low-resource languages but also identifies the shortcomings of in-context label alignment, and introduces a more effective alternative: query alignment. Moreover, we provide valuable insights into various facets of ICL for low-resource languages. Our study concludes the significance of few-shot in-cont
    
[^3]: 电信语言模型：它们必须庞大吗？

    Telecom Language Models: Must They Be Large?

    [https://arxiv.org/abs/2403.04666](https://arxiv.org/abs/2403.04666)

    小型语言模型Phi-2在电信领域展示出与大型对应模型相媲美的性能，通过检索增强生成方法提升了其能力。

    

    电信部门对庞大语言模型（LLMs）的日益关注凸显了它们在改变运营效率方面的潜力。然而，部署这些复杂模型往往受到其巨大体积和计算需求的影响，引发了对它们在资源受限环境中可行性的担忧。为了解决这一挑战，最近的进展出现了一批小型语言模型，令人惊讶的是它们在许多任务中表现与其较大对应物相当，比如编码和常识推理。Phi-2是一种紧凑但功能强大的模型，它体现了这一系列高效小型语言模型的新浪潮。本文对Phi-2在电信领域内在本质上的理解进行了全面评估。鉴于规模相关限制，我们通过检索增强生成方法，精心增强了Phi-2的能力。

    arXiv:2403.04666v1 Announce Type: new  Abstract: The increasing interest in Large Language Models (LLMs) within the telecommunications sector underscores their potential to revolutionize operational efficiency. However, the deployment of these sophisticated models is often hampered by their substantial size and computational demands, raising concerns about their viability in resource-constrained environments. Addressing this challenge, recent advancements have seen the emergence of small language models that surprisingly exhibit performance comparable to their larger counterparts in many tasks, such as coding and common-sense reasoning. Phi-2, a compact yet powerful model, exemplifies this new wave of efficient small language models. This paper conducts a comprehensive evaluation of Phi-2's intrinsic understanding of the telecommunications domain. Recognizing the scale-related limitations, we enhance Phi-2's capabilities through a Retrieval-Augmented Generation approach, meticulously i
    
[^4]: 基于人类阅读过程的多跳问题回答中促进显式和隐式知识

    Prompting Explicit and Implicit Knowledge for Multi-hop Question Answering Based on Human Reading Process

    [https://arxiv.org/abs/2402.19350](https://arxiv.org/abs/2402.19350)

    该研究引入了一个促进显式和隐式知识的框架，用于多跳问题回答，从人类阅读过程的角度连接输入文段和预训练知识。

    

    预训练语言模型（PLMs）利用思维链（CoT）模拟人类推理和推断过程，实现了在多跳QA方面高效的性能。然而，当处理复杂问题时，PLMs的推理能力和人类之间仍存在差距。心理学研究表明，在阅读过程中，输入文段中的显式信息与人类先验知识之间存在重要联系。然而，当前的研究未能充分关注从人类认知研究的角度链接输入文段和基于PLMs预训练知识。在本研究中，我们引入了一个促进显式和隐式知识（PEI）框架，使用提示连接显式和隐式知识，与人类阅读过程对齐，用于多跳QA。我们将输入文段视为显式知识，利用它们通过统一提示推导隐式知识。

    arXiv:2402.19350v1 Announce Type: new  Abstract: Pre-trained language models (PLMs) leverage chains-of-thought (CoT) to simulate human reasoning and inference processes, achieving proficient performance in multi-hop QA. However, a gap persists between PLMs' reasoning abilities and those of humans when tackling complex problems. Psychological studies suggest a vital connection between explicit information in passages and human prior knowledge during reading. Nevertheless, current research has given insufficient attention to linking input passages and PLMs' pre-training-based knowledge from the perspective of human cognition studies. In this study, we introduce a \textbf{P}rompting \textbf{E}xplicit and \textbf{I}mplicit knowledge (PEI) framework, which uses prompts to connect explicit and implicit knowledge, aligning with human reading process for multi-hop QA. We consider the input passages as explicit knowledge, employing them to elicit implicit knowledge through unified prompt reason
    
[^5]: 用大型语言模型生成忠实且高质量的病人总结的数据中心方法

    A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models

    [https://arxiv.org/abs/2402.15422](https://arxiv.org/abs/2402.15422)

    本研究探讨了使用大型语言模型基于医生笔记生成患者总结的潜力，通过严格的标记协议和医学专家标记实验发现，在无幻觉数据上进行微调能有效减少幻觉的生成，并保留相关信息。

    

    患者经常面临难以理解其住院情况的困难，而医护人员资源有限以提供解释。在这项工作中，我们研究了大型语言模型基于医生笔记生成患者总结的潜力，并研究了训练数据对生成总结的忠实性和质量的影响。为此，我们开发了严格的标记协议用于幻觉，让两位医学专家标记了100个真实总结和100个生成的总结。我们展示了在无幻觉数据进行微调可以有效地减少Llama 2每个总结的幻觉从2.60降低到1.55，同时保留相关信息。虽然效果仍然存在，但当使用五个例子提示GPT-4时，该效果要小得多（0.70降至0.40）。我们还对无幻觉和改进的训练数据进行了定性评估。即使在幻觉自由数据下，GPT-4也展现出非常好的结果。

    arXiv:2402.15422v1 Announce Type: cross  Abstract: Patients often face difficulties in understanding their hospitalizations, while healthcare workers have limited resources to provide explanations. In this work, we investigate the potential of large language models to generate patient summaries based on doctors' notes and study the effect of training data on the faithfulness and quality of the generated summaries. To this end, we develop a rigorous labeling protocol for hallucinations, and have two medical experts annotate 100 real-world summaries and 100 generated summaries. We show that fine-tuning on hallucination-free data effectively reduces hallucinations from 2.60 to 1.55 per summary for Llama 2, while preserving relevant information. Although the effect is still present, it is much smaller for GPT-4 when prompted with five examples (0.70 to 0.40). We also conduct a qualitative evaluation using hallucination-free and improved training data. GPT-4 shows very good results even in 
    
[^6]: MT-Bench-101: 用于评估大型语言模型在多轮对话中的细粒度基准

    MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues

    [https://arxiv.org/abs/2402.14762](https://arxiv.org/abs/2402.14762)

    提出了MT-Bench-101用于评估大型语言模型在多轮对话中的细粒度能力，构建了包含4208轮对话数据的三级分层能力分类，并评估了21种流行的语言模型，发现它们在不同对话轮次中表现出不同的趋势。

    

    大型语言模型（LLMs）的出现大大增强了对话系统。然而，全面评估LLMs的对话能力仍然是一个挑战。以往的基准主要集中在单轮对话或者提供粗粒度和不完整的多轮对话评估，忽视了真实对话的复杂性和细微的差异。为了解决这个问题，我们引入了MT-Bench-101，专门设计用于评估LLMs在多轮对话中的细粒度能力。通过对真实多轮对话数据进行详细分析，我们构建了一个包含13个不同任务中1388个多轮对话中的4208轮的三级分层能力分类。然后我们基于MT-Bench-101评估了21个流行的LLMs，从能力和任务两个角度进行全面分析，并观察到LLMs在对话轮次中表现出不同的趋势。

    arXiv:2402.14762v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has drastically enhanced dialogue systems. However, comprehensively evaluating the dialogue abilities of LLMs remains a challenge. Previous benchmarks have primarily focused on single-turn dialogues or provided coarse-grained and incomplete assessments of multi-turn dialogues, overlooking the complexity and fine-grained nuances of real-life dialogues. To address this issue, we introduce MT-Bench-101, specifically designed to evaluate the fine-grained abilities of LLMs in multi-turn dialogues. By conducting a detailed analysis of real multi-turn dialogue data, we construct a three-tier hierarchical ability taxonomy comprising 4208 turns across 1388 multi-turn dialogues in 13 distinct tasks. We then evaluate 21 popular LLMs based on MT-Bench-101, conducting comprehensive analyses from both ability and task perspectives and observing differing trends in LLMs performance across dialogue turns with
    
[^7]: 将大型语言模型用作事后校正器

    Harnessing Large Language Models as Post-hoc Correctors

    [https://arxiv.org/abs/2402.13414](https://arxiv.org/abs/2402.13414)

    通过提出的无需训练的框架 LlmCorr，本文展示了一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。

    

    随着机器学习（ML）模型的规模增长并需求更高质量的训练数据，与对这些模型进行重新训练和微调相关的费用正在迅速增加。受最近大型语言模型（LLMs）在不同领域取得的令人瞩目成就启发，本文探讨了一个问题：LLMs能否以极低成本有效地改善ML的性能？我们展示了，通过我们提出的无需训练的框架 LlmCorr，一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。特别是，我们通过整合数据集的标签信息和ML模型对验证集的预测来形成一个上下文知识数据库。利用LLMs的上下文学习能力，我们要求LLM总结ML模型犯错误的实例以及主要预测与真实标签之间的相关性。随后，LLM可以

    arXiv:2402.13414v1 Announce Type: cross  Abstract: As Machine Learning (ML) models grow in size and demand higher-quality training data, the expenses associated with re-training and fine-tuning these models are escalating rapidly. Inspired by recent impressive achievements of Large Language Models (LLMs) in different fields, this paper delves into the question: can LLMs efficiently improve an ML's performance at a minimal cost? We show that, through our proposed training-free framework LlmCorr, an LLM can work as a post-hoc corrector to propose corrections for the predictions of an arbitrary ML model. In particular, we form a contextual knowledge database by incorporating the dataset's label information and the ML model's predictions on the validation dataset. Leveraging the in-context learning capability of LLMs, we ask the LLM to summarise the instances in which the ML model makes mistakes and the correlation between primary predictions and true labels. Following this, the LLM can tr
    
[^8]: 指令链：大型语言模型的组合指令调整

    Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models

    [https://arxiv.org/abs/2402.11532](https://arxiv.org/abs/2402.11532)

    提出了一种名为指令链（CoI）的新概念，通过逐步解决每个子任务来处理由多个子任务组成的指令，进而提高了大型语言模型（LLMs）的泛化能力和多语言摘要性能

    

    使用一系列大型和多样化的指令对大型语言模型（LLMs）进行微调，提高了模型对不同任务的泛化能力，甚至对未曾见过的任务也适用。本研究提出了一种称为指令链（CoI）的新概念，其中一个指令的输出成为下一个指令的输入，就像一条链条。与解决单一指令任务的传统做法不同，我们提出的方法鼓励模型逐步解决每个子任务，直至得出最终答案。CoI调整（即使用CoI指令进行微调）提高了模型处理由多个子任务组成的指令能力。经CoI调整的模型在多语言摘要上也优于基准模型，证明....

    arXiv:2402.11532v1 Announce Type: new  Abstract: Fine-tuning large language models (LLMs) with a collection of large and diverse instructions has improved the model's generalization to different tasks, even for unseen tasks. However, most existing instruction datasets include only single instructions, and they struggle to follow complex instructions composed of multiple subtasks (Wang et al., 2023a). In this work, we propose a novel concept of compositional instructions called chain-of-instructions (CoI), where the output of one instruction becomes an input for the next like a chain. Unlike the conventional practice of solving single instruction tasks, our proposed method encourages a model to solve each subtask step by step until the final answer is reached. CoI-tuning (i.e., fine-tuning with CoI instructions) improves the model's ability to handle instructions composed of multiple subtasks. CoI-tuned models also outperformed baseline models on multilingual summarization, demonstratin
    
[^9]: 通过基于政策的自我判断来对齐大型语言模型

    Aligning Large Language Models by On-Policy Self-Judgment

    [https://arxiv.org/abs/2402.11253](https://arxiv.org/abs/2402.11253)

    本文提出了一个新颖的对齐框架SELF-JUDGE，通过增加式监督微调（JSFT）训练一个同时充当策略和评判器的单一模型，实现了参数高效的基于政策学习，无需额外的奖励模型。

    

    为了使大型语言模型与人类偏好保持一致，现有研究要么利用单独的奖励模型（RM）执行基于政策的学习，要么通过放弃基于政策的学习和对独立RM的需求简化训练过程。在本文中，我们提出了一个新颖的对齐框架SELF-JUDGE，它既是(1) 基于政策的学习，又是(2) 参数高效的，因为它不需要额外的RM来评估样本进行基于政策的学习。为此，我们提出了增强式监督微调（JSFT）来训练一个单一模型，作为策略和评判器。具体来说，我们将一对一判断任务视为指导式任务的特殊情况，从响应对中选择更好的响应。因此，得到的模型可以评判当前策略的即时响应偏好，从自身初始化。实验结果显示了SELF-JUDGE的有效性，优于基线模型。

    arXiv:2402.11253v1 Announce Type: cross  Abstract: To align large language models with human preferences, existing research either utilizes a separate reward model (RM) to perform on-policy learning or simplifies the training procedure by discarding the on-policy learning and the need for a separate RM. In this paper, we present a novel alignment framework, SELF-JUDGE that is (1) on-policy learning and 2) parameter efficient, as it does not require an additional RM for evaluating the samples for on-policy learning. To this end, we propose Judge-augmented Supervised Fine-Tuning (JSFT) to train a single model acting as both a policy and a judge. Specifically, we view the pairwise judgment task as a special case of the instruction-following task, choosing the better response from a response pair. Thus, the resulting model can judge preferences of on-the-fly responses from current policy initialized from itself. Experimental results show the efficacy of SELF-JUDGE, outperforming baselines 
    
[^10]: GLoRe: 何时、何地以及如何通过全局和局部的改进来提高LLM推理能力

    GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements

    [https://arxiv.org/abs/2402.10963](https://arxiv.org/abs/2402.10963)

    提出了Stepwise ORMs (SORMs)，它们在合成数据上训练，以近似预测最优策略的未来预期奖励

    

    最先进的语言模型在数学、科学或编码任务中展现出令人印象深刻的推理改进能力。然而，最近的研究表明，即使最好的模型也很难在没有外部反馈的情况下确定何时何地进行改进。基于结果的奖励模型(ORMs)，被训练来预测最终答案的正确性，指示何时进行改进，为决定何时进行改进提供了一种便利的解决方案。基于过程的奖励模型(PRMs)受过训练，用以预测中间步骤的正确性，然后可以用来指示何处进行改进。但它们很昂贵，需要大量的人工注释。在本文中，我们提出了逐步ORMs(SORMs)，它们只在合成数据上受过训练，以近似预测最优策略或$V^{\star}$的未来预期奖励。更具体地说，SORMs受训练来预测当取样时最终答案的正确性

    arXiv:2402.10963v1 Announce Type: new  Abstract: State-of-the-art language models can exhibit impressive reasoning refinement capabilities on math, science or coding tasks. However, recent work demonstrates that even the best models struggle to identify \textit{when and where to refine} without access to external feedback. Outcome-based Reward Models (\textbf{ORMs}), trained to predict correctness of the final answer indicating when to refine, offer one convenient solution for deciding when to refine. Process Based Reward Models (\textbf{PRMs}), trained to predict correctness of intermediate steps, can then be used to indicate where to refine. But they are expensive to train, requiring extensive human annotations. In this paper, we propose Stepwise ORMs (\textbf{SORMs}) which are trained, only on synthetic data, to approximate the expected future reward of the optimal policy or $V^{\star}$. More specifically, SORMs are trained to predict the correctness of the final answer when samplin
    
[^11]: 通用提示优化器用于安全文本到图像生成

    Universal Prompt Optimizer for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2402.10882](https://arxiv.org/abs/2402.10882)

    提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。

    

    文本到图像（T2I）模型在根据文字提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，从而生成不安全内容，如色情、骚扰和非法活动图像。基于图像检查器、模型微调和嵌入式阻止的现有研究在真实世界应用中不可行。因此，我们提出了第一个用于黑盒场景中安全 T2I 生成的通用提示优化器。

    arXiv:2402.10882v1 Announce Type: cross  Abstract: Text-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, \textit{we propose the first universal prompt optimizer for safe T2I generation in black-box scenario}. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating in
    
[^12]: 在语言模型训练数据中检测版权内容的方法：DE-COP

    DE-COP: Detecting Copyrighted Content in Language Models Training Data

    [https://arxiv.org/abs/2402.09910](https://arxiv.org/abs/2402.09910)

    DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。

    

    在考虑到训练数据通常是保密的情况下，我们如何检测语言模型的训练过程中是否使用了版权内容？我们的动机是基于一个语言模型很可能能够识别出其训练文本中的独文摘录。我们提出了一种称为DE-COP的方法，用于确定是否在训练中包含了一段版权内容。DE-COP的核心方法是通过多项选择问题对语言模型进行探测，选择项包括独文本和它们的释义。我们构建了一个基准数据集BookTection，其中包含了在模型训练截止日期之前和之后出版的165本书的摘录以及它们的释义。实验证明，DE-COP在模型的逻辑可用时，检测性能（AUC）超过之前的最佳方法9.6%。此外，DE-COP在完全黑盒模型上检测可疑书籍的平均准确率达到72%，而之前的方法只有$。

    arXiv:2402.09910v1 Announce Type: new  Abstract: How can we detect if copyrighted content was used in the training process of a language model, considering that the training data is typically undisclosed? We are motivated by the premise that a language model is likely to identify verbatim excerpts from its training text. We propose DE-COP, a method to determine whether a piece of copyrighted content was included in training. DE-COP's core approach is to probe an LLM with multiple-choice questions, whose options include both verbatim text and their paraphrases. We construct BookTection, a benchmark with excerpts from 165 books published prior and subsequent to a model's training cutoff, along with their paraphrases. Our experiments show that DE-COP surpasses the prior best method by 9.6% in detection performance (AUC) on models with logits available. Moreover, DE-COP also achieves an average accuracy of 72% for detecting suspect books on fully black-box models where prior methods give $
    
[^13]: MiMiC：表示空间中最小修改的对抗事实

    MiMiC: Minimally Modified Counterfactuals in the Representation Space

    [https://arxiv.org/abs/2402.09631](https://arxiv.org/abs/2402.09631)

    提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。

    

    arXiv:2402.09631v1 公告类型：交叉学科 简介：语言模型经常表现出不良行为，如性别偏见或有毒语言。通过对表示空间进行干预，可以有效减轻这些问题，但两种常见的干预技术，即线性擦除和定向向量，并不能提供高度可控和表达丰富度。因此，我们提出了一种新颖的干预方法，旨在在表示空间中生成富有表达力的对抗事实，使源类别（例如“有毒”）的表示与目标类别（例如“非有毒”）的表示相似。这种方法利用高斯假设下的闭式解决方案，在地球移动问题方面提供了理论上的保证，并对表示空间的几何组织提供了进一步的改进。

    arXiv:2402.09631v1 Announce Type: cross  Abstract: Language models often exhibit undesirable behaviors, such as gender bias or toxic language. Interventions in the representation space were shown effective in mitigating such issues by altering the LM behavior. We first show that two prominent intervention techniques, Linear Erasure and Steering Vectors, do not enable a high degree of control and are limited in expressivity.   We then propose a novel intervention methodology for generating expressive counterfactuals in the representation space, aiming to make representations of a source class (e.g., ``toxic'') resemble those of a target class (e.g., ``non-toxic''). This approach, generalizing previous linear intervention techniques, utilizes a closed-form solution for the Earth Mover's problem under Gaussian assumptions and provides theoretical guarantees on the representation space's geometric organization. We further build on this technique and derive a nonlinear intervention that ena
    
[^14]: Aligner: 通过弱到强校正实现高效对齐

    Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction

    [https://arxiv.org/abs/2402.02416](https://arxiv.org/abs/2402.02416)

    Aligner是一种通过学习校正残差来实现高效对齐的方法，相比于传统的强化学习方法，Aligner具有参数高效、弱到强泛化以及即插即用的优势。

    

    对于大型语言模型（LLMs），通过强化学习来进行对齐的努力主要是通过人类反馈的强化学习方法进行的。然而，强化学习面临着主要的挑战，包括训练奖励模型、演员-评论家工程以及重要的是，需要访问LLM参数。在这里，我们介绍了一种新的高效对齐范式Aligner，它通过学习对齐和未对齐答案之间的校正残差来绕过整个强化学习过程。我们的Aligner具有几个关键优势。首先，它是一个基于自监督学习的自动回归seq2seq模型，通过训练查询-答案-校正数据集，提供了一种参数高效的对齐解决方案，并且对资源需求较少。其次，Aligner实现了从弱到强的泛化；通过Aligner的监督信号来微调大型预训练模型，可以显著提升性能。第三，Aligner作为一个模型不可知的即插即用模块，可以直接应用于…

    Efforts to align Large Language Models (LLMs) are mainly conducted via Reinforcement Learning from Human Feedback (RLHF) methods. However, RLHF encounters major challenges including training reward models, actor-critic engineering, and importantly, it requires access to LLM parameters. Here we introduce Aligner, a new efficient alignment paradigm that bypasses the whole RLHF process by learning the correctional residuals between the aligned and the unaligned answers. Our Aligner offers several key advantages. Firstly, it is an autoregressive seq2seq model that is trained on the query-answer-correction dataset via supervised learning; this offers a parameter-efficient alignment solution with minimal resources. Secondly, the Aligner facilitates weak-to-strong generalization; finetuning large pretrained models by Aligner's supervisory signals demonstrates strong performance boost. Thirdly, Aligner functions as a model-agnostic plug-and-play module, allowing for its direct application on d
    
[^15]: 通过整合外延知识和内涵知识嵌入本体

    Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge

    [https://arxiv.org/abs/2402.01677](https://arxiv.org/abs/2402.01677)

    本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。

    

    本体包含领域内丰富的知识，可以分为两个类别，即外延知识和内涵知识。外延知识提供关于本体中特定概念所属的具体实例的信息，而内涵知识详细描述了概念之间的内在属性、特征和语义关联。然而，现有的本体嵌入方法未能同时充分考虑外延知识和内涵知识。在本文中，我们提出了一种名为EIKE（Extensional and Intensional Knowledge Embedding）的新型本体嵌入方法，通过在外延空间和内涵空间中表示本体。EIKE提出了一个统一的框架，用于将实例、概念及其关系嵌入到本体中，采用基于几何的方法对外延知识进行建模，并使用预训练的语言模型对内涵知识进行建模。

    Ontologies contain rich knowledge within domain, which can be divided into two categories, namely extensional knowledge and intensional knowledge. Extensional knowledge provides information about the concrete instances that belong to specific concepts in the ontology, while intensional knowledge details inherent properties, characteristics, and semantic associations among concepts. However, existing ontology embedding approaches fail to take both extensional knowledge and intensional knowledge into fine consideration simultaneously. In this paper, we propose a novel ontology embedding approach named EIKE (Extensional and Intensional Knowledge Embedding) by representing ontologies in two spaces, called extensional space and intensional space. EIKE presents a unified framework for embedding instances, concepts and their relations in an ontology, applying a geometry-based method to model extensional knowledge and a pretrained language model to model intensional knowledge, which can captur
    
[^16]: 一个带有嵌入式历时语义变化模型的论文与一个关于古希腊的案例研究

    An Embedded Diachronic Sense Change Model with a Case Study from Ancient Greek. (arXiv:2311.00541v1 [cs.CL])

    [http://arxiv.org/abs/2311.00541](http://arxiv.org/abs/2311.00541)

    本论文介绍了一个嵌入式历时语义变化模型（EDiSC），结合了词嵌入和DiSC模型，通过无监督学习分析古希腊文本中目标词汇的意义变化。实验证明EDiSC具有优越的性能。

    

    词汇的意义随着时间的推移而变化，词义在这个过程中会演变、出现或消失。对于古代语言来说，由于语料库通常较小、稀疏且嘈杂，准确建模这种变化变得具有挑战性，因此对于意义变化估计的不确定性进行量化变得重要。GASC和DiSC是现有的生成模型，已经被用来分析古希腊文本语料库中目标词汇的意义变化，使用了无监督学习并没有借助任何预训练的帮助。这些模型将给定目标词汇（如"kosmos"，意为装饰、秩序或世界）的意义表示为上下文词汇的分布，并将意义的普遍性表示为意义的分布。这些模型使用马尔科夫链蒙特卡洛方法进行拟合，以测量这些表示中的时间变化。在本文中，我们介绍了EDiSC，这是DiSC的嵌入版本，它将词嵌入与DiSC相结合，提供了更优秀的模型性能。我们通过实验证明，EDiSC提供了改进的性能。

    Word meanings change over time, and word senses evolve, emerge or die out in the process. For ancient languages, where the corpora are often small, sparse and noisy, modelling such changes accurately proves challenging, and quantifying uncertainty in sense-change estimates consequently becomes important. GASC and DiSC are existing generative models that have been used to analyse sense change for target words from an ancient Greek text corpus, using unsupervised learning without the help of any pre-training. These models represent the senses of a given target word such as "kosmos" (meaning decoration, order or world) as distributions over context words, and sense prevalence as a distribution over senses. The models are fitted using MCMC methods to measure temporal changes in these representations. In this paper, we introduce EDiSC, an embedded version of DiSC, which combines word embeddings with DiSC to provide superior model performance. We show empirically that EDiSC offers improved p
    
[^17]: NExT-GPT: 任何到任何的多模态语言模型

    NExT-GPT: Any-to-Any Multimodal LLM. (arXiv:2309.05519v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.05519](http://arxiv.org/abs/2309.05519)

    NExT-GPT是一个任何到任何的多模态语言模型系统，通过连接多模态适配器和不同扩散解码器，能够接受和生成任意组合的文本、图像、视频和音频内容。

    

    最近，多模态大型语言模型（MM-LLM）取得了令人振奋的进展，但它们主要存在一个限制，即只能在输入端进行多模态理解，无法以多种模式生成内容。由于我们人类总是通过各种模态感知世界和与人交流，因此开发能够接受和传递任何模态内容的任何到任何的MM-LLM系统对于实现人级AI至关重要。为了填补这一空白，我们提出了一个端到端的通用任何到任何的多模态语言模型系统，NExT-GPT。我们通过连接一个含有多模态适配器和不同扩散解码器的LLM，使得NExT-GPT能够以任意的文本、图像、视频和音频的组合进行输入和输出。通过利用现有训练有素的高性能编码器和解码器，NExT-GPT仅通过调整某些投影层的少量参数（1%）进行调优，这不仅有利于低成本训练，还有助于方便的扩展性。

    While recently Multimodal Large Language Models (MM-LLMs) have made exciting strides, they mostly fall prey to the limitation of only input-side multimodal understanding, without the ability to produce content in multiple modalities. As we humans always perceive the world and communicate with people through various modalities, developing any-to-any MM-LLMs capable of accepting and delivering content in any modality becomes essential to human-level AI. To fill the gap, we present an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT. We connect an LLM with multimodal adaptors and different diffusion decoders, enabling NExT-GPT to perceive inputs and generate outputs in arbitrary combinations of text, images, videos, and audio. By leveraging the existing well-trained highly-performing encoders and decoders, NExT-GPT is tuned with only a small amount of parameter (1%) of certain projection layers, which not only benefits low-cost training and also facilitates convenient expansi
    
[^18]: S3HQA：一种用于文本-表格混合问答的三阶段框架

    S$^3$HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering. (arXiv:2305.11725v1 [cs.CL])

    [http://arxiv.org/abs/2305.11725](http://arxiv.org/abs/2305.11725)

    本文提出了一种三阶段的文本表格问答框架S3HQA，该框架采用训练精细的检索器来解决标签噪声问题，采用混合选择器来选择最相关的实际知识并采用基于生成的推理器来获取答案。在WikiTableQuestions和ComplexWebQuestions数据集上实现了最先进的性能，并在很大程度上优于以前的方法。

    

    回答涉及文本和表格混合事实知识的多跳问题(TextTableQA)是一项具有挑战性的任务。现有模型主要采用检索器-阅读器框架，存在几个问题，如训练检索器的嘈杂标签、对文本和表格的异构信息利用不足以及不同推理操作能力不足。本文提出了一个三阶段文本表格问答框架S3HQA，包括检索器、选择器和推理器。我们采用一个训练精细的检索器来解决嘈杂标签的问题。然后，使用一个混合选择器来考虑异构数据之间的链接关系，以选择最相关的实际知识。在最后一个阶段，我们采用一个基于生成的推理器来获取答案，而不是像之前的方法一样使用阅读理解模块。其中包括两种方法：一种是按行生成器，一种是LLM提示生成器（在这个任务中第一次使用）。在WikiTableQuestions和ComplexWebQuestions数据集上的实验结果显示，我们的S3HQA实现了最先进的性能，并在很大程度上优于以前的方法。

    Answering multi-hop questions over hybrid factual knowledge from the given text and table (TextTableQA) is a challenging task. Existing models mainly adopt a retriever-reader framework, which have several deficiencies, such as noisy labeling in training retriever, insufficient utilization of heterogeneous information over text and table, and deficient ability for different reasoning operations. In this paper, we propose a three-stage TextTableQA framework S3HQA, which comprises of retriever, selector, and reasoner. We use a retriever with refinement training to solve the noisy labeling problem. Then, a hybrid selector considers the linked relationships between heterogeneous data to select the most relevant factual knowledge. For the final stage, instead of adapting a reading comprehension module like in previous methods, we employ a generation-based reasoner to obtain answers. This includes two approaches: a row-wise generator and an LLM prompting generator~(first time used in this tas
    

