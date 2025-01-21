# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models](https://arxiv.org/abs/2403.20262) | 该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。 |
| [^2] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^3] | [Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training](https://arxiv.org/abs/2403.03728) | 通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。 |
| [^4] | [Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates](https://arxiv.org/abs/2402.18540) | 提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。 |
| [^5] | [Two Types of AI Existential Risk: Decisive and Accumulative](https://arxiv.org/abs/2401.07836) | 本文对比了传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”，指出人工智能可能带来的灭绝性灾难有两种可能路径：一种是突然发生的AI接管，另一种是逐渐积累的威胁。 |
| [^6] | [A Systematic Evaluation of Large Language Models on Out-of-Distribution Logical Reasoning Tasks.](http://arxiv.org/abs/2310.09430) | 通过对大型语言模型在非分布式逻辑推理任务上进行系统评估，我们发现这些模型在处理我们新构建的数据集时都存在困难，尽管它们在其他自然语言处理任务上表现良好。这表明这些模型在逻辑推理方面的泛化和鲁棒性仍需要进一步研究。 |
| [^7] | [Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models.](http://arxiv.org/abs/2309.10444) | 本文提出了一个自我强化大型语言模型框架，自动生成和评估学生生成的解释，用于改进学生资源共享中学生生成的多项选择题的解释质量。 |
| [^8] | [Large Process Models: Business Process Management in the Age of Generative AI.](http://arxiv.org/abs/2309.00900) | 大型过程模型（LPM）结合了大规模信息语料库和基于知识系统的方法的优势，旨在为组织提供过程建议和优化方案。 |
| [^9] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |
| [^10] | [Automated Machine Learning for Remaining Useful Life Predictions.](http://arxiv.org/abs/2306.12215) | 本文介绍了一种自动化的机器学习方法，名为AutoRUL，用于自动预测工程系统的剩余使用寿命（RUL）。该方法将微调的标准回归方法与高预测能力的集成相结合，并通过八个真实世界的和合成数据集的评估，证明AutoML提供了一种可行的选择。 |
| [^11] | [Can AI-Generated Text be Reliably Detected?.](http://arxiv.org/abs/2303.11156) | 本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。 |

# 详细

[^1]: ELITR-Bench: 面向长上下文语言模型的会议助理基准

    ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models

    [https://arxiv.org/abs/2403.20262](https://arxiv.org/abs/2403.20262)

    该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。

    

    最近，对大型语言模型（LLMs）的研究越来越受到关注，主要致力于扩展模型的上下文大小，以更好地捕捉长文档内部的依赖关系。尽管已经提出了用于评估长距离能力的基准，但现有的努力主要考虑的是不一定与现实应用相关的通用任务。相反，我们的工作提出了一个针对实际会议助理场景的长上下文LLMs的新基准。在这种情景下，长上下文由自动语音识别获得的转录组成，由于这些数据的固有嘈杂性和口语特性，这为LLMs提出了独特的挑战。我们的基准，名为ELITR-Bench，通过271个手工制作的问题及其真实答案来增强现有的ELITR语料库的转录。我们在ELITR-Bench上对最新的长上下文LLMs进行的实验凸显了开源模型和专有模型之间的差距。

    arXiv:2403.20262v1 Announce Type: cross  Abstract: Research on Large Language Models (LLMs) has recently witnessed an increasing interest in extending models' context size to better capture dependencies within long documents. While benchmarks have been proposed to assess long-range abilities, existing efforts primarily considered generic tasks that are not necessarily aligned with real-world applications. In contrast, our work proposes a new benchmark for long-context LLMs focused on a practical meeting assistant scenario. In this scenario, the long contexts consist of transcripts obtained by automatic speech recognition, presenting unique challenges for LLMs due to the inherent noisiness and oral nature of such data. Our benchmark, named ELITR-Bench, augments the existing ELITR corpus' transcripts with 271 manually crafted questions and their ground-truth answers. Our experiments with recent long-context LLMs on ELITR-Bench highlight a gap between open-source and proprietary models, e
    
[^2]: 与人类判断相一致：大型语言模型评估中成对偏好的作用

    Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

    [https://arxiv.org/abs/2403.16950](https://arxiv.org/abs/2403.16950)

    在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。

    

    大型语言模型（LLMs）作为自动评估器在评估生成的自然语言质量方面表现出有希望的能力。然而，LLMs在评估中仍存在偏见，常常难以生成与人类评估一致的连贯评估。在这项工作中，我们首先对LLM评估器与人类判断之间的不一致进行系统研究，揭示现有旨在减轻偏见的校准方法不足以有效将LLM评估器对齐。受到RLHF中对偏好数据的使用的启发，我们将评估形式化为一个排序问题，并引入Pairwise-preference Search（PAIRS），这是一种以LLMs进行成对比较并有效对候选文本进行排序的基于不确定性引导的搜索方法。PAIRS在代表性评估任务上实现了最先进的性能，并且显示出比直接打分有显著改进。

    arXiv:2403.16950v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated promising capabilities as automatic evaluators in assessing the quality of generated natural language. However, LLMs still exhibit biases in evaluation and often struggle to generate coherent evaluations that align with human assessments. In this work, we first conduct a systematic study of the misalignment between LLM evaluators and human judgement, revealing that existing calibration methods aimed at mitigating biases are insufficient for effectively aligning LLM evaluators. Inspired by the use of preference data in RLHF, we formulate the evaluation as a ranking problem and introduce Pairwise-preference Search (PAIRS), an uncertainty-guided search method that employs LLMs to conduct pairwise comparisons and efficiently ranks candidate texts. PAIRS achieves state-of-the-art performance on representative evaluation tasks and demonstrates significant improvements over direct scoring. Furthe
    
[^3]: 通过自监督预训练在主动学习中弥合多样性与不确定性

    Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training

    [https://arxiv.org/abs/2403.03728](https://arxiv.org/abs/2403.03728)

    通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。

    

    本研究探讨了在主动学习中集成基于多样性和基于不确定性的采样策略，特别是在自监督预训练模型的背景下。我们引入了一个称为TCM的简单启发式方法，可以缓解冷启动问题，同时在各种数据水平上保持强大性能。通过首先应用TypiClust进行多样性采样，随后过渡到使用Margin进行不确定性采样，我们的方法有效地结合了两种策略的优势。我们的实验表明，TCM在低数据和高数据情况下始终优于现有方法。

    arXiv:2403.03728v1 Announce Type: cross  Abstract: This study addresses the integration of diversity-based and uncertainty-based sampling strategies in active learning, particularly within the context of self-supervised pre-trained models. We introduce a straightforward heuristic called TCM that mitigates the cold start problem while maintaining strong performance across various data levels. By initially applying TypiClust for diversity sampling and subsequently transitioning to uncertainty sampling with Margin, our approach effectively combines the strengths of both strategies. Our experiments demonstrate that TCM consistently outperforms existing methods across various datasets in both low and high data regimes.
    
[^4]: 在微调后保持LLMs的对齐性:提示模板的关键作用

    Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates

    [https://arxiv.org/abs/2402.18540](https://arxiv.org/abs/2402.18540)

    提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。

    

    公共LLMs，如Llama 2-Chat，推动了LLM研究的巨大活动。这些模型经历了对齐性训练，被认为是安全的。最近，齐等人（2023年）报告称，即使是良性的微调（例如，在看似安全的数据集上）也可能导致模型产生不安全的行为。本文介绍了减轻这种对齐性丢失的方法和最佳实践。通过对几个聊天模型（Meta的Llama 2-Chat，Mistral AI的Mistral 7B Instruct v0.2和OpenAI的GPT-3.5 Turbo）进行广泛实验，本文发现微调和推理过程中使用的提示模板在保持安全对齐性方面起着至关重要的作用，并提出了“纯粹调优，安全测试”（PTST）原则 - 在测试时不使用安全提示进行模型微调，但在测试时包含它。对GSM8K，ChatDoctor和OpenOrca进行的微调实验表明，PTST显着减少了不安全行为的增加，甚至几乎消除了它们。

    arXiv:2402.18540v1 Announce Type: cross  Abstract: Public LLMs such as the Llama 2-Chat have driven huge activity in LLM research. These models underwent alignment training and were considered safe. Recently Qi et al. (2023) reported that even benign fine-tuning (e.g., on seemingly safe datasets) can give rise to unsafe behaviors in the models. The current paper is about methods and best practices to mitigate such loss of alignment. Through extensive experiments on several chat models (Meta's Llama 2-Chat, Mistral AI's Mistral 7B Instruct v0.2, and OpenAI's GPT-3.5 Turbo), this paper uncovers that the prompt templates used during fine-tuning and inference play a crucial role in preserving safety alignment, and proposes the "Pure Tuning, Safe Testing" (PTST) principle -- fine-tune models without a safety prompt, but include it at test time. Fine-tuning experiments on GSM8K, ChatDoctor, and OpenOrca show that PTST significantly reduces the rise of unsafe behaviors, and even almost elimin
    
[^5]: 两种类型的人工智能存在风险：决定性和累积性

    Two Types of AI Existential Risk: Decisive and Accumulative

    [https://arxiv.org/abs/2401.07836](https://arxiv.org/abs/2401.07836)

    本文对比了传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”，指出人工智能可能带来的灭绝性灾难有两种可能路径：一种是突然发生的AI接管，另一种是逐渐积累的威胁。

    

    传统上对人工智能(AI)引起的存在风险(x-risks)的讨论通常集中在由先进的AI系统引起的突然、严重事件上，尤其是那些可能达到或超过人类水平智能的系统。这些事件将带来严重后果，要么导致人类灭绝，要么无法逆转地使人类文明陷入无法恢复的状态。然而，这种讨论经常忽视AI x-risk逐渐通过一系列较小但相互关联的中断逐渐显现出来的严重可能性，随着时间的推移逐渐跨越关键阈值。该论文将传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”进行对比。前者描绘了一种明显的AI接管路径，其特征是无法控制的超级智能等情景，而后者则提出了另一种导致灭绝性灾难的因果路径。这涉及到由AI引起的严重威胁的逐渐累积，例如严重的漏洞和系统性问题

    The conventional discourse on existential risks (x-risks) from AI typically focuses on abrupt, dire events caused by advanced AI systems, particularly those that might achieve or surpass human-level intelligence. These events have severe consequences that either lead to human extinction or irreversibly cripple human civilization to a point beyond recovery. This discourse, however, often neglects the serious possibility of AI x-risks manifesting incrementally through a series of smaller yet interconnected disruptions, gradually crossing critical thresholds over time. This paper contrasts the conventional "decisive AI x-risk hypothesis" with an "accumulative AI x-risk hypothesis." While the former envisions an overt AI takeover pathway, characterized by scenarios like uncontrollable superintelligence, the latter suggests a different causal pathway to existential catastrophes. This involves a gradual accumulation of critical AI-induced threats such as severe vulnerabilities and systemic e
    
[^6]: 对大型语言模型在非分布式逻辑推理任务上的系统评估

    A Systematic Evaluation of Large Language Models on Out-of-Distribution Logical Reasoning Tasks. (arXiv:2310.09430v1 [cs.CL])

    [http://arxiv.org/abs/2310.09430](http://arxiv.org/abs/2310.09430)

    通过对大型语言模型在非分布式逻辑推理任务上进行系统评估，我们发现这些模型在处理我们新构建的数据集时都存在困难，尽管它们在其他自然语言处理任务上表现良好。这表明这些模型在逻辑推理方面的泛化和鲁棒性仍需要进一步研究。

    

    大型语言模型（LLMs），如GPT-3.5和GPT-4，已经将人工系统在各种自然语言处理任务上的性能提升到接近人类水平。然而，它们在逻辑推理方面的泛化和鲁棒性仍未得到充分评估。为了探索这种能力，我们提出了三个新的逻辑推理数据集，分别名为"ReClor-plus"、"LogiQA-plus"和"LogiQAv2-plus"，每个数据集都包含三个子集：第一个是选项随机打乱，第二个是将正确选项替换为"没有其他选项是正确的"，第三个是前两个子集的组合。我们在这些数据集上进行了实验，使用了鉴别和生成型的LLMs，并表明这些简单的技巧极大地阻碍了语言模型的性能。尽管在原始的公开可用数据集上表现出优秀的性能，但我们发现所有模型都很难回答我们新构建的数据集。我们展示了通过扰动引入任务变化可以提高模型的性能。

    Large language models (LLMs), such as GPT-3.5 and GPT-4, have greatly advanced the performance of artificial systems on various natural language processing tasks to human-like levels. However, their generalisation and robustness to perform logical reasoning remain under-evaluated. To probe this ability, we propose three new logical reasoning datasets named "ReClor-plus", "LogiQA-plus" and "LogiQAv2-plus", each featuring three subsets: the first with randomly shuffled options, the second with the correct choices replaced by "none of the other options are correct", and a combination of the previous two subsets. We carry out experiments on these datasets with both discriminative and generative LLMs and show that these simple tricks greatly hinder the performance of the language models. Despite their superior performance on the original publicly available datasets, we find that all models struggle to answer our newly constructed datasets. We show that introducing task variations by perturb
    
[^7]: 利用大型语言模型探索自我强化以改进学生生成的多项选择题解释

    Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models. (arXiv:2309.10444v1 [cs.AI])

    [http://arxiv.org/abs/2309.10444](http://arxiv.org/abs/2309.10444)

    本文提出了一个自我强化大型语言模型框架，自动生成和评估学生生成的解释，用于改进学生资源共享中学生生成的多项选择题的解释质量。

    

    学生资源共享涉及学生生成和分享学习资源。在学生生成多项选择题时，创建解释是一个关键步骤，因为它有助于对相关概念的深入理解。然而，学生往往由于主题理解有限和仅仅重申问题、干扰因素和正确答案的倾向而难以编写有效的解释。为了帮助支撑这个任务，在这项工作中，我们提出了一个自我强化的大型语言模型框架，旨在自动生成和评估解释。该框架由三个模块组成，生成与学生对齐的解释，评估这些解释以确保其质量，并迭代增强解释。如果一个解释的评估分数低于定义的阈值，框架会迭代地优化和重新评估解释。重要的是，我们的框架模拟了一个学生学习的过程。

    Learnersourcing involves students generating and sharing learning resources with their peers. When learnersourcing multiple-choice questions, creating explanations for the generated questions is a crucial step as it facilitates a deeper understanding of the related concepts. However, it is often difficult for students to craft effective explanations due to limited subject understanding and a tendency to merely restate the question stem, distractors, and correct answer. To help scaffold this task, in this work we propose a self-reinforcement large-language-model framework, with the goal of generating and evaluating explanations automatically. Comprising three modules, the framework generates student-aligned explanations, evaluates these explanations to ensure their quality and iteratively enhances the explanations. If an explanation's evaluation score falls below a defined threshold, the framework iteratively refines and reassesses the explanation. Importantly, our framework emulates th
    
[^8]: 大型过程模型：生成式人工智能时代的业务流程管理

    Large Process Models: Business Process Management in the Age of Generative AI. (arXiv:2309.00900v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2309.00900](http://arxiv.org/abs/2309.00900)

    大型过程模型（LPM）结合了大规模信息语料库和基于知识系统的方法的优势，旨在为组织提供过程建议和优化方案。

    

    大型语言模型（LLMs）和其他生成式人工智能方法的持续成功突显了大规模信息语料库相对于严格定义的符号模型的优势，但也证明了纯统计方法在安全性和可信度方面面临的挑战。为了对LLMs和其他基于基础模型的技术的潜力和限制进行框架化，我们提出了大型过程模型（LPM）的概念，它将LLMs的相关性能力与基于知识系统和自动推理方法的分析精度和可靠性相结合。LPMs被设想为直接利用专家积累的过程管理经验和具有不同特征（例如规模、地区或行业）的组织的过程绩效数据。在这个愿景中，提出的LPM将允许组织接收到对其特征进行了建模的过程建议和优化方案

    The continued success of Large Language Models (LLMs) and other generative artificial intelligence approaches highlights the advantages that large information corpora can have over rigidly defined symbolic models, but also serves as a proof-point of the challenges that purely statistics-based approaches have in terms of safety and trustworthiness. As a framework for contextualizing the potential, as well as the limitations of LLMs and other foundation model-based technologies, we propose the concept of a Large Process Model (LPM) that combines the correlation power of LLMs with the analytical precision and reliability of knowledge-based systems and automated reasoning approaches. LPMs are envisioned to directly utilize the wealth of process management experience that experts have accumulated, as well as process performance data of organizations with diverse characteristics, e.g., regarding size, region, or industry. In this vision, the proposed LPM would allow organizations to receive 
    
[^9]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    
[^10]: 面向剩余使用寿命预测的自动化机器学习

    Automated Machine Learning for Remaining Useful Life Predictions. (arXiv:2306.12215v1 [cs.LG])

    [http://arxiv.org/abs/2306.12215](http://arxiv.org/abs/2306.12215)

    本文介绍了一种自动化的机器学习方法，名为AutoRUL，用于自动预测工程系统的剩余使用寿命（RUL）。该方法将微调的标准回归方法与高预测能力的集成相结合，并通过八个真实世界的和合成数据集的评估，证明AutoML提供了一种可行的选择。

    

    预测工程系统的剩余使用寿命（RUL）是预测与健康管理中的重要任务。最近，数据驱动的方法在RUL预测中普及，相比模型驱动的方法不需要工程系统的物理知识。但是，这只是将需要的物理专业知识替换成机器学习（ML）专业知识，而这种专业知识通常也不可得。自动化机器学习（AutoML）承诺自动构建端到端的ML管道，使领域专家而非ML专家能够创建自己的模型。本文介绍了AutoRUL，一种AutoML驱动的端到端方法，用于自动RUL预测。AutoRUL将微调的标准回归方法与高预测能力的集成相结合。通过将所提出的方法用于八个真实世界的和合成数据集，与最先进的手工模型进行比较，我们表明AutoML提供了一种可行的选择。

    Being able to predict the remaining useful life (RUL) of an engineering system is an important task in prognostics and health management. Recently, data-driven approaches to RUL predictions are becoming prevalent over model-based approaches since no underlying physical knowledge of the engineering system is required. Yet, this just replaces required expertise of the underlying physics with machine learning (ML) expertise, which is often also not available. Automated machine learning (AutoML) promises to build end-to-end ML pipelines automatically enabling domain experts without ML expertise to create their own models. This paper introduces AutoRUL, an AutoML-driven end-to-end approach for automatic RUL predictions. AutoRUL combines fine-tuned standard regression methods to an ensemble with high predictive power. By evaluating the proposed method on eight real-world and synthetic datasets against state-of-the-art hand-crafted models, we show that AutoML provides a viable alternative to 
    
[^11]: AI生成的文本是否可靠地检测出来？

    Can AI-Generated Text be Reliably Detected?. (arXiv:2303.11156v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.11156](http://arxiv.org/abs/2303.11156)

    本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。

    

    本文从实证和理论两个方面表明，在实际场景中，几种AI文本检测器并不可靠。从实践上来说，我们证明了轻量级的改写器应用在大型语言模型（LLM）上可以破解一系列的检测器，包括使用水印方案、神经网络检测器和零样本分类器。我们的实验表明，旨在躲避改写攻击的基于检索的检测器仍然容易受到递归改写的攻击。然后，我们提出了一个理论上的不可能结果，指出随着语言模型变得越来越复杂和更擅长模仿人类文本，在最好的检测器性能会下降。对于一个足够先进的语言模型来模仿人类文本，即使最佳的检测器的表现只比随机分类器好上一点点。我们的结果足够概括特定的场景，如改写攻击。

    In this paper, both empirically and theoretically, we show that several AI-text detectors are not reliable in practical scenarios. Empirically, we show that paraphrasing attacks, where a light paraphraser is applied on top of a large language model (LLM), can break a whole range of detectors, including ones using watermarking schemes as well as neural network-based detectors and zero-shot classifiers. Our experiments demonstrate that retrieval-based detectors, designed to evade paraphrasing attacks, are still vulnerable to recursive paraphrasing. We then provide a theoretical impossibility result indicating that as language models become more sophisticated and better at emulating human text, the performance of even the best-possible detector decreases. For a sufficiently advanced language model seeking to imitate human text, even the best-possible detector may only perform marginally better than a random classifier. Our result is general enough to capture specific scenarios such as par
    

