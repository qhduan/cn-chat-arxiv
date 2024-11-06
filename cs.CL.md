# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AraTrust: An Evaluation of Trustworthiness for LLMs in Arabic](https://arxiv.org/abs/2403.09017) | AraTrust是第一个阿拉伯语大型语言模型的全面信誉基准，解决了缺乏全面信誉评估基准的问题，帮助准确评估和提高LLMs的安全性。 |
| [^2] | [An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers](https://arxiv.org/abs/2403.02839) | 精调评判模型在领域内测试上表现出色，但泛化能力和公平性不及GPT4。 |
| [^3] | [ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors](https://arxiv.org/abs/2402.16444) | ShieldLM是一个基于LLM的安全检测器，符合一般人类安全标准，支持定制化的检测规则，并提供决策解释。 |
| [^4] | [MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues](https://arxiv.org/abs/2402.14762) | 提出了MT-Bench-101用于评估大型语言模型在多轮对话中的细粒度能力，构建了包含4208轮对话数据的三级分层能力分类，并评估了21种流行的语言模型，发现它们在不同对话轮次中表现出不同的趋势。 |
| [^5] | [Multi-modal preference alignment remedies regression of visual instruction tuning on language model](https://arxiv.org/abs/2402.10884) | 通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。 |
| [^6] | [Limits of Transformer Language Models on Algorithmic Learning](https://arxiv.org/abs/2402.05785) | Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。 |
| [^7] | [A Natural Language Processing-Based Classification and Mode-Based Ranking of Musculoskeletal Disorder Risk Factors](https://arxiv.org/abs/2312.11517) | 本研究利用自然语言处理和基于模式的排名方法对肌肉骨骼疾病的风险因素进行了分类和排名，以提高对其理解、分类和优先考虑预防和治疗的能力。 |
| [^8] | [Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation.](http://arxiv.org/abs/2401.06477) | Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。 |
| [^9] | [Detecting Misinformation with LLM-Predicted Credibility Signals and Weak Supervision.](http://arxiv.org/abs/2309.07601) | 本文研究了使用大型语言模型和弱监督的方式来检测虚假信息，证明了这种方法在两个数据集上的效果优于当前最先进的分类器。 |

# 详细

[^1]: AraTrust：阿拉伯语大型语言模型信誉评估

    AraTrust: An Evaluation of Trustworthiness for LLMs in Arabic

    [https://arxiv.org/abs/2403.09017](https://arxiv.org/abs/2403.09017)

    AraTrust是第一个阿拉伯语大型语言模型的全面信誉基准，解决了缺乏全面信誉评估基准的问题，帮助准确评估和提高LLMs的安全性。

    

    arXiv:2403.09017v1 公告类型：新摘要：人工智能系统的迅速发展和广泛接受凸显了理解人工智能的能力和潜在风险的迫切需要。鉴于阿拉伯语在人工智能研究中的语言复杂性、文化丰富性和地位不高，有必要专注于阿拉伯语相关任务的大型语言模型（LLMs）的性能和安全性。尽管它们的发展取得了一些进展，但缺乏全面的信誉评估基准是准确评估和提高在阿拉伯语提示时LLMs的安全性面临的主要挑战。本文介绍了AraTrust 1，这是第一个针对阿拉伯语大型语言模型的全面的信誉基准。AraTrust 包含了516个人工编写的多项选择题，涉及与真实性、道德、安全性、身体健康、心理健康、不公平行为、非法活动相关的多个维度。

    arXiv:2403.09017v1 Announce Type: new  Abstract: The swift progress and widespread acceptance of artificial intelligence (AI) systems highlight a pressing requirement to comprehend both the capabilities and potential risks associated with AI. Given the linguistic complexity, cultural richness, and underrepresented status of Arabic in AI research, there is a pressing need to focus on Large Language Models (LLMs) performance and safety for Arabic related tasks. Despite some progress in their development, there is a lack of comprehensive trustworthiness evaluation benchmarks which presents a major challenge in accurately assessing and improving the safety of LLMs when prompted in Arabic. In this paper, we introduce AraTrust 1, the first comprehensive trustworthiness benchmark for LLMs in Arabic. AraTrust comprises 516 human-written multiple-choice questions addressing diverse dimensions related to truthfulness, ethics, safety, physical health, mental health, unfairness, illegal activities
    
[^2]: 作为评判器的LLM的实证研究：精调评判器模型是特定任务的分类器

    An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers

    [https://arxiv.org/abs/2403.02839](https://arxiv.org/abs/2403.02839)

    精调评判模型在领域内测试上表现出色，但泛化能力和公平性不及GPT4。

    

    最近，利用大型语言模型（LLM）评估其他LLM质量的趋势日益增长。许多研究采用专有的闭源模型，尤其是GPT4，作为评估器。另外，其他研究利用开源LLM来精调评判模型作为评估器。在本研究中，我们对不同的评判模型进行了实证研究。我们的发现表明，尽管精调的评判模型在领域内测试集上能够达到较高的准确性，甚至超过GPT4，但它们本质上是特定任务的分类器，其泛化能力和公平性远低于GPT4。

    arXiv:2403.02839v1 Announce Type: new  Abstract: Recently, there has been a growing trend of utilizing Large Language Model (LLM) to evaluate the quality of other LLMs. Many studies have employed proprietary close-source models, especially GPT4, as the evaluator. Alternatively, other works have fine-tuned judge models based on open-source LLMs as the evaluator. In this study, we conduct an empirical study of different judge models on their evaluation capability. Our findings indicate that although the fine-tuned judge models achieve high accuracy on in-domain test sets, even surpassing GPT4, they are inherently task-specific classifiers, and their generalizability and fairness severely underperform GPT4.
    
[^3]: ShieldLM: 使LLMs成为对齐、可定制和可解释的安全检测器

    ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors

    [https://arxiv.org/abs/2402.16444](https://arxiv.org/abs/2402.16444)

    ShieldLM是一个基于LLM的安全检测器，符合一般人类安全标准，支持定制化的检测规则，并提供决策解释。

    

    大型语言模型（LLMs）的安全性近年来受到越来越多关注，但在对LLMs的响应中检测安全问题的方法仍然缺乏一个全面的、对齐、可定制和可解释的方法。在本文中，我们提出了ShieldLM，一个基于LLM的安全检测器，它与一般人类安全标准相符，支持定制化的检测规则，并为其决策提供解释。为了训练ShieldLM，我们编制了一个包含14,387个查询-响应对的大型双语数据集，根据各种安全标准对响应的安全性进行了注释。通过大量实验，我们证明了ShieldLM在四个测试集上超越了强基线，展示了出色的定制性和可解释性。除了在标准检测数据集上表现良好外，ShieldLM还被证明在实际情况中作为先进LLMs的安全评估器是有效的。

    arXiv:2402.16444v1 Announce Type: new  Abstract: The safety of Large Language Models (LLMs) has gained increasing attention in recent years, but there still lacks a comprehensive approach for detecting safety issues within LLMs' responses in an aligned, customizable and explainable manner. In this paper, we propose ShieldLM, an LLM-based safety detector, which aligns with general human safety standards, supports customizable detection rules, and provides explanations for its decisions. To train ShieldLM, we compile a large bilingual dataset comprising 14,387 query-response pairs, annotating the safety of responses based on various safety standards. Through extensive experiments, we demonstrate that ShieldLM surpasses strong baselines across four test sets, showcasing remarkable customizability and explainability. Besides performing well on standard detection datasets, ShieldLM has also been shown to be effective in real-world situations as a safety evaluator for advanced LLMs. We relea
    
[^4]: MT-Bench-101: 用于评估大型语言模型在多轮对话中的细粒度基准

    MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues

    [https://arxiv.org/abs/2402.14762](https://arxiv.org/abs/2402.14762)

    提出了MT-Bench-101用于评估大型语言模型在多轮对话中的细粒度能力，构建了包含4208轮对话数据的三级分层能力分类，并评估了21种流行的语言模型，发现它们在不同对话轮次中表现出不同的趋势。

    

    大型语言模型（LLMs）的出现大大增强了对话系统。然而，全面评估LLMs的对话能力仍然是一个挑战。以往的基准主要集中在单轮对话或者提供粗粒度和不完整的多轮对话评估，忽视了真实对话的复杂性和细微的差异。为了解决这个问题，我们引入了MT-Bench-101，专门设计用于评估LLMs在多轮对话中的细粒度能力。通过对真实多轮对话数据进行详细分析，我们构建了一个包含13个不同任务中1388个多轮对话中的4208轮的三级分层能力分类。然后我们基于MT-Bench-101评估了21个流行的LLMs，从能力和任务两个角度进行全面分析，并观察到LLMs在对话轮次中表现出不同的趋势。

    arXiv:2402.14762v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has drastically enhanced dialogue systems. However, comprehensively evaluating the dialogue abilities of LLMs remains a challenge. Previous benchmarks have primarily focused on single-turn dialogues or provided coarse-grained and incomplete assessments of multi-turn dialogues, overlooking the complexity and fine-grained nuances of real-life dialogues. To address this issue, we introduce MT-Bench-101, specifically designed to evaluate the fine-grained abilities of LLMs in multi-turn dialogues. By conducting a detailed analysis of real multi-turn dialogue data, we construct a three-tier hierarchical ability taxonomy comprising 4208 turns across 1388 multi-turn dialogues in 13 distinct tasks. We then evaluate 21 popular LLMs based on MT-Bench-101, conducting comprehensive analyses from both ability and task perspectives and observing differing trends in LLMs performance across dialogue turns with
    
[^5]: 多模式偏好对齐修复了语言模型在视觉指令调整上的回归

    Multi-modal preference alignment remedies regression of visual instruction tuning on language model

    [https://arxiv.org/abs/2402.10884](https://arxiv.org/abs/2402.10884)

    通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。

    

    在实际应用中，多模式大型语言模型（MLLMs）被期望能够支持图像和文本模态的交换式多轮查询。然而，当前使用视觉问题回答（VQA）数据集训练的MLLMs可能会出现退化，因为VQA数据集缺乏原始文本指令数据集的多样性和复杂性，后者是底层语言模型训练的数据集。为了解决这一具有挑战性的退化问题，我们首先收集了一个轻量级（6k条记录）的VQA偏好数据集，其中答案由Gemini以细粒度方式注释了5个质量指标，然后研究了标准的监督微调、拒绝抽样、直接偏好优化（DPO）和SteerLM。我们的研究结果表明，通过DPO，我们能够超越语言模型的指导能力，实现了6.73的MT-Bench分数，而Vicuna的6.57和LLaVA的5.99，尽管数据规模较小。

    arXiv:2402.10884v1 Announce Type: cross  Abstract: In production, multi-modal large language models (MLLMs) are expected to support multi-turn queries of interchanging image and text modalities. However, the current MLLMs trained with visual-question-answering (VQA) datasets could suffer from degradation, as VQA datasets lack the diversity and complexity of the original text instruction datasets which the underlying language model had been trained with. To address this challenging degradation, we first collect a lightweight (6k entries) VQA preference dataset where answers were annotated by Gemini for 5 quality metrics in a granular fashion, and investigate standard Supervised Fine-tuning, rejection sampling, Direct Preference Optimization (DPO), and SteerLM. Our findings indicate that the with DPO we are able to surpass instruction-following capabilities of the language model, achieving a 6.73 score on MT-Bench, compared to Vicuna's 6.57 and LLaVA's 5.99 despite small data scale. This
    
[^6]: Transformer语言模型在算法学习上的限制

    Limits of Transformer Language Models on Algorithmic Learning

    [https://arxiv.org/abs/2402.05785](https://arxiv.org/abs/2402.05785)

    Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。

    

    我们分析了Transformer语言模型在学习离散算法方面的能力。为此，我们引入了两个要求组合多个离散子任务的新任务。我们通过从头开始训练LLaMA模型和在GPT-4和Gemini上提示来衡量学习学习原语的组合。我们观察到，目前最先进的Transformer语言模型的组合能力非常有限，并且在样本规模方面比为新的算法组合重新学习所有子任务效果更差。我们还提出了一个复杂性理论的定理，证明了记忆前馈模型上的梯度下降可以指数级地浪费数据。

    We analyze the capabilities of Transformer language models on learning discrete algorithms. To this end, we introduce two new tasks demanding the composition of several discrete sub-tasks. On both training LLaMA models from scratch and prompting on GPT-4 and Gemini we measure learning compositions of learned primitives. We observe that the compositional capabilities of state-of-the-art Transformer language models are very limited and sample-wise scale worse than relearning all sub-tasks for a new algorithmic composition. We also present a theorem in complexity theory, showing that gradient descent on memorizing feedforward models can be exponentially data inefficient.
    
[^7]: 基于自然语言处理的肌肉骨骼疾病风险因素分类与基于模式的排名

    A Natural Language Processing-Based Classification and Mode-Based Ranking of Musculoskeletal Disorder Risk Factors

    [https://arxiv.org/abs/2312.11517](https://arxiv.org/abs/2312.11517)

    本研究利用自然语言处理和基于模式的排名方法对肌肉骨骼疾病的风险因素进行了分类和排名，以提高对其理解、分类和优先考虑预防和治疗的能力。

    

    本研究探讨了肌肉骨骼疾病（MSD）风险因素，使用自然语言处理（NLP）和基于模式的排名相结合。旨在精细化理解、分类和优先考虑针对性预防和治疗。评估了八个NLP模型，结合预训练的转换器、余弦相似度和距离度量将因素分类为个人、生物力学、工作场所、心理和组织等类别。BERT与余弦相似度达到28%的准确率；句子转换器与欧氏、布雷曲蒂斯和闵可夫斯基距离得分为100%。通过10倍交叉验证，统计检验确保鲁棒结果。调查数据和基于模式的排名确定了严重性等级，与文献相一致。"工作姿势"是最严重的，凸显了姿势的作用。调查结果强调了"工作不稳定性"、"工作努力和回报不平衡"和"员工设施差"等因素的显著性。

    arXiv:2312.11517v3 Announce Type: replace  Abstract: This research delves into Musculoskeletal Disorder (MSD) risk factors, using a blend of Natural Language Processing (NLP) and mode-based ranking. The aim is to refine understanding, classification, and prioritization for focused prevention and treatment. Eight NLP models are evaluated, combining pre-trained transformers, cosine similarity, and distance metrics to categorize factors into personal, biomechanical, workplace, psychological, and organizational classes. BERT with cosine similarity achieves 28% accuracy; sentence transformer with Euclidean, Bray-Curtis, and Minkowski distances scores 100%. With 10-fold cross-validation, statistical tests ensure robust results. Survey data and mode-based ranking determine severity hierarchy, aligning with the literature. "Working posture" is the most severe, highlighting posture's role. Survey insights emphasize "Job insecurity," "Effort reward imbalance," and "Poor employee facility" as sig
    
[^8]: Kun: 使用指令反向翻译的中国自对齐问题的答案优化方法

    Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation. (arXiv:2401.06477v1 [cs.CL])

    [http://arxiv.org/abs/2401.06477](http://arxiv.org/abs/2401.06477)

    Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。

    

    在本文中，我们介绍了一种名为Kun的新方法，用于在不依赖手动注释的情况下为大型语言模型（LLMs）创建高质量的指导调整数据集。Kun利用来自吾道、完卷和SkyPile等多个来源的未标记数据，采用基于指令反向翻译和答案优化的自我训练算法，生成了一个超过一百万个中文指导数据点的大规模数据集。该方法通过使用自我筛选过程来完善和选择最有效的指令-输出对，显著偏离传统方法。我们在多个基准测试上对6B参数的Yi模型进行了实验，结果表明Kun具有鲁棒性和可扩展性。我们方法的核心贡献在于算法的改进，增强了数据的保留和清晰度，并且创新的数据生成方法极大地减少了对昂贵和耗时的手动注释的依赖。这种方法ological方法提出了一种解决中文自对齐问题的方法，并提高了数据的准确性和质量。

    In this paper, we introduce Kun, a novel approach for creating high-quality instruction-tuning datasets for large language models (LLMs) without relying on manual annotations. Adapting a self-training algorithm based on instruction back-translation and answer polishment, Kun leverages unlabelled data from diverse sources such as Wudao, Wanjuan, and SkyPile to generate a substantial dataset of over a million Chinese instructional data points. This approach significantly deviates from traditional methods by using a self-curation process to refine and select the most effective instruction-output pairs. Our experiments with the 6B-parameter Yi model across various benchmarks demonstrate Kun's robustness and scalability. Our method's core contributions lie in its algorithmic advancement, which enhances data retention and clarity, and its innovative data generation approach that substantially reduces the reliance on costly and time-consuming manual annotations. This methodology presents a sc
    
[^9]: 使用LLM预测的可信度信号和弱监督检测虚假信息

    Detecting Misinformation with LLM-Predicted Credibility Signals and Weak Supervision. (arXiv:2309.07601v1 [cs.CL])

    [http://arxiv.org/abs/2309.07601](http://arxiv.org/abs/2309.07601)

    本文研究了使用大型语言模型和弱监督的方式来检测虚假信息，证明了这种方法在两个数据集上的效果优于当前最先进的分类器。

    

    可信度信号代表了记者和事实核查员通常用来评估在线内容真实性的一系列启发式方法。然而，自动化可信度信号提取的任务非常具有挑战性，因为它需要训练高准确率的特定信号提取器，而目前没有足够大的数据集对所有可信度信号进行注释。本文研究了是否可以有效地用一组18个可信度信号来提示大型语言模型（LLMs），以产生每个信号的弱标签。然后，我们使用弱监督的方式对这些潜在的噪声标签进行聚合，以预测内容的真实性。我们证明了我们的方法，即结合了零-shot LLM可信度信号标注和弱监督的方法，在两个虚假信息数据集上优于最先进的分类器，而没有使用任何训练标签。

    Credibility signals represent a wide range of heuristics that are typically used by journalists and fact-checkers to assess the veracity of online content. Automating the task of credibility signal extraction, however, is very challenging as it requires high-accuracy signal-specific extractors to be trained, while there are currently no sufficiently large datasets annotated with all credibility signals. This paper investigates whether large language models (LLMs) can be prompted effectively with a set of 18 credibility signals to produce weak labels for each signal. We then aggregate these potentially noisy labels using weak supervision in order to predict content veracity. We demonstrate that our approach, which combines zero-shot LLM credibility signal labeling and weak supervision, outperforms state-of-the-art classifiers on two misinformation datasets without using any ground-truth labels for training. We also analyse the contribution of the individual credibility signals towards p
    

