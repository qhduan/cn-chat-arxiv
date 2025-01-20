# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models](https://arxiv.org/abs/2403.20262) | 该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。 |
| [^2] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^3] | [Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates](https://arxiv.org/abs/2402.18540) | 提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。 |
| [^4] | [A Systematic Evaluation of Large Language Models on Out-of-Distribution Logical Reasoning Tasks.](http://arxiv.org/abs/2310.09430) | 通过对大型语言模型在非分布式逻辑推理任务上进行系统评估，我们发现这些模型在处理我们新构建的数据集时都存在困难，尽管它们在其他自然语言处理任务上表现良好。这表明这些模型在逻辑推理方面的泛化和鲁棒性仍需要进一步研究。 |
| [^5] | [Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models.](http://arxiv.org/abs/2309.10444) | 本文提出了一个自我强化大型语言模型框架，自动生成和评估学生生成的解释，用于改进学生资源共享中学生生成的多项选择题的解释质量。 |
| [^6] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |
| [^7] | [ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain.](http://arxiv.org/abs/2304.11960) | 本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。 |
| [^8] | [Can AI-Generated Text be Reliably Detected?.](http://arxiv.org/abs/2303.11156) | 本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。 |

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
    
[^3]: 在微调后保持LLMs的对齐性:提示模板的关键作用

    Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates

    [https://arxiv.org/abs/2402.18540](https://arxiv.org/abs/2402.18540)

    提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。

    

    公共LLMs，如Llama 2-Chat，推动了LLM研究的巨大活动。这些模型经历了对齐性训练，被认为是安全的。最近，齐等人（2023年）报告称，即使是良性的微调（例如，在看似安全的数据集上）也可能导致模型产生不安全的行为。本文介绍了减轻这种对齐性丢失的方法和最佳实践。通过对几个聊天模型（Meta的Llama 2-Chat，Mistral AI的Mistral 7B Instruct v0.2和OpenAI的GPT-3.5 Turbo）进行广泛实验，本文发现微调和推理过程中使用的提示模板在保持安全对齐性方面起着至关重要的作用，并提出了“纯粹调优，安全测试”（PTST）原则 - 在测试时不使用安全提示进行模型微调，但在测试时包含它。对GSM8K，ChatDoctor和OpenOrca进行的微调实验表明，PTST显着减少了不安全行为的增加，甚至几乎消除了它们。

    arXiv:2402.18540v1 Announce Type: cross  Abstract: Public LLMs such as the Llama 2-Chat have driven huge activity in LLM research. These models underwent alignment training and were considered safe. Recently Qi et al. (2023) reported that even benign fine-tuning (e.g., on seemingly safe datasets) can give rise to unsafe behaviors in the models. The current paper is about methods and best practices to mitigate such loss of alignment. Through extensive experiments on several chat models (Meta's Llama 2-Chat, Mistral AI's Mistral 7B Instruct v0.2, and OpenAI's GPT-3.5 Turbo), this paper uncovers that the prompt templates used during fine-tuning and inference play a crucial role in preserving safety alignment, and proposes the "Pure Tuning, Safe Testing" (PTST) principle -- fine-tune models without a safety prompt, but include it at test time. Fine-tuning experiments on GSM8K, ChatDoctor, and OpenOrca show that PTST significantly reduces the rise of unsafe behaviors, and even almost elimin
    
[^4]: 对大型语言模型在非分布式逻辑推理任务上的系统评估

    A Systematic Evaluation of Large Language Models on Out-of-Distribution Logical Reasoning Tasks. (arXiv:2310.09430v1 [cs.CL])

    [http://arxiv.org/abs/2310.09430](http://arxiv.org/abs/2310.09430)

    通过对大型语言模型在非分布式逻辑推理任务上进行系统评估，我们发现这些模型在处理我们新构建的数据集时都存在困难，尽管它们在其他自然语言处理任务上表现良好。这表明这些模型在逻辑推理方面的泛化和鲁棒性仍需要进一步研究。

    

    大型语言模型（LLMs），如GPT-3.5和GPT-4，已经将人工系统在各种自然语言处理任务上的性能提升到接近人类水平。然而，它们在逻辑推理方面的泛化和鲁棒性仍未得到充分评估。为了探索这种能力，我们提出了三个新的逻辑推理数据集，分别名为"ReClor-plus"、"LogiQA-plus"和"LogiQAv2-plus"，每个数据集都包含三个子集：第一个是选项随机打乱，第二个是将正确选项替换为"没有其他选项是正确的"，第三个是前两个子集的组合。我们在这些数据集上进行了实验，使用了鉴别和生成型的LLMs，并表明这些简单的技巧极大地阻碍了语言模型的性能。尽管在原始的公开可用数据集上表现出优秀的性能，但我们发现所有模型都很难回答我们新构建的数据集。我们展示了通过扰动引入任务变化可以提高模型的性能。

    Large language models (LLMs), such as GPT-3.5 and GPT-4, have greatly advanced the performance of artificial systems on various natural language processing tasks to human-like levels. However, their generalisation and robustness to perform logical reasoning remain under-evaluated. To probe this ability, we propose three new logical reasoning datasets named "ReClor-plus", "LogiQA-plus" and "LogiQAv2-plus", each featuring three subsets: the first with randomly shuffled options, the second with the correct choices replaced by "none of the other options are correct", and a combination of the previous two subsets. We carry out experiments on these datasets with both discriminative and generative LLMs and show that these simple tricks greatly hinder the performance of the language models. Despite their superior performance on the original publicly available datasets, we find that all models struggle to answer our newly constructed datasets. We show that introducing task variations by perturb
    
[^5]: 利用大型语言模型探索自我强化以改进学生生成的多项选择题解释

    Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models. (arXiv:2309.10444v1 [cs.AI])

    [http://arxiv.org/abs/2309.10444](http://arxiv.org/abs/2309.10444)

    本文提出了一个自我强化大型语言模型框架，自动生成和评估学生生成的解释，用于改进学生资源共享中学生生成的多项选择题的解释质量。

    

    学生资源共享涉及学生生成和分享学习资源。在学生生成多项选择题时，创建解释是一个关键步骤，因为它有助于对相关概念的深入理解。然而，学生往往由于主题理解有限和仅仅重申问题、干扰因素和正确答案的倾向而难以编写有效的解释。为了帮助支撑这个任务，在这项工作中，我们提出了一个自我强化的大型语言模型框架，旨在自动生成和评估解释。该框架由三个模块组成，生成与学生对齐的解释，评估这些解释以确保其质量，并迭代增强解释。如果一个解释的评估分数低于定义的阈值，框架会迭代地优化和重新评估解释。重要的是，我们的框架模拟了一个学生学习的过程。

    Learnersourcing involves students generating and sharing learning resources with their peers. When learnersourcing multiple-choice questions, creating explanations for the generated questions is a crucial step as it facilitates a deeper understanding of the related concepts. However, it is often difficult for students to craft effective explanations due to limited subject understanding and a tendency to merely restate the question stem, distractors, and correct answer. To help scaffold this task, in this work we propose a self-reinforcement large-language-model framework, with the goal of generating and evaluating explanations automatically. Comprising three modules, the framework generates student-aligned explanations, evaluates these explanations to ensure their quality and iteratively enhances the explanations. If an explanation's evaluation score falls below a defined threshold, the framework iteratively refines and reassesses the explanation. Importantly, our framework emulates th
    
[^6]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    
[^7]: ThreatCrawl：基于BERT的网络安全焦点爬虫

    ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain. (arXiv:2304.11960v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2304.11960](http://arxiv.org/abs/2304.11960)

    本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。

    

    可公开获取的信息对于网络威胁情报（CTI）来说包含有价值的信息。这可以用于预防已经在其他系统上发生的攻击。但是，虽然有不同的标准来交流这些信息，但很多信息是以非标准化的方式在文章或博客帖子中共享的。手动浏览多个在线门户和新闻页面以发现新威胁并提取它们是一项耗时的任务。为了自动化这个扫描过程的一部分，多篇论文提出了使用自然语言处理（NLP）从文档中提取威胁指示器（IOCs）的提取器。然而，虽然这已经解决了从文档中提取信息的问题，但很少考虑搜索这些文档。本文提出了一种新的焦点爬虫ThreatCrawl，它使用双向编码器表示（BERT）搜索网络安全领域中的相关文档。ThreatCrawl使用主题建模和关键词提取技术来识别相关网站和网页，然后应用基于BERT的分类器来优先考虑最可能包含有价值CTI信息的网页。

    Publicly available information contains valuable information for Cyber Threat Intelligence (CTI). This can be used to prevent attacks that have already taken place on other systems. Ideally, only the initial attack succeeds and all subsequent ones are detected and stopped. But while there are different standards to exchange this information, a lot of it is shared in articles or blog posts in non-standardized ways. Manually scanning through multiple online portals and news pages to discover new threats and extracting them is a time-consuming task. To automize parts of this scanning process, multiple papers propose extractors that use Natural Language Processing (NLP) to extract Indicators of Compromise (IOCs) from documents. However, while this already solves the problem of extracting the information out of documents, the search for these documents is rarely considered. In this paper, a new focused crawler is proposed called ThreatCrawl, which uses Bidirectional Encoder Representations 
    
[^8]: AI生成的文本是否可靠地检测出来？

    Can AI-Generated Text be Reliably Detected?. (arXiv:2303.11156v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.11156](http://arxiv.org/abs/2303.11156)

    本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。

    

    本文从实证和理论两个方面表明，在实际场景中，几种AI文本检测器并不可靠。从实践上来说，我们证明了轻量级的改写器应用在大型语言模型（LLM）上可以破解一系列的检测器，包括使用水印方案、神经网络检测器和零样本分类器。我们的实验表明，旨在躲避改写攻击的基于检索的检测器仍然容易受到递归改写的攻击。然后，我们提出了一个理论上的不可能结果，指出随着语言模型变得越来越复杂和更擅长模仿人类文本，在最好的检测器性能会下降。对于一个足够先进的语言模型来模仿人类文本，即使最佳的检测器的表现只比随机分类器好上一点点。我们的结果足够概括特定的场景，如改写攻击。

    In this paper, both empirically and theoretically, we show that several AI-text detectors are not reliable in practical scenarios. Empirically, we show that paraphrasing attacks, where a light paraphraser is applied on top of a large language model (LLM), can break a whole range of detectors, including ones using watermarking schemes as well as neural network-based detectors and zero-shot classifiers. Our experiments demonstrate that retrieval-based detectors, designed to evade paraphrasing attacks, are still vulnerable to recursive paraphrasing. We then provide a theoretical impossibility result indicating that as language models become more sophisticated and better at emulating human text, the performance of even the best-possible detector decreases. For a sufficiently advanced language model seeking to imitate human text, even the best-possible detector may only perform marginally better than a random classifier. Our result is general enough to capture specific scenarios such as par
    

