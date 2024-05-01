# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) | 本文通过比较数据取代和数据积累两种情况，发现累积数据可以防止模型崩溃。 |
| [^2] | [NUMTEMP: A real-world benchmark to verify claims with statistical and temporal expressions](https://arxiv.org/abs/2403.17169) | NUMTEMP是一个真实世界基准，专注于验证复杂的数字论点，量化了现有解决方案的局限性，并提供了一种解决真实世界数字论点验证挑战的方法。 |
| [^3] | [Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation](https://arxiv.org/abs/2403.16771) | 本文提出了用于鲁棒性混合代码翻译的合成数据生成和联合学习方法，包括开发了Hinglish到英语的平行语料库以及提出的能够处理噪声的联合训练模型RCMT。 |
| [^4] | [Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines](https://arxiv.org/abs/2403.11585) | Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。 |
| [^5] | [LAB: Large-Scale Alignment for ChatBots](https://arxiv.org/abs/2403.01081) | 介绍了一种名为LAB的方法，旨在克服大型语言模型训练中的可扩展性挑战，通过分类法指导的合成数据生成和多阶段调整框架，实现了对昂贵人工标注和GPT-4等专有模型依赖较少的大规模对齐，提供了一种可扩展、具有成本效益的解决方案，不会出现灾难性遗忘情况，进一步增强了LLM的训练效率。 |
| [^6] | [Entity-Aware Multimodal Alignment Framework for News Image Captioning](https://arxiv.org/abs/2402.19404) | 设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。 |
| [^7] | [Stick to your Role! Stability of Personal Values Expressed in Large Language Models](https://arxiv.org/abs/2402.14846) | 本文提出研究在大型语言模型中个人价值在不同背景下的表达稳定性，通过模拟对话的方式进行评估，对19个LLMs进行比较研究。 |
| [^8] | [End-to-end multilingual fact-checking at scale](https://arxiv.org/abs/2402.12147) | 使用Factiverse AI模型，可以进行跨语言的端到端事实核查，并且通过实验证明，为事实核查任务进行微调的模型优于大型语言模型。 |
| [^9] | [The Essential Role of Causality in Foundation World Models for Embodied AI](https://arxiv.org/abs/2402.06665) | 基于因果关系的基础世界模型对于具身人工智能的发展至关重要，当前的基础模型无法准确建模与现实世界的物理相互作用。因果关系的研究有助于构建真实世界模型，提高对可能相互作用结果的准确预测能力。 |
| [^10] | [Detecting Generated Native Ads in Conversational Search](https://arxiv.org/abs/2402.04889) | 本论文研究了LLM是否可以用作对抗生成式原生广告的对策，并通过构建广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验证明。 |
| [^11] | [Multimodal Clinical Pseudo-notes for Emergency Department Prediction Tasks using Multiple Embedding Model for EHR (MEME)](https://arxiv.org/abs/2402.00160) | 本研究提出了一种名为“MEME”的多重嵌入模型，将电子健康记录视为多模态数据。通过结合“伪笔记”和多模态方法，该模型在紧急科室预测任务中表现出优越性能，超过了单模态嵌入方法和传统机器学习方法。然而，该模型在不同医院机构之间存在泛化能力方面的局限性。 |
| [^12] | [NOLA: Networks as Linear Combination of Low Rank Random Basis.](http://arxiv.org/abs/2310.02556) | 该论文介绍了一种名为NOLA的方法，该方法通过将网络表示为低秩随机基向量的线性组合来减少语言模型的参数数量，从而实现高效的适应和存储。 |

# 详细

[^1]: 模型崩溃是否不可避免？通过累积真实和合成数据打破递归的诅咒

    Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data

    [https://arxiv.org/abs/2404.01413](https://arxiv.org/abs/2404.01413)

    本文通过比较数据取代和数据积累两种情况，发现累积数据可以防止模型崩溃。

    

    随着生成模型的激增，以及在网络规模数据上的预训练，一个及时的问题浮出水面：当这些模型被训练在它们自己生成的输出上时会发生什么？最近对模型数据反馈循环的研究发现，这样的循环可能导致模型崩溃，即性能随着每次模型拟合迭代逐渐下降，直到最新的模型变得无用。然而，最近几篇研究模型崩溃的论文都假设随着时间推移，新数据会取代旧数据，而不是假设数据会随时间累积。在本文中，我们比较了这两种情况，并表明积累数据可以防止模型崩溃。我们首先研究了一个解析可处理的设置，其中一系列线性模型拟合到先前模型的预测。先前的工作表明，如果数据被替换，测试误差会随着模型拟合迭代次数线性增加；我们扩展了这个研究探讨了数据逐渐累积的情况下会发生什么。

    arXiv:2404.01413v1 Announce Type: cross  Abstract: The proliferation of generative models, combined with pretraining on web-scale data, raises a timely question: what happens when these models are trained on their own generated outputs? Recent investigations into model-data feedback loops discovered that such loops can lead to model collapse, a phenomenon where performance progressively degrades with each model-fitting iteration until the latest model becomes useless. However, several recent papers studying model collapse assumed that new data replace old data over time rather than assuming data accumulate over time. In this paper, we compare these two settings and show that accumulating data prevents model collapse. We begin by studying an analytically tractable setup in which a sequence of linear models are fit to the previous models' predictions. Previous work showed if data are replaced, the test error increases linearly with the number of model-fitting iterations; we extend this r
    
[^2]: NUMTEMP：一个用于验证带有统计和时间表达式的论点的真实世界基准

    NUMTEMP: A real-world benchmark to verify claims with statistical and temporal expressions

    [https://arxiv.org/abs/2403.17169](https://arxiv.org/abs/2403.17169)

    NUMTEMP是一个真实世界基准，专注于验证复杂的数字论点，量化了现有解决方案的局限性，并提供了一种解决真实世界数字论点验证挑战的方法。

    

    自动事实检查在数字时代应对不断增长的错误信息方面引起了极大兴趣。现有系统主要专注于维基百科上的合成论点，并且在真实世界论点上也取得了显著进展。在本文中，我们发布了Numtemp，一个多样化、多领域的数据集，专门关注数字论点，包括时间、统计和多样化方面的细粒度元数据，并且具有不泄露的证据收集。这解决了验证真实世界数字论点的挑战，这些论点复杂，往往缺乏精确信息，这是现有作品主要关注合成论点未解决的问题。我们评估并量化了现有解决方案在验证数字论点任务中的局限性。我们还评估了基于论点分解的方法、基于数字理解的模型，我们的最佳基线实现了58.32的宏F1分数。这证明了Numtemp的关键价值。

    arXiv:2403.17169v1 Announce Type: cross  Abstract: Automated fact checking has gained immense interest to tackle the growing misinformation in the digital era. Existing systems primarily focus on synthetic claims on Wikipedia, and noteworthy progress has also been made on real-world claims. In this work, we release Numtemp, a diverse, multi-domain dataset focused exclusively on numerical claims, encompassing temporal, statistical and diverse aspects with fine-grained metadata and an evidence collection without leakage. This addresses the challenge of verifying real-world numerical claims, which are complex and often lack precise information, not addressed by existing works that mainly focus on synthetic claims. We evaluate and quantify the limitations of existing solutions for the task of verifying numerical claims. We also evaluate claim decomposition based methods, numerical understanding based models and our best baselines achieves a macro-F1 of 58.32. This demonstrates that Numtemp
    
[^3]: 用于鲁棒性混合代码翻译的合成数据生成和联合学习

    Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation

    [https://arxiv.org/abs/2403.16771](https://arxiv.org/abs/2403.16771)

    本文提出了用于鲁棒性混合代码翻译的合成数据生成和联合学习方法，包括开发了Hinglish到英语的平行语料库以及提出的能够处理噪声的联合训练模型RCMT。

    

    现代多语言世界中的广泛网络交流为在单个话语中混合多种语言（又称混合代码语言）提供了机会。由于标注数据的稀缺和噪音的存在，这给计算模型带来了严峻挑战。在资源匮乏的环境中缓解数据稀缺问题的潜在解决方案是通过翻译利用资源丰富语言中的现有数据。本文针对混合代码（印地语和孟加拉语）到英语的机器翻译问题。首先，我们合成开发了HINMIX一个印地语到英语的平行语料库，包含约420万个句对。随后，我们提出了RCMT，一种基于强健扰动的联合训练模型，通过在干净和带噪声单词之间共享参数，学习处理现实世界混合代码文本中的噪声。此外，我们展示了RCMT在零-shot设置中对孟加拉语的适应能力。

    arXiv:2403.16771v1 Announce Type: new  Abstract: The widespread online communication in a modern multilingual world has provided opportunities to blend more than one language (aka code-mixed language) in a single utterance. This has resulted a formidable challenge for the computational models due to the scarcity of annotated data and presence of noise. A potential solution to mitigate the data scarcity problem in low-resource setup is to leverage existing data in resource-rich language through translation. In this paper, we tackle the problem of code-mixed (Hinglish and Bengalish) to English machine translation. First, we synthetically develop HINMIX, a parallel corpus of Hinglish to English, with ~4.2M sentence pairs. Subsequently, we propose RCMT, a robust perturbation based joint-training model that learns to handle noise in the real-world code-mixed text by parameter sharing across clean and noisy words. Further, we show the adaptability of RCMT in a zero-shot setup for Bengalish t
    
[^4]: Linguacodus：一种在机器学习流水线中进行变革性代码生成的协同框架

    Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines

    [https://arxiv.org/abs/2403.11585](https://arxiv.org/abs/2403.11585)

    Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。

    

    在不断发展的机器学习领域中，将自然语言描述无缝转化为可执行代码仍然是一个巨大的挑战。本文介绍了Linguacodus，这是一个创新性框架，旨在通过部署一个动态流水线，通过高级数据塑形指令，将自然语言任务描述迭代地转换为代码来应对这一挑战。Linguacodus的核心是一个经过精细调整的大型语言模型（LLM），能够评估各种问题的多样解决方案，并为特定任务选择最合适的解决方案。本文详细介绍了精细调整过程，并阐明了如何将自然语言描述转化为功能性代码。Linguacodus代表了自动化代码生成的重大飞跃，有效地弥合了任务描述和可执行代码之间的差距。它对推进跨不同领域的机器学习应用具有巨大潜力。

    arXiv:2403.11585v1 Announce Type: cross  Abstract: In the ever-evolving landscape of machine learning, seamless translation of natural language descriptions into executable code remains a formidable challenge. This paper introduces Linguacodus, an innovative framework designed to tackle this challenge by deploying a dynamic pipeline that iteratively transforms natural language task descriptions into code through high-level data-shaping instructions. The core of Linguacodus is a fine-tuned large language model (LLM), empowered to evaluate diverse solutions for various problems and select the most fitting one for a given task. This paper details the fine-tuning process, and sheds light on how natural language descriptions can be translated into functional code. Linguacodus represents a substantial leap towards automated code generation, effectively bridging the gap between task descriptions and executable code. It holds great promise for advancing machine learning applications across div
    
[^5]: LAB：针对ChatBots的大规模对齐

    LAB: Large-Scale Alignment for ChatBots

    [https://arxiv.org/abs/2403.01081](https://arxiv.org/abs/2403.01081)

    介绍了一种名为LAB的方法，旨在克服大型语言模型训练中的可扩展性挑战，通过分类法指导的合成数据生成和多阶段调整框架，实现了对昂贵人工标注和GPT-4等专有模型依赖较少的大规模对齐，提供了一种可扩展、具有成本效益的解决方案，不会出现灾难性遗忘情况，进一步增强了LLM的训练效率。

    

    这项工作介绍了LAB（ChatBots的大规模对齐），这是一种旨在克服大型语言模型（LLM）训练中指令调整阶段的可扩展性挑战的创新方法。通过利用基于分类法的合成数据生成过程和多阶段调整框架，LAB显著减少对昂贵的人类注释和诸如GPT-4之类的专有模型的依赖。我们证明，使用LAB训练的模型在几个基准测试中的性能可以与使用传统人类注释或GPT-4生成的合成数据训练的模型相比具有竞争力。因此，在不会出现灾难性遗忘的情况下，提供了一种可扩展、具有成本效益的解决方案，以增强LLM的能力和指令遵循行为，标志着在高效训练各种应用的LLM方面迈出了一步。

    arXiv:2403.01081v1 Announce Type: new  Abstract: This work introduces LAB (Large-scale Alignment for chatBots), a novel methodology designed to overcome the scalability challenges in the instruction-tuning phase of large language model (LLM) training. Leveraging a taxonomy-guided synthetic data generation process and a multi-phase tuning framework, LAB significantly reduces reliance on expensive human annotations and proprietary models like GPT-4. We demonstrate that LAB-trained models can achieve competitive performance across several benchmarks compared to models trained with traditional human-annotated or GPT-4 generated synthetic data. Thus offering a scalable, cost-effective solution for enhancing LLM capabilities and instruction-following behaviors without the drawbacks of catastrophic forgetting, marking a step forward in the efficient training of LLMs for a wide range of applications.
    
[^6]: 面向实体的多模态对齐框架用于新闻图像字幕生成

    Entity-Aware Multimodal Alignment Framework for News Image Captioning

    [https://arxiv.org/abs/2402.19404](https://arxiv.org/abs/2402.19404)

    设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。

    

    新闻图像字幕生成任务是图像字幕生成任务的一个变体，要求模型生成一个更具信息性的字幕，其中包含新闻图像和相关新闻文章。近年来，多模态大型语言模型发展迅速，并在新闻图像字幕生成任务中表现出前景。然而，根据我们的实验，常见的多模态大型语言模型在零样本设定下生成实体方面表现不佳。即使在新闻图像字幕生成数据集上进行简单微调，它们处理实体信息的能力仍然有限。为了获得一个更强大的模型来处理多模态实体信息，我们设计了两个多模态实体感知对齐任务和一个对齐框架，以对齐模型并生成新闻图像字幕。我们的方法在GoodNews数据集上将CIDEr分数提高到86.29（从72.33），在NYTimes800k数据集上将其提高到85.61（从70.83），优于先前的最先进模型。

    arXiv:2402.19404v1 Announce Type: cross  Abstract: News image captioning task is a variant of image captioning task which requires model to generate a more informative caption with news image and the associated news article. Multimodal Large Language models have developed rapidly in recent years and is promising in news image captioning task. However, according to our experiments, common MLLMs are not good at generating the entities in zero-shot setting. Their abilities to deal with the entities information are still limited after simply fine-tuned on news image captioning dataset. To obtain a more powerful model to handle the multimodal entity information, we design two multimodal entity-aware alignment tasks and an alignment framework to align the model and generate the news image captions. Our method achieves better results than previous state-of-the-art models in CIDEr score (72.33 -> 86.29) on GoodNews dataset and (70.83 -> 85.61) on NYTimes800k dataset.
    
[^7]: 坚持你的角色！个人价值在大型语言模型中的稳定性

    Stick to your Role! Stability of Personal Values Expressed in Large Language Models

    [https://arxiv.org/abs/2402.14846](https://arxiv.org/abs/2402.14846)

    本文提出研究在大型语言模型中个人价值在不同背景下的表达稳定性，通过模拟对话的方式进行评估，对19个LLMs进行比较研究。

    

    通过基准测试或心理问卷的标准方式研究大型语言模型(LLMs)是提供许多来源于类似最小背景的不同查询（例如多项选择问题）。然而，由于LLM高度依赖于背景，因此从这种最小背景评估中得出的结论可能对模型在部署中的行为（在那里它将暴露于许多新背景）的说明很少。我们认为，依赖于背景的特性应该作为LLM比较的另一个维度来研究，而不是其他维度，如认知能力、知识或模型大小。在本文中，我们提出了一个关于在不同背景下（模拟对不同话题的对话）价值表达稳定性的案例研究，并使用标准心理学问卷（PVQ）和行为下游任务进行测量。我们考虑了来自五个家族的19个开源LLM。借鉴心理学方法，我们研究了等级稳定性。

    arXiv:2402.14846v1 Announce Type: cross  Abstract: The standard way to study Large Language Models (LLMs) through benchmarks or psychology questionnaires is to provide many different queries from similar minimal contexts (e.g. multiple choice questions). However, due to LLM's highly context-dependent nature, conclusions from such minimal-context evaluations may be little informative about the model's behavior in deployment (where it will be exposed to many new contexts). We argue that context-dependence should be studied as another dimension of LLM comparison alongside others such as cognitive abilities, knowledge, or model size. In this paper, we present a case-study about the stability of value expression over different contexts (simulated conversations on different topics), and as measured using a standard psychology questionnaire (PVQ) and a behavioral downstream task. We consider 19 open-sourced LLMs from five families. Reusing methods from psychology, we study Rank-order stabilit
    
[^8]: 跨语言规模的端到端事实核查

    End-to-end multilingual fact-checking at scale

    [https://arxiv.org/abs/2402.12147](https://arxiv.org/abs/2402.12147)

    使用Factiverse AI模型，可以进行跨语言的端到端事实核查，并且通过实验证明，为事实核查任务进行微调的模型优于大型语言模型。

    

    在本文中，我们描述了如何使用Factiverse AI模型在100多种语言中进行端到端事实核查。我们还通过实验性基准测试展示，为事实核查任务进行微调的模型胜过GPT-4、GPT-3.5-Turbo和Mistral-7b等大型语言模型。

    arXiv:2402.12147v1 Announce Type: cross  Abstract: In this article, we describe how you can perform end-to-end fact-checking in over 100 languages using Factiverse AI models. We also show through an experimental benchmark that fine-tuned models tailored for fact-checking tasks outperform Large Language Models such as GPT-4, GPT-3.5-Turbo, and Mistral-7b.
    
[^9]: 基于因果关系的基础世界模型在具身人工智能中的重要作用

    The Essential Role of Causality in Foundation World Models for Embodied AI

    [https://arxiv.org/abs/2402.06665](https://arxiv.org/abs/2402.06665)

    基于因果关系的基础世界模型对于具身人工智能的发展至关重要，当前的基础模型无法准确建模与现实世界的物理相互作用。因果关系的研究有助于构建真实世界模型，提高对可能相互作用结果的准确预测能力。

    

    最近在基础模型中取得的进展，尤其是在大型多模态模型和对话代理方面，引发了对具备普遍能力的具身代理人潜力的兴趣。这样的代理人需要能够在许多不同的真实世界环境中执行新任务。然而，当前的基础模型未能准确建模与现实世界的物理相互作用，因此对于具身人工智能而言是不够的。因果关系的研究有助于构建真实世界模型，这对于准确预测可能相互作用的结果至关重要。本文着重探讨了为即将到来的具身代理生成基础世界模型的前景，并对其中的因果关系的重要性提出了新的观点。我们认为整合因果关系是促进与世界的有意义的物理相互作用至关重要的。最后，我们揭示了这一背景下对因果关系的误解，并展示了我们对未来的展望。

    Recent advances in foundation models, especially in large multi-modal models and conversational agents, have ignited interest in the potential of generally capable embodied agents. Such agents would require the ability to perform new tasks in many different real-world environments. However, current foundation models fail to accurately model physical interactions with the real world thus not sufficient for Embodied AI. The study of causality lends itself to the construction of veridical world models, which are crucial for accurately predicting the outcomes of possible interactions. This paper focuses on the prospects of building foundation world models for the upcoming generation of embodied agents and presents a novel viewpoint on the significance of causality within these. We posit that integrating causal considerations is vital to facilitate meaningful physical interactions with the world. Finally, we demystify misconceptions about causality in this context and present our outlook fo
    
[^10]: 发现对话式搜索中的生成式原生广告

    Detecting Generated Native Ads in Conversational Search

    [https://arxiv.org/abs/2402.04889](https://arxiv.org/abs/2402.04889)

    本论文研究了LLM是否可以用作对抗生成式原生广告的对策，并通过构建广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验证明。

    

    对话式搜索引擎如YouChat和Microsoft Copilot使用大型语言模型（LLM）为查询生成答案。将此技术用于生成并整合广告，而不是将广告与有机搜索结果分开放置，只是一小步。这种类型的广告类似于原生广告和产品放置，两者都是非常有效的微妙和操纵性广告形式。在考虑到与LLM相关的高计算成本时，信息搜索者将很可能在不久的将来面临这种LLM技术的使用，因此供应商需要开发可持续的商业模式。本文研究了LLM是否也可以用作对抗生成式原生广告的对策，即阻止它们。为此，我们编制了一个大型的广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验。

    Conversational search engines such as YouChat and Microsoft Copilot use large language models (LLMs) to generate answers to queries. It is only a small step to also use this technology to generate and integrate advertising within these answers - instead of placing ads separately from the organic search results. This type of advertising is reminiscent of native advertising and product placement, both of which are very effective forms of subtle and manipulative advertising. It is likely that information seekers will be confronted with such use of LLM technology in the near future, especially when considering the high computational costs associated with LLMs, for which providers need to develop sustainable business models. This paper investigates whether LLMs can also be used as a countermeasure against generated native ads, i.e., to block them. For this purpose we compile a large dataset of ad-prone queries and of generated answers with automatically integrated ads to experiment with fin
    
[^11]: 使用多重嵌入模型的多模式临床伪笔记用于紧急科室预测任务的医疗电子健康记录（EHR）翻译

    Multimodal Clinical Pseudo-notes for Emergency Department Prediction Tasks using Multiple Embedding Model for EHR (MEME)

    [https://arxiv.org/abs/2402.00160](https://arxiv.org/abs/2402.00160)

    本研究提出了一种名为“MEME”的多重嵌入模型，将电子健康记录视为多模态数据。通过结合“伪笔记”和多模态方法，该模型在紧急科室预测任务中表现出优越性能，超过了单模态嵌入方法和传统机器学习方法。然而，该模型在不同医院机构之间存在泛化能力方面的局限性。

    

    在这项工作中，我们引入了针对电子健康记录（EHR）的多重嵌入模型（MEME），这种方法将EHR视为多模态数据。该方法包括“伪笔记”，即对表格形式的EHR概念（如诊断和药物）进行文本表示，使我们能够有效地使用大型语言模型（LLM）进行EHR表示。该框架还采用了多模态方法，分别嵌入每个EHR模态。我们通过在多个医院系统的急诊科中应用MEME来证明其有效性。我们的研究结果表明，MEME在性能上超过了单模态嵌入方法和传统的机器学习方法。然而，我们还观察到所有测试模型在不同医院机构之间的泛化能力方面存在明显的局限性。

    In this work, we introduce Multiple Embedding Model for EHR (MEME), an approach that views Electronic Health Records (EHR) as multimodal data. This approach incorporates "pseudo-notes", textual representations of tabular EHR concepts such as diagnoses and medications, and allows us to effectively employ Large Language Models (LLMs) for EHR representation. This framework also adopts a multimodal approach, embedding each EHR modality separately. We demonstrate the effectiveness of MEME by applying it to several tasks within the Emergency Department across multiple hospital systems. Our findings show that MEME surpasses the performance of both single modality embedding methods and traditional machine learning approaches. However, we also observe notable limitations in generalizability across hospital institutions for all tested models.
    
[^12]: NOLA: 网络作为低秩随机基向量的线性组合

    NOLA: Networks as Linear Combination of Low Rank Random Basis. (arXiv:2310.02556v1 [cs.CL])

    [http://arxiv.org/abs/2310.02556](http://arxiv.org/abs/2310.02556)

    该论文介绍了一种名为NOLA的方法，该方法通过将网络表示为低秩随机基向量的线性组合来减少语言模型的参数数量，从而实现高效的适应和存储。

    

    最近，由于大型语言模型（LLMs）在各种下游任务上表现出的惊人少样本性能，它们受到了广泛关注。然而，由于检查点的庞大大小（例如GPT-3的350GB），对所有参数进行微调并为每个下游任务或领域存储一个唯一模型变得不切实际。当前的文献，例如LoRA，展示了对LLM的原始权重进行低秩修改的潜力，从而实现了针对特定任务的模型的高效适应和存储。这些方法可以将微调LLM所需的参数数量减少几个数量级。然而，这些方法面临两个主要限制：1）参数减少受到秩一分解的下界限制，2）减少的程度受到模型架构和选择的秩的严重影响。例如，在更大模型中，即使是秩一分解，参数的数量也可能超过真正需要进行适应的参数数量。在这篇论文中，我们介绍了NOLA，它通过将网络表示为低秩随机基向量的线性组合，解决了这些限制。

    Large Language Models (LLMs) have recently gained popularity due to their impressive few-shot performance across various downstream tasks. However, fine-tuning all parameters and storing a unique model for each downstream task or domain becomes impractical because of the massive size of checkpoints (e.g., 350GB in GPT-3). Current literature, such as LoRA, showcases the potential of low-rank modifications to the original weights of an LLM, enabling efficient adaptation and storage for task-specific models. These methods can reduce the number of parameters needed to fine-tune an LLM by several orders of magnitude. Yet, these methods face two primary limitations: 1) the parameter reduction is lower-bounded by the rank one decomposition, and 2) the extent of reduction is heavily influenced by both the model architecture and the chosen rank. For instance, in larger models, even a rank one decomposition might exceed the number of parameters truly needed for adaptation. In this paper, we intr
    

