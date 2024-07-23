# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can MLLMs Perform Text-to-Image In-Context Learning?](https://rss.arxiv.org/abs/2402.01293) | 本论文探索了将上下文学习扩展到多模态的文本到图像转换任务，并提出了第一个T2I-ICL基准数据集CoBSAT。研究发现MLLMs在解决T2I-ICL问题时面临着多模态和图像生成的固有复杂性，并通过微调和思维链提示等策略取得了显著改进。 |
| [^2] | [Will the Real Linda Please Stand up...to Large Language Models? Examining the Representativeness Heuristic in LLMs](https://arxiv.org/abs/2404.01461) | 该研究调查了代表性启发式对大型语言模型推理的影响，并创建了专门的数据集进行实验验证 |
| [^3] | [ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models](https://arxiv.org/abs/2403.20262) | 该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。 |
| [^4] | [Can LLMs Converse Formally? Automatically Assessing LLMs in Translating and Interpreting Formal Specifications](https://arxiv.org/abs/2403.18327) | 本文提出了一种方法，利用两个LLM的副本与验证器结合使用，能够自动评估其在自然语言描述和正式规范之间转换的能力，无需额外的人工输入。 |
| [^5] | [FEEL: A Framework for Evaluating Emotional Support Capability with Large Language Models](https://arxiv.org/abs/2403.15699) | 提出了一个基于大型语言模型的框架FEEL，用于评估情感支持能力，解决了当前非人工方法在评估情感支持能力方面面临的挑战，并采用了概率分布方法和集成学习以获得更稳定和全面的结果。 |
| [^6] | [Emotion Detection with Transformers: A Comparative Study](https://arxiv.org/abs/2403.15454) | 本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。 |
| [^7] | [Automatic Interactive Evaluation for Large Language Models with State Aware Patient Simulator](https://arxiv.org/abs/2403.08495) | 介绍了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），以动态、真实的平台评估LLMs，弥补传统评估方法无法满足临床任务需求的不足。 |
| [^8] | [Beyond Memorization: The Challenge of Random Memory Access in Language Models](https://arxiv.org/abs/2403.07805) | 本文研究了语言模型在访问内存时的挑战，发现通过背诵和置换等技术可以改善语言模型的随机内存访问能力，从而在开放域问题回答任务中取得显著改进。 |
| [^9] | [Query-OPT: Optimizing Inference of Large Language Models via Multi-Query Instructions in Meeting Summarization](https://arxiv.org/abs/2403.00067) | 本研究旨在通过将相同输入上下文的查询组合为单个提示，以最小化重复调用来优化使用大型语言模型在会议摘要中的推理。 |
| [^10] | [Meta-Task Prompting Elicits Embedding from Large Language Models](https://arxiv.org/abs/2402.18458) | 提出了一种新的无监督嵌入方法MetaEOL，通过元任务提示引导大型语言模型生成高质量句子嵌入，无需模型微调或特定任务工程，实验表明其在语义文本相似性测试和下游任务中表现出色 |
| [^11] | [OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist Autonomous Agents for Desktop and Web](https://arxiv.org/abs/2402.17553) | OmniACT是一个针对代理生成可执行程序完成计算机任务能力的数据集和基准，超越了传统Web自动化，涵盖了各种桌面应用。 |
| [^12] | [Mysterious Projections: Multimodal LLMs Gain Domain-Specific Visual Capabilities Without Richer Cross-Modal Projections](https://arxiv.org/abs/2402.16832) | MLLMs通过微调获得了特定领域的视觉能力，但投影并未提取相关的领域特定视觉属性。 |
| [^13] | [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822) | Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。 |
| [^14] | [Language Models as Science Tutors](https://arxiv.org/abs/2402.11111) | 介绍了TutorEval和TutorChat，通过TutorEval基准可以衡量LMs作为科学助手的实际可用性，TutorChat数据集用于微调模型。 |
| [^15] | [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) | ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。 |
| [^16] | [MAPLE: Multilingual Evaluation of Parameter Efficient Finetuning of Large Language Models](https://arxiv.org/abs/2401.07598) | MAPLE通过在两个多语言指令调整数据集上对LLama-2-7B和Mistral-7B模型进行微调，在六项涵盖40种语言的下游任务中发现了微调对模型性能的影响。 |
| [^17] | [Efficient Pre-training for Localized Instruction Generation of Videos](https://arxiv.org/abs/2311.15964) | 提出了一种名为Sieve-&-Swap的技术，通过自动筛选出不相关文本并用人类编写的说明替换文本转录，从而实现视频本地化指令生成的高效预训练。 |
| [^18] | [Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access.](http://arxiv.org/abs/2401.09967) | 本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。 |
| [^19] | [Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages.](http://arxiv.org/abs/2401.05811) | 本文引入了对比对齐指令（AlignInstruct），通过使用统计词对齐构建的跨语言鉴别器实现了跨语言监督，解决了机器翻译中的两个挑战：将支持的语言扩展到未知语言和低资源语言中数据缺乏的问题。实验结果表明，LLMs通过MTInstruct可以有效地翻译未知语言，并且使用AlignInstruct在涉及英语的48个翻译方向上能够持续改善翻译质量。基于鉴别器的指令优于生成型指令。 |
| [^20] | [On Detecting Cherry-picking in News Coverage Using Large Language Models.](http://arxiv.org/abs/2401.05650) | 本研究介绍了Cherry，一种创新的方法，通过找出目标新闻报道中缺失的重要陈述来自动检测新闻文章中的摘选陈述。Cherry利用多个来源的新闻报道分析来识别摘选实例。 |
| [^21] | [Cross-Speaker Encoding Network for Multi-Talker Speech Recognition.](http://arxiv.org/abs/2401.04152) | 本文提出了一种叫做Cross-Speaker Encoding（CSE）的网络，用于解决多说话人语音识别中的局限性，通过聚合跨说话人表示。通过与SOT结合，该模型在两个说话人的数据集上实验证明比SIMO基准模型的词错误率（WER）分别降低了8%和10%。 |
| [^22] | [Fast and Optimal Weight Update for Pruned Large Language Models.](http://arxiv.org/abs/2401.02938) | 本研究提出了一种基于交替方向乘积算法(ADMM)的快速且最优的修剪层权重更新算法，结合简单的迭代修剪掩码选择，在广泛的大型语言模型范围内实现了最先进的修剪性能。 |
| [^23] | [Video Understanding with Large Language Models: A Survey.](http://arxiv.org/abs/2312.17432) | 这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。 |
| [^24] | [General-Purpose Retrieval-Enhanced Medical Prediction Model Using Near-Infinite History.](http://arxiv.org/abs/2310.20204) | 基于电子健康记录，我们提出了一种称为REMed的检索增强医学预测模型，通过无限评估临床事件并自动选择相关事件进行预测，消除了人工特征选择和观察窗口的限制，并在实验中表现出优异的效果。 |
| [^25] | [Multiscale Superpixel Structured Difference Graph Convolutional Network for VL Representation.](http://arxiv.org/abs/2310.13447) | 本文提出了一种多尺度超像素结构差异图卷积网络（MDGCN）用于视觉语言表征，通过聚类感知相似像素，减少了后续处理的视觉基元数量，并挖掘了更精确的拓扑关系。 |
| [^26] | [Evaluating large language models' ability to understand metaphor and sarcasm using a screening test for Asperger syndrome.](http://arxiv.org/abs/2309.10744) | 该研究使用一个评分测试来评估大型语言模型（LLMs）理解人类微妙交流的能力。研究结果发现，随着模型参数数量的增加，LLMs对隐喻理解能力有所改善，但对讽刺理解能力的改进并未观察到。 |
| [^27] | [Long-Term Memorability On Advertisements.](http://arxiv.org/abs/2309.00378) | 本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。 |
| [^28] | [Long-range Language Modeling with Self-retrieval.](http://arxiv.org/abs/2306.13421) | 本论文提出了一种名为Retrieval-Pretrained Transformer的模型，可以从头开始联合训练语言模型和检索器来模拟长文本。模型可以计算文本块的查询表示，并将其用于检索前面的块，从而融合信息以预测下一个目标块。检索器使用一个语义目标进行训练，目标是检索那些增加下一个块概率的块。 |
| [^29] | [BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks.](http://arxiv.org/abs/2305.17100) | BiomedGPT是一种面向视觉、语言和多模态任务的通用生物医学生成预训练Transformer，在多个临床任务中取得了16个最新的最优结果，包括超过了OpenAI的GPT-4V和Google的Med-PaLM M（12B）。同时，BiomedGPT还支持零-shot迁移学习。 |
| [^30] | [MLRegTest: A Benchmark for the Machine Learning of Regular Languages.](http://arxiv.org/abs/2304.07687) | 本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。 |
| [^31] | [Diagnostic Benchmark and Iterative Inpainting for Layout-Guided Image Generation.](http://arxiv.org/abs/2304.06671) | 本文提出了布局引导下图像生成的诊断基准LayoutBench，对数量、位置、大小和形状四种空间控制技能进行了研究，发现好的ID布局控制在任意布局的野外环境下可能不具有良好的推广性。接着，我们提出了一种新的基准方法IterInpaint通过修复逐步生成前景和背景区域，显现出在OOD布局方面更强的通用性。 |
| [^32] | [Active Prompting with Chain-of-Thought for Large Language Models.](http://arxiv.org/abs/2302.12246) | 本论文提出了一种新的方法Active-Prompt，它使用任务特定的示例提示适应大型语言模型中的不同任务，提高模型性能与效率。 |
| [^33] | [Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals.](http://arxiv.org/abs/2302.04449) | 本论文提出了阅读并奖励的框架，通过阅读Atari游戏开发者发布的指导手册，以提高强化学习算法在Atari游戏中的效率。该框架包含一个QA提取模块和一个推理模块，能够从指导手册中提取关键信息，并评估物体与智能体的交互效果。 |
| [^34] | [Explaining Machine Learning Models in Natural Conversations: Towards a Conversational XAI Agent.](http://arxiv.org/abs/2209.02552) | 本研究将解释人工智能（XAI）融入到一个对话代理中，设计具有自然语言理解和生成组件的标准模型。通过扩展XAI问题库并提供解释方法，实现了关于机器学习模型的真正自然对话。 |
| [^35] | [Empowering Fake-News Mitigation: Insights from Sharers' Social Media Post-Histories.](http://arxiv.org/abs/2203.10560) | 本论文提出消费者的社交媒体帖子历史是研究分享虚假新闻动机的一种被低估的数据来源。通过对帖子历史提取的文本线索，我们发现虚假新闻分享者在言辞上更多涉及愤怒、宗教和权力。并且，通过将帖子历史中的文本线索加入模型，可以提高预测分享虚假新闻的准确性。此外，通过激活宗教价值观和减少愤怒，可以减少虚假新闻的分享和更广泛的分享。 |

# 详细

[^1]: MLLMs能否进行上下文学习的文本到图像转换？

    Can MLLMs Perform Text-to-Image In-Context Learning?

    [https://rss.arxiv.org/abs/2402.01293](https://rss.arxiv.org/abs/2402.01293)

    本论文探索了将上下文学习扩展到多模态的文本到图像转换任务，并提出了第一个T2I-ICL基准数据集CoBSAT。研究发现MLLMs在解决T2I-ICL问题时面临着多模态和图像生成的固有复杂性，并通过微调和思维链提示等策略取得了显著改进。

    

    从大型语言模型（LLMs）发展到多模式大型语言模型（MLLMs）推动了将上下文学习（ICL）扩展到多模式的研究。现有的研究主要集中在图像到文本的ICL上。然而，文本到图像的ICL（T2I-ICL）具有独特的特性和潜在的应用，但仍然少有研究。为了填补这个空白，我们正式定义了T2I-ICL任务，并提出了CoBSAT，第一个包含十个任务的T2I-ICL基准数据集。利用我们的数据集评估了六个最先进的MLLMs，我们发现MLLMs在解决T2I-ICL问题时面临着相当大的困难。我们确定了多模态和图像生成的固有复杂性是主要挑战。为了克服这些挑战，我们探索了微调和思维链提示等策略，并取得了显著的改进。我们的代码和数据集可以在\url{https://github.com/UW-Madison-Lee-Lab/CoBSAT}上获得。

    The evolution from Large Language Models (LLMs) to Multimodal Large Language Models (MLLMs) has spurred research into extending In-Context Learning (ICL) to its multimodal counterpart. Existing such studies have primarily concentrated on image-to-text ICL. However, the Text-to-Image ICL (T2I-ICL), with its unique characteristics and potential applications, remains underexplored. To address this gap, we formally define the task of T2I-ICL and present CoBSAT, the first T2I-ICL benchmark dataset, encompassing ten tasks. Utilizing our dataset to benchmark six state-of-the-art MLLMs, we uncover considerable difficulties MLLMs encounter in solving T2I-ICL. We identify the primary challenges as the inherent complexity of multimodality and image generation. To overcome these challenges, we explore strategies like fine-tuning and Chain-of-Thought prompting, demonstrating notable improvements. Our code and dataset are available at \url{https://github.com/UW-Madison-Lee-Lab/CoBSAT}.
    
[^2]: 请真正的琳达站出来...面对大语言模型？在LLMs中审视代表性启发式

    Will the Real Linda Please Stand up...to Large Language Models? Examining the Representativeness Heuristic in LLMs

    [https://arxiv.org/abs/2404.01461](https://arxiv.org/abs/2404.01461)

    该研究调查了代表性启发式对大型语言模型推理的影响，并创建了专门的数据集进行实验验证

    

    尽管大型语言模型（LLMs）在理解文本和生成类似人类文本方面表现出色，但它们可能会展现出从训练数据中获得的偏见。具体而言，LLMs可能会容易受到人类决策中的一种常见认知陷阱影响，即代表性启发式。这是心理学中的一个概念，指的是根据事件与一个众所周知的原型或典型例子的相似程度来判断事件发生的可能性，而不考虑更广泛的事实或统计证据。本研究调查了代表性启发式对LLM推理的影响。我们创建了REHEAT（Representativeness Heuristic AI Testing），一个包含涵盖六种常见代表性启发式类型问题的数据集。实验显示，应用于REHEAT的四个LLMs都表现出代表性启发式偏见。我们进一步确定了模型的推理步骤

    arXiv:2404.01461v1 Announce Type: new  Abstract: Although large language models (LLMs) have demonstrated remarkable proficiency in understanding text and generating human-like text, they may exhibit biases acquired from training data in doing so. Specifically, LLMs may be susceptible to a common cognitive trap in human decision-making called the representativeness heuristic. This is a concept in psychology that refers to judging the likelihood of an event based on how closely it resembles a well-known prototype or typical example versus considering broader facts or statistical evidence. This work investigates the impact of the representativeness heuristic on LLM reasoning. We created REHEAT (Representativeness Heuristic AI Testing), a dataset containing a series of problems spanning six common types of representativeness heuristics. Experiments reveal that four LLMs applied to REHEAT all exhibited representativeness heuristic biases. We further identify that the model's reasoning steps
    
[^3]: ELITR-Bench: 面向长上下文语言模型的会议助理基准

    ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models

    [https://arxiv.org/abs/2403.20262](https://arxiv.org/abs/2403.20262)

    该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。

    

    最近，对大型语言模型（LLMs）的研究越来越受到关注，主要致力于扩展模型的上下文大小，以更好地捕捉长文档内部的依赖关系。尽管已经提出了用于评估长距离能力的基准，但现有的努力主要考虑的是不一定与现实应用相关的通用任务。相反，我们的工作提出了一个针对实际会议助理场景的长上下文LLMs的新基准。在这种情景下，长上下文由自动语音识别获得的转录组成，由于这些数据的固有嘈杂性和口语特性，这为LLMs提出了独特的挑战。我们的基准，名为ELITR-Bench，通过271个手工制作的问题及其真实答案来增强现有的ELITR语料库的转录。我们在ELITR-Bench上对最新的长上下文LLMs进行的实验凸显了开源模型和专有模型之间的差距。

    arXiv:2403.20262v1 Announce Type: cross  Abstract: Research on Large Language Models (LLMs) has recently witnessed an increasing interest in extending models' context size to better capture dependencies within long documents. While benchmarks have been proposed to assess long-range abilities, existing efforts primarily considered generic tasks that are not necessarily aligned with real-world applications. In contrast, our work proposes a new benchmark for long-context LLMs focused on a practical meeting assistant scenario. In this scenario, the long contexts consist of transcripts obtained by automatic speech recognition, presenting unique challenges for LLMs due to the inherent noisiness and oral nature of such data. Our benchmark, named ELITR-Bench, augments the existing ELITR corpus' transcripts with 271 manually crafted questions and their ground-truth answers. Our experiments with recent long-context LLMs on ELITR-Bench highlight a gap between open-source and proprietary models, e
    
[^4]: LLM可以进行正式对话吗？自动评估LLMs在转换和解释正式规范中的表现

    Can LLMs Converse Formally? Automatically Assessing LLMs in Translating and Interpreting Formal Specifications

    [https://arxiv.org/abs/2403.18327](https://arxiv.org/abs/2403.18327)

    本文提出了一种方法，利用两个LLM的副本与验证器结合使用，能够自动评估其在自然语言描述和正式规范之间转换的能力，无需额外的人工输入。

    

    利益相关者经常用自然语言描述系统需求，然后由领域专家将其转换为形式化语法，从而增加设计成本。本文评估了大型语言模型（LLMs）在自然语言描述和正式规范之间转换的能力。我们提出了一种方法，可以利用两个LLM的副本与现成的验证器结合使用，无需任何额外的人工输入就可以自动评估其翻译能力。我们的方法使用语言语法生成形式化语法，自动生成数据集。我们进行了经验评估以衡量这种翻译任务的准确性。

    arXiv:2403.18327v1 Announce Type: cross  Abstract: Stakeholders often describe system requirements using natural language which are then converted to formal syntax by a domain-expert leading to increased design costs. This paper assesses the capabilities of Large Language Models (LLMs) in converting between natural language descriptions and formal specifications. Existing work has evaluated the capabilities of LLMs in generating formal syntax such as source code but such experiments are typically hand-crafted and use problems that are likely to be in the training set of LLMs, and often require human-annotated datasets. We propose an approach that can use two copies of an LLM in conjunction with an off-the-shelf verifier to automatically evaluate its translation abilities without any additional human input. Our approach generates formal syntax using language grammars to automatically generate a dataset. We conduct an empirical evaluation to measure the accuracy of this translation task 
    
[^5]: FEEL：用于评估大型语言模型情感支持能力的框架

    FEEL: A Framework for Evaluating Emotional Support Capability with Large Language Models

    [https://arxiv.org/abs/2403.15699](https://arxiv.org/abs/2403.15699)

    提出了一个基于大型语言模型的框架FEEL，用于评估情感支持能力，解决了当前非人工方法在评估情感支持能力方面面临的挑战，并采用了概率分布方法和集成学习以获得更稳定和全面的结果。

    

    情感支持对话（ESC）是一种典型的对话，可以有效地帮助用户缓解情感压力。然而，由于情感分析中涉及固有主观性，当前非人工方法在有效评估情感支持能力方面面临挑战。这些指标与人类判断之间存在很低的相关性。同时，手动评估方法将导致很高的成本。为解决这些问题，我们提出了一个新型模型FEEL（用大型语言模型评估情感支持能力的框架），采用大型语言模型（LLMs）作为评估者来评估情感支持能力。该模型周密考虑ESC的各种评估方面，应用更全面和准确的ESC评估方法。此外，它采用概率分布方法以获得更稳定的结果，并集成了集成学习。

    arXiv:2403.15699v1 Announce Type: new  Abstract: Emotional Support Conversation (ESC) is a typical dialogue that can effec-tively assist the user in mitigating emotional pressures. However, owing to the inherent subjectivity involved in analyzing emotions, current non-artificial methodologies face challenges in effectively appraising the emo-tional support capability. These metrics exhibit a low correlation with human judgments. Concurrently, manual evaluation methods extremely will cause high costs. To solve these problems, we propose a novel model FEEL (Framework for Evaluating Emotional Support Capability with Large Lan-guage Models), employing Large Language Models (LLMs) as evaluators to assess emotional support capabilities. The model meticulously considers var-ious evaluative aspects of ESC to apply a more comprehensive and accurate evaluation method for ESC. Additionally, it employs a probability distribu-tion approach for a more stable result and integrates an ensemble learnin
    
[^6]: 使用Transformer进行情感检测：一项比较研究

    Emotion Detection with Transformers: A Comparative Study

    [https://arxiv.org/abs/2403.15454](https://arxiv.org/abs/2403.15454)

    本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。

    

    在这项研究中，我们探讨了基于Transformer模型在文本数据情感分类中的应用。我们使用不同变体的Transformer对Emotion数据集进行训练和评估。论文还分析了一些影响模型性能的因素，比如Transformer层的微调、层的可训练性以及文本数据的预处理。我们的分析表明，常用技术如去除标点符号和停用词可能会阻碍模型的性能。这可能是因为Transformer的优势在于理解文本内的语境关系。像标点符号和停用词这样的元素仍然可以传达情感或强调，去除它们可能会破坏这种上下文。

    arXiv:2403.15454v1 Announce Type: new  Abstract: In this study, we explore the application of transformer-based models for emotion classification on text data. We train and evaluate several pre-trained transformer models, on the Emotion dataset using different variants of transformers. The paper also analyzes some factors that in-fluence the performance of the model, such as the fine-tuning of the transformer layer, the trainability of the layer, and the preprocessing of the text data. Our analysis reveals that commonly applied techniques like removing punctuation and stop words can hinder model performance. This might be because transformers strength lies in understanding contextual relationships within text. Elements like punctuation and stop words can still convey sentiment or emphasis and removing them might disrupt this context.
    
[^7]: 具有状态感知病人模拟器的大型语言模型自动互动评估

    Automatic Interactive Evaluation for Large Language Models with State Aware Patient Simulator

    [https://arxiv.org/abs/2403.08495](https://arxiv.org/abs/2403.08495)

    介绍了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），以动态、真实的平台评估LLMs，弥补传统评估方法无法满足临床任务需求的不足。

    

    大型语言模型（LLMs）在人机互动中表现出色，但在医疗领域的应用仍未得到充分探索。本文引入了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），旨在弥补传统LLM评估与临床实践的微妙需求之间的差距。

    arXiv:2403.08495v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable proficiency in human interactions, yet their application within the medical field remains insufficiently explored. Previous works mainly focus on the performance of medical knowledge with examinations, which is far from the realistic scenarios, falling short in assessing the abilities of LLMs on clinical tasks. In the quest to enhance the application of Large Language Models (LLMs) in healthcare, this paper introduces the Automated Interactive Evaluation (AIE) framework and the State-Aware Patient Simulator (SAPS), targeting the gap between traditional LLM evaluations and the nuanced demands of clinical practice. Unlike prior methods that rely on static medical knowledge assessments, AIE and SAPS provide a dynamic, realistic platform for assessing LLMs through multi-turn doctor-patient simulations. This approach offers a closer approximation to real clinical scenarios and allows f
    
[^8]: 超越死记硬背：语言模型中的随机内存访问挑战

    Beyond Memorization: The Challenge of Random Memory Access in Language Models

    [https://arxiv.org/abs/2403.07805](https://arxiv.org/abs/2403.07805)

    本文研究了语言模型在访问内存时的挑战，发现通过背诵和置换等技术可以改善语言模型的随机内存访问能力，从而在开放域问题回答任务中取得显著改进。

    

    最近语言模型(LMs)的发展展示了它们在NLP任务中的有效性，尤其是在知识密集型任务中。然而，在其参数内部的知识存储和内存访问机制仍然令人费解。本文探讨了生成式语言模型（如GPT-2）是否能够顺序或随机地访问其内存。通过精心设计的合成任务，涵盖全面背诵、选择性背诵和基于问题回答的情景，我们揭示了LMs能够顺序访问其内存，同时在随机访问已记忆内容时遇到挑战。我们发现，通过背诵和置换等技术可以提高LMs的随机内存访问能力。此外，通过将这种干预应用于开放域问题回答的现实场景，我们验证了通过背诵来增强随机访问技术对问题回答能力的显著改进。

    arXiv:2403.07805v1 Announce Type: cross  Abstract: Recent developments in Language Models (LMs) have shown their effectiveness in NLP tasks, particularly in knowledge-intensive tasks. However, the mechanisms underlying knowledge storage and memory access within their parameters remain elusive. In this paper, we investigate whether a generative LM (e.g., GPT-2) is able to access its memory sequentially or randomly. Through carefully-designed synthetic tasks, covering the scenarios of full recitation, selective recitation and grounded question answering, we reveal that LMs manage to sequentially access their memory while encountering challenges in randomly accessing memorized content. We find that techniques including recitation and permutation improve the random memory access capability of LMs. Furthermore, by applying this intervention to realistic scenarios of open-domain question answering, we validate that enhancing random access by recitation leads to notable improvements in questi
    
[^9]: Query-OPT：通过多查询指令优化大型语言模型在会议摘要中的推理

    Query-OPT: Optimizing Inference of Large Language Models via Multi-Query Instructions in Meeting Summarization

    [https://arxiv.org/abs/2403.00067](https://arxiv.org/abs/2403.00067)

    本研究旨在通过将相同输入上下文的查询组合为单个提示，以最小化重复调用来优化使用大型语言模型在会议摘要中的推理。

    

    这项工作关注基于查询的会议摘要任务，在此任务中，针对特定查询对上下文（会议记录）生成摘要。使用大型语言模型（LLMs）进行此任务时，即使上下文保持不变，每个新查询也需要对LLM推理端点/API进行一次新调用。然而，反复调用LLM推理端点会显著增加在生产中使用它们的成本，这使得许多实际用例中LLMs都不切实际。为解决这一问题，在本文中，我们研究了是否可以成功地将相同输入上下文的查询组合为单个提示以最小化重复调用，在会议摘要中使用。在这方面，我们通过比较各种流行的LLM（GPT-4、PaLM-2、LLaMA-2、Mistral和FLAN-T5）在单查询和多查询设置中的表现进行了广泛实验。

    arXiv:2403.00067v1 Announce Type: new  Abstract: This work focuses on the task of query-based meeting summarization in which the summary of a context (meeting transcript) is generated in response to a specific query. When using Large Language Models (LLMs) for this task, a new call to the LLM inference endpoint/API is required for each new query even if the context stays the same. However, repeated calls to the LLM inference endpoints would significantly increase the costs of using them in production, making LLMs impractical for many real-world use cases. To address this problem, in this paper, we investigate whether combining the queries for the same input context in a single prompt to minimize repeated calls can be successfully used in meeting summarization. In this regard, we conduct extensive experiments by comparing the performance of various popular LLMs: GPT-4, PaLM-2, LLaMA-2, Mistral, and FLAN-T5 in single-query and multi-query settings. We observe that while most LLMs tend to
    
[^10]: 使用元任务提示从大型语言模型中引出嵌入

    Meta-Task Prompting Elicits Embedding from Large Language Models

    [https://arxiv.org/abs/2402.18458](https://arxiv.org/abs/2402.18458)

    提出了一种新的无监督嵌入方法MetaEOL，通过元任务提示引导大型语言模型生成高质量句子嵌入，无需模型微调或特定任务工程，实验表明其在语义文本相似性测试和下游任务中表现出色

    

    在这项工作中，我们引入了一种新的无监督嵌入方法，即带显式单词限制的元任务提示（MetaEOL），用于从大型语言模型（LLMs）中生成高质量的句子嵌入，无需对模型进行微调或特定任务的工程。通过利用元任务提示，MetaEOL引导LLMs通过一系列精心设计的提示生成嵌入，这些提示涵盖了多个表示方面。我们全面的实验表明，从各种元任务平均得到的嵌入在语义文本相似性（STS）基准测试中表现出竞争力，并在下游任务中表现卓越，超越了对比训练模型。我们的发现提出了一种嵌入生成的新的扩展定律，为跨多种以句子为中心的场景中的嵌入提取提供了一种多才多艺、资源高效的方法。

    arXiv:2402.18458v1 Announce Type: new  Abstract: In this work, we introduce a new unsupervised embedding method, Meta-Task Prompting with Explicit One-Word Limitation (MetaEOL), for generating high-quality sentence embeddings from Large Language Models (LLMs) without the need for model fine-tuning or task-specific engineering. Leveraging meta-task prompting, MetaEOL guides LLMs to produce embeddings through a series of carefully designed prompts that address multiple representational aspects. Our comprehensive experiments demonstrate that embeddings averaged from various meta-tasks yield competitive performance on Semantic Textual Similarity (STS) benchmarks and excel in downstream tasks, surpassing contrastive-trained models. Our findings suggest a new scaling law for embedding generation, offering a versatile, resource-efficient approach for embedding extraction across diverse sentence-centric scenarios.
    
[^11]: OmniACT：用于启用桌面和Web多模式通用主动智能体的数据集和基准

    OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist Autonomous Agents for Desktop and Web

    [https://arxiv.org/abs/2402.17553](https://arxiv.org/abs/2402.17553)

    OmniACT是一个针对代理生成可执行程序完成计算机任务能力的数据集和基准，超越了传统Web自动化，涵盖了各种桌面应用。

    

    几十年来，人机交互从根本上一直是手动的。即使在今天，几乎所有在计算机上进行的高效工作都需要人类在每一步都提供输入。虚拟主动智能代表了自动化许多这些琐碎任务的一个激动人心的步骤。虚拟代理将使技术能力有限的用户能够充分利用计算机系统的各种可能性。它们还可以实现高效地简化许多计算机任务，从日历管理到复杂的旅行预订，减少人类干预。在这篇论文中，我们介绍了 OmniACT，这是一个用于评估代理生成可执行程序来完成计算机任务能力的首个数据集和基准。我们的范围超越了传统的Web自动化，涵盖了各种桌面应用。该数据集包含诸如"播放下一首歌"之类的基本任务，以及更为长期的任务

    arXiv:2402.17553v1 Announce Type: new  Abstract: For decades, human-computer interaction has fundamentally been manual. Even today, almost all productive work done on the computer necessitates human input at every step. Autonomous virtual agents represent an exciting step in automating many of these menial tasks. Virtual agents would empower users with limited technical proficiency to harness the full possibilities of computer systems. They could also enable the efficient streamlining of numerous computer tasks, ranging from calendar management to complex travel bookings, with minimal human intervention. In this paper, we introduce OmniACT, the first-of-a-kind dataset and benchmark for assessing an agent's capability to generate executable programs to accomplish computer tasks. Our scope extends beyond traditional web automation, covering a diverse range of desktop applications. The dataset consists of fundamental tasks such as "Play the next song", as well as longer horizon tasks such
    
[^12]: 神秘的投影：多模态LLMs在没有更丰富的跨模态投影的情况下获得特定领域的视觉能力

    Mysterious Projections: Multimodal LLMs Gain Domain-Specific Visual Capabilities Without Richer Cross-Modal Projections

    [https://arxiv.org/abs/2402.16832](https://arxiv.org/abs/2402.16832)

    MLLMs通过微调获得了特定领域的视觉能力，但投影并未提取相关的领域特定视觉属性。

    

    多模态大型语言模型（MLLMs）如LLaVA和GPT-4(V)使得可以进行关于图像的通用对话。然而，现成的MLLMs可能在诸如皮肤病学和农业等领域的图像上具有有限的能力，因此必须进行微调以解锁特定领域的应用。通过对4个数据集进行实验，在两种微调设置下，我们发现随着MLLM的微调，它确实获得了特定领域的视觉能力，但这些更新并没有导致投影提取相关的领域特定视觉属性。

    arXiv:2402.16832v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) like LLaVA and GPT-4(V) enable general-purpose conversations about images with the language modality. As off-the-shelf MLLMs may have limited capabilities on images from domains like dermatology and agriculture, they must be fine-tuned to unlock domain-specific applications. The prevalent architecture of current open-source MLLMs comprises two major modules: an image-language (cross-modal) projection network and a large language model. It is desirable to understand the roles of these two modules in modeling domain-specific visual attributes to inform the design of future models and streamline the interpretability efforts on the current models. To this end, via experiments on 4 datasets and under 2 fine-tuning settings, we find that as the MLLM is fine-tuned, it indeed gains domain-specific visual capabilities, but the updates do not lead to the projection extracting relevant domain-specific visual
    
[^13]: 彩虹团队：多样化对抗性提示的开放式生成

    Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts

    [https://arxiv.org/abs/2402.16822](https://arxiv.org/abs/2402.16822)

    Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。

    

    随着大型语言模型（LLMs）在许多现实世界应用中变得越来越普遍，理解和增强它们对用户输入的稳健性至关重要。现有的用于识别敌对提示的方法往往专注于特定领域，缺乏多样性，或需要大量人工注释。为了解决这些限制，我们提出了彩虹团队，一种用于生成多样化对抗性提示的新方法。彩虹团队将对抗性提示生成视为一个质量 - 多样性问题，并使用开放式搜索来生成既有效又多样的提示。它可以揭示模型在广泛领域内的脆弱性，包括本文中的安全性、问答和网络安全。我们还证明，对由彩虹团队生成的合成数据进行微调可以提高最先进的LLMs的安全性，而不损害它们的一般能力。

    arXiv:2402.16822v1 Announce Type: new  Abstract: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to user inputs is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. It can uncover a model's vulnerabilities across a broad range of domains including, in this paper, safety, question answering, and cybersecurity. We also demonstrate that fine-tuning on synthetic data generated by Rainbow Teaming improves the safety of state-of-the-art LLMs without hurting their general capabilities 
    
[^14]: 语言模型作为科学导师

    Language Models as Science Tutors

    [https://arxiv.org/abs/2402.11111](https://arxiv.org/abs/2402.11111)

    介绍了TutorEval和TutorChat，通过TutorEval基准可以衡量LMs作为科学助手的实际可用性，TutorChat数据集用于微调模型。

    

    NLP最近取得了令人兴奋的进展，朝着训练具有较强科学问题解决能力的语言模型（LMs）的方向发展。然而，模型的发展并没有专注于LMs在科学教育中的实际用例，包括需要处理长篇科学文档的应用。为了解决这个问题，我们引入了TutorEval和TutorChat。TutorEval是一个多样化的问答基准，其中包含有关STEM教科书长篇章节的问题，由专家编写。TutorEval有助于衡量LMs作为科学助手的实际可用性，它是第一个结合长上下文、自由生成和多学科科学知识的基准。此外，我们展示了使用现有对话数据集对基础模型进行微调会导致TutorEval性能不佳。因此，我们创建了TutorChat，这是一个包含80,000个关于教科书的长合成对话的数据集。我们使用TutorChat来微调Llemma模型。

    arXiv:2402.11111v1 Announce Type: cross  Abstract: NLP has recently made exciting progress toward training language models (LMs) with strong scientific problem-solving skills. However, model development has not focused on real-life use-cases of LMs for science, including applications in education that require processing long scientific documents. To address this, we introduce TutorEval and TutorChat. TutorEval is a diverse question-answering benchmark consisting of questions about long chapters from STEM textbooks, written by experts. TutorEval helps measure real-life usability of LMs as scientific assistants, and it is the first benchmark combining long contexts, free-form generation, and multi-disciplinary scientific knowledge. Moreover, we show that fine-tuning base models with existing dialogue datasets leads to poor performance on TutorEval. Therefore, we create TutorChat, a dataset of 80,000 long synthetic dialogues about textbooks. We use TutorChat to fine-tune Llemma models wit
    
[^15]: 一种具有长期上下文概要记忆的人工智能阅读代理

    A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

    [https://arxiv.org/abs/2402.09727](https://arxiv.org/abs/2402.09727)

    ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。

    

    当前的大型语言模型不仅限制在某个最大上下文长度内，而且无法稳定地处理长输入。为了解决这些限制，我们提出了ReadAgent，一个增加了有效上下文长度的语言模型代理系统，在我们的实验中可以达到20倍。受到人类交互式阅读长文档的启发，我们将ReadAgent实现为一个简单的提示系统，利用LLM的高级语言能力来：（1）决定将哪些内容存储在一个记忆片段中，（2）将这些记忆片段压缩成为称为概要记忆的短时记忆，（3）在需要时通过原始文本查找段落来提醒自己相关细节以完成任务。我们使用检索方法、使用原始长上下文以及使用概要记忆来评估ReadAgent与基线的性能。这些评估是在三个长文档阅读理解任务上进行的。

    arXiv:2402.09727v1 Announce Type: cross  Abstract: Current Large Language Models (LLMs) are not only limited to some maximum context length, but also are not able to robustly consume long inputs. To address these limitations, we propose ReadAgent, an LLM agent system that increases effective context length up to 20x in our experiments. Inspired by how humans interactively read long documents, we implement ReadAgent as a simple prompting system that uses the advanced language capabilities of LLMs to (1) decide what content to store together in a memory episode, (2) compress those memory episodes into short episodic memories called gist memories, and (3) take actions to look up passages in the original text if ReadAgent needs to remind itself of relevant details to complete a task. We evaluate ReadAgent against baselines using retrieval methods, using the original long contexts, and using the gist memories. These evaluations are performed on three long-document reading comprehension task
    
[^16]: MAPLE: 多语言参数高效微调大型语言模型的评估

    MAPLE: Multilingual Evaluation of Parameter Efficient Finetuning of Large Language Models

    [https://arxiv.org/abs/2401.07598](https://arxiv.org/abs/2401.07598)

    MAPLE通过在两个多语言指令调整数据集上对LLama-2-7B和Mistral-7B模型进行微调，在六项涵盖40种语言的下游任务中发现了微调对模型性能的影响。

    

    Parameter Efficient Finetuning (PEFT)已经成为改善大型语言模型（LLMs）性能的可行解决方案，而无需大量资源和计算。本文在两个合成多语言指令调整数据集上微调LLama-2-7B和Mistral-7B模型，以确定其对六个涵盖四十种语言的下游任务上模型性能的影响。此外，我们尝试不同的参数，例如用于低秩适应的秩和量化值，以确定它们对下游性能的影响，发现更高的秩和更高的hig

    arXiv:2401.07598v2 Announce Type: replace  Abstract: Parameter Efficient Finetuning (PEFT) has emerged as a viable solution for improving the performance of Large Language Models (LLMs) without requiring massive resources and compute. Prior work on multilingual evaluation has shown that there is a large gap between the performance of LLMs on English and other languages. Further, there is also a large gap between the performance of smaller open-source models and larger LLMs. Finetuning can be an effective way to bridge this gap and make language models more equitable. In this work, we finetune the LLama-2-7B and Mistral-7B models on two synthetic multilingual instruction tuning datasets to determine its effect on model performance on six downstream tasks covering forty languages in all. Additionally, we experiment with various parameters, such as rank for low-rank adaptation and values of quantisation to determine their effects on downstream performance and find that higher rank and hig
    
[^17]: 视频本地化指令生成的高效预训练方法

    Efficient Pre-training for Localized Instruction Generation of Videos

    [https://arxiv.org/abs/2311.15964](https://arxiv.org/abs/2311.15964)

    提出了一种名为Sieve-&-Swap的技术，通过自动筛选出不相关文本并用人类编写的说明替换文本转录，从而实现视频本地化指令生成的高效预训练。

    

    过程视频展示了诸如食谱准备等任务的逐步演示。理解此类视频具有挑战性，需要对步骤进行精确定位并生成文字说明。手动注释步骤并编写说明成本高昂，这限制了当前数据集的规模并阻碍了有效学习。利用大规模但嘈杂的视频-文本数据集进行预训练可以提升性能，但需要大量计算资源。此外，文本转录包含无关内容，与人类注释员编写的说明相比存在风格变化。为了缓解这两个问题，我们提出了一种技术，Sieve-&-Swap，通过自动筛选出不相关文本和使用文本食谱数据集中人类编写的说明自动替换文本转录以增强文字指令的质量。

    arXiv:2311.15964v2 Announce Type: replace-cross  Abstract: Procedural videos show step-by-step demonstrations of tasks like recipe preparation. Understanding such videos is challenging, involving the precise localization of steps and the generation of textual instructions. Manually annotating steps and writing instructions is costly, which limits the size of current datasets and hinders effective learning. Leveraging large but noisy video-transcript datasets for pre-training can boost performance, but demands significant computational resources. Furthermore, transcripts contain irrelevant content and exhibit style variation compared to instructions written by human annotators. To mitigate both issues, we propose a technique, Sieve-&-Swap, to automatically curate a smaller dataset: (i) Sieve filters irrelevant transcripts and (ii) Swap enhances the quality of the text instruction by automatically replacing the transcripts with human-written instructions from a text-only recipe dataset. 
    
[^18]: 无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码

    Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access. (arXiv:2401.09967v1 [cs.CL])

    [http://arxiv.org/abs/2401.09967](http://arxiv.org/abs/2401.09967)

    本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。

    

    有限约束在语言模型输出的控制上提供了一种不需要重新训练或架构修改的方式，但通常只适用于拥有逻辑回归访问权限的模型，这对于黑盒大型语言模型存在限制。本文引入了一种新颖的基于草图引导的黑盒大型语言模型约束解码（SGCD）方法，无需访问黑盒语言模型的逻辑回归。SGCD利用本地辅助模型来优化无约束黑盒语言模型的输出，将其作为进一步扩展的“草图”。此方法可与传统的基于逻辑回归的技术相互补充，使有限约束解码在无法完全透明的模型环境中应用。通过实验展示了SGCD的有效性。

    Constrained decoding, a technique for enforcing constraints on language model outputs, offers a way to control text generation without retraining or architectural modifications. Its application is, however, typically restricted to models that give users access to next-token distributions (usually via softmax logits), which poses a limitation with blackbox large language models (LLMs). This paper introduces sketch-guided constrained decoding (SGCD), a novel approach to constrained decoding for blackbox LLMs, which operates without access to the logits of the blackbox LLM. SGCD utilizes a locally hosted auxiliary model to refine the output of an unconstrained blackbox LLM, effectively treating this initial output as a "sketch" for further elaboration. This approach is complementary to traditional logit-based techniques and enables the application of constrained decoding in settings where full model transparency is unavailable. We demonstrate the efficacy of SGCD through experiments in cl
    
[^19]: 使用对比对齐指令调整LLMs以解决机器翻译中的未知、低资源语言问题

    Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages. (arXiv:2401.05811v1 [cs.CL])

    [http://arxiv.org/abs/2401.05811](http://arxiv.org/abs/2401.05811)

    本文引入了对比对齐指令（AlignInstruct），通过使用统计词对齐构建的跨语言鉴别器实现了跨语言监督，解决了机器翻译中的两个挑战：将支持的语言扩展到未知语言和低资源语言中数据缺乏的问题。实验结果表明，LLMs通过MTInstruct可以有效地翻译未知语言，并且使用AlignInstruct在涉及英语的48个翻译方向上能够持续改善翻译质量。基于鉴别器的指令优于生成型指令。

    

    本文介绍了对比对齐指令（AlignInstruct）来解决大型语言模型（LLMs）上机器翻译（MT）中的两个挑战。一个是将支持的语言扩展到之前未见过的语言。第二个与低资源语言中缺乏数据有关。通过MT指令（MTInstruct）对模型进行微调是应对第一个挑战的一种直接方法。然而，MTInstruct受到第二个挑战中固有的弱语言跨度信号的限制。AlignInstruct通过使用基于统计词对齐构建的跨语言鉴别器来强调跨语言监督。我们基于在多达24种未知语言上对BLOOMZ模型（1b1、3b和7b1）进行微调的结果表明：（1）LLMs可以使用MTInstruct有效地翻译未知语言；（2）AlignInstruct在涉及英语的48个翻译方向上提高了翻译质量的一致性；（3）基于鉴别器的指令优于生成型指令。

    This article introduces contrastive alignment instructions (AlignInstruct) to address two challenges in machine translation (MT) on large language models (LLMs). One is the expansion of supported languages to previously unseen ones. The second relates to the lack of data in low-resource languages. Model fine-tuning through MT instructions (MTInstruct) is a straightforward approach to the first challenge. However, MTInstruct is limited by weak cross-lingual signals inherent in the second challenge. AlignInstruct emphasizes cross-lingual supervision via a cross-lingual discriminator built using statistical word alignments. Our results based on fine-tuning the BLOOMZ models (1b1, 3b, and 7b1) in up to 24 unseen languages showed that: (1) LLMs can effectively translate unseen languages using MTInstruct; (2) AlignInstruct led to consistent improvements in translation quality across 48 translation directions involving English; (3) Discriminator-based instructions outperformed their generativ
    
[^20]: 通过使用大型语言模型检测新闻报道中的摘选

    On Detecting Cherry-picking in News Coverage Using Large Language Models. (arXiv:2401.05650v1 [cs.CL])

    [http://arxiv.org/abs/2401.05650](http://arxiv.org/abs/2401.05650)

    本研究介绍了Cherry，一种创新的方法，通过找出目标新闻报道中缺失的重要陈述来自动检测新闻文章中的摘选陈述。Cherry利用多个来源的新闻报道分析来识别摘选实例。

    

    摘选是指有意选择有利于特定观点的证据或事实，同时忽视或扭曲支持相反观点的证据。在新闻报道中手动识别摘选陈述可能会很具有挑战性，特别是当相反观点的报道缺失时。这项研究引入了一种创新的方法，名为Cherry，用于通过找出目标新闻报道中缺失的重要陈述来自动检测新闻文章中的摘选陈述。Cherry利用多个来源的新闻报道分析来识别摘选实例。我们的方法依赖于考虑来自其他新闻来源的语境信息的语言模型，根据陈述对目标新闻报道所涵盖事件的重要性进行分类。此外，这项研究引入了一个专门用于摘选检测的新数据集，用于训练和评估模型的性能。

    Cherry-picking refers to the deliberate selection of evidence or facts that favor a particular viewpoint while ignoring or distorting evidence that supports an opposing perspective. Manually identifying instances of cherry-picked statements in news stories can be challenging, particularly when the opposing viewpoint's story is absent. This study introduces Cherry, an innovative approach for automatically detecting cherry-picked statements in news articles by finding missing important statements in the target news story. Cherry utilizes the analysis of news coverage from multiple sources to identify instances of cherry-picking. Our approach relies on language models that consider contextual information from other news sources to classify statements based on their importance to the event covered in the target news story. Furthermore, this research introduces a novel dataset specifically designed for cherry-picking detection, which was used to train and evaluate the performance of the mod
    
[^21]: 跨说话人编码网络用于多说话人语音识别

    Cross-Speaker Encoding Network for Multi-Talker Speech Recognition. (arXiv:2401.04152v1 [cs.SD])

    [http://arxiv.org/abs/2401.04152](http://arxiv.org/abs/2401.04152)

    本文提出了一种叫做Cross-Speaker Encoding（CSE）的网络，用于解决多说话人语音识别中的局限性，通过聚合跨说话人表示。通过与SOT结合，该模型在两个说话人的数据集上实验证明比SIMO基准模型的词错误率（WER）分别降低了8%和10%。

    

    端到端的多说话人语音识别已经引起了极大的兴趣，作为一种直接转录多个说话人重叠语音的有效方法。目前的方法通常采用1）带有分支编码器的单输入多输出（SIMO）模型，或者2）基于注意力机制的编码器-解码器架构和序列化输出训练（SOT）的单输入单输出（SISO）模型。在这项工作中，我们提出了一种叫做Cross-Speaker Encoding（CSE）的网络来解决SIMO模型的局限性，通过聚合跨说话人表示。此外，CSE模型与SOT相结合，既发挥了SIMO和SISO的优势，又缓解了它们的缺点。据我们所知，该工作代表了将SIMO和SISO集成到多说话人语音识别中的早期工作。在两个说话人的LibrispeechMix数据集上进行的实验表明，CES模型相比于SIMO基准模型将词错误率（WER）降低了8%。CSE-SOT模型将WER降低了10%

    End-to-end multi-talker speech recognition has garnered great interest as an effective approach to directly transcribe overlapped speech from multiple speakers. Current methods typically adopt either 1) single-input multiple-output (SIMO) models with a branched encoder, or 2) single-input single-output (SISO) models based on attention-based encoder-decoder architecture with serialized output training (SOT). In this work, we propose a Cross-Speaker Encoding (CSE) network to address the limitations of SIMO models by aggregating cross-speaker representations. Furthermore, the CSE model is integrated with SOT to leverage both the advantages of SIMO and SISO while mitigating their drawbacks. To the best of our knowledge, this work represents an early effort to integrate SIMO and SISO for multi-talker speech recognition. Experiments on the two-speaker LibrispeechMix dataset show that the CES model reduces word error rate (WER) by 8% over the SIMO baseline. The CSE-SOT model reduces WER by 10
    
[^22]: 快速且最优的修剪大型语言模型的权重更新。

    Fast and Optimal Weight Update for Pruned Large Language Models. (arXiv:2401.02938v1 [cs.CL])

    [http://arxiv.org/abs/2401.02938](http://arxiv.org/abs/2401.02938)

    本研究提出了一种基于交替方向乘积算法(ADMM)的快速且最优的修剪层权重更新算法，结合简单的迭代修剪掩码选择，在广泛的大型语言模型范围内实现了最先进的修剪性能。

    

    修剪大型语言模型(LLMs)是一个具有挑战性的任务，因为它们的规模庞大。主要困难在于修剪后的模型微调，这是为了恢复因删除权重而导致的性能损失。最近的方法要么完全忽略了微调，专注于高效的修剪标准，要么尝试逐层权重更新，保持每个层的行为。然而，即使是逐层权重更新对LLMs来说也可能代价高昂，之前的工作不得不采用各种近似方法。在我们的论文中，我们提出了一种基于交替方向乘积算法(ADMM)的快速且最优的修剪层权重更新算法。结合简单的迭代修剪掩码选择，我们的算法在广泛的LLMs范围内实现了最先进的修剪性能。代码可以在https://github.com/fmfi-compbio/admm-pruning获得。

    Pruning large language models (LLMs) is a challenging task due to their enormous size. The primary difficulty is fine-tuning the model after pruning, which is needed to recover the lost performance caused by dropping weights. Recent approaches have either ignored fine-tuning entirely, focusing on efficient pruning criteria, or attempted layer-wise weight updates, preserving the behavior of each layer. However, even layer-wise weight updates can be costly for LLMs, and previous works have resorted to various approximations.  In our paper, we propose a fast and optimal weight update algorithm for pruned layers based on the Alternating Direction Method of Multipliers (ADMM). Coupled with a simple iterative pruning mask selection, our algorithm achieves state-of-the-art pruning performance across a wide range of LLMs. Code is available at https://github.com/fmfi-compbio/admm-pruning.
    
[^23]: 大型语言模型在视频理解中的应用：一项调查研究

    Video Understanding with Large Language Models: A Survey. (arXiv:2312.17432v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17432](http://arxiv.org/abs/2312.17432)

    这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。

    

    随着在线视频平台的不断增长和视频内容的不断增多，对熟练的视频理解工具的需求显著增加。鉴于大型语言模型在语言和多模态任务中的卓越能力，本调查提供了对利用大型语言模型（Vid-LLMs）技术进行视频理解的最新进展的详细概述。Vid-LLMs的新兴能力令人惊讶，尤其是它们在开放式时空推理和常识知识方面的能力，为未来的视频理解提供了一个有前途的方向。本调查对Vid-LLMs的独特特点和能力进行了分类，分为四种主要类型：基于LLM的视频代理、Vid-LLMs的预训练、Vid-LLMs的指令调整和混合方法。此外，本调查对Vid-LLMs的任务、数据集和评估方法进行了全面的研究。另外，它还探讨了Vid-LLMs技术的局限性和未来的挑战。

    With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of Large Language Models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey presents a comprehensive study of the tasks, datasets, and evaluation methodologies for Vid-LLMs. Additionally, it explores 
    
[^24]: 利用近无限历史的通用检索增强医学预测模型

    General-Purpose Retrieval-Enhanced Medical Prediction Model Using Near-Infinite History. (arXiv:2310.20204v1 [cs.LG])

    [http://arxiv.org/abs/2310.20204](http://arxiv.org/abs/2310.20204)

    基于电子健康记录，我们提出了一种称为REMed的检索增强医学预测模型，通过无限评估临床事件并自动选择相关事件进行预测，消除了人工特征选择和观察窗口的限制，并在实验中表现出优异的效果。

    

    基于电子健康记录（EHRs）开发临床预测模型（例如死亡预测）通常依赖于专家意见进行特征选择和调整观测窗口大小。这给专家带来负担并在开发过程中造成瓶颈。我们提出了一种检索增强的医学预测模型（REMed），以应对这些挑战。REMed可以基本评估无限量的临床事件，选择相关的事件并进行预测。这种方法有效地消除了需要手动进行特征选择并实时观察的需要。我们通过对27个临床任务和两个公开可用的EHR数据集的独立队列实验验证了这些特性，结果显示REMed优于其他现代架构，它们旨在处理尽可能多的事件。值得注意的是，我们发现REMed的偏好与医学专家的偏好密切相似。我们期望我们的方法能显著加速该领域的发展。

    Developing clinical prediction models (e.g., mortality prediction) based on electronic health records (EHRs) typically relies on expert opinion for feature selection and adjusting observation window size. This burdens experts and creates a bottleneck in the development process. We propose Retrieval-Enhanced Medical prediction model (REMed) to address such challenges. REMed can essentially evaluate an unlimited number of clinical events, select the relevant ones, and make predictions. This approach effectively eliminates the need for manual feature selection and enables an unrestricted observation window. We verified these properties through experiments on 27 clinical tasks and two independent cohorts from publicly available EHR datasets, where REMed outperformed other contemporary architectures that aim to handle as many events as possible. Notably, we found that the preferences of REMed align closely with those of medical experts. We expect our approach to significantly expedite the d
    
[^25]: 多尺度超像素结构差异图卷积网络用于视觉语言表征

    Multiscale Superpixel Structured Difference Graph Convolutional Network for VL Representation. (arXiv:2310.13447v1 [cs.CV])

    [http://arxiv.org/abs/2310.13447](http://arxiv.org/abs/2310.13447)

    本文提出了一种多尺度超像素结构差异图卷积网络（MDGCN）用于视觉语言表征，通过聚类感知相似像素，减少了后续处理的视觉基元数量，并挖掘了更精确的拓扑关系。

    

    在多模态领域中，整合视觉和语言的关键在于建立一个良好的对齐策略。最近，受到自监督学习成功的启发，基于预训练模型的视觉和语言的多模态语义表征取得了重大进展。然而，视觉语义表征仍有改进的空间。当前基于像素或块的方法在准确提取复杂场景边界方面存在空间语义连贯性不足和对噪声的脆弱性的挑战。为此，本文将超像素作为可学习图像数据的综合紧凑表征，通过对感知相似像素进行聚类，有效地减少了后续处理的视觉基元数量。为了挖掘更精确的拓扑关系，我们提出了一种多尺度差异图卷积网络（MDGCN）。它将整个图像解析为细到粗的层次结构，从而实现了整个图像的解析。

    Within the multimodal field, the key to integrating vision and language lies in establishing a good alignment strategy. Recently, benefiting from the success of self-supervised learning, significant progress has been made in multimodal semantic representation based on pre-trained models for vision and language. However, there is still room for improvement in visual semantic representation. The lack of spatial semantic coherence and vulnerability to noise makes it challenging for current pixel or patch-based methods to accurately extract complex scene boundaries. To this end, this paper develops superpixel as a comprehensive compact representation of learnable image data, which effectively reduces the number of visual primitives for subsequent processing by clustering perceptually similar pixels. To mine more precise topological relations, we propose a Multiscale Difference Graph Convolutional Network (MDGCN). It parses the entire image as a fine-to-coarse hierarchical structure of cons
    
[^26]: 评估大型语言模型利用亚斯伯格综合征筛选测试理解隐喻和讽刺的能力

    Evaluating large language models' ability to understand metaphor and sarcasm using a screening test for Asperger syndrome. (arXiv:2309.10744v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.10744](http://arxiv.org/abs/2309.10744)

    该研究使用一个评分测试来评估大型语言模型（LLMs）理解人类微妙交流的能力。研究结果发现，随着模型参数数量的增加，LLMs对隐喻理解能力有所改善，但对讽刺理解能力的改进并未观察到。

    

    隐喻和讽刺是我们高度进化的社交沟通技巧的珍贵成果。然而，亚斯伯格综合征的儿童众所周知在理解讽刺方面存在困难，即使他们具有足够理解隐喻的口语智商水平。鉴于此，已经使用了一个评分测试来评估理解隐喻和讽刺的能力，以区分亚斯伯格综合征和其他表现相似外部行为的症状（例如注意力缺陷/多动障碍）。本研究使用标准化测试来研究最近大型语言模型（LLMs）理解人类微妙交流的能力。结果显示，随着模型参数数量的增加，它们理解隐喻的能力得到了改善，但并没有观察到对讽刺理解的改进。这意味着有必要采取其他方法来使LLMs具备理解讽刺的能力，这已与亚斯伯格综合征相关。

    Metaphors and sarcasm are precious fruits of our highly-evolved social communication skills. However, children with Asperger syndrome are known to have difficulties in comprehending sarcasm, even if they possess a certain level of verbal IQ sufficient for understanding metaphors. Given that, a screening test that scores the ability to understand metaphor and sarcasm has been used to differentiate Asperger syndrome from other symptoms exhibiting akin external behaviors (e.g., attention-deficit/hyperactivity disorder). This study uses the standardized test to examine the capability of recent large language models (LLMs) in understanding human nuanced communication. The results divulged that, whereas their ability to comprehend metaphors has been improved with the increase of the number of model parameters, the improvement in sarcasm understanding was not observed. This implies that an alternative approach is imperative to imbue LLMs with the capacity to grasp sarcasm, which has been asso
    
[^27]: 广告的长期记忆性研究

    Long-Term Memorability On Advertisements. (arXiv:2309.00378v1 [cs.CL])

    [http://arxiv.org/abs/2309.00378](http://arxiv.org/abs/2309.00378)

    本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。

    

    市场营销人员花费数十亿美元在广告上，但是投入到广告上的金钱能起多大作用呢？当顾客在购买时无法辨认出他们看过的品牌的话，花在广告上的钱基本上就被浪费了。尽管在营销中很重要，但迄今为止，在机器学习的文献中还没有关于广告记忆力的研究。大多数研究都是对特定内容类型（如物体和动作视频）进行短期回忆（<5分钟）的研究。另一方面，广告行业只关心长期记忆（几个小时或更长时间），而且广告几乎总是高度多模式化，通过不同的形式（文本、图像和视频）来讲故事。基于这一动机，我们进行了首个大规模记忆性研究，共有1203名参与者和2205个广告涵盖了276个品牌。在不同参与者子群体和广告类型上进行统计测试，我们发现了许多有关什么使广告难忘的有趣见解-无论是内容还是

    Marketers spend billions of dollars on advertisements but to what end? At the purchase time, if customers cannot recognize a brand for which they saw an ad, the money spent on the ad is essentially wasted. Despite its importance in marketing, until now, there has been no study on the memorability of ads in the ML literature. Most studies have been conducted on short-term recall (<5 mins) on specific content types like object and action videos. On the other hand, the advertising industry only cares about long-term memorability (a few hours or longer), and advertisements are almost always highly multimodal, depicting a story through its different modalities (text, images, and videos). With this motivation, we conduct the first large scale memorability study consisting of 1203 participants and 2205 ads covering 276 brands. Running statistical tests over different participant subpopulations and ad-types, we find many interesting insights into what makes an ad memorable - both content and h
    
[^28]: 自检索的长文本语言模型

    Long-range Language Modeling with Self-retrieval. (arXiv:2306.13421v1 [cs.CL])

    [http://arxiv.org/abs/2306.13421](http://arxiv.org/abs/2306.13421)

    本论文提出了一种名为Retrieval-Pretrained Transformer的模型，可以从头开始联合训练语言模型和检索器来模拟长文本。模型可以计算文本块的查询表示，并将其用于检索前面的块，从而融合信息以预测下一个目标块。检索器使用一个语义目标进行训练，目标是检索那些增加下一个块概率的块。

    

    近期，基于检索辅助的语言模型受到了广泛关注。但是，通常检索器并不是作为语言模型的本地组件进行联合训练的，而是被添加到已经预训练好的语言模型中，这限制了语言模型和检索器相互适应的能力。在本文中，我们提出了Retrieval-Pretrained Transformer (RPT)，一种从头开始训练检索辅助的语言模型的架构和训练方法，用于模拟长文本。给定一个最近在长文档中生成的文本块，语言模型计算查询表示，然后用它来检索文档中更早的块，这些块可能跨越数万个标记。检索到的块中的信息被融合到语言模型表示中，以预测下一个目标块。我们用一个语义目标来训练检索器组件，该目标的目的是检索增加下一个块概率的块，根据参考语言模型。我们评估了...

    Retrieval-augmented language models (LMs) have received much attention recently. However, typically the retriever is not trained jointly as a native component of the LM, but added to an already-pretrained LM, which limits the ability of the LM and the retriever to adapt to one another. In this work, we propose the Retrieval-Pretrained Transformer (RPT), an architecture and training procedure for jointly training a retrieval-augmented LM from scratch for the task of modeling long texts. Given a recently generated text chunk in a long document, the LM computes query representations, which are then used to retrieve earlier chunks in the document, located potentially tens of thousands of tokens before. Information from retrieved chunks is fused into the LM representations to predict the next target chunk. We train the retriever component with a semantic objective, where the goal is to retrieve chunks that increase the probability of the next chunk, according to a reference LM. We evaluate 
    
[^29]: BiomedGPT：一种面向视觉、语言和多模态任务的统一且通用的生物医学生成预训练Transformer

    BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks. (arXiv:2305.17100v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17100](http://arxiv.org/abs/2305.17100)

    BiomedGPT是一种面向视觉、语言和多模态任务的通用生物医学生成预训练Transformer，在多个临床任务中取得了16个最新的最优结果，包括超过了OpenAI的GPT-4V和Google的Med-PaLM M（12B）。同时，BiomedGPT还支持零-shot迁移学习。

    

    传统的任务和模态特定的人工智能模型在生物医学领域的实际应用和维护中不够灵活。与此同时，生物医学数据的不断增加，结合现代多模态多任务人工智能技术的进展，为通用的生物医学人工智能解决方案的出现铺平了道路。这些解决方案有潜力解释不同的医疗模态，并产生如自由文本报告或疾病诊断等表达性输出。本文提出了BiomedGPT，这是第一个面向多样化生物医学任务的开源通用视觉语言人工智能模型。BiomedGPT在26个数据集的五个临床重要任务中实现了16个最新的结果。值得注意的是，在放射学人员评估中，它超越了OpenAI的GPT-4 with vision（GPT-4V），并在乳腺癌诊断和医学视觉问题回答方面超过了Google的Med-PaLM M（12B）。此外，BiomedGPT还支持零-shot迁移学习。

    Conventional task- and modality-specific artificial intelligence (AI) models are inflexible in real-world deployment and maintenance for biomedicine. At the same time, the growing availability of biomedical data, coupled with the advancements in modern multi-modal multi-task AI techniques, has paved the way for the emergence of generalist biomedical AI solutions. These solutions hold the potential to interpret different medical modalities and produce expressive outputs such as free-text reports or disease diagnosis. Here, we propose BiomedGPT, the first open-source and generalist visual language AI for diverse biomedical tasks. BiomedGPT achieved 16 state-of-the-art results across five clinically significant tasks on 26 datasets. Notably, it outperformed OpenAI's GPT-4 with vision (GPT-4V) in radiology human evaluation and surpassed Google's Med-PaLM M (12B) in breast cancer diagnosis and medical visual question answering. Moreover, BiomedGPT facilitates zero-shot transfer learning, gr
    
[^30]: MLRegTest：机器学习正则语言的基准测试

    MLRegTest: A Benchmark for the Machine Learning of Regular Languages. (arXiv:2304.07687v1 [cs.LG])

    [http://arxiv.org/abs/2304.07687](http://arxiv.org/abs/2304.07687)

    本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。

    

    评估机器学习系统对已知分类器的学习能力允许细致地检查它们可以学习哪些模式，并在将它们应用于未知分类器的学习时建立信心。本文提出了一个名为MLRegTest的新的序列分类机器学习系统基准测试，其中包含来自1,800个正则语言的训练、开发和测试集。不同类型的形式语言代表着不同种类的长距离依赖，并正确地识别序列中的长距离依赖是机器学习系统成功泛化的已知挑战。MLRegTest根据它们的逻辑复杂度（单调二阶，一阶，命题或单项式表达式）和逻辑文字的种类（字符串，定级字符串，子序列或两者的组合）组织其语言。逻辑复杂度和文字的选择提供了一种系统方法来理解不同种类的长距离依赖和机器学习系统在处理它们时的性能。

    Evaluating machine learning (ML) systems on their ability to learn known classifiers allows fine-grained examination of the patterns they can learn, which builds confidence when they are applied to the learning of unknown classifiers. This article presents a new benchmark for ML systems on sequence classification called MLRegTest, which contains training, development, and test sets from 1,800 regular languages.  Different kinds of formal languages represent different kinds of long-distance dependencies, and correctly identifying long-distance dependencies in sequences is a known challenge for ML systems to generalize successfully. MLRegTest organizes its languages according to their logical complexity (monadic second order, first order, propositional, or monomial expressions) and the kind of logical literals (string, tier-string, subsequence, or combinations thereof). The logical complexity and choice of literal provides a systematic way to understand different kinds of long-distance d
    
[^31]: 布局引导下的图像生成的诊断基准和迭代修复

    Diagnostic Benchmark and Iterative Inpainting for Layout-Guided Image Generation. (arXiv:2304.06671v1 [cs.CV])

    [http://arxiv.org/abs/2304.06671](http://arxiv.org/abs/2304.06671)

    本文提出了布局引导下图像生成的诊断基准LayoutBench，对数量、位置、大小和形状四种空间控制技能进行了研究，发现好的ID布局控制在任意布局的野外环境下可能不具有良好的推广性。接着，我们提出了一种新的基准方法IterInpaint通过修复逐步生成前景和背景区域，显现出在OOD布局方面更强的通用性。

    

    空间控制是可控图像生成的核心能力。在布局引导下的图像生成方面的进展已经显示出在具有类似空间配置的内分布（ID）数据集上有良好的结果。然而，当面对任意不确定的布局的离线分布样本时，这些模型的表现还不清楚。在本文中，我们提出了LayoutBench，这是一种对布局引导下的图像生成进行诊断的基准，它检查了四种空间控制技能：数量，位置，大小和形状。我们对两种最近代表性的布局引导下的图像生成方法进行了基准测试，并观察到良好的ID布局控制可能无法很好地推广到任意布局的野外环境（例如，边界上的对象）。接下来，我们提出了一个新的基准方法IterInpaint，它通过修复逐步生成前景和背景区域，展示出在LayoutBench的OOD布局上更强的通用性。我们进行了数量和定性评估，表明IterInpaint相对于现有方法具有更好的生成多样和视觉上令人愉悦的图像和可控的空间布局。

    Spatial control is a core capability in controllable image generation. Advancements in layout-guided image generation have shown promising results on in-distribution (ID) datasets with similar spatial configurations. However, it is unclear how these models perform when facing out-of-distribution (OOD) samples with arbitrary, unseen layouts. In this paper, we propose LayoutBench, a diagnostic benchmark for layout-guided image generation that examines four categories of spatial control skills: number, position, size, and shape. We benchmark two recent representative layout-guided image generation methods and observe that the good ID layout control may not generalize well to arbitrary layouts in the wild (e.g., objects at the boundary). Next, we propose IterInpaint, a new baseline that generates foreground and background regions in a step-by-step manner via inpainting, demonstrating stronger generalizability than existing models on OOD layouts in LayoutBench. We perform quantitative and q
    
[^32]: 大型语言模型的思维链主动提示

    Active Prompting with Chain-of-Thought for Large Language Models. (arXiv:2302.12246v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.12246](http://arxiv.org/abs/2302.12246)

    本论文提出了一种新的方法Active-Prompt，它使用任务特定的示例提示适应大型语言模型中的不同任务，提高模型性能与效率。

    

    大型语言模型的规模日益增大，为各种需要推理的复杂任务（如算术和常识推理）带来了新的能力。众所周知，任务特定提示的有效设计对LLMs产生高质量答案的能力至关重要。特别是，对于复杂的问答任务，一种有效的方法是基于示例的思维链（CoT）推导提示，它大大提高了LLMs的性能。但是，当前的CoT方法依赖于一组固定的人类注释示例，这些示例不一定是不同任务的最有效示例。本文提出了一种新方法，Active-Prompt，使用任务特定的示例提示（人为设计的CoT推理注释）来适应LLMs不同的任务。为此，我们提出了一个解决方案，确定哪些问题从任务特定查询池中注释最重要和有用。通过借鉴主动学习的方法，我们提出了一个主动提示(Acitve-Prompt)的方法，将最相关的问题作为任务特定提示添加给LLMs，从而改善其性能。

    The increasing scale of large language models (LLMs) brings emergent abilities to various complex tasks requiring reasoning, such as arithmetic and commonsense reasoning. It is known that the effective design of task-specific prompts is critical for LLMs' ability to produce high-quality answers. In particular, an effective approach for complex question-and-answer tasks is example-based prompting with chain-of-thought (CoT) reasoning, which significantly improves the performance of LLMs. However, current CoT methods rely on a fixed set of human-annotated exemplars, which are not necessarily the most effective examples for different tasks. This paper proposes a new method, Active-Prompt, to adapt LLMs to different tasks with task-specific example prompts (annotated with human-designed CoT reasoning). For this purpose, we propose a solution to the key problem of determining which questions are the most important and helpful ones to annotate from a pool of task-specific queries. By borrowi
    
[^33]: 阅读并获得回报：在与指导手册的帮助下学习玩Atari游戏

    Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals. (arXiv:2302.04449v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04449](http://arxiv.org/abs/2302.04449)

    本论文提出了阅读并奖励的框架，通过阅读Atari游戏开发者发布的指导手册，以提高强化学习算法在Atari游戏中的效率。该框架包含一个QA提取模块和一个推理模块，能够从指导手册中提取关键信息，并评估物体与智能体的交互效果。

    

    长期以来，高样本复杂性一直是强化学习面临的挑战。然而，人类学习执行任务的方式不仅仅是通过交互或演示，还包括阅读非结构化文本文档，例如指导手册。指导手册和维基页面是最丰富的数据之一，它们可以提供有关宝贵特征、策略、任务特定的环境动态和奖励结构的信息，因此我们假设利用人写的指导手册来帮助学习特定任务的策略将导致更高效和更优秀的智能体。我们提出了阅读并奖励的框架。阅读并奖励通过阅读Atari游戏开发者发布的指导手册来加速RL算法。我们的框架包括一个QA提取模块，用于提取和总结指导手册中的相关信息，以及一个推理模块，根据指导手册中的信息评估物体-智能体的交互效果。一个辅助的反馈机制可以提高效果。

    High sample complexity has long been a challenge for RL. On the other hand, humans learn to perform tasks not only from interaction or demonstrations, but also by reading unstructured text documents, e.g., instruction manuals. Instruction manuals and wiki pages are among the most abundant data that could inform agents of valuable features and policies or task-specific environmental dynamics and reward structures. Therefore, we hypothesize that the ability to utilize human-written instruction manuals to assist learning policies for specific tasks should lead to a more efficient and better-performing agent. We propose the Read and Reward framework. Read and Reward speeds up RL algorithms on Atari games by reading manuals released by the Atari game developers. Our framework consists of a QA Extraction module that extracts and summarizes relevant information from the manual and a Reasoning module that evaluates object-agent interactions based on information from the manual. An auxiliary re
    
[^34]: 自然对话中解释机器学习模型：走向对话式XAI代理

    Explaining Machine Learning Models in Natural Conversations: Towards a Conversational XAI Agent. (arXiv:2209.02552v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2209.02552](http://arxiv.org/abs/2209.02552)

    本研究将解释人工智能（XAI）融入到一个对话代理中，设计具有自然语言理解和生成组件的标准模型。通过扩展XAI问题库并提供解释方法，实现了关于机器学习模型的真正自然对话。

    

    可解释人工智能（XAI）的目标是设计方法来揭示黑盒模型（如深度神经网络）的推理过程，以便向人类解释。社会科学研究指出，这样的解释应该是对话式的，类似于人与人之间的解释。在这项工作中，我们展示了如何将XAI融入到一个对话代理中，使用了一个包括自然语言理解和生成组件的标准设计。我们根据质控的释义重述扩展了一个XAI问题库，以理解用户的信息需求。我们进一步系统地调查了适合提供答案信息的解释方法的文献，并提出了一个全面的建议列表。我们的工作是实现关于机器学习模型的真正自然对话的第一步，与一个解释代理有关的全面的XAI问题列表和相应的解释方法。

    The goal of Explainable AI (XAI) is to design methods to provide insights into the reasoning process of black-box models, such as deep neural networks, in order to explain them to humans. Social science research states that such explanations should be conversational, similar to human-to-human explanations. In this work, we show how to incorporate XAI in a conversational agent, using a standard design for the agent comprising natural language understanding and generation components. We build upon an XAI question bank which we extend by quality-controlled paraphrases to understand the user's information needs. We further systematically survey the literature for suitable explanation methods that provide the information to answer those questions, and present a comprehensive list of suggestions. Our work is the first step towards truly natural conversations about machine learning models with an explanation agent. The comprehensive list of XAI questions and the corresponding explanation meth
    
[^35]: 提升虚假新闻缓解：来自分享者社交媒体帖子历史的洞察力

    Empowering Fake-News Mitigation: Insights from Sharers' Social Media Post-Histories. (arXiv:2203.10560v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2203.10560](http://arxiv.org/abs/2203.10560)

    本论文提出消费者的社交媒体帖子历史是研究分享虚假新闻动机的一种被低估的数据来源。通过对帖子历史提取的文本线索，我们发现虚假新闻分享者在言辞上更多涉及愤怒、宗教和权力。并且，通过将帖子历史中的文本线索加入模型，可以提高预测分享虚假新闻的准确性。此外，通过激活宗教价值观和减少愤怒，可以减少虚假新闻的分享和更广泛的分享。

    

    虚假信息是一个全球性问题，限制其传播对保护民主、公共卫生和消费者至关重要。我们认为消费者自己的社交媒体帖子历史是一个被低估的数据来源，用于研究是什么导致他们分享虚假新闻链接。在第一项研究中，我们探讨了从帖子历史中提取的文本线索如何区分虚假新闻的分享者和随机社交媒体用户以及其他在误导信息生态系统中的人。在两个数据集中，我们发现虚假新闻的分享者使用更多与愤怒、宗教和权力相关的词汇。在第二项研究中，我们展示了从帖子历史中添加文本线索如何提高模型预测谁有可能分享虚假新闻的准确性。在第三项研究中，我们对从第一项研究中推导出的两种缓解策略进行了初步测试，即激活宗教价值观和减少愤怒，发现它们可以减少虚假新闻的分享和更广泛的分享。在第四项研究中，我们将调查结果与用户的验证推特結合在一起。

    Misinformation is a global concern and limiting its spread is critical for protecting democracy, public health, and consumers. We propose that consumers' own social media post-histories are an underutilized data source to study what leads them to share links to fake-news. In Study 1, we explore how textual cues extracted from post-histories distinguish fake-news sharers from random social media users and others in the misinformation ecosystem. Among other results, we find across two datasets that fake-news sharers use more words related to anger, religion and power. In Study 2, we show that adding textual cues from post-histories improves the accuracy of models to predict who is likely to share fake-news. In Study 3, we provide a preliminary test of two mitigation strategies deduced from Study 1 - activating religious values and reducing anger - and find that they reduce fake-news sharing and sharing more generally. In Study 4, we combine survey responses with users' verified Twitter p
    

