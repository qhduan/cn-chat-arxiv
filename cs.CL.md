# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs](https://rss.arxiv.org/abs/2312.05356) | 这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。 |
| [^2] | [Beyond Accuracy: Evaluating the Reasoning Behavior of Large Language Models -- A Survey](https://arxiv.org/abs/2404.01869) | 本文通过综述超越任务准确性的研究，提供对大型语言模型推理过程更深入了解，并强调了LLMs倾向于依赖于训练数据中的表面模式和相关性。 |
| [^3] | [A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning](https://arxiv.org/abs/2403.14972) | 提出了一种演绎式的图谱辩论方法（BDoG），在多模态推理中防止意见陈腐化和减少由图像引入的分心概念，实验证明其在科学问答和MMBench上取得了最先进的结果。 |
| [^4] | [Boosting Disfluency Detection with Large Language Model as Disfluency Generator](https://arxiv.org/abs/2403.08229) | 本研究提出了一种利用大型语言模型生成不流畅句子作为数据增强的轻量级方法，通过数据过滤和小型模型训练实现了不流畅检测性能的提升。 |
| [^5] | [FakeNewsGPT4: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs](https://arxiv.org/abs/2403.01988) | FakeNewsGPT4是一个新颖框架，通过增加特定于伪造的知识，提升了LVLMs在多模态假新闻检测中的效果。 |
| [^6] | [GAOKAO-MM: A Chinese Human-Level Benchmark for Multimodal Models Evaluation](https://arxiv.org/abs/2402.15745) | GAOKAO-MM 是基于中国高考的多模态基准，为模型的能力设定人类水平要求，评估结果显示目前的LVLMs的准确率普遍不足50%。 |
| [^7] | [Distilling Large Language Models for Text-Attributed Graph Learning](https://arxiv.org/abs/2402.12022) | 本研究旨在将大型语言模型和图模型的优势相结合，通过将LLMs的能力压缩到 TAG 学习的本地图模型中，解决它们之间的固有差距。 |
| [^8] | [ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation](https://arxiv.org/abs/2402.04492) | 本文介绍了ColorSwap数据集，用于评估和改进多模态模型在匹配物体和颜色方面的能力。通过将颜色词重新排序以修改不同的对象，该数据集可以测试模型在这项任务上的鲁棒性。尽管目前的模型在这个任务上仍不够稳定，但通过更先进的提示技术可能会有所改善。 |
| [^9] | [Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering](https://arxiv.org/abs/2401.06824) | 通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。 |
| [^10] | [A Simple Framework to Accelerate Multilingual Language Model for Monolingual Text Generation.](http://arxiv.org/abs/2401.10660) | 这项研究介绍了一种新颖的框架，旨在加速非英语语言的文本生成。通过预测更大的语言单元并针对目标语言进行调整，该框架降低了解码步骤的数量，并将生成速度提高了1.9倍。 |
| [^11] | [ARIES: A Corpus of Scientific Paper Edits Made in Response to Peer Reviews.](http://arxiv.org/abs/2306.12587) | ARIES是一份包含科学论文修订的语料库，为训练和评估大型语言模型提供了工具。通过评估模型，发现其在寻找对应的修订方面仍存在困难，同时在生成修订时过分遵循反馈的措辞，而不是考虑整体的语义。 |
| [^12] | [Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective.](http://arxiv.org/abs/2306.10512) | 本研究提出了一种自适应测试框架，用于高效测量语言模型的认知能力。通过动态调整测试问题的特性，能够更准确地评估模型的能力，并使用更少的问题。同时，该框架使得语言模型能够与人类进行轻松比较。 |
| [^13] | [Towards the Automatic Generation of Conversational Interfaces to Facilitate the Exploration of Tabular Data.](http://arxiv.org/abs/2305.11326) | 本文提出了使用聊天机器人作为自动化创造的会话接口来方便大众探索表格数据的方法。 |
| [^14] | [Tool Learning with Foundation Models.](http://arxiv.org/abs/2304.08354) | 基于基础模型的工具学习结合了专用工具和基础模型的优势，实现了问题解决的增强精度、效率和自动化。本文对工具学习进行了系统研究，提出了涵盖两种类型学习的通用工具学习框架，并分析了它们的独特挑战、机会和未来方向。 |

# 详细

[^1]: Neuron Patching: 神经元层面的模型编辑与代码生成

    Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs

    [https://rss.arxiv.org/abs/2312.05356](https://rss.arxiv.org/abs/2312.05356)

    这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。

    

    大型语言模型在软件工程中得到了成功应用，特别是在代码生成方面。更新这些模型的新知识非常昂贵，通常需要全面实现其价值。在本文中，我们提出了一种新颖有效的模型编辑方法MENT，用于在编码任务中修补LLM模型。基于生成式LLM的机制，MENT可以在预测下一个令牌时进行模型编辑，并进一步支持常见的编码任务。MENT具有高效、有效和可靠的特点。它可以通过修补1或2个神经元来纠正神经模型。作为神经元层面上生成模型编辑的先驱工作，我们规范了编辑过程并介绍了相关概念。此外，我们还引入了新的衡量方法来评估其泛化能力，并建立了一个用于进一步研究的基准。我们的方法在三个编码任务上进行了评估，包括API序列推荐、行级代码生成和伪代码到代码转换。

    Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction
    
[^2]: 超越准确性：评估大型语言模型的推理行为--一项调查

    Beyond Accuracy: Evaluating the Reasoning Behavior of Large Language Models -- A Survey

    [https://arxiv.org/abs/2404.01869](https://arxiv.org/abs/2404.01869)

    本文通过综述超越任务准确性的研究，提供对大型语言模型推理过程更深入了解，并强调了LLMs倾向于依赖于训练数据中的表面模式和相关性。

    

    大型语言模型（LLMs）最近在涉及推理的任务中表现出色，引发了关于这些模型是否具有类似于人类的推理能力的激烈讨论。然而，尽管取得了成功，但LLMs的推理能力的深度仍然存在不确定性。这种不确定性部分源自对模型推理行为的深入调查而非仅仅通过表面准确性指标来衡量任务表现。本文旨在通过综述超越任务准确性的研究，提供对模型推理过程更深入的了解来弥补这一差距。此外，我们调查了评估LLMs推理行为的主要方法论，强调了当前对更细致推理分析的趋势和努力。我们的综述表明，LLMs倾向于依赖于训练数据中的表面模式和相关性。

    arXiv:2404.01869v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently shown impressive performance on tasks involving reasoning, leading to a lively debate on whether these models possess reasoning capabilities similar to humans. However, despite these successes, the depth of LLMs' reasoning abilities remains uncertain. This uncertainty partly stems from the predominant focus on task performance, measured through shallow accuracy metrics, rather than a thorough investigation of the models' reasoning behavior. This paper seeks to address this gap by providing a comprehensive review of studies that go beyond task accuracy, offering deeper insights into the models' reasoning processes. Furthermore, we survey prevalent methodologies to evaluate the reasoning behavior of LLMs, emphasizing current trends and efforts towards more nuanced reasoning analyses. Our review suggests that LLMs tend to rely on surface-level patterns and correlations in their training data, rat
    
[^3]: 一图胜千言：多模态推理中的图谱辩论

    A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning

    [https://arxiv.org/abs/2403.14972](https://arxiv.org/abs/2403.14972)

    提出了一种演绎式的图谱辩论方法（BDoG），在多模态推理中防止意见陈腐化和减少由图像引入的分心概念，实验证明其在科学问答和MMBench上取得了最先进的结果。

    

    本文介绍了一项旨在将多智能体辩论引入多模态推理的试点研究。该研究解决了两个关键挑战：由于过度总结而导致意见陈腐化，以及由于图像引入转移性概念而导致注意力分散的问题。这些挑战源自现有辩论方案的归纳（自下而上）性质。为解决这一问题，我们提出了一种演绎（自上而下）的辩论方法，称为图谱辩论（BDoG）。在BDoG中，辩论仅限于蓝图图中，以防止通过世界级摘要而导致意见陈腐化。此外，通过在图中的分支中存储证据，BDoG缓解了频繁但无关的概念带来的分散注意力现象。大量实验验证了BDoG，在科学问答和MMBench中取得了最新成果，并相较于先前的方法具有显著改进。

    arXiv:2403.14972v1 Announce Type: new  Abstract: This paper presents a pilot study aimed at introducing multi-agent debate into multimodal reasoning. The study addresses two key challenges: the trivialization of opinions resulting from excessive summarization and the diversion of focus caused by distractor concepts introduced from images. These challenges stem from the inductive (bottom-up) nature of existing debating schemes. To address the issue, we propose a deductive (top-down) debating approach called Blueprint Debate on Graphs (BDoG). In BDoG, debates are confined to a blueprint graph to prevent opinion trivialization through world-level summarization. Moreover, by storing evidence in branches within the graph, BDoG mitigates distractions caused by frequent but irrelevant concepts. Extensive experiments validate BDoG, achieving state-of-the-art results in Science QA and MMBench with significant improvements over previous methods.
    
[^4]: 利用大型语言模型作为语篇生成器提升不流畅检测

    Boosting Disfluency Detection with Large Language Model as Disfluency Generator

    [https://arxiv.org/abs/2403.08229](https://arxiv.org/abs/2403.08229)

    本研究提出了一种利用大型语言模型生成不流畅句子作为数据增强的轻量级方法，通过数据过滤和小型模型训练实现了不流畅检测性能的提升。

    

    当前的不流畅检测方法严重依赖昂贵且稀缺的人工标注数据。为了解决这一问题，一些方法采用启发式或统计特征来生成不流畅句子，部分提高了检测性能。然而，这些句子常常偏离真实场景，限制了整体模型改善。本研究提出了一种轻量级数据增强方法，利用大型语言模型（LLM）卓越的生成和语义理解能力生成不流畅句子作为增强数据。我们利用LLM生成多样且更真实的句子，通过具体提示进行引导，无需对LLM进行微调。随后，我们应用一种基于不确定性的数据过滤方法来提高生成句子的质量，用于训练小型检测模型以提高性能。

    arXiv:2403.08229v1 Announce Type: new  Abstract: Current disfluency detection methods heavily rely on costly and scarce human-annotated data. To tackle this issue, some approaches employ heuristic or statistical features to generate disfluent sentences, partially improving detection performance. However, these sentences often deviate from real-life scenarios, constraining overall model enhancement. In this study, we propose a lightweight data augmentation approach for disfluency detection, utilizing the superior generative and semantic understanding capabilities of large language model (LLM) to generate disfluent sentences as augmentation data. We leverage LLM to generate diverse and more realistic sentences guided by specific prompts, without the need for fine-tuning the LLM. Subsequently, we apply an uncertainty-aware data filtering approach to improve the quality of the generated sentences, utilized in training a small detection model for improved performance. Experiments using enha
    
[^5]: FakeNewsGPT4：通过知识增强的LVLMs推进多模态假新闻检测

    FakeNewsGPT4: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs

    [https://arxiv.org/abs/2403.01988](https://arxiv.org/abs/2403.01988)

    FakeNewsGPT4是一个新颖框架，通过增加特定于伪造的知识，提升了LVLMs在多模态假新闻检测中的效果。

    

    大规模生成的多模态假新闻存在实质性的分布差异，促使需要广义检测器。然而，训练在特定领域内的孤立性限制了传统检测器获得开放世界事实的能力。本文提出了FakeNewsGPT4，这是一个新颖的框架，通过增添特定于伪造的知识来增强大规模视觉-语言模型（LVLMs）进行操纵推理，同时继承丰富的世界知识作为补充。FakeNewsGPT4中的知识增强涉及获取两种伪造特定知识，即语义相关和工件追踪，将它们合并到LVLMs中。

    arXiv:2403.01988v1 Announce Type: new  Abstract: The massive generation of multimodal fake news exhibits substantial distribution discrepancies, prompting the need for generalized detectors. However, the insulated nature of training within specific domains restricts the capability of classical detectors to obtain open-world facts. In this paper, we propose FakeNewsGPT4, a novel framework that augments Large Vision-Language Models (LVLMs) with forgery-specific knowledge for manipulation reasoning while inheriting extensive world knowledge as complementary. Knowledge augmentation in FakeNewsGPT4 involves acquiring two types of forgery-specific knowledge, i.e., semantic correlation and artifact trace, and merging them into LVLMs. Specifically, we design a multi-level cross-modal reasoning module that establishes interactions across modalities for extracting semantic correlations. Concurrently, a dual-branch fine-grained verification module is presented to comprehend localized details to e
    
[^6]: GAOKAO-MM: 一个用于多模态模型评估的中国人类水平基准

    GAOKAO-MM: A Chinese Human-Level Benchmark for Multimodal Models Evaluation

    [https://arxiv.org/abs/2402.15745](https://arxiv.org/abs/2402.15745)

    GAOKAO-MM 是基于中国高考的多模态基准，为模型的能力设定人类水平要求，评估结果显示目前的LVLMs的准确率普遍不足50%。

    

    大型视觉语言模型（LVLMs）已经在图像感知和语言理解方面展示出了极大的能力。然而，现有的多模态基准主要关注基本的感知能力和常识知识，这些无法充分反映出LVLMs的全面能力。我们提出了GAOKAO-MM，一个基于中国高考的多模态基准，包括8个科目和12种类型的图片，如图表、函数图、地图和照片。GAOKAO-MM来源于中国本土背景，并为模型的能力设定了人类水平的要求，包括感知、理解、知识和推理。我们评估了10个LVLMs，发现它们的准确率都低于50%，其中GPT-4-Vision（48.1%）、Qwen-VL-Plus（41.2%）和Gemini-Pro-Vision（35.1%）位列前三名。我们的多维分析结果表明，LVLMs具有适度的

    arXiv:2402.15745v1 Announce Type: cross  Abstract: The Large Vision-Language Models (LVLMs) have demonstrated great abilities in image perception and language understanding. However, existing multimodal benchmarks focus on primary perception abilities and commonsense knowledge which are insufficient to reflect the comprehensive capabilities of LVLMs. We propose GAOKAO-MM, a multimodal benchmark based on the Chinese College Entrance Examination (GAOKAO), comprising of 8 subjects and 12 types of images, such as diagrams, function graphs, maps and photos. GAOKAO-MM derives from native Chinese context and sets human-level requirements for the model's abilities, including perception, understanding, knowledge and reasoning. We evaluate 10 LVLMs and find that the accuracies of all of them are lower than 50%, with GPT-4-Vison (48.1%), Qwen-VL-Plus (41.2%) and Gemini-Pro-Vision (35.1%) ranking in the top three positions. The results of our multi-dimension analysis indicate that LVLMs have moder
    
[^7]: 将大型语言模型压缩用于文本属性图学习

    Distilling Large Language Models for Text-Attributed Graph Learning

    [https://arxiv.org/abs/2402.12022](https://arxiv.org/abs/2402.12022)

    本研究旨在将大型语言模型和图模型的优势相结合，通过将LLMs的能力压缩到 TAG 学习的本地图模型中，解决它们之间的固有差距。

    

    文本属性图（TAGs）是连接的文本文档图。图模型可以有效学习TAGs，但它们的训练严重依赖于人工标注的标签，在许多应用中这些标签很少或甚至不可用。大型语言模型（LLMs）最近在少样本和零样本TAG学习中展示了显著能力，但它们存在可伸缩性、成本和隐私问题。因此，在这项工作中，我们专注于通过将LLMs的能力传授给TAG学习中的本地图模型，从而协同LLMs和图模型的互补优势。

    arXiv:2402.12022v1 Announce Type: new  Abstract: Text-Attributed Graphs (TAGs) are graphs of connected textual documents. Graph models can efficiently learn TAGs, but their training heavily relies on human-annotated labels, which are scarce or even unavailable in many applications. Large language models (LLMs) have recently demonstrated remarkable capabilities in few-shot and zero-shot TAG learning, but they suffer from scalability, cost, and privacy issues. Therefore, in this work, we focus on synergizing LLMs and graph models with their complementary strengths by distilling the power of LLMs to a local graph model on TAG learning. To address the inherent gaps between LLMs (generative models for texts) and graph models (discriminative models for graphs), we propose first to let LLMs teach an interpreter with rich textual rationale and then let a student model mimic the interpreter's reasoning without LLMs' textual rationale. Extensive experiments validate the efficacy of our proposed 
    
[^8]: ColorSwap: 一个用于多模态评估的颜色和单词排序数据集

    ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation

    [https://arxiv.org/abs/2402.04492](https://arxiv.org/abs/2402.04492)

    本文介绍了ColorSwap数据集，用于评估和改进多模态模型在匹配物体和颜色方面的能力。通过将颜色词重新排序以修改不同的对象，该数据集可以测试模型在这项任务上的鲁棒性。尽管目前的模型在这个任务上仍不够稳定，但通过更先进的提示技术可能会有所改善。

    

    本文介绍了ColorSwap数据集，旨在评估和改进多模态模型在匹配物体和其颜色方面的熟练程度。该数据集包含2000个独特的图像-标题对，分为1000个示例。每个示例包括一个标题-图像对，以及一个“颜色交换”对。我们遵循Winoground方案：示例中的两个标题具有相同的单词，但颜色单词被重新排列以修改不同的对象。该数据集通过自动化的标题和图像生成与人类的交互创造而成。我们评估图像-文本匹配（ITM）和视觉语言模型（VLMs）发现即使是最新的模型在这个任务上仍然不够稳健。GPT-4V和LLaVA在我们的主要VLM指标上得分分别为72%和42%，尽管它们可能通过更先进的提示技术来提升。在主要的ITM指标上，像CLIP和SigLIP这样的对比模型接近于随机猜测（分别为12%和30%），尽管非对比模型在这个任务上表现得更好。

    This paper introduces the ColorSwap dataset, designed to assess and improve the proficiency of multimodal models in matching objects with their colors. The dataset is comprised of 2,000 unique image-caption pairs, grouped into 1,000 examples. Each example includes a caption-image pair, along with a ``color-swapped'' pair. We follow the Winoground schema: the two captions in an example have the same words, but the color words have been rearranged to modify different objects. The dataset was created through a novel blend of automated caption and image generation with humans in the loop. We evaluate image-text matching (ITM) and visual language models (VLMs) and find that even the latest ones are still not robust at this task. GPT-4V and LLaVA score 72% and 42% on our main VLM metric, although they may improve with more advanced prompting techniques. On the main ITM metric, contrastive models such as CLIP and SigLIP perform close to chance (at 12% and 30%, respectively), although the non-
    
[^9]: 打开LLMs的潘多拉魔盒：通过表示工程对LLMs进行越狱

    Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering

    [https://arxiv.org/abs/2401.06824](https://arxiv.org/abs/2401.06824)

    通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。

    

    越狱技术旨在通过诱使大型语言模型（LLMs）生成对恶意查询产生有毒响应，来探索LLMs安全性边界，这在LLMs社区内是一个重要关注点。我们提出一种名为通过表示工程对LLMs进行越狱（Jailbreaking LLMs through Representation Engineering，JRE）的新颖越狱方法，其仅需要少量查询对以提取可用于规避目标模型防御的“安全模式”，实现了前所未有的越狱性能。

    arXiv:2401.06824v2 Announce Type: replace-cross  Abstract: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation conf
    
[^10]: 一种用于单语文本生成的加速多语言语言模型的简单框架

    A Simple Framework to Accelerate Multilingual Language Model for Monolingual Text Generation. (arXiv:2401.10660v1 [cs.CL])

    [http://arxiv.org/abs/2401.10660](http://arxiv.org/abs/2401.10660)

    这项研究介绍了一种新颖的框架，旨在加速非英语语言的文本生成。通过预测更大的语言单元并针对目标语言进行调整，该框架降低了解码步骤的数量，并将生成速度提高了1.9倍。

    

    最近大型语言模型的进展不仅在英语而且在非英语语言中都促进了复杂的语言任务的执行。然而，大多数语言模型的标记器（如Llama）在以英语为中心的语料库上训练，倾向于在非英语语言中过分分割标记。这个问题在非罗马字母语言中尤为明显，这些语言通常在字符或Unicode级别上被划分，导致文本生成速度较慢。为了解决这个问题，我们的研究介绍了一个新颖的框架，旨在加速这些语言的文本生成。该框架预测比传统的多语言标记器更大的语言单元，并且专门针对目标语言进行了调整，从而减少了解码所需的步骤数。我们的实证结果表明，与标准解码相比，所提出的框架将生成速度提高了1.9倍，同时保持了预先训练模型的性能。

    Recent advancements in large language models have facilitated the execution of complex language tasks, not only in English but also in non-English languages. However, the tokenizers of most language models, such as Llama, trained on English-centric corpora, tend to excessively fragment tokens in non-English languages. This issue is especially pronounced in non-roman alphabetic languages, which are often divided at a character or even Unicode level, leading to slower text generation. To address this, our study introduces a novel framework designed to expedite text generation in these languages. This framework predicts larger linguistic units than those of conventional multilingual tokenizers and is specifically tailored to the target language, thereby reducing the number of decoding steps required. Our empirical results demonstrate that the proposed framework increases the generation speed by a factor of 1.9 compared to standard decoding while maintaining the performance of a pre-traine
    
[^11]: ARIES: 一份包含科学论文修订的语料库，这些修订是作为对同行评审的回应而进行的

    ARIES: A Corpus of Scientific Paper Edits Made in Response to Peer Reviews. (arXiv:2306.12587v1 [cs.CL])

    [http://arxiv.org/abs/2306.12587](http://arxiv.org/abs/2306.12587)

    ARIES是一份包含科学论文修订的语料库，为训练和评估大型语言模型提供了工具。通过评估模型，发现其在寻找对应的修订方面仍存在困难，同时在生成修订时过分遵循反馈的措辞，而不是考虑整体的语义。

    

    根据同行反馈修改科学论文是一项具有挑战性的任务，需要深厚的科学知识和推理能力，同时还需要识别高级反馈中的隐含意义，并在众多可能的方式中选择最佳的方式来更新手稿。我们为大语言模型提出了这个任务，并发布了ARIES数据集，其中包含了评论及其相应的论文修订，以便进行训练和评估模型。我们研究了任务的两个版本：评论-修订对齐和修订生成，并评估了几个基线模型，包括GPT-4。我们发现即使在评论以间接方式表述或修订涉及评论的主旨而非精确要求的情况下，模型仍然难以确定对应于评论的修订。在生成修订时，GPT-4通常能够在表面上处理好评论，但它过分遵循反馈的措辞，而不是考虑整体的语义。

    Revising scientific papers based on peer feedback is a challenging task that requires not only deep scientific knowledge and reasoning, but also the ability to recognize the implicit requests in high-level feedback and to choose the best of many possible ways to update the manuscript in response. We introduce this task for large language models and release ARIES, a dataset of review comments and their corresponding paper edits, to enable training and evaluating models. We study two versions of the task: comment-edit alignment and edit generation, and evaluate several baselines, including GPT-4. We find that models struggle even to identify the edits that correspond to a comment, especially in cases where the comment is phrased in an indirect way or where the edit addresses the spirit of a comment but not the precise request. When tasked with generating edits, GPT-4 often succeeds in addressing comments on a surface level, but it rigidly follows the wording of the feedback rather than t
    
[^12]: 高效测量语言模型的认知能力：自适应测试视角

    Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective. (arXiv:2306.10512v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.10512](http://arxiv.org/abs/2306.10512)

    本研究提出了一种自适应测试框架，用于高效测量语言模型的认知能力。通过动态调整测试问题的特性，能够更准确地评估模型的能力，并使用更少的问题。同时，该框架使得语言模型能够与人类进行轻松比较。

    

    大型语言模型（LLMs），如ChatGPT，展现了一些类似于人类的认知能力。为了比较不同模型的这些能力，通常采用来自不同领域（如文学、生物学和心理学）的多个基准（即标准测试问题集），并报告传统度量指标（如准确率、召回率和F1）。然而，从认知科学的角度来看，这种评估LLMs的方法可能效率低下且不准确。受心理测量学中计算机自适应测试（CAT）的启发，我们提出了一种适用于LLM评估的自适应测试框架。该方法根据模型的表现动态调整测试问题的特性（如难度），而不是使用标准的测试集并简单报告准确率。这使得能更准确地估计模型的能力，并使用更少的问题。更重要的是，它使LLMs能够与人类进行轻松比较，这是至关重要的。

    Large language models (LLMs), like ChatGPT, have shown some human-like cognitive abilities. For comparing these abilities of different models, several benchmarks (i.e. sets of standard test questions) from different fields (e.g., Literature, Biology and Psychology) are often adopted and the test results under traditional metrics such as accuracy, recall and F1, are reported. However, such way for evaluating LLMs can be inefficient and inaccurate from the cognitive science perspective. Inspired by Computerized Adaptive Testing (CAT) used in psychometrics, we propose an adaptive testing framework for LLM evaluation. Rather than using a standard test set and simply reporting accuracy, this approach dynamically adjusts the characteristics of the test questions, such as difficulty, based on the model's performance. This allows for a more accurate estimation of the model's abilities, using fewer questions. More importantly, it allows LLMs to be compared with humans easily, which is essential
    
[^13]: 自动生成会话接口以便于探索表格数据

    Towards the Automatic Generation of Conversational Interfaces to Facilitate the Exploration of Tabular Data. (arXiv:2305.11326v1 [cs.CL])

    [http://arxiv.org/abs/2305.11326](http://arxiv.org/abs/2305.11326)

    本文提出了使用聊天机器人作为自动化创造的会话接口来方便大众探索表格数据的方法。

    

    表格数据是在线发布和交换结构化数据的最常见格式。一个明确的例子是各种类型的公共行政机构发布的开放数据门户数量的增长。但是，这些数据源的利用目前仅限于能够以程序方式处理和消化此类数据的技术人员。作为替代方案，我们建议使用聊天机器人提供会话接口，以便于探索表格数据源。通过我们的方法，任何普通公民都可以从中受益并利用它们。此外，我们的聊天机器人不是手动创建的：相反，它们是通过实例化可配置的对话模式集从数据源本身自动生成的。

    Tabular data is the most common format to publish and exchange structured data online. A clear example is the growing number of open data portals published by all types of public administrations. However, exploitation of these data sources is currently limited to technical people able to programmatically manipulate and digest such data. As an alternative, we propose the use of chatbots to offer a conversational interface to facilitate the exploration of tabular data sources. With our approach, any regular citizen can benefit and leverage them. Moreover, our chatbots are not manually created: instead, they are automatically generated from the data source itself thanks to the instantiation of a configurable collection of conversation patterns.
    
[^14]: 基于基础模型的工具学习

    Tool Learning with Foundation Models. (arXiv:2304.08354v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.08354](http://arxiv.org/abs/2304.08354)

    基于基础模型的工具学习结合了专用工具和基础模型的优势，实现了问题解决的增强精度、效率和自动化。本文对工具学习进行了系统研究，提出了涵盖两种类型学习的通用工具学习框架，并分析了它们的独特挑战、机会和未来方向。

    

    人类拥有非凡的创造和利用工具的能力，使得他们能够克服物理限制并探索新的领域。随着基础模型的出现，AI系统有望像人类一样熟练地使用工具。这种范式即基于基础模型的工具学习，结合了专用工具和基础模型的优势，实现了问题解决的增强精度、效率和自动化。尽管具有巨大潜力，但该领域仍缺乏对关键挑战、机会和未来发展的全面理解。针对这一问题，本文对工具学习进行了系统研究。首先介绍了工具学习的背景，包括其认知起源、基础模型的范式转换和工具和模型的互补作用。然后，我们回顾了现有的工具学习研究，包括基于工具和面向工具的学习。我们制定了一个涵盖两种类型学习的通用工具学习框架，并分析了它们的独特挑战、机会和未来方向。我们预计这种系统的探索将为未来开发具有复杂工具学习能力的AI系统提供一个跳板。

    Humans possess an extraordinary ability to create and utilize tools, allowing them to overcome physical limitations and explore new frontiers. With the advent of foundation models, AI systems have the potential to be equally adept in tool use as humans. This paradigm, i.e., tool learning with foundation models, combines the strengths of specialized tools and foundation models to achieve enhanced accuracy, efficiency, and automation in problem-solving. Despite its immense potential, there is still a lack of a comprehensive understanding of key challenges, opportunities, and future endeavors in this field. To this end, we present a systematic investigation of tool learning in this paper. We first introduce the background of tool learning, including its cognitive origins, the paradigm shift of foundation models, and the complementary roles of tools and models. Then we recapitulate existing tool learning research into tool-augmented and tool-oriented learning. We formulate a general tool l
    

